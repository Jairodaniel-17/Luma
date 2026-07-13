use anyhow::{anyhow, Result};
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

type EmbedCache = Arc<Mutex<LruCache<String, Vec<f32>>>>;

/// Default per-request timeout (connect + full round trip) applied to every
/// embedding HTTP client. Mirrors `Config::request_timeout_secs`'s default so
/// behaviour is unchanged unless a caller overrides it via `with_timeout`.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;

/// Build a reqwest client with connect + total request timeouts so a hung or
/// slow embedding provider can never pin a request (and the inflight/ingest
/// permits held around it) forever. Falls back to a default client only if the
/// builder rejects the options, which should not happen for plain timeouts.
fn build_http_client(timeout_secs: u64) -> reqwest::Client {
    let total = Duration::from_secs(timeout_secs.max(1));
    let connect = Duration::from_secs(timeout_secs.min(10).max(1));
    reqwest::Client::builder()
        .timeout(total)
        .connect_timeout(connect)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

#[derive(Debug, Clone)]
pub enum EmbeddingProvider {
    OpenAI {
        api_url: String,
        api_key: String,
        model: String,
    },
    AzureOpenAI {
        api_base: String,
        deployment: String,
        api_key: String,
        api_version: String,
    },
    Cohere {
        api_url: String,
        api_key: String,
        model: String,
        input_type: String,
    },
    HuggingFace {
        api_url: String,
        api_key: String,
        model: String,
    },
    Ollama {
        api_url: String,
        model: String,
    },
    Mock {
        dim: usize,
    },
    None,
}

#[derive(Clone)]
pub struct EmbeddingClient {
    provider: EmbeddingProvider,
    client: reqwest::Client,
    /// Optional LRU cache: key = provider/model/dim/text namespace.
    cache: Option<EmbedCache>,
    /// Metrics handle for tracking cache hits/misses (optional).
    metrics: Option<Arc<crate::engine::metrics::Metrics>>,
    max_inflight_requests: usize,
    /// Bounds concurrent outbound HTTP requests to the embedding provider.
    inflight_semaphore: Option<Arc<Semaphore>>,
    /// Expected output dimension — included in cache keys to prevent collisions
    /// when the same model is used at different dims (e.g. text-embedding-3-large
    /// supports variable dimensions via the `dimensions` API parameter).
    dim: usize,
    /// Max retry attempts for transient provider errors (min 1 = no retry).
    retry_attempts: u32,
    /// Initial backoff delay in ms; doubles each retry.
    retry_initial_ms: u64,
}

impl Default for EmbeddingClient {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::None,
            client: build_http_client(DEFAULT_REQUEST_TIMEOUT_SECS),
            cache: None,
            metrics: None,
            max_inflight_requests: 16,
            inflight_semaphore: None,
            dim: 0,
            retry_attempts: 1,
            retry_initial_ms: 200,
        }
    }
}

impl EmbeddingClient {
    pub fn new(provider: EmbeddingProvider) -> Self {
        Self {
            provider,
            client: build_http_client(DEFAULT_REQUEST_TIMEOUT_SECS),
            cache: None,
            metrics: None,
            max_inflight_requests: 16,
            inflight_semaphore: None,
            dim: 0,
            retry_attempts: 1,
            retry_initial_ms: 200,
        }
    }

    pub fn with_limits(
        provider: EmbeddingProvider,
        cache_size: usize,
        max_inflight_requests: usize,
        metrics: Option<Arc<crate::engine::metrics::Metrics>>,
    ) -> Self {
        Self::with_limits_and_dim(provider, cache_size, max_inflight_requests, metrics, 0)
    }

    pub fn with_limits_and_dim(
        provider: EmbeddingProvider,
        cache_size: usize,
        max_inflight_requests: usize,
        metrics: Option<Arc<crate::engine::metrics::Metrics>>,
        dim: usize,
    ) -> Self {
        let n = max_inflight_requests.max(1);
        let cache =
            NonZeroUsize::new(cache_size).map(|cap| Arc::new(Mutex::new(LruCache::new(cap))));
        Self {
            provider,
            client: build_http_client(DEFAULT_REQUEST_TIMEOUT_SECS),
            cache,
            metrics,
            max_inflight_requests: n,
            inflight_semaphore: Some(Arc::new(Semaphore::new(n))),
            dim,
            retry_attempts: 1,
            retry_initial_ms: 200,
        }
    }

    pub fn with_retry(mut self, attempts: u32, initial_ms: u64) -> Self {
        self.retry_attempts = attempts.max(1);
        self.retry_initial_ms = initial_ms;
        self
    }

    /// Override the per-request HTTP timeout (seconds). Additive: callers that
    /// want config-driven timeouts (e.g. `Config::request_timeout_secs`) can
    /// chain this after construction.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.client = build_http_client(timeout_secs);
        self
    }

    /// Backward-compatible helper for tests/callers that only care about cache size.
    pub fn with_cache(
        provider: EmbeddingProvider,
        cache_size: usize,
        metrics: Option<Arc<crate::engine::metrics::Metrics>>,
    ) -> Self {
        Self::with_limits(provider, cache_size, 16, metrics)
    }

    /// Embed a single text string. Uses the LRU cache when enabled.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cache) = &self.cache {
            let key = self.cache_key(text);
            {
                let mut guard = cache.lock();
                if let Some(cached) = cache_lookup(&mut guard, &key) {
                    if let Some(m) = &self.metrics {
                        m.inc_embed_cache_hit();
                    }
                    return Ok(cached);
                }
            }
            if let Some(m) = &self.metrics {
                m.inc_embed_cache_miss();
            }
            let vec = self.embed_uncached(text).await?;
            cache_store(&mut cache.lock(), key, vec.clone());
            return Ok(vec);
        }
        self.embed_uncached(text).await
    }

    /// Embed a batch of texts in a single provider call where possible.
    /// PR3: true batching for OpenAI; parallel for Ollama; trivial for Mock.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check cache for all inputs first
        if let Some(cache) = &self.cache {
            let mut result: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
            let mut missing_indices: Vec<usize> = Vec::new();

            {
                let mut guard = cache.lock();
                for (i, text) in texts.iter().enumerate() {
                    let key = self.cache_key(text);
                    if let Some(cached) = cache_lookup(&mut guard, &key) {
                        if let Some(m) = &self.metrics {
                            m.inc_embed_cache_hit();
                        }
                        result[i] = Some(cached);
                    } else {
                        missing_indices.push(i);
                    }
                }
            }

            if missing_indices.is_empty() {
                return Ok(result
                    .into_iter()
                    .map(|v| v.expect("all values are present"))
                    .collect());
            }

            // Embed the missing ones in batch
            let missing_texts: Vec<String> =
                missing_indices.iter().map(|&i| texts[i].clone()).collect();
            let missing_vecs = self.embed_batch_uncached(&missing_texts).await?;
            if missing_vecs.len() != missing_indices.len() {
                return Err(anyhow!(
                    "Batch embedding response size mismatch: expected {}, got {}",
                    missing_indices.len(),
                    missing_vecs.len()
                ));
            }

            // Update cache and fill result
            {
                let mut guard = cache.lock();
                for (&idx, vec) in missing_indices.iter().zip(missing_vecs.iter()) {
                    if let Some(m) = &self.metrics {
                        m.inc_embed_cache_miss();
                    }
                    let key = self.cache_key(&texts[idx]);
                    cache_store(&mut guard, key, vec.clone());
                    result[idx] = Some(vec.clone());
                }
            }

            return result
                .into_iter()
                .map(|v| v.ok_or_else(|| anyhow!("Missing embedding result after batch merge")))
                .collect();
        }

        self.embed_batch_uncached(texts).await
    }

    /// Sleep with exponential backoff + jitter before a retry attempt.
    /// Shared by the single-embed and batch-embed retry loops.
    async fn retry_backoff_sleep(&self, attempt: u32) {
        let base_ms = self.retry_initial_ms << (attempt - 1).min(5);
        let jitter_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_millis() as u64)
            .unwrap_or(0)
            % (base_ms / 4 + 1);
        tokio::time::sleep(Duration::from_millis(base_ms + jitter_ms)).await;
    }

    async fn embed_uncached(&self, text: &str) -> Result<Vec<f32>> {
        let _permit = if let Some(sem) = &self.inflight_semaphore {
            Some(
                sem.acquire()
                    .await
                    .map_err(|e| anyhow!("semaphore closed: {e}"))?,
            )
        } else {
            None
        };
        let is_network = !matches!(
            &self.provider,
            EmbeddingProvider::None | EmbeddingProvider::Mock { .. }
        );
        let max_attempts = if is_network {
            self.retry_attempts.max(1)
        } else {
            1
        };
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                self.retry_backoff_sleep(attempt).await;
            }
            let result = self.call_provider_single(text).await;
            match result {
                Ok(v) => return Ok(v),
                Err(e) => {
                    tracing::warn!(
                        attempt = attempt + 1,
                        max = max_attempts,
                        error = %e,
                        "embedding attempt failed"
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("embedding failed after {max_attempts} attempts")))
    }

    async fn call_provider_single(&self, text: &str) -> Result<Vec<f32>> {
        match &self.provider {
            EmbeddingProvider::None => Err(anyhow::anyhow!("Embeddings are not configured")),
            EmbeddingProvider::OpenAI {
                api_url,
                api_key,
                model,
            } => self.embed_openai(api_url, api_key, model, text).await,
            EmbeddingProvider::AzureOpenAI {
                api_base,
                deployment,
                api_key,
                api_version,
            } => {
                self.embed_azure(api_base, deployment, api_key, api_version, text)
                    .await
            }
            EmbeddingProvider::Cohere {
                api_url,
                api_key,
                model,
                input_type,
            } => {
                self.embed_cohere_single(api_url, api_key, model, input_type, text)
                    .await
            }
            EmbeddingProvider::HuggingFace {
                api_url,
                api_key,
                model,
            } => {
                self.embed_huggingface_single(api_url, api_key, model, text)
                    .await
            }
            EmbeddingProvider::Ollama { api_url, model } => {
                self.embed_ollama(api_url, model, text).await
            }
            EmbeddingProvider::Mock { dim } => Ok(self.embed_mock(*dim, text)),
        }
    }

    /// Batch embedding with the same retry/backoff policy as `embed_uncached`.
    async fn embed_batch_uncached(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let is_network = !matches!(
            &self.provider,
            EmbeddingProvider::None | EmbeddingProvider::Mock { .. }
        );
        let max_attempts = if is_network {
            self.retry_attempts.max(1)
        } else {
            1
        };
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                self.retry_backoff_sleep(attempt).await;
            }
            match self.embed_batch_dispatch(texts).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    tracing::warn!(
                        attempt = attempt + 1,
                        max = max_attempts,
                        error = %e,
                        "batch embedding attempt failed"
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| anyhow!("batch embedding failed after {max_attempts} attempts")))
    }

    async fn embed_batch_dispatch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &self.provider {
            EmbeddingProvider::None => Err(anyhow::anyhow!("Embeddings are not configured")),
            EmbeddingProvider::OpenAI {
                api_url,
                api_key,
                model,
            } => {
                self.embed_batch_openai(api_url, api_key, model, texts)
                    .await
            }
            EmbeddingProvider::AzureOpenAI {
                api_base,
                deployment,
                api_key,
                api_version,
            } => {
                self.embed_batch_azure(api_base, deployment, api_key, api_version, texts)
                    .await
            }
            EmbeddingProvider::Cohere {
                api_url,
                api_key,
                model,
                input_type,
            } => {
                self.embed_batch_cohere(api_url, api_key, model, input_type, texts)
                    .await
            }
            EmbeddingProvider::HuggingFace {
                api_url,
                api_key,
                model,
            } => {
                self.embed_batch_huggingface(api_url, api_key, model, texts)
                    .await
            }
            EmbeddingProvider::Ollama { api_url, model } => {
                self.embed_batch_ollama(api_url, model, texts).await
            }
            EmbeddingProvider::Mock { dim } => {
                Ok(texts.iter().map(|t| self.embed_mock(*dim, t)).collect())
            }
        }
    }

    /// Validate a returned embedding's length against the configured `dim`.
    /// A `dim` of 0 means "unspecified" and disables the check. Prevents a
    /// misconfigured/misbehaving provider from silently poisoning the index
    /// with wrong-width vectors.
    fn check_dim(&self, got: usize) -> Result<()> {
        if self.dim != 0 && got != self.dim {
            return Err(anyhow!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dim,
                got
            ));
        }
        Ok(())
    }

    fn check_dims(&self, vecs: &[Vec<f32>]) -> Result<()> {
        for v in vecs {
            self.check_dim(v.len())?;
        }
        Ok(())
    }

    fn embed_mock(&self, dim: usize, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0; dim];
        if dim > 0 {
            let sum: u32 = text.bytes().map(|b| b as u32).sum();
            vec[0] = (sum % 100) as f32 / 100.0;
        }
        vec
    }

    async fn embed_openai(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        text: &str,
    ) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a str,
        }

        let resp = self
            .client
            .post(api_url)
            .bearer_auth(api_key)
            .json(&Req { model, input: text })
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("OpenAI API error: {}", resp.text().await?));
        }

        #[derive(Deserialize)]
        struct Resp {
            data: Vec<Data>,
        }
        #[derive(Deserialize)]
        struct Data {
            embedding: Vec<f32>,
        }

        let mut parsed: Resp = resp.json().await?;
        if parsed.data.is_empty() {
            return Err(anyhow::anyhow!("No embeddings returned from OpenAI"));
        }
        let embedding = parsed.data.remove(0).embedding;
        self.check_dim(embedding.len())?;
        Ok(embedding)
    }

    /// OpenAI batch — splits into chunks of at most 96 texts to stay within API limits.
    async fn embed_batch_openai(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        const OPENAI_BATCH_LIMIT: usize = 96;

        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [String],
        }
        #[derive(Deserialize)]
        struct Resp {
            data: Vec<Data>,
        }
        #[derive(Deserialize)]
        struct Data {
            embedding: Vec<f32>,
            index: usize,
        }

        let mut all_vecs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(OPENAI_BATCH_LIMIT) {
            let _permit = if let Some(sem) = &self.inflight_semaphore {
                Some(
                    sem.acquire()
                        .await
                        .map_err(|e| anyhow!("semaphore closed: {e}"))?,
                )
            } else {
                None
            };
            let resp = self
                .client
                .post(api_url)
                .bearer_auth(api_key)
                .json(&Req {
                    model,
                    input: chunk,
                })
                .send()
                .await?;

            if !resp.status().is_success() {
                return Err(anyhow::anyhow!("OpenAI API error: {}", resp.text().await?));
            }

            let parsed: Resp = resp.json().await?;
            if parsed.data.is_empty() {
                return Err(anyhow!("No embeddings returned from OpenAI"));
            }
            let rows: Vec<(usize, Vec<f32>)> = parsed
                .data
                .into_iter()
                .map(|row| (row.index, row.embedding))
                .collect();
            let ordered = reorder_openai_embeddings(rows, chunk.len())?;
            self.check_dims(&ordered)?;
            all_vecs.extend(ordered);
        }

        Ok(all_vecs)
    }

    async fn embed_ollama(&self, api_url: &str, model: &str, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            prompt: &'a str,
        }

        let resp = self
            .client
            .post(format!("{}/api/embeddings", api_url.trim_end_matches('/')))
            .json(&Req {
                model,
                prompt: text,
            })
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!("Ollama API error: {}", resp.text().await?));
        }

        #[derive(Deserialize)]
        struct Resp {
            embedding: Vec<f32>,
        }

        let parsed: Resp = resp.json().await?;
        self.check_dim(parsed.embedding.len())?;
        Ok(parsed.embedding)
    }

    /// PR3: Ollama doesn't support native batch — parallelize individual requests.
    async fn embed_batch_ollama(
        &self,
        api_url: &str,
        model: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        use futures_util::stream::{self, StreamExt, TryStreamExt};

        let rows: Vec<(usize, Vec<f32>)> = stream::iter(texts.iter().cloned().enumerate())
            .map(|(index, text)| async move {
                self.embed_ollama(api_url, model, &text)
                    .await
                    .map(|embedding| (index, embedding))
            })
            .buffer_unordered(self.max_inflight_requests)
            .try_collect()
            .await?;
        reorder_openai_embeddings(rows, texts.len())
    }

    fn cache_key(&self, text: &str) -> String {
        format!("{}::{}::{text}", self.provider_cache_namespace(), self.dim)
    }

    fn provider_cache_namespace(&self) -> String {
        match &self.provider {
            EmbeddingProvider::OpenAI { api_url, model, .. } => {
                format!("openai::{api_url}::{model}")
            }
            EmbeddingProvider::AzureOpenAI {
                api_base,
                deployment,
                ..
            } => {
                format!("azure::{api_base}::{deployment}")
            }
            EmbeddingProvider::Cohere { api_url, model, .. } => {
                format!("cohere::{api_url}::{model}")
            }
            EmbeddingProvider::HuggingFace { api_url, model, .. } => {
                format!("huggingface::{api_url}::{model}")
            }
            EmbeddingProvider::Ollama { api_url, model } => {
                format!("ollama::{api_url}::{model}")
            }
            EmbeddingProvider::Mock { dim } => format!("mock::{dim}"),
            EmbeddingProvider::None => "none".to_string(),
        }
    }

    // ── Azure OpenAI ─────────────────────────────────────────────────────────

    async fn embed_azure(
        &self,
        api_base: &str,
        deployment: &str,
        api_key: &str,
        api_version: &str,
        text: &str,
    ) -> Result<Vec<f32>> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            api_base.trim_end_matches('/'),
            deployment,
            api_version
        );
        #[derive(Serialize)]
        struct Req<'a> {
            input: &'a str,
        }
        let resp = self
            .client
            .post(&url)
            .header("api-key", api_key)
            .json(&Req { input: text })
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(anyhow!("Azure OpenAI error: {}", resp.text().await?));
        }
        #[derive(Deserialize)]
        struct Resp {
            data: Vec<Data>,
        }
        #[derive(Deserialize)]
        struct Data {
            embedding: Vec<f32>,
        }
        let mut parsed: Resp = resp.json().await?;
        let embedding = parsed
            .data
            .pop()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow!("Azure OpenAI returned no embeddings"))?;
        self.check_dim(embedding.len())?;
        Ok(embedding)
    }

    async fn embed_batch_azure(
        &self,
        api_base: &str,
        deployment: &str,
        api_key: &str,
        api_version: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            api_base.trim_end_matches('/'),
            deployment,
            api_version
        );
        const AZURE_BATCH_LIMIT: usize = 96;
        #[derive(Serialize)]
        struct Req<'a> {
            input: &'a [String],
        }
        #[derive(Deserialize)]
        struct Resp {
            data: Vec<Data>,
        }
        #[derive(Deserialize)]
        struct Data {
            embedding: Vec<f32>,
            index: usize,
        }

        let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(AZURE_BATCH_LIMIT) {
            let _permit = if let Some(sem) = &self.inflight_semaphore {
                Some(
                    sem.acquire()
                        .await
                        .map_err(|e| anyhow!("semaphore closed: {e}"))?,
                )
            } else {
                None
            };
            let resp = self
                .client
                .post(&url)
                .header("api-key", api_key)
                .json(&Req { input: chunk })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!("Azure OpenAI batch error: {}", resp.text().await?));
            }
            let parsed: Resp = resp.json().await?;
            let rows: Vec<(usize, Vec<f32>)> = parsed
                .data
                .into_iter()
                .map(|d| (d.index, d.embedding))
                .collect();
            let ordered = reorder_openai_embeddings(rows, chunk.len())?;
            self.check_dims(&ordered)?;
            all.extend(ordered);
        }
        Ok(all)
    }

    // ── Cohere ───────────────────────────────────────────────────────────────

    async fn embed_cohere_single(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        input_type: &str,
        text: &str,
    ) -> Result<Vec<f32>> {
        let vecs = self
            .embed_batch_cohere(api_url, api_key, model, input_type, &[text.to_string()])
            .await?;
        vecs.into_iter()
            .next()
            .ok_or_else(|| anyhow!("Cohere returned no embeddings"))
    }

    async fn embed_batch_cohere(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        input_type: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        const COHERE_BATCH_LIMIT: usize = 96;
        let url = format!("{}/v1/embed", api_url.trim_end_matches('/'));
        #[derive(Serialize)]
        struct Req<'a> {
            texts: &'a [String],
            model: &'a str,
            input_type: &'a str,
        }
        #[derive(Deserialize)]
        struct Resp {
            embeddings: Vec<Vec<f32>>,
        }

        let mut all: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(COHERE_BATCH_LIMIT) {
            let _permit = if let Some(sem) = &self.inflight_semaphore {
                Some(
                    sem.acquire()
                        .await
                        .map_err(|e| anyhow!("semaphore closed: {e}"))?,
                )
            } else {
                None
            };
            let resp = self
                .client
                .post(&url)
                .bearer_auth(api_key)
                .json(&Req {
                    texts: chunk,
                    model,
                    input_type,
                })
                .send()
                .await?;
            if !resp.status().is_success() {
                return Err(anyhow!("Cohere error: {}", resp.text().await?));
            }
            let parsed: Resp = resp.json().await?;
            if parsed.embeddings.len() != chunk.len() {
                return Err(anyhow!(
                    "Cohere batch response size mismatch: expected {}, got {}",
                    chunk.len(),
                    parsed.embeddings.len()
                ));
            }
            all.extend(parsed.embeddings);
        }
        self.check_dims(&all)?;
        Ok(all)
    }

    // ── HuggingFace Inference API ─────────────────────────────────────────────

    async fn embed_huggingface_single(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        text: &str,
    ) -> Result<Vec<f32>> {
        let vecs = self
            .embed_batch_huggingface(api_url, api_key, model, &[text.to_string()])
            .await?;
        vecs.into_iter()
            .next()
            .ok_or_else(|| anyhow!("HuggingFace returned no embeddings"))
    }

    async fn embed_batch_huggingface(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let url = format!(
            "{}/pipeline/feature-extraction/{}",
            api_url.trim_end_matches('/'),
            model
        );
        let _permit = if let Some(sem) = &self.inflight_semaphore {
            Some(
                sem.acquire()
                    .await
                    .map_err(|e| anyhow!("semaphore closed: {e}"))?,
            )
        } else {
            None
        };
        let resp = self
            .client
            .post(&url)
            .bearer_auth(api_key)
            .json(texts)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(anyhow!("HuggingFace error: {}", resp.text().await?));
        }
        // HF returns [[f32]] for batch or [f32] for single
        let body: serde_json::Value = resp.json().await?;
        if let Some(arr) = body.as_array() {
            if arr.is_empty() {
                return Ok(Vec::new());
            }
            if arr[0].is_array() {
                // batch: [[f32]]
                let vecs = arr
                    .iter()
                    .map(|row| {
                        serde_json::from_value::<Vec<f32>>(row.clone())
                            .map_err(|e| anyhow!("HuggingFace parse error: {e}"))
                    })
                    .collect::<Result<Vec<Vec<f32>>>>()?;
                self.check_dims(&vecs)?;
                Ok(vecs)
            } else {
                // single text returned as [f32]
                let vec = serde_json::from_value::<Vec<f32>>(body)
                    .map_err(|e| anyhow!("HuggingFace parse error: {e}"))?;
                self.check_dim(vec.len())?;
                Ok(vec![vec])
            }
        } else {
            Err(anyhow!("HuggingFace returned unexpected response format"))
        }
    }
}

fn cache_lookup(cache: &mut LruCache<String, Vec<f32>>, key: &str) -> Option<Vec<f32>> {
    cache.get(key).cloned()
}

fn cache_store(cache: &mut LruCache<String, Vec<f32>>, key: String, embedding: Vec<f32>) {
    cache.put(key, embedding);
}

fn reorder_openai_embeddings(
    data: Vec<(usize, Vec<f32>)>,
    expected_len: usize,
) -> Result<Vec<Vec<f32>>> {
    if data.len() != expected_len {
        return Err(anyhow!(
            "OpenAI batch response size mismatch: expected {}, got {}",
            expected_len,
            data.len()
        ));
    }

    let mut ordered = vec![None; expected_len];
    for (index, embedding) in data {
        if index >= expected_len {
            return Err(anyhow!(
                "OpenAI batch response index {} out of range for {} inputs",
                index,
                expected_len
            ));
        }
        if ordered[index].is_some() {
            return Err(anyhow!(
                "OpenAI batch response contains duplicate index {}",
                index
            ));
        }
        ordered[index] = Some(embedding);
    }

    ordered
        .into_iter()
        .enumerate()
        .map(|(idx, item)| {
            item.ok_or_else(|| anyhow!("OpenAI batch response missing index {}", idx))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{reorder_openai_embeddings, EmbeddingClient, EmbeddingProvider};

    #[test]
    fn reorder_openai_embeddings_restores_order() {
        let data = vec![(1, vec![2.0]), (0, vec![1.0])];

        let ordered = reorder_openai_embeddings(data, 2).unwrap();
        assert_eq!(ordered, vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn reorder_openai_embeddings_rejects_missing_rows() {
        let err = reorder_openai_embeddings(vec![(0, vec![1.0])], 2).unwrap_err();

        assert!(err.to_string().contains("size mismatch"));
    }

    #[test]
    fn check_dim_rejects_mismatched_width() {
        let client =
            EmbeddingClient::with_limits_and_dim(EmbeddingProvider::Mock { dim: 4 }, 0, 4, None, 4);
        // Exact match passes.
        assert!(client.check_dim(4).is_ok());
        assert!(client.check_dims(&[vec![0.0; 4], vec![1.0; 4]]).is_ok());
        // Wrong width is rejected.
        let err = client.check_dim(3).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
        assert!(client.check_dims(&[vec![0.0; 4], vec![0.0; 3]]).is_err());
    }

    #[test]
    fn check_dim_disabled_when_dim_zero() {
        // dim == 0 means "unspecified": any width is accepted.
        let client = EmbeddingClient::new(EmbeddingProvider::Mock { dim: 8 });
        assert!(client.check_dim(8).is_ok());
        assert!(client.check_dim(123).is_ok());
    }

    #[tokio::test]
    async fn embed_batch_with_cache_size_zero_still_returns_all_vectors() {
        let client = EmbeddingClient::with_cache(EmbeddingProvider::Mock { dim: 4 }, 0, None);
        let texts = vec!["alpha".to_string(), "beta".to_string(), "alpha".to_string()];

        let vectors = client.embed_batch(&texts).await.unwrap();

        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vectors[2]);
        assert_ne!(vectors[0], vectors[1]);
    }
}
