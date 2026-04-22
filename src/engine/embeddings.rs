use anyhow::{anyhow, Result};
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::Semaphore;

type EmbedCache = Arc<Mutex<LruCache<String, Vec<f32>>>>;

#[derive(Debug, Clone)]
pub enum EmbeddingProvider {
    OpenAI {
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
}

impl Default for EmbeddingClient {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::None,
            client: reqwest::Client::new(),
            cache: None,
            metrics: None,
            max_inflight_requests: 16,
            inflight_semaphore: None,
        }
    }
}

impl EmbeddingClient {
    pub fn new(provider: EmbeddingProvider) -> Self {
        Self {
            provider,
            client: reqwest::Client::new(),
            cache: None,
            metrics: None,
            max_inflight_requests: 16,
            inflight_semaphore: None,
        }
    }

    pub fn with_limits(
        provider: EmbeddingProvider,
        cache_size: usize,
        max_inflight_requests: usize,
        metrics: Option<Arc<crate::engine::metrics::Metrics>>,
    ) -> Self {
        let n = max_inflight_requests.max(1);
        let cache = NonZeroUsize::new(cache_size)
            .map(|cap| Arc::new(Mutex::new(LruCache::new(cap))));
        Self {
            provider,
            client: reqwest::Client::new(),
            cache,
            metrics,
            max_inflight_requests: n,
            inflight_semaphore: Some(Arc::new(Semaphore::new(n))),
        }
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
                return Ok(result.into_iter().map(|v| v.unwrap()).collect());
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

    async fn embed_uncached(&self, text: &str) -> Result<Vec<f32>> {
        let _permit = if let Some(sem) = &self.inflight_semaphore {
            Some(sem.acquire().await.map_err(|e| anyhow!("semaphore closed: {e}"))?)
        } else {
            None
        };
        match &self.provider {
            EmbeddingProvider::None => Err(anyhow::anyhow!("Embeddings are not configured")),
            EmbeddingProvider::OpenAI {
                api_url,
                api_key,
                model,
            } => self.embed_openai(api_url, api_key, model, text).await,
            EmbeddingProvider::Ollama { api_url, model } => {
                self.embed_ollama(api_url, model, text).await
            }
            EmbeddingProvider::Mock { dim } => Ok(self.embed_mock(*dim, text)),
        }
    }

    async fn embed_batch_uncached(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
            EmbeddingProvider::Ollama { api_url, model } => {
                self.embed_batch_ollama(api_url, model, texts).await
            }
            EmbeddingProvider::Mock { dim } => {
                Ok(texts.iter().map(|t| self.embed_mock(*dim, t)).collect())
            }
        }
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
        Ok(parsed.data.remove(0).embedding)
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
                Some(sem.acquire().await.map_err(|e| anyhow!("semaphore closed: {e}"))?)
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
        format!("{}::{text}", self.provider_cache_namespace())
    }

    fn provider_cache_namespace(&self) -> String {
        match &self.provider {
            EmbeddingProvider::OpenAI { api_url, model, .. } => {
                format!("openai::{api_url}::{model}")
            }
            EmbeddingProvider::Ollama { api_url, model } => {
                format!("ollama::{api_url}::{model}")
            }
            EmbeddingProvider::Mock { dim } => format!("mock::{dim}"),
            EmbeddingProvider::None => "none".to_string(),
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
