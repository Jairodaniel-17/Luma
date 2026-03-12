use anyhow::Result;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Mutex;

type EmbedCache = std::sync::Arc<Mutex<LruCache<u64, Vec<f32>>>>;

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

/// FNV-1a 64-bit hash — fast, no external dependency.
fn fnv1a_hash(text: &str) -> u64 {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    let mut hash = FNV_OFFSET;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[derive(Clone)]
pub struct EmbeddingClient {
    provider: EmbeddingProvider,
    client: reqwest::Client,
    /// Optional LRU cache: key = FNV-1a hash of input text.
    /// `None` means caching is disabled.
    cache: Option<EmbedCache>,
    /// Metrics handle for tracking cache hits/misses (optional).
    metrics: Option<std::sync::Arc<crate::engine::metrics::Metrics>>,
}

impl Default for EmbeddingClient {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::None,
            client: reqwest::Client::new(),
            cache: None,
            metrics: None,
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
        }
    }

    /// Create a client with an LRU embedding cache.
    /// `cache_size = 0` disables the cache.
    pub fn with_cache(
        provider: EmbeddingProvider,
        cache_size: usize,
        metrics: Option<std::sync::Arc<crate::engine::metrics::Metrics>>,
    ) -> Self {
        let cache: Option<EmbedCache> = if cache_size > 0 {
            NonZeroUsize::new(cache_size).map(|n| std::sync::Arc::new(Mutex::new(LruCache::new(n))))
        } else {
            None
        };
        Self {
            provider,
            client: reqwest::Client::new(),
            cache,
            metrics,
        }
    }

    /// Embed a single text string. Uses the LRU cache when enabled.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cache) = &self.cache {
            let key = fnv1a_hash(text);
            {
                let mut guard = cache.lock().unwrap();
                if let Some(cached) = guard.get(&key) {
                    if let Some(m) = &self.metrics {
                        m.inc_embed_cache_hit();
                    }
                    return Ok(cached.clone());
                }
            }
            if let Some(m) = &self.metrics {
                m.inc_embed_cache_miss();
            }
            let vec = self.embed_uncached(text).await?;
            cache.lock().unwrap().put(key, vec.clone());
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
                let mut guard = cache.lock().unwrap();
                for (i, text) in texts.iter().enumerate() {
                    let key = fnv1a_hash(text);
                    if let Some(cached) = guard.get(&key) {
                        if let Some(m) = &self.metrics {
                            m.inc_embed_cache_hit();
                        }
                        result[i] = Some(cached.clone());
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

            // Update cache and fill result
            {
                let mut guard = cache.lock().unwrap();
                for (&idx, vec) in missing_indices.iter().zip(missing_vecs.iter()) {
                    if let Some(m) = &self.metrics {
                        m.inc_embed_cache_miss();
                    }
                    let key = fnv1a_hash(&texts[idx]);
                    guard.put(key, vec.clone());
                    result[idx] = Some(vec.clone());
                }
            }

            return Ok(result.into_iter().map(|v| v.unwrap()).collect());
        }

        self.embed_batch_uncached(texts).await
    }

    async fn embed_uncached(&self, text: &str) -> Result<Vec<f32>> {
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

    /// PR3: OpenAI batch — `input` accepts an array, one HTTP request for N texts.
    async fn embed_batch_openai(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [String],
        }

        let resp = self
            .client
            .post(api_url)
            .bearer_auth(api_key)
            .json(&Req {
                model,
                input: texts,
            })
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
            index: usize,
        }

        let mut parsed: Resp = resp.json().await?;
        if parsed.data.is_empty() {
            return Err(anyhow::anyhow!("No embeddings returned from OpenAI"));
        }
        // OpenAI returns data sorted by index — ensure correct ordering
        parsed.data.sort_by_key(|d| d.index);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
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
        let futures: Vec<_> = texts
            .iter()
            .map(|text| self.embed_ollama(api_url, model, text))
            .collect();
        futures_util::future::try_join_all(futures).await
    }
}
