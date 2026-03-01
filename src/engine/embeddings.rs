use anyhow::Result;
use serde::{Deserialize, Serialize};

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
}

impl Default for EmbeddingClient {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::None,
            client: reqwest::Client::new(),
        }
    }
}

impl EmbeddingClient {
    pub fn new(provider: EmbeddingProvider) -> Self {
        Self {
            provider,
            client: reqwest::Client::new(),
        }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
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

    fn embed_mock(&self, dim: usize, text: &str) -> Vec<f32> {
        // Simple deterministic vector for tests
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
}
