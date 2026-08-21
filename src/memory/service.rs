use crate::config::Config;
use crate::engine::embeddings::EmbeddingHandle;
use crate::engine::Engine;
use crate::memory::consolidator::Consolidator;
use crate::memory::graph::GraphService;
use crate::memory::llm::{InferenceClient, InferenceProvider};
use crate::sqlite::memory_schema::ensure_memory_schema;
use crate::sqlite::SqliteService;
use std::sync::Arc;
use tokio::sync::OnceCell;

#[derive(Clone)]
pub struct MemoryService {
    pub(crate) engine: Arc<Engine>,
    pub(crate) sqlite: Option<Arc<SqliteService>>,
    pub(crate) embeddings: EmbeddingHandle,
    pub(crate) llm: InferenceClient,
    pub(crate) config: Config,
    pub(crate) schema_ready: Arc<OnceCell<()>>,
    pub(crate) consolidator: Consolidator,
    pub(crate) graph: Option<GraphService>,
}

impl MemoryService {
    pub fn new(
        engine: Arc<Engine>,
        sqlite: Option<Arc<SqliteService>>,
        embeddings: EmbeddingHandle,
        config: Config,
    ) -> Self {
        let graph = sqlite.as_ref().map(|sq| GraphService::new(sq.clone()));
        Self {
            engine,
            sqlite,
            consolidator: Consolidator,
            embeddings,
            llm: init_inference_client(&config),
            config,
            schema_ready: Arc::new(OnceCell::new()),
            graph,
        }
    }

    pub(crate) async fn ensure_schema(&self) -> anyhow::Result<()> {
        let Some(sqlite) = &self.sqlite else {
            anyhow::bail!("sqlite module is required for memory APIs");
        };
        self.schema_ready
            .get_or_try_init(|| async { ensure_memory_schema(sqlite).await })
            .await
            .map(|_| ())
    }

    pub(crate) fn default_limit(&self, limit: Option<usize>) -> usize {
        limit
            .unwrap_or(self.config.memory_default_limit)
            .clamp(1, self.config.memory_max_evidence.max(1))
    }

    pub(crate) fn working_ttl_ms(&self) -> u64 {
        self.config.memory_working_ttl_secs.saturating_mul(1000)
    }
}

fn init_inference_client(config: &Config) -> InferenceClient {
    let provider = match config.llm_provider.to_ascii_lowercase().as_str() {
        "openai" => InferenceProvider::OpenAI {
            api_url: config.llm_url.clone(),
            api_key: config.llm_api_key.clone(),
            model: config.llm_model.clone(),
        },
        "ollama" => InferenceProvider::Ollama {
            api_url: config.llm_url.clone(),
            model: config.llm_model.clone(),
        },
        "mock" => InferenceProvider::Mock,
        _ => InferenceProvider::None,
    };
    InferenceClient::new(provider)
}
