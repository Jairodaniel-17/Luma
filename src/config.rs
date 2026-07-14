use serde::{Deserialize, Serialize};
use std::net::IpAddr;

pub mod resolve;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub bind_addr: IpAddr,
    /// Secret: never persisted to luma.toml. Loaded from env at runtime.
    #[serde(default, skip_serializing)]
    pub api_key: String,
    pub data_dir: Option<String>,
    pub snapshot_interval_secs: u64,
    pub event_buffer_size: usize,
    pub live_broadcast_capacity: usize,
    pub wal_segment_max_bytes: u64,
    pub wal_retention_segments: usize,
    pub request_timeout_secs: u64,
    pub max_body_bytes: usize,
    pub max_key_len: usize,
    pub max_collection_len: usize,
    pub max_id_len: usize,
    pub max_vector_dim: usize,
    pub max_k: usize,
    pub max_json_bytes: usize,
    pub max_state_batch: usize,
    pub max_vector_batch: usize,
    pub max_doc_find: usize,
    pub cors_allowed_origins: Option<String>,
    pub sqlite_enabled: bool,
    pub sqlite_path: Option<String>,
    pub search_threads: usize,
    pub parallel_probe: bool,
    pub parallel_probe_min_segments: usize,
    pub simd_enabled: bool,
    pub index_kind: String,
    pub ivf_clusters: usize,
    pub ivf_nprobe: usize,
    pub ivf_training_sample: usize,
    pub ivf_min_train_vectors: usize,
    pub ivf_retrain_min_deltas: usize,
    pub q8_refine_topk: usize,
    pub diskann_max_degree: usize,
    pub diskann_build_threads: usize,
    pub diskann_search_list_size: usize,
    pub run_target_bytes: u64,
    pub run_retention: usize,
    pub compaction_trigger_tombstone_ratio: f32,
    pub compaction_max_bytes_per_pass: u64,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_url: String,
    /// Secret: never persisted to luma.toml. Loaded from env at runtime.
    #[serde(default, skip_serializing)]
    pub embedding_api_key: String,
    pub llm_provider: String,
    pub llm_model: String,
    pub llm_url: String,
    /// Secret: never persisted to luma.toml. Loaded from env at runtime.
    #[serde(default, skip_serializing)]
    pub llm_api_key: String,
    pub embedding_dim: usize,
    /// PR2: LRU embedding cache size (0 = disabled). Default 10_000.
    pub embedding_cache_size: usize,
    /// Maximum concurrent provider calls when native batching is unavailable.
    pub embedding_max_inflight_requests: usize,
    pub hub_ingest_max_inflight: usize,
    pub hub_hydration_max_inflight: usize,
    pub hub_sql_prefilter_max_candidates: usize,
    pub hub_sql_prefilter_selectivity_threshold: f32,
    pub hub_vector_first_candidate_multiplier: usize,
    pub memory_consolidation_enabled: bool,
    pub memory_working_ttl_secs: u64,
    pub memory_default_limit: usize,
    pub memory_max_evidence: usize,
    pub memory_procedural_max_nodes: usize,
    pub memory_fact_promotion_threshold: f32,
    /// Max BFS hops in semantic walk (default 2).
    pub memory_walk_max_hops: usize,
    /// Cosine similarity cutoff for semantic walk pruning (default 0.65).
    pub memory_walk_min_similarity: f32,
    /// Max nodes to explore per semantic walk query (default 40).
    pub memory_walk_max_nodes: usize,
    /// Enable PageRank centrality scoring in semantic walk (default true).
    pub memory_centrality_enabled: bool,
    /// How often to recompute centrality scores in seconds (default 300).
    pub memory_centrality_update_interval_secs: u64,
    /// Enable exponential decay of semantic facts (default false).
    pub memory_decay_enabled: bool,
    /// Half-life in days for semantic fact decay (default 30.0).
    pub memory_decay_half_life_days: f64,
    /// Archive facts whose decay_score drops below this threshold (default 0.1).
    pub memory_decay_archive_threshold: f32,
    /// How often to run the decay background task in seconds (default 3600).
    pub memory_decay_interval_secs: u64,
    /// PR4: HNSW M parameter (connections per node). Default 16.
    pub hnsw_m: usize,
    /// PR4: HNSW ef_construction parameter. Default 200.
    pub hnsw_ef_construction: usize,
    /// Background HNSW segment compaction is opt-in because it trades write latency for cleanup.
    pub hnsw_segment_compaction_enabled: bool,
    /// Tombstone ratio threshold for in-memory HNSW segment rebuilds.
    pub hnsw_segment_compaction_threshold: f32,
    /// Interval for checking HNSW compaction candidates.
    pub hnsw_segment_compaction_interval_secs: u64,
    /// PR5: WAL sync mode: "per_write" (fsync each write) or "group" (buffered flush).
    pub wal_sync_mode: String,
    /// PR5: Group commit flush interval in ms. Default 10.
    pub wal_flush_interval_ms: u64,
    /// PR5: Group commit batch size (flush after N events). Default 64.
    pub wal_batch_size: usize,
    /// Process role: "node" (default, runs the engine) or "router" (forwards
    /// requests to `router_nodes` by namespace-sharding). See `crate::router`.
    #[serde(default = "default_role")]
    pub role: String,
    /// Backend node base URLs when `role == "router"` (e.g. http://10.0.0.1:1234).
    #[serde(default)]
    pub router_nodes: Vec<String>,
    /// Maximum vectors per collection (0 = unlimited). Applies to add and upsert.
    pub max_collection_vectors: usize,
    /// Emit a tracing::warn! for vector searches that exceed this threshold in ms (0 = disabled).
    pub slow_query_threshold_ms: u64,
    /// Max filtered candidates to use brute-force pre-search instead of HNSW + post-filter.
    /// Default 10000. Higher values trade accuracy for speed on filtered corpora.
    pub pre_filter_threshold: usize,
    /// Max embedding retry attempts for transient provider errors (default 3, min 1).
    pub embedding_retry_attempts: u32,
    /// Initial backoff delay in ms for embedding retries (default 200).
    pub embedding_retry_initial_ms: u64,
    /// Azure OpenAI: base URL (e.g. https://my.openai.azure.com).
    pub embedding_azure_api_base: String,
    /// Azure OpenAI: deployment name.
    pub embedding_azure_deployment: String,
    /// Azure OpenAI: API version (e.g. 2024-02-01).
    pub embedding_azure_api_version: String,
    /// Cohere embed input_type: "search_document" (upsert) or "search_query" (query).
    pub embedding_cohere_input_type: String,
    /// Max IDs fetched by sql_filter pre-scan (prevents OOM on large collections). Default 50_000.
    pub hub_sql_filter_max_ids: usize,
    /// Path to TLS certificate (PEM). Both cert+key required to enable TLS.
    pub tls_cert_path: Option<String>,
    /// Path to TLS private key (PEM).
    pub tls_key_path: Option<String>,
    /// Rate limit: max requests per second per IP address (0 = disabled).
    pub rate_limit_rps: u32,
    /// Rate limit burst size (default 10× rate_limit_rps).
    pub rate_limit_burst: u32,
    /// libSQL/Turso remote URL (e.g. https://db-name.turso.io).
    /// When set, Luma routes all SQL through the Hrana HTTP protocol instead of local SQLite.
    /// Enables active-active HA via Turso's global replication — zero code changes required.
    pub libsql_url: Option<String>,
    /// Auth token for libSQL/Turso remote database (Bearer token).
    /// Secret: never persisted to luma.toml. Loaded from env at runtime.
    #[serde(default, skip_serializing)]
    pub libsql_auth_token: String,
    /// Enable the periodic background backup task (default false).
    #[serde(default)]
    pub backup_enabled: bool,
    /// Directory where timestamped backups are written (default "backups").
    #[serde(default = "default_backup_dir")]
    pub backup_dir: String,
    /// Interval between automatic backups, in seconds (default 86400 = daily).
    #[serde(default = "default_backup_interval_secs")]
    pub backup_interval_secs: u64,
    /// Number of most-recent backups to retain; older ones are pruned (default 7).
    #[serde(default = "default_backup_retention")]
    pub backup_retention: usize,
}

fn default_backup_dir() -> String {
    "backups".to_string()
}
fn default_backup_interval_secs() -> u64 {
    86_400
}
fn default_backup_retention() -> usize {
    7
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 8080,
            bind_addr: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
            api_key: "dev".to_string(),
            data_dir: Some("data".to_string()),
            snapshot_interval_secs: 30,
            event_buffer_size: 10_000,
            live_broadcast_capacity: 4096,
            wal_segment_max_bytes: 64 * 1024 * 1024,
            wal_retention_segments: 10,
            request_timeout_secs: 30,
            max_body_bytes: 10 * 1024 * 1024,
            max_key_len: 256,
            max_collection_len: 64,
            max_id_len: 128,
            max_vector_dim: 1536,
            max_k: 100,
            max_json_bytes: 1024 * 1024,
            max_state_batch: 256,
            max_vector_batch: 256,
            max_doc_find: 100,
            cors_allowed_origins: None,
            sqlite_enabled: true,
            sqlite_path: None,
            search_threads: 0,
            parallel_probe: true,
            parallel_probe_min_segments: 4,
            simd_enabled: true,
            index_kind: "IVF_FLAT_Q8".to_string(),
            ivf_clusters: 4096,
            ivf_nprobe: 16,
            ivf_training_sample: 200_000,
            ivf_min_train_vectors: 1_024,
            ivf_retrain_min_deltas: 50_000,
            q8_refine_topk: 512,
            diskann_max_degree: 48,
            diskann_build_threads: 1,
            diskann_search_list_size: 64,
            run_target_bytes: 134_217_728,
            run_retention: 8,
            compaction_trigger_tombstone_ratio: 0.2,
            compaction_max_bytes_per_pass: 1_073_741_824,
            embedding_provider: "none".to_string(),
            embedding_model: "".to_string(),
            embedding_url: "".to_string(),
            embedding_api_key: "".to_string(),
            llm_provider: "none".to_string(),
            llm_model: "".to_string(),
            llm_url: "".to_string(),
            llm_api_key: "".to_string(),
            embedding_dim: 384,
            embedding_cache_size: 10_000,
            embedding_max_inflight_requests: 16,
            hub_ingest_max_inflight: 32,
            hub_hydration_max_inflight: 32,
            hub_sql_prefilter_max_candidates: 20_000,
            hub_sql_prefilter_selectivity_threshold: 0.2,
            hub_vector_first_candidate_multiplier: 8,
            memory_consolidation_enabled: false,
            memory_working_ttl_secs: 3600,
            memory_default_limit: 10,
            memory_max_evidence: 10,
            memory_procedural_max_nodes: 128,
            memory_fact_promotion_threshold: 0.85,
            memory_walk_max_hops: 2,
            memory_walk_min_similarity: 0.65,
            memory_walk_max_nodes: 40,
            memory_centrality_enabled: true,
            memory_centrality_update_interval_secs: 300,
            memory_decay_enabled: false,
            memory_decay_half_life_days: 30.0,
            memory_decay_archive_threshold: 0.1,
            memory_decay_interval_secs: 3600,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_segment_compaction_enabled: false,
            hnsw_segment_compaction_threshold: 0.35,
            hnsw_segment_compaction_interval_secs: 300,
            // Group commit by default: the WAL fsync is amortized across a batch
            // instead of paid on every write, which is the dominant single-node
            // write bottleneck. Durability of acked writes is preserved because
            // the state store (redb) and vector segments still fsync immediately,
            // so a lost WAL-buffer tail is always recoverable from them on replay.
            // Set WAL_SYNC_MODE=per_write for a fully synchronous WAL.
            wal_sync_mode: "group".to_string(),
            wal_flush_interval_ms: 10,
            wal_batch_size: 64,
            role: default_role(),
            router_nodes: Vec::new(),
            max_collection_vectors: 0,
            slow_query_threshold_ms: 0,
            pre_filter_threshold: 10_000,
            embedding_retry_attempts: 3,
            embedding_retry_initial_ms: 200,
            embedding_azure_api_base: String::new(),
            embedding_azure_deployment: String::new(),
            embedding_azure_api_version: "2024-02-01".to_string(),
            embedding_cohere_input_type: "search_document".to_string(),
            hub_sql_filter_max_ids: 50_000,
            tls_cert_path: None,
            tls_key_path: None,
            // Rate limiting on by default for brute-force protection; set to 0 to disable.
            rate_limit_rps: 100,
            rate_limit_burst: 0,
            libsql_url: None,
            libsql_auth_token: String::new(),
            backup_enabled: false,
            backup_dir: default_backup_dir(),
            backup_interval_secs: default_backup_interval_secs(),
            backup_retention: default_backup_retention(),
        }
    }
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let path = std::path::Path::new("luma.toml");
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let mut config: Config = toml::from_str(&content)?;
            // Secrets are never written to luma.toml (see `#[serde(skip_serializing)]`).
            // Overlay them from the environment so they come from env at runtime only.
            config.overlay_secrets_from_env();
            return Ok(config);
        }

        let config = Self::from_env()?;
        if let Err(e) = config.save() {
            tracing::warn!("Could not automatically generate luma.toml: {}", e);
        } else {
            tracing::info!("Auto-generated default luma.toml configuration file.");
        }
        Ok(config)
    }

    /// Overlay secret fields from the environment, overwriting any values loaded
    /// from `luma.toml`. Uses the same env lookups as [`Config::from_env`]. A
    /// secret is only overwritten when its env var is actually set, so an unset
    /// var leaves the (empty) deserialized default in place.
    fn overlay_secrets_from_env(&mut self) {
        if let Ok(v) = std::env::var("LUMA_API_KEY").or_else(|_| std::env::var("API_KEY")) {
            self.api_key = v;
        }
        if let Ok(v) = std::env::var("EMBEDDING_API_KEY") {
            self.embedding_api_key = v;
        }
        if let Ok(v) = std::env::var("LLM_API_KEY") {
            self.llm_api_key = v;
        }
        if let Ok(v) = std::env::var("LIBSQL_AUTH_TOKEN") {
            self.libsql_auth_token = v;
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write("luma.toml", content)?;
        Ok(())
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let port = resolve_port();
        let bind_addr = resolve_bind_addr();

        let api_key = std::env::var("LUMA_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .unwrap_or_else(|_| "".to_string());
        let data_dir = resolve_data_dir();

        let snapshot_interval_secs = std::env::var("SNAPSHOT_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let event_buffer_size = std::env::var("EVENT_BUFFER_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);

        let live_broadcast_capacity = std::env::var("LIVE_BROADCAST_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        let wal_segment_max_bytes = std::env::var("WAL_SEGMENT_MAX_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 * 1024 * 1024);

        let wal_retention_segments = resolve::resolve_wal_retention_segments();
        let request_timeout_secs = resolve::resolve_request_timeout_secs();
        let max_body_bytes = resolve::resolve_max_body_mb();
        let max_key_len = resolve::resolve_max_key_len();
        let max_collection_len = resolve::resolve_max_collection_len();

        let max_id_len = std::env::var("MAX_ID_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);

        let max_vector_dim = resolve::resolve_max_vector_dim();
        let max_k = resolve::resolve_max_k();
        let max_json_bytes = resolve::resolve_max_json_mb();

        let max_state_batch = std::env::var("MAX_STATE_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        let max_vector_batch = std::env::var("MAX_VECTOR_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        let max_doc_find = std::env::var("MAX_DOC_FIND")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);

        let search_threads = std::env::var("SEARCH_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let parallel_probe = parse_env_bool("PARALLEL_PROBE", true);

        let parallel_probe_min_segments = std::env::var("PARALLEL_PROBE_MIN_SEGMENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        let simd_enabled = parse_env_bool("SIMD_ENABLED", true);

        let index_kind = std::env::var("INDEX_KIND").unwrap_or_else(|_| "IVF_FLAT_Q8".to_string());

        let ivf_clusters = std::env::var("IVF_CLUSTERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        let ivf_nprobe = std::env::var("IVF_NPROBE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);

        let ivf_training_sample = std::env::var("IVF_TRAINING_SAMPLE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);
        let ivf_min_train_vectors = std::env::var("IVF_MIN_TRAIN_VECTORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_024);
        let ivf_retrain_min_deltas = std::env::var("IVF_RETRAIN_MIN_DELTAS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);
        let q8_refine_topk = std::env::var("Q8_REFINE_TOPK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let diskann_max_degree = std::env::var("DISKANN_MAX_DEGREE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48);
        let diskann_build_threads = std::env::var("DISKANN_BUILD_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1)
            });
        let diskann_search_list_size = std::env::var("DISKANN_SEARCH_LIST_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);

        let run_target_bytes = std::env::var("RUN_TARGET_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(134_217_728);

        let run_retention = std::env::var("RUN_RETENTION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);

        let compaction_trigger_tombstone_ratio =
            std::env::var("COMPACTION_TRIGGER_TOMBSTONE_RATIO")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2);

        let compaction_max_bytes_per_pass = std::env::var("COMPACTION_MAX_BYTES_PER_PASS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_073_741_824);

        let cors_allowed_origins = std::env::var("CORS_ALLOWED_ORIGINS").ok();
        let sqlite_enabled = resolve_sqlite_enabled();
        let sqlite_path = std::env::var("SQLITE_DB_PATH").ok();

        let embedding_provider =
            std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "none".to_string());
        let embedding_model = std::env::var("EMBEDDING_MODEL").unwrap_or_default();
        let embedding_url = std::env::var("EMBEDDING_URL").unwrap_or_default();
        let embedding_api_key = std::env::var("EMBEDDING_API_KEY").unwrap_or_default();
        let llm_provider = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "none".to_string());
        let llm_model = std::env::var("LLM_MODEL").unwrap_or_default();
        let llm_url = std::env::var("LLM_URL").unwrap_or_default();
        let llm_api_key = std::env::var("LLM_API_KEY").unwrap_or_default();
        let embedding_dim = std::env::var("EMBEDDING_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(384);
        let embedding_cache_size = std::env::var("EMBEDDING_CACHE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);
        let embedding_max_inflight_requests = std::env::var("EMBEDDING_MAX_INFLIGHT_REQUESTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let hub_ingest_max_inflight = std::env::var("HUB_INGEST_MAX_INFLIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let hub_hydration_max_inflight = std::env::var("HUB_HYDRATION_MAX_INFLIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let hub_sql_prefilter_max_candidates = std::env::var("HUB_SQL_PREFILTER_MAX_CANDIDATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20_000);
        let hub_sql_prefilter_selectivity_threshold =
            std::env::var("HUB_SQL_PREFILTER_SELECTIVITY_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2);
        let hub_vector_first_candidate_multiplier =
            std::env::var("HUB_VECTOR_FIRST_CANDIDATE_MULTIPLIER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
        let memory_consolidation_enabled = parse_env_bool("MEMORY_CONSOLIDATION_ENABLED", false);
        let memory_working_ttl_secs = std::env::var("MEMORY_WORKING_TTL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3600);
        let memory_default_limit = std::env::var("MEMORY_DEFAULT_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let memory_max_evidence = std::env::var("MEMORY_MAX_EVIDENCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let memory_procedural_max_nodes = std::env::var("MEMORY_PROCEDURAL_MAX_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let memory_fact_promotion_threshold = std::env::var("MEMORY_FACT_PROMOTION_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85);
        let memory_walk_max_hops = std::env::var("MEMORY_WALK_MAX_HOPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let memory_walk_min_similarity = std::env::var("MEMORY_WALK_MIN_SIMILARITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.65_f32);
        let memory_walk_max_nodes = std::env::var("MEMORY_WALK_MAX_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        let memory_centrality_enabled = parse_env_bool("MEMORY_CENTRALITY_ENABLED", true);
        let memory_centrality_update_interval_secs =
            std::env::var("MEMORY_CENTRALITY_UPDATE_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300);
        let memory_decay_enabled = parse_env_bool("MEMORY_DECAY_ENABLED", false);
        let memory_decay_half_life_days = std::env::var("MEMORY_DECAY_HALF_LIFE_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30.0_f64);
        let memory_decay_archive_threshold = std::env::var("MEMORY_DECAY_ARCHIVE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.1_f32);
        let memory_decay_interval_secs = std::env::var("MEMORY_DECAY_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3600_u64);
        let hnsw_m = std::env::var("HNSW_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let hnsw_ef_construction = std::env::var("HNSW_EF_CONSTRUCTION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let hnsw_segment_compaction_enabled =
            parse_env_bool("HNSW_SEGMENT_COMPACTION_ENABLED", false);
        let hnsw_segment_compaction_threshold = std::env::var("HNSW_SEGMENT_COMPACTION_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.35);
        let hnsw_segment_compaction_interval_secs =
            std::env::var("HNSW_SEGMENT_COMPACTION_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300);
        let wal_sync_mode =
            std::env::var("WAL_SYNC_MODE").unwrap_or_else(|_| "group".to_string());
        let wal_flush_interval_ms = std::env::var("WAL_FLUSH_INTERVAL_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let wal_batch_size = std::env::var("WAL_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let role = std::env::var("ROLE").unwrap_or_else(|_| default_role());
        let router_nodes = std::env::var("ROUTER_NODES")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            port,
            bind_addr,
            api_key,
            data_dir,
            snapshot_interval_secs,
            event_buffer_size,
            live_broadcast_capacity,
            wal_segment_max_bytes,
            wal_retention_segments,
            request_timeout_secs,
            max_body_bytes,
            max_key_len,
            max_collection_len,
            max_id_len,
            max_vector_dim,
            max_k,
            max_json_bytes,
            max_state_batch,
            max_vector_batch,
            max_doc_find,
            cors_allowed_origins,
            sqlite_enabled,
            sqlite_path,
            search_threads,
            parallel_probe,
            parallel_probe_min_segments,
            simd_enabled,
            index_kind,
            ivf_clusters,
            ivf_nprobe,
            ivf_training_sample,
            ivf_min_train_vectors,
            ivf_retrain_min_deltas,
            q8_refine_topk,
            diskann_max_degree,
            diskann_build_threads,
            diskann_search_list_size,
            run_target_bytes,
            run_retention,
            compaction_trigger_tombstone_ratio,
            compaction_max_bytes_per_pass,
            embedding_provider,
            embedding_model,
            embedding_url,
            embedding_api_key,
            llm_provider,
            llm_model,
            llm_url,
            llm_api_key,
            embedding_dim,
            embedding_cache_size,
            embedding_max_inflight_requests,
            hub_ingest_max_inflight,
            hub_hydration_max_inflight,
            hub_sql_prefilter_max_candidates,
            hub_sql_prefilter_selectivity_threshold,
            hub_vector_first_candidate_multiplier,
            memory_consolidation_enabled,
            memory_working_ttl_secs,
            memory_default_limit,
            memory_max_evidence,
            memory_procedural_max_nodes,
            memory_fact_promotion_threshold,
            memory_walk_max_hops,
            memory_walk_min_similarity,
            memory_walk_max_nodes,
            memory_centrality_enabled,
            memory_centrality_update_interval_secs,
            memory_decay_enabled,
            memory_decay_half_life_days,
            memory_decay_archive_threshold,
            memory_decay_interval_secs,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_segment_compaction_enabled,
            hnsw_segment_compaction_threshold,
            hnsw_segment_compaction_interval_secs,
            wal_sync_mode,
            wal_flush_interval_ms,
            wal_batch_size,
            role,
            router_nodes,
            max_collection_vectors: std::env::var("MAX_COLLECTION_VECTORS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            slow_query_threshold_ms: std::env::var("SLOW_QUERY_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            pre_filter_threshold: std::env::var("PRE_FILTER_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000),
            embedding_retry_attempts: std::env::var("EMBEDDING_RETRY_ATTEMPTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
            embedding_retry_initial_ms: std::env::var("EMBEDDING_RETRY_INITIAL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(200),
            embedding_azure_api_base: std::env::var("EMBEDDING_AZURE_API_BASE").unwrap_or_default(),
            embedding_azure_deployment: std::env::var("EMBEDDING_AZURE_DEPLOYMENT")
                .unwrap_or_default(),
            embedding_azure_api_version: std::env::var("EMBEDDING_AZURE_API_VERSION")
                .unwrap_or_else(|_| "2024-02-01".to_string()),
            embedding_cohere_input_type: std::env::var("EMBEDDING_COHERE_INPUT_TYPE")
                .unwrap_or_else(|_| "search_document".to_string()),
            hub_sql_filter_max_ids: std::env::var("HUB_SQL_FILTER_MAX_IDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50_000),
            tls_cert_path: std::env::var("TLS_CERT_PATH").ok(),
            tls_key_path: std::env::var("TLS_KEY_PATH").ok(),
            rate_limit_rps: std::env::var("RATE_LIMIT_RPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),
            rate_limit_burst: std::env::var("RATE_LIMIT_BURST")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            libsql_url: std::env::var("LIBSQL_URL").ok(),
            libsql_auth_token: std::env::var("LIBSQL_AUTH_TOKEN").unwrap_or_default(),
            backup_enabled: parse_env_bool("BACKUP_ENABLED", false),
            backup_dir: std::env::var("BACKUP_DIR").unwrap_or_else(|_| default_backup_dir()),
            backup_interval_secs: std::env::var("BACKUP_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(default_backup_interval_secs),
            backup_retention: std::env::var("BACKUP_RETENTION")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(default_backup_retention),
        })
    }
}

fn default_role() -> String {
    "node".to_string()
}

fn resolve_port() -> u16 {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--port" {
            if let Some(value) = args.next() {
                if let Ok(port) = value.parse::<u16>() {
                    return port;
                }
                eprintln!(
                    "Valor de puerto invalido `{value}` para --port. Cayendo a otras fuentes."
                );
            } else {
                eprintln!("`--port` requiere un valor. Cayendo a otras fuentes.");
            }
            continue;
        }
    }

    if let Ok(value) = std::env::var("PORT_LUMA_VDB") {
        if let Ok(port) = value.parse::<u16>() {
            return port;
        }
        eprintln!(
            "Valor de puerto invalido `{value}` para `PORT_LUMA_VDB`. Usando valor por defecto."
        );
    }

    1234
}

fn resolve_data_dir() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--data" || arg == "--data-dir" || arg == "--DATA_DIR" {
            if let Some(path) = args.next() {
                return Some(path);
            } else {
                eprintln!("`--data` requiere un valor. Ignorando flag.");
                // Fallback to default
            }
        }
    }
    std::env::var("DATA_DIR")
        .ok()
        .or_else(|| Some("./data".to_string()))
}

fn resolve_bind_addr() -> IpAddr {
    use std::net::{IpAddr, Ipv4Addr};
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--unsafe-bind" {
            eprintln!(
                "`--unsafe-bind` habilitado: exponiendo en 0.0.0.0. Usa un proxy/autenticación externa."
            );
            return IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0));
        }
        if arg == "--bind" || arg == "--host" {
            let Some(value) = args.next() else {
                eprintln!("`--bind` requiere un valor. Usando 127.0.0.1.");
                return IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
            };
            match value.parse::<IpAddr>() {
                Ok(addr) => return addr,
                Err(_) => {
                    eprintln!("Valor de bind inválido `{value}` para --bind. Usando 127.0.0.1.");
                    return IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
                }
            }
        }
    }

    if let Ok(value) = std::env::var("BIND_ADDR") {
        if let Ok(addr) = value.parse::<IpAddr>() {
            if addr == IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)) {
                eprintln!(
                    "BIND_ADDR=0.0.0.0 detectado. Preferimos `--bind 0.0.0.0` o `--unsafe-bind` para hacerlo explícito."
                );
            }
            return addr;
        }
        eprintln!("Valor de bind inválido `{value}` para `BIND_ADDR`. Usando 127.0.0.1.");
    }

    IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))
}

fn resolve_sqlite_enabled() -> bool {
    let args = std::env::args().skip(1);
    for arg in args {
        if arg == "--no-sqlite" {
            return false;
        }
        if arg == "--sqlite" || arg == "--sqlite-enabled" {
            return true;
        }
    }
    match std::env::var("SQLITE_ENABLED").ok().as_deref() {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => true,
    }
}

fn parse_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => default,
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_never_serialized() {
        let mut c = Config::default();
        c.api_key = "super-secret-api-value".to_string();
        c.embedding_api_key = "embed-secret-value".to_string();
        c.llm_api_key = "llm-secret-value".to_string();
        c.libsql_auth_token = "turso-secret-value".to_string();
        let toml_str = toml::to_string_pretty(&c).unwrap();
        assert!(!toml_str.contains("super-secret-api-value"));
        assert!(!toml_str.contains("embed-secret-value"));
        assert!(!toml_str.contains("llm-secret-value"));
        assert!(!toml_str.contains("turso-secret-value"));
    }

    #[test]
    fn roundtrip_omits_secrets_and_still_parses() {
        // A serialized config (no secret fields) must deserialize thanks to
        // `#[serde(default)]`, coming back with empty secrets.
        let mut c = Config::default();
        c.api_key = "secret-that-should-vanish".to_string();
        let s = toml::to_string_pretty(&c).unwrap();
        let parsed: Config = toml::from_str(&s).unwrap();
        assert_eq!(parsed.api_key, "");
        // MED fix: rate limiting is on by default.
        assert_eq!(parsed.rate_limit_rps, 100);
    }

    #[test]
    fn default_rate_limit_is_enabled() {
        assert_eq!(Config::default().rate_limit_rps, 100);
    }
}
