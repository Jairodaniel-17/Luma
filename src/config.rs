use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
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
    pub cors_allowed_origins: Option<String>,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8080);

        let api_key = std::env::var("API_KEY").unwrap_or_else(|_| "dev".to_string());
        let data_dir = std::env::var("DATA_DIR").ok();

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

        let wal_retention_segments = std::env::var("WAL_RETENTION_SEGMENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);

        let request_timeout_secs = std::env::var("REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let max_body_bytes = std::env::var("MAX_BODY_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_048_576);

        let max_key_len = std::env::var("MAX_KEY_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);

        let max_collection_len = std::env::var("MAX_COLLECTION_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);

        let max_id_len = std::env::var("MAX_ID_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);

        let max_vector_dim = std::env::var("MAX_VECTOR_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);

        let max_k = std::env::var("MAX_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        let max_json_bytes = std::env::var("MAX_JSON_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 * 1024);

        let cors_allowed_origins = std::env::var("CORS_ALLOWED_ORIGINS").ok();

        Ok(Self {
            port,
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
            cors_allowed_origins,
        })
    }
}
