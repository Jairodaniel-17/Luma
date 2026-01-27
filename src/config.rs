use serde::{Deserialize, Serialize};
use std::net::IpAddr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub port: u16,
    pub bind_addr: IpAddr,
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
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let port = resolve_port();
        let bind_addr = resolve_bind_addr();

        let api_key = std::env::var("RUSTKISS_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .unwrap_or_else(|_| "dev".to_string());
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
        let sqlite_enabled = matches!(
            std::env::var("SQLITE_ENABLED")
                .ok()
                .as_deref()
                .map(|s| s.to_ascii_lowercase()),
            Some(ref v) if v == "1" || v == "true" || v == "on"
        );
        let sqlite_path = std::env::var("SQLITE_DB_PATH").ok();

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
        })
    }
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

    if let Ok(value) = std::env::var("PORT_RUST_KISS_VDB") {
        if let Ok(port) = value.parse::<u16>() {
            return port;
        }
        eprintln!(
            "Valor de puerto invalido `{value}` para `PORT_RUST_KISS_VDB`. Usando valor por defecto."
        );
    }

    9917
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
    std::env::var("DATA_DIR").ok().or_else(|| Some("./data".to_string()))
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
