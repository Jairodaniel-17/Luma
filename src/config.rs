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
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let port = resolve_port();
        let bind_addr = resolve_bind_addr();

        let api_key = std::env::var("RUSTKISS_API_KEY")
            .or_else(|_| std::env::var("API_KEY"))
            .unwrap_or_else(|_| "dev".to_string());
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
        if arg == "--bind" {
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
