pub fn resolve_max_body_mb() -> usize {
    resolve_mb("--max-body-mb", "MAX_BODY_MB", 100.0)
}

pub fn resolve_max_json_mb() -> usize {
    resolve_mb("--max-json-mb", "MAX_JSON_MB", 100.0)
}

pub fn resolve_max_vector_dim() -> usize {
    resolve_usize("--max-vector-dim", "MAX_VECTOR_DIM", 4096)
}

pub fn resolve_max_k() -> usize {
    resolve_usize("--max-k", "MAX_K", 256)
}

pub fn resolve_max_key_len() -> usize {
    resolve_usize("--max-key-len", "MAX_KEY_LEN", 512)
}

pub fn resolve_max_collection_len() -> usize {
    resolve_usize("--max-collection-len", "MAX_COLLECTION_LEN", 64)
}

pub fn resolve_wal_retention_segments() -> usize {
    resolve_usize("--wal-retention", "WAL_RETENTION_SEGMENTS", 8)
}

pub fn resolve_request_timeout_secs() -> u64 {
    resolve_u64("--request-timeout-secs", "REQUEST_TIMEOUT_SECS", 30)
}

// Helpers

fn cli_arg(flag: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            return args.next();
        }
    }
    None
}

fn resolve_mb(flag: &str, env: &str, default_mb: f64) -> usize {
    if let Some(val_str) = cli_arg(flag) {
        if let Ok(mb) = val_str.parse::<f64>() {
            return (mb * 1024.0 * 1024.0) as usize;
        }
    }

    if let Ok(val_str) = std::env::var(env) {
        if let Ok(mb) = val_str.parse::<f64>() {
            return (mb * 1024.0 * 1024.0) as usize;
        }
    }

    (default_mb * 1024.0 * 1024.0) as usize
}

fn resolve_usize(flag: &str, env: &str, default: usize) -> usize {
    if let Some(val_str) = cli_arg(flag) {
        if let Ok(v) = val_str.parse::<usize>() {
            return v;
        }
    }
    if let Ok(val_str) = std::env::var(env) {
        if let Ok(v) = val_str.parse::<usize>() {
            return v;
        }
    }
    default
}

fn resolve_u64(flag: &str, env: &str, default: u64) -> u64 {
    if let Some(val_str) = cli_arg(flag) {
        if let Ok(v) = val_str.parse::<u64>() {
            return v;
        }
    }
    if let Ok(val_str) = std::env::var(env) {
        if let Ok(v) = val_str.parse::<u64>() {
            return v;
        }
    }
    default
}
