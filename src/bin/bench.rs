use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::vector::{Metric, SearchRequest, VectorItem};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "dev".to_string(),
        data_dir: None,
        snapshot_interval_secs: 30,
        event_buffer_size: 10_000,
        live_broadcast_capacity: 4096,
        wal_segment_max_bytes: 64 * 1024 * 1024,
        wal_retention_segments: 8,
        request_timeout_secs: 30,
        max_body_bytes: 1_048_576,
        max_key_len: 512,
        max_collection_len: 64,
        max_id_len: 128,
        max_vector_dim: 4096,
        max_k: 256,
        max_json_bytes: 64 * 1024,
        max_state_batch: 256,
        max_vector_batch: 256,
        max_doc_find: 100,
        cors_allowed_origins: None,
        sqlite_enabled: false,
        sqlite_path: None,
    };
    let engine = Engine::new(config)?;

    let n = 20_000usize;
    let mut put_lat = Vec::with_capacity(n);
    for i in 0..n {
        let key = format!("bench:{}", i);
        let start = Instant::now();
        let _ = engine.put_state(key, serde_json::json!({"i": i}), None, None);
        put_lat.push(start.elapsed());
    }
    report("state.put", &put_lat);

    let mut get_lat = Vec::with_capacity(n);
    for i in 0..n {
        let key = format!("bench:{}", i);
        let start = Instant::now();
        let _ = engine.get_state(&key);
        get_lat.push(start.elapsed());
    }
    report("state.get", &get_lat);

    engine.create_vector_collection("bench", 8, Metric::Cosine)?;
    for i in 0..50_000usize {
        let v = vec![i as f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        engine.vector_upsert(
            "bench",
            &format!("id{}", i),
            VectorItem {
                vector: v,
                meta: serde_json::json!({"i": i}),
            },
        )?;
    }

    let mut search_lat = Vec::with_capacity(2000);
    for _ in 0..2000usize {
        let start = Instant::now();
        let _ = engine.vector_search(
            "bench",
            SearchRequest {
                vector: vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                k: 10,
                filters: None,
                include_meta: Some(false),
            },
        )?;
        search_lat.push(start.elapsed());
    }
    report("vector.search(bruteforce)", &search_lat);

    Ok(())
}

fn report(name: &str, samples: &[Duration]) {
    let mut v: Vec<u128> = samples.iter().map(|d| d.as_micros()).collect();
    v.sort_unstable();
    let p50 = percentile(&v, 50.0);
    let p95 = percentile(&v, 95.0);
    println!("{name}: n={} p50={}us p95={}us", v.len(), p50, p95);
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
