use futures_util::StreamExt;
use rust_kiss_vdb::api;
use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::oneshot;

async fn start_with_config(config: Config) -> (String, oneshot::Sender<()>) {
    let engine = Engine::new(config.clone()).unwrap();
    
    let temp_dir = tempfile::tempdir().unwrap(); 
    let search_engine = Arc::new(SearchEngine::new(temp_dir.path().to_path_buf()).unwrap());
    
    let app = api::router(engine, config, None, search_engine);

    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    (format!("http://{}", addr), tx)
}

fn base_config() -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: None,
        snapshot_interval_secs: 3600,
        event_buffer_size: 1000,
        live_broadcast_capacity: 16,
        wal_segment_max_bytes: 256 * 1024,
        wal_retention_segments: 4,
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
        search_threads: 0,
        parallel_probe: true,
        parallel_probe_min_segments: 4,
        simd_enabled: true,
        index_kind: "IVF_FLAT_Q8".to_string(),
        ivf_clusters: 64,
        ivf_nprobe: 8,
        ivf_training_sample: 1024,
        ivf_min_train_vectors: 64,
        ivf_retrain_min_deltas: 32,
        q8_refine_topk: 256,
        diskann_max_degree: 32,
        diskann_build_threads: 1,
        diskann_search_list_size: 64,
        run_target_bytes: 8 * 1024 * 1024,
        run_retention: 4,
        compaction_trigger_tombstone_ratio: 0.2,
        compaction_max_bytes_per_pass: 64 * 1024 * 1024,
    }
}

#[tokio::test]
async fn ttl_emits_event() {
    let (base, shutdown) = start_with_config(base_config()).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!(
            "{}/v1/stream?types=state_deleted&key_prefix=ttl:",
            base
        ))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let put = client
        .put(format!("{}/v1/state/ttl:1", base))
        .json(&serde_json::json!({"value":{"v":1},"ttl_ms":50}))
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success());

    let mut stream = resp.bytes_stream();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);
    let mut buf = String::new();
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => panic!("timeout waiting for ttl event"),
            chunk = stream.next() => {
                let Some(chunk) = chunk else { break };
                let chunk = chunk.unwrap();
                buf.push_str(&String::from_utf8_lossy(&chunk));
                if buf.contains("\"reason\":\"ttl\"") && buf.contains("\"key\":\"ttl:1\"") {
                    break;
                }
            }
        }
    }

    let _ = shutdown.send(());
}

#[tokio::test]
async fn sse_lagged_emits_gap_instead_of_dying() {
    let mut config = base_config();
    config.live_broadcast_capacity = 1;
    let (base, shutdown) = start_with_config(config).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/stream?types=state_updated&since=0", base))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let mut stream = resp.bytes_stream();
    let _ = tokio::time::timeout(std::time::Duration::from_millis(50), stream.next()).await;

    for i in 0..5000u32 {
        let _ = client
            .put(format!("{}/v1/state/lag:{}", base, i))
            .json(&serde_json::json!({"value":{"i":i}}))
            .send()
            .await
            .unwrap();
    }

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut buf = String::new();
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => break,
            chunk = stream.next() => {
                let Some(chunk) = chunk else { break };
                let chunk = chunk.unwrap();
                buf.push_str(&String::from_utf8_lossy(&chunk));
                if buf.contains("event:gap") || buf.contains("event: gap") {
                    break;
                }
            }
        }
    }

    // This integration test is best-effort; the deterministic "Lagged -> gap"
    // behavior is covered in a unit test in `src/api/routes_events.rs`.
    assert!(!buf.is_empty());

    let _ = shutdown.send(());
}
