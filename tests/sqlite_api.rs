use rust_kiss_vdb::api;
use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::oneshot;

async fn start_with_sqlite(data_dir: String) -> (String, oneshot::Sender<()>) {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir.clone()),
        snapshot_interval_secs: 30,
        event_buffer_size: 1000,
        live_broadcast_capacity: 1024,
        wal_segment_max_bytes: 4 * 1024 * 1024,
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
        sqlite_enabled: true,
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
    };
    let engine = Engine::new(config.clone()).unwrap();
    let sqlite = Some(
        rust_kiss_vdb::sqlite::SqliteService::new(
            config.data_dir.as_ref().unwrap().to_string() + "/sqlite/rustkiss.db",
        )
        .unwrap(),
    );
    let search_dir = PathBuf::from(&data_dir);
    let search_engine = Arc::new(SearchEngine::new(search_dir).unwrap());
    let app = api::router(engine, config, sqlite, search_engine);

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

#[tokio::test]
async fn sqlite_exec_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let (base, shutdown) = start_with_sqlite(data_dir).await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{}/v1/sql/exec", base))
        .json(&serde_json::json!({"sql":"CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, body TEXT)","params":[]}))
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    let insert = client
        .post(format!("{}/v1/sql/exec", base))
        .json(&serde_json::json!({"sql":"INSERT INTO notes(body) VALUES (?)","params":["hola"]}))
        .send()
        .await
        .unwrap();
    assert!(insert.status().is_success());

    let query = client
        .post(format!("{}/v1/sql/query", base))
        .json(&serde_json::json!({"sql":"SELECT body FROM notes","params":[]}))
        .send()
        .await
        .unwrap();
    assert!(query.status().is_success());
    let body: serde_json::Value = query.json().await.unwrap();
    assert_eq!(body["rows"][0]["body"], "hola");

    let _ = shutdown.send(());
}
