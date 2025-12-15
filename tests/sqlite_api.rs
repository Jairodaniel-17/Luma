use rust_kiss_vdb::api;
use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use std::net::SocketAddr;
use tokio::sync::oneshot;

async fn start_with_sqlite(data_dir: String) -> (String, oneshot::Sender<()>) {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir),
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
    };
    let engine = Engine::new(config.clone()).unwrap();
    let sqlite = Some(
        rust_kiss_vdb::sqlite::SqliteService::new(
            config.data_dir.as_ref().unwrap().to_string() + "/sqlite/rustkiss.db",
        )
        .unwrap(),
    );
    let app = api::router(engine, config, sqlite);

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
