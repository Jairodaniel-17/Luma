use rust_kiss_vdb::api;
use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use std::net::SocketAddr;
use tokio::sync::oneshot;

async fn start() -> (String, oneshot::Sender<()>) {
    let config = Config {
        port: 0,
        api_key: "test".to_string(),
        data_dir: None,
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
        cors_allowed_origins: None,
    };
    let engine = Engine::new(config.clone()).unwrap();
    let app = api::router(engine, config);

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
async fn state_put_get() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let put = client
        .put(format!("{}/v1/state/job:1", base))
        .header("Authorization", "Bearer test")
        .json(&serde_json::json!({"value":{"progress":1}}))
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success());

    let got = client
        .get(format!("{}/v1/state/job:1", base))
        .header("Authorization", "Bearer test")
        .send()
        .await
        .unwrap();
    assert!(got.status().is_success());
    let v: serde_json::Value = got.json().await.unwrap();
    assert_eq!(v["key"], "job:1");
    assert_eq!(v["value"]["progress"], 1);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn vector_create_upsert_search() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{}/v1/vector/docs", base))
        .header("Authorization", "Bearer test")
        .json(&serde_json::json!({"dim":2,"metric":"cosine"}))
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    let upsert = client
        .post(format!("{}/v1/vector/docs/upsert", base))
        .header("Authorization", "Bearer test")
        .json(&serde_json::json!({"id":"a","vector":[1.0,0.0],"meta":{"tag":"x"}}))
        .send()
        .await
        .unwrap();
    assert!(upsert.status().is_success());

    let search = client
        .post(format!("{}/v1/vector/docs/search", base))
        .header("Authorization", "Bearer test")
        .json(&serde_json::json!({"vector":[0.9,0.1],"k":1,"include_meta":true}))
        .send()
        .await
        .unwrap();
    assert!(search.status().is_success());
    let v: serde_json::Value = search.json().await.unwrap();
    assert_eq!(v["hits"][0]["id"], "a");

    let _ = shutdown.send(());
}
