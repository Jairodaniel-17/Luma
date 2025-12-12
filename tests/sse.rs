use futures_util::StreamExt;
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
async fn sse_receives_state_updated() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/events?types=state_updated&since=0", base))
        .header("Authorization", "Bearer test")
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let put_fut = client
        .put(format!("{}/v1/state/job:sse", base))
        .header("Authorization", "Bearer test")
        .json(&serde_json::json!({"value":{"progress":1}}))
        .send();

    let mut stream = resp.bytes_stream();
    let _put = put_fut.await.unwrap();

    let mut buf = String::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => panic!("timeout waiting for sse"),
            chunk = stream.next() => {
                let Some(chunk) = chunk else { break };
                let chunk = chunk.unwrap();
                buf.push_str(&String::from_utf8_lossy(&chunk));
                if buf.contains("event:state_updated") || buf.contains("event: state_updated") {
                    break;
                }
            }
        }
    }

    let _ = shutdown.send(());
}
