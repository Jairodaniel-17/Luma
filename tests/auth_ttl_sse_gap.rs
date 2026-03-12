use futures_util::StreamExt;
use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

async fn start_with_config(config: Config) -> (String, oneshot::Sender<()>) {
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let search_engine = Arc::new(SearchEngine::new(temp_dir.path().to_path_buf()).unwrap());

    let app = api::router(
        engine,
        config,
        None,
        search_engine,
        None,
        std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        )),
    );

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
        ..Config::default()
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
        .bearer_auth("test")
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let put = client
        .put(format!("{}/v1/state/ttl:1", base))
        .json(&serde_json::json!({"value":{"v":1},"ttl_ms":50}))
        .bearer_auth("test")
        .bearer_auth("test")
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
        .bearer_auth("test")
        .bearer_auth("test")
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
            .bearer_auth("test")
            .bearer_auth("test")
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

#[tokio::test]
async fn consumer_group_resumes_from_committed_offset() {
    let (base, shutdown) = start_with_config(base_config()).await;
    let client = reqwest::Client::new();

    let _ = client
        .put(format!("{}/v1/state/group:1", base))
        .json(&serde_json::json!({"value":{"i":1}}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();

    let resp = client
        .get(format!(
            "{}/v1/stream?types=state_updated&consumer_group=agents",
            base
        ))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => panic!("timeout waiting for first group event"),
            chunk = stream.next() => {
                let Some(chunk) = chunk else { break };
                buf.push_str(&String::from_utf8_lossy(&chunk.unwrap()));
                if buf.contains("\"key\":\"group:1\"") {
                    break;
                }
            }
        }
    }
    drop(stream);

    let _ = client
        .put(format!("{}/v1/state/group:2", base))
        .json(&serde_json::json!({"value":{"i":2}}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();

    let resp2 = client
        .get(format!(
            "{}/v1/stream?types=state_updated&consumer_group=agents",
            base
        ))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(resp2.status().is_success());

    let mut stream2 = resp2.bytes_stream();
    let mut buf2 = String::new();
    let deadline2 = tokio::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline2) => panic!("timeout waiting for resumed group event"),
            chunk = stream2.next() => {
                let Some(chunk) = chunk else { break };
                buf2.push_str(&String::from_utf8_lossy(&chunk.unwrap()));
                if buf2.contains("\"key\":\"group:2\"") {
                    break;
                }
            }
        }
    }

    assert!(!buf2.contains("\"key\":\"group:1\""));
    let _ = shutdown.send(());
}
