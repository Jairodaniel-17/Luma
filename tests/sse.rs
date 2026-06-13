use futures_util::StreamExt;
use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

async fn start() -> (String, oneshot::Sender<()>) {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: None,
        snapshot_interval_secs: 30,
        ..Config::default()
    };
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();

    let temp_dir = tempfile::tempdir().unwrap();
    let search_engine = Arc::new(SearchEngine::new(temp_dir.path().to_path_buf()).unwrap());

    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: None,
        search_engine,
        auth_store: None,
        embeddings: std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        )),
        audit_log: None,
        rbac: None,
    });

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
        .bearer_auth("test")
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    let put_fut = client
        .put(format!("{}/v1/state/job:sse", base))
        .json(&serde_json::json!({"value":{"progress":1}}))
        .bearer_auth("test")
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
