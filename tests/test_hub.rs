use axum::http::StatusCode;
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
        compaction_max_bytes_per_pass: 10 * 1024 * 1024,
        ..Config::default()
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let search_engine = Arc::new(SearchEngine::new(temp_dir.path().to_path_buf()).unwrap());

    let embeddings = std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
        luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
    ));
    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: None,
        search_engine,
        auth_store: None,
        embeddings,
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
async fn hub_ingest_endpoint() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    // Ingest a fake document without real embeddings configured (should return error saying not configured)
    let ingest = client
        .post(format!("{}/v1/db/my_namespace/ingest", base))
        .json(&serde_json::json!({
            "text": "This is a contract file.",
            "metadata": { "year": 2024 }
        }))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();

    // We expect success because Ollama with granite-embedding:30m is configured
    if ingest.status() != StatusCode::OK {
        let body: serde_json::Value = ingest.json().await.unwrap();
        panic!("Expected 200 OK, got: {:?}", body);
    }

    let body: serde_json::Value = ingest.json().await.unwrap();
    assert_eq!(body["status"], "success");
    assert_eq!(body["namespace"], "my_namespace");
    assert!(body["doc_id"].as_str().is_some());

    let _ = shutdown.send(());
}

#[tokio::test]
async fn hub_search_endpoint() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    // 1. Ingest a document
    let ingest = client
        .post(format!("{}/v1/db/my_namespace/ingest", base))
        .json(&serde_json::json!({
            "id": "doc_123",
            "text": "The quick brown fox jumps over the lazy dog. This is a very important test document for the search feature.",
            "metadata": { "category": "animal" }
        }))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();

    assert_eq!(ingest.status(), StatusCode::OK);

    // 2. Search document
    let search = client
        .post(format!("{}/v1/db/my_namespace/search", base))
        .json(&serde_json::json!({
            "query": "quick fox",
            "limit": 5
        }))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();

    assert_eq!(search.status(), StatusCode::OK);
    let search_body: serde_json::Value = search.json().await.unwrap();

    // Validate response structure
    let results = search_body["results"]
        .as_array()
        .expect("Results should be an array");
    assert!(!results.is_empty(), "Should find the ingested document");

    let first_hit = &results[0];
    assert_eq!(first_hit["id"], "doc_123");
    assert!(first_hit["score"].as_f64().is_some());
    assert!(!first_hit["snippets"].as_array().unwrap().is_empty());
    assert_eq!(first_hit["document"]["text"].as_str().unwrap(), "The quick brown fox jumps over the lazy dog. This is a very important test document for the search feature.");

    let _ = shutdown.send(());
}
