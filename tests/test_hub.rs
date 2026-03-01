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
        compaction_max_bytes_per_pass: 10 * 1024 * 1024,
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let search_engine = Arc::new(SearchEngine::new(temp_dir.path().to_path_buf()).unwrap());

    let embeddings = std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
        luma::engine::embeddings::EmbeddingProvider::Ollama {
            api_url: "http://localhost:11434".to_string(),
            model: "granite-embedding:30m".to_string(),
        },
    ));
    let app = api::router(engine, config, None, search_engine, None, embeddings);

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
    assert!(first_hit["snippets"].as_array().unwrap().len() > 0);
    assert_eq!(first_hit["document"]["text"].as_str().unwrap(), "The quick brown fox jumps over the lazy dog. This is a very important test document for the search feature.");

    let _ = shutdown.send(());
}
