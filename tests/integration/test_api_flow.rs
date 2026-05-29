use axum::body::Body;
use axum::http::{Request, StatusCode};
use luma::api::router;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use serde_json::{json, Value};
use std::sync::Arc;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt; // for `oneshot`

#[tokio::test]
async fn test_api_vector_flow() {
    let dir = tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        api_key: "test-key".to_string(),
        ..Config::default()
    };
    let token = CancellationToken::new();
    let engine = Engine::new(config.clone(), token).unwrap();
    let search_engine = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());

    // Minimal app setup without SQLite/Auth for vector test
    let app = router(
        engine,
        config.clone(),
        None,
        search_engine,
        None,
        std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        )),
        None,
    );

    // 1. Create Collection
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/vector/test_col")
                .header("Authorization", "Bearer test-key")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "dim": 4,
                        "metric": "cosine"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 2. Add Vector
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/vector/test_col/add")
                .header("Authorization", "Bearer test-key")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "id": "vec1",
                        "vector": [1.0, 0.0, 0.0, 0.0],
                        "meta": {"tag": "A"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 3. Search
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/vector/test_col/search")
                .header("Authorization", "Bearer test-key")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "vector": [1.0, 0.0, 0.0, 0.0],
                        "k": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();

    // Search returns {"hits": [...]}
    assert_eq!(body["hits"][0]["id"], "vec1");
}

