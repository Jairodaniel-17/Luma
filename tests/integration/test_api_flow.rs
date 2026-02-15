use axum::body::Body;
use axum::http::{Request, StatusCode};
use luma::api::router;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use luma::sqlite::SqliteService;
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
    let app = router(engine, config.clone(), None, search_engine, None);

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
                        "metric": "dot"
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

    // 2.5 Verify Count
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/vector/test_col")
                .header("Authorization", "Bearer test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["count"], 1, "Collection count should be 1 after add");

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
    if body.get("hits").is_none() {
        panic!("Response missing 'hits' field. Body: {}", body);
    }
    let hits = body["hits"].as_array().expect("hits should be array");
    if hits.is_empty() {
        panic!("Search returned 0 hits. Body: {}", body);
    }
    assert_eq!(hits[0]["id"], "vec1");
}

#[tokio::test]
async fn test_api_sql_flow() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        api_key: "test-key".to_string(),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_str().unwrap().to_string()),
        ..Config::default()
    };

    let token = CancellationToken::new();
    let engine = Engine::new(config.clone(), token).unwrap();
    let sqlite = SqliteService::new(&db_path).unwrap();
    let search_engine = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());

    let app = router(engine, config.clone(), Some(sqlite), search_engine, None);

    // 1. Create Table (Exec)
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sql/exec")
                .header("Authorization", "Bearer test-key")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "sql": "CREATE TABLE items (id INTEGER, name TEXT)",
                        "params": []
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 2. Insert (Exec)
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sql/exec")
                .header("Authorization", "Bearer test-key")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "sql": "INSERT INTO items (id, name) VALUES (?, ?)",
                        "params": [1, "foo"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Check affected rows
    let body_bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["rows_affected"], 1);

    // 3. Query
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sql/query")
                .header("Authorization", "Bearer test-key")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    json!({
                        "sql": "SELECT * FROM items WHERE id = ?",
                        "params": [1]
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

    assert_eq!(body["rows"][0]["name"], "foo");
}
