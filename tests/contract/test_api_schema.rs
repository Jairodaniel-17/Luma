use axum::body::Body;
use axum::http::{Request, StatusCode};
use luma::api::router;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use serde_json::json;
use serde_json::Value;
use std::sync::Arc;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

#[tokio::test]
async fn test_health_check_contract() {
    let dir = tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        ..Config::default()
    };
    let token = CancellationToken::new();
    let engine = Engine::new(config.clone(), token).unwrap();
    let search_engine = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());

    let app = router(engine, config, None, search_engine, None);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/health")
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

    // Contract: Health check returns {"status": "ok", ...}
    assert_eq!(body["status"], "ok");
    assert!(body.get("version").is_some());
    assert!(body.get("uptime_secs").is_some());
}

#[tokio::test]
async fn test_error_schema_contract() {
    let dir = tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        api_key: "test-key".to_string(),
        ..Config::default()
    };
    let token = CancellationToken::new();
    let engine = Engine::new(config.clone(), token).unwrap();
    let search_engine = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());

    let app = router(engine, config, None, search_engine, None);

    // Request with missing auth
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/state")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body_bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap();

    // Contract: Errors return {"error": "...", "message": "..."}
    assert!(body.get("error").is_some());
    assert!(body.get("message").is_some());
}
