use axum::http::StatusCode;
use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use luma::sqlite::SqliteService;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

struct TestApp {
    base: String,
    shutdown: oneshot::Sender<()>,
    _dir: TempDir,
}

async fn start() -> TestApp {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("hybrid.db");
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.path().to_string_lossy().to_string()),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    let sqlite = SqliteService::new(&db_path).unwrap();
    let search_engine = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());
    let embeddings = Arc::new(luma::engine::embeddings::EmbeddingClient::new(
        luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
    ));

    let app = api::router(
        engine,
        config,
        Some(sqlite),
        search_engine,
        None,
        embeddings,
    );

    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown, rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    TestApp {
        base: format!("http://{}", addr),
        shutdown,
        _dir: dir,
    }
}

async fn ingest_docs(client: &reqwest::Client, base: &str) {
    for idx in 0..12usize {
        let tenant = if idx < 2 { "acme" } else { "globex" };
        let group = if idx < 6 { "keep" } else { "drop" };
        let resp = client
            .post(format!("{}/v1/db/hybrid/ingest", base))
            .bearer_auth("test")
            .json(&serde_json::json!({
                "id": format!("doc-{idx}"),
                "text": format!("policy document {idx} for tenant {tenant}"),
                "metadata": {
                    "tenant": tenant,
                    "group": group,
                    "idx": idx,
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn hybrid_search_exposes_distinct_sql_and_vector_paths() {
    let app = start().await;
    let client = reqwest::Client::new();
    ingest_docs(&client, &app.base).await;

    let sql_first = client
        .post(format!("{}/v1/db/hybrid/search", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "policy",
            "sql_filter": "json_extract(metadata, '$.tenant') = 'acme'",
            "limit": 3,
            "include_plan": true,
            "include_diagnostics": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(sql_first.status(), StatusCode::OK);
    let sql_first_body: serde_json::Value = sql_first.json().await.unwrap();
    assert_eq!(sql_first_body["_plan"]["strategy"], "sql_first");
    assert_eq!(sql_first_body["_plan"]["filter_application"], "pre_vector");
    assert_eq!(sql_first_body["_plan"]["estimated_sql_candidates"], 2);
    assert_eq!(sql_first_body["_diagnostics"]["sql_candidates"], 2);

    let vector_first = client
        .post(format!("{}/v1/db/hybrid/search", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "policy",
            "sql_filter": "json_extract(metadata, '$.tenant') = 'globex'",
            "limit": 1,
            "include_plan": true,
            "include_diagnostics": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(vector_first.status(), StatusCode::OK);
    let vector_first_body: serde_json::Value = vector_first.json().await.unwrap();
    assert_eq!(vector_first_body["_plan"]["strategy"], "vector_first");
    assert_eq!(
        vector_first_body["_plan"]["filter_application"],
        "post_vector"
    );
    assert!(
        vector_first_body["_diagnostics"]["ranked_docs_before_filter"]
            .as_u64()
            .unwrap()
            >= vector_first_body["_diagnostics"]["ranked_docs_after_filter"]
                .as_u64()
                .unwrap()
    );

    let _ = app.shutdown.send(());
}

#[tokio::test]
async fn vector_first_returns_diagnostics_without_exposing_plan() {
    let app = start().await;
    let client = reqwest::Client::new();
    ingest_docs(&client, &app.base).await;

    let response = client
        .post(format!("{}/v1/db/hybrid/search", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "policy",
            "sql_filter": "json_extract(metadata, '$.group') = 'keep'",
            "limit": 1,
            "include_diagnostics": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.unwrap();

    assert!(body.get("_plan").is_none());
    assert!(body.get("_diagnostics").is_some());
    assert_eq!(body["_diagnostics"]["ranked_docs_before_filter"], 12);
    assert_eq!(body["_diagnostics"]["ranked_docs_after_filter"], 6);

    let _ = app.shutdown.send(());
}
