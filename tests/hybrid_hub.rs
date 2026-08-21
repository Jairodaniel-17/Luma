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
    let embeddings = luma::engine::embeddings::EmbeddingHandle::new(
        luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        ),
    );

    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: Some(sqlite),
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

// ─── M3.3: reindex a collection under a new embedding model ──────────────────

/// Poll a reindex job until it leaves `running`, with a bounded wait so a hung
/// job fails the test instead of blocking the suite forever.
async fn await_reindex(
    client: &reqwest::Client,
    base: &str,
    collection: &str,
    job_id: &str,
) -> serde_json::Value {
    for _ in 0..100 {
        let resp = client
            .get(format!(
                "{}/v1/vector/{}/reindex/{}",
                base, collection, job_id
            ))
            .bearer_auth("test")
            .send()
            .await
            .unwrap();
        if resp.status() == StatusCode::OK {
            let body: serde_json::Value = resp.json().await.unwrap();
            if body["status"] != "running" {
                return body;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("reindex job {job_id} never reached a terminal state");
}

#[tokio::test]
async fn reindex_reembeds_every_chunk_into_a_new_collection() {
    let app = start().await;
    let client = reqwest::Client::new();
    ingest_docs(&client, &app.base).await;

    let before: serde_json::Value = client
        .get(format!("{}/v1/vector/hybrid", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let source_count = before["count"].as_u64().unwrap();
    assert!(source_count > 0, "fixture should have produced vectors");

    let start_resp = client
        .post(format!("{}/v1/vector/hybrid/reindex", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "target": "hybrid_v2", "batch_size": 5 }))
        .send()
        .await
        .unwrap();
    assert_eq!(start_resp.status(), StatusCode::ACCEPTED);
    let started: serde_json::Value = start_resp.json().await.unwrap();
    let job_id = started["job_id"].as_str().unwrap().to_string();
    assert_eq!(started["target"], "hybrid_v2");

    let done = await_reindex(&client, &app.base, "hybrid", &job_id).await;
    assert_eq!(done["status"], "done", "job report: {done}");
    assert_eq!(done["processed"].as_u64().unwrap(), source_count);
    assert_eq!(done["reembedded"].as_u64().unwrap(), source_count);
    // Every vector here came through the hub, so all of them carry chunk text.
    assert_eq!(done["skipped_no_text"].as_u64().unwrap(), 0);
    assert_eq!(done["target_dim"].as_u64().unwrap(), 384);

    // The source is untouched — that is the whole point of writing to a new
    // collection rather than rewriting in place.
    let after_source: serde_json::Value = client
        .get(format!("{}/v1/vector/hybrid", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(after_source["count"].as_u64().unwrap(), source_count);

    // And the target holds the same population, searchable.
    let target: serde_json::Value = client
        .get(format!("{}/v1/vector/hybrid_v2", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(target["count"].as_u64().unwrap(), source_count);

    let search = client
        .post(format!("{}/v1/vector/hybrid_v2/search", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "vector": vec![0.1f32; 384], "k": 3 }))
        .send()
        .await
        .unwrap();
    assert_eq!(search.status(), StatusCode::OK);
    let hits: serde_json::Value = search.json().await.unwrap();
    assert!(
        !hits["hits"].as_array().unwrap().is_empty(),
        "reindexed collection must be searchable"
    );

    let _ = app.shutdown.send(());
}

#[tokio::test]
async fn reindex_rejects_rewriting_in_place() {
    let app = start().await;
    let client = reqwest::Client::new();
    ingest_docs(&client, &app.base).await;

    // Rewriting in place would require dropping the source first, so a provider
    // failure midway would leave no vectors and no way back. Refused up front.
    let resp = client
        .post(format!("{}/v1/vector/hybrid/reindex", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "target": "hybrid" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let _ = app.shutdown.send(());
}

#[tokio::test]
async fn reindex_unknown_collection_is_404_and_unknown_job_is_404() {
    let app = start().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/v1/vector/does_not_exist/reindex", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    let status = client
        .get(format!("{}/v1/vector/hybrid/reindex/no-such-job", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::NOT_FOUND);

    let _ = app.shutdown.send(());
}
