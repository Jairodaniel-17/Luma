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
    start_with_config(|cfg| cfg).await
}

async fn start_with_config(mut f: impl FnMut(Config) -> Config) -> TestApp {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("memory.db");
    let config = f(Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.path().to_string_lossy().to_string()),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    });

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

#[tokio::test]
async fn episodic_ingest_promotes_semantic_fact_when_consolidation_enabled() {
    let app = start_with_config(|mut cfg| {
        cfg.memory_consolidation_enabled = true;
        cfg.llm_provider = "mock".to_string();
        cfg
    })
    .await;
    let client = reqwest::Client::new();

    let ingest = client
        .post(format!("{}/v1/memory/agents/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-2",
            "text": "El usuario prefiere alertas por correo",
            "metadata": { "channel": "chat" }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ingest.status(), StatusCode::OK);

    let query = client
        .post(format!("{}/v1/memory/agents/query", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "¿Qué prefiere el usuario?",
            "entity_id": "user-2",
            "include_evidence": true
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(query.status(), StatusCode::OK);
    let body: serde_json::Value = query.json().await.unwrap();
    assert!(
        body["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["record"]["kind"] == "semantic")
    );

    let _ = app.shutdown.send(());
}

#[tokio::test]
async fn memory_recall_and_timeline_endpoints_work() {
    let app = start().await;
    let client = reqwest::Client::new();

    let ingest = client
        .post(format!("{}/v1/memory/agents/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-1",
            "text": "El usuario pidió activar alertas por correo",
            "metadata": { "channel": "chat" },
            "session_id": "sess-1"
        }))
        .send()
        .await
        .unwrap();
    let ingest_status = ingest.status();
    let ingest_body = ingest.text().await.unwrap();
    assert_eq!(ingest_status, StatusCode::OK, "{ingest_body}");

    let fact = client
        .post(format!("{}/v1/memory/agents/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-1",
            "fact_key": "notification_preference",
            "content": "Prefiere alertas por correo",
            "metadata": { "category": "preferences" }
        }))
        .send()
        .await
        .unwrap();
    let fact_status = fact.status();
    let fact_body = fact.text().await.unwrap();
    assert_eq!(fact_status, StatusCode::OK, "{fact_body}");

    let query = client
        .post(format!("{}/v1/memory/agents/query", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "¿Qué recuerda del usuario sobre notificaciones?",
            "entity_id": "user-1",
            "include_evidence": true
        }))
        .send()
        .await
        .unwrap();
    let query_status = query.status();
    let query_body = query.text().await.unwrap();
    assert_eq!(query_status, StatusCode::OK, "{query_body}");
    let body: serde_json::Value = serde_json::from_str(&query_body).unwrap();
    assert_eq!(body["mode"], "recall");
    assert!(body["results"].as_array().unwrap().len() >= 1);
    assert!(body["evidence"].as_array().unwrap().len() >= 1);

    let timeline = client
        .get(format!("{}/v1/memory/agents/timeline/user-1", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let timeline_status = timeline.status();
    let timeline_body_text = timeline.text().await.unwrap();
    assert_eq!(timeline_status, StatusCode::OK, "{timeline_body_text}");
    let timeline_body: serde_json::Value = serde_json::from_str(&timeline_body_text).unwrap();
    assert_eq!(timeline_body["entity_id"], "user-1");
    assert_eq!(timeline_body["events"].as_array().unwrap().len(), 1);

    let _ = app.shutdown.send(());
}

#[tokio::test]
async fn memory_procedural_next_step_is_deterministic() {
    let app = start().await;
    let client = reqwest::Client::new();

    let procedure = client
        .post(format!("{}/v1/memory/ops/upsert_procedure", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "procedure_id": "approve_refund",
            "name": "Approve refund",
            "nodes": [
                { "node_id": "start", "kind": "start", "label": "Start", "payload": {} },
                { "node_id": "validate", "kind": "action", "label": "Validate request", "payload": {} },
                { "node_id": "approve", "kind": "goal", "label": "Approve", "payload": {} }
            ],
            "edges": [
                { "from_node_id": "start", "to_node_id": "validate", "priority": 10, "condition": null },
                {
                    "from_node_id": "validate",
                    "to_node_id": "approve",
                    "priority": 10,
                    "condition": { "field": "request.amount", "op": "lte", "value": 500 }
                }
            ],
            "constraints": [
                {
                    "constraint_id": "role-check",
                    "target_node_id": "approve",
                    "condition": { "field": "actor.role", "op": "eq", "value": "manager" },
                    "message": "manager role required"
                }
            ]
        }))
        .send()
        .await
        .unwrap();
    let procedure_status = procedure.status();
    let procedure_body = procedure.text().await.unwrap();
    assert_eq!(procedure_status, StatusCode::OK, "{procedure_body}");

    let first_step = client
        .post(format!("{}/v1/memory/ops/next_step", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "procedure_id": "approve_refund",
            "context": { "request": { "amount": 200 }, "actor": { "role": "manager" } }
        }))
        .send()
        .await
        .unwrap();
    let first_status = first_step.status();
    let first_body_text = first_step.text().await.unwrap();
    assert_eq!(first_status, StatusCode::OK, "{first_body_text}");
    let first_body: serde_json::Value = serde_json::from_str(&first_body_text).unwrap();
    assert_eq!(first_body["next_node"]["node_id"], "validate");

    let second_step = client
        .post(format!("{}/v1/memory/ops/next_step", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "procedure_id": "approve_refund",
            "current_node_id": "validate",
            "context": { "request": { "amount": 200 }, "actor": { "role": "manager" } }
        }))
        .send()
        .await
        .unwrap();
    let second_status = second_step.status();
    let second_body_text = second_step.text().await.unwrap();
    assert_eq!(second_status, StatusCode::OK, "{second_body_text}");
    let second_body: serde_json::Value = serde_json::from_str(&second_body_text).unwrap();
    assert_eq!(second_body["next_node"]["node_id"], "approve");

    let blocked = client
        .post(format!("{}/v1/memory/ops/next_step", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "procedure_id": "approve_refund",
            "current_node_id": "validate",
            "context": { "request": { "amount": 200 }, "actor": { "role": "agent" } }
        }))
        .send()
        .await
        .unwrap();
    let blocked_status = blocked.status();
    let blocked_body_text = blocked.text().await.unwrap();
    assert_eq!(blocked_status, StatusCode::OK, "{blocked_body_text}");
    let blocked_body: serde_json::Value = serde_json::from_str(&blocked_body_text).unwrap();
    assert_eq!(blocked_body["blocked_reason"], "manager role required");

    let _ = app.shutdown.send(());
}
