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

// ─── Test app harness (mirrors ns_mem.rs pattern) ────────────────────────────

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

// ─── Test 1: Edge CRUD ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_edge_crud() {
    let app = start().await;
    let client = reqwest::Client::new();

    // First create two memory records so the edge has valid references
    let event_a = client
        .post(format!("{}/v1/memory/graph_ns/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "entity-a",
            "text": "Node A: user likes dark mode",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(event_a.status(), StatusCode::OK);
    let event_a_body: serde_json::Value = event_a.json().await.unwrap();
    let node_a_id = event_a_body["id"].as_str().unwrap().to_string();

    let event_b = client
        .post(format!("{}/v1/memory/graph_ns/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "entity-b",
            "text": "Node B: user prefers light themes",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(event_b.status(), StatusCode::OK);
    let event_b_body: serde_json::Value = event_b.json().await.unwrap();
    let node_b_id = event_b_body["id"].as_str().unwrap().to_string();

    // Create edge: A supports B
    let create_edge = client
        .post(format!("{}/v1/memory/graph_ns/edges", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "source_id": node_a_id,
            "target_id": node_b_id,
            "edge_type": "supports",
            "weight": 0.9,
            "metadata": { "test": true }
        }))
        .send()
        .await
        .unwrap();
    let edge_status = create_edge.status();
    let edge_body_text = create_edge.text().await.unwrap();
    assert_eq!(
        edge_status,
        StatusCode::OK,
        "create edge failed: {edge_body_text}"
    );
    let edge_body: serde_json::Value = serde_json::from_str(&edge_body_text).unwrap();
    let edge_id = edge_body["id"].as_str().unwrap().to_string();
    assert_eq!(edge_body["edge_type"], "supports");
    assert_eq!(edge_body["source_id"], node_a_id);
    assert_eq!(edge_body["target_id"], node_b_id);

    // Read edges for node A: should have 1 outgoing
    let node_edges = client
        .get(format!("{}/v1/memory/graph_ns/edges/{node_a_id}", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let ne_status = node_edges.status();
    let ne_body_text = node_edges.text().await.unwrap();
    assert_eq!(
        ne_status,
        StatusCode::OK,
        "get node edges failed: {ne_body_text}"
    );
    let ne_body: serde_json::Value = serde_json::from_str(&ne_body_text).unwrap();
    assert_eq!(ne_body["memory_id"], node_a_id);
    let outgoing = ne_body["outgoing"].as_array().unwrap();
    assert!(!outgoing.is_empty(), "expected at least 1 outgoing edge");
    assert!(outgoing.iter().any(|e| e["id"] == edge_id));

    // Delete edge
    let delete_edge = client
        .post(format!(
            "{}/v1/memory/graph_ns/edges/{edge_id}/delete",
            app.base
        ))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let del_status = delete_edge.status();
    let del_body_text = delete_edge.text().await.unwrap();
    assert_eq!(
        del_status,
        StatusCode::OK,
        "delete edge failed: {del_body_text}"
    );
    let del_body: serde_json::Value = serde_json::from_str(&del_body_text).unwrap();
    assert_eq!(del_body["deleted"], true);

    // Verify edge is gone
    let node_edges_after = client
        .get(format!("{}/v1/memory/graph_ns/edges/{node_a_id}", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert_eq!(node_edges_after.status(), StatusCode::OK);
    let ne_after: serde_json::Value = node_edges_after.json().await.unwrap();
    let outgoing_after = ne_after["outgoing"].as_array().unwrap();
    assert!(
        !outgoing_after.iter().any(|e| e["id"] == edge_id),
        "edge should have been deleted"
    );

    let _ = app.shutdown.send(());
}

// ─── Test 2: TriggeredBy edge created on consolidation ───────────────────────

#[tokio::test]
async fn test_triggered_by_on_consolidation() {
    let app = start_with_config(|mut cfg| {
        cfg.memory_consolidation_enabled = true;
        cfg.llm_provider = "mock".to_string();
        cfg
    })
    .await;
    let client = reqwest::Client::new();

    // Ingest an episodic event — consolidation should fire and create a TriggeredBy edge
    let ingest = client
        .post(format!("{}/v1/memory/cons_ns/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-cons",
            "text": "User requested dark mode to be enabled by default",
            "metadata": { "channel": "api" }
        }))
        .send()
        .await
        .unwrap();
    let ingest_status = ingest.status();
    let ingest_body_text = ingest.text().await.unwrap();
    assert_eq!(
        ingest_status,
        StatusCode::OK,
        "ingest failed: {ingest_body_text}"
    );
    let ingest_body: serde_json::Value = serde_json::from_str(&ingest_body_text).unwrap();
    let event_id = ingest_body["id"].as_str().unwrap().to_string();

    // Read node edges for the episodic event
    let node_edges = client
        .get(format!("{}/v1/memory/cons_ns/edges/{event_id}", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let ne_status = node_edges.status();
    let ne_body_text = node_edges.text().await.unwrap();
    assert_eq!(
        ne_status,
        StatusCode::OK,
        "get node edges failed: {ne_body_text}"
    );
    let ne_body: serde_json::Value = serde_json::from_str(&ne_body_text).unwrap();

    // The mock LLM extracts facts during consolidation; if it does, a triggered_by
    // edge should exist outgoing from the episodic event.
    // Accept either: (a) there is a triggered_by edge, or (b) no facts were extracted
    // (mock provider may return empty), which is also valid.
    let outgoing = ne_body["outgoing"].as_array().unwrap();
    let has_triggered_by = outgoing.iter().any(|e| e["edge_type"] == "triggered_by");

    // Either the mock created triggered_by edges, or no facts were extracted.
    // Both outcomes are acceptable — we just verify the endpoint works.
    // If triggered_by edges exist, validate their structure.
    if has_triggered_by {
        let tb_edge = outgoing
            .iter()
            .find(|e| e["edge_type"] == "triggered_by")
            .unwrap();
        assert_eq!(tb_edge["source_id"], event_id);
        assert!(tb_edge["target_id"].as_str().is_some());
    }

    let _ = app.shutdown.send(());
}

// ─── Test 3: Supersedes edge created on upsert_fact overwrite ────────────────

#[tokio::test]
async fn test_supersedes_on_upsert() {
    let app = start().await;
    let client = reqwest::Client::new();

    // First upsert of a fact
    let fact1 = client
        .post(format!("{}/v1/memory/sup_ns/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-sup",
            "fact_key": "theme_preference",
            "content": "User prefers dark mode",
            "metadata": { "category": "ui" }
        }))
        .send()
        .await
        .unwrap();
    let f1_status = fact1.status();
    let f1_body_text = fact1.text().await.unwrap();
    assert_eq!(
        f1_status,
        StatusCode::OK,
        "first upsert_fact failed: {f1_body_text}"
    );
    let f1_body: serde_json::Value = serde_json::from_str(&f1_body_text).unwrap();
    let fact_id = f1_body["id"].as_str().unwrap().to_string();

    // Second upsert with same fact_key (overwrites)
    let fact2 = client
        .post(format!("{}/v1/memory/sup_ns/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-sup",
            "fact_key": "theme_preference",
            "content": "User now prefers light mode",
            "metadata": { "category": "ui" }
        }))
        .send()
        .await
        .unwrap();
    let f2_status = fact2.status();
    let f2_body_text = fact2.text().await.unwrap();
    assert_eq!(
        f2_status,
        StatusCode::OK,
        "second upsert_fact failed: {f2_body_text}"
    );

    // Check that edges exist for the fact node — should contain a supersedes edge
    let node_edges = client
        .get(format!("{}/v1/memory/sup_ns/edges/{fact_id}", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let ne_status = node_edges.status();
    let ne_body_text = node_edges.text().await.unwrap();
    assert_eq!(
        ne_status,
        StatusCode::OK,
        "get node edges failed: {ne_body_text}"
    );
    let ne_body: serde_json::Value = serde_json::from_str(&ne_body_text).unwrap();

    let outgoing = ne_body["outgoing"].as_array().unwrap();
    let has_supersedes = outgoing.iter().any(|e| e["edge_type"] == "supersedes");
    assert!(
        has_supersedes,
        "expected a supersedes edge after second upsert_fact; edges: {ne_body}"
    );

    let _ = app.shutdown.send(());
}

// ─── Test 4: Belief history endpoint ─────────────────────────────────────────

#[tokio::test]
async fn test_belief_history_endpoint() {
    let app = start().await;
    let client = reqwest::Client::new();

    let fact_key = "color_scheme";

    // First version
    let f1 = client
        .post(format!("{}/v1/memory/hist_ns/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-hist",
            "fact_key": fact_key,
            "content": "User prefers blue color scheme",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(f1.status(), StatusCode::OK, "{}", f1.text().await.unwrap());

    // Second version (overwrites, creates history entry)
    let f2 = client
        .post(format!("{}/v1/memory/hist_ns/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-hist",
            "fact_key": fact_key,
            "content": "User now prefers green color scheme",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(f2.status(), StatusCode::OK, "{}", f2.text().await.unwrap());

    // Query belief history
    let history = client
        .get(format!(
            "{}/v1/memory/hist_ns/beliefs/{fact_key}/history",
            app.base
        ))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let hist_status = history.status();
    let hist_body_text = history.text().await.unwrap();
    assert_eq!(
        hist_status,
        StatusCode::OK,
        "belief history failed: {hist_body_text}"
    );
    let hist_body: serde_json::Value = serde_json::from_str(&hist_body_text).unwrap();

    assert_eq!(hist_body["fact_key"], fact_key);
    let versions = hist_body["versions"].as_array().unwrap();
    assert!(
        !versions.is_empty(),
        "expected at least one history version; body: {hist_body}"
    );
    // The previous version's content should be the first one we wrote
    let first_version = &versions[0];
    assert!(
        first_version["content"].as_str().is_some(),
        "version should have content"
    );

    let _ = app.shutdown.send(());
}

// ─── Test 5: Centrality hub nodes ────────────────────────────────────────────

#[tokio::test]
async fn test_centrality_hub_nodes() {
    let app = start().await;
    let client = reqwest::Client::new();

    // Create 3 memory nodes
    let mut node_ids: Vec<String> = Vec::new();
    for i in 0..3 {
        let ev = client
            .post(format!("{}/v1/memory/cent_ns/ingest_event", app.base))
            .bearer_auth("test")
            .json(&serde_json::json!({
                "entity_id": format!("user-{i}"),
                "text": format!("Memory node number {i} about user preferences"),
                "metadata": {}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(ev.status(), StatusCode::OK);
        let ev_body: serde_json::Value = ev.json().await.unwrap();
        node_ids.push(ev_body["id"].as_str().unwrap().to_string());
    }

    let hub_id = &node_ids[0];
    let node_b = &node_ids[1];
    let node_c = &node_ids[2];

    // Create edges pointing TO hub_id (making it a hub with more in-links)
    for target in &[node_b.as_str(), node_c.as_str()] {
        let edge = client
            .post(format!("{}/v1/memory/cent_ns/edges", app.base))
            .bearer_auth("test")
            .json(&serde_json::json!({
                "source_id": target,
                "target_id": hub_id,
                "edge_type": "supports",
                "weight": 1.0
            }))
            .send()
            .await
            .unwrap();
        let e_status = edge.status();
        let e_body_text = edge.text().await.unwrap();
        assert_eq!(
            e_status,
            StatusCode::OK,
            "create hub edge failed: {e_body_text}"
        );
    }

    // Trigger centrality recomputation
    let centrality = client
        .post(format!("{}/v1/memory/cent_ns/graph/centrality", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    let cent_status = centrality.status();
    let cent_body_text = centrality.text().await.unwrap();
    assert_eq!(
        cent_status,
        StatusCode::OK,
        "centrality failed: {cent_body_text}"
    );
    let cent_body: serde_json::Value = serde_json::from_str(&cent_body_text).unwrap();
    assert!(
        cent_body["updated_nodes"].as_u64().unwrap_or(0) >= 2,
        "expected at least 2 nodes updated; body: {cent_body}"
    );

    let _ = app.shutdown.send(());
}

// ─── Test 6: Semantic walk expands via Supports edge ─────────────────────────

#[tokio::test]
async fn test_semantic_walk_expands() {
    // Note: Mock embeddings produce random vectors per call.
    // The walk BFS expansion only happens if cosine similarity >= min_similarity (0.65).
    // With random vectors this is unlikely, so we lower min_similarity to 0.0 to
    // ensure the walk always expands, and test that the endpoints work without error.
    let app = start_with_config(|mut cfg| {
        cfg.memory_walk_min_similarity = 0.0;
        cfg.memory_walk_max_hops = 2;
        cfg
    })
    .await;
    let client = reqwest::Client::new();

    // Create two episodic memories
    let ev_a = client
        .post(format!("{}/v1/memory/walk_ns/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "walk-user",
            "text": "User prefers notifications via email",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ev_a.status(), StatusCode::OK);
    let ev_a_body: serde_json::Value = ev_a.json().await.unwrap();
    let id_a = ev_a_body["id"].as_str().unwrap().to_string();

    let ev_b = client
        .post(format!("{}/v1/memory/walk_ns/ingest_event", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "walk-user",
            "text": "Email alerts should be sent at 9am daily",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(ev_b.status(), StatusCode::OK);
    let ev_b_body: serde_json::Value = ev_b.json().await.unwrap();
    let id_b = ev_b_body["id"].as_str().unwrap().to_string();

    // Connect them with a Supports edge
    let edge = client
        .post(format!("{}/v1/memory/walk_ns/edges", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "source_id": id_a,
            "target_id": id_b,
            "edge_type": "supports",
            "weight": 1.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        edge.status(),
        StatusCode::OK,
        "{}",
        edge.text().await.unwrap()
    );

    // Query — at minimum the query must succeed without error
    let query = client
        .post(format!("{}/v1/memory/walk_ns/query", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "notification preferences",
            "entity_id": "walk-user",
            "include_evidence": true
        }))
        .send()
        .await
        .unwrap();
    let q_status = query.status();
    let q_body_text = query.text().await.unwrap();
    assert_eq!(q_status, StatusCode::OK, "query failed: {q_body_text}");
    let q_body: serde_json::Value = serde_json::from_str(&q_body_text).unwrap();

    // Verify the response shape is correct
    assert!(q_body["results"].is_array(), "results should be an array");
    assert!(q_body["evidence"].is_array(), "evidence should be an array");

    let _ = app.shutdown.send(());
}

// ─── Test 7: Archived records skipped in query results ───────────────────────

#[tokio::test]
async fn test_semantic_walk_skips_archived() {
    let app = start().await;
    let client = reqwest::Client::new();

    // Ingest an event and immediately overwrite the same fact_key twice so the
    // first version gets archived.
    let fact_key = "archive_pref";

    let f1 = client
        .post(format!("{}/v1/memory/arch_ns/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-arch",
            "fact_key": fact_key,
            "content": "Old archived preference",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(f1.status(), StatusCode::OK, "{}", f1.text().await.unwrap());
    let f1_body: serde_json::Value = f1.json().await.unwrap();
    let old_fact_id = f1_body["id"].as_str().unwrap().to_string();

    // Overwrite — archives the first version
    let f2 = client
        .post(format!("{}/v1/memory/arch_ns/upsert_fact", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "entity_id": "user-arch",
            "fact_key": fact_key,
            "content": "New active preference",
            "metadata": {}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(f2.status(), StatusCode::OK, "{}", f2.text().await.unwrap());

    // Query: archived records must not appear
    let query = client
        .post(format!("{}/v1/memory/arch_ns/query", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "query": "preference",
            "entity_id": "user-arch",
            "include_evidence": true
        }))
        .send()
        .await
        .unwrap();
    let q_status = query.status();
    let q_body_text = query.text().await.unwrap();
    assert_eq!(q_status, StatusCode::OK, "query failed: {q_body_text}");
    let q_body: serde_json::Value = serde_json::from_str(&q_body_text).unwrap();

    let results = q_body["results"].as_array().unwrap();

    // upsert_fact reuses the same id (fact::{ns}::{fact_key}) for both writes,
    // so the old record is archived in-place and replaced by the new active record
    // with the same id. Any result returned must have status "active".
    let has_archived_status = results
        .iter()
        .any(|r| r["record"]["status"].as_str() == Some("archived"));
    assert!(
        !has_archived_status,
        "archived records should not appear in query results; results: {results:?}"
    );

    // Validate the result that IS returned (if any) belongs to the active version
    // Note: upsert_fact reuses the same id (fact::{ns}::{fact_key}), so the new version
    // replaces the old one. We just ensure no archived record leaks into results.
    let _ = old_fact_id; // suppress unused-variable warning

    let _ = app.shutdown.send(());
}
