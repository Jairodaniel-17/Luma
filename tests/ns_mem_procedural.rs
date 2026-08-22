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
    let db_path = dir.path().join("proc_test.db");
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
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 4 },
        ),
    );
    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: Some(sqlite),
        search_engine,
        auth_store: None,
        embeddings,
        resp_metrics: None,
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

async fn upsert(app: &TestApp, body: serde_json::Value) -> serde_json::Value {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/memory/ns/upsert_procedure", app.base))
        .bearer_auth("test")
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "upsert failed: {text}");
    serde_json::from_str(&text).unwrap()
}

async fn next_step(app: &TestApp, body: serde_json::Value) -> serde_json::Value {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/memory/ns/next_step", app.base))
        .bearer_auth("test")
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "next_step failed: {text}");
    serde_json::from_str(&text).unwrap()
}

// ───────────────────────────────────────────────────────────
// Happy path: linear DAG with no conditions
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn linear_dag_steps_through_all_nodes() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "linear",
            "name": "Linear",
            "nodes": [
                { "node_id": "s", "kind": "start",  "label": "Start"  },
                { "node_id": "a", "kind": "action", "label": "Action" },
                { "node_id": "g", "kind": "goal",   "label": "Goal"   }
            ],
            "edges": [
                { "from_node_id": "s", "to_node_id": "a", "priority": 0 },
                { "from_node_id": "a", "to_node_id": "g", "priority": 0 }
            ],
            "constraints": []
        }),
    )
    .await;

    let step1 = next_step(
        &app,
        serde_json::json!({ "procedure_id": "linear", "context": {} }),
    )
    .await;
    assert_eq!(step1["next_node"]["node_id"], "a");
    assert_eq!(step1["blocked_reason"], serde_json::Value::Null);

    let step2 = next_step(
        &app,
        serde_json::json!({ "procedure_id": "linear", "current_node_id": "a", "context": {} }),
    )
    .await;
    assert_eq!(step2["next_node"]["node_id"], "g");

    // Goal node has no outgoing edges — should be blocked
    let blocked = next_step(
        &app,
        serde_json::json!({ "procedure_id": "linear", "current_node_id": "g", "context": {} }),
    )
    .await;
    assert!(
        !blocked["blocked_reason"].is_null(),
        "expected blocked at terminal node"
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Conditional branching: decision node → two paths
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn conditional_branching_takes_correct_path() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "branch",
            "name": "Branch",
            "nodes": [
                { "node_id": "s",    "kind": "start",    "label": "Start"    },
                { "node_id": "dec",  "kind": "decision", "label": "Decide"   },
                { "node_id": "high", "kind": "action",   "label": "High"     },
                { "node_id": "low",  "kind": "action",   "label": "Low"      }
            ],
            "edges": [
                { "from_node_id": "s",   "to_node_id": "dec",  "priority": 0 },
                {
                    "from_node_id": "dec",
                    "to_node_id": "high",
                    "priority": 10,
                    "condition": { "field": "amount", "op": "gt", "value": 100.0 }
                },
                {
                    "from_node_id": "dec",
                    "to_node_id": "low",
                    "priority": 5,
                    "condition": { "field": "amount", "op": "lte", "value": 100.0 }
                }
            ],
            "constraints": []
        }),
    )
    .await;

    // Advance to decision node
    next_step(
        &app,
        serde_json::json!({ "procedure_id": "branch", "context": {} }),
    )
    .await;

    let high_path = next_step(
        &app,
        serde_json::json!({ "procedure_id": "branch", "current_node_id": "dec", "context": { "amount": 200 } }),
    )
    .await;
    assert_eq!(high_path["next_node"]["node_id"], "high");

    let low_path = next_step(
        &app,
        serde_json::json!({ "procedure_id": "branch", "current_node_id": "dec", "context": { "amount": 50 } }),
    )
    .await;
    assert_eq!(low_path["next_node"]["node_id"], "low");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Edge priority: higher priority edge wins when both match
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn edge_priority_selects_higher_priority_first() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "priority",
            "name": "Priority",
            "nodes": [
                { "node_id": "s",    "kind": "start",  "label": "Start"   },
                { "node_id": "fast", "kind": "action", "label": "Fast"    },
                { "node_id": "slow", "kind": "action", "label": "Slow"    }
            ],
            "edges": [
                { "from_node_id": "s", "to_node_id": "slow", "priority": 1  },
                { "from_node_id": "s", "to_node_id": "fast", "priority": 99 }
            ],
            "constraints": []
        }),
    )
    .await;

    let step = next_step(
        &app,
        serde_json::json!({ "procedure_id": "priority", "context": {} }),
    )
    .await;
    assert_eq!(
        step["next_node"]["node_id"], "fast",
        "higher priority edge should be selected"
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Node-specific constraint: hard block when condition fails
//
// Constraints are hard stops — when a constraint fails for a target node
// the whole step is blocked (no fallthrough to other edges).
// Use edge CONDITIONS (not constraints) to route to alternative nodes.
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn node_specific_constraint_hard_blocks_on_failure() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "node_constraint",
            "name": "NodeConstraint",
            "nodes": [
                { "node_id": "s", "kind": "start",  "label": "Start"    },
                { "node_id": "a", "kind": "action", "label": "Action A" }
            ],
            "edges": [
                { "from_node_id": "s", "to_node_id": "a", "priority": 0 }
            ],
            "constraints": [
                {
                    "constraint_id": "admin-only",
                    "target_node_id": "a",
                    "condition": { "field": "role", "op": "eq", "value": "admin" },
                    "message": "admin required for a"
                }
            ]
        }),
    )
    .await;

    // Non-admin: constraint on "a" fails → hard block with message
    let blocked = next_step(
        &app,
        serde_json::json!({ "procedure_id": "node_constraint", "context": { "role": "user" } }),
    )
    .await;
    assert_eq!(blocked["blocked_reason"], "admin required for a");
    assert!(blocked["next_node"].is_null());

    // Admin: constraint passes → reaches "a"
    let allowed = next_step(
        &app,
        serde_json::json!({ "procedure_id": "node_constraint", "context": { "role": "admin" } }),
    )
    .await;
    assert_eq!(allowed["next_node"]["node_id"], "a");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Edge CONDITIONS (not constraints) provide fallthrough routing:
// if an edge condition fails, the engine skips to the next edge.
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn edge_condition_failure_falls_through_to_next_edge() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "fallthrough",
            "name": "Fallthrough",
            "nodes": [
                { "node_id": "s",        "kind": "start",  "label": "Start"    },
                { "node_id": "admin_path","kind": "action", "label": "Admin"   },
                { "node_id": "user_path", "kind": "action", "label": "User"    }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "admin_path", "priority": 10,
                    "condition": { "field": "role", "op": "eq", "value": "admin" }
                },
                {
                    "from_node_id": "s", "to_node_id": "user_path", "priority": 5
                }
            ],
            "constraints": []
        }),
    )
    .await;

    // Non-admin: admin edge condition fails → falls through to user_path
    let user = next_step(
        &app,
        serde_json::json!({ "procedure_id": "fallthrough", "context": { "role": "user" } }),
    )
    .await;
    assert_eq!(user["next_node"]["node_id"], "user_path");

    // Admin: admin edge condition passes → takes admin_path
    let admin = next_step(
        &app,
        serde_json::json!({ "procedure_id": "fallthrough", "context": { "role": "admin" } }),
    )
    .await;
    assert_eq!(admin["next_node"]["node_id"], "admin_path");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Global constraint (no target_node_id) blocks all transitions
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn global_constraint_blocks_all_transitions() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "global_constraint",
            "name": "GlobalConstraint",
            "nodes": [
                { "node_id": "s", "kind": "start",  "label": "Start"  },
                { "node_id": "a", "kind": "action", "label": "Action" }
            ],
            "edges": [
                { "from_node_id": "s", "to_node_id": "a", "priority": 0 }
            ],
            "constraints": [
                {
                    "constraint_id": "system-active",
                    "condition": { "field": "system.active", "op": "eq", "value": true },
                    "message": "system must be active"
                }
            ]
        }),
    )
    .await;

    // System not active — global constraint fails, all transitions blocked
    let blocked = next_step(
        &app,
        serde_json::json!({
            "procedure_id": "global_constraint",
            "context": { "system": { "active": false } }
        }),
    )
    .await;
    assert_eq!(blocked["blocked_reason"], "system must be active");
    assert!(blocked["next_node"].is_null());

    // System active — proceeds normally
    let allowed = next_step(
        &app,
        serde_json::json!({
            "procedure_id": "global_constraint",
            "context": { "system": { "active": true } }
        }),
    )
    .await;
    assert_eq!(allowed["next_node"]["node_id"], "a");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Procedure with no Start node
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn no_start_node_returns_blocked_with_reason() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "no_start",
            "name": "NoStart",
            "nodes": [
                { "node_id": "a", "kind": "action", "label": "Action" }
            ],
            "edges": [],
            "constraints": []
        }),
    )
    .await;

    // No current_node_id and no start node → blocked
    let resp = next_step(
        &app,
        serde_json::json!({ "procedure_id": "no_start", "context": {} }),
    )
    .await;
    assert!(resp["blocked_reason"]
        .as_str()
        .unwrap_or("")
        .contains("start"));

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Unknown procedure returns error
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_procedure_returns_error() {
    let app = start().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/v1/memory/ns/next_step", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "procedure_id": "does_not_exist", "context": {} }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected error status for unknown procedure, got {}",
        resp.status()
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Archived procedure: not loaded by next_step
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn archived_procedure_is_not_executed() {
    let app = start().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/v1/memory/ns/upsert_procedure", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "procedure_id": "archived_proc",
            "name": "Archived",
            "status": "archived",
            "nodes": [
                { "node_id": "s", "kind": "start",  "label": "Start" },
                { "node_id": "a", "kind": "action", "label": "Act"   }
            ],
            "edges": [{ "from_node_id": "s", "to_node_id": "a", "priority": 0 }],
            "constraints": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let step = client
        .post(format!("{}/v1/memory/ns/next_step", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "procedure_id": "archived_proc", "context": {} }))
        .send()
        .await
        .unwrap();

    // Archived procedure should not be found → error response
    assert!(
        step.status().is_client_error() || step.status().is_server_error(),
        "expected error for archived procedure, got {}",
        step.status()
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Draft procedure: not loaded by next_step
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn draft_procedure_is_not_executed() {
    let app = start().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/v1/memory/ns/upsert_procedure", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({
            "procedure_id": "draft_proc",
            "name": "Draft",
            "status": "draft",
            "nodes": [
                { "node_id": "s", "kind": "start",  "label": "Start" },
                { "node_id": "a", "kind": "action", "label": "Act"   }
            ],
            "edges": [{ "from_node_id": "s", "to_node_id": "a", "priority": 0 }],
            "constraints": []
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let step = client
        .post(format!("{}/v1/memory/ns/next_step", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "procedure_id": "draft_proc", "context": {} }))
        .send()
        .await
        .unwrap();

    assert!(
        step.status().is_client_error() || step.status().is_server_error(),
        "expected error for draft procedure, got {}",
        step.status()
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Upsert replaces the procedure completely
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn upsert_replaces_previous_definition() {
    let app = start().await;

    // V1: start → old_node
    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "replace_me",
            "name": "V1",
            "nodes": [
                { "node_id": "s",        "kind": "start",  "label": "Start"    },
                { "node_id": "old_node", "kind": "action", "label": "Old node" }
            ],
            "edges": [{ "from_node_id": "s", "to_node_id": "old_node", "priority": 0 }],
            "constraints": []
        }),
    )
    .await;

    // V2: start → new_node (old_node no longer exists)
    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "replace_me",
            "name": "V2",
            "nodes": [
                { "node_id": "s",        "kind": "start",  "label": "Start"    },
                { "node_id": "new_node", "kind": "action", "label": "New node" }
            ],
            "edges": [{ "from_node_id": "s", "to_node_id": "new_node", "priority": 0 }],
            "constraints": []
        }),
    )
    .await;

    let step = next_step(
        &app,
        serde_json::json!({ "procedure_id": "replace_me", "context": {} }),
    )
    .await;
    assert_eq!(
        step["next_node"]["node_id"], "new_node",
        "old definition should be replaced"
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Numeric operators: gte, lte, lt, gt
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn numeric_edge_conditions_are_evaluated_correctly() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "numeric_ops",
            "name": "NumericOps",
            "nodes": [
                { "node_id": "s",   "kind": "start",  "label": "Start"  },
                { "node_id": "gte", "kind": "action", "label": "GTE 10" },
                { "node_id": "lt",  "kind": "action", "label": "LT 10"  }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "gte", "priority": 10,
                    "condition": { "field": "v", "op": "gte", "value": 10.0 }
                },
                {
                    "from_node_id": "s", "to_node_id": "lt", "priority": 5,
                    "condition": { "field": "v", "op": "lt", "value": 10.0 }
                }
            ],
            "constraints": []
        }),
    )
    .await;

    for (v, expected) in [(10, "gte"), (15, "gte"), (9, "lt"), (0, "lt")] {
        let step = next_step(
            &app,
            serde_json::json!({ "procedure_id": "numeric_ops", "context": { "v": v } }),
        )
        .await;
        assert_eq!(
            step["next_node"]["node_id"], expected,
            "v={v} should route to {expected}"
        );
    }

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Operator: neq
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn neq_condition_routes_correctly() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "neq_test",
            "name": "NeqTest",
            "nodes": [
                { "node_id": "s",      "kind": "start",  "label": "Start"      },
                { "node_id": "not_a",  "kind": "action", "label": "Not A"      },
                { "node_id": "is_a",   "kind": "action", "label": "Is A"       }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "not_a", "priority": 10,
                    "condition": { "field": "val", "op": "neq", "value": "a" }
                },
                {
                    "from_node_id": "s", "to_node_id": "is_a", "priority": 5,
                    "condition": { "field": "val", "op": "eq", "value": "a" }
                }
            ],
            "constraints": []
        }),
    )
    .await;

    let not_a = next_step(
        &app,
        serde_json::json!({ "procedure_id": "neq_test", "context": { "val": "b" } }),
    )
    .await;
    assert_eq!(not_a["next_node"]["node_id"], "not_a");

    let is_a = next_step(
        &app,
        serde_json::json!({ "procedure_id": "neq_test", "context": { "val": "a" } }),
    )
    .await;
    assert_eq!(is_a["next_node"]["node_id"], "is_a");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Operator: in (check if value is in array)
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn in_operator_matches_array_membership() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "in_test",
            "name": "InTest",
            "nodes": [
                { "node_id": "s",     "kind": "start",  "label": "Start"  },
                { "node_id": "vip",   "kind": "action", "label": "VIP"    },
                { "node_id": "other", "kind": "action", "label": "Other"  }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "vip", "priority": 10,
                    "condition": { "field": "tier", "op": "in", "value": ["gold", "platinum"] }
                },
                {
                    "from_node_id": "s", "to_node_id": "other", "priority": 0
                }
            ],
            "constraints": []
        }),
    )
    .await;

    let vip = next_step(
        &app,
        serde_json::json!({ "procedure_id": "in_test", "context": { "tier": "gold" } }),
    )
    .await;
    assert_eq!(vip["next_node"]["node_id"], "vip");

    let other = next_step(
        &app,
        serde_json::json!({ "procedure_id": "in_test", "context": { "tier": "bronze" } }),
    )
    .await;
    assert_eq!(other["next_node"]["node_id"], "other");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Operator: contains (substring check)
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn contains_operator_matches_substring() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "contains_test",
            "name": "ContainsTest",
            "nodes": [
                { "node_id": "s",      "kind": "start",  "label": "Start"  },
                { "node_id": "match",  "kind": "action", "label": "Match"  },
                { "node_id": "nomatch","kind": "action", "label": "No"     }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "match", "priority": 10,
                    "condition": { "field": "msg", "op": "contains", "value": "error" }
                },
                { "from_node_id": "s", "to_node_id": "nomatch", "priority": 0 }
            ],
            "constraints": []
        }),
    )
    .await;

    let matched = next_step(
        &app,
        serde_json::json!({ "procedure_id": "contains_test", "context": { "msg": "fatal error occurred" } }),
    )
    .await;
    assert_eq!(matched["next_node"]["node_id"], "match");

    let not_matched = next_step(
        &app,
        serde_json::json!({ "procedure_id": "contains_test", "context": { "msg": "all good" } }),
    )
    .await;
    assert_eq!(not_matched["next_node"]["node_id"], "nomatch");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Missing context field: condition evaluates to false → edge skipped
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn missing_context_field_skips_conditional_edge() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "missing_field",
            "name": "MissingField",
            "nodes": [
                { "node_id": "s",        "kind": "start",  "label": "Start"     },
                { "node_id": "guarded",  "kind": "action", "label": "Guarded"  },
                { "node_id": "fallback", "kind": "action", "label": "Fallback" }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "guarded", "priority": 10,
                    "condition": { "field": "nonexistent.field", "op": "eq", "value": "x" }
                },
                { "from_node_id": "s", "to_node_id": "fallback", "priority": 0 }
            ],
            "constraints": []
        }),
    )
    .await;

    let step = next_step(
        &app,
        serde_json::json!({ "procedure_id": "missing_field", "context": {} }),
    )
    .await;
    assert_eq!(
        step["next_node"]["node_id"], "fallback",
        "missing field should cause conditional edge to be skipped"
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Diamond DAG: fork and rejoin
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn diamond_dag_both_paths_converge_at_goal() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "diamond",
            "name": "Diamond",
            "nodes": [
                { "node_id": "s",    "kind": "start",  "label": "Start"  },
                { "node_id": "left", "kind": "action", "label": "Left"   },
                { "node_id": "right","kind": "action", "label": "Right"  },
                { "node_id": "g",    "kind": "goal",   "label": "Goal"   }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "left", "priority": 10,
                    "condition": { "field": "path", "op": "eq", "value": "left" }
                },
                {
                    "from_node_id": "s", "to_node_id": "right", "priority": 5,
                    "condition": { "field": "path", "op": "eq", "value": "right" }
                },
                { "from_node_id": "left",  "to_node_id": "g", "priority": 0 },
                { "from_node_id": "right", "to_node_id": "g", "priority": 0 }
            ],
            "constraints": []
        }),
    )
    .await;

    for chosen_path in ["left", "right"] {
        let fork = next_step(
            &app,
            serde_json::json!({ "procedure_id": "diamond", "context": { "path": chosen_path } }),
        )
        .await;
        assert_eq!(
            fork["next_node"]["node_id"].as_str().unwrap(),
            chosen_path,
            "should fork to {chosen_path}"
        );

        let merge = next_step(
            &app,
            serde_json::json!({
                "procedure_id": "diamond",
                "current_node_id": chosen_path,
                "context": {}
            }),
        )
        .await;
        assert_eq!(merge["next_node"]["node_id"], "g");
    }

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Constraint message falls back to default when not set
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn constraint_without_message_returns_default_reason() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "no_msg",
            "name": "NoMsg",
            "nodes": [
                { "node_id": "s", "kind": "start",  "label": "Start" },
                { "node_id": "a", "kind": "action", "label": "Act"   }
            ],
            "edges": [{ "from_node_id": "s", "to_node_id": "a", "priority": 0 }],
            "constraints": [
                {
                    "constraint_id": "no-msg-constraint",
                    "target_node_id": "a",
                    "condition": { "field": "ok", "op": "eq", "value": true }
                }
            ]
        }),
    )
    .await;

    let blocked = next_step(
        &app,
        serde_json::json!({ "procedure_id": "no_msg", "context": { "ok": false } }),
    )
    .await;
    assert!(
        !blocked["blocked_reason"].is_null(),
        "blocked_reason should be present even without a custom message"
    );

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Explicit current_node_id overrides start node lookup
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn explicit_current_node_id_is_used_instead_of_start() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "explicit_node",
            "name": "ExplicitNode",
            "nodes": [
                { "node_id": "s",   "kind": "start",  "label": "Start"    },
                { "node_id": "mid", "kind": "action", "label": "Mid"      },
                { "node_id": "end", "kind": "action", "label": "End"      }
            ],
            "edges": [
                { "from_node_id": "s",   "to_node_id": "mid", "priority": 0 },
                { "from_node_id": "mid", "to_node_id": "end", "priority": 0 }
            ],
            "constraints": []
        }),
    )
    .await;

    // Jump directly to "mid" without going through start
    let step = next_step(
        &app,
        serde_json::json!({
            "procedure_id": "explicit_node",
            "current_node_id": "mid",
            "context": {}
        }),
    )
    .await;
    assert_eq!(step["next_node"]["node_id"], "end");

    let _ = app.shutdown.send(());
}

// ───────────────────────────────────────────────────────────
// Dot-notation context path traversal
// ───────────────────────────────────────────────────────────

#[tokio::test]
async fn deep_nested_context_field_is_resolved_correctly() {
    let app = start().await;

    upsert(
        &app,
        serde_json::json!({
            "procedure_id": "deep_ctx",
            "name": "DeepCtx",
            "nodes": [
                { "node_id": "s",   "kind": "start",  "label": "Start" },
                { "node_id": "hit", "kind": "action", "label": "Hit"   },
                { "node_id": "miss","kind": "action", "label": "Miss"  }
            ],
            "edges": [
                {
                    "from_node_id": "s", "to_node_id": "hit", "priority": 10,
                    "condition": { "field": "a.b.c", "op": "eq", "value": "deep" }
                },
                { "from_node_id": "s", "to_node_id": "miss", "priority": 0 }
            ],
            "constraints": []
        }),
    )
    .await;

    let hit = next_step(
        &app,
        serde_json::json!({
            "procedure_id": "deep_ctx",
            "context": { "a": { "b": { "c": "deep" } } }
        }),
    )
    .await;
    assert_eq!(hit["next_node"]["node_id"], "hit");

    let miss = next_step(
        &app,
        serde_json::json!({
            "procedure_id": "deep_ctx",
            "context": { "a": { "b": { "c": "shallow" } } }
        }),
    )
    .await;
    assert_eq!(miss["next_node"]["node_id"], "miss");

    let _ = app.shutdown.send(());
}
