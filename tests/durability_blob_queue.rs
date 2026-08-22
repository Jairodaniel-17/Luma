//! Durability of the two primitives that commit straight to the filesystem.
//!
//! W1.2 of `docs/PLAN-MAESTRO.md`. Blob PUT and queue enqueue used to do
//! `write` + `rename`: atomic but not durable, since both the file contents and
//! the directory entry could still be only in the page cache. These tests pin
//! the observable contract — a confirmed write is on disk and survives the
//! process that wrote it — plus the invariants that the temp-file dance must
//! not break.
//!
//! Killing the process mid-write is a different, harder question and belongs to
//! the W1.1 crash-recovery matrix.

use axum::http::StatusCode;
use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

struct TestApp {
    base: String,
    shutdown: oneshot::Sender<()>,
    serve: tokio::task::JoinHandle<()>,
    token: CancellationToken,
}

impl TestApp {
    /// Stop the server *and* release the engine.
    ///
    /// Dropping the HTTP listener is not enough: the router holds an `Engine`
    /// clone, and redb refuses a second open while any handle is alive. So the
    /// background tasks are cancelled and the serve future is awaited to
    /// completion, which is what actually drops the state.
    async fn stop(self) {
        let _ = self.shutdown.send(());
        self.token.cancel();
        let _ = self.serve.await;
    }
}

/// Start a server over an existing directory, so a test can stop it and start a
/// fresh one on the same `data_dir` — the cheapest honest way to ask "did this
/// actually reach the disk?".
async fn start_on(dir: &Path) -> TestApp {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string_lossy().to_string()),
        sqlite_enabled: false,
        snapshot_interval_secs: 3600,
        ..Config::default()
    };

    let token = CancellationToken::new();
    // A previous phase's engine may still be winding down; redb reports that as
    // a lock error. Retry briefly so the test measures durability rather than
    // shutdown timing, but keep the bound tight so a handle that never releases
    // still fails.
    let engine = {
        let mut attempt = 0;
        loop {
            match Engine::new(config.clone(), token.clone()) {
                Ok(engine) => break engine,
                Err(e) if attempt < 50 => {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    if attempt == 50 {
                        panic!("engine never became openable: {e}");
                    }
                }
                Err(e) => panic!("engine open failed: {e}"),
            }
        }
    };
    let search_engine = Arc::new(SearchEngine::new(dir.to_path_buf()).unwrap());
    let embeddings = luma::engine::embeddings::EmbeddingHandle::new(
        luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 8 },
        ),
    );

    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: None,
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
    let serve = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    TestApp {
        base: format!("http://{}", addr),
        shutdown,
        serve,
        token,
    }
}

/// Any `.tmp-*` still present means a durable write leaked its scratch file.
fn temp_leftovers(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(".tmp-"))
            {
                found.push(path);
            }
        }
    }
    found
}

#[tokio::test]
async fn blob_survives_a_restart_and_leaves_no_temp_files() {
    let dir = tempfile::tempdir().unwrap();
    let client = reqwest::Client::new();
    let payload: Vec<u8> = (0u8..=255).cycle().take(64 * 1024).collect();

    {
        let app = start_on(dir.path()).await;
        let resp = client
            .put(format!("{}/v1/blob/assets/nested/logo.bin", app.base))
            .bearer_auth("test")
            .body(payload.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["size"].as_u64().unwrap(), payload.len() as u64);

        // The object is on disk under the expected path, not merely in a buffer.
        let on_disk = dir.path().join("blobs/assets/nested/logo.bin");
        assert!(on_disk.exists(), "blob not written to {on_disk:?}");
        assert_eq!(std::fs::read(&on_disk).unwrap(), payload);

        app.stop().await;
    }

    assert!(
        temp_leftovers(dir.path()).is_empty(),
        "durable write left temp files: {:?}",
        temp_leftovers(dir.path())
    );

    // A brand-new process over the same data_dir must still serve it.
    let app = start_on(dir.path()).await;
    let resp = client
        .get(format!("{}/v1/blob/assets/nested/logo.bin", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.bytes().await.unwrap().to_vec(), payload);
    app.stop().await;
}

#[tokio::test]
async fn blob_overwrite_replaces_content_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let app = start_on(dir.path()).await;
    let client = reqwest::Client::new();

    for body in [b"first-write".to_vec(), b"second".to_vec()] {
        let resp = client
            .put(format!("{}/v1/blob/assets/k", app.base))
            .bearer_auth("test")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Shorter second write must replace, not truncate-in-place leaving a
        // tail of the first — the rename is what guarantees that.
        let got = client
            .get(format!("{}/v1/blob/assets/k", app.base))
            .bearer_auth("test")
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap()
            .to_vec();
        assert_eq!(got, body);
    }

    app.stop().await;
}

#[tokio::test]
async fn enqueued_message_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let client = reqwest::Client::new();

    let message_id = {
        let app = start_on(dir.path()).await;
        let resp = client
            .post(format!("{}/v1/queue/jobs", app.base))
            .bearer_auth("test")
            .json(&serde_json::json!({ "body": { "task": "reindex", "id": 42 } }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: serde_json::Value = resp.json().await.unwrap();
        let id = body["id"].as_str().unwrap().to_string();

        let stats: serde_json::Value = client
            .get(format!("{}/v1/queue/jobs", app.base))
            .bearer_auth("test")
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(stats["depth"].as_u64().unwrap(), 1);

        app.stop().await;
        id
    };

    assert!(
        temp_leftovers(dir.path()).is_empty(),
        "enqueue left temp files: {:?}",
        temp_leftovers(dir.path())
    );

    // The confirmed enqueue must be there for a new process: this is the
    // property the missing fsync silently broke.
    let app = start_on(dir.path()).await;
    let received: serde_json::Value = client
        .post(format!("{}/v1/queue/jobs/receive", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "max": 10, "visibility_secs": 30 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let messages = received["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1, "message lost across restart: {received}");
    assert_eq!(messages[0]["id"].as_str().unwrap(), message_id);
    assert_eq!(messages[0]["body"]["task"], "reindex");

    // Ack removes it for good, so a further restart finds nothing.
    let ack = client
        .delete(format!("{}/v1/queue/jobs/{}", app.base, message_id))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert_eq!(ack.status(), StatusCode::OK);
    app.stop().await;

    let app = start_on(dir.path()).await;
    let stats: serde_json::Value = client
        .get(format!("{}/v1/queue/jobs", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stats["depth"].as_u64().unwrap(), 0, "ack was not durable");
    app.stop().await;
}

#[tokio::test]
async fn unacked_message_becomes_visible_again() {
    let dir = tempfile::tempdir().unwrap();
    let app = start_on(dir.path()).await;
    let client = reqwest::Client::new();

    client
        .post(format!("{}/v1/queue/retry", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "body": { "n": 1 } }))
        .send()
        .await
        .unwrap();

    // Lease with a visibility window of 0 so it expires immediately: the
    // at-least-once contract says an unacked message comes back, with its
    // attempt count incremented. That counter is what lets a consumer spot a
    // poison message instead of looping on it forever.
    let first: serde_json::Value = client
        .post(format!("{}/v1/queue/retry/receive", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "max": 1, "visibility_secs": 0 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(first["messages"].as_array().unwrap().len(), 1);
    let first_attempts = first["messages"][0]["attempts"].as_u64().unwrap();

    let second: serde_json::Value = client
        .post(format!("{}/v1/queue/retry/receive", app.base))
        .bearer_auth("test")
        .json(&serde_json::json!({ "max": 1, "visibility_secs": 30 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let redelivered = second["messages"].as_array().unwrap();
    assert_eq!(redelivered.len(), 1, "unacked message was not redelivered");
    assert!(
        redelivered[0]["attempts"].as_u64().unwrap() > first_attempts,
        "redelivery must increment attempts so a consumer can detect a poison message"
    );

    app.stop().await;
}

// ─── F0.1: raw byte values in the KV store ───────────────────────────────────

#[tokio::test]
async fn raw_kv_value_roundtrips_and_survives_a_restart() {
    use base64::Engine as _;

    let dir = tempfile::tempdir().unwrap();
    let client = reqwest::Client::new();
    // Bytes that cannot appear in a JSON string: the case the Value-only store
    // could not represent at all.
    let payload: Vec<u8> = vec![0x00, 0xFF, 0xFE, 0x80, 0x7F, 0x01];
    let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);

    {
        let app = start_on(dir.path()).await;
        let resp = client
            .put(format!("{}/v1/state/binary-key", app.base))
            .bearer_auth("test")
            .json(&serde_json::json!({
                "value": { "__luma_raw": encoded, "content_type": "application/octet-stream" }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // A plain JSON value under a different key must be unaffected — the
        // marker is what distinguishes them, nothing else.
        let json_resp = client
            .put(format!("{}/v1/state/json-key", app.base))
            .bearer_auth("test")
            .json(&serde_json::json!({ "value": { "content_type": "still a document" } }))
            .send()
            .await
            .unwrap();
        assert_eq!(json_resp.status(), StatusCode::OK);
        app.stop().await;
    }

    // Restart: the bytes must come back through WAL replay / redb, not just
    // from memory.
    let app = start_on(dir.path()).await;
    let got: serde_json::Value = client
        .get(format!("{}/v1/state/binary-key", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        got["value"]["__luma_raw"].as_str(),
        Some(encoded.as_str()),
        "raw bytes must round-trip symmetrically after a restart: {got}"
    );
    assert_eq!(
        got["value"]["content_type"].as_str(),
        Some("application/octet-stream")
    );

    let json_got: serde_json::Value = client
        .get(format!("{}/v1/state/json-key", app.base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        json_got["value"]["content_type"].as_str(),
        Some("still a document"),
        "a document that merely has a content_type field must stay a document"
    );

    app.stop().await;
}
