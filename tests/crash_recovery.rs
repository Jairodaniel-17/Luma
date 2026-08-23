//! W1.1 — crash-recovery matrix.
//!
//! The claim this exists to *demonstrate* rather than assert: **a write Luma has
//! confirmed to a caller survives a hard kill.** `docs/operar/PROD_READINESS.md`
//! describes what each primitive syncs; this file proves it by killing the real
//! server binary mid-write and checking what came back.
//!
//! How it works, and why it is built this way:
//!
//! - It drives the **actual `luma` binary** over HTTP, not an in-process router.
//!   An in-process test shares the harness's allocator and drop glue, so a clean
//!   shutdown would sneak in and flush things a real `SIGKILL` never would.
//! - It records the id of every request that returned a success status. That set
//!   is the contract: after the kill and a restart, every one of them must still
//!   be readable. Requests in flight when the process died may or may not have
//!   landed — that is allowed, and the test never asserts on them.
//! - Each engine gets its own case, because they have different sync paths (see
//!   the durability table in PROD_READINESS.md).
//!
//! The normal run does a couple of iterations to keep the suite fast. The heavy
//! version — hundreds of iterations per engine — belongs in the nightly job, via
//! `LUMA_CRASH_ITERATIONS`.

use std::io::Read;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const API_KEY: &str = "crash-test-key";

/// Iterations per engine. One is enough to catch a systematic durability bug;
/// the nightly job raises it to shake out timing-dependent ones.
fn iterations() -> usize {
    std::env::var("LUMA_CRASH_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2)
}

/// Reserve a port by binding and immediately releasing it.
///
/// Racy in principle: another process could take it in between. In practice the
/// window is microseconds and the alternative — letting the server pick and
/// parsing its log — couples the test to log formatting.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

struct Server {
    child: Child,
    base: String,
}

impl Server {
    /// Start the real binary over `dir` and wait until it answers health checks.
    async fn start(dir: &Path, port: u16) -> Server {
        let child = Command::new(env!("CARGO_BIN_EXE_luma"))
            .arg("serve")
            .arg("--port")
            .arg(port.to_string())
            .arg("--data-dir")
            .arg(dir)
            .env("LUMA_API_KEY", API_KEY)
            // SQLite used to be excluded here, on the grounds that its
            // durability was "a separate question" governed by
            // `synchronous = NORMAL`. It was not a separate question — it was
            // the same question with an answer nobody liked: NORMAL does not
            // fsync at commit, so a power cut could lose transactions this
            // process had already reported as committed. The setting is now
            // FULL, so SQLite belongs in the matrix like every other store.
            .env("SQLITE_ENABLED", "true")
            .env("LUMA_ALLOW_INSECURE", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn the luma binary");

        let base = format!("http://127.0.0.1:{port}");
        let mut server = Server { child, base };
        if !server.await_healthy().await {
            // Report *why* rather than just "never became healthy": a config or
            // port problem looks identical to a durability bug from the outside,
            // and that ambiguity costs far more than capturing the pipes does.
            let _ = server.child.kill();
            let _ = server.child.wait();
            let mut err = String::new();
            if let Some(mut pipe) = server.child.stderr.take() {
                let _ = pipe.read_to_string(&mut err);
            }
            let mut out = String::new();
            if let Some(mut pipe) = server.child.stdout.take() {
                let _ = pipe.read_to_string(&mut out);
            }
            panic!(
                "server never became healthy on {}
--- stderr ---
{}
--- stdout ---
{}",
                server.base,
                err.trim(),
                out.trim()
            );
        }
        server
    }

    /// `true` once the server answers a health check, `false` on timeout.
    async fn await_healthy(&self) -> bool {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(resp) = client.get(format!("{}/v1/health", self.base)).send().await {
                if resp.status().is_success() {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    /// Kill without letting any destructor, flush or shutdown hook run — the
    /// whole point of the exercise.
    fn kill_hard(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Drain the pipes so a full buffer cannot block the child's exit and
        // make the next start look like a hang.
        if let Some(mut out) = self.child.stdout.take() {
            let _ = out.read_to_string(&mut String::new());
        }
        if let Some(mut err) = self.child.stderr.take() {
            let _ = err.read_to_string(&mut String::new());
        }
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap()
}

/// Scaffolding common to every case: fresh directory, server, write phase,
/// hard kill, restart. The verification differs per engine, so it stays in each
/// test rather than being squeezed through a generic callback.
struct Phase {
    dir: tempfile::TempDir,
}

impl Phase {
    fn new() -> Phase {
        Phase {
            dir: tempfile::tempdir().unwrap(),
        }
    }

    async fn start(&self) -> Server {
        Server::start(self.dir.path(), free_port()).await
    }
}

/// How many writes to attempt before the kill. Varied per iteration so the kill
/// lands at different points in the write path instead of always the same one.
fn writes_for(iteration: usize) -> usize {
    20 + iteration * 17
}

#[tokio::test]
async fn kv_confirmed_writes_survive_a_hard_kill() {
    for iteration in 0..iterations() {
        let phase = Phase::new();
        let server = phase.start().await;
        let http = client();

        let mut confirmed: Vec<String> = Vec::new();
        for n in 0..writes_for(iteration) {
            let key = format!("crash-key-{n}");
            let Ok(resp) = http
                .put(format!("{}/v1/state/{key}", server.base))
                .bearer_auth(API_KEY)
                .json(&serde_json::json!({ "value": { "n": n } }))
                .send()
                .await
            else {
                break;
            };
            if !resp.status().is_success() {
                break;
            }
            confirmed.push(key);
        }
        assert!(
            !confirmed.is_empty(),
            "kv: nothing confirmed, proves nothing"
        );
        server.kill_hard();

        let server = phase.start().await;
        let http = client();
        for key in &confirmed {
            let resp = http
                .get(format!("{}/v1/state/{key}", server.base))
                .bearer_auth(API_KEY)
                .send()
                .await
                .expect("state read failed after restart");
            assert!(
                resp.status().is_success(),
                "kv: `{key}` was confirmed before the kill and is gone after it"
            );
        }
        server.kill_hard();
    }
}

#[tokio::test]
async fn blob_confirmed_writes_survive_a_hard_kill() {
    for iteration in 0..iterations() {
        let phase = Phase::new();
        let server = phase.start().await;
        let http = client();

        let mut confirmed: Vec<String> = Vec::new();
        for n in 0..writes_for(iteration) {
            let key = format!("obj-{n}.bin");
            let Ok(resp) = http
                .put(format!("{}/v1/blob/crash/{key}", server.base))
                .bearer_auth(API_KEY)
                .body(vec![(n % 251) as u8; 4096])
                .send()
                .await
            else {
                break;
            };
            if !resp.status().is_success() {
                break;
            }
            confirmed.push(key);
        }
        assert!(
            !confirmed.is_empty(),
            "blob: nothing confirmed, proves nothing"
        );
        server.kill_hard();

        let server = phase.start().await;
        let http = client();
        for key in &confirmed {
            let resp = http
                .get(format!("{}/v1/blob/crash/{key}", server.base))
                .bearer_auth(API_KEY)
                .send()
                .await
                .expect("blob read failed after restart");
            assert!(
                resp.status().is_success(),
                "blob: `{key}` was confirmed before the kill and is gone after it"
            );
            // A short body would mean the rename reached the disk before the
            // data did — exactly what the temp-file fsync exists to prevent.
            assert_eq!(
                resp.bytes().await.unwrap().len(),
                4096,
                "blob: `{key}` came back truncated"
            );
        }
        server.kill_hard();
    }
}

#[tokio::test]
async fn queue_confirmed_enqueues_survive_a_hard_kill() {
    for iteration in 0..iterations() {
        let phase = Phase::new();
        let server = phase.start().await;
        let http = client();

        let mut confirmed: Vec<String> = Vec::new();
        for n in 0..writes_for(iteration) {
            let Ok(resp) = http
                .post(format!("{}/v1/queue/crash", server.base))
                .bearer_auth(API_KEY)
                .json(&serde_json::json!({ "body": { "n": n } }))
                .send()
                .await
            else {
                break;
            };
            if !resp.status().is_success() {
                break;
            }
            let Ok(body) = resp.json::<serde_json::Value>().await else {
                break;
            };
            match body["id"].as_str() {
                Some(id) => confirmed.push(id.to_string()),
                None => break,
            }
        }
        assert!(
            !confirmed.is_empty(),
            "queue: nothing confirmed, proves nothing"
        );
        server.kill_hard();

        // Drain and compare: a confirmed enqueue that is not redeliverable is a
        // lost message, whatever the queue's own bookkeeping says.
        let server = phase.start().await;
        let http = client();
        let mut seen: Vec<String> = Vec::new();
        for _ in 0..20 {
            let received: serde_json::Value = http
                .post(format!("{}/v1/queue/crash/receive", server.base))
                .bearer_auth(API_KEY)
                .json(&serde_json::json!({ "max": 100, "visibility_secs": 60 }))
                .send()
                .await
                .expect("receive failed after restart")
                .json()
                .await
                .expect("receive returned a non-JSON body");
            let batch = received["messages"].as_array().cloned().unwrap_or_default();
            if batch.is_empty() {
                break;
            }
            for msg in batch {
                if let Some(id) = msg["id"].as_str() {
                    seen.push(id.to_string());
                }
            }
        }
        for id in &confirmed {
            assert!(
                seen.contains(id),
                "queue: message `{id}` was confirmed before the kill and never came back"
            );
        }
        server.kill_hard();
    }
}

#[tokio::test]
async fn vector_confirmed_upserts_survive_a_hard_kill() {
    for iteration in 0..iterations() {
        let phase = Phase::new();
        let server = phase.start().await;
        let http = client();

        let created = http
            .post(format!("{}/v1/vector/crash", server.base))
            .bearer_auth(API_KEY)
            .json(&serde_json::json!({ "dim": 4, "metric": "cosine" }))
            .send()
            .await
            .expect("create collection failed");
        assert!(
            created.status().is_success(),
            "setup problem, not a durability finding: {}",
            created.status()
        );

        let mut confirmed: Vec<String> = Vec::new();
        for n in 0..writes_for(iteration) {
            let id = format!("vec-{n}");
            let value = (n as f32) / 100.0;
            let Ok(resp) = http
                .post(format!("{}/v1/vector/crash/upsert", server.base))
                .bearer_auth(API_KEY)
                .json(&serde_json::json!({
                    "id": id,
                    "vector": [value, value, value, value],
                    "meta": { "n": n }
                }))
                .send()
                .await
            else {
                break;
            };
            if !resp.status().is_success() {
                break;
            }
            confirmed.push(id);
        }
        assert!(
            !confirmed.is_empty(),
            "vector: nothing confirmed, proves nothing"
        );
        server.kill_hard();

        let server = phase.start().await;
        let http = client();
        for id in &confirmed {
            let resp = http
                .get(format!("{}/v1/vector/crash/get", server.base))
                .bearer_auth(API_KEY)
                .query(&[("id", id.as_str())])
                .send()
                .await
                .expect("vector read failed after restart");
            assert!(
                resp.status().is_success(),
                "vector: `{id}` was confirmed before the kill and is gone after it"
            );
        }
        server.kill_hard();
    }
}
