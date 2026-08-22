//! Forward compatibility of on-disk data.
//!
//! Rule 1 of the data compatibility policy in `docs/SPEC-producto.md`: **every
//! version reads what the previous one wrote.** Rule 4 says CI has to enforce
//! it, because a rule nothing checks is a wish.
//!
//! `tests/fixtures/golden_data_dir/` is a real `data_dir` produced by a build of
//! this project — WAL segment, snapshot, redb state file, vector manifest and
//! runs, a blob, a queued message. This test starts the current binary over a
//! copy of it and reads every record back. If a format changes in a way that
//! makes old data unreadable, this fails.
//!
//! **Maintaining it:** regenerate the fixture on each release, from that
//! release's binary, and commit it. Never edit it by hand and never regenerate
//! it just to make this test pass — a failure here means either the format
//! change needs a migration, or it does not belong in v1.

use std::io::Read;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The api key the fixture was created with. Irrelevant to the stored bytes —
/// auth is not persisted — but the requests below need one.
const API_KEY: &str = "golden-key";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_data_dir")
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Recursive copy, so the committed fixture is never mutated by a test run. A
/// test that writes to its own input stops being a fixture after the first run.
fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let target = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

struct Server {
    child: Child,
    base: String,
}

impl Server {
    async fn start(dir: &Path) -> Server {
        let port = free_port();
        let child = Command::new(env!("CARGO_BIN_EXE_luma"))
            .arg("serve")
            .arg("--port")
            .arg(port.to_string())
            .arg("--data-dir")
            .arg(dir)
            .env("LUMA_API_KEY", API_KEY)
            .env("SQLITE_ENABLED", "false")
            .env("LUMA_ALLOW_INSECURE", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn the luma binary");

        let mut server = Server {
            child,
            base: format!("http://127.0.0.1:{port}"),
        };

        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(resp) = client
                .get(format!("{}/v1/health", server.base))
                .send()
                .await
            {
                if resp.status().is_success() {
                    return server;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Failing to boot over old data is the most likely way this test breaks,
        // so surface the reason instead of a bare timeout.
        let _ = server.child.kill();
        let _ = server.child.wait();
        let mut err = String::new();
        if let Some(mut pipe) = server.child.stderr.take() {
            let _ = pipe.read_to_string(&mut err);
        }
        panic!(
            "the current binary could not start over the golden data_dir.\n\
             That means this build cannot read data written by an earlier one.\n\
             --- stderr ---\n{}",
            err.trim()
        );
    }

    fn stop(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test]
async fn current_binary_reads_a_data_dir_written_by_an_earlier_build() {
    let fixture = fixture_dir();
    assert!(
        fixture.join("snapshot.json").exists(),
        "golden fixture missing at {fixture:?} — regenerate it from a release \
         binary and commit it; without it this rule is unenforced"
    );

    let temp = tempfile::tempdir().unwrap();
    let data_dir = temp.path().join("data");
    copy_tree(&fixture, &data_dir);

    let server = Server::start(&data_dir).await;
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();

    // ── KV: reconstructed from the WAL and/or the redb state file ────────────
    let kv: serde_json::Value = http
        .get(format!("{}/v1/state/golden-kv", server.base))
        .bearer_auth(API_KEY)
        .send()
        .await
        .expect("kv read failed")
        .json()
        .await
        .expect("kv response was not JSON");
    assert_eq!(
        kv["value"]["marker"], "kv-v4.24.0",
        "KV value from the fixture did not survive: {kv}"
    );

    // ── Blob: a plain file, but under the layout the old build chose ─────────
    let blob = http
        .get(format!("{}/v1/blob/golden/obj.bin", server.base))
        .bearer_auth(API_KEY)
        .send()
        .await
        .expect("blob read failed");
    assert!(blob.status().is_success(), "blob missing from the fixture");
    assert_eq!(blob.bytes().await.unwrap().to_vec(), b"golden-blob-bytes");

    // ── Queue: the message file format has to still parse ───────────────────
    let received: serde_json::Value = http
        .post(format!("{}/v1/queue/golden/receive", server.base))
        .bearer_auth(API_KEY)
        .json(&serde_json::json!({ "max": 10, "visibility_secs": 30 }))
        .send()
        .await
        .expect("receive failed")
        .json()
        .await
        .expect("receive response was not JSON");
    let messages = received["messages"].as_array().unwrap();
    assert_eq!(
        messages.len(),
        1,
        "queued message from the fixture was not readable: {received}"
    );
    assert_eq!(messages[0]["body"]["marker"], "queue-v4.24.0");

    // ── Vector: manifest plus run file, the richest format of the set ────────
    let vector: serde_json::Value = http
        .get(format!("{}/v1/vector/golden/get", server.base))
        .bearer_auth(API_KEY)
        .query(&[("id", "golden-vec")])
        .send()
        .await
        .expect("vector read failed")
        .json()
        .await
        .expect("vector response was not JSON");
    assert_eq!(
        vector["meta"]["marker"], "vector-v4.24.0",
        "vector from the fixture did not survive: {vector}"
    );

    // A manifest written before the embedding-provenance fields existed must
    // load with them absent rather than failing the collection open.
    let detail: serde_json::Value = http
        .get(format!("{}/v1/vector/golden", server.base))
        .bearer_auth(API_KEY)
        .send()
        .await
        .expect("collection detail failed")
        .json()
        .await
        .expect("collection detail was not JSON");
    assert_eq!(detail["dim"].as_u64(), Some(4));

    server.stop();
}
