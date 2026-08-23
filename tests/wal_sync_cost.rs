//! What `per_write` durability actually costs against `group` buffering.
//!
//! Diagnostic, ignored by default. The choice it informs: `group` is the default
//! WAL sync mode, and `append_event` returns before the record is on disk — the
//! persist module's own unit test asserts the WAL file is still empty
//! afterwards. So a write confirmed to a caller can be lost if the process dies
//! inside the flush window, which is 10 ms or 64 records, whichever comes first.
//!
//! That contradicts the rule `src/durability.rs` states in its first paragraph:
//! *a write that has been confirmed to a caller must survive a crash.* Before
//! changing the default, this measures what the safe mode costs — through
//! `put_state`, which is what an HTTP request actually calls.
//!
//! ## Running it
//!
//! ```text
//! cargo test --release --test wal_sync_cost -- --ignored --nocapture
//! ```

use luma::config::Config;
use luma::engine::Engine;
use std::time::Instant;
use tokio_util::sync::CancellationToken;

fn config(dir: &std::path::Path, mode: &str) -> Config {
    Config {
        port: 0,
        data_dir: Some(dir.to_string_lossy().to_string()),
        sqlite_enabled: false,
        snapshot_interval_secs: 3600,
        wal_sync_mode: mode.to_string(),
        ..Config::default()
    }
}

fn measure(label: &str, mode: &str, n: u64) {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(config(dir.path(), mode), CancellationToken::new()).unwrap();

    let started = Instant::now();
    for i in 0..n {
        engine
            .put_state(
                format!("k{i}"),
                serde_json::json!({ "value": "a payload of a realistic size for a KV write" }),
                None,
                None,
            )
            .unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{label:<30} {n} writes in {:?} = {:.0}/s",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "diagnostic: cargo test --release --test wal_sync_cost -- --ignored --nocapture"]
async fn what_durability_costs() {
    let n = 3_000;
    measure("group (buffered, current)", "group", n);
    measure("per_write (fsync each)", "per_write", n);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "diagnostic: cargo test --release --test wal_sync_cost -- --ignored --nocapture"]
async fn where_the_write_cost_actually_is() {
    // Isolates the three layers. `data_dir = None` gives no WAL and no redb —
    // the in-memory `StateStore` that already exists for that case. The gap
    // between it and the persistent modes is what a batching pipeline could
    // recover, so it is the number that decides whether the work is worth it.
    let n = 3_000;

    let started = Instant::now();
    let engine = Engine::new(
        Config {
            port: 0,
            data_dir: None,
            sqlite_enabled: false,
            snapshot_interval_secs: 3600,
            ..Config::default()
        },
        CancellationToken::new(),
    )
    .unwrap();
    for i in 0..n {
        engine
            .put_state(
                format!("k{i}"),
                serde_json::json!({ "value": "a payload of a realistic size for a KV write" }),
                None,
                None,
            )
            .unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<30} {n} writes in {:?} = {:.0}/s",
        "in-memory only (no WAL, no redb)",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "diagnostic: cargo test --release --test wal_sync_cost -- --ignored --nocapture"]
async fn concurrent_writers_with_the_commit_pipeline() {
    // The number the RESP benchmark cannot separate from the network. Many
    // writers in-process, so what is measured is the pipeline and nothing else.
    // If this tracks the ~8 000/s seen over RESP, the ceiling is inside Luma; if
    // it is far higher, the ceiling is the socket.
    let dir = tempfile::tempdir().unwrap();
    let engine = std::sync::Arc::new(
        Engine::new(config(dir.path(), "per_write"), CancellationToken::new()).unwrap(),
    );

    for concurrency in [1usize, 8, 32, 128] {
        let per_task = 500;
        let started = Instant::now();
        let mut handles = Vec::new();
        for task in 0..concurrency {
            let engine = engine.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..per_task {
                    engine
                        .put_state(
                            format!("t{task}-{i}"),
                            serde_json::json!({ "value": "a payload of a realistic size" }),
                            None,
                            None,
                        )
                        .unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let elapsed = started.elapsed();
        let total = concurrency * per_task;
        println!(
            "{concurrency:>4} writers: {total:>6} writes in {:>10?} = {:>8.0}/s",
            elapsed,
            total as f64 / elapsed.as_secs_f64()
        );
    }
}
