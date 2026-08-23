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
