//! How much write throughput the serialized SQLite writer actually sustains.
//!
//! Diagnostic, not a benchmark suite: it exists to turn "SQLite is single-writer
//! so it must be the bottleneck" into a number. Ignored by default so it never
//! runs in the ordinary cycle.
//!
//! The architecture it measures: WAL mode, a 10-connection reader pool, and one
//! writer thread fed by an MPSC channel. Serializing writes is not a workaround
//! for SQLite's limits — it *is* SQLite's model. One writer plus many concurrent
//! readers is what WAL mode provides, and funnelling writes through one thread
//! removes `SQLITE_BUSY` entirely instead of retrying around it.
//!
//! ## Running it
//!
//! ```text
//! cargo test --release --test sqlite_write_throughput -- --ignored --nocapture
//! ```

use luma::sqlite::SqliteService;
use std::sync::Arc;
use std::time::Instant;

async fn service() -> (Arc<SqliteService>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let service = Arc::new(SqliteService::new(dir.path().join("t.db")).unwrap());
    service
        .execute(
            "CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY, payload TEXT)".into(),
            vec![],
        )
        .await
        .unwrap();
    (service, dir)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "diagnostic: cargo test --release --test sqlite_write_throughput -- --ignored --nocapture"]
async fn one_writer_at_a_time() {
    let (service, _dir) = service().await;
    let n = 2_000;
    let started = Instant::now();
    for i in 0..n {
        service
            .execute(
                "INSERT INTO t (id, payload) VALUES (?, ?)".into(),
                vec![
                    serde_json::json!(format!("k{i}")),
                    serde_json::json!("some payload of a realistic size for a metadata row"),
                ],
            )
            .await
            .unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "serial writes: {n} in {:?} = {:.0}/s",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "diagnostic: cargo test --release --test sqlite_write_throughput -- --ignored --nocapture"]
async fn many_callers_at_once() {
    // The case the "no multi-writer" worry is really about: many concurrent
    // requests all wanting to write. They queue at the actor rather than
    // contending on the file, so what this measures is whether the queue drains
    // fast enough to matter.
    let (service, _dir) = service().await;
    let concurrency = 64;
    let per_task = 100;
    let started = Instant::now();

    let mut handles = Vec::new();
    for task in 0..concurrency {
        let service = service.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..per_task {
                service
                    .execute(
                        "INSERT INTO t (id, payload) VALUES (?, ?)".into(),
                        vec![
                            serde_json::json!(format!("t{task}-{i}")),
                            serde_json::json!("some payload of a realistic size"),
                        ],
                    )
                    .await
                    .unwrap();
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    let elapsed = started.elapsed();
    let total = concurrency * per_task;
    println!(
        "{concurrency} concurrent writers: {total} writes in {:?} = {:.0}/s",
        elapsed,
        total as f64 / elapsed.as_secs_f64()
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "diagnostic: cargo test --release --test sqlite_write_throughput -- --ignored --nocapture"]
async fn reads_do_not_queue_behind_writes() {
    // The half of the worry that is simply not true here: reads go to a
    // 10-connection pool, not through the writer. If they queued behind writes,
    // a read-heavy workload would stall whenever anything was being written.
    let (service, _dir) = service().await;
    for i in 0..500 {
        service
            .execute(
                "INSERT INTO t (id, payload) VALUES (?, ?)".into(),
                vec![serde_json::json!(format!("r{i}")), serde_json::json!("x")],
            )
            .await
            .unwrap();
    }

    // Hammer writes in the background while timing reads.
    let writer = {
        let service = service.clone();
        tokio::spawn(async move {
            for i in 0..2_000 {
                let _ = service
                    .execute(
                        "INSERT INTO t (id, payload) VALUES (?, ?)".into(),
                        vec![serde_json::json!(format!("w{i}")), serde_json::json!("y")],
                    )
                    .await;
            }
        })
    };

    let started = Instant::now();
    let reads = 1_000;
    let mut handles = Vec::new();
    for _ in 0..16 {
        let service = service.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..(reads / 16) {
                service
                    .query("SELECT COUNT(*) FROM t".into(), vec![])
                    .await
                    .unwrap();
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "reads while writing: {reads} in {:?} = {:.0}/s",
        elapsed,
        reads as f64 / elapsed.as_secs_f64()
    );
    writer.await.unwrap();
}
