//! What redb itself can do, so the write path can be judged against it.
//!
//! Diagnostic, ignored by default. The measurements above it left one question:
//! a KV write costs ~0.5 ms with the WAL fsync taken out of the picture, and
//! redb is the only thing left in that path. Is that redb's floor, or is it the
//! *three* transactions Luma opens per write — `applied_offset()`,
//! `prepare_put_revision()`, and the write itself?
//!
//! The answer decides the fix. If redb's own commit is the floor, only batching
//! or an in-memory index helps. If the extra transactions dominate, folding them
//! into the write transaction is a much smaller change.
//!
//! ```text
//! cargo test --release --test redb_ceiling -- --ignored --nocapture
//! ```

use redb::{Database, Durability, ReadableTable, TableDefinition};
use std::io::Write;
use std::time::Instant;

const T: TableDefinition<&[u8], &[u8]> = TableDefinition::new("t");
const META: TableDefinition<&[u8], u64> = TableDefinition::new("meta");

fn db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::create(dir.path().join("t.redb")).unwrap();
    let wtx = db.begin_write().unwrap();
    {
        let _ = wtx.open_table(T).unwrap();
        let _ = wtx.open_table(META).unwrap();
    }
    wtx.commit().unwrap();
    (db, dir)
}

#[test]
#[ignore = "diagnostic: cargo test --release --test redb_ceiling -- --ignored --nocapture"]
fn one_eventual_commit_per_write() {
    // The shape Luma uses today: one write transaction per event, Eventual
    // durability, two tables open, a read of the previous value, an insert, and
    // a meta update.
    let (db, _dir) = db();
    let n = 3_000;
    let started = Instant::now();
    for i in 0..n {
        let key = format!("k{i}");
        let mut wtx = db.begin_write().unwrap();
        wtx.set_durability(Durability::Eventual);
        {
            let mut t = wtx.open_table(T).unwrap();
            let mut m = wtx.open_table(META).unwrap();
            let _prev = t.get(key.as_bytes()).unwrap().map(|v| v.value().len());
            t.insert(key.as_bytes(), b"a payload of a realistic size".as_slice())
                .unwrap();
            m.insert(b"applied".as_slice(), &(i as u64)).unwrap();
        }
        wtx.commit().unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<38} {n} in {:?} = {:.0}/s",
        "one txn per write (today's shape)",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test redb_ceiling -- --ignored --nocapture"]
fn plus_the_two_extra_read_transactions() {
    // Today's shape *plus* the two read transactions Luma opens first:
    // `applied_offset()` and `prepare_put_revision()`. The difference between
    // this and the test above is what folding them into the write would save.
    let (db, _dir) = db();
    let n = 3_000;
    let started = Instant::now();
    for i in 0..n {
        let key = format!("k{i}");

        // applied_offset()
        {
            let rtx = db.begin_read().unwrap();
            let m = rtx.open_table(META).unwrap();
            let _ = m.get(b"applied".as_slice()).unwrap();
        }
        // prepare_put_revision()
        {
            let rtx = db.begin_read().unwrap();
            let t = rtx.open_table(T).unwrap();
            let _ = t.get(key.as_bytes()).unwrap();
        }

        let mut wtx = db.begin_write().unwrap();
        wtx.set_durability(Durability::Eventual);
        {
            let mut t = wtx.open_table(T).unwrap();
            let mut m = wtx.open_table(META).unwrap();
            let _prev = t.get(key.as_bytes()).unwrap().map(|v| v.value().len());
            t.insert(key.as_bytes(), b"a payload of a realistic size".as_slice())
                .unwrap();
            m.insert(b"applied".as_slice(), &(i as u64)).unwrap();
        }
        wtx.commit().unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<38} {n} in {:?} = {:.0}/s",
        "+ the two extra read txns",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test redb_ceiling -- --ignored --nocapture"]
fn batched_into_one_transaction() {
    // What a commit pipeline would achieve: many writes sharing one transaction.
    // If this is far above the per-write number, batching is the fix and no
    // amount of trimming the per-write path will get close.
    let (db, _dir) = db();
    let n = 3_000;
    let batch = 32; // the concurrency redis-benchmark used
    let started = Instant::now();
    let mut i = 0u64;
    while i < n {
        let mut wtx = db.begin_write().unwrap();
        wtx.set_durability(Durability::Eventual);
        {
            let mut t = wtx.open_table(T).unwrap();
            let mut m = wtx.open_table(META).unwrap();
            for _ in 0..batch {
                if i >= n {
                    break;
                }
                let key = format!("k{i}");
                let _prev = t.get(key.as_bytes()).unwrap().map(|v| v.value().len());
                t.insert(key.as_bytes(), b"a payload of a realistic size".as_slice())
                    .unwrap();
                i += 1;
            }
            m.insert(b"applied".as_slice(), &i).unwrap();
        }
        wtx.commit().unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<38} {n} in {:?} = {:.0}/s",
        format!("batched {batch} per txn"),
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test redb_ceiling -- --ignored --nocapture"]
fn durability_none_because_redb_is_only_a_projection() {
    // `Durability::None` gives up redb's own crash-consistency. Normally that
    // would be reckless; here it is exactly right, because redb is not the
    // source of truth. The WAL is, and `state_db.rs` says so: redb is a
    // projection that replay rebuilds from `applied_offset`. A redb file left
    // inconsistent by a crash costs a rebuild, not data.
    let (db, _dir) = db();
    let n = 3_000;
    let started = Instant::now();
    for i in 0..n {
        let key = format!("k{i}");
        let mut wtx = db.begin_write().unwrap();
        wtx.set_durability(Durability::None);
        {
            let mut t = wtx.open_table(T).unwrap();
            let mut m = wtx.open_table(META).unwrap();
            let _prev = t.get(key.as_bytes()).unwrap().map(|v| v.value().len());
            t.insert(key.as_bytes(), b"a payload of a realistic size".as_slice())
                .unwrap();
            m.insert(b"applied".as_slice(), &(i as u64)).unwrap();
        }
        wtx.commit().unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<38} {n} in {:?} = {:.0}/s",
        "Durability::None, one txn per write",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test redb_ceiling -- --ignored --nocapture"]
fn a_sequential_append_for_comparison() {
    // The shape a log has: append the record, no page tree to rewrite. This is
    // what the WAL already does, and what a key-to-offset index would make the
    // *only* write on the path. It is the ceiling any B-tree is measured
    // against, and the gap is write amplification.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap();
    let n = 3_000;
    let started = Instant::now();
    for i in 0..n {
        writeln!(
            file,
            "{{\"k\":\"k{i}\",\"v\":\"a payload of a realistic size\"}}"
        )
        .unwrap();
    }
    file.flush().unwrap();
    let elapsed = started.elapsed();
    println!(
        "{:<38} {n} in {:?} = {:.0}/s",
        "sequential append (no fsync)",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test redb_ceiling -- --ignored --nocapture"]
fn batched_and_durability_none_together() {
    let (db, _dir) = db();
    let n = 3_000;
    let batch = 32;
    let started = Instant::now();
    let mut i = 0u64;
    while i < n {
        let mut wtx = db.begin_write().unwrap();
        wtx.set_durability(Durability::None);
        {
            let mut t = wtx.open_table(T).unwrap();
            let mut m = wtx.open_table(META).unwrap();
            for _ in 0..batch {
                if i >= n {
                    break;
                }
                let key = format!("k{i}");
                let _prev = t.get(key.as_bytes()).unwrap().map(|v| v.value().len());
                t.insert(key.as_bytes(), b"a payload of a realistic size".as_slice())
                    .unwrap();
                i += 1;
            }
            m.insert(b"applied".as_slice(), &i).unwrap();
        }
        wtx.commit().unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<38} {n} in {:?} = {:.0}/s",
        "batched 32 + Durability::None",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}
