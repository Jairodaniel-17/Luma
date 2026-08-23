//! Option 5: an LSM store for the projection, measured against options 1+2 and 3.
//!
//! Diagnostic, ignored by default, and `fjall` is a **dev-dependency only** —
//! nothing ships until the numbers say it should.
//!
//! ## Why this is on the table at all
//!
//! It was dismissed too quickly as "a new dependency and a data migration". The
//! dependency part is real; the migration is not, and that changes the whole
//! calculation: **redb holds no data of its own.** It is a projection of the
//! WAL, rebuilt from `applied_offset` on replay. Swapping it means deleting a
//! file and replaying — there is nothing to migrate.
//!
//! ## What an LSM does differently
//!
//! A log-structured merge tree writes into an in-memory table and flushes it to
//! a sorted file; a background job merges those files. So a write never rewrites
//! a page path from leaf to root, which is exactly what makes redb's
//! copy-on-write B-tree cost 16 KB for a 30-byte value.
//!
//! Where that lands against the alternatives, on the axes that were actually
//! asked for:
//!
//! - **Speed.** Measured below.
//! - **RAM.** Bounded by the memtable and the block cache, both configured — not
//!   by the key count, which is what makes option 3 cost 150 bytes per key
//!   forever and 15 GiB at 100M keys.
//! - **Ordered iteration.** Native: an LSM is sorted, so `KEYS`, `SCAN` and
//!   `list_range` work. Option 3's 37-byte hash index cannot do this at all.
//! - **Compaction.** Already written and already tested by somebody else. That
//!   is the risky core of option 3, and the reason option 3 stopped at a
//!   prototype.
//! - **All Rust, one binary.** `fjall` is pure Rust, so the constraint holds.
//!
//! ```text
//! cargo test --release --test lsm_ceiling -- --ignored --nocapture --test-threads=1
//! ```

use std::sync::Arc;
use std::time::Instant;

use fjall::{Config, PartitionCreateOptions, PersistMode};

fn key_at(i: usize) -> String {
    format!("session:tenant-acme:user-{i:012}")
}

fn rss() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_processes();
    sysinfo::get_current_pid()
        .ok()
        .and_then(|p| sys.process(p))
        .map(|p| p.memory())
        .unwrap_or(0)
}

#[test]
#[ignore = "diagnostic: cargo test --release --test lsm_ceiling -- --ignored --nocapture --test-threads=1"]
fn what_an_lsm_reaches_for_a_projection() {
    // No `persist` inside the loop, deliberately, and it is not cheating: the
    // WAL is the durable record and this store is a projection of it. That is
    // the same reasoning that put redb on `Durability::None`, applied to the
    // same role. A crash costs a rebuild, not data.
    let dir = tempfile::tempdir().unwrap();
    let keyspace = Config::new(dir.path()).open().unwrap();
    let partition = keyspace
        .open_partition("state", PartitionCreateOptions::default())
        .unwrap();
    let value = vec![b'x'; 200];

    for n in [3_000usize, 50_000] {
        let before = rss();
        let started = Instant::now();
        for i in 0..n {
            partition.insert(key_at(i), &value).unwrap();
        }
        let elapsed = started.elapsed();
        println!(
            "{:<34} {n:>7} in {:>11?} = {:>8.0}/s   RSS +{:.1} MiB",
            "LSM insert (projection role)",
            elapsed,
            n as f64 / elapsed.as_secs_f64(),
            (rss().saturating_sub(before)) as f64 / (1024.0 * 1024.0)
        );
    }

    // And with a durability barrier per batch, for the case where the store is
    // asked to be durable in its own right.
    let n = 20_000;
    let batch = 32;
    let started = Instant::now();
    let mut i = 0usize;
    while i < n {
        for _ in 0..batch {
            if i >= n {
                break;
            }
            partition.insert(format!("b{i}"), &value).unwrap();
            i += 1;
        }
        keyspace.persist(PersistMode::SyncAll).unwrap();
    }
    let elapsed = started.elapsed();
    println!(
        "{:<34} {n:>7} in {:>11?} = {:>8.0}/s",
        format!("LSM + fsync every {batch}"),
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test lsm_ceiling -- --ignored --nocapture --test-threads=1"]
fn concurrent_writers_against_an_lsm() {
    // The shape that matters: many writers at once, which is what the commit
    // pipeline delivers to the projection.
    let dir = tempfile::tempdir().unwrap();
    let keyspace = Config::new(dir.path()).open().unwrap();
    let partition = Arc::new(
        keyspace
            .open_partition("state", PartitionCreateOptions::default())
            .unwrap(),
    );

    for concurrency in [1usize, 8, 32, 128] {
        let per_task = 500;
        let started = Instant::now();
        let mut handles = Vec::new();
        for task in 0..concurrency {
            let partition = partition.clone();
            handles.push(std::thread::spawn(move || {
                let value = vec![b'x'; 200];
                for i in 0..per_task {
                    partition.insert(format!("t{task}-{i}"), &value).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let elapsed = started.elapsed();
        let total = concurrency * per_task;
        println!(
            "{concurrency:>4} writers: {total:>6} inserts in {:>10?} = {:>8.0}/s",
            elapsed,
            total as f64 / elapsed.as_secs_f64()
        );
    }
}

#[test]
#[ignore = "diagnostic: cargo test --release --test lsm_ceiling -- --ignored --nocapture --test-threads=1"]
fn ordered_iteration_which_option_three_cannot_do() {
    // `KEYS`, `SCAN` and `list_range` all walk keys in order. An LSM is sorted,
    // so this is native and cheap; a hash index cannot do it at any price, and
    // the ordered map that can costs 150 bytes per key forever.
    let dir = tempfile::tempdir().unwrap();
    let keyspace = Config::new(dir.path()).open().unwrap();
    let partition = keyspace
        .open_partition("state", PartitionCreateOptions::default())
        .unwrap();
    for i in 0..50_000 {
        partition.insert(key_at(i), b"v").unwrap();
    }

    let started = Instant::now();
    let prefix = "session:tenant-acme:user-0000000001";
    let found = partition.prefix(prefix).count();
    println!(
        "{:<34} {found} keys under a prefix in {:?}",
        "LSM prefix scan",
        started.elapsed()
    );
    assert!(found > 0, "the scan must find the keys it inserted");
}
