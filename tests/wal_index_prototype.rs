//! Option 3, measured before it is built: the WAL as the value store, with an
//! in-memory index instead of a B-tree projection.
//!
//! Diagnostic, ignored by default. This is a **prototype for measurement**, not
//! a backend — the point is to know what the design is worth before committing
//! a multi-day change to the write path.
//!
//! ## The design being measured
//!
//! The WAL already contains every value, so redb is a second copy of data that
//! is already on disk, written through a copy-on-write B-tree that rewrites the
//! page path from leaf to root on every insert. Option 3 removes that copy:
//! writes append to the log and update an in-memory `key -> (segment, offset,
//! len)`; reads look up the location and read the record back.
//!
//! ## Why a `BTreeMap` and not the 37-byte hash index
//!
//! A `HashMap<u64, Location>` measured at 37 bytes per key against a
//! `BTreeMap<String, Location>`'s 150 (`tests/ram_cost.rs`), and dropping the
//! key text is only free because a read has to fetch the record anyway and can
//! verify the key there.
//!
//! But Luma needs **ordered iteration**: `KEYS`, `SCAN` and `list_range` all
//! walk keys in order, and a hash of the key cannot be walked in the order of
//! the key. So the ordered map is the honest structure, and the 37-byte figure
//! only applies to a store that gives up scans. 150 bytes per key with the
//! values on disk is still the trade that was asked for: a 1 KB value costs 150
//! bytes of RAM instead of 1 KB.
//!
//! ```text
//! cargo test --release --test wal_index_prototype -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};

/// Where a value lives in the log.
///
/// `segment` is carried and never read here — one segment is enough to measure
/// the design, and leaving the field in keeps the struct the size a real index
/// would pay for.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Location {
    segment: u32,
    offset: u64,
    len: u32,
}

/// The prototype store: an append-only log plus an ordered index.
struct WalIndex {
    /// One lock for the log and the index together, held only for the append and
    /// the index update — the same shape as the ordering guard in
    /// `engine::commit`, which is microseconds of work.
    log: Mutex<LogState>,
    /// Held only by the thread performing a flush.
    sync_handle: Mutex<std::fs::File>,
    /// Signalled when `synced` advances.
    synced: Condvar,
    index: Mutex<BTreeMap<String, Location>>,
    path: std::path::PathBuf,
}

struct LogState {
    file: std::fs::File,
    offset: u64,
    /// Records appended so far, and how many of them are on the medium. The gap
    /// between the two is what a shared fsync closes.
    appended: u64,
    synced: u64,
    /// True while somebody is flushing, so the rest wait instead of each paying
    /// for their own.
    syncing: bool,
}

impl WalIndex {
    fn open(dir: &std::path::Path) -> WalIndex {
        let path = dir.join("log-000001");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        // A second handle, so the flush happens outside the append lock. With
        // one handle the fsync would hold up every appender and there would be
        // nothing to batch — which is exactly what the first version of this
        // prototype measured: 3 157/s, the per-write fsync rate, not the design.
        let sync_handle = file.try_clone().unwrap();
        WalIndex {
            log: Mutex::new(LogState {
                file,
                offset: 0,
                appended: 0,
                synced: 0,
                syncing: false,
            }),
            sync_handle: Mutex::new(sync_handle),
            synced: Condvar::new(),
            index: Mutex::new(BTreeMap::new()),
            path,
        }
    }

    /// Append and index one record, then make the batch durable.
    ///
    /// The fsync is shared the way `engine::commit` shares it: whoever finds
    /// unsynced records flushes them all, so a burst costs one flush.
    fn put(&self, key: &str, value: &[u8]) -> std::io::Result<()> {
        let record = format!("{}\t{}\n", key, String::from_utf8_lossy(value));
        let (location, my_seq) = {
            let mut log = self.log.lock();
            let offset = log.offset;
            log.file.write_all(record.as_bytes())?;
            log.offset += record.len() as u64;
            log.appended += 1;
            (
                Location {
                    segment: 1,
                    offset,
                    len: record.len() as u32,
                },
                log.appended,
            )
        };
        self.index.lock().insert(key.to_string(), location);

        // Leader/follower, the same shape as `engine::commit`: whoever finds no
        // flush in progress flushes everything appended so far, everyone else's
        // records included, and the rest wait for their own position.
        loop {
            let mut log = self.log.lock();
            if log.synced >= my_seq {
                return Ok(());
            }
            if log.syncing {
                self.synced.wait(&mut log);
                continue;
            }
            log.syncing = true;
            let target = log.appended;
            drop(log);

            let outcome = self.sync_handle.lock().sync_data();

            let mut log = self.log.lock();
            log.syncing = false;
            match outcome {
                Ok(()) => log.synced = log.synced.max(target),
                Err(err) => {
                    self.synced.notify_all();
                    return Err(err);
                }
            }
            self.synced.notify_all();
        }
    }

    /// Read one value back through the index.
    fn get(&self, key: &str) -> std::io::Result<Option<Vec<u8>>> {
        let Some(location) = self.index.lock().get(key).copied() else {
            return Ok(None);
        };
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(location.offset))?;
        let mut buf = vec![0u8; location.len as usize];
        file.read_exact(&mut buf)?;
        let text = String::from_utf8_lossy(&buf);
        // The record carries the key, so a lookup verifies it rather than
        // trusting the index — which is what would make a hash index exact.
        let (stored_key, value) = text.trim_end().split_once('\t').unwrap();
        assert_eq!(stored_key, key, "the index pointed at the wrong record");
        Ok(Some(value.as_bytes().to_vec()))
    }
}

#[test]
#[ignore = "diagnostic: cargo test --release --test wal_index_prototype -- --ignored --nocapture"]
fn what_option_three_would_reach() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(WalIndex::open(dir.path()));
    let value = b"a payload of a realistic size for a KV write";

    for concurrency in [1usize, 8, 32, 128] {
        let per_task = 500;
        let started = Instant::now();
        let mut handles = Vec::new();
        for task in 0..concurrency {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..per_task {
                    store.put(&format!("t{task}-{i}"), value).unwrap();
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

    // And a read, to show the design's other half is not free either: it is a
    // seek and a read where redb served from its page cache.
    let started = Instant::now();
    let reads = 20_000;
    for i in 0..reads {
        let key = format!("t0-{}", i % 500);
        assert!(store.get(&key).unwrap().is_some());
    }
    let elapsed = started.elapsed();
    println!(
        "     reads: {reads:>6} in {:>10?} = {:>8.0}/s (open + seek + read each)",
        elapsed,
        reads as f64 / elapsed.as_secs_f64()
    );
}
