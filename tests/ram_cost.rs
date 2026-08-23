//! What each candidate write path actually costs in RAM.
//!
//! Diagnostic, ignored by default. The two designs on the table hold very
//! different things in memory, and the difference decides which one fits a
//! machine:
//!
//! - **Batched redb with `Durability::None`** holds *dirty pages* until a
//!   checkpoint. Bounded by how often you checkpoint, not by the dataset.
//! - **A key-to-offset index** (Bitcask) holds *one entry per live key*
//!   forever, and the value size does not matter. Bounded by the key count.
//!
//! Estimating either is easy to get wrong by an order of magnitude, so both are
//! measured against the process's real resident set.
//!
//! ```text
//! cargo test --release --test ram_cost -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::time::Instant;

use redb::{Database, Durability, TableDefinition};

const T: TableDefinition<&[u8], &[u8]> = TableDefinition::new("t");

/// Resident set of this process, in bytes.
///
/// Same call shape as `engine::read_process_memory_rss`, so the numbers here are
/// comparable to what `/v1/metrics` reports.
fn rss() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_processes();
    sysinfo::get_current_pid()
        .ok()
        .and_then(|p| sys.process(p))
        .map(|p| p.memory())
        .unwrap_or(0)
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// A realistic key: long enough to be a real name, not a counter.
fn key_at(i: usize) -> String {
    format!("session:tenant-acme:user-{i:012}")
}

#[test]
#[ignore = "diagnostic: cargo test --release --test ram_cost -- --ignored --nocapture"]
fn dirty_pages_held_by_durability_none() {
    // Option 1+2: how much memory redb holds between checkpoints. This is the
    // number that decides how often a checkpoint has to run.
    let dir = tempfile::tempdir().unwrap();
    let db = Database::create(dir.path().join("t.redb")).unwrap();
    {
        let wtx = db.begin_write().unwrap();
        let _ = wtx.open_table(T).unwrap();
        wtx.commit().unwrap();
    }

    let value = vec![b'x'; 200];
    for n in [10_000usize, 50_000, 200_000] {
        let before = rss();
        let started = Instant::now();
        let batch = 32;
        let mut i = 0usize;
        while i < n {
            let mut wtx = db.begin_write().unwrap();
            wtx.set_durability(Durability::None);
            {
                let mut t = wtx.open_table(T).unwrap();
                for _ in 0..batch {
                    if i >= n {
                        break;
                    }
                    t.insert(key_at(i).as_bytes(), value.as_slice()).unwrap();
                    i += 1;
                }
            }
            wtx.commit().unwrap();
        }
        let elapsed = started.elapsed();
        let after = rss();
        println!(
            "redb None, {n:>7} writes uncheckpointed: RSS +{:>7.1} MiB  ({:.0} bytes/write, {:.0}/s)",
            mib(after.saturating_sub(before)),
            after.saturating_sub(before) as f64 / n as f64,
            n as f64 / elapsed.as_secs_f64()
        );
        // A durable commit is the checkpoint. Whatever it releases is what the
        // checkpoint interval is buying back.
        let released_before = rss();
        {
            let wtx = db.begin_write().unwrap();
            wtx.commit().unwrap();
        }
        println!(
            "    after a durable commit:            RSS {:+.1} MiB",
            mib(rss()) - mib(released_before)
        );
    }
}

#[test]
#[ignore = "diagnostic: cargo test --release --test ram_cost -- --ignored --nocapture"]
fn a_key_to_offset_index_per_live_key() {
    // Option 3: one entry per live key, for the lifetime of the process. The
    // value never enters memory, so a 200-byte value and a 2 MB value cost the
    // same here — which is the whole point of the design.
    // Deliberately the full shape a real index would hold, so the measured
    // bytes-per-key is not an understatement. `segment` and `len` are never read
    // here — they are being *sized*, which is the whole point.
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    struct Location {
        segment: u32,
        offset: u64,
        len: u32,
    }

    for n in [100_000usize, 1_000_000] {
        let before = rss();
        let started = Instant::now();
        let mut index: HashMap<String, Location> = HashMap::with_capacity(n);
        for i in 0..n {
            index.insert(
                key_at(i),
                Location {
                    segment: 1,
                    offset: (i as u64) * 256,
                    len: 200,
                },
            );
        }
        let after = rss();
        println!(
            "key→offset index, {n:>9} keys:       RSS +{:>7.1} MiB  ({:.0} bytes/key, built in {:?})",
            mib(after.saturating_sub(before)),
            after.saturating_sub(before) as f64 / n as f64,
            started.elapsed()
        );
        // Read it back so the optimiser cannot discard the map.
        let probe = index.get(&key_at(n / 2)).map(|l| l.offset).unwrap_or(0);
        assert!(probe > 0);
        drop(index);
    }
}

#[test]
#[ignore = "diagnostic: cargo test --release --test ram_cost -- --ignored --nocapture"]
fn what_redb_holds_for_reads() {
    // The other half of option 1+2's memory: redb's page cache, which serves
    // reads. It is a cache, so it is reclaimable — but it is configured, and a
    // default that is large is memory the process will take if it can.
    println!(
        "redb default cache: whatever `Database::create` sets — see \
         Builder::set_cache_size in the redb version pinned in Cargo.lock"
    );
}

#[test]
#[ignore = "diagnostic: cargo test --release --test ram_cost -- --ignored --nocapture --test-threads=1"]
fn a_compacted_index_that_stores_a_hash_instead_of_the_key() {
    // Where the 150 bytes went: almost all of it is the key *text*, plus
    // `String`'s 24-byte header, plus what the allocator rounds up. The location
    // itself is 16 bytes.
    //
    // And the key text does not need to be there. A lookup has to read the log
    // record anyway to get the value, and that record carries the full key — so
    // the index can hold a hash and the read verifies it. Collisions become a
    // correctness detail handled for free instead of a reason to keep the text.
    //
    // Packing further: a global byte offset makes segment+offset one u64, and
    // the length is in the record's own header. So the whole location is 8
    // bytes and the entry is u64 -> u64.
    let n = 1_000_000usize;

    let before = rss();
    let mut by_hash: HashMap<u64, u64> = HashMap::with_capacity(n);
    for i in 0..n {
        let key = key_at(i);
        // Any good 64-bit hash; this stands in for xxhash/ahash.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in key.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        by_hash.insert(h, (i as u64) * 256);
    }
    let after = rss();
    println!(
        "hash -> offset,   {n:>9} keys:       RSS +{:>7.1} MiB  ({:.0} bytes/key)",
        mib(after.saturating_sub(before)),
        after.saturating_sub(before) as f64 / n as f64
    );
    assert_eq!(by_hash.len(), n, "a collision would show up as a short map");
    drop(by_hash);
}
