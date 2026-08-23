//! Group commit: many mutations share one WAL fsync and one projection
//! transaction.
//!
//! ## What this replaces, and why
//!
//! Every mutation used to hold one global lock across *both* expensive halves of
//! a write — the WAL fsync and the redb transaction — so the whole process did
//! one at a time. Measured through RESP: **785 SET/s**, and pipelining the
//! client did not help, which is what a serialization looks like from outside.
//!
//! The measurements that located it, layer by layer
//! (`tests/redb_ceiling.rs`, `tests/wal_sync_cost.rs`):
//!
//! ```text
//! sequential append, no fsync                194 151/s
//! redb batched 32/txn + Durability::None     235 679/s
//! redb batched 32/txn (Eventual)              19 057/s
//! redb Durability::None, one txn per write    11 401/s
//! redb Eventual, one txn per write             1 308/s
//! ```
//!
//! So the cost was neither the fsync (removing it entirely bought 2x) nor the
//! extra read transactions (1 308 vs 1 288 with them). It was **one redb
//! transaction per operation**: redb is a copy-on-write B-tree, and a single-key
//! insert rewrites the page path from leaf to root — about 16 KB written for a
//! 30-byte value.
//!
//! ## Leader/follower, not a committer thread
//!
//! No dedicated thread and no shutdown protocol: whichever writer arrives to
//! find no commit in progress *becomes* the leader and commits the whole queued
//! batch, including everyone else's records. The rest wait for their own offset
//! to be reported applied. Postgres and MySQL do the same thing, for the same
//! reason — the work has to happen on some thread, and a request thread is
//! already there.
//!
//! ## What is preserved, and where each property comes from
//!
//! - **WAL file order == offset order.** The queue is FIFO and records are
//!   pushed under the caller's ordering guard, so a batch is always a contiguous
//!   run of offsets. Replay depends on this: it skips `offset <= applied`, so an
//!   out-of-order record would be dropped instead of applied.
//! - **A confirmed write is durable.** `wait_for` returns only once the record's
//!   offset is both in the fsynced WAL and applied to the projection.
//! - **Read-your-write.** Same reason: the apply happens before the caller is
//!   told OK, so a `SET` followed by a `GET` cannot miss.
//! - **Publish order == offset order.** The leader publishes as it applies, in
//!   queue order, so a subscriber never sees a gap that later fills in.
//! - **A failure is not a hang.** If a leader's commit fails, the error is
//!   recorded and every waiter past the failed point returns it rather than
//!   waiting for an offset that will never arrive.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

use super::EventRecord;

/// A queued mutation, and everything needed to commit it.
struct Queued {
    event: EventRecord,
    /// The serialized WAL line, built by the caller so the leader does no
    /// per-record encoding work while holding up everyone else.
    line: Vec<u8>,
}

struct Queue {
    pending: VecDeque<Queued>,
    /// Highest offset that is in the fsynced WAL *and* applied to the
    /// projections *and* published.
    applied: u64,
    /// True while a leader is committing. Followers wait rather than starting a
    /// second, interleaved commit.
    committing: bool,
    /// A leader's failure. Waiters past `applied` return this instead of
    /// waiting forever for an offset that was dropped.
    failure: Option<String>,
}

struct Inner {
    queue: Mutex<Queue>,
    /// Signalled whenever `applied`, `committing` or `failure` changes.
    progress: Condvar,
}

/// The commit pipeline. Cheap to clone; all clones share one queue.
#[derive(Clone)]
pub struct Commits(Arc<Inner>);

/// What a leader has to do with one batch.
///
/// Passed in rather than stored so this module stays free of the WAL, redb and
/// the vector store: it owns the batching and the waiting, and nothing else.
/// `apply` runs per record, in order, and is where the projection write and the
/// publish happen.
/// Writes a batch of encoded records to the WAL with one fsync.
pub type WriteWal<'a> = &'a (dyn Fn(&[Vec<u8>]) -> std::io::Result<()> + Send + Sync);

/// Applies and publishes a batch, returning the highest offset that landed and
/// the failure, if any.
pub type ApplyBatch<'a> =
    &'a (dyn Fn(&[EventRecord]) -> (u64, Option<anyhow::Error>) + Send + Sync);

pub struct Commit<'a> {
    /// Write these lines to the WAL and fsync once.
    pub write_wal: WriteWal<'a>,
    /// Apply and publish the **whole batch**, in offset order.
    ///
    /// A batch and not one record at a time, and that is the point. Batching the
    /// WAL fsync alone moved RESP `SET` from 785/s to 4 648/s and stopped there,
    /// because the projection was still opening one redb transaction per record.
    /// Handing the batch over lets them share one.
    ///
    /// Returns the highest offset that landed **and** the failure, if any — not
    /// one or the other. A partially applied batch has records that are durable
    /// and visible, and telling their callers the write failed is the more
    /// dangerous of the two possible lies.
    pub apply_batch: ApplyBatch<'a>,
}

impl Default for Commits {
    fn default() -> Self {
        Self::new()
    }
}

impl Commits {
    pub fn new() -> Commits {
        Commits(Arc::new(Inner {
            queue: Mutex::new(Queue {
                pending: VecDeque::new(),
                applied: 0,
                committing: false,
                failure: None,
            }),
            progress: Condvar::new(),
        }))
    }

    /// Queue a record. **Must be called under the caller's ordering guard**, so
    /// that queue order is offset order.
    pub fn enqueue(&self, event: EventRecord, line: Vec<u8>) {
        self.0
            .queue
            .lock()
            .pending
            .push_back(Queued { event, line });
    }

    /// Commit until `offset` is durable, applied and published.
    ///
    /// Becomes the leader if nobody else is committing; otherwise waits for the
    /// leader that will cover this offset. Either way it returns only when the
    /// record is safe to acknowledge.
    pub fn wait_for(&self, offset: u64, commit: Commit<'_>) -> anyhow::Result<()> {
        loop {
            let batch = {
                let mut q = self.0.queue.lock();
                if q.applied >= offset {
                    return Ok(());
                }
                if let Some(reason) = q.failure.clone() {
                    // Reported once per waiter, then cleared so a later batch is
                    // judged on its own outcome rather than inheriting this one.
                    q.failure = None;
                    self.0.progress.notify_all();
                    anyhow::bail!("the commit covering this write failed: {reason}");
                }
                if q.committing {
                    self.0.progress.wait(&mut q);
                    continue;
                }
                // Become the leader and take everything queued so far — other
                // callers' records included. That sharing is the whole point.
                q.committing = true;
                std::mem::take(&mut q.pending)
            };

            if batch.is_empty() {
                // Nothing queued yet our offset is not applied: the record was
                // taken by a batch that failed and dropped it.
                let mut q = self.0.queue.lock();
                q.committing = false;
                self.0.progress.notify_all();
                anyhow::bail!("this write was dropped by a failed commit");
            }

            let (highest, failure) = Self::commit_batch(&batch, &commit);

            let mut q = self.0.queue.lock();
            q.committing = false;
            // Credit what landed *before* recording the failure, and check
            // `applied` before `failure` at the top of the loop: a waiter whose
            // record made it through must be told OK even though a later record
            // in the same batch did not.
            if highest > 0 {
                q.applied = q.applied.max(highest);
            }
            if let Some(err) = failure {
                // The records that did not land are *not* put back. The caller's
                // mutation never became visible — the WAL write is what precedes
                // it — so a later batch persisting them would apply a change
                // that was already reported as failed.
                q.failure = Some(err.to_string());
            }
            self.0.progress.notify_all();
        }
    }

    /// One batch: a single WAL write with a single fsync, then the applies in
    /// order.
    ///
    /// Returns *both* the highest offset that made it all the way through and
    /// the failure, if any — not one or the other. On an apply failure the
    /// records before it are already durable and applied, and discarding that
    /// progress would tell their callers a write failed that in fact succeeded,
    /// which is the more dangerous of the two lies.
    fn commit_batch(batch: &VecDeque<Queued>, commit: &Commit<'_>) -> (u64, Option<anyhow::Error>) {
        let lines: Vec<Vec<u8>> = batch.iter().map(|q| q.line.clone()).collect();
        if let Err(err) = (commit.write_wal)(&lines) {
            // Nothing is durable, so nothing is applied and nothing is credited.
            return (0, Some(err.into()));
        }

        let events: Vec<EventRecord> = batch.iter().map(|q| q.event.clone()).collect();
        (commit.apply_batch)(&events)
    }

    /// Highest offset durably committed.
    ///
    /// Only the tests need it: the WAL retention floor comes from the
    /// projection's own `flush()`, not from here, and exposing a second source
    /// for the same number is how the two would disagree.
    #[cfg(test)]
    fn applied(&self) -> u64 {
        self.0.queue.lock().applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn event(offset: u64) -> EventRecord {
        EventRecord {
            offset,
            event_type: "state_updated".to_string(),
            ts_ms: 0,
            data: serde_json::json!({ "key": format!("k{offset}") }),
        }
    }

    #[test]
    fn a_single_writer_commits_its_own_record() {
        let commits = Commits::new();
        let written = AtomicUsize::new(0);
        let applied = AtomicUsize::new(0);

        commits.enqueue(event(1), b"line1".to_vec());
        let write = |lines: &[Vec<u8>]| {
            written.fetch_add(lines.len(), Ordering::SeqCst);
            Ok(())
        };
        let apply = |evs: &[EventRecord]| {
            applied.fetch_add(evs.len(), Ordering::SeqCst);
            (evs.last().map(|e| e.offset).unwrap_or(0), None)
        };
        commits
            .wait_for(
                1,
                Commit {
                    write_wal: &write,
                    apply_batch: &apply,
                },
            )
            .unwrap();

        assert_eq!(written.load(Ordering::SeqCst), 1);
        assert_eq!(applied.load(Ordering::SeqCst), 1);
        assert_eq!(commits.applied(), 1);
    }

    #[test]
    fn one_writer_commits_the_whole_queue_including_other_records() {
        // The point of the design: the leader pays one fsync for everybody
        // queued, so throughput rises with concurrency instead of flattening.
        let commits = Commits::new();
        for offset in 1..=32 {
            commits.enqueue(event(offset), format!("line{offset}").into_bytes());
        }

        let wal_calls = AtomicUsize::new(0);
        let write = |lines: &[Vec<u8>]| {
            wal_calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(lines.len(), 32, "the batch must be taken whole");
            Ok(())
        };
        let apply = |evs: &[EventRecord]| (evs.last().map(|e| e.offset).unwrap_or(0), None);

        commits
            .wait_for(
                1,
                Commit {
                    write_wal: &write,
                    apply_batch: &apply,
                },
            )
            .unwrap();

        assert_eq!(
            wal_calls.load(Ordering::SeqCst),
            1,
            "32 records must cost one WAL write, not 32"
        );
        assert_eq!(commits.applied(), 32, "all of them are now acknowledgeable");
    }

    #[test]
    fn records_are_applied_in_offset_order() {
        // Replay skips `offset <= applied`, so a record applied out of order
        // would be dropped rather than applied. This is the property that keeps
        // the WAL replayable.
        let commits = Commits::new();
        for offset in 1..=8 {
            commits.enqueue(event(offset), b"l".to_vec());
        }
        let seen = Mutex::new(Vec::new());
        let write = |_: &[Vec<u8>]| Ok(());
        let apply = |evs: &[EventRecord]| {
            for ev in evs {
                seen.lock().push(ev.offset);
            }
            (evs.last().map(|e| e.offset).unwrap_or(0), None)
        };
        commits
            .wait_for(
                8,
                Commit {
                    write_wal: &write,
                    apply_batch: &apply,
                },
            )
            .unwrap();
        assert_eq!(*seen.lock(), (1..=8).collect::<Vec<_>>());
    }

    #[test]
    fn a_waiter_already_covered_does_no_work() {
        let commits = Commits::new();
        commits.enqueue(event(1), b"l".to_vec());
        let write = |_: &[Vec<u8>]| Ok(());
        let apply = |evs: &[EventRecord]| (evs.last().map(|e| e.offset).unwrap_or(0), None);
        commits
            .wait_for(
                1,
                Commit {
                    write_wal: &write,
                    apply_batch: &apply,
                },
            )
            .unwrap();

        // A second call for the same offset must return immediately without
        // committing anything: the queue is empty and the offset is applied.
        let calls = AtomicUsize::new(0);
        let write_again = |_: &[Vec<u8>]| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };
        commits
            .wait_for(
                1,
                Commit {
                    write_wal: &write_again,
                    apply_batch: &apply,
                },
            )
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_failed_wal_write_is_reported_and_does_not_hang() {
        // The failure mode that matters most: a waiter must not sit forever on
        // an offset whose record was dropped.
        let commits = Commits::new();
        commits.enqueue(event(1), b"l".to_vec());
        let write = |_: &[Vec<u8>]| Err(std::io::Error::other("disk full"));
        let apply = |evs: &[EventRecord]| (evs.last().map(|e| e.offset).unwrap_or(0), None);

        let err = commits
            .wait_for(
                1,
                Commit {
                    write_wal: &write,
                    apply_batch: &apply,
                },
            )
            .expect_err("a failed WAL write must surface");
        assert!(err.to_string().contains("disk full"), "{err}");
        assert_eq!(commits.applied(), 0, "nothing may be reported as applied");
    }

    #[test]
    fn an_apply_failure_still_credits_the_records_that_landed() {
        // The first records of the batch are already durable and applied.
        // Telling their callers the write failed would be a lie in the more
        // dangerous direction.
        let commits = Commits::new();
        for offset in 1..=4 {
            commits.enqueue(event(offset), b"l".to_vec());
        }
        let write = |_: &[Vec<u8>]| Ok(());
        let apply = |evs: &[EventRecord]| {
            let mut highest = 0;
            for ev in evs {
                if ev.offset >= 3 {
                    return (
                        highest,
                        Some(anyhow::anyhow!("projection rejected offset {}", ev.offset)),
                    );
                }
                highest = ev.offset;
            }
            (highest, None)
        };

        // Offset 1 is covered by the partial batch, so it succeeds.
        assert!(commits
            .wait_for(
                1,
                Commit {
                    write_wal: &write,
                    apply_batch: &apply,
                },
            )
            .is_ok());
        assert_eq!(commits.applied(), 2, "1 and 2 landed, 3 did not");
    }

    #[test]
    fn concurrent_writers_share_one_commit() {
        // The end-to-end property, with real threads: N writers, far fewer than
        // N WAL writes.
        let commits = Commits::new();
        let wal_calls = Arc::new(AtomicUsize::new(0));
        let total = 64u64;

        // Enqueue everything first so the leader has a full queue to take,
        // mirroring what concurrent requests produce under the ordering guard.
        for offset in 1..=total {
            commits.enqueue(event(offset), b"l".to_vec());
        }

        let mut handles = Vec::new();
        for offset in 1..=total {
            let commits = commits.clone();
            let wal_calls = wal_calls.clone();
            handles.push(std::thread::spawn(move || {
                let write = move |_: &[Vec<u8>]| {
                    wal_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                };
                let apply = |evs: &[EventRecord]| (evs.last().map(|e| e.offset).unwrap_or(0), None);
                commits
                    .wait_for(
                        offset,
                        Commit {
                            write_wal: &write,
                            apply_batch: &apply,
                        },
                    )
                    .unwrap();
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(commits.applied(), total);
        let calls = wal_calls.load(Ordering::SeqCst);
        assert!(
            calls < total as usize,
            "{calls} WAL writes for {total} records means nothing was shared"
        );
    }
}
