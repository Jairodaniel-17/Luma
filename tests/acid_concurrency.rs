//! ACID under concurrency: the guarantees a database owes its callers when
//! many of them write at once, checked against the real `Engine` with the WAL
//! on.
//!
//! Each case is a property any engine Luma is compared with (Postgres, Redis,
//! SQLite) holds, and each one failed here before its fix:
//!
//! - **No lost confirmed write.** A put that returned OK is readable, now and
//!   after a restart, whatever other keys were being written or deleted at the
//!   same time. Two WAL write paths (the group-commit queue and the direct
//!   append used by deletes, batches and vectors) used to overtake each other;
//!   the projection skips offsets at or below the last one it applied, so the
//!   overtaken put was dropped while its caller had been told OK.
//! - **Compare-and-swap is atomic.** Of N writers racing with the same expected
//!   revision exactly one wins. The revision used to be checked before the
//!   write took any lock, so several could pass the check and all commit.
//! - **Revisions are unique and increasing per key**, which is what makes them
//!   usable for optimistic concurrency at all.
//! - **Create-if-absent** (`if_revision = 0`) admits exactly one creator.
//! - **Document indexes stay consistent** with their documents under
//!   concurrent writes: every document is found by every value it holds.

use luma::config::Config;
use luma::engine::Engine;
use std::collections::HashSet;
use std::sync::{Arc, Barrier};
use tokio_util::sync::CancellationToken;

fn config(dir: &std::path::Path) -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    }
}

fn engine(dir: &std::path::Path) -> Engine {
    Engine::new(config(dir), CancellationToken::new()).unwrap()
}

/// Puts on some keys race deletes on other keys. Every confirmed put must be
/// readable live and after reopening the data directory.
#[test]
fn confirmed_puts_survive_concurrent_deletes_and_batches() {
    let dir = tempfile::tempdir().unwrap();
    let e = Arc::new(engine(dir.path()));
    for i in 0..400 {
        e.put_state(format!("victim:{i}"), serde_json::json!(i), None, None)
            .unwrap();
    }

    const WRITERS: usize = 8;
    const PER_WRITER: usize = 300;
    let barrier = Arc::new(Barrier::new(WRITERS + 2));
    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let e = e.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let mut ok = Vec::new();
            for i in 0..PER_WRITER {
                let key = format!("w{w}:{i}");
                if e.put_state(key.clone(), serde_json::json!(i), None, None)
                    .is_ok()
                {
                    ok.push(key);
                }
            }
            ok
        }));
    }
    // A deleter and a batcher on disjoint keys, so nothing they do can
    // legitimately remove a writer's key.
    let deleter = {
        let e = e.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            barrier.wait();
            for i in 0..400 {
                let _ = e.delete_state(&format!("victim:{i}"));
            }
        })
    };
    let batcher = {
        let e = e.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            barrier.wait();
            for i in 0..200 {
                let _ = e.put_state_batch(vec![
                    luma::engine::StateOp::Put {
                        key: format!("batch:{i}:a"),
                        value: serde_json::json!(i).into(),
                        ttl_ms: None,
                        if_revision: None,
                    },
                    luma::engine::StateOp::Put {
                        key: format!("batch:{i}:b"),
                        value: serde_json::json!(i).into(),
                        ttl_ms: None,
                        if_revision: None,
                    },
                ]);
            }
        })
    };

    let mut confirmed = Vec::new();
    for h in handles {
        confirmed.extend(h.join().unwrap());
    }
    deleter.join().unwrap();
    batcher.join().unwrap();

    let missing_live: Vec<&String> = confirmed
        .iter()
        .filter(|k| e.get_state(k).is_none())
        .collect();
    assert!(
        missing_live.is_empty(),
        "{} confirmed puts are not readable right after they returned OK (e.g. {:?})",
        missing_live.len(),
        missing_live.first()
    );

    drop(e);
    let reopened = engine(dir.path());
    let missing_after: Vec<&String> = confirmed
        .iter()
        .filter(|k| reopened.get_state(k).is_none())
        .collect();
    assert!(
        missing_after.is_empty(),
        "{} confirmed puts were lost by a restart (e.g. {:?})",
        missing_after.len(),
        missing_after.first()
    );
    for i in 0..200 {
        assert!(reopened.get_state(&format!("batch:{i}:a")).is_some());
        assert!(reopened.get_state(&format!("batch:{i}:b")).is_some());
    }
}

/// A counter incremented by optimistic CAS from many threads must end at
/// exactly the number of successful CAS writes.
#[test]
fn compare_and_swap_admits_one_winner_per_revision() {
    let dir = tempfile::tempdir().unwrap();
    let e = Arc::new(engine(dir.path()));
    e.put_state("counter".into(), serde_json::json!(0), None, None)
        .unwrap();

    const THREADS: usize = 8;
    const INCREMENTS: usize = 150;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let e = e.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let mut done = 0;
            while done < INCREMENTS {
                let cur = e.get_state("counter").unwrap();
                let n = cur.value.as_json().unwrap().as_i64().unwrap();
                if e.put_state(
                    "counter".into(),
                    serde_json::json!(n + 1),
                    None,
                    Some(cur.revision),
                )
                .is_ok()
                {
                    done += 1;
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let item = e.get_state("counter").unwrap();
    let value = item.value.as_json().unwrap().as_i64().unwrap();
    assert_eq!(
        value,
        (THREADS * INCREMENTS) as i64,
        "lost updates: CAS let two writers commit over the same revision"
    );
    assert_eq!(item.revision, 1 + (THREADS * INCREMENTS) as u64);
}

/// Unconditional puts on one key from many threads: every returned revision
/// is distinct, and the final revision is the highest of them.
#[test]
fn revisions_are_unique_and_monotonic_per_key() {
    let dir = tempfile::tempdir().unwrap();
    let e = Arc::new(engine(dir.path()));
    const THREADS: usize = 8;
    const PUTS: usize = 150;
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let e = e.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            (0..PUTS)
                .map(|i| {
                    e.put_state("hot".into(), serde_json::json!([t, i]), None, None)
                        .unwrap()
                        .revision
                })
                .collect::<Vec<u64>>()
        }));
    }
    let mut all = Vec::new();
    for h in handles {
        all.extend(h.join().unwrap());
    }
    let unique: HashSet<u64> = all.iter().copied().collect();
    assert_eq!(unique.len(), all.len(), "two writes got the same revision");
    let final_rev = e.get_state("hot").unwrap().revision;
    assert_eq!(final_rev, *all.iter().max().unwrap());
    assert_eq!(final_rev, (THREADS * PUTS) as u64);
}

/// `if_revision = 0` means "only if the key does not exist": of many racing
/// creators exactly one succeeds.
#[test]
fn create_if_absent_admits_exactly_one_creator() {
    let dir = tempfile::tempdir().unwrap();
    let e = Arc::new(engine(dir.path()));
    for round in 0..40 {
        const THREADS: usize = 8;
        let barrier = Arc::new(Barrier::new(THREADS));
        let key = format!("once:{round}");
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let e = e.clone();
                let barrier = barrier.clone();
                let key = key.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    e.put_state(key, serde_json::json!(t), None, Some(0))
                        .is_ok()
                })
            })
            .collect();
        let winners = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|ok| *ok)
            .count();
        assert_eq!(winners, 1, "round {round}: {winners} creators won");
        assert_eq!(e.get_state(&key).unwrap().revision, 1);
    }
}

/// Documents written concurrently with the same field value must all be found
/// through the index, and an update must move a document between index
/// entries without leaving it in the old one.
#[test]
fn document_indexes_stay_consistent_under_concurrent_writes() {
    let dir = tempfile::tempdir().unwrap();
    let e = Arc::new(engine(dir.path()));
    const THREADS: usize = 8;
    const DOCS: usize = 40;
    let barrier = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let e = e.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..DOCS {
                    let id = format!("d{t}-{i}");
                    luma::docstore::put_doc(
                        &e,
                        "orders",
                        &id,
                        serde_json::json!({ "estado": "abierto", "n": i }),
                    )
                    .expect("put_doc must not fail under contention");
                    if i % 2 == 0 {
                        luma::docstore::put_doc(
                            &e,
                            "orders",
                            &id,
                            serde_json::json!({ "estado": "cerrado", "n": i }),
                        )
                        .expect("update must not fail under contention");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let open = luma::docstore::find_docs(
        &e,
        "orders",
        Some(&serde_json::json!({ "estado": "abierto" })),
        10_000,
    )
    .unwrap();
    let closed = luma::docstore::find_docs(
        &e,
        "orders",
        Some(&serde_json::json!({ "estado": "cerrado" })),
        10_000,
    )
    .unwrap();
    assert_eq!(
        open.len(),
        THREADS * DOCS / 2,
        "open documents missing from the index"
    );
    assert_eq!(
        closed.len(),
        THREADS * DOCS / 2,
        "closed documents missing from the index"
    );

    // The index itself, not just the filtered result: an id left behind in the
    // wrong entry is a stale index, even if `find` filters it out.
    let stale = e
        .get_state("docidx:orders:estado:abierto")
        .map(|item| {
            item.value.as_json().unwrap()["ids"]
                .as_array()
                .unwrap()
                .len()
        })
        .unwrap_or(0);
    assert_eq!(
        stale,
        THREADS * DOCS / 2,
        "closed documents left in the 'abierto' index"
    );
}

/// Indexes written by an older Luma may be missing documents (the lost-id
/// bug). The startup migration rebuilds them from the documents, once.
#[test]
fn startup_migration_rebuilds_stale_document_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let e = engine(dir.path());
    // What the old bug left behind: documents whose ids are missing from
    // their index entry, and an entry pointing at a value no document has.
    for id in ["a", "b:1"] {
        e.put_state(
            format!("doc:orders:{id}"),
            serde_json::json!({ "estado": "abierto" }),
            None,
            None,
        )
        .unwrap();
    }
    e.put_state(
        "docidx:orders:estado:cerrado".into(),
        serde_json::json!({ "ids": ["a"] }),
        None,
        None,
    )
    .unwrap();

    assert_eq!(luma::docstore::migrate_indexes(&e).unwrap(), 2);
    let open = luma::docstore::find_docs(
        &e,
        "orders",
        Some(&serde_json::json!({ "estado": "abierto" })),
        100,
    )
    .unwrap();
    let mut ids: Vec<_> = open.iter().map(|d| d.id.as_str()).collect();
    ids.sort();
    assert_eq!(ids, ["a", "b:1"]);
    assert!(e.get_state("docidx:orders:estado:cerrado").is_none());
    // Once per index version.
    assert_eq!(luma::docstore::migrate_indexes(&e).unwrap(), 0);
}

/// Write skew, the anomaly snapshot isolation lets through: two doctors on
/// call, each checks that the other one is still on call and goes off. Run
/// both at once and nobody is on call, although each transaction alone kept
/// the rule. With the key it read but does not write guarded by `Check`, the
/// second to commit sees the first one's write and fails: serializable.
#[test]
fn guarded_reads_prevent_write_skew() {
    use luma::engine::StateOp;
    let dir = tempfile::tempdir().unwrap();
    let e = Arc::new(engine(dir.path()));

    for round in 0..200 {
        let (alice, bob) = (
            format!("oncall:{round}:alice"),
            format!("oncall:{round}:bob"),
        );
        e.put_state(alice.clone(), serde_json::json!(true), None, None)
            .unwrap();
        e.put_state(bob.clone(), serde_json::json!(true), None, None)
            .unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [(alice.clone(), bob.clone()), (bob.clone(), alice.clone())]
            .into_iter()
            .map(|(me, other)| {
                let (e, barrier) = (e.clone(), barrier.clone());
                std::thread::spawn(move || {
                    let mine = e.get_state(&me).unwrap();
                    let theirs = e.get_state(&other).unwrap();
                    barrier.wait();
                    if theirs.value.as_json() != Some(&serde_json::json!(true)) {
                        return false;
                    }
                    e.state_txn(vec![
                        StateOp::Check {
                            key: other,
                            revision: theirs.revision,
                        },
                        StateOp::Put {
                            key: me,
                            value: serde_json::json!(false).into(),
                            ttl_ms: None,
                            if_revision: Some(mine.revision),
                        },
                    ])
                    .is_ok()
                })
            })
            .collect();
        let went_off = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|&ok| ok)
            .count();
        assert_eq!(
            went_off, 1,
            "round {round}: {went_off} doctors went off call"
        );
        let on_call = [&alice, &bob]
            .iter()
            .filter(|k| e.get_state(k).unwrap().value.as_json() == Some(&serde_json::json!(true)))
            .count();
        assert_eq!(on_call, 1, "round {round}: nobody left on call");
    }
}

/// All or nothing: one failed guard at the end of a transaction leaves the
/// writes before it unwritten, live and after a restart.
#[test]
fn a_failed_transaction_writes_nothing() {
    use luma::engine::StateOp;
    let dir = tempfile::tempdir().unwrap();
    let put = |key: &str| StateOp::Put {
        key: key.into(),
        value: serde_json::json!(1).into(),
        ttl_ms: None,
        if_revision: None,
    };
    {
        let e = engine(dir.path());
        e.put_state("stock".into(), serde_json::json!(5), None, None)
            .unwrap();
        let err = e.state_txn(vec![
            put("invoice:1"),
            put("ledger:1"),
            StateOp::Delete {
                key: "stock".into(),
                if_revision: None,
            },
            StateOp::Check {
                key: "price".into(),
                revision: 7,
            },
        ]);
        assert!(err.is_err(), "a failed check must fail the transaction");
        for key in ["invoice:1", "ledger:1"] {
            assert!(
                e.get_state(key).is_none(),
                "{key} written by a failed transaction"
            );
        }
        assert!(
            e.get_state("stock").is_some(),
            "stock deleted by a failed transaction"
        );

        let revisions = e
            .state_txn(vec![
                put("invoice:2"),
                StateOp::Check {
                    key: "stock".into(),
                    revision: 1,
                },
            ])
            .unwrap();
        assert_eq!(revisions, vec![Some(1), Some(1)]);
    }
    let e = engine(dir.path());
    assert!(e.get_state("invoice:1").is_none() && e.get_state("ledger:1").is_none());
    assert!(e.get_state("invoice:2").is_some() && e.get_state("stock").is_some());
}
