//! Recovery, one WAL record type at a time.
//!
//! F4.5 of `docs/PLAN-MAESTRO.md`. `tests/crash_recovery.rs` kills the real
//! binary and asks "did a confirmed write survive?", which is the right question
//! per *engine*. This asks it per **record type**, which is where the answers
//! actually differ: each type has its own apply arm on the replay path, and a
//! type with no arm replays as nothing at all.
//!
//! That is not hypothetical. The `state_batch` arm was missing from the redb
//! replay path when it was first written, so every atomic move would have been
//! lost on the first restart after a crash — the exact failure the batch record
//! exists to prevent, reintroduced one layer down. Nothing caught it until a
//! test replayed that record type specifically.
//!
//! ## The three questions, for every type
//!
//! 1. **Intact.** The record replays and the mutation is there.
//! 2. **Torn.** A half-written record — what a crash mid-append leaves behind.
//! 3. **Corrupt.** A record whose checksum does not match. It must not be
//!    applied: skipping it and continuing would apply later records on top of
//!    state that was never built, which is silent divergence — worse than
//!    stopping short and saying so.
//!
//! And one more, once: everything written *before* the damaged record must still
//! replay. An implementation that threw away the whole segment would be safe and
//! useless.
//!
//! ## The answer to (2) is not the same for every type, and that is the point
//!
//! For the KV records the WAL is the only source of truth, so a torn record is
//! as if it never happened. For the vector records it is not: the vector store
//! writes its own manifest and segment files, which are durable in their own
//! right, so a mutation that already reached disk is *not* undone by losing its
//! WAL record. That is stronger than the WAL-only guarantee, not weaker.
//!
//! The one that reads oddly is `vector_collection_dropped`: with its record
//! gone, the earlier `vector_collection_created` record replays and the
//! collection comes back, even though its directory was deleted. Each case
//! therefore states its own expectation and why, so a change in **either**
//! direction fails here instead of being discovered by a user.
//!
//! ## Replaying from the WAL alone
//!
//! Every case copies only the WAL into a fresh directory, leaving the redb
//! projection behind. That is the crash scenario stated exactly: the write order
//! is WAL first and projection second, and the projection commits with
//! `Durability::Eventual`, so a crash rolls it back and the WAL is what recovery
//! has. Truncating the WAL of a cleanly closed directory would prove nothing —
//! a clean close flushes the projection, and replay then skips the record as
//! already applied.

use luma::config::Config;
use luma::engine::Engine;
use luma::vector::{Metric, VectorItem};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

fn config_for(dir: &Path) -> Config {
    Config {
        data_dir: Some(dir.to_str().unwrap().to_string()),
        // No snapshot: it would fold the WAL away and leave nothing to damage.
        snapshot_interval_secs: 86_400,
        // Per-write, not the default group commit. Group commit holds records in
        // a buffer, so whether one is on disk at a given instant is a timing
        // question — and a durability test whose subject may or may not be there
        // is not a test.
        wal_sync_mode: "per_write".to_string(),
        sqlite_enabled: false,
        ..Config::default()
    }
}

fn wal_segments(dir: &Path) -> Vec<PathBuf> {
    let mut segments: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("data dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("events-") && n.ends_with(".log"))
        })
        .collect();
    segments.sort();
    segments
}

/// Copy only the WAL, plus the `vectors/` directory it needs.
///
/// Vector collections keep their own files alongside the WAL, so replaying a
/// vector record with no `vectors/` is not a crash — it is a different and
/// harsher scenario, and conflating the two would make a pass here mean less
/// than it appears to.
fn copy_wal(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).expect("data dir").flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_wal = name.starts_with("events-") && name.ends_with(".log");
        if is_wal || name == "snapshot.json" {
            std::fs::copy(entry.path(), to.join(&name)).expect("copy");
        } else if name == "vectors" {
            copy_tree(&entry.path(), &to.join(&name));
        }
    }
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).ok();
    for entry in std::fs::read_dir(from).into_iter().flatten().flatten() {
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            let _ = std::fs::copy(entry.path(), target);
        }
    }
}

/// The `event.type` of one WAL line, or an empty string when it cannot be read.
fn record_type(line: &str) -> String {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .and_then(|value| {
            value
                .get("event")
                .and_then(|event| event.get("type"))
                .and_then(|kind| kind.as_str())
                .map(|kind| kind.to_string())
        })
        .unwrap_or_default()
}

/// Non-empty lines of the last WAL segment.
fn segment_lines(dir: &Path) -> (PathBuf, Vec<String>) {
    let segment = wal_segments(dir).pop().expect("a WAL segment must exist");
    let text = std::fs::read_to_string(&segment).unwrap();
    let lines = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.to_string())
        .collect();
    (segment, lines)
}

/// Damage the first record of `record`, discarding everything after it.
///
/// Everything after is dropped because replay stops at the damaged record
/// anyway: keeping those lines would test nothing while inviting the reader to
/// believe they had been considered.
fn damage_first(dir: &Path, record: &str, damage: impl Fn(&str) -> String) {
    let (segment, lines) = segment_lines(dir);
    let at = lines
        .iter()
        .position(|line| record_type(line) == record)
        .unwrap_or_else(|| panic!("no {record} record in the segment"));
    write_lines(&segment, &lines[..at], &damage(&lines[at]));
}

/// Damage the final record, keeping every earlier one.
fn damage_last(dir: &Path, damage: impl Fn(&str) -> String) {
    let (segment, mut lines) = segment_lines(dir);
    let last = lines.pop().expect("a record to damage");
    write_lines(&segment, &lines, &damage(&last));
}

fn write_lines(segment: &Path, keep: &[String], tail: &str) {
    let mut rebuilt = keep.join("\n");
    if !rebuilt.is_empty() {
        rebuilt.push('\n');
    }
    rebuilt.push_str(tail);
    std::fs::write(segment, rebuilt).unwrap();
}

/// Break a record's checksum without touching anything else.
///
/// The record stays well-formed JSON on purpose: otherwise this would be testing
/// the JSON parser, and a checksum that is never consulted would still pass.
fn break_checksum(line: &str) -> String {
    let mut record: serde_json::Value = serde_json::from_str(line).expect("valid record");
    let current = record
        .get("crc32")
        .and_then(|v| v.as_u64())
        .expect("records carry a crc32");
    record["crc32"] = serde_json::json!(current ^ 0xFFFF_FFFFu64);
    record.to_string()
}

fn halve(line: &str) -> String {
    line[..line.len() / 2].to_string()
}

/// One record type, and the mutation that produces it.
struct Case {
    /// The `event_type` under test, asserted during capture so a renamed record
    /// type fails loudly rather than quietly testing whatever else was written.
    record: &'static str,
    /// Whatever the mutation needs in place first.
    prepare: fn(&Engine),
    /// The mutation whose record is the subject.
    mutate: fn(&Engine),
    /// True once the mutation is visible. For a deletion, that means the value
    /// is gone.
    applied: fn(&Engine) -> bool,
    /// Whether the mutation still holds when its WAL record is damaged, and why.
    /// Stated per case rather than assumed uniform, because it genuinely differs
    /// by where the durable artefact lives.
    survives_damage: bool,
    why: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            record: "state_updated",
            prepare: |_| {},
            mutate: |e| {
                e.put_state("k".into(), serde_json::json!("v"), None, None)
                    .unwrap();
            },
            applied: |e| e.get_state("k").is_some(),
            survives_damage: false,
            why: "the WAL is the only source of truth for the KV store",
        },
        Case {
            record: "state_deleted",
            prepare: |e| {
                e.put_state("gone".into(), serde_json::json!(1), None, None)
                    .unwrap();
            },
            mutate: |e| {
                assert!(e.delete_state("gone").unwrap());
            },
            applied: |e| e.get_state("gone").is_none(),
            survives_damage: false,
            why: "losing the delete record brings the value back, which is the WAL being \
             the source of truth",
        },
        Case {
            record: "state_batch",
            prepare: |e| {
                use luma::engine::structures::{Structure, Structures};
                Structures::new(e)
                    .mutate("queue", Structure::empty_list, |s| {
                        s.rpush(vec![b"job".to_vec(), b"other".to_vec()])
                    })
                    .unwrap();
            },
            mutate: |e| {
                use luma::engine::structures::Structures;
                Structures::new(e)
                    .move_element("queue", "inflight", false, true)
                    .unwrap()
                    .expect("something to move");
            },
            applied: |e| {
                use luma::engine::structures::Structures;
                Structures::new(e).load("inflight").unwrap().is_some()
            },
            survives_damage: false,
            why: "the whole batch is one record: it is entirely present or entirely absent",
        },
        Case {
            record: "vector_collection_created",
            prepare: |_| {},
            mutate: |e| {
                e.create_vector_collection("c", 2, Metric::Cosine).unwrap();
            },
            applied: |e| e.vector_collection_info("c").is_some(),
            survives_damage: true,
            why: "the collection's manifest is a file of its own, durable without the WAL",
        },
        Case {
            record: "vector_upserted",
            prepare: |e| {
                e.create_vector_collection("c", 2, Metric::Cosine).unwrap();
            },
            mutate: |e| {
                e.vector_upsert("c", "v1", unit_vector()).unwrap();
            },
            applied: |e| e.vector_get("c", "v1").ok().flatten().is_some(),
            survives_damage: true,
            why: "the vector is already in the collection's segment file on disk",
        },
        Case {
            record: "vector_deleted",
            prepare: |e| {
                e.create_vector_collection("c", 2, Metric::Cosine).unwrap();
                e.vector_upsert("c", "v1", unit_vector()).unwrap();
            },
            mutate: |e| {
                e.vector_delete("c", "v1").unwrap();
            },
            applied: |e| e.vector_get("c", "v1").ok().flatten().is_none(),
            survives_damage: true,
            why: "the removal is already reflected in the collection's own files",
        },
        Case {
            record: "vector_collection_dropped",
            prepare: |e| {
                e.create_vector_collection("doomed", 2, Metric::Cosine)
                    .unwrap();
            },
            mutate: |e| {
                e.drop_vector_collection("doomed").unwrap();
            },
            applied: |e| e.vector_collection_info("doomed").is_none(),
            survives_damage: false,
            why: "with the drop record gone, the earlier create record replays and the \
             collection returns even though its directory was removed",
        },
    ]
}

fn unit_vector() -> VectorItem {
    VectorItem {
        vector: vec![0.5, 0.5],
        meta: serde_json::Value::Null,
        mmap_offset: None,
    }
}

/// Run one case and return the WAL as a crash would have left it.
///
/// The copy is taken while the engine is still up: a clean close takes a
/// snapshot and rotates the WAL away, which is exactly what a crash does not do.
///
/// The record under test is not always the last one — creating a vector
/// collection writes `vector_collection_created` and then a `state_updated` for
/// the manifest — so its position is found by type rather than assumed.
fn capture(case: &Case) -> tempfile::TempDir {
    let live = tempfile::tempdir().unwrap();
    let captured = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    {
        let engine = Engine::new(config_for(live.path()), shutdown.clone()).unwrap();
        (case.prepare)(&engine);
        (case.mutate)(&engine);

        let (_, lines) = segment_lines(live.path());
        assert!(
            lines.iter().any(|line| record_type(line) == case.record),
            "no {} record was written; the types are {:?}",
            case.record,
            lines.iter().map(|l| record_type(l)).collect::<Vec<_>>()
        );

        copy_wal(live.path(), captured.path());
        shutdown.cancel();
    }
    captured
}

fn replay(dir: &Path) -> (Engine, CancellationToken) {
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(dir), shutdown.clone()).unwrap();
    (engine, shutdown)
}

#[tokio::test]
async fn every_record_type_replays_when_intact() {
    for case in cases() {
        let captured = capture(&case);
        let (engine, shutdown) = replay(captured.path());
        assert!(
            (case.applied)(&engine),
            "{} did not replay: a record type with no apply arm replays as nothing",
            case.record
        );
        shutdown.cancel();
    }
}

#[tokio::test]
async fn a_torn_record_has_the_documented_effect_for_its_type() {
    for case in cases() {
        let captured = capture(&case);
        damage_first(captured.path(), case.record, halve);

        let (engine, shutdown) = replay(captured.path());
        assert_eq!(
            (case.applied)(&engine),
            case.survives_damage,
            "{}: a torn record — expected the mutation to {} survive, because {}",
            case.record,
            if case.survives_damage { "" } else { "not" },
            case.why
        );
        shutdown.cancel();
    }
}

#[tokio::test]
async fn a_broken_checksum_behaves_exactly_like_a_torn_record() {
    for case in cases() {
        let captured = capture(&case);
        damage_first(captured.path(), case.record, break_checksum);

        let (engine, shutdown) = replay(captured.path());
        assert_eq!(
            (case.applied)(&engine),
            case.survives_damage,
            "{}: a broken checksum must have the same effect as a torn record, \
             and the mutation should {} survive, because {}",
            case.record,
            if case.survives_damage { "" } else { "not" },
            case.why
        );
        shutdown.cancel();
    }
}

#[tokio::test]
async fn everything_before_a_damaged_record_still_replays() {
    // Recovery stops at the bad record; it does not throw away the good ones
    // before it. An implementation that refused the whole segment would be safe
    // and useless.
    let live = tempfile::tempdir().unwrap();
    let captured = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    {
        let engine = Engine::new(config_for(live.path()), shutdown.clone()).unwrap();
        for i in 0..5 {
            engine
                .put_state(format!("keep-{i}"), serde_json::json!(i), None, None)
                .unwrap();
        }
        engine
            .put_state("torn".into(), serde_json::json!("last"), None, None)
            .unwrap();
        copy_wal(live.path(), captured.path());
        shutdown.cancel();
    }
    damage_last(captured.path(), halve);

    let (engine, shutdown) = replay(captured.path());
    for i in 0..5 {
        assert!(
            engine.get_state(&format!("keep-{i}")).is_some(),
            "keep-{i} was written before the damaged record and must survive"
        );
    }
    assert!(
        engine.get_state("torn").is_none(),
        "the damaged record itself must not be applied"
    );
    shutdown.cancel();
}
