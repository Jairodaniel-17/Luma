//! `RPOPLPUSH`/`LMOVE` move an element between two keys atomically.
//!
//! The guarantee under test is the one a client buys by using `BRPOPLPUSH` for
//! reliable delivery: kombu's unacked queue exists so a task is not lost when a
//! worker dies mid-delivery. Implemented as a pop followed by a push, a process
//! death in between dropped the element — the exact failure the primitive is
//! meant to prevent.
//!
//! ## Why this does not kill a process
//!
//! `tests/crash_recovery.rs` kills the real binary, which is right for "a
//! confirmed write survives". It is the wrong instrument here, because landing
//! the kill inside the microseconds between two writes is timing-dependent: a
//! green run would prove nothing, and the same test could pass on a build where
//! the window is wide open.
//!
//! So this demonstrates the mechanism instead, deterministically:
//!
//! 1. A move writes **one** WAL record naming both keys, not two records.
//! 2. Truncating that record — a torn write, which is what a crash mid-append
//!    leaves behind — replays as if the move never happened. The element is
//!    still in the source, and in one place only.
//! 3. The intact record replays with the element in the destination, once.
//!
//! Together those say: there is no reachable state in which the element is in
//! neither list, or in both.
//!
//! ## Replaying from the WAL alone
//!
//! Cases 2 and 3 copy **only the WAL** into a fresh directory, leaving the redb
//! projection behind. That is not a shortcut; it is the crash scenario stated
//! exactly. The write order is WAL first, projection second, and the projection
//! commits with `Durability::Eventual` — so a crash rolls it back to its last
//! checkpoint and the WAL is what remains. Truncating the WAL of a *cleanly
//! closed* directory would prove nothing, because a clean close flushes the
//! projection and replay then skips the record as already applied.

use luma::config::Config;
use luma::engine::structures::Structures;
use luma::engine::Engine;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

fn config_for(dir: &Path) -> Config {
    Config {
        data_dir: Some(dir.to_str().unwrap().to_string()),
        // No snapshots during the test: a snapshot would fold the WAL away and
        // there would be nothing left to truncate.
        snapshot_interval_secs: 86_400,
        // Per-write rather than the default group commit. Group commit holds
        // records in a buffer until a batch fills or an interval passes, so
        // whether a record is on disk at any instant is a timing question — and
        // a durability test whose subject may or may not be there is not a test.
        // The window group commit opens is a separate, documented property; this
        // file is about what happens to a record that *did* reach the disk.
        wal_sync_mode: "per_write".to_string(),
        ..Config::default()
    }
}

/// The WAL segments in a data directory, in order.
fn wal_segments(dir: &Path) -> Vec<PathBuf> {
    let mut segments: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("data dir must exist")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("events-") && n.ends_with(".log"))
        })
        .collect();
    segments.sort();
    segments
}

/// Every WAL line across every segment.
fn wal_lines(dir: &Path) -> Vec<String> {
    wal_segments(dir)
        .into_iter()
        .flat_map(|path| {
            std::fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|l| !l.trim().is_empty())
        .collect()
}

/// Copy only the WAL (and any snapshot) into a fresh directory.
///
/// Deliberately leaves the redb projection behind: after a crash it has rolled
/// back to its last checkpoint, and the WAL is what recovery actually has to
/// work from. Copying the projection too would test a clean shutdown wearing a
/// crash costume.
fn copy_wal(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from)
        .expect("data dir must exist")
        .flatten()
    {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_wal = name.starts_with("events-") && name.ends_with(".log");
        if is_wal || name == "snapshot.json" {
            std::fs::copy(entry.path(), to.join(name.as_ref())).expect("copy");
        }
    }
}

fn list_contents(engine: &Engine, key: &str) -> Vec<String> {
    Structures::new(engine)
        .load(key)
        .expect("load must not error")
        .map(|(structure, _)| {
            structure
                .as_list()
                .expect("must be a list")
                .iter()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_move_is_one_wal_record_naming_both_keys() {
    let dir = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
    let structures = Structures::new(&engine);

    structures
        .mutate(
            "queue",
            luma::engine::structures::Structure::empty_list,
            |s| s.rpush(vec![b"job-1".to_vec()]),
        )
        .unwrap();
    let before = wal_lines(dir.path()).len();

    let moved = structures
        .move_element("queue", "inflight", false, true)
        .unwrap();
    assert_eq!(moved.as_deref(), Some(b"job-1".as_slice()));

    let added: Vec<String> = wal_lines(dir.path()).into_iter().skip(before).collect();
    assert_eq!(
        added.len(),
        1,
        "a move must be one record, not a pop and a push: {added:?}"
    );
    let record: serde_json::Value = serde_json::from_str(&added[0]).expect("valid WAL JSON");
    // The envelope carries the event; find it whatever the wrapper is named.
    let text = record.to_string();
    assert!(
        text.contains("state_batch"),
        "the record must be a batch: {text}"
    );
    assert!(
        text.contains("queue") && text.contains("inflight"),
        "the single record must name both keys: {text}"
    );

    shutdown.cancel();
}

#[tokio::test]
async fn a_torn_move_record_replays_as_if_it_never_happened() {
    let dir = tempfile::tempdir().unwrap();

    // ── write the move, and take the WAL while the process is still up ──────
    //
    // Copied before shutting down on purpose: a clean close takes a snapshot and
    // rotates the WAL away, which is precisely what a crash does not do.
    let replayed = tempfile::tempdir().unwrap();
    {
        let shutdown = CancellationToken::new();
        let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
        let structures = Structures::new(&engine);
        structures
            .mutate(
                "queue",
                luma::engine::structures::Structure::empty_list,
                |s| s.rpush(vec![b"job-1".to_vec()]),
            )
            .unwrap();
        structures
            .move_element("queue", "inflight", false, true)
            .unwrap();
        copy_wal(dir.path(), replayed.path());

        shutdown.cancel();
    }

    // ── replay the WAL alone, with the last record torn ─────────────────────
    let segment = wal_segments(replayed.path())
        .pop()
        .expect("there must be a WAL segment");
    let text = std::fs::read_to_string(&segment).unwrap();
    let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let last = lines.pop().expect("the move record must be there");
    assert!(
        last.contains("state_batch"),
        "the last record should be the move: {last}"
    );
    // Half of it reached the disk. This is what a torn write looks like.
    let mut rebuilt = lines.join("\n");
    rebuilt.push('\n');
    rebuilt.push_str(&last[..last.len() / 2]);
    std::fs::write(&segment, rebuilt).unwrap();

    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(replayed.path()), shutdown.clone()).unwrap();

    assert_eq!(
        list_contents(&engine, "queue"),
        vec!["job-1".to_string()],
        "a torn move must leave the element in the source"
    );
    assert!(
        list_contents(&engine, "inflight").is_empty(),
        "and it must not appear in the destination as well"
    );
    shutdown.cancel();
}

#[tokio::test]
async fn an_intact_move_record_replays_the_element_exactly_once() {
    let dir = tempfile::tempdir().unwrap();
    let replayed = tempfile::tempdir().unwrap();
    {
        let shutdown = CancellationToken::new();
        let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
        let structures = Structures::new(&engine);
        structures
            .mutate(
                "queue",
                luma::engine::structures::Structure::empty_list,
                |s| s.rpush(vec![b"job-1".to_vec(), b"job-2".to_vec()]),
            )
            .unwrap();
        structures
            .move_element("queue", "inflight", false, true)
            .unwrap();
        copy_wal(dir.path(), replayed.path());
        shutdown.cancel();
    }

    // The same WAL-only replay as the torn case, so the two differ in exactly
    // one thing: whether the record is complete.
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(replayed.path()), shutdown.clone()).unwrap();
    assert_eq!(list_contents(&engine, "queue"), vec!["job-1".to_string()]);
    assert_eq!(
        list_contents(&engine, "inflight"),
        vec!["job-2".to_string()],
        "the moved element must be in the destination exactly once"
    );
    shutdown.cancel();
}

#[tokio::test]
async fn emptying_the_source_deletes_it_in_the_same_record() {
    // The source list had one element, so after the move it is gone — and that
    // deletion has to be inside the same record, or a crash between the two
    // leaves an empty husk that makes EXISTS and LLEN disagree.
    let dir = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
    let structures = Structures::new(&engine);
    structures
        .mutate(
            "queue",
            luma::engine::structures::Structure::empty_list,
            |s| s.rpush(vec![b"only".to_vec()]),
        )
        .unwrap();
    let before = wal_lines(dir.path()).len();
    structures
        .move_element("queue", "inflight", false, true)
        .unwrap();

    let added: Vec<String> = wal_lines(dir.path()).into_iter().skip(before).collect();
    assert_eq!(added.len(), 1, "still one record: {added:?}");
    assert!(
        added[0].contains("delete"),
        "the emptied source must be deleted in the same record: {}",
        added[0]
    );
    assert!(structures.load("queue").unwrap().is_none());
    shutdown.cancel();
}

#[tokio::test]
async fn a_rejected_destination_does_not_consume_the_element() {
    let dir = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
    let structures = Structures::new(&engine);
    structures
        .mutate(
            "queue",
            luma::engine::structures::Structure::empty_list,
            |s| s.rpush(vec![b"job".to_vec()]),
        )
        .unwrap();
    structures
        .mutate(
            "taken",
            luma::engine::structures::Structure::empty_set,
            |s| s.sadd(vec![b"member".to_vec()]),
        )
        .unwrap();

    assert!(
        structures
            .move_element("queue", "taken", false, true)
            .is_err(),
        "a destination holding a set must be refused"
    );
    assert_eq!(
        list_contents(&engine, "queue"),
        vec!["job".to_string()],
        "and the refusal must not cost the source its element"
    );
    shutdown.cancel();
}

#[tokio::test]
async fn a_self_move_rotates_without_losing_or_duplicating() {
    // `RPOPLPUSH mylist mylist` is the documented round-robin idiom, and the one
    // case where source and destination are the same key. Treated as two keys it
    // would prepare two revisions for one key and the second write would fail
    // its own compare-and-swap.
    let dir = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
    let structures = Structures::new(&engine);
    structures
        .mutate(
            "ring",
            luma::engine::structures::Structure::empty_list,
            |s| s.rpush(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]),
        )
        .unwrap();

    let moved = structures
        .move_element("ring", "ring", false, true)
        .unwrap();
    assert_eq!(moved.as_deref(), Some(b"c".as_slice()));
    assert_eq!(
        list_contents(&engine, "ring"),
        vec!["c".to_string(), "a".to_string(), "b".to_string()]
    );
    shutdown.cancel();
}

#[tokio::test]
async fn an_empty_source_moves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config_for(dir.path()), shutdown.clone()).unwrap();
    let structures = Structures::new(&engine);
    assert_eq!(
        structures
            .move_element("nothing", "somewhere", false, true)
            .unwrap(),
        None
    );
    assert!(structures.load("somewhere").unwrap().is_none());
    shutdown.cancel();
}
