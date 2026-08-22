//! Assisted failover: the health check a proxy uses, and epoch fencing.
//!
//! W2.3 of `docs/PLAN-MAESTRO.md`. **Not automatic HA** — nothing here elects
//! anything. What it does is make a manual promotion safe enough to document
//! without a paragraph of warnings:
//!
//! * A proxy can tell a primary from a replica by status code alone.
//! * A promoted replica claims the next epoch, and the old primary stops
//!   shipping on its next pass rather than interleaving its WAL with the new
//!   one's.
//!
//! The window is one shipping interval, not zero. Closing it needs a lease with
//! a real quorum, which is a consensus system, which the plan keeps in the
//! backlog behind an explicit entry criterion.

use luma::fencing::{claim_next_epoch, local_epoch, set_local_epoch, standing, Standing};
use object_store::memory::InMemory;
use object_store::ObjectStore;
use std::sync::Arc;

fn store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

#[tokio::test]
async fn shipping_stops_once_another_node_claims_the_prefix() {
    // The property that matters. Two nodes shipping into one prefix interleave
    // their segments, and replay then reads two histories spliced together —
    // which is not a mess that can be untangled afterwards.
    let old = tempfile::tempdir().unwrap();
    let store = store();

    // The old primary owns epoch 1 and has a segment to ship.
    let epoch = claim_next_epoch(&store, "luma").await.unwrap();
    set_local_epoch(old.path(), epoch).unwrap();
    std::fs::write(old.path().join("events-000001.log"), b"{}\n").unwrap();

    let mut state = luma::wal_ship::ShipState::default();
    let report =
        luma::wal_ship::ship_once(&store, "luma", old.path(), &mut state, |b| Ok(b.to_vec()))
            .await
            .unwrap();
    assert!(!report.fenced, "the owner must be allowed to ship");
    assert!(
        !report.uploaded.is_empty(),
        "and must actually have shipped: {report:?}"
    );

    // Somebody promotes a replica.
    claim_next_epoch(&store, "luma").await.unwrap();

    // The old primary has a new segment and tries again.
    std::fs::write(old.path().join("events-000002.log"), b"{}\n").unwrap();
    let mut state = luma::wal_ship::ShipState::default();
    let report =
        luma::wal_ship::ship_once(&store, "luma", old.path(), &mut state, |b| Ok(b.to_vec()))
            .await
            .unwrap();
    assert!(
        report.fenced,
        "a superseded node must report that it was fenced: {report:?}"
    );
    assert_eq!(
        report.uploaded.len(),
        0,
        "and must not have written anything: {report:?}"
    );
}

#[tokio::test]
async fn being_fenced_is_distinguishable_from_having_nothing_to_ship() {
    // An empty report and a fenced report would look identical without the flag,
    // and "quiet" and "locked out" are very different things to page someone
    // about.
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    let epoch = claim_next_epoch(&store, "luma").await.unwrap();
    set_local_epoch(dir.path(), epoch).unwrap();

    // Nothing to ship, but not fenced.
    let mut state = luma::wal_ship::ShipState::default();
    let quiet =
        luma::wal_ship::ship_once(&store, "luma", dir.path(), &mut state, |b| Ok(b.to_vec()))
            .await
            .unwrap();
    assert!(!quiet.fenced);
    assert!(quiet.uploaded.is_empty());
}

#[tokio::test]
async fn a_first_primary_on_an_empty_prefix_is_not_fenced() {
    // Epoch 0 against a missing object. Treating a fresh prefix as hostile would
    // stop the very first node from ever shipping.
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    assert_eq!(local_epoch(dir.path()), 0);
    assert_eq!(
        standing(&store, "luma", dir.path()).await.unwrap(),
        Standing::Current
    );
}

#[tokio::test]
async fn the_epoch_survives_a_restart() {
    // It lives on disk for the same reason the replica marker does: a node that
    // forgot its epoch on restart would read the remote one, find it higher, and
    // fence itself out of its own prefix.
    let dir = tempfile::tempdir().unwrap();
    let store = store();
    let epoch = claim_next_epoch(&store, "luma").await.unwrap();
    set_local_epoch(dir.path(), epoch).unwrap();

    // Simulate a restart: nothing in memory, only what is on disk.
    assert_eq!(local_epoch(dir.path()), epoch);
    assert_eq!(
        standing(&store, "luma", dir.path()).await.unwrap(),
        Standing::Current
    );
}
