//! Epoch fencing, so a promoted replica can stop the old primary writing.
//!
//! W2.3 of `docs/PLAN-MAESTRO.md`. This is the part that makes promotion safe
//! enough to document without a paragraph of warnings.
//!
//! ## The failure it prevents
//!
//! Promotion removes a marker on the new primary. It does nothing to the old
//! one. If the old primary is still alive — a network partition, a host that
//! looked dead and was not, an operator who promoted before stopping it — both
//! nodes then ship WAL segments into the same prefix. Their segments interleave,
//! and the result is not a mess that can be untangled later: replay reads one
//! stream, and the stream is now two histories spliced together.
//!
//! ## How the epoch stops it
//!
//! One small object in the remote prefix holds a counter. Promotion bumps it.
//! Every shipping pass reads it first, and a node whose own epoch is behind the
//! remote one stops shipping and marks itself read-only.
//!
//! The window is one shipping interval, not zero. That is an honest bound and it
//! is the reason this is called *assisted* failover: a lease-based scheme with a
//! real quorum could close it, and that is a consensus system, which the plan
//! puts in the backlog behind an explicit entry criterion.
//!
//! ## Why an object and not a lock
//!
//! Object stores do not offer locks that survive a client dying, and a lock that
//! outlives its holder is worse than none. A monotonic counter needs no liveness:
//! whoever bumped it last wins, and the loser finds out by reading.

use anyhow::{Context, Result};
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use std::path::Path;
use std::sync::Arc;

/// Name of the epoch object inside the remote prefix.
const EPOCH_OBJECT: &str = "EPOCH";
/// Name of the local record of which epoch this node believes it owns.
const EPOCH_FILE: &str = "EPOCH";

/// The epoch this data directory believes it owns.
///
/// A fresh directory is epoch 0. That is deliberately the lowest value: a brand
/// new node must lose to any established primary rather than fencing it off by
/// existing.
pub fn local_epoch(data_dir: &Path) -> u64 {
    std::fs::read_to_string(data_dir.join(EPOCH_FILE))
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(0)
}

/// Record the epoch this node owns.
pub fn set_local_epoch(data_dir: &Path, epoch: u64) -> Result<()> {
    std::fs::create_dir_all(data_dir)?;
    // Atomic, because a torn epoch file reads as 0 and would silently fence the
    // node off from its own prefix.
    let bytes = format!("{epoch}\n").into_bytes();
    let path = data_dir.join(EPOCH_FILE);
    let temp = data_dir.join(format!("{EPOCH_FILE}.tmp"));
    std::fs::write(&temp, &bytes)?;
    std::fs::rename(&temp, &path)?;
    Ok(())
}

/// The epoch currently recorded in the remote prefix.
///
/// A missing object is epoch 0, not an error: the first primary to ship into an
/// empty prefix has nobody to lose to.
pub async fn remote_epoch(store: &Arc<dyn ObjectStore>, prefix: &str) -> Result<u64> {
    let path = ObjectPath::from(format!("{}/{EPOCH_OBJECT}", prefix.trim_end_matches('/')));
    match store.get(&path).await {
        Ok(result) => {
            let bytes = result.bytes().await.context("reading epoch object")?;
            let text = String::from_utf8_lossy(&bytes);
            Ok(text.trim().parse().unwrap_or(0))
        }
        Err(object_store::Error::NotFound { .. }) => Ok(0),
        Err(e) => Err(e).context("fetching epoch object"),
    }
}

/// Claim the next epoch: read the remote value, add one, write it back.
///
/// Returns the epoch now owned. Not atomic against another claimer — object
/// stores give no compare-and-set here — so two simultaneous promotions can land
/// on the same number. Both would then fence *the old primary* and neither would
/// fence the other, which is the failure this cannot solve and consensus can.
/// Promotion is a manual, one-operator act precisely so that case does not arise.
pub async fn claim_next_epoch(store: &Arc<dyn ObjectStore>, prefix: &str) -> Result<u64> {
    let next = remote_epoch(store, prefix).await?.saturating_add(1);
    let path = ObjectPath::from(format!("{}/{EPOCH_OBJECT}", prefix.trim_end_matches('/')));
    store
        .put(&path, PutPayload::from(format!("{next}\n").into_bytes()))
        .await
        .context("writing epoch object")?;
    Ok(next)
}

/// Whether this node may still write to the prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// This node's epoch is at least the remote one.
    Current,
    /// Somebody else claimed a later epoch. This node must stop writing.
    Fenced { local: u64, remote: u64 },
}

/// Compare this node's epoch against the prefix.
///
/// Equal counts as current: the ordinary case is one primary whose local epoch
/// matches what it wrote, and treating equality as fenced would stop every node
/// including the legitimate one.
pub async fn standing(
    store: &Arc<dyn ObjectStore>,
    prefix: &str,
    data_dir: &Path,
) -> Result<Standing> {
    let local = local_epoch(data_dir);
    let remote = remote_epoch(store, prefix).await?;
    Ok(if local >= remote {
        Standing::Current
    } else {
        Standing::Fenced { local, remote }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn store() -> Arc<dyn ObjectStore> {
        Arc::new(InMemory::new())
    }

    #[tokio::test]
    async fn an_empty_prefix_is_epoch_zero() {
        // The first primary has nobody to lose to, so a missing object must not
        // be an error that stops it shipping.
        let store = store();
        assert_eq!(remote_epoch(&store, "luma").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn claiming_advances_the_epoch() {
        let store = store();
        assert_eq!(claim_next_epoch(&store, "luma").await.unwrap(), 1);
        assert_eq!(claim_next_epoch(&store, "luma").await.unwrap(), 2);
        assert_eq!(remote_epoch(&store, "luma").await.unwrap(), 2);
    }

    #[tokio::test]
    async fn a_fresh_directory_loses_to_an_established_primary() {
        // Epoch 0 for a new node is the point: appearing must not fence off a
        // running primary.
        let dir = tempfile::tempdir().unwrap();
        let store = store();
        claim_next_epoch(&store, "luma").await.unwrap();
        assert_eq!(
            standing(&store, "luma", dir.path()).await.unwrap(),
            Standing::Fenced {
                local: 0,
                remote: 1
            }
        );
    }

    #[tokio::test]
    async fn the_current_owner_is_not_fenced_by_its_own_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let store = store();
        let epoch = claim_next_epoch(&store, "luma").await.unwrap();
        set_local_epoch(dir.path(), epoch).unwrap();
        assert_eq!(
            standing(&store, "luma", dir.path()).await.unwrap(),
            Standing::Current
        );
    }

    #[tokio::test]
    async fn a_promotion_fences_the_previous_primary() {
        // The whole point, end to end: the old primary keeps its epoch, the new
        // one claims the next, and the old one discovers it on its next read.
        let old = tempfile::tempdir().unwrap();
        let new = tempfile::tempdir().unwrap();
        let store = store();

        let first = claim_next_epoch(&store, "luma").await.unwrap();
        set_local_epoch(old.path(), first).unwrap();
        assert_eq!(
            standing(&store, "luma", old.path()).await.unwrap(),
            Standing::Current
        );

        // The replica is promoted.
        let second = claim_next_epoch(&store, "luma").await.unwrap();
        set_local_epoch(new.path(), second).unwrap();

        assert_eq!(
            standing(&store, "luma", new.path()).await.unwrap(),
            Standing::Current,
            "the new primary owns the prefix"
        );
        assert_eq!(
            standing(&store, "luma", old.path()).await.unwrap(),
            Standing::Fenced {
                local: first,
                remote: second
            },
            "and the old one must stop writing"
        );
    }

    #[test]
    fn a_torn_epoch_file_reads_as_zero() {
        // Zero is the safe direction: it fences this node rather than letting it
        // believe it owns an epoch it does not. Written atomically so it should
        // not happen, but the reader must not guess high.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(EPOCH_FILE), b"12").unwrap();
        assert_eq!(local_epoch(dir.path()), 12);
        std::fs::write(dir.path().join(EPOCH_FILE), b"not a number").unwrap();
        assert_eq!(local_epoch(dir.path()), 0);
    }

    #[test]
    fn the_local_epoch_survives_a_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        set_local_epoch(dir.path(), 3).unwrap();
        set_local_epoch(dir.path(), 4).unwrap();
        assert_eq!(local_epoch(dir.path()), 4);
    }
}
