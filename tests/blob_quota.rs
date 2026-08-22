//! Object-store bytes per organization — the rest of W5.2.
//!
//! `max_keys` was enforceable straight away because KV keys carry a tenant
//! prefix, so an org's usage is a prefix scan. Blob bytes were not, and the first
//! attempt at this guard was deliberately deleted rather than shipped: with no
//! way to say which bytes belong to whom it would have charged one org for
//! another's storage, which is the exact failure the acceptance criterion
//! forbids.
//!
//! What makes it possible is that the tenant-isolation middleware already records
//! ownership in `sys_collections` on first touch, so every bucket has one owning
//! org.
//!
//! The properties worth pinning, in order of what would hurt most:
//!
//! 1. **One org at its limit does not degrade another.** The acceptance criterion.
//! 2. **A delete gives the bytes back.** Otherwise the quota only ever goes up
//!    and an org that stays within its limit still eventually cannot write.
//! 3. **An overwrite is charged the difference.** Charging the full size makes
//!    updating an object impossible at the limit, which is not what a storage
//!    limit means.
//! 4. **507, not 429.** The caller is out of room; retrying cannot help.

use luma::api::quotas::{guard_blob_write, BlobUsage};
use luma::api::TenantContext;
use std::sync::Arc;

/// Just the pieces the guard needs: SQLite, the ownership registry and where
/// the blobs live. No router, because the guard does not need one — which is
/// why it takes them as arguments rather than an `AppState`.
struct Harness {
    sqlite: luma::sqlite::SqliteService,
    accounts: Arc<luma::api::accounts::AccountsService>,
    blobs_root: std::path::PathBuf,
    _dir: tempfile::TempDir,
}

impl Harness {
    fn store(&self) -> luma::api::quotas::BlobQuotaStore<'_> {
        luma::api::quotas::BlobQuotaStore {
            sqlite: Some(&self.sqlite),
            accounts: Some(&self.accounts),
            blobs_root: &self.blobs_root,
        }
    }
}

async fn start() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let blobs_root = dir.path().join("data").join("blobs");
    std::fs::create_dir_all(&blobs_root).unwrap();
    let sqlite = luma::sqlite::SqliteService::new(dir.path().join("meta.db")).unwrap();
    let accounts = Arc::new(luma::api::accounts::AccountsService::new(Arc::new(
        sqlite.clone(),
    )));
    Harness {
        sqlite,
        accounts,
        blobs_root,
        _dir: dir,
    }
}
fn ctx(org: &str, limit_bytes: Option<u64>) -> TenantContext {
    let quotas = match limit_bytes {
        Some(limit) => serde_json::json!({ "max_blob_bytes": limit }),
        None => serde_json::json!({}),
    };
    TenantContext {
        tenant_id: Some(org.to_string()),
        user_id: None,
        role: "member".to_string(),
        platform_admin: false,
        permissions: serde_json::json!({}),
        quotas,
    }
}

/// Put `bytes` bytes into a bucket the org owns, updating the accounting the way
/// the route does.
async fn store(harness: &Harness, org: &str, bucket: &str, key: &str, bytes: usize) {
    harness.accounts.register_collection(bucket, org).await.ok();
    let directory = harness.blobs_root.join(bucket);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(key);
    let replacing = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    std::fs::write(&path, vec![b'x'; bytes]).unwrap();
    luma::api::quotas::record_blob_delta(
        Some(&harness.sqlite),
        &ctx(org, None),
        bytes as i64 - replacing as i64,
    )
    .await;
}

async fn usage_of(harness: &Harness, org: &str) -> u64 {
    let owned = harness.accounts.names_owned_by(org).await.unwrap();
    BlobUsage::new(harness.sqlite.clone())
        .bytes_for(org, &harness.blobs_root, &owned)
        .await
        .unwrap()
}

#[tokio::test]
async fn one_org_at_its_limit_does_not_degrade_another() {
    // The acceptance criterion from the SPEC, stated as a test.
    let harness = start().await;
    store(&harness, "acme", "acme-files", "a", 1000).await;
    store(&harness, "globex", "globex-files", "a", 10).await;

    assert!(
        guard_blob_write(harness.store(), &ctx("acme", Some(1000)), 1, 0)
            .await
            .is_err(),
        "acme holds 1000 of 1000 and must be refused"
    );
    assert!(
        guard_blob_write(harness.store(), &ctx("globex", Some(1000)), 500, 0)
            .await
            .is_ok(),
        "globex holds 10 of 1000 and must be unaffected by acme being full"
    );
}

#[tokio::test]
async fn a_delete_gives_the_bytes_back() {
    // Without this the quota only ever climbs, and an org that stays well within
    // its limit still eventually cannot write.
    let harness = start().await;
    store(&harness, "acme", "files", "big", 900).await;
    assert_eq!(usage_of(&harness, "acme").await, 900);
    assert!(
        guard_blob_write(harness.store(), &ctx("acme", Some(1000)), 200, 0)
            .await
            .is_err()
    );

    // Delete, accounted the way the route does.
    let path = harness.blobs_root.join("files").join("big");
    let freed = std::fs::metadata(&path).unwrap().len();
    std::fs::remove_file(&path).unwrap();
    luma::api::quotas::record_blob_delta(
        Some(&harness.sqlite),
        &ctx("acme", None),
        -(freed as i64),
    )
    .await;

    assert_eq!(usage_of(&harness, "acme").await, 0);
    assert!(
        guard_blob_write(harness.store(), &ctx("acme", Some(1000)), 200, 0)
            .await
            .is_ok(),
        "after freeing the space the write must be allowed"
    );
}

#[tokio::test]
async fn an_overwrite_is_charged_only_the_difference() {
    // An org exactly at its limit must still be able to replace what it has.
    // Charging the full size would make the data read-only the moment the limit
    // is reached.
    let harness = start().await;
    store(&harness, "acme", "files", "obj", 1000).await;
    assert_eq!(usage_of(&harness, "acme").await, 1000);

    // Same size, replacing itself: net zero, so it fits.
    assert!(
        guard_blob_write(harness.store(), &ctx("acme", Some(1000)), 1000, 1000)
            .await
            .is_ok(),
        "replacing an object with one of the same size costs nothing"
    );
    // One byte larger does not.
    assert!(
        guard_blob_write(harness.store(), &ctx("acme", Some(1000)), 1001, 1000)
            .await
            .is_err()
    );
    // And smaller is fine.
    assert!(
        guard_blob_write(harness.store(), &ctx("acme", Some(1000)), 10, 1000)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn being_out_of_room_answers_507_not_429() {
    // 429 invites a retry loop that cannot succeed.
    let harness = start().await;
    store(&harness, "acme", "files", "a", 100).await;
    let error = guard_blob_write(harness.store(), &ctx("acme", Some(100)), 1, 0)
        .await
        .unwrap_err();
    let response = axum::response::IntoResponse::into_response(error);
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INSUFFICIENT_STORAGE
    );
}

#[tokio::test]
async fn no_limit_configured_means_no_limit() {
    // Keeps this from behaving like a global switch: the limit belongs to the
    // caller, not to the instance.
    let harness = start().await;
    store(&harness, "acme", "files", "a", 10_000).await;
    assert!(
        guard_blob_write(harness.store(), &ctx("acme", None), 1_000_000, 0)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn a_platform_caller_is_not_charged_to_any_org() {
    // No tenant means there is nothing to attribute the bytes to. Inventing an
    // attribution is how one org ends up paying for another's storage, which is
    // why the first version of this guard was deleted instead of shipped.
    let harness = start().await;
    store(&harness, "acme", "files", "a", 1000).await;
    let platform = TenantContext {
        tenant_id: None,
        user_id: None,
        role: "admin".to_string(),
        platform_admin: true,
        permissions: serde_json::json!({}),
        quotas: serde_json::json!({ "max_blob_bytes": 1 }),
    };
    assert!(guard_blob_write(harness.store(), &platform, 10_000, 0)
        .await
        .is_ok());
}

#[tokio::test]
async fn usage_is_seeded_from_the_filesystem_the_first_time() {
    // The stored total starts from a walk, so an org that existed before this
    // accounting did is charged for what it already holds rather than starting
    // from zero.
    let harness = start().await;
    harness
        .accounts
        .register_collection("legacy", "acme")
        .await
        .ok();
    let directory = harness.blobs_root.join("legacy");
    std::fs::create_dir_all(directory.join("nested")).unwrap();
    std::fs::write(directory.join("a"), vec![b'x'; 300]).unwrap();
    std::fs::write(directory.join("nested").join("b"), vec![b'x'; 700]).unwrap();

    // No `record_blob_delta` was called, so this can only come from the walk.
    assert_eq!(
        usage_of(&harness, "acme").await,
        1000,
        "the seed must count nested objects too"
    );
}

#[tokio::test]
async fn a_recount_repairs_a_total_that_drifted() {
    // Files can change out of band — a restore, a human with rm. The stored
    // total then disagrees with the disk, and a quota that is quietly wrong is
    // worse than one that is absent.
    let harness = start().await;
    store(&harness, "acme", "files", "a", 1000).await;
    assert_eq!(usage_of(&harness, "acme").await, 1000);

    // Remove it behind the accounting's back.
    std::fs::remove_file(harness.blobs_root.join("files").join("a")).unwrap();
    assert_eq!(
        usage_of(&harness, "acme").await,
        1000,
        "the stored total does not notice, which is exactly why recount exists"
    );

    let owned = harness.accounts.names_owned_by("acme").await.unwrap();
    let repaired = BlobUsage::new(harness.sqlite.clone())
        .recount("acme", &harness.blobs_root, &owned)
        .await
        .unwrap();
    assert_eq!(repaired, 0);
    assert_eq!(usage_of(&harness, "acme").await, 0);
}

#[tokio::test]
async fn bytes_are_attributed_per_bucket_owner_not_per_writer() {
    // Two orgs, two buckets. The ownership registry is what makes the split
    // real; without it this whole guard would be charging the wrong org.
    let harness = start().await;
    store(&harness, "acme", "acme-a", "x", 100).await;
    store(&harness, "acme", "acme-b", "x", 200).await;
    store(&harness, "globex", "globex-a", "x", 5000).await;

    assert_eq!(usage_of(&harness, "acme").await, 300);
    assert_eq!(usage_of(&harness, "globex").await, 5000);
}

// ─── Queue messages and vectors ──────────────────────────────────────────────

#[tokio::test]
async fn one_orgs_full_queue_does_not_block_another() {
    // The acceptance criterion again, for messages. Queues are already isolated
    // by directory, so this is checking that the guard reads the *right*
    // subtree — a guard that counted the whole root would refuse org B because
    // org A is full.
    let harness = start().await;
    let queues = harness.blobs_root.parent().unwrap().join("queues");
    for org in ["acme", "globex"] {
        std::fs::create_dir_all(queues.join(format!("t_{org}")).join("jobs")).unwrap();
    }
    for i in 0..5 {
        std::fs::write(
            queues.join("t_acme").join("jobs").join(format!("m{i}")),
            b"{}",
        )
        .unwrap();
    }
    std::fs::write(queues.join("t_globex").join("jobs").join("m0"), b"{}").unwrap();

    assert!(
        luma::api::quotas::guard_queue_write(&queues, &ctx_with(&acme_quota(5)), 1).is_err(),
        "acme holds 5 of 5 and must be refused"
    );
    assert!(
        luma::api::quotas::guard_queue_write(&queues, &globex_ctx(5), 1).is_ok(),
        "globex holds 1 of 5 and must be unaffected"
    );
}

#[tokio::test]
async fn a_platform_caller_is_not_charged_for_queue_messages() {
    // A platform caller writes to the shared top-level namespace, not to any
    // org's subtree, so there is nothing to charge.
    let harness = start().await;
    let queues = harness.blobs_root.parent().unwrap().join("queues");
    std::fs::create_dir_all(&queues).unwrap();
    let platform = TenantContext {
        tenant_id: None,
        user_id: None,
        role: "admin".to_string(),
        platform_admin: true,
        permissions: serde_json::json!({}),
        quotas: serde_json::json!({ "max_queue_messages": 1 }),
    };
    assert!(luma::api::quotas::guard_queue_write(&queues, &platform, 100).is_ok());
}

#[tokio::test]
async fn a_missing_queue_directory_counts_as_empty() {
    // A brand-new org has no directory yet. Treating an unreadable directory as
    // an error would refuse the very first enqueue.
    let harness = start().await;
    let queues = harness.blobs_root.parent().unwrap().join("queues");
    assert!(luma::api::quotas::guard_queue_write(&queues, &ctx_with(&acme_quota(1)), 1).is_ok());
}

fn acme_quota(limit: u64) -> serde_json::Value {
    serde_json::json!({ "max_queue_messages": limit })
}

fn ctx_with(quotas: &serde_json::Value) -> TenantContext {
    TenantContext {
        tenant_id: Some("acme".to_string()),
        user_id: None,
        role: "member".to_string(),
        platform_admin: false,
        permissions: serde_json::json!({}),
        quotas: quotas.clone(),
    }
}

fn globex_ctx(limit: u64) -> TenantContext {
    TenantContext {
        tenant_id: Some("globex".to_string()),
        user_id: None,
        role: "member".to_string(),
        platform_admin: false,
        permissions: serde_json::json!({ "": "" }),
        quotas: serde_json::json!({ "max_queue_messages": limit }),
    }
}
