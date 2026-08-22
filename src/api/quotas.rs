//! Per-organization quotas.
//!
//! W5.2 of `docs/PLAN-MAESTRO.md` (B.1 of the older roadmap). Multi-tenancy that
//! isolates but does not limit is only half of it: one organization can still
//! fill the disk, and every other organization finds out at the same moment the
//! operator does.
//!
//! `TenantContext.quotas` already travelled from the api key record as an
//! untyped JSON blob and was never read by anything. This gives it a type and
//! enforces it.
//!
//! ## What "exceeded" means here
//!
//! A quota rejects the *write that would cross the line*, not writes after it.
//! So a limit of 100 keys admits the hundredth and refuses the hundred-and-first
//! — which is what makes the number in the config the number an operator can
//! reason about.
//!
//! Reads are never refused. An organization at its limit can still get its data
//! out, which is the whole point of telling it to clean up.

use serde::{Deserialize, Serialize};

/// Limits for one organization. `None` means unlimited.
///
/// Every field is optional and defaults to unlimited, so an existing api key
/// record with `{}` — which is what they all carry today — keeps behaving
/// exactly as before.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Quotas {
    /// Maximum keys in the organization's keyspace.
    #[serde(default)]
    pub max_keys: Option<u64>,
    /// Maximum vectors across the organization's collections.
    #[serde(default)]
    pub max_vectors: Option<u64>,
    /// Maximum total bytes stored in the object store.
    #[serde(default)]
    pub max_blob_bytes: Option<u64>,
    /// Maximum messages held across the organization's queues.
    #[serde(default)]
    pub max_queue_messages: Option<u64>,
}

impl Quotas {
    /// Parse from the untyped value carried on `TenantContext`.
    ///
    /// Unparseable content yields unlimited rather than an error. That is the
    /// deliberate choice: a malformed quota record must not lock an
    /// organization out of its own data, and the alternative — refusing every
    /// write until someone fixes a JSON blob — turns a config typo into an
    /// outage. The mistake is loud in the logs instead.
    pub fn from_value(value: &serde_json::Value) -> Self {
        if value.is_null() {
            return Self::default();
        }
        match serde_json::from_value::<Quotas>(value.clone()) {
            Ok(quotas) => quotas,
            Err(e) => {
                tracing::warn!(
                    "quota record is not readable, treating as unlimited: {e}; value = {value}"
                );
                Self::default()
            }
        }
    }

    /// Whether anything is limited at all. Lets a caller skip the cost of
    /// measuring usage for the common unlimited case.
    pub fn is_unlimited(&self) -> bool {
        self.max_keys.is_none()
            && self.max_vectors.is_none()
            && self.max_blob_bytes.is_none()
            && self.max_queue_messages.is_none()
    }
}

/// What a caller is trying to consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resource {
    Keys,
    Vectors,
    BlobBytes,
    QueueMessages,
}

impl Resource {
    pub fn name(self) -> &'static str {
        match self {
            Resource::Keys => "keys",
            Resource::Vectors => "vectors",
            Resource::BlobBytes => "blob_bytes",
            Resource::QueueMessages => "queue_messages",
        }
    }
}

/// A quota that would be crossed.
#[derive(Clone, Debug, PartialEq)]
pub struct Exceeded {
    pub resource: Resource,
    pub limit: u64,
    pub current: u64,
    pub requested: u64,
}

impl std::fmt::Display for Exceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Names all three numbers, because "quota exceeded" alone leaves the
        // caller unable to tell whether they need to delete one thing or a
        // thousand.
        write!(
            f,
            "organization quota for {} exceeded: {} in use, {} requested, limit {}",
            self.resource.name(),
            self.current,
            self.requested,
            self.limit
        )
    }
}

impl Quotas {
    /// Check whether consuming `requested` more of `resource` stays within the
    /// limit, given `current` usage.
    pub fn check(&self, resource: Resource, current: u64, requested: u64) -> Result<(), Exceeded> {
        let limit = match resource {
            Resource::Keys => self.max_keys,
            Resource::Vectors => self.max_vectors,
            Resource::BlobBytes => self.max_blob_bytes,
            Resource::QueueMessages => self.max_queue_messages,
        };
        let Some(limit) = limit else {
            return Ok(());
        };
        // saturating_add so a caller cannot get under the limit by overflowing.
        if current.saturating_add(requested) > limit {
            return Err(Exceeded {
                resource,
                limit,
                current,
                requested,
            });
        }
        Ok(())
    }
}

// ── enforcement ──────────────────────────────────────────────────────────────

use crate::api::errors::ApiError;
use crate::api::TenantContext;
use crate::engine::Engine;
use axum::http::StatusCode;

/// Map an exceeded quota to the HTTP response.
///
/// 507 Insufficient Storage rather than 429: the caller is not being rate
/// limited, they are out of room, and retrying the identical request will not
/// help. A 429 would invite exactly the retry loop that does nothing.
pub fn to_api_error(exceeded: Exceeded) -> ApiError {
    ApiError::new(
        StatusCode::INSUFFICIENT_STORAGE,
        "quota_exceeded",
        exceeded.to_string(),
    )
}

/// Number of keys currently in the caller's keyspace.
///
/// Counted rather than tracked, which is honest about the cost: it is a scan
/// bounded by the same walk limit the keyspace commands use. Tracking a running
/// total would be cheaper but has to survive replay and compaction to stay
/// truthful, and a quota that drifts is worse than one that costs a scan.
fn key_count(engine: &Engine, ctx: &TenantContext) -> u64 {
    let prefix = ctx.tenant_id.as_ref().map(|t| format!("{t}:"));
    engine.list_state(prefix.as_deref(), MAX_USAGE_WALK).len() as u64
}

/// Upper bound on a usage scan. A caller past this is over any realistic quota
/// anyway, so the count saturating here cannot let a write through that should
/// have been refused.
const MAX_USAGE_WALK: usize = 1_000_000;

/// Refuse a key write that would cross the organization's key quota.
///
/// Skips the scan entirely when nothing is limited, which is every deployment
/// that has not opted in.
/// Takes the engine rather than the whole `AppState`: counting keys needs
/// nothing else, and the narrower dependency is what makes this testable
/// without standing up a router.
pub fn guard_key_write(
    engine: &Engine,
    ctx: &TenantContext,
    new_keys: u64,
) -> Result<(), ApiError> {
    let quotas = Quotas::from_value(&ctx.quotas);
    if quotas.max_keys.is_none() {
        return Ok(());
    }
    quotas
        .check(Resource::Keys, key_count(engine, ctx), new_keys)
        .map_err(|exceeded| {
            tracing::warn!(
                tenant = ?ctx.tenant_id,
                resource = exceeded.resource.name(),
                "write refused: {exceeded}"
            );
            to_api_error(exceeded)
        })
}

// ── not yet enforced: blob bytes and queue messages ──────────────────────────
//
// `max_blob_bytes` and `max_queue_messages` parse and are honoured by
// `Quotas::check`, but nothing calls them yet, and that is deliberate rather
// than unfinished.
//
// Keys are enforceable because the keyspace is tenant-prefixed, so counting a
// caller's usage is a prefix scan. Blobs are not: the layout is
// `blobs/{bucket}/…` with ownership recorded separately in `sys_collections`,
// so a directory walk measures *every* organization's bytes. Charging one
// organization for another's storage would refuse org B's write because org A
// filled the disk — precisely the failure the acceptance criterion for this
// item forbids, and worse than having no quota at all.
//
// Enforcing them needs the ownership index consulted per bucket, which is an
// async lookup this synchronous guard cannot make. That is the next step, not
// a shortcut to take now.

// ── Object-store bytes per organization ──────────────────────────────────────
//
// The remaining half of W5.2. `max_keys` was enforceable straight away because
// KV keys carry a tenant prefix, so an org's usage is a prefix scan. Blob bytes
// were not, and the first attempt at a guard was deliberately deleted: without a
// way to say which bytes belong to whom, it would have charged one org for
// another's storage — the exact failure the acceptance criterion forbids.
//
// What makes it possible now is that the tenant-isolation middleware already
// records ownership in `sys_collections` on first touch, so every bucket has
// exactly one owning org.
//
// ## Why the total is stored rather than measured
//
// Measuring means walking the org's buckets on every write, which is O(files) on
// a hot path — an org with a hundred thousand objects would pay for all of them
// to store one. So the total lives in SQLite and is adjusted by each write and
// delete.
//
// In SQLite rather than in memory on purpose. An in-memory counter has to be
// rebuilt on every restart, and a rebuild is the walk we were avoiding; worse, a
// process that restarts often would spend its life walking. The cost is one
// small query per blob write, which is far below the cost of the write itself.
//
// The total is seeded by walking **once**, the first time an org is seen. If the
// files change out of band — someone deletes from the filesystem — the total
// drifts, and `recount` exists for exactly that. Said plainly because a quota
// that is quietly wrong is worse than one that is absent.

/// Bytes an organization holds in the object store.
pub struct BlobUsage {
    sqlite: crate::sqlite::SqliteService,
}

impl BlobUsage {
    pub fn new(sqlite: crate::sqlite::SqliteService) -> Self {
        Self { sqlite }
    }

    async fn ensure_table(&self) -> anyhow::Result<()> {
        self.sqlite
            .execute(
                "CREATE TABLE IF NOT EXISTS sys_blob_usage (
                    org_id TEXT PRIMARY KEY,
                    bytes INTEGER NOT NULL
                )"
                .to_string(),
                vec![],
            )
            .await?;
        Ok(())
    }

    /// Current bytes for an org, seeding from a walk the first time.
    pub async fn bytes_for(
        &self,
        org: &str,
        blobs_root: &std::path::Path,
        owned_buckets: &[String],
    ) -> anyhow::Result<u64> {
        self.ensure_table().await?;
        let rows = self
            .sqlite
            .query(
                "SELECT bytes FROM sys_blob_usage WHERE org_id = ?".to_string(),
                vec![serde_json::json!(org)],
            )
            .await?;
        if let Some(row) = rows.first() {
            let bytes = row
                .get("bytes")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                .max(0) as u64;
            return Ok(bytes);
        }

        // First sighting: measure once, then remember.
        let measured = walk_bytes(blobs_root, owned_buckets);
        self.sqlite
            .execute(
                "INSERT OR REPLACE INTO sys_blob_usage (org_id, bytes) VALUES (?, ?)".to_string(),
                vec![serde_json::json!(org), serde_json::json!(measured as i64)],
            )
            .await?;
        Ok(measured)
    }

    /// Adjust the stored total. `delta` may be negative for a delete.
    ///
    /// Clamped at zero: a total that went negative would make the next write
    /// look free, and a stuck-at-zero quota is a quota that is not enforced.
    pub async fn adjust(&self, org: &str, delta: i64) -> anyhow::Result<()> {
        self.ensure_table().await?;
        self.sqlite
            .execute(
                "UPDATE sys_blob_usage SET bytes = MAX(0, bytes + ?) WHERE org_id = ?".to_string(),
                vec![serde_json::json!(delta), serde_json::json!(org)],
            )
            .await?;
        Ok(())
    }

    /// Re-measure from the filesystem and overwrite the stored total.
    ///
    /// For when the two have diverged: files removed out of band, a restore, a
    /// bug here. Exposed so an operator can fix a wrong quota without editing
    /// the database.
    pub async fn recount(
        &self,
        org: &str,
        blobs_root: &std::path::Path,
        owned_buckets: &[String],
    ) -> anyhow::Result<u64> {
        self.ensure_table().await?;
        let measured = walk_bytes(blobs_root, owned_buckets);
        self.sqlite
            .execute(
                "INSERT OR REPLACE INTO sys_blob_usage (org_id, bytes) VALUES (?, ?)".to_string(),
                vec![serde_json::json!(org), serde_json::json!(measured as i64)],
            )
            .await?;
        Ok(measured)
    }
}

/// Total bytes under the named buckets.
///
/// `owned_buckets` comes from the ownership registry, which is keyed by name
/// across every scoped resource — so it also contains vector collections and doc
/// namespaces. Those simply have no directory under `blobs/` and contribute
/// nothing, which is why the list is used as-is rather than filtered: a filter
/// would need to know every resource kind and would go stale when one is added.
fn walk_bytes(blobs_root: &std::path::Path, owned_buckets: &[String]) -> u64 {
    owned_buckets
        .iter()
        .map(|bucket| directory_bytes(&blobs_root.join(bucket)))
        .sum()
}

fn directory_bytes(directory: &std::path::Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(directory) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += directory_bytes(&path);
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

/// Refuse a blob write that would take the organization past its byte limit.
///
/// **507, not 429.** The caller is out of room, not being rate limited: the
/// identical request will never succeed, and a 429 invites exactly the retry
/// loop that cannot help.
///
/// Returns `Ok(())` with no limit configured, with no organization (a
/// platform-level caller), or with no accounts layer — in each case there is
/// nothing to attribute the bytes to, and inventing an attribution is how one
/// org ends up paying for another's storage.
///
/// Takes the pieces it needs rather than the whole `AppState`: the guard depends
/// on SQLite, the ownership registry and where the blobs live, and saying so
/// makes it callable from a test without standing up a router.
pub async fn guard_blob_write(
    store: BlobQuotaStore<'_>,
    ctx: &TenantContext,
    incoming: u64,
    replacing: u64,
) -> Result<(), ApiError> {
    let quotas = Quotas::from_value(&ctx.quotas);
    let Some(limit) = quotas.max_blob_bytes else {
        return Ok(());
    };
    let Some(org) = ctx.tenant_id.as_deref() else {
        return Ok(());
    };
    let (Some(sqlite), Some(accounts)) = (store.sqlite, store.accounts) else {
        return Ok(());
    };

    let owned = accounts
        .names_owned_by(org)
        .await
        .map_err(|e| internal(format!("ownership lookup failed: {e}")))?;
    let root = store.blobs_root;
    let usage = BlobUsage::new(sqlite.clone());
    let current = usage
        .bytes_for(org, root, &owned)
        .await
        .map_err(|e| internal(format!("usage lookup failed: {e}")))?;

    // An overwrite only costs the difference. Charging the full size would make
    // updating an object in place impossible at the limit, which is not what a
    // storage limit means.
    let after = current.saturating_sub(replacing).saturating_add(incoming);
    if after > limit {
        return Err(to_api_error(Exceeded {
            resource: Resource::BlobBytes,
            limit,
            current,
            requested: incoming,
        }));
    }
    Ok(())
}

/// Record the byte change after a successful write or delete.
///
/// Best effort, and deliberately so: the object is already committed, and
/// failing the request now would tell the caller their write did not happen when
/// it did. A logged warning plus `recount` is the honest trade.
pub async fn record_blob_delta(
    sqlite: Option<&crate::sqlite::SqliteService>,
    ctx: &TenantContext,
    delta: i64,
) {
    if delta == 0 {
        return;
    }
    let Some(org) = ctx.tenant_id.as_deref() else {
        return;
    };
    let Some(sqlite) = sqlite else {
        return;
    };
    if let Err(e) = BlobUsage::new(sqlite.clone()).adjust(org, delta).await {
        tracing::warn!(
            org = %org,
            "blob usage accounting failed; the quota for this org may drift until a recount: {e}"
        );
    }
}

/// What the blob quota guard needs, and nothing else.
pub struct BlobQuotaStore<'a> {
    pub sqlite: Option<&'a crate::sqlite::SqliteService>,
    pub accounts: Option<&'a crate::api::accounts::AccountsService>,
    pub blobs_root: &'a std::path::Path,
}

fn internal(message: String) -> ApiError {
    ApiError::new(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "internal",
        message,
    )
}

// ── Queue messages and vectors per organization ──────────────────────────────
//
// The last two of W5.2. Both are cheaper than blob bytes, for opposite reasons:
//
// * **Queues** are already isolated by directory (`queues/t_{org}/…`), so an
//   org's messages are a walk of one subtree it exclusively owns. No ownership
//   registry and no stored counter: the answer is right there and small.
// * **Vectors** are counted by the vector store already — `live_count` per
//   collection — so the only question is which collections belong to the org,
//   and the ownership registry answers that.
//
// Neither needs the stored-total machinery blob bytes needed, and adding it
// would be cost with no benefit: a counter that can drift, guarding a number
// that was never expensive to compute.

/// Refuse an enqueue that would take the organization past its message limit.
///
/// Returns `Ok(())` with no limit and for a platform caller, who writes to the
/// shared top-level namespace rather than to any org's subtree.
pub fn guard_queue_write(
    queues_root: &std::path::Path,
    ctx: &TenantContext,
    incoming: u64,
) -> Result<(), ApiError> {
    let quotas = Quotas::from_value(&ctx.quotas);
    let Some(limit) = quotas.max_queue_messages else {
        return Ok(());
    };
    let Some(org) = ctx.tenant_id.as_deref() else {
        return Ok(());
    };

    let current = count_files(&queues_root.join(format!("t_{org}")));
    if current.saturating_add(incoming) > limit {
        return Err(to_api_error(Exceeded {
            resource: Resource::QueueMessages,
            limit,
            current,
            requested: incoming,
        }));
    }
    Ok(())
}

/// Messages held under a directory tree, one file each.
fn count_files(directory: &std::path::Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(directory) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += count_files(&path);
        } else {
            total += 1;
        }
    }
    total
}

/// Refuse a vector upsert that would take the organization past its limit.
///
/// Counts `live_count`, not total records: a tombstoned vector is not storage
/// the caller can read back, and charging for it would make a collection
/// permanently full after enough deletes.
pub async fn guard_vector_write(
    engine: &Engine,
    accounts: Option<&crate::api::accounts::AccountsService>,
    ctx: &TenantContext,
    incoming: u64,
) -> Result<(), ApiError> {
    let quotas = Quotas::from_value(&ctx.quotas);
    let Some(limit) = quotas.max_vectors else {
        return Ok(());
    };
    let Some(org) = ctx.tenant_id.as_deref() else {
        return Ok(());
    };
    let Some(accounts) = accounts else {
        return Ok(());
    };

    let owned = accounts.names_owned_by(org).await.map_err(|e| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            format!("ownership lookup failed: {e}"),
        )
    })?;
    // The registry is keyed by name across every scoped resource, so this list
    // also holds blob buckets and doc namespaces. Those are not collections and
    // contribute nothing — used as-is rather than filtered, because a filter
    // would need to know every resource kind and would go stale when one is
    // added.
    let current: u64 = owned
        .iter()
        .filter_map(|name| engine.vector_collection_info(name))
        .map(|info| info.live_count as u64)
        .sum();

    if current.saturating_add(incoming) > limit {
        return Err(to_api_error(Exceeded {
            resource: Resource::Vectors,
            limit,
            current,
            requested: incoming,
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn an_empty_record_is_unlimited() {
        // Every api key today carries `{}`; enforcing anything on those would
        // break every existing deployment on upgrade.
        let quotas = Quotas::from_value(&json!({}));
        assert!(quotas.is_unlimited());
        assert!(quotas.check(Resource::Keys, u64::MAX - 1, 1).is_ok());
    }

    #[test]
    fn the_legacy_unlimited_record_parses_as_unlimited() {
        // The shape the auth layer has been emitting: string values, not
        // numbers. It must not be read as a limit of zero.
        let quotas = Quotas::from_value(&json!({"storage_bytes": "unlimited", "qps": "unlimited"}));
        assert!(quotas.is_unlimited());
    }

    #[test]
    fn a_malformed_record_is_unlimited_not_a_lockout() {
        // A config typo must not become an outage: refusing every write until
        // someone fixes a JSON blob is worse than not enforcing the quota.
        let quotas = Quotas::from_value(&json!({"max_keys": "not a number"}));
        assert!(quotas.is_unlimited());
        assert!(quotas.check(Resource::Keys, 1_000_000, 1).is_ok());
    }

    #[test]
    fn null_is_unlimited() {
        assert!(Quotas::from_value(&serde_json::Value::Null).is_unlimited());
    }

    #[test]
    fn a_limit_admits_the_last_allowed_write_and_refuses_the_next() {
        // A limit of 100 keys means the hundredth lands and the hundred-and-
        // first does not, which is what makes the configured number the number
        // an operator can reason about.
        let quotas = Quotas::from_value(&json!({"max_keys": 100}));
        assert!(quotas.check(Resource::Keys, 99, 1).is_ok(), "the 100th");
        assert!(quotas.check(Resource::Keys, 100, 1).is_err(), "the 101st");
    }

    #[test]
    fn a_batch_that_would_cross_the_line_is_refused_whole() {
        // Partially applying a batch would leave the caller unable to tell what
        // landed.
        let quotas = Quotas::from_value(&json!({"max_keys": 100}));
        assert!(quotas.check(Resource::Keys, 95, 5).is_ok());
        assert!(quotas.check(Resource::Keys, 95, 6).is_err());
    }

    #[test]
    fn the_error_names_every_number_the_caller_needs() {
        let quotas = Quotas::from_value(&json!({"max_blob_bytes": 1000}));
        let err = quotas.check(Resource::BlobBytes, 900, 200).unwrap_err();
        assert_eq!(err.limit, 1000);
        assert_eq!(err.current, 900);
        assert_eq!(err.requested, 200);

        let message = err.to_string();
        // "quota exceeded" alone leaves the caller unable to tell whether to
        // delete one thing or a thousand.
        assert!(message.contains("900"), "{message}");
        assert!(message.contains("200"), "{message}");
        assert!(message.contains("1000"), "{message}");
        assert!(message.contains("blob_bytes"), "{message}");
    }

    #[test]
    fn overflow_cannot_be_used_to_slip_under_a_limit() {
        let quotas = Quotas::from_value(&json!({"max_keys": 10}));
        assert!(quotas.check(Resource::Keys, u64::MAX, 1).is_err());
        assert!(quotas.check(Resource::Keys, 5, u64::MAX).is_err());
    }

    #[test]
    fn a_zero_limit_refuses_everything_including_the_first_write() {
        // Zero is a real setting — a suspended organization — and must not be
        // confused with unlimited.
        let quotas = Quotas::from_value(&json!({"max_keys": 0}));
        assert!(!quotas.is_unlimited());
        assert!(quotas.check(Resource::Keys, 0, 1).is_err());
    }

    #[test]
    fn limits_are_independent_per_resource() {
        // Hitting the key limit must not refuse a blob write, or an operator
        // debugging one problem gets misled about another.
        let quotas = Quotas::from_value(&json!({"max_keys": 1}));
        assert!(quotas.check(Resource::Keys, 1, 1).is_err());
        assert!(quotas.check(Resource::BlobBytes, u64::MAX - 1, 1).is_ok());
        assert!(quotas.check(Resource::Vectors, 999_999, 1).is_ok());
    }

    #[test]
    fn partial_records_only_limit_what_they_mention() {
        let quotas = Quotas::from_value(&json!({"max_vectors": 5}));
        assert!(!quotas.is_unlimited());
        assert!(quotas.check(Resource::Vectors, 5, 1).is_err());
        assert!(quotas.check(Resource::Keys, 10_000, 1).is_ok());
    }

    #[test]
    fn quotas_round_trip_through_json() {
        // They are persisted on the api key record, so the serialized form has
        // to survive.
        let original = Quotas {
            max_keys: Some(10),
            max_vectors: None,
            max_blob_bytes: Some(1024),
            max_queue_messages: Some(0),
        };
        let text = serde_json::to_string(&original).unwrap();
        assert_eq!(serde_json::from_str::<Quotas>(&text).unwrap(), original);
    }

    // ── enforcement against real state ───────────────────────────────────────

    fn engine() -> (crate::engine::Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::config::Config {
            data_dir: Some(dir.path().to_str().unwrap().to_string()),
            ..crate::config::Config::default()
        };
        (
            crate::engine::Engine::new(config, tokio_util::sync::CancellationToken::new()).unwrap(),
            dir,
        )
    }

    fn ctx(tenant: &str, quotas: serde_json::Value) -> TenantContext {
        TenantContext {
            tenant_id: Some(tenant.to_string()),
            user_id: None,
            role: "member".to_string(),
            platform_admin: false,
            permissions: json!({}),
            quotas,
        }
    }

    #[test]
    fn one_org_at_its_limit_does_not_degrade_another() {
        // The acceptance criterion from the SPEC, as a test. Usage is measured
        // over the caller's own key prefix, so a full neighbour is invisible.
        let (engine, _dir) = engine();
        for i in 0..3 {
            engine
                .put_state(format!("acme:k{i}"), json!(i), None, None)
                .unwrap();
        }
        engine
            .put_state("globex:only".to_string(), json!(1), None, None)
            .unwrap();

        let limit = json!({ "max_keys": 3 });
        assert!(
            guard_key_write(&engine, &ctx("acme", limit.clone()), 1).is_err(),
            "acme holds 3 of 3 and must be refused"
        );
        assert!(
            guard_key_write(&engine, &ctx("globex", limit), 1).is_ok(),
            "globex holds 1 of 3 and must be unaffected by acme being full"
        );
    }

    #[test]
    fn an_exhausted_quota_answers_507_not_429() {
        // The caller is out of room, not being rate limited: retrying the
        // identical request will never succeed, and a 429 invites exactly that
        // retry loop.
        let (engine, _dir) = engine();
        engine
            .put_state("t:a".to_string(), json!(1), None, None)
            .unwrap();

        let err = guard_key_write(&engine, &ctx("t", json!({"max_keys": 1})), 1).unwrap_err();
        let response = axum::response::IntoResponse::into_response(err);
        assert_eq!(response.status(), StatusCode::INSUFFICIENT_STORAGE);
    }

    #[test]
    fn an_unlimited_context_is_never_refused_over_the_same_state() {
        // Keeps this from behaving like a global switch: the limit belongs to
        // the caller, not to the instance.
        let (engine, _dir) = engine();
        for i in 0..50 {
            engine
                .put_state(format!("t:k{i}"), json!(i), None, None)
                .unwrap();
        }
        assert!(guard_key_write(&engine, &ctx("t", json!({})), 1).is_ok());
        assert!(guard_key_write(&engine, &ctx("t", json!({"max_keys": 10})), 1).is_err());
    }

    #[test]
    fn a_batch_is_charged_as_a_whole() {
        // Half-applying a batch would leave the caller unable to tell what
        // landed, so the check is for the whole request.
        let (engine, _dir) = engine();
        engine
            .put_state("t:a".to_string(), json!(1), None, None)
            .unwrap();

        let context = ctx("t", json!({"max_keys": 5}));
        assert!(guard_key_write(&engine, &context, 4).is_ok(), "1 + 4 = 5");
        assert!(guard_key_write(&engine, &context, 5).is_err(), "1 + 5 > 5");
    }

    #[test]
    fn a_platform_connection_counts_the_whole_keyspace() {
        // No tenant prefix means the caller sees everything, so its usage is
        // everything — a superuser with a limit is limited on the real total.
        let (engine, _dir) = engine();
        for i in 0..4 {
            engine
                .put_state(format!("anything{i}"), json!(i), None, None)
                .unwrap();
        }
        let platform = TenantContext {
            tenant_id: None,
            user_id: None,
            role: "admin".to_string(),
            platform_admin: true,
            permissions: json!({}),
            quotas: json!({"max_keys": 4}),
        };
        assert!(guard_key_write(&engine, &platform, 1).is_err());
    }
}
