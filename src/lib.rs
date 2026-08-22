//! # Luma (`rust-kiss-vdb`)
//!
//! A high-performance convergent data engine for AI applications. Luma unifies
//! vector search, key-value state, relational SQL, and pub/sub events behind a
//! single Rust binary, exposed over three progressively higher-level API tiers:
//!
//! - **Level 1** — primitive operations on individual subsystems
//!   (`/v1/vector`, `/v1/state`, `/v1/doc`, `/v1/sql`, `/v1/events`).
//! - **Level 2** — the [`engine::hub`] `LumaDatabase` orchestrator
//!   (`/v1/db`): auto-chunking, embedding generation, and hybrid SQL+vector search.
//! - **Level 3** — the NS-Mem agent memory layer ([`memory`], `/v1/memory`):
//!   episodic, semantic, procedural, and working memory over a typed graph.
//!
//! Each top-level module corresponds to one subsystem of the engine.

/// HTTP layer: Axum router, authentication, and per-concern route handlers.
// ── `unsafe` is confined to `vector` ─────────────────────────────────────────
//
// Every module below except `vector` carries `#[forbid(unsafe_code)]`. That is
// not decoration: it turns "no module outside the vector engine uses unsafe"
// from something that happens to be true today into something the compiler
// refuses to let change.
//
// `vector` is the exception because memory-mapped segment files and the SIMD
// dot products genuinely need it — 16 sites across four files, inventoried with
// their justifications in `docs/SECURITY.md`. Marking the exception explicitly
// is the point: an `unsafe` block appearing anywhere else is a compile error,
// not a review comment somebody might miss.
//
// `tests/unsafe_inventory.rs` checks that this list has not silently lost a
// module, because a new `pub mod` added without the attribute would compile
// perfectly well.
#[forbid(unsafe_code)]
pub mod api;
/// Consistent on-disk backups (SQLite + snapshot + WAL) and restore.
#[forbid(unsafe_code)]
pub mod backup;

#[forbid(unsafe_code)]
pub mod backup_remote;
/// Configuration loading and defaults (`luma.toml` + environment overrides).
#[forbid(unsafe_code)]
pub mod config;
/// Encryption-at-rest (AEAD) and password hashing (Argon2id).
#[forbid(unsafe_code)]
pub mod crypto;
/// Document store: chunking, storage, and retrieval of ingested documents.
#[forbid(unsafe_code)]
pub mod docstore;

#[forbid(unsafe_code)]
pub mod durability;
/// Core engine: subsystem coordination, event sourcing, WAL, and persistence.
#[forbid(unsafe_code)]
pub mod engine;
/// NS-Mem: the agent memory layer (episodic, semantic, procedural, working).
#[forbid(unsafe_code)]
pub mod memory;

#[forbid(unsafe_code)]
pub mod resp;

#[forbid(unsafe_code)]
pub mod replica;

#[forbid(unsafe_code)]
pub mod router;
/// Search primitives shared across the vector and hybrid query paths.
#[forbid(unsafe_code)]
pub mod search;
/// Embedded SQLite access via the thread-safe async actor pattern.
#[forbid(unsafe_code)]
pub mod sqlite;
#[forbid(unsafe_code)]
pub mod telemetry;
/// Vector store: segmented storage with HNSW, IVF-FLAT-Q8, and DiskANN indexing.
pub mod vector;

#[forbid(unsafe_code)]
pub mod wal_ship;

/// Install the rustls cryptography provider, once per process.
///
/// rustls 0.23 refuses to pick a provider when more than one is compiled in,
/// and the refusal is a **panic** at the first handshake rather than an error at
/// startup. Both `ring` and `aws-lc-rs` reach this build through different
/// dependency paths, so leaving the choice implicit means TLS works or panics
/// depending on which crates happen to be in the graph — including which *test*
/// crates are.
///
/// `ring` is the pick: it needs no C toolchain, which is what makes the Windows
/// build reproducible.
///
/// Called from every path that builds a TLS config. A second call is a no-op:
/// `install_default` reports that someone else won the race, and the provider is
/// what matters, not who installed it.
pub fn install_crypto_provider() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
