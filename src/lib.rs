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
pub mod api;
/// Consistent on-disk backups (SQLite + snapshot + WAL) and restore.
pub mod backup;
/// Configuration loading and defaults (`luma.toml` + environment overrides).
pub mod config;
/// Encryption-at-rest (AEAD) and password hashing (Argon2id).
pub mod crypto;
/// Document store: chunking, storage, and retrieval of ingested documents.
pub mod docstore;

pub mod durability;
/// Core engine: subsystem coordination, event sourcing, WAL, and persistence.
pub mod engine;
/// NS-Mem: the agent memory layer (episodic, semantic, procedural, working).
pub mod memory;

pub mod router;
/// Search primitives shared across the vector and hybrid query paths.
pub mod search;
/// Embedded SQLite access via the thread-safe async actor pattern.
pub mod sqlite;
/// Vector store: segmented storage with HNSW, IVF-FLAT-Q8, and DiskANN indexing.
pub mod vector;
