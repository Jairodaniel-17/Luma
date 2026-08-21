//! Durable, disk-backed message queue (Cloudflare Queues-equivalent).
//!
//! Messages are stored as JSON files under `{data_dir}/queues/{queue}/`, named
//! with a lexicographically-sortable id (zero-padded enqueue millis + a short
//! uuid suffix) so a directory listing sorts oldest-first. Delivery is
//! at-least-once: `receive` makes messages invisible for `visibility_secs` and
//! increments `attempts`; `delete` acks them. A per-queue mutex serializes
//! `receive` so two callers never hand out the same message.
//!
//! These routes live in the authenticated router chain, so every request
//! requires a valid Bearer api_key.

use crate::api::errors::ApiError;
use crate::api::{AppState, TenantContext};
use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

/// Maximum messages returnable from a single `receive`.
const MAX_RECEIVE: usize = 100;
/// Default `receive` batch size.
const DEFAULT_RECEIVE: usize = 1;
/// Default visibility timeout (seconds).
const DEFAULT_VISIBILITY_SECS: u64 = 30;
/// Cap on enqueue delay / visibility timeout (24h) to reject absurd inputs.
const MAX_DELAY_SECS: u64 = 24 * 60 * 60;

/// Per-queue receive locks, keyed by queue name. Guards the read-then-rewrite
/// in `receive` so concurrent receivers don't claim the same message.
static RECEIVE_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();

async fn receive_lock(queue: &str) -> Arc<Mutex<()>> {
    let mut map = RECEIVE_LOCKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .await;
    map.entry(queue.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

#[derive(Debug, Serialize, Deserialize)]
struct StoredMessage {
    id: String,
    body: Value,
    enqueued_at: u64,
    visible_at: u64,
    attempts: u64,
}

#[derive(Debug, Deserialize)]
pub struct EnqueueRequest {
    body: Value,
    #[serde(default)]
    delay_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct EnqueueResponse {
    id: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct ReceiveRequest {
    #[serde(default)]
    max: Option<usize>,
    #[serde(default)]
    visibility_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ReceivedMessage {
    id: String,
    body: Value,
    attempts: u64,
}

#[derive(Debug, Serialize)]
pub struct ReceiveResponse {
    messages: Vec<ReceivedMessage>,
}

#[derive(Debug, Serialize)]
pub struct AckResponse {
    ok: bool,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    queue: String,
    depth: usize,
    visible: usize,
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn queues_root(state: &AppState) -> PathBuf {
    let data_dir = state.config.data_dir.as_deref().unwrap_or("data");
    PathBuf::from(data_dir).join("queues")
}

/// Validate a single path component (queue name or message id).
/// Allowed charset is `[A-Za-z0-9._-]`; `.`, `..`, empty and any other
/// character (including `/`, `\`, NUL, control chars) are rejected.
fn validate_segment(seg: &str) -> Result<(), ApiError> {
    if seg.is_empty() || seg == "." || seg == ".." {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "invalid path segment",
        ));
    }
    if !seg
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "path segment contains illegal characters",
        ));
    }
    Ok(())
}

fn validate_queue(queue: &str) -> Result<(), ApiError> {
    if queue.len() > 128 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "queue name too long",
        ));
    }
    validate_segment(queue)
}

fn validate_id(id: &str) -> Result<(), ApiError> {
    if id.len() > 128 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "message id too long",
        ));
    }
    validate_segment(id)
}

/// Defense-in-depth: confirm a built path is lexically contained within `root`.
fn ensure_within_root(root: &StdPath, candidate: &StdPath) -> Result<(), ApiError> {
    crate::api::pathsafe::ensure_within_root(root, candidate).map_err(|why| match why {
        crate::api::pathsafe::PathRejection::Traversal => ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "path traversal detected",
        ),
        crate::api::pathsafe::PathRejection::Escapes => ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "path escapes queue root",
        ),
    })
}

/// Per-tenant isolation directory (`t_{tenant}`) under the queues root, or
/// `None` for a platform admin (who uses the shared top-level namespace).
/// The tenant id is validated as a path segment to preclude traversal.
fn tenant_queue_dir(ctx: &TenantContext) -> Result<Option<String>, ApiError> {
    match &ctx.tenant_id {
        None => Ok(None),
        Some(t) => {
            validate_segment(t)?;
            Ok(Some(format!("t_{t}")))
        }
    }
}

/// Lock key that keeps two tenants' identically-named queues from sharing a
/// receive lock.
fn scoped_lock_key(ctx: &TenantContext, queue: &str) -> String {
    match &ctx.tenant_id {
        Some(t) => format!("t_{t}/{queue}"),
        None => format!("_global/{queue}"),
    }
}

fn resolve_queue_dir(
    state: &AppState,
    ctx: &TenantContext,
    queue: &str,
) -> Result<PathBuf, ApiError> {
    validate_queue(queue)?;
    let root = queues_root(state);
    let mut path = root.clone();
    if let Some(td) = tenant_queue_dir(ctx)? {
        path = path.join(td);
    }
    let path = path.join(queue);
    ensure_within_root(&root, &path)?;
    Ok(path)
}

fn resolve_message_path(
    state: &AppState,
    ctx: &TenantContext,
    queue: &str,
    id: &str,
) -> Result<PathBuf, ApiError> {
    validate_queue(queue)?;
    validate_id(id)?;
    let root = queues_root(state);
    let mut path = root.clone();
    if let Some(td) = tenant_queue_dir(ctx)? {
        path = path.join(td);
    }
    let path = path.join(queue).join(format!("{id}.json"));
    ensure_within_root(&root, &path)?;
    Ok(path)
}

fn io_error(msg: &'static str) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "io_error", msg)
}

fn check_delay(secs: u64) -> Result<(), ApiError> {
    if secs > MAX_DELAY_SECS {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "delay/visibility exceeds maximum",
        ));
    }
    Ok(())
}

/// Commit one message to disk durably.
///
/// An enqueue that returns OK must survive a crash, so this is a full durable
/// write (fsync of the file, then of the directory after the rename) and not
/// just an atomic rename: the rename was already atomic, but its directory
/// entry could still be lost with the page cache, silently dropping a message
/// the producer believes was accepted.
async fn write_message_atomic(dir: &StdPath, msg: &StoredMessage) -> Result<(), ApiError> {
    let bytes = serde_json::to_vec(msg).map_err(|_| io_error("failed to serialize message"))?;
    let final_path = dir.join(format!("{}.json", msg.id));
    if let Err(e) = crate::durability::write_atomic(&final_path, &bytes).await {
        tracing::error!("queue message commit failed: {}", e);
        return Err(io_error("failed to commit message"));
    }
    Ok(())
}

/// POST /v1/queue/:queue — enqueue a message.
pub async fn enqueue(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(queue): Path<String>,
    axum::Json(req): axum::Json<EnqueueRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let dir = resolve_queue_dir(&state, &ctx, &queue)?;
    let delay = req.delay_secs.unwrap_or(0);
    check_delay(delay)?;

    let now = now_millis();
    // Lexicographically-sortable id: zero-padded enqueue millis + short uuid
    // suffix to break ties within the same millisecond.
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    let id = format!("{now:013}-{}", &suffix[..12]);

    let msg = StoredMessage {
        id: id.clone(),
        body: req.body,
        enqueued_at: now,
        visible_at: now + delay * 1000,
        attempts: 0,
    };

    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|_| io_error("failed to create queue directory"))?;
    write_message_atomic(&dir, &msg).await?;

    Ok(axum::Json(EnqueueResponse { id }))
}

/// Read and parse one stored message, skipping temp files and unreadable entries.
async fn load_message(path: &StdPath) -> Option<StoredMessage> {
    let bytes = tokio::fs::read(path).await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Collect message file paths in a queue dir, sorted oldest-first by file name
/// (file names are lexicographically-sortable ids).
async fn list_message_files(dir: &StdPath) -> Result<Vec<PathBuf>, ApiError> {
    let mut files: Vec<PathBuf> = Vec::new();
    if tokio::fs::metadata(dir).await.is_err() {
        return Ok(files);
    }
    let mut rd = tokio::fs::read_dir(dir)
        .await
        .map_err(|_| io_error("failed to list queue"))?;
    while let Some(entry) = rd
        .next_entry()
        .await
        .map_err(|_| io_error("failed to list queue"))?
    {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name.starts_with(".tmp-") || !name.ends_with(".json") {
            continue;
        }
        files.push(path);
    }
    files.sort();
    Ok(files)
}

/// POST /v1/queue/:queue/receive — claim up to `max` visible messages.
pub async fn receive(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(queue): Path<String>,
    body: Option<axum::Json<ReceiveRequest>>,
) -> Result<impl IntoResponse, ApiError> {
    let dir = resolve_queue_dir(&state, &ctx, &queue)?;
    let req = body.map(|b| b.0).unwrap_or_default();

    let max = req.max.unwrap_or(DEFAULT_RECEIVE).clamp(1, MAX_RECEIVE);
    let visibility = req.visibility_secs.unwrap_or(DEFAULT_VISIBILITY_SECS);
    check_delay(visibility)?;

    // Serialize receives per (tenant, queue) so two callers can't claim the
    // same message, without colliding across tenants sharing a queue name.
    let lock = receive_lock(&scoped_lock_key(&ctx, &queue)).await;
    let _guard = lock.lock().await;

    let now = now_millis();
    let files = list_message_files(&dir).await?;

    let mut out: Vec<ReceivedMessage> = Vec::new();
    for path in files {
        if out.len() >= max {
            break;
        }
        let mut msg = match load_message(&path).await {
            Some(m) => m,
            None => continue,
        };
        if msg.visible_at > now {
            continue;
        }
        msg.visible_at = now + visibility * 1000;
        msg.attempts += 1;
        // Rewrite atomically to reflect the in-flight state.
        write_message_atomic(&dir, &msg).await?;
        out.push(ReceivedMessage {
            id: msg.id,
            body: msg.body,
            attempts: msg.attempts,
        });
    }

    Ok(axum::Json(ReceiveResponse { messages: out }))
}

/// DELETE /v1/queue/:queue/:id — ack/delete a message (idempotent).
pub async fn ack(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path((queue, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let path = resolve_message_path(&state, &ctx, &queue, &id)?;
    match tokio::fs::remove_file(&path).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(io_error("failed to delete message")),
    }
    Ok(axum::Json(AckResponse { ok: true }))
}

/// GET /v1/queue/:queue — queue stats.
pub async fn stats(
    State(state): State<AppState>,
    Extension(ctx): Extension<TenantContext>,
    Path(queue): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let dir = resolve_queue_dir(&state, &ctx, &queue)?;
    let now = now_millis();
    let files = list_message_files(&dir).await?;

    let depth = files.len();
    let mut visible = 0usize;
    for path in &files {
        if let Some(msg) = load_message(path).await {
            if msg.visible_at <= now {
                visible += 1;
            }
        }
    }

    Ok(axum::Json(StatsResponse {
        queue,
        depth,
        visible,
    }))
}
