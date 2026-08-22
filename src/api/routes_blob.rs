//! S3-like object/blob storage backed by the local filesystem.
//!
//! Binary objects are stored on disk under `{data_dir}/blobs/{bucket}/{key}` so
//! applications can persist file attachments without embedding base64 strings in
//! JSON documents.
//!
//! These routes live in the same authenticated router chain as the document
//! routes, so every request requires a valid Bearer api_key.

use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use serde::Serialize;
use std::path::{Component, Path as StdPath, PathBuf};

/// Fallback maximum object size when `config.max_body_bytes` is unset/zero (100 MB).
const DEFAULT_MAX_BLOB_BYTES: usize = 100 * 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct BlobPutResponse {
    pub bucket: String,
    pub key: String,
    pub size: u64,
    pub etag: String,
}

#[derive(Debug, Serialize)]
pub struct BlobDeleteResponse {
    pub ok: bool,
}

#[derive(Debug, Serialize)]
pub struct BlobListResponse {
    pub keys: Vec<String>,
    pub count: usize,
}

/// Resolve `{data_dir}/blobs`, defaulting to `data/blobs` when `data_dir` is None.
fn blobs_root(state: &AppState) -> PathBuf {
    let data_dir = state.config.data_dir.as_deref().unwrap_or("data");
    PathBuf::from(data_dir).join("blobs")
}

/// Validate a single path component (bucket name or one key segment).
///
/// Allowed charset is `[A-Za-z0-9._-]`. Empty components, `.`, `..`, and any
/// other character (including `/`, `\`, NUL, control chars) are rejected.
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

/// Validate the bucket name (a single safe segment).
fn validate_bucket(bucket: &str) -> Result<(), ApiError> {
    if bucket.len() > 128 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "bucket too long",
        ));
    }
    validate_segment(bucket)
}

/// Validate a key, which may contain `/` to denote nested directories.
///
/// Each `/`-separated segment must independently pass [`validate_segment`], so
/// `..`, leading/trailing/empty segments, and absolute paths are all rejected.
fn validate_key(key: &str) -> Result<(), ApiError> {
    if key.len() > 1024 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "key too long",
        ));
    }
    if key.starts_with('/') || key.contains('\\') {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_argument",
            "invalid key",
        ));
    }
    for seg in key.split('/') {
        validate_segment(seg)?;
    }
    Ok(())
}

/// Defense-in-depth: confirm a built path is lexically contained within `root`.
///
/// Because the on-disk path may not exist yet (PUT), we cannot rely on
/// `canonicalize`; instead we reject any path containing `..`/root-dir/prefix
/// components and verify it still starts with `root`.
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
            "path escapes blob root",
        ),
    })
}

/// Resolve and validate the full on-disk path for a bucket+key.
fn resolve_blob_path(state: &AppState, bucket: &str, key: &str) -> Result<PathBuf, ApiError> {
    validate_bucket(bucket)?;
    validate_key(key)?;
    let root = blobs_root(state);
    let mut path = root.join(bucket);
    for seg in key.split('/') {
        path.push(seg);
    }
    ensure_within_root(&root, &path)?;
    Ok(path)
}

/// Resolve and validate the on-disk directory for a bucket.
fn resolve_bucket_dir(state: &AppState, bucket: &str) -> Result<PathBuf, ApiError> {
    validate_bucket(bucket)?;
    let root = blobs_root(state);
    let path = root.join(bucket);
    ensure_within_root(&root, &path)?;
    Ok(path)
}

fn max_blob_bytes(state: &AppState) -> usize {
    if state.config.max_body_bytes > 0 {
        state.config.max_body_bytes
    } else {
        DEFAULT_MAX_BLOB_BYTES
    }
}

/// The pieces the blob quota guard needs, gathered from the router state.
fn blob_quota_store(state: &AppState) -> crate::api::quotas::BlobQuotaStore<'_> {
    crate::api::quotas::BlobQuotaStore {
        sqlite: state.sqlite.as_ref(),
        accounts: state.accounts.as_deref(),
        blobs_root: BLOBS_ROOT.get_or_init(|| blobs_root(state)),
    }
}

/// Resolved once: the path is derived from config, which does not change while
/// the process runs, and the guard needs a borrow that outlives the call.
static BLOBS_ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

fn io_error(msg: &'static str) -> ApiError {
    ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "io_error", msg)
}

/// PUT /v1/blob/:bucket/:key — store raw bytes atomically.
pub async fn put(
    State(state): State<AppState>,
    axum::extract::Extension(ctx): axum::extract::Extension<crate::api::TenantContext>,
    Path((bucket, key)): Path<(String, String)>,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    let path = resolve_blob_path(&state, &bucket, &key)?;

    let limit = max_blob_bytes(&state);
    if body.len() > limit {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "blob too large",
        ));
    }

    // What this write replaces, so an overwrite is charged the difference rather
    // than its full size. Charging the full size would make updating an object
    // impossible once the org is at its limit, which is not what a storage limit
    // means.
    let replacing = tokio::fs::metadata(&path)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    crate::api::quotas::guard_blob_write(
        blob_quota_store(&state),
        &ctx,
        body.len() as u64,
        replacing,
    )
    .await?;

    // Durable: temp file -> fsync the file -> rename -> fsync the directory.
    // The rename alone was atomic but not durable: a crash right after a
    // confirmed PUT could lose the object because the directory entry was still
    // only in the page cache.
    if let Err(e) = crate::durability::write_atomic(&path, &body).await {
        tracing::error!("blob commit failed: {}", e);
        return Err(io_error("failed to commit blob"));
    }

    let size = body.len() as u64;
    // After the commit: the bytes are on disk, so this is what the org now holds.
    // Accounted after rather than before because a failed write must not leave a
    // charge behind.
    crate::api::quotas::record_blob_delta(
        state.sqlite.as_ref(),
        &ctx,
        size as i64 - replacing as i64,
    )
    .await;
    let etag = format!("{:08x}-{}", crc32fast::hash(&body), size);

    Ok(axum::Json(BlobPutResponse {
        bucket,
        key,
        size,
        etag,
    }))
}

/// GET /v1/blob/:bucket/:key — return the raw bytes.
pub async fn get(
    State(state): State<AppState>,
    Path((bucket, key)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let path = resolve_blob_path(&state, &bucket, &key)?;

    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "blob not found",
            ));
        }
        Err(_) => return Err(io_error("failed to read blob")),
    };

    let len = bytes.len();
    let headers = [
        (header::CONTENT_TYPE, "application/octet-stream".to_string()),
        (header::CONTENT_LENGTH, len.to_string()),
    ];
    Ok((StatusCode::OK, headers, bytes))
}

/// DELETE /v1/blob/:bucket/:key — idempotent delete.
pub async fn delete(
    State(state): State<AppState>,
    axum::extract::Extension(ctx): axum::extract::Extension<crate::api::TenantContext>,
    Path((bucket, key)): Path<(String, String)>,
) -> Result<impl IntoResponse, ApiError> {
    let path = resolve_blob_path(&state, &bucket, &key)?;

    // Read the size before removing it: afterwards there is nothing to measure,
    // and a delete that does not give the bytes back is a quota that only ever
    // goes up.
    let freed = tokio::fs::metadata(&path)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);

    match tokio::fs::remove_file(&path).await {
        Ok(()) => {
            crate::api::quotas::record_blob_delta(state.sqlite.as_ref(), &ctx, -(freed as i64))
                .await;
        }
        // Idempotent: deleting what is not there frees nothing.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err(io_error("failed to delete blob")),
    }

    Ok(axum::Json(BlobDeleteResponse { ok: true }))
}

/// GET /v1/blob/:bucket — list keys in the bucket (recursing nested dirs).
pub async fn list(
    State(state): State<AppState>,
    Path(bucket): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let dir = resolve_bucket_dir(&state, &bucket)?;

    let mut keys: Vec<String> = Vec::new();

    // Bucket dir may not exist yet -> empty list.
    if tokio::fs::metadata(&dir).await.is_ok() {
        collect_keys(&dir, &dir, &mut keys).await?;
    }

    keys.sort();
    let count = keys.len();
    Ok(axum::Json(BlobListResponse { keys, count }))
}

/// Recursively collect relative keys under `base`, skipping in-flight temp files.
async fn collect_keys(
    base: &StdPath,
    dir: &StdPath,
    out: &mut Vec<String>,
) -> Result<(), ApiError> {
    // Iterative BFS to avoid async recursion / boxing.
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let mut rd = tokio::fs::read_dir(&current)
            .await
            .map_err(|_| io_error("failed to list bucket"))?;
        while let Some(entry) = rd
            .next_entry()
            .await
            .map_err(|_| io_error("failed to list bucket"))?
        {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .await
                .map_err(|_| io_error("failed to list bucket"))?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() {
                // Skip transient temp files from interrupted PUTs.
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(".tmp-") {
                        continue;
                    }
                }
                if let Ok(rel) = path.strip_prefix(base) {
                    // Normalize to forward-slash keys.
                    let key = rel
                        .components()
                        .filter_map(|c| match c {
                            Component::Normal(s) => s.to_str(),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("/");
                    if !key.is_empty() {
                        out.push(key);
                    }
                }
            }
        }
    }
    Ok(())
}
