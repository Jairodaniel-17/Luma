//! On-the-fly image transform of blobs already in the store (Cloudflare
//! Images-equivalent). Stateless: reads source bytes from
//! `{data_dir}/blobs/{bucket}/{key}`, decodes, optionally resizes to fit within
//! `w`/`h` preserving aspect ratio, re-encodes to png/jpeg, and returns bytes.
//!
//! These routes live in the authenticated router chain, so every request
//! requires a valid Bearer api_key.

use crate::api::errors::ApiError;
use crate::api::AppState;
use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use image::imageops::FilterType;
use image::{ImageFormat, ImageReader};
use serde::Deserialize;
use std::io::Cursor;
use std::path::{Path as StdPath, PathBuf};

/// Cap on requested output dimension (px) to reject absurd resizes / OOM.
const MAX_DIMENSION: u32 = 5000;

#[derive(Debug, Deserialize)]
pub struct TransformParams {
    w: Option<u32>,
    h: Option<u32>,
    format: Option<String>,
    quality: Option<u8>,
}

fn blobs_root(state: &AppState) -> PathBuf {
    let data_dir = state.config.data_dir.as_deref().unwrap_or("data");
    PathBuf::from(data_dir).join("blobs")
}

/// Validate a single path component (bucket name or one key segment).
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

fn bad_request(msg: &'static str) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, "invalid_argument", msg)
}

/// GET /v1/image/:bucket/:key — transform a stored image on the fly.
pub async fn transform(
    State(state): State<AppState>,
    Path((bucket, key)): Path<(String, String)>,
    Query(params): Query<TransformParams>,
) -> Result<impl IntoResponse, ApiError> {
    let path = resolve_blob_path(&state, &bucket, &key)?;

    // Validate requested dimensions before doing any work.
    for dim in [params.w, params.h].into_iter().flatten() {
        if dim == 0 || dim > MAX_DIMENSION {
            return Err(bad_request("requested dimension out of range"));
        }
    }
    if let Some(q) = params.quality {
        if !(1..=100).contains(&q) {
            return Err(bad_request("quality must be 1-100"));
        }
    }

    // Resolve target encoding. Default: keep source format, falling back to jpeg.
    let target_format = match params.format.as_deref() {
        None => None,
        Some("png") => Some(ImageFormat::Png),
        Some("jpeg") | Some("jpg") => Some(ImageFormat::Jpeg),
        Some(_) => return Err(bad_request("unsupported format (png|jpeg)")),
    };

    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "not_found",
                "source image not found",
            ));
        }
        Err(_) => {
            return Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "io_error",
                "failed to read source image",
            ))
        }
    };

    // CPU-bound decode/resize/encode: run off the async runtime.
    let (out, content_type) = tokio::task::spawn_blocking(move || {
        process(&bytes, params.w, params.h, target_format, params.quality)
    })
    .await
    .map_err(|_| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "image task failed",
        )
    })??;

    let len = out.len();
    let headers = [
        (header::CONTENT_TYPE, content_type.to_string()),
        (header::CONTENT_LENGTH, len.to_string()),
    ];
    Ok((StatusCode::OK, headers, out))
}

/// Decode, optionally resize-to-fit, and re-encode. Returns (bytes, content-type).
fn process(
    bytes: &[u8],
    w: Option<u32>,
    h: Option<u32>,
    target_format: Option<ImageFormat>,
    quality: Option<u8>,
) -> Result<(Vec<u8>, &'static str), ApiError> {
    // Detect source format from content (not the key extension).
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| bad_request("failed to read image"))?;
    let source_format = reader.format();
    let mut img = reader
        .decode()
        .map_err(|_| bad_request("undecodable image"))?;

    // Resize to fit within the given bounds, preserving aspect ratio. Only when
    // at least one dimension is requested; missing bound uses MAX_DIMENSION.
    if w.is_some() || h.is_some() {
        let bound_w = w.unwrap_or(MAX_DIMENSION);
        let bound_h = h.unwrap_or(MAX_DIMENSION);
        img = img.resize(bound_w, bound_h, FilterType::Lanczos3);
    }

    // Choose output format: explicit > keep source (png/jpeg only) > jpeg.
    let out_format = target_format.unwrap_or(match source_format {
        Some(ImageFormat::Png) => ImageFormat::Png,
        _ => ImageFormat::Jpeg,
    });

    let mut buf: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    match out_format {
        ImageFormat::Jpeg => {
            let q = quality.unwrap_or(85);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, q);
            // JPEG has no alpha channel; convert to RGB8.
            enc.encode_image(&img.to_rgb8())
                .map_err(|_| bad_request("failed to encode jpeg"))?;
            Ok((buf, "image/jpeg"))
        }
        ImageFormat::Png => {
            img.write_to(&mut cursor, ImageFormat::Png)
                .map_err(|_| bad_request("failed to encode png"))?;
            Ok((buf, "image/png"))
        }
        // Unreachable: out_format is only ever Jpeg or Png by construction.
        _ => Err(bad_request("unsupported output format")),
    }
}
