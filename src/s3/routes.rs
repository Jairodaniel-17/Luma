//! The S3-compatible HTTP surface.
//!
//! W3.2 of `docs/PLAN-MAESTRO.md`. Objects live where the native blob API keeps
//! them — `{data_dir}/blobs/{bucket}/{key}` — so the same bytes are reachable
//! through either door, the ownership registry applies to both, and an org's
//! quota counts them once. Two stores would have meant two quotas, two backup
//! paths and two ways to be inconsistent.
//!
//! ## Mounted on its own port
//!
//! S3 owns the root of the path space: `GET /` is ListBuckets and
//! `GET /{bucket}/{key}` is any object. That cannot share a router with
//! `/v1/...` without one of them shadowing the other, so it listens separately
//! (`s3_port`, off by default).
//!
//! ## What is supported, and what is not
//!
//! Supported: ListBuckets, CreateBucket, DeleteBucket, ListObjectsV2 (with
//! prefix, delimiter and continuation), PutObject, GetObject, HeadObject,
//! DeleteObject, DeleteObjects, and multipart upload.
//!
//! **Not** supported, and refused with S3's own `NotImplemented` rather than a
//! quiet success: versioning, lifecycle, ACLs, replication, object locking,
//! server-side encryption headers, tagging, CORS configuration and website
//! hosting. A client that sets an ACL and gets a 200 believes the object is
//! private when it is not, which is worse than being told no.

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::api::AppState;
use crate::s3::credentials::S3Credentials;
use crate::s3::sigv4;
use crate::s3::xml;

/// An S3 error, rendered as the XML document clients parse.
pub struct S3Error {
    pub status: StatusCode,
    pub code: &'static str,
    message: String,
    resource: String,
    /// Headers the error itself has to carry.
    ///
    /// Only one error needs this today, and it genuinely needs it: a `416` must
    /// answer with `Content-Range: bytes */{size}` so the client learns the
    /// actual length instead of having to guess and retry.
    headers: Vec<(axum::http::HeaderName, String)>,
}

impl S3Error {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            resource: String::new(),
            headers: Vec::new(),
        }
    }

    fn at(mut self, resource: impl Into<String>) -> Self {
        self.resource = resource.into();
        self
    }

    fn with_header(mut self, name: axum::http::HeaderName, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    fn no_such_key(resource: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "NoSuchKey",
            "The specified key does not exist.",
        )
        .at(resource)
    }

    fn no_such_bucket(resource: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "NoSuchBucket",
            "The specified bucket does not exist.",
        )
        .at(resource)
    }

    fn access_denied(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, "AccessDenied", message)
    }

    fn not_implemented(what: &str) -> Self {
        Self::new(
            StatusCode::NOT_IMPLEMENTED,
            "NotImplemented",
            format!("{what} is not supported by this server."),
        )
    }
}

impl IntoResponse for S3Error {
    fn into_response(self) -> Response {
        let body = xml::error(self.code, &self.message, &self.resource);
        let mut response = (
            self.status,
            [(axum::http::header::CONTENT_TYPE, "application/xml")],
            body,
        )
            .into_response();
        for (name, value) in self.headers {
            if let Ok(value) = axum::http::HeaderValue::from_str(&value) {
                response.headers_mut().insert(name, value);
            }
        }
        response
    }
}

type S3Result<T> = Result<T, S3Error>;

/// The authenticated caller.
struct Caller {
    org_id: String,
    /// Present only when the body arrives chunk-framed
    /// (`STREAMING-AWS4-HMAC-SHA256-PAYLOAD`). See `unframe_body`.
    streaming: Option<sigv4::StreamingContext>,
}

/// The object's actual bytes, given what arrived on the wire.
///
/// For an ordinary request the two are the same. For a chunk-framed one they are
/// not, and writing the wire bytes stores the chunk sizes and signatures *inside
/// the object* — a corrupt object with a 200 in reply. So the framing is removed
/// here, and since it has to be parsed anyway, each chunk's signature is checked
/// on the way through.
fn unframe_body(caller: &Caller, body: &Bytes) -> Result<Vec<u8>, S3Error> {
    let Some(ctx) = &caller.streaming else {
        return Ok(body.to_vec());
    };
    sigv4::dechunk_and_verify(body, ctx).map_err(|verdict| match verdict {
        sigv4::Verdict::Mismatch => S3Error::access_denied("a chunk signature did not match"),
        sigv4::Verdict::Malformed(reason) => {
            S3Error::new(StatusCode::BAD_REQUEST, "InvalidRequest", reason)
        }
        // `dechunk_and_verify` returns only the two above on failure.
        sigv4::Verdict::Ok => S3Error::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "InternalError",
            "unreachable",
        ),
    })
}

/// Verify SigV4 and resolve the organization.
///
/// Anonymous access is not supported at all: S3 allows it for public buckets,
/// and a multi-tenant store where a missing signature means "public" is one
/// misconfiguration away from being an open bucket.
/// `method` must be the method that **arrived**, never a literal chosen by the
/// route.
///
/// SigV4 signs the method as part of the canonical request, so signing "GET" for
/// a request the client sent as "HEAD" produces a mismatch and a 403 that reads
/// exactly like a wrong secret. Every handler used to pass a constant, which was
/// fine only while each route served one method — and axum answers HEAD with the
/// GET handler when there is no HEAD route, which is precisely what `HeadBucket`
/// does. MinIO's mint suite failed on its first test because of it.
async fn authenticate(
    state: &AppState,
    method: &str,
    path: &str,
    query: &str,
    headers: &HeaderMap,
) -> S3Result<Caller> {
    let Some(sqlite) = state.sqlite.clone() else {
        return Err(S3Error::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "ServiceUnavailable",
            "the credential store is not available",
        ));
    };

    // A presigned URL carries its signature in the query string and has no
    // Authorization header at all, so it is checked first: otherwise a valid
    // presigned request would be rejected for "requiring a signed request".
    let presigned = sigv4::parse_presigned(query);
    let credential = match (&presigned, headers.get(axum::http::header::AUTHORIZATION)) {
        (Some(p), _) => p.credential.clone(),
        (None, Some(header)) => {
            let text = header
                .to_str()
                .map_err(|_| S3Error::access_denied("malformed Authorization header"))?;
            sigv4::parse_authorization(text)
                .map_err(|reason| S3Error::access_denied(reason.to_string()))?
        }
        (None, None) => {
            // A SigV2 presigned URL has this shape. Only SigV4 is supported,
            // and saying which scheme was seen is the difference between a
            // one-line client fix and an afternoon: the generic message sends
            // the reader looking at their credentials, which are fine.
            if query.contains("AWSAccessKeyId=") && query.contains("Signature=") {
                return Err(S3Error::access_denied(
                    "this URL is signed with SigV2, which this server does not support.                      Configure the client for SigV4 — for boto3,                      Config(signature_version='s3v4')",
                ));
            }
            return Err(S3Error::access_denied(
                "this server requires a signed request",
            ));
        }
    };

    let store = S3Credentials::new(std::sync::Arc::new(sqlite));
    let found = store
        .lookup(&credential.access_key_id)
        .await
        .map_err(|e| {
            S3Error::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                e.to_string(),
            )
        })?
        // An unknown key id and a bad signature give the same answer on purpose:
        // distinguishing them tells an attacker which key ids exist.
        .ok_or_else(|| S3Error::access_denied("the signature did not match"))?;

    let header_pairs: Vec<(String, String)> = headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_ascii_lowercase(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    let payload_hash = headers
        .get("x-amz-content-sha256")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(sigv4::EMPTY_PAYLOAD_HASH)
        .to_string();

    let verdict = match &presigned {
        Some(p) => sigv4::verify_presigned(
            method,
            path,
            query,
            &header_pairs,
            p,
            &found.secret_access_key,
            std::time::SystemTime::now(),
        ),
        None => {
            let request = sigv4::Request {
                method,
                path,
                query,
                headers: &header_pairs,
                payload_hash: &payload_hash,
            };
            sigv4::verify(&request, &credential, &found.secret_access_key)
        }
    };

    match verdict {
        sigv4::Verdict::Ok => Ok(Caller {
            org_id: found.org_id,
            // Only built for a chunk-framed body, and only after the request's
            // own signature verified — the chain's seed *is* that signature.
            streaming: if sigv4::is_streaming_payload(&payload_hash) {
                sigv4::StreamingContext::from_verified(
                    &credential,
                    &found.secret_access_key,
                    &header_pairs,
                )
            } else {
                None
            },
        }),
        sigv4::Verdict::Mismatch => Err(S3Error::access_denied("the signature did not match")),
        sigv4::Verdict::Malformed(reason) => Err(S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidRequest",
            reason,
        )),
    }
}

fn blobs_root(state: &AppState) -> PathBuf {
    let data_dir = state
        .config
        .data_dir
        .clone()
        .unwrap_or_else(|| "data".into());
    PathBuf::from(data_dir).join("blobs")
}

/// Reject a bucket or key that could escape the store.
///
/// The same rule the native blob API uses, restated here rather than shared,
/// because S3 accepts key shapes the native API never sees — a client can send
/// `..%2F..` and expect it to be a literal key name.
fn safe_segment(segment: &str) -> S3Result<()> {
    if segment.is_empty()
        || segment.contains("..")
        || segment.starts_with('/')
        || segment.contains('\\')
        || segment.contains('\0')
    {
        return Err(S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidBucketName",
            "the name contains characters this server refuses",
        ));
    }
    Ok(())
}

/// Confirm the org owns this bucket, or claim it on first touch.
async fn own_bucket(state: &AppState, caller: &Caller, bucket: &str) -> S3Result<()> {
    let Some(accounts) = state.accounts.clone() else {
        return Ok(());
    };
    match accounts.collection_owner(bucket).await {
        // Another org's bucket is reported as missing, not forbidden: telling a
        // caller "that exists but is not yours" is a way to enumerate other
        // tenants' bucket names.
        Ok(Some(owner)) if owner != caller.org_id => Err(S3Error::no_such_bucket(bucket)),
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            let _ = accounts.register_collection(bucket, &caller.org_id).await;
            Ok(())
        }
        Err(e) => Err(S3Error::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "InternalError",
            e.to_string(),
        )),
    }
}

/// Query parameters that mean "operate on a subresource".
///
/// Present and unsupported, they must be refused rather than ignored: a client
/// that sets a lifecycle policy and gets a 200 believes objects will expire.
const UNSUPPORTED_SUBRESOURCES: &[&str] = &[
    "versioning",
    "lifecycle",
    "acl",
    "replication",
    "object-lock",
    "tagging",
    "cors",
    "website",
    "policy",
    "encryption",
    "logging",
    "notification",
    "requestPayment",
    "accelerate",
    "analytics",
    "inventory",
    "metrics",
    "publicAccessBlock",
];

fn refuse_unsupported(params: &BTreeMap<String, String>) -> S3Result<()> {
    for name in UNSUPPORTED_SUBRESOURCES {
        if params.contains_key(*name) {
            return Err(S3Error::not_implemented(name));
        }
    }
    Ok(())
}

fn iso8601(time: std::time::SystemTime) -> String {
    let secs = time
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // S3 uses this exact shape; a client comparing timestamps as strings — and
    // some do — needs the milliseconds and the trailing Z.
    let days = secs / 86_400;
    let time_of_day = secs % 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.000Z",
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60
    )
}

/// An HTTP date, which is **not** the same shape as the one S3 puts in XML.
///
/// `Last-Modified` is an HTTP header, so RFC 9110 fixes its format:
/// `Sun, 06 Nov 1994 08:49:37 GMT`, always GMT, always those English
/// abbreviations. This server sent the ISO 8601 string it uses inside the
/// listing XML instead, and the two are not interchangeable — a listing that
/// said `2026-08-23T17:29:14.000Z` in a header made minio-py fail with "time
/// data does not match HTTP header format" on its first `PutObject`. boto3 does
/// not parse the header, which is why 16 checks against it never noticed.
///
/// The XML uses [`iso8601`] and keeps doing so; only the header changed.
fn http_date(time: std::time::SystemTime) -> String {
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let secs = time
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let time_of_day = secs % 86_400;
    let (year, month, day) = civil_from_days(days as i64);
    // 1970-01-01 was a Thursday, which is index 4 with Sunday at 0.
    let weekday = DAYS[((days + 4) % 7) as usize];
    let month_name = MONTHS[(month as usize).saturating_sub(1).min(11)];
    format!(
        "{weekday}, {day:02} {month_name} {year:04} {:02}:{:02}:{:02} GMT",
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60
    )
}

/// Days since the epoch to a calendar date (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The ETag of a single-part upload: the MD5 of the body, hex, as S3 defines it.
///
/// It used to be the first 16 bytes of a SHA-256, on the reasoning that adding a
/// dependency on a cryptographically broken hash for a checksum was not worth
/// it. Two things were wrong with that. `md-5` was already in the dependency
/// tree, so the dependency cost nothing; and MD5 here is not a security
/// primitive but a wire format — a client that recomputes the ETag to verify a
/// download is checking for corruption, and it either matches or the download
/// looks corrupt.
fn etag_of(bytes: &[u8]) -> String {
    hex_lower(&md5_digest(bytes))
}

fn md5_digest(bytes: &[u8]) -> [u8; 16] {
    use md5::{Digest, Md5};
    Md5::digest(bytes).into()
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The ETag of a completed multipart upload.
///
/// Not the MD5 of the assembled object, which is what a naive implementation
/// produces and what no S3 client expects. S3's form is the MD5 of the
/// *concatenated binary digests of the parts*, then a dash and the part count —
/// so a client can tell a multipart object from a single-part one by the dash
/// alone, and knows not to compare it against the MD5 of what it downloaded.
fn multipart_etag(part_bodies: &[Vec<u8>]) -> String {
    let mut digests = Vec::with_capacity(part_bodies.len() * 16);
    for body in part_bodies {
        digests.extend_from_slice(&md5_digest(body));
    }
    format!("{}-{}", hex_lower(&md5_digest(&digests)), part_bodies.len())
}

// ── handlers ─────────────────────────────────────────────────────────────────

pub async fn list_buckets(
    State(state): State<AppState>,
    method: axum::http::Method,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    let caller = authenticate(&state, method.as_str(), "/", "", &headers).await?;
    let Some(accounts) = state.accounts.clone() else {
        return Ok(xml_response(StatusCode::OK, xml::list_buckets(&[])));
    };
    let names = accounts
        .names_owned_by(&caller.org_id)
        .await
        .unwrap_or_default();

    let root = blobs_root(&state);
    // Only the names that are actually buckets on disk: the ownership registry
    // is keyed by name across every scoped resource, so it also holds vector
    // collections and doc namespaces.
    let buckets: Vec<(String, String)> = names
        .into_iter()
        .filter(|name| root.join(name).is_dir())
        .map(|name| {
            let created = std::fs::metadata(root.join(&name))
                .and_then(|m| m.created())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            (name, iso8601(created))
        })
        .collect();
    Ok(xml_response(StatusCode::OK, xml::list_buckets(&buckets)))
}

pub async fn bucket_get(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path(bucket): Path<String>,
    Query(params): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    refuse_unsupported(&params)?;
    safe_segment(&bucket)?;
    let query = raw_query(&params);
    let caller = authenticate(
        &state,
        method.as_str(),
        &format!("/{bucket}"),
        &query,
        &headers,
    )
    .await?;
    own_bucket(&state, &caller, &bucket).await?;

    let directory = blobs_root(&state).join(&bucket);
    if !directory.is_dir() {
        return Err(S3Error::no_such_bucket(&bucket));
    }

    let prefix = params.get("prefix").cloned().unwrap_or_default();
    let delimiter = params.get("delimiter").cloned().unwrap_or_default();
    let max_keys: usize = params
        .get("max-keys")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000)
        .min(1000);
    let after = params
        .get("continuation-token")
        .or_else(|| params.get("start-after"))
        .cloned()
        .unwrap_or_default();

    let mut keys = Vec::new();
    collect_keys(&directory, &directory, &mut keys);
    keys.sort();

    let mut objects = Vec::new();
    let mut common: Vec<String> = Vec::new();
    let mut truncated = false;
    let mut next_token = None;

    for key in keys {
        if !key.starts_with(&prefix) {
            continue;
        }
        // The continuation token is the last key returned, so resuming is
        // "everything after this" — which is exactly what sorted order makes
        // cheap and what a client expects when objects are added mid-listing.
        if !after.is_empty() && key <= after {
            continue;
        }
        if !delimiter.is_empty() {
            if let Some(at) = key[prefix.len()..].find(&delimiter) {
                let group = format!("{}{}", &key[..prefix.len() + at], delimiter);
                if !common.contains(&group) {
                    common.push(group);
                }
                continue;
            }
        }
        if objects.len() >= max_keys {
            truncated = true;
            next_token = objects.last().map(|o: &xml::ObjectEntry| o.key.clone());
            break;
        }
        let path = directory.join(&key);
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        objects.push(xml::ObjectEntry {
            key,
            size: meta.len(),
            last_modified: iso8601(meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)),
            etag: std::fs::read(&path)
                .map(|b| etag_of(&b))
                .unwrap_or_default(),
        });
    }

    Ok(xml_response(
        StatusCode::OK,
        xml::list_objects_v2(
            &bucket,
            &prefix,
            &delimiter,
            max_keys,
            truncated,
            next_token.as_deref(),
            &objects,
            &common,
        ),
    ))
}

pub async fn bucket_put(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path(bucket): Path<String>,
    Query(params): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    refuse_unsupported(&params)?;
    safe_segment(&bucket)?;
    let caller = authenticate(&state, method.as_str(), &format!("/{bucket}"), "", &headers).await?;
    own_bucket(&state, &caller, &bucket).await?;

    std::fs::create_dir_all(blobs_root(&state).join(&bucket)).map_err(|e| {
        S3Error::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "InternalError",
            e.to_string(),
        )
    })?;
    Ok((StatusCode::OK, [("Location", format!("/{bucket}"))]).into_response())
}

pub async fn bucket_delete(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path(bucket): Path<String>,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    safe_segment(&bucket)?;
    let caller = authenticate(&state, method.as_str(), &format!("/{bucket}"), "", &headers).await?;
    own_bucket(&state, &caller, &bucket).await?;

    let directory = blobs_root(&state).join(&bucket);
    if !directory.is_dir() {
        return Err(S3Error::no_such_bucket(&bucket));
    }
    let mut keys = Vec::new();
    collect_keys(&directory, &directory, &mut keys);
    if !keys.is_empty() {
        // S3 refuses, and so must this: deleting a bucket's contents because
        // somebody deleted the bucket is not a recoverable mistake.
        return Err(S3Error::new(
            StatusCode::CONFLICT,
            "BucketNotEmpty",
            "The bucket you tried to delete is not empty.",
        )
        .at(bucket));
    }
    let _ = std::fs::remove_dir(&directory);
    Ok(StatusCode::NO_CONTENT.into_response())
}

fn collect_keys(root: &std::path::Path, directory: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_keys(root, &path, out);
        } else if let Ok(relative) = path.strip_prefix(root) {
            // Forward slashes regardless of platform: the key is what the client
            // sent, and a Windows server must not hand back backslashes.
            out.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn raw_query(params: &BTreeMap<String, String>) -> String {
    params
        .iter()
        .map(|(k, v)| {
            if v.is_empty() {
                k.clone()
            } else {
                format!("{k}={v}")
            }
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn xml_response(status: StatusCode, body: String) -> Response {
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/xml")],
        body,
    )
        .into_response()
}

// ── objects ─────────────────────────────────────────────────────────────────

fn object_path(state: &AppState, bucket: &str, key: &str) -> S3Result<PathBuf> {
    safe_segment(bucket)?;
    if key.is_empty() || key.contains("..") || key.contains('\0') {
        return Err(S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidArgument",
            "the key contains characters this server refuses",
        ));
    }
    Ok(blobs_root(state).join(bucket).join(key))
}

/// Where the composite ETag of a multipart object is remembered.
///
/// A parallel tree rather than a sidecar next to the object: listings walk the
/// blob directory, so a `key.etag` file living beside `key` would show up as an
/// object of its own, and reserving a suffix would make `report.etag` an
/// unstorable key name.
fn etag_sidecar(state: &AppState, bucket: &str, key: &str) -> S3Result<PathBuf> {
    safe_segment(bucket)?;
    if key.is_empty() || key.contains("..") || key.contains('\0') {
        return Err(S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidArgument",
            "the key contains characters this server refuses",
        ));
    }
    let data_dir = state
        .config
        .data_dir
        .clone()
        .unwrap_or_else(|| "data".into());
    Ok(PathBuf::from(data_dir)
        .join("s3-meta")
        .join("etags")
        .join(bucket)
        .join(key))
}

/// What this server remembers about an object beyond its bytes.
///
/// Persisted as JSON beside the object tree (not inside it — listings walk the
/// blob directory, so a sidecar living next to `key` would show up as an object
/// of its own).
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct ObjectMeta {
    /// `md5(concat(part digests))-N` for a multipart object; absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    composite_etag: Option<String>,
    /// The `Content-Type` the caller sent, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content_type: Option<String>,
    /// `x-amz-meta-*`, with the prefix stripped and the name lowercased.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    user: BTreeMap<String, String>,
    /// The MD5 of the bytes this record describes.
    ///
    /// What makes the record **self-validating**: it is believed only while the
    /// object still hashes to this. The alternative was to invalidate it from
    /// every path that can write an object, and those are not all in this file —
    /// the same bytes are reachable through the native `/v1/blob` API, which
    /// knows nothing about S3 metadata. A check that cannot be forgotten beats a
    /// list of call sites that can.
    guard: String,
}

/// Read the metadata record for an object, if it still describes these bytes.
fn read_meta(state: &AppState, bucket: &str, key: &str, bytes: &[u8]) -> Option<ObjectMeta> {
    let path = meta_sidecar(state, bucket, key).ok()?;
    let text = std::fs::read_to_string(&path).ok()?;
    let meta: ObjectMeta = serde_json::from_str(&text).ok()?;
    (meta.guard == etag_of(bytes)).then_some(meta)
}

/// The ETag to report for an object that already exists on disk.
///
/// S3's multipart ETag is `md5(concat(part digests))-N`, and it is **stable**:
/// what `CompleteMultipartUpload` returns is what every later `HEAD` and `GET`
/// returns. This server computed the composite at completion, returned it, and
/// then threw it away — so the same object answered with a composite ETag once
/// and with the plain MD5 of its bytes forever after.
///
/// That is not a cosmetic difference. `aws s3 sync`, rclone and every cache that
/// validates on ETag would see the object change on every check, and re-transfer
/// it. It survived because the check that covers this
/// (`a_multipart_etag_has_the_dash_and_part_count`) only ever read the ETag out
/// of the completion *response* and never asked the server again.
///
/// A missing record falls back to the MD5 of the bytes, which is both correct
/// for single-part objects and exactly the previous behaviour — so data written
/// by an older build, or restored from a backup taken before the sidecar
/// existed, keeps answering as it did instead of failing.
fn stored_etag(state: &AppState, bucket: &str, key: &str, bytes: &[u8]) -> String {
    read_meta(state, bucket, key, bytes)
        .and_then(|m| m.composite_etag)
        .unwrap_or_else(|| etag_of(bytes))
}

/// The headers an object answers with beyond its ETag: its content type and any
/// `x-amz-meta-*` the caller stored with it.
fn stored_headers(
    state: &AppState,
    bucket: &str,
    key: &str,
    bytes: &[u8],
) -> (String, BTreeMap<String, String>) {
    match read_meta(state, bucket, key, bytes) {
        Some(meta) => (
            meta.content_type
                .unwrap_or_else(|| "application/octet-stream".to_string()),
            meta.user,
        ),
        // S3's own default for an object stored without one.
        None => ("application/octet-stream".to_string(), BTreeMap::new()),
    }
}

/// The `Content-Type` a request carried, if it named one worth keeping.
///
/// Clients that send no body type still send `application/octet-stream`, and
/// recording that is the same as recording nothing — it is already the default.
fn content_type_of(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && *v != "application/octet-stream")
        .map(|v| v.to_string())
}

/// Write the stored `x-amz-meta-*` back onto a response.
fn put_user_metadata(headers: &mut HeaderMap, user: &BTreeMap<String, String>) {
    for (name, value) in user {
        let full = format!("x-amz-meta-{name}");
        if let (Ok(name), Ok(value)) = (
            axum::http::HeaderName::try_from(full.as_str()),
            axum::http::HeaderValue::from_str(value),
        ) {
            headers.insert(name, value);
        }
    }
}

/// The `x-amz-meta-*` headers of a request, prefix stripped and names lowercased.
///
/// Lowercased because HTTP header names are case-insensitive while a JSON map's
/// keys are not: a client that sends `X-Amz-Meta-Testing` and reads back
/// `x-amz-meta-testing` must find it.
fn user_metadata(headers: &HeaderMap) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (name, value) in headers {
        let name = name.as_str().to_ascii_lowercase();
        if let Some(rest) = name.strip_prefix("x-amz-meta-") {
            if let Ok(text) = value.to_str() {
                out.insert(rest.to_string(), text.to_string());
            }
        }
    }
    out
}

/// Where an object's metadata record lives.
fn meta_sidecar(state: &AppState, bucket: &str, key: &str) -> S3Result<PathBuf> {
    etag_sidecar(state, bucket, key)
}

/// Write an object's metadata record, or remove it.
///
/// `None` clears it, which every write that replaces an object must do: the old
/// content type, user metadata and composite ETag all describe bytes that are no
/// longer there, and a stale record would be worse than none.
fn remember_meta(state: &AppState, bucket: &str, key: &str, meta: Option<ObjectMeta>) {
    let Ok(path) = meta_sidecar(state, bucket, key) else {
        return;
    };
    match meta {
        Some(meta) => {
            // Nothing worth remembering: skip the file rather than leave an
            // empty record for every ordinary PUT.
            if meta.composite_etag.is_none() && meta.content_type.is_none() && meta.user.is_empty()
            {
                let _ = std::fs::remove_file(&path);
                return;
            }
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(text) = serde_json::to_string(&meta) {
                let _ = std::fs::write(&path, text);
            }
        }
        None => {
            let _ = std::fs::remove_file(&path);
        }
    }
}

pub async fn object_put(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path((bucket, key)): Path<(String, String)>,
    Query(params): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, S3Error> {
    refuse_unsupported(&params)?;
    let caller = authenticate(
        &state,
        method.as_str(),
        &format!("/{bucket}/{key}"),
        &raw_query(&params),
        &headers,
    )
    .await?;
    own_bucket(&state, &caller, &bucket).await?;

    // `CopyObject` is a `PUT` with a header and no body, so without this it fell
    // through to the ordinary write path and **stored an empty object** —
    // answering 200 with an XML-less body while destroying the destination. An
    // `aws s3 cp` between two keys lost the data and reported success.
    if let Some(source) = headers.get("x-amz-copy-source") {
        let source = source
            .to_str()
            .map_err(|_| {
                S3Error::new(
                    StatusCode::BAD_REQUEST,
                    "InvalidArgument",
                    "x-amz-copy-source is not valid text",
                )
            })?
            .to_string();
        return copy_object(&state, &caller, &bucket, &key, &source, &headers).await;
    }

    // Before anything looks at the bytes: for a chunk-framed body the wire
    // content is not the object, and every length, hash and write below has to
    // see the unframed payload.
    let body = Bytes::from(unframe_body(&caller, &body)?);

    // Multipart part upload arrives on the same path with these parameters.
    if let (Some(upload_id), Some(part)) = (params.get("uploadId"), params.get("partNumber")) {
        return upload_part(&state, &caller, &bucket, &key, upload_id, part, &body).await;
    }

    let path = object_path(&state, &bucket, &key)?;
    let replacing = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // The same quota the native API enforces, because it is the same storage.
    let ctx = enforcing_context(&state, &caller).await;
    crate::api::quotas::guard_blob_write(
        crate::api::quotas::BlobQuotaStore {
            sqlite: state.sqlite.as_ref(),
            accounts: state.accounts.as_deref(),
            blobs_root: &blobs_root(&state),
        },
        &ctx,
        body.len() as u64,
        replacing,
    )
    .await
    .map_err(|_| {
        S3Error::new(
            StatusCode::INSUFFICIENT_STORAGE,
            "QuotaExceeded",
            "the organization is out of object storage",
        )
    })?;

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    crate::durability::write_atomic(&path, &body)
        .await
        .map_err(|e| {
            S3Error::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                e.to_string(),
            )
        })?;
    // Replaces whatever was remembered: an ordinary PUT is not multipart, so any
    // composite ETag is retired, and the content type and user metadata are the
    // ones this request carried.
    remember_meta(
        &state,
        &bucket,
        &key,
        Some(ObjectMeta {
            composite_etag: None,
            content_type: content_type_of(&headers),
            user: user_metadata(&headers),
            guard: etag_of(&body),
        }),
    );
    crate::api::quotas::record_blob_delta(
        state.sqlite.as_ref(),
        &ctx,
        body.len() as i64 - replacing as i64,
    )
    .await;

    let etag = etag_of(&body);
    Ok((StatusCode::OK, [("ETag", format!("\"{etag}\""))]).into_response())
}

/// A context for accounting: who the bytes belong to, no limits attached.
///
/// Enough for `record_blob_delta`, which only needs the organization. Anything
/// that has to *refuse* a write wants `enforcing_context` instead.
fn tenant_context(caller: &Caller) -> crate::api::TenantContext {
    crate::api::TenantContext {
        tenant_id: Some(caller.org_id.clone()),
        user_id: None,
        role: "member".to_string(),
        platform_admin: false,
        permissions: serde_json::json!({}),
        quotas: serde_json::json!({}),
    }
}

/// The same, with the organization's real quota resolved.
///
/// SigV4 identifies a caller by access key id, and that record carries an org
/// and nothing else — so this door used to run with empty quotas: the bytes were
/// counted, keeping usage correct, but nothing was ever refused. The limit is
/// read from the org's api keys, where it already lives; see
/// `quotas::quotas_for_org` for why the widest one is the right choice.
async fn enforcing_context(state: &AppState, caller: &Caller) -> crate::api::TenantContext {
    let mut ctx = tenant_context(caller);
    ctx.quotas =
        crate::api::quotas::quotas_for_org(state.auth_store.as_deref(), &caller.org_id).await;
    ctx
}

/// `CopyObject`: server-side copy, which is also how a client changes an
/// object's metadata without re-uploading it.
///
/// The metadata directive is the part that is easy to get wrong. `COPY` (the
/// default) carries the source's content type and `x-amz-meta-*` across;
/// `REPLACE` takes them from this request. Treating every copy as `REPLACE`
/// would silently strip metadata on the operation clients use precisely to
/// preserve it.
async fn copy_object(
    state: &AppState,
    caller: &Caller,
    bucket: &str,
    key: &str,
    source: &str,
    headers: &HeaderMap,
) -> Result<Response, S3Error> {
    let (src_bucket, src_key) = parse_copy_source(source).ok_or_else(|| {
        S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidArgument",
            "x-amz-copy-source must be /bucket/key",
        )
    })?;

    // The source bucket is checked too: a copy reads it, so letting it through
    // unchecked would be a cross-tenant read wearing a write's clothing.
    own_bucket(state, caller, &src_bucket).await?;

    let src_path = object_path(state, &src_bucket, &src_key)?;
    let bytes = std::fs::read(&src_path)
        .map_err(|_| S3Error::no_such_key(&format!("/{src_bucket}/{src_key}")))?;

    let replace = headers
        .get("x-amz-metadata-directive")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("REPLACE"));

    // Copying a key onto itself is only meaningful when it changes something,
    // which is what S3 says too.
    if src_bucket == bucket && src_key == key && !replace {
        return Err(S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidRequest",
            "the source and destination are the same and no metadata is being replaced",
        ));
    }

    let dst_path = object_path(state, bucket, key)?;
    if let Some(parent) = dst_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let replacing = std::fs::metadata(&dst_path).map(|m| m.len()).unwrap_or(0);
    crate::durability::write_atomic(&dst_path, &bytes)
        .await
        .map_err(|e| {
            S3Error::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                e.to_string(),
            )
        })?;

    let source_meta = read_meta(state, &src_bucket, &src_key, &bytes);
    let (content_type, user) = if replace {
        (content_type_of(headers), user_metadata(headers))
    } else {
        match &source_meta {
            Some(m) => (m.content_type.clone(), m.user.clone()),
            None => (None, BTreeMap::new()),
        }
    };
    // A copy of a multipart object is a fresh single object: its ETag is the MD5
    // of the bytes, which is what S3 reports too. Carrying the composite across
    // would describe part boundaries the copy does not have.
    remember_meta(
        state,
        bucket,
        key,
        Some(ObjectMeta {
            composite_etag: None,
            content_type,
            user,
            guard: etag_of(&bytes),
        }),
    );

    crate::api::quotas::record_blob_delta(
        state.sqlite.as_ref(),
        &tenant_context(caller),
        bytes.len() as i64 - replacing as i64,
    )
    .await;

    let modified = std::fs::metadata(&dst_path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    Ok(xml_response(
        StatusCode::OK,
        xml::copy_object(&etag_of(&bytes), &iso8601(modified)),
    ))
}

/// Split `x-amz-copy-source` into bucket and key.
///
/// The header is percent-encoded and may or may not have a leading slash, and it
/// may carry a `?versionId=` this server has no versions for. Accepting only one
/// of those shapes would reject clients that are behaving correctly.
fn parse_copy_source(raw: &str) -> Option<(String, String)> {
    let trimmed = raw.trim().trim_start_matches('/');
    let without_version = trimmed.split('?').next().unwrap_or(trimmed);
    let (bucket, key) = without_version.split_once('/')?;
    if bucket.is_empty() || key.is_empty() {
        return None;
    }
    Some((percent_decode(bucket), percent_decode(key)))
}

/// Decode `%XX` escapes, leaving anything malformed as written.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(byte) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub async fn object_get(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path((bucket, key)): Path<(String, String)>,
    Query(params): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    let caller = authenticate(
        &state,
        method.as_str(),
        &format!("/{bucket}/{key}"),
        &raw_query(&params),
        &headers,
    )
    .await?;
    own_bucket(&state, &caller, &bucket).await?;

    let path = object_path(&state, &bucket, &key)?;
    let bytes =
        std::fs::read(&path).map_err(|_| S3Error::no_such_key(&format!("/{bucket}/{key}")))?;
    let meta = std::fs::metadata(&path).ok();
    let etag = stored_etag(&state, &bucket, &key, &bytes);
    let modified = meta
        .and_then(|m| m.modified().ok())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let total = bytes.len() as u64;
    let range = headers
        .get(axum::http::header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(|raw| parse_range(raw, total));

    // `Some(None)` is a syntactically valid range that this object cannot
    // satisfy, which S3 answers with 416 and a `Content-Range` naming the size
    // — not with the whole object, and not with an empty 206.
    if range == Some(None) {
        return Err(S3Error::new(
            StatusCode::RANGE_NOT_SATISFIABLE,
            "InvalidRange",
            "The requested range is not satisfiable",
        )
        .at(format!("/{bucket}/{key}"))
        .with_header(
            axum::http::header::CONTENT_RANGE,
            format!("bytes */{total}"),
        ));
    }

    // Resolved before the body is moved out: the metadata record is keyed on the
    // hash of the *whole* object, so it has to be read while all of it is still
    // here, not from whatever slice a range request asked for.
    let (content_type, user) = stored_headers(&state, &bucket, &key, &bytes);

    let (status, body, content_range) = match range.flatten() {
        Some((first, last)) => (
            StatusCode::PARTIAL_CONTENT,
            bytes[first as usize..=last as usize].to_vec(),
            Some(format!("bytes {first}-{last}/{total}")),
        ),
        None => (StatusCode::OK, bytes, None),
    };
    let mut response = (
        status,
        [
            (axum::http::header::CONTENT_TYPE, content_type),
            (axum::http::header::ETAG, format!("\"{etag}\"")),
            (axum::http::header::LAST_MODIFIED, http_date(modified)),
            (axum::http::header::ACCEPT_RANGES, "bytes".to_string()),
        ],
        body,
    )
        .into_response();
    put_user_metadata(response.headers_mut(), &user);
    if let Some(value) = content_range {
        if let Ok(value) = axum::http::HeaderValue::from_str(&value) {
            response
                .headers_mut()
                .insert(axum::http::header::CONTENT_RANGE, value);
        }
    }
    Ok(response)
}

/// Parse one HTTP byte range against a known object size.
///
/// Three-valued on purpose, because the three cases have three different
/// answers and collapsing any two of them is a bug:
///
/// - `None` — not a range this server acts on, so serve the whole object with
///   `200`. That covers a malformed header (RFC 9110 says to ignore one) and a
///   multi-range request, which real S3 also answers with the entire object
///   rather than a multipart body.
/// - `Some(None)` — well formed, unsatisfiable: `416`.
/// - `Some(Some((first, last)))` — inclusive, clamped, and never empty: `206`.
///
/// Before this existed the `Range` header was not read at all, so every ranged
/// `GET` returned `200` and the whole body. That is worse than refusing the
/// header: boto3 downloads a large object as concurrent ranged `GET`s and
/// writes each reply at its own offset, so an ignored `Range` does not fail —
/// it writes the whole object into every slot and hands the caller a corrupt
/// file with no error anywhere. `tests/e2e/s3_scale.py` is what caught it.
fn parse_range(raw: &str, total: u64) -> Option<Option<(u64, u64)>> {
    let spec = raw.trim().strip_prefix("bytes=")?;
    if spec.contains(',') {
        return None;
    }
    let (start, end) = spec.split_once('-')?;
    let (start, end) = (start.trim(), end.trim());

    let (first, last) = if start.is_empty() {
        // `bytes=-N`: the final N bytes. A suffix of zero asks for nothing,
        // which is unsatisfiable rather than an empty success; a suffix larger
        // than the object simply means the whole object.
        let suffix: u64 = end.parse().ok()?;
        if suffix == 0 {
            return Some(None);
        }
        (total.saturating_sub(suffix), total.saturating_sub(1))
    } else {
        let first: u64 = start.parse().ok()?;
        // An open end, and an end past the object, both mean "to the last byte".
        let last = if end.is_empty() {
            total.saturating_sub(1)
        } else {
            end.parse::<u64>().ok()?.min(total.saturating_sub(1))
        };
        (first, last)
    };

    if total == 0 || first >= total || first > last {
        return Some(None);
    }
    Some(Some((first, last)))
}

pub async fn object_head(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path((bucket, key)): Path<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    let caller = authenticate(
        &state,
        method.as_str(),
        &format!("/{bucket}/{key}"),
        "",
        &headers,
    )
    .await?;
    own_bucket(&state, &caller, &bucket).await?;

    let path = object_path(&state, &bucket, &key)?;
    let meta =
        std::fs::metadata(&path).map_err(|_| S3Error::no_such_key(&format!("/{bucket}/{key}")))?;
    // Read once and derive both: HEAD must answer with the same ETag, content
    // type and user metadata as GET, and deriving them from different reads is
    // how the two drift.
    let bytes = std::fs::read(&path).unwrap_or_default();
    let etag = stored_etag(&state, &bucket, &key, &bytes);
    let (content_type, user) = stored_headers(&state, &bucket, &key, &bytes);

    let mut response = (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_LENGTH, meta.len().to_string()),
            (axum::http::header::CONTENT_TYPE, content_type),
            (axum::http::header::ETAG, format!("\"{etag}\"")),
            (
                axum::http::header::LAST_MODIFIED,
                http_date(meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)),
            ),
            (axum::http::header::ACCEPT_RANGES, "bytes".to_string()),
        ],
    )
        .into_response();
    put_user_metadata(response.headers_mut(), &user);
    Ok(response)
}

pub async fn object_delete(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path((bucket, key)): Path<(String, String)>,
    Query(params): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
) -> Result<Response, S3Error> {
    let caller = authenticate(
        &state,
        method.as_str(),
        &format!("/{bucket}/{key}"),
        &raw_query(&params),
        &headers,
    )
    .await?;
    own_bucket(&state, &caller, &bucket).await?;

    if let Some(upload_id) = params.get("uploadId") {
        // Abort a multipart upload: drop the parts and say nothing else.
        let _ = std::fs::remove_dir_all(multipart_dir(&state, upload_id));
        return Ok(StatusCode::NO_CONTENT.into_response());
    }

    let path = object_path(&state, &bucket, &key)?;
    let freed = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    // S3 delete is idempotent: a missing object is a 204, not a 404.
    if std::fs::remove_file(&path).is_ok() {
        crate::api::quotas::record_blob_delta(
            state.sqlite.as_ref(),
            &tenant_context(&caller),
            -(freed as i64),
        )
        .await;
    }
    // Unconditionally, not only when the object was there: a record left behind
    // would hand a later object at the same key the previous one's metadata.
    remember_meta(&state, &bucket, &key, None);
    Ok(StatusCode::NO_CONTENT.into_response())
}

// ── multipart ───────────────────────────────────────────────────────────────

fn multipart_dir(state: &AppState, upload_id: &str) -> PathBuf {
    let data_dir = state
        .config
        .data_dir
        .clone()
        .unwrap_or_else(|| "data".into());
    PathBuf::from(data_dir).join("s3-multipart").join(upload_id)
}

pub async fn object_post(
    State(state): State<AppState>,
    method: axum::http::Method,
    Path((bucket, key)): Path<(String, String)>,
    Query(params): Query<BTreeMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, S3Error> {
    let caller = authenticate(
        &state,
        method.as_str(),
        &format!("/{bucket}/{key}"),
        &raw_query(&params),
        &headers,
    )
    .await?;
    own_bucket(&state, &caller, &bucket).await?;

    if params.contains_key("uploads") {
        let upload_id = uuid::Uuid::new_v4().simple().to_string();
        let dir = multipart_dir(&state, &upload_id);
        std::fs::create_dir_all(&dir).map_err(|e| {
            S3Error::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                e.to_string(),
            )
        })?;
        // S3 takes a multipart object's content type and `x-amz-meta-*` from
        // this request, not from the parts and not from the completion call, so
        // they have to survive until then. `meta.json` cannot collide with a
        // part file: those are named by zero-padded part number.
        let pending = ObjectMeta {
            composite_etag: None,
            content_type: content_type_of(&headers),
            user: user_metadata(&headers),
            guard: String::new(),
        };
        if let Ok(text) = serde_json::to_string(&pending) {
            let _ = std::fs::write(dir.join("meta.json"), text);
        }
        return Ok(xml_response(
            StatusCode::OK,
            xml::initiate_multipart(&bucket, &key, &upload_id),
        ));
    }

    if let Some(upload_id) = params.get("uploadId") {
        return complete_multipart(&state, &caller, &bucket, &key, upload_id, &body).await;
    }

    Err(S3Error::not_implemented("this POST operation"))
}

async fn upload_part(
    state: &AppState,
    caller: &Caller,
    _bucket: &str,
    _key: &str,
    upload_id: &str,
    part: &str,
    body: &Bytes,
) -> Result<Response, S3Error> {
    let number: u32 = part.parse().map_err(|_| {
        S3Error::new(
            StatusCode::BAD_REQUEST,
            "InvalidArgument",
            "partNumber must be a number",
        )
    })?;
    let directory = multipart_dir(state, upload_id);
    if !directory.is_dir() {
        return Err(S3Error::new(
            StatusCode::NOT_FOUND,
            "NoSuchUpload",
            "The specified multipart upload does not exist.",
        ));
    }
    // Multipart had no quota guard at all, which made it the way around the
    // one `PutObject` enforces: upload the same bytes in parts and nothing
    // counted them. Charged per part, because the parts are what is on disk —
    // `complete` only concatenates what a guard already admitted.
    let ctx = enforcing_context(state, caller).await;
    let replacing = std::fs::metadata(directory.join(format!("{number:05}")))
        .map(|m| m.len())
        .unwrap_or(0);
    crate::api::quotas::guard_blob_write(
        crate::api::quotas::BlobQuotaStore {
            sqlite: state.sqlite.as_ref(),
            accounts: state.accounts.as_deref(),
            blobs_root: &blobs_root(state),
        },
        &ctx,
        body.len() as u64,
        replacing,
    )
    .await
    .map_err(|_| {
        S3Error::new(
            StatusCode::INSUFFICIENT_STORAGE,
            "QuotaExceeded",
            "the organization is out of object storage",
        )
    })?;

    // Zero-padded so the parts sort in numeric order on disk, which is the order
    // they have to be concatenated in.
    let path = directory.join(format!("{number:05}"));
    crate::durability::write_atomic(&path, body)
        .await
        .map_err(|e| {
            S3Error::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                e.to_string(),
            )
        })?;
    let etag = etag_of(body);
    Ok((StatusCode::OK, [("ETag", format!("\"{etag}\""))]).into_response())
}

async fn complete_multipart(
    state: &AppState,
    caller: &Caller,
    bucket: &str,
    key: &str,
    upload_id: &str,
    body: &Bytes,
) -> Result<Response, S3Error> {
    let directory = multipart_dir(state, upload_id);
    if !directory.is_dir() {
        return Err(S3Error::new(
            StatusCode::NOT_FOUND,
            "NoSuchUpload",
            "The specified multipart upload does not exist.",
        ));
    }

    // The client states the order. Using the order on disk instead would be
    // right by accident whenever parts were uploaded in sequence, and silently
    // wrong whenever they were not — which is the whole reason to use multipart.
    let wanted = xml::parse_complete_parts(&String::from_utf8_lossy(body));
    let numbers: Vec<u32> = if wanted.is_empty() {
        let mut found: Vec<u32> = std::fs::read_dir(&directory)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_string_lossy().parse().ok())
            .collect();
        found.sort_unstable();
        found
    } else {
        wanted
    };

    // Parts kept separately as well as concatenated: the multipart ETag is the
    // MD5 of the parts' digests, not of the assembled object, so the boundaries
    // are part of the answer.
    let mut part_bodies: Vec<Vec<u8>> = Vec::with_capacity(numbers.len());
    let mut assembled = Vec::new();
    for number in &numbers {
        let part = directory.join(format!("{number:05}"));
        let bytes = std::fs::read(&part).map_err(|_| {
            S3Error::new(
                StatusCode::BAD_REQUEST,
                "InvalidPart",
                format!("part {number} was never uploaded"),
            )
        })?;
        assembled.extend_from_slice(&bytes);
        part_bodies.push(bytes);
    }

    let path = object_path(state, bucket, key)?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let replacing = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    crate::durability::write_atomic(&path, &assembled)
        .await
        .map_err(|e| {
            S3Error::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                e.to_string(),
            )
        })?;
    // Before the directory goes: this is where CreateMultipartUpload left the
    // content type and `x-amz-meta-*` that S3 takes from the initiating request.
    let initiated: ObjectMeta = std::fs::read_to_string(directory.join("meta.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let _ = std::fs::remove_dir_all(&directory);

    crate::api::quotas::record_blob_delta(
        state.sqlite.as_ref(),
        &tenant_context(caller),
        assembled.len() as i64 - replacing as i64,
    )
    .await;

    let etag = multipart_etag(&part_bodies);
    remember_meta(
        state,
        bucket,
        key,
        Some(ObjectMeta {
            composite_etag: Some(etag.clone()),
            content_type: initiated.content_type,
            user: initiated.user,
            guard: etag_of(&assembled),
        }),
    );
    Ok(xml_response(
        StatusCode::OK,
        xml::complete_multipart(&format!("/{bucket}/{key}"), bucket, key, &etag),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_etag_is_the_md5_s3_clients_expect() {
        // Known answers, so a wrong hash cannot pass by agreeing with itself.
        // A client that recomputes the ETag to check a download is looking for
        // exactly these bytes.
        assert_eq!(etag_of(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(etag_of(b"hello world"), "5eb63bbbe01eeed093cb22bb8f5acdc3");
    }

    #[test]
    fn the_last_modified_header_is_an_http_date_not_the_xml_one() {
        // Two formats, two places, and they are not interchangeable: RFC 9110
        // fixes the header's shape, S3 fixes the listing's. Sending the listing
        // shape in the header is what made minio-py refuse a PutObject reply
        // with "time data does not match HTTP header format".
        let epoch = std::time::UNIX_EPOCH;
        assert_eq!(http_date(epoch), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(iso8601(epoch), "1970-01-01T00:00:00.000Z");

        // A known instant, so a wrong weekday or month cannot agree with itself:
        // 2001-09-09T01:46:40Z was a Sunday.
        let billennium = epoch + std::time::Duration::from_secs(1_000_000_000);
        assert_eq!(http_date(billennium), "Sun, 09 Sep 2001 01:46:40 GMT");

        // Leap day, because the calendar arithmetic is hand-rolled.
        let leap = epoch + std::time::Duration::from_secs(1_709_164_800);
        assert_eq!(http_date(leap), "Thu, 29 Feb 2024 00:00:00 GMT");
    }

    #[test]
    fn range_parsing_separates_the_three_answers() {
        // `None` = serve the whole object with 200, `Some(None)` = 416,
        // `Some(Some(_))` = 206. Collapsing any two of those is a bug, and one
        // of them was: the header was not parsed at all, so every ranged GET
        // took the 200 path and returned the entire object.
        let ten = 10;

        // Ordinary closed range, inclusive on both ends.
        assert_eq!(parse_range("bytes=2-5", ten), Some(Some((2, 5))));
        // Open ended, and an end past the object: both run to the last byte.
        assert_eq!(parse_range("bytes=7-", ten), Some(Some((7, 9))));
        assert_eq!(parse_range("bytes=0-100", ten), Some(Some((0, 9))));
        // Suffix form counts back from the end.
        assert_eq!(parse_range("bytes=-3", ten), Some(Some((7, 9))));
        // A suffix longer than the object is the whole object, not an error.
        assert_eq!(parse_range("bytes=-99", ten), Some(Some((0, 9))));
        // Single byte, including the last one.
        assert_eq!(parse_range("bytes=9-9", ten), Some(Some((9, 9))));

        // Unsatisfiable: starts past the end, inverted, empty suffix, or any
        // range at all against an empty object.
        assert_eq!(parse_range("bytes=50-60", ten), Some(None));
        assert_eq!(parse_range("bytes=5-2", ten), Some(None));
        assert_eq!(parse_range("bytes=-0", ten), Some(None));
        assert_eq!(parse_range("bytes=0-0", 0), Some(None));

        // Not acted on, so the caller serves the whole object: a unit this
        // server does not implement, a malformed value, and multiple ranges —
        // which real S3 also answers with the entire object rather than a
        // multipart/byteranges body.
        assert_eq!(parse_range("rows=1-2", ten), None);
        assert_eq!(parse_range("bytes=abc", ten), None);
        assert_eq!(parse_range("bytes=1-2,5-6", ten), None);
        assert_eq!(parse_range("", ten), None);
    }

    #[test]
    fn a_multipart_etag_is_the_digest_of_digests_with_a_part_count() {
        // Not the MD5 of the assembled object, which is what a naive
        // implementation produces and what no client expects. The dash and the
        // count are how a client knows not to compare it against the MD5 of
        // what it downloaded.
        let parts = vec![b"aaaa".to_vec(), b"bbbb".to_vec()];
        let etag = multipart_etag(&parts);
        assert!(etag.ends_with("-2"), "{etag}");

        let mut digests = Vec::new();
        digests.extend_from_slice(&md5_digest(b"aaaa"));
        digests.extend_from_slice(&md5_digest(b"bbbb"));
        assert_eq!(etag, format!("{}-2", hex_lower(&md5_digest(&digests))));

        // And it is not the MD5 of the concatenation, which is the mistake.
        assert_ne!(etag, etag_of(b"aaaabbbb"));
    }

    #[test]
    fn the_part_boundaries_change_a_multipart_etag() {
        // Same bytes, different split: S3 gives different ETags, because the
        // digest is taken over the parts. Collapsing them would make the ETag
        // claim the object was uploaded differently than it was.
        let one = multipart_etag(&[b"aaaabbbb".to_vec()]);
        let two = multipart_etag(&[b"aaaa".to_vec(), b"bbbb".to_vec()]);
        assert_ne!(one, two);
        assert!(one.ends_with("-1"));
    }

    #[test]
    fn a_traversing_bucket_name_is_refused() {
        assert!(safe_segment("..").is_err());
        assert!(safe_segment("a/../b").is_err());
        assert!(safe_segment("/etc").is_err());
        assert!(safe_segment("a\\b").is_err());
        assert!(safe_segment("").is_err());
        assert!(safe_segment("normal-bucket").is_ok());
    }

    #[test]
    fn unsupported_subresources_are_refused_rather_than_ignored() {
        // A client that sets an ACL and gets a 200 believes the object is
        // private when it is not.
        for name in ["acl", "versioning", "lifecycle", "tagging"] {
            let mut params = BTreeMap::new();
            params.insert(name.to_string(), String::new());
            let refused = refuse_unsupported(&params);
            assert!(refused.is_err(), "{name} must be refused");
            assert_eq!(refused.err().unwrap().code, "NotImplemented");
        }
        assert!(refuse_unsupported(&BTreeMap::new()).is_ok());
    }

    #[test]
    fn the_timestamp_has_the_shape_s3_uses() {
        // Some clients compare these as strings, so the milliseconds and the
        // trailing Z are not decoration.
        let stamp = iso8601(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_787_000_000));
        assert!(stamp.ends_with(".000Z"), "{stamp}");
        assert_eq!(stamp.len(), 24, "{stamp}");
        assert_eq!(iso8601(std::time::UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn keys_are_reported_with_forward_slashes_on_every_platform() {
        // A Windows server must not hand back backslashes: the key is what the
        // client sent, and it sent slashes.
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("a").join("b")).unwrap();
        std::fs::write(root.path().join("a").join("b").join("c.txt"), b"x").unwrap();
        let mut keys = Vec::new();
        collect_keys(root.path(), root.path(), &mut keys);
        assert_eq!(keys, vec!["a/b/c.txt".to_string()]);
    }
}
