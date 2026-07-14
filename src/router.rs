//! Namespace-sharding router (horizontal write scaling).
//!
//! A router instance forwards each request to a backend Luma node chosen by
//! rendezvous (highest-random-weight) hashing over the request's shard key —
//! the namespace / collection / bucket in the path. Every backend runs the
//! normal single-node engine **unchanged**; shard-local writes therefore scale
//! roughly linearly with the number of nodes, since each node owns a disjoint
//! slice of the keyspace and keeps its own WAL / offset space.
//!
//! Shared metadata (accounts, RBAC, tenant ownership) must live in a common
//! remote libSQL/Turso backend (`libsql_url`) so auth is consistent across
//! nodes; requests that carry no shard key (auth, admin, health, event streams)
//! are sent to a fixed primary node (index 0).
//!
//! Limitations of this first increment (documented, not hidden):
//! - No automatic failover or rebalancing: a downed node makes its shard
//!   unavailable, and changing the node list remaps the keys that hashed to the
//!   added/removed node (rendezvous hashing minimizes this, but data does not
//!   move on its own).
//! - The proxy buffers each request/response body, so streaming endpoints
//!   (SSE `/v1/events`, `/v1/stream`) are not streamed through the router; point
//!   clients at a node directly for those.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderName, Request, Response, StatusCode};
use axum::response::IntoResponse;
use axum::Router;
use std::sync::Arc;

/// Path prefixes whose next segment is the shard key.
const SHARDED_PREFIXES: [&str; 6] = [
    "/v1/db/",
    "/v1/vector/",
    "/v1/doc/",
    "/v1/memory/",
    "/v1/blob/",
    "/v1/image/",
];

#[derive(Clone)]
pub struct RouterState {
    nodes: Arc<Vec<String>>,
    http: reqwest::Client,
    max_body_bytes: usize,
}

impl RouterState {
    pub fn new(nodes: Vec<String>, max_body_bytes: usize, request_timeout_secs: u64) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(request_timeout_secs.max(1)))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            nodes: Arc::new(nodes),
            http,
            max_body_bytes,
        }
    }
}

/// Extract the shard key: the first path segment after a shardable prefix.
/// Returns `None` for non-sharded paths (auth/admin/health/etc).
pub fn shard_key(path: &str) -> Option<&str> {
    for prefix in SHARDED_PREFIXES {
        if let Some(rest) = path.strip_prefix(prefix) {
            let seg = rest.split('/').next().unwrap_or("");
            if !seg.is_empty() {
                return Some(seg);
            }
        }
    }
    None
}

/// Stable weight for (node, key) used by rendezvous hashing. crc32 is fixed
/// across builds/versions, so a namespace always maps to the same node for a
/// given node set — data does not silently move on a version upgrade.
fn hrw_weight(node: &str, key: &str) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(node.as_bytes());
    h.update(b"\x00");
    h.update(key.as_bytes());
    h.finalize()
}

/// Rendezvous (HRW) hashing: pick the node index with the highest weight for
/// `key`. Deterministic; adding/removing a node only remaps keys that hashed to
/// that node, not the whole keyspace. Empty `nodes` is a programmer error
/// (validated at startup) — defaults to 0.
pub fn pick_node(key: &str, nodes: &[String]) -> usize {
    let mut best = 0usize;
    let mut best_w = 0u32;
    for (i, node) in nodes.iter().enumerate() {
        let w = hrw_weight(node, key);
        // Tie-break on index for determinism (crc32 collisions are astronomically rare).
        if i == 0 || w > best_w {
            best_w = w;
            best = i;
        }
    }
    best
}

/// Choose the backend index for a request path: the shard owner, or the primary
/// (node 0) for non-sharded paths.
fn target_index(path: &str, nodes: &[String]) -> usize {
    match shard_key(path) {
        Some(key) => pick_node(key, nodes),
        None => 0,
    }
}

/// Build the router's axum app: every request is proxied to its backend node.
pub fn build_app(state: RouterState) -> Router {
    Router::new().fallback(proxy).with_state(state)
}

async fn proxy(State(state): State<RouterState>, req: Request<Body>) -> Response<Body> {
    let path = req.uri().path().to_string();
    let idx = target_index(&path, &state.nodes);
    let base = state.nodes[idx].trim_end_matches('/');
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(&path);
    let url = format!("{base}{path_and_query}");

    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, state.max_body_bytes).await {
        Ok(b) => b,
        Err(_) => {
            return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response();
        }
    };

    let mut builder = state.http.request(parts.method, &url);
    let host = HeaderName::from_static("host");
    for (name, value) in parts.headers.iter() {
        // Drop Host so reqwest sets it for the upstream; forward everything else
        // (Authorization, Content-Type, X-Forwarded-For, …).
        if name != host {
            builder = builder.header(name, value);
        }
    }
    builder = builder.body(body_bytes.to_vec());

    match builder.send().await {
        Ok(resp) => {
            let status = resp.status();
            let headers = resp.headers().clone();
            let bytes = resp.bytes().await.unwrap_or_default();
            let mut out = Response::builder().status(status);
            for (name, value) in headers.iter() {
                // Skip hop-by-hop headers that don't survive re-framing.
                let n = name.as_str();
                if n.eq_ignore_ascii_case("transfer-encoding")
                    || n.eq_ignore_ascii_case("content-length")
                    || n.eq_ignore_ascii_case("connection")
                {
                    continue;
                }
                out = out.header(name, value);
            }
            out.body(Body::from(bytes))
                .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
        }
        Err(err) => (
            StatusCode::BAD_GATEWAY,
            format!("upstream node {idx} error: {err}"),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::{pick_node, shard_key};

    #[test]
    fn shard_key_extracts_namespace() {
        assert_eq!(shard_key("/v1/db/docs/ingest"), Some("docs"));
        assert_eq!(shard_key("/v1/db/docs/search"), Some("docs"));
        assert_eq!(shard_key("/v1/vector/mycoll/points"), Some("mycoll"));
        assert_eq!(shard_key("/v1/memory/ns1/upsert_fact"), Some("ns1"));
        assert_eq!(shard_key("/v1/blob/bucket-a/key"), Some("bucket-a"));
        // Non-sharded paths → primary.
        assert_eq!(shard_key("/v1/auth/login"), None);
        assert_eq!(shard_key("/v1/admin/stats"), None);
        assert_eq!(shard_key("/v1/health"), None);
        assert_eq!(shard_key("/v1/db/"), None); // missing key
    }

    #[test]
    fn pick_node_is_deterministic_and_in_range() {
        let nodes = vec![
            "http://a:1234".to_string(),
            "http://b:1234".to_string(),
            "http://c:1234".to_string(),
        ];
        for key in ["docs", "users", "ns-42", "bucket"] {
            let a = pick_node(key, &nodes);
            let b = pick_node(key, &nodes);
            assert_eq!(a, b, "same key must map to same node");
            assert!(a < nodes.len());
        }
    }

    #[test]
    fn pick_node_spreads_keys_across_nodes() {
        let nodes = vec![
            "http://a:1234".to_string(),
            "http://b:1234".to_string(),
            "http://c:1234".to_string(),
        ];
        let mut counts = [0usize; 3];
        for i in 0..300 {
            counts[pick_node(&format!("ns-{i}"), &nodes)] += 1;
        }
        // Every node should get a non-trivial share (not perfectly even, but no
        // node should be starved with 300 keys over 3 nodes).
        assert!(counts.iter().all(|&c| c > 40), "uneven: {counts:?}");
    }

    #[test]
    fn removing_a_node_only_remaps_its_own_keys() {
        let full = vec![
            "http://a:1234".to_string(),
            "http://b:1234".to_string(),
            "http://c:1234".to_string(),
        ];
        // Drop node "c" (index 2).
        let reduced = vec!["http://a:1234".to_string(), "http://b:1234".to_string()];
        let mut moved = 0;
        let mut kept = 0;
        for i in 0..500 {
            let key = format!("ns-{i}");
            let before = pick_node(&key, &full);
            let after = pick_node(&key, &reduced);
            if before == 2 {
                // was on the removed node → must move to a or b
                assert!(after < 2);
                moved += 1;
            } else {
                // was on a surviving node → must stay put (HRW stability)
                assert_eq!(before, after, "key {key} moved unnecessarily");
                kept += 1;
            }
        }
        assert!(moved > 0 && kept > 0, "moved={moved} kept={kept}");
    }
}
