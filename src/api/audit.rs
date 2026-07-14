use crate::api::AppState;
use crate::sqlite::SqliteService;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

/// One buffered audit record, flushed to SQLite in batches by a background task.
struct AuditRow {
    ts_ms: i64,
    api_key_id: Option<String>,
    ip: Option<String>,
    method: String,
    path: String,
    status: u16,
    latency_ms: u64,
}

/// How many buffered records to coalesce into a single transaction.
const AUDIT_MAX_BATCH: usize = 256;
/// Bounded buffer; when full, records are dropped rather than blocking requests.
const AUDIT_BUFFER: usize = 10_000;

#[derive(Clone)]
pub struct AuditLog {
    sqlite: Arc<SqliteService>,
    tx: mpsc::Sender<AuditRow>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: i64,
    pub ts_ms: i64,
    pub api_key_id: Option<String>,
    pub ip: Option<String>,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub latency_ms: u64,
}

impl AuditLog {
    pub fn new(sqlite: Arc<SqliteService>) -> Self {
        let (tx, rx) = mpsc::channel(AUDIT_BUFFER);
        // Background flusher: batches buffered records into one transaction per
        // drain, so audit writes never sit on the request path or hammer the
        // single SQLite writer one INSERT at a time.
        tokio::spawn(audit_flush_loop(sqlite.clone(), rx));
        Self { sqlite, tx }
    }

    pub async fn init(&self) -> anyhow::Result<()> {
        self.sqlite
            .execute(
                "CREATE TABLE IF NOT EXISTS sys_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    api_key_id TEXT,
                    ip TEXT,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL
                )"
                .to_string(),
                vec![],
            )
            .await?;
        self.sqlite
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_ts ON sys_audit_log(ts_ms)".to_string(),
                vec![],
            )
            .await?;
        Ok(())
    }

    /// Enqueue an audit record. Non-blocking and best-effort: if the buffer is
    /// full the record is dropped rather than slowing the request. The actual
    /// SQLite write happens in `audit_flush_loop`, off the request path.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        ts_ms: i64,
        api_key_id: Option<&str>,
        ip: Option<&str>,
        method: &str,
        path: &str,
        status: u16,
        latency_ms: u64,
    ) {
        let row = AuditRow {
            ts_ms,
            api_key_id: api_key_id.map(str::to_string),
            ip: ip.map(str::to_string),
            method: method.to_string(),
            path: path.to_string(),
            status,
            latency_ms,
        };
        if self.tx.try_send(row).is_err() {
            // Buffer full (or flusher gone): drop rather than block the request.
            tracing::debug!("audit buffer full; dropping record");
        }
    }

    pub async fn query(
        &self,
        from_ms: Option<i64>,
        to_ms: Option<i64>,
        api_key_id: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut conditions = vec!["1=1".to_string()];
        let mut params: Vec<serde_json::Value> = Vec::new();

        if let Some(from) = from_ms {
            conditions.push("ts_ms >= ?".to_string());
            params.push(json!(from));
        }
        if let Some(to) = to_ms {
            conditions.push("ts_ms <= ?".to_string());
            params.push(json!(to));
        }
        if let Some(key) = api_key_id {
            conditions.push("api_key_id = ?".to_string());
            params.push(json!(key));
        }

        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT id, ts_ms, api_key_id, ip, method, path, status, latency_ms
             FROM sys_audit_log
             WHERE {where_clause}
             ORDER BY ts_ms DESC
             LIMIT {limit}"
        );

        self.sqlite.query(sql, params).await
    }
}

/// Drains buffered audit records and writes them to SQLite in batches (one
/// transaction per drain). Each drain collects exactly what queued while the
/// previous batch was being written, so it self-batches under load and stays a
/// single INSERT under light load — always off the request path.
async fn audit_flush_loop(sqlite: Arc<SqliteService>, mut rx: mpsc::Receiver<AuditRow>) {
    const INSERT: &str = "INSERT INTO sys_audit_log \
         (ts_ms, api_key_id, ip, method, path, status, latency_ms) \
         VALUES (?, ?, ?, ?, ?, ?, ?)";
    loop {
        let Some(first) = rx.recv().await else {
            return; // all senders dropped
        };
        let mut batch = vec![first];
        while batch.len() < AUDIT_MAX_BATCH {
            match rx.try_recv() {
                Ok(row) => batch.push(row),
                Err(_) => break,
            }
        }
        let stmts: Vec<(String, Vec<serde_json::Value>)> = batch
            .into_iter()
            .map(|r| {
                (
                    INSERT.to_string(),
                    vec![
                        json!(r.ts_ms),
                        json!(r.api_key_id),
                        json!(r.ip),
                        json!(r.method),
                        json!(r.path),
                        json!(r.status),
                        json!(r.latency_ms),
                    ],
                )
            })
            .collect();
        if let Err(e) = sqlite.execute_tx(stmts).await {
            tracing::debug!("audit batch flush failed: {e}");
        }
    }
}

/// Axum middleware that records every request to the audit log.
pub async fn audit_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let ip = request
        .headers()
        .get("x-forwarded-for")
        .or_else(|| request.headers().get("x-real-ip"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let api_key_id = request
        .extensions()
        .get::<AuditKeyId>()
        .map(|k| k.0.clone());

    let started = Instant::now();
    let response = next.run(request).await;
    let latency_ms = started.elapsed().as_millis() as u64;
    let status = response.status().as_u16();

    if let Some(log) = &state.audit_log {
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        log.record(
            ts_ms,
            api_key_id.as_deref(),
            ip.as_deref(),
            &method,
            &path,
            status,
            latency_ms,
        );
    }

    response
}

/// Newtype wrapper to pass the resolved API key ID through request extensions.
#[derive(Clone)]
pub struct AuditKeyId(pub String);
