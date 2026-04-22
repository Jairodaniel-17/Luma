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

#[derive(Clone)]
pub struct AuditLog {
    sqlite: Arc<SqliteService>,
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
        Self { sqlite }
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

    #[allow(clippy::too_many_arguments)]
    pub async fn record(
        &self,
        ts_ms: i64,
        api_key_id: Option<&str>,
        ip: Option<&str>,
        method: &str,
        path: &str,
        status: u16,
        latency_ms: u64,
    ) {
        let _ = self
            .sqlite
            .execute(
                "INSERT INTO sys_audit_log (ts_ms, api_key_id, ip, method, path, status, latency_ms)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
                    .to_string(),
                vec![
                    json!(ts_ms),
                    json!(api_key_id),
                    json!(ip),
                    json!(method),
                    json!(path),
                    json!(status),
                    json!(latency_ms),
                ],
            )
            .await;
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
        )
        .await;
    }

    response
}

/// Newtype wrapper to pass the resolved API key ID through request extensions.
#[derive(Clone)]
pub struct AuditKeyId(pub String);
