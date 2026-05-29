use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use rusqlite::types::{Value, ValueRef};
use rusqlite::{Connection, Row};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

mod actor;
pub mod memory_schema;
mod pool;
use actor::{SqliteActor, SqliteCommand};
use pool::SqliteReaderPool;

#[derive(Clone)]
pub struct SqliteService {
    sender: mpsc::Sender<SqliteCommand>,
    reader_pool: Arc<SqliteReaderPool>,
    path: PathBuf,
    planner_stats: Arc<Mutex<HashMap<String, CachedCount>>>,
}

#[derive(Clone, Copy)]
struct CachedCount {
    count: usize,
    expires_at_ms: u64,
}

impl SqliteService {
    pub fn new(db_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&db_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        let (sender, receiver) = mpsc::channel(1000);
        let actor = SqliteActor::new(conn, receiver);

        std::thread::spawn(move || {
            actor.run();
        });

        // Concurrency improvement: 10 readers
        let reader_pool = Arc::new(SqliteReaderPool::new(db_path.clone(), 10)?);

        Ok(Self {
            sender,
            reader_pool,
            path: db_path,
            planner_stats: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub async fn query(
        &self,
        sql: String,
        params: Vec<serde_json::Value>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let values = json_params_to_values(params)?;
        self.reader_pool.query(sql, values).await
    }

    pub async fn execute(
        &self,
        sql: String,
        params: Vec<serde_json::Value>,
    ) -> anyhow::Result<u64> {
        let (respond_to, receiver) = oneshot::channel();
        let values = json_params_to_values(params)?;

        let msg = SqliteCommand::Execute {
            sql,
            params: values,
            respond_to,
        };

        if self.sender.send(msg).await.is_err() {
            return Err(anyhow::anyhow!("sqlite actor channel closed"));
        }

        receiver
            .await
            .map_err(|_| anyhow::anyhow!("sqlite actor dropped response channel"))?
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn estimate_count_cached(
        &self,
        cache_key: String,
        sql: String,
        ttl_ms: u64,
    ) -> anyhow::Result<usize> {
        let now = now_ms();
        if let Some(cached) = self.planner_stats.lock().unwrap().get(&cache_key).copied() {
            if cached.expires_at_ms > now {
                return Ok(cached.count);
            }
        }
        let rows = self.query(sql, vec![]).await?;
        let count = rows
            .first()
            .and_then(|row| row.get("count"))
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_i64().map(|v| v.max(0) as u64))
            })
            .unwrap_or(0) as usize;
        self.planner_stats.lock().unwrap().insert(
            cache_key,
            CachedCount {
                count,
                expires_at_ms: now.saturating_add(ttl_ms.max(1)),
            },
        );
        Ok(count)
    }
}

fn json_params_to_values(params: Vec<serde_json::Value>) -> anyhow::Result<Vec<Value>> {
    let mut out = Vec::new();
    for value in params {
        let sql_value = match value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Integer(if b { 1 } else { 0 }),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Real(f)
                } else {
                    Value::Null
                }
            }
            serde_json::Value::String(s) => Value::Text(s),
            _ => {
                return Err(anyhow::anyhow!(
                    "unsupported parameter type (only null, bool, number, string)"
                ));
            }
        };
        out.push(sql_value);
    }
    Ok(out)
}

fn row_to_json(row: &Row<'_>, columns: &[String]) -> anyhow::Result<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    for (idx, name) in columns.iter().enumerate() {
        let value = row.get_ref(idx)?;
        obj.insert(name.clone(), sqlite_value_to_json(value)?);
    }
    Ok(serde_json::Value::Object(obj))
}

fn sqlite_value_to_json(value: ValueRef<'_>) -> anyhow::Result<serde_json::Value> {
    Ok(match value {
        ValueRef::Null => serde_json::Value::Null,
        ValueRef::Integer(i) => serde_json::json!(i),
        ValueRef::Real(f) => match serde_json::Number::from_f64(f) {
            Some(num) => serde_json::Value::Number(num),
            None => serde_json::Value::Null,
        },
        ValueRef::Text(t) => serde_json::Value::String(String::from_utf8_lossy(t).to_string()),
        ValueRef::Blob(b) => serde_json::Value::String(STANDARD_NO_PAD.encode(b)),
    })
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
