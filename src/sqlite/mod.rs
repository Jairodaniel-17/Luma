use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use rusqlite::types::{Value, ValueRef};
use rusqlite::{params_from_iter, Connection, Row};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct SqliteService {
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl SqliteService {
    pub fn new(db_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&db_path)?;
        conn.pragma_update(None, "journal_mode", &"WAL")?;
        conn.pragma_update(None, "synchronous", &"NORMAL")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            path: db_path,
        })
    }

    pub async fn query(
        &self,
        sql: String,
        params: Vec<serde_json::Value>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|_| anyhow::anyhow!("sqlite lock poisoned"))?;
            let mut stmt = conn.prepare(&sql)?;
            let columns = stmt
                .column_names()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
            let values = json_params_to_values(params)?;
            let mut rows = stmt.query(params_from_iter(values.iter()))?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(row_to_json(row, &columns)?);
            }
            Ok(out)
        })
        .await
        .map_err(|err| anyhow::anyhow!(err))?
    }

    pub async fn execute(
        &self,
        sql: String,
        params: Vec<serde_json::Value>,
    ) -> anyhow::Result<u64> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|_| anyhow::anyhow!("sqlite lock poisoned"))?;
            let values = json_params_to_values(params)?;
            let affected = conn.execute(&sql, params_from_iter(values.iter()))?;
            Ok(affected as u64)
        })
        .await
        .map_err(|err| anyhow::anyhow!(err))?
    }

    pub fn path(&self) -> &Path {
        &self.path
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
