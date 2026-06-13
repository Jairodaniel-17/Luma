use parking_lot::Mutex;
use rusqlite::Connection;
use std::sync::Arc;
use tokio::sync::oneshot;

pub struct SqliteReaderPool {
    connections: Arc<Mutex<Vec<Connection>>>,
    path: std::path::PathBuf,
}

impl SqliteReaderPool {
    pub fn new(path: std::path::PathBuf, pool_size: usize) -> anyhow::Result<Self> {
        let mut conns = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let conn = Connection::open(&path)?;
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            conn.busy_timeout(std::time::Duration::from_secs(5))?;
            conns.push(conn);
        }
        Ok(Self {
            connections: Arc::new(Mutex::new(conns)),
            path,
        })
    }

    pub async fn query(
        &self,
        sql: String,
        params: Vec<rusqlite::types::Value>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let pool = self.connections.clone();
        let path = self.path.clone();

        let (tx, rx) = oneshot::channel();

        tokio::task::spawn_blocking(move || {
            let mut conn = None;
            {
                let mut guard = pool.lock();
                if let Some(c) = guard.pop() {
                    conn = Some(c);
                }
            }

            let c = match conn {
                Some(c) => c,
                None => {
                    // Pool is empty, open a temporary connection or block?
                    // Let's just open a temporary one for simplicity and robustness.
                    match Connection::open(&path) {
                        Ok(c) => {
                            let _ = c.pragma_update(None, "journal_mode", "WAL");
                            let _ = c.pragma_update(None, "synchronous", "NORMAL");
                            let _ = c.busy_timeout(std::time::Duration::from_secs(5));
                            c
                        }
                        Err(e) => {
                            let _ = tx.send(Err(anyhow::anyhow!("Failed to open db: {}", e)));
                            return;
                        }
                    }
                }
            };

            let res = (|| -> anyhow::Result<Vec<serde_json::Value>> {
                let mut stmt = c.prepare(&sql)?;
                let columns = stmt
                    .column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
                let mut out = Vec::new();
                while let Some(row) = rows.next()? {
                    out.push(super::row_to_json(row, &columns)?);
                }
                Ok(out)
            })();

            // Return to pool
            {
                let mut guard = pool.lock();
                if guard.len() < 10 {
                    // Max size
                    guard.push(c);
                }
            }

            let _ = tx.send(res);
        });

        rx.await
            .unwrap_or_else(|_| Err(anyhow::anyhow!("spawn_blocking failed")))
    }
}
