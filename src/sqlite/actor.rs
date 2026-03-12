use rusqlite::Connection;
use tokio::sync::{mpsc, oneshot};

pub enum SqliteCommand {
    Execute {
        sql: String,
        params: Vec<rusqlite::types::Value>,
        respond_to: oneshot::Sender<anyhow::Result<u64>>,
    },
    Query {
        sql: String,
        params: Vec<rusqlite::types::Value>,
        respond_to: oneshot::Sender<anyhow::Result<Vec<serde_json::Value>>>,
    },
}

pub struct SqliteActor {
    conn: Connection,
    receiver: mpsc::Receiver<SqliteCommand>,
}

impl SqliteActor {
    pub fn new(conn: Connection, receiver: mpsc::Receiver<SqliteCommand>) -> Self {
        Self { conn, receiver }
    }

    pub fn run(mut self) {
        while let Some(msg) = self.receiver.blocking_recv() {
            match msg {
                SqliteCommand::Execute {
                    sql,
                    params,
                    respond_to,
                } => {
                    let result = self.handle_execute(sql, params);
                    let _ = respond_to.send(result);
                }
                SqliteCommand::Query {
                    sql,
                    params,
                    respond_to,
                } => {
                    let result = self.handle_query(sql, params);
                    let _ = respond_to.send(result);
                }
            }
        }
    }

    fn handle_execute(
        &mut self,
        sql: String,
        params: Vec<rusqlite::types::Value>,
    ) -> anyhow::Result<u64> {
        let affected = self
            .conn
            .prepare_cached(&sql)?
            .execute(rusqlite::params_from_iter(params.iter()))?;
        Ok(affected as u64)
    }

    fn handle_query(
        &mut self,
        sql: String,
        params: Vec<rusqlite::types::Value>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut stmt = self.conn.prepare_cached(&sql)?;
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
    }
}
