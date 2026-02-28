use rusqlite::Connection;
use tokio::sync::{mpsc, oneshot};

pub enum SqliteCommand {
    Execute {
        sql: String,
        params: Vec<rusqlite::types::Value>,
        respond_to: oneshot::Sender<anyhow::Result<u64>>,
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
                SqliteCommand::Execute { sql, params, respond_to } => {
                    let result = self.handle_execute(sql, params);
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
        let affected = self.conn.execute(&sql, rusqlite::params_from_iter(params.iter()))?;
        Ok(affected as u64)
    }
}
