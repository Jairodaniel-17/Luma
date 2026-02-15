use crate::engine::traits::BaseEngine;
use crate::engine::{Engine, EngineError};
use crate::sqlite::SqliteService;
use crate::vector::{SearchRequest, VectorError, VectorItem};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
impl BaseEngine for Engine {
    async fn add(&self, collection: &str, id: &str, data: Value) -> Result<()> {
        // Assume data is VectorItem structure: {"vector": [...], "meta": ...}
        let item: VectorItem = serde_json::from_value(data)?;
        self.vector_add(collection, id, item).map_err(Into::into)
    }

    async fn search(&self, collection: &str, query: Value) -> Result<Vec<Value>> {
        // Assume query is SearchRequest
        let req: SearchRequest = serde_json::from_value(query)?;
        let hits = self.vector_search(collection, req)?;
        Ok(serde_json::to_value(hits)?.as_array().unwrap().clone())
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<bool> {
        match self.vector_delete(collection, id) {
            Ok(_) => Ok(true),
            Err(EngineError::Vector(VectorError::IdNotFound)) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    async fn health(&self) -> bool {
        self.health() == "ok"
    }

    async fn stats(&self) -> Value {
        // Metrics text? Or structured?
        // Metrics is unstructured text currently.
        serde_json::json!({"metrics": self.metrics_text()})
    }
}

#[async_trait]
impl BaseEngine for SqliteService {
    async fn add(&self, table: &str, _id: &str, data: Value) -> Result<()> {
        // data is { "columns": [...], "values": [...] } or object?
        // Simple mapping: data is object { "col": val, ... }
        // INSERT INTO table (keys) VALUES (vals)
        let obj = data
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("data must be object"))?;
        let columns: Vec<String> = obj.keys().cloned().collect();
        let placeholders: Vec<String> = columns.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table,
            columns.join(", "),
            placeholders.join(", ")
        );
        let params: Vec<Value> = obj.values().cloned().collect();
        self.execute(sql, params).await?;
        Ok(())
    }

    async fn search(&self, table: &str, query: Value) -> Result<Vec<Value>> {
        // query is { "filter": { "col": val } } => SELECT * FROM table WHERE col = val
        // or raw SQL?
        // Let's support simple filter object
        let obj = query
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("query must be object"))?;
        let mut where_clauses = Vec::new();
        let mut params = Vec::new();
        for (k, v) in obj {
            where_clauses.push(format!("{} = ?", k));
            params.push(v.clone());
        }
        let where_sql = if where_clauses.is_empty() {
            "".to_string()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let sql = format!("SELECT * FROM {} {}", table, where_sql);
        self.query(sql, params).await
    }

    async fn delete(&self, table: &str, id: &str) -> Result<bool> {
        // Assume PK is "id"
        let sql = format!("DELETE FROM {} WHERE id = ?", table);
        let affected = self
            .execute(sql, vec![serde_json::Value::String(id.to_string())])
            .await?;
        Ok(affected > 0)
    }

    async fn health(&self) -> bool {
        self.query("SELECT 1".to_string(), vec![]).await.is_ok()
    }

    async fn stats(&self) -> Value {
        serde_json::json!({"status": "ok"})
    }
}
