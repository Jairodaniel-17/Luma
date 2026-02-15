use crate::engine::Engine;
use crate::search::engine::SearchEngine;
use crate::sqlite::SqliteService;
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub struct MetaEngine {
    core: Arc<Engine>,
    sqlite: Option<Arc<SqliteService>>,
    #[allow(dead_code)]
    search: Arc<SearchEngine>,
}

impl MetaEngine {
    pub fn new(
        core: Arc<Engine>,
        sqlite: Option<Arc<SqliteService>>,
        search: Arc<SearchEngine>,
    ) -> Self {
        Self {
            core,
            sqlite,
            search,
        }
    }

    pub async fn execute(&self, collection: &str, query: Value) -> Result<Vec<Value>> {
        let query_type = query
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("vector");

        match query_type {
            "vector" => self.execute_vector(collection, &query).await,
            "sql" => self.execute_sql(&query).await,
            "hybrid" => self.execute_hybrid(collection, &query).await,
            _ => Err(anyhow::anyhow!("unknown query type: {}", query_type)),
        }
    }

    async fn execute_vector(&self, collection: &str, query: &Value) -> Result<Vec<Value>> {
        // Expect query to be SearchRequest compatible structure
        // But we need to map generic Value to SearchRequest
        let req: crate::vector::SearchRequest = serde_json::from_value(query.clone())?;
        let hits = self.core.vector_search(collection, req)?;
        let mut results = Vec::new();
        for hit in hits {
            results.push(serde_json::to_value(hit)?);
        }
        Ok(results)
    }

    async fn execute_sql(&self, query: &Value) -> Result<Vec<Value>> {
        let Some(svc) = &self.sqlite else {
            return Err(anyhow::anyhow!("sqlite not enabled"));
        };
        // Expect "sql" and optional "params"
        let sql = query
            .get("sql")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing sql field"))?;
        let params = query
            .get("params")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        // Safety: ensure it is SELECT if this is strictly a query method?
        // But MetaEngine might be general.
        // Assuming execute implies read-only for "query" type in this context?
        // Let's assume it calls query()
        svc.query(sql.to_string(), params).await
    }

    async fn execute_hybrid(&self, collection: &str, query: &Value) -> Result<Vec<Value>> {
        // Hybrid: Run Vector Search AND SQL Search (or another source) and merge via RRF.
        // Expect "vector_query": {...}, "sql_query": {...}, "rrf_k": 60

        let vector_query = query
            .get("vector_query")
            .ok_or_else(|| anyhow::anyhow!("missing vector_query"))?;
        let sql_query = query.get("sql_query");

        let vector_results = self.execute_vector(collection, vector_query).await?;

        let mut sql_results = Vec::new();
        if let Some(sq) = sql_query {
            sql_results = self.execute_sql(sq).await?;
        }

        // RRF Merge
        // We need a common ID to merge on.
        // Vector results have "id".
        // SQL results should have "id" column.

        let rrf_k = query.get("rrf_k").and_then(|v| v.as_u64()).unwrap_or(60) as f64;
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut docs: HashMap<String, Value> = HashMap::new();

        // Process Vector Results
        for (rank, doc) in vector_results.iter().enumerate() {
            if let Some(id) = doc.get("id").and_then(|v| v.as_str()) {
                let score = 1.0 / (rrf_k + (rank as f64 + 1.0));
                *scores.entry(id.to_string()).or_default() += score;
                docs.entry(id.to_string()).or_insert(doc.clone());
            }
        }

        // Process SQL Results
        for (rank, doc) in sql_results.iter().enumerate() {
            // SQL results might use "id" or "ID" or similar.
            // We assume "id" field exists.
            if let Some(id) = doc
                .get("id")
                .and_then(|v| v.as_str().or(doc.get("ID").and_then(|v| v.as_str())))
            {
                let score = 1.0 / (rrf_k + (rank as f64 + 1.0));
                *scores.entry(id.to_string()).or_default() += score;
                // Prefer vector doc if exists, or merge?
                // Simple strategy: keep existing or insert
                docs.entry(id.to_string()).or_insert(doc.clone());
            }
        }

        // Sort by RRF score
        let mut merged: Vec<(String, f64)> = scores.into_iter().collect();
        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut final_results = Vec::new();
        for (id, score) in merged {
            if let Some(mut doc) = docs.remove(&id) {
                // Inject RRF score
                if let Some(obj) = doc.as_object_mut() {
                    obj.insert("rrf_score".to_string(), serde_json::Value::from(score));
                }
                final_results.push(doc);
            }
        }

        Ok(final_results)
    }
}
