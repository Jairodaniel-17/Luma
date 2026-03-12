use crate::engine::chunking::ChunkingEngine;
use crate::engine::embeddings::EmbeddingClient;
use crate::engine::Engine;
use crate::search::engine::SearchEngine;
use crate::sqlite::SqliteService;
use std::sync::Arc;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ChunkMetadata {
    pub parent_id: String,
    pub chunk_index: usize,
    pub text_snippet: String,
    #[serde(flatten)]
    pub metadata: Option<serde_json::Value>,
}

pub struct LumaDatabase {
    pub engine: Arc<Engine>,
    pub sqlite: Option<Arc<SqliteService>>,
    pub search_engine: Arc<SearchEngine>,
    pub embeddings: EmbeddingClient,
    pub chunking: ChunkingEngine,
    pub schema_queue: tokio::sync::mpsc::Sender<(String, String)>,
}

#[derive(serde::Serialize)]
struct HydratedResult {
    id: String,
    score: f32,
    snippets: Vec<String>,
    document: Option<serde_json::Value>,
}

impl LumaDatabase {
    pub fn new(
        engine: Arc<Engine>,
        sqlite: Option<Arc<SqliteService>>,
        search_engine: Arc<SearchEngine>,
        embeddings: EmbeddingClient,
        chunking: ChunkingEngine,
    ) -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, String)>(10000);
        let sql_opt = sqlite.clone();
        tokio::spawn(async move {
            let Some(sql) = sql_opt else {
                return;
            };
            while let Some((ns, key)) = rx.recv().await {
                if key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    let index_name = format!("idx_{}_{}", ns, key);
                    let index_sql = format!(
                        "CREATE INDEX IF NOT EXISTS {} ON {}_docs(json_extract(metadata, '$.{}'))",
                        index_name, ns, key
                    );
                    let _ = sql.execute(index_sql, vec![]).await;
                }
            }
        });

        Self {
            engine,
            sqlite,
            search_engine,
            embeddings,
            chunking,
            schema_queue: tx,
        }
    }

    pub async fn ingest_document(
        &self,
        namespace: &str,
        doc_id: &str,
        text: &str,
        raw_json: serde_json::Value,
        metadata: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        let ingest_start = std::time::Instant::now();

        // 1. Save original document to DocStore (state store for now)
        let doc_key = format!("doc:{}:{}", namespace, doc_id);
        self.engine
            .put_state(doc_key.clone(), raw_json, None, None)?;

        // 2. Chunking
        let chunks = self.chunking.split_text(text);

        if chunks.is_empty() {
            return Ok(()); // Nothing to embed
        }

        // 3. Embeddings — PR3: batch all chunks in a single call
        let chunk_strs: Vec<String> = chunks.iter().map(|c| c.to_string()).collect();
        let embed_start = std::time::Instant::now();
        let vectors = self.embeddings.embed_batch(&chunk_strs).await?;
        let embed_us = embed_start.elapsed().as_micros() as u64;
        self.engine.metrics().embed_latency.record_us(embed_us);

        if vectors.is_empty() {
            return Ok(());
        }

        let detected_dim = vectors[0].len();

        // Auto-create or validate collection
        let collections = self.engine.list_vector_collections();
        let collection_exists = collections.iter().any(|c| c.collection == namespace);

        if !collection_exists {
            // Auto create collection with detected dimension
            self.engine.create_vector_collection(
                namespace,
                detected_dim,
                crate::vector::Metric::Cosine,
            )?;
        }

        // Explicit Consistency Model: Strategy C - Compensating Actions.
        // We write in order: KV State -> Vector Engine -> SQLite.
        // If Vector Engine fails, we rollback KV State.
        // If SQLite fails, we rollback Vector Engine and KV State.
        for (i, (chunk, vector)) in chunks.iter().zip(vectors.into_iter()).enumerate() {
            let chunk_id = format!("{}#{}", doc_id, i);
            let meta_struct = ChunkMetadata {
                parent_id: doc_id.to_string(),
                chunk_index: i,
                text_snippet: chunk.to_string(),
                metadata: metadata.clone(),
            };

            let item = crate::vector::VectorItem {
                vector,
                meta: serde_json::to_value(meta_struct).unwrap_or(serde_json::Value::Null),
                mmap_offset: None,
            };

            if let Err(e) = self.engine.vector_upsert(namespace, &chunk_id, item) {
                // Compensation: Rollback state and previously inserted vectors
                let _ = self.engine.delete_state(&doc_key);
                for j in 0..i {
                    let rollback_id = format!("{}#{}", doc_id, j);
                    let _ = self.engine.vector_delete(namespace, &rollback_id);
                }
                return Err(anyhow::anyhow!(
                    "Vector insertion failed, rolled back: {}",
                    e
                ));
            }
        }

        // 4. Relational Data (SQLite)
        if let Some(sql) = &self.sqlite {
            let create_table = format!(
                "CREATE TABLE IF NOT EXISTS {}_docs (id TEXT PRIMARY KEY, metadata JSON)",
                namespace
            );
            if let Err(e) = sql.execute(create_table, vec![]).await {
                let _ = self.engine.delete_state(&doc_key);
                for i in 0..chunks.len() {
                    let _ = self
                        .engine
                        .vector_delete(namespace, &format!("{}#{}", doc_id, i));
                }
                return Err(anyhow::anyhow!(
                    "SQLite create table failed, rolled back: {}",
                    e
                ));
            }

            let meta_json_str = metadata
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "{}".to_string());
            let insert_sql = format!(
                "INSERT OR REPLACE INTO {}_docs (id, metadata) VALUES (?, ?)",
                namespace
            );
            if let Err(e) = sql
                .execute(
                    insert_sql,
                    vec![
                        serde_json::Value::String(doc_id.to_string()),
                        serde_json::Value::String(meta_json_str),
                    ],
                )
                .await
            {
                // Compensation
                let _ = self.engine.delete_state(&doc_key);
                for i in 0..chunks.len() {
                    let _ = self
                        .engine
                        .vector_delete(namespace, &format!("{}#{}", doc_id, i));
                }
                return Err(anyhow::anyhow!("SQLite insert failed, rolled back: {}", e));
            }

            if let Some(serde_json::Value::Object(map)) = metadata {
                let ns_clone = namespace.to_string();
                for key in map.keys() {
                    let _ = self.schema_queue.try_send((ns_clone.clone(), key.clone()));
                }
            }
        }

        let ingest_us = ingest_start.elapsed().as_micros() as u64;
        self.engine.metrics().ingest_latency.record_us(ingest_us);

        Ok(())
    }

    pub async fn search(
        &self,
        namespace: &str,
        query: &str,
        sql_filter: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let search_start = std::time::Instant::now();

        // PR7: Query planner heuristics
        // Use collection_size cached from live_count (no COUNT(*) query needed)
        let collection_size = self
            .engine
            .vector_collection_info(namespace)
            .map(|info| info.live_count)
            .unwrap_or(0);

        let has_sql_filter = sql_filter.is_some() && self.sqlite.is_some();
        let (strategy, _reason) = choose_strategy(limit, collection_size, has_sql_filter);

        let mut allowed_ids: Option<std::collections::HashSet<String>> = None;

        match strategy {
            "sql_first" => {
                // 1a. SQL pre-filter first (reduces vector search space)
                if let (Some(filter_str), Some(sql)) = (sql_filter, &self.sqlite) {
                    let query_sql =
                        format!("SELECT id FROM {}_docs WHERE {}", namespace, filter_str);
                    match sql.query(query_sql, vec![]).await {
                        Ok(rows) => {
                            let mut ids = std::collections::HashSet::new();
                            for row in rows {
                                if let Some(id_str) = row.get("id").and_then(|v| v.as_str()) {
                                    ids.insert(id_str.to_string());
                                }
                            }
                            if ids.is_empty() {
                                return Ok(Vec::new());
                            }
                            allowed_ids = Some(ids);
                        }
                        Err(e) => return Err(anyhow::anyhow!("SQL filter error: {}", e)),
                    }
                }
            }
            _ => {
                // "vector_first": run SQL filter concurrently or after (for collation)
                if let (Some(filter_str), Some(sql)) = (sql_filter, &self.sqlite) {
                    let query_sql =
                        format!("SELECT id FROM {}_docs WHERE {}", namespace, filter_str);
                    match sql.query(query_sql, vec![]).await {
                        Ok(rows) => {
                            let mut ids = std::collections::HashSet::new();
                            for row in rows {
                                if let Some(id_str) = row.get("id").and_then(|v| v.as_str()) {
                                    ids.insert(id_str.to_string());
                                }
                            }
                            if ids.is_empty() {
                                return Ok(Vec::new());
                            }
                            allowed_ids = Some(ids);
                        }
                        Err(e) => return Err(anyhow::anyhow!("SQL filter error: {}", e)),
                    }
                }
            }
        }

        // 2. Text to Vector Conversion
        let query_vector: Vec<f32> = self.embeddings.embed(query).await?;

        // 3. Vector Search
        let req = crate::vector::SearchRequest {
            vector: query_vector,
            k: limit,
            options: crate::vector::SearchOptions {
                filters: None,
                include_meta: true,
                allowed_ids: allowed_ids.clone(),
            },
        };

        let hits = self.engine.vector_search(namespace, req)?;

        // 4. Grouping & Collation
        let mut collapsed_results: std::collections::HashMap<String, HydratedResult> =
            std::collections::HashMap::new();

        for hit in hits {
            if let Some(meta) = hit.meta {
                if let Ok(chunk_meta) = serde_json::from_value::<ChunkMetadata>(meta) {
                    let parent_id = chunk_meta.parent_id;

                    if let Some(ref allowed) = allowed_ids {
                        if !allowed.contains(&parent_id) {
                            continue;
                        }
                    }

                    let entry = collapsed_results
                        .entry(parent_id.clone())
                        .or_insert_with(|| HydratedResult {
                            id: parent_id,
                            score: hit.score,
                            snippets: Vec::new(),
                            document: None,
                        });

                    if hit.score > entry.score {
                        entry.score = hit.score;
                    }

                    if entry.snippets.len() < 3 {
                        entry.snippets.push(chunk_meta.text_snippet);
                    }
                }
            }
        }

        // Sort by score descending
        let mut final_results: Vec<HydratedResult> = collapsed_results.into_values().collect();
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 5. Fetch Full Documents (Hydration)
        final_results.truncate(limit);
        for result in final_results.iter_mut() {
            let doc_key = format!("doc:{}:{}", namespace, result.id);
            if let Some(state_item) = self.engine.get_state(&doc_key) {
                result.document = Some(state_item.value);
            }
        }

        let search_us = search_start.elapsed().as_micros() as u64;
        self.engine.metrics().search_latency.record_us(search_us);

        Ok(final_results
            .into_iter()
            .map(|r| serde_json::to_value(r).unwrap_or(serde_json::Value::Null))
            .collect())
    }
}

/// PR7: Simple heuristic-based query planner.
///
/// Returns (strategy, reason):
/// - `"sql_first"`: run SQL filter before vector search (reduces search space for large collections with selective filters)
/// - `"vector_first"`: run vector search directly (better for small k relative to collection, or no filter)
fn choose_strategy(
    limit: usize,
    collection_size: usize,
    has_sql_filter: bool,
) -> (&'static str, &'static str) {
    if !has_sql_filter {
        return ("vector_first", "no_sql_filter");
    }
    // If collection is large and we have a filter, pre-filter with SQL first
    if collection_size > 100_000 {
        return ("sql_first", "large_collection_with_filter");
    }
    // If limit is a very small fraction of the collection, vector_first is better
    if collection_size > 0 {
        let ratio = limit as f64 / collection_size as f64;
        if ratio < 0.01 {
            return ("vector_first", "small_k_ratio");
        }
    }
    ("sql_first", "has_filter")
}

#[cfg(test)]
mod tests {
    use super::choose_strategy;

    #[test]
    fn choose_strategy_without_filter_stays_vector_first() {
        assert_eq!(
            choose_strategy(10, 1_000, false),
            ("vector_first", "no_sql_filter")
        );
    }

    #[test]
    fn choose_strategy_large_filtered_collection_prefers_sql_first() {
        assert_eq!(
            choose_strategy(10, 150_000, true),
            ("sql_first", "large_collection_with_filter")
        );
    }
}
