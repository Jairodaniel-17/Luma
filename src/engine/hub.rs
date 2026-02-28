use crate::engine::Engine;
use crate::sqlite::SqliteService;
use crate::search::engine::SearchEngine;
use crate::engine::embeddings::EmbeddingClient;
use crate::engine::chunking::ChunkingEngine;
use std::sync::Arc;

pub struct LumaDatabase {
    pub engine: Arc<Engine>,
    pub sqlite: Option<Arc<SqliteService>>,
    pub search_engine: Arc<SearchEngine>,
    pub embeddings: EmbeddingClient,
    pub chunking: ChunkingEngine,
}

impl LumaDatabase {
    pub fn new(
        engine: Arc<Engine>,
        sqlite: Option<Arc<SqliteService>>,
        search_engine: Arc<SearchEngine>,
        embeddings: EmbeddingClient,
        chunking: ChunkingEngine,
    ) -> Self {
        Self {
            engine,
            sqlite,
            search_engine,
            embeddings,
            chunking,
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
        // 1. Save original document to DocStore (state store for now)
        let doc_key = format!("doc:{}:{}", namespace, doc_id);
        self.engine.put_state(doc_key, raw_json, None, None)?;

        // 2. Chunking
        let chunks = self.chunking.split_text(text);
        
        if chunks.is_empty() {
            return Ok(()); // Nothing to embed
        }

        // 3. Embeddings & Vector Insert
        // We embed the first chunk to detect the dimension dynamically
        let first_vector: Vec<f32> = self.embeddings.embed(&chunks[0]).await?;
        let detected_dim = first_vector.len() as usize;

        // Auto-create or validate collection
        let collections = self.engine.list_vector_collections();
        let collection_exists = collections.iter().any(|c| c.collection == namespace);
        
        if !collection_exists {
            // Auto create collection with detected dimension
            self.engine.create_vector_collection(namespace, detected_dim, crate::vector::Metric::Cosine)?;
        }

        // Upsert first chunk
        let mut first_meta = metadata.clone().unwrap_or_else(|| serde_json::json!({}));
        first_meta["parent_id"] = serde_json::json!(doc_id);
        first_meta["chunk_index"] = serde_json::json!(0);
        first_meta["text_snippet"] = serde_json::json!(&chunks[0]);
        
        let chunk_0_id = format!("{}#0", doc_id);
        let first_item = crate::vector::VectorItem {
            vector: first_vector,
            meta: first_meta,
        };

        self.engine.vector_upsert(
            namespace,
            &chunk_0_id,
            first_item,
        )?;

        // Process remaining chunks
        for (i, chunk) in chunks.iter().enumerate().skip(1) {
            let vector: Vec<f32> = self.embeddings.embed(chunk).await?;
            let chunk_id = format!("{}#{}", doc_id, i);
            
            let mut meta = metadata.clone().unwrap_or_else(|| serde_json::json!({}));
            meta["parent_id"] = serde_json::json!(doc_id);
            meta["chunk_index"] = serde_json::json!(i);
            meta["text_snippet"] = serde_json::json!(chunk);

            let item = crate::vector::VectorItem {
                vector,
                meta,
            };

            self.engine.vector_upsert(namespace, &chunk_id, item)?;
        }

        // 4. Relational Data (SQLite)
        if let Some(sql) = &self.sqlite {
            let create_table = format!("CREATE TABLE IF NOT EXISTS {}_docs (id TEXT PRIMARY KEY, metadata JSON)", namespace);
            let _ = sql.execute(create_table, vec![]).await?;

            let insert_sql = format!("INSERT OR REPLACE INTO {}_docs (id, metadata) VALUES (?, ?)", namespace);
            let meta_json_str = metadata.unwrap_or_else(|| serde_json::json!({})).to_string();
            let _ = sql.execute(insert_sql, vec![
                serde_json::Value::String(doc_id.to_string()),
                serde_json::Value::String(meta_json_str)
            ]).await?;
        }

        Ok(())
    }

    pub async fn search(
        &self,
        namespace: &str,
        query: &str,
        sql_filter: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        // 1. Relational Pre-filtering (Hard Filter)
        let mut allowed_ids: Option<std::collections::HashSet<String>> = None;
        
        if let (Some(filter_str), Some(sql)) = (sql_filter, &self.sqlite) {
            let query_sql = format!("SELECT id FROM {}_docs WHERE {}", namespace, filter_str);
            
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
                Err(e) => {
                    return Err(anyhow::anyhow!("SQL filter error: {}", e));
                }
            }
        }

        // 2. Text to Vector Conversion
        let query_vector: Vec<f32> = self.embeddings.embed(query).await?;

        // 3. Vector Search
        let req = crate::vector::SearchRequest {
            vector: query_vector,
            k: limit * 3, // Overfetch to account for chunk grouping and pre-filtering
            filters: None, // We do pre-filtering via SQL instead of vector metadata
            include_meta: Some(true),
        };
        
        let hits = self.engine.vector_search(namespace, req)?;

        // 4. Grouping & Collation
        let mut collapsed_results = std::collections::HashMap::new();
        
        for hit in hits {
            if let Some(meta) = hit.meta {
                if let Some(parent_id) = meta.get("parent_id").and_then(|v| v.as_str()) {
                    // Check if it passes our SQL pre-filter
                    if let Some(ref allowed) = allowed_ids {
                        if !allowed.contains(parent_id) {
                            continue;
                        }
                    }

                    // Group by parent_id, keep the highest score and collect snippets
                    let entry = collapsed_results.entry(parent_id.to_string()).or_insert_with(|| {
                        serde_json::json!({
                            "id": parent_id,
                            "score": hit.score,
                            "snippets": [],
                            "document": serde_json::Value::Null
                        })
                    });
                    
                    // Update score if higher
                    if let Some(current_score) = entry["score"].as_f64() {
                        if (hit.score as f64) > current_score {
                            entry["score"] = serde_json::json!(hit.score);
                        }
                    }

                    // Add snippet
                    if let Some(snippet) = meta.get("text_snippet") {
                        if let Some(arr) = entry["snippets"].as_array_mut() {
                            if arr.len() < 3 { // Limit to top 3 snippets per doc
                                arr.push(snippet.clone());
                            }
                        }
                    }
                }
            }
        }

        // Sort by score descending
        let mut final_results: Vec<serde_json::Value> = collapsed_results.into_values().collect();
        final_results.sort_by(|a, b| {
            let score_a = a["score"].as_f64().unwrap_or(0.0);
            let score_b = b["score"].as_f64().unwrap_or(0.0);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // 5. Fetch Full Documents (Hydration)
        final_results.truncate(limit);
        for result in final_results.iter_mut() {
            if let Some(doc_id) = result["id"].as_str() {
                let doc_key = format!("doc:{}:{}", namespace, doc_id);
                if let Some(state_item) = self.engine.get_state(&doc_key) {
                    result["document"] = state_item.value;
                }
            }
        }

        Ok(final_results)
    }
}
