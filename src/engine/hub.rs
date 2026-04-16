use crate::config::Config;
use crate::engine::chunking::ChunkingEngine;
use crate::engine::embeddings::EmbeddingClient;
use crate::engine::Engine;
use crate::sqlite::SqliteService;
use anyhow::Context;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

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
    pub embeddings: EmbeddingClient,
    pub chunking: ChunkingEngine,
    pub schema_queue: mpsc::Sender<(String, String)>,
    config: Config,
    ingest_limit: Arc<Semaphore>,
    hydration_limit: Arc<Semaphore>,
}

#[derive(Clone, serde::Serialize)]
struct RankedDocument {
    id: String,
    score: f32,
    snippets: Vec<String>,
    hit_count: usize,
    score_sum: f32,
    best_chunk_score: f32,
}

#[derive(serde::Serialize)]
struct HydratedResult {
    id: String,
    score: f32,
    snippets: Vec<String>,
    document: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryStrategy {
    SqlFirst,
    VectorFirst,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterApplication {
    None,
    PreVector,
    PostVector,
}

#[derive(Clone, serde::Serialize)]
pub struct QueryPlan {
    pub strategy: QueryStrategy,
    pub reason: &'static str,
    pub filter_application: FilterApplication,
    pub collection_size: usize,
    pub limit: usize,
    pub vector_k: usize,
    pub sql_filter: bool,
    pub estimated_sql_candidates: Option<usize>,
    pub estimated_selectivity: Option<f32>,
}

#[derive(Clone, Default, serde::Serialize)]
pub struct QueryDiagnostics {
    pub planner_ms: u64,
    pub sql_prefilter_ms: u64,
    pub embedding_ms: u64,
    pub vector_search_ms: u64,
    pub hydration_ms: u64,
    pub sql_candidates: usize,
    pub vector_hits: usize,
    pub ranked_docs_before_filter: usize,
    pub ranked_docs_after_filter: usize,
    pub hydrated_docs: usize,
}

#[derive(Clone, serde::Serialize)]
pub struct SearchOutcome {
    pub results: Vec<serde_json::Value>,
    pub plan: QueryPlan,
    pub diagnostics: QueryDiagnostics,
}

enum SqlFilterIds {
    NotRequested,
    Empty,
    Values(HashSet<String>),
}

struct SqlInspection {
    estimated_matches: usize,
    selectivity: Option<f32>,
}

impl LumaDatabase {
    pub fn new(
        engine: Arc<Engine>,
        sqlite: Option<Arc<SqliteService>>,
        embeddings: EmbeddingClient,
        chunking: ChunkingEngine,
        config: Config,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<(String, String)>(10_000);
        let sql_opt = sqlite.clone();
        tokio::spawn(async move {
            let Some(sql) = sql_opt else {
                return;
            };
            while let Some((ns, key)) = rx.recv().await {
                if key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    let index_name = format!("idx_{}_{}", ns, key);
                    let index_sql = format!(
                        "CREATE INDEX IF NOT EXISTS {} ON {}(json_extract(metadata, '$.{}'))",
                        index_name,
                        sql_doc_table(&ns),
                        key
                    );
                    let _ = sql.execute(index_sql, vec![]).await;
                }
            }
        });

        Self {
            engine,
            sqlite,
            embeddings,
            chunking,
            schema_queue: tx,
            ingest_limit: Arc::new(Semaphore::new(config.hub_ingest_max_inflight.max(1))),
            hydration_limit: Arc::new(Semaphore::new(config.hub_hydration_max_inflight.max(1))),
            config,
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
        let _permit = self.ingest_limit.clone().acquire_owned().await?;
        let ingest_start = std::time::Instant::now();
        let doc_key = format!("doc:{}:{}", namespace, doc_id);
        self.engine
            .put_state(doc_key.clone(), raw_json, None, None)
            .context("store source document")?;

        let chunk_start = std::time::Instant::now();
        let chunks = self.chunking.split_text(text);
        self.engine
            .metrics()
            .hybrid_chunking_latency
            .record_us(chunk_start.elapsed().as_micros() as u64);

        if chunks.is_empty() {
            self.engine
                .metrics()
                .ingest_latency
                .record_us(ingest_start.elapsed().as_micros() as u64);
            return Ok(());
        }

        let chunk_strs: Vec<String> = chunks.iter().map(|chunk| chunk.to_string()).collect();
        let embed_start = std::time::Instant::now();
        let vectors = match self.embeddings.embed_batch(&chunk_strs).await {
            Ok(vectors) => vectors,
            Err(err) => {
                self.engine.metrics().inc_embed_failure();
                let _ = self.engine.delete_state(&doc_key);
                return Err(err.context("embed batch"));
            }
        };
        self.engine
            .metrics()
            .embed_latency
            .record_us(embed_start.elapsed().as_micros() as u64);

        if vectors.is_empty() {
            self.engine
                .metrics()
                .ingest_latency
                .record_us(ingest_start.elapsed().as_micros() as u64);
            return Ok(());
        }

        let detected_dim = vectors[0].len();
        self.ensure_collection(namespace, detected_dim)?;
        self.ensure_sqlite_table(namespace).await?;

        let vector_stage = std::time::Instant::now();
        let mut inserted_ids = Vec::with_capacity(chunks.len());
        for (i, (chunk, vector)) in chunks.iter().zip(vectors.into_iter()).enumerate() {
            let chunk_id = format!("{}#{}", doc_id, i);
            let meta = ChunkMetadata {
                parent_id: doc_id.to_string(),
                chunk_index: i,
                text_snippet: chunk.to_string(),
                metadata: metadata.clone(),
            };
            let item = crate::vector::VectorItem {
                vector,
                meta: serde_json::to_value(meta).unwrap_or(serde_json::Value::Null),
                mmap_offset: None,
            };
            if let Err(err) = self.engine.vector_upsert(namespace, &chunk_id, item) {
                self.rollback_ingest(namespace, &doc_key, &inserted_ids);
                return Err(anyhow::anyhow!(
                    "vector insertion failed, rolled back: {}",
                    err
                ));
            }
            inserted_ids.push(chunk_id);
        }
        self.engine
            .metrics()
            .hybrid_vector_write_latency
            .record_us(vector_stage.elapsed().as_micros() as u64);

        let sql_stage = std::time::Instant::now();
        if let Err(err) = self
            .write_sqlite_document(namespace, doc_id, metadata.clone())
            .await
        {
            self.rollback_ingest(namespace, &doc_key, &inserted_ids);
            return Err(err);
        }
        self.engine
            .metrics()
            .hybrid_sql_write_latency
            .record_us(sql_stage.elapsed().as_micros() as u64);

        self.enqueue_metadata_indexes(namespace, metadata.as_ref());
        tracing::info!(
            namespace,
            doc_id,
            chunks = chunks.len(),
            total_ms = ingest_start.elapsed().as_millis() as u64,
            "hybrid ingest completed"
        );
        self.engine
            .metrics()
            .ingest_latency
            .record_us(ingest_start.elapsed().as_micros() as u64);
        Ok(())
    }

    pub async fn search(
        &self,
        namespace: &str,
        query: &str,
        sql_filter: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        Ok(self
            .search_with_plan(namespace, query, sql_filter, limit)
            .await?
            .results)
    }

    pub async fn search_with_plan(
        &self,
        namespace: &str,
        query: &str,
        sql_filter: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<SearchOutcome> {
        let plan_start = std::time::Instant::now();
        let collection_size = self
            .engine
            .vector_collection_info(namespace)
            .map(|info| info.live_count)
            .unwrap_or(0);
        let plan = self
            .plan_query(namespace, sql_filter, limit, collection_size)
            .await?;
        let mut diagnostics = QueryDiagnostics {
            planner_ms: plan_start.elapsed().as_millis() as u64,
            ..QueryDiagnostics::default()
        };

        let ranked = match plan.strategy {
            QueryStrategy::SqlFirst => {
                self.execute_sql_first(namespace, query, sql_filter, &plan, &mut diagnostics)
                    .await?
            }
            QueryStrategy::VectorFirst => {
                self.execute_vector_first(namespace, query, sql_filter, &plan, &mut diagnostics)
                    .await?
            }
        };

        let hydrated = self
            .hydrate_ranked_documents(namespace, ranked, limit, &mut diagnostics)
            .await;
        self.engine.metrics().observe_hybrid_search(
            matches!(plan.strategy, QueryStrategy::SqlFirst),
            diagnostics.sql_candidates,
            diagnostics.vector_hits,
            diagnostics.ranked_docs_after_filter,
            diagnostics.hydrated_docs,
        );
        tracing::info!(
            strategy = ?plan.strategy,
            filter_application = ?plan.filter_application,
            reason = plan.reason,
            collection_size = plan.collection_size,
            sql_candidates = diagnostics.sql_candidates,
            vector_hits = diagnostics.vector_hits,
            ranked_docs_before_filter = diagnostics.ranked_docs_before_filter,
            ranked_docs_after_filter = diagnostics.ranked_docs_after_filter,
            hydrated_docs = diagnostics.hydrated_docs,
            planner_ms = diagnostics.planner_ms,
            sql_prefilter_ms = diagnostics.sql_prefilter_ms,
            embedding_ms = diagnostics.embedding_ms,
            vector_search_ms = diagnostics.vector_search_ms,
            hydration_ms = diagnostics.hydration_ms,
            "hybrid query executed"
        );

        Ok(SearchOutcome {
            results: hydrated
                .into_iter()
                .map(|row| serde_json::to_value(row).unwrap_or(serde_json::Value::Null))
                .collect(),
            plan,
            diagnostics,
        })
    }

    async fn plan_query(
        &self,
        namespace: &str,
        sql_filter: Option<&str>,
        limit: usize,
        collection_size: usize,
    ) -> anyhow::Result<QueryPlan> {
        let has_sql_filter = sql_filter.is_some() && self.sqlite.is_some();
        if !has_sql_filter {
            return Ok(QueryPlan {
                strategy: QueryStrategy::VectorFirst,
                reason: "no_sql_filter",
                filter_application: FilterApplication::None,
                collection_size,
                limit,
                vector_k: limit.max(1),
                sql_filter: false,
                estimated_sql_candidates: None,
                estimated_selectivity: None,
            });
        }

        let inspection = self
            .inspect_sql_filter(namespace, sql_filter.unwrap(), collection_size)
            .await?;
        if inspection.estimated_matches == 0 {
            return Ok(QueryPlan {
                strategy: QueryStrategy::SqlFirst,
                reason: "sql_filter_empty",
                filter_application: FilterApplication::PreVector,
                collection_size,
                limit,
                vector_k: 0,
                sql_filter: true,
                estimated_sql_candidates: Some(0),
                estimated_selectivity: inspection.selectivity,
            });
        }

        let limit = limit.max(1);
        let max_prefilter = self.config.hub_sql_prefilter_max_candidates.max(limit);
        let selective = inspection
            .selectivity
            .is_some_and(|ratio| ratio <= self.config.hub_sql_prefilter_selectivity_threshold);
        let small_prefilter = inspection.estimated_matches <= limit.saturating_mul(4);
        let bounded_prefilter = inspection.estimated_matches <= max_prefilter;

        if bounded_prefilter && (selective || small_prefilter) {
            return Ok(QueryPlan {
                strategy: QueryStrategy::SqlFirst,
                reason: if selective {
                    "selective_sql_filter"
                } else {
                    "small_prefilter_candidate_set"
                },
                filter_application: FilterApplication::PreVector,
                collection_size,
                limit,
                vector_k: limit,
                sql_filter: true,
                estimated_sql_candidates: Some(inspection.estimated_matches),
                estimated_selectivity: inspection.selectivity,
            });
        }

        Ok(QueryPlan {
            strategy: QueryStrategy::VectorFirst,
            reason: if bounded_prefilter {
                "postfilter_for_small_topk"
            } else {
                "prefilter_too_broad"
            },
            filter_application: FilterApplication::PostVector,
            collection_size,
            limit,
            vector_k: expand_vector_first_limit(
                limit,
                collection_size,
                self.config.hub_vector_first_candidate_multiplier,
            ),
            sql_filter: true,
            estimated_sql_candidates: Some(inspection.estimated_matches),
            estimated_selectivity: inspection.selectivity,
        })
    }

    async fn execute_sql_first(
        &self,
        namespace: &str,
        query: &str,
        sql_filter: Option<&str>,
        plan: &QueryPlan,
        diagnostics: &mut QueryDiagnostics,
    ) -> anyhow::Result<Vec<RankedDocument>> {
        if plan.estimated_sql_candidates == Some(0) {
            return Ok(Vec::new());
        }

        let embed_future = async {
            let start = std::time::Instant::now();
            let vector = self.embeddings.embed(query).await;
            (vector, start.elapsed())
        };
        let sql_future = async {
            let start = std::time::Instant::now();
            let ids = self.fetch_allowed_ids_all(namespace, sql_filter).await;
            (ids, start.elapsed())
        };
        let ((query_vector, embed_elapsed), (allowed_ids, sql_elapsed)) =
            tokio::join!(embed_future, sql_future);
        let query_vector = query_vector.inspect_err(|_| {
            self.engine.metrics().inc_embed_failure();
        })?;
        let allowed_ids = allowed_ids?;
        diagnostics.embedding_ms = embed_elapsed.as_millis() as u64;
        diagnostics.sql_prefilter_ms = sql_elapsed.as_millis() as u64;
        self.engine
            .metrics()
            .embed_latency
            .record_us(embed_elapsed.as_micros() as u64);
        self.engine
            .metrics()
            .hybrid_sql_prefilter_latency
            .record_us(sql_elapsed.as_micros() as u64);

        let allowed_ids = match allowed_ids {
            SqlFilterIds::NotRequested => None,
            SqlFilterIds::Empty => return Ok(Vec::new()),
            SqlFilterIds::Values(ids) => {
                diagnostics.sql_candidates = ids.len();
                Some(ids)
            }
        };

        let vector_start = std::time::Instant::now();
        let vector_started = std::time::Instant::now();
        let (hits, vector_stats) = self.engine.vectors().search_with_stats(
            namespace,
            crate::vector::SearchRequest {
                vector: query_vector,
                k: plan.vector_k,
                options: crate::vector::SearchOptions {
                    filters: None,
                    include_meta: true,
                    allowed_ids,
                },
            },
        )?;
        let elapsed_us = vector_started.elapsed().as_micros() as u64;
        self.engine.metrics().observe_vector_search(
            vector_stats.candidate_expansion_steps,
            vector_stats.final_candidate_k,
            vector_stats.recall_estimate,
            elapsed_us,
        );
        diagnostics.vector_search_ms = vector_start.elapsed().as_millis() as u64;
        diagnostics.vector_hits = hits.len();
        self.engine
            .metrics()
            .hybrid_vector_latency
            .record_us(vector_start.elapsed().as_micros() as u64);

        let ranked = collapse_hits(hits);
        diagnostics.ranked_docs_before_filter = ranked.len();
        diagnostics.ranked_docs_after_filter = ranked.len();
        Ok(ranked)
    }

    async fn execute_vector_first(
        &self,
        namespace: &str,
        query: &str,
        sql_filter: Option<&str>,
        plan: &QueryPlan,
        diagnostics: &mut QueryDiagnostics,
    ) -> anyhow::Result<Vec<RankedDocument>> {
        let embed_start = std::time::Instant::now();
        let query_vector = self.embeddings.embed(query).await.inspect_err(|_| {
            self.engine.metrics().inc_embed_failure();
        })?;
        diagnostics.embedding_ms = embed_start.elapsed().as_millis() as u64;
        self.engine
            .metrics()
            .embed_latency
            .record_us(embed_start.elapsed().as_micros() as u64);

        let vector_start = std::time::Instant::now();
        let vector_started = std::time::Instant::now();
        let (hits, vector_stats) = self.engine.vectors().search_with_stats(
            namespace,
            crate::vector::SearchRequest {
                vector: query_vector,
                k: plan.vector_k,
                options: crate::vector::SearchOptions {
                    filters: None,
                    include_meta: true,
                    allowed_ids: None,
                },
            },
        )?;
        let elapsed_us = vector_started.elapsed().as_micros() as u64;
        self.engine.metrics().observe_vector_search(
            vector_stats.candidate_expansion_steps,
            vector_stats.final_candidate_k,
            vector_stats.recall_estimate,
            elapsed_us,
        );
        diagnostics.vector_search_ms = vector_start.elapsed().as_millis() as u64;
        diagnostics.vector_hits = hits.len();
        self.engine
            .metrics()
            .hybrid_vector_latency
            .record_us(vector_start.elapsed().as_micros() as u64);

        let ranked = collapse_hits(hits);
        diagnostics.ranked_docs_before_filter = ranked.len();

        if !plan.sql_filter {
            diagnostics.ranked_docs_after_filter = ranked.len();
            return Ok(ranked);
        }

        let sql_start = std::time::Instant::now();
        let doc_ids: Vec<String> = ranked.iter().map(|doc| doc.id.clone()).collect();
        let allowed_ids = self
            .filter_candidate_ids(namespace, sql_filter.unwrap_or_default(), &doc_ids)
            .await?;
        diagnostics.sql_prefilter_ms = sql_start.elapsed().as_millis() as u64;
        diagnostics.sql_candidates = allowed_ids.len();
        self.engine
            .metrics()
            .hybrid_sql_prefilter_latency
            .record_us(sql_start.elapsed().as_micros() as u64);

        let filtered = ranked
            .into_iter()
            .filter(|doc| allowed_ids.contains(&doc.id))
            .collect::<Vec<_>>();
        diagnostics.ranked_docs_after_filter = filtered.len();
        Ok(filtered)
    }

    async fn hydrate_ranked_documents(
        &self,
        namespace: &str,
        ranked: Vec<RankedDocument>,
        limit: usize,
        diagnostics: &mut QueryDiagnostics,
    ) -> Vec<HydratedResult> {
        let hydration_start = std::time::Instant::now();
        let namespace = namespace.to_string();
        let docs = ranked.into_iter().take(limit).collect::<Vec<_>>();
        diagnostics.hydrated_docs = docs.len();

        let mut tasks = tokio::task::JoinSet::new();
        for doc in docs {
            let permit = self.hydration_limit.clone();
            let engine = self.engine.clone();
            let namespace = namespace.clone();
            tasks.spawn(async move {
                let _permit = permit.acquire_owned().await.ok();
                let key = format!("doc:{}:{}", namespace, doc.id);
                let document = engine.get_state(&key).map(|state| state.value);
                HydratedResult {
                    id: doc.id,
                    score: doc.score,
                    snippets: doc.snippets,
                    document,
                }
            });
        }

        let mut results = Vec::new();
        while let Some(result) = tasks.join_next().await {
            if let Ok(row) = result {
                results.push(row);
            }
        }
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        diagnostics.hydration_ms = hydration_start.elapsed().as_millis() as u64;
        self.engine
            .metrics()
            .hybrid_hydration_latency
            .record_us(hydration_start.elapsed().as_micros() as u64);
        results
    }

    fn ensure_collection(&self, namespace: &str, detected_dim: usize) -> anyhow::Result<()> {
        let collection_exists = self
            .engine
            .list_vector_collections()
            .iter()
            .any(|c| c.collection == namespace);
        if !collection_exists {
            self.engine.create_vector_collection(
                namespace,
                detected_dim,
                crate::vector::Metric::Cosine,
            )?;
        }
        Ok(())
    }

    async fn ensure_sqlite_table(&self, namespace: &str) -> anyhow::Result<()> {
        let Some(sql) = &self.sqlite else {
            return Ok(());
        };
        let create_table = format!(
            "CREATE TABLE IF NOT EXISTS {} (id TEXT PRIMARY KEY, metadata JSON)",
            sql_doc_table(namespace)
        );
        sql.execute(create_table, vec![])
            .await
            .map(|_| ())
            .map_err(|err| anyhow::anyhow!("SQLite create table failed: {}", err))
    }

    async fn write_sqlite_document(
        &self,
        namespace: &str,
        doc_id: &str,
        metadata: Option<serde_json::Value>,
    ) -> anyhow::Result<()> {
        let Some(sql) = &self.sqlite else {
            return Ok(());
        };
        let meta_json = metadata
            .as_ref()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "{}".to_string());
        let insert_sql = format!(
            "INSERT OR REPLACE INTO {} (id, metadata) VALUES (?, ?)",
            sql_doc_table(namespace)
        );
        sql.execute(
            insert_sql,
            vec![
                serde_json::Value::String(doc_id.to_string()),
                serde_json::Value::String(meta_json),
            ],
        )
        .await
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("SQLite insert failed, rolled back: {}", err))
    }

    fn rollback_ingest(&self, namespace: &str, doc_key: &str, inserted_ids: &[String]) {
        let _ = self.engine.delete_state(doc_key);
        for chunk_id in inserted_ids {
            let _ = self.engine.vector_delete(namespace, chunk_id);
        }
    }

    fn enqueue_metadata_indexes(&self, namespace: &str, metadata: Option<&serde_json::Value>) {
        let Some(serde_json::Value::Object(map)) = metadata else {
            return;
        };
        for key in map.keys() {
            let _ = self
                .schema_queue
                .try_send((namespace.to_string(), key.clone()));
        }
    }

    async fn inspect_sql_filter(
        &self,
        namespace: &str,
        filter: &str,
        collection_size: usize,
    ) -> anyhow::Result<SqlInspection> {
        let Some(sql) = &self.sqlite else {
            return Ok(SqlInspection {
                estimated_matches: 0,
                selectivity: None,
            });
        };
        let query = format!(
            "SELECT COUNT(*) AS count FROM {} WHERE {}",
            sql_doc_table(namespace),
            filter
        );
        let estimated_matches = sql
            .estimate_count_cached(format!("count::{namespace}::{filter}"), query, 15_000)
            .await?;
        let selectivity = (collection_size > 0)
            .then_some(estimated_matches as f32 / collection_size.max(1) as f32);
        Ok(SqlInspection {
            estimated_matches,
            selectivity,
        })
    }

    async fn fetch_allowed_ids_all(
        &self,
        namespace: &str,
        sql_filter: Option<&str>,
    ) -> anyhow::Result<SqlFilterIds> {
        let (Some(filter), Some(sql)) = (sql_filter, &self.sqlite) else {
            return Ok(SqlFilterIds::NotRequested);
        };
        let query = format!(
            "SELECT id FROM {} WHERE {}",
            sql_doc_table(namespace),
            filter
        );
        let rows = sql.query(query, vec![]).await?;
        let mut ids = HashSet::new();
        for row in rows {
            if let Some(id) = row.get("id").and_then(|value| value.as_str()) {
                ids.insert(id.to_string());
            }
        }
        if ids.is_empty() {
            Ok(SqlFilterIds::Empty)
        } else {
            Ok(SqlFilterIds::Values(ids))
        }
    }

    async fn filter_candidate_ids(
        &self,
        namespace: &str,
        filter: &str,
        doc_ids: &[String],
    ) -> anyhow::Result<HashSet<String>> {
        let Some(sql) = &self.sqlite else {
            return Ok(HashSet::new());
        };
        if doc_ids.is_empty() {
            return Ok(HashSet::new());
        }

        let mut matched = HashSet::new();
        for chunk in doc_ids.chunks(250) {
            let placeholders = std::iter::repeat_n("?", chunk.len())
                .collect::<Vec<_>>()
                .join(", ");
            let query = format!(
                "SELECT id FROM {} WHERE ({}) AND id IN ({})",
                sql_doc_table(namespace),
                filter,
                placeholders
            );
            let params = chunk
                .iter()
                .cloned()
                .map(serde_json::Value::String)
                .collect::<Vec<_>>();
            for row in sql.query(query, params).await? {
                if let Some(id) = row.get("id").and_then(|value| value.as_str()) {
                    matched.insert(id.to_string());
                }
            }
        }
        Ok(matched)
    }
}

fn collapse_hits(hits: Vec<crate::vector::SearchHit>) -> Vec<RankedDocument> {
    let mut collapsed = HashMap::<String, RankedDocument>::new();
    for hit in hits {
        let Some(meta) = hit.meta else {
            continue;
        };
        let Ok(chunk_meta) = serde_json::from_value::<ChunkMetadata>(meta) else {
            continue;
        };
        let entry = collapsed
            .entry(chunk_meta.parent_id.clone())
            .or_insert_with(|| RankedDocument {
                id: chunk_meta.parent_id.clone(),
                score: hit.score,
                snippets: Vec::new(),
                hit_count: 0,
                score_sum: 0.0,
                best_chunk_score: hit.score,
            });
        entry.hit_count = entry.hit_count.saturating_add(1);
        entry.score_sum += hit.score;
        if hit.score > entry.best_chunk_score {
            entry.best_chunk_score = hit.score;
        }
        let supporting_score = (entry.score_sum - entry.best_chunk_score).max(0.0) * 0.35;
        let coverage_bonus = ((entry.hit_count.saturating_sub(1)) as f32) * 0.05;
        entry.score = entry.best_chunk_score + supporting_score + coverage_bonus;
        if entry.snippets.len() < 3 {
            entry.snippets.push(chunk_meta.text_snippet);
        }
    }

    let mut ranked = collapsed.into_values().collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

fn expand_vector_first_limit(limit: usize, collection_size: usize, multiplier: usize) -> usize {
    let multiplier = multiplier.max(2);
    let floor = limit
        .saturating_mul(multiplier)
        .max(limit.saturating_add(32));
    let ceiling = collection_size.min(limit.saturating_mul(multiplier.saturating_mul(4)).max(256));
    floor.min(ceiling.max(limit))
}

fn sql_doc_table(namespace: &str) -> String {
    format!("docs_{}", sanitize_sql_identifier(namespace))
}

fn sanitize_sql_identifier(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return "namespace".to_string();
    }
    if out.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        out.insert(0, 'n');
        out.insert(1, '_');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        collapse_hits, expand_vector_first_limit, sql_doc_table, ChunkMetadata, FilterApplication,
        QueryPlan, QueryStrategy,
    };

    #[test]
    fn vector_first_expands_candidate_limit() {
        assert_eq!(expand_vector_first_limit(10, 1_000, 8), 80);
        assert_eq!(expand_vector_first_limit(100, 150, 8), 150);
    }

    #[test]
    fn query_plan_serializes_filter_application() {
        let plan = QueryPlan {
            strategy: QueryStrategy::SqlFirst,
            reason: "selective_sql_filter",
            filter_application: FilterApplication::PreVector,
            collection_size: 1_000,
            limit: 10,
            vector_k: 10,
            sql_filter: true,
            estimated_sql_candidates: Some(12),
            estimated_selectivity: Some(0.012),
        };
        let json = serde_json::to_value(plan).unwrap();
        assert_eq!(json["filter_application"], "pre_vector");
    }

    #[test]
    fn sql_doc_table_sanitizes_tenant_scoped_namespaces() {
        assert_eq!(
            sql_doc_table("tenant__tenant-a__shared"),
            "docs_tenant__tenant_a__shared"
        );
        assert_eq!(sql_doc_table("123-demo"), "docs_n_123_demo");
    }

    #[test]
    fn document_ranking_aggregates_multiple_chunk_hits() {
        let hits = vec![
            crate::vector::SearchHit {
                id: "doc-a#0".to_string(),
                score: 0.9,
                meta: Some(
                    serde_json::to_value(ChunkMetadata {
                        parent_id: "doc-a".to_string(),
                        chunk_index: 0,
                        text_snippet: "alpha".to_string(),
                        metadata: None,
                    })
                    .unwrap(),
                ),
            },
            crate::vector::SearchHit {
                id: "doc-a#1".to_string(),
                score: 0.8,
                meta: Some(
                    serde_json::to_value(ChunkMetadata {
                        parent_id: "doc-a".to_string(),
                        chunk_index: 1,
                        text_snippet: "beta".to_string(),
                        metadata: None,
                    })
                    .unwrap(),
                ),
            },
            crate::vector::SearchHit {
                id: "doc-b#0".to_string(),
                score: 0.95,
                meta: Some(
                    serde_json::to_value(ChunkMetadata {
                        parent_id: "doc-b".to_string(),
                        chunk_index: 0,
                        text_snippet: "gamma".to_string(),
                        metadata: None,
                    })
                    .unwrap(),
                ),
            },
        ];
        let ranked = collapse_hits(hits);
        assert_eq!(ranked[0].id, "doc-a");
        assert!(ranked[0].score > ranked[1].score);
    }
}
