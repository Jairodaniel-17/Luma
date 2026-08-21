use crate::config::Config;
use crate::engine::chunking::ChunkingEngine;
use crate::engine::embeddings::EmbeddingHandle;
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
    pub embeddings: EmbeddingHandle,
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
        embeddings: EmbeddingHandle,
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

        // Rebuild the SQLite doc-filter projection from event-sourced state so a
        // crash/restore can't leave docs_<ns> diverged from the WAL. Additive
        // (INSERT OR REPLACE, never drops rows), so legacy docs that predate
        // projection entries keep working. Runs once, shortly after startup.
        // ponytail: O(docs) reconciliation on every boot; a dirty-marker to skip
        // clean restarts is the upgrade path if startup latency ever matters.
        {
            let engine = engine.clone();
            let sql_opt = sqlite.clone();
            tokio::spawn(async move {
                if let Some(sql) = sql_opt {
                    if let Err(e) = rebuild_sql_projection(&engine, &sql).await {
                        tracing::error!("SQL projection rebuild failed: {}", e);
                    }
                }
            });
        }

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
        // H6: put_state fsyncs the WAL / writes SQLite — run it on the blocking
        // pool so it doesn't stall a Tokio worker. Owned data is moved in.
        {
            let engine = self.engine.clone();
            let put_key = doc_key.clone();
            tokio::task::spawn_blocking(move || engine.put_state(put_key, raw_json, None, None))
                .await
                .context("join put_state task")?
                .context("store source document")?;
        }

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
        // ponytail: H7b — the ingest permit (`_permit`) is held across this await, but
        // embed_batch is now bounded: every provider HTTP call uses a client built with
        // connect + total request timeouts (see embeddings::build_http_client), and the
        // batch makes a finite number of such calls. So the permit is always released
        // promptly and can't be exhausted forever — no extra tokio::time::timeout needed.
        let embedder = self.embeddings.current();
        let vectors = match embedder.embed_batch(&chunk_strs).await {
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

        // SQLite written first: if the process crashes after this point the document
        // metadata row exists but has no vectors — a recoverable state on restart.
        // The inverse (orphaned vectors with no metadata) is harder to detect and
        // reconcile, so we prefer this ordering.
        let sql_stage = std::time::Instant::now();
        if let Err(err) = self
            .write_sqlite_document(namespace, doc_id, metadata.clone())
            .await
        {
            let _ = self.engine.delete_state(&doc_key);
            return Err(err);
        }
        self.engine
            .metrics()
            .hybrid_sql_write_latency
            .record_us(sql_stage.elapsed().as_micros() as u64);

        let vector_stage = std::time::Instant::now();
        // Build the (chunk_id, item) pairs up front so the blocking task owns everything.
        let chunk_items: Vec<(String, crate::vector::VectorItem)> = chunks
            .iter()
            .zip(vectors)
            .enumerate()
            .map(|(i, (chunk, vector))| {
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
                (chunk_id, item)
            })
            .collect();

        // H6: vector_upsert appends to the (fsync'd) WAL — run the whole insert loop on
        // the blocking pool instead of stalling a Tokio worker. On failure it returns the
        // partially-inserted ids so we can roll back (which touches async SQLite) here.
        let upsert_outcome = {
            let engine = self.engine.clone();
            let ns = namespace.to_string();
            tokio::task::spawn_blocking(move || {
                let mut inserted = Vec::with_capacity(chunk_items.len());
                for (chunk_id, item) in chunk_items {
                    if let Err(err) = engine.vector_upsert(&ns, &chunk_id, item) {
                        return Err((err, inserted));
                    }
                    inserted.push(chunk_id);
                }
                Ok(inserted)
            })
            .await
            .context("join vector_upsert task")?
        };
        if let Err((err, inserted_ids)) = upsert_outcome {
            self.rollback_ingest(namespace, &doc_key, &inserted_ids);
            if let Some(sql_svc) = &self.sqlite {
                let table = sql_doc_table(namespace);
                let _ = sql_svc
                    .execute(
                        format!("DELETE FROM {table} WHERE id = ?"),
                        vec![serde_json::Value::String(doc_id.to_string())],
                    )
                    .await;
            }
            return Err(anyhow::anyhow!(
                "vector insertion failed, rolled back: {}",
                err
            ));
        }
        self.engine
            .metrics()
            .hybrid_vector_write_latency
            .record_us(vector_stage.elapsed().as_micros() as u64);

        // Event-source the SQL filter projection. The docs_<ns> row is derived
        // data (id + metadata), so persist it as WAL-backed state keyed by
        // (table, id). On restart the SQLite table is rebuilt from these entries
        // (see LumaDatabase::new), so the filter store can never silently diverge
        // from the WAL after a crash/restore. Written last, only on full success,
        // so an aborted/rolled-back ingest leaves no projection entry behind.
        if self.sqlite.is_some() {
            let table = sql_doc_table(namespace);
            let proj_key = sql_projection_key(&table, doc_id);
            let proj_val = serde_json::json!({
                "table": table,
                "id": doc_id,
                "metadata": metadata.clone().unwrap_or_else(|| serde_json::json!({})),
            });
            let engine = self.engine.clone();
            tokio::task::spawn_blocking(move || engine.put_state(proj_key, proj_val, None, None))
                .await
                .context("join sqlproj put_state task")?
                .context("persist sql projection entry")?;
        }

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
            .inspect_sql_filter(
                namespace,
                sql_filter.expect("sql_filter is checked to be some"),
                collection_size,
            )
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
            let vector = self.embeddings.current().embed(query).await;
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
                    filter: None,
                    min_score: None,
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
        let query_vector = self
            .embeddings
            .current()
            .embed(query)
            .await
            .inspect_err(|_| {
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
                    filter: None,
                    min_score: None,
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
                // H6: get_state reads SQLite in state_db mode — offload to the blocking
                // pool. On join error we fall back to no document rather than panicking.
                let document = tokio::task::spawn_blocking(move || engine.get_state(&key))
                    .await
                    .ok()
                    .flatten()
                    .map(|state| state.value);
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
        // Reject before writing, not after: a dim or model mismatch here would
        // otherwise land vectors that are numerically valid but incomparable
        // with their neighbours, which degrades recall silently instead of
        // failing loudly. Stamps the provenance on first text ingest.
        let embedder = self.embeddings.current();
        self.engine.vectors().check_and_stamp_embedding(
            namespace,
            detected_dim,
            embedder.provider_name(),
            embedder.model_name(),
        )?;
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
        if let Err(e) = self.engine.delete_state(doc_key) {
            tracing::error!("Rollback failed for state doc_key {}: {}", doc_key, e);
        }
        for chunk_id in inserted_ids {
            if let Err(e) = self.engine.vector_delete(namespace, chunk_id) {
                tracing::error!("Rollback failed for vector chunk {}: {}", chunk_id, e);
            }
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
        validate_sql_filter(filter)?;
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
        validate_sql_filter(filter)?;
        let max_ids = self.config.hub_sql_filter_max_ids;
        // Defense in depth (see validate_sql_filter): wrap the user filter in parens and
        // put LIMIT on its own line so a trailing comment can't swallow the row cap.
        let query = format!(
            "SELECT id FROM {} WHERE ({})\nLIMIT {}",
            sql_doc_table(namespace),
            filter,
            max_ids
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
        validate_sql_filter(filter)?;
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

/// State key under which a document's SQL-filter projection row is event-sourced.
/// The value is `{table, id, metadata}`; `rebuild_sql_projection` replays these
/// into the docs_<ns> tables on startup so SQLite can't diverge from the WAL.
fn sql_projection_key(table: &str, doc_id: &str) -> String {
    format!("__sqlproj__:{table}:{doc_id}")
}

/// Whether a table name is a safe, self-generated docs_<ns> identifier. Defends
/// the rebuild against a tampered projection entry before interpolating it.
fn is_safe_doc_table(table: &str) -> bool {
    table.starts_with("docs_")
        && table
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Rebuild the SQLite doc-filter tables from the event-sourced `__sqlproj__:`
/// state entries. Idempotent and additive (CREATE TABLE IF NOT EXISTS +
/// INSERT OR REPLACE); it never drops rows, so documents that predate projection
/// entries are left intact. This makes the SQLite filter store a reconstructible
/// projection of the WAL-backed state rather than a separate, divergeable store.
async fn rebuild_sql_projection(engine: &Engine, sql: &SqliteService) -> anyhow::Result<()> {
    let entries = engine.list_state(Some("__sqlproj__:"), usize::MAX);
    let mut created: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut restored = 0usize;
    for item in &entries {
        let (Some(table), Some(id)) = (
            item.value.get("table").and_then(|v| v.as_str()),
            item.value.get("id").and_then(|v| v.as_str()),
        ) else {
            continue;
        };
        if !is_safe_doc_table(table) {
            tracing::warn!(%table, "skipping projection entry with unsafe table name");
            continue;
        }
        if created.insert(table.to_string()) {
            sql.execute(
                format!("CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, metadata JSON)"),
                vec![],
            )
            .await
            .map_err(|e| anyhow::anyhow!("projection rebuild create table: {e}"))?;
        }
        let meta_str = item
            .value
            .get("metadata")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}))
            .to_string();
        sql.execute(
            format!("INSERT OR REPLACE INTO {table} (id, metadata) VALUES (?, ?)"),
            vec![
                serde_json::Value::String(id.to_string()),
                serde_json::Value::String(meta_str),
            ],
        )
        .await
        .map_err(|e| anyhow::anyhow!("projection rebuild insert: {e}"))?;
        restored += 1;
    }
    if restored > 0 {
        tracing::info!(
            docs = restored,
            tables = created.len(),
            "rebuilt SQLite doc-filter projection from WAL-backed state"
        );
    }
    Ok(())
}

/// Validate a user-supplied sql_filter expression using full AST parsing.
///
/// Wraps the filter in `SELECT 1 FROM __t__ WHERE <filter>` and parses it with
/// sqlparser (SQLite dialect). Rejects:
/// - Statement separators (`;`)
/// - Compound queries at the top level (UNION/INTERSECT/EXCEPT)
/// - Subqueries anywhere in the expression tree (Subquery, EXISTS, IN (SELECT ...))
/// - Dangerous function names (load_extension, readfile, writefile)
fn validate_sql_filter(filter: &str) -> anyhow::Result<()> {
    use sqlparser::ast::{SetExpr, Statement};
    use sqlparser::dialect::SQLiteDialect;
    use sqlparser::parser::Parser;

    if filter.trim().is_empty() {
        anyhow::bail!("sql_filter cannot be empty");
    }
    if filter.contains(';') {
        anyhow::bail!("invalid sql_filter: statement separator ';' is not allowed");
    }
    // Reject SQL comment tokens: a trailing `--` (or a `/* */` block) would otherwise
    // comment out anything appended after the filter — e.g. the `LIMIT` guard in
    // fetch_allowed_ids_all — letting a caller bypass the row cap.
    for token in ["--", "/*", "*/"] {
        if filter.contains(token) {
            anyhow::bail!("invalid sql_filter: SQL comments ('{token}') are not allowed");
        }
    }

    let sql = format!("SELECT 1 FROM __t__ WHERE {filter}");
    let mut stmts = Parser::parse_sql(&SQLiteDialect {}, &sql)
        .map_err(|e| anyhow::anyhow!("invalid sql_filter: parse error: {e}"))?;

    let stmt = stmts
        .pop()
        .ok_or_else(|| anyhow::anyhow!("invalid sql_filter: empty parse result"))?;

    let Statement::Query(query) = stmt else {
        anyhow::bail!("invalid sql_filter: not a valid WHERE expression");
    };

    // Reject UNION / INTERSECT / EXCEPT at the top level
    let SetExpr::Select(select) = query.body.as_ref() else {
        anyhow::bail!("invalid sql_filter: compound queries are not allowed");
    };

    let Some(where_expr) = &select.selection else {
        anyhow::bail!("invalid sql_filter: could not parse as a WHERE expression");
    };

    reject_subqueries_in_filter(where_expr)
}

/// Recursively walk an expression tree and reject any subquery or dangerous function.
fn reject_subqueries_in_filter(expr: &sqlparser::ast::Expr) -> anyhow::Result<()> {
    use sqlparser::ast::Expr;
    match expr {
        Expr::Subquery(_) | Expr::Exists { .. } | Expr::InSubquery { .. } => {
            anyhow::bail!("invalid sql_filter: subqueries are not allowed");
        }
        Expr::Function(f) => {
            let name = f.name.to_string().to_lowercase();
            const BLOCKED_FNS: &[&str] = &["load_extension", "readfile", "writefile"];
            if BLOCKED_FNS.iter().any(|b| name.contains(b)) {
                anyhow::bail!("invalid sql_filter: function '{name}' is not allowed");
            }
            // Belt-and-suspenders: catch nested SELECT in function body string form
            let body = f.to_string().to_lowercase();
            if body.contains("select ") || body.contains("(select") {
                anyhow::bail!(
                    "invalid sql_filter: subqueries in function arguments are not allowed"
                );
            }
            Ok(())
        }
        Expr::BinaryOp { left, right, .. } => {
            reject_subqueries_in_filter(left)?;
            reject_subqueries_in_filter(right)
        }
        Expr::UnaryOp { expr, .. } | Expr::Nested(expr) => reject_subqueries_in_filter(expr),
        Expr::IsNull(e) | Expr::IsNotNull(e) => reject_subqueries_in_filter(e),
        Expr::Between {
            expr, low, high, ..
        } => {
            reject_subqueries_in_filter(expr)?;
            reject_subqueries_in_filter(low)?;
            reject_subqueries_in_filter(high)
        }
        Expr::InList { expr, list, .. } => {
            reject_subqueries_in_filter(expr)?;
            list.iter().try_for_each(reject_subqueries_in_filter)
        }
        Expr::Cast { expr, .. } => reject_subqueries_in_filter(expr),
        _ => Ok(()),
    }
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
        collapse_hits, expand_vector_first_limit, is_safe_doc_table, rebuild_sql_projection,
        sql_doc_table, sql_projection_key, validate_sql_filter, ChunkMetadata, FilterApplication,
        QueryPlan, QueryStrategy,
    };

    #[test]
    fn projection_key_and_table_guard() {
        assert_eq!(
            sql_projection_key("docs_ns", "doc-1"),
            "__sqlproj__:docs_ns:doc-1"
        );
        assert!(is_safe_doc_table("docs_tenant__a__ns"));
        assert!(!is_safe_doc_table("users")); // wrong prefix
        assert!(!is_safe_doc_table("docs_ns; DROP TABLE x")); // injection chars
    }

    // The core of the event-sourcing redesign: the SQLite filter row is derived
    // data, so rebuilding from a WAL-backed `__sqlproj__:` state entry must
    // recreate it even when SQLite has no such row (the post-crash divergence).
    #[tokio::test]
    async fn rebuild_restores_doc_row_from_state_projection() {
        use crate::config::Config;
        use crate::engine::Engine;
        use crate::sqlite::SqliteService;
        use tokio_util::sync::CancellationToken;

        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            data_dir: Some(dir.path().to_str().unwrap().to_string()),
            ..Config::default()
        };
        let engine = Engine::new(config, CancellationToken::new()).unwrap();
        let sql = SqliteService::new(dir.path().join("meta.db")).unwrap();

        // Event-sourced projection entry exists, but SQLite has NO docs table/row
        // (simulates a crash/restore where the filter store fell behind the WAL).
        engine
            .put_state(
                sql_projection_key("docs_ns", "doc-1"),
                serde_json::json!({
                    "table": "docs_ns",
                    "id": "doc-1",
                    "metadata": {"status": "active"}
                }),
                None,
                None,
            )
            .unwrap();

        rebuild_sql_projection(&engine, &sql).await.unwrap();

        let rows = sql
            .query(
                "SELECT metadata FROM docs_ns WHERE id = ?".to_string(),
                vec![serde_json::Value::String("doc-1".to_string())],
            )
            .await
            .unwrap();
        assert_eq!(
            rows.len(),
            1,
            "row should be reconstructed from the WAL projection"
        );
        let meta = rows[0]["metadata"].as_str().unwrap();
        assert!(meta.contains("active"), "metadata preserved: {meta}");
    }

    #[test]
    fn validate_sql_filter_rejects_comment_tokens() {
        // A trailing line comment would otherwise comment out the appended LIMIT.
        let err = validate_sql_filter("status = 'active' --").unwrap_err();
        assert!(err.to_string().contains("comments"));
        assert!(validate_sql_filter("a = 1 /* x */").is_err());
        assert!(validate_sql_filter("a = 1 */").is_err());
    }

    #[test]
    fn validate_sql_filter_accepts_plain_predicate() {
        assert!(validate_sql_filter("json_extract(metadata, '$.status') = 'active'").is_ok());
    }

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
