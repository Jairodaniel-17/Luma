use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// Histogram bucket upper bounds in milliseconds
const BUCKET_BOUNDS_MS: &[u64] = &[1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];
const NUM_BUCKETS: usize = 13; // 12 bounds + 1 overflow (+Inf)

pub struct LatencyHistogram {
    /// counts[i] = events that fell in bucket i (i.e., ms <= BUCKET_BOUNDS_MS[i] and > previous)
    /// counts[NUM_BUCKETS-1] = overflow (above last bound)
    counts: [AtomicU64; NUM_BUCKETS],
    total_count: AtomicU64,
    /// Sum in microseconds for precision (divide by 1000 for ms output)
    total_sum_us: AtomicU64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self {
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
            total_count: AtomicU64::new(0),
            total_sum_us: AtomicU64::new(0),
        }
    }
}

impl LatencyHistogram {
    pub fn record_us(&self, us: u64) {
        let ms = us / 1000;
        // Find first bucket where bound >= ms (i.e., ms fits in that bucket)
        let bucket = BUCKET_BOUNDS_MS
            .iter()
            .position(|&b| b >= ms)
            .unwrap_or(NUM_BUCKETS - 1);
        self.counts[bucket.min(NUM_BUCKETS - 1)].fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Compute approximate percentile (0–100) in milliseconds.
    pub fn percentile_ms(&self, p: f64) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let target = (total as f64 * p / 100.0).ceil() as u64;
        let mut running = 0u64;
        for (i, &bound_ms) in BUCKET_BOUNDS_MS.iter().enumerate() {
            running += self.counts[i].load(Ordering::Relaxed);
            if running >= target {
                return bound_ms as f64;
            }
        }
        // In overflow bucket
        running += self.counts[NUM_BUCKETS - 1].load(Ordering::Relaxed);
        if running >= target {
            return (BUCKET_BOUNDS_MS.last().copied().unwrap_or(5000) * 2) as f64;
        }
        0.0
    }

    /// Emit Prometheus histogram format (cumulative buckets).
    pub fn render_prometheus(&self, name: &str, out: &mut String) {
        let total = self.total_count.load(Ordering::Relaxed);
        let sum_us = self.total_sum_us.load(Ordering::Relaxed);

        out.push_str(&format!("# TYPE {name} histogram\n"));
        let mut running = 0u64;
        for (i, &bound_ms) in BUCKET_BOUNDS_MS.iter().enumerate() {
            running += self.counts[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound_ms}\"}} {running}\n"));
        }
        // +Inf bucket (cumulative total)
        running += self.counts[NUM_BUCKETS - 1].load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {running}\n"));
        out.push_str(&format!("{name}_sum {:.3}\n", sum_us as f64 / 1000.0));
        out.push_str(&format!("{name}_count {total}\n"));
    }
}

#[derive(Default)]
pub struct Metrics {
    state_put_total: AtomicU64,
    state_delete_total: AtomicU64,
    vector_ops_total: AtomicU64,
    events_total: AtomicU64,
    sse_clients: AtomicU64,

    // PR1: latency histograms
    pub search_latency: LatencyHistogram,
    pub ingest_latency: LatencyHistogram,
    pub embed_latency: LatencyHistogram,
    pub hybrid_sql_prefilter_latency: LatencyHistogram,
    pub hybrid_vector_latency: LatencyHistogram,
    pub hybrid_hydration_latency: LatencyHistogram,
    pub hybrid_chunking_latency: LatencyHistogram,
    pub hybrid_vector_write_latency: LatencyHistogram,
    pub hybrid_sql_write_latency: LatencyHistogram,
    pub vector_ann_latency: LatencyHistogram,

    // PR2: embedding cache counters
    embed_cache_hits: AtomicU64,
    embed_cache_misses: AtomicU64,
    embed_failures_total: AtomicU64,
    hybrid_sql_first_total: AtomicU64,
    hybrid_vector_first_total: AtomicU64,
    hybrid_last_sql_candidates: AtomicU64,
    hybrid_last_vector_candidates: AtomicU64,
    hybrid_last_doc_candidates: AtomicU64,
    hybrid_last_hydrated_docs: AtomicU64,
    hybrid_sql_candidates_total: AtomicU64,
    hybrid_vector_candidates_total: AtomicU64,
    hybrid_doc_candidates_total: AtomicU64,
    hybrid_hydrated_docs_total: AtomicU64,
    vector_candidate_expansion_total: AtomicU64,
    vector_search_total: AtomicU64,
    vector_last_candidate_expansion_steps: AtomicU64,
    vector_last_final_candidate_k: AtomicU64,
    vector_last_recall_estimate_milli: AtomicU64,
    wal_replay_applied_total: AtomicU64,
    wal_replay_duplicates_total: AtomicU64,
    wal_replay_corrupt_total: AtomicU64,
    wal_gap_total: AtomicU64,
    wal_rotation_total: AtomicU64,
    snapshot_total: AtomicU64,
    snapshot_failed_total: AtomicU64,
    ttl_expired_total: AtomicU64,
    hnsw_compaction_total: AtomicU64,
    last_metrics_render: Mutex<String>,
}

#[derive(Clone, Debug, Default)]
pub struct VectorMetricsSnapshot {
    pub collection: String,
    pub live_count: u64,
    pub segments: u64,
    pub deleted_count: u64,
    pub fragmentation_score: f64,
}

impl Metrics {
    pub fn inc_state_put(&self) {
        self.state_put_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_state_delete(&self) {
        self.state_delete_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_vector_op(&self) {
        self.vector_ops_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_events(&self) {
        self.events_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_sse_clients(&self) {
        self.sse_clients.fetch_add(1, Ordering::Relaxed);
    }
    pub fn dec_sse_clients(&self) {
        self.sse_clients.fetch_sub(1, Ordering::Relaxed);
    }
    pub fn inc_embed_cache_hit(&self) {
        self.embed_cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_embed_cache_miss(&self) {
        self.embed_cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_embed_failure(&self) {
        self.embed_failures_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn observe_hybrid_search(
        &self,
        sql_first: bool,
        sql_candidates: usize,
        vector_candidates: usize,
        doc_candidates: usize,
        hydrated_docs: usize,
    ) {
        if sql_first {
            self.hybrid_sql_first_total.fetch_add(1, Ordering::Relaxed);
        } else {
            self.hybrid_vector_first_total
                .fetch_add(1, Ordering::Relaxed);
        }
        self.hybrid_last_sql_candidates
            .store(sql_candidates as u64, Ordering::Relaxed);
        self.hybrid_last_vector_candidates
            .store(vector_candidates as u64, Ordering::Relaxed);
        self.hybrid_last_doc_candidates
            .store(doc_candidates as u64, Ordering::Relaxed);
        self.hybrid_last_hydrated_docs
            .store(hydrated_docs as u64, Ordering::Relaxed);
        self.hybrid_sql_candidates_total
            .fetch_add(sql_candidates as u64, Ordering::Relaxed);
        self.hybrid_vector_candidates_total
            .fetch_add(vector_candidates as u64, Ordering::Relaxed);
        self.hybrid_doc_candidates_total
            .fetch_add(doc_candidates as u64, Ordering::Relaxed);
        self.hybrid_hydrated_docs_total
            .fetch_add(hydrated_docs as u64, Ordering::Relaxed);
    }
    pub fn observe_vector_search(
        &self,
        expansion_steps: usize,
        final_candidate_k: usize,
        recall_estimate: f32,
        elapsed_us: u64,
    ) {
        self.vector_search_total.fetch_add(1, Ordering::Relaxed);
        self.vector_candidate_expansion_total
            .fetch_add(expansion_steps as u64, Ordering::Relaxed);
        self.vector_last_candidate_expansion_steps
            .store(expansion_steps as u64, Ordering::Relaxed);
        self.vector_last_final_candidate_k
            .store(final_candidate_k as u64, Ordering::Relaxed);
        self.vector_last_recall_estimate_milli.store(
            (recall_estimate.clamp(0.0, 1.0) * 1000.0) as u64,
            Ordering::Relaxed,
        );
        self.vector_ann_latency.record_us(elapsed_us);
    }
    pub fn observe_replay(&self, applied: u64, duplicates: u64, corrupt: u64, gap: bool) {
        self.wal_replay_applied_total
            .fetch_add(applied, Ordering::Relaxed);
        self.wal_replay_duplicates_total
            .fetch_add(duplicates, Ordering::Relaxed);
        self.wal_replay_corrupt_total
            .fetch_add(corrupt, Ordering::Relaxed);
        if gap {
            self.wal_gap_total.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn inc_wal_gap(&self) {
        self.wal_gap_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_wal_rotation(&self) {
        self.wal_rotation_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_snapshot_ok(&self) {
        self.snapshot_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn inc_snapshot_failed(&self) {
        self.snapshot_failed_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn add_ttl_expired(&self, count: u64) {
        self.ttl_expired_total.fetch_add(count, Ordering::Relaxed);
    }
    pub fn inc_hnsw_compaction(&self) {
        self.hnsw_compaction_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn cached_render(&self) -> String {
        self.last_metrics_render.lock().clone()
    }

    /// Render Prometheus-format metrics. Active collection/vector counts and RSS
    /// are passed in from the Engine which has access to the vector store and sysinfo.
    pub fn render(
        &self,
        active_collections: u64,
        total_vectors: u64,
        memory_rss_bytes: u64,
        vector_metrics: &[VectorMetricsSnapshot],
    ) -> String {
        let state_put = self.state_put_total.load(Ordering::Relaxed);
        let state_delete = self.state_delete_total.load(Ordering::Relaxed);
        let vector_ops = self.vector_ops_total.load(Ordering::Relaxed);
        let events = self.events_total.load(Ordering::Relaxed);
        let sse_clients = self.sse_clients.load(Ordering::Relaxed);
        let cache_hits = self.embed_cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.embed_cache_misses.load(Ordering::Relaxed);
        let embed_failures = self.embed_failures_total.load(Ordering::Relaxed);
        let hybrid_sql_first = self.hybrid_sql_first_total.load(Ordering::Relaxed);
        let hybrid_vector_first = self.hybrid_vector_first_total.load(Ordering::Relaxed);
        let hybrid_last_sql_candidates = self.hybrid_last_sql_candidates.load(Ordering::Relaxed);
        let hybrid_last_vector_candidates =
            self.hybrid_last_vector_candidates.load(Ordering::Relaxed);
        let hybrid_last_doc_candidates = self.hybrid_last_doc_candidates.load(Ordering::Relaxed);
        let hybrid_last_hydrated_docs = self.hybrid_last_hydrated_docs.load(Ordering::Relaxed);
        let hybrid_sql_candidates_total = self.hybrid_sql_candidates_total.load(Ordering::Relaxed);
        let hybrid_vector_candidates_total =
            self.hybrid_vector_candidates_total.load(Ordering::Relaxed);
        let hybrid_doc_candidates_total = self.hybrid_doc_candidates_total.load(Ordering::Relaxed);
        let hybrid_hydrated_docs_total = self.hybrid_hydrated_docs_total.load(Ordering::Relaxed);
        let vector_candidate_expansion_total = self
            .vector_candidate_expansion_total
            .load(Ordering::Relaxed);
        let vector_search_total = self.vector_search_total.load(Ordering::Relaxed);
        let vector_last_candidate_expansion_steps = self
            .vector_last_candidate_expansion_steps
            .load(Ordering::Relaxed);
        let vector_last_final_candidate_k =
            self.vector_last_final_candidate_k.load(Ordering::Relaxed);
        let vector_last_recall_estimate = self
            .vector_last_recall_estimate_milli
            .load(Ordering::Relaxed) as f64
            / 1000.0;
        let replay_applied = self.wal_replay_applied_total.load(Ordering::Relaxed);
        let replay_duplicates = self.wal_replay_duplicates_total.load(Ordering::Relaxed);
        let replay_corrupt = self.wal_replay_corrupt_total.load(Ordering::Relaxed);
        let wal_gaps = self.wal_gap_total.load(Ordering::Relaxed);
        let wal_rotations = self.wal_rotation_total.load(Ordering::Relaxed);
        let snapshots = self.snapshot_total.load(Ordering::Relaxed);
        let snapshot_failures = self.snapshot_failed_total.load(Ordering::Relaxed);
        let ttl_expired = self.ttl_expired_total.load(Ordering::Relaxed);
        let hnsw_compactions = self.hnsw_compaction_total.load(Ordering::Relaxed);

        let mut out = String::with_capacity(2048);

        // Counters
        out.push_str("# TYPE state_put_total counter\n");
        out.push_str(&format!("state_put_total {state_put}\n"));
        out.push_str("# TYPE state_delete_total counter\n");
        out.push_str(&format!("state_delete_total {state_delete}\n"));
        out.push_str("# TYPE vector_ops_total counter\n");
        out.push_str(&format!("vector_ops_total {vector_ops}\n"));
        out.push_str("# TYPE events_total counter\n");
        out.push_str(&format!("events_total {events}\n"));
        out.push_str("# TYPE sse_clients gauge\n");
        out.push_str(&format!("sse_clients {sse_clients}\n"));

        // System gauges
        out.push_str("# TYPE active_collections gauge\n");
        out.push_str(&format!("active_collections {active_collections}\n"));
        out.push_str("# TYPE total_vectors gauge\n");
        out.push_str(&format!("total_vectors {total_vectors}\n"));
        out.push_str("# TYPE memory_rss_bytes gauge\n");
        out.push_str(&format!("memory_rss_bytes {memory_rss_bytes}\n"));

        // Embedding cache counters
        out.push_str("# TYPE embed_cache_hits_total counter\n");
        out.push_str(&format!("embed_cache_hits_total {cache_hits}\n"));
        out.push_str("# TYPE embed_cache_misses_total counter\n");
        out.push_str(&format!("embed_cache_misses_total {cache_misses}\n"));
        out.push_str("# TYPE embed_failures_total counter\n");
        out.push_str(&format!("embed_failures_total {embed_failures}\n"));
        out.push_str("# TYPE hybrid_sql_first_total counter\n");
        out.push_str(&format!("hybrid_sql_first_total {hybrid_sql_first}\n"));
        out.push_str("# TYPE hybrid_vector_first_total counter\n");
        out.push_str(&format!(
            "hybrid_vector_first_total {hybrid_vector_first}\n"
        ));
        out.push_str("# TYPE hybrid_last_sql_candidates gauge\n");
        out.push_str(&format!(
            "hybrid_last_sql_candidates {hybrid_last_sql_candidates}\n"
        ));
        out.push_str("# TYPE hybrid_last_vector_candidates gauge\n");
        out.push_str(&format!(
            "hybrid_last_vector_candidates {hybrid_last_vector_candidates}\n"
        ));
        out.push_str("# TYPE hybrid_last_doc_candidates gauge\n");
        out.push_str(&format!(
            "hybrid_last_doc_candidates {hybrid_last_doc_candidates}\n"
        ));
        out.push_str("# TYPE hybrid_last_hydrated_docs gauge\n");
        out.push_str(&format!(
            "hybrid_last_hydrated_docs {hybrid_last_hydrated_docs}\n"
        ));
        out.push_str("# TYPE hybrid_sql_candidates_total counter\n");
        out.push_str(&format!(
            "hybrid_sql_candidates_total {hybrid_sql_candidates_total}\n"
        ));
        out.push_str("# TYPE hybrid_vector_candidates_total counter\n");
        out.push_str(&format!(
            "hybrid_vector_candidates_total {hybrid_vector_candidates_total}\n"
        ));
        out.push_str("# TYPE hybrid_doc_candidates_total counter\n");
        out.push_str(&format!(
            "hybrid_doc_candidates_total {hybrid_doc_candidates_total}\n"
        ));
        out.push_str("# TYPE hybrid_hydrated_docs_total counter\n");
        out.push_str(&format!(
            "hybrid_hydrated_docs_total {hybrid_hydrated_docs_total}\n"
        ));
        out.push_str("# TYPE vector_search_total counter\n");
        out.push_str(&format!("vector_search_total {vector_search_total}\n"));
        out.push_str("# TYPE vector_candidate_expansion_total counter\n");
        out.push_str(&format!(
            "vector_candidate_expansion_total {vector_candidate_expansion_total}\n"
        ));
        out.push_str("# TYPE vector_last_candidate_expansion_steps gauge\n");
        out.push_str(&format!(
            "vector_last_candidate_expansion_steps {vector_last_candidate_expansion_steps}\n"
        ));
        out.push_str("# TYPE vector_last_final_candidate_k gauge\n");
        out.push_str(&format!(
            "vector_last_final_candidate_k {vector_last_final_candidate_k}\n"
        ));
        out.push_str("# TYPE vector_last_recall_estimate gauge\n");
        out.push_str(&format!(
            "vector_last_recall_estimate {:.3}\n",
            vector_last_recall_estimate
        ));
        out.push_str("# TYPE wal_replay_applied_total counter\n");
        out.push_str(&format!("wal_replay_applied_total {replay_applied}\n"));
        out.push_str("# TYPE wal_replay_duplicates_total counter\n");
        out.push_str(&format!(
            "wal_replay_duplicates_total {replay_duplicates}\n"
        ));
        out.push_str("# TYPE wal_replay_corrupt_total counter\n");
        out.push_str(&format!("wal_replay_corrupt_total {replay_corrupt}\n"));
        out.push_str("# TYPE wal_gap_total counter\n");
        out.push_str(&format!("wal_gap_total {wal_gaps}\n"));
        out.push_str("# TYPE wal_rotation_total counter\n");
        out.push_str(&format!("wal_rotation_total {wal_rotations}\n"));
        out.push_str("# TYPE snapshot_total counter\n");
        out.push_str(&format!("snapshot_total {snapshots}\n"));
        out.push_str("# TYPE snapshot_failed_total counter\n");
        out.push_str(&format!("snapshot_failed_total {snapshot_failures}\n"));
        out.push_str("# TYPE ttl_expired_total counter\n");
        out.push_str(&format!("ttl_expired_total {ttl_expired}\n"));
        out.push_str("# TYPE hnsw_compaction_total counter\n");
        out.push_str(&format!("hnsw_compaction_total {hnsw_compactions}\n"));

        // Latency histograms
        self.search_latency
            .render_prometheus("search_duration_ms", &mut out);
        self.ingest_latency
            .render_prometheus("ingest_duration_ms", &mut out);
        self.embed_latency
            .render_prometheus("embed_duration_ms", &mut out);
        self.hybrid_sql_prefilter_latency
            .render_prometheus("hybrid_sql_prefilter_duration_ms", &mut out);
        self.hybrid_vector_latency
            .render_prometheus("hybrid_vector_duration_ms", &mut out);
        self.hybrid_hydration_latency
            .render_prometheus("hybrid_hydration_duration_ms", &mut out);
        self.hybrid_chunking_latency
            .render_prometheus("hybrid_chunking_duration_ms", &mut out);
        self.hybrid_vector_write_latency
            .render_prometheus("hybrid_vector_write_duration_ms", &mut out);
        self.hybrid_sql_write_latency
            .render_prometheus("hybrid_sql_write_duration_ms", &mut out);
        self.vector_ann_latency
            .render_prometheus("vector_ann_duration_ms", &mut out);

        // Human-readable percentiles for convenience
        out.push_str("# Approximate latency percentiles (ms)\n");
        out.push_str(&format!(
            "search_p50_ms {:.1}\nsearch_p95_ms {:.1}\nsearch_p99_ms {:.1}\n",
            self.search_latency.percentile_ms(50.0),
            self.search_latency.percentile_ms(95.0),
            self.search_latency.percentile_ms(99.0),
        ));
        out.push_str(&format!(
            "ingest_p50_ms {:.1}\ningest_p95_ms {:.1}\ningest_p99_ms {:.1}\n",
            self.ingest_latency.percentile_ms(50.0),
            self.ingest_latency.percentile_ms(95.0),
            self.ingest_latency.percentile_ms(99.0),
        ));
        out.push_str(&format!(
            "embed_p50_ms {:.1}\nembed_p95_ms {:.1}\nembed_p99_ms {:.1}\n",
            self.embed_latency.percentile_ms(50.0),
            self.embed_latency.percentile_ms(95.0),
            self.embed_latency.percentile_ms(99.0),
        ));

        for collection in vector_metrics {
            out.push_str(&format!(
                "vector_collection_live_count{{collection=\"{}\"}} {}\n",
                collection.collection, collection.live_count
            ));
            out.push_str(&format!(
                "vector_collection_segments{{collection=\"{}\"}} {}\n",
                collection.collection, collection.segments
            ));
            out.push_str(&format!(
                "vector_collection_deleted_count{{collection=\"{}\"}} {}\n",
                collection.collection, collection.deleted_count
            ));
            out.push_str(&format!(
                "vector_collection_fragmentation_score{{collection=\"{}\"}} {:.4}\n",
                collection.collection, collection.fragmentation_score
            ));
        }

        *self.last_metrics_render.lock() = out.clone();
        out
    }
}
