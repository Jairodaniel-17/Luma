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

    // PR2: embedding cache counters
    embed_cache_hits: AtomicU64,
    embed_cache_misses: AtomicU64,
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

    /// Render Prometheus-format metrics. Active collection/vector counts and RSS
    /// are passed in from the Engine which has access to the vector store and sysinfo.
    pub fn render(
        &self,
        active_collections: u64,
        total_vectors: u64,
        memory_rss_bytes: u64,
    ) -> String {
        let state_put = self.state_put_total.load(Ordering::Relaxed);
        let state_delete = self.state_delete_total.load(Ordering::Relaxed);
        let vector_ops = self.vector_ops_total.load(Ordering::Relaxed);
        let events = self.events_total.load(Ordering::Relaxed);
        let sse_clients = self.sse_clients.load(Ordering::Relaxed);
        let cache_hits = self.embed_cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.embed_cache_misses.load(Ordering::Relaxed);

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

        // Latency histograms
        self.search_latency
            .render_prometheus("search_duration_ms", &mut out);
        self.ingest_latency
            .render_prometheus("ingest_duration_ms", &mut out);
        self.embed_latency
            .render_prometheus("embed_duration_ms", &mut out);

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

        out
    }
}
