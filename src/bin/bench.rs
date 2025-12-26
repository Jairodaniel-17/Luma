use anyhow::Context;
use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::vector::index::DiskAnnBuildParams;
use rust_kiss_vdb::vector::{Metric, SearchRequest, VectorItem};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use sysinfo::System;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};

const DATASET_MAGIC: u32 = 0x524B_4441; // "RKDA"

#[derive(Parser, Debug)]
#[command(
    name = "rustkiss bench",
    version,
    about = "Benchmarks for RustKissVDB vector engine"
)]
struct BenchCli {
    /// Dimensions to benchmark, comma-separated (e.g. 768,1024,4096)
    #[arg(long, default_value = "768,1024,4096")]
    dims: String,
    /// Number of vectors to insert per dimension
    #[arg(long, default_value_t = 100_000)]
    rows: usize,
    /// Number of search queries per dimension
    #[arg(long, default_value_t = 2_000)]
    search_queries: usize,
    /// Top-k for search
    #[arg(long, default_value_t = 10)]
    k: usize,
    /// Metric to benchmark
    #[arg(long, value_enum, default_value_t = MetricArg::Cosine)]
    metric: MetricArg,
    /// Modes to execute (baseline_f32,ivf_flat_q8,diskann)
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [BenchMode::BaselineF32, BenchMode::IvfFlatQ8]
    )]
    modes: Vec<BenchMode>,
    /// Optional directory to store benchmark data. Defaults to temp dir.
    #[arg(long)]
    data_dir: Option<String>,
    /// Random seed used for reproducible vectors
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Keep generated data when using the temp dir (useful for inspecting artifacts)
    #[arg(long)]
    keep_data: bool,
    /// Reuse on-disk datasets between runs if available
    #[arg(long)]
    reuse_data: bool,
    /// Reuse previously built indexes (skips rebuild if metadata matches)
    #[arg(long)]
    reuse_index: bool,
    /// Run entirely in-memory (no on-disk persistence, reuse flags ignored)
    #[arg(long)]
    in_mem: bool,
    /// IVF cluster count (centroids)
    #[arg(long, default_value_t = 512)]
    ivf_clusters: usize,
    /// IVF nprobe value
    #[arg(long, default_value_t = 16)]
    ivf_nprobe: usize,
    /// Sample size for IVF training
    #[arg(long, default_value_t = 50_000)]
    ivf_training_sample: usize,
    /// Minimum live vectors before first training
    #[arg(long, default_value_t = 1_024)]
    ivf_min_train_vectors: usize,
    /// Upsert delta before retraining
    #[arg(long, default_value_t = 50_000)]
    ivf_retrain_min_deltas: usize,
    /// Refinement window for Q8 → f32 scoring
    #[arg(long, default_value_t = 512)]
    q8_refine_topk: usize,
    /// DiskANN max degree (graph fanout)
    #[arg(long, default_value_t = 64)]
    diskann_max_degree: usize,
    /// DiskANN build threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    diskann_build_threads: usize,
    /// DiskANN search_list size
    #[arg(long, default_value_t = 128)]
    diskann_search_list_size: usize,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum MetricArg {
    Cosine,
    Dot,
}

impl From<MetricArg> for Metric {
    fn from(value: MetricArg) -> Self {
        match value {
            MetricArg::Cosine => Metric::Cosine,
            MetricArg::Dot => Metric::Dot,
        }
    }
}

impl MetricArg {
    fn label(self) -> &'static str {
        match self {
            MetricArg::Cosine => "cosine",
            MetricArg::Dot => "dot",
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum BenchMode {
    BaselineF32,
    IvfFlatQ8,
    DiskAnn,
}

impl BenchMode {
    fn label(self) -> &'static str {
        match self {
            BenchMode::BaselineF32 => "baseline_f32",
            BenchMode::IvfFlatQ8 => "ivf_flat_q8",
            BenchMode::DiskAnn => "diskann",
        }
    }

    fn dir_name(self) -> &'static str {
        self.label()
    }

    fn index_kind(self) -> &'static str {
        match self {
            BenchMode::BaselineF32 => "HNSW",
            BenchMode::IvfFlatQ8 => "IVF_FLAT_Q8",
            BenchMode::DiskAnn => "DISKANN",
        }
    }

    fn is_baseline(self) -> bool {
        matches!(self, BenchMode::BaselineF32)
    }
}

#[derive(Clone)]
struct DatasetInfo {
    path: PathBuf,
    dim: usize,
    rows: usize,
}

impl DatasetInfo {
    fn stream<F>(&self, mut f: F) -> anyhow::Result<()>
    where
        F: FnMut(usize, Vec<f32>) -> anyhow::Result<()>,
    {
        let mut file = File::open(&self.path)
            .with_context(|| format!("open dataset {}", self.path.display()))?;
        let mut header = [0u8; 16];
        file.read_exact(&mut header)
            .context("read dataset header")?;
        let magic = u32::from_le_bytes(header[..4].try_into().unwrap());
        if magic != DATASET_MAGIC {
            anyhow::bail!("invalid dataset header");
        }
        let dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let rows = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        if dim != self.dim || rows != self.rows {
            anyhow::bail!("dataset header mismatch");
        }
        let mut buf = vec![0u8; dim * 4];
        for idx in 0..rows {
            file.read_exact(&mut buf)
                .context("read dataset vector bytes")?;
            let mut vec = Vec::with_capacity(dim);
            for chunk in buf.chunks_exact(4) {
                vec.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            f(idx, vec)?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct BuildMeta {
    fingerprint: String,
    insert_stats: StoredLatencyStats,
    mem_delta: u64,
}

#[derive(Serialize, Deserialize)]
struct StoredLatencyStats {
    p50: u128,
    p95: u128,
    p99: u128,
    throughput: f64,
}

impl From<LatencyStats> for StoredLatencyStats {
    fn from(value: LatencyStats) -> Self {
        Self {
            p50: value.p50,
            p95: value.p95,
            p99: value.p99,
            throughput: value.throughput,
        }
    }
}

impl From<StoredLatencyStats> for LatencyStats {
    fn from(value: StoredLatencyStats) -> Self {
        Self {
            p50: value.p50,
            p95: value.p95,
            p99: value.p99,
            throughput: value.throughput,
        }
    }
}

fn prepare_dataset(base_dir: &Path, dim: usize, cli: &BenchCli) -> anyhow::Result<DatasetInfo> {
    let datasets_dir = base_dir.join("datasets");
    fs::create_dir_all(&datasets_dir)?;
    let file_name = dataset_file_name(dim, cli.rows, cli.metric.label(), cli.seed);
    let path = datasets_dir.join(file_name);
    let mut reused = false;
    if path.exists() && cli.reuse_data {
        reused = dataset_matches(&path, dim, cli.rows)?;
    }
    if reused {
        println!(
            "[dataset] dim={} rows={} metric={} reuse path={}",
            dim,
            cli.rows,
            cli.metric.label(),
            path.display()
        );
    } else {
        println!(
            "[dataset] dim={} rows={} metric={} generating at {}",
            dim,
            cli.rows,
            cli.metric.label(),
            path.display()
        );
        if path.exists() {
            let _ = fs::remove_file(&path);
        }
        generate_dataset_file(&path, dim, cli.rows, dataset_seed(cli.seed, dim))?;
    }
    Ok(DatasetInfo {
        path,
        dim,
        rows: cli.rows,
    })
}

fn dataset_file_name(dim: usize, rows: usize, metric: &str, seed: u64) -> String {
    format!("dataset_dim{dim}_rows{rows}_{metric}_seed{seed}.bin")
}

fn dataset_seed(seed: u64, dim: usize) -> u64 {
    seed ^ ((dim as u64).wrapping_mul(0x9E37_79B1_85EB_CA87))
}

fn dataset_matches(path: &Path, dim: usize, rows: usize) -> anyhow::Result<bool> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let mut header = [0u8; 16];
    if file.read_exact(&mut header).is_err() {
        return Ok(false);
    }
    let magic = u32::from_le_bytes(header[..4].try_into().unwrap());
    if magic != DATASET_MAGIC {
        return Ok(false);
    }
    let stored_dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let stored_rows = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    Ok(stored_dim == dim && stored_rows == rows)
}

fn generate_dataset_file(path: &Path, dim: usize, rows: usize, seed: u64) -> anyhow::Result<()> {
    let mut file =
        File::create(path).with_context(|| format!("create dataset {}", path.display()))?;
    file.write_all(&DATASET_MAGIC.to_le_bytes())?;
    file.write_all(&(dim as u32).to_le_bytes())?;
    file.write_all(&(rows as u64).to_le_bytes())?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut buf = vec![0u8; dim * 4];
    for _ in 0..rows {
        for j in 0..dim {
            let value: f32 = rng.gen();
            buf[j * 4..(j + 1) * 4].copy_from_slice(&value.to_le_bytes());
        }
        file.write_all(&buf)?;
    }
    Ok(())
}

fn build_fingerprint(dim: usize, metric: Metric, mode: BenchMode, cli: &BenchCli) -> String {
    let mut base = format!(
        "dim={dim};rows={};metric={:?};mode={};seed={};clusters={};nprobe={};train_sample={};min_train={};retrain={};q8ref={}",
        cli.rows,
        metric,
        mode.label(),
        cli.seed,
        cli.ivf_clusters,
        cli.ivf_nprobe,
        cli.ivf_training_sample,
        cli.ivf_min_train_vectors,
        cli.ivf_retrain_min_deltas,
        cli.q8_refine_topk,
    );
    if matches!(mode, BenchMode::DiskAnn) {
        base.push_str(&format!(
            ";diskann_degree={};diskann_threads={};diskann_search={}",
            cli.diskann_max_degree, cli.diskann_build_threads, cli.diskann_search_list_size
        ));
    }
    base
}

fn read_build_meta(path: &Path) -> anyhow::Result<BuildMeta> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

fn write_build_meta(path: &Path, meta: &BuildMeta) -> anyhow::Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, meta)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = BenchCli::parse();
    run_vector_bench(&cli)
}

fn timestamp_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string())
}

fn run_vector_bench(cli: &BenchCli) -> anyhow::Result<()> {
    if cli.in_mem && cli.reuse_index {
        println!("[bench] aviso: --reuse-index se ignora en modo --in-mem");
    }
    let dims = parse_dims(&cli.dims)?;
    let base_dir = cli
        .data_dir
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("rustkiss-bench-{}", now_ms())));
    if base_dir.exists() && cli.data_dir.is_none() {
        fs::remove_dir_all(&base_dir)?;
    }
    fs::create_dir_all(&base_dir)?;

    let mode_labels: Vec<&str> = cli.modes.iter().map(|m| m.label()).collect();
    let bench_start = Instant::now();
    let bench_start_ts = timestamp_now();
    println!(
        "starting bench: dims={:?} rows={} queries={} metric={:?} modes={:?} start_ts={}",
        dims, cli.rows, cli.search_queries, cli.metric, mode_labels, bench_start_ts
    );

    for dim in dims {
        let dataset = prepare_dataset(&base_dir, dim, cli)?;
        run_single_dimension(&base_dir, dim, cli, &dataset)?;
    }

    if cli.data_dir.is_none() && !cli.keep_data {
        let _ = fs::remove_dir_all(&base_dir);
    } else {
        println!("bench data stored under {}", base_dir.display());
    }
    let bench_end_ts = timestamp_now();
    println!(
        "bench completed: end_ts={} elapsed={:.2}s",
        bench_end_ts,
        bench_start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_single_dimension(
    base_dir: &Path,
    dim: usize,
    cli: &BenchCli,
    dataset: &DatasetInfo,
) -> anyhow::Result<()> {
    let metric: Metric = cli.metric.into();
    let dir = base_dir.join(format!("dim_{dim}"));
    if dir.exists() && !cli.reuse_index {
        fs::remove_dir_all(&dir)?;
    }
    fs::create_dir_all(&dir)?;
    let query_seed = cli.seed ^ 0xBAD0FACEu64 ^ dim as u64;
    let queries = generate_queries(dim, cli.search_queries, query_seed);

    let mut eval_order = cli.modes.clone();
    let mut baseline_hidden = false;
    if !eval_order.iter().any(|m| m.is_baseline()) {
        eval_order.insert(0, BenchMode::BaselineF32);
        baseline_hidden = true;
    } else if let Some(pos) = eval_order.iter().position(|m| m.is_baseline()) {
        eval_order.swap(0, pos);
    }

    let mut reports = Vec::new();
    let mut baseline_hits: Option<Vec<Vec<String>>> = None;

    let dim_start_ts = timestamp_now();
    let dim_start = Instant::now();
    println!(
        "dimension {} start_ts={} metric={:?} rows={} queries={}",
        dim, dim_start_ts, metric, cli.rows, cli.search_queries
    );

    for mode in eval_order {
        let (mut metrics, hits) =
            run_mode_for_dimension(&dir, dim, metric, cli, mode, dataset, &queries)?;
        if mode.is_baseline() && baseline_hits.is_none() {
            baseline_hits = Some(hits);
            metrics.recall = Some(1.0);
        } else if let Some(base) = baseline_hits.as_ref() {
            metrics.recall = recall_against(base, &hits);
        }
        reports.push(metrics);
    }

    if baseline_hidden {
        reports.retain(|m| !m.mode.is_baseline());
    }

    reports.sort_by_key(|m| {
        cli.modes
            .iter()
            .position(|mode| mode == &m.mode)
            .unwrap_or(usize::MAX)
    });

    print_dimension_report(dim, metric, cli.rows, cli.search_queries, cli.k, &reports);
    let dim_end_ts = timestamp_now();
    println!(
        "dimension {} completed: end_ts={} elapsed={:.2}s",
        dim,
        dim_end_ts,
        dim_start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_mode_for_dimension(
    dim_dir: &Path,
    dim: usize,
    metric: Metric,
    cli: &BenchCli,
    mode: BenchMode,
    dataset: &DatasetInfo,
    queries: &[Vec<f32>],
) -> anyhow::Result<(ModeMetrics, Vec<Vec<String>>)> {
    if cli.in_mem && matches!(mode, BenchMode::DiskAnn) {
        return Err(anyhow::anyhow!(
            "DiskANN bench requiere almacenamiento persistente; remueve --in-mem"
        ));
    }
    let mode_dir = dim_dir.join(mode.dir_name());
    let collection = format!("bench_{dim}_{}", mode.dir_name());
    let mode_start_ts = timestamp_now();
    let mode_start = Instant::now();
    println!(
        "[mode-start] dim={} metric={:?} mode={} start_ts={}",
        dim,
        metric,
        mode.label(),
        mode_start_ts
    );

    if cli.in_mem {
        if mode_dir.exists() {
            fs::remove_dir_all(&mode_dir)?;
        }
        let config = bench_config(cli, None, cli.k, mode);
        let engine = Engine::new(config.clone())?;
        engine.create_vector_collection(&collection, dim, metric)?;
        let (insert_stats, mem_delta) = ingest_dataset(&engine, &collection, dataset)?;
        if matches!(mode, BenchMode::DiskAnn) {
            let params = bench_diskann_params(cli);
            engine.vector_build_disk_index(&collection, params)?;
        }
        let (search_stats, recorded_hits) = execute_queries(&engine, &collection, cli, queries)?;
        println!(
            "[mode-end] dim={} metric={:?} mode={} end_ts={} elapsed={:.2}s",
            dim,
            metric,
            mode.label(),
            timestamp_now(),
            mode_start.elapsed().as_secs_f64()
        );
        return Ok((
            ModeMetrics {
                mode,
                insert: insert_stats,
                search: search_stats,
                mem_delta,
                disk_bytes: 0,
                recall: None,
            },
            recorded_hits,
        ));
    }

    fs::create_dir_all(&mode_dir)?;
    let fingerprint = build_fingerprint(dim, metric, mode, cli);
    let meta_path = mode_dir.join("build-meta.json");
    let mut insert_stats: Option<LatencyStats> = None;
    let mut mem_delta = 0u64;
    let mut reused = false;

    if cli.reuse_index {
        if let Ok(meta) = read_build_meta(&meta_path) {
            if meta.fingerprint == fingerprint {
                insert_stats = Some(meta.insert_stats.into());
                mem_delta = meta.mem_delta;
                reused = true;
                println!(
                    "[build] dim={} mode={} reuse existing index",
                    dim,
                    mode.label()
                );
            }
        }
    }

    if !reused {
        if mode_dir.exists() {
            fs::remove_dir_all(&mode_dir)?;
        }
        fs::create_dir_all(&mode_dir)?;
        let build_start = Instant::now();
        println!(
            "[build] dim={} mode={} start_ts={} reuse=false",
            dim,
            mode.label(),
            timestamp_now()
        );
        let config = bench_config(cli, Some(&mode_dir), cli.k, mode);
        let engine = Engine::new(config.clone())?;
        engine.create_vector_collection(&collection, dim, metric)?;
        let (stats, mem) = ingest_dataset(&engine, &collection, dataset)?;
        insert_stats = Some(stats);
        mem_delta = mem;
        if matches!(mode, BenchMode::DiskAnn) {
            let params = bench_diskann_params(cli);
            engine.vector_build_disk_index(&collection, params)?;
        }
        drop(engine);
        let meta = BuildMeta {
            fingerprint: fingerprint.clone(),
            insert_stats: stats.into(),
            mem_delta,
        };
        write_build_meta(&meta_path, &meta)?;
        println!(
            "[build] dim={} mode={} end_ts={} elapsed={:.2}s",
            dim,
            mode.label(),
            timestamp_now(),
            build_start.elapsed().as_secs_f64()
        );
    }

    let config = bench_config(cli, Some(&mode_dir), cli.k, mode);
    let engine = Engine::new(config.clone())?;
    let (search_stats, recorded_hits) = execute_queries(&engine, &collection, cli, queries)?;
    drop(engine);
    let disk_bytes = dir_size(&mode_dir);

    println!(
        "[mode-end] dim={} metric={:?} mode={} end_ts={} elapsed={:.2}s",
        dim,
        metric,
        mode.label(),
        timestamp_now(),
        mode_start.elapsed().as_secs_f64()
    );

    Ok((
        ModeMetrics {
            mode,
            insert: insert_stats.expect("insert stats available"),
            search: search_stats,
            mem_delta,
            disk_bytes,
            recall: None,
        },
        recorded_hits,
    ))
}

fn ingest_dataset(
    engine: &Engine,
    collection: &str,
    dataset: &DatasetInfo,
) -> anyhow::Result<(LatencyStats, u64)> {
    let mem_before = process_memory_bytes();
    let mut insert_lat = Vec::with_capacity(dataset.rows);
    dataset.stream(|i, vector| {
        let start = Instant::now();
        engine.vector_upsert(
            collection,
            &format!("vec-{i}"),
            VectorItem {
                vector,
                meta: json!({ "seq": i }),
            },
        )?;
        insert_lat.push(start.elapsed());
        Ok(())
    })?;
    let mem_after = process_memory_bytes();
    let mem_delta = mem_after.saturating_sub(mem_before);
    Ok((LatencyStats::from_samples(&insert_lat), mem_delta))
}

fn execute_queries(
    engine: &Engine,
    collection: &str,
    cli: &BenchCli,
    queries: &[Vec<f32>],
) -> anyhow::Result<(LatencyStats, Vec<Vec<String>>)> {
    let mut search_lat = Vec::with_capacity(queries.len());
    let mut recorded_hits = Vec::with_capacity(queries.len());
    for (qid, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let hits = engine.vector_search(
            collection,
            SearchRequest {
                vector: query.clone(),
                k: cli.k,
                filters: None,
                include_meta: Some(false),
            },
        )?;
        search_lat.push(start.elapsed());
        let ids = hits.iter().map(|hit| hit.id.clone()).collect();
        recorded_hits.push(ids);
        if qid + 1 >= cli.search_queries {
            break;
        }
    }
    Ok((LatencyStats::from_samples(&search_lat), recorded_hits))
}

fn parse_dims(raw: &str) -> anyhow::Result<Vec<usize>> {
    let mut dims = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let dim = trimmed
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("invalid dimension `{trimmed}`"))?;
        if dim == 0 {
            return Err(anyhow::anyhow!("dimension must be > 0"));
        }
        dims.push(dim);
    }
    if dims.is_empty() {
        return Err(anyhow::anyhow!("at least one dimension is required"));
    }
    Ok(dims)
}

struct ModeMetrics {
    mode: BenchMode,
    insert: LatencyStats,
    search: LatencyStats,
    mem_delta: u64,
    disk_bytes: u64,
    recall: Option<f32>,
}

fn print_dimension_report(
    dim: usize,
    metric: Metric,
    rows: usize,
    queries: usize,
    k: usize,
    reports: &[ModeMetrics],
) {
    println!("--- dimension {dim} ({:?}) ---", metric);
    for metrics in reports {
        print_mode_report(dim, rows, queries, k, metrics);
    }
}

fn print_mode_report(dim: usize, rows: usize, queries: usize, k: usize, metrics: &ModeMetrics) {
    println!("# mode={} dim={}", metrics.mode.label(), dim);
    println!(
        "insert: n={} p50={}us p95={}us p99={}us throughput={:.2} vec/s",
        rows, metrics.insert.p50, metrics.insert.p95, metrics.insert.p99, metrics.insert.throughput
    );
    println!(
        "search: n={} p50={}us p95={}us p99={}us qps={:.2}",
        queries,
        metrics.search.p50,
        metrics.search.p95,
        metrics.search.p99,
        metrics.search.throughput
    );
    if let Some(recall) = metrics.recall {
        println!("recall@{}={:.2}%", k, recall * 100.0);
    } else {
        println!("recall@{}=n/a (baseline missing)", k);
    }
    println!(
        "usage: ram~{:.2} MiB, disk~{:.2} MiB",
        metrics.mem_delta as f64 / (1024.0 * 1024.0),
        metrics.disk_bytes as f64 / (1024.0 * 1024.0)
    );
}

fn recall_against(baseline: &[Vec<String>], current: &[Vec<String>]) -> Option<f32> {
    if baseline.is_empty() || baseline.len() != current.len() {
        return None;
    }
    let mut hits = 0f32;
    let mut total = 0f32;
    for (base, candidate) in baseline.iter().zip(current.iter()) {
        if base.is_empty() {
            continue;
        }
        let set: HashSet<&str> = base.iter().map(|s| s.as_str()).collect();
        total += base.len() as f32;
        hits += candidate
            .iter()
            .filter(|id| set.contains(id.as_str()))
            .count() as f32;
    }
    if total <= f32::EPSILON {
        None
    } else {
        Some(hits / total)
    }
}

#[derive(Clone, Copy)]
struct LatencyStats {
    p50: u128,
    p95: u128,
    p99: u128,
    throughput: f64,
}

impl LatencyStats {
    fn from_samples(samples: &[Duration]) -> Self {
        if samples.is_empty() {
            return Self {
                p50: 0,
                p95: 0,
                p99: 0,
                throughput: 0.0,
            };
        }
        let mut micros: Vec<u128> = samples.iter().map(|d| d.as_micros()).collect();
        micros.sort_unstable();
        let p50 = percentile(&micros, 50.0);
        let p95 = percentile(&micros, 95.0);
        let p99 = percentile(&micros, 99.0);
        let total = samples.iter().fold(Duration::ZERO, |acc, v| acc + *v);
        let throughput = if total.as_secs_f64() > 0.0 {
            samples.len() as f64 / total.as_secs_f64()
        } else {
            0.0
        };
        Self {
            p50,
            p95,
            p99,
            throughput,
        }
    }
}

fn percentile(sorted_micros: &[u128], p: f64) -> u128 {
    if sorted_micros.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_micros.len() as f64 - 1.0)).round() as usize;
    sorted_micros[idx.min(sorted_micros.len() - 1)]
}

fn generate_queries(dim: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| random_vector(dim, &mut rng)).collect()
}

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen()).collect()
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_file() {
                total = total.saturating_add(meta.len());
            } else if meta.is_dir() {
                total = total.saturating_add(dir_size(&path));
            }
        }
    }
    total
}

fn process_memory_bytes() -> u64 {
    let pid = match sysinfo::get_current_pid() {
        Ok(pid) => pid,
        Err(_) => return 0,
    };
    let mut system = System::new();
    if !system.refresh_process(pid) {
        system.refresh_processes();
    }
    system
        .process(pid)
        .map(|process| process.memory())
        .unwrap_or(0)
}

fn now_ms() -> u128 {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    dur.as_millis()
}

fn bench_config(cli: &BenchCli, data_dir: Option<&Path>, k: usize, mode: BenchMode) -> Config {
    let data_dir = if cli.in_mem {
        None
    } else {
        data_dir.map(|p| p.to_string_lossy().to_string())
    };
    let diskann_threads = if cli.diskann_build_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        cli.diskann_build_threads
    };
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "bench".to_string(),
        data_dir,
        snapshot_interval_secs: 3600,
        event_buffer_size: 10_000,
        live_broadcast_capacity: 4096,
        wal_segment_max_bytes: 256 * 1024 * 1024,
        wal_retention_segments: 8,
        request_timeout_secs: 30,
        max_body_bytes: 4 * 1_048_576,
        max_key_len: 512,
        max_collection_len: 128,
        max_id_len: 128,
        max_vector_dim: 8192,
        max_k: k.max(256),
        max_json_bytes: 64 * 1024,
        max_state_batch: 512,
        max_vector_batch: 1024,
        max_doc_find: 100,
        cors_allowed_origins: None,
        sqlite_enabled: false,
        sqlite_path: None,
        search_threads: 0,
        parallel_probe: true,
        parallel_probe_min_segments: 4,
        simd_enabled: true,
        index_kind: mode.index_kind().to_string(),
        ivf_clusters: cli.ivf_clusters,
        ivf_nprobe: cli.ivf_nprobe,
        ivf_training_sample: cli.ivf_training_sample,
        ivf_min_train_vectors: cli.ivf_min_train_vectors,
        ivf_retrain_min_deltas: cli.ivf_retrain_min_deltas,
        q8_refine_topk: cli.q8_refine_topk,
        diskann_max_degree: cli.diskann_max_degree.max(4),
        diskann_build_threads: diskann_threads.max(1),
        diskann_search_list_size: cli.diskann_search_list_size.max(k.max(8)),
        run_target_bytes: 32 * 1024 * 1024,
        run_retention: 4,
        compaction_trigger_tombstone_ratio: 0.2,
        compaction_max_bytes_per_pass: 256 * 1024 * 1024,
    }
}

fn bench_diskann_params(cli: &BenchCli) -> DiskAnnBuildParams {
    let threads = if cli.diskann_build_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        cli.diskann_build_threads
    };
    DiskAnnBuildParams {
        max_degree: cli.diskann_max_degree.max(4),
        build_threads: threads.max(1),
        search_list_size: cli.diskann_search_list_size.max(cli.k.max(8)),
    }
    .sanitized()
}
