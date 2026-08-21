use criterion::{criterion_group, criterion_main, Criterion};
use luma::config::Config;
use luma::engine::chunking::ChunkingEngine;
use luma::engine::embeddings::{EmbeddingClient, EmbeddingHandle, EmbeddingProvider};
use luma::engine::hub::LumaDatabase;
use luma::engine::Engine;
use luma::sqlite::SqliteService;
use std::sync::Arc;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

fn bench_hybrid_queries(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("hybrid-bench.db");
    let config = Config {
        data_dir: Some(dir.path().to_string_lossy().to_string()),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };

    let hub = rt.block_on(async {
        let engine = Arc::new(Engine::new(config.clone(), CancellationToken::new()).unwrap());
        let sqlite = Arc::new(SqliteService::new(&db_path).unwrap());
        Arc::new(LumaDatabase::new(
            engine,
            Some(sqlite),
            EmbeddingHandle::new(EmbeddingClient::new(EmbeddingProvider::Mock { dim: 384 })),
            ChunkingEngine::default(),
            config,
        ))
    });

    rt.block_on(async {
        for idx in 0..500usize {
            let tenant = if idx < 12 { "selective" } else { "broad" };
            let metadata = serde_json::json!({
                "tenant": tenant,
                "bucket": if idx % 2 == 0 { "even" } else { "odd" },
            });
            hub.ingest_document(
                "bench",
                &format!("doc-{idx}"),
                &format!("policy document {idx} for {tenant}"),
                serde_json::json!({
                    "id": format!("doc-{idx}"),
                    "text": format!("policy document {idx} for {tenant}"),
                    "metadata": metadata,
                }),
                Some(metadata),
            )
            .await
            .unwrap();
        }
    });

    let mut group = c.benchmark_group("HybridQueries");
    group.bench_function("sql_first_selective_filter", |b| {
        b.to_async(&rt).iter(|| async {
            hub.search_with_plan(
                "bench",
                "policy",
                Some("json_extract(metadata, '$.tenant') = 'selective'"),
                5,
            )
            .await
            .unwrap();
        })
    });
    group.bench_function("vector_first_broad_filter", |b| {
        b.to_async(&rt).iter(|| async {
            hub.search_with_plan(
                "bench",
                "policy",
                Some("json_extract(metadata, '$.tenant') = 'broad'"),
                2,
            )
            .await
            .unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, bench_hybrid_queries);
criterion_main!(benches);
