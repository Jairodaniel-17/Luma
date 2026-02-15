use criterion::{criterion_group, criterion_main, Criterion};
use luma::sqlite::SqliteService;
use serde_json::json;
use tempfile::tempdir;

fn bench_sqlite_ops(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("bench.db");
    // SqliteService::new returns runtime::Runtime needed?
    // No, it's async so I need `to_async(tokio::runtime::Runtime::new().unwrap())` for async benchmark?
    // Or I can use `block_on`. `SqliteService` methods are async.

    // Criterion supports async benchmarks with `to_async`.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let svc = SqliteService::new(&db_path).unwrap();
    let _ = rt
        .block_on(svc.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)".to_string(),
            vec![],
        ))
        .unwrap();

    let mut group = c.benchmark_group("SqliteOps");

    let params = vec![json!("test_val")];
    let sql = "INSERT INTO items (val) VALUES (?)".to_string();

    group.bench_function("insert_sqlite", |b| {
        b.to_async(&rt).iter(|| async {
            svc.execute(sql.clone(), params.clone()).await.unwrap();
        })
    });

    // Pre-fill
    for _ in 0..100 {
        let _ = rt.block_on(svc.execute(
            "INSERT INTO items (val) VALUES (?)".to_string(),
            vec![json!("test")],
        ));
    }

    let query_sql = "SELECT * FROM items LIMIT 10".to_string();
    group.bench_function("query_sqlite", |b| {
        b.to_async(&rt).iter(|| async {
            svc.query(query_sql.clone(), vec![]).await.unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_sqlite_ops);
criterion_main!(benches);
