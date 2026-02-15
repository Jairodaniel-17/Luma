use criterion::{criterion_group, criterion_main, Criterion};
use luma::vector::{IndexKind, Metric, SearchRequest, VectorItem, VectorSettings, VectorStore};
use serde_json::json;
use tempfile::tempdir;

fn bench_vector_ops(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    // Use default settings but force HNSW for predictable in-memory benchmarking
    let settings = VectorSettings {
        index_kind: IndexKind::Hnsw,
        ..VectorSettings::default()
    };
    // Re-create dir structure as open expects it
    let path = dir.path();
    let store = VectorStore::open_with_settings(path, settings).unwrap();

    let collection = "bench_col";
    store
        .create_collection(collection, 128, Metric::Cosine)
        .unwrap();

    let mut group = c.benchmark_group("VectorOps");

    // Benchmark Add
    // Note: This benchmark adds continuously, so it measures throughput but the state grows.
    // For pure insertion speed, this is acceptable.
    let mut i = 0;
    let vector_128 = vec![0.1; 128];
    let item_template = VectorItem {
        vector: vector_128.clone(),
        meta: json!({}),
    };

    group.bench_function("add_vector", |b| {
        b.iter(|| {
            let id = format!("vec_{}", i);
            let _ = store.add(collection, &id, item_template.clone());
            i += 1;
        })
    });

    // Benchmark Search
    // Pre-populate 1000 items
    for j in 0..1000 {
        let id = format!("search_target_{}", j);
        let _ = store.add(collection, &id, item_template.clone());
    }

    let req = SearchRequest {
        vector: vector_128.clone(),
        k: 10,
        filters: None,
        include_meta: None,
    };

    group.bench_function("search_vector", |b| {
        b.iter(|| {
            store.search(collection, req.clone()).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_vector_ops);
criterion_main!(benches);
