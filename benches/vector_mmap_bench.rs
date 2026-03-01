use criterion::{criterion_group, criterion_main, Criterion};
use tempfile::tempdir;
use luma::vector::mmap::VectorMmap;
use rand::Rng;

fn bench_mmap_append(c: &mut Criterion) {
    let temp_dir = tempdir().unwrap();
    let mmap_path = temp_dir.path().join("vectors.mmap");
    let dim = 1536;
    let mut store = VectorMmap::create_or_open(&mmap_path, dim, 1000).unwrap();
    let mut rng = rand::thread_rng();
    
    let vector: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

    c.bench_function("mmap_append", |b| b.iter(|| {
        store.append(&vector).unwrap()
    }));
}

criterion_group!(benches, bench_mmap_append);
criterion_main!(benches);
