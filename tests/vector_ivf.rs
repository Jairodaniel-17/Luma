use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::vector::{
    IndexKind, Metric, SearchRequest, VectorItem, VectorSettings, VectorStore,
};
use serde_json::json;
use std::fs;

fn config_with_dir(dir: &str) -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string()),
        snapshot_interval_secs: 3600,
        event_buffer_size: 1000,
        live_broadcast_capacity: 1024,
        wal_segment_max_bytes: 256 * 1024,
        wal_retention_segments: 4,
        request_timeout_secs: 30,
        max_body_bytes: 1_048_576,
        max_key_len: 512,
        max_collection_len: 64,
        max_id_len: 128,
        max_vector_dim: 4096,
        max_k: 256,
        max_json_bytes: 64 * 1024,
        max_state_batch: 256,
        max_vector_batch: 256,
        max_doc_find: 100,
        cors_allowed_origins: None,
        sqlite_enabled: false,
        sqlite_path: None,
        search_threads: 0,
        parallel_probe: true,
        parallel_probe_min_segments: 4,
        simd_enabled: true,
        index_kind: "IVF_FLAT_Q8".to_string(),
        ivf_clusters: 2,
        ivf_nprobe: 1,
        ivf_training_sample: 128,
        ivf_min_train_vectors: 8,
        ivf_retrain_min_deltas: 4,
        q8_refine_topk: 64,
        diskann_max_degree: 32,
        diskann_build_threads: 1,
        diskann_search_list_size: 64,
        run_target_bytes: 8 * 1024 * 1024,
        run_retention: 4,
        compaction_trigger_tombstone_ratio: 0.2,
        compaction_max_bytes_per_pass: 64 * 1024 * 1024,
    }
}

#[tokio::test]
async fn ivf_centroids_persist_and_filter() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone()).unwrap();
    engine
        .create_vector_collection("docs", 2, Metric::Cosine)
        .unwrap();
    for i in 0..32 {
        let vec = if i % 2 == 0 {
            vec![1.0, 0.05 * i as f32]
        } else {
            vec![0.05 * i as f32, 1.0]
        };
        engine
            .vector_upsert(
                "docs",
                &format!("vec-{i}"),
                VectorItem {
                    vector: vec,
                    meta: json!({ "cluster": if i % 2 == 0 { "a" } else { "b" } }),
                },
            )
            .unwrap();
    }
    drop(engine);

    // centroids persisted
    assert!(std::path::Path::new(&data_dir)
        .join("vectors/docs/centroids.bin")
        .exists());
    assert!(std::path::Path::new(&data_dir)
        .join("vectors/docs/centroids.json")
        .exists());

    let engine2 = Engine::new(config).unwrap();
    let hits_a = engine2
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![1.0, 0.0],
                k: 4,
                filters: None,
                include_meta: Some(true),
            },
        )
        .unwrap();
    assert!(!hits_a.is_empty());
    assert!(hits_a
        .iter()
        .all(|hit| hit.meta.as_ref().unwrap()["cluster"] == "a"));

    let hits_b = engine2
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![0.0, 1.0],
                k: 4,
                filters: None,
                include_meta: Some(true),
            },
        )
        .unwrap();
    assert!(!hits_b.is_empty());
    assert!(hits_b
        .iter()
        .all(|hit| hit.meta.as_ref().unwrap()["cluster"] == "b"));
}

#[tokio::test]
async fn ivf_config_clamps_values() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.ivf_clusters = 1;
    config.ivf_nprobe = 0;
    config.q8_refine_topk = 0;
    config.ivf_min_train_vectors = 0;
    config.ivf_retrain_min_deltas = 0;

    let engine = Engine::new(config.clone()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    for idx in 0..8 {
        engine
            .vector_upsert(
                "docs",
                &format!("vec-{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 0.0, 1.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    drop(engine);

    let manifest_path = std::path::Path::new(&data_dir).join("vectors/docs/manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["ivf_clusters"].as_u64().unwrap(), 2);
    assert_eq!(manifest["ivf_nprobe"].as_u64().unwrap(), 1);
    assert_eq!(manifest["q8_refine_topk"].as_u64().unwrap(), 1);
    assert!(manifest["ivf_min_train_vectors"].as_u64().unwrap() >= 2);
    assert!(manifest["ivf_retrain_min_vectors"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn ivf_retrain_updates_manifest_and_results() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.ivf_min_train_vectors = 4;
    config.ivf_retrain_min_deltas = 2;
    config.ivf_clusters = 2;
    config.ivf_nprobe = 1;

    let engine = Engine::new(config.clone()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    for idx in 0..16 {
        engine
            .vector_upsert(
                "docs",
                &format!("v{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 1.0, 0.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    let before = engine
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![8.0, 1.0, 0.0],
                k: 2,
                filters: None,
                include_meta: Some(false),
            },
        )
        .unwrap();
    engine.vector_retrain_ivf("docs", true).unwrap();
    let after = engine
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![8.0, 1.0, 0.0],
                k: 2,
                filters: None,
                include_meta: Some(false),
            },
        )
        .unwrap();
    assert_eq!(before.first().map(|h| &h.id), after.first().map(|h| &h.id));
    drop(engine);

    let manifest_path = std::path::Path::new(&data_dir).join("vectors/docs/manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let trained_at = manifest["centroids_trained_at_ms"].as_u64().unwrap();

    let engine2 = Engine::new(config).unwrap();
    for idx in 16..24 {
        engine2
            .vector_upsert(
                "docs",
                &format!("v{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 1.0, 0.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    drop(engine2);
    let manifest2: serde_json::Value =
        serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert!(manifest2["centroids_trained_at_ms"].as_u64().unwrap() >= trained_at);
    assert!(
        manifest2["ivf_last_trained_upsert"].as_u64().unwrap()
            >= manifest["ivf_last_trained_upsert"].as_u64().unwrap()
    );
}

#[cfg_attr(
    not(feature = "ivf_stress_tests"),
    ignore = "enable `--features ivf_stress_tests` to run the 1M-vector stress test"
)]
#[test]
fn ivf_large_dataset_retrain_consistent() {
    let mut settings = VectorSettings::default();
    settings.index_kind = IndexKind::IvfFlatQ8;
    settings.ivf.clusters = 128;
    settings.ivf.nprobe = 8;
    settings.ivf.min_train_vectors = 512;
    settings.ivf.retrain_min_deltas = 200_000;
    settings.q8_refine_topk = 128;
    settings.hnsw_fallback_enabled = false;
    let store = VectorStore::with_settings(settings);
    store.create_collection("big", 2, Metric::Cosine).unwrap();
    let total = 1_000_000u32;
    for i in 0..total {
        let v0 = ((i % 10_000) as f32) / 10_000.0;
        let v1 = (((i * 7) % 10_000) as f32) / 10_000.0;
        store
            .upsert(
                "big",
                &format!("vec-{i}"),
                VectorItem {
                    vector: vec![v0, v1],
                    meta: serde_json::Value::Null,
                },
            )
            .unwrap();
    }
    let query = SearchRequest {
        vector: vec![0.123, 0.456],
        k: 5,
        filters: None,
        include_meta: Some(false),
    };
    let before = store.search("big", query.clone()).unwrap();
    assert!(!before.is_empty());
    store.retrain_ivf("big", true).unwrap();
    let after = store.search("big", query).unwrap();
    assert_eq!(before.first().map(|h| &h.id), after.first().map(|h| &h.id));
}
