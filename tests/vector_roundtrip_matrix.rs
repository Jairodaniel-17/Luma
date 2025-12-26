use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::vector::{Metric, SearchRequest, VectorItem};

fn config_for(dir: &str) -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string()),
        snapshot_interval_secs: 3600,
        event_buffer_size: 1024,
        live_broadcast_capacity: 512,
        wal_segment_max_bytes: 256 * 1024,
        wal_retention_segments: 4,
        request_timeout_secs: 30,
        max_body_bytes: 1_048_576,
        max_key_len: 512,
        max_collection_len: 64,
        max_id_len: 128,
        max_vector_dim: 8192,
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
        ivf_clusters: 64,
        ivf_nprobe: 8,
        ivf_training_sample: 1024,
        ivf_min_train_vectors: 64,
        ivf_retrain_min_deltas: 32,
        q8_refine_topk: 256,
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
async fn vector_roundtrip_matrix_dims() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_for(&data_dir);
    let engine = Engine::new(config.clone()).unwrap();

    let dims = [8usize, 128, 384, 768];
    for &dim in &dims {
        let collection = format!("roundtrip_{dim}");
        engine
            .create_vector_collection(&collection, dim, Metric::Cosine)
            .unwrap();
        for idx in 0..32usize {
            let mut vector = vec![0.0f32; dim];
            vector[idx % dim] = 1.0;
            engine
                .vector_upsert(
                    &collection,
                    &format!("id-{idx}"),
                    VectorItem {
                        vector,
                        meta: serde_json::json!({ "dim": dim, "idx": idx }),
                    },
                )
                .unwrap();
        }
    }
    drop(engine);

    let reopened = Engine::new(config).unwrap();
    for &dim in &dims {
        let collection = format!("roundtrip_{dim}");
        let hits = reopened
            .vector_search(
                &collection,
                SearchRequest {
                    vector: vec![1.0; dim],
                    k: 5,
                    filters: None,
                    include_meta: Some(true),
                },
            )
            .unwrap();
        assert!(
            !hits.is_empty(),
            "expected hits for dim {dim} after restart"
        );
        assert!(
            hits.iter().all(|hit| hit.meta.as_ref().is_some()),
            "expected metadata for dim {dim}"
        );
    }
}
