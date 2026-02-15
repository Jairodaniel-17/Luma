use luma::config::Config;
use luma::engine::Engine;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn snapshot_and_wal_replay_no_loss() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();

    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir.clone()),
        snapshot_interval_secs: 3600,
        event_buffer_size: 1000,
        live_broadcast_capacity: 1024,
        wal_segment_max_bytes: 256 * 1024,
        wal_retention_segments: 16,
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
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();

    for i in 0..200u32 {
        engine
            .put_state(format!("k:{i}"), serde_json::json!({ "i": i }), None, None)
            .unwrap();
    }

    engine.force_snapshot().unwrap();

    for i in 200..400u32 {
        engine
            .put_state(format!("k:{i}"), serde_json::json!({ "i": i }), None, None)
            .unwrap();
    }

    drop(engine);

    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
    for i in 0..400u32 {
        let item = engine2.get_state(&format!("k:{i}")).unwrap();
        assert_eq!(item.value["i"], i);
    }
}

#[tokio::test]
async fn state_survives_restart_without_wal() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();

    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir.clone()),
        snapshot_interval_secs: 3600,
        event_buffer_size: 1000,
        live_broadcast_capacity: 1024,
        wal_segment_max_bytes: 256 * 1024,
        wal_retention_segments: 16,
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
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    for i in 0..2000u32 {
        engine
            .put_state(
                format!("big:{i}"),
                serde_json::json!({ "i": i }),
                None,
                None,
            )
            .unwrap();
    }

    drop(engine);

    for entry in std::fs::read_dir(&data_dir).unwrap().flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with("events-") || name == "snapshot.json" {
            let _ = std::fs::remove_file(&path);
        }
    }

    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
    for i in 0..2000u32 {
        let item = engine2.get_state(&format!("big:{i}")).unwrap();
        assert_eq!(item.value["i"], i);
    }
}
