use rust_kiss_vdb::config::Config;
use rust_kiss_vdb::engine::Engine;
use rust_kiss_vdb::vector::{Metric, SearchRequest, VectorItem};
use serde_json::json;

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
    }
}

#[tokio::test]
async fn vector_persistence_restart_search() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "persisted",
            VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: json!({"tag": "persist"}),
            },
        )
        .unwrap();
    drop(engine);

    let engine2 = Engine::new(config).unwrap();
    let hits = engine2
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                k: 1,
                filters: None,
                include_meta: Some(true),
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "persisted");
    assert_eq!(hits[0].meta.as_ref().unwrap()["tag"], "persist");
}

#[tokio::test]
async fn vector_rebuild_handles_many_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone()).unwrap();
    engine
        .create_vector_collection("docs", 2, Metric::Cosine)
        .unwrap();
    for i in 0..64 {
        let weight = i as f32 / 63_f32.max(1.0);
        engine
            .vector_upsert(
                "docs",
                &format!("id{i}"),
                VectorItem {
                    vector: vec![weight, 1.0 - weight],
                    meta: json!({ "i": i }),
                },
            )
            .unwrap();
    }
    drop(engine);

    let engine2 = Engine::new(config).unwrap();
    let hits = engine2
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![0.72, 0.28],
                k: 3,
                filters: None,
                include_meta: Some(false),
            },
        )
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].id.starts_with("id"));
}

#[tokio::test]
async fn vector_delete_update_persisted() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "keep",
            VectorItem {
                vector: vec![0.0, 1.0, 0.0],
                meta: json!({"state": "keep"}),
            },
        )
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "gone",
            VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: json!({"state": "gone"}),
            },
        )
        .unwrap();
    engine.vector_delete("docs", "gone").unwrap();
    engine
        .vector_update("docs", "keep", Some(vec![0.0, 0.0, 1.0]), None)
        .unwrap();
    drop(engine);

    let engine2 = Engine::new(config).unwrap();
    assert!(engine2.vector_get("docs", "gone").unwrap().is_none());
    let hits = engine2
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![0.0, 0.0, 1.0],
                k: 1,
                filters: None,
                include_meta: Some(false),
            },
        )
        .unwrap();
    assert_eq!(hits.first().map(|h| h.id.as_str()), Some("keep"));
}
