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
        ..Config::default()
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
        ..Config::default()
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

#[tokio::test]
async fn group_commit_flushes_pending_events_on_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();

    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir.clone()),
        snapshot_interval_secs: 3600,
        wal_sync_mode: "group".to_string(),
        wal_batch_size: 128,
        wal_flush_interval_ms: 60_000,
        ..Config::default()
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    for i in 0..8u32 {
        engine
            .put_state(
                format!("pending:{i}"),
                serde_json::json!({ "i": i }),
                None,
                None,
            )
            .unwrap();
    }

    engine.shutdown();
    drop(engine);

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    for i in 0..8u32 {
        let item = reopened.get_state(&format!("pending:{i}")).unwrap();
        assert_eq!(item.value["i"], i);
    }
}

#[tokio::test]
async fn vector_state_replays_without_state_db_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();

    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, luma::vector::Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "doc-1",
            luma::vector::VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: serde_json::json!({"kind":"primary"}),
                mmap_offset: None,
            },
        )
        .unwrap();
    drop(engine);

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    let hit_ids: Vec<_> = reopened
        .vector_search(
            "docs",
            luma::vector::SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                k: 5,
                options: luma::vector::SearchOptions {
                    filters: None,
                    include_meta: false,
                    allowed_ids: None,
                },
            },
        )
        .unwrap()
        .into_iter()
        .map(|hit| hit.id)
        .collect();
    assert!(hit_ids.iter().any(|id| id == "doc-1"));
}

#[tokio::test]
async fn truncated_wal_tail_does_not_drop_valid_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();

    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir.clone()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .put_state("safe:1".to_string(), serde_json::json!({"v":1}), None, None)
        .unwrap();
    engine
        .put_state("safe:2".to_string(), serde_json::json!({"v":2}), None, None)
        .unwrap();
    drop(engine);

    let wal_path = std::fs::read_dir(&data_dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("events-"))
        })
        .unwrap();
    let mut wal = std::fs::read(&wal_path).unwrap();
    wal.extend_from_slice(b"{\"broken\":");
    std::fs::write(&wal_path, wal).unwrap();

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    assert_eq!(
        reopened.get_state("safe:1").unwrap().value,
        serde_json::json!({"v":1})
    );
    assert_eq!(
        reopened.get_state("safe:2").unwrap().value,
        serde_json::json!({"v":2})
    );
}
