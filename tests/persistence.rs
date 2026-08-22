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
        assert_eq!(item.value.get("i"), Some(&serde_json::json!(i)));
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
        assert_eq!(item.value.get("i"), Some(&serde_json::json!(i)));
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
        assert_eq!(item.value.get("i"), Some(&serde_json::json!(i)));
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
                    filter: None,
                    min_score: None,
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
async fn vectors_survive_restart_with_compact_event_wal() {
    // Write dim-768 vectors, confirm the event WAL stores each vector as a compact
    // base64 blob (not a fat JSON number array), then reopen from the same data dir
    // and assert every upsert replayed and is still searchable. This is the
    // crash-recovery/replay guarantee for the reduced write-amp encoding.
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

    let dim = 768usize;
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", dim, luma::vector::Metric::Cosine)
        .unwrap();

    // Each doc gets a unique dominant dimension so it is unambiguously its own
    // nearest neighbour (avoids ties between near-identical vectors), while still
    // carrying arbitrary non-power-of-two f32 values that exercise the encoding.
    let make_vec = |seed: usize| -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.013 + 0.17).fract())
            .collect();
        v[seed % dim] += 5.0;
        v
    };
    for n in 0..64usize {
        engine
            .vector_upsert(
                "docs",
                &format!("doc-{n}"),
                luma::vector::VectorItem {
                    vector: make_vec(n),
                    meta: serde_json::json!({"n": n}),
                    mmap_offset: None,
                },
            )
            .unwrap();
    }
    drop(engine);

    // Inspect the event WAL: it must use the compact base64 vector encoding, and
    // each dim-768 vector record must be far smaller than the ~15 KB JSON array.
    let wal_bytes: Vec<u8> = std::fs::read_dir(&data_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("events-"))
        })
        .flat_map(|p| std::fs::read(p).unwrap())
        .collect();
    let wal_text = String::from_utf8(wal_bytes).unwrap();
    assert!(
        wal_text.contains("vec_b64"),
        "event WAL must store vectors as compact base64 blobs"
    );
    let max_upsert_line = wal_text
        .lines()
        .filter(|l| l.contains("vector_upserted"))
        .map(|l| l.len())
        .max()
        .unwrap();
    assert!(
        max_upsert_line < 6000,
        "compact dim-768 upsert record should be well under 15 KB, got {max_upsert_line} bytes"
    );

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    for n in 0..64usize {
        let hit_ids: Vec<_> = reopened
            .vector_search(
                "docs",
                luma::vector::SearchRequest {
                    vector: make_vec(n),
                    k: 1,
                    options: luma::vector::SearchOptions {
                        filters: None,
                        filter: None,
                        min_score: None,
                        include_meta: false,
                        allowed_ids: None,
                    },
                },
            )
            .unwrap()
            .into_iter()
            .map(|hit| hit.id)
            .collect();
        assert!(
            hit_ids.iter().any(|id| id == &format!("doc-{n}")),
            "doc-{n} must survive restart and be searchable"
        );
    }
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
        reopened.get_state("safe:1").unwrap().value.as_json(),
        Some(&serde_json::json!({"v":1}))
    );
    assert_eq!(
        reopened.get_state("safe:2").unwrap().value.as_json(),
        Some(&serde_json::json!({"v":2}))
    );
}
