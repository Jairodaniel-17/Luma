use luma::config::Config;
use luma::engine::Engine;
use tokio_util::sync::CancellationToken;
use luma::vector::index::DiskAnnBuildParams;
use luma::vector::{Metric, SearchRequest, VectorItem};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

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

fn first_run_file(data_dir: &str, collection: &str) -> Option<PathBuf> {
    let runs_dir = Path::new(data_dir)
        .join("vectors")
        .join(collection)
        .join("runs");
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(&runs_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                files.push(entry.path());
            }
        }
    }
    files.sort();
    files.into_iter().next()
}

fn list_run_files(data_dir: &str, collection: &str) -> Vec<PathBuf> {
    let runs_dir = Path::new(data_dir)
        .join("vectors")
        .join(collection)
        .join("runs");
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(&runs_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                files.push(entry.path());
            }
        }
    }
    files.sort();
    files
}

fn run_file_names(paths: &[PathBuf]) -> HashSet<String> {
    paths
        .iter()
        .filter_map(|p| p.file_name().map(|s| s.to_string_lossy().to_string()))
        .collect()
}

fn read_manifest_json(data_dir: &str, collection: &str) -> serde_json::Value {
    let manifest_path = Path::new(data_dir)
        .join("vectors")
        .join(collection)
        .join("manifest.json");
    let contents = fs::read_to_string(manifest_path).unwrap();
    serde_json::from_str(&contents).unwrap()
}

#[tokio::test]
async fn vector_persistence_restart_search() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
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

    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
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

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
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

    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
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

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
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

    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
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

#[tokio::test]
async fn vector_runs_tail_truncation_safe() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 4, Metric::Cosine)
        .unwrap();
    for idx in 0..32usize {
        engine
            .vector_upsert(
                "docs",
                &format!("id-{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 1.0, 0.0, 0.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    drop(engine);

    let run_file = first_run_file(&data_dir, "docs").expect("run file should exist");
    let len = fs::metadata(&run_file).unwrap().len();
    assert!(len > 64, "run file too small for truncation test");
    let mut file = OpenOptions::new().write(true).open(&run_file).unwrap();
    let new_len = len - 7;
    file.set_len(new_len).unwrap();
    file.flush().unwrap();
    drop(file);

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    assert!(reopened.vector_get("docs", "id-0").unwrap().is_some());
    let hits = reopened
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![1.0, 1.0, 0.0, 0.0],
                k: 5,
                filters: None,
                include_meta: Some(false),
            },
        )
        .unwrap();
    assert!(
        !hits.is_empty(),
        "truncation should not corrupt earlier data"
    );
}

#[tokio::test]
async fn vector_runs_checksum_detection() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 4, Metric::Cosine)
        .unwrap();
    for idx in 0..8usize {
        engine
            .vector_upsert(
                "docs",
                &format!("crc-{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 0.0, 1.0, 0.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    drop(engine);

    let run_file = first_run_file(&data_dir, "docs").expect("run file should exist");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&run_file)
        .unwrap();
    let len = file.metadata().unwrap().len();
    assert!(len > 32, "run file too small for corruption test");
    file.seek(SeekFrom::End(-1)).unwrap();
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 0xFF;
    file.seek(SeekFrom::End(-1)).unwrap();
    file.write_all(&byte).unwrap();
    file.flush().unwrap();
    drop(file);

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    assert!(
        reopened.vector_get("docs", "crc-0").unwrap().is_some(),
        "earlier records must still load"
    );
    assert!(
        reopened.vector_get("docs", "crc-7").unwrap().is_none(),
        "last record should be skipped after checksum failure"
    );
}

#[tokio::test]
async fn vector_q8_run_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.ivf_clusters = 2;
    config.ivf_nprobe = 1;

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "north",
            VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: json!({"dir": "north"}),
            },
        )
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "east",
            VectorItem {
                vector: vec![0.0, 1.0, 0.0],
                meta: json!({"dir": "east"}),
            },
        )
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "west",
            VectorItem {
                vector: vec![0.0, -1.0, 0.0],
                meta: json!({"dir": "west"}),
            },
        )
        .unwrap();
    drop(engine);

    let run_file = first_run_file(&data_dir, "docs").expect("run file should exist");
    assert!(run_file.exists());
    let legacy = Path::new(&data_dir)
        .join("vectors")
        .join("docs")
        .join("vectors.bin");
    if legacy.exists() {
        fs::remove_file(&legacy).unwrap();
    }

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    let hits = reopened
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
    assert_eq!(hits[0].id, "north");
    assert_eq!(hits[0].meta.as_ref().unwrap()["dir"], "north");
}

#[tokio::test]
async fn vector_run_retention_compacts_old_runs() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.run_target_bytes = 512;
    config.run_retention = 1;

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 6, Metric::Cosine)
        .unwrap();
    for idx in 0..200usize {
        engine
            .vector_upsert(
                "docs",
                &format!("doc-{idx}"),
                VectorItem {
                    vector: vec![
                        idx as f32,
                        1.0,
                        0.0,
                        (idx % 3) as f32,
                        0.0,
                        1.0 - (idx as f32 / 200.0),
                    ],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }

    let runs_before = list_run_files(&data_dir, "docs");
    assert!(
        runs_before.len() > config.run_retention,
        "expected retention trigger"
    );
    engine
        .vector_force_compact_collection("docs")
        .expect("retention compaction");
    let runs_after = list_run_files(&data_dir, "docs");
    let before_set = run_file_names(&runs_before);
    let after_set = run_file_names(&runs_after);
    let removed = before_set.difference(&after_set).count();
    assert!(removed > 0, "old run files should be removed");
    drop(engine);
    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    assert!(
        reopened.vector_get("docs", "doc-0").unwrap().is_some(),
        "data must survive retention compaction"
    );
}

#[tokio::test]
async fn vector_compaction_triggers_on_tombstones() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.run_target_bytes = 4 * 1024 * 1024;
    config.run_retention = 8;
    config.compaction_trigger_tombstone_ratio = 0.01;

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    for idx in 0..24usize {
        engine
            .vector_upsert(
                "docs",
                &format!("keep-{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 1.0, 0.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    let before = list_run_files(&data_dir, "docs");
    assert_eq!(before.len(), 1, "expected a single run initially");
    let before_name = before[0].file_name().unwrap().to_string_lossy().to_string();

    for idx in 0..16usize {
        engine
            .vector_delete("docs", &format!("keep-{idx}"))
            .unwrap();
    }
    drop(engine);

    let after = list_run_files(&data_dir, "docs");
    assert_eq!(after.len(), 1, "compaction should rewrite to one run");
    let after_name = after[0].file_name().unwrap().to_string_lossy().to_string();
    assert_ne!(
        before_name, after_name,
        "tombstone ratio trigger should rewrite runs"
    );

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    for idx in 0..16usize {
        assert!(
            reopened
                .vector_get("docs", &format!("keep-{idx}"))
                .unwrap()
                .is_none(),
            "deleted ids must not return after compaction"
        );
    }
    assert!(
        reopened.vector_get("docs", "keep-20").unwrap().is_some(),
        "live ids should remain"
    );
}

#[tokio::test]
async fn vector_manifest_settings_persisted() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.run_target_bytes = 2048;
    config.run_retention = 3;
    config.compaction_trigger_tombstone_ratio = 0.35;
    config.compaction_max_bytes_per_pass = 16 * 1024;

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 4, Metric::Cosine)
        .unwrap();
    for idx in 0..32usize {
        engine
            .vector_upsert(
                "docs",
                &format!("persist-{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 0.0, 1.0, 0.5],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    drop(engine);

    let manifest = read_manifest_json(&data_dir, "docs");
    assert_eq!(manifest["run_target_bytes"].as_u64().unwrap(), 2048);
    assert_eq!(manifest["run_retention"].as_u64().unwrap(), 3);
    assert!(
        (manifest["compaction_trigger_tombstone_ratio"]
            .as_f64()
            .unwrap()
            - 0.35)
            .abs()
            < 1e-6
    );
    assert_eq!(
        manifest["compaction_max_bytes_per_pass"].as_u64().unwrap(),
        16 * 1024
    );

    let mut config2 = config.clone();
    config2.run_target_bytes = 1024;
    let engine2 = Engine::new(config2.clone(), CancellationToken::new()).unwrap();
    engine2
        .vector_upsert(
            "docs",
            "extra",
            VectorItem {
                vector: vec![1.0, 1.0, 1.0, 1.0],
                meta: json!({ "state": "extra" }),
            },
        )
        .unwrap();
    drop(engine2);
    let manifest2 = read_manifest_json(&data_dir, "docs");
    assert_eq!(manifest2["run_target_bytes"].as_u64().unwrap(), 1024);
    let runs = list_run_files(&data_dir, "docs");
    assert!(
        runs.len() >= 2,
        "small run target should create multiple runs"
    );
}

#[tokio::test]
async fn vector_compaction_budget_multiple_passes() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.run_target_bytes = 512;
    config.run_retention = 32;
    config.compaction_trigger_tombstone_ratio = 0.1;
    config.compaction_max_bytes_per_pass = 600;

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    for idx in 0..90usize {
        engine
            .vector_upsert(
                "docs",
                &format!("keep-{idx}"),
                VectorItem {
                    vector: vec![idx as f32, 0.0, 1.0],
                    meta: json!({ "idx": idx }),
                },
            )
            .unwrap();
    }
    for idx in 0..60usize {
        engine
            .vector_delete("docs", &format!("keep-{idx}"))
            .unwrap();
    }

    let runs_before = list_run_files(&data_dir, "docs");
    assert!(runs_before.len() >= 3);

    let before_names = run_file_names(&runs_before);
    let first = engine.vector_force_compact_collection("docs").unwrap();
    assert!(first, "expected compaction to run");
    let runs_mid = list_run_files(&data_dir, "docs");
    let mid_names = run_file_names(&runs_mid);
    let removed_first = before_names.difference(&mid_names).count();
    assert!(removed_first > 0, "first pass should drop some runs");

    let second = engine.vector_force_compact_collection("docs").unwrap();
    assert!(second, "expected second compaction pass");
    let runs_after = list_run_files(&data_dir, "docs");
    let after_names = run_file_names(&runs_after);
    let removed_second = mid_names.difference(&after_names).count();
    assert!(
        removed_second > 0,
        "second pass should drop additional runs"
    );

    let remaining = engine.vector_get("docs", "keep-80").unwrap();
    assert!(remaining.is_some());
    let search_hits = engine
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![80.0, 0.0, 1.0],
                k: 1,
                filters: None,
                include_meta: Some(false),
            },
        )
        .unwrap();
    assert!(
        !search_hits.is_empty(),
        "expected hits after multi-pass compaction"
    );
}

#[tokio::test]
async fn vector_disk_index_manifest_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "a",
            VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: json!({"k": "a"}),
            },
        )
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "b",
            VectorItem {
                vector: vec![0.0, 1.0, 0.0],
                meta: json!({"k": "b"}),
            },
        )
        .unwrap();
    let params = DiskAnnBuildParams {
        max_degree: 32,
        build_threads: 2,
        search_list_size: 64,
    };
    engine
        .vector_build_disk_index("docs", params.clone())
        .unwrap();
    let status = engine.vector_disk_index_status("docs").unwrap();
    assert!(status.available, "disk index should report availability");
    assert!(
        !status.graph_files.is_empty(),
        "graph files must be recorded in status"
    );
    drop(engine);

    let manifest = read_manifest_json(&data_dir, "docs");
    assert_eq!(
        manifest["disk_index"]["graph_files"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(manifest["disk_index"]["build_params"]
        .get("max_degree")
        .is_some());
    let graph_path = manifest["disk_index"]["graph_files"][0].as_str().unwrap();
    let graph_full = Path::new(&data_dir)
        .join("vectors")
        .join("docs")
        .join(graph_path);
    assert!(graph_full.exists(), "disk graph file should exist");

    let reopened = Engine::new(config, CancellationToken::new()).unwrap();
    reopened.vector_drop_disk_index("docs").unwrap();
    let manifest2 = read_manifest_json(&data_dir, "docs");
    assert!(
        manifest2["disk_index"]["graph_files"]
            .as_array()
            .unwrap()
            .is_empty(),
        "dropping the disk index removes files from manifest"
    );
}

#[tokio::test]
async fn vector_diskann_search_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.index_kind = "DISKANN".to_string();

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "north",
            VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: json!({"dir": "north"}),
            },
        )
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "east",
            VectorItem {
                vector: vec![0.0, 1.0, 0.0],
                meta: json!({"dir": "east"}),
            },
        )
        .unwrap();
    engine
        .vector_upsert(
            "docs",
            "west",
            VectorItem {
                vector: vec![0.0, -1.0, 0.0],
                meta: json!({"dir": "west"}),
            },
        )
        .unwrap();
    let params = DiskAnnBuildParams {
        max_degree: 8,
        build_threads: 2,
        search_list_size: 64,
    };
    engine
        .vector_build_disk_index("docs", params)
        .expect("build disk index");
    drop(engine);

    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
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
    assert_eq!(hits[0].id, "north");
    assert_eq!(hits[0].meta.as_ref().unwrap()["dir"], "north");
}
