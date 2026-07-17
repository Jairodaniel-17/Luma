use luma::config::Config;
use luma::engine::Engine;
use luma::vector::index::DiskAnnBuildParams;
use luma::vector::{Metric, SearchOptions, SearchRequest, VectorItem};
use serde_json::json;
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

fn config_with_dir(dir: &str) -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string()),
        snapshot_interval_secs: 3600,
        wal_retention_segments: 16,
        ..Config::default()
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
                mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: true,
                    allowed_ids: None,
                },
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
                    mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: false,
                    allowed_ids: None,
                },
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: false,
                    allowed_ids: None,
                },
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
                    mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: false,
                    allowed_ids: None,
                },
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
                    mmap_offset: None,
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: true,
                    allowed_ids: None,
                },
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
                    mmap_offset: None,
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
                    mmap_offset: None,
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
                    mmap_offset: None,
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
                mmap_offset: None,
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
                    mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: false,
                    allowed_ids: None,
                },
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                mmap_offset: None,
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
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: true,
                    allowed_ids: None,
                },
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "north");
    assert_eq!(hits[0].meta.as_ref().unwrap()["dir"], "north");
}

#[tokio::test]
async fn vector_diskann_auto_builds_on_disk_graph() {
    // A DiskANN collection should build (and keep rebuilding) its on-disk graph
    // automatically as vectors are inserted, with NO explicit build call.
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.index_kind = "DISKANN".to_string();
    // Small thresholds so the test is fast: first build at 4 live vectors, and
    // rebuild on every subsequent upsert (delta gate = 1) so the graph stays
    // fresh through the last insert.
    config.diskann_auto_build_min_vectors = 4;
    config.diskann_rebuild_min_deltas = 1;

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 3, Metric::Cosine)
        .unwrap();

    let vectors = [
        ("north", [1.0f32, 0.0, 0.0]),
        ("east", [0.0, 1.0, 0.0]),
        ("west", [0.0, -1.0, 0.0]),
        ("up", [0.0, 0.0, 1.0]),
        ("down", [0.0, 0.0, -1.0]),
        ("south", [-1.0, 0.0, 0.0]),
    ];
    for (id, v) in vectors.iter() {
        engine
            .vector_upsert(
                "docs",
                id,
                VectorItem {
                    vector: v.to_vec(),
                    meta: json!({ "dir": id }),
                    mmap_offset: None,
                },
            )
            .unwrap();
    }

    // No explicit vector_build_disk_index call — the graph is built off the write
    // path by the background maintenance pass (which the engine also runs on a
    // timer). Drive it directly here so the assertion is deterministic.
    let built = engine.vectors().maintain_disk_indexes();
    assert_eq!(built, vec!["docs".to_string()]);
    let status = engine.vector_disk_index_status("docs").unwrap();
    assert!(
        status.available && !status.graph_files.is_empty(),
        "disk index should have auto-built without an explicit build call"
    );

    // Search still returns the correct top-1 for a known vector.
    let hits = engine
        .vector_search(
            "docs",
            SearchRequest {
                vector: vec![1.0, 0.0, 0.0],
                k: 1,
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: true,
                    allowed_ids: None,
                },
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "north");

    drop(engine);

    // The persisted manifest records the auto-build.
    let manifest = read_manifest_json(&data_dir, "docs");
    assert!(
        !manifest["disk_index"]["graph_files"]
            .as_array()
            .unwrap()
            .is_empty(),
        "manifest should record the auto-built graph files"
    );
    assert!(
        manifest["diskann_last_built_upsert"].as_u64().unwrap() > 0,
        "diskann_last_built_upsert marker should be set after auto-build"
    );
}

#[tokio::test]
async fn vector_diskann_background_build_off_write_path() {
    // The DiskANN graph (re)build runs off the write path: a build produces the
    // on-disk graph via the snapshot->build->swap maintenance pass, and writes
    // that happen *around* the build keep succeeding (the build never holds the
    // write lock for its full duration).
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let mut config = config_with_dir(&data_dir);
    config.index_kind = "DISKANN".to_string();
    config.diskann_auto_build_min_vectors = 10;
    config.diskann_rebuild_min_deltas = 1;

    let engine = Engine::new(config, CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("docs", 2, Metric::Cosine)
        .unwrap();

    // 50 vectors on a dense 1-D gradient from [0,1] to [1,0]. `grad-49` = [49, 0]
    // is the unique cosine top-1 for the query [1, 0], and the gradient is densely
    // connected so the approximate graph search reliably reaches it.
    let query = vec![1.0f32, 0.0];
    const N: u32 = 50;
    for i in 0..N {
        engine
            .vector_upsert(
                "docs",
                &format!("grad-{i:02}"),
                VectorItem {
                    vector: vec![i as f32, (N - 1 - i) as f32],
                    meta: json!({ "i": i }),
                    mmap_offset: None,
                },
            )
            .unwrap();
    }

    // Phase 1 (no concurrency): the maintenance pass must produce a graph.
    let built = engine.vectors().maintain_disk_indexes();
    assert_eq!(built, vec!["docs".to_string()]);
    assert!(
        engine.vector_disk_index_status("docs").unwrap().available,
        "graph should be available after the background build"
    );

    // Phase 2: inserts keep succeeding *while* builds run — the build does not
    // hold the write lock for its whole duration. Hammer inserts from another
    // thread while repeatedly kicking the build on this thread. The bg vectors sit
    // on the [1,1] diagonal (cosine 0.707 vs. the query), so they never outrank
    // `grad-49`.
    let writer = engine.clone();
    let handle = std::thread::spawn(move || {
        for i in 0..60u32 {
            writer
                .vector_upsert(
                    "docs",
                    &format!("bg-{i}"),
                    VectorItem {
                        vector: vec![(i % 5) as f32 + 1.0, (i % 5) as f32 + 1.0],
                        meta: json!({ "bg": i }),
                        mmap_offset: None,
                    },
                )
                .expect("concurrent insert should succeed during a build");
        }
    });
    for _ in 0..25 {
        engine.vectors().maintain_disk_indexes();
    }
    handle
        .join()
        .expect("writer thread should not panic/deadlock");

    // Every concurrent insert landed.
    let info = engine.vector_collection_info("docs").unwrap();
    assert_eq!(info.live_count, (N as usize) + 60);

    // A final build (no writers now) yields a fresh graph over all vectors, and
    // search returns the correct top-1 through the DiskANN path.
    let built = engine.vectors().maintain_disk_indexes();
    assert_eq!(built, vec!["docs".to_string()]);
    assert!(engine.vector_disk_index_status("docs").unwrap().available);
    let hits = engine
        .vector_search(
            "docs",
            SearchRequest {
                vector: query,
                k: 1,
                options: SearchOptions {
                    filters: None,
                    filter: None,
                    min_score: None,
                    include_meta: false,
                    allowed_ids: None,
                },
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "grad-49");
}

#[tokio::test]
async fn vector_collection_drop_persists_across_restart() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let config = config_with_dir(&data_dir);

    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .create_vector_collection("todrop", 3, Metric::Cosine)
        .unwrap();
    engine
        .vector_upsert(
            "todrop",
            "v1",
            VectorItem {
                vector: vec![1.0, 0.0, 0.0],
                meta: json!({}),
                mmap_offset: None,
            },
        )
        .unwrap();
    let col_dir = Path::new(&data_dir).join("vectors").join("todrop");
    assert!(col_dir.exists(), "collection dir should exist after create");

    // Drop it: gone from memory + disk.
    assert!(engine.drop_vector_collection("todrop").unwrap());
    assert!(engine.vector_collection_info("todrop").is_none());
    assert!(!col_dir.exists(), "collection dir should be deleted");
    // Dropping a nonexistent collection returns false.
    assert!(!engine.drop_vector_collection("todrop").unwrap());
    drop(engine);

    // After restart (WAL replay), the drop must stick — no resurrection.
    let engine2 = Engine::new(config, CancellationToken::new()).unwrap();
    assert!(
        engine2.vector_collection_info("todrop").is_none(),
        "dropped collection must not be resurrected by replay"
    );
}
