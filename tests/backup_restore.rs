//! Verifies consistent backup + restore of the SQLite database.

use luma::config::Config;

#[test]
fn backup_then_restore_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let db_path = data_dir.join("rustkiss.db");

    // Seed a database with a known row.
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT); INSERT INTO t (v) VALUES ('hello');",
        )
        .unwrap();
    }

    let backup_dir = dir.path().join("backups");
    let config = Config {
        data_dir: Some(data_dir.to_string_lossy().to_string()),
        sqlite_path: Some(db_path.to_string_lossy().to_string()),
        backup_dir: backup_dir.to_string_lossy().to_string(),
        backup_retention: 3,
        ..Config::default()
    };

    // Back up.
    let dest = luma::backup::run_backup(&config).unwrap();
    assert!(dest.join("rustkiss.db").exists(), "backup contains the db");

    // Corrupt the live DB (drop the row) to prove restore recovers it.
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch("DELETE FROM t;").unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0, "row deleted");
    }

    // Restore from the backup directory.
    luma::backup::restore(&config, dest.to_str().unwrap()).unwrap();
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let v: String = conn
        .query_row("SELECT v FROM t LIMIT 1", [], |r| r.get(0))
        .unwrap();
    assert_eq!(v, "hello", "restore recovers the original row");
}

// ─── W1.4: what the backup actually covers ───────────────────────────────────

/// Build a data dir holding one of everything, the way a running server would.
fn populate(data: &std::path::Path) {
    use std::fs;
    fs::create_dir_all(data).unwrap();
    fs::write(data.join("snapshot.json"), b"{}").unwrap();
    fs::write(data.join("events-000001.log"), b"{}\n").unwrap();
    fs::write(data.join("events-000002.log"), b"{}\n").unwrap();
    fs::write(data.join("state.redb"), b"redb-bytes").unwrap();

    let collection = data.join("vectors").join("docs");
    fs::create_dir_all(collection.join("runs")).unwrap();
    fs::write(collection.join("manifest.json"), b"{\"dim\":4}").unwrap();
    fs::write(collection.join("runs").join("run-000001.log"), b"run").unwrap();

    fs::create_dir_all(data.join("blobs").join("assets").join("nested")).unwrap();
    fs::write(
        data.join("blobs")
            .join("assets")
            .join("nested")
            .join("o.bin"),
        b"object-bytes",
    )
    .unwrap();

    fs::create_dir_all(data.join("queues").join("jobs")).unwrap();
    fs::write(data.join("queues").join("jobs").join("m1.json"), b"{}").unwrap();
}

#[test]
fn backup_covers_vectors_blobs_and_queues() {
    // The regression this pins: `run_backup` used to copy SQLite, the snapshot
    // and the WAL only. Vectors, blobs and queues were silently excluded, so a
    // restore came back without a single collection or stored object — and
    // blobs and queues are not in the WAL at all, so for them the loss was
    // permanent rather than merely expensive.
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    populate(&data);

    let config = luma::config::Config {
        data_dir: Some(data.to_string_lossy().to_string()),
        sqlite_enabled: false,
        backup_dir: dir.path().join("backups").to_string_lossy().to_string(),
        backup_retention: 3,
        ..luma::config::Config::default()
    };

    let dest = luma::backup::run_backup(&config).unwrap();

    assert!(
        dest.join("vectors/docs/manifest.json").exists(),
        "vector collections must be in the backup"
    );
    assert!(
        dest.join("vectors/docs/runs/run-000001.log").exists(),
        "nested vector run files must be copied, not just the top level"
    );
    assert!(
        dest.join("blobs/assets/nested/o.bin").exists(),
        "blobs must be in the backup — they are not in the WAL, so this is the \
         only copy"
    );
    assert!(
        dest.join("queues/jobs/m1.json").exists(),
        "queued messages must be in the backup"
    );
    assert!(dest.join("state.redb").exists());

    let manifest = luma::backup::read_manifest(&dest).unwrap();
    assert_eq!(manifest.wal_segments, 2);
    assert_eq!(manifest.vector_collections, 1);
    assert_eq!(manifest.blob_files, 1);
    assert_eq!(manifest.queue_files, 1);
    assert!(!manifest.luma_version.is_empty());
}

#[test]
fn restore_brings_back_every_primitive() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    populate(&data);

    let config = luma::config::Config {
        data_dir: Some(data.to_string_lossy().to_string()),
        sqlite_enabled: false,
        backup_dir: dir.path().join("backups").to_string_lossy().to_string(),
        backup_retention: 3,
        ..luma::config::Config::default()
    };
    let dest = luma::backup::run_backup(&config).unwrap();

    // Destroy the live data entirely, the situation a restore exists for.
    std::fs::remove_dir_all(&data).unwrap();

    luma::backup::restore(&config, dest.to_str().unwrap()).unwrap();

    assert_eq!(
        std::fs::read(data.join("blobs/assets/nested/o.bin")).unwrap(),
        b"object-bytes"
    );
    assert_eq!(
        std::fs::read(data.join("vectors/docs/manifest.json")).unwrap(),
        b"{\"dim\":4}"
    );
    assert!(data.join("queues/jobs/m1.json").exists());
    assert!(data.join("events-000002.log").exists());
    assert!(data.join("state.redb").exists());
}

#[test]
fn verify_accepts_a_good_backup() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    populate(&data);
    let config = luma::config::Config {
        data_dir: Some(data.to_string_lossy().to_string()),
        sqlite_enabled: false,
        backup_dir: dir.path().join("backups").to_string_lossy().to_string(),
        backup_retention: 3,
        ..luma::config::Config::default()
    };
    let dest = luma::backup::run_backup(&config).unwrap();
    let manifest = luma::backup::verify(&dest).expect("a fresh backup must verify");
    assert_eq!(manifest.blob_files, 1);
}

#[test]
fn verify_catches_a_backup_that_lost_files() {
    // The whole point: a backup that looks present but is incomplete must be
    // caught now, not during an incident.
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    populate(&data);
    let config = luma::config::Config {
        data_dir: Some(data.to_string_lossy().to_string()),
        sqlite_enabled: false,
        backup_dir: dir.path().join("backups").to_string_lossy().to_string(),
        backup_retention: 3,
        ..luma::config::Config::default()
    };
    let dest = luma::backup::run_backup(&config).unwrap();

    std::fs::remove_file(dest.join("blobs/assets/nested/o.bin")).unwrap();
    let err = luma::backup::verify(&dest).unwrap_err().to_string();
    assert!(
        err.contains("blob count mismatch"),
        "verify must name what is missing, got: {err}"
    );
}

#[test]
fn verify_rejects_a_directory_with_no_manifest() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("not-a-backup")).unwrap();
    let err = luma::backup::verify(&dir.path().join("not-a-backup"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("manifest"), "got: {err}");
}

#[test]
fn verify_catches_a_missing_wal_segment() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    populate(&data);
    let config = luma::config::Config {
        data_dir: Some(data.to_string_lossy().to_string()),
        sqlite_enabled: false,
        backup_dir: dir.path().join("backups").to_string_lossy().to_string(),
        backup_retention: 3,
        ..luma::config::Config::default()
    };
    let dest = luma::backup::run_backup(&config).unwrap();

    std::fs::remove_file(dest.join("events-000001.log")).unwrap();
    let err = luma::backup::verify(&dest).unwrap_err().to_string();
    assert!(err.contains("WAL segment count mismatch"), "got: {err}");
}
