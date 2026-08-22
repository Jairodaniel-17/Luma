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
    // Deliberately absent: redb is a projection of the WAL, it is open and
    // mapped by the running engine, and copying it live is either a hard
    // failure (Windows) or a torn read (Linux). The restore rebuilds it.
    assert!(
        !dest.join("state.redb").exists(),
        "a live redb must not be copied into the backup"
    );

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
    // Not restored either — it is rebuilt from the WAL on next start.
    assert!(data.join("events-000002.log").exists());
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

// ─── F4.2: structures must survive a backup/restore cycle ────────────────────

#[test]
fn structures_survive_backup_and_restore() {
    // Structures are stored as values in the KV under a `struct:` prefix, so in
    // principle they ride along in the WAL and snapshot that the backup already
    // copies. "In principle" is exactly the kind of claim that turns out to be
    // false, so this proves it end to end rather than reasoning about it.
    use luma::engine::structures::{Structure, Structures};

    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let config = luma::config::Config {
        data_dir: Some(data.to_string_lossy().to_string()),
        sqlite_enabled: false,
        backup_dir: dir.path().join("backups").to_string_lossy().to_string(),
        backup_retention: 3,
        ..luma::config::Config::default()
    };

    // The token must be cancelled before the data dir can be removed: the
    // engine's background tasks hold redb open, and on Windows an open mapping
    // makes the directory undeletable rather than merely stale.
    let token = tokio_util::sync::CancellationToken::new();
    let dest = {
        let engine = luma::engine::Engine::new(config.clone(), token.clone()).unwrap();
        let structures = Structures::new(&engine);

        structures
            .mutate("jobs", Structure::empty_list, |s| {
                s.rpush(vec![b"first".to_vec(), vec![0x80, 0xFF]])
            })
            .unwrap();
        structures
            .mutate("unacked", Structure::empty_hash, |s| {
                s.hset(vec![(b"task-1".to_vec(), b"payload".to_vec())])
            })
            .unwrap();
        structures
            .mutate("due", Structure::empty_zset, |s| {
                s.as_zset_mut()?.add(b"job-a".to_vec(), 1.5)?;
                // An infinite score, because JSON has no infinity: this used to
                // serialize as `null` and made the whole sorted set unreadable
                // on the way back. A restore is exactly where that would first
                // be noticed, and far too late.
                s.as_zset_mut()?.add(b"never".to_vec(), f64::INFINITY)?;
                s.as_zset_mut()?
                    .add(b"always".to_vec(), f64::NEG_INFINITY)
                    .map(|_| ())
            })
            .unwrap();
        structures
            .mutate("tags", Structure::empty_set, |s| {
                s.sadd(vec![b"alpha".to_vec(), b"beta".to_vec()])
            })
            .unwrap();

        // Force a snapshot so the backup has something beyond the WAL to carry.
        let _ = engine.force_snapshot();
        let dest = luma::backup::run_backup(&config).unwrap();
        token.cancel();
        drop(engine);
        dest
    };
    luma::backup::verify(&dest).expect("the backup must verify");

    // Destroy the live data, the situation a restore exists for. Retried
    // briefly because the previous engine may still be releasing its handles.
    for attempt in 0..50 {
        match std::fs::remove_dir_all(&data) {
            Ok(()) => break,
            Err(e) if attempt == 49 => panic!("data dir never became removable: {e}"),
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
    luma::backup::restore(&config, dest.to_str().unwrap()).unwrap();

    let engine =
        luma::engine::Engine::new(config, tokio_util::sync::CancellationToken::new()).unwrap();
    let structures = Structures::new(&engine);

    let (list, _) = structures.load("jobs").unwrap().expect("list must survive");
    assert_eq!(
        list.lrange(0, -1).unwrap(),
        vec![b"first".to_vec(), vec![0x80, 0xFF]],
        "a binary list member must come back byte for byte"
    );

    let (hash, _) = structures
        .load("unacked")
        .unwrap()
        .expect("hash must survive");
    assert_eq!(
        hash.as_hash().unwrap().get(&b"task-1".to_vec()),
        Some(&b"payload".to_vec())
    );

    let (zset, _) = structures.load("due").unwrap().expect("zset must survive");
    let zset = zset.as_zset().unwrap();
    assert_eq!(
        zset.score(b"job-a"),
        Some(1.5),
        "a float score must survive the JSON round trip exactly"
    );
    assert_eq!(
        zset.score(b"never"),
        Some(f64::INFINITY),
        "an infinite score must survive too, or the whole sorted set is lost"
    );
    assert_eq!(zset.score(b"always"), Some(f64::NEG_INFINITY));

    let (set, _) = structures.load("tags").unwrap().expect("set must survive");
    let members = set.as_set().unwrap();
    assert!(members.contains(&b"alpha".to_vec()) && members.contains(&b"beta".to_vec()));
}
