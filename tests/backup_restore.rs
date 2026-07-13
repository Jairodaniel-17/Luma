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
