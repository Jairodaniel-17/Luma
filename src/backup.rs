//! Consistent on-disk backups of the SQLite database, snapshot, and WAL.
//!
//! A backup writes a timestamped directory under `backup_dir` containing:
//! - `rustkiss.db` — a consistent copy produced via SQLite `VACUUM INTO`
//! - `snapshot.json` — the latest engine snapshot (if present)
//! - `events-*.log` — the current WAL segments (if present)
//!
//! Old backups beyond `backup_retention` are pruned (oldest first).
//!
//! `luma backup` and `luma restore <path>` drive this from the CLI, and an
//! opt-in background task (`backup_enabled`) runs it on an interval.

use crate::config::Config;
use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Resolve the SQLite database file path from config (mirrors `server::init_sqlite`).
pub fn sqlite_db_path(config: &Config) -> Option<PathBuf> {
    config.sqlite_path.clone().map(PathBuf::from).or_else(|| {
        config
            .data_dir
            .as_ref()
            .map(|d| PathBuf::from(format!("{d}/sqlite/rustkiss.db")))
    })
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Perform a single backup. Returns the created backup directory.
pub fn run_backup(config: &Config) -> Result<PathBuf> {
    let backup_root = PathBuf::from(&config.backup_dir);
    fs::create_dir_all(&backup_root)?;

    let stamp = now_ms();
    let dest = backup_root.join(format!("backup-{stamp}"));
    fs::create_dir_all(&dest)?;

    // 1. SQLite consistent copy via VACUUM INTO.
    if let Some(db_path) = sqlite_db_path(config) {
        if db_path.exists() {
            let target = dest.join("rustkiss.db");
            // Use a short-lived direct connection so we don't disturb the actor.
            let conn = rusqlite::Connection::open(&db_path)?;
            // VACUUM INTO does not accept bound params for the path in all
            // versions; the path is server-generated (not user input) so a
            // single-quote escape is safe here.
            let target_str = target.to_string_lossy().replace('\'', "''");
            conn.execute_batch(&format!("VACUUM INTO '{target_str}'"))?;
        }
    }

    // 2. Snapshot + WAL segments from the data dir.
    if let Some(data_dir) = &config.data_dir {
        let data = Path::new(data_dir);
        let snapshot = data.join("snapshot.json");
        if snapshot.exists() {
            fs::copy(&snapshot, dest.join("snapshot.json"))?;
        }
        if let Ok(entries) = fs::read_dir(data) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("events-") && name.ends_with(".log") {
                    let _ = fs::copy(entry.path(), dest.join(name.as_ref()));
                }
            }
        }
    }

    prune_old_backups(&backup_root, config.backup_retention)?;
    tracing::info!("backup written to {}", dest.display());
    Ok(dest)
}

/// Keep only the `retention` most recent `backup-*` directories.
fn prune_old_backups(root: &Path, retention: usize) -> Result<()> {
    if retention == 0 {
        return Ok(());
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(root)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("backup-"))
                    .unwrap_or(false)
        })
        .collect();
    // Names embed a monotonic millisecond stamp, so lexicographic sort = chronological.
    dirs.sort();
    if dirs.len() > retention {
        for old in &dirs[..dirs.len() - retention] {
            let _ = fs::remove_dir_all(old);
            tracing::info!("pruned old backup {}", old.display());
        }
    }
    Ok(())
}

/// Restore a backup directory back into the live data locations.
///
/// This overwrites the SQLite database, snapshot, and WAL segments. The server
/// should be stopped before restoring.
pub fn restore(config: &Config, backup_path: &str) -> Result<()> {
    let src = PathBuf::from(backup_path);
    if !src.is_dir() {
        return Err(anyhow!("backup path is not a directory: {backup_path}"));
    }

    if let Some(db_path) = sqlite_db_path(config) {
        let src_db = src.join("rustkiss.db");
        if src_db.exists() {
            if let Some(parent) = db_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&src_db, &db_path)?;
            tracing::info!("restored SQLite database to {}", db_path.display());
        }
    }

    if let Some(data_dir) = &config.data_dir {
        let data = Path::new(data_dir);
        fs::create_dir_all(data)?;
        let src_snap = src.join("snapshot.json");
        if src_snap.exists() {
            fs::copy(&src_snap, data.join("snapshot.json"))?;
        }
        if let Ok(entries) = fs::read_dir(&src) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("events-") && name.ends_with(".log") {
                    let _ = fs::copy(entry.path(), data.join(name.as_ref()));
                }
            }
        }
    }

    tracing::info!("restore from {} complete", src.display());
    Ok(())
}

/// Spawn the opt-in periodic backup task. No-op unless `backup_enabled`.
pub fn spawn_backup_task(config: Config) {
    if !config.backup_enabled {
        return;
    }
    let interval = config.backup_interval_secs.max(60);
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(interval));
        // Skip the immediate first tick; back up after one interval.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let cfg = config.clone();
            let result = tokio::task::spawn_blocking(move || run_backup(&cfg)).await;
            match result {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => tracing::error!("scheduled backup failed: {e}"),
                Err(e) => tracing::error!("backup task join error: {e}"),
            }
        }
    });
}
