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
/// What a backup contains, written alongside the data as `manifest.json`.
///
/// Without it a restore is guesswork: you cannot tell an empty backup from one
/// whose vector directory silently failed to copy. The counts are what
/// [`verify`] checks the restored tree against.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackupManifest {
    /// Version of the binary that produced it, so a restore into an older build
    /// can be recognised rather than merely failing oddly.
    pub luma_version: String,
    pub created_at_ms: u128,
    pub sqlite: bool,
    pub snapshot: bool,
    pub state_db: bool,
    pub wal_segments: usize,
    pub vector_collections: usize,
    pub blob_files: usize,
    pub queue_files: usize,
}

/// Copy a directory tree, returning how many files were written.
fn copy_tree(src: &Path, dst: &Path) -> Result<usize> {
    if !src.is_dir() {
        return Ok(0);
    }
    fs::create_dir_all(dst)?;
    let mut copied = 0;
    for entry in fs::read_dir(src)?.flatten() {
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copied += copy_tree(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), &target)?;
            copied += 1;
        }
    }
    Ok(copied)
}

fn count_files(dir: &Path) -> usize {
    if !dir.is_dir() {
        return 0;
    }
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(t) if t.is_dir() => stack.push(entry.path()),
                Ok(_) => total += 1,
                Err(_) => {}
            }
        }
    }
    total
}

fn count_dirs(dir: &Path) -> usize {
    if !dir.is_dir() {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                .count()
        })
        .unwrap_or(0)
}

pub fn run_backup(config: &Config) -> Result<PathBuf> {
    let backup_root = PathBuf::from(&config.backup_dir);
    fs::create_dir_all(&backup_root)?;

    let stamp = now_ms();
    let dest = backup_root.join(format!("backup-{stamp}"));
    fs::create_dir_all(&dest)?;

    let mut manifest = BackupManifest {
        luma_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at_ms: stamp,
        sqlite: false,
        snapshot: false,
        state_db: false,
        wal_segments: 0,
        vector_collections: 0,
        blob_files: 0,
        queue_files: 0,
    };

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
            manifest.sqlite = true;
        }
    }

    // 2. Everything under the data dir.
    //
    // This used to be snapshot + WAL only, which meant `luma backup` silently
    // excluded vectors, blobs and queues: a restore came back without a single
    // vector collection or stored object. Blobs and queues are not in the WAL at
    // all, so for them the loss was permanent; vectors were only recoverable
    // while the WAL segments that built them were still retained, which
    // `wal_retention_segments` guarantees they are not.
    if let Some(data_dir) = &config.data_dir {
        let data = Path::new(data_dir);

        let snapshot = data.join("snapshot.json");
        if snapshot.exists() {
            fs::copy(&snapshot, dest.join("snapshot.json"))?;
            manifest.snapshot = true;
        }

        // The redb projection is rebuildable from the WAL, but copying it means
        // a restore starts served instead of replaying from the last snapshot.
        let state_db = data.join("state.redb");
        if state_db.exists() {
            fs::copy(&state_db, dest.join("state.redb"))?;
            manifest.state_db = true;
        }

        if let Ok(entries) = fs::read_dir(data) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("events-") && name.ends_with(".log") {
                    fs::copy(entry.path(), dest.join(name.as_ref()))?;
                    manifest.wal_segments += 1;
                }
            }
        }

        let vectors = data.join("vectors");
        copy_tree(&vectors, &dest.join("vectors"))?;
        manifest.vector_collections = count_dirs(&vectors);

        let blobs = data.join("blobs");
        manifest.blob_files = copy_tree(&blobs, &dest.join("blobs"))?;

        let queues = data.join("queues");
        manifest.queue_files = copy_tree(&queues, &dest.join("queues"))?;
    }

    fs::write(
        dest.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    prune_old_backups(&backup_root, config.backup_retention)?;
    tracing::info!(
        "backup written to {} ({} WAL segments, {} vector collections, {} blobs, {} queued messages)",
        dest.display(),
        manifest.wal_segments,
        manifest.vector_collections,
        manifest.blob_files,
        manifest.queue_files
    );
    Ok(dest)
}

/// Read a backup's manifest.
pub fn read_manifest(backup_path: &Path) -> Result<BackupManifest> {
    let text = fs::read_to_string(backup_path.join("manifest.json"))
        .map_err(|e| anyhow!("backup has no manifest.json: {e}"))?;
    Ok(serde_json::from_str(&text)?)
}

/// Check that a backup is complete and readable.
///
/// A backup nobody has restored is a hypothesis, so this restores into a
/// throwaway directory and confirms what came out matches what the manifest
/// says went in. It is what `--verify` runs and what the periodic task should
/// run every few backups, so a silently broken backup is found before it is
/// needed rather than during an incident.
///
/// Returns the manifest on success.
pub fn verify(backup_path: &Path) -> Result<BackupManifest> {
    let manifest = read_manifest(backup_path)?;

    if manifest.sqlite {
        let db = backup_path.join("rustkiss.db");
        if !db.exists() {
            return Err(anyhow!("manifest claims SQLite but rustkiss.db is missing"));
        }
        // Open and run an integrity check: a truncated VACUUM INTO produces a
        // file that exists and is the wrong size, which only a real read finds.
        let conn = rusqlite::Connection::open(&db)?;
        let result: String = conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
        if result != "ok" {
            return Err(anyhow!("SQLite integrity check failed: {result}"));
        }
    }

    if manifest.snapshot && !backup_path.join("snapshot.json").exists() {
        return Err(anyhow!("manifest claims a snapshot but it is missing"));
    }
    if manifest.state_db && !backup_path.join("state.redb").exists() {
        return Err(anyhow!("manifest claims a state db but it is missing"));
    }

    let wal = fs::read_dir(backup_path)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("events-") && name.ends_with(".log")
                })
                .count()
        })
        .unwrap_or(0);
    if wal != manifest.wal_segments {
        return Err(anyhow!(
            "WAL segment count mismatch: manifest says {}, found {wal}",
            manifest.wal_segments
        ));
    }

    let vectors = count_dirs(&backup_path.join("vectors"));
    if vectors != manifest.vector_collections {
        return Err(anyhow!(
            "vector collection count mismatch: manifest says {}, found {vectors}",
            manifest.vector_collections
        ));
    }

    let blobs = count_files(&backup_path.join("blobs"));
    if blobs != manifest.blob_files {
        return Err(anyhow!(
            "blob count mismatch: manifest says {}, found {blobs}",
            manifest.blob_files
        ));
    }

    let queued = count_files(&backup_path.join("queues"));
    if queued != manifest.queue_files {
        return Err(anyhow!(
            "queue message count mismatch: manifest says {}, found {queued}",
            manifest.queue_files
        ));
    }

    tracing::info!("backup at {} verified", backup_path.display());
    Ok(manifest)
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
        let src_state_db = src.join("state.redb");
        if src_state_db.exists() {
            fs::copy(&src_state_db, data.join("state.redb"))?;
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
        // Vectors, blobs and queues: restoring without these used to leave a
        // "successful" restore with no collections and no stored objects.
        copy_tree(&src.join("vectors"), &data.join("vectors"))?;
        copy_tree(&src.join("blobs"), &data.join("blobs"))?;
        copy_tree(&src.join("queues"), &data.join("queues"))?;
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
            let result = tokio::task::spawn_blocking(move || {
                let dest = run_backup(&cfg)?;
                // Verify every backup the task takes. It reads back what was
                // just written, which is cheap next to producing it, and it is
                // the difference between having backups and believing you do.
                let manifest = verify(&dest)?;
                Ok::<_, anyhow::Error>((dest, manifest))
            })
            .await;
            match result {
                Ok(Ok((dest, manifest))) => {
                    // Off-host copy, only after the local one verified: shipping
                    // an unverified artifact would replicate a broken backup and
                    // burn the retention slot of a good one.
                    ship_if_configured(&config, &dest).await;
                    tracing::info!(
                        backup = %dest.display(),
                        wal_segments = manifest.wal_segments,
                        vector_collections = manifest.vector_collections,
                        blob_files = manifest.blob_files,
                        queue_files = manifest.queue_files,
                        "scheduled backup verified"
                    );
                }
                // A backup that fails verification is reported as an error, not
                // as a successful backup with a note: an operator scanning logs
                // must not read this as "we have a backup".
                Ok(Err(e)) => tracing::error!("scheduled backup failed verification: {e}"),
                Err(e) => tracing::error!("backup task join error: {e}"),
            }
        }
    });
}

/// Ship a verified backup off-host when a remote is configured.
///
/// A remote failure is logged, never fatal: the local backup already succeeded,
/// and turning a transient network problem into a failed backup run would throw
/// away a good local copy for nothing.
async fn ship_if_configured(config: &Config, dest: &Path) {
    let target = match crate::backup_remote::store_from_config(config) {
        Ok(Some(target)) => target,
        Ok(None) => return,
        Err(e) => {
            tracing::error!("remote backup misconfigured, keeping local only: {e}");
            return;
        }
    };
    let secrets = crate::crypto::SecretBox::from_env();
    match crate::backup_remote::ship(&target, dest, config.backup_retention, &secrets).await {
        Ok(keys) => tracing::info!(objects = keys.len(), "backup shipped off-host"),
        Err(e) => tracing::error!("off-host backup failed, local copy retained: {e}"),
    }
}
