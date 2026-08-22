//! Durable file writes shared by every primitive that commits to disk.
//!
//! The rule this module exists to enforce: **a write that has been confirmed to
//! a caller must survive a crash.** `write` + `rename` alone does not give that
//! — both the file's data and the directory entry created by the rename can sit
//! in the page cache when the machine loses power. Two flushes are needed, and
//! getting them right differs per platform, so the knowledge lives here once
//! rather than in each caller.
//!
//! See `docs/operar/PROD_READINESS.md` for what each primitive guarantees when it
//! returns OK.

use std::io;
use std::path::Path;

/// Flush a directory entry so a preceding `rename` survives a crash.
///
/// Unix opens the directory read-only and fsyncs the handle.
///
/// Windows can do neither: a plain `File::open` on a directory fails with
/// `PermissionDenied`, and even with `FILE_FLAG_BACKUP_SEMANTICS` — which does
/// yield a handle — `FlushFileBuffers` on that handle is rejected the same way.
/// That is the platform saying the operation does not exist, not an I/O failure,
/// so it must not fail the write that asked for it. The durability of the
/// rename's directory entry then rests on NTFS metadata journaling instead of on
/// a flush we perform, which is why Linux is the deployment target.
#[cfg(not(windows))]
pub fn fsync_dir(dir: &Path) -> io::Result<()> {
    let handle = std::fs::File::open(dir)?;
    match handle.sync_all() {
        Ok(()) => Ok(()),
        // Some filesystems reject fsync on a directory handle.
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(windows)]
pub fn fsync_dir(dir: &Path) -> io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;
    // Required to obtain a handle to a directory at all on Windows.
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

    let handle = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(dir)?;
    match handle.sync_all() {
        Ok(()) => Ok(()),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::InvalidInput | io::ErrorKind::PermissionDenied
            ) =>
        {
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Replace a file's contents **atomically but not durably**.
///
/// Temp file plus rename, so no reader ever sees a half-written record — but no
/// fsync, so a crash may leave the previous contents. That is the right trade
/// for exactly one kind of value: state whose loss is already a permitted
/// outcome.
///
/// The case it exists for is a queue's visibility deadline. It is a *lease*: if
/// a crash loses it, the message becomes visible again and is redelivered, which
/// is the at-least-once contract the queue already offers. Paying two fsyncs per
/// message to durably record a lease whose loss is allowed is pure cost — and it
/// was measurable, not theoretical: `receive` on a few dozen messages did ~74
/// fsyncs and intermittently took longer than the crash-recovery test's
/// five-second timeout on a CI runner.
///
/// **Not for data.** A confirmed enqueue, a stored object, a WAL record: those
/// use `write_atomic`. The question to ask is whether losing this write is a
/// state the system already handles; if the answer needs a moment's thought, the
/// answer is no.
pub async fn replace_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "destination has no parent directory",
        )
    })?;
    create_dir_all_durable(parent).await?;

    let tmp = parent.join(format!(".tmp-{}", uuid::Uuid::new_v4()));
    let result = async {
        tokio::fs::write(&tmp, bytes).await?;
        tokio::fs::rename(&tmp, path).await
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

/// Create a directory and every missing ancestor, durably.
///
/// `create_dir_all` is not enough, and the gap is not theoretical: step 4 of
/// `write_atomic` fsyncs the file's own directory, which makes the file's entry
/// durable *in that directory*. It says nothing about whether that directory's
/// own entry reached the disk. If `create_dir_all` just made three levels —
/// `queues/`, `queues/t_acme/`, `queues/t_acme/jobs/` — a crash can take all
/// three, and every message inside them, while the enqueue has already returned
/// OK to the producer.
///
/// It hid on Windows and showed on Linux, which is the usual shape for this
/// class of bug: NTFS journals directory metadata aggressively enough that the
/// entries were there anyway.
///
/// So each newly created level is fsynced **in its parent**, top down. Levels
/// that already existed are left alone — their entries are already durable, and
/// fsyncing the whole chain on every write would be a syscall per level per
/// message.
pub async fn create_dir_all_durable(dir: &Path) -> io::Result<()> {
    // Nothing to do for a directory that is already there, which is the common
    // case: this runs on the write path of every queued message.
    if tokio::fs::metadata(dir).await.is_ok() {
        return Ok(());
    }

    // Top down, so a parent always exists before its child is created.
    let mut ancestors: Vec<&Path> = dir.ancestors().collect();
    ancestors.reverse();

    for level in ancestors {
        if tokio::fs::metadata(level).await.is_ok() {
            continue;
        }
        match tokio::fs::create_dir(level).await {
            Ok(()) => {}
            // Another task created it between the check and the call. Its
            // creator is responsible for the fsync.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
        let Some(parent) = level.parent().map(|p| p.to_path_buf()) else {
            continue;
        };
        tokio::task::spawn_blocking(move || fsync_dir(&parent))
            .await
            .map_err(|e| io::Error::other(format!("fsync_dir task failed: {e}")))??;
    }
    Ok(())
}

/// Write `bytes` to `path` atomically **and durably**.
///
/// The sequence, and why each step is there:
///
/// 1. write into a unique temp file in the same directory — same directory so
///    the rename is atomic (a cross-device rename is not);
/// 2. `sync_all` the temp file, so its contents are on the medium before
///    anything points at them. Without this a crash can leave the final name
///    resolving to a zero-length or partially written file, which is worse than
///    the write never having happened;
/// 3. rename over the destination, which is atomic;
/// 4. fsync the directory, so the rename itself is durable.
///
/// A failure at any point removes the temp file and leaves the destination as it
/// was. Returns `io::Result` rather than a domain error so callers can map it to
/// whatever their surface needs.
pub async fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "destination has no parent directory",
        )
    })?;
    create_dir_all_durable(parent).await?;

    let tmp = parent.join(format!(".tmp-{}", uuid::Uuid::new_v4()));
    match write_and_commit(&tmp, path, parent, bytes).await {
        Ok(()) => Ok(()),
        Err(e) => {
            // Best-effort cleanup: leaving a .tmp-* behind is untidy but must
            // not mask the original error.
            let _ = tokio::fs::remove_file(&tmp).await;
            Err(e)
        }
    }
}

async fn write_and_commit(
    tmp: &Path,
    final_path: &Path,
    parent: &Path,
    bytes: &[u8],
) -> io::Result<()> {
    {
        let mut file = tokio::fs::File::create(tmp).await?;
        tokio::io::AsyncWriteExt::write_all(&mut file, bytes).await?;
        // Data and metadata of the temp file, before anything points at it.
        file.sync_all().await?;
    }
    tokio::fs::rename(tmp, final_path).await?;

    // fsync_dir is blocking, so it goes to the blocking pool rather than
    // stalling a Tokio worker for the duration of a disk flush.
    let parent = parent.to_path_buf();
    tokio::task::spawn_blocking(move || fsync_dir(&parent))
        .await
        .map_err(|e| io::Error::other(format!("fsync_dir task failed: {e}")))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_lease_replacement_is_atomic_and_leaves_no_temp_file() {
        // Not durable, but a reader must never see a half-written record, and a
        // failed or interrupted replacement must not litter the directory with
        // `.tmp-*` files that the queue's own listing would then have to skip.
        let root = tmp_dir("lease_replace");
        std::fs::create_dir_all(&root).unwrap();
        let target = root.join("message.json");

        write_atomic(&target, b"{\"visible_at\":1}").await.unwrap();
        replace_atomic(&target, b"{\"visible_at\":2}")
            .await
            .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"{\"visible_at\":2}");

        let leftovers: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_lease_replacement_creates_its_directory_too() {
        // The first receive on a queue whose directory was never written to has
        // to work, not fail on a missing parent.
        let root = tmp_dir("lease_new_dir");
        let target = root.join("queues").join("jobs").join("m.json");
        replace_atomic(&target, b"{}").await.unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"{}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn a_write_creates_every_missing_level_of_its_path() {
        // Three levels at once is exactly the queue case: `queues/t_acme/jobs/`
        // did not exist, and the enqueue that made it returned OK to the
        // producer. Whether each level's entry is durable is what a crash
        // decides; this at least fixes that the whole chain is created and
        // written through, which is what regressed the moment `create_dir_all`
        // was doing it in one unchecked call.
        let root = tmp_dir("deep_chain");
        let nested = root.join("queues").join("t_acme").join("jobs");
        let target = nested.join("message.json");

        write_atomic(&target, b"{}").await.expect("write");

        assert!(nested.is_dir(), "every level must exist");
        assert_eq!(std::fs::read(&target).unwrap(), b"{}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn creating_a_directory_that_already_exists_is_not_an_error() {
        // The common case on the write path: this runs for every queued
        // message, and all but the first find the directory already there.
        let root = tmp_dir("existing_chain");
        std::fs::create_dir_all(root.join("a").join("b")).unwrap();

        create_dir_all_durable(&root.join("a").join("b"))
            .await
            .expect("an existing directory is fine");
        // And a partially existing chain: `a/b` is there, `a/b/c` is not.
        create_dir_all_durable(&root.join("a").join("b").join("c"))
            .await
            .expect("a partial chain is fine");
        assert!(root.join("a").join("b").join("c").is_dir());
        let _ = std::fs::remove_dir_all(&root);
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("luma_durability_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn fsync_dir_succeeds_on_a_real_directory() {
        // The regression this guards: on Windows the previous implementation
        // used a plain File::open, which fails with PermissionDenied and made
        // every durable write fail.
        let dir = tmp_dir("fsync_ok");
        fsync_dir(&dir).expect("fsync_dir must succeed on every supported platform");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fsync_dir_reports_a_missing_directory() {
        let dir = tmp_dir("fsync_missing");
        let missing = dir.join("nope");
        assert!(
            fsync_dir(&missing).is_err(),
            "a missing directory is a real error, not an unsupported operation"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_atomic_roundtrip_and_overwrite() {
        let dir = tmp_dir("write_atomic");
        let path = dir.join("nested").join("value.bin");

        write_atomic(&path, b"first").await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");

        // Overwriting must land the new content, not append or fail.
        write_atomic(&path, b"second-and-longer").await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second-and-longer");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_atomic_leaves_no_temp_files_behind() {
        let dir = tmp_dir("write_atomic_clean");
        let path = dir.join("v");
        write_atomic(&path, b"x").await.unwrap();

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with(".tmp-"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn write_atomic_rejects_a_path_with_no_parent() {
        // A bare root path cannot be committed atomically because there is no
        // directory to place the temp file in or to fsync afterwards.
        let err = write_atomic(Path::new("/"), b"x").await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
