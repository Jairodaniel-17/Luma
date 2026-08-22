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
    tokio::fs::create_dir_all(parent).await?;

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
