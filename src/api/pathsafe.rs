//! Containment check for caller-supplied path fragments.
//!
//! Blob keys, queue names and image keys all become filesystem paths, so each
//! one has to be proven to stay inside its root before it is touched. This lived
//! as three identical copies in `routes_blob`, `routes_queue` and
//! `routes_image`; they are one function now, because a containment check with
//! three copies is a containment check that will eventually be fixed in two
//! places.

use std::path::{Component, Path};

/// Reason a candidate path was refused. The caller maps it to its own error
/// type and wording (blob root vs queue root).
#[derive(Debug, PartialEq, Eq)]
pub enum PathRejection {
    /// The caller-supplied portion contained something other than plain
    /// segments — `..`, a root, or a drive prefix.
    Traversal,
    /// The candidate is not under the root at all.
    Escapes,
}

/// Verify that `candidate` is `root` plus plain path segments.
///
/// The check applies to the portion **below** the root, not to the whole path.
/// That distinction is the entire point: `candidate` is built by joining onto
/// `root`, so when `root` is absolute it legitimately contributes a `RootDir`
/// component (and a `Prefix` such as `C:` on Windows). Scanning the full path
/// and rejecting every non-`Normal` component therefore rejected *every*
/// request whenever `data_dir` was an absolute path — which is exactly what the
/// shipped Docker setup uses (`DATA_DIR: /data`), so object storage, queues and
/// image transforms all returned "path traversal detected" for every call there
/// while working fine under a relative `data_dir`.
///
/// Below the root, only `Normal` components are allowed. `ParentDir` (`..`) is
/// the traversal this exists to stop; `RootDir`, `Prefix` and `CurDir` cannot
/// legitimately appear in a segment the caller supplied.
pub fn ensure_within_root(root: &Path, candidate: &Path) -> Result<(), PathRejection> {
    let relative = candidate
        .strip_prefix(root)
        .map_err(|_| PathRejection::Escapes)?;
    for comp in relative.components() {
        match comp {
            Component::Normal(_) => {}
            _ => return Err(PathRejection::Traversal),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn accepts_a_plain_key_under_a_relative_root() {
        let root = PathBuf::from("data/blobs");
        let candidate = root.join("assets").join("logo.png");
        assert_eq!(ensure_within_root(&root, &candidate), Ok(()));
    }

    #[test]
    fn accepts_a_plain_key_under_an_absolute_root() {
        // The regression this pins: with `DATA_DIR=/data` every blob, queue and
        // image request was refused as traversal, because the absolute root's
        // own RootDir component was being counted against the caller.
        let root = if cfg!(windows) {
            PathBuf::from(r"C:\data\blobs")
        } else {
            PathBuf::from("/data/blobs")
        };
        let candidate = root.join("assets").join("logo.png");
        assert_eq!(
            ensure_within_root(&root, &candidate),
            Ok(()),
            "an absolute data_dir must not make every request look like traversal"
        );
    }

    #[test]
    fn accepts_nested_segments() {
        let root = PathBuf::from("/data/blobs");
        let candidate = root.join("assets").join("2026").join("08").join("logo.png");
        assert_eq!(ensure_within_root(&root, &candidate), Ok(()));
    }

    #[test]
    fn accepts_the_root_itself() {
        let root = PathBuf::from("/data/queues");
        assert_eq!(ensure_within_root(&root, &root), Ok(()));
    }

    #[test]
    fn rejects_parent_dir_below_the_root() {
        let root = PathBuf::from("/data/blobs");
        let candidate = root.join("assets").join("..").join("..").join("etc");
        assert_eq!(
            ensure_within_root(&root, &candidate),
            Err(PathRejection::Traversal),
            ".. below the root is the traversal this must stop"
        );
    }

    #[test]
    fn rejects_a_single_parent_dir() {
        let root = PathBuf::from("/data/blobs");
        assert_eq!(
            ensure_within_root(&root, &root.join("..")),
            Err(PathRejection::Traversal)
        );
    }

    #[test]
    fn rejects_a_sibling_of_the_root() {
        let root = PathBuf::from("/data/blobs");
        assert_eq!(
            ensure_within_root(&root, Path::new("/data/queues/x")),
            Err(PathRejection::Escapes)
        );
    }

    #[test]
    fn rejects_a_prefix_match_that_is_not_a_path_ancestor() {
        // `/data/blobs-evil` starts with the root's *string* but is not under
        // the root directory. strip_prefix compares components, not bytes,
        // which is why this is refused.
        let root = PathBuf::from("/data/blobs");
        assert_eq!(
            ensure_within_root(&root, Path::new("/data/blobs-evil/x")),
            Err(PathRejection::Escapes)
        );
    }

    #[test]
    fn rejects_an_absolute_candidate_under_a_relative_root() {
        let root = PathBuf::from("data/blobs");
        assert_eq!(
            ensure_within_root(&root, Path::new("/etc/passwd")),
            Err(PathRejection::Escapes)
        );
    }

    #[cfg(windows)]
    #[test]
    fn rejects_a_drive_relative_escape_on_windows() {
        let root = PathBuf::from(r"C:\data\blobs");
        assert_eq!(
            ensure_within_root(&root, Path::new(r"D:\evil")),
            Err(PathRejection::Escapes)
        );
    }
}
