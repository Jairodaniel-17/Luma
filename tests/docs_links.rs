//! Every relative link in the documentation points at a file that exists.
//!
//! W5.5 moved twenty documents into `empezar/`, `operar/`, `integrar/` and
//! `referencia/`. A move like that breaks links silently: the markdown still
//! renders, the link still looks like a link, and it only fails for whoever
//! clicks it — which is a reader who has already decided they need that page.
//!
//! So the reorganisation ships with the check that makes it safe to do again.
//! A broken link in an index is worse than no index at all: it sends somebody
//! looking for something that appears to be there.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Markdown files to check: everything under `docs/`, plus the two READMEs.
fn documents(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for name in ["README.md", "README.en.md", "CLAUDE.md", "SECURITY.md"] {
        let path = root.join(name);
        if path.is_file() {
            found.push(path);
        }
    }
    collect(&root.join("docs"), &mut found);
    found
}

fn collect(dir: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, into);
        } else if path.extension().is_some_and(|e| e == "md") {
            into.push(path);
        }
    }
}

/// Pull the targets out of `[text](target)` links.
///
/// Deliberately small. It skips anything with a scheme, anchors, and links
/// inside code fences — the goal is to catch a moved file, not to reimplement a
/// markdown parser.
fn links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let bytes: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != ']' || i + 1 >= bytes.len() || bytes[i + 1] != '(' {
                i += 1;
                continue;
            }
            let start = i + 2;
            let Some(end) = bytes[start..].iter().position(|&c| c == ')') else {
                break;
            };
            let target: String = bytes[start..start + end].iter().collect();
            out.push(target);
            i = start + end + 1;
        }
    }
    out
}

fn is_relative_file_link(target: &str) -> bool {
    let target = target.trim();
    if target.is_empty() || target.starts_with('#') {
        return false;
    }
    // A scheme of any kind, and protocol-relative URLs.
    if target.contains("://") || target.starts_with("//") || target.starts_with("mailto:") {
        return false;
    }
    // A reference-style or templated target is not a path.
    !target.contains('<') && !target.contains('{')
}

#[test]
fn every_relative_link_in_the_docs_resolves() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut broken: BTreeSet<String> = BTreeSet::new();
    let mut checked = 0;

    for document in documents(&root) {
        let Ok(text) = std::fs::read_to_string(&document) else {
            continue;
        };
        let base = document.parent().unwrap().to_path_buf();
        for target in links(&text) {
            if !is_relative_file_link(&target) {
                continue;
            }
            // Drop an anchor and any title after the path.
            let path_part = target
                .split(&[' ', '#'][..])
                .next()
                .unwrap_or_default()
                .to_string();
            if path_part.is_empty() {
                continue;
            }
            checked += 1;
            let resolved = base.join(&path_part);
            if !resolved.exists() {
                broken.insert(format!(
                    "{} → {}",
                    document.strip_prefix(&root).unwrap_or(&document).display(),
                    path_part
                ));
            }
        }
    }

    assert!(
        checked > 20,
        "only {checked} relative links were checked, which means the scan is not finding the \
         documentation — a test that checks nothing passes for the wrong reason"
    );
    assert!(
        broken.is_empty(),
        "these documentation links point at files that do not exist:\n  {}",
        broken.into_iter().collect::<Vec<_>>().join("\n  ")
    );
}

#[test]
fn the_index_lists_every_document_and_every_document_is_in_the_index() {
    // An index that drifts from the folder is how a page becomes unreachable
    // while still existing. Both directions, because each has its own failure:
    // a missing entry hides a page, and a stale entry sends somebody nowhere.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let index = std::fs::read_to_string(root.join("docs").join("README.md"))
        .expect("docs/README.md is the index W5.5 promised");

    let mut missing = Vec::new();
    for section in ["empezar", "operar", "integrar", "referencia"] {
        let dir = root.join("docs").join(section);
        let mut found = Vec::new();
        collect(&dir, &mut found);
        for document in found {
            let name = document.file_name().unwrap().to_string_lossy().to_string();
            let reference = format!("{section}/{name}");
            if !index.contains(&reference) {
                missing.push(reference);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "these documents exist but the index does not mention them: {missing:?}"
    );
}
