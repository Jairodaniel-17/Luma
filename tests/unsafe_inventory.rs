//! `unsafe` stays where it is declared to be.
//!
//! W5.3 of `docs/PLAN-MAESTRO.md`. Every module in `src/lib.rs` except `vector`
//! carries `#[forbid(unsafe_code)]`, which makes an `unsafe` block anywhere else
//! a compile error rather than something a reviewer has to catch.
//!
//! That attribute is the real guarantee; this file guards the one hole it leaves.
//! A **new** `pub mod` added without the attribute compiles perfectly well, and
//! nothing else would notice — the protection would quietly stop covering the
//! newest code, which is exactly where it is most wanted.
//!
//! So this reads `src/lib.rs` and checks that every module is either marked or
//! listed here as a known exception. Adding an exception is a deliberate,
//! reviewable line; forgetting the attribute is not.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Modules allowed to contain `unsafe`, with the reason.
///
/// One entry. Extending this list means accepting `unsafe` in a new place, which
/// should be an argued decision rather than an oversight.
const EXCEPTIONS: &[(&str, &str)] = &[(
    "vector",
    "memory-mapped segment files and the SIMD dot products; 16 sites across \
     four files, inventoried in docs/SECURITY.md",
)];

fn repo(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// `(module, has_forbid)` for every `pub mod` in `src/lib.rs`.
fn declared_modules() -> Vec<(String, bool)> {
    let text = std::fs::read_to_string(repo("src/lib.rs")).expect("src/lib.rs must be readable");
    let lines: Vec<&str> = text.lines().collect();
    let mut modules = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("pub mod ") else {
            continue;
        };
        let Some(name) = rest.strip_suffix(';') else {
            // `pub mod foo { ... }` — an inline module, which this does not
            // handle. Fail loudly rather than skipping something unexamined.
            panic!("unexpected inline module declaration: {trimmed}");
        };
        // The attribute sits on the line above, possibly after doc comments.
        let mut marked = false;
        for previous in lines[..index].iter().rev() {
            let previous = previous.trim();
            if previous.contains("forbid(unsafe_code)") {
                marked = true;
                break;
            }
            // Doc comments and blank lines can sit between the attribute and the
            // declaration; anything else means there was no attribute.
            if previous.is_empty() || previous.starts_with("///") || previous.starts_with("//") {
                continue;
            }
            break;
        }
        modules.push((name.to_string(), marked));
    }
    modules
}

#[test]
fn every_module_either_forbids_unsafe_or_is_a_listed_exception() {
    let allowed: BTreeSet<&str> = EXCEPTIONS.iter().map(|(name, _)| *name).collect();
    let modules = declared_modules();
    assert!(
        modules.len() > 5,
        "only {} modules found; the parse must have broken",
        modules.len()
    );

    let unguarded: Vec<&String> = modules
        .iter()
        .filter(|(name, marked)| !marked && !allowed.contains(name.as_str()))
        .map(|(name, _)| name)
        .collect();

    assert!(
        unguarded.is_empty(),
        "these modules neither forbid unsafe nor are listed as exceptions in \
         tests/unsafe_inventory.rs: {unguarded:?}\n\
         Add `#[forbid(unsafe_code)]` above the declaration, or add an exception \
         with a reason if the module genuinely needs unsafe."
    );
}

#[test]
fn the_exceptions_are_the_only_modules_that_actually_use_unsafe() {
    // The list must not grow stale in the other direction either. A module that
    // stopped needing `unsafe` should lose its exception and gain the attribute,
    // or the exception becomes a licence nobody is using and nobody notices.
    for (module, _) in EXCEPTIONS {
        let uses = module_uses_unsafe(module);
        assert!(
            uses,
            "`{module}` is listed as needing unsafe but no longer contains any. \
             Remove the exception and add #[forbid(unsafe_code)]."
        );
    }
}

/// Whether any `.rs` file under `src/<module>` (or `src/<module>.rs`) uses
/// `unsafe` as a block, function, impl or trait.
///
/// Deliberately not a bare substring search: `--unsafe-bind`, an `unsafe-inline`
/// in a CSP header and the word in a comment all contain it, and counting those
/// would make the inventory a number nobody trusts.
fn module_uses_unsafe(module: &str) -> bool {
    let mut roots = vec![repo(&format!("src/{module}.rs"))];
    let directory = repo(&format!("src/{module}"));
    if directory.is_dir() {
        collect_rust_files(&directory, &mut roots);
    }
    roots.iter().any(|path| {
        std::fs::read_to_string(path)
            .map(|text| uses_unsafe(&text))
            .unwrap_or(false)
    })
}

fn collect_rust_files(directory: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `unsafe` followed by `{`, `fn`, `impl` or `trait` — the four forms that
/// actually introduce unsafety.
fn uses_unsafe(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut at = 0;
    while let Some(found) = text[at..].find("unsafe") {
        let start = at + found;
        at = start + "unsafe".len();
        // Not part of a longer identifier such as `unsafe_code`.
        let before_is_word = start
            .checked_sub(1)
            .map(|i| bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
            .unwrap_or(false);
        if before_is_word {
            continue;
        }
        let rest = text[at..].trim_start();
        if rest.starts_with('{')
            || rest.starts_with("fn ")
            || rest.starts_with("impl")
            || rest.starts_with("trait")
        {
            return true;
        }
    }
    false
}

#[test]
fn the_unsafe_detector_is_not_fooled_by_the_word_appearing_in_prose() {
    // The three shapes that made a naive substring count wrong in this very
    // repository.
    assert!(!uses_unsafe("let flag = \"--unsafe-bind\";"));
    assert!(!uses_unsafe("style-src 'self' 'unsafe-inline';"));
    assert!(!uses_unsafe("#[forbid(unsafe_code)]"));
    assert!(!uses_unsafe("// this is unsafe in the colloquial sense"));
    // And the four that are real.
    assert!(uses_unsafe("let x = unsafe { ptr.read() };"));
    assert!(uses_unsafe("unsafe fn danger() {}"));
    assert!(uses_unsafe("unsafe impl Send for X {}"));
    assert!(uses_unsafe("unsafe trait Marker {}"));
}
