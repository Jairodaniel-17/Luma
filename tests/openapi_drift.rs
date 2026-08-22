//! The committed OpenAPI spec must describe the routes that actually exist.
//!
//! W3.3 of `docs/PLAN-MAESTRO.md`, in the form that delivers the benefit.
//!
//! ## What this is, and what it is not
//!
//! The plan asks for a spec **generated** from the code. This is not that: the
//! spec is still written by hand, and the schemas in it are not checked against
//! anything. What is checked is the path-and-method surface — every route the
//! router exposes appears in the spec, and every path the spec documents exists.
//!
//! That is the smaller claim, and it is worth being precise about, because it
//! covers the drift that actually bites. Its first run found:
//!
//! * `PUT /v1/auth/domain-orgs` documented as `POST`. A client following the
//!   documentation would get a 405 and have nothing to go on.
//! * `DELETE /v1/vector/{collection}` — a real operation, undocumented.
//! * `GET /v1/admin/resp` — added days earlier and never written down.
//!
//! Full generation (utoipa annotations on every handler) would also pin request
//! and response schemas. It is a large, mechanical change across roughly a
//! hundred handlers and it is **not done**; this guard is what stands in for it,
//! and saying so is better than letting the checkbox imply more than it covers.
//!
//! ## Why the router is read from source
//!
//! `axum::Router` does not expose its routes for introspection. The declarations
//! all live in one function, so reading them from the file is reliable — and the
//! alternative, maintaining a second hand-written list, would drift in exactly
//! the way this test exists to prevent.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const VERBS: &[&str] = &["get", "post", "put", "delete", "patch", "head"];

/// Routes that are not API surface and are deliberately absent from the spec.
///
/// The panel and the docs viewer are pages, not endpoints. Listing them
/// explicitly rather than pattern-matching on the path keeps the exclusion
/// auditable: a new `/v1/...` route can never fall through it by accident.
const NOT_API: &[(&str, &str)] = &[
    ("GET", "/"),
    ("GET", "/index.html"),
    ("GET", "/docs"),
    ("GET", "/docs/openapi.yaml"),
    ("GET", "/openapi.yaml"),
];

fn repo(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// Every `(VERB, path)` declared with `.route("path", verb(handler)…)`.
///
/// Scans to the balanced closing paren rather than matching one regex over the
/// whole call: handler lists are chained (`get(a).post(b)`) and wrap across
/// lines, so a lazy match stops at the first `)` inside the list and loses
/// every verb after it. That mistake made a first pass see 50 of 111 routes.
fn router_routes() -> BTreeSet<(String, String)> {
    let mut found = BTreeSet::new();
    for source in ["src/api/mod.rs", "src/api/routes_docs.rs"] {
        let text = std::fs::read_to_string(repo(source)).expect("router source must be readable");
        let bytes = text.as_bytes();
        let mut from = 0;
        while let Some(at) = text[from..].find(".route(") {
            let start = from + at + ".route(".len();
            from = start;
            // The quoted path.
            let Some(open) = text[start..].find('"') else {
                continue;
            };
            let path_start = start + open + 1;
            let Some(close) = text[path_start..].find('"') else {
                continue;
            };
            let path = &text[path_start..path_start + close];

            // Everything up to the balanced close of `.route(`.
            let mut depth = 1usize;
            let mut cursor = path_start + close + 1;
            while cursor < bytes.len() && depth > 0 {
                match bytes[cursor] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                cursor += 1;
            }
            let handlers = &text[path_start + close + 1..cursor];
            for verb in VERBS {
                if handlers.contains(&format!("{verb}(")) {
                    found.insert((verb.to_uppercase(), normalise(path)));
                }
            }
        }
    }
    found
}

/// axum spells parameters `:id` and wildcards `*key`; OpenAPI uses `{id}`.
fn normalise(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ':' || c == '*' {
            out.push('{');
            while let Some(&next) = chars.peek() {
                if next.is_alphanumeric() || next == '_' {
                    out.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            out.push('}');
        } else {
            out.push(c);
        }
    }
    out
}

/// Every `(VERB, path)` the spec documents.
///
/// Parsed with a small line reader rather than a YAML crate: the spec is five
/// thousand lines and adding a parser dependency to read two levels of keys is
/// not a trade worth making. The shape it relies on — `paths:` at column 0, each
/// path at two spaces, each verb at four — is checked by the assertion on the
/// count, so a reformatted spec fails loudly instead of silently matching
/// nothing.
fn documented_paths() -> BTreeSet<(String, String)> {
    let text = std::fs::read_to_string(repo("docs/openapi.yaml")).expect("spec must be readable");
    let mut out = BTreeSet::new();
    let mut in_paths = false;
    let mut current: Option<String> = None;
    for line in text.lines() {
        if line.starts_with("paths:") {
            in_paths = true;
            continue;
        }
        if !in_paths {
            continue;
        }
        // A new top-level key ends the paths section.
        if !line.starts_with(' ') && !line.trim().is_empty() {
            break;
        }
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim_end();
        if indent == 2 && trimmed.ends_with(':') && trimmed.trim_start().starts_with('/') {
            current = Some(trimmed.trim().trim_end_matches(':').to_string());
            continue;
        }
        if indent == 4 && trimmed.ends_with(':') {
            let verb = trimmed.trim().trim_end_matches(':').to_lowercase();
            if VERBS.contains(&verb.as_str()) {
                if let Some(path) = &current {
                    out.insert((verb.to_uppercase(), path.clone()));
                }
            }
        }
    }
    out
}

#[test]
fn the_spec_and_the_router_describe_the_same_surface() {
    let router = router_routes();
    let spec = documented_paths();

    // Both parsers are heuristic; a broken one would make this test pass by
    // comparing two empty sets.
    assert!(
        router.len() > 80,
        "only {} routes parsed from the router — the extractor is broken",
        router.len()
    );
    assert!(
        spec.len() > 80,
        "only {} paths parsed from the spec — the extractor is broken",
        spec.len()
    );

    let excluded: BTreeSet<(String, String)> = NOT_API
        .iter()
        .map(|(v, p)| (v.to_string(), p.to_string()))
        .collect();

    let undocumented: Vec<&(String, String)> = router
        .iter()
        .filter(|route| !spec.contains(route) && !excluded.contains(route))
        .collect();
    let phantom: Vec<&(String, String)> = spec.difference(&router).collect();

    assert!(
        undocumented.is_empty(),
        "these routes exist but are not in docs/openapi.yaml:\n{}",
        undocumented
            .iter()
            .map(|(v, p)| format!("  {v:6} {p}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        phantom.is_empty(),
        "docs/openapi.yaml documents these, and they do not exist — a client \
         following the spec would get a 404 or a 405:\n{}",
        phantom
            .iter()
            .map(|(v, p)| format!("  {v:6} {p}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_exclusion_list_only_holds_non_api_routes() {
    // An exclusion is a hole in the guard. Keeping `/v1/` out of it means the
    // API surface can never be excused by accident.
    for (_, path) in NOT_API {
        assert!(
            !path.starts_with("/v1/"),
            "{path} is API surface and must be documented, not excluded"
        );
    }
}

#[test]
fn the_axum_to_openapi_path_translation_is_right() {
    assert_eq!(normalise("/v1/state/:key"), "/v1/state/{key}");
    assert_eq!(
        normalise("/v1/blob/:bucket/*key"),
        "/v1/blob/{bucket}/{key}"
    );
    assert_eq!(normalise("/v1/health"), "/v1/health");
    assert_eq!(
        normalise("/v1/memory/:namespace/beliefs/:fact_key/history"),
        "/v1/memory/{namespace}/beliefs/{fact_key}/history"
    );
}

// ── schemas ──────────────────────────────────────────────────────────────────
//
// W3.3 asked for the spec to be *generated* from the code. It is not, and
// annotating ~100 handlers with utoipa is a large mechanical change that buys
// the goal indirectly. The goal is "no drift", and generation was one proposed
// means. These check the same thing against the document itself, which is
// cheaper and catches the failures a reader actually hits.
//
// What they still do not check: that a response's *values* match its documented
// schema. That needs a running server and a fixture per endpoint, and it is the
// honest remaining gap.

/// The spec, parsed properly rather than line by line.
fn spec() -> serde_yaml::Value {
    let text = std::fs::read_to_string(repo("docs/openapi.yaml")).expect("spec must be readable");
    serde_yaml::from_str(&text).expect("the spec must be valid YAML")
}

/// Every `$ref` in the document, as the schema name it points at.
fn collect_refs(node: &serde_yaml::Value, out: &mut Vec<String>) {
    match node {
        serde_yaml::Value::Mapping(map) => {
            for (key, value) in map {
                if key.as_str() == Some("$ref") {
                    if let Some(target) = value.as_str() {
                        out.push(target.to_string());
                    }
                } else {
                    collect_refs(value, out);
                }
            }
        }
        serde_yaml::Value::Sequence(items) => {
            for item in items {
                collect_refs(item, out);
            }
        }
        _ => {}
    }
}

fn defined_schemas(spec: &serde_yaml::Value) -> BTreeSet<String> {
    spec.get("components")
        .and_then(|c| c.get("schemas"))
        .and_then(|s| s.as_mapping())
        .map(|m| {
            m.keys()
                .filter_map(|k| k.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn every_ref_in_the_spec_resolves() {
    // A dangling `$ref` is not a cosmetic problem: every OpenAPI client
    // generator fails outright on one, so a single bad reference makes the whole
    // document unusable rather than partly wrong.
    let spec = spec();
    let defined = defined_schemas(&spec);
    let mut refs = Vec::new();
    collect_refs(&spec, &mut refs);
    assert!(
        refs.len() > 50,
        "only {} refs found — the walker is not reaching the document",
        refs.len()
    );

    let mut dangling: BTreeSet<String> = BTreeSet::new();
    for target in &refs {
        let Some(name) = target.strip_prefix("#/components/schemas/") else {
            // Anything else is a form this spec does not use, and silently
            // accepting it would let a typo through as "not a schema ref".
            dangling.insert(format!("{target} (unsupported ref form)"));
            continue;
        };
        if !defined.contains(name) {
            dangling.insert(target.clone());
        }
    }
    assert!(
        dangling.is_empty(),
        "these $refs point at schemas that do not exist:\n  {}",
        dangling.into_iter().collect::<Vec<_>>().join("\n  ")
    );
}

#[test]
fn every_defined_schema_is_used() {
    // An orphaned schema is documentation nothing points at, so nothing keeps it
    // honest: it is the piece that drifts first and is noticed last.
    let spec = spec();
    let defined = defined_schemas(&spec);
    let mut refs = Vec::new();
    collect_refs(&spec, &mut refs);
    let used: BTreeSet<String> = refs
        .iter()
        .filter_map(|r| r.strip_prefix("#/components/schemas/").map(str::to_string))
        .collect();

    let orphans: Vec<&String> = defined
        .iter()
        .filter(|name| !used.contains(*name))
        .collect();
    assert!(
        orphans.is_empty(),
        "these schemas are defined and never referenced: {orphans:?}"
    );
}

#[test]
fn every_documented_response_body_has_a_schema() {
    // A documented `200` with a content type and no schema tells a client
    // nothing at all, while looking in an index exactly like one that does.
    let spec = spec();
    let mut missing: Vec<String> = Vec::new();

    let Some(paths) = spec.get("paths").and_then(|p| p.as_mapping()) else {
        panic!("the spec has no paths");
    };
    for (path, operations) in paths {
        let Some(operations) = operations.as_mapping() else {
            continue;
        };
        for (verb, operation) in operations {
            let Some(verb) = verb.as_str() else { continue };
            if !VERBS.contains(&verb) {
                continue;
            }
            let Some(responses) = operation.get("responses").and_then(|r| r.as_mapping()) else {
                missing.push(format!(
                    "{} {} has no responses at all",
                    verb.to_uppercase(),
                    path.as_str().unwrap_or("?")
                ));
                continue;
            };
            for (status, response) in responses {
                let Some(content) = response.get("content").and_then(|c| c.as_mapping()) else {
                    // No body documented is fine — a 204 or a 401 need none.
                    continue;
                };
                for (media_type, body) in content {
                    if body.get("schema").is_none() {
                        missing.push(format!(
                            "{} {} → {} ({}) declares a body with no schema",
                            verb.to_uppercase(),
                            path.as_str().unwrap_or("?"),
                            status.as_str().unwrap_or("?"),
                            media_type.as_str().unwrap_or("?")
                        ));
                    }
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} documented bodies have no schema:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
