//! The dashboard and alert rules must name metrics that actually exist.
//!
//! W5.1 of `docs/PLAN-MAESTRO.md`. A dashboard panel that silently plots
//! nothing looks like an idle system, and an alert rule on a renamed metric
//! never fires — which is indistinguishable from everything being fine. Both
//! are worse than not having them, so this test reads the committed files and
//! checks every metric they reference against what `/v1/metrics` actually
//! serves.
//!
//! It fails when a metric is renamed or removed without updating the
//! observability assets, which is exactly when it should.

use luma::config::Config;
use luma::engine::Engine;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

fn observability_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/observability")
}

/// Metric names the running server exposes.
///
/// Histograms are recorded under their base name; a query against
/// `search_duration_ms_bucket` is satisfied by `search_duration_ms` existing,
/// so the suffix is trimmed before comparing.
fn exported_metrics() -> BTreeSet<String> {
    let dir = tempfile::tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        ..Config::default()
    };
    let engine = Engine::new(config, CancellationToken::new()).unwrap();

    // Exercise a few engines so the counters that are only emitted once they
    // have a value are present.
    engine
        .put_state("warm".to_string(), serde_json::json!(1), None, None)
        .unwrap();

    let body = engine.metrics_text();
    let mut names = BTreeSet::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            if let Some(name) = rest.split_whitespace().next() {
                names.insert(name.to_string());
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        // `name{label="x"} 1` or `name 1`
        let name = line
            .split(|c: char| c == '{' || c.is_whitespace())
            .next()
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        for suffix in ["_bucket", "_sum", "_count"] {
            if let Some(base) = name.strip_suffix(suffix) {
                names.insert(base.to_string());
            }
        }
        names.insert(name.to_string());
    }
    names
}

/// Metric names referenced by a PromQL expression.
///
/// Deliberately simple: identifiers that are not PromQL functions or keywords.
/// A parser would be more precise, but a false positive here fails loudly and
/// is fixed in one line, whereas a missed reference is the thing this test
/// exists to catch.
fn metrics_in_expr(expr: &str) -> BTreeSet<String> {
    const NOT_METRICS: &[&str] = &[
        "rate",
        "increase",
        "sum",
        "avg",
        "max",
        "min",
        "count",
        "histogram_quantile",
        "by",
        "on",
        "and",
        "or",
        "unless",
        "le",
        "irate",
        "delta",
        "abs",
        "ceil",
        "floor",
        "clamp_max",
        "clamp_min",
        "topk",
        "bottomk",
        "quantile",
        "without",
        "group_left",
        "group_right",
        "m",
        "h",
        "s",
        "d",
    ];
    let mut found = BTreeSet::new();
    let mut current = String::new();
    for ch in expr.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            take_identifier(&mut current, &mut found, NOT_METRICS);
        }
    }
    take_identifier(&mut current, &mut found, NOT_METRICS);
    found
}

fn take_identifier(current: &mut String, found: &mut BTreeSet<String>, skip: &[&str]) {
    let word = std::mem::take(current);
    if word.is_empty() || skip.contains(&word.as_str()) {
        return;
    }
    // Numbers, and the `5m` of a range selector, are not metric names.
    if word.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return;
    }
    // Metric names are snake_case identifiers; anything without an underscore
    // in these files is a label or a function we did not list.
    if !word.contains('_') {
        return;
    }
    found.insert(word);
}

/// Strip the histogram suffix so a `_bucket` query maps to its base metric.
fn base_name(metric: &str) -> String {
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(base) = metric.strip_suffix(suffix) {
            return base.to_string();
        }
    }
    metric.to_string()
}

#[test]
fn every_dashboard_metric_exists() {
    let path = observability_dir().join("dashboard.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("dashboard missing at {path:?}: {e}"));
    let dashboard: serde_json::Value =
        serde_json::from_str(&text).expect("dashboard.json must be valid JSON");

    let exported = exported_metrics();
    let mut referenced = BTreeSet::new();
    collect_exprs(&dashboard, &mut referenced);
    assert!(
        !referenced.is_empty(),
        "found no PromQL in the dashboard — the panel structure must have changed"
    );

    let mut missing = Vec::new();
    for expr in &referenced {
        for metric in metrics_in_expr(expr) {
            let base = base_name(&metric);
            // RESP metrics are emitted only when the listener runs, and this
            // harness starts an engine without one. They are covered by the
            // listener's own metrics test instead.
            if base.starts_with("resp_") {
                continue;
            }
            if !exported.contains(&base) {
                missing.push(format!("{metric} (in `{expr}`)"));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "the dashboard references metrics the server does not export:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn every_alert_metric_exists() {
    let path = observability_dir().join("alerts.yml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("alerts missing at {path:?}: {e}"));

    let exported = exported_metrics();
    let mut checked = 0;
    let mut missing = Vec::new();
    for line in text.lines() {
        let Some(expr) = line.trim().strip_prefix("expr: ") else {
            continue;
        };
        checked += 1;
        for metric in metrics_in_expr(expr) {
            let base = base_name(&metric);
            if base.starts_with("resp_") {
                continue;
            }
            if !exported.contains(&base) {
                missing.push(format!("{metric} (in `{expr}`)"));
            }
        }
    }
    assert!(checked > 0, "found no alert expressions to check");
    assert!(
        missing.is_empty(),
        "alert rules reference metrics the server does not export — these \
         alerts would never fire:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn the_demo_stack_files_are_all_present_and_parse() {
    // The acceptance criterion is that `docker compose up` shows the dashboard
    // without editing anything, which needs every provisioning file to exist.
    let dir = observability_dir();
    for name in [
        "docker-compose.yml",
        "prometheus.yml",
        "alerts.yml",
        "grafana-datasource.yml",
        "grafana-dashboards.yml",
        "dashboard.json",
    ] {
        let path = dir.join(name);
        assert!(path.exists(), "missing {path:?}");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.trim().is_empty(), "{name} is empty");
    }

    // The compose file must actually mount the dashboard, or Grafana comes up
    // blank and the whole point is lost.
    let compose = std::fs::read_to_string(dir.join("docker-compose.yml")).unwrap();
    assert!(compose.contains("dashboard.json"), "dashboard not mounted");
    assert!(compose.contains("alerts.yml"), "alert rules not mounted");

    let prometheus = std::fs::read_to_string(dir.join("prometheus.yml")).unwrap();
    assert!(
        prometheus.contains("/v1/metrics"),
        "the scrape config must point at Luma's metrics path"
    );
}

#[test]
fn the_metrics_endpoint_is_valid_prometheus_text() {
    // Every non-comment line must be `name value` or `name{labels} value`, and
    // the value must parse as a number. Prometheus rejects the whole scrape on
    // one malformed line, so a single bad metric takes down all of them.
    let body = exported_metrics();
    assert!(!body.is_empty(), "no metrics exported at all");

    let dir = tempfile::tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        ..Config::default()
    };
    let engine = Engine::new(config, CancellationToken::new()).unwrap();
    let text = engine.metrics_text();

    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value = line
            .rsplit_once(char::is_whitespace)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("line {} has no value: `{line}`", n + 1));
        assert!(
            value.parse::<f64>().is_ok() || value == "NaN" || value == "+Inf" || value == "-Inf",
            "line {} has a non-numeric value: `{line}`",
            n + 1
        );
    }
}

/// Walk the dashboard JSON collecting every `expr` field.
fn collect_exprs(value: &serde_json::Value, out: &mut BTreeSet<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, child) in map {
                if key == "expr" {
                    if let Some(expr) = child.as_str() {
                        out.insert(expr.to_string());
                    }
                }
                collect_exprs(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_exprs(item, out);
            }
        }
        _ => {}
    }
}
