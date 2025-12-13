use axum::{
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};

use super::AppState;

const OPENAPI_SPEC: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/openapi.yaml"));

#[derive(Clone, Default)]
struct BasicOperation {
    method: String,
    summary: String,
    description: String,
    deprecated: bool,
}

#[derive(Clone, Default)]
struct BasicPathItem {
    path: String,
    operations: Vec<BasicOperation>,
}

#[derive(Clone, Default)]
struct BasicSpec {
    title: String,
    version: String,
    paths: Vec<BasicPathItem>,
}

fn escape_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(c),
        }
    }
    out
}

fn anchor_for_path(path: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a 64-bit
    for b in path.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("p{hash:x}")
}

fn parse_scalar_value(raw: &str) -> String {
    raw.trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn is_http_method_key(s: &str) -> bool {
    matches!(
        s,
        "get" | "put" | "post" | "delete" | "options" | "head" | "patch" | "trace"
    )
}

fn parse_openapi_basic(yaml: &str) -> BasicSpec {
    let mut out = BasicSpec::default();

    let mut in_info = false;
    let mut in_paths = false;

    let mut current_path: Option<BasicPathItem> = None;
    let mut current_op: Option<BasicOperation> = None;

    for raw_line in yaml.lines() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let indent = line.chars().take_while(|c| *c == ' ').count();

        if indent == 0 {
            in_info = trimmed == "info:";
            in_paths = trimmed == "paths:";
            continue;
        }

        if in_info && indent == 2 {
            if let Some(v) = trimmed.strip_prefix("title:") {
                out.title = parse_scalar_value(v);
            } else if let Some(v) = trimmed.strip_prefix("version:") {
                out.version = parse_scalar_value(v);
            }
            continue;
        }

        if in_paths {
            if indent == 2 && trimmed.ends_with(':') && trimmed.starts_with('/') {
                if let Some(op) = current_op.take() {
                    if let Some(ref mut p) = current_path {
                        p.operations.push(op);
                    }
                }
                if let Some(p) = current_path.take() {
                    out.paths.push(p);
                }
                let path = trimmed.trim_end_matches(':').to_string();
                current_path = Some(BasicPathItem {
                    path,
                    operations: Vec::new(),
                });
                continue;
            }

            if indent == 4 && trimmed.ends_with(':') {
                let key = trimmed.trim_end_matches(':');
                if is_http_method_key(key) {
                    if let Some(op) = current_op.take() {
                        if let Some(ref mut p) = current_path {
                            p.operations.push(op);
                        }
                    }
                    current_op = Some(BasicOperation {
                        method: key.to_string(),
                        ..Default::default()
                    });
                }
                continue;
            }

            if indent == 6 {
                if let Some(ref mut op) = current_op {
                    if let Some(v) = trimmed.strip_prefix("summary:") {
                        op.summary = parse_scalar_value(v);
                    } else if let Some(v) = trimmed.strip_prefix("description:") {
                        op.description = parse_scalar_value(v);
                    } else if let Some(v) = trimmed.strip_prefix("deprecated:") {
                        op.deprecated = parse_scalar_value(v).eq_ignore_ascii_case("true");
                    }
                }
                continue;
            }
        }
    }

    if let Some(op) = current_op.take() {
        if let Some(ref mut p) = current_path {
            p.operations.push(op);
        }
    }
    if let Some(p) = current_path.take() {
        out.paths.push(p);
    }

    if out.title.is_empty() {
        out.title = "API Docs".to_string();
    }

    for p in &mut out.paths {
        p.operations.sort_by(|a, b| a.method.cmp(&b.method));
    }
    out.paths.sort_by(|a, b| a.path.cmp(&b.path));

    out
}

fn render_docs_html(spec: &BasicSpec) -> String {
    let title = spec.title.as_str();
    let version = spec.version.as_str();

    let mut html = String::new();
    html.push_str("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />");
    html.push_str("<title>");
    html.push_str(&escape_html(title));
    html.push_str("</title>");
    html.push_str(
        r#"<style>
        :root { color-scheme: light dark; }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif; margin: 0; }
        header { padding: 20px 24px; border-bottom: 1px solid rgba(127,127,127,.35); }
        header h1 { margin: 0 0 6px 0; font-size: 20px; }
        header .meta { opacity: .8; font-size: 13px; display: flex; gap: 12px; flex-wrap: wrap; }
        main { display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 74px); }
        nav { border-right: 1px solid rgba(127,127,127,.35); padding: 14px 16px; overflow: auto; }
        section { padding: 14px 18px; overflow: auto; }
        a { color: inherit; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .path { padding: 8px 10px; border-radius: 8px; }
        .path:hover { background: rgba(127,127,127,.12); }
        .op { display: flex; gap: 10px; align-items: baseline; padding: 10px 12px; border: 1px solid rgba(127,127,127,.35); border-radius: 10px; margin: 10px 0; }
        .method { font-weight: 700; width: 62px; text-transform: uppercase; font-size: 12px; opacity: .9; }
        .summary { font-weight: 600; }
        .muted { opacity: .8; font-size: 13px; }
        pre { white-space: pre-wrap; word-break: break-word; padding: 12px; border: 1px solid rgba(127,127,127,.35); border-radius: 10px; }
        details { border: 1px solid rgba(127,127,127,.35); border-radius: 10px; padding: 10px 12px; margin: 12px 0; }
        details > summary { cursor: pointer; font-weight: 700; }
        @media (max-width: 900px) { main { grid-template-columns: 1fr; } nav { border-right: 0; border-bottom: 1px solid rgba(127,127,127,.35);} }
        </style>"#,
    );
    html.push_str("</head><body>");

    html.push_str("<header><h1>");
    html.push_str(&escape_html(title));
    html.push_str("</h1><div class=\"meta\">");
    if !version.is_empty() {
        html.push_str("<span>version: ");
        html.push_str(&escape_html(version));
        html.push_str("</span>");
    }
    html.push_str("<a href=\"/openapi.yaml\">openapi.yaml</a>");
    html.push_str("</div></header>");

    html.push_str("<main><nav>");
    html.push_str("<div class=\"muted\" style=\"margin: 6px 0 10px 0;\">Endpoints</div>");
    if spec.paths.is_empty() {
        html.push_str("<div class=\"muted\">No paths found in OpenAPI spec.</div>");
    }
    for p in &spec.paths {
        let anchor = anchor_for_path(&p.path);
        html.push_str("<div class=\"path\"><a href=\"#");
        html.push_str(&anchor);
        html.push_str("\">");
        html.push_str(&escape_html(&p.path));
        html.push_str("</a></div>");
    }
    html.push_str("</nav><section>");

    for p in &spec.paths {
        let anchor = anchor_for_path(&p.path);
        html.push_str("<h2 id=\"");
        html.push_str(&anchor);
        html.push_str("\" style=\"margin: 14px 0 8px 0; font-size: 18px;\">");
        html.push_str(&escape_html(&p.path));
        html.push_str("</h2>");

        for op in &p.operations {
            html.push_str("<div class=\"op\">");
            html.push_str("<div class=\"method\">");
            html.push_str(&escape_html(&op.method));
            html.push_str("</div>");
            html.push_str("<div>");
            html.push_str("<div class=\"summary\">");
            if op.deprecated {
                html.push_str("[DEPRECATED] ");
            }
            if op.summary.trim().is_empty() {
                html.push_str("<span class=\"muted\">(no summary)</span>");
            } else {
                html.push_str(&escape_html(&op.summary));
            }
            html.push_str("</div>");

            if !op.description.trim().is_empty() {
                html.push_str("<div class=\"muted\" style=\"margin-top: 4px;\">");
                html.push_str(&escape_html(&op.description));
                html.push_str("</div>");
            }
            html.push_str("</div></div>");
        }
    }

    html.push_str("<details><summary>Raw OpenAPI (YAML)</summary><pre>");
    html.push_str(&escape_html(OPENAPI_SPEC));
    html.push_str("</pre></details>");

    html.push_str("</section></main></body></html>");
    html
}

pub async fn openapi_yaml() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/yaml; charset=utf-8")
        .body(axum::body::Body::from(OPENAPI_SPEC))
        .unwrap()
}

pub async fn docs_html() -> impl IntoResponse {
    let spec = parse_openapi_basic(OPENAPI_SPEC);
    Html(render_docs_html(&spec)).into_response()
}

pub fn routes_docs() -> Router<AppState> {
    Router::<AppState>::new()
        .route("/docs", get(docs_html))
        .route("/docs/openapi.yaml", get(openapi_yaml))
        .route("/openapi.yaml", get(openapi_yaml))
}
