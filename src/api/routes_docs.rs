use axum::{
    http::{header, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
    routing::get,
    Router,
};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use super::AppState;

const OPENAPI_SPEC: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/openapi.yaml"));

#[derive(Debug, Clone, Default, Deserialize)]
struct OpenApiDoc {
    openapi: Option<String>,
    info: Option<Info>,
    servers: Option<Vec<Server>>,
    components: Option<Components>,
    paths: Option<IndexMap<String, PathItem>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Info {
    title: Option<String>,
    version: Option<String>,
    description: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Server {
    url: Option<String>,
    description: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Components {
    #[serde(rename = "securitySchemes")]
    security_schemes: Option<IndexMap<String, SecurityScheme>>,
    schemas: Option<IndexMap<String, Schema>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct SecurityScheme {
    #[serde(rename = "type")]
    ty: Option<String>,
    scheme: Option<String>,
    #[serde(rename = "bearerFormat")]
    bearer_format: Option<String>,
    description: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Schema {
    #[serde(rename = "type")]
    ty: Option<String>,
    required: Option<Vec<String>>,
    properties: Option<IndexMap<String, Value>>,
    #[serde(rename = "enum")]
    enum_values: Option<Vec<Value>>,
    items: Option<Value>,
    format: Option<String>,
    minimum: Option<Value>,
    nullable: Option<bool>,
    #[serde(rename = "$ref")]
    r#ref: Option<String>,
    #[serde(flatten)]
    extra: IndexMap<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PathItem {
    get: Option<Operation>,
    post: Option<Operation>,
    put: Option<Operation>,
    patch: Option<Operation>,
    delete: Option<Operation>,
    options: Option<Operation>,
    head: Option<Operation>,
    trace: Option<Operation>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Operation {
    summary: Option<String>,
    description: Option<String>,
    deprecated: Option<bool>,
    security: Option<Vec<IndexMap<String, Vec<String>>>>,
    parameters: Option<Vec<Parameter>>,
    #[serde(rename = "requestBody")]
    request_body: Option<RequestBody>,
    responses: Option<IndexMap<String, ApiResponse>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct Parameter {
    #[serde(rename = "in")]
    location: Option<String>,
    name: Option<String>,
    required: Option<bool>,
    description: Option<String>,
    schema: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RequestBody {
    required: Option<bool>,
    content: Option<IndexMap<String, MediaType>>,
    description: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct MediaType {
    schema: Option<Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ApiResponse {
    description: Option<String>,
    content: Option<IndexMap<String, MediaType>>,
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
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in path.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("p{hash:x}")
}

fn method_badge(method: &str) -> &'static str {
    match method {
        "get" => "get",
        "post" => "post",
        "put" => "put",
        "patch" => "patch",
        "delete" => "delete",
        _ => "other",
    }
}

fn op_requires_auth(op: &Operation) -> bool {
    op.security.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
}

fn extract_schema_ref(v: &Value) -> Option<String> {
    v.get("$ref")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string())
}

fn pretty_json_one_line(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.to_string(),
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

fn op_security_labels(op: &Operation, doc: &OpenApiDoc) -> Vec<String> {
    let mut labels = Vec::new();
    let Some(requirements) = op.security.as_ref() else {
        return labels;
    };
    let schemes = doc
        .components
        .as_ref()
        .and_then(|c| c.security_schemes.as_ref());

    for requirement in requirements {
        for key in requirement.keys() {
            let mut label = key.clone();
            if let Some(scheme) = schemes.and_then(|all| all.get(key)) {
                if let Some(ty) = scheme.ty.as_deref() {
                    let mut detail = ty.to_string();
                    if ty == "http" {
                        if let Some(http_scheme) = scheme.scheme.as_deref() {
                            detail = format!("http {http_scheme}");
                            if let Some(format) = scheme.bearer_format.as_deref() {
                                detail.push_str(" (");
                                detail.push_str(format);
                                detail.push(')');
                            }
                        }
                    }
                    label.push_str(" (");
                    label.push_str(&detail);
                    label.push(')');
                }
            }
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
    }

    labels
}

fn parse_openapi(yaml: &str) -> OpenApiDoc {
    serde_yaml::from_str::<OpenApiDoc>(yaml).unwrap_or_default()
}

fn render_docs_html(doc: &OpenApiDoc) -> String {
    let title = doc
        .info
        .as_ref()
        .and_then(|i| i.title.as_deref())
        .unwrap_or("API Docs");
    let version = doc
        .info
        .as_ref()
        .and_then(|i| i.version.as_deref())
        .unwrap_or("");
    let openapi_v = doc.openapi.as_deref().unwrap_or("");
    let info_description = doc
        .info
        .as_ref()
        .and_then(|i| i.description.as_deref())
        .unwrap_or("");
    let servers = doc.servers.clone().unwrap_or_default();
    let base_url = servers.get(0).and_then(|s| s.url.as_deref()).unwrap_or("");
    let security_schemes = doc
        .components
        .as_ref()
        .and_then(|c| c.security_schemes.as_ref())
        .cloned()
        .unwrap_or_default();
    let scheme_names = security_schemes.keys().cloned().collect::<Vec<_>>();
    let schemas = doc
        .components
        .as_ref()
        .and_then(|c| c.schemas.as_ref())
        .cloned()
        .unwrap_or_default();
    let paths = doc.paths.clone().unwrap_or_default();

    let mut op_count = 0usize;
    let mut auth_ops = 0usize;
    for item in paths.values() {
        for op in [
            &item.get,
            &item.post,
            &item.put,
            &item.patch,
            &item.delete,
            &item.options,
            &item.head,
            &item.trace,
        ] {
            if let Some(op) = op {
                op_count += 1;
                if op_requires_auth(op) {
                    auth_ops += 1;
                }
            }
        }
    }

    let mut html = String::new();
    html.push_str("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />");
    html.push_str("<title>");
    html.push_str(&escape_html(title));
    html.push_str("</title>");
    html.push_str(
        r#"
<style>
:root {
  color-scheme: light dark;

  --bg: #0b1020;
  --bg2: #0f1630;
  --panel: rgba(255,255,255,.06);
  --panel2: rgba(255,255,255,.08);
  --border: rgba(255,255,255,.12);
  --border2: rgba(255,255,255,.18);
  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.70);
  --muted2: rgba(255,255,255,.55);

  --accent: #4da3ff;
  --accent2: #87c7ff;

  --shadow: 0 10px 30px rgba(0,0,0,.25);
  --radius: 14px;

  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #f6f8ff;
    --bg2: #ffffff;
    --panel: rgba(0,0,0,.04);
    --panel2: rgba(0,0,0,.06);
    --border: rgba(0,0,0,.10);
    --border2: rgba(0,0,0,.16);
    --text: rgba(0,0,0,.88);
    --muted: rgba(0,0,0,.62);
    --muted2: rgba(0,0,0,.48);
    --shadow: 0 10px 30px rgba(0,0,0,.10);
  }
}

* { box-sizing: border-box; }
html {
  min-height: 100%;
  background:
    radial-gradient(1200px 500px at 20% -10%, rgba(77,163,255,.20), transparent 60%),
    radial-gradient(900px 500px at 90% 0%, rgba(135,199,255,.14), transparent 60%),
    linear-gradient(180deg, var(--bg), var(--bg2));
  background-attachment: fixed;
}

body {
  min-height: 100%;
  margin: 0;
  font-family: var(--sans);
  background: transparent; /* 🔑 */
}


a { color: inherit; text-decoration: none; }
a:hover { text-decoration: underline; }

header {
  position: sticky;
  top: 0;
  z-index: 20;
  backdrop-filter: blur(12px);
  background: linear-gradient(180deg, rgba(0,0,0,.25), rgba(0,0,0,0));
  border-bottom: 1px solid var(--border);
}
@media (prefers-color-scheme: light) {
  header { background: rgba(255,255,255,.75); }
}

.header-inner {
  display: flex;
  gap: 16px;
  align-items: center;
  justify-content: space-between;
  padding: 16px 18px;
  max-width: 1200px;
  margin: 0 auto;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 12px;
  min-width: 0;
}
.brand h1 {
  margin: 0;
  font-size: 18px;
  letter-spacing: .2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.pill {
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--panel);
  border: 1px solid var(--border);
  color: var(--muted);
}

.actions {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--panel);
  box-shadow: none;
  font-size: 13px;
  color: var(--text);
}
.btn:hover { background: var(--panel2); text-decoration: none; }

main {
  max-width: 1200px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 14px;
  padding: 14px 18px 26px 18px;
}

.sidebar {
  position: sticky;
  top: 68px;
  align-self: start;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: rgba(255,255,255,.03);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.sidebar-top {
  padding: 14px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,0));
}
.sidebar-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.sidebar-title .muted { color: var(--muted); font-size: 13px; }

.search {
  width: 100%;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(0,0,0,.12);
  color: var(--text);
  outline: none;
}
@media (prefers-color-scheme: light) {
  .search { background: rgba(255,255,255,.8); }
}
.search:focus { border-color: var(--border2); box-shadow: 0 0 0 3px rgba(77,163,255,.18); }

.navlist {
  max-height: calc(100vh - 68px - 160px);
  overflow: auto;
  padding: 10px;
}
.navitem {
  display: block;
  padding: 10px 10px;
  border-radius: 12px;
  border: 1px solid transparent;
  color: var(--text);
}
.navitem:hover { background: var(--panel); text-decoration: none; }
.navitem.active { border-color: var(--border2); background: var(--panel2); }

.navpath { font-family: var(--mono); font-size: 12px; color: var(--muted); }
.navmeta { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 6px; }

.content {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: rgba(255,255,255,.03);
  box-shadow: var(--shadow);
  overflow: hidden;
}

.section {
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
}
.section:last-child { border-bottom: 0; }

.h2row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin: 4px 0 10px 0;
}
h2 {
  margin: 0;
  font-size: 16px;
  letter-spacing: .2px;
}
.pathcode {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
}

.opcard {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 12px;
  background: rgba(0,0,0,.10);
  margin: 10px 0;
}
@media (prefers-color-scheme: light) {
  .opcard { background: rgba(255,255,255,.80); }
}
.ophead {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.opleft {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}
.badge {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: .6px;
  text-transform: uppercase;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border2);
  background: var(--panel);
  white-space: nowrap;
}
.badge.get    { border-color: rgba(77,163,255,.45); }
.badge.post   { border-color: rgba(76,228,165,.45); }
.badge.put    { border-color: rgba(255,224,102,.45); }
.badge.patch  { border-color: rgba(255,109,189,.45); }
.badge.delete { border-color: rgba(255,79,121,.45); }
.badge.other  { border-color: rgba(180,180,180,.35); }

.summary {
  font-weight: 650;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.deprecated {
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,79,121,.35);
  background: rgba(255,79,121,.10);
  color: rgba(255,79,121,.95);
}

.desc { margin-top: 8px; color: var(--muted); font-size: 13px; line-height: 1.45; }
.small { color: var(--muted2); font-size: 12px; }

.kv { display:flex; gap:10px; flex-wrap:wrap; }
.card { border:1px solid var(--border); background:rgba(255,255,255,.03); border-radius:14px; padding:12px; }
.k { color: var(--muted2); font-size:12px; }
.v { color: var(--text); font-size:13px; font-family: var(--mono); }
.table { width:100%; border-collapse: collapse; margin-top:8px; }
.table th, .table td { text-align:left; padding:8px 10px; border-top:1px solid var(--border); vertical-align: top; }
.table th { color: var(--muted); font-size:12px; font-weight:650; }
.code { font-family: var(--mono); font-size:12px; color: var(--muted); }
.schema-chip { display:inline-flex; gap:8px; align-items:center; padding:6px 10px; border:1px solid var(--border); border-radius:999px; background:var(--panel); font-family: var(--mono); font-size:12px; }
.auth { border-color: rgba(255,224,102,.35); background: rgba(255,224,102,.10); color: rgba(255,224,102,.95); }

details {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(255,255,255,.02);
}
details > summary { cursor: pointer; font-weight: 650; }
pre {
  margin: 10px 0 0 0;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(0,0,0,.18);
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.5;
}
@media (prefers-color-scheme: light) {
  pre { background: rgba(0,0,0,.04); }
}

@media (max-width: 960px) {
  main { grid-template-columns: 1fr; }
  .sidebar { position: relative; top: 0; }
  .navlist { max-height: 260px; }
}
</style>
"#,
    );
    html.push_str("</head><body>");

    html.push_str("<header><div class=\"header-inner\">");
    html.push_str("<div class=\"brand\"><h1>");
    html.push_str(&escape_html(title));
    html.push_str("</h1>");
    if !version.is_empty() {
        html.push_str("<span class=\"pill\">v");
        html.push_str(&escape_html(version));
        html.push_str("</span>");
    }
    if !openapi_v.is_empty() {
        html.push_str("<span class=\"pill\">OpenAPI ");
        html.push_str(&escape_html(openapi_v));
        html.push_str("</span>");
    }
    html.push_str("</div>");

    html.push_str("<div class=\"actions\">");
    html.push_str("<a class=\"btn\" href=\"/openapi.yaml\">openapi.yaml</a>");
    html.push_str("<a class=\"btn\" href=\"#schemas\">schemas</a>");
    html.push_str("</div>");
    html.push_str("</div></header>");

    html.push_str("<main>");

    html.push_str("<aside class=\"sidebar\">");
    html.push_str("<div class=\"sidebar-top\">");
    html.push_str("<div class=\"sidebar-title\"><div class=\"muted\">Navigation</div></div>");
    html.push_str("<input class=\"search\" id=\"q\" placeholder=\"Search endpoints…\" autocomplete=\"off\" />");
    html.push_str("</div>");
    html.push_str("<div class=\"navlist\" id=\"nav\">");
    html.push_str(
        "<a class=\"navitem\" href=\"#overview\"><div class=\"navpath\">Overview</div></a>",
    );
    for (path, item) in &paths {
        let anchor = anchor_for_path(path);
        html.push_str("<a class=\"navitem\" data-path=\"");
        html.push_str(&escape_html(path));
        html.push_str("\" href=\"#");
        html.push_str(&anchor);
        html.push_str("\"><div class=\"navpath\">");
        html.push_str(&escape_html(path));
        html.push_str("</div><div class=\"navmeta\">");
        for (method, op) in [
            ("get", &item.get),
            ("post", &item.post),
            ("put", &item.put),
            ("patch", &item.patch),
            ("delete", &item.delete),
        ] {
            if op.is_some() {
                html.push_str("<span class=\"badge ");
                html.push_str(method_badge(method));
                html.push_str("\">");
                html.push_str(&escape_html(method));
                html.push_str("</span>");
            }
        }
        html.push_str("</div></a>");
    }
    html.push_str(
        "<a class=\"navitem\" href=\"#schemas\"><div class=\"navpath\">Schemas</div></a>",
    );
    html.push_str("<a class=\"navitem\" href=\"#raw\"><div class=\"navpath\">Raw YAML</div></a>");
    html.push_str("</div></aside>");

    html.push_str("<div class=\"content\" id=\"content\">");

    html.push_str("<div class=\"section\" id=\"overview\">");
    html.push_str("<div class=\"h2row\"><h2>Overview</h2><div class=\"pathcode\">");
    html.push_str(&escape_html(&format!(
        "{op_count} operations • {auth_ops} protected"
    )));
    html.push_str("</div></div>");
    if !info_description.is_empty() {
        html.push_str("<div class=\"desc\">");
        html.push_str(&escape_html(info_description));
        html.push_str("</div>");
    }
    html.push_str("<div class=\"kv\">");
    html.push_str("<div class=\"card\"><div class=\"k\">Base URL</div><div class=\"v\">");
    if base_url.is_empty() {
        html.push_str("<span class=\"code\">(none)</span>");
    } else {
        html.push_str(&escape_html(base_url));
    }
    html.push_str("</div></div>");
    html.push_str("<div class=\"card\"><div class=\"k\">Security Schemes</div><div class=\"v\">");
    if scheme_names.is_empty() {
        html.push_str("<span class=\"code\">(none)</span>");
    } else {
        html.push_str(&escape_html(&scheme_names.join(", ")));
    }
    html.push_str("</div></div>");
    html.push_str("<div class=\"card\"><div class=\"k\">Schemas</div><div class=\"v\">");
    html.push_str(&escape_html(&schemas.len().to_string()));
    html.push_str("</div></div>");
    html.push_str("</div>");
    if !servers.is_empty() {
        html.push_str("<div class=\"card\" style=\"margin-top:12px;\">");
        html.push_str("<div class=\"k\">Servers</div>");
        html.push_str("<table class=\"table\"><thead><tr><th>url</th><th>description</th></tr></thead><tbody>");
        for s in &servers {
            html.push_str("<tr><td class=\"code\">");
            html.push_str(&escape_html(s.url.as_deref().unwrap_or("")));
            html.push_str("</td><td>");
            html.push_str(&escape_html(s.description.as_deref().unwrap_or("")));
            html.push_str("</td></tr>");
        }
        html.push_str("</tbody></table></div>");
    }
    if !security_schemes.is_empty() {
        html.push_str("<div class=\"card\" style=\"margin-top:12px;\">");
        html.push_str("<div class=\"k\">Security Schemes</div>");
        html.push_str("<table class=\"table\"><thead><tr><th>name</th><th>type</th><th>description</th></tr></thead><tbody>");
        for (name, scheme) in &security_schemes {
            html.push_str("<tr><td class=\"code\">");
            html.push_str(&escape_html(name));
            html.push_str("</td><td class=\"code\">");
            let mut detail = String::new();
            if let Some(ty) = scheme.ty.as_deref() {
                detail.push_str(ty);
            }
            if let Some(http_scheme) = scheme.scheme.as_deref() {
                if !detail.is_empty() {
                    detail.push(' ');
                }
                detail.push_str(http_scheme);
            }
            if let Some(format) = scheme.bearer_format.as_deref() {
                if !detail.is_empty() {
                    detail.push(' ');
                }
                detail.push('(');
                detail.push_str(format);
                detail.push(')');
            }
            if detail.is_empty() {
                detail.push_str("(unspecified)");
            }
            html.push_str(&escape_html(&detail));
            html.push_str("</td><td>");
            html.push_str(&escape_html(scheme.description.as_deref().unwrap_or("")));
            html.push_str("</td></tr>");
        }
        html.push_str("</tbody></table></div>");
    }
    html.push_str("</div>");

    for (path, item) in &paths {
        let anchor = anchor_for_path(path);
        html.push_str("<div class=\"section\" id=\"");
        html.push_str(&anchor);
        html.push_str("\">");
        html.push_str("<div class=\"h2row\"><h2>");
        html.push_str(&escape_html(path));
        html.push_str("</h2><div class=\"pathcode\">path</div></div>");

        let ops: [(&str, &Option<Operation>); 8] = [
            ("get", &item.get),
            ("post", &item.post),
            ("put", &item.put),
            ("patch", &item.patch),
            ("delete", &item.delete),
            ("options", &item.options),
            ("head", &item.head),
            ("trace", &item.trace),
        ];

        for (method, op_opt) in ops {
            let Some(op) = op_opt.as_ref() else {
                continue;
            };
            let summary = op.summary.as_deref().unwrap_or("");
            let desc = op.description.as_deref().unwrap_or("");
            let deprecated = op.deprecated.unwrap_or(false);
            let auth = op_requires_auth(op);
            let security_labels = op_security_labels(op, doc);

            html.push_str("<div class=\"opcard\" data-op=\"1\" data-method=\"");
            html.push_str(&escape_html(method));
            html.push_str("\">");
            html.push_str("<div class=\"ophead\"><div class=\"opleft\">");
            html.push_str("<span class=\"badge ");
            html.push_str(method_badge(method));
            html.push_str("\">");
            html.push_str(&escape_html(method));
            html.push_str("</span>");
            html.push_str("<div style=\"min-width:0;\">");
            html.push_str("<div class=\"summary\">");
            if summary.is_empty() {
                html.push_str("<span class=\"small\">(no summary)</span>");
            } else {
                html.push_str(&escape_html(summary));
            }
            html.push_str("</div></div></div>");

            if auth {
                if security_labels.is_empty() {
                    html.push_str("<span class=\"schema-chip auth\">AUTH required</span>");
                } else {
                    for label in &security_labels {
                        html.push_str("<span class=\"schema-chip auth\">");
                        html.push_str(&escape_html(label));
                        html.push_str("</span>");
                    }
                }
            }
            if deprecated {
                html.push_str("<span class=\"deprecated\">DEPRECATED</span>");
            }
            html.push_str("</div>");

            if !desc.is_empty() {
                html.push_str("<div class=\"desc\">");
                html.push_str(&escape_html(desc));
                html.push_str("</div>");
            }

            if let Some(params) = op.parameters.as_ref() {
                if !params.is_empty() {
                    html.push_str("<div style=\"margin-top:10px;\" class=\"k\">Parameters</div>");
                    html.push_str("<table class=\"table\"><thead><tr><th>in</th><th>name</th><th>required</th><th>schema</th><th>description</th></tr></thead><tbody>");
                    for p in params {
                        html.push_str("<tr><td class=\"code\">");
                        html.push_str(&escape_html(p.location.as_deref().unwrap_or("")));
                        html.push_str("</td><td class=\"code\">");
                        html.push_str(&escape_html(p.name.as_deref().unwrap_or("")));
                        html.push_str("</td><td>");
                        html.push_str(if p.required.unwrap_or(false) {
                            "true"
                        } else {
                            "false"
                        });
                        html.push_str("</td><td class=\"code\">");
                        if let Some(s) = p.schema.as_ref() {
                            if let Some(r) = extract_schema_ref(s) {
                                html.push_str(&escape_html(&r));
                            } else {
                                html.push_str(&escape_html(&pretty_json_one_line(s)));
                            }
                        }
                        html.push_str("</td><td>");
                        html.push_str(&escape_html(p.description.as_deref().unwrap_or("")));
                        html.push_str("</td></tr>");
                    }
                    html.push_str("</tbody></table>");
                }
            }

            if let Some(rb) = op.request_body.as_ref() {
                html.push_str("<div style=\"margin-top:10px;\" class=\"k\">Request Body</div>");
                if let Some(desc) = rb.description.as_deref() {
                    if !desc.is_empty() {
                        html.push_str("<div class=\"small\">");
                        html.push_str(&escape_html(desc));
                        html.push_str("</div>");
                    }
                }
                html.push_str("<div class=\"small\">required: ");
                html.push_str(if rb.required.unwrap_or(false) {
                    "true"
                } else {
                    "false"
                });
                html.push_str("</div>");
                if let Some(content) = rb.content.as_ref() {
                    html.push_str("<table class=\"table\"><thead><tr><th>content-type</th><th>schema</th></tr></thead><tbody>");
                    for (ct, mt) in content {
                        html.push_str("<tr><td class=\"code\">");
                        html.push_str(&escape_html(ct));
                        html.push_str("</td><td class=\"code\">");
                        if let Some(s) = mt.schema.as_ref() {
                            if let Some(r) = extract_schema_ref(s) {
                                html.push_str(&escape_html(&r));
                            } else {
                                html.push_str(&escape_html(&pretty_json_one_line(s)));
                            }
                        }
                        html.push_str("</td></tr>");
                    }
                    html.push_str("</tbody></table>");
                }
            }

            if let Some(resps) = op.responses.as_ref() {
                if !resps.is_empty() {
                    html.push_str("<div style=\"margin-top:10px;\" class=\"k\">Responses</div>");
                    html.push_str("<table class=\"table\"><thead><tr><th>status</th><th>description</th><th>content</th></tr></thead><tbody>");
                    for (code, r) in resps {
                        html.push_str("<tr><td class=\"code\">");
                        html.push_str(&escape_html(code));
                        html.push_str("</td><td>");
                        html.push_str(&escape_html(r.description.as_deref().unwrap_or("")));
                        html.push_str("</td><td>");
                        if let Some(content) = r.content.as_ref() {
                            let mut first = true;
                            for (ct, mt) in content {
                                if !first {
                                    html.push_str("<br/>");
                                }
                                first = false;
                                html.push_str("<span class=\"code\">");
                                html.push_str(&escape_html(ct));
                                html.push_str("</span>");
                                if let Some(s) = mt.schema.as_ref() {
                                    html.push_str(" — <span class=\"code\">");
                                    if let Some(rf) = extract_schema_ref(s) {
                                        html.push_str(&escape_html(&rf));
                                    } else {
                                        html.push_str(&escape_html(&pretty_json_one_line(s)));
                                    }
                                    html.push_str("</span>");
                                }
                            }
                        } else {
                            html.push_str("<span class=\"small\">(no body)</span>");
                        }
                        html.push_str("</td></tr>");
                    }
                    html.push_str("</tbody></table>");
                }
            }

            html.push_str("</div>");
        }

        html.push_str("</div>");
    }

    html.push_str("<div class=\"section\" id=\"schemas\">");
    html.push_str("<div class=\"h2row\"><h2>Schemas</h2><div class=\"pathcode\">components/schemas</div></div>");
    if schemas.is_empty() {
        html.push_str("<div class=\"small\">No schemas found.</div>");
    } else {
        for (name, schema) in &schemas {
            html.push_str("<details style=\"margin:10px 0;\"><summary>");
            html.push_str(&escape_html(name));
            html.push_str("</summary>");
            html.push_str(
                "<div class=\"small\" style=\"margin-top:8px;\">type: <span class=\"code\">",
            );
            html.push_str(&escape_html(schema.ty.as_deref().unwrap_or("")));
            html.push_str("</span>");
            if let Some(req) = schema.required.as_ref() {
                if !req.is_empty() {
                    html.push_str(" • required: <span class=\"code\">");
                    html.push_str(&escape_html(&req.join(", ")));
                    html.push_str("</span>");
                }
            }
            if let Some(ev) = schema.enum_values.as_ref() {
                if !ev.is_empty() {
                    let values = ev
                        .iter()
                        .map(pretty_json_one_line)
                        .collect::<Vec<_>>()
                        .join(", ");
                    html.push_str(" • enum: <span class=\"code\">");
                    html.push_str(&escape_html(&values));
                    html.push_str("</span>");
                }
            }
            if schema.nullable.unwrap_or(false) {
                html.push_str(" • <span class=\"code\">nullable</span>");
            }
            if let Some(format) = schema.format.as_deref() {
                html.push_str(" • format: <span class=\"code\">");
                html.push_str(&escape_html(format));
                html.push_str("</span>");
            }
            if let Some(minimum) = schema.minimum.as_ref() {
                html.push_str(" • minimum: <span class=\"code\">");
                html.push_str(&escape_html(&pretty_json_one_line(minimum)));
                html.push_str("</span>");
            }
            if let Some(reference) = schema.r#ref.as_deref() {
                html.push_str(" • ref: <span class=\"code\">");
                html.push_str(&escape_html(reference));
                html.push_str("</span>");
            }
            html.push_str("</div>");
            if let Some(props) = schema.properties.as_ref() {
                if !props.is_empty() {
                    html.push_str("<table class=\"table\"><thead><tr><th>property</th><th>schema</th></tr></thead><tbody>");
                    for (prop, value) in props {
                        html.push_str("<tr><td class=\"code\">");
                        html.push_str(&escape_html(prop));
                        html.push_str("</td><td class=\"code\">");
                        if let Some(rf) = extract_schema_ref(value) {
                            html.push_str(&escape_html(&rf));
                        } else {
                            html.push_str(&escape_html(&pretty_json_one_line(value)));
                        }
                        html.push_str("</td></tr>");
                    }
                    html.push_str("</tbody></table>");
                }
            }
            if let Some(items) = schema.items.as_ref() {
                html.push_str(
                    "<div class=\"small\" style=\"margin-top:6px;\">items: <span class=\"code\">",
                );
                if let Some(rf) = extract_schema_ref(items) {
                    html.push_str(&escape_html(&rf));
                } else {
                    html.push_str(&escape_html(&pretty_json_one_line(items)));
                }
                html.push_str("</span></div>");
            }
            if !schema.extra.is_empty() {
                let json_obj = Value::Object(
                    schema
                        .extra
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                );
                html.push_str(
                    "<div class=\"small\" style=\"margin-top:6px;\">extra: <span class=\"code\">",
                );
                html.push_str(&escape_html(&pretty_json_one_line(&json_obj)));
                html.push_str("</span></div>");
            }
            html.push_str("</details>");
        }
    }
    html.push_str("</div>");

    html.push_str("<div class=\"section\" id=\"raw\">");
    html.push_str("<details open><summary>Raw OpenAPI (YAML)</summary><pre>");
    html.push_str(&escape_html(OPENAPI_SPEC));
    html.push_str("</pre></details></div>");

    html.push_str("</div>");
    html.push_str("</main>");

    html.push_str(
        r#"
<script>
(function(){
  const q = document.getElementById('q');
  if(!q) return;
  q.addEventListener('input', () => {
    const query = (q.value||'').trim().toLowerCase();
    document.querySelectorAll('.section[id]').forEach(sec => {
      if(sec.id === 'overview' || sec.id === 'schemas' || sec.id === 'raw') return;
      const text = sec.innerText.toLowerCase();
      sec.style.display = (!query || text.includes(query)) ? '' : 'none';
    });
  });
})();
</script>
"#,
    );

    html.push_str("</body></html>");
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
    let doc = parse_openapi(OPENAPI_SPEC);
    Html(render_docs_html(&doc)).into_response()
}

pub fn routes_docs() -> Router<AppState> {
    Router::<AppState>::new()
        .route("/", get(|| async { Redirect::to("/docs") }))
        .route("/docs", get(docs_html))
        .route("/docs/openapi.yaml", get(openapi_yaml))
        .route("/openapi.yaml", get(openapi_yaml))
}
