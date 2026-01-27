use axum::{
    extract::Host,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Router,
};
use serde_yaml;

use super::AppState;

// We embed the static openapi.yaml file into the binary.
const OPENAPI_SPEC: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/openapi.yaml"));

/// Handler to serve the raw openapi.yaml file.
/// This is used by the Scalar UI.
pub async fn openapi_yaml() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/yaml; charset=utf-8")
        .body(axum::body::Body::from(OPENAPI_SPEC))
        .unwrap()
}

/// Handler that serves the beautiful Scalar API documentation UI.
/// It dynamically replaces the server URL in the OpenAPI spec to match the
/// currently running server.
pub async fn docs_html(Host(host): Host) -> impl IntoResponse {
    // Parse the static YAML file into a generic JSON value.
    // Using serde_json::Value is easier than defining all the structs.
    let mut openapi_value: serde_json::Value =
        serde_yaml::from_str(OPENAPI_SPEC).unwrap_or_default();

    // Dynamically set the server URL based on the Host header.
    if let Some(obj) = openapi_value.as_object_mut() {
        let new_server = serde_json::json!([{
            "url": format!("http://{}", host),
            "description": "Current Server"
        }]);
        obj.insert("servers".to_string(), new_server);
    }

    // Serialize the modified spec back to a JSON string.
    let dynamic_spec_json = serde_json::to_string(&openapi_value).unwrap_or_default();

    // The HTML response that embeds the Scalar UI.
    // It's configured to use the dynamically generated spec content.
    let html = format!(
        r#"
<!doctype html>
<html>
<head>
  <title>API Docs</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      margin: 0;
    }}
  </style>
</head>
<body>
  <script id="api-reference" type="application/json">
    {spec_content}
  </script>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>
"#,
        spec_content = dynamic_spec_json
    );

    Html(html)
}

/// Sets up all the routes for serving the documentation.
/// - Redirects `/` to `/docs`
/// - Serves the Scalar UI at `/docs`
/// - Serves the raw openapi.yaml file for tooling or manual inspection.
pub fn routes_docs() -> Router<AppState> {
    Router::<AppState>::new()
        // The beautiful, dynamic Scalar UI
        .route("/docs", get(docs_html))
        // The original, static openapi.yaml file
        .route("/docs/openapi.yaml", get(openapi_yaml))
        .route("/openapi.yaml", get(openapi_yaml))
}