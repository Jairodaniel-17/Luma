use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

/// The compiled admin SPA (React + Vite + TypeScript) built from `admin-ui/`
/// into `ui/dist` and embedded into the binary. The server therefore ships the
/// panel with no need for Node or the source tree at runtime.
#[derive(RustEmbed)]
#[folder = "ui/dist/"]
struct Assets;

pub struct StaticFile<T>(pub T);

/// Vite emits content-hashed asset filenames (e.g. `index-DMmLYY8W.js`), so those
/// are safe to cache forever. The HTML shell must NEVER be cached hard, or a
/// browser keeps requesting stale, already-deleted asset hashes after a deploy
/// and the app renders broken. This is the fix for exactly that.
fn cache_control(path: &str) -> &'static str {
    if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        // index.html and anything else: always revalidate.
        "no-cache"
    }
}

impl<T> IntoResponse for StaticFile<T>
where
    T: Into<String>,
{
    fn into_response(self) -> Response {
        let path = self.0.into();
        match Assets::get(path.as_str()) {
            Some(content) => {
                let mime = mime_guess::from_path(&path).first_or_octet_stream();
                (
                    [
                        (header::CONTENT_TYPE, mime.as_ref()),
                        (header::CACHE_CONTROL, cache_control(&path)),
                    ],
                    Body::from(content.data),
                )
                    .into_response()
            }
            None => serve_index(),
        }
    }
}

/// Serve the SPA entry point (`index.html`) with `no-cache` so new deploys are
/// always picked up.
fn serve_index() -> Response {
    match Assets::get("index.html") {
        Some(content) => (
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            Body::from(content.data),
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "404 Not Found").into_response(),
    }
}

/// Handler for `/` and `/index.html`.
pub async fn handler(uri: Uri) -> impl IntoResponse {
    let mut path = uri.path().trim_start_matches('/').to_string();
    if path.is_empty() {
        path = "index.html".to_string();
    }
    StaticFile(path)
}

/// Catch-all SPA fallback: any unmatched route that is not an API path serves
/// `index.html` so client-side navigation and refreshes work. API paths
/// (`/v1/*`, `/docs`, …) are matched by explicit routes and never reach here.
pub async fn spa_fallback(uri: Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/').to_string();
    // Serve known static assets directly; otherwise fall back to the SPA shell.
    if !path.is_empty() && Assets::get(&path).is_some() {
        StaticFile(path).into_response()
    } else {
        serve_index()
    }
}
