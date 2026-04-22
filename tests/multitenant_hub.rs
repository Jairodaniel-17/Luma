use axum::http::StatusCode;
use luma::api;
use luma::api::auth_store::AuthStore;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use luma::sqlite::SqliteService;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

struct TenantApp {
    base: String,
    shutdown: oneshot::Sender<()>,
    _dir: TempDir,
}

async fn start() -> (TenantApp, Arc<AuthStore>) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("tenant.db");
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "bootstrap".to_string(),
        data_dir: Some(dir.path().to_string_lossy().to_string()),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    let sqlite = SqliteService::new(&db_path).unwrap();
    let auth_store = Arc::new(AuthStore::new(Arc::new(sqlite.clone())));
    auth_store.init().await.unwrap();
    let search_engine = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());
    let app = api::router(
        engine,
        config,
        Some(sqlite),
        search_engine,
        Some(auth_store.clone()),
        Arc::new(luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        )),
        None,
    );

    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown, rx) = oneshot::channel();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    (
        TenantApp {
            base: format!("http://{}", addr),
            shutdown,
            _dir: dir,
        },
        auth_store,
    )
}

#[tokio::test]
async fn tenant_keys_isolate_hub_namespaces() {
    let (app, auth_store) = start().await;
    auth_store
        .create_key(
            "Tenant A",
            Some("tenant-a"),
            "user",
            "tenant-a-key",
            serde_json::json!({"allow":"*"}),
            serde_json::json!({"storage_bytes":1000,"qps":10}),
        )
        .await
        .unwrap();
    auth_store
        .create_key(
            "Tenant B",
            Some("tenant-b"),
            "user",
            "tenant-b-key",
            serde_json::json!({"allow":"*"}),
            serde_json::json!({"storage_bytes":1000,"qps":10}),
        )
        .await
        .unwrap();

    let client = reqwest::Client::new();
    for (token, doc_id, tenant) in [
        ("tenant-a-key", "doc-a", "tenant-a"),
        ("tenant-b-key", "doc-b", "tenant-b"),
    ] {
        let resp = client
            .post(format!("{}/v1/db/shared/ingest", app.base))
            .bearer_auth(token)
            .json(&serde_json::json!({
                "id": doc_id,
                "text": format!("hello from {tenant}"),
                "metadata": { "tenant": tenant }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    let search_a = client
        .post(format!("{}/v1/db/shared/search", app.base))
        .bearer_auth("tenant-a-key")
        .json(&serde_json::json!({"query":"hello","limit":5}))
        .send()
        .await
        .unwrap();
    let body_a: serde_json::Value = search_a.json().await.unwrap();
    assert_eq!(body_a["results"][0]["id"], "doc-a");

    let search_b = client
        .post(format!("{}/v1/db/shared/search", app.base))
        .bearer_auth("tenant-b-key")
        .json(&serde_json::json!({"query":"hello","limit":5}))
        .send()
        .await
        .unwrap();
    let body_b: serde_json::Value = search_b.json().await.unwrap();
    assert_eq!(body_b["results"][0]["id"], "doc-b");

    let _ = app.shutdown.send(());
}
