use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

async fn start_with_sqlite(data_dir: String) -> (String, oneshot::Sender<()>) {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data_dir.clone()),
        snapshot_interval_secs: 30,
        ..Config::default()
    };
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    let sqlite = Some(
        luma::sqlite::SqliteService::new(
            config.data_dir.as_ref().unwrap().to_string() + "/sqlite/rustkiss.db",
        )
        .unwrap(),
    );
    let search_dir = PathBuf::from(&data_dir);
    let search_engine = Arc::new(SearchEngine::new(search_dir).unwrap());
    let app = api::router(
        engine,
        config,
        sqlite,
        search_engine,
        None,
        std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        )),
        None,
    );

    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    (format!("http://{}", addr), tx)
}

#[tokio::test]
async fn sqlite_exec_and_query() {
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_string_lossy().to_string();
    let (base, shutdown) = start_with_sqlite(data_dir).await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{}/v1/sql/exec", base))
        .json(&serde_json::json!({"sql":"CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, body TEXT)","params":[]}))
        .bearer_auth("test").bearer_auth("test").send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    let insert = client
        .post(format!("{}/v1/sql/exec", base))
        .json(&serde_json::json!({"sql":"INSERT INTO notes(body) VALUES (?)","params":["hola"]}))
        .bearer_auth("test")
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(insert.status().is_success());

    let query = client
        .post(format!("{}/v1/sql/query", base))
        .json(&serde_json::json!({"sql":"SELECT body FROM notes","params":[]}))
        .bearer_auth("test")
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(query.status().is_success());
    let body: serde_json::Value = query.json().await.unwrap();
    assert_eq!(body["rows"][0]["body"], "hola");

    let _ = shutdown.send(());
}
