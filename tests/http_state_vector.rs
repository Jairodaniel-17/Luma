use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

async fn start() -> (String, oneshot::Sender<()>) {
    start_with_config(base_test_config()).await
}

async fn start_with_diskann() -> (String, oneshot::Sender<()>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = base_test_config();
    config.data_dir = Some(dir.path().to_string_lossy().to_string());
    let (base, shutdown) = start_with_config(config).await;
    (base, shutdown, dir)
}

async fn start_with_config(config: Config) -> (String, oneshot::Sender<()>) {
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();

    // For tests not using search, a temporary dropped dir is fine.
    let temp_dir = tempfile::tempdir().unwrap();
    let search_engine = Arc::new(SearchEngine::new(temp_dir.path().to_path_buf()).unwrap());

    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: None,
        search_engine,
        auth_store: None,
        embeddings: std::sync::Arc::new(luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 384 },
        )),
        audit_log: None,
        rbac: None,
    });

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
async fn state_put_get() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let put = client
        .put(format!("{}/v1/state/job:1", base))
        .json(&serde_json::json!({"value":{"progress":1}}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success());

    let got = client
        .get(format!("{}/v1/state/job:1", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(got.status().is_success());
    let v: serde_json::Value = got.json().await.unwrap();
    assert_eq!(v["key"], "job:1");
    assert_eq!(v["value"]["progress"], 1);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn vector_create_upsert_search() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{}/v1/vector/docs", base))
        .json(&serde_json::json!({"dim":2,"metric":"cosine"}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    let upsert = client
        .post(format!("{}/v1/vector/docs/upsert", base))
        .json(&serde_json::json!({"id":"a","vector":[1.0,0.0],"meta":{"tag":"x"}}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(upsert.status().is_success());

    let search = client
        .post(format!("{}/v1/vector/docs/search", base))
        .json(&serde_json::json!({"vector":[0.9,0.1],"k":1,"options":{"include_meta":true}}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(search.status().is_success());
    let v: serde_json::Value = search.json().await.unwrap();
    assert_eq!(v["hits"][0]["id"], "a");

    let _ = shutdown.send(());
}

fn base_test_config() -> Config {
    Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: None,
        snapshot_interval_secs: 30,
        ..Config::default()
    }
}

#[tokio::test]
async fn vector_list_collections_endpoint() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let initial = client
        .get(format!("{}/v1/vector", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(initial.status().is_success());
    let value: serde_json::Value = initial.json().await.unwrap();
    assert!(value["collections"].as_array().unwrap().is_empty());

    let create = client
        .post(format!("{}/v1/vector/docs", base))
        .json(&serde_json::json!({"dim":2,"metric":"cosine"}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    let listed = client
        .get(format!("{}/v1/vector", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(listed.status().is_success());
    let body: serde_json::Value = listed.json().await.unwrap();
    let collections = body["collections"].as_array().unwrap();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0]["collection"], "docs");
    assert_eq!(collections[0]["dim"], 2);

    let _ = shutdown.send(());
}

#[tokio::test]
async fn vector_collection_detail_endpoint() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let missing = client
        .get(format!("{}/v1/vector/none", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);
    let body: serde_json::Value = missing.json().await.unwrap();
    assert_eq!(body["error"], "not_found");

    let create = client
        .post(format!("{}/v1/vector/docs", base))
        .json(&serde_json::json!({"dim":2,"metric":"cosine"}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    let detail = client
        .get(format!("{}/v1/vector/docs", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(detail.status().is_success());
    let value: serde_json::Value = detail.json().await.unwrap();
    assert_eq!(value["collection"], "docs");
    assert_eq!(value["dim"], 2);
    assert_eq!(value["metric"], "cosine");
    assert_eq!(value["count"], 0);
    assert!(value["manifest"].is_object());

    let _ = shutdown.send(());
}

#[tokio::test]
async fn docstore_put_get_find() {
    let (base, shutdown) = start().await;
    let client = reqwest::Client::new();

    let put = client
        .put(format!("{}/v1/doc/users/u1", base))
        .json(&serde_json::json!({"name":"Ada","role":"admin"}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(put.status().is_success());

    let get = client
        .get(format!("{}/v1/doc/users/u1", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(get.status().is_success());
    let doc: serde_json::Value = get.json().await.unwrap();
    assert_eq!(doc["doc"]["name"], "Ada");

    let find = client
        .post(format!("{}/v1/doc/users/find", base))
        .json(&serde_json::json!({"filter":{"role":"admin"},"limit":10}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(find.status().is_success());
    let v: serde_json::Value = find.json().await.unwrap();
    assert!(!v["documents"].as_array().unwrap().is_empty());
    assert_eq!(v["documents"][0]["doc"]["role"], "admin");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn vector_diskann_build_status_and_tune() {
    let (base, shutdown, _dir) = start_with_diskann().await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{}/v1/vector/docs", base))
        .json(&serde_json::json!({"dim":3,"metric":"cosine"}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success());

    for (id, vector) in [("a", vec![1.0, 0.0, 0.0]), ("b", vec![0.0, 1.0, 0.0])] {
        let upsert = client
            .post(format!("{}/v1/vector/docs/upsert", base))
            .json(&serde_json::json!({"id":id,"vector":vector}))
            .bearer_auth("test")
            .send()
            .await
            .unwrap();
        assert!(upsert.status().is_success());
    }

    let build = client
        .post(format!("{}/v1/vector/docs/diskann/build", base))
        .json(&serde_json::json!({"max_degree":16,"search_list_size":48}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(build.status().is_success());
    let body: serde_json::Value = build.json().await.unwrap();
    assert_eq!(body["status"]["available"], true);
    assert_eq!(body["params"]["max_degree"], 16);

    let status = client
        .get(format!("{}/v1/vector/docs/diskann/status", base))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(status.status().is_success());
    let status_body: serde_json::Value = status.json().await.unwrap();
    assert_eq!(status_body["available"], true);

    let tune = client
        .post(format!("{}/v1/vector/docs/diskann/tune", base))
        .json(&serde_json::json!({"search_list_size":96}))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert!(tune.status().is_success());
    let tune_body: serde_json::Value = tune.json().await.unwrap();
    assert_eq!(tune_body["params"]["search_list_size"], 96);

    let _ = shutdown.send(());
}
