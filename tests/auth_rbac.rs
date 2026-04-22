use luma::api;
use luma::config::Config;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

async fn start_server(data_dir: &str) -> (String, oneshot::Sender<()>, String) {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "admin-static".to_string(),
        data_dir: Some(data_dir.to_string()),
        sqlite_enabled: true,
        ..Config::default()
    };
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    let sqlite = luma::sqlite::SqliteService::new(format!("{}/rustkiss.db", data_dir)).unwrap();
    let sqlite_arc = Arc::new(sqlite.clone());

    let auth_store = Arc::new(luma::api::auth_store::AuthStore::new(sqlite_arc.clone()));
    auth_store.init().await.unwrap();
    auth_store
        .ensure_bootstrap_key(&config.api_key)
        .await
        .unwrap();

    let audit_log = Arc::new(luma::api::audit::AuditLog::new(sqlite_arc.clone()));
    audit_log.init().await.unwrap();

    let embeddings = Arc::new(luma::engine::embeddings::EmbeddingClient::new(
        luma::engine::embeddings::EmbeddingProvider::Mock { dim: 4 },
    ));
    let search_engine = Arc::new(SearchEngine::new(PathBuf::from(data_dir)).unwrap());

    let admin_key = config.api_key.clone();
    let app = api::router(
        engine,
        config,
        Some(sqlite),
        search_engine,
        Some(auth_store),
        embeddings,
        Some(audit_log),
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

    (format!("http://{}", addr), tx, admin_key)
}

#[tokio::test]
async fn missing_token_returns_401() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/auth/keys", base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "no token should be 401");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn invalid_token_returns_401() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/auth/keys", base))
        .bearer_auth("totally-wrong-key-xyz")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "invalid token should be 401");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn user_role_forbidden_on_list_keys() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Create a user-role key using the admin key
    let create_resp = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({
            "name": "user-key",
            "role": "user"
        }))
        .send()
        .await
        .unwrap();
    assert!(create_resp.status().is_success(), "create key failed");
    let body: serde_json::Value = create_resp.json().await.unwrap();
    let user_key = body["key"].as_str().unwrap().to_string();

    // User-role key should get 403 on list_keys
    let resp = client
        .get(format!("{}/v1/auth/keys", base))
        .bearer_auth(&user_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403, "user role should be forbidden");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn user_role_forbidden_on_create_key() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Create a user-role key
    let create_resp = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "user-key2", "role": "user"}))
        .send()
        .await
        .unwrap();
    assert!(create_resp.status().is_success());
    let body: serde_json::Value = create_resp.json().await.unwrap();
    let user_key = body["key"].as_str().unwrap().to_string();

    // User-role key should get 403 on create_key
    let resp = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&user_key)
        .json(&serde_json::json!({"name": "attempt", "role": "admin"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403, "user cannot create keys");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn user_role_forbidden_on_revoke_key() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Create two keys: one user, one target
    let user_resp = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "user-revoker", "role": "user"}))
        .send()
        .await
        .unwrap();
    let user_key = user_resp.json::<serde_json::Value>().await.unwrap()["key"]
        .as_str()
        .unwrap()
        .to_string();

    let target_resp = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "target-key", "role": "user"}))
        .send()
        .await
        .unwrap();
    let target_id = target_resp.json::<serde_json::Value>().await.unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();

    // User-role should get 403 on revoke
    let resp = client
        .delete(format!("{}/v1/auth/keys/{}", base, target_id))
        .bearer_auth(&user_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403, "user cannot revoke keys");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn admin_key_can_list_and_create_keys() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let list = client
        .get(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert!(list.status().is_success(), "admin can list keys");

    let create = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "new-key"}))
        .send()
        .await
        .unwrap();
    assert!(create.status().is_success(), "admin can create keys");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn revoked_key_returns_401() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Create and then revoke a key
    let create_resp = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "ephemeral", "role": "user"}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = create_resp.json().await.unwrap();
    let key_id = body["id"].as_str().unwrap().to_string();
    let key_val = body["key"].as_str().unwrap().to_string();

    // Verify key works before revocation
    let before = client
        .get(format!("{}/v1/health", base))
        .bearer_auth(&key_val)
        .send()
        .await
        .unwrap();
    assert!(
        before.status().is_success(),
        "key should work before revocation"
    );

    // Revoke
    let revoke = client
        .delete(format!("{}/v1/auth/keys/{}", base, key_id))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert_eq!(revoke.status(), 204, "revoke should succeed");

    // Key should now be invalid — 401 on a protected endpoint
    let after = client
        .get(format!("{}/v1/auth/keys", base))
        .bearer_auth(&key_val)
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), 401, "revoked key should be 401");
    let _ = shutdown.send(());
}
