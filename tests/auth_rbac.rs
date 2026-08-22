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

    let rbac = Arc::new(luma::api::rbac::RbacService::new(sqlite_arc.clone()));
    rbac.init().await.unwrap();

    let embeddings = luma::engine::embeddings::EmbeddingHandle::new(
        luma::engine::embeddings::EmbeddingClient::new(
            luma::engine::embeddings::EmbeddingProvider::Mock { dim: 4 },
        ),
    );
    let search_engine = Arc::new(SearchEngine::new(PathBuf::from(data_dir)).unwrap());

    let admin_key = config.api_key.clone();
    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: Some(sqlite),
        search_engine,
        auth_store: Some(auth_store),
        embeddings,
        resp_metrics: None,
        audit_log: Some(audit_log),
        rbac: Some(rbac),
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

// ---- RBAC role management ----

#[tokio::test]
async fn admin_can_list_roles() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/auth/roles", base))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "admin can list roles");
    let body: serde_json::Value = resp.json().await.unwrap();
    let roles = body["roles"].as_array().unwrap();
    // Seeded: admin, user, readonly
    assert!(roles.len() >= 3, "at least 3 system roles should be seeded");
    let names: Vec<&str> = roles.iter().filter_map(|r| r["name"].as_str()).collect();
    assert!(names.contains(&"admin"));
    assert!(names.contains(&"user"));
    assert!(names.contains(&"readonly"));
    let _ = shutdown.send(());
}

#[tokio::test]
async fn user_role_forbidden_on_list_roles() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let create = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "u", "role": "user"}))
        .send()
        .await
        .unwrap();
    let user_key = create.json::<serde_json::Value>().await.unwrap()["key"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = client
        .get(format!("{}/v1/auth/roles", base))
        .bearer_auth(&user_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403, "user cannot list roles");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn admin_can_create_and_delete_custom_role() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Create
    let create = client
        .post(format!("{}/v1/auth/roles", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({
            "name": "operator",
            "parent_role_id": "user",
            "description": "ops role"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create.status(), 201, "create custom role should be 201");
    let body: serde_json::Value = create.json().await.unwrap();
    let role_id = body["id"].as_str().unwrap().to_string();

    // List — new role should appear
    let list = client
        .get(format!("{}/v1/auth/roles", base))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    let roles = list.json::<serde_json::Value>().await.unwrap();
    let names: Vec<&str> = roles["roles"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["name"].as_str())
        .collect();
    assert!(names.contains(&"operator"));

    // Delete
    let del = client
        .delete(format!("{}/v1/auth/roles/{}", base, role_id))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert_eq!(del.status(), 204, "delete custom role should be 204");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn cannot_delete_system_role() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let resp = client
        .delete(format!("{}/v1/auth/roles/admin", base))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 422, "deleting system role should be 422");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn admin_can_list_role_permissions() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{}/v1/auth/roles/readonly/permissions", base))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    let perms = body["permissions"].as_array().unwrap();
    // readonly must have at least read on vector
    assert!(
        perms
            .iter()
            .any(|p| p["resource"] == "vector" && p["action"] == "read"),
        "readonly should have vector:read"
    );
    let _ = shutdown.send(());
}

#[tokio::test]
async fn permission_check_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // admin can do everything
    let resp = client
        .get(format!(
            "{}/v1/auth/roles/check?role=admin&resource=auth&action=admin",
            base
        ))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["allowed"], true, "admin should have auth:admin");

    // readonly cannot write
    let resp2 = client
        .get(format!(
            "{}/v1/auth/roles/check?role=readonly&resource=vector&action=write",
            base
        ))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    let body2: serde_json::Value = resp2.json().await.unwrap();
    assert_eq!(
        body2["allowed"], false,
        "readonly should not have vector:write"
    );

    // user inherits read from readonly
    let resp3 = client
        .get(format!(
            "{}/v1/auth/roles/check?role=user&resource=vector&action=read",
            base
        ))
        .bearer_auth(&admin_key)
        .send()
        .await
        .unwrap();
    let body3: serde_json::Value = resp3.json().await.unwrap();
    assert_eq!(
        body3["allowed"], true,
        "user inherits vector:read from readonly"
    );
    let _ = shutdown.send(());
}

#[tokio::test]
async fn admin_can_update_key_role() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, admin_key) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Create a user-role key
    let create = client
        .post(format!("{}/v1/auth/keys", base))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"name": "promote-me", "role": "user"}))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = create.json().await.unwrap();
    let key_id = body["id"].as_str().unwrap().to_string();

    // User key cannot list keys initially
    let key_val = body["key"].as_str().unwrap().to_string();
    let before = client
        .get(format!("{}/v1/auth/keys", base))
        .bearer_auth(&key_val)
        .send()
        .await
        .unwrap();
    assert_eq!(before.status(), 403);

    // Promote to admin
    let update = client
        .put(format!("{}/v1/auth/keys/{}/role", base, key_id))
        .bearer_auth(&admin_key)
        .json(&serde_json::json!({"role": "admin"}))
        .send()
        .await
        .unwrap();
    assert_eq!(update.status(), 204, "role update should be 204");

    // Now the key should be able to list keys
    let after = client
        .get(format!("{}/v1/auth/keys", base))
        .bearer_auth(&key_val)
        .send()
        .await
        .unwrap();
    assert!(
        after.status().is_success(),
        "promoted key should have admin access"
    );
    let _ = shutdown.send(());
}
