//! Enterprise security tests: multi-tenant isolation, password/session auth,
//! RBAC, security headers, XSS-in-JSON, and SQL-injection resistance.

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
        api_key: "admin-static-key".to_string(),
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

    let embeddings = Arc::new(luma::engine::embeddings::EmbeddingClient::new(
        luma::engine::embeddings::EmbeddingProvider::Mock { dim: 4 },
    ));
    let search_engine = Arc::new(SearchEngine::new(PathBuf::from(data_dir)).unwrap());

    let admin_key = config.api_key.clone();
    let app = api::router(api::RouterDeps {
        engine,
        config,
        sqlite: Some(sqlite),
        search_engine,
        auth_store: Some(auth_store),
        embeddings,
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

async fn register(base: &str, org: &str, email: &str, pass: &str) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!("{base}/v1/auth/register"))
        .json(&serde_json::json!({"org_name": org, "email": email, "password": pass}))
        .send()
        .await
        .unwrap()
}

async fn login(base: &str, email: &str, pass: &str) -> String {
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/auth/login"))
        .json(&serde_json::json!({"email": email, "password": pass}))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "login should succeed");
    let body: serde_json::Value = resp.json().await.unwrap();
    body["token"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn register_login_session_flow() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;

    let reg = register(&base, "Acme", "owner@acme.com", "s3cret-pass").await;
    assert_eq!(reg.status(), 201, "register returns 201");

    let token = login(&base, "owner@acme.com", "s3cret-pass").await;
    assert!(token.starts_with("lums_"), "session token prefix");

    // The session token authenticates protected endpoints.
    let listed = reqwest::Client::new()
        .get(format!("{base}/v1/vector"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert!(
        listed.status().is_success(),
        "session token authorizes reads"
    );

    // Wrong password is rejected.
    let bad = reqwest::Client::new()
        .post(format!("{base}/v1/auth/login"))
        .json(&serde_json::json!({"email": "owner@acme.com", "password": "wrong"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 401, "wrong password is 401");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn invalid_and_revoked_sessions_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // A bogus session token is rejected.
    let bogus = client
        .get(format!("{base}/v1/vector"))
        .bearer_auth("lums_not-a-real-token")
        .send()
        .await
        .unwrap();
    assert_eq!(bogus.status(), 401, "bogus session token is 401");

    register(&base, "Beta", "b@beta.com", "s3cret-pass").await;
    let token = login(&base, "b@beta.com", "s3cret-pass").await;

    // Logout revokes the session.
    let logout = client
        .post(format!("{base}/v1/auth/logout"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(logout.status(), 204);

    let after = client
        .get(format!("{base}/v1/vector"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap();
    assert_eq!(after.status(), 401, "revoked session is 401");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn org_isolation_blocks_cross_tenant_access() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    register(&base, "OrgA", "a@a.com", "s3cret-pass").await;
    register(&base, "OrgB", "b@b.com", "s3cret-pass").await;
    let token_a = login(&base, "a@a.com", "s3cret-pass").await;
    let token_b = login(&base, "b@b.com", "s3cret-pass").await;

    // Org A creates a collection — this records ownership.
    let create = client
        .post(format!("{base}/v1/vector/iso_col"))
        .bearer_auth(&token_a)
        .json(&serde_json::json!({"dim": 4, "metric": "cosine"}))
        .send()
        .await
        .unwrap();
    assert!(
        create.status().is_success(),
        "org A can create its collection"
    );

    // Org A can read its own collection.
    let a_read = client
        .get(format!("{base}/v1/vector/iso_col"))
        .bearer_auth(&token_a)
        .send()
        .await
        .unwrap();
    assert!(
        a_read.status().is_success(),
        "org A reads its own collection"
    );

    // Org B is blocked from the same collection name (existence hidden → 404).
    let b_read = client
        .get(format!("{base}/v1/vector/iso_col"))
        .bearer_auth(&token_b)
        .send()
        .await
        .unwrap();
    assert_eq!(b_read.status(), 404, "org B cannot read org A's collection");

    // Org B cannot hijack the name either.
    let b_create = client
        .post(format!("{base}/v1/vector/iso_col"))
        .bearer_auth(&token_b)
        .json(&serde_json::json!({"dim": 4, "metric": "cosine"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        b_create.status(),
        404,
        "org B cannot write org A's collection"
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn rbac_viewer_cannot_perform_admin_actions() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    register(&base, "Corp", "owner@corp.com", "s3cret-pass").await;
    let owner = login(&base, "owner@corp.com", "s3cret-pass").await;

    // Owner (admin-level) creates a viewer user in their org.
    let created = client
        .post(format!("{base}/v1/admin/users"))
        .bearer_auth(&owner)
        .json(
            &serde_json::json!({"email":"view@corp.com","password":"s3cret-pass","role":"viewer"}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 201, "owner can create users");

    let viewer = login(&base, "view@corp.com", "s3cret-pass").await;

    // Viewer cannot create users (needs admin).
    let denied = client
        .post(format!("{base}/v1/admin/users"))
        .bearer_auth(&viewer)
        .json(&serde_json::json!({"email":"x@corp.com","password":"s3cret-pass","role":"member"}))
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 403, "viewer cannot create users");

    // Viewer cannot list users either.
    let denied_list = client
        .get(format!("{base}/v1/admin/users"))
        .bearer_auth(&viewer)
        .send()
        .await
        .unwrap();
    assert_eq!(denied_list.status(), 403, "viewer cannot list users");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn security_headers_and_csp_present() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;

    let resp = reqwest::Client::new()
        .get(format!("{base}/v1/health"))
        .send()
        .await
        .unwrap();
    let h = resp.headers();
    assert_eq!(
        h.get("x-content-type-options").unwrap(),
        "nosniff",
        "nosniff present"
    );
    assert_eq!(
        h.get("x-frame-options").unwrap(),
        "DENY",
        "frame-options DENY"
    );
    assert!(
        h.get("referrer-policy").is_some(),
        "referrer-policy present"
    );
    assert!(h.get("strict-transport-security").is_some(), "HSTS present");
    let csp = h.get("content-security-policy").unwrap().to_str().unwrap();
    assert!(csp.contains("object-src 'none'"), "CSP blocks objects");
    assert!(csp.contains("frame-ancestors 'none'"), "CSP blocks framing");
    // No 'unsafe-inline' in script-src — blocks inline script injection.
    assert!(csp.contains("script-src 'self'"), "scripts locked to self");
    assert!(
        !csp.contains("script-src 'self' 'unsafe-inline'"),
        "inline scripts must not be allowed"
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn xss_payload_served_as_escaped_json() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    let payload = "<script>alert('xss')</script>";
    register(&base, payload, "owner@xss.com", "s3cret-pass").await;
    let owner = login(&base, "owner@xss.com", "s3cret-pass").await;

    let resp = client
        .get(format!("{base}/v1/admin/orgs"))
        .bearer_auth(&owner)
        .send()
        .await
        .unwrap();
    // Content is JSON, never HTML — the browser will not execute the payload,
    // and nosniff prevents content-type sniffing into HTML.
    let ct = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        ct.starts_with("application/json"),
        "org list is JSON, got {ct}"
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    // The payload round-trips as data (a string value), not as markup.
    let found = body["orgs"]
        .as_array()
        .unwrap()
        .iter()
        .any(|o| o["name"] == payload);
    assert!(found, "payload stored/returned as inert JSON string data");

    let _ = shutdown.send(());
}

#[tokio::test]
async fn admin_panel_spa_is_served() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;
    let client = reqwest::Client::new();

    // Root serves the embedded React shell (public, no auth needed).
    let root = client.get(format!("{base}/")).send().await.unwrap();
    assert!(root.status().is_success(), "SPA root served");
    let ct = root
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(ct.contains("text/html"), "root is HTML, got {ct}");
    let html = root.text().await.unwrap();
    assert!(html.contains("id=\"root\""), "React mount point present");
    assert!(html.contains("/assets/"), "references built assets");

    // Deep-link fallback also serves the shell.
    let deep = client
        .get(format!("{base}/dashboard"))
        .send()
        .await
        .unwrap();
    assert!(
        deep.status().is_success(),
        "SPA deep-link falls back to shell"
    );

    let _ = shutdown.send(());
}

#[tokio::test]
async fn sql_injection_in_inputs_is_neutralized() {
    let dir = tempfile::tempdir().unwrap();
    let (base, shutdown, _) = start_server(dir.path().to_str().unwrap()).await;

    // Attempt an injection via the email field of registration.
    let evil = "evil@x.com'; DROP TABLE sys_users;--";
    let reg = register(&base, "Injected", evil, "s3cret-pass").await;
    assert_eq!(
        reg.status(),
        201,
        "injection payload stored as literal data"
    );

    // If the table had been dropped, a subsequent registration + login would
    // fail. Prove the schema is intact by registering and logging in normally.
    register(&base, "Safe", "safe@x.com", "s3cret-pass").await;
    let token = login(&base, "safe@x.com", "s3cret-pass").await;
    assert!(
        token.starts_with("lums_"),
        "users table intact after injection"
    );

    let _ = shutdown.send(());
}
