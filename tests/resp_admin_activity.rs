//! `GET /v1/admin/resp` — RESP activity, broken down by organization.
//!
//! F4.2 of `docs/PLAN-MAESTRO.md`. Two things here are easy to get wrong in ways
//! that look fine, so both are pinned:
//!
//! 1. **The per-org connection gauge must come back down.** `serve_inner`
//!    returns from a dozen places — a framing error, a quit, an idle timeout, a
//!    broken pipe — and a missed decrement leaves a phantom connection in the
//!    panel forever. That reads as a leak in the server rather than a bug in the
//!    accounting, which is why the count is held by a guard whose `Drop` does
//!    the work. The listener already learned this once with `drop_connection`,
//!    where the test checked the visible symptom instead of the registry.
//!
//! 2. **Re-authenticating to another org moves the connection, not copies it.**

use luma::api::auth_store::AuthStore;
use luma::config::Config;
use luma::engine::Engine;
use luma::resp::listener::{spawn, RespMetrics};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

struct Harness {
    port: u16,
    metrics: Arc<RespMetrics>,
    store: Arc<AuthStore>,
    shutdown: CancellationToken,
    _dir: tempfile::TempDir,
}

async fn start() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        resp_port: 0,
        api_key: "instance-secret-key-long".to_string(),
        ..Config::default()
    };
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    config.resp_port = probe.local_addr().unwrap().port();
    drop(probe);

    let sqlite = Arc::new(
        luma::sqlite::SqliteService::new(dir.path().join("keys.db")).expect("sqlite must open"),
    );
    let store = Arc::new(AuthStore::new(sqlite));
    store.init().await.unwrap();

    let shutdown = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown.clone()).unwrap();
    let metrics = Arc::new(RespMetrics::default());
    let port = spawn(
        &config,
        engine,
        Arc::clone(&metrics),
        Some(Arc::clone(&store)),
        shutdown.clone(),
    )
    .await
    .unwrap()
    .expect("listener must bind");

    Harness {
        port,
        metrics,
        store,
        shutdown,
        _dir: dir,
    }
}

async fn key_for(harness: &Harness, org: &str) -> String {
    let plain = harness.store.generate_api_key();
    harness
        .store
        .create_key(
            org,
            Some(org),
            "user",
            &plain,
            serde_json::json!({}),
            serde_json::json!({}),
        )
        .await
        .expect("key creation");
    plain
}

fn connect(port: u16) -> TcpStream {
    let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream.set_nodelay(true).unwrap();
    stream
}

fn call(stream: &mut TcpStream, frame: &[u8]) -> String {
    stream.write_all(frame).unwrap();
    let mut buffer = [0u8; 4096];
    let read = stream.read(&mut buffer).unwrap_or(0);
    String::from_utf8_lossy(&buffer[..read]).to_string()
}

fn auth_frame(secret: &str) -> Vec<u8> {
    format!("*2\r\n$4\r\nAUTH\r\n${}\r\n{}\r\n", secret.len(), secret).into_bytes()
}

fn open_for(harness: &Harness, org: &str) -> u64 {
    harness
        .metrics
        .per_org_snapshot()
        .get(org)
        .map(|c| c.connections_open)
        .unwrap_or(0)
}

/// Wait for a per-org gauge to reach a value.
///
/// The decrement happens when the server-side task drops its guard, which is
/// after the client's socket is closed — so polling briefly is the difference
/// between testing the accounting and testing the scheduler.
fn wait_for_open(harness: &Harness, org: &str, want: u64) -> u64 {
    for _ in 0..100 {
        let now = open_for(harness, org);
        if now == want {
            return now;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    open_for(harness, org)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_connection_is_attributed_to_the_org_it_authenticated_as() {
    let harness = start().await;
    let acme = key_for(&harness, "acme").await;

    let mut client = connect(harness.port);
    // Before AUTH the connection is not attributable to any org.
    assert_eq!(open_for(&harness, "acme"), 0);

    assert_eq!(call(&mut client, &auth_frame(&acme)), "+OK\r\n");
    assert_eq!(wait_for_open(&harness, "acme", 1), 1);

    call(&mut client, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n");
    let counters = harness.metrics.per_org_snapshot();
    let acme_counters = counters.get("acme").expect("acme must be tracked");
    assert!(
        acme_counters.commands_total >= 1,
        "the command must be counted against the org: {acme_counters:?}"
    );
    assert_eq!(acme_counters.connections_total, 1);

    harness.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_gauge_comes_back_down_when_the_connection_closes() {
    // The property a missed decrement would break, and the one that reads as a
    // server leak rather than a metrics bug.
    let harness = start().await;
    let acme = key_for(&harness, "acme").await;
    {
        let mut client = connect(harness.port);
        assert_eq!(call(&mut client, &auth_frame(&acme)), "+OK\r\n");
        assert_eq!(wait_for_open(&harness, "acme", 1), 1);
    }
    assert_eq!(
        wait_for_open(&harness, "acme", 0),
        0,
        "a closed connection must not stay in the gauge"
    );
    // The cumulative counter keeps its history, which is the point of having
    // both.
    assert_eq!(
        harness
            .metrics
            .per_org_snapshot()
            .get("acme")
            .map(|c| c.connections_total),
        Some(1)
    );
    harness.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_quit_also_releases_the_gauge() {
    // A different exit path through `serve_inner`, which is exactly the class of
    // bug the guard exists to prevent: one forgotten `return` and the count
    // never comes down.
    let harness = start().await;
    let acme = key_for(&harness, "acme").await;
    let mut client = connect(harness.port);
    assert_eq!(call(&mut client, &auth_frame(&acme)), "+OK\r\n");
    assert_eq!(wait_for_open(&harness, "acme", 1), 1);

    call(&mut client, b"*1\r\n$4\r\nQUIT\r\n");
    assert_eq!(wait_for_open(&harness, "acme", 0), 0);
    harness.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn re_authenticating_moves_the_connection_rather_than_copying_it() {
    let harness = start().await;
    let acme = key_for(&harness, "acme").await;
    let globex = key_for(&harness, "globex").await;

    let mut client = connect(harness.port);
    assert_eq!(call(&mut client, &auth_frame(&acme)), "+OK\r\n");
    assert_eq!(wait_for_open(&harness, "acme", 1), 1);

    assert_eq!(call(&mut client, &auth_frame(&globex)), "+OK\r\n");
    assert_eq!(
        wait_for_open(&harness, "globex", 1),
        1,
        "the connection must be counted under its new org"
    );
    assert_eq!(
        wait_for_open(&harness, "acme", 0),
        0,
        "and no longer under the old one"
    );
    harness.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn the_platform_credential_is_named_not_hidden() {
    // Hiding it would make the panel disagree with the totals for a reason
    // nobody could see.
    let harness = start().await;
    let mut client = connect(harness.port);
    assert_eq!(
        call(&mut client, &auth_frame("instance-secret-key-long")),
        "+OK\r\n"
    );
    assert_eq!(wait_for_open(&harness, "(platform)", 1), 1);
    harness.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn two_orgs_are_counted_separately() {
    let harness = start().await;
    let acme = key_for(&harness, "acme").await;
    let globex = key_for(&harness, "globex").await;

    let mut a1 = connect(harness.port);
    let mut a2 = connect(harness.port);
    let mut g1 = connect(harness.port);
    call(&mut a1, &auth_frame(&acme));
    call(&mut a2, &auth_frame(&acme));
    call(&mut g1, &auth_frame(&globex));

    assert_eq!(wait_for_open(&harness, "acme", 2), 2);
    assert_eq!(wait_for_open(&harness, "globex", 1), 1);

    // Errors are attributed too, so a panel can show which org is misbehaving.
    call(&mut g1, b"*1\r\n$7\r\nNOSUCHC\r\n");
    let counters = harness.metrics.per_org_snapshot();
    assert!(
        counters.get("globex").map(|c| c.errors_total).unwrap_or(0) >= 1,
        "an error must be attributed to the org that caused it: {counters:?}"
    );
    assert_eq!(
        counters.get("acme").map(|c| c.errors_total),
        Some(0),
        "and not to a bystander"
    );
    harness.shutdown.cancel();
}
