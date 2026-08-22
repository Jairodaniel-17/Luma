//! End-to-end tests over a real TCP socket against the RESP listener.
//!
//! Block 4 of `docs/PLAN-MAESTRO.md`. The unit tests in `src/resp` cover the
//! parser and the command semantics; these cover what only shows up on a
//! socket — pipelining, partial writes, the connection cap, idle timeouts, and
//! that a framing error closes the connection instead of desynchronising it.
//!
//! **What these do not cover:** the differential suite against a real Redis 7,
//! which `docs/SPEC-resp.md` calls the source of truth for semantics. That needs
//! docker and belongs in CI, not here. These tests pin the wire behaviour we
//! intend; the differential suite is what proves it matches Redis.

use luma::config::Config;
use luma::engine::Engine;
use luma::resp::listener::{spawn, RespMetrics};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_util::sync::CancellationToken;

struct Server {
    port: u16,
    metrics: Arc<RespMetrics>,
    shutdown: CancellationToken,
    _dir: tempfile::TempDir,
}

async fn start(mut tune: impl FnMut(&mut Config)) -> Server {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        // Port 0 lets the OS pick, avoiding a race with another test.
        resp_port: 0,
        api_key: String::new(),
        ..Config::default()
    };
    tune(&mut config);
    // `resp_port: 0` means "disabled" in production, so the test binds
    // explicitly and reads back the port the OS chose.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    config.resp_port = listener.local_addr().unwrap().port();
    drop(listener);

    let shutdown = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown.clone()).unwrap();
    let metrics = Arc::new(RespMetrics::default());
    let port = spawn(&config, engine, Arc::clone(&metrics), shutdown.clone())
        .await
        .unwrap()
        .expect("listener must bind");

    Server {
        port,
        metrics,
        shutdown,
        _dir: dir,
    }
}

async fn connect(server: &Server) -> TcpStream {
    let stream = TcpStream::connect(("127.0.0.1", server.port))
        .await
        .unwrap();
    stream.set_nodelay(true).unwrap();
    stream
}

/// Send raw bytes and read whatever comes back within a short window.
async fn exchange(stream: &mut TcpStream, request: &[u8]) -> Vec<u8> {
    stream.write_all(request).await.unwrap();
    let mut out = Vec::new();
    let mut chunk = [0u8; 4096];
    // One read is enough for these replies; the timeout keeps a bug from
    // hanging the suite.
    if let Ok(Ok(n)) = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut chunk)).await {
        out.extend_from_slice(&chunk[..n]);
    }
    out
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

#[tokio::test]
async fn responds_to_an_inline_ping() {
    // The `nc host 6379` smoke test a human reaches for.
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    assert_eq!(text(&exchange(&mut client, b"PING\r\n").await), "+PONG\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn responds_to_a_typed_command() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(
        &mut client,
        b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$5\r\nvalue\r\n",
    )
    .await;
    assert_eq!(text(&reply), "+OK\r\n");

    let reply = exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await;
    assert_eq!(text(&reply), "$5\r\nvalue\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_frame_split_across_writes_is_reassembled() {
    // TCP is free to deliver a frame in pieces. A server that treats one read
    // as one command works until it meets a real network.
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;

    client.write_all(b"*2\r\n$3\r\nGET").await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let reply = exchange(&mut client, b"\r\n$1\r\nk\r\n").await;
    assert_eq!(text(&reply), "$-1\r\n", "reply after reassembly: {reply:?}");
    server.shutdown.cancel();
}

#[tokio::test]
async fn pipelined_commands_all_get_replies() {
    // A pipelining client sends N commands before reading any reply. Answering
    // only the first stalls it forever.
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(
        &mut client,
        b"*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nPING\r\n*1\r\n$4\r\nPING\r\n",
    )
    .await;
    assert_eq!(text(&reply), "+PONG\r\n+PONG\r\n+PONG\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn auth_is_required_when_a_password_is_configured() {
    let server = start(|c| c.api_key = "s3cret".to_string()).await;
    let mut client = connect(&server).await;

    let reply = exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await;
    assert!(
        text(&reply).starts_with("-NOAUTH"),
        "expected NOAUTH, got {}",
        text(&reply)
    );

    let reply = exchange(&mut client, b"*2\r\n$4\r\nAUTH\r\n$6\r\ns3cret\r\n").await;
    assert_eq!(text(&reply), "+OK\r\n");

    let reply = exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await;
    assert_eq!(text(&reply), "$-1\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_wrong_password_is_counted_and_refused() {
    let server = start(|c| c.api_key = "s3cret".to_string()).await;
    let mut client = connect(&server).await;
    let reply = exchange(&mut client, b"*2\r\n$4\r\nAUTH\r\n$5\r\nwrong\r\n").await;
    assert!(text(&reply).starts_with("-WRONGPASS"), "{}", text(&reply));
    assert_eq!(
        server
            .metrics
            .auth_failures_total
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "a rejected AUTH must be visible in metrics — it is the signal for a \
         brute-force attempt"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_framing_error_closes_the_connection() {
    // A desynchronised stream cannot be recovered by skipping bytes: we no
    // longer know where the next frame begins, so continuing would risk
    // executing something nobody sent.
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(&mut client, b"$abc\r\n").await;
    assert!(
        text(&reply).starts_with("-Protocol error"),
        "got {}",
        text(&reply)
    );

    // The socket must now be closed: a further read returns 0 bytes.
    let mut chunk = [0u8; 64];
    let n = tokio::time::timeout(Duration::from_secs(2), client.read(&mut chunk))
        .await
        .expect("read should not hang")
        .unwrap();
    assert_eq!(n, 0, "the connection must be closed after a framing error");
    server.shutdown.cancel();
}

#[tokio::test]
async fn quit_replies_then_closes() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(&mut client, b"*1\r\n$4\r\nQUIT\r\n").await;
    assert_eq!(text(&reply), "+OK\r\n");

    let mut chunk = [0u8; 64];
    let n = tokio::time::timeout(Duration::from_secs(2), client.read(&mut chunk))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(n, 0);
    server.shutdown.cancel();
}

#[tokio::test]
async fn the_connection_cap_refuses_with_a_reason() {
    // A bare socket reset looks like a network fault and gets retried forever;
    // saying why lets a client back off.
    let server = start(|c| c.resp_max_clients = 1).await;
    let mut first = connect(&server).await;
    assert_eq!(text(&exchange(&mut first, b"PING\r\n").await), "+PONG\r\n");

    let mut second = connect(&server).await;
    let mut chunk = [0u8; 256];
    let n = tokio::time::timeout(Duration::from_secs(2), second.read(&mut chunk))
        .await
        .expect("the server must answer rather than hang")
        .unwrap();
    assert!(
        text(&chunk[..n]).starts_with("-ERR max number of clients"),
        "got {}",
        text(&chunk[..n])
    );
    assert_eq!(
        server
            .metrics
            .rejected_at_limit_total
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn an_idle_connection_is_closed() {
    // Otherwise dead peers consume the max_clients budget indefinitely.
    let server = start(|c| c.resp_idle_timeout_secs = 1).await;
    let mut client = connect(&server).await;
    assert_eq!(text(&exchange(&mut client, b"PING\r\n").await), "+PONG\r\n");

    let mut chunk = [0u8; 64];
    let n = tokio::time::timeout(Duration::from_secs(5), client.read(&mut chunk))
        .await
        .expect("the idle timeout should have fired")
        .unwrap();
    assert_eq!(n, 0, "an idle connection must be closed by the server");
    server.shutdown.cancel();
}

#[tokio::test]
async fn metrics_count_commands_and_connections() {
    use std::sync::atomic::Ordering;
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    exchange(&mut client, b"PING\r\n").await;
    exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await;

    assert_eq!(server.metrics.connections_total.load(Ordering::Relaxed), 1);
    assert_eq!(server.metrics.commands_total.load(Ordering::Relaxed), 2);

    let mut rendered = String::new();
    server.metrics.render(&mut rendered);
    assert!(rendered.contains("resp_commands_total 2"), "{rendered}");
    assert!(
        rendered.contains("# TYPE resp_connections_open gauge"),
        "{rendered}"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_disabled_listener_binds_nothing() {
    // The default. Starting to listen on upgrade would be a surprise, and on a
    // shared host a conflict with the real Redis.
    let dir = tempfile::tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        resp_port: 0,
        ..Config::default()
    };
    let shutdown = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown.clone()).unwrap();
    let bound = spawn(
        &config,
        engine,
        Arc::new(RespMetrics::default()),
        shutdown.clone(),
    )
    .await
    .unwrap();
    assert!(bound.is_none());
    shutdown.cancel();
}

// ─── block 6: blocking reads over a real socket ──────────────────────────────

#[tokio::test]
async fn blpop_returns_immediately_when_data_is_already_there() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    exchange(
        &mut client,
        b"*3\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$3\r\njob\r\n",
    )
    .await;

    let reply = exchange(&mut client, b"*3\r\n$5\r\nBLPOP\r\n$1\r\nq\r\n$1\r\n0\r\n").await;
    // Redis replies [key, element] so a multi-key waiter knows which queue
    // served it.
    assert_eq!(text(&reply), "*2\r\n$1\r\nq\r\n$3\r\njob\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn blpop_parks_until_another_client_pushes() {
    // The property that makes it a queue rather than a poll loop: the waiter is
    // released by the push, not by its own timeout.
    let server = start(|_| {}).await;
    let mut waiter = connect(&server).await;

    waiter
        .write_all(b"*3\r\n$5\r\nBLPOP\r\n$4\r\nwork\r\n$1\r\n5\r\n")
        .await
        .unwrap();
    // Give it time to actually park rather than racing the push.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut producer = connect(&server).await;
    let started = std::time::Instant::now();
    exchange(
        &mut producer,
        b"*3\r\n$5\r\nRPUSH\r\n$4\r\nwork\r\n$5\r\nhello\r\n",
    )
    .await;

    let mut chunk = [0u8; 256];
    let n = tokio::time::timeout(Duration::from_secs(3), waiter.read(&mut chunk))
        .await
        .expect("the waiter should have been woken by the push")
        .unwrap();
    assert_eq!(text(&chunk[..n]), "*2\r\n$4\r\nwork\r\n$5\r\nhello\r\n");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the push should wake it well before the 5s timeout"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn blpop_times_out_with_a_null_array() {
    // Null, not empty: the client distinguishes "timed out" from "served an
    // empty value".
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(
        &mut client,
        b"*3\r\n$5\r\nBLPOP\r\n$4\r\nidle\r\n$3\r\n0.2\r\n",
    )
    .await;
    assert_eq!(text(&reply), "*-1\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn one_push_wakes_exactly_one_of_two_waiters() {
    // Otherwise a single job would wake every worker, and all but one would
    // find an empty queue and re-park — the thundering herd the notifier
    // exists to prevent.
    let server = start(|_| {}).await;
    let mut first = connect(&server).await;
    let mut second = connect(&server).await;

    for client in [&mut first, &mut second] {
        client
            .write_all(b"*3\r\n$5\r\nBLPOP\r\n$1\r\nq\r\n$1\r\n2\r\n")
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut producer = connect(&server).await;
    exchange(
        &mut producer,
        b"*3\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$3\r\none\r\n",
    )
    .await;

    let mut served = 0;
    for client in [&mut first, &mut second] {
        let mut chunk = [0u8; 256];
        if let Ok(Ok(n)) =
            tokio::time::timeout(Duration::from_millis(900), client.read(&mut chunk)).await
        {
            if n > 0 && text(&chunk[..n]).starts_with("*2") {
                served += 1;
            }
        }
    }
    assert_eq!(served, 1, "exactly one waiter may be served by one push");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_multi_key_blpop_reports_which_key_served_it() {
    let server = start(|_| {}).await;
    let mut waiter = connect(&server).await;
    waiter
        .write_all(b"*4\r\n$5\r\nBLPOP\r\n$2\r\nq1\r\n$2\r\nq2\r\n$1\r\n3\r\n")
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut producer = connect(&server).await;
    exchange(
        &mut producer,
        b"*3\r\n$5\r\nRPUSH\r\n$2\r\nq2\r\n$1\r\nx\r\n",
    )
    .await;

    let mut chunk = [0u8; 256];
    let n = tokio::time::timeout(Duration::from_secs(3), waiter.read(&mut chunk))
        .await
        .expect("should be woken")
        .unwrap();
    assert_eq!(text(&chunk[..n]), "*2\r\n$2\r\nq2\r\n$1\r\nx\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_transaction_round_trips_over_the_socket() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    assert_eq!(text(&exchange(&mut client, b"MULTI\r\n").await), "+OK\r\n");
    assert_eq!(
        text(&exchange(&mut client, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\n7\r\n").await),
        "+QUEUED\r\n"
    );
    assert_eq!(
        text(&exchange(&mut client, b"EXEC\r\n").await),
        "*1\r\n+OK\r\n"
    );
    assert_eq!(
        text(&exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await),
        "$1\r\n7\r\n"
    );
    server.shutdown.cancel();
}
