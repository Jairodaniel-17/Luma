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
    let port = spawn(
        &config,
        engine,
        Arc::clone(&metrics),
        None,
        shutdown.clone(),
    )
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
        None,
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

// ─── block 6: pub/sub over a real socket ─────────────────────────────────────

#[tokio::test]
async fn a_subscriber_is_pushed_a_published_message() {
    // The property that makes it Pub/Sub rather than polling: the message
    // arrives without the subscriber sending anything.
    let server = start(|_| {}).await;
    let mut subscriber = connect(&server).await;
    let reply = exchange(&mut subscriber, b"*2\r\n$9\r\nSUBSCRIBE\r\n$4\r\nnews\r\n").await;
    assert_eq!(
        text(&reply),
        "*3\r\n$9\r\nsubscribe\r\n$4\r\nnews\r\n:1\r\n"
    );

    let mut publisher = connect(&server).await;
    let count = exchange(
        &mut publisher,
        b"*3\r\n$7\r\nPUBLISH\r\n$4\r\nnews\r\n$5\r\nhello\r\n",
    )
    .await;
    assert_eq!(text(&count), ":1\r\n", "PUBLISH reports receivers");

    let mut chunk = [0u8; 256];
    let n = tokio::time::timeout(Duration::from_secs(3), subscriber.read(&mut chunk))
        .await
        .expect("the subscriber should be pushed the message")
        .unwrap();
    assert_eq!(
        text(&chunk[..n]),
        "*3\r\n$7\r\nmessage\r\n$4\r\nnews\r\n$5\r\nhello\r\n"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_pattern_subscriber_gets_a_pmessage_with_four_elements() {
    // A pmessage carries the pattern as well, so a client subscribed to several
    // patterns can route what it receives.
    let server = start(|_| {}).await;
    let mut subscriber = connect(&server).await;
    exchange(
        &mut subscriber,
        b"*2\r\n$10\r\nPSUBSCRIBE\r\n$6\r\nnews.*\r\n",
    )
    .await;

    let mut publisher = connect(&server).await;
    exchange(
        &mut publisher,
        b"*3\r\n$7\r\nPUBLISH\r\n$10\r\nnews.sport\r\n$4\r\ngoal\r\n",
    )
    .await;

    let mut chunk = [0u8; 256];
    let n = tokio::time::timeout(Duration::from_secs(3), subscriber.read(&mut chunk))
        .await
        .expect("should be pushed")
        .unwrap();
    assert_eq!(
        text(&chunk[..n]),
        "*4\r\n$8\r\npmessage\r\n$6\r\nnews.*\r\n$10\r\nnews.sport\r\n$4\r\ngoal\r\n"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_fanout_reaches_every_worker() {
    // Celery's fanout exchange: every worker gets the message, not one of them.
    let server = start(|_| {}).await;
    let mut first = connect(&server).await;
    let mut second = connect(&server).await;
    for worker in [&mut first, &mut second] {
        exchange(worker, b"*2\r\n$9\r\nSUBSCRIBE\r\n$4\r\nfany\r\n").await;
    }

    let mut publisher = connect(&server).await;
    let count = exchange(
        &mut publisher,
        b"*3\r\n$7\r\nPUBLISH\r\n$4\r\nfany\r\n$2\r\nhi\r\n",
    )
    .await;
    assert_eq!(text(&count), ":2\r\n");

    for worker in [&mut first, &mut second] {
        let mut chunk = [0u8; 256];
        let n = tokio::time::timeout(Duration::from_secs(3), worker.read(&mut chunk))
            .await
            .expect("every subscriber must receive it")
            .unwrap();
        assert!(
            text(&chunk[..n]).contains("message"),
            "{}",
            text(&chunk[..n])
        );
    }
    server.shutdown.cancel();
}

#[tokio::test]
async fn unsubscribing_stops_delivery() {
    let server = start(|_| {}).await;
    let mut subscriber = connect(&server).await;
    exchange(&mut subscriber, b"*2\r\n$9\r\nSUBSCRIBE\r\n$1\r\nc\r\n").await;
    let reply = exchange(&mut subscriber, b"*2\r\n$11\r\nUNSUBSCRIBE\r\n$1\r\nc\r\n").await;
    assert!(text(&reply).contains("unsubscribe"), "{}", text(&reply));

    let mut publisher = connect(&server).await;
    let count = exchange(
        &mut publisher,
        b"*3\r\n$7\r\nPUBLISH\r\n$1\r\nc\r\n$1\r\nx\r\n",
    )
    .await;
    assert_eq!(text(&count), ":0\r\n", "nobody is listening any more");
    server.shutdown.cancel();
}

#[tokio::test]
async fn pubsub_channels_lists_only_live_channels() {
    let server = start(|_| {}).await;
    let mut subscriber = connect(&server).await;
    exchange(&mut subscriber, b"*2\r\n$9\r\nSUBSCRIBE\r\n$4\r\nlive\r\n").await;

    let mut admin = connect(&server).await;
    let reply = exchange(&mut admin, b"*2\r\n$6\r\nPUBSUB\r\n$8\r\nCHANNELS\r\n").await;
    assert_eq!(text(&reply), "*1\r\n$4\r\nlive\r\n");

    let reply = exchange(
        &mut admin,
        b"*3\r\n$6\r\nPUBSUB\r\n$6\r\nNUMSUB\r\n$4\r\nlive\r\n",
    )
    .await;
    assert_eq!(text(&reply), "*2\r\n$4\r\nlive\r\n:1\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_disconnected_subscriber_is_forgotten() {
    // Otherwise the registry accumulates one dead entry per disconnect and a
    // long-lived server leaks steadily.
    let server = start(|_| {}).await;
    {
        let mut subscriber = connect(&server).await;
        exchange(&mut subscriber, b"*2\r\n$9\r\nSUBSCRIBE\r\n$4\r\ngone\r\n").await;
        // Dropping the stream closes the socket.
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut publisher = connect(&server).await;
    let count = exchange(
        &mut publisher,
        b"*3\r\n$7\r\nPUBLISH\r\n$4\r\ngone\r\n$1\r\nx\r\n",
    )
    .await;
    assert_eq!(text(&count), ":0\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_subscribed_connection_still_answers_commands() {
    // Redis restricts a subscriber to a subset; keeping ordinary commands
    // working is a superset, and a client that sends PING to keep the
    // connection alive must get a reply rather than silence.
    //
    // **The reply is an array, not `+PONG`.** This test asserted `+PONG` and was
    // wrong: inside a subscription Redis answers `["pong", ""]`. ioredis uses
    // PING as its subscriber keepalive, read the simple string where it expected
    // two elements, and desynchronised its command queue — it died with "Command
    // queue state error" the moment anything was published. So this assertion
    // was actively holding the bug in place.
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    exchange(&mut client, b"*2\r\n$9\r\nSUBSCRIBE\r\n$1\r\nc\r\n").await;
    assert_eq!(
        text(&exchange(&mut client, b"PING\r\n").await),
        "*2\r\n$4\r\npong\r\n$0\r\n\r\n"
    );
    // With an argument the second element is that argument, still as an array.
    assert_eq!(
        text(&exchange(&mut client, b"*2\r\n$4\r\nPING\r\n$2\r\nhi\r\n").await),
        "*2\r\n$4\r\npong\r\n$2\r\nhi\r\n"
    );
    // And once the subscription is dropped it goes back to the simple string.
    exchange(&mut client, b"*2\r\n$11\r\nUNSUBSCRIBE\r\n$1\r\nc\r\n").await;
    assert_eq!(text(&exchange(&mut client, b"PING\r\n").await), "+PONG\r\n");
    server.shutdown.cancel();
}

// ─── F1.2: AUTH binds a connection to the org that owns the api key ──────────
//
// Until this was wired the listener only ever accepted the instance-wide
// password and left `tenant` unset, so every RESP client shared one flat
// keyspace and an org's api key did not work over the protocol at all. The
// prefixing code existed and nothing reached it. These tests exist so that
// cannot happen again silently.

/// A server with a real api-key store behind it.
///
/// `AuthStore` needs SQLite, so this cannot use the plain `start` helper. The
/// keys are created through the store's own API rather than by inserting rows,
/// so the test exercises the same hashing the server does.
struct AuthServer {
    inner: Server,
    store: Arc<luma::api::auth_store::AuthStore>,
}

async fn start_with_keys(instance_password: &str) -> AuthServer {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        resp_port: 0,
        api_key: instance_password.to_string(),
        ..Config::default()
    };
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    config.resp_port = listener.local_addr().unwrap().port();
    drop(listener);

    let sqlite = Arc::new(
        luma::sqlite::SqliteService::new(dir.path().join("keys.db")).expect("sqlite must open"),
    );
    let store = Arc::new(luma::api::auth_store::AuthStore::new(sqlite));
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

    AuthServer {
        inner: Server {
            port,
            metrics,
            shutdown,
            _dir: dir,
        },
        store,
    }
}

/// Mint a key for an org and return the plaintext secret plus its row id.
async fn key_for(server: &AuthServer, org: &str) -> (String, String) {
    let plain = server.store.generate_api_key();
    let id = server
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
        .expect("key creation must succeed");
    (plain, id)
}

#[tokio::test]
async fn an_api_key_authenticates_over_resp() {
    // The whole point: a credential minted for an org works on the Redis port,
    // not just over HTTP.
    let server = start_with_keys("instance-secret-key-long").await;
    let (secret, _) = key_for(&server, "acme").await;
    let mut client = connect(&server.inner).await;

    // Unauthenticated commands are refused first, so a pass cannot be mistaken
    // for the server not requiring auth at all.
    let reply = exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await;
    assert!(
        text(&reply).starts_with("-NOAUTH"),
        "expected NOAUTH before AUTH, got {:?}",
        text(&reply)
    );

    let reply = exchange(&mut client, &auth_frame(&secret)).await;
    assert_eq!(text(&reply), "+OK\r\n");
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn two_orgs_with_their_own_keys_cannot_see_each_other() {
    // The acceptance criterion from the SPEC. Both write the same key name and
    // must read back their own value.
    let server = start_with_keys("instance-secret-key-long").await;
    let (acme, _) = key_for(&server, "acme").await;
    let (globex, _) = key_for(&server, "globex").await;

    let mut a = connect(&server.inner).await;
    assert_eq!(text(&exchange(&mut a, &auth_frame(&acme)).await), "+OK\r\n");
    assert_eq!(
        text(&exchange(&mut a, b"*3\r\n$3\r\nSET\r\n$5\r\nshare\r\n$4\r\nmine\r\n").await),
        "+OK\r\n"
    );

    let mut g = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut g, &auth_frame(&globex)).await),
        "+OK\r\n"
    );
    // A nil, not acme's value: the same key name in another org is a different
    // key.
    assert_eq!(
        text(&exchange(&mut g, b"*2\r\n$3\r\nGET\r\n$5\r\nshare\r\n").await),
        "$-1\r\n",
        "globex must not read acme's value"
    );
    assert_eq!(
        text(&exchange(&mut g, b"*3\r\n$3\r\nSET\r\n$5\r\nshare\r\n$5\r\ntheir\r\n").await),
        "+OK\r\n"
    );
    // And acme's value survived globex writing the same name.
    assert_eq!(
        text(&exchange(&mut a, b"*2\r\n$3\r\nGET\r\n$5\r\nshare\r\n").await),
        "$4\r\nmine\r\n",
        "acme's value must not have been overwritten by globex"
    );

    // KEYS must not leak the neighbour either, and must name the key the client
    // used rather than the internal prefixed form.
    let listed = text(&exchange(&mut g, b"*2\r\n$4\r\nKEYS\r\n$1\r\n*\r\n").await);
    assert!(
        listed.contains("share") && !listed.contains("acme"),
        "KEYS leaked another org or the internal prefix: {listed:?}"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn the_instance_password_stays_platform_wide() {
    // The static key is the platform credential, exactly as over HTTP: it sees
    // the unprefixed keyspace. Losing this would break every existing RESP
    // deployment on upgrade.
    let server = start_with_keys("instance-secret-key-long").await;
    let mut client = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut client, &auth_frame("instance-secret-key-long")).await),
        "+OK\r\n"
    );
    assert_eq!(
        text(&exchange(&mut client, b"*3\r\n$3\r\nSET\r\n$4\r\nflat\r\n$1\r\n1\r\n").await),
        "+OK\r\n"
    );
    assert_eq!(
        text(&exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$4\r\nflat\r\n").await),
        "$1\r\n1\r\n"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn a_wrong_secret_is_refused_and_counted() {
    let server = start_with_keys("instance-secret-key-long").await;
    let mut client = connect(&server.inner).await;
    let reply = exchange(&mut client, &auth_frame("not-a-key")).await;
    assert!(
        text(&reply).starts_with("-WRONGPASS"),
        "expected WRONGPASS, got {:?}",
        text(&reply)
    );
    assert_eq!(
        server
            .inner
            .metrics
            .auth_failures_total
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "a rejected AUTH must be visible in the metrics"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn the_two_argument_auth_form_works() {
    // redis-py sends `AUTH <username> <password>` whenever a username is
    // configured. The username means nothing here, but rejecting it would fail
    // the connection over a name.
    let server = start_with_keys("instance-secret-key-long").await;
    let (secret, _) = key_for(&server, "acme").await;
    let mut client = connect(&server.inner).await;
    let frame = format!(
        "*3\r\n$4\r\nAUTH\r\n$7\r\ndefault\r\n${}\r\n{}\r\n",
        secret.len(),
        secret
    );
    assert_eq!(
        text(&exchange(&mut client, frame.as_bytes()).await),
        "+OK\r\n"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn a_revoked_key_stops_working_on_the_next_command() {
    // The SPEC's wording, as a test. The connection stays open — a client has
    // to be told to re-authenticate, and a bare socket reset reads as a network
    // fault and gets retried forever.
    let server = start_with_keys("instance-secret-key-long").await;
    let (secret, key_id) = key_for(&server, "acme").await;
    let mut client = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut client, &auth_frame(&secret)).await),
        "+OK\r\n"
    );
    assert_eq!(
        text(&exchange(&mut client, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\n1\r\n").await),
        "+OK\r\n"
    );

    server.store.revoke_key(&key_id, None).await.unwrap();

    let reply = exchange(&mut client, b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n").await;
    assert!(
        text(&reply).starts_with("-NOAUTH"),
        "a revoked key must stop serving on the very next command, got {:?}",
        text(&reply)
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn revoking_one_key_does_not_cut_another_connection() {
    // The revocation epoch is global, so every connection re-checks after any
    // revocation. It must re-check and *survive*, not assume the worst.
    let server = start_with_keys("instance-secret-key-long").await;
    let (acme, acme_id) = key_for(&server, "acme").await;
    let (globex, _) = key_for(&server, "globex").await;

    let mut a = connect(&server.inner).await;
    assert_eq!(text(&exchange(&mut a, &auth_frame(&acme)).await), "+OK\r\n");
    let mut g = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut g, &auth_frame(&globex)).await),
        "+OK\r\n"
    );

    server.store.revoke_key(&acme_id, None).await.unwrap();

    assert!(text(&exchange(&mut a, b"*1\r\n$6\r\nDBSIZE\r\n").await).starts_with("-NOAUTH"));
    assert!(
        !text(&exchange(&mut g, b"*1\r\n$6\r\nDBSIZE\r\n").await).starts_with("-NOAUTH"),
        "globex's connection must survive acme's key being revoked"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn a_blocking_pop_replies_with_the_key_the_client_asked_for() {
    // BLPOP answers `[key, element]` and kombu matches that key against the one
    // it sent. Handing back the internal `acme<US>jobs` both leaks the layout
    // and breaks the client — which is what happened while `unscope` returned
    // its input unchanged in both branches.
    let server = start_with_keys("instance-secret-key-long").await;
    let (secret, _) = key_for(&server, "acme").await;
    let mut client = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut client, &auth_frame(&secret)).await),
        "+OK\r\n"
    );
    assert_eq!(
        text(
            &exchange(
                &mut client,
                b"*3\r\n$5\r\nLPUSH\r\n$4\r\njobs\r\n$1\r\na\r\n"
            )
            .await
        ),
        ":1\r\n"
    );
    let reply = text(
        &exchange(
            &mut client,
            b"*3\r\n$5\r\nBLPOP\r\n$4\r\njobs\r\n$1\r\n1\r\n",
        )
        .await,
    );
    assert_eq!(
        reply, "*2\r\n$4\r\njobs\r\n$1\r\na\r\n",
        "BLPOP must name the key the client sent, not the prefixed one"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn a_tenant_key_containing_a_colon_cannot_collide_with_another_org() {
    // With `:` as the separator, org `a` holding `b:c` and org `a:b` holding
    // `c` are the same physical key. The unit separator cannot appear in an org
    // id, so the split is unambiguous.
    let server = start_with_keys("instance-secret-key-long").await;
    let (outer, _) = key_for(&server, "a").await;
    let (inner, _) = key_for(&server, "a:b").await;

    let mut one = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut one, &auth_frame(&outer)).await),
        "+OK\r\n"
    );
    assert_eq!(
        text(&exchange(&mut one, b"*3\r\n$3\r\nSET\r\n$3\r\nb:c\r\n$5\r\nouter\r\n").await),
        "+OK\r\n"
    );

    let mut two = connect(&server.inner).await;
    assert_eq!(
        text(&exchange(&mut two, &auth_frame(&inner)).await),
        "+OK\r\n"
    );
    assert_eq!(
        text(&exchange(&mut two, b"*2\r\n$3\r\nGET\r\n$1\r\nc\r\n").await),
        "$-1\r\n",
        "org `a:b` reading `c` must not see org `a`'s `b:c`"
    );
    server.inner.shutdown.cancel();
}

/// One-argument `AUTH` frame for a secret of any length.
fn auth_frame(secret: &str) -> Vec<u8> {
    format!("*2\r\n$4\r\nAUTH\r\n${}\r\n{}\r\n", secret.len(), secret).into_bytes()
}

// ─── F3.1 remainder: BLMOVE, BRPOPLPUSH, BZPOPMIN, BZPOPMAX ──────────────────

#[tokio::test]
async fn brpoplpush_moves_an_element_that_is_already_there() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    exchange(
        &mut client,
        b"*3\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$3\r\njob\r\n",
    )
    .await;
    // Replies with the element alone — the client already knows both keys.
    assert_eq!(
        text(
            &exchange(
                &mut client,
                b"*4\r\n$10\r\nBRPOPLPUSH\r\n$1\r\nq\r\n$2\r\nip\r\n$1\r\n1\r\n"
            )
            .await
        ),
        "$3\r\njob\r\n"
    );
    assert_eq!(
        text(
            &exchange(
                &mut client,
                b"*4\r\n$6\r\nLRANGE\r\n$2\r\nip\r\n$1\r\n0\r\n$2\r\n-1\r\n"
            )
            .await
        ),
        "*1\r\n$3\r\njob\r\n"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn brpoplpush_parks_until_another_client_pushes() {
    // The whole point of the blocking form: a worker waits on an empty queue
    // and is served the moment a producer arrives.
    let server = start(|_| {}).await;
    let mut worker = connect(&server).await;
    let mut producer = connect(&server).await;

    let waiting = tokio::spawn(async move {
        let reply = exchange(
            &mut worker,
            b"*4\r\n$10\r\nBRPOPLPUSH\r\n$1\r\nq\r\n$2\r\nip\r\n$1\r\n5\r\n",
        )
        .await;
        text(&reply)
    });

    // Give the worker time to park before the push, so this exercises the
    // wakeup rather than the already-there fast path.
    tokio::time::sleep(Duration::from_millis(150)).await;
    exchange(
        &mut producer,
        b"*3\r\n$5\r\nLPUSH\r\n$1\r\nq\r\n$4\r\nlate\r\n",
    )
    .await;

    let served = tokio::time::timeout(Duration::from_secs(3), waiting)
        .await
        .expect("the parked worker must be woken, not time out")
        .unwrap();
    assert_eq!(served, "$4\r\nlate\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn brpoplpush_times_out_with_a_null_array() {
    // A null array, not an empty one and not a nil bulk: redis-py distinguishes
    // "nothing arrived" from "served an empty value".
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(
        &mut client,
        b"*4\r\n$10\r\nBRPOPLPUSH\r\n$5\r\nempty\r\n$2\r\nip\r\n$1\r\n1\r\n",
    )
    .await;
    assert_eq!(text(&reply), "*-1\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn blmove_honours_the_sides_it_was_given() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    exchange(
        &mut client,
        b"*4\r\n$5\r\nRPUSH\r\n$3\r\nsrc\r\n$1\r\na\r\n$1\r\nb\r\n",
    )
    .await;
    // LEFT from the source: 'a', not 'b'.
    assert_eq!(
        text(&exchange(
            &mut client,
            b"*6\r\n$6\r\nBLMOVE\r\n$3\r\nsrc\r\n$3\r\ndst\r\n$4\r\nLEFT\r\n$5\r\nRIGHT\r\n$1\r\n1\r\n"
        )
        .await),
        "$1\r\na\r\n"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn blmove_rejects_a_side_it_does_not_understand() {
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let reply = exchange(
        &mut client,
        b"*6\r\n$6\r\nBLMOVE\r\n$3\r\nsrc\r\n$3\r\ndst\r\n$2\r\nUP\r\n$5\r\nRIGHT\r\n$1\r\n1\r\n",
    )
    .await;
    assert!(
        text(&reply).starts_with("-ERR syntax error"),
        "an unknown side must be a syntax error, not a silent block: {:?}",
        text(&reply)
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn bzpopmin_replies_with_three_elements() {
    // The trap for a client written against BLPOP: [key, member, score], not
    // [key, member].
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    exchange(
        &mut client,
        b"*4\r\n$4\r\nZADD\r\n$1\r\nz\r\n$1\r\n2\r\n$3\r\nlow\r\n",
    )
    .await;
    exchange(
        &mut client,
        b"*4\r\n$4\r\nZADD\r\n$1\r\nz\r\n$1\r\n9\r\n$4\r\nhigh\r\n",
    )
    .await;
    assert_eq!(
        text(
            &exchange(
                &mut client,
                b"*3\r\n$8\r\nBZPOPMIN\r\n$1\r\nz\r\n$1\r\n1\r\n"
            )
            .await
        ),
        "*3\r\n$1\r\nz\r\n$3\r\nlow\r\n$1\r\n2\r\n"
    );
    assert_eq!(
        text(
            &exchange(
                &mut client,
                b"*3\r\n$8\r\nBZPOPMAX\r\n$1\r\nz\r\n$1\r\n1\r\n"
            )
            .await
        ),
        "*3\r\n$1\r\nz\r\n$4\r\nhigh\r\n$1\r\n9\r\n"
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn bzpopmin_parks_and_is_woken_by_a_zadd() {
    let server = start(|_| {}).await;
    let mut worker = connect(&server).await;
    let mut producer = connect(&server).await;

    let waiting = tokio::spawn(async move {
        text(
            &exchange(
                &mut worker,
                b"*3\r\n$8\r\nBZPOPMIN\r\n$1\r\nz\r\n$1\r\n5\r\n",
            )
            .await,
        )
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    exchange(
        &mut producer,
        b"*4\r\n$4\r\nZADD\r\n$1\r\nz\r\n$1\r\n1\r\n$1\r\nm\r\n",
    )
    .await;

    let served = tokio::time::timeout(Duration::from_secs(3), waiting)
        .await
        .expect("a ZADD must wake a parked BZPOPMIN")
        .unwrap();
    assert_eq!(served, "*3\r\n$1\r\nz\r\n$1\r\nm\r\n$1\r\n1\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_blocking_move_stays_inside_its_own_org() {
    // Both keys are prefixed, so one org's BRPOPLPUSH cannot drain another's
    // queue even when both are called `q`.
    let server = start_with_keys("instance-secret-key-long").await;
    let (acme, _) = key_for(&server, "acme").await;
    let (globex, _) = key_for(&server, "globex").await;

    let mut a = connect(&server.inner).await;
    exchange(&mut a, &auth_frame(&acme)).await;
    exchange(&mut a, b"*3\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$4\r\nmine\r\n").await;

    let mut g = connect(&server.inner).await;
    exchange(&mut g, &auth_frame(&globex)).await;
    // globex's queue is empty, so this must time out rather than steal.
    assert_eq!(
        text(
            &exchange(
                &mut g,
                b"*4\r\n$10\r\nBRPOPLPUSH\r\n$1\r\nq\r\n$2\r\nip\r\n$1\r\n1\r\n"
            )
            .await
        ),
        "*-1\r\n",
        "one org must not be served from another's queue"
    );
    // And acme's element is still there.
    assert_eq!(
        text(&exchange(&mut a, b"*2\r\n$4\r\nLLEN\r\n$1\r\nq\r\n").await),
        ":1\r\n"
    );
    server.inner.shutdown.cancel();
}

#[tokio::test]
async fn a_move_wakes_a_worker_parked_on_the_destination() {
    // The chain kombu builds: one worker waits on the in-flight list while
    // another moves a job into it. Without notifying the *destination* of a
    // move, the second worker sleeps until its own timeout with the job sitting
    // right there — a hang that looks like an idle system.
    let server = start(|_| {}).await;
    let mut waiter = connect(&server).await;
    let mut mover = connect(&server).await;
    let mut producer = connect(&server).await;

    exchange(
        &mut producer,
        b"*3\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$3\r\njob\r\n",
    )
    .await;

    let parked = tokio::spawn(async move {
        text(&exchange(&mut waiter, b"*3\r\n$5\r\nBLPOP\r\n$2\r\nip\r\n$1\r\n5\r\n").await)
    });
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Non-blocking move: the notify must still fire.
    assert_eq!(
        text(
            &exchange(
                &mut mover,
                b"*3\r\n$9\r\nRPOPLPUSH\r\n$1\r\nq\r\n$2\r\nip\r\n"
            )
            .await
        ),
        "$3\r\njob\r\n"
    );

    let served = tokio::time::timeout(Duration::from_secs(3), parked)
        .await
        .expect("a move into the watched key must wake the parked worker")
        .unwrap();
    assert_eq!(served, "*2\r\n$2\r\nip\r\n$3\r\njob\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_blocking_move_wakes_a_worker_parked_on_its_destination() {
    // Same property through the blocking form, which resolves in a different
    // arm of the connection loop and therefore needs its own notify.
    let server = start(|_| {}).await;
    let mut waiter = connect(&server).await;
    let mut mover = connect(&server).await;
    let mut producer = connect(&server).await;

    let parked = tokio::spawn(async move {
        text(&exchange(&mut waiter, b"*3\r\n$5\r\nBLPOP\r\n$2\r\nip\r\n$1\r\n5\r\n").await)
    });
    let moving = tokio::spawn(async move {
        text(
            &exchange(
                &mut mover,
                b"*4\r\n$10\r\nBRPOPLPUSH\r\n$1\r\nq\r\n$2\r\nip\r\n$1\r\n5\r\n",
            )
            .await,
        )
    });
    tokio::time::sleep(Duration::from_millis(150)).await;
    exchange(
        &mut producer,
        b"*3\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$4\r\nlate\r\n",
    )
    .await;

    let moved = tokio::time::timeout(Duration::from_secs(3), moving)
        .await
        .expect("the mover must be woken by the push")
        .unwrap();
    assert_eq!(moved, "$4\r\nlate\r\n");
    let served = tokio::time::timeout(Duration::from_secs(3), parked)
        .await
        .expect("the second worker must be woken by the move")
        .unwrap();
    assert_eq!(served, "*2\r\n$2\r\nip\r\n$4\r\nlate\r\n");
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_deeply_nested_frame_is_refused_with_a_protocol_error() {
    // Small enough to fit in one write, so the refusal can be read back rather
    // than racing the server's close.
    let server = start(|_| {}).await;
    let mut client = connect(&server).await;
    let mut frame = Vec::new();
    for _ in 0..200 {
        frame.extend_from_slice(b"*1\r\n");
    }
    frame.extend_from_slice(b"$1\r\na\r\n");
    let reply = exchange(&mut client, &frame).await;
    assert!(
        text(&reply).starts_with("-Protocol error"),
        "expected a protocol error, got {:?}",
        text(&reply)
    );
    server.shutdown.cancel();
}

#[tokio::test]
async fn a_huge_nested_frame_does_not_take_the_server_down() {
    // 40 KB of `*1` used to overflow the parser's stack, and a stack overflow
    // cannot be caught: the process died for the price of one packet, from an
    // unauthenticated peer, before AUTH was ever consulted.
    //
    // The refusal itself is checked above. The only claim here is that the
    // *server* survives — this connection may well be reset mid-write, because
    // the server refuses as soon as it has seen enough bytes rather than
    // politely reading all forty kilobytes first.
    let server = start(|_| {}).await;
    {
        let mut client = connect(&server).await;
        let mut frame = Vec::new();
        for _ in 0..10_000 {
            frame.extend_from_slice(b"*1\r\n");
        }
        frame.extend_from_slice(b"$1\r\na\r\n");
        let _ = exchange(&mut client, &frame).await;
    }

    let mut other = connect(&server).await;
    assert_eq!(
        text(&exchange(&mut other, b"PING\r\n").await),
        "+PONG\r\n",
        "the server must still be serving everyone else"
    );
    server.shutdown.cancel();
}
