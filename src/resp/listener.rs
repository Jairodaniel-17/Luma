//! The RESP TCP listener.
//!
//! F1.1 (connection loop), F1.4 (metrics) and F4.1 (limits) of
//! `docs/PLAN-MAESTRO.md`.
//!
//! F4.1 was pulled forward from block 8 on purpose — deviation D-1 in the plan.
//! Exposing a new TCP port with no connection cap, no idle timeout and no bound
//! on a per-connection buffer is opening a denial-of-service vector and closing
//! it four releases later. The limits ship with the listener or the listener
//! does not ship.
//!
//! ## Off by default
//!
//! `resp_port` is 0 unless configured. A convergent engine that starts
//! listening on 6379 the moment you upgrade is a surprise, and on a shared host
//! it is a port conflict with the actual Redis.

use crate::engine::Engine;
use crate::resp::commands::{dispatch, Dispatch, Session};
use crate::resp::protocol::{Decoder, ProtocolError, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;

/// Live counters, exposed through `/v1/metrics` alongside the HTTP ones.
#[derive(Default)]
pub struct RespMetrics {
    pub connections_open: AtomicU64,
    pub connections_total: AtomicU64,
    pub commands_total: AtomicU64,
    pub errors_total: AtomicU64,
    pub auth_failures_total: AtomicU64,
    pub rejected_at_limit_total: AtomicU64,
}

impl RespMetrics {
    /// Render in the Prometheus text format, appended to the existing output.
    pub fn render(&self, out: &mut String) {
        use std::fmt::Write as _;
        let _ = writeln!(
            out,
            "# HELP resp_connections_open Currently open RESP connections\n\
             # TYPE resp_connections_open gauge\n\
             resp_connections_open {}",
            self.connections_open.load(Ordering::Relaxed)
        );
        for (name, help, value) in [
            (
                "resp_connections_total",
                "RESP connections accepted",
                &self.connections_total,
            ),
            (
                "resp_commands_total",
                "RESP commands executed",
                &self.commands_total,
            ),
            (
                "resp_errors_total",
                "RESP commands that replied with an error",
                &self.errors_total,
            ),
            (
                "resp_auth_failures_total",
                "Rejected RESP AUTH attempts",
                &self.auth_failures_total,
            ),
            (
                "resp_rejected_at_limit_total",
                "Connections refused because max_clients was reached",
                &self.rejected_at_limit_total,
            ),
        ] {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}",
                value.load(Ordering::Relaxed)
            );
        }
    }
}

/// Everything a connection needs.
pub struct RespServer {
    pub engine: Engine,
    pub metrics: Arc<RespMetrics>,
    /// Static password, or empty for no authentication.
    pub password: String,
    pub max_clients: usize,
    pub idle_timeout: Duration,
    /// Hard cap on a single connection's read buffer. A peer that never
    /// completes a frame must not be able to grow it without bound.
    pub max_buffer_bytes: usize,
}

impl RespServer {
    fn requires_auth(&self) -> bool {
        !self.password.is_empty()
    }
}

/// Start the listener. Returns immediately; the accept loop runs in a task.
pub async fn spawn(
    config: &crate::config::Config,
    engine: Engine,
    metrics: Arc<RespMetrics>,
    shutdown: CancellationToken,
) -> std::io::Result<Option<u16>> {
    if config.resp_port == 0 {
        return Ok(None);
    }
    let addr = std::net::SocketAddr::new(config.bind_addr, config.resp_port);
    let listener = TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?.port();

    let server = Arc::new(RespServer {
        engine,
        metrics,
        password: config.api_key.clone(),
        max_clients: config.resp_max_clients.max(1),
        idle_timeout: Duration::from_secs(config.resp_idle_timeout_secs.max(1)),
        max_buffer_bytes: config.resp_max_buffer_bytes.max(1024),
    });

    tracing::info!(
        port = bound,
        max_clients = server.max_clients,
        auth = server.requires_auth(),
        "RESP listener started"
    );
    if !server.requires_auth() {
        // Loud, because an unauthenticated Redis port is how a lot of data gets
        // stolen; the HTTP side warns the same way about a missing api key.
        tracing::warn!(
            "RESP listener has no password: any client that reaches the port has full access"
        );
    }

    tokio::spawn(async move {
        loop {
            let accepted = tokio::select! {
                result = listener.accept() => result,
                _ = shutdown.cancelled() => break,
            };
            let (stream, peer) = match accepted {
                Ok(pair) => pair,
                // An accept error is usually transient (fd exhaustion); killing
                // the loop would silently stop serving for the process lifetime.
                Err(e) => {
                    tracing::warn!("RESP accept failed: {e}");
                    continue;
                }
            };

            let open = server.metrics.connections_open.load(Ordering::Relaxed) as usize;
            if open >= server.max_clients {
                server
                    .metrics
                    .rejected_at_limit_total
                    .fetch_add(1, Ordering::Relaxed);
                // Tell the client why rather than dropping the socket: a bare
                // reset looks like a network fault and gets retried forever.
                let mut stream = stream;
                let _ = stream
                    .write_all(&Value::Error("ERR max number of clients reached".into()).to_bytes())
                    .await;
                let _ = stream.shutdown().await;
                continue;
            }

            let server = Arc::clone(&server);
            tokio::spawn(async move {
                server
                    .metrics
                    .connections_open
                    .fetch_add(1, Ordering::Relaxed);
                server
                    .metrics
                    .connections_total
                    .fetch_add(1, Ordering::Relaxed);
                if let Err(e) = serve_connection(&server, stream).await {
                    tracing::debug!(%peer, "RESP connection ended: {e}");
                }
                server
                    .metrics
                    .connections_open
                    .fetch_sub(1, Ordering::Relaxed);
            });
        }
        tracing::info!("RESP listener stopped");
    });

    Ok(Some(bound))
}

async fn serve_connection(server: &RespServer, mut stream: TcpStream) -> std::io::Result<()> {
    // Nagle batches small writes, which for a request/response protocol means
    // adding latency to every reply for no throughput gain.
    let _ = stream.set_nodelay(true);

    let mut session = Session::new(server.requires_auth());
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 16 * 1024];

    loop {
        // Decode everything already buffered before reading again: one read can
        // carry several pipelined commands, and replying to only the first
        // would stall a client that is waiting for all of them.
        let mut out = Vec::new();
        loop {
            match Decoder::decode(&buf) {
                Ok(Some((value, used))) => {
                    buf.drain(..used);
                    let args = match value.into_command() {
                        Ok(args) => args,
                        Err(e) => {
                            reply_protocol_error(&mut stream, &e).await?;
                            return Ok(());
                        }
                    };
                    if args.is_empty() {
                        continue;
                    }
                    server
                        .metrics
                        .commands_total
                        .fetch_add(1, Ordering::Relaxed);

                    let was_authenticated = session.authenticated;
                    let password = server.password.clone();
                    let outcome = dispatch(&server.engine, &mut session, &args, |_user, given| {
                        // Static password for now; the api-key/role mapping of
                        // D3 arrives with the accounts wiring.
                        (given == password).then_some(None)
                    });

                    match outcome {
                        Dispatch::Reply(value) => {
                            if matches!(value, Value::Error(_)) {
                                server.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                                if !was_authenticated && !session.authenticated {
                                    server
                                        .metrics
                                        .auth_failures_total
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            value.encode(&mut out);
                        }
                        Dispatch::Quit => {
                            Value::ok().encode(&mut out);
                            stream.write_all(&out).await?;
                            let _ = stream.shutdown().await;
                            return Ok(());
                        }
                    }
                }
                Ok(None) => break,
                // A framing error is unrecoverable: the stream cannot be
                // resynchronised by skipping bytes, because we no longer know
                // where the next frame starts.
                Err(e) => {
                    reply_protocol_error(&mut stream, &e).await?;
                    return Ok(());
                }
            }
        }
        if !out.is_empty() {
            stream.write_all(&out).await?;
        }

        if buf.len() > server.max_buffer_bytes {
            let _ = stream
                .write_all(&Value::Error("ERR Protocol error: request too large".into()).to_bytes())
                .await;
            return Ok(());
        }

        let read = match tokio::time::timeout(server.idle_timeout, stream.read(&mut chunk)).await {
            Ok(result) => result?,
            // An idle connection is closed rather than held open forever, which
            // is what keeps max_clients from being consumed by dead peers.
            Err(_) => return Ok(()),
        };
        if read == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..read]);
    }
}

async fn reply_protocol_error(
    stream: &mut TcpStream,
    error: &ProtocolError,
) -> std::io::Result<()> {
    let _ = stream
        .write_all(&Value::Error(error.to_string()).to_bytes())
        .await;
    let _ = stream.shutdown().await;
    Ok(())
}
