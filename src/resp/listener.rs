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
use crate::resp::commands::{dispatch, AuthBinding, BlockKind, Dispatch, Session};
use crate::resp::protocol::{Decoder, ProtocolError, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
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
    /// Static instance password, or empty for no authentication.
    ///
    /// It authenticates to the *unprefixed* keyspace, matching what the same
    /// key does over HTTP: it is the platform credential, not an org's. Org
    /// scoping comes from an api key, below.
    pub password: String,
    /// Api keys, so `AUTH` can bind a connection to the org that owns the key.
    ///
    /// Optional because the listener has to work in tests and in a build with
    /// no accounts layer initialised; without it only the static password is
    /// accepted and every connection is platform-wide.
    pub auth_store: Option<Arc<crate::api::auth_store::AuthStore>>,
    pub max_clients: usize,
    pub idle_timeout: Duration,
    /// Hard cap on a single connection's read buffer. A peer that never
    /// completes a frame must not be able to grow it without bound.
    pub max_buffer_bytes: usize,
    /// Whether FLUSHDB/FLUSHALL are permitted. Off by default: an accidental
    /// flush from a misconfigured client is unrecoverable without a restore.
    pub allow_flush: bool,
    /// Wakeups for blocking reads. Shared across connections, because the
    /// pusher and the waiter are different clients by definition.
    pub notifier: Arc<crate::engine::notify::KeyNotifier>,
    /// Pub/Sub broker, likewise shared: publisher and subscriber are different
    /// connections by definition.
    pub pubsub: Arc<crate::resp::pubsub::PubSub>,
    /// Inbox depth per subscriber. Bounded so one subscriber that stops reading
    /// cannot grow a publisher's memory without limit.
    pub pubsub_inbox: usize,
    /// TLS, when configured. Absent means the port speaks plaintext RESP, which
    /// is what Redis does by default and what every client assumes unless told
    /// otherwise.
    pub tls: Option<tokio_rustls::TlsAcceptor>,
}

impl RespServer {
    fn requires_auth(&self) -> bool {
        !self.password.is_empty()
    }

    /// Resolve an `AUTH` credential.
    ///
    /// The static instance password wins first and binds no tenant. Otherwise
    /// the secret is looked up as an api key and the connection is bound to
    /// that key's org, which is what makes the `{org}<US>{key}` prefixing real
    /// rather than a code path nothing reaches.
    ///
    /// The username is accepted and ignored. Clients send `default` when only a
    /// password is set, and redis-py sends whatever username it was configured
    /// with; rejecting a mismatch would fail connections over a name that means
    /// nothing here.
    async fn resolve(&self, password: &str) -> Option<AuthBinding> {
        if !self.password.is_empty() && password == self.password {
            return Some(AuthBinding::default());
        }
        let store = self.auth_store.as_ref()?;
        match store.validate_key(password).await {
            Ok(Some(record)) => Some(AuthBinding {
                tenant: record.tenant_id,
                key_id: Some(record.id),
            }),
            Ok(None) => None,
            // A database error is not a valid credential. Failing open here
            // would turn a transient SQLite fault into an authentication
            // bypass.
            Err(e) => {
                tracing::warn!("RESP auth lookup failed: {e}");
                None
            }
        }
    }

    /// Current revocation count, or 0 with no store.
    fn revocation_epoch(&self) -> u64 {
        self.auth_store
            .as_ref()
            .map(|s| s.revocation_epoch())
            .unwrap_or(0)
    }
}

/// Drop the session's credential if the api key behind it was revoked.
///
/// The plan's acceptance criterion is that a revoked key stops working on the
/// *next* command, which rules out caching the answer for the connection's
/// lifetime. Asking the database every command would put a SQLite round-trip in
/// front of every `GET`, so the epoch is the fast path: an atomic load, and a
/// real query only after somebody actually revoked something.
async fn enforce_revocation(server: &RespServer, session: &mut Session, seen_epoch: &mut u64) {
    let Some(key_id) = session.key_id.clone() else {
        return;
    };
    let current = server.revocation_epoch();
    if current == *seen_epoch {
        return;
    }
    let Some(store) = server.auth_store.as_ref() else {
        return;
    };
    match store.key_is_live(&key_id).await {
        Ok(true) => *seen_epoch = current,
        Ok(false) => {
            tracing::info!(key_id = %key_id, "RESP connection deauthenticated: api key revoked");
            session.deauthenticate();
        }
        // Leave the epoch behind on error so the next command retries. Assuming
        // the key is dead would cut live connections on a transient fault;
        // assuming it is alive would keep a revoked one working indefinitely.
        Err(e) => tracing::warn!("RESP revocation check failed: {e}"),
    }
}

/// Build a TLS acceptor from a PEM certificate chain and a PKCS#8 key.
///
/// Deliberately strict about the key format, with a message that says how to
/// convert: an RSA key in the old PEM form loads as an empty list and the
/// failure otherwise reads as "no key in the file you are looking at".
fn load_tls(cert_path: &str, key_path: &str) -> std::io::Result<tokio_rustls::TlsAcceptor> {
    use rustls_pemfile::{certs, pkcs8_private_keys};
    use std::io::BufReader;

    // Before building a config: rustls panics at the first handshake if the
    // provider is ambiguous, and a panic in an accept loop is a dead port.
    crate::install_crypto_provider();

    let invalid = |message: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, message);

    let cert_file = std::fs::File::open(cert_path)
        .map_err(|e| invalid(format!("cannot open RESP TLS cert '{cert_path}': {e}")))?;
    let chain: Vec<_> = certs(&mut BufReader::new(cert_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| invalid(format!("invalid RESP TLS cert '{cert_path}': {e}")))?;
    if chain.is_empty() {
        return Err(invalid(format!("no certificate found in '{cert_path}'")));
    }

    let key_file = std::fs::File::open(key_path)
        .map_err(|e| invalid(format!("cannot open RESP TLS key '{key_path}': {e}")))?;
    let mut keys: Vec<_> = pkcs8_private_keys(&mut BufReader::new(key_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| invalid(format!("invalid RESP TLS key '{key_path}': {e}")))?;
    if keys.is_empty() {
        return Err(invalid(format!(
            "no PKCS#8 private key in '{key_path}'. Convert with \
             `openssl pkcs8 -topk8 -nocrypt`"
        )));
    }

    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            chain,
            rustls::pki_types::PrivateKeyDer::Pkcs8(keys.remove(0)),
        )
        .map_err(|e| invalid(format!("RESP TLS configuration rejected: {e}")))?;
    Ok(tokio_rustls::TlsAcceptor::from(Arc::new(config)))
}

/// Start the listener. Returns immediately; the accept loop runs in a task.
pub async fn spawn(
    config: &crate::config::Config,
    engine: Engine,
    metrics: Arc<RespMetrics>,
    auth_store: Option<Arc<crate::api::auth_store::AuthStore>>,
    shutdown: CancellationToken,
) -> std::io::Result<Option<u16>> {
    if config.resp_port == 0 {
        return Ok(None);
    }
    // Resolve TLS before binding: refusing to start is the right answer to a
    // missing certificate, and doing it after the port is open would leave a
    // window in which the listener served plaintext.
    let tls = match config.resp_tls_paths() {
        None => None,
        Some(Ok((cert, key))) => Some(load_tls(&cert, &key)?),
        Some(Err(reason)) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                reason,
            ))
        }
    };

    let addr = std::net::SocketAddr::new(config.bind_addr, config.resp_port);
    let listener = TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?.port();

    let server = Arc::new(RespServer {
        engine,
        metrics,
        password: config.api_key.clone(),
        auth_store,
        max_clients: config.resp_max_clients.max(1),
        idle_timeout: Duration::from_secs(config.resp_idle_timeout_secs.max(1)),
        max_buffer_bytes: config.resp_max_buffer_bytes.max(1024),
        allow_flush: config.resp_allow_flush,
        notifier: Arc::new(crate::engine::notify::KeyNotifier::new()),
        pubsub: Arc::new(crate::resp::pubsub::PubSub::new()),
        pubsub_inbox: config.resp_pubsub_inbox.max(1),
        tls,
    });

    tracing::info!(
        port = bound,
        max_clients = server.max_clients,
        auth = server.requires_auth(),
        tls = server.tls.is_some(),
        "RESP listener started"
    );
    if server.requires_auth() && server.tls.is_none() {
        // `AUTH` carries an api key that binds the connection to an
        // organization. In the clear, anyone on the path has it.
        tracing::warn!(
            "RESP listener has no TLS: AUTH credentials cross the network in \
             plaintext. Set resp_tls_enabled unless the port is on a trusted \
             network."
        );
    }
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
                // Nagle batches small writes, which for a request/response
                // protocol adds latency to every reply for no throughput gain.
                // Set before the handshake, while the socket is still a socket.
                let _ = stream.set_nodelay(true);
                let outcome = match &server.tls {
                    Some(acceptor) => match acceptor.accept(stream).await {
                        Ok(tls_stream) => serve_connection(&server, tls_stream).await,
                        // A failed handshake is a client problem — a plaintext
                        // client, or a name it would not accept — and logging it
                        // at debug keeps a scanner from filling the log.
                        Err(e) => {
                            tracing::debug!(%peer, "RESP TLS handshake failed: {e}");
                            Ok(())
                        }
                    },
                    None => serve_connection(&server, stream).await,
                };
                if let Err(e) = outcome {
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

/// Serve one connection to completion.
///
/// Deregisters any Pub/Sub subscription on the way out — every return path,
/// including the error ones, which is why the body is wrapped rather than
/// having a `drop_connection` call before each `return`.
async fn serve_connection<S>(server: &RespServer, stream: S) -> std::io::Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut subscriber: Option<crate::resp::pubsub::Subscriber> = None;
    let result = serve_inner(server, stream, &mut subscriber).await;
    if let Some(sub) = subscriber {
        server.pubsub.drop_connection(sub.id);
    }
    result
}

async fn serve_inner<S>(
    server: &RespServer,
    mut stream: S,
    subscriber: &mut Option<crate::resp::pubsub::Subscriber>,
) -> std::io::Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut session = Session::new(server.requires_auth());
    // Revocation count this connection has already reconciled against. Starts
    // at the current value: a key revoked *before* this connection existed
    // cannot have authenticated it.
    let mut seen_epoch = server.revocation_epoch();
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

                    // Before anything else: a key revoked since the last
                    // command must not serve this one.
                    enforce_revocation(server, &mut session, &mut seen_epoch).await;

                    let was_authenticated = session.authenticated;
                    // `AUTH` is resolved here, not inside dispatch: looking a
                    // secret up in the api-key store is async and dispatch is
                    // synchronous on purpose. The epoch is read *before* the
                    // lookup so a revocation racing this auth is caught on the
                    // next command rather than skipped.
                    let resolved = if args[0].eq_ignore_ascii_case(b"auth") {
                        let epoch_before = server.revocation_epoch();
                        let binding = match crate::resp::commands::auth_credential(&args[1..]) {
                            Some((_user, password)) => server.resolve(&password).await,
                            // Wrong arity: let dispatch produce the arity error
                            // rather than reporting a bad password.
                            None => None,
                        };
                        if binding.is_some() {
                            seen_epoch = epoch_before;
                        }
                        binding
                    } else {
                        None
                    };
                    // A push has to wake a parked reader; the keys are
                    // captured before dispatch and notified after, so the
                    // element is already there when the waiter re-checks.
                    let pushed_keys = pushed_list_keys(&session, &args);
                    let outcome = dispatch(
                        &server.engine,
                        &mut session,
                        &args,
                        &|_user, _given| resolved.clone(),
                        Some(server.pubsub.as_ref()),
                        server.allow_flush,
                    );

                    match outcome {
                        Dispatch::Block {
                            keys,
                            timeout,
                            kind,
                        } => {
                            // Flush replies queued so far before parking: a
                            // client that pipelined `SET` then `BLPOP` is
                            // waiting on the SET reply, and holding it until the
                            // block resolves would deadlock the pair.
                            if !out.is_empty() {
                                stream.write_all(&out).await?;
                                out.clear();
                            }
                            let value = block_on_keys(
                                server,
                                &keys,
                                session.tenant.as_deref(),
                                timeout,
                                &kind,
                            )
                            .await;
                            // A blocking move fills its destination, and
                            // another connection may be parked on it. The
                            // notify has to happen here as well as in the
                            // `Reply` arm: a `BRPOPLPUSH` that resolves is a
                            // push, and the chain of workers feeding each other
                            // is exactly the shape kombu uses.
                            for key in &pushed_keys {
                                server.notifier.notify_one(key);
                            }
                            value.encode(&mut out);
                        }
                        Dispatch::PubSub(command) => {
                            for reply in handle_pubsub(server, &session, subscriber, command) {
                                reply.encode(&mut out);
                            }
                        }
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
                            for key in &pushed_keys {
                                server.notifier.notify_one(key);
                            }
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

        // Once subscribed the connection has two sources: the socket and the
        // inbox. A plain read would hold a published message until the client
        // happened to send something, which for a subscriber is never.
        let read = match subscriber.as_mut() {
            Some(sub) => {
                let mut pushed = Vec::new();
                let outcome = tokio::select! {
                    result = tokio::time::timeout(server.idle_timeout, stream.read(&mut chunk)) => {
                        Some(result)
                    }
                    Some(message) = sub.receiver.recv() => {
                        encode_delivery(message).encode(&mut pushed);
                        None
                    }
                };
                if !pushed.is_empty() {
                    stream.write_all(&pushed).await?;
                }
                match outcome {
                    Some(Ok(result)) => result?,
                    // A subscribed connection is never idle-closed: it is
                    // waiting for messages by design, which is exactly what an
                    // idle timeout would otherwise punish.
                    Some(Err(_)) => return Ok(()),
                    None => continue,
                }
            }
            None => {
                match tokio::time::timeout(server.idle_timeout, stream.read(&mut chunk)).await {
                    Ok(result) => result?,
                    // An idle connection is closed rather than held open forever,
                    // which keeps max_clients from being consumed by dead peers.
                    Err(_) => return Ok(()),
                }
            }
        };
        if read == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..read]);
    }
}

async fn reply_protocol_error<S>(stream: &mut S, error: &ProtocolError) -> std::io::Result<()>
where
    S: tokio::io::AsyncWrite + Unpin,
{
    let _ = stream
        .write_all(&Value::Error(error.to_string()).to_bytes())
        .await;
    let _ = stream.shutdown().await;
    Ok(())
}

/// Keys a command pushed onto, so the connection loop knows what to notify.
///
/// Returns empty for everything else. Deliberately conservative: a missed
/// notification only means a waiter falls back to its timeout, whereas a
/// spurious one merely wakes a reader that finds nothing and re-parks.
fn pushed_list_keys(session: &Session, args: &[Vec<u8>]) -> Vec<String> {
    let Some(name) = args.first() else {
        return Vec::new();
    };
    let name = String::from_utf8_lossy(name).to_ascii_uppercase();
    let keys: Vec<&Vec<u8>> = match name.as_str() {
        // A push makes the key non-empty, which is what a parked BLPOP waits
        // for.
        "LPUSH" | "RPUSH" | "LPUSHX" | "RPUSHX" => args[1..].iter().take(1).collect(),
        // A ZADD does the same for a parked BZPOPMIN/BZPOPMAX. Leaving this out
        // is not a missed optimisation: the waiter sleeps until its own timeout
        // and the element sits there unserved, which looks like a hung worker.
        "ZADD" | "ZINCRBY" => args[1..].iter().take(1).collect(),
        // A move fills its *destination*, and another connection may be parked
        // on it — the BRPOPLPUSH chain, where one worker feeds the next.
        "RPOPLPUSH" => args.get(2).into_iter().collect(),
        "LMOVE" | "BLMOVE" | "BRPOPLPUSH" => args.get(2).into_iter().collect(),
        _ => return Vec::new(),
    };
    keys.into_iter()
        .map(|key| scope_key(session, key))
        .collect()
}

/// Same tenant scoping the command layer uses, duplicated here only because the
/// notifier keys must match the store keys exactly.
fn scope_key(session: &Session, key: &[u8]) -> String {
    let key = String::from_utf8_lossy(key);
    match &session.tenant {
        Some(tenant) => format!("{tenant}{}{key}", super::commands::TENANT_SEP),
        None => key.to_string(),
    }
}

/// Park until one of `keys` has something, then serve it.
///
/// Registers interest *before* the first check, which is what makes a push
/// landing in that window wake us rather than being missed. On timeout Redis
/// replies with a null array, distinct from an empty one.
async fn block_on_keys(
    server: &RespServer,
    keys: &[String],
    tenant: Option<&str>,
    timeout: Option<Duration>,
    kind: &BlockKind,
) -> Value {
    let deadline = timeout.map(|t| std::time::Instant::now() + t);
    loop {
        let waiter = server.notifier.waiter(keys);

        // Re-check after registering: an element that arrived before we parked
        // is already here, and waiting for a notification that has been and
        // gone would hang until the timeout.
        for key in keys {
            match try_serve(server, key, tenant, kind) {
                Served::Value(value) => return value,
                Served::Empty => continue,
            }
        }

        let remaining = match deadline {
            Some(at) => {
                let now = std::time::Instant::now();
                if now >= at {
                    // A null array, not an empty one: the client distinguishes
                    // "timed out" from "served an empty value".
                    return Value::Array(None);
                }
                Some(at - now)
            }
            None => None,
        };
        if waiter.wait(remaining).await.is_none() && deadline.is_some() {
            return Value::Array(None);
        }
    }
}

/// Outcome of one non-blocking attempt against one key.
enum Served {
    /// A reply for the client — a value, or an error worth reporting now rather
    /// than blocking until the timeout on a key the client got wrong.
    Value(Value),
    /// Nothing there yet; try the next key or park.
    Empty,
}

fn try_serve(server: &RespServer, key: &str, tenant: Option<&str>, kind: &BlockKind) -> Served {
    use crate::engine::structures::{Structure, Structures};
    let structures = Structures::new(&server.engine);

    match kind {
        BlockKind::Pop { left } => {
            let popped = structures.mutate(key, Structure::empty_list, |s| {
                if *left {
                    s.lpop(1)
                } else {
                    s.rpop(1)
                }
            });
            match popped {
                Ok(applied) => match applied.value.into_iter().next() {
                    // Redis replies [key, element] so a multi-key waiter knows
                    // which queue served it.
                    Some(bytes) => Served::Value(Value::Array(Some(vec![
                        Value::bulk(crate::resp::commands::unscope_key(tenant, key)),
                        Value::bulk(bytes),
                    ]))),
                    None => Served::Empty,
                },
                Err(e) => Served::Value(Value::Error(format!("WRONGTYPE {e}"))),
            }
        }
        BlockKind::ZPop { min } => {
            let popped = structures.mutate(key, Structure::empty_zset, |s| {
                Ok(s.as_zset_mut()?.pop(1, *min))
            });
            match popped {
                Ok(applied) => match applied.value.into_iter().next() {
                    // Three elements, not two: [key, member, score].
                    Some((member, score)) => Served::Value(Value::Array(Some(vec![
                        Value::bulk(crate::resp::commands::unscope_key(tenant, key)),
                        Value::bulk(member),
                        Value::bulk(crate::resp::structures_cmd::format_score(score)),
                    ]))),
                    None => Served::Empty,
                },
                Err(e) => Served::Value(Value::Error(format!("WRONGTYPE {e}"))),
            }
        }
        BlockKind::Move {
            destination,
            from_left,
            to_left,
        } => {
            // Atomic across both keys, like the synchronous `LMOVE`. The
            // blocking form is where it matters most: the element arrives at
            // an arbitrary moment, so the window a pop-then-push leaves open
            // is the one a dying worker is most likely to land in.
            match structures.move_element(key, destination, *from_left, *to_left) {
                Ok(Some(element)) => Served::Value(Value::bulk(element)),
                Ok(None) => Served::Empty,
                Err(e) => Served::Value(Value::Error(format!("WRONGTYPE {e}"))),
            }
        }
    }
}

/// Execute a Pub/Sub command for this connection.
///
/// Lives in the listener rather than the command module because it needs the
/// connection's subscriber id and inbox, which only exist here.
fn handle_pubsub(
    server: &RespServer,
    session: &Session,
    subscriber: &mut Option<crate::resp::pubsub::Subscriber>,
    command: crate::resp::commands::PubSubCommand,
) -> Vec<Value> {
    use crate::resp::commands::PubSubCommand as P;

    // Registering lazily means a connection that never subscribes costs no
    // inbox — most connections are plain command clients.
    let id = match subscriber {
        Some(existing) => existing.id,
        None => {
            let new = server.pubsub.register(server.pubsub_inbox);
            let id = new.id;
            *subscriber = Some(new);
            id
        }
    };
    let tenant = session.tenant.as_deref();

    match command {
        P::Subscribe(channels) => channels
            .into_iter()
            .map(|channel| {
                let count = server.pubsub.subscribe(id, tenant, &channel);
                // Redis confirms each subscription individually, in order, so a
                // client that subscribed to three channels reads three replies.
                Value::Array(Some(vec![
                    Value::bulk("subscribe"),
                    Value::bulk(channel),
                    Value::Integer(count as i64),
                ]))
            })
            .collect(),
        P::PSubscribe(patterns) => patterns
            .into_iter()
            .map(|pattern| {
                let count = server.pubsub.psubscribe(id, tenant, &pattern);
                Value::Array(Some(vec![
                    Value::bulk("psubscribe"),
                    Value::bulk(pattern),
                    Value::Integer(count as i64),
                ]))
            })
            .collect(),
        P::Unsubscribe(targets) => match targets {
            Some(channels) => channels
                .into_iter()
                .map(|channel| {
                    let count = server.pubsub.unsubscribe(id, tenant, Some(&channel));
                    Value::Array(Some(vec![
                        Value::bulk("unsubscribe"),
                        Value::bulk(channel),
                        Value::Integer(count as i64),
                    ]))
                })
                .collect(),
            None => {
                let count = server.pubsub.unsubscribe(id, tenant, None);
                vec![Value::Array(Some(vec![
                    Value::bulk("unsubscribe"),
                    Value::nil(),
                    Value::Integer(count as i64),
                ]))]
            }
        },
        P::PUnsubscribe(targets) => match targets {
            Some(patterns) => patterns
                .into_iter()
                .map(|pattern| {
                    let count = server.pubsub.punsubscribe(id, tenant, Some(&pattern));
                    Value::Array(Some(vec![
                        Value::bulk("punsubscribe"),
                        Value::bulk(pattern),
                        Value::Integer(count as i64),
                    ]))
                })
                .collect(),
            None => {
                let count = server.pubsub.punsubscribe(id, tenant, None);
                vec![Value::Array(Some(vec![
                    Value::bulk("punsubscribe"),
                    Value::nil(),
                    Value::Integer(count as i64),
                ]))]
            }
        },
        P::Publish { channel, payload } => {
            // The count is receivers *in this tenant*: a global one would leak
            // the existence of other tenants' subscribers.
            let delivered = server.pubsub.publish(tenant, &channel, &payload);
            vec![Value::Integer(delivered as i64)]
        }
        P::Channels(pattern) => vec![Value::Array(Some(
            server
                .pubsub
                .channels(tenant, pattern.as_deref())
                .into_iter()
                .map(Value::bulk)
                .collect(),
        ))],
        P::NumSub(channels) => {
            let mut out = Vec::with_capacity(channels.len() * 2);
            for channel in channels {
                let count = server.pubsub.subscriber_count(tenant, &channel);
                out.push(Value::bulk(channel));
                out.push(Value::Integer(count as i64));
            }
            vec![Value::Array(Some(out))]
        }
    }
}

/// Encode a delivered message in the shape the subscriber expects.
///
/// A pattern delivery carries four elements rather than three, and the extra
/// one is the pattern — a client that subscribed to several patterns needs it
/// to route the message.
fn encode_delivery(message: crate::resp::pubsub::Delivery) -> Value {
    match message.pattern {
        Some(pattern) => Value::Array(Some(vec![
            Value::bulk("pmessage"),
            Value::bulk(pattern),
            Value::bulk(message.channel),
            Value::bulk(message.payload),
        ])),
        None => Value::Array(Some(vec![
            Value::bulk("message"),
            Value::bulk(message.channel),
            Value::bulk(message.payload),
        ])),
    }
}
