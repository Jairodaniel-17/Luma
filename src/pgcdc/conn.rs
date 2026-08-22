//! A Postgres connection that can be put into logical replication mode.
//!
//! `tokio-postgres` is used elsewhere in this module for ordinary work —
//! catalog queries and the backfill `COPY`. It cannot be used here: the
//! released 0.7.18 rejects `replication=database` as an unknown connection-
//! string key, has no `copy_both_simple`, and its message parser does not know
//! the `CopyBothResponse` tag. All three live only on an unreleased branch.
//! `src/pgcdc/pgoutput.rs` carries the full survey.
//!
//! So the framing is here, and it is deliberately the smallest thing that
//! works: startup, authentication, simple query, and the COPY-BOTH duplex. The
//! one piece **not** hand-rolled is SCRAM-SHA-256, which comes from the
//! published `postgres-protocol`. Authentication is where a subtle mistake is a
//! security bug rather than a parse error.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use bytes::BytesMut;
use postgres_protocol::authentication::sasl::{ChannelBinding, ScramSha256};
use postgres_protocol::message::frontend;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::rustls::pki_types::ServerName;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

use super::pgoutput::PG_EPOCH_OFFSET_SECS;

/// Whether the connection must be encrypted.
///
/// Only two settings, not libpq's five. `prefer` — try TLS, silently continue
/// without it — is the one that looks safe in a config file and is not: a
/// downgrade produces no error anywhere. Replication traffic is the entire
/// contents of the database, so this is a yes-or-no question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SslMode {
    Disable,
    Require,
}

impl SslMode {
    pub fn parse(text: &str) -> Result<SslMode> {
        match text.trim().to_ascii_lowercase().as_str() {
            "disable" => Ok(SslMode::Disable),
            "require" => Ok(SslMode::Require),
            "prefer" => bail!(
                "sslmode=prefer is not supported: it silently continues unencrypted when the \
                 server declines TLS, and a downgrade of a replication stream should not be \
                 something that happens without an error. Use require or disable."
            ),
            other => bail!("unknown sslmode {other:?}: expected require or disable"),
        }
    }
}

/// Where and how to connect.
#[derive(Debug, Clone)]
pub struct PgConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub database: String,
    pub ssl_mode: SslMode,
    /// Whether to ask for a replication connection. A replication connection
    /// cannot run ordinary queries and an ordinary one cannot stream, so this
    /// is fixed when the connection is opened.
    pub replication: bool,
    pub connect_timeout: Duration,
}

impl PgConfig {
    /// Parse a libpq-style URL: `postgres://user:pass@host:5432/db?sslmode=require`.
    pub fn from_url(url: &str) -> Result<PgConfig> {
        let rest = url
            .strip_prefix("postgres://")
            .or_else(|| url.strip_prefix("postgresql://"))
            .ok_or_else(|| {
                anyhow!("a Postgres URL must start with postgres:// or postgresql://")
            })?;

        let (authority, tail) = match rest.split_once('/') {
            Some((a, t)) => (a, t),
            None => (rest, ""),
        };
        let (credentials, hostport) = match authority.rsplit_once('@') {
            Some((c, h)) => (c, h),
            None => ("", authority),
        };
        let (user, password) = match credentials.split_once(':') {
            Some((u, p)) => (percent_decode(u), percent_decode(p)),
            None => (percent_decode(credentials), String::new()),
        };
        let (host, port) = match hostport.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse()
                    .context("the port in the Postgres URL is not a number")?,
            ),
            None => (hostport.to_string(), 5432),
        };
        let (database, query) = match tail.split_once('?') {
            Some((d, q)) => (d.to_string(), q),
            None => (tail.to_string(), ""),
        };

        let mut ssl_mode = SslMode::Require;
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            if let Some((k, v)) = pair.split_once('=') {
                if k.eq_ignore_ascii_case("sslmode") {
                    ssl_mode = SslMode::parse(v)?;
                }
            }
        }

        if user.is_empty() {
            bail!("the Postgres URL names no user");
        }
        if database.is_empty() {
            bail!("the Postgres URL names no database");
        }

        Ok(PgConfig {
            host,
            port,
            user,
            password,
            database,
            ssl_mode,
            replication: false,
            connect_timeout: Duration::from_secs(10),
        })
    }
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A TCP stream that may or may not have been upgraded to TLS.
enum Stream {
    Plain(TcpStream),
    Tls(Box<tokio_rustls::client::TlsStream<TcpStream>>),
}

impl Stream {
    async fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        match self {
            Stream::Plain(s) => s.write_all(bytes).await,
            Stream::Tls(s) => s.write_all(bytes).await,
        }
    }

    async fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Stream::Plain(s) => s.flush().await,
            Stream::Tls(s) => s.flush().await,
        }
    }

    async fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        match self {
            Stream::Plain(s) => s.read_exact(buf).await.map(|_| ()),
            Stream::Tls(s) => s.read_exact(buf).await.map(|_| ()),
        }
    }
}

/// One backend message, framed but not interpreted.
#[derive(Debug)]
struct Raw {
    tag: u8,
    body: Vec<u8>,
}

/// A message on the replication stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamMessage {
    /// WAL contents. `data` is one pgoutput message.
    XLogData {
        start_lsn: u64,
        end_lsn: u64,
        clock: i64,
        data: Vec<u8>,
    },
    /// The server checking we are still here.
    ///
    /// `reply_requested` is not advisory: ignoring it lets the server decide
    /// the standby is gone and drop the connection, which reads downstream as a
    /// mysterious periodic disconnect.
    Keepalive {
        end_lsn: u64,
        clock: i64,
        reply_requested: bool,
    },
}

/// The largest single backend message we will buffer.
///
/// A `Relation` message for a wide table or one `XLogData` carrying a large
/// TOASTed value are both legitimately big, so this is generous. It exists
/// because the length is a number the peer chooses, and without a ceiling a
/// hostile or corrupt one is an allocation of up to 4 GiB.
const MAX_MESSAGE_BYTES: usize = 256 * 1024 * 1024;

pub struct PgConnection {
    stream: Stream,
    /// Set once `START_REPLICATION` has been accepted. Guards the two halves
    /// apart: a simple query on a streaming connection would be interleaved
    /// into the COPY data and desynchronize both sides.
    streaming: bool,
}

impl PgConnection {
    /// Open, authenticate, and wait for the server to be ready.
    pub async fn connect(config: &PgConfig) -> Result<PgConnection> {
        let addr = format!("{}:{}", config.host, config.port);
        let tcp = tokio::time::timeout(config.connect_timeout, TcpStream::connect(&addr))
            .await
            .with_context(|| format!("timed out connecting to {addr}"))?
            .with_context(|| format!("could not connect to {addr}"))?;
        tcp.set_nodelay(true).ok();

        let mut stream = Stream::Plain(tcp);
        if config.ssl_mode == SslMode::Require {
            stream = upgrade_to_tls(stream, &config.host).await?;
        }

        let mut conn = PgConnection {
            stream,
            streaming: false,
        };
        conn.startup(config).await?;
        Ok(conn)
    }

    async fn startup(&mut self, config: &PgConfig) -> Result<()> {
        let mut buf = BytesMut::new();
        let mut params: Vec<(&str, &str)> = vec![
            ("user", config.user.as_str()),
            ("database", config.database.as_str()),
            ("application_name", "luma-cdc"),
            // Dates and numbers arrive as text and are parsed downstream, so
            // the server's locale must not decide their shape.
            ("DateStyle", "ISO, YMD"),
            ("client_encoding", "UTF8"),
        ];
        if config.replication {
            // The whole reason this connection is hand-rolled: the released
            // tokio-postgres rejects this key outright.
            params.push(("replication", "database"));
        }
        frontend::startup_message(params.into_iter(), &mut buf)
            .context("encoding the startup message")?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        self.authenticate(config).await?;

        // Then parameter status, key data, and finally ReadyForQuery.
        loop {
            let msg = self.read_message().await?;
            match msg.tag {
                b'S' | b'K' | b'N' => continue,
                b'Z' => return Ok(()),
                b'E' => bail!("Postgres refused the connection: {}", error_text(&msg.body)),
                other => bail!("unexpected message {:?} while starting up", other as char),
            }
        }
    }

    /// The server's first word after startup decides everything here, so this
    /// reads exactly one message rather than looping: any method other than
    /// SCRAM is refused outright, and SCRAM runs its own exchange.
    async fn authenticate(&mut self, config: &PgConfig) -> Result<()> {
        let msg = self.read_message().await?;
        if msg.tag == b'E' {
            bail!("authentication failed: {}", error_text(&msg.body));
        }
        if msg.tag != b'R' {
            bail!(
                "expected an authentication message, got {:?}",
                msg.tag as char
            );
        }
        let kind =
            read_i32(&msg.body, 0).ok_or_else(|| anyhow!("truncated authentication message"))?;
        match kind {
            0 => Ok(()),
            10 => self.scram(config, &msg.body[4..]).await,
            3 => bail!(
                "the server asked for a cleartext password. Luma will not send one: set the \
                 role's password_encryption to scram-sha-256."
            ),
            5 => bail!(
                "the server asked for md5 authentication, which Luma does not implement. md5 \
                 password verification has been superseded since Postgres 10; set the role's \
                 password_encryption to scram-sha-256 and reset its password."
            ),
            other => bail!("unsupported authentication method {other}"),
        }
    }

    /// SCRAM-SHA-256, via `postgres-protocol`.
    ///
    /// Channel binding is declared unsupported rather than unrequested. The
    /// difference is not cosmetic: `unrequested` tells the server the client
    /// cannot see a TLS channel at all, and a server configured to require
    /// binding must be able to refuse us rather than silently accept a session
    /// that is not bound to its channel.
    async fn scram(&mut self, config: &PgConfig, mechanisms: &[u8]) -> Result<()> {
        let offered: Vec<String> = mechanisms
            .split(|&b| b == 0)
            .filter(|m| !m.is_empty())
            .map(|m| String::from_utf8_lossy(m).into_owned())
            .collect();
        if !offered.iter().any(|m| m == "SCRAM-SHA-256") {
            bail!("the server offered {offered:?}, none of which Luma implements");
        }

        let mut scram = ScramSha256::new(config.password.as_bytes(), ChannelBinding::unsupported());
        let mut buf = BytesMut::new();
        frontend::sasl_initial_response("SCRAM-SHA-256", scram.message(), &mut buf)?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        let msg = self.read_message().await?;
        if msg.tag == b'E' {
            bail!("authentication failed: {}", error_text(&msg.body));
        }
        let kind = read_i32(&msg.body, 0).unwrap_or(-1);
        if msg.tag != b'R' || kind != 11 {
            bail!("expected a SASL continue, got {:?}", msg.tag as char);
        }
        scram
            .update(&msg.body[4..])
            .context("the server's SCRAM challenge was rejected")?;

        buf.clear();
        frontend::sasl_response(scram.message(), &mut buf)?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        let msg = self.read_message().await?;
        if msg.tag == b'E' {
            bail!("authentication failed: {}", error_text(&msg.body));
        }
        let kind = read_i32(&msg.body, 0).unwrap_or(-1);
        if msg.tag != b'R' || kind != 12 {
            bail!("expected a SASL final, got {:?}", msg.tag as char);
        }
        // Verifying the server's signature is what makes SCRAM mutual: without
        // it we have proven ourselves to whoever answered, and learned nothing
        // about who that was.
        scram
            .finish(&msg.body[4..])
            .context("the server failed SCRAM verification — it does not hold this password")?;

        let msg = self.read_message().await?;
        if msg.tag != b'R' || read_i32(&msg.body, 0) != Some(0) {
            bail!("expected AuthenticationOk after SCRAM");
        }
        Ok(())
    }

    /// Run a simple query and return its rows as text.
    ///
    /// Text rather than typed: everything this is used for — `IDENTIFY_SYSTEM`,
    /// `CREATE_REPLICATION_SLOT`, catalog lookups — is either already text or
    /// is about to be turned into JSON anyway.
    pub async fn simple_query(&mut self, sql: &str) -> Result<Vec<Vec<Option<String>>>> {
        if self.streaming {
            bail!("this connection is streaming replication and cannot run queries");
        }
        let mut buf = BytesMut::new();
        frontend::query(sql, &mut buf).context("encoding a query")?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        let mut rows = Vec::new();
        let mut failure: Option<String> = None;
        loop {
            let msg = self.read_message().await?;
            match msg.tag {
                b'T' | b'C' | b'S' | b'N' | b'I' => continue,
                b'D' => rows.push(parse_data_row(&msg.body)?),
                b'E' => failure = Some(error_text(&msg.body)),
                // ReadyForQuery ends the exchange whether it succeeded or not,
                // so the error is remembered and reported here — returning at
                // 'E' would leave the unread ReadyForQuery to corrupt the next
                // query on this connection.
                b'Z' => {
                    return match failure {
                        Some(text) => Err(anyhow!("{text}")),
                        None => Ok(rows),
                    }
                }
                other => bail!("unexpected message {:?} during a query", other as char),
            }
        }
    }

    /// Run `COPY ... TO STDOUT` and hand each row to a callback.
    ///
    /// Streamed rather than collected: a backfill's whole point is a table that
    /// does not fit in memory, and a function that returned `Vec<Vec<_>>` would
    /// make the size of the table the size of the process.
    ///
    /// The callback may stop the copy by returning an error, and the connection
    /// is left usable — the remaining frames are drained rather than abandoned,
    /// because a half-read COPY desynchronizes everything after it.
    pub async fn copy_out<F>(&mut self, sql: &str, mut on_row: F) -> Result<u64>
    where
        F: FnMut(Vec<Option<String>>) -> Result<()>,
    {
        if self.streaming {
            bail!("this connection is streaming replication and cannot run queries");
        }
        let mut buf = BytesMut::new();
        frontend::query(sql, &mut buf).context("encoding a COPY")?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        let mut rows = 0u64;
        let mut failure: Option<anyhow::Error> = None;
        // Text format arrives as whole lines, but a CopyData frame is not
        // guaranteed to be one: it can carry several rows or half of one.
        let mut pending: Vec<u8> = Vec::new();
        loop {
            let msg = self.read_message().await?;
            match msg.tag {
                // CopyOutResponse, then the data, then CopyDone.
                b'H' | b'c' | b'C' | b'S' | b'N' | b'T' => continue,
                b'd' => {
                    if failure.is_some() {
                        continue;
                    }
                    pending.extend_from_slice(&msg.body);
                    while let Some(at) = pending.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = pending.drain(..=at).collect();
                        let line = &line[..line.len() - 1];
                        // The trailer of a text-format copy.
                        if line == b"\\." {
                            continue;
                        }
                        match decode_copy_row(line) {
                            Ok(row) => match on_row(row) {
                                Ok(()) => rows += 1,
                                Err(e) => failure = Some(e),
                            },
                            Err(e) => failure = Some(e),
                        }
                        if failure.is_some() {
                            break;
                        }
                    }
                }
                b'E' => failure = Some(anyhow!("{}", error_text(&msg.body))),
                b'Z' => {
                    return match failure {
                        Some(e) => Err(e),
                        None => Ok(rows),
                    }
                }
                other => bail!("unexpected message {:?} during a COPY", other as char),
            }
        }
    }

    /// Ask the server to start sending WAL, and switch to streaming mode.
    ///
    /// `start_lsn` of 0 means "wherever the slot left off", which is what a
    /// restart wants. A saved LSN is passed through verbatim.
    pub async fn start_replication(
        &mut self,
        slot: &str,
        publications: &[String],
        start_lsn: u64,
    ) -> Result<()> {
        let names = publications
            .iter()
            .map(|p| quote_identifier(p))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "START_REPLICATION SLOT {} LOGICAL {} (proto_version '1', publication_names '{}')",
            quote_identifier(slot),
            super::pgoutput::format_lsn(start_lsn),
            names.replace('\'', "''"),
        );

        let mut buf = BytesMut::new();
        frontend::query(&sql, &mut buf)?;
        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        loop {
            let msg = self.read_message().await?;
            match msg.tag {
                // CopyBothResponse: the tag the released tokio-postgres does
                // not know, and the reason this file exists.
                b'W' => {
                    self.streaming = true;
                    return Ok(());
                }
                b'N' | b'S' => continue,
                b'E' => bail!("START_REPLICATION was refused: {}", error_text(&msg.body)),
                other => bail!(
                    "expected CopyBothResponse, got {:?} — is the slot in use?",
                    other as char
                ),
            }
        }
    }

    /// The next message on the replication stream.
    pub async fn next_stream_message(&mut self) -> Result<StreamMessage> {
        loop {
            let msg = self.read_message().await?;
            match msg.tag {
                b'd' => {
                    if msg.body.is_empty() {
                        bail!("an empty CopyData frame arrived");
                    }
                    match msg.body[0] {
                        b'w' => {
                            if msg.body.len() < 25 {
                                bail!("an XLogData frame was too short for its header");
                            }
                            return Ok(StreamMessage::XLogData {
                                start_lsn: read_u64(&msg.body, 1).unwrap(),
                                end_lsn: read_u64(&msg.body, 9).unwrap(),
                                clock: read_i64(&msg.body, 17).unwrap(),
                                data: msg.body[25..].to_vec(),
                            });
                        }
                        b'k' => {
                            if msg.body.len() < 18 {
                                bail!("a keepalive frame was too short");
                            }
                            return Ok(StreamMessage::Keepalive {
                                end_lsn: read_u64(&msg.body, 1).unwrap(),
                                clock: read_i64(&msg.body, 9).unwrap(),
                                reply_requested: msg.body[17] != 0,
                            });
                        }
                        other => bail!("unknown replication frame {:?}", other as char),
                    }
                }
                b'N' => continue,
                b'E' => bail!("the replication stream failed: {}", error_text(&msg.body)),
                b'c' => bail!("the server ended the replication stream"),
                other => bail!("unexpected message {:?} while streaming", other as char),
            }
        }
    }

    /// Tell the server how far we have got.
    ///
    /// This is what lets Postgres release WAL. A connector that streams happily
    /// and never sends one of these will hold every WAL segment since the slot
    /// was created and eventually fill the primary's disk — the failure mode
    /// that gives replication slots their reputation.
    pub async fn send_standby_status(
        &mut self,
        write_lsn: u64,
        flush_lsn: u64,
        apply_lsn: u64,
        reply_requested: bool,
    ) -> Result<()> {
        let mut body = Vec::with_capacity(34);
        body.push(b'r');
        // The protocol wants the position of the *next* byte expected, which is
        // one past what we have. Sending the last received byte instead makes
        // the server resend one record after every restart.
        body.extend_from_slice(&(write_lsn + 1).to_be_bytes());
        body.extend_from_slice(&(flush_lsn + 1).to_be_bytes());
        body.extend_from_slice(&(apply_lsn + 1).to_be_bytes());
        body.extend_from_slice(&pg_now_micros().to_be_bytes());
        body.push(u8::from(reply_requested));

        // CopyData, framed here: `postgres-protocol` encodes `copy_done` but
        // not `copy_data`, because the released client never sends any.
        let mut frame = Vec::with_capacity(body.len() + 5);
        frame.push(b'd');
        frame.extend_from_slice(&((body.len() + 4) as u32).to_be_bytes());
        frame.extend_from_slice(&body);
        self.stream.write_all(&frame).await?;
        self.stream.flush().await?;
        Ok(())
    }

    async fn read_message(&mut self) -> Result<Raw> {
        let mut header = [0u8; 5];
        self.stream
            .read_exact(&mut header)
            .await
            .context("the Postgres connection closed")?;
        let tag = header[0];
        let len = u32::from_be_bytes([header[1], header[2], header[3], header[4]]) as usize;
        if len < 4 {
            bail!("a message claimed a length of {len}, which cannot include its own header");
        }
        let body_len = len - 4;
        if body_len > MAX_MESSAGE_BYTES {
            bail!(
                "a message claimed {body_len} bytes, past the {MAX_MESSAGE_BYTES}-byte ceiling; \
                 refusing to allocate for it"
            );
        }
        let mut body = vec![0u8; body_len];
        if body_len > 0 {
            self.stream
                .read_exact(&mut body)
                .await
                .context("the Postgres connection closed mid-message")?;
        }
        Ok(Raw { tag, body })
    }
}

async fn upgrade_to_tls(stream: Stream, host: &str) -> Result<Stream> {
    let Stream::Plain(mut tcp) = stream else {
        bail!("the connection is already encrypted");
    };
    // SSLRequest: a length and the magic number, with no message tag. It is the
    // one frame in the protocol that does not have one.
    let mut request = Vec::with_capacity(8);
    request.extend_from_slice(&8i32.to_be_bytes());
    request.extend_from_slice(&80_877_103i32.to_be_bytes());
    tcp.write_all(&request).await?;
    tcp.flush().await?;

    let mut answer = [0u8; 1];
    tcp.read_exact(&mut answer).await?;
    if answer[0] != b'S' {
        bail!(
            "the server declined TLS (answered {:?}) and sslmode is require",
            answer[0] as char
        );
    }

    crate::install_crypto_provider();
    let roots = RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = ServerName::try_from(host.to_string())
        .with_context(|| format!("{host:?} is not a valid TLS server name"))?;
    let tls = TlsConnector::from(Arc::new(config))
        .connect(name, tcp)
        .await
        .context("the TLS handshake with Postgres failed")?;
    Ok(Stream::Tls(Box::new(tls)))
}

/// Quote an identifier for a replication command.
///
/// These commands are not ordinary SQL and do not accept parameters, so the
/// slot and publication names are interpolated. They come from a config file
/// rather than a request, but a config file is still not a reason to build a
/// command by concatenation.
fn quote_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn parse_data_row(body: &[u8]) -> Result<Vec<Option<String>>> {
    let count = read_i16(body, 0).ok_or_else(|| anyhow!("truncated data row"))? as usize;
    let mut values = Vec::with_capacity(count);
    let mut at = 2;
    for _ in 0..count {
        let len = read_i32(body, at).ok_or_else(|| anyhow!("truncated data row"))?;
        at += 4;
        if len < 0 {
            values.push(None);
            continue;
        }
        let len = len as usize;
        if at + len > body.len() {
            bail!("a data row column ran past the end of the message");
        }
        values.push(Some(
            String::from_utf8_lossy(&body[at..at + len]).into_owned(),
        ));
        at += len;
    }
    Ok(values)
}

/// Decode one line of `COPY ... TO STDOUT` text format.
///
/// Two passes, and the order is Postgres's rather than the convenient one.
/// Fields are split first — on unescaped tabs, since a literal tab inside a
/// value arrives as an escape — and only then de-escaped. The null marker is
/// compared against the **raw** field, before de-escaping.
///
/// Both halves of that were bugs here first, and the second was caught by this
/// file's own tests. Splitting after de-escaping breaks a value containing a
/// tab into two columns and shifts every value after it, invisible in a tidy
/// table and corrupting the first row of free text. And treating a leading
/// null marker as NULL makes the ordinary value that de-escapes to "Nope"
/// arrive as a missing field.
fn decode_copy_row(line: &[u8]) -> Result<Vec<Option<String>>> {
    let mut columns = Vec::new();
    for field in split_copy_fields(line)? {
        if field == NULL_MARKER {
            columns.push(None);
        } else {
            columns.push(Some(unescape_copy_field(field)?));
        }
    }
    Ok(columns)
}

/// What `COPY ... TO STDOUT` writes for a NULL, in text format.
const NULL_MARKER: &[u8] = b"\\N";

/// Split a COPY line on its unescaped tabs.
fn split_copy_fields(line: &[u8]) -> Result<Vec<&[u8]>> {
    let mut fields = Vec::new();
    let mut start = 0;
    let mut at = 0;
    while at < line.len() {
        match line[at] {
            b'\t' => {
                fields.push(&line[start..at]);
                at += 1;
                start = at;
            }
            // Whatever follows a backslash belongs to this field, tab included.
            b'\\' => {
                if at + 1 >= line.len() {
                    bail!("a COPY row ended on a backslash");
                }
                at += 2;
            }
            _ => at += 1,
        }
    }
    fields.push(&line[start..]);
    Ok(fields)
}

/// Undo the backslash escaping of one COPY field.
fn unescape_copy_field(field: &[u8]) -> Result<String> {
    let mut out: Vec<u8> = Vec::with_capacity(field.len());
    let mut at = 0;
    while at < field.len() {
        if field[at] != b'\\' {
            out.push(field[at]);
            at += 1;
            continue;
        }
        let next = *field
            .get(at + 1)
            .ok_or_else(|| anyhow!("a COPY field ended on a backslash"))?;
        at += 2;
        match next {
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'v' => out.push(0x0b),
            b'\\' => out.push(b'\\'),
            b'x' => {
                let mut value = 0u8;
                let mut digits = 0;
                while digits < 2 {
                    match field.get(at).and_then(|c| (*c as char).to_digit(16)) {
                        Some(d) => {
                            value = value * 16 + d as u8;
                            at += 1;
                            digits += 1;
                        }
                        None => break,
                    }
                }
                if digits == 0 {
                    out.push(b'x');
                } else {
                    out.push(value);
                }
            }
            b'0'..=b'7' => {
                let mut value = (next - b'0') as u32;
                let mut digits = 1;
                while digits < 3 {
                    match field.get(at).filter(|c| (b'0'..=b'7').contains(c)) {
                        Some(c) => {
                            value = value * 8 + (c - b'0') as u32;
                            at += 1;
                            digits += 1;
                        }
                        None => break,
                    }
                }
                out.push(value as u8);
            }
            // Postgres's documented fallback: an unrecognised escape is the
            // character itself.
            other => out.push(other),
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// The human-readable part of an ErrorResponse.
///
/// Postgres sends a dozen fields; the message and the detail are what an
/// operator needs, and dumping the rest buries them.
fn error_text(body: &[u8]) -> String {
    let mut message = String::new();
    let mut detail = String::new();
    let mut hint = String::new();
    let mut at = 0;
    while at < body.len() && body[at] != 0 {
        let kind = body[at];
        at += 1;
        let end = match body[at..].iter().position(|&b| b == 0) {
            Some(e) => at + e,
            None => break,
        };
        let value = String::from_utf8_lossy(&body[at..end]).into_owned();
        at = end + 1;
        match kind {
            b'M' => message = value,
            b'D' => detail = value,
            b'H' => hint = value,
            _ => {}
        }
    }
    let mut text = if message.is_empty() {
        "the server reported an error with no message".to_string()
    } else {
        message
    };
    if !detail.is_empty() {
        text.push_str(&format!(" ({detail})"));
    }
    if !hint.is_empty() {
        text.push_str(&format!(" — {hint}"));
    }
    text
}

fn read_i16(body: &[u8], at: usize) -> Option<i16> {
    let b = body.get(at..at + 2)?;
    Some(i16::from_be_bytes([b[0], b[1]]))
}

fn read_i32(body: &[u8], at: usize) -> Option<i32> {
    let b = body.get(at..at + 4)?;
    Some(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_i64(body: &[u8], at: usize) -> Option<i64> {
    let b = body.get(at..at + 8)?;
    Some(i64::from_be_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn read_u64(body: &[u8], at: usize) -> Option<u64> {
    read_i64(body, at).map(|v| v as u64)
}

/// Now, in the microseconds-since-2000 Postgres expects.
fn pg_now_micros() -> i64 {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    unix - PG_EPOCH_OFFSET_SECS * 1_000_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_url_parses_into_its_parts() {
        let c = PgConfig::from_url("postgres://luma:secret@db.internal:15432/orders").unwrap();
        assert_eq!(c.host, "db.internal");
        assert_eq!(c.port, 15432);
        assert_eq!(c.user, "luma");
        assert_eq!(c.password, "secret");
        assert_eq!(c.database, "orders");
        // TLS unless the URL says otherwise: the default for a stream carrying
        // the whole database should not be plaintext.
        assert_eq!(c.ssl_mode, SslMode::Require);
    }

    #[test]
    fn a_password_with_an_at_sign_survives() {
        // rsplit on '@' rather than split: a password containing one is
        // ordinary, and splitting on the first would take half of it as the
        // host.
        let c = PgConfig::from_url("postgres://luma:p%40ss@host/db").unwrap();
        assert_eq!(c.password, "p@ss");
        assert_eq!(c.host, "host");
        assert_eq!(c.port, 5432);
    }

    #[test]
    fn sslmode_prefer_is_refused_rather_than_honoured() {
        // It is the setting that looks safe in a config file and is not: with
        // prefer, a server that declines TLS gets an unencrypted replication
        // stream and nothing anywhere reports it.
        let err = SslMode::parse("prefer").unwrap_err().to_string();
        assert!(err.contains("silently"), "{err}");
        assert_eq!(SslMode::parse("disable").unwrap(), SslMode::Disable);
        assert_eq!(SslMode::parse("REQUIRE").unwrap(), SslMode::Require);
        assert!(SslMode::parse("verify-full").is_err());
    }

    #[test]
    fn a_url_without_a_database_is_refused_at_parse_time() {
        // Rather than connecting and failing later with a message about the
        // database being named after the user.
        assert!(PgConfig::from_url("postgres://luma@host").is_err());
        assert!(PgConfig::from_url("postgres://host/db").is_err());
        assert!(PgConfig::from_url("mysql://luma@host/db").is_err());
    }

    #[test]
    fn an_identifier_with_a_quote_cannot_end_its_own_quoting() {
        assert_eq!(quote_identifier("slot"), "\"slot\"");
        assert_eq!(quote_identifier("we\"ird"), "\"we\"\"ird\"");
    }

    #[test]
    fn an_error_response_reduces_to_the_part_an_operator_reads() {
        let mut body = Vec::new();
        for (kind, value) in [
            (b'S', "ERROR"),
            (b'C', "55006"),
            (b'M', "replication slot \"s\" is active"),
            (b'H', "stop the other consumer"),
        ] {
            body.push(kind);
            body.extend_from_slice(value.as_bytes());
            body.push(0);
        }
        body.push(0);
        let text = error_text(&body);
        assert!(text.contains("is active"), "{text}");
        assert!(text.contains("stop the other consumer"), "{text}");
        assert!(
            !text.contains("55006"),
            "the code buries the message: {text}"
        );
    }

    #[test]
    fn a_malformed_error_response_still_produces_something() {
        // An error while parsing an error must not be what the operator sees.
        assert!(!error_text(&[]).is_empty());
        assert!(!error_text(b"M").is_empty());
        assert!(!error_text(b"Mno terminator").is_empty());
    }

    #[test]
    fn a_data_row_reads_nulls_apart_from_empty_strings() {
        // -1 is NULL and 0 is the empty string. Reading one as the other turns
        // an absent LSN into "0/0", which is a valid position in the WAL.
        let mut body = Vec::new();
        body.extend_from_slice(&3i16.to_be_bytes());
        body.extend_from_slice(&(-1i32).to_be_bytes());
        body.extend_from_slice(&0i32.to_be_bytes());
        body.extend_from_slice(&3i32.to_be_bytes());
        body.extend_from_slice(b"abc");
        let row = parse_data_row(&body).unwrap();
        assert_eq!(row, vec![None, Some(String::new()), Some("abc".into())]);
    }

    #[test]
    fn a_data_row_that_overruns_is_an_error_not_a_panic() {
        let mut body = Vec::new();
        body.extend_from_slice(&1i16.to_be_bytes());
        body.extend_from_slice(&999i32.to_be_bytes());
        body.extend_from_slice(b"ab");
        assert!(parse_data_row(&body).is_err());
        assert!(parse_data_row(&[]).is_err());
    }

    #[test]
    fn a_copy_row_splits_on_real_tabs_and_not_on_escaped_ones() {
        // The bug this guards: splitting on tabs before undoing the escaping
        // breaks a value that contains one into two columns and shifts every
        // value after it. Invisible in a tidy table; corrupts the first row of
        // free text.
        let row = decode_copy_row(b"1\\tone\tsecond").unwrap();
        assert_eq!(
            row,
            vec![Some("1\tone".into()), Some("second".into())],
            "the escaped tab belongs inside the first column"
        );
    }

    #[test]
    fn a_copy_null_is_not_the_two_character_string() {
        let row = decode_copy_row(b"a\t\\N\t").unwrap();
        assert_eq!(row, vec![Some("a".into()), None, Some(String::new())]);
        // And a value that merely starts with it is text, not NULL: `\Nope` is
        // the escape fallback applied to N, giving "Nope".
        let row = decode_copy_row(b"\\Nope").unwrap();
        assert_eq!(row, vec![Some("Nope".into())]);
    }

    #[test]
    fn the_copy_escapes_postgres_documents_all_decode() {
        let row = decode_copy_row(b"a\\nb\\rc\\\\d\\x41\\101").unwrap();
        assert_eq!(row, vec![Some("a\nb\rc\\dAA".into())]);
    }

    #[test]
    fn a_copy_row_ending_on_a_backslash_is_an_error_not_a_panic() {
        assert!(decode_copy_row(b"value\\").is_err());
        // An empty line is one empty column, which is what Postgres means by it.
        assert_eq!(decode_copy_row(b"").unwrap(), vec![Some(String::new())]);
    }

    #[test]
    fn the_postgres_clock_is_offset_from_the_unix_one() {
        // A standby status update dated 1970 tells the server the standby is
        // thirty years behind, which is not a state it handles gracefully.
        let now = pg_now_micros();
        let unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        assert!(now < unix, "the Postgres epoch is later than the Unix one");
        assert_eq!((unix - now) / 1_000_000, PG_EPOCH_OFFSET_SECS);
    }
}
