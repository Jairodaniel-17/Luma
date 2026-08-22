//! RESP command dispatch: strings, keys, and the connection commands.
//!
//! F1.2 and F1.3 of `docs/PLAN-MAESTRO.md`. Commands execute against the
//! existing key-value store, so TTL, compare-and-swap, WAL durability and crash
//! replay are the same machinery the HTTP API uses — a `SET` through RESP and a
//! `PUT /v1/state` land in the same place with the same guarantees.
//!
//! ## Keyspace isolation
//!
//! Every key is stored as `{tenant}:{key}` when the connection is bound to an
//! organization, which is exactly what the HTTP layer's `scope_key` does. Two
//! organizations using the key `celery` therefore cannot see each other, and the
//! isolation is the same code path that is already tested for HTTP rather than a
//! second implementation that could drift.
//!
//! ## Errors
//!
//! Error strings start with the Redis error code — `ERR`, `WRONGTYPE`,
//! `NOAUTH` — because clients match on that prefix. redis-py raises different
//! exception classes based on it, so a reworded message is a behaviour change.

use crate::engine::stored::StoredVal;
use crate::engine::Engine;
use crate::resp::protocol::Value;

/// Per-connection state.
pub struct Session {
    /// Organization this connection is bound to, or `None` for an unscoped
    /// (platform) connection.
    pub tenant: Option<String>,
    /// Whether the connection has authenticated. When the server requires a
    /// password, an unauthenticated connection may only run the handshake
    /// commands — the same subset Redis allows under `requirepass`.
    pub authenticated: bool,
    /// Set by `CLIENT SETNAME`; purely informational, but clients send it and a
    /// server that errors on it looks broken.
    pub name: Option<String>,
}

impl Session {
    pub fn new(requires_auth: bool) -> Self {
        Self {
            tenant: None,
            authenticated: !requires_auth,
            name: None,
        }
    }
}

/// Commands allowed before authenticating. Anything else gets `NOAUTH`, which
/// is what a client uses to know it must send `AUTH` rather than that the
/// command does not exist.
const PRE_AUTH: &[&str] = &["AUTH", "HELLO", "PING", "QUIT", "COMMAND", "RESET"];

fn err(message: impl Into<String>) -> Value {
    Value::Error(message.into())
}

/// Uppercase ASCII name of a command, for dispatch.
fn command_name(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw).to_ascii_uppercase()
}

/// Prefix a key with the connection's tenant, mirroring the HTTP layer.
fn scope(session: &Session, key: &[u8]) -> String {
    let key = String::from_utf8_lossy(key);
    match &session.tenant {
        Some(tenant) => format!("{tenant}:{key}"),
        None => key.to_string(),
    }
}

/// Outcome of dispatching one command.
pub enum Dispatch {
    Reply(Value),
    /// The client asked to close the connection.
    Quit,
}

/// Execute one command.
pub fn dispatch(
    engine: &Engine,
    session: &mut Session,
    args: &[Vec<u8>],
    authenticate: impl Fn(&str, &str) -> Option<Option<String>>,
) -> Dispatch {
    let Some(first) = args.first() else {
        // An empty inline line: Redis answers nothing at all.
        return Dispatch::Reply(Value::Array(Some(Vec::new())));
    };
    let name = command_name(first);
    let rest = &args[1..];

    if !session.authenticated && !PRE_AUTH.contains(&name.as_str()) {
        return Dispatch::Reply(err("NOAUTH Authentication required."));
    }

    match name.as_str() {
        // ── connection ───────────────────────────────────────────────────────
        "QUIT" => Dispatch::Quit,
        "PING" => Dispatch::Reply(match rest.len() {
            0 => Value::Simple("PONG".into()),
            // With an argument PING echoes it, as a bulk string rather than a
            // simple string — clients that use PING as an echo rely on that.
            1 => Value::Bulk(Some(rest[0].clone())),
            _ => err("ERR wrong number of arguments for 'ping' command"),
        }),
        "ECHO" => Dispatch::Reply(match rest.len() {
            1 => Value::Bulk(Some(rest[0].clone())),
            _ => err("ERR wrong number of arguments for 'echo' command"),
        }),
        "AUTH" => Dispatch::Reply(auth(session, rest, authenticate)),
        "HELLO" => Dispatch::Reply(hello(session, rest)),
        "SELECT" => Dispatch::Reply(select(rest)),
        "CLIENT" => Dispatch::Reply(client(session, rest)),
        "COMMAND" => Dispatch::Reply(Value::Array(Some(Vec::new()))),
        "RESET" => {
            session.name = None;
            Dispatch::Reply(Value::Simple("RESET".into()))
        }

        // ── strings ──────────────────────────────────────────────────────────
        "SET" => Dispatch::Reply(set(engine, session, rest)),
        "GET" => Dispatch::Reply(get(engine, session, rest)),
        "GETDEL" => Dispatch::Reply(getdel(engine, session, rest)),
        "SETEX" => Dispatch::Reply(setex(engine, session, rest, 1000)),
        "PSETEX" => Dispatch::Reply(setex(engine, session, rest, 1)),
        "SETNX" => Dispatch::Reply(setnx(engine, session, rest)),
        "STRLEN" => Dispatch::Reply(strlen(engine, session, rest)),
        "INCR" => Dispatch::Reply(incr_by(engine, session, rest, 1)),
        "DECR" => Dispatch::Reply(incr_by(engine, session, rest, -1)),
        "INCRBY" => Dispatch::Reply(incr_by_arg(engine, session, rest, 1)),
        "DECRBY" => Dispatch::Reply(incr_by_arg(engine, session, rest, -1)),
        "MGET" => Dispatch::Reply(mget(engine, session, rest)),
        "MSET" => Dispatch::Reply(mset(engine, session, rest)),

        // ── keys ─────────────────────────────────────────────────────────────
        "DEL" | "UNLINK" => Dispatch::Reply(del(engine, session, rest)),
        "EXISTS" => Dispatch::Reply(exists(engine, session, rest)),
        "TYPE" => Dispatch::Reply(type_of(engine, session, rest)),
        "TTL" => Dispatch::Reply(ttl(engine, session, rest, 1000)),
        "PTTL" => Dispatch::Reply(ttl(engine, session, rest, 1)),
        "EXPIRE" => Dispatch::Reply(expire(engine, session, rest, 1000)),
        "PEXPIRE" => Dispatch::Reply(expire(engine, session, rest, 1)),
        "PERSIST" => Dispatch::Reply(persist(engine, session, rest)),

        other => Dispatch::Reply(err(format!(
            "ERR unknown command '{other}', with args beginning with: "
        ))),
    }
}

// ── connection commands ──────────────────────────────────────────────────────

fn auth(
    session: &mut Session,
    args: &[Vec<u8>],
    authenticate: impl Fn(&str, &str) -> Option<Option<String>>,
) -> Value {
    // Redis 6 added the two-argument form; the one-argument form is still what
    // most clients send when only a password is configured.
    let (user, password) = match args.len() {
        1 => (
            "default".to_string(),
            String::from_utf8_lossy(&args[0]).to_string(),
        ),
        2 => (
            String::from_utf8_lossy(&args[0]).to_string(),
            String::from_utf8_lossy(&args[1]).to_string(),
        ),
        _ => return err("ERR wrong number of arguments for 'auth' command"),
    };

    match authenticate(&user, &password) {
        Some(tenant) => {
            session.authenticated = true;
            session.tenant = tenant;
            Value::ok()
        }
        None => err("WRONGPASS invalid username-password pair or user is disabled."),
    }
}

fn hello(session: &Session, args: &[Vec<u8>]) -> Value {
    // A client may ask for RESP3. We answer the handshake either way but keep
    // speaking RESP2, which every targeted client understands; claiming RESP3
    // and then sending RESP2 replies would break them in ways that only show up
    // under load.
    if let Some(version) = args.first() {
        let requested = String::from_utf8_lossy(version).parse::<i64>().unwrap_or(2);
        if !(2..=3).contains(&requested) {
            return err("NOPROTO unsupported protocol version");
        }
    }
    Value::Map(vec![
        (Value::bulk("server"), Value::bulk("luma")),
        (
            Value::bulk("version"),
            Value::bulk(env!("CARGO_PKG_VERSION")),
        ),
        (Value::bulk("proto"), Value::Integer(2)),
        (Value::bulk("id"), Value::Integer(0)),
        (Value::bulk("mode"), Value::bulk("standalone")),
        (Value::bulk("role"), Value::bulk("master")),
        (
            Value::bulk("name"),
            match &session.name {
                Some(name) => Value::bulk(name.clone()),
                None => Value::bulk(""),
            },
        ),
    ])
}

fn select(args: &[Vec<u8>]) -> Value {
    // One logical database, like Redis configured with `databases 1`. Answering
    // OK for any index would silently share one keyspace between databases a
    // client believes are separate.
    match args.first().map(|a| String::from_utf8_lossy(a).to_string()) {
        Some(index) if index == "0" => Value::ok(),
        Some(_) => err("ERR DB index is out of range"),
        None => err("ERR wrong number of arguments for 'select' command"),
    }
}

fn client(session: &mut Session, args: &[Vec<u8>]) -> Value {
    match args.first().map(|a| command_name(a)).as_deref() {
        Some("SETNAME") => {
            session.name = args.get(1).map(|n| String::from_utf8_lossy(n).to_string());
            Value::ok()
        }
        Some("GETNAME") => match &session.name {
            Some(name) => Value::bulk(name.clone()),
            None => Value::nil(),
        },
        // SETINFO is what redis-py sends on connect to report its library
        // version. Erroring on it makes the client log a warning on every
        // connection for no reason.
        Some("SETINFO") => Value::ok(),
        Some("ID") => Value::Integer(0),
        _ => Value::ok(),
    }
}

// ── string commands ──────────────────────────────────────────────────────────

/// Read a key's raw bytes, whatever form it is stored in.
///
/// A value written through the HTTP API is JSON; one written by `SET` is raw
/// bytes. `GET` has to answer both, because the whole point is that the two
/// surfaces address one store.
fn read_bytes(engine: &Engine, key: &str) -> Option<Vec<u8>> {
    let item = engine.get_state(key)?;
    match &item.value {
        StoredVal::Raw { bytes, .. } => Some(bytes.clone()),
        StoredVal::Json(value) => Some(match value {
            // A JSON string reads back as its contents, not as a quoted literal:
            // `SET k v` then a GET must return `v`, not `"v"`.
            serde_json::Value::String(text) => text.clone().into_bytes(),
            other => other.to_string().into_bytes(),
        }),
    }
}

fn set(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return err("ERR wrong number of arguments for 'set' command");
    }
    let key = scope(session, &args[0]);
    let mut ttl_ms: Option<u64> = None;
    let mut only_if_absent = false;
    let mut only_if_present = false;
    let mut keep_ttl = false;

    let mut i = 2;
    while i < args.len() {
        match command_name(&args[i]).as_str() {
            "EX" | "PX" => {
                let Some(raw) = args.get(i + 1) else {
                    return err("ERR syntax error");
                };
                let Ok(n) = String::from_utf8_lossy(raw).parse::<u64>() else {
                    return err("ERR value is not an integer or out of range");
                };
                let multiplier = if command_name(&args[i]) == "EX" {
                    1000
                } else {
                    1
                };
                ttl_ms = Some(n.saturating_mul(multiplier));
                i += 2;
            }
            "NX" => {
                only_if_absent = true;
                i += 1;
            }
            "XX" => {
                only_if_present = true;
                i += 1;
            }
            "KEEPTTL" => {
                keep_ttl = true;
                i += 1;
            }
            _ => return err("ERR syntax error"),
        }
    }
    if only_if_absent && only_if_present {
        return err("ERR syntax error");
    }

    let existing = engine.get_state(&key);
    if only_if_absent && existing.is_some() {
        return Value::nil();
    }
    if only_if_present && existing.is_none() {
        return Value::nil();
    }
    if keep_ttl {
        // Carry the remaining TTL rather than clearing it, which is what
        // KEEPTTL means and what a session-refresh pattern depends on.
        ttl_ms = existing
            .as_ref()
            .and_then(|item| item.expires_at_ms)
            .map(|at| at.saturating_sub(now_ms()));
    }

    match engine.put_state(key, StoredVal::raw(args[1].clone(), None), ttl_ms, None) {
        Ok(_) => Value::ok(),
        Err(e) => err(format!("ERR {e}")),
    }
}

fn get(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    match args.len() {
        1 => match read_bytes(engine, &scope(session, &args[0])) {
            Some(bytes) => Value::Bulk(Some(bytes)),
            None => Value::nil(),
        },
        _ => err("ERR wrong number of arguments for 'get' command"),
    }
}

fn getdel(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'getdel' command");
    }
    let key = scope(session, &args[0]);
    let value = read_bytes(engine, &key);
    let _ = engine.delete_state(&key);
    match value {
        Some(bytes) => Value::Bulk(Some(bytes)),
        None => Value::nil(),
    }
}

fn setex(engine: &Engine, session: &Session, args: &[Vec<u8>], multiplier: u64) -> Value {
    if args.len() != 3 {
        return err("ERR wrong number of arguments for 'setex' command");
    }
    let Ok(n) = String::from_utf8_lossy(&args[1]).parse::<u64>() else {
        return err("ERR value is not an integer or out of range");
    };
    if n == 0 {
        return err("ERR invalid expire time in 'setex' command");
    }
    match engine.put_state(
        scope(session, &args[0]),
        StoredVal::raw(args[2].clone(), None),
        Some(n.saturating_mul(multiplier)),
        None,
    ) {
        Ok(_) => Value::ok(),
        Err(e) => err(format!("ERR {e}")),
    }
}

fn setnx(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'setnx' command");
    }
    let key = scope(session, &args[0]);
    if engine.get_state(&key).is_some() {
        return Value::Integer(0);
    }
    match engine.put_state(key, StoredVal::raw(args[1].clone(), None), None, None) {
        Ok(_) => Value::Integer(1),
        Err(_) => Value::Integer(0),
    }
}

fn strlen(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'strlen' command");
    }
    Value::Integer(
        read_bytes(engine, &scope(session, &args[0]))
            .map(|b| b.len() as i64)
            .unwrap_or(0),
    )
}

fn incr_by(engine: &Engine, session: &Session, args: &[Vec<u8>], delta: i64) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments");
    }
    apply_delta(engine, &scope(session, &args[0]), delta)
}

fn incr_by_arg(engine: &Engine, session: &Session, args: &[Vec<u8>], sign: i64) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments");
    }
    let Ok(n) = String::from_utf8_lossy(&args[1]).parse::<i64>() else {
        return err("ERR value is not an integer or out of range");
    };
    apply_delta(engine, &scope(session, &args[0]), n.saturating_mul(sign))
}

fn apply_delta(engine: &Engine, key: &str, delta: i64) -> Value {
    let current = match read_bytes(engine, key) {
        Some(bytes) => match std::str::from_utf8(&bytes)
            .ok()
            .and_then(|t| t.trim().parse::<i64>().ok())
        {
            Some(n) => n,
            // Redis is explicit here rather than resetting to zero: silently
            // discarding a non-numeric value would destroy data.
            None => return err("ERR value is not an integer or out of range"),
        },
        None => 0,
    };
    let Some(next) = current.checked_add(delta) else {
        return err("ERR increment or decrement would overflow");
    };
    match engine.put_state(
        key.to_string(),
        StoredVal::raw(next.to_string().into_bytes(), None),
        None,
        None,
    ) {
        Ok(_) => Value::Integer(next),
        Err(e) => err(format!("ERR {e}")),
    }
}

fn mget(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() {
        return err("ERR wrong number of arguments for 'mget' command");
    }
    Value::Array(Some(
        args.iter()
            .map(|key| match read_bytes(engine, &scope(session, key)) {
                Some(bytes) => Value::Bulk(Some(bytes)),
                // A missing key is a nil *within* the array, not a shorter
                // array: the reply is positional and clients zip it with their
                // key list.
                None => Value::nil(),
            })
            .collect(),
    ))
}

fn mset(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() || !args.len().is_multiple_of(2) {
        return err("ERR wrong number of arguments for 'mset' command");
    }
    for pair in args.chunks_exact(2) {
        if let Err(e) = engine.put_state(
            scope(session, &pair[0]),
            StoredVal::raw(pair[1].clone(), None),
            None,
            None,
        ) {
            return err(format!("ERR {e}"));
        }
    }
    Value::ok()
}

// ── key commands ─────────────────────────────────────────────────────────────

fn del(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() {
        return err("ERR wrong number of arguments for 'del' command");
    }
    let removed = args
        .iter()
        .filter(|key| engine.delete_state(&scope(session, key)).unwrap_or(false))
        .count();
    Value::Integer(removed as i64)
}

fn exists(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() {
        return err("ERR wrong number of arguments for 'exists' command");
    }
    // Counts occurrences, not distinct keys: `EXISTS k k` is 2 in Redis.
    let count = args
        .iter()
        .filter(|key| engine.get_state(&scope(session, key)).is_some())
        .count();
    Value::Integer(count as i64)
}

fn type_of(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'type' command");
    }
    match engine.get_state(&scope(session, &args[0])) {
        Some(_) => Value::Simple("string".into()),
        // A missing key is `none`, a simple string — not an error and not a nil.
        None => Value::Simple("none".into()),
    }
}

fn ttl(engine: &Engine, session: &Session, args: &[Vec<u8>], divisor: u64) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'ttl' command");
    }
    match engine.get_state(&scope(session, &args[0])) {
        // The two negative sentinels are part of the contract: -2 means the key
        // does not exist, -1 means it exists with no expiry. A client cannot
        // tell them apart any other way.
        None => Value::Integer(-2),
        Some(item) => match item.expires_at_ms {
            None => Value::Integer(-1),
            Some(at) => {
                let remaining = at.saturating_sub(now_ms());
                Value::Integer((remaining / divisor) as i64)
            }
        },
    }
}

fn expire(engine: &Engine, session: &Session, args: &[Vec<u8>], multiplier: u64) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'expire' command");
    }
    let Ok(n) = String::from_utf8_lossy(&args[1]).parse::<i64>() else {
        return err("ERR value is not an integer or out of range");
    };
    let key = scope(session, &args[0]);
    let Some(item) = engine.get_state(&key) else {
        return Value::Integer(0);
    };
    if n <= 0 {
        // A non-positive expiry deletes the key immediately, as Redis does.
        let _ = engine.delete_state(&key);
        return Value::Integer(1);
    }
    let ttl_ms = (n as u64).saturating_mul(multiplier);
    match engine.put_state(key, item.value, Some(ttl_ms), None) {
        Ok(_) => Value::Integer(1),
        Err(_) => Value::Integer(0),
    }
}

fn persist(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'persist' command");
    }
    let key = scope(session, &args[0]);
    let Some(item) = engine.get_state(&key) else {
        return Value::Integer(0);
    };
    if item.expires_at_ms.is_none() {
        return Value::Integer(0);
    }
    match engine.put_state(key, item.value, None, None) {
        Ok(_) => Value::Integer(1),
        Err(_) => Value::Integer(0),
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use tokio_util::sync::CancellationToken;

    fn engine() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            data_dir: Some(dir.path().to_str().unwrap().to_string()),
            ..Config::default()
        };
        (Engine::new(config, CancellationToken::new()).unwrap(), dir)
    }

    /// Accepts any credential, binding to no tenant. Auth policy is the server's
    /// business; these tests are about command semantics.
    fn allow_all(_user: &str, _password: &str) -> Option<Option<String>> {
        Some(None)
    }

    fn run(engine: &Engine, session: &mut Session, argv: &[&str]) -> Value {
        let args: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
        match dispatch(engine, session, &args, allow_all) {
            Dispatch::Reply(value) => value,
            Dispatch::Quit => Value::Simple("QUIT".into()),
        }
    }

    fn open() -> (Engine, tempfile::TempDir, Session) {
        let (engine, dir) = engine();
        (engine, dir, Session::new(false))
    }

    // ── connection ───────────────────────────────────────────────────────────

    #[test]
    fn ping_and_echo() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["PING"]), Value::Simple("PONG".into()));
        // With an argument PING replies with a *bulk* string, not a simple one.
        assert_eq!(run(&e, &mut s, &["PING", "hi"]), Value::bulk("hi"));
        assert_eq!(run(&e, &mut s, &["ECHO", "hi"]), Value::bulk("hi"));
    }

    #[test]
    fn command_names_are_case_insensitive() {
        // Clients send lowercase as often as uppercase.
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["ping"]), Value::Simple("PONG".into()));
        assert_eq!(run(&e, &mut s, &["PiNg"]), Value::Simple("PONG".into()));
    }

    #[test]
    fn select_accepts_only_database_zero() {
        // One logical database. Answering OK for any index would silently share
        // one keyspace between databases a client believes are separate.
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["SELECT", "0"]), Value::ok());
        assert!(matches!(
            run(&e, &mut s, &["SELECT", "3"]),
            Value::Error(m) if m.starts_with("ERR DB index")
        ));
    }

    #[test]
    fn client_setinfo_is_accepted() {
        // redis-py sends this on every connect to report its version. Erroring
        // makes it log a warning on each connection for no reason.
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["CLIENT", "SETINFO", "LIB-NAME", "redis-py"]),
            Value::ok()
        );
        assert_eq!(
            run(&e, &mut s, &["CLIENT", "SETNAME", "worker-1"]),
            Value::ok()
        );
        assert_eq!(
            run(&e, &mut s, &["CLIENT", "GETNAME"]),
            Value::bulk("worker-1")
        );
    }

    #[test]
    fn hello_reports_resp2_even_when_three_is_requested() {
        // Claiming RESP3 and then sending RESP2 replies breaks clients in ways
        // that only surface under load, so the handshake stays honest.
        let (e, _d, mut s) = open();
        let Value::Map(pairs) = run(&e, &mut s, &["HELLO", "3"]) else {
            panic!("HELLO must reply with a map");
        };
        let proto = pairs
            .iter()
            .find(|(k, _)| *k == Value::bulk("proto"))
            .map(|(_, v)| v.clone());
        assert_eq!(proto, Some(Value::Integer(2)));
    }

    #[test]
    fn hello_rejects_an_unknown_protocol_version() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["HELLO", "9"]),
            Value::Error(m) if m.starts_with("NOPROTO")
        ));
    }

    // ── auth ─────────────────────────────────────────────────────────────────

    #[test]
    fn an_unauthenticated_connection_can_only_handshake() {
        let (e, _d) = engine();
        let mut s = Session::new(true);

        // NOAUTH, not "unknown command": the client uses the code to know it
        // must authenticate rather than that the command is unsupported.
        assert!(matches!(
            run(&e, &mut s, &["GET", "k"]),
            Value::Error(m) if m.starts_with("NOAUTH")
        ));
        assert_eq!(run(&e, &mut s, &["PING"]), Value::Simple("PONG".into()));

        assert_eq!(run(&e, &mut s, &["AUTH", "secret"]), Value::ok());
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::nil());
    }

    #[test]
    fn a_rejected_auth_leaves_the_connection_locked() {
        let (e, _d) = engine();
        let mut s = Session::new(true);
        let args = vec![b"AUTH".to_vec(), b"wrong".to_vec()];
        let reply = match dispatch(&e, &mut s, &args, |_, _| None) {
            Dispatch::Reply(v) => v,
            Dispatch::Quit => panic!(),
        };
        assert!(matches!(reply, Value::Error(m) if m.starts_with("WRONGPASS")));
        assert!(!s.authenticated);
    }

    #[test]
    fn tenants_cannot_see_each_other() {
        // The isolation that makes multi-tenancy real: two orgs using the key
        // `celery` must not collide.
        let (e, _d) = engine();
        let mut acme = Session::new(false);
        acme.tenant = Some("acme".into());
        let mut globex = Session::new(false);
        globex.tenant = Some("globex".into());

        run(&e, &mut acme, &["SET", "celery", "acme-value"]);
        run(&e, &mut globex, &["SET", "celery", "globex-value"]);

        assert_eq!(
            run(&e, &mut acme, &["GET", "celery"]),
            Value::bulk("acme-value")
        );
        assert_eq!(
            run(&e, &mut globex, &["GET", "celery"]),
            Value::bulk("globex-value")
        );
        // And a delete in one org leaves the other untouched.
        run(&e, &mut acme, &["DEL", "celery"]);
        assert_eq!(run(&e, &mut acme, &["GET", "celery"]), Value::nil());
        assert_eq!(
            run(&e, &mut globex, &["GET", "celery"]),
            Value::bulk("globex-value")
        );
    }

    // ── strings ──────────────────────────────────────────────────────────────

    #[test]
    fn set_and_get_roundtrip_including_binary() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["SET", "k", "v"]), Value::ok());
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("v"));

        // Binary: a pickled Celery body is not valid UTF-8 and must survive.
        let args = vec![
            b"SET".to_vec(),
            b"bin".to_vec(),
            vec![0x80, 0x04, 0x00, 0xFF],
        ];
        let Dispatch::Reply(reply) = dispatch(&e, &mut s, &args, allow_all) else {
            panic!()
        };
        assert_eq!(reply, Value::ok());
        assert_eq!(
            run(&e, &mut s, &["GET", "bin"]),
            Value::Bulk(Some(vec![0x80, 0x04, 0x00, 0xFF]))
        );
    }

    #[test]
    fn a_missing_key_is_nil_not_an_empty_string() {
        // The distinction clients branch on: nil means absent, an empty bulk
        // string means present and empty.
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["GET", "nope"]), Value::nil());
        run(&e, &mut s, &["SET", "empty", ""]);
        assert_eq!(run(&e, &mut s, &["GET", "empty"]), Value::bulk(""));
    }

    #[test]
    fn set_nx_and_xx_gate_on_existence() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["SET", "k", "1", "NX"]), Value::ok());
        // NX on an existing key replies nil, not an error: it is a conditional,
        // and a lock built on SET NX depends on telling those apart.
        assert_eq!(run(&e, &mut s, &["SET", "k", "2", "NX"]), Value::nil());
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("1"));

        assert_eq!(run(&e, &mut s, &["SET", "k", "3", "XX"]), Value::ok());
        assert_eq!(run(&e, &mut s, &["SET", "absent", "x", "XX"]), Value::nil());
    }

    #[test]
    fn set_with_nx_and_xx_together_is_a_syntax_error() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["SET", "k", "v", "NX", "XX"]),
            Value::Error(m) if m.starts_with("ERR syntax")
        ));
    }

    #[test]
    fn set_ex_sets_a_ttl_and_ttl_reports_it() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v", "EX", "100"]);
        let Value::Integer(remaining) = run(&e, &mut s, &["TTL", "k"]) else {
            panic!("TTL must be an integer");
        };
        assert!(
            (95..=100).contains(&remaining),
            "TTL should be about 100s, got {remaining}"
        );
    }

    #[test]
    fn ttl_sentinels_distinguish_absent_from_immortal() {
        // -2 and -1 are the only way a client can tell "no such key" from
        // "exists, never expires".
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["TTL", "nope"]), Value::Integer(-2));
        run(&e, &mut s, &["SET", "k", "v"]);
        assert_eq!(run(&e, &mut s, &["TTL", "k"]), Value::Integer(-1));
    }

    #[test]
    fn incr_starts_at_zero_and_refuses_non_numbers() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["INCR", "n"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["INCRBY", "n", "9"]), Value::Integer(10));
        assert_eq!(run(&e, &mut s, &["DECR", "n"]), Value::Integer(9));
        assert_eq!(run(&e, &mut s, &["DECRBY", "n", "9"]), Value::Integer(0));

        run(&e, &mut s, &["SET", "word", "hello"]);
        // An error, not a reset to zero: silently discarding the value would
        // destroy data.
        assert!(matches!(
            run(&e, &mut s, &["INCR", "word"]),
            Value::Error(m) if m.contains("not an integer")
        ));
        assert_eq!(run(&e, &mut s, &["GET", "word"]), Value::bulk("hello"));
    }

    #[test]
    fn incr_overflow_is_an_error_not_a_wrap() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "n", &i64::MAX.to_string()]);
        assert!(matches!(
            run(&e, &mut s, &["INCR", "n"]),
            Value::Error(m) if m.contains("overflow")
        ));
    }

    #[test]
    fn mget_returns_a_positional_array_with_nils() {
        // A missing key is a nil *inside* the array, never a shorter array:
        // clients zip the reply against the key list they sent.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MSET", "a", "1", "c", "3"]);
        assert_eq!(
            run(&e, &mut s, &["MGET", "a", "b", "c"]),
            Value::Array(Some(vec![Value::bulk("1"), Value::nil(), Value::bulk("3")]))
        );
    }

    #[test]
    fn mset_with_an_odd_number_of_arguments_is_an_error() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["MSET", "a", "1", "b"]),
            Value::Error(m) if m.contains("wrong number of arguments")
        ));
    }

    #[test]
    fn getdel_returns_the_value_and_removes_the_key() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v"]);
        assert_eq!(run(&e, &mut s, &["GETDEL", "k"]), Value::bulk("v"));
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::nil());
        assert_eq!(run(&e, &mut s, &["GETDEL", "k"]), Value::nil());
    }

    #[test]
    fn setnx_reports_whether_it_wrote() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["SETNX", "k", "1"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["SETNX", "k", "2"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("1"));
    }

    // ── keys ─────────────────────────────────────────────────────────────────

    #[test]
    fn del_counts_keys_actually_removed() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MSET", "a", "1", "b", "2"]);
        assert_eq!(
            run(&e, &mut s, &["DEL", "a", "b", "missing"]),
            Value::Integer(2)
        );
    }

    #[test]
    fn exists_counts_occurrences_not_distinct_keys() {
        // `EXISTS k k` is 2 in Redis. Deduplicating would be a silent behaviour
        // difference in a command clients use for counting.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v"]);
        assert_eq!(run(&e, &mut s, &["EXISTS", "k", "k"]), Value::Integer(2));
        assert_eq!(run(&e, &mut s, &["EXISTS", "nope"]), Value::Integer(0));
    }

    #[test]
    fn type_of_a_missing_key_is_none() {
        // A simple string `none`, not an error and not a nil.
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["TYPE", "nope"]),
            Value::Simple("none".into())
        );
        run(&e, &mut s, &["SET", "k", "v"]);
        assert_eq!(
            run(&e, &mut s, &["TYPE", "k"]),
            Value::Simple("string".into())
        );
    }

    #[test]
    fn expire_with_a_non_positive_ttl_deletes_the_key() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v"]);
        assert_eq!(run(&e, &mut s, &["EXPIRE", "k", "0"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::nil());
    }

    #[test]
    fn expire_on_a_missing_key_reports_zero() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["EXPIRE", "nope", "10"]),
            Value::Integer(0)
        );
    }

    #[test]
    fn persist_clears_a_ttl_and_reports_whether_it_did() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v", "EX", "100"]);
        assert_eq!(run(&e, &mut s, &["PERSIST", "k"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["TTL", "k"]), Value::Integer(-1));
        // Already persistent: nothing to do, reported as 0.
        assert_eq!(run(&e, &mut s, &["PERSIST", "k"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["PERSIST", "nope"]), Value::Integer(0));
    }

    #[test]
    fn expire_preserves_the_value() {
        // EXPIRE rewrites the entry to attach a TTL; losing the value while
        // doing so would be a spectacular way to fail.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "payload"]);
        run(&e, &mut s, &["EXPIRE", "k", "100"]);
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("payload"));
    }

    // ── cross-surface ────────────────────────────────────────────────────────

    #[test]
    fn a_value_written_over_http_is_readable_over_resp() {
        // The whole premise: both surfaces address one store. A JSON string
        // reads back as its contents, not as a quoted literal.
        let (e, _d, mut s) = open();
        e.put_state(
            "shared".to_string(),
            serde_json::json!("from-http"),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            run(&e, &mut s, &["GET", "shared"]),
            Value::bulk("from-http")
        );

        e.put_state("num".to_string(), serde_json::json!(42), None, None)
            .unwrap();
        assert_eq!(run(&e, &mut s, &["GET", "num"]), Value::bulk("42"));
    }

    #[test]
    fn an_unknown_command_reports_its_name() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["FLUXCAPACITOR"]),
            Value::Error(m) if m.contains("unknown command 'FLUXCAPACITOR'")
        ));
    }

    #[test]
    fn quit_asks_the_connection_to_close() {
        let (e, _d) = engine();
        let mut s = Session::new(false);
        let args = vec![b"QUIT".to_vec()];
        assert!(matches!(
            dispatch(&e, &mut s, &args, allow_all),
            Dispatch::Quit
        ));
    }
}
