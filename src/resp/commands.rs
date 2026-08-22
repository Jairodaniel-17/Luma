//! RESP command dispatch: strings, keys, and the connection commands.
//!
//! F1.2 and F1.3 of `docs/PLAN-MAESTRO.md`. Commands execute against the
//! existing key-value store, so TTL, compare-and-swap, WAL durability and crash
//! replay are the same machinery the HTTP API uses — a `SET` through RESP and a
//! `PUT /v1/state` land in the same place with the same guarantees.
//!
//! ## Keyspace isolation
//!
//! Every key is stored as `{tenant}<US>{key}` when the connection is bound to an
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
    /// Commands queued by `MULTI`, or `None` outside a transaction.
    ///
    /// `Some(vec![])` is a started-but-empty transaction, which is different
    /// from not being in one at all — `EXEC` there returns an empty array
    /// rather than an error.
    pub queued: Option<Vec<Vec<Vec<u8>>>>,
    /// Set when a queued command failed to parse. `EXEC` must then abort with
    /// `EXECABORT` instead of running a transaction the client already knows
    /// is broken.
    pub queue_error: bool,
    /// Keys under `WATCH`, with the revision they had when watched. `EXEC`
    /// compares against these; any change aborts the transaction.
    pub watched: Vec<(String, u64)>,
    /// Row id of the api key this connection authenticated with.
    ///
    /// Kept so revocation can be re-checked later without holding the secret
    /// itself in memory for the connection's lifetime. `None` when the
    /// connection used the static instance password, which has no row and
    /// cannot be revoked without a restart.
    pub key_id: Option<String>,
}

/// What a successful `AUTH` established.
#[derive(Debug, Clone, Default)]
pub struct AuthBinding {
    /// Org the connection is bound to. `None` is a platform-wide connection
    /// that sees the unprefixed keyspace.
    pub tenant: Option<String>,
    /// Api key row id, or `None` for the static instance password.
    pub key_id: Option<String>,
}

impl Session {
    pub fn new(requires_auth: bool) -> Self {
        Self {
            tenant: None,
            authenticated: !requires_auth,
            name: None,
            queued: None,
            queue_error: false,
            watched: Vec::new(),
            key_id: None,
        }
    }

    /// Drop everything the credential granted, leaving the connection open.
    ///
    /// Used when a key is revoked mid-connection: the socket stays up and the
    /// next command gets `NOAUTH`, which is what tells a client to
    /// re-authenticate instead of treating it as a network fault. Clearing the
    /// tenant matters as much as the flag — a stale tenant on a connection that
    /// somehow reached a keyed command would read another org's data.
    pub fn deauthenticate(&mut self) {
        self.authenticated = false;
        self.tenant = None;
        self.key_id = None;
        self.queued = None;
        self.queue_error = false;
        self.watched.clear();
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

/// Separator between a tenant and the key it owns.
///
/// ASCII unit separator rather than `:`, which is what Redis users put *inside*
/// their own keys (`user:1000:sessions`). With a colon, tenant `a` holding key
/// `b:c` and tenant `a:b` holding key `c` collapse to the same physical key —
/// one org reading another's data. A control character cannot appear in an org
/// id, so the split is unambiguous. Pub/Sub channels already use this byte.
pub const TENANT_SEP: char = '\u{1f}';

/// Prefix a key with the connection's tenant, mirroring the HTTP layer.
pub fn scope_key(tenant: Option<&str>, key: &str) -> String {
    match tenant {
        Some(t) => format!("{t}{TENANT_SEP}{key}"),
        None => key.to_string(),
    }
}

fn scope(session: &Session, key: &[u8]) -> String {
    scope_key(session.tenant.as_deref(), &String::from_utf8_lossy(key))
}

/// What a key holds, looking in **both** places a value can live.
///
/// ## Why this exists
///
/// Structures are stored in the same KV store under a `struct:` prefix, so
/// physically one name can hold a string *and* a list at once. Redis has one
/// keyspace with one type per key, and clients lean on that hard: `SET lock 1`
/// followed by `LPUSH lock x` has to fail. Without this, both succeeded and the
/// name carried two unrelated values that no command could reconcile — `TYPE`
/// said `none` for every structure, `EXISTS` said 0, `SETNX` succeeded on a key
/// that was plainly taken, and `KEYS *` handed the client raw `struct:` names.
///
/// Found by the differential suite against a real Redis 7. It could not have
/// been found by our own tests, which encoded the same wrong model as the code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Held {
    Nothing,
    /// A plain value in the string keyspace.
    Str,
    /// A structure, carrying the type name `TYPE` reports.
    Structure(&'static str),
}

impl Held {
    fn exists(self) -> bool {
        self != Held::Nothing
    }
}

/// Resolve what an already-scoped key holds.
///
/// The structure side is checked first: a `SET` over a structure deletes it, so
/// the two can only coexist through a bug, and reporting the structure makes
/// that bug visible rather than silently preferring the string.
pub fn held(engine: &Engine, scoped: &str) -> Held {
    let structure_key = crate::engine::structures::structure_key(scoped);
    if engine.get_state(&structure_key).is_some() {
        let name = crate::engine::structures::Structures::new(engine)
            .load(scoped)
            .ok()
            .flatten()
            .map(|(structure, _)| structure.type_name())
            .unwrap_or("string");
        return Held::Structure(name);
    }
    if engine.get_state(scoped).is_some() {
        return Held::Str;
    }
    Held::Nothing
}

/// String commands that read or extend an existing value, and therefore must
/// refuse a key holding a structure.
///
/// `SET`/`SETEX`/`PSETEX`/`MSET` are deliberately absent: Redis lets them
/// replace any type. `MGET` is absent too — it answers nil per wrongly-typed
/// key rather than failing the whole command. `SETNX` is absent because it
/// reports 0 for a taken key of any type, not an error.
/// Each with the exact argument count it takes, so the guard can stay quiet on
/// a malformed command and let the command report the arity error itself.
/// Redis validates shape before type, and telling a client "wrong type" when it
/// actually forgot an argument sends it looking in the wrong place.
const STRING_ONLY: &[(&str, usize)] = &[
    ("GET", 1),
    ("GETDEL", 1),
    ("GETSET", 2),
    ("STRLEN", 1),
    ("APPEND", 2),
    ("INCR", 1),
    ("DECR", 1),
    ("INCRBY", 2),
    ("DECRBY", 2),
    ("INCRBYFLOAT", 2),
];

/// Refuse a command aimed at a key whose type it cannot operate on.
fn keyspace_guard(
    engine: &Engine,
    session: &Session,
    name: &str,
    rest: &[Vec<u8>],
) -> Option<Value> {
    let key = rest.first()?;
    let scoped = scope(session, key);
    let wrong_type = || err("WRONGTYPE Operation against a key holding the wrong kind of value");

    if let Some((_, arity)) = STRING_ONLY.iter().find(|(c, _)| *c == name) {
        if rest.len() != *arity {
            return None;
        }
        return matches!(held(engine, &scoped), Held::Structure(_)).then(wrong_type);
    }
    // A move has a second key, and a string sitting at the destination is a type
    // error too. Checking only the source would let the pop happen and then fail
    // the push, which is the one path where an element can be lost outside a
    // crash. The source itself is caught by `Structures::load`, after the
    // command has checked its own arity.
    if matches!(name, "RPOPLPUSH" | "LMOVE" | "BRPOPLPUSH" | "BLMOVE") {
        let expected = match name {
            "RPOPLPUSH" => 2,
            "BRPOPLPUSH" => 3,
            "LMOVE" => 4,
            _ => 5,
        };
        if rest.len() == expected {
            if let Some(destination) = rest.get(1) {
                if matches!(held(engine, &scope(session, destination)), Held::Str) {
                    return Some(wrong_type());
                }
            }
        }
    }
    None
}

/// Drop a structure a string command is about to overwrite.
///
/// Only for the commands Redis defines as type-agnostic replacements. Leaving
/// the structure behind would resurrect it the moment the string was deleted.
fn clear_replaced_structures(engine: &Engine, session: &Session, name: &str, rest: &[Vec<u8>]) {
    match name {
        "SET" | "SETEX" | "PSETEX" | "GETSET" => {
            if let Some(key) = rest.first() {
                clear_structure(engine, &scope(session, key));
            }
        }
        "MSET" => {
            for pair in rest.chunks_exact(2) {
                clear_structure(engine, &scope(session, &pair[0]));
            }
        }
        _ => {}
    }
}

/// Delete whatever else lives under a key before writing a string to it.
///
/// `SET` is type-agnostic in Redis: it replaces a list with a string without
/// complaint. Leaving the structure behind would resurrect it the moment the
/// string was deleted.
fn clear_structure(engine: &Engine, scoped: &str) {
    if matches!(held(engine, scoped), Held::Structure(_)) {
        crate::engine::structures::Structures::new(engine).delete(scoped);
    }
}

/// Remove the tenant prefix so a reply names the key the client actually sent.
///
/// A client that asked for `jobs` must be told `jobs`; handing back the
/// internal `acme<US>jobs` both leaks the layout and breaks clients that match
/// the returned key against the one they requested — kombu and arq both do
/// exactly that after a blocking pop.
pub fn unscope_key(tenant: Option<&str>, key: &str) -> String {
    match tenant {
        Some(t) => key
            .strip_prefix(t)
            .and_then(|rest| rest.strip_prefix(TENANT_SEP))
            .unwrap_or(key)
            .to_string(),
        None => key.to_string(),
    }
}

/// Outcome of dispatching one command.
#[derive(Debug)]
pub enum Dispatch {
    Reply(Value),
    /// The client asked to close the connection.
    Quit,
    /// A blocking read. Dispatch stays synchronous and hands the wait back to
    /// the connection loop, which is the only place that can await without
    /// holding a lock on the store.
    Block {
        /// Keys to watch, in the order the client gave them — a `BLPOP` that
        /// wakes must pop from the *first* ready key in argument order.
        keys: Vec<String>,
        /// `None` means wait forever, which is how Redis spells `timeout 0`.
        timeout: Option<std::time::Duration>,
        /// What to do once a key has something.
        kind: BlockKind,
    },
    /// A Pub/Sub command. Handled by the connection loop, which owns the
    /// subscriber inbox and the socket it has to be pushed to.
    PubSub(PubSubCommand),
}

/// What a blocking command does when one of its keys becomes non-empty.
///
/// Kept as data rather than a closure because the wait happens in the
/// connection loop, which is the only place that can await — the command layer
/// decides *what* to do and hands it over.
#[derive(Debug, Clone)]
pub enum BlockKind {
    /// `BLPOP` / `BRPOP`. Replies `[key, element]`.
    Pop { left: bool },
    /// `BLMOVE` / `BRPOPLPUSH`. Pops from the watched key and pushes to
    /// `destination`; replies with the element alone, since the client already
    /// knows both keys.
    Move {
        destination: String,
        from_left: bool,
        to_left: bool,
    },
    /// `BZPOPMIN` / `BZPOPMAX`. Replies `[key, member, score]` — three
    /// elements, not two, which is the trap for a client written against
    /// `BLPOP`.
    ZPop { min: bool },
}

/// A Pub/Sub request, parsed but not yet executed.
#[derive(Debug)]
pub enum PubSubCommand {
    Subscribe(Vec<Vec<u8>>),
    PSubscribe(Vec<Vec<u8>>),
    /// `None` means every channel, which is what a bare UNSUBSCRIBE does.
    Unsubscribe(Option<Vec<Vec<u8>>>),
    PUnsubscribe(Option<Vec<Vec<u8>>>),
    Publish {
        channel: Vec<u8>,
        payload: Vec<u8>,
    },
    Channels(Option<Vec<u8>>),
    NumSub(Vec<Vec<u8>>),
}

/// Execute one command.
pub fn dispatch(
    engine: &Engine,
    session: &mut Session,
    args: &[Vec<u8>],
    // A trait object rather than `impl Fn`: `exec` calls back into
    // dispatch, and a generic parameter would monomorphize `&&&&...` forever.
    authenticate: &dyn Fn(&str, &str) -> Option<AuthBinding>,
    allow_flush: bool,
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

    // Inside MULTI everything except the transaction controls is queued rather
    // than run, and the client is told so with +QUEUED.
    if session.queued.is_some() && !TRANSACTION_CONTROL.contains(&name.as_str()) {
        if is_unknown_command(&name) {
            // Redis reports the error now *and* refuses the EXEC later: a
            // client that mistyped a command must not have the rest of its
            // transaction applied.
            session.queue_error = true;
            return Dispatch::Reply(err(format!(
                "ERR unknown command '{name}', with args beginning with: "
            )));
        }
        if let Some(queue) = session.queued.as_mut() {
            queue.push(args.to_vec());
        }
        return Dispatch::Reply(Value::Simple("QUEUED".into()));
    }

    // One keyspace, one type per key. Enforced here rather than inside each
    // command so the policy is auditable in one place — it previously lived
    // nowhere, and every string command silently succeeded on a key holding a
    // list.
    if let Some(refusal) = keyspace_guard(engine, session, &name, rest) {
        return Dispatch::Reply(refusal);
    }
    // `SET` and friends replace whatever a key held, including a structure.
    clear_replaced_structures(engine, session, &name, rest);

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
        "INFO" => Dispatch::Reply(info(engine, session, rest)),
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

        // ── transactions ─────────────────────────────────────────────────────
        "MULTI" => Dispatch::Reply(if session.queued.is_some() {
            // Redis refuses to nest; silently accepting would make the inner
            // EXEC run only part of what the client thinks it queued.
            err("ERR MULTI calls can not be nested")
        } else {
            session.queued = Some(Vec::new());
            session.queue_error = false;
            Value::ok()
        }),
        "EXEC" => Dispatch::Reply(exec(engine, session, authenticate, allow_flush)),
        "DISCARD" => Dispatch::Reply(discard(session)),
        "WATCH" => Dispatch::Reply(watch(engine, session, rest)),
        "UNWATCH" => {
            session.watched.clear();
            Dispatch::Reply(Value::ok())
        }

        // ── blocking reads ───────────────────────────────────────────────────
        "BLPOP" | "BRPOP" => blocking(
            session,
            rest,
            &name,
            BlockKind::Pop {
                left: name == "BLPOP",
            },
        ),
        "BZPOPMIN" | "BZPOPMAX" => blocking(
            session,
            rest,
            &name,
            BlockKind::ZPop {
                min: name == "BZPOPMIN",
            },
        ),
        "BRPOPLPUSH" => {
            // `BRPOPLPUSH source destination timeout`: exactly one source, so
            // the shared parser cannot be reused for the key list.
            if rest.len() != 3 {
                return Dispatch::Reply(err(
                    "ERR wrong number of arguments for 'brpoplpush' command",
                ));
            }
            blocking(
                session,
                &[rest[0].clone(), rest[2].clone()],
                &name,
                BlockKind::Move {
                    destination: scope(session, &rest[1]),
                    from_left: false,
                    to_left: true,
                },
            )
        }
        "BLMOVE" => {
            // `BLMOVE source destination LEFT|RIGHT LEFT|RIGHT timeout`.
            if rest.len() != 5 {
                return Dispatch::Reply(err("ERR wrong number of arguments for 'blmove' command"));
            }
            let (Some(from_left), Some(to_left)) = (parse_side(&rest[2]), parse_side(&rest[3]))
            else {
                return Dispatch::Reply(err("ERR syntax error"));
            };
            blocking(
                session,
                &[rest[0].clone(), rest[4].clone()],
                &name,
                BlockKind::Move {
                    destination: scope(session, &rest[1]),
                    from_left,
                    to_left,
                },
            )
        }

        // ── pub/sub ──────────────────────────────────────────────────────────
        "SUBSCRIBE" | "PSUBSCRIBE" => {
            if rest.is_empty() {
                return Dispatch::Reply(err(format!(
                    "ERR wrong number of arguments for '{}' command",
                    name.to_lowercase()
                )));
            }
            Dispatch::PubSub(if name == "SUBSCRIBE" {
                PubSubCommand::Subscribe(rest.to_vec())
            } else {
                PubSubCommand::PSubscribe(rest.to_vec())
            })
        }
        "UNSUBSCRIBE" | "PUNSUBSCRIBE" => {
            let targets = (!rest.is_empty()).then(|| rest.to_vec());
            Dispatch::PubSub(if name == "UNSUBSCRIBE" {
                PubSubCommand::Unsubscribe(targets)
            } else {
                PubSubCommand::PUnsubscribe(targets)
            })
        }
        "PUBLISH" => {
            if rest.len() != 2 {
                return Dispatch::Reply(err("ERR wrong number of arguments for 'publish' command"));
            }
            Dispatch::PubSub(PubSubCommand::Publish {
                channel: rest[0].clone(),
                payload: rest[1].clone(),
            })
        }
        "PUBSUB" => match rest.first().map(|a| command_name(a)).as_deref() {
            Some("CHANNELS") => Dispatch::PubSub(PubSubCommand::Channels(rest.get(1).cloned())),
            Some("NUMSUB") => Dispatch::PubSub(PubSubCommand::NumSub(rest[1..].to_vec())),
            // NUMPAT is answered as 0 rather than refused: a dashboard that
            // polls it should not see an error it cannot act on.
            Some("NUMPAT") => Dispatch::Reply(Value::Integer(0)),
            _ => Dispatch::Reply(err("ERR Unknown PUBSUB subcommand")),
        },
        "EXPIREAT" => Dispatch::Reply(expire_at(engine, session, rest, 1000)),
        "PEXPIREAT" => Dispatch::Reply(expire_at(engine, session, rest, 1)),
        "RENAME" => Dispatch::Reply(rename(engine, session, rest, false)),
        "RENAMENX" => Dispatch::Reply(rename(engine, session, rest, true)),
        "APPEND" => Dispatch::Reply(append(engine, session, rest)),
        "GETSET" => Dispatch::Reply(getset(engine, session, rest)),
        "INCRBYFLOAT" => Dispatch::Reply(incrbyfloat(engine, session, rest)),
        "KEYS" => Dispatch::Reply(keys(engine, session, rest)),
        "SCAN" => Dispatch::Reply(scan(engine, session, rest)),
        "DBSIZE" => Dispatch::Reply(dbsize(engine, session)),
        "RANDOMKEY" => Dispatch::Reply(randomkey(engine, session)),
        "FLUSHDB" | "FLUSHALL" => Dispatch::Reply(flushdb(engine, session, allow_flush)),

        other => {
            // Structure commands live in their own module; they are tried
            // before reporting an unknown command so the two sets stay
            // independent without a second dispatch table to keep in sync.
            match crate::resp::structures_cmd::dispatch(engine, session, other, rest) {
                Some(reply) => Dispatch::Reply(reply),
                None => Dispatch::Reply(err(format!(
                    "ERR unknown command '{other}', with args beginning with: "
                ))),
            }
        }
    }
}

// ── connection commands ──────────────────────────────────────────────────────

/// Split `AUTH`'s arguments into username and password.
///
/// Redis 6 added the two-argument form; the one-argument form is still what
/// most clients send when only a password is configured. Exposed because the
/// connection loop has to read the credential before dispatch — resolving it
/// against the api-key store needs an `await`, and dispatch is synchronous.
pub fn auth_credential(args: &[Vec<u8>]) -> Option<(String, String)> {
    match args.len() {
        1 => Some((
            "default".to_string(),
            String::from_utf8_lossy(&args[0]).to_string(),
        )),
        2 => Some((
            String::from_utf8_lossy(&args[0]).to_string(),
            String::from_utf8_lossy(&args[1]).to_string(),
        )),
        _ => None,
    }
}

fn auth(
    session: &mut Session,
    args: &[Vec<u8>],
    authenticate: &dyn Fn(&str, &str) -> Option<AuthBinding>,
) -> Value {
    let Some((user, password)) = auth_credential(args) else {
        return err("ERR wrong number of arguments for 'auth' command");
    };

    match authenticate(&user, &password) {
        Some(binding) => {
            session.authenticated = true;
            session.tenant = binding.tenant;
            session.key_id = binding.key_id;
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

/// `INFO [section]`.
///
/// kombu reads this to decide how to talk to the broker, and monitoring
/// dashboards scrape it, so an error here looks like a broken server rather
/// than a missing feature. The numbers are real — reporting plausible constants
/// would be worse than not implementing it, because an operator would trust
/// them.
///
/// Lines end with CRLF, as Redis does: some INFO parsers split on `\r\n`
/// specifically, and bare LF leaves them with a trailing `\r` on every value.
fn info(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    let requested = args.first().map(|a| command_name(a));
    let want = |section: &str| {
        requested.is_none()
            || requested.as_deref() == Some("ALL")
            || requested.as_deref() == Some("DEFAULT")
            || requested.as_deref() == Some(&section.to_ascii_uppercase())
    };

    let mut out = String::new();
    if want("server") {
        out.push_str("# Server\r\n");
        out.push_str("redis_version:7.0.0\r\n");
        // The real identity, right after the compatibility version: a client
        // that checks `redis_version` keeps working, and a human reading INFO
        // is not misled about what they are talking to.
        out.push_str("server_name:luma\r\n");
        out.push_str(&format!("luma_version:{}\r\n", env!("CARGO_PKG_VERSION")));
        out.push_str("redis_mode:standalone\r\n");
        out.push_str(&format!("process_id:{}\r\n", std::process::id()));
        out.push_str("\r\n");
    }
    if want("clients") {
        out.push_str("# Clients\r\n");
        out.push_str("connected_clients:1\r\n\r\n");
    }
    if want("stats") {
        out.push_str("# Stats\r\n");
        out.push_str("total_connections_received:0\r\n");
        out.push_str("total_commands_processed:0\r\n\r\n");
    }
    if want("keyspace") {
        out.push_str("# Keyspace\r\n");
        let keys = visible_keys(engine, session, MAX_KEYSPACE_WALK).len();
        if keys > 0 {
            out.push_str(&format!("db0:keys={keys},expires=0,avg_ttl=0\r\n"));
        }
        out.push_str("\r\n");
    }
    // A bulk string, not a simple one: the body contains CRLF, which only a
    // length-prefixed reply can carry.
    Value::bulk(out)
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
    // Taken is taken, whatever type took it.
    if held(engine, &key).exists() {
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
            .map(|key| {
                let scoped = scope(session, key);
                // Redis answers nil for a wrongly-typed key here rather than
                // failing the whole command, so one bad key does not cost the
                // caller the other ninety-nine.
                if matches!(held(engine, &scoped), Held::Structure(_)) {
                    return Value::nil();
                }
                match read_bytes(engine, &scoped) {
                    Some(bytes) => Value::Bulk(Some(bytes)),
                    // A missing key is a nil *within* the array, not a shorter
                    // array: the reply is positional and clients zip it with their
                    // key list.
                    None => Value::nil(),
                }
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
    // Both slots: a `DEL` that left the structure behind reported 0 and the
    // key stayed alive, so the client could not remove its own data.
    let removed = args
        .iter()
        .filter(|key| {
            let scoped = scope(session, key);
            let structure = crate::engine::structures::Structures::new(engine).delete(&scoped);
            let plain = engine.delete_state(&scoped).unwrap_or(false);
            structure || plain
        })
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
        .filter(|key| held(engine, &scope(session, key)).exists())
        .count();
    Value::Integer(count as i64)
}

fn type_of(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'type' command");
    }
    match held(engine, &scope(session, &args[0])) {
        Held::Str => Value::Simple("string".into()),
        Held::Structure(name) => Value::Simple(name.into()),
        // A missing key is `none`, a simple string — not an error and not a nil.
        Held::Nothing => Value::Simple("none".into()),
    }
}

fn ttl(engine: &Engine, session: &Session, args: &[Vec<u8>], divisor: u64) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'ttl' command");
    }
    match engine.get_state(&physical_key(engine, &scope(session, &args[0]))) {
        // The two negative sentinels are part of the contract: -2 means the key
        // does not exist, -1 means it exists with no expiry. A client cannot
        // tell them apart any other way.
        None => Value::Integer(-2),
        Some(item) => match item.expires_at_ms {
            None => Value::Integer(-1),
            Some(at) => {
                let remaining = at.saturating_sub(now_ms());
                // Round up. Redis answers 100 right after `SETEX k 100 v`;
                // truncating gives 99, and a client asserting on the value it
                // just set sees an off-by-one it cannot explain.
                Value::Integer(remaining.div_ceil(divisor) as i64)
            }
        },
    }
}

/// Where a key's value physically lives: the structure slot when it holds a
/// structure, the plain slot otherwise.
///
/// Expiry, renaming and deletion all act on the stored item, so they need the
/// physical name; everything the client sees stays the logical one.
fn physical_key(engine: &Engine, scoped: &str) -> String {
    match held(engine, scoped) {
        Held::Structure(_) => crate::engine::structures::structure_key(scoped),
        _ => scoped.to_string(),
    }
}

fn expire(engine: &Engine, session: &Session, args: &[Vec<u8>], multiplier: u64) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'expire' command");
    }
    let Ok(n) = String::from_utf8_lossy(&args[1]).parse::<i64>() else {
        return err("ERR value is not an integer or out of range");
    };
    let key = physical_key(engine, &scope(session, &args[0]));
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
    let key = physical_key(engine, &scope(session, &args[0]));
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

// ── keyspace scanning ────────────────────────────────────────────────────────

/// Glob matcher for `KEYS` and `SCAN MATCH`, supporting `*`, `?` and `[...]`.
///
/// Written out rather than pulled from a regex crate because Redis globs are
/// their own small language — `[a-c]`, `[^a]`, `\*` — and translating them into
/// a regex correctly is more code than matching them directly, with more ways to
/// be subtly wrong on a pattern a client sends.
pub fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    // Position to resume from if the current `*` turns out to have matched too
    // little — the standard backtracking pair, which keeps this linear in
    // practice without recursion.
    let (mut star, mut resume) = (None, 0usize);

    while t < text.len() {
        match pattern.get(p) {
            Some(b'*') => {
                star = Some(p);
                resume = t;
                p += 1;
            }
            Some(b'?') => {
                p += 1;
                t += 1;
            }
            Some(b'[') => match match_class(pattern, p, text[t]) {
                Some((matched, next)) if matched => {
                    p = next;
                    t += 1;
                }
                Some((_, next)) => match star {
                    Some(s) => {
                        p = s + 1;
                        resume += 1;
                        t = resume;
                    }
                    None => {
                        let _ = next;
                        return false;
                    }
                },
                // An unterminated class matches literally, as Redis does.
                None => {
                    if pattern[p] == text[t] {
                        p += 1;
                        t += 1;
                    } else {
                        return false;
                    }
                }
            },
            Some(b'\\') if p + 1 < pattern.len() => {
                if pattern[p + 1] == text[t] {
                    p += 2;
                    t += 1;
                } else if let Some(s) = star {
                    p = s + 1;
                    resume += 1;
                    t = resume;
                } else {
                    return false;
                }
            }
            Some(&c) if c == text[t] => {
                p += 1;
                t += 1;
            }
            _ => match star {
                Some(s) => {
                    p = s + 1;
                    resume += 1;
                    t = resume;
                }
                None => return false,
            },
        }
    }
    // Trailing stars can absorb nothing.
    while pattern.get(p) == Some(&b'*') {
        p += 1;
    }
    p == pattern.len()
}

/// Match one `[...]` class against `c`. Returns `(matched, index after the
/// class)`, or `None` when the class is unterminated.
fn match_class(pattern: &[u8], start: usize, c: u8) -> Option<(bool, usize)> {
    let mut i = start + 1;
    let negated = pattern.get(i) == Some(&b'^');
    if negated {
        i += 1;
    }
    let mut matched = false;
    let mut first = true;
    while i < pattern.len() {
        if pattern[i] == b']' && !first {
            return Some((matched != negated, i + 1));
        }
        first = false;
        // A range like `a-c`, but only when the `-` is not the last character.
        if i + 2 < pattern.len() && pattern[i + 1] == b'-' && pattern[i + 2] != b']' {
            if pattern[i] <= c && c <= pattern[i + 2] {
                matched = true;
            }
            i += 3;
        } else {
            if pattern[i] == c {
                matched = true;
            }
            i += 1;
        }
    }
    None
}

/// Keys visible to this session, already unscoped, capped at `limit`.
fn visible_keys(engine: &Engine, session: &Session, limit: usize) -> Vec<String> {
    let tenant = session.tenant.as_ref().map(|t| format!("{t}{TENANT_SEP}"));
    let structure_prefix = crate::engine::structures::structure_key("");

    // Plain keys, minus the structure slots, which are storage detail and must
    // never be handed to a client — a platform-level `KEYS *` used to return
    // `struct:myhash`.
    let mut names: Vec<String> = engine
        .list_state(tenant.as_deref(), limit)
        .into_iter()
        .filter_map(|item| {
            if item.key.starts_with(&structure_prefix) {
                return None;
            }
            match &tenant {
                // Strip the tenant prefix so a caller only ever sees its own
                // keys, exactly as the HTTP layer does.
                Some(p) => item.key.strip_prefix(p.as_str()).map(|k| k.to_string()),
                None => Some(item.key),
            }
        })
        .collect();

    // Structures, listed under their own prefix and reported by their logical
    // name. One keyspace to the client, two slots underneath.
    let structure_scan = match &tenant {
        Some(t) => format!("{structure_prefix}{t}"),
        None => structure_prefix.clone(),
    };
    names.extend(
        engine
            .list_state(Some(&structure_scan), limit)
            .into_iter()
            .filter_map(|item| {
                let logical = item.key.strip_prefix(structure_prefix.as_str())?;
                match &tenant {
                    Some(p) => logical.strip_prefix(p.as_str()).map(|k| k.to_string()),
                    None => Some(logical.to_string()),
                }
            }),
    );
    names.truncate(limit);
    names
}

/// Upper bound on keys a single `KEYS`/`SCAN`/`DBSIZE` will walk.
///
/// `KEYS *` on a large keyspace is the classic way to stall Redis; bounding it
/// means a careless client degrades its own result rather than the server.
const MAX_KEYSPACE_WALK: usize = 100_000;

fn keys(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return err("ERR wrong number of arguments for 'keys' command");
    }
    let matched = visible_keys(engine, session, MAX_KEYSPACE_WALK)
        .into_iter()
        .filter(|key| glob_match(&args[0], key.as_bytes()))
        .map(Value::bulk)
        .collect();
    Value::Array(Some(matched))
}

/// `SCAN cursor [MATCH pattern] [COUNT n]`.
///
/// The cursor is an index into the sorted key list rather than Redis's hash
/// bucket. That keeps the guarantee clients actually rely on — a full iteration
/// returns every key present throughout it — while being far simpler than
/// emulating the bucket layout of a hash table we do not have.
fn scan(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    let Some(cursor_raw) = args.first() else {
        return err("ERR wrong number of arguments for 'scan' command");
    };
    let Ok(cursor) = String::from_utf8_lossy(cursor_raw).parse::<usize>() else {
        return err("ERR invalid cursor");
    };

    let mut pattern: Option<Vec<u8>> = None;
    let mut count = 10usize;
    let mut i = 1;
    while i < args.len() {
        match command_name(&args[i]).as_str() {
            "MATCH" => {
                let Some(p) = args.get(i + 1) else {
                    return err("ERR syntax error");
                };
                pattern = Some(p.clone());
                i += 2;
            }
            "COUNT" => {
                let Some(n) = args.get(i + 1) else {
                    return err("ERR syntax error");
                };
                let Ok(n) = String::from_utf8_lossy(n).parse::<usize>() else {
                    return err("ERR value is not an integer or out of range");
                };
                count = n.clamp(1, 10_000);
                i += 2;
            }
            // TYPE is accepted and ignored: everything here is a string, and
            // erroring would break a client that passes it defensively.
            "TYPE" => i += 2,
            _ => return err("ERR syntax error"),
        }
    }

    let mut all = visible_keys(engine, session, MAX_KEYSPACE_WALK);
    all.sort();

    let end = cursor.saturating_add(count).min(all.len());
    let page: Vec<Value> = all[cursor.min(all.len())..end]
        .iter()
        .filter(|key| match &pattern {
            Some(p) => glob_match(p, key.as_bytes()),
            None => true,
        })
        .map(|key| Value::bulk(key.clone()))
        .collect();

    // Cursor 0 means the iteration is finished — a client loops until it sees
    // it, so returning a non-zero cursor at the end would loop forever.
    let next = if end >= all.len() { 0 } else { end };
    Value::Array(Some(vec![
        Value::bulk(next.to_string()),
        Value::Array(Some(page)),
    ]))
}

fn dbsize(engine: &Engine, session: &Session) -> Value {
    Value::Integer(visible_keys(engine, session, MAX_KEYSPACE_WALK).len() as i64)
}

fn randomkey(engine: &Engine, session: &Session) -> Value {
    let keys = visible_keys(engine, session, 1);
    match keys.into_iter().next() {
        Some(key) => Value::bulk(key),
        None => Value::nil(),
    }
}

fn flushdb(engine: &Engine, session: &Session, allowed: bool) -> Value {
    // Gated behind config: an accidental FLUSHDB from a misconfigured client is
    // unrecoverable without a restore, so it is off unless explicitly enabled.
    if !allowed {
        return err("ERR FLUSHDB is disabled on this server (set resp_allow_flush = true)");
    }
    for key in visible_keys(engine, session, MAX_KEYSPACE_WALK) {
        let scoped = scope(session, key.as_bytes());
        clear_structure(engine, &scoped);
        let _ = engine.delete_state(&scoped);
    }
    Value::ok()
}

fn rename(engine: &Engine, session: &Session, args: &[Vec<u8>], only_if_absent: bool) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'rename' command");
    }
    let logical_from = scope(session, &args[0]);
    let logical_to = scope(session, &args[1]);
    // A structure moves as a structure: renaming only the plain slot would
    // leave the data behind under the old name and create nothing under the new
    // one, which is what `RENAME` on a list used to do.
    let from = physical_key(engine, &logical_from);
    let to = match held(engine, &logical_from) {
        Held::Structure(_) => crate::engine::structures::structure_key(&logical_to),
        _ => logical_to.clone(),
    };
    let Some(item) = engine.get_state(&from) else {
        // Redis distinguishes these: RENAME errors on a missing source, RENAMENX
        // reports 0. A client uses the difference to tell "gone" from "taken".
        return if only_if_absent {
            Value::Integer(0)
        } else {
            err("ERR no such key")
        };
    };
    if only_if_absent && held(engine, &logical_to).exists() {
        return Value::Integer(0);
    }
    // The destination is overwritten whatever type it held.
    if !only_if_absent {
        clear_structure(engine, &logical_to);
        let _ = engine.delete_state(&logical_to);
    }
    // The TTL travels with the value, as Redis specifies.
    let ttl_ms = item
        .expires_at_ms
        .map(|at| at.saturating_sub(now_ms()))
        .filter(|remaining| *remaining > 0);
    match engine.put_state(to, item.value, ttl_ms, None) {
        Ok(_) => {
            let _ = engine.delete_state(&from);
            if only_if_absent {
                Value::Integer(1)
            } else {
                Value::ok()
            }
        }
        Err(e) => err(format!("ERR {e}")),
    }
}

fn append(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'append' command");
    }
    let key = scope(session, &args[0]);
    let mut current = read_bytes(engine, &key).unwrap_or_default();
    current.extend_from_slice(&args[1]);
    let len = current.len() as i64;
    match engine.put_state(key, StoredVal::raw(current, None), None, None) {
        Ok(_) => Value::Integer(len),
        Err(e) => err(format!("ERR {e}")),
    }
}

fn getset(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'getset' command");
    }
    let key = scope(session, &args[0]);
    let previous = read_bytes(engine, &key);
    match engine.put_state(key, StoredVal::raw(args[1].clone(), None), None, None) {
        Ok(_) => match previous {
            Some(bytes) => Value::Bulk(Some(bytes)),
            None => Value::nil(),
        },
        Err(e) => err(format!("ERR {e}")),
    }
}

fn incrbyfloat(engine: &Engine, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'incrbyfloat' command");
    }
    let Ok(delta) = String::from_utf8_lossy(&args[1]).parse::<f64>() else {
        return err("ERR value is not a valid float");
    };
    if !delta.is_finite() {
        return err("ERR value is not a valid float");
    }
    let key = scope(session, &args[0]);
    let current = match read_bytes(engine, &key) {
        Some(bytes) => match std::str::from_utf8(&bytes)
            .ok()
            .and_then(|t| t.trim().parse::<f64>().ok())
        {
            Some(n) => n,
            None => return err("ERR value is not a valid float"),
        },
        None => 0.0,
    };
    let next = current + delta;
    if !next.is_finite() {
        return err("ERR increment would produce NaN or Infinity");
    }
    // Redis trims trailing zeros so `1.0 + 0.5` reads back as `1.5`, not
    // `1.5000000000000000`.
    let rendered = format_float(next);
    match engine.put_state(
        key,
        StoredVal::raw(rendered.clone().into_bytes(), None),
        None,
        None,
    ) {
        Ok(_) => Value::bulk(rendered),
        Err(e) => err(format!("ERR {e}")),
    }
}

fn format_float(value: f64) -> String {
    let mut text = format!("{value:.17}");
    if text.contains('.') {
        text = text.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    text
}

/// `EXPIREAT`/`PEXPIREAT`: an absolute deadline rather than a duration.
fn expire_at(engine: &Engine, session: &Session, args: &[Vec<u8>], multiplier: u64) -> Value {
    if args.len() != 2 {
        return err("ERR wrong number of arguments for 'expireat' command");
    }
    let Ok(at) = String::from_utf8_lossy(&args[1]).parse::<i64>() else {
        return err("ERR value is not an integer or out of range");
    };
    let key = scope(session, &args[0]);
    let Some(item) = engine.get_state(&key) else {
        return Value::Integer(0);
    };
    let deadline_ms = (at.max(0) as u64).saturating_mul(multiplier);
    let now = now_ms();
    if deadline_ms <= now {
        // A deadline in the past deletes immediately, as Redis does.
        let _ = engine.delete_state(&key);
        return Value::Integer(1);
    }
    match engine.put_state(key, item.value, Some(deadline_ms - now), None) {
        Ok(_) => Value::Integer(1),
        Err(_) => Value::Integer(0),
    }
}

// ── transactions ─────────────────────────────────────────────────────────────

/// Commands that act on the transaction itself and are therefore executed
/// immediately rather than queued.
const TRANSACTION_CONTROL: &[&str] = &["MULTI", "EXEC", "DISCARD", "WATCH", "UNWATCH", "QUIT"];

/// The revision a key currently has, or 0 when it does not exist.
///
/// Absent is deliberately a *value* rather than a special case: `WATCH`ing a
/// key that does not exist yet and having it created before `EXEC` must abort
/// the transaction, which only works if "absent" and "revision 1" compare
/// unequal.
fn revision_of(engine: &Engine, key: &str) -> u64 {
    engine.get_state(key).map(|item| item.revision).unwrap_or(0)
}

fn watch(engine: &Engine, session: &mut Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() {
        return err("ERR wrong number of arguments for 'watch' command");
    }
    // Redis refuses this: watching after MULTI could not affect the outcome,
    // and silently accepting it would give a false sense of protection.
    if session.queued.is_some() {
        return err("ERR WATCH inside MULTI is not allowed");
    }
    for key in args {
        let scoped = scope(session, key);
        let revision = revision_of(engine, &scoped);
        session.watched.push((scoped, revision));
    }
    Value::ok()
}

fn exec(
    engine: &Engine,
    session: &mut Session,
    authenticate: &dyn Fn(&str, &str) -> Option<AuthBinding>,
    allow_flush: bool,
) -> Value {
    let Some(queued) = session.queued.take() else {
        return err("ERR EXEC without MULTI");
    };
    let watched = std::mem::take(&mut session.watched);

    if session.queue_error {
        session.queue_error = false;
        // The client already saw the queueing error, so running the rest would
        // execute a transaction it knows is incomplete.
        return err("EXECABORT Transaction discarded because of previous errors.");
    }

    // Optimistic concurrency: if anything we watched moved, the whole
    // transaction is abandoned and the client retries. A *null array* is the
    // signal — not an error and not an empty array, both of which a client
    // would read as "it ran".
    for (key, revision) in &watched {
        if revision_of(engine, key) != *revision {
            return Value::Array(None);
        }
    }

    let mut replies = Vec::with_capacity(queued.len());
    for args in queued {
        match dispatch(engine, session, &args, authenticate, allow_flush) {
            Dispatch::Reply(value) => replies.push(value),
            // QUIT is in TRANSACTION_CONTROL so it never reaches the queue;
            // treating it as a reply keeps this total without a panic.
            Dispatch::Quit => replies.push(Value::ok()),
            Dispatch::Block { .. } => {
                // A blocking command inside MULTI must not block: Redis runs it
                // with a zero timeout, so it behaves as its non-blocking twin.
                replies.push(Value::Array(None));
            }
            // Pub/Sub needs the connection's inbox, which EXEC does not have.
            // Redis likewise refuses subscribe commands inside a transaction.
            Dispatch::PubSub(_) => replies.push(Value::Error(
                "ERR SUBSCRIBE is not allowed in transactions".into(),
            )),
        }
    }
    Value::Array(Some(replies))
}

fn discard(session: &mut Session) -> Value {
    if session.queued.take().is_none() {
        return err("ERR DISCARD without MULTI");
    }
    session.queue_error = false;
    session.watched.clear();
    Value::ok()
}

/// Parse `BLPOP key [key ...] timeout` into a wait request.
///
/// The timeout is seconds and may be fractional, which is how Redis spells
/// sub-second waits. `0` means wait forever.
/// `LEFT`/`RIGHT` for `LMOVE` and `BLMOVE`.
fn parse_side(raw: &[u8]) -> Option<bool> {
    if raw.eq_ignore_ascii_case(b"LEFT") {
        Some(true)
    } else if raw.eq_ignore_ascii_case(b"RIGHT") {
        Some(false)
    } else {
        None
    }
}

/// Parse `<key...> <timeout>` and hand the wait to the connection loop.
///
/// Every blocking command shares this tail, including the ones whose keys are
/// not a variadic list — those pass the single source plus the timeout.
fn blocking(session: &Session, args: &[Vec<u8>], name: &str, kind: BlockKind) -> Dispatch {
    if args.len() < 2 {
        return Dispatch::Reply(err(format!(
            "ERR wrong number of arguments for '{}' command",
            name.to_lowercase()
        )));
    }
    let (keys, timeout_raw) = args.split_at(args.len() - 1);
    let Some(seconds) = std::str::from_utf8(&timeout_raw[0])
        .ok()
        .and_then(|t| t.trim().parse::<f64>().ok())
    else {
        return Dispatch::Reply(err("ERR timeout is not a float or out of range"));
    };
    if seconds < 0.0 || !seconds.is_finite() {
        return Dispatch::Reply(err("ERR timeout is negative"));
    }
    Dispatch::Block {
        keys: keys.iter().map(|k| scope(session, k)).collect(),
        // Redis spells "wait forever" as timeout 0.
        timeout: (seconds > 0.0).then(|| std::time::Duration::from_secs_f64(seconds)),
        kind,
    }
}

/// Whether a name is one this server does not implement.
///
/// Used only to reject inside MULTI. Kept as one list so a command added to the
/// dispatcher but forgotten here would merely be queued rather than silently
/// mis-reported — failing towards accepting a real command rather than
/// rejecting it.
fn is_unknown_command(name: &str) -> bool {
    const KNOWN: &[&str] = &[
        "PING",
        "ECHO",
        "AUTH",
        "HELLO",
        "SELECT",
        "CLIENT",
        "COMMAND",
        "RESET",
        "SET",
        "GET",
        "GETDEL",
        "SETEX",
        "PSETEX",
        "SETNX",
        "STRLEN",
        "INCR",
        "DECR",
        "INCRBY",
        "DECRBY",
        "MGET",
        "MSET",
        "DEL",
        "UNLINK",
        "EXISTS",
        "TYPE",
        "TTL",
        "PTTL",
        "EXPIRE",
        "PEXPIRE",
        "PERSIST",
        "EXPIREAT",
        "PEXPIREAT",
        "RENAME",
        "RENAMENX",
        "APPEND",
        "GETSET",
        "INCRBYFLOAT",
        "KEYS",
        "SCAN",
        "DBSIZE",
        "RANDOMKEY",
        "FLUSHDB",
        "FLUSHALL",
        "BLPOP",
        "BRPOP",
        "LPUSH",
        "RPUSH",
        "LPOP",
        "RPOP",
        "LLEN",
        "LRANGE",
        "LREM",
        "HSET",
        "HGET",
        "HMGET",
        "HDEL",
        "HGETALL",
        "HLEN",
        "HEXISTS",
        "HKEYS",
        "HVALS",
        "HINCRBY",
        "SADD",
        "SREM",
        "SMEMBERS",
        "SISMEMBER",
        "SCARD",
        "ZADD",
        "ZREM",
        "ZSCORE",
        "ZCARD",
        "ZRANGE",
        "ZRANGEBYSCORE",
        "ZRANK",
        "ZREVRANGE",
        "LPUSHX",
        "RPUSHX",
        "LINDEX",
        "LSET",
        "LTRIM",
        "RPOPLPUSH",
        "LMOVE",
        "HSETNX",
        "HSCAN",
        "SPOP",
        "SRANDMEMBER",
        "SSCAN",
        "ZREVRANK",
        "ZMSCORE",
        "ZCOUNT",
        "ZINCRBY",
        "ZREVRANGEBYSCORE",
        "ZREMRANGEBYSCORE",
        "ZREMRANGEBYRANK",
        "ZPOPMIN",
        "ZPOPMAX",
        "ZSCAN",
        "BLMOVE",
        "BRPOPLPUSH",
        "BZPOPMIN",
        "BZPOPMAX",
    ];
    !KNOWN.contains(&name)
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
    fn allow_all(_user: &str, _password: &str) -> Option<AuthBinding> {
        Some(AuthBinding::default())
    }

    fn run(engine: &Engine, session: &mut Session, argv: &[&str]) -> Value {
        let args: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
        match dispatch(engine, session, &args, &allow_all, true) {
            Dispatch::Reply(value) => value,
            Dispatch::Quit => Value::Simple("QUIT".into()),
            // The test helper is synchronous; a blocking command is reported
            // rather than silently turned into something else.
            Dispatch::Block { .. } => Value::Simple("BLOCK".into()),
            Dispatch::PubSub(_) => Value::Simple("PUBSUB".into()),
        }
    }

    /// Run a command on a fresh session — the "another client" in the WATCH
    /// tests, and the way to read state without disturbing a transaction in
    /// progress on the session under test.
    fn run_outside(engine: &Engine, argv: &[&str]) -> Value {
        let mut other = Session::new(false);
        run(engine, &mut other, argv)
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
        let reply = match dispatch(&e, &mut s, &args, &|_, _| None, true) {
            Dispatch::Reply(v) => v,
            Dispatch::Quit => panic!(),
            Dispatch::Block { .. } => panic!("unexpected block"),
            Dispatch::PubSub(_) => panic!("unexpected pubsub"),
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
        let Dispatch::Reply(reply) = dispatch(&e, &mut s, &args, &allow_all, true) else {
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
            dispatch(&e, &mut s, &args, &allow_all, true),
            Dispatch::Quit
        ));
    }

    // ── glob matching ────────────────────────────────────────────────────────

    #[test]
    fn glob_handles_the_redis_pattern_language() {
        let cases: &[(&str, &str, bool)] = &[
            ("*", "anything", true),
            ("*", "", true),
            ("h?llo", "hello", true),
            ("h?llo", "hllo", false),
            ("h*llo", "heeeello", true),
            ("h[ae]llo", "hallo", true),
            ("h[ae]llo", "hillo", false),
            ("h[^e]llo", "hallo", true),
            ("h[^e]llo", "hello", false),
            ("h[a-c]llo", "hbllo", true),
            ("h[a-c]llo", "hdllo", false),
            ("celery*", "celery-queue", true),
            ("celery*", "arq-queue", false),
            ("*-queue", "celery-queue", true),
            ("a*b*c", "axxbyyc", true),
            ("a*b*c", "axxbyy", false),
            // A literal star, escaped.
            ("a\\*b", "a*b", true),
            ("a\\*b", "axb", false),
            // Exact matches with no wildcards at all.
            ("plain", "plain", true),
            ("plain", "plainer", false),
        ];
        for (pattern, text, expected) in cases {
            assert_eq!(
                glob_match(pattern.as_bytes(), text.as_bytes()),
                *expected,
                "pattern `{pattern}` against `{text}`"
            );
        }
    }

    #[test]
    fn glob_backtracks_rather_than_giving_up_on_the_first_star() {
        // The classic failure: `*abc` against `aabc` needs the star to give
        // back a character. A greedy non-backtracking matcher reports no match.
        assert!(glob_match(b"*abc", b"aabc"));
        assert!(glob_match(b"*a*b", b"zzazzb"));
    }

    // ── keyspace ─────────────────────────────────────────────────────────────

    #[test]
    fn keys_filters_by_pattern_and_stays_inside_the_tenant() {
        let (e, _d) = engine();
        let mut acme = Session::new(false);
        acme.tenant = Some("acme".into());
        let mut globex = Session::new(false);
        globex.tenant = Some("globex".into());

        run(
            &e,
            &mut acme,
            &["MSET", "job:1", "a", "job:2", "b", "other", "c"],
        );
        run(&e, &mut globex, &["SET", "job:9", "z"]);

        let Value::Array(Some(found)) = run(&e, &mut acme, &["KEYS", "job:*"]) else {
            panic!("KEYS must return an array");
        };
        let mut names: Vec<String> = found
            .iter()
            .map(|v| match v {
                Value::Bulk(Some(b)) => String::from_utf8_lossy(b).to_string(),
                _ => panic!("keys must be bulk strings"),
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec!["job:1".to_string(), "job:2".to_string()],
            "another tenant's job:9 must be invisible, and the keys must come \
             back unprefixed"
        );
    }

    #[test]
    fn scan_walks_the_whole_keyspace_and_terminates() {
        // The property a client depends on: loop until the cursor is 0 and you
        // have seen every key. A cursor that never reaches 0 loops forever.
        let (e, _d, mut s) = open();
        for i in 0..25 {
            run(&e, &mut s, &["SET", &format!("k{i:02}"), "v"]);
        }

        let mut cursor = "0".to_string();
        let mut seen = Vec::new();
        let mut rounds = 0;
        loop {
            rounds += 1;
            assert!(rounds < 100, "SCAN did not terminate");
            let Value::Array(Some(reply)) = run(&e, &mut s, &["SCAN", &cursor, "COUNT", "7"])
            else {
                panic!("SCAN must return an array");
            };
            let Value::Bulk(Some(next)) = &reply[0] else {
                panic!("cursor must be a bulk string");
            };
            let Value::Array(Some(page)) = &reply[1] else {
                panic!("page must be an array");
            };
            for item in page {
                if let Value::Bulk(Some(b)) = item {
                    seen.push(String::from_utf8_lossy(b).to_string());
                }
            }
            cursor = String::from_utf8_lossy(next).to_string();
            if cursor == "0" {
                break;
            }
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 25, "SCAN must eventually return every key");
    }

    #[test]
    fn scan_honours_match() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MSET", "a:1", "x", "a:2", "x", "b:1", "x"]);
        let Value::Array(Some(reply)) =
            run(&e, &mut s, &["SCAN", "0", "MATCH", "a:*", "COUNT", "100"])
        else {
            panic!()
        };
        let Value::Array(Some(page)) = &reply[1] else {
            panic!()
        };
        assert_eq!(page.len(), 2);
    }

    #[test]
    fn scan_rejects_a_bad_cursor_and_bad_syntax() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["SCAN", "notanumber"]),
            Value::Error(m) if m.contains("invalid cursor")
        ));
        assert!(matches!(
            run(&e, &mut s, &["SCAN", "0", "NONSENSE"]),
            Value::Error(m) if m.contains("syntax")
        ));
    }

    #[test]
    fn dbsize_and_randomkey_are_tenant_scoped() {
        let (e, _d) = engine();
        let mut acme = Session::new(false);
        acme.tenant = Some("acme".into());
        let mut globex = Session::new(false);
        globex.tenant = Some("globex".into());

        run(&e, &mut acme, &["MSET", "a", "1", "b", "2"]);
        run(&e, &mut globex, &["SET", "c", "3"]);

        assert_eq!(run(&e, &mut acme, &["DBSIZE"]), Value::Integer(2));
        assert_eq!(run(&e, &mut globex, &["DBSIZE"]), Value::Integer(1));

        let Value::Bulk(Some(key)) = run(&e, &mut globex, &["RANDOMKEY"]) else {
            panic!("RANDOMKEY must return a key")
        };
        assert_eq!(String::from_utf8_lossy(&key), "c");
    }

    #[test]
    fn randomkey_on_an_empty_keyspace_is_nil() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["RANDOMKEY"]), Value::nil());
    }

    #[test]
    fn flushdb_is_refused_unless_enabled() {
        // An accidental flush from a misconfigured client is unrecoverable
        // without a restore, so it is off unless explicitly turned on.
        let (e, _d) = engine();
        let mut s = Session::new(false);
        run(&e, &mut s, &["SET", "k", "v"]);

        let args = vec![b"FLUSHDB".to_vec()];
        let Dispatch::Reply(reply) = dispatch(&e, &mut s, &args, &allow_all, false) else {
            panic!()
        };
        assert!(matches!(reply, Value::Error(m) if m.contains("disabled")));
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("v"));

        let Dispatch::Reply(reply) = dispatch(&e, &mut s, &args, &allow_all, true) else {
            panic!()
        };
        assert_eq!(reply, Value::ok());
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::nil());
    }

    #[test]
    fn flushdb_only_clears_the_calling_tenant() {
        let (e, _d) = engine();
        let mut acme = Session::new(false);
        acme.tenant = Some("acme".into());
        let mut globex = Session::new(false);
        globex.tenant = Some("globex".into());
        run(&e, &mut acme, &["SET", "k", "acme"]);
        run(&e, &mut globex, &["SET", "k", "globex"]);

        let args = vec![b"FLUSHDB".to_vec()];
        let _ = dispatch(&e, &mut acme, &args, &allow_all, true);

        assert_eq!(run(&e, &mut acme, &["GET", "k"]), Value::nil());
        assert_eq!(
            run(&e, &mut globex, &["GET", "k"]),
            Value::bulk("globex"),
            "one org's FLUSHDB must never touch another's keyspace"
        );
    }

    // ── rename, append, getset, float ────────────────────────────────────────

    #[test]
    fn rename_moves_the_value_and_its_ttl() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "old", "payload", "EX", "100"]);
        assert_eq!(run(&e, &mut s, &["RENAME", "old", "new"]), Value::ok());
        assert_eq!(run(&e, &mut s, &["GET", "old"]), Value::nil());
        assert_eq!(run(&e, &mut s, &["GET", "new"]), Value::bulk("payload"));

        let Value::Integer(remaining) = run(&e, &mut s, &["TTL", "new"]) else {
            panic!()
        };
        assert!(
            remaining > 0,
            "the TTL must travel with the value, got {remaining}"
        );
    }

    #[test]
    fn rename_and_renamenx_differ_on_a_missing_source() {
        // The difference is how a client tells "gone" from "taken".
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["RENAME", "nope", "x"]),
            Value::Error(m) if m.contains("no such key")
        ));
        assert_eq!(
            run(&e, &mut s, &["RENAMENX", "nope", "x"]),
            Value::Integer(0)
        );

        run(&e, &mut s, &["MSET", "a", "1", "b", "2"]);
        assert_eq!(run(&e, &mut s, &["RENAMENX", "a", "b"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["GET", "a"]), Value::bulk("1"));
    }

    #[test]
    fn append_creates_then_extends_and_returns_the_length() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["APPEND", "k", "abc"]), Value::Integer(3));
        assert_eq!(run(&e, &mut s, &["APPEND", "k", "de"]), Value::Integer(5));
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("abcde"));
    }

    #[test]
    fn getset_returns_the_previous_value() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["GETSET", "k", "first"]), Value::nil());
        assert_eq!(
            run(&e, &mut s, &["GETSET", "k", "second"]),
            Value::bulk("first")
        );
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::bulk("second"));
    }

    #[test]
    fn incrbyfloat_trims_trailing_zeros() {
        // Redis renders `1.5`, not `1.50000000000000000`. A client that parses
        // the reply and re-sends it would otherwise drift.
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["INCRBYFLOAT", "f", "1.0"]),
            Value::bulk("1")
        );
        assert_eq!(
            run(&e, &mut s, &["INCRBYFLOAT", "f", "0.5"]),
            Value::bulk("1.5")
        );
        assert_eq!(
            run(&e, &mut s, &["INCRBYFLOAT", "f", "-1.5"]),
            Value::bulk("0")
        );
    }

    #[test]
    fn incrbyfloat_refuses_non_numbers_and_infinities() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["INCRBYFLOAT", "f", "abc"]),
            Value::Error(m) if m.contains("not a valid float")
        ));
        assert!(matches!(
            run(&e, &mut s, &["INCRBYFLOAT", "f", "inf"]),
            Value::Error(m) if m.contains("not a valid float")
        ));
        run(&e, &mut s, &["SET", "word", "hello"]);
        assert!(matches!(
            run(&e, &mut s, &["INCRBYFLOAT", "word", "1"]),
            Value::Error(m) if m.contains("not a valid float")
        ));
    }

    #[test]
    fn expireat_in_the_past_deletes_immediately() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v"]);
        assert_eq!(run(&e, &mut s, &["EXPIREAT", "k", "1"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["GET", "k"]), Value::nil());
    }

    #[test]
    fn expireat_in_the_future_sets_a_ttl() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "v"]);
        let deadline = (now_ms() / 1000) + 100;
        assert_eq!(
            run(&e, &mut s, &["EXPIREAT", "k", &deadline.to_string()]),
            Value::Integer(1)
        );
        let Value::Integer(remaining) = run(&e, &mut s, &["TTL", "k"]) else {
            panic!()
        };
        assert!((90..=100).contains(&remaining), "got {remaining}");
    }

    // ── transactions ─────────────────────────────────────────────────────────

    #[test]
    fn multi_queues_then_exec_runs_in_order() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["MULTI"]), Value::ok());
        // Queued commands reply +QUEUED, not their result: a client that saw a
        // real reply here would act on a value that has not been written yet.
        assert_eq!(
            run(&e, &mut s, &["SET", "k", "1"]),
            Value::Simple("QUEUED".into())
        );
        assert_eq!(
            run(&e, &mut s, &["INCR", "k"]),
            Value::Simple("QUEUED".into())
        );
        // Nothing has run yet.
        assert_eq!(run_outside(&e, &["GET", "k"]), Value::nil());

        assert_eq!(
            run(&e, &mut s, &["EXEC"]),
            Value::Array(Some(vec![Value::ok(), Value::Integer(2)]))
        );
        assert_eq!(run_outside(&e, &["GET", "k"]), Value::bulk("2"));
    }

    #[test]
    fn discard_throws_the_queue_away() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "k", "1"]);
        assert_eq!(run(&e, &mut s, &["DISCARD"]), Value::ok());
        assert_eq!(run_outside(&e, &["GET", "k"]), Value::nil());
        // And we are out of the transaction, so EXEC is now an error.
        assert!(matches!(
            run(&e, &mut s, &["EXEC"]),
            Value::Error(m) if m.contains("EXEC without MULTI")
        ));
    }

    #[test]
    fn exec_and_discard_without_multi_are_errors() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["EXEC"]),
            Value::Error(m) if m.contains("EXEC without MULTI")
        ));
        assert!(matches!(
            run(&e, &mut s, &["DISCARD"]),
            Value::Error(m) if m.contains("DISCARD without MULTI")
        ));
    }

    #[test]
    fn multi_cannot_be_nested() {
        // Accepting it silently would make the inner EXEC run only part of what
        // the client thinks it queued.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MULTI"]);
        assert!(matches!(
            run(&e, &mut s, &["MULTI"]),
            Value::Error(m) if m.contains("can not be nested")
        ));
    }

    #[test]
    fn an_empty_transaction_execs_to_an_empty_array() {
        // Started-but-empty is not the same as not being in a transaction.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MULTI"]);
        assert_eq!(run(&e, &mut s, &["EXEC"]), Value::Array(Some(Vec::new())));
    }

    #[test]
    fn a_bad_command_inside_multi_aborts_the_whole_exec() {
        // EXECABORT, not a partial application: a client that mistyped one
        // command must not have the rest of its transaction land.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "k", "1"]);
        assert!(matches!(
            run(&e, &mut s, &["NOSUCHCOMMAND"]),
            Value::Error(m) if m.contains("unknown command")
        ));
        assert!(matches!(
            run(&e, &mut s, &["EXEC"]),
            Value::Error(m) if m.starts_with("EXECABORT")
        ));
        assert_eq!(run_outside(&e, &["GET", "k"]), Value::nil());
    }

    // ── WATCH ────────────────────────────────────────────────────────────────

    #[test]
    fn watch_aborts_exec_when_the_key_changed() {
        // The optimistic-concurrency contract. A *null array* is the abort
        // signal — not an error and not an empty array, both of which a client
        // would read as "it ran".
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "1"]);
        assert_eq!(run(&e, &mut s, &["WATCH", "k"]), Value::ok());

        // Another client writes.
        run_outside(&e, &["SET", "k", "999"]);

        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "k", "2"]);
        assert_eq!(
            run(&e, &mut s, &["EXEC"]),
            Value::Array(None),
            "a watched key that moved must abort the transaction"
        );
        assert_eq!(run_outside(&e, &["GET", "k"]), Value::bulk("999"));
    }

    #[test]
    fn watch_lets_exec_through_when_nothing_changed() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "1"]);
        run(&e, &mut s, &["WATCH", "k"]);
        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "k", "2"]);
        assert_eq!(
            run(&e, &mut s, &["EXEC"]),
            Value::Array(Some(vec![Value::ok()]))
        );
        assert_eq!(run_outside(&e, &["GET", "k"]), Value::bulk("2"));
    }

    #[test]
    fn watching_a_key_that_is_then_created_aborts() {
        // "Absent" and "revision 1" must compare unequal, or a client watching
        // for a key to *not* appear would be silently defeated.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["WATCH", "newkey"]);
        run_outside(&e, &["SET", "newkey", "sneaked-in"]);
        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "other", "1"]);
        assert_eq!(run(&e, &mut s, &["EXEC"]), Value::Array(None));
    }

    #[test]
    fn unwatch_clears_the_guard() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "1"]);
        run(&e, &mut s, &["WATCH", "k"]);
        assert_eq!(run(&e, &mut s, &["UNWATCH"]), Value::ok());
        run_outside(&e, &["SET", "k", "999"]);

        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "k", "2"]);
        assert_eq!(
            run(&e, &mut s, &["EXEC"]),
            Value::Array(Some(vec![Value::ok()])),
            "after UNWATCH the change must no longer abort"
        );
    }

    #[test]
    fn exec_clears_the_watch_list() {
        // Otherwise a stale watch from a completed transaction would abort the
        // next one for no reason.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "k", "1"]);
        run(&e, &mut s, &["WATCH", "k"]);
        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["EXEC"]);

        run(&e, &mut s, &["MULTI"]);
        run(&e, &mut s, &["SET", "other", "1"]);
        assert_eq!(
            run(&e, &mut s, &["EXEC"]),
            Value::Array(Some(vec![Value::ok()]))
        );
    }

    #[test]
    fn watch_inside_multi_is_refused() {
        // It could not affect the outcome, and accepting it would give a false
        // sense of protection.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MULTI"]);
        assert!(matches!(
            run(&e, &mut s, &["WATCH", "k"]),
            Value::Error(m) if m.contains("WATCH inside MULTI")
        ));
    }

    #[test]
    fn a_hundred_concurrent_transactions_sum_exactly() {
        // The property the whole mechanism exists for: with WATCH/MULTI/EXEC and
        // a retry loop, N clients incrementing one key land on exactly N.
        let (e, _d, mut writer) = open();
        run(&e, &mut writer, &["SET", "counter", "0"]);

        for _ in 0..100 {
            loop {
                let mut s = Session::new(false);
                run(&e, &mut s, &["WATCH", "counter"]);
                let current = match run(&e, &mut s, &["GET", "counter"]) {
                    Value::Bulk(Some(b)) => String::from_utf8_lossy(&b).parse::<i64>().unwrap(),
                    other => panic!("unexpected GET reply {other:?}"),
                };
                run(&e, &mut s, &["MULTI"]);
                run(&e, &mut s, &["SET", "counter", &(current + 1).to_string()]);
                if !matches!(run(&e, &mut s, &["EXEC"]), Value::Array(None)) {
                    break;
                }
                // Aborted: another writer got there first, so retry.
            }
        }
        assert_eq!(run_outside(&e, &["GET", "counter"]), Value::bulk("100"));
    }

    // ── blocking ─────────────────────────────────────────────────────────────

    #[test]
    fn blpop_reports_a_block_rather_than_replying() {
        // Dispatch stays synchronous; the wait is handed to the connection loop,
        // which is the only place that can await without holding the store.
        let (e, _d, mut s) = open();
        let args: Vec<Vec<u8>> = ["BLPOP", "a", "b", "0"]
            .iter()
            .map(|a| a.as_bytes().to_vec())
            .collect();
        match dispatch(&e, &mut s, &args, &allow_all, true) {
            Dispatch::Block {
                keys,
                timeout,
                kind,
            } => {
                assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
                assert!(timeout.is_none(), "timeout 0 means wait forever");
                assert!(matches!(kind, BlockKind::Pop { left: true }));
            }
            other => panic!("expected a block, got {other:?}"),
        }
    }

    #[test]
    fn blpop_parses_a_fractional_timeout() {
        let (e, _d, mut s) = open();
        let args: Vec<Vec<u8>> = ["BRPOP", "q", "0.25"]
            .iter()
            .map(|a| a.as_bytes().to_vec())
            .collect();
        match dispatch(&e, &mut s, &args, &allow_all, true) {
            Dispatch::Block { timeout, kind, .. } => {
                assert_eq!(timeout, Some(std::time::Duration::from_millis(250)));
                assert!(
                    matches!(kind, BlockKind::Pop { left: false }),
                    "BRPOP pops from the tail"
                );
            }
            other => panic!("expected a block, got {other:?}"),
        }
    }

    #[test]
    fn blpop_rejects_a_negative_or_unparseable_timeout() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["BLPOP", "q", "-1"]),
            Value::Error(m) if m.contains("negative")
        ));
        assert!(matches!(
            run(&e, &mut s, &["BLPOP", "q", "soon"]),
            Value::Error(m) if m.contains("not a float")
        ));
        assert!(matches!(
            run(&e, &mut s, &["BLPOP", "q"]),
            Value::Error(m) if m.contains("wrong number of arguments")
        ));
    }

    #[test]
    fn info_reports_real_numbers_and_its_own_identity() {
        // kombu reads INFO to decide how to talk to the broker, and dashboards
        // scrape it. Plausible constants would be worse than nothing, because
        // an operator would trust them.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["MSET", "a", "1", "b", "2"]);

        let Value::Bulk(Some(body)) = run(&e, &mut s, &["INFO"]) else {
            panic!("INFO must be a bulk string: the body contains CRLF");
        };
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("redis_version:"), "{text}");
        // CRLF, not bare LF: parsers that split on CRLF would otherwise leave a
        // trailing carriage return on every value.
        assert!(
            text.contains("redis_version:7.0.0\r\n"),
            "INFO lines must end with CRLF"
        );
        // The real identity sits right beside the compatibility version.
        assert!(text.contains("server_name:luma"), "{text}");
        assert!(
            text.contains("db0:keys=2"),
            "the keyspace count must be measured, not invented: {text}"
        );
    }

    #[test]
    fn info_honours_a_section_argument() {
        let (e, _d, mut s) = open();
        let Value::Bulk(Some(body)) = run(&e, &mut s, &["INFO", "server"]) else {
            panic!()
        };
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("# Server"));
        assert!(!text.contains("# Keyspace"), "asked for one section only");
    }

    #[test]
    fn info_keyspace_is_tenant_scoped() {
        // Otherwise one org could read another's key count out of INFO.
        let (e, _d) = engine();
        let mut acme = Session::new(false);
        acme.tenant = Some("acme".into());
        let mut globex = Session::new(false);
        globex.tenant = Some("globex".into());
        run(&e, &mut acme, &["MSET", "a", "1", "b", "2"]);
        run(&e, &mut globex, &["SET", "c", "3"]);

        let Value::Bulk(Some(body)) = run(&e, &mut globex, &["INFO", "keyspace"]) else {
            panic!()
        };
        assert!(
            String::from_utf8_lossy(&body).contains("db0:keys=1"),
            "globex must see only its own key"
        );
    }
}
