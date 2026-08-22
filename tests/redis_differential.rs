//! Differential suite against a real Redis 7.
//!
//! `docs/SPEC-resp.md` names this the source of truth for RESP semantics — not
//! the Redis documentation, and not our own tests. The distinction earned its
//! place: the first run of this suite found nine divergences that the in-tree
//! tests could not have caught, because those tests encoded the same wrong model
//! as the code they were checking. The worst of them was structural — a name
//! could hold a string *and* a list at once, so `TYPE` answered `none` for every
//! structure, `EXISTS` answered 0, `DEL` could not delete, and `KEYS *` handed
//! clients the internal `struct:` prefix.
//!
//! ## Running it
//!
//! Ignored by default because it needs a Redis that this process does not own:
//!
//! ```text
//! docker run -d --name luma-diff-redis -p 16379:6379 redis:7-alpine
//! LUMA_DIFF_REDIS=127.0.0.1:16379 cargo test --test redis_differential -- --ignored
//! ```
//!
//! Without `LUMA_DIFF_REDIS` the test refuses to run rather than passing
//! vacuously: a suite that goes green when its subject is absent is worse than
//! no suite, because it reports coverage it does not have.
//!
//! Luma is started in-process on an ephemeral port, so only Redis is external.
//!
//! ## What "identical" means here
//!
//! Raw bytes on both sides. Going through a client library would let its parser
//! smooth over a malformed reply, which is one of the things this is looking
//! for. Three comparisons are deliberately looser, each for a stated reason:
//! unordered collections, error text beyond the leading token, and the cases
//! `docs/RESP.md` records as accepted divergences.

use luma::config::Config;
use luma::engine::Engine;
use luma::resp::listener::{spawn, RespMetrics};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// How strictly one case is compared.
#[derive(Clone, Copy, PartialEq)]
enum Compare {
    /// Byte-for-byte.
    Exact,
    /// Same elements, any order — `SMEMBERS`, `HGETALL` and friends promise no
    /// ordering, and Redis's own is a hash-table artefact.
    Unordered,
    /// Only the leading error token (`ERR`, `WRONGTYPE`, `WRONGPASS`). Clients
    /// branch on the token; the prose after it is not a contract.
    ErrorToken,
}

struct Case {
    argv: &'static [&'static str],
    compare: Compare,
}

const fn exact(argv: &'static [&'static str]) -> Case {
    Case {
        argv,
        compare: Compare::Exact,
    }
}

const fn unordered(argv: &'static [&'static str]) -> Case {
    Case {
        argv,
        compare: Compare::Unordered,
    }
}

const fn token(argv: &'static [&'static str]) -> Case {
    Case {
        argv,
        compare: Compare::ErrorToken,
    }
}

/// A blocking RESP connection that reads exactly one reply at a time.
struct Peer {
    name: &'static str,
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Peer {
    fn connect(addr: &str, name: &'static str) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_nodelay(true)?;
        Ok(Self {
            name,
            stream,
            buf: Vec::new(),
        })
    }

    fn call(&mut self, argv: &[&str]) -> Vec<u8> {
        let mut frame = format!("*{}\r\n", argv.len()).into_bytes();
        for arg in argv {
            frame.extend_from_slice(format!("${}\r\n{arg}\r\n", arg.len()).as_bytes());
        }
        self.stream.write_all(&frame).expect("write");
        self.read_reply()
    }

    fn read_reply(&mut self) -> Vec<u8> {
        loop {
            if let Some(end) = self.complete(0) {
                let reply = self.buf[..end].to_vec();
                self.buf.drain(..end);
                return reply;
            }
            let mut chunk = [0u8; 16 * 1024];
            let read = self
                .stream
                .read(&mut chunk)
                .unwrap_or_else(|e| panic!("{} read failed: {e}", self.name));
            assert!(read > 0, "{} closed the connection", self.name);
            self.buf.extend_from_slice(&chunk[..read]);
        }
    }

    /// Index just past a complete reply starting at `start`, or `None` when more
    /// bytes are needed. Recursive for arrays, which nest.
    fn complete(&self, start: usize) -> Option<usize> {
        let kind = *self.buf.get(start)?;
        let line_end = self.buf[start..].windows(2).position(|w| w == b"\r\n")? + start;
        let payload = &self.buf[start + 1..line_end];
        let after = line_end + 2;
        match kind {
            b'+' | b'-' | b':' => Some(after),
            b'$' => {
                let n: i64 = std::str::from_utf8(payload).ok()?.parse().ok()?;
                if n < 0 {
                    return Some(after);
                }
                let end = after + n as usize + 2;
                (self.buf.len() >= end).then_some(end)
            }
            b'*' => {
                let n: i64 = std::str::from_utf8(payload).ok()?.parse().ok()?;
                if n < 0 {
                    return Some(after);
                }
                let mut at = after;
                for _ in 0..n {
                    at = self.complete(at)?;
                }
                Some(at)
            }
            other => panic!("{} sent an unknown reply type {other:?}", self.name),
        }
    }
}

/// Leading token of an error reply, or `None` when the reply is not an error.
fn error_token(reply: &[u8]) -> Option<&[u8]> {
    let rest = reply.strip_prefix(b"-")?;
    Some(rest.split(|b| *b == b' ').next().unwrap_or(rest))
}

/// Top-level elements of an array reply, sorted, for order-insensitive cases.
fn sorted_elements(reply: &[u8]) -> Option<Vec<Vec<u8>>> {
    if !reply.starts_with(b"*") {
        return None;
    }
    let mut parts: Vec<Vec<u8>> = reply
        .split(|b| *b == b'\r')
        .filter(|p| {
            let p = p.strip_prefix(b"\n").unwrap_or(p);
            !p.is_empty() && !p.starts_with(b"*") && !p.starts_with(b"$")
        })
        .map(|p| p.strip_prefix(b"\n").unwrap_or(p).to_vec())
        .collect();
    parts.sort();
    Some(parts)
}

fn start_luma() -> (String, CancellationToken, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        // No password: the corpus would otherwise need an AUTH that Redis does
        // not, and the two conversations must be identical.
        api_key: String::new(),
        resp_allow_flush: true,
        ..Config::default()
    };
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    config.resp_port = probe.local_addr().unwrap().port();
    drop(probe);

    let shutdown = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown.clone()).unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let port = runtime
        .block_on(spawn(
            &config,
            engine,
            Arc::new(RespMetrics::default()),
            None,
            shutdown.clone(),
        ))
        .unwrap()
        .expect("listener must bind");
    // The runtime has to outlive the listener task it is driving.
    std::mem::forget(runtime);
    (format!("127.0.0.1:{port}"), shutdown, dir)
}

#[test]
#[ignore = "needs a real Redis 7: set LUMA_DIFF_REDIS=host:port and run with --ignored"]
fn luma_answers_byte_for_byte_like_redis_7() {
    let redis_addr = std::env::var("LUMA_DIFF_REDIS").unwrap_or_default();
    assert!(
        !redis_addr.is_empty(),
        "LUMA_DIFF_REDIS is unset. This suite must not pass without a real Redis \
         to compare against — a green run with no subject reports coverage that \
         does not exist. Start one with:\n  \
         docker run -d -p 16379:6379 redis:7-alpine\n  \
         LUMA_DIFF_REDIS=127.0.0.1:16379 cargo test --test redis_differential -- --ignored"
    );

    let (luma_addr, shutdown, _dir) = start_luma();
    let mut redis = Peer::connect(&redis_addr, "redis")
        .unwrap_or_else(|e| panic!("cannot reach Redis at {redis_addr}: {e}"));
    let mut luma = Peer::connect(&luma_addr, "luma").expect("cannot reach Luma");

    // Both start empty, or the first divergence is just leftover state.
    redis.call(&["FLUSHALL"]);
    luma.call(&["FLUSHALL"]);

    let mut divergences = Vec::new();
    for case in corpus() {
        let expected = redis.call(case.argv);
        let actual = luma.call(case.argv);
        let same = match case.compare {
            Compare::Exact => expected == actual,
            Compare::ErrorToken => error_token(&expected) == error_token(&actual),
            Compare::Unordered => sorted_elements(&expected) == sorted_elements(&actual),
        };
        if !same {
            divergences.push(format!(
                "  {}\n    redis: {:?}\n    luma : {:?}",
                case.argv.join(" "),
                String::from_utf8_lossy(&expected),
                String::from_utf8_lossy(&actual),
            ));
        }
    }

    shutdown.cancel();
    assert!(
        divergences.is_empty(),
        "{} of {} commands diverged from Redis 7:\n{}",
        divergences.len(),
        corpus().len(),
        divergences.join("\n")
    );
}

/// The command corpus.
///
/// Grouped by the trap each group is aimed at rather than by command family:
/// nil versus empty array, `-WRONGTYPE` across every type pairing, arity and
/// syntax errors, and the one-keyspace rule.
fn corpus() -> Vec<Case> {
    let mut c = vec![
        // ── strings and keys ────────────────────────────────────────────────
        exact(&["SET", "k", "v"]),
        exact(&["GET", "k"]),
        exact(&["GET", "missing"]),
        exact(&["SET", "k", "v2"]),
        exact(&["GET", "k"]),
        exact(&["SETNX", "k", "nope"]),
        exact(&["SETNX", "fresh", "yes"]),
        exact(&["GET", "fresh"]),
        exact(&["STRLEN", "k"]),
        exact(&["STRLEN", "missing"]),
        exact(&["APPEND", "k", "!"]),
        exact(&["GET", "k"]),
        exact(&["EXISTS", "k"]),
        exact(&["EXISTS", "missing"]),
        exact(&["EXISTS", "k", "missing", "k"]),
        exact(&["TYPE", "k"]),
        exact(&["TYPE", "missing"]),
        exact(&["SET", "n", "10"]),
        exact(&["INCR", "n"]),
        exact(&["INCRBY", "n", "5"]),
        exact(&["DECR", "n"]),
        exact(&["DECRBY", "n", "3"]),
        exact(&["GET", "n"]),
        token(&["INCR", "k"]),
        exact(&["SET", "f", "1.5"]),
        exact(&["INCRBYFLOAT", "f", "0.25"]),
        exact(&["GETSET", "k", "replaced"]),
        exact(&["GET", "k"]),
        exact(&["MSET", "a", "1", "b", "2"]),
        exact(&["MGET", "a", "b", "missing"]),
        exact(&["DEL", "a"]),
        exact(&["DEL", "a"]),
        exact(&["DEL", "b", "missing"]),
        exact(&["SETEX", "ttl", "100", "v"]),
        // Rounded up: truncation answers 99 to the value just set.
        exact(&["TTL", "ttl"]),
        exact(&["TTL", "k"]),
        exact(&["TTL", "missing"]),
        exact(&["PERSIST", "ttl"]),
        exact(&["TTL", "ttl"]),
        exact(&["EXPIRE", "k", "100"]),
        exact(&["TTL", "k"]),
        exact(&["PERSIST", "k"]),
        exact(&["RENAME", "fresh", "renamed"]),
        exact(&["GET", "renamed"]),
        token(&["RENAME", "missing", "x"]),
        exact(&["RENAMENX", "renamed", "n"]),
        exact(&["GETDEL", "renamed"]),
        exact(&["GET", "renamed"]),
        // ── lists ───────────────────────────────────────────────────────────
        exact(&["RPUSH", "L", "a", "b", "c"]),
        exact(&["LRANGE", "L", "0", "-1"]),
        exact(&["LRANGE", "L", "1", "1"]),
        exact(&["LRANGE", "L", "5", "9"]),
        exact(&["LRANGE", "missing", "0", "-1"]),
        exact(&["LLEN", "L"]),
        exact(&["LLEN", "missing"]),
        exact(&["LINDEX", "L", "0"]),
        exact(&["LINDEX", "L", "-1"]),
        exact(&["LINDEX", "L", "99"]),
        exact(&["LINDEX", "missing", "0"]),
        exact(&["LSET", "L", "1", "B"]),
        exact(&["LRANGE", "L", "0", "-1"]),
        token(&["LSET", "L", "99", "x"]),
        token(&["LSET", "missing", "0", "x"]),
        exact(&["LPUSH", "L", "z"]),
        exact(&["LPUSHX", "L", "y"]),
        exact(&["LPUSHX", "nolist", "y"]),
        exact(&["EXISTS", "nolist"]),
        exact(&["RPUSHX", "L", "w"]),
        exact(&["RPUSHX", "nolist", "w"]),
        exact(&["LRANGE", "L", "0", "-1"]),
        exact(&["LPOP", "L"]),
        exact(&["RPOP", "L"]),
        exact(&["LPOP", "L", "2"]),
        exact(&["LPOP", "missing"]),
        // Null array, not an empty one: "no such queue" versus "drained".
        exact(&["LPOP", "missing", "2"]),
        exact(&["RPUSH", "R", "a", "b", "a", "c", "a"]),
        exact(&["LREM", "R", "2", "a"]),
        exact(&["LRANGE", "R", "0", "-1"]),
        exact(&["LREM", "R", "0", "a"]),
        exact(&["LRANGE", "R", "0", "-1"]),
        exact(&["RPUSH", "T", "1", "2", "3", "4", "5"]),
        exact(&["LTRIM", "T", "1", "3"]),
        exact(&["LRANGE", "T", "0", "-1"]),
        exact(&["LTRIM", "T", "9", "10"]),
        exact(&["EXISTS", "T"]),
        exact(&["LTRIM", "missing", "0", "1"]),
        exact(&["RPUSH", "M", "x", "y"]),
        exact(&["RPOPLPUSH", "M", "M2"]),
        exact(&["LRANGE", "M", "0", "-1"]),
        exact(&["LRANGE", "M2", "0", "-1"]),
        exact(&["RPOPLPUSH", "empty", "M2"]),
        exact(&["LMOVE", "M", "M2", "LEFT", "RIGHT"]),
        exact(&["LRANGE", "M2", "0", "-1"]),
        // Source and destination the same key: the round-robin idiom.
        exact(&["LMOVE", "M2", "M2", "LEFT", "RIGHT"]),
        exact(&["LRANGE", "M2", "0", "-1"]),
        token(&["LMOVE", "M2", "M2", "UP", "RIGHT"]),
        // ── hashes ──────────────────────────────────────────────────────────
        exact(&["HSET", "H", "f1", "v1"]),
        exact(&["HSET", "H", "f1", "v2"]),
        exact(&["HSET", "H", "f2", "v2", "f3", "v3"]),
        exact(&["HGET", "H", "f1"]),
        exact(&["HGET", "H", "nope"]),
        exact(&["HGET", "missing", "f"]),
        exact(&["HMGET", "H", "f1", "nope", "f2"]),
        exact(&["HMGET", "missing", "f1", "f2"]),
        exact(&["HLEN", "H"]),
        exact(&["HLEN", "missing"]),
        exact(&["HEXISTS", "H", "f1"]),
        exact(&["HEXISTS", "H", "nope"]),
        unordered(&["HKEYS", "H"]),
        unordered(&["HVALS", "H"]),
        unordered(&["HGETALL", "H"]),
        exact(&["HGETALL", "missing"]),
        exact(&["HDEL", "H", "f3"]),
        exact(&["HDEL", "H", "f3"]),
        exact(&["HSETNX", "H", "f1", "blocked"]),
        exact(&["HGET", "H", "f1"]),
        exact(&["HSETNX", "H", "brand", "new"]),
        exact(&["HGET", "H", "brand"]),
        exact(&["HSET", "HN", "n", "5"]),
        exact(&["HINCRBY", "HN", "n", "3"]),
        exact(&["HINCRBY", "HN", "fresh", "2"]),
        exact(&["HINCRBY", "H", "f1", "1"]),
        // ── sets ────────────────────────────────────────────────────────────
        exact(&["SADD", "S", "a", "b", "c"]),
        exact(&["SADD", "S", "a"]),
        exact(&["SCARD", "S"]),
        exact(&["SCARD", "missing"]),
        exact(&["SISMEMBER", "S", "a"]),
        exact(&["SISMEMBER", "S", "zz"]),
        exact(&["SISMEMBER", "missing", "a"]),
        unordered(&["SMEMBERS", "S"]),
        exact(&["SMEMBERS", "missing"]),
        exact(&["SREM", "S", "a"]),
        exact(&["SREM", "S", "a"]),
        // SPOP/SRANDMEMBER pick at random in Redis and in stored order here —
        // a divergence recorded in docs/RESP.md — so only the empty cases,
        // where there is nothing to choose between, are comparable.
        exact(&["SPOP", "missing"]),
        exact(&["SPOP", "missing", "2"]),
        exact(&["SRANDMEMBER", "missing"]),
        exact(&["SRANDMEMBER", "missing", "2"]),
        // ── sorted sets ─────────────────────────────────────────────────────
        exact(&["ZADD", "Z", "1", "a", "2", "b", "3", "c"]),
        exact(&["ZADD", "Z", "9", "a"]),
        exact(&["ZADD", "Z", "CH", "8", "a"]),
        exact(&["ZADD", "Z", "CH", "8", "a"]),
        exact(&["ZCARD", "Z"]),
        exact(&["ZCARD", "missing"]),
        exact(&["ZSCORE", "Z", "b"]),
        exact(&["ZSCORE", "Z", "nope"]),
        exact(&["ZSCORE", "missing", "b"]),
        exact(&["ZMSCORE", "Z", "b", "nope"]),
        exact(&["ZMSCORE", "missing", "b", "c"]),
        exact(&["ZRANGE", "Z", "0", "-1"]),
        exact(&["ZRANGE", "Z", "0", "-1", "WITHSCORES"]),
        // `0 -1 REV` is the ordinary way to ask for everything descending, and
        // it returned an empty array until this suite ran.
        exact(&["ZRANGE", "Z", "0", "-1", "REV"]),
        exact(&["ZRANGE", "missing", "0", "-1"]),
        exact(&["ZREVRANGE", "Z", "0", "-1"]),
        exact(&["ZREVRANGE", "Z", "0", "1", "WITHSCORES"]),
        exact(&["ZRANGEBYSCORE", "Z", "2", "3"]),
        exact(&["ZRANGEBYSCORE", "Z", "-inf", "+inf"]),
        exact(&["ZRANGEBYSCORE", "Z", "-inf", "+inf", "WITHSCORES"]),
        exact(&["ZREVRANGEBYSCORE", "Z", "3", "2"]),
        exact(&["ZRANGE", "Z", "2", "3", "BYSCORE"]),
        exact(&["ZRANGE", "Z", "3", "2", "BYSCORE", "REV"]),
        exact(&["ZRANGE", "Z", "-inf", "+inf", "BYSCORE", "LIMIT", "1", "2"]),
        exact(&["ZRANGE", "Z", "-inf", "+inf", "BYSCORE", "LIMIT", "1", "-1"]),
        token(&["ZRANGE", "Z", "0", "-1", "LIMIT", "0", "1"]),
        exact(&["ZRANK", "Z", "b"]),
        exact(&["ZRANK", "Z", "nope"]),
        exact(&["ZREVRANK", "Z", "b"]),
        exact(&["ZREVRANK", "Z", "nope"]),
        exact(&["ZCOUNT", "Z", "2", "3"]),
        exact(&["ZCOUNT", "Z", "-inf", "+inf"]),
        exact(&["ZCOUNT", "missing", "0", "9"]),
        exact(&["ZINCRBY", "Z", "1.5", "b"]),
        exact(&["ZINCRBY", "Z", "2", "brandnew"]),
        exact(&["ZSCORE", "Z", "brandnew"]),
        exact(&["ZADD", "Z", "NX", "99", "b"]),
        exact(&["ZSCORE", "Z", "b"]),
        exact(&["ZADD", "Z", "XX", "4", "neverseen"]),
        exact(&["ZSCORE", "Z", "neverseen"]),
        exact(&["ZADD", "Z2", "GT", "5", "g"]),
        exact(&["ZADD", "Z2", "GT", "3", "g"]),
        exact(&["ZSCORE", "Z2", "g"]),
        exact(&["ZADD", "Z2", "GT", "7", "g"]),
        exact(&["ZSCORE", "Z2", "g"]),
        exact(&["ZADD", "Z2", "LT", "9", "g"]),
        exact(&["ZSCORE", "Z2", "g"]),
        exact(&["ZADD", "Z2", "INCR", "2", "g"]),
        // Blocked by NX: a nil, not a 0, which would be a real score.
        exact(&["ZADD", "Z2", "NX", "INCR", "2", "g"]),
        exact(&["ZADD", "Z2", "XX", "INCR", "2", "nothere"]),
        token(&["ZADD", "Z2", "NX", "XX", "1", "g"]),
        token(&["ZADD", "Z2", "GT", "LT", "1", "g"]),
        exact(&["ZADD", "ZL", "0", "a", "0", "b", "0", "c", "0", "d"]),
        exact(&["ZRANGE", "ZL", "-", "+", "BYLEX"]),
        exact(&["ZRANGE", "ZL", "[b", "[c", "BYLEX"]),
        exact(&["ZRANGE", "ZL", "(b", "(d", "BYLEX"]),
        token(&["ZRANGE", "ZL", "b", "c", "BYLEX"]),
        token(&["ZRANGE", "ZL", "-", "+", "BYLEX", "WITHSCORES"]),
        exact(&["ZADD", "ZP", "1", "a", "2", "b", "3", "c"]),
        exact(&["ZPOPMIN", "ZP"]),
        exact(&["ZPOPMAX", "ZP"]),
        exact(&["ZPOPMIN", "ZP", "5"]),
        exact(&["ZPOPMIN", "ZP"]),
        exact(&["ZPOPMIN", "missing"]),
        exact(&["ZADD", "ZR", "1", "a", "2", "b", "3", "c", "4", "d"]),
        exact(&["ZREMRANGEBYSCORE", "ZR", "2", "3"]),
        exact(&["ZRANGE", "ZR", "0", "-1"]),
        exact(&["ZREMRANGEBYRANK", "ZR", "0", "0"]),
        exact(&["ZRANGE", "ZR", "0", "-1"]),
        exact(&["ZREM", "ZR", "d"]),
        exact(&["ZREM", "ZR", "d"]),
        // An infinite score used to be written as JSON null, which made the
        // whole sorted set unreadable on the next command.
        exact(&["ZADD", "ZI", "+inf", "high", "-inf", "low"]),
        exact(&["ZSCORE", "ZI", "high"]),
        exact(&["ZSCORE", "ZI", "low"]),
        exact(&["ZRANGE", "ZI", "0", "-1", "WITHSCORES"]),
    ];

    // ── one keyspace, one type per key ──────────────────────────────────────
    c.extend([
        exact(&["FLUSHALL"]),
        exact(&["SET", "mix", "string"]),
        exact(&["TYPE", "mix"]),
        token(&["RPUSH", "mix", "x"]),
        exact(&["DEL", "mix"]),
        exact(&["RPUSH", "mix", "x"]),
        exact(&["TYPE", "mix"]),
        // SET is the one command that replaces any type.
        exact(&["SET", "mix", "now-a-string"]),
        exact(&["TYPE", "mix"]),
        token(&["LLEN", "mix"]),
        exact(&["GET", "mix"]),
        exact(&["DEL", "mix"]),
        exact(&["RPUSH", "onlylist", "a"]),
        exact(&["EXISTS", "onlylist"]),
        exact(&["TYPE", "onlylist"]),
        exact(&["SETNX", "onlylist", "v"]),
        // MGET answers nil for a wrongly-typed key rather than failing.
        exact(&["MGET", "onlylist"]),
        exact(&["DEL", "onlylist"]),
        exact(&["EXISTS", "onlylist"]),
        exact(&["RPUSH", "ttllist", "a"]),
        exact(&["EXPIRE", "ttllist", "100"]),
        exact(&["TTL", "ttllist"]),
        exact(&["PERSIST", "ttllist"]),
        exact(&["TTL", "ttllist"]),
        exact(&["RENAME", "ttllist", "movedlist"]),
        exact(&["LRANGE", "movedlist", "0", "-1"]),
        exact(&["EXISTS", "ttllist"]),
        exact(&["HSET", "onlyhash", "f", "v"]),
        exact(&["TYPE", "onlyhash"]),
        exact(&["SADD", "onlyset", "m"]),
        exact(&["TYPE", "onlyset"]),
        exact(&["ZADD", "onlyzset", "1", "m"]),
        exact(&["TYPE", "onlyzset"]),
        exact(&["FLUSHALL"]),
        exact(&["SET", "plain", "1"]),
        exact(&["RPUSH", "alist", "a"]),
        exact(&["HSET", "ahash", "f", "v"]),
        exact(&["DBSIZE"]),
        // `KEYS *` used to hand back the internal `struct:` names.
        unordered(&["KEYS", "*"]),
    ]);

    // ── WRONGTYPE across every pairing ──────────────────────────────────────
    c.extend([
        exact(&["FLUSHALL"]),
        exact(&["SET", "str", "x"]),
        exact(&["RPUSH", "lst", "x"]),
        exact(&["HSET", "hsh", "f", "v"]),
        exact(&["SADD", "st", "m"]),
        exact(&["ZADD", "zst", "1", "m"]),
        token(&["LPUSH", "str", "v"]),
        token(&["LRANGE", "str", "0", "-1"]),
        token(&["LINDEX", "hsh", "0"]),
        token(&["LSET", "hsh", "0", "v"]),
        token(&["LTRIM", "hsh", "0", "1"]),
        token(&["RPOPLPUSH", "hsh", "x"]),
        token(&["RPOPLPUSH", "lst", "str"]),
        token(&["HSET", "lst", "f", "v"]),
        token(&["HGET", "lst", "f"]),
        token(&["HGETALL", "lst"]),
        token(&["HSETNX", "lst", "f", "v"]),
        token(&["SADD", "lst", "m"]),
        token(&["SMEMBERS", "lst"]),
        token(&["SPOP", "lst"]),
        token(&["SRANDMEMBER", "lst"]),
        // Cardinality used to answer the *other* type's length instead of an
        // error: a real answer to a question the client did not ask.
        token(&["SCARD", "lst"]),
        token(&["ZCARD", "lst"]),
        token(&["HLEN", "lst"]),
        token(&["LLEN", "hsh"]),
        token(&["ZADD", "lst", "1", "m"]),
        token(&["ZSCORE", "lst", "m"]),
        token(&["ZRANGE", "lst", "0", "-1"]),
        token(&["ZMSCORE", "lst", "m"]),
        token(&["ZCOUNT", "lst", "0", "1"]),
        token(&["ZINCRBY", "lst", "1", "m"]),
        token(&["ZPOPMIN", "lst"]),
        token(&["ZREVRANK", "lst", "m"]),
        token(&["ZREMRANGEBYRANK", "lst", "0", "1"]),
        token(&["GET", "lst"]),
        token(&["INCR", "lst"]),
        token(&["APPEND", "lst", "x"]),
        token(&["STRLEN", "lst"]),
        token(&["GETDEL", "lst"]),
    ]);

    // ── transactions ────────────────────────────────────────────────────────
    //
    // This whole group was missing, and its absence let a real bug through: a
    // `PUBLISH` queued inside `MULTI` was rejected as an unknown command, which
    // is exactly what a redis-py pipeline sends. Celery's result backend writes
    // its result with a pipelined `SETEX` + `PUBLISH`, so a real worker executed
    // the task and then hung forever waiting for a result that was never stored.
    c.extend([
        exact(&["FLUSHALL"]),
        exact(&["MULTI"]),
        exact(&["SET", "tx", "1"]),
        exact(&["INCR", "tx"]),
        exact(&["EXEC"]),
        exact(&["GET", "tx"]),
        // A transaction that queues nothing still runs.
        exact(&["MULTI"]),
        exact(&["EXEC"]),
        exact(&["MULTI"]),
        exact(&["SET", "tx", "9"]),
        exact(&["DISCARD"]),
        exact(&["GET", "tx"]),
        token(&["EXEC"]),
        token(&["DISCARD"]),
        // The shape redis-py's pipeline sends, and the one Celery depends on.
        exact(&["MULTI"]),
        exact(&["SETEX", "celery-task-meta-x", "60", "payload"]),
        exact(&["PUBLISH", "celery-task-meta-x", "payload"]),
        exact(&["EXEC"]),
        exact(&["GET", "celery-task-meta-x"]),
        // `SUBSCRIBE` inside `MULTI` is deliberately *not* in this corpus.
        // Redis queues it and `EXEC` runs it, which puts that connection into
        // subscriber mode while Luma's stays out — and from there every
        // subsequent command diverges for a reason already recorded in
        // docs/RESP.md (Luma accepts the full command set while subscribed).
        // Comparing it here would report one documented divergence as a dozen.
        // Luma's own behaviour is pinned in `resp::commands`.
        exact(&["PUBLISH", "nobody-listening", "x"]),
        // WATCH aborts the transaction when the key moved underneath it.
        exact(&["SET", "guarded", "1"]),
        exact(&["WATCH", "guarded"]),
        exact(&["MULTI"]),
        exact(&["GET", "guarded"]),
        exact(&["EXEC"]),
        exact(&["UNWATCH"]),
    ]);

    // ── arity and syntax, which Redis reports *before* a type conflict ──────
    c.extend([
        token(&["GET"]),
        token(&["SET", "only"]),
        token(&["LPUSH", "str"]),
        token(&["HSET", "str", "f"]),
        token(&["ZADD", "str", "1"]),
        token(&["LRANGE", "str", "0"]),
        token(&["EXPIRE", "str"]),
        token(&["SETEX", "str", "notanumber", "v"]),
        token(&["LPOP", "str", "notanumber"]),
        token(&["ZADD", "str", "notafloat", "m"]),
        token(&["ZRANGE", "str", "0", "-1", "SIDEWAYS"]),
    ]);

    c
}
