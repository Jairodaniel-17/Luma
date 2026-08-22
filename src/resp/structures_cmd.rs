//! RESP commands for lists, hashes, sets and sorted sets.
//!
//! Block 5 of `docs/PLAN-MAESTRO.md` (F2.1–F2.4). The semantics live in
//! `engine::structures`, pinned by its own tests; this module is the protocol
//! surface over them, so what it has to get right is the *reply shapes* rather
//! than the data structures.
//!
//! ## The two traps this module exists to avoid
//!
//! **nil versus empty.** `LRANGE` on a missing key is an empty array, not a null
//! one. `LPOP` on a missing key is a null bulk string, not an empty one. Clients
//! branch on the difference, and getting it backwards produces a client that
//! hangs or crashes rather than one that reports an error.
//!
//! **`-WRONGTYPE` on a type mismatch.** Not a generic error: redis-py raises a
//! distinct exception for it, and kombu relies on that to detect a fouled
//! queue key rather than treating it as a transient failure.

use crate::engine::structures::{Structure, StructureError, Structures};
use crate::engine::Engine;
use crate::resp::commands::Session;
use crate::resp::protocol::Value;

/// Every command whose first argument names a key that must hold a structure.
///
/// Kept beside the dispatch table on purpose: the one-keyspace guard in
/// `commands.rs` reads this to know that a plain string under that name is a
/// type error. A command added to the table and forgotten here loses the guard
/// silently, so the two belong within sight of each other.
///
/// The blocking commands are included even though they are dispatched in
/// `commands.rs`: they operate on lists and sorted sets like the rest.
pub const STRUCTURE_COMMANDS: &[&str] = &[
    "BLMOVE",
    "BLPOP",
    "BRPOP",
    "BRPOPLPUSH",
    "BZPOPMAX",
    "BZPOPMIN",
    "HDEL",
    "HEXISTS",
    "HGET",
    "HGETALL",
    "HINCRBY",
    "HKEYS",
    "HLEN",
    "HMGET",
    "HSCAN",
    "HSET",
    "HSETNX",
    "HVALS",
    "LINDEX",
    "LLEN",
    "LMOVE",
    "LPOP",
    "LPUSH",
    "LPUSHX",
    "LRANGE",
    "LREM",
    "LSET",
    "LTRIM",
    "RPOP",
    "RPOPLPUSH",
    "RPUSH",
    "RPUSHX",
    "SADD",
    "SCARD",
    "SISMEMBER",
    "SMEMBERS",
    "SPOP",
    "SRANDMEMBER",
    "SREM",
    "SSCAN",
    "ZADD",
    "ZCARD",
    "ZCOUNT",
    "ZINCRBY",
    "ZMSCORE",
    "ZPOPMAX",
    "ZPOPMIN",
    "ZRANGE",
    "ZRANGEBYSCORE",
    "ZRANK",
    "ZREM",
    "ZREMRANGEBYRANK",
    "ZREMRANGEBYSCORE",
    "ZREVRANGE",
    "ZREVRANGEBYSCORE",
    "ZREVRANK",
    "ZSCAN",
    "ZSCORE",
];

fn err(message: impl Into<String>) -> Value {
    Value::Error(message.into())
}

/// Render a structure error as the Redis error a client expects.
fn structure_error(e: StructureError) -> Value {
    match e {
        // The exact string matters: clients match on the leading token.
        StructureError::WrongType => {
            err("WRONGTYPE Operation against a key holding the wrong kind of value")
        }
        StructureError::NotAnInteger => err("ERR value is not an integer or out of range"),
        StructureError::HashNotAnInteger => err("ERR hash value is not an integer"),
        StructureError::NotANumber => err("ERR value is not a valid float"),
        StructureError::MemberTooLong => err("ERR value is too large"),
        StructureError::TooManyEntries => err("ERR structure is at its configured entry limit"),
        // Redis's exact wording, which redis-py maps to its own exception.
        StructureError::IndexOutOfRange => err("ERR index out of range"),
        // Not WRONGTYPE: the client did nothing wrong and retrying with
        // another command will not help.
        StructureError::Corrupt(detail) => err(format!("ERR {detail}")),
    }
}

fn scoped(session: &Session, key: &[u8]) -> String {
    let key = String::from_utf8_lossy(key);
    match &session.tenant {
        Some(tenant) => format!("{tenant}{}{key}", super::commands::TENANT_SEP),
        None => key.to_string(),
    }
}

fn wrong_args(command: &str) -> Value {
    err(format!(
        "ERR wrong number of arguments for '{}' command",
        command.to_lowercase()
    ))
}

fn parse_i64(raw: &[u8]) -> Option<i64> {
    std::str::from_utf8(raw).ok()?.trim().parse().ok()
}

fn parse_f64(raw: &[u8]) -> Option<f64> {
    let text = std::str::from_utf8(raw).ok()?.trim();
    // Redis accepts these spellings for the bounds of a score range.
    match text.to_ascii_lowercase().as_str() {
        "+inf" | "inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        _ => text.parse().ok(),
    }
}

/// Dispatch a structure command, or `None` if the name is not one of ours.
pub fn dispatch(engine: &Engine, session: &Session, name: &str, args: &[Vec<u8>]) -> Option<Value> {
    let structures = Structures::new(engine);
    Some(match name {
        // ── lists ────────────────────────────────────────────────────────────
        "LPUSH" | "RPUSH" => push(&structures, session, args, name == "LPUSH"),
        "LPUSHX" | "RPUSHX" => pushx(&structures, session, args, name == "LPUSHX"),
        "LPOP" | "RPOP" => pop(&structures, session, args, name == "LPOP", name),
        "LLEN" => cardinality(&structures, session, args, "llen"),
        "LRANGE" => lrange(&structures, session, args),
        "LREM" => lrem(&structures, session, args),
        "LINDEX" => lindex(&structures, session, args),
        "LSET" => lset(&structures, session, args),
        "LTRIM" => ltrim(&structures, session, args),
        "RPOPLPUSH" => lmove(&structures, session, args, true),
        "LMOVE" => lmove(&structures, session, args, false),

        // ── hashes ───────────────────────────────────────────────────────────
        "HSET" => hset(&structures, session, args),
        "HGET" => hget(&structures, session, args),
        "HMGET" => hmget(&structures, session, args),
        "HDEL" => hdel(&structures, session, args),
        "HGETALL" => hgetall(&structures, session, args),
        "HLEN" => cardinality(&structures, session, args, "hlen"),
        "HEXISTS" => hexists(&structures, session, args),
        "HKEYS" | "HVALS" => hkeys_or_vals(&structures, session, args, name == "HKEYS"),
        "HINCRBY" => hincrby(&structures, session, args),
        "HSETNX" => hsetnx(&structures, session, args),
        "HSCAN" => hscan(&structures, session, args),

        // ── sets ─────────────────────────────────────────────────────────────
        "SADD" => sadd(&structures, session, args),
        "SREM" => srem(&structures, session, args),
        "SMEMBERS" => smembers(&structures, session, args),
        "SISMEMBER" => sismember(&structures, session, args),
        "SCARD" => cardinality(&structures, session, args, "scard"),
        "SPOP" => spop(&structures, session, args),
        "SRANDMEMBER" => srandmember(&structures, session, args),
        "SSCAN" => sscan(&structures, session, args),

        // ── sorted sets ──────────────────────────────────────────────────────
        "ZADD" => zadd(&structures, session, args),
        "ZREM" => zrem(&structures, session, args),
        "ZSCORE" => zscore(&structures, session, args),
        "ZCARD" => cardinality(&structures, session, args, "zcard"),
        "ZRANGE" => zrange(&structures, session, args),
        // `ZREVRANGE key start stop [WITHSCORES]` is `ZRANGE ... REV`. Kept as
        // its own name because clients written before Redis 6.2 send it, and a
        // rewrite here is cheaper than a second implementation to keep in sync.
        "ZREVRANGE" => {
            let mut rewritten = args.to_vec();
            rewritten.push(b"REV".to_vec());
            zrange(&structures, session, &rewritten)
        }
        "ZRANGEBYSCORE" => zrangebyscore(&structures, session, args),
        "ZRANK" => zrank(&structures, session, args),
        "ZREVRANK" => zrevrank(&structures, session, args),
        "ZMSCORE" => zmscore(&structures, session, args),
        "ZCOUNT" => zcount(&structures, session, args),
        "ZINCRBY" => zincrby(&structures, session, args),
        "ZREVRANGEBYSCORE" => zrevrangebyscore(&structures, session, args),
        "ZREMRANGEBYSCORE" => zremrangebyscore(&structures, session, args),
        "ZREMRANGEBYRANK" => zremrangebyrank(&structures, session, args),
        "ZPOPMIN" => zpop(&structures, session, args, true),
        "ZPOPMAX" => zpop(&structures, session, args, false),
        "ZSCAN" => zscan(&structures, session, args),

        _ => return None,
    })
}

/// Read a structure, mapping "absent" to the caller's chosen empty reply.
fn with_structure<T>(
    structures: &Structures<'_>,
    key: &str,
    absent: T,
    read: impl FnOnce(&Structure) -> Result<T, StructureError>,
) -> Result<T, StructureError> {
    match structures.load(key)? {
        Some((structure, _)) => read(&structure),
        None => Ok(absent),
    }
}

// ── lists ────────────────────────────────────────────────────────────────────

fn push(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>], left: bool) -> Value {
    if args.len() < 2 {
        return wrong_args(if left { "lpush" } else { "rpush" });
    }
    let values: Vec<Vec<u8>> = args[1..].to_vec();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_list, |s| {
        if left {
            s.lpush(values.clone())
        } else {
            s.rpush(values.clone())
        }
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn pop(
    structures: &Structures<'_>,
    session: &Session,
    args: &[Vec<u8>],
    left: bool,
    command: &str,
) -> Value {
    if args.is_empty() || args.len() > 2 {
        return wrong_args(command);
    }
    // Without COUNT the reply is a single bulk string; with COUNT it is an
    // array. A client that sent COUNT expects an array even for one element.
    let count = match args.get(1) {
        Some(raw) => match parse_i64(raw) {
            Some(n) if n >= 0 => Some(n as usize),
            _ => return err("ERR value is out of range, must be positive"),
        },
        None => None,
    };

    let key = scoped(session, &args[0]);
    let existed = matches!(structures.load(&key), Ok(Some(_)));
    let result = structures.mutate(&key, Structure::empty_list, |s| {
        if left {
            s.lpop(count.unwrap_or(1))
        } else {
            s.rpop(count.unwrap_or(1))
        }
    });

    match result {
        Ok(applied) => match count {
            // A *missing key* with COUNT is a null array, while an existing but
            // exhausted list is an empty one. Redis keeps them distinct and a
            // client uses the difference to tell "no such queue" from "queue is
            // drained".
            Some(_) if !existed => Value::Array(None),
            Some(_) => Value::Array(Some(applied.value.into_iter().map(Value::bulk).collect())),
            None => match applied.value.into_iter().next() {
                Some(bytes) => Value::bulk(bytes),
                // A missing element is nil, never an empty bulk string.
                None => Value::nil(),
            },
        },
        Err(e) => structure_error(e),
    }
}

fn lrange(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("lrange");
    }
    let (Some(start), Some(stop)) = (parse_i64(&args[1]), parse_i64(&args[2])) else {
        return err("ERR value is not an integer or out of range");
    };
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        s.lrange(start, stop)
    }) {
        // An empty array, never a null one: a client distinguishes "no
        // elements" from "no such thing".
        Ok(items) => Value::Array(Some(items.into_iter().map(Value::bulk).collect())),
        Err(e) => structure_error(e),
    }
}

fn lrem(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("lrem");
    }
    let Some(count) = parse_i64(&args[1]) else {
        return err("ERR value is not an integer or out of range");
    };
    let value = args[2].clone();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_list, |s| {
        s.lrem(count, &value)
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

// ── shared ───────────────────────────────────────────────────────────────────

fn cardinality(
    structures: &Structures<'_>,
    session: &Session,
    args: &[Vec<u8>],
    command: &str,
) -> Value {
    if args.len() != 1 {
        return wrong_args(command);
    }
    // The variant has to be checked, not just the length. `SCARD` on a list
    // used to return the list's length instead of `-WRONGTYPE`: a real answer
    // to a question the client did not ask, which is worse than an error
    // because nothing looks broken.
    let expected = match command {
        "llen" => "list",
        "hlen" => "hash",
        "scard" => "set",
        _ => "zset",
    };
    // A missing key is 0, not an error: `LLEN` of nothing is nothing.
    match with_structure(structures, &scoped(session, &args[0]), 0, |s| {
        if s.type_name() != expected {
            return Err(StructureError::WrongType);
        }
        Ok(s.len())
    }) {
        Ok(len) => Value::Integer(len as i64),
        Err(e) => structure_error(e),
    }
}

// ── hashes ───────────────────────────────────────────────────────────────────

fn hset(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 3 || !(args.len() - 1).is_multiple_of(2) {
        return wrong_args("hset");
    }
    let pairs: Vec<(Vec<u8>, Vec<u8>)> = args[1..]
        .chunks_exact(2)
        .map(|c| (c[0].clone(), c[1].clone()))
        .collect();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_hash, |s| {
        s.hset(pairs.clone())
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn hget(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("hget");
    }
    match with_structure(structures, &scoped(session, &args[0]), None, |s| {
        Ok(s.as_hash()?.get(&args[1]).cloned())
    }) {
        Ok(Some(bytes)) => Value::bulk(bytes),
        Ok(None) => Value::nil(),
        Err(e) => structure_error(e),
    }
}

fn hmget(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return wrong_args("hmget");
    }
    let fields = &args[1..];
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        let hash = s.as_hash()?;
        Ok(fields.iter().map(|f| hash.get(f).cloned()).collect())
    }) {
        Ok(values) => {
            // Positional, like MGET: a missing field is a nil inside the array.
            let mut out: Vec<Value> = values
                .into_iter()
                .map(|v: Option<Vec<u8>>| match v {
                    Some(bytes) => Value::bulk(bytes),
                    None => Value::nil(),
                })
                .collect();
            // A missing key still returns one nil per requested field.
            if out.is_empty() {
                out = fields.iter().map(|_| Value::nil()).collect();
            }
            Value::Array(Some(out))
        }
        Err(e) => structure_error(e),
    }
}

fn hdel(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return wrong_args("hdel");
    }
    let fields: Vec<Vec<u8>> = args[1..].to_vec();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_hash, |s| {
        let hash = s.as_hash_mut()?;
        Ok(fields.iter().filter(|f| hash.remove(*f).is_some()).count())
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn hgetall(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return wrong_args("hgetall");
    }
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        Ok(s.as_hash()?
            .iter()
            .flat_map(|(k, v)| [Value::bulk(k.clone()), Value::bulk(v.clone())])
            .collect())
    }) {
        // A flat field,value,field,value array — not a map, because RESP2
        // clients parse it pairwise.
        Ok(items) => Value::Array(Some(items)),
        Err(e) => structure_error(e),
    }
}

fn hexists(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("hexists");
    }
    match with_structure(structures, &scoped(session, &args[0]), false, |s| {
        Ok(s.as_hash()?.contains_key(&args[1]))
    }) {
        Ok(found) => Value::Integer(found as i64),
        Err(e) => structure_error(e),
    }
}

fn hkeys_or_vals(
    structures: &Structures<'_>,
    session: &Session,
    args: &[Vec<u8>],
    keys: bool,
) -> Value {
    if args.len() != 1 {
        return wrong_args(if keys { "hkeys" } else { "hvals" });
    }
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        let hash = s.as_hash()?;
        Ok(if keys {
            hash.keys().cloned().map(Value::bulk).collect()
        } else {
            hash.values().cloned().map(Value::bulk).collect()
        })
    }) {
        Ok(items) => Value::Array(Some(items)),
        Err(e) => structure_error(e),
    }
}

fn hincrby(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("hincrby");
    }
    let Some(delta) = parse_i64(&args[2]) else {
        return err("ERR value is not an integer or out of range");
    };
    let field = args[1].clone();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_hash, |s| {
        s.hincrby(field.clone(), delta)
    }) {
        Ok(applied) => Value::Integer(applied.value),
        Err(e) => structure_error(e),
    }
}

// ── sets ─────────────────────────────────────────────────────────────────────

fn sadd(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return wrong_args("sadd");
    }
    let members: Vec<Vec<u8>> = args[1..].to_vec();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_set, |s| {
        s.sadd(members.clone())
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn srem(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return wrong_args("srem");
    }
    let members: Vec<Vec<u8>> = args[1..].to_vec();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_set, |s| {
        s.srem(&members)
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn smembers(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 1 {
        return wrong_args("smembers");
    }
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        Ok(s.as_set()?.iter().cloned().map(Value::bulk).collect())
    }) {
        Ok(items) => Value::Array(Some(items)),
        Err(e) => structure_error(e),
    }
}

fn sismember(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("sismember");
    }
    match with_structure(structures, &scoped(session, &args[0]), false, |s| {
        Ok(s.as_set()?.contains(&args[1]))
    }) {
        Ok(found) => Value::Integer(found as i64),
        Err(e) => structure_error(e),
    }
}

// ── sorted sets ──────────────────────────────────────────────────────────────

fn zadd(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 3 {
        return wrong_args("zadd");
    }
    let (flags, consumed) = match ZAddFlags::parse(&args[1..]) {
        Ok(parsed) => parsed,
        Err(e) => return e,
    };
    let rest = &args[1 + consumed..];
    if rest.is_empty() || !rest.len().is_multiple_of(2) {
        return wrong_args("zadd");
    }
    if flags.incr && rest.len() != 2 {
        // Redis refuses this: INCR replies with one score, so it cannot be
        // given several members to increment.
        return err("ERR INCR option supports a single increment-element pair");
    }

    let mut pairs = Vec::new();
    for chunk in rest.chunks_exact(2) {
        let Some(score) = parse_f64(&chunk[0]) else {
            return err("ERR value is not a valid float");
        };
        pairs.push((score, chunk[1].clone()));
    }

    let result = structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        let zset = s.as_zset_mut()?;
        let mut added = 0i64;
        let mut changed = 0i64;
        let mut incremented = None;
        for (score, member) in &pairs {
            let current = zset.score(member);
            let target = if flags.incr {
                current.unwrap_or(0.0) + score
            } else {
                *score
            };
            if !flags.allows(current, target) {
                continue;
            }
            if target.is_nan() {
                return Err(StructureError::NotANumber);
            }
            if zset.add(member.clone(), target)? {
                added += 1;
                changed += 1;
            } else if current != Some(target) {
                changed += 1;
            }
            incremented = Some(target);
        }
        Ok((added, changed, incremented))
    });

    match result {
        Ok(applied) => {
            let (added, changed, incremented) = applied.value;
            if flags.incr {
                // A nil, not 0: NX or XX blocked the write, and 0 would be
                // indistinguishable from a score that really is 0.
                return match incremented {
                    Some(score) => Value::bulk(format_score(score)),
                    None => Value::nil(),
                };
            }
            // Without CH this counts members *added*, not updated — which is
            // what a client uses to tell a new job from a rescheduled one.
            Value::Integer(if flags.ch { changed } else { added })
        }
        Err(e) => structure_error(e),
    }
}

fn zrem(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return wrong_args("zrem");
    }
    let members: Vec<Vec<u8>> = args[1..].to_vec();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        let zset = s.as_zset_mut()?;
        Ok(members.iter().filter(|m| zset.remove(m)).count())
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn zscore(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("zscore");
    }
    match with_structure(structures, &scoped(session, &args[0]), None, |s| {
        Ok(s.as_zset()?.score(&args[1]))
    }) {
        // Scores come back as bulk strings, not integers: Redis scores are
        // doubles and a client parses the string.
        Ok(Some(score)) => Value::bulk(format_score(score)),
        Ok(None) => Value::nil(),
        Err(e) => structure_error(e),
    }
}

fn zrange(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 3 {
        return wrong_args("zrange");
    }
    let options = match ZRangeOptions::parse(&args[3..]) {
        Ok(parsed) => parsed,
        Err(e) => return e,
    };

    let all = match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        Ok(s.as_zset()?
            .range()
            .map(|(m, score)| (m.clone(), score))
            .collect::<Vec<_>>())
    }) {
        Ok(all) => all,
        Err(e) => return structure_error(e),
    };

    // `REV` means two different things depending on how the bounds are read,
    // and conflating them is why `ZRANGE key 0 -1 REV` — the ordinary way to
    // ask for everything descending — came back empty:
    //
    // * With **ranks**, the indexes address the *reversed* ordering. `0 -1`
    //   still means "all of it"; the list is flipped first and the bounds are
    //   applied as given.
    // * With **BYSCORE/BYLEX**, the client itself states the bounds backwards
    //   (`max min`). They are swapped, matched against the ascending set, and
    //   the result is flipped at the end.
    let (low, high) = if options.rev && options.by != RangeBy::Rank {
        (&args[2], &args[1])
    } else {
        (&args[1], &args[2])
    };

    let mut selected: Vec<(Vec<u8>, f64)> = match options.by {
        RangeBy::Rank => {
            let (Some(start), Some(stop)) = (parse_i64(low), parse_i64(high)) else {
                return err("ERR value is not an integer or out of range");
            };
            let mut ordered = all;
            if options.rev {
                ordered.reverse();
            }
            let len = ordered.len() as i64;
            let (from, to) = normalize_range(start, stop, len);
            if from > to {
                Vec::new()
            } else {
                ordered
                    .into_iter()
                    .skip(from as usize)
                    .take((to - from + 1) as usize)
                    .collect()
            }
        }
        RangeBy::Score => {
            let (Some(min), Some(max)) = (parse_f64(low), parse_f64(high)) else {
                return err("ERR min or max is not a float");
            };
            all.into_iter()
                .filter(|(_, score)| *score >= min && *score <= max)
                .collect()
        }
        RangeBy::Lex => {
            let (Some(min), Some(max)) = (LexBound::parse(low), LexBound::parse(high)) else {
                return err("ERR min or max not valid string range item");
            };
            all.into_iter()
                .filter(|(member, _)| min.accepts_low(member) && max.accepts_high(member))
                .collect()
        }
    };

    // Already handled above for ranks; flipping twice would undo it.
    if options.rev && options.by != RangeBy::Rank {
        selected.reverse();
    }

    if let Some((offset, count)) = options.limit {
        if offset < 0 {
            // Redis returns nothing rather than treating it as from-the-end.
            selected.clear();
        } else {
            let mut windowed: Vec<(Vec<u8>, f64)> =
                selected.into_iter().skip(offset as usize).collect();
            // A negative count means "everything from the offset".
            if count >= 0 {
                windowed.truncate(count as usize);
            }
            selected = windowed;
        }
    }

    let mut out = Vec::new();
    for (member, score) in selected {
        out.push(Value::bulk(member));
        if options.with_scores {
            out.push(Value::bulk(format_score(score)));
        }
    }
    Value::Array(Some(out))
}

fn pushx(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>], left: bool) -> Value {
    if args.len() < 2 {
        return wrong_args(if left { "lpushx" } else { "rpushx" });
    }
    let key = scoped(session, &args[0]);
    // The whole point of the X variants: a missing key is left missing, and the
    // reply is 0. Creating it would defeat the "only if it already exists"
    // contract a caller is relying on.
    match structures.load(&key) {
        Ok(None) => return Value::Integer(0),
        Err(e) => return structure_error(e),
        Ok(Some(_)) => {}
    }
    let values: Vec<Vec<u8>> = args[1..].to_vec();
    match structures.mutate(&key, Structure::empty_list, |s| {
        if left {
            s.lpush(values.clone())
        } else {
            s.rpush(values.clone())
        }
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn zrangebyscore(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 3 {
        return wrong_args("zrangebyscore");
    }
    let (Some(min), Some(max)) = (parse_f64(&args[1]), parse_f64(&args[2])) else {
        return err("ERR min or max is not a float");
    };
    let with_scores = args
        .iter()
        .skip(3)
        .any(|a| a.eq_ignore_ascii_case(b"WITHSCORES"));

    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        Ok(s.as_zset()?.range_by_score(min, max))
    }) {
        Ok(hits) => {
            let mut out = Vec::new();
            for (member, score) in hits {
                out.push(Value::bulk(member));
                if with_scores {
                    out.push(Value::bulk(format_score(score)));
                }
            }
            Value::Array(Some(out))
        }
        Err(e) => structure_error(e),
    }
}

fn zrank(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("zrank");
    }
    match with_structure(structures, &scoped(session, &args[0]), None, |s| {
        Ok(s.as_zset()?.rank(&args[1]))
    }) {
        Ok(Some(rank)) => Value::Integer(rank as i64),
        Ok(None) => Value::nil(),
        Err(e) => structure_error(e),
    }
}

/// Redis renders a score as the shortest string that round-trips, so `1` rather
/// than `1.0000000000000000` — a client that reads a score and re-sends it must
/// not drift.
pub fn format_score(score: f64) -> String {
    if score.is_infinite() {
        return if score.is_sign_positive() {
            "inf"
        } else {
            "-inf"
        }
        .to_string();
    }
    let mut text = format!("{score:.17}");
    if text.contains('.') {
        text = text.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    text
}

/// Redis index semantics shared by LRANGE and ZRANGE: negative counts back from
/// the end, out-of-range clamps.
fn normalize_range(start: i64, stop: i64, len: i64) -> (i64, i64) {
    if len == 0 {
        return (0, -1);
    }
    let from = if start < 0 {
        (len + start).max(0)
    } else {
        start
    };
    let to = if stop < 0 {
        len + stop
    } else {
        stop.min(len - 1)
    };
    (from, to)
}

// ── list commands added in the SPEC backfill ─────────────────────────────────

fn lindex(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("lindex");
    }
    let Some(index) = parse_i64(&args[1]) else {
        return err("ERR value is not an integer or out of range");
    };
    match with_structure(structures, &scoped(session, &args[0]), None, |s| {
        s.lindex(index)
    }) {
        Ok(Some(bytes)) => Value::bulk(bytes),
        // Out of range is a nil, not an error: the client asked what is there.
        Ok(None) => Value::nil(),
        Err(e) => structure_error(e),
    }
}

fn lset(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("lset");
    }
    let Some(index) = parse_i64(&args[1]) else {
        return err("ERR value is not an integer or out of range");
    };
    let key = scoped(session, &args[0]);
    // A missing key is "no such key", not an empty list that then rejects the
    // index — the two errors send a client down different paths.
    match structures.load(&key) {
        Ok(None) => return err("ERR no such key"),
        Err(e) => return structure_error(e),
        Ok(Some(_)) => {}
    }
    let value = args[2].clone();
    match structures.mutate(&key, Structure::empty_list, |s| {
        s.lset(index, value.clone())
    }) {
        Ok(_) => Value::ok(),
        Err(e) => structure_error(e),
    }
}

fn ltrim(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("ltrim");
    }
    let (Some(start), Some(stop)) = (parse_i64(&args[1]), parse_i64(&args[2])) else {
        return err("ERR value is not an integer or out of range");
    };
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_list, |s| {
        s.ltrim(start, stop)
    }) {
        // `LTRIM` on a missing key is `+OK`, matching Redis: trimming nothing
        // succeeded.
        Ok(_) => Value::ok(),
        Err(e) => structure_error(e),
    }
}

/// `LMOVE src dst LEFT|RIGHT LEFT|RIGHT` and its older spelling `RPOPLPUSH`.
///
/// Atomic across both keys: one WAL record, guarded by a compare-and-swap on
/// each. It used to be a pop followed by a push with a compensating push-back
/// on failure, which handled a rejected destination but not a process death in
/// between — and losing the element there is the one thing a client using
/// `RPOPLPUSH` for reliable delivery is paying to avoid.
fn lmove(
    structures: &Structures<'_>,
    session: &Session,
    args: &[Vec<u8>],
    rpoplpush: bool,
) -> Value {
    let (source, destination, from_left, to_left) = if rpoplpush {
        if args.len() != 2 {
            return wrong_args("rpoplpush");
        }
        (&args[0], &args[1], false, true)
    } else {
        if args.len() != 4 {
            return wrong_args("lmove");
        }
        let (Some(from_left), Some(to_left)) = (side(&args[2]), side(&args[3])) else {
            return err("ERR syntax error");
        };
        (&args[0], &args[1], from_left, to_left)
    };

    match structures.move_element(
        &scoped(session, source),
        &scoped(session, destination),
        from_left,
        to_left,
    ) {
        Ok(Some(element)) => Value::bulk(element),
        // An empty source is a nil, and nothing was moved.
        Ok(None) => Value::nil(),
        Err(e) => structure_error(e),
    }
}
/// `LEFT` → true, `RIGHT` → false, anything else → a syntax error.
fn side(raw: &[u8]) -> Option<bool> {
    if raw.eq_ignore_ascii_case(b"LEFT") {
        Some(true)
    } else if raw.eq_ignore_ascii_case(b"RIGHT") {
        Some(false)
    } else {
        None
    }
}

// ── hash commands added in the SPEC backfill ─────────────────────────────────

fn hsetnx(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("hsetnx");
    }
    let (field, value) = (args[1].clone(), args[2].clone());
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_hash, |s| {
        s.hsetnx(field.clone(), value.clone())
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

// ── set commands added in the SPEC backfill ──────────────────────────────────

fn spop(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() || args.len() > 2 {
        return wrong_args("spop");
    }
    // As with LPOP: no COUNT means a single bulk reply, COUNT means an array,
    // even for one element.
    let count = match args.get(1) {
        Some(raw) => match parse_i64(raw) {
            Some(n) if n >= 0 => Some(n as usize),
            _ => return err("ERR value is out of range, must be positive"),
        },
        None => None,
    };
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_set, |s| {
        s.spop(count.unwrap_or(1))
    }) {
        Ok(applied) => match count {
            Some(_) => Value::Array(Some(applied.value.into_iter().map(Value::bulk).collect())),
            None => match applied.value.into_iter().next() {
                Some(bytes) => Value::bulk(bytes),
                None => Value::nil(),
            },
        },
        Err(e) => structure_error(e),
    }
}

fn srandmember(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.is_empty() || args.len() > 2 {
        return wrong_args("srandmember");
    }
    let count = match args.get(1) {
        Some(raw) => match parse_i64(raw) {
            Some(n) => Some(n),
            None => return err("ERR value is not an integer or out of range"),
        },
        None => None,
    };
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        s.srandmember(count.unwrap_or(1))
    }) {
        Ok(members) => match count {
            Some(_) => Value::Array(Some(members.into_iter().map(Value::bulk).collect())),
            None => match members.into_iter().next() {
                Some(bytes) => Value::bulk(bytes),
                None => Value::nil(),
            },
        },
        Err(e) => structure_error(e),
    }
}

// ── sorted-set commands added in the SPEC backfill ───────────────────────────

fn zmscore(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 2 {
        return wrong_args("zmscore");
    }
    let members: Vec<Vec<u8>> = args[1..].to_vec();
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        let zset = s.as_zset()?;
        Ok(members.iter().map(|m| zset.score(m)).collect::<Vec<_>>())
    }) {
        Ok(scores) if scores.is_empty() => {
            // A missing key answers one nil per member, not an empty array.
            Value::Array(Some(members.iter().map(|_| Value::nil()).collect()))
        }
        Ok(scores) => Value::Array(Some(
            scores
                .into_iter()
                .map(|s| match s {
                    Some(score) => Value::bulk(format_score(score)),
                    None => Value::nil(),
                })
                .collect(),
        )),
        Err(e) => structure_error(e),
    }
}

fn zcount(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("zcount");
    }
    let (Some(min), Some(max)) = (parse_f64(&args[1]), parse_f64(&args[2])) else {
        return err("ERR min or max is not a float");
    };
    match with_structure(structures, &scoped(session, &args[0]), 0, |s| {
        Ok(s.as_zset()?.count_by_score(min, max))
    }) {
        Ok(count) => Value::Integer(count as i64),
        Err(e) => structure_error(e),
    }
}

fn zincrby(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("zincrby");
    }
    let Some(delta) = parse_f64(&args[1]) else {
        return err("ERR value is not a valid float");
    };
    let member = args[2].clone();
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        s.as_zset_mut()?.incr_by(member.clone(), delta)
    }) {
        // A bulk string, not a double: RESP2 has no float type and clients
        // parse the bulk.
        Ok(applied) => Value::bulk(format_score(applied.value)),
        Err(e) => structure_error(e),
    }
}

fn zrevrangebyscore(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() < 3 {
        return wrong_args("zrevrangebyscore");
    }
    // Note the argument order: `ZREVRANGEBYSCORE key max min`. Reversing it
    // silently returns an empty array, which reads as "no data" rather than
    // "you sent it backwards".
    let (Some(max), Some(min)) = (parse_f64(&args[1]), parse_f64(&args[2])) else {
        return err("ERR min or max is not a float");
    };
    let with_scores = args
        .iter()
        .skip(3)
        .any(|a| a.eq_ignore_ascii_case(b"WITHSCORES"));
    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        Ok(s.as_zset()?.range_by_score(min, max))
    }) {
        Ok(mut hits) => {
            hits.reverse();
            let mut out = Vec::new();
            for (member, score) in hits {
                out.push(Value::bulk(member));
                if with_scores {
                    out.push(Value::bulk(format_score(score)));
                }
            }
            Value::Array(Some(out))
        }
        Err(e) => structure_error(e),
    }
}

fn zremrangebyscore(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("zremrangebyscore");
    }
    let (Some(min), Some(max)) = (parse_f64(&args[1]), parse_f64(&args[2])) else {
        return err("ERR min or max is not a float");
    };
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        Ok(s.as_zset_mut()?.remove_range_by_score(min, max))
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn zremrangebyrank(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 3 {
        return wrong_args("zremrangebyrank");
    }
    let (Some(start), Some(stop)) = (parse_i64(&args[1]), parse_i64(&args[2])) else {
        return err("ERR value is not an integer or out of range");
    };
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        Ok(s.as_zset_mut()?.remove_range_by_rank(start, stop))
    }) {
        Ok(applied) => Value::Integer(applied.value as i64),
        Err(e) => structure_error(e),
    }
}

fn zrevrank(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    if args.len() != 2 {
        return wrong_args("zrevrank");
    }
    match with_structure(structures, &scoped(session, &args[0]), None, |s| {
        Ok(s.as_zset()?.rev_rank(&args[1]))
    }) {
        Ok(Some(rank)) => Value::Integer(rank as i64),
        Ok(None) => Value::nil(),
        Err(e) => structure_error(e),
    }
}

fn zpop(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>], min: bool) -> Value {
    let command = if min { "zpopmin" } else { "zpopmax" };
    if args.is_empty() || args.len() > 2 {
        return wrong_args(command);
    }
    let count = match args.get(1) {
        Some(raw) => match parse_i64(raw) {
            Some(n) if n >= 0 => n as usize,
            _ => return err("ERR value is out of range, must be positive"),
        },
        None => 1,
    };
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        Ok(s.as_zset_mut()?.pop(count, min))
    }) {
        // Flat `[member, score, member, score, ...]`, which is what the clients
        // parse — not an array of pairs.
        Ok(applied) => Value::Array(Some(
            applied
                .value
                .into_iter()
                .flat_map(|(member, score)| [Value::bulk(member), Value::bulk(format_score(score))])
                .collect(),
        )),
        Err(e) => structure_error(e),
    }
}

// ── HSCAN / SSCAN / ZSCAN ────────────────────────────────────────────────────

/// Parse the trailing `MATCH pattern` and `COUNT n` of a `*SCAN`.
///
/// `COUNT` is accepted and used as the page size. Redis treats it as a hint;
/// honouring it as a bound is stricter, never returns more than asked, and keeps
/// a `COUNT 10` from walking a million entries.
fn scan_options(args: &[Vec<u8>]) -> Result<(Option<Vec<u8>>, usize), Value> {
    let mut pattern = None;
    let mut count = 10usize;
    let mut i = 0;
    while i < args.len() {
        if args[i].eq_ignore_ascii_case(b"MATCH") && i + 1 < args.len() {
            pattern = Some(args[i + 1].clone());
            i += 2;
        } else if args[i].eq_ignore_ascii_case(b"COUNT") && i + 1 < args.len() {
            match parse_i64(&args[i + 1]) {
                Some(n) if n > 0 => count = n as usize,
                _ => return Err(err("ERR syntax error")),
            }
            i += 2;
        } else {
            return Err(err("ERR syntax error"));
        }
    }
    Ok((pattern, count))
}

/// One page of a structure scan.
///
/// The cursor is an index into the structure's stored order, which is a
/// `BTreeMap`/`BTreeSet` and therefore stable. That gives the guarantee clients
/// actually rely on — an element present for the whole iteration is returned at
/// least once — without pretending to be Redis's hash-bucket cursor. Recorded
/// as a divergence in `docs/RESP.md`, same as the top-level `SCAN`.
fn scan_page(
    structures: &Structures<'_>,
    session: &Session,
    args: &[Vec<u8>],
    command: &str,
    entries: impl Fn(&Structure) -> Result<Vec<(Vec<u8>, Option<Vec<u8>>)>, StructureError>,
) -> Value {
    if args.len() < 2 {
        return wrong_args(command);
    }
    let Some(cursor) = parse_i64(&args[1]).and_then(|c| (c >= 0).then_some(c as usize)) else {
        return err("ERR invalid cursor");
    };
    let (pattern, count) = match scan_options(&args[2..]) {
        Ok(parsed) => parsed,
        Err(e) => return e,
    };

    let all = match with_structure(structures, &scoped(session, &args[0]), Vec::new(), &entries) {
        Ok(all) => all,
        Err(e) => return structure_error(e),
    };

    let mut out = Vec::new();
    let mut index = cursor;
    while index < all.len() && out.len() / 2 < count {
        let (name, value) = &all[index];
        index += 1;
        // MATCH filters but still consumes the page, exactly as Redis does: a
        // page can legitimately come back empty with a non-zero cursor.
        if let Some(pattern) = &pattern {
            if !crate::resp::commands::glob_match(pattern, name) {
                continue;
            }
        }
        out.push(Value::bulk(name.clone()));
        if let Some(value) = value {
            out.push(Value::bulk(value.clone()));
        }
    }
    // Cursor 0 means "iteration finished", which is why the end must report 0
    // and not the final index.
    let next = if index >= all.len() { 0 } else { index };
    Value::Array(Some(vec![
        Value::bulk(next.to_string()),
        Value::Array(Some(out)),
    ]))
}

fn hscan(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    scan_page(structures, session, args, "hscan", |s| {
        Ok(s.as_hash()?
            .iter()
            .map(|(f, v)| (f.clone(), Some(v.clone())))
            .collect())
    })
}

fn sscan(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    scan_page(structures, session, args, "sscan", |s| {
        Ok(s.as_set()?.iter().map(|m| (m.clone(), None)).collect())
    })
}

fn zscan(structures: &Structures<'_>, session: &Session, args: &[Vec<u8>]) -> Value {
    scan_page(structures, session, args, "zscan", |s| {
        Ok(s.as_zset()?
            .range()
            .map(|(m, score)| (m.clone(), Some(format_score(score).into_bytes())))
            .collect())
    })
}

// ── ZADD modifiers and ZRANGE options ────────────────────────────────────────

/// The `NX XX GT LT CH INCR` flags of `ZADD`.
#[derive(Default, Debug, Clone, Copy)]
struct ZAddFlags {
    /// Only add new members; never update an existing score.
    nx: bool,
    /// Only update existing members; never add.
    xx: bool,
    /// Only update when the new score is greater than the current one.
    gt: bool,
    /// Only update when the new score is less than the current one.
    lt: bool,
    /// Count members *changed* rather than added.
    ch: bool,
    /// Treat the score as an increment and reply with the resulting score.
    incr: bool,
}

impl ZAddFlags {
    /// Parse leading flags, returning them and how many arguments they used.
    ///
    /// Returns an error `Value` for the combinations Redis refuses, rather than
    /// picking a winner: `NX` with `XX` is a contradiction, and `NX` with
    /// `GT`/`LT` is one too — `NX` already means "do not update", so a
    /// conditional update on top of it can only be a client bug.
    fn parse(args: &[Vec<u8>]) -> Result<(Self, usize), Value> {
        let mut flags = Self::default();
        let mut used = 0;
        for arg in args {
            let set = if arg.eq_ignore_ascii_case(b"NX") {
                &mut flags.nx
            } else if arg.eq_ignore_ascii_case(b"XX") {
                &mut flags.xx
            } else if arg.eq_ignore_ascii_case(b"GT") {
                &mut flags.gt
            } else if arg.eq_ignore_ascii_case(b"LT") {
                &mut flags.lt
            } else if arg.eq_ignore_ascii_case(b"CH") {
                &mut flags.ch
            } else if arg.eq_ignore_ascii_case(b"INCR") {
                &mut flags.incr
            } else {
                break;
            };
            *set = true;
            used += 1;
        }
        if flags.nx && flags.xx {
            return Err(err(
                "ERR XX and NX options at the same time are not compatible",
            ));
        }
        if flags.nx && (flags.gt || flags.lt) {
            return Err(err(
                "ERR GT, LT, and/or NX options at the same time are not compatible",
            ));
        }
        if flags.gt && flags.lt {
            return Err(err(
                "ERR GT, LT, and/or NX options at the same time are not compatible",
            ));
        }
        Ok((flags, used))
    }

    /// Whether a member with `current` score may be written at `proposed`.
    fn allows(&self, current: Option<f64>, proposed: f64) -> bool {
        match current {
            None => !self.xx,
            Some(current) => {
                if self.nx {
                    return false;
                }
                if self.gt && proposed <= current {
                    return false;
                }
                if self.lt && proposed >= current {
                    return false;
                }
                true
            }
        }
    }
}

/// How `ZRANGE` should interpret its two bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RangeBy {
    /// Ranks, with Redis's negative indexing. The default.
    Rank,
    /// Scores.
    Score,
    /// Member bytes, lexicographically.
    Lex,
}

/// The parsed tail of a `ZRANGE`.
struct ZRangeOptions {
    by: RangeBy,
    rev: bool,
    with_scores: bool,
    /// `LIMIT offset count`, where a negative count means "all from offset".
    limit: Option<(i64, i64)>,
}

impl ZRangeOptions {
    fn parse(args: &[Vec<u8>]) -> Result<Self, Value> {
        let mut parsed = ZRangeOptions {
            by: RangeBy::Rank,
            rev: false,
            with_scores: false,
            limit: None,
        };
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if arg.eq_ignore_ascii_case(b"BYSCORE") {
                parsed.by = RangeBy::Score;
            } else if arg.eq_ignore_ascii_case(b"BYLEX") {
                parsed.by = RangeBy::Lex;
            } else if arg.eq_ignore_ascii_case(b"REV") {
                parsed.rev = true;
            } else if arg.eq_ignore_ascii_case(b"WITHSCORES") {
                parsed.with_scores = true;
            } else if arg.eq_ignore_ascii_case(b"LIMIT") {
                let (Some(offset), Some(count)) = (
                    args.get(i + 1).and_then(|a| parse_i64(a)),
                    args.get(i + 2).and_then(|a| parse_i64(a)),
                ) else {
                    return Err(err("ERR syntax error"));
                };
                parsed.limit = Some((offset, count));
                i += 2;
            } else {
                return Err(err("ERR syntax error"));
            }
            i += 1;
        }
        if parsed.limit.is_some() && parsed.by == RangeBy::Rank {
            // Redis refuses this rather than ignoring it: with ranks the bounds
            // already are the window, so a LIMIT can only mean the caller
            // thought it was doing something else.
            return Err(err(
                "ERR syntax error, LIMIT is only supported in combination with either BYSCORE or BYLEX",
            ));
        }
        if parsed.with_scores && parsed.by == RangeBy::Lex {
            return Err(err(
                "ERR syntax error, WITHSCORES not supported in combination with BYLEX",
            ));
        }
        Ok(parsed)
    }
}

/// A `BYLEX` bound: `[member` inclusive, `(member` exclusive, `-` and `+` for
/// the extremes.
enum LexBound {
    Min,
    Max,
    Inclusive(Vec<u8>),
    Exclusive(Vec<u8>),
}

impl LexBound {
    fn parse(raw: &[u8]) -> Option<Self> {
        match raw.first()? {
            b'-' if raw.len() == 1 => Some(LexBound::Min),
            b'+' if raw.len() == 1 => Some(LexBound::Max),
            b'[' => Some(LexBound::Inclusive(raw[1..].to_vec())),
            b'(' => Some(LexBound::Exclusive(raw[1..].to_vec())),
            // A bare member is a syntax error in Redis, not an inclusive bound:
            // accepting it would silently change what a typo'd query returns.
            _ => None,
        }
    }

    fn accepts_low(&self, member: &[u8]) -> bool {
        match self {
            LexBound::Min => true,
            LexBound::Max => false,
            LexBound::Inclusive(b) => member >= b.as_slice(),
            LexBound::Exclusive(b) => member > b.as_slice(),
        }
    }

    fn accepts_high(&self, member: &[u8]) -> bool {
        match self {
            LexBound::Max => true,
            LexBound::Min => false,
            LexBound::Inclusive(b) => member <= b.as_slice(),
            LexBound::Exclusive(b) => member < b.as_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::resp::commands::{dispatch as command_dispatch, Dispatch};
    use tokio_util::sync::CancellationToken;

    fn engine() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            data_dir: Some(dir.path().to_str().unwrap().to_string()),
            ..Config::default()
        };
        (Engine::new(config, CancellationToken::new()).unwrap(), dir)
    }

    fn open() -> (Engine, tempfile::TempDir, Session) {
        let (e, d) = engine();
        (e, d, Session::new(false))
    }

    /// Run through the *real* dispatcher, so these tests also prove the
    /// structure commands are reachable from the protocol surface rather than
    /// only callable directly.
    fn run(engine: &Engine, session: &mut Session, argv: &[&str]) -> Value {
        let args: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
        match command_dispatch(
            engine,
            session,
            &args,
            &|_, _| Some(Default::default()),
            None,
            true,
        ) {
            Dispatch::Reply(value) => value,
            Dispatch::Quit => panic!("unexpected quit"),
            Dispatch::Block { .. } => panic!("unexpected block"),
            Dispatch::PubSub(_) => panic!("unexpected pubsub"),
        }
    }

    fn bulks(value: &Value) -> Vec<String> {
        match value {
            Value::Array(Some(items)) => items
                .iter()
                .map(|item| match item {
                    Value::Bulk(Some(b)) => String::from_utf8_lossy(b).to_string(),
                    Value::Bulk(None) => "<nil>".to_string(),
                    other => panic!("expected bulk, got {other:?}"),
                })
                .collect(),
            other => panic!("expected an array, got {other:?}"),
        }
    }

    // ── the nil / empty distinction ──────────────────────────────────────────

    #[test]
    fn lrange_on_a_missing_key_is_an_empty_array_not_a_null_one() {
        // Clients branch on this. Returning a null array makes a worker treat an
        // idle queue as an error condition.
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["LRANGE", "nope", "0", "-1"]),
            Value::Array(Some(Vec::new()))
        );
    }

    #[test]
    fn lpop_on_a_missing_key_is_nil_not_an_empty_bulk() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["LPOP", "nope"]), Value::nil());
        // With COUNT the reply is a *null* array for a key that is not there.
        // This asserted an empty array until the differential run against a
        // real Redis 7 showed otherwise, and it matters: a client tells "no such
        // queue" from "queue is drained" by exactly this difference.
        assert_eq!(run(&e, &mut s, &["LPOP", "nope", "2"]), Value::Array(None));

        // An empty array is in fact unreachable here, and that is worth
        // pinning: popping the last element deletes the key, so a drained list
        // is a missing list on the next call. Verified against Redis 7, which
        // does the same.
        run(&e, &mut s, &["RPUSH", "real", "x"]);
        run(&e, &mut s, &["LPOP", "real", "1"]);
        assert_eq!(run(&e, &mut s, &["EXISTS", "real"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["LPOP", "real", "2"]), Value::Array(None));
    }

    #[test]
    fn cardinality_of_a_missing_key_is_zero_not_an_error() {
        let (e, _d, mut s) = open();
        for command in ["LLEN", "HLEN", "SCARD", "ZCARD"] {
            assert_eq!(
                run(&e, &mut s, &[command, "nope"]),
                Value::Integer(0),
                "{command} of a missing key"
            );
        }
    }

    #[test]
    fn hget_and_zscore_of_a_missing_key_are_nil() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["HGET", "nope", "f"]), Value::nil());
        assert_eq!(run(&e, &mut s, &["ZSCORE", "nope", "m"]), Value::nil());
        assert_eq!(run(&e, &mut s, &["ZRANK", "nope", "m"]), Value::nil());
    }

    #[test]
    fn hmget_on_a_missing_key_returns_one_nil_per_field() {
        // Positional, like MGET: a shorter array would desynchronise a client
        // zipping the reply with its field list.
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["HMGET", "nope", "a", "b", "c"]),
            Value::Array(Some(vec![Value::nil(), Value::nil(), Value::nil()]))
        );
    }

    // ── WRONGTYPE ────────────────────────────────────────────────────────────

    #[test]
    fn operating_on_the_wrong_type_replies_wrongtype() {
        // Not a generic error: redis-py raises a distinct exception, and kombu
        // uses it to spot a fouled queue key rather than a transient failure.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["LPUSH", "mylist", "a"]);

        for argv in [
            vec!["HSET", "mylist", "f", "v"],
            vec!["SADD", "mylist", "m"],
            vec!["ZADD", "mylist", "1", "m"],
            vec!["HGETALL", "mylist"],
            vec!["SMEMBERS", "mylist"],
        ] {
            let reply = run(&e, &mut s, &argv);
            assert!(
                matches!(&reply, Value::Error(m) if m.starts_with("WRONGTYPE")),
                "{argv:?} should be WRONGTYPE, got {reply:?}"
            );
        }
        // And the list is untouched.
        assert_eq!(run(&e, &mut s, &["LLEN", "mylist"]), Value::Integer(1));
    }

    // ── lists ────────────────────────────────────────────────────────────────

    #[test]
    fn push_pop_and_range_behave_like_redis() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["RPUSH", "l", "a", "b", "c"]),
            Value::Integer(3)
        );
        // LPUSH pushes one at a time, so `x y` leaves y,x at the head.
        assert_eq!(
            run(&e, &mut s, &["LPUSH", "l", "x", "y"]),
            Value::Integer(5)
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            vec!["y", "x", "a", "b", "c"]
        );
        assert_eq!(run(&e, &mut s, &["LPOP", "l"]), Value::bulk("y"));
        assert_eq!(run(&e, &mut s, &["RPOP", "l"]), Value::bulk("c"));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            vec!["x", "a", "b"]
        );
    }

    #[test]
    fn lrem_direction_follows_the_sign() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "l", "x", "a", "x", "b", "x"]);
        assert_eq!(run(&e, &mut s, &["LREM", "l", "1", "x"]), Value::Integer(1));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            vec!["a", "x", "b", "x"]
        );
        assert_eq!(run(&e, &mut s, &["LREM", "l", "0", "x"]), Value::Integer(2));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            vec!["a", "b"]
        );
    }

    #[test]
    fn emptying_a_list_removes_the_key() {
        // Redis has no empty list, so LLEN and EXISTS must agree.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "l", "only"]);
        run(&e, &mut s, &["LPOP", "l"]);
        assert_eq!(run(&e, &mut s, &["LLEN", "l"]), Value::Integer(0));
    }

    // ── hashes ───────────────────────────────────────────────────────────────

    #[test]
    fn hash_commands_roundtrip() {
        let (e, _d, mut s) = open();
        // HSET counts fields *created*, so a second write of the same field is 0.
        assert_eq!(
            run(&e, &mut s, &["HSET", "h", "a", "1", "b", "2"]),
            Value::Integer(2)
        );
        assert_eq!(run(&e, &mut s, &["HSET", "h", "a", "9"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["HGET", "h", "a"]), Value::bulk("9"));
        assert_eq!(run(&e, &mut s, &["HEXISTS", "h", "a"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["HEXISTS", "h", "zz"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["HLEN", "h"]), Value::Integer(2));
        assert_eq!(bulks(&run(&e, &mut s, &["HKEYS", "h"])), vec!["a", "b"]);
        assert_eq!(bulks(&run(&e, &mut s, &["HVALS", "h"])), vec!["9", "2"]);
        assert_eq!(
            run(&e, &mut s, &["HDEL", "h", "a", "nope"]),
            Value::Integer(1)
        );
    }

    #[test]
    fn hgetall_is_a_flat_field_value_array() {
        // RESP2 clients parse it pairwise; a map reply would break them.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["HSET", "h", "a", "1", "b", "2"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["HGETALL", "h"])),
            vec!["a", "1", "b", "2"]
        );
    }

    #[test]
    fn hincrby_starts_at_zero_and_rejects_non_integers() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["HINCRBY", "h", "n", "5"]),
            Value::Integer(5)
        );
        assert_eq!(
            run(&e, &mut s, &["HINCRBY", "h", "n", "-2"]),
            Value::Integer(3)
        );
        run(&e, &mut s, &["HSET", "h", "word", "abc"]);
        assert!(matches!(
            run(&e, &mut s, &["HINCRBY", "h", "word", "1"]),
            Value::Error(m) if m.contains("not an integer")
        ));
    }

    // ── sets ─────────────────────────────────────────────────────────────────

    #[test]
    fn set_commands_count_actual_changes() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["SADD", "s", "a", "b", "a"]),
            Value::Integer(2)
        );
        assert_eq!(run(&e, &mut s, &["SISMEMBER", "s", "a"]), Value::Integer(1));
        assert_eq!(
            run(&e, &mut s, &["SISMEMBER", "s", "zz"]),
            Value::Integer(0)
        );
        assert_eq!(bulks(&run(&e, &mut s, &["SMEMBERS", "s"])), vec!["a", "b"]);
        assert_eq!(
            run(&e, &mut s, &["SREM", "s", "a", "nope"]),
            Value::Integer(1)
        );
        assert_eq!(run(&e, &mut s, &["SCARD", "s"]), Value::Integer(1));
    }

    // ── sorted sets ──────────────────────────────────────────────────────────

    #[test]
    fn zadd_counts_added_not_updated() {
        // The difference between a new job and a rescheduled one.
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b"]),
            Value::Integer(2)
        );
        assert_eq!(run(&e, &mut s, &["ZADD", "z", "9", "a"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("9"));
        assert_eq!(run(&e, &mut s, &["ZCARD", "z"]), Value::Integer(2));
    }

    #[test]
    fn zrange_orders_by_score_then_member() {
        // The tie order arq pages by; unstable ordering makes it skip jobs.
        let (e, _d, mut s) = open();
        run(
            &e,
            &mut s,
            &["ZADD", "z", "1", "delta", "1", "alpha", "0", "zzz"],
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1"])),
            vec!["zzz", "alpha", "delta"]
        );
    }

    #[test]
    fn zrange_withscores_interleaves_member_and_score() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1.5", "a"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1", "WITHSCORES"])),
            vec!["a", "1.5"]
        );
    }

    #[test]
    fn zrangebyscore_is_inclusive_and_accepts_infinity() {
        // arq polls for due jobs with an inclusive upper bound and often uses
        // `-inf` as the lower one.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGEBYSCORE", "z", "1", "2"])),
            vec!["a", "b"]
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGEBYSCORE", "z", "-inf", "+inf"])),
            vec!["a", "b", "c"]
        );
        assert!(bulks(&run(&e, &mut s, &["ZRANGEBYSCORE", "z", "9", "10"])).is_empty());
    }

    #[test]
    fn scores_render_without_trailing_zeros() {
        // A client that reads a score and re-sends it must not drift.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2.5", "b"]);
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("1"));
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "b"]), Value::bulk("2.5"));
    }

    #[test]
    fn zrank_is_zero_based_and_follows_removals() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(run(&e, &mut s, &["ZRANK", "z", "a"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["ZRANK", "z", "c"]), Value::Integer(2));
        run(&e, &mut s, &["ZREM", "z", "b"]);
        assert_eq!(run(&e, &mut s, &["ZRANK", "z", "c"]), Value::Integer(1));
    }

    #[test]
    fn a_nan_score_is_refused() {
        let (e, _d, mut s) = open();
        assert!(matches!(
            run(&e, &mut s, &["ZADD", "z", "nan", "m"]),
            Value::Error(m) if m.contains("not a valid float")
        ));
    }

    // ── isolation and persistence ────────────────────────────────────────────

    #[test]
    fn structures_are_isolated_per_tenant() {
        let (e, _d) = engine();
        let mut acme = Session::new(false);
        acme.tenant = Some("acme".into());
        let mut globex = Session::new(false);
        globex.tenant = Some("globex".into());

        run(&e, &mut acme, &["RPUSH", "celery", "acme-job"]);
        run(&e, &mut globex, &["RPUSH", "celery", "globex-job"]);

        assert_eq!(
            bulks(&run(&e, &mut acme, &["LRANGE", "celery", "0", "-1"])),
            vec!["acme-job"]
        );
        assert_eq!(
            bulks(&run(&e, &mut globex, &["LRANGE", "celery", "0", "-1"])),
            vec!["globex-job"]
        );
    }

    #[test]
    fn a_binary_member_survives() {
        // A pickled Celery body is not valid UTF-8.
        let (e, _d, mut s) = open();
        let payload = vec![0x80u8, 0x04, 0x00, 0xFF];
        let args = vec![b"RPUSH".to_vec(), b"q".to_vec(), payload.clone()];
        let Dispatch::Reply(_) = command_dispatch(
            &e,
            &mut s,
            &args,
            &|_, _| Some(Default::default()),
            None,
            true,
        ) else {
            panic!()
        };
        let Value::Array(Some(items)) = run(&e, &mut s, &["LRANGE", "q", "0", "-1"]) else {
            panic!()
        };
        assert_eq!(items, vec![Value::Bulk(Some(payload))]);
    }

    /// One keyspace, one type per key — in both directions.
    ///
    /// This test used to assert that `SET q v` and `RPUSH q x` "address
    /// different namespaces, so neither silently overwrites the other". The
    /// differential run against a real Redis 7 showed that is not a feature but
    /// a divergence clients cannot survive: two unrelated values under one name,
    /// with `TYPE` reporting `none` and `EXISTS` reporting 0 for the structure.
    #[test]
    fn a_name_holds_one_type_at_a_time() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "q", "plain"]);
        // A structure command on a string is a type error, not a second value.
        assert!(matches!(
            run(&e, &mut s, &["RPUSH", "q", "item"]),
            Value::Error(m) if m.starts_with("WRONGTYPE")
        ));
        assert_eq!(run(&e, &mut s, &["GET", "q"]), Value::bulk("plain"));
        assert_eq!(
            run(&e, &mut s, &["TYPE", "q"]),
            Value::Simple("string".into())
        );

        // `SET` is the one command that replaces any type, so it clears the way.
        run(&e, &mut s, &["DEL", "q"]);
        run(&e, &mut s, &["RPUSH", "q", "item"]);
        assert_eq!(
            run(&e, &mut s, &["TYPE", "q"]),
            Value::Simple("list".into())
        );
        assert_eq!(run(&e, &mut s, &["EXISTS", "q"]), Value::Integer(1));
        // And a string command on a list is the mirror-image error.
        assert!(matches!(
            run(&e, &mut s, &["GET", "q"]),
            Value::Error(m) if m.starts_with("WRONGTYPE")
        ));
        assert_eq!(run(&e, &mut s, &["SETNX", "q", "v"]), Value::Integer(0));

        // SET replaces the list outright, and the list does not come back.
        assert_eq!(run(&e, &mut s, &["SET", "q", "now-a-string"]), Value::ok());
        assert_eq!(
            run(&e, &mut s, &["TYPE", "q"]),
            Value::Simple("string".into())
        );
        assert_eq!(run(&e, &mut s, &["GET", "q"]), Value::bulk("now-a-string"));
        run(&e, &mut s, &["DEL", "q"]);
        assert_eq!(run(&e, &mut s, &["EXISTS", "q"]), Value::Integer(0));
    }

    #[test]
    fn del_removes_a_structure_and_keys_report_it_by_its_own_name() {
        // `DEL` used to leave the structure behind and report 0, so a client
        // could not delete its own data; `KEYS *` handed back the internal
        // `struct:` name.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "jobs", "a"]);
        run(&e, &mut s, &["SET", "flag", "1"]);
        let listed = bulks(&run(&e, &mut s, &["KEYS", "*"]));
        assert!(
            listed.contains(&"jobs".to_string()) && listed.contains(&"flag".to_string()),
            "both keys must be listed by the name the client used: {listed:?}"
        );
        assert!(
            !listed.iter().any(|k| k.starts_with("struct:")),
            "the storage prefix must never reach a client: {listed:?}"
        );
        assert_eq!(run(&e, &mut s, &["DEL", "jobs"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["EXISTS", "jobs"]), Value::Integer(0));
        assert_eq!(
            run(&e, &mut s, &["TYPE", "jobs"]),
            Value::Simple("none".into())
        );
    }

    #[test]
    fn expire_and_rename_reach_a_structure() {
        // All three used to see only the plain slot, so a list could not be
        // given a TTL and could not be renamed.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "q", "a"]);
        assert_eq!(run(&e, &mut s, &["EXPIRE", "q", "100"]), Value::Integer(1));
        // Rounded up: right after setting 100 seconds the answer is 100, not 99.
        assert_eq!(run(&e, &mut s, &["TTL", "q"]), Value::Integer(100));
        assert_eq!(run(&e, &mut s, &["PERSIST", "q"]), Value::Integer(1));
        assert_eq!(run(&e, &mut s, &["TTL", "q"]), Value::Integer(-1));

        assert_eq!(run(&e, &mut s, &["RENAME", "q", "moved"]), Value::ok());
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "moved", "0", "-1"])),
            ["a"]
        );
        assert_eq!(run(&e, &mut s, &["EXISTS", "q"]), Value::Integer(0));
    }

    #[test]
    fn wrong_arity_is_reported_per_command() {
        let (e, _d, mut s) = open();
        for argv in [
            vec!["LPUSH", "k"],
            vec!["HSET", "k", "onlyfield"],
            vec!["ZADD", "k", "1"],
            vec!["LRANGE", "k", "0"],
        ] {
            let reply = run(&e, &mut s, &argv);
            assert!(
                matches!(&reply, Value::Error(m) if m.contains("wrong number of arguments")),
                "{argv:?} gave {reply:?}"
            );
        }
    }

    // ── SPEC backfill: the commands that were missing ────────────────────────
    //
    // The two traps `SPEC-resp.md` names explicitly are nil-vs-empty-array and
    // `-WRONGTYPE` across types, so every command below is checked against both
    // where it can hit them.

    #[test]
    fn lindex_reads_by_position_and_from_the_end() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "l", "a", "b", "c"]);
        assert_eq!(run(&e, &mut s, &["LINDEX", "l", "0"]), Value::bulk("a"));
        assert_eq!(run(&e, &mut s, &["LINDEX", "l", "-1"]), Value::bulk("c"));
        // Out of range is a nil, not an error: the client asked what is there.
        assert_eq!(run(&e, &mut s, &["LINDEX", "l", "9"]), Value::nil());
        assert_eq!(run(&e, &mut s, &["LINDEX", "l", "-9"]), Value::nil());
        // A missing key is a nil too, not an empty bulk.
        assert_eq!(run(&e, &mut s, &["LINDEX", "gone", "0"]), Value::nil());
    }

    #[test]
    fn lset_overwrites_in_place_and_refuses_out_of_range() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "l", "a", "b"]);
        assert_eq!(run(&e, &mut s, &["LSET", "l", "1", "B"]), Value::ok());
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            ["a", "B"]
        );
        // A write out of range is an error, unlike the read: silently appending
        // would corrupt the client's idea of the list.
        assert!(matches!(
            run(&e, &mut s, &["LSET", "l", "5", "x"]),
            Value::Error(m) if m.starts_with("ERR index out of range")
        ));
        // And a missing key is "no such key", a different error a client routes
        // differently.
        assert!(matches!(
            run(&e, &mut s, &["LSET", "gone", "0", "x"]),
            Value::Error(m) if m.contains("no such key")
        ));
    }

    #[test]
    fn ltrim_keeps_only_the_requested_window() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "l", "a", "b", "c", "d", "e"]);
        assert_eq!(run(&e, &mut s, &["LTRIM", "l", "1", "3"]), Value::ok());
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            ["b", "c", "d"]
        );
        // A range that selects nothing empties the list, and an emptied list is
        // gone — LLEN 0, not a surviving empty structure.
        assert_eq!(run(&e, &mut s, &["LTRIM", "l", "5", "9"]), Value::ok());
        assert_eq!(run(&e, &mut s, &["LLEN", "l"]), Value::Integer(0));
        // Trimming a key that does not exist succeeded at trimming nothing.
        assert_eq!(run(&e, &mut s, &["LTRIM", "gone", "0", "1"]), Value::ok());
    }

    #[test]
    fn rpoplpush_moves_the_tail_to_the_head_of_the_other_list() {
        // kombu's reliable-delivery shape: take a task off the queue and put it
        // on the in-flight list in one call.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "q", "first", "second"]);
        assert_eq!(
            run(&e, &mut s, &["RPOPLPUSH", "q", "inflight"]),
            Value::bulk("second")
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "q", "0", "-1"])),
            ["first"]
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "inflight", "0", "-1"])),
            ["second"]
        );
        // An empty source moves nothing and replies nil.
        run(&e, &mut s, &["RPOPLPUSH", "q", "inflight"]);
        assert_eq!(
            run(&e, &mut s, &["RPOPLPUSH", "q", "inflight"]),
            Value::nil()
        );
    }

    #[test]
    fn rpoplpush_onto_itself_rotates() {
        // The documented Redis idiom for a round-robin worklist, and the one
        // case where source and destination are the same key.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "l", "a", "b", "c"]);
        assert_eq!(run(&e, &mut s, &["RPOPLPUSH", "l", "l"]), Value::bulk("c"));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            ["c", "a", "b"],
            "a self-rotation must not drop or duplicate the element"
        );
    }

    #[test]
    fn lmove_honours_both_sides() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "src", "a", "b", "c"]);
        assert_eq!(
            run(&e, &mut s, &["LMOVE", "src", "dst", "LEFT", "RIGHT"]),
            Value::bulk("a")
        );
        assert_eq!(
            run(&e, &mut s, &["LMOVE", "src", "dst", "RIGHT", "RIGHT"]),
            Value::bulk("c")
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "dst", "0", "-1"])),
            ["a", "c"]
        );
        assert!(matches!(
            run(&e, &mut s, &["LMOVE", "src", "dst", "SIDEWAYS", "RIGHT"]),
            Value::Error(m) if m.starts_with("ERR syntax error")
        ));
    }

    #[test]
    fn a_failed_lmove_push_puts_the_element_back() {
        // The element must not evaporate because the destination held the wrong
        // type. This is the compensating push-back, and the reason it exists.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "src", "only"]);
        run(&e, &mut s, &["SADD", "dst", "member"]);
        assert!(matches!(
            run(&e, &mut s, &["RPOPLPUSH", "src", "dst"]),
            Value::Error(m) if m.starts_with("WRONGTYPE")
        ));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "src", "0", "-1"])),
            ["only"],
            "a rejected destination must not consume the element"
        );
    }

    #[test]
    fn hsetnx_only_writes_when_the_field_is_absent() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["HSETNX", "h", "f", "1"]),
            Value::Integer(1)
        );
        assert_eq!(
            run(&e, &mut s, &["HSETNX", "h", "f", "2"]),
            Value::Integer(0)
        );
        assert_eq!(run(&e, &mut s, &["HGET", "h", "f"]), Value::bulk("1"));
    }

    #[test]
    fn spop_removes_what_it_returns() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SADD", "s", "a", "b", "c"]);
        let popped = run(&e, &mut s, &["SPOP", "s"]);
        assert!(matches!(popped, Value::Bulk(Some(_))), "got {popped:?}");
        assert_eq!(run(&e, &mut s, &["SCARD", "s"]), Value::Integer(2));
        // With COUNT the reply is an array even for one element.
        assert_eq!(bulks(&run(&e, &mut s, &["SPOP", "s", "2"])).len(), 2);
        assert_eq!(run(&e, &mut s, &["SCARD", "s"]), Value::Integer(0));
        // Popping an empty set is a nil, and with COUNT an empty array.
        assert_eq!(run(&e, &mut s, &["SPOP", "s"]), Value::nil());
        assert_eq!(
            run(&e, &mut s, &["SPOP", "s", "3"]),
            Value::Array(Some(vec![]))
        );
    }

    #[test]
    fn srandmember_leaves_the_set_alone() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SADD", "s", "a", "b"]);
        assert!(matches!(
            run(&e, &mut s, &["SRANDMEMBER", "s"]),
            Value::Bulk(Some(_))
        ));
        assert_eq!(run(&e, &mut s, &["SCARD", "s"]), Value::Integer(2));
        // A positive count is "distinct, at most that many": asking for more
        // than exist returns what exists.
        assert_eq!(bulks(&run(&e, &mut s, &["SRANDMEMBER", "s", "5"])).len(), 2);
        // A negative count allows repeats and returns exactly that many.
        assert_eq!(
            bulks(&run(&e, &mut s, &["SRANDMEMBER", "s", "-5"])).len(),
            5
        );
        assert_eq!(run(&e, &mut s, &["SCARD", "s"]), Value::Integer(2));
    }

    #[test]
    fn zmscore_answers_one_slot_per_member() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZMSCORE", "z", "a", "missing", "b"])),
            ["1", "<nil>", "2"]
        );
        // A missing key answers nils, not an empty array — a client indexes the
        // reply positionally against what it asked for.
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZMSCORE", "gone", "a", "b"])),
            ["<nil>", "<nil>"]
        );
    }

    #[test]
    fn zcount_and_zincrby() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(
            run(&e, &mut s, &["ZCOUNT", "z", "2", "3"]),
            Value::Integer(2)
        );
        assert_eq!(
            run(&e, &mut s, &["ZCOUNT", "z", "-inf", "+inf"]),
            Value::Integer(3)
        );
        assert_eq!(
            run(&e, &mut s, &["ZCOUNT", "gone", "0", "9"]),
            Value::Integer(0)
        );

        assert_eq!(
            run(&e, &mut s, &["ZINCRBY", "z", "1.5", "a"]),
            Value::bulk("2.5")
        );
        // Incrementing an absent member creates it at the delta.
        assert_eq!(
            run(&e, &mut s, &["ZINCRBY", "z", "7", "new"]),
            Value::bulk("7")
        );
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("2.5"));
    }

    #[test]
    fn zincrby_refuses_to_produce_a_nan_score() {
        // +inf plus -inf. Storing NaN would break the total ordering the whole
        // sorted set is built on, so it has to be refused rather than clamped.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "+inf", "a"]);
        // The round-trip through JSON has to survive the infinity first: this
        // used to come back WRONGTYPE because serde_json writes inf as null.
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("inf"));
        assert!(matches!(
            run(&e, &mut s, &["ZINCRBY", "z", "-inf", "a"]),
            Value::Error(m) if m.contains("not a valid float")
        ));
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("inf"));
    }

    #[test]
    fn zrevrank_counts_from_the_high_end() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(run(&e, &mut s, &["ZRANK", "z", "a"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["ZREVRANK", "z", "a"]), Value::Integer(2));
        assert_eq!(run(&e, &mut s, &["ZREVRANK", "z", "c"]), Value::Integer(0));
        assert_eq!(run(&e, &mut s, &["ZREVRANK", "z", "gone"]), Value::nil());
    }

    #[test]
    fn zrevrangebyscore_takes_max_before_min() {
        // The argument order is reversed relative to ZRANGEBYSCORE. Getting it
        // wrong returns an empty array, which reads as "no data" rather than
        // "you sent it backwards" — so it is pinned here.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZREVRANGEBYSCORE", "z", "3", "2"])),
            ["c", "b"]
        );
        assert_eq!(
            bulks(&run(
                &e,
                &mut s,
                &["ZREVRANGEBYSCORE", "z", "3", "1", "WITHSCORES"]
            )),
            ["c", "3", "b", "2", "a", "1"]
        );
    }

    #[test]
    fn zremrange_by_score_and_by_rank() {
        let (e, _d, mut s) = open();
        run(
            &e,
            &mut s,
            &["ZADD", "z", "1", "a", "2", "b", "3", "c", "4", "d"],
        );
        assert_eq!(
            run(&e, &mut s, &["ZREMRANGEBYSCORE", "z", "2", "3"]),
            Value::Integer(2)
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1"])),
            ["a", "d"]
        );

        run(&e, &mut s, &["ZADD", "z", "5", "e", "6", "f"]);
        // Rank 0..0 is the lowest-scoring member only.
        assert_eq!(
            run(&e, &mut s, &["ZREMRANGEBYRANK", "z", "0", "0"]),
            Value::Integer(1)
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1"])),
            ["d", "e", "f"]
        );
        // Negative indexing reaches the top end.
        assert_eq!(
            run(&e, &mut s, &["ZREMRANGEBYRANK", "z", "-1", "-1"]),
            Value::Integer(1)
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1"])),
            ["d", "e"]
        );
    }

    #[test]
    fn zpopmin_and_zpopmax_take_from_opposite_ends() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        // Flat [member, score], not a nested pair — which is what the clients
        // parse.
        assert_eq!(bulks(&run(&e, &mut s, &["ZPOPMIN", "z"])), ["a", "1"]);
        assert_eq!(bulks(&run(&e, &mut s, &["ZPOPMAX", "z"])), ["c", "3"]);
        assert_eq!(bulks(&run(&e, &mut s, &["ZPOPMIN", "z", "5"])), ["b", "2"]);
        // Popping an empty set is an empty array, never a nil.
        assert_eq!(
            run(&e, &mut s, &["ZPOPMIN", "z"]),
            Value::Array(Some(vec![]))
        );
    }

    #[test]
    fn a_scan_returns_every_entry_exactly_once_across_pages() {
        // The guarantee clients rely on. A page size of 1 forces the cursor to
        // be walked rather than short-circuited by everything fitting in one.
        let (e, _d, mut s) = open();
        for i in 0..7 {
            run(
                &e,
                &mut s,
                &["HSET", "h", &format!("f{i}"), &format!("v{i}")],
            );
        }
        let mut seen = Vec::new();
        let mut cursor = "0".to_string();
        let mut pages = 0;
        loop {
            let reply = run(&e, &mut s, &["HSCAN", "h", &cursor, "COUNT", "1"]);
            let (next, entries) = match &reply {
                Value::Array(Some(items)) if items.len() == 2 => {
                    let next = match &items[0] {
                        Value::Bulk(Some(b)) => String::from_utf8_lossy(b).to_string(),
                        other => panic!("expected a cursor, got {other:?}"),
                    };
                    (next, bulks(&items[1]))
                }
                other => panic!("expected [cursor, entries], got {other:?}"),
            };
            seen.extend(entries);
            pages += 1;
            assert!(pages < 50, "the cursor is not advancing");
            if next == "0" {
                break;
            }
            cursor = next;
        }
        assert!(pages > 1, "COUNT 1 must produce more than one page");
        let fields: Vec<&String> = seen.iter().step_by(2).collect();
        assert_eq!(fields.len(), 7, "every field exactly once: {seen:?}");
    }

    #[test]
    fn scan_match_filters_and_the_final_cursor_is_zero() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SADD", "s", "job:1", "job:2", "other"]);
        let reply = run(&e, &mut s, &["SSCAN", "s", "0", "MATCH", "job:*"]);
        match reply {
            Value::Array(Some(items)) => {
                assert_eq!(
                    items[0],
                    Value::bulk("0"),
                    "a finished iteration must report cursor 0"
                );
                let mut found = bulks(&items[1]);
                found.sort();
                assert_eq!(found, ["job:1", "job:2"]);
            }
            other => panic!("expected [cursor, members], got {other:?}"),
        }
    }

    #[test]
    fn zscan_carries_the_score_with_each_member() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b"]);
        let reply = run(&e, &mut s, &["ZSCAN", "z", "0"]);
        match reply {
            Value::Array(Some(items)) => {
                assert_eq!(bulks(&items[1]), ["a", "1", "b", "2"]);
            }
            other => panic!("expected [cursor, pairs], got {other:?}"),
        }
    }

    #[test]
    fn a_scan_on_a_missing_key_is_a_finished_empty_iteration() {
        // Not an error and not a non-zero cursor: a client loops until the
        // cursor is 0, and anything else would spin forever.
        let (e, _d, mut s) = open();
        let reply = run(&e, &mut s, &["HSCAN", "gone", "0"]);
        assert_eq!(
            reply,
            Value::Array(Some(vec![Value::bulk("0"), Value::Array(Some(vec![]))]))
        );
    }

    #[test]
    fn every_new_command_reports_wrongtype_across_types() {
        // The second trap the SPEC names. One structure of each kind, then every
        // new command aimed at the wrong one.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["RPUSH", "list", "x"]);
        run(&e, &mut s, &["HSET", "hash", "f", "v"]);
        run(&e, &mut s, &["SADD", "set", "m"]);
        run(&e, &mut s, &["ZADD", "zset", "1", "m"]);

        let cases: &[&[&str]] = &[
            &["LINDEX", "hash", "0"],
            &["LSET", "hash", "0", "v"],
            &["LTRIM", "hash", "0", "1"],
            &["RPOPLPUSH", "hash", "other"],
            &["LMOVE", "set", "other", "LEFT", "LEFT"],
            &["HSETNX", "list", "f", "v"],
            &["HSCAN", "list", "0"],
            &["SPOP", "list"],
            &["SRANDMEMBER", "list"],
            &["SSCAN", "list", "0"],
            &["ZMSCORE", "list", "m"],
            &["ZCOUNT", "list", "0", "1"],
            &["ZINCRBY", "list", "1", "m"],
            &["ZREVRANGEBYSCORE", "list", "1", "0"],
            &["ZREMRANGEBYSCORE", "list", "0", "1"],
            &["ZREMRANGEBYRANK", "list", "0", "1"],
            &["ZREVRANK", "list", "m"],
            &["ZPOPMIN", "list"],
            &["ZPOPMAX", "list"],
            &["ZSCAN", "list", "0"],
        ];
        for case in cases {
            let reply = run(&e, &mut s, case);
            assert!(
                matches!(&reply, Value::Error(m) if m.starts_with("WRONGTYPE")),
                "{case:?} must be WRONGTYPE, got {reply:?}"
            );
        }
    }

    // ── ZADD modifiers ───────────────────────────────────────────────────────

    #[test]
    fn zadd_nx_only_adds_and_xx_only_updates() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "NX", "1", "a"]),
            Value::Integer(1)
        );
        // NX must not overwrite: the score stays 1.
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "NX", "9", "a"]),
            Value::Integer(0)
        );
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("1"));

        // XX must not create.
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "XX", "5", "new"]),
            Value::Integer(0)
        );
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "new"]), Value::nil());
        // But it does update.
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "XX", "7", "a"]),
            Value::Integer(0)
        );
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("7"));
    }

    #[test]
    fn zadd_gt_and_lt_only_move_the_score_one_way() {
        // arq's "reschedule only if sooner" shape. Getting the comparison
        // backwards would silently push jobs into the future.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "5", "job"]);
        run(&e, &mut s, &["ZADD", "z", "GT", "3", "job"]);
        assert_eq!(
            run(&e, &mut s, &["ZSCORE", "z", "job"]),
            Value::bulk("5"),
            "GT must refuse a lower score"
        );
        run(&e, &mut s, &["ZADD", "z", "GT", "9", "job"]);
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "job"]), Value::bulk("9"));

        run(&e, &mut s, &["ZADD", "z", "LT", "11", "job"]);
        assert_eq!(
            run(&e, &mut s, &["ZSCORE", "z", "job"]),
            Value::bulk("9"),
            "LT must refuse a higher score"
        );
        run(&e, &mut s, &["ZADD", "z", "LT", "2", "job"]);
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "job"]), Value::bulk("2"));

        // GT and LT still *create* an absent member, which is the part that is
        // easy to get wrong: there is no current score to compare against.
        run(&e, &mut s, &["ZADD", "z", "GT", "4", "fresh"]);
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "fresh"]), Value::bulk("4"));
    }

    #[test]
    fn zadd_ch_counts_updates_as_well_as_additions() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["ZADD", "z", "1", "a"]), Value::Integer(1));
        // Without CH an update counts 0; with CH it counts 1.
        assert_eq!(run(&e, &mut s, &["ZADD", "z", "2", "a"]), Value::Integer(0));
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "CH", "3", "a"]),
            Value::Integer(1)
        );
        // A write that changes nothing is not a change, even with CH.
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "CH", "3", "a"]),
            Value::Integer(0)
        );
    }

    #[test]
    fn zadd_incr_behaves_like_zincrby_and_reports_a_blocked_write_as_nil() {
        let (e, _d, mut s) = open();
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "INCR", "5", "a"]),
            Value::bulk("5")
        );
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "INCR", "2.5", "a"]),
            Value::bulk("7.5")
        );
        // NX on an existing member blocks the write, and the reply is nil —
        // not 0, which would be indistinguishable from a real score of 0.
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "NX", "INCR", "1", "a"]),
            Value::nil()
        );
        assert_eq!(run(&e, &mut s, &["ZSCORE", "z", "a"]), Value::bulk("7.5"));
        // XX on an absent member likewise.
        assert_eq!(
            run(&e, &mut s, &["ZADD", "z", "XX", "INCR", "1", "gone"]),
            Value::nil()
        );
    }

    #[test]
    fn zadd_refuses_contradictory_flags() {
        let (e, _d, mut s) = open();
        for flags in [
            vec!["NX", "XX"],
            vec!["NX", "GT"],
            vec!["NX", "LT"],
            vec!["GT", "LT"],
        ] {
            let mut argv = vec!["ZADD", "z"];
            argv.extend(flags.iter().copied());
            argv.extend(["1", "m"]);
            let reply = run(&e, &mut s, &argv);
            assert!(
                matches!(&reply, Value::Error(m) if m.contains("not compatible")),
                "{flags:?} must be refused, got {reply:?}"
            );
        }
        // INCR takes exactly one pair.
        assert!(matches!(
            run(&e, &mut s, &["ZADD", "z", "INCR", "1", "a", "2", "b"]),
            Value::Error(m) if m.contains("single increment")
        ));
    }

    // ── ZRANGE options ───────────────────────────────────────────────────────

    #[test]
    fn zrange_rev_reverses_and_takes_its_bounds_reversed_too() {
        // With REV the client states `high low`. Applying the bounds as given
        // would return an empty array, which reads as "no data".
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1"])),
            ["a", "b", "c"]
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "0", "-1", "REV"])),
            ["c", "b", "a"]
        );
    }

    #[test]
    fn zrange_byscore_selects_on_score_not_rank() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "10", "a", "20", "b", "30", "c"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "20", "30", "BYSCORE"])),
            ["b", "c"]
        );
        // Rank 20..30 is empty; score 20..30 is not. This is the whole point of
        // the option.
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "20", "30"])),
            [] as [&str; 0]
        );
        assert_eq!(
            bulks(&run(
                &e,
                &mut s,
                &["ZRANGE", "z", "-inf", "+inf", "BYSCORE", "WITHSCORES"]
            )),
            ["a", "10", "b", "20", "c", "30"]
        );
        // REV with BYSCORE takes `max min`.
        assert_eq!(
            bulks(&run(
                &e,
                &mut s,
                &["ZRANGE", "z", "30", "20", "BYSCORE", "REV"]
            )),
            ["c", "b"]
        );
    }

    #[test]
    fn zrange_bylex_honours_inclusive_exclusive_and_the_extremes() {
        let (e, _d, mut s) = open();
        // All the same score, which is when BYLEX is meaningful at all.
        run(
            &e,
            &mut s,
            &["ZADD", "z", "0", "a", "0", "b", "0", "c", "0", "d"],
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "-", "+", "BYLEX"])),
            ["a", "b", "c", "d"]
        );
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "[b", "[c", "BYLEX"])),
            ["b", "c"]
        );
        // `(` excludes the bound.
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZRANGE", "z", "(b", "(d", "BYLEX"])),
            ["c"]
        );
        // A bare member is a syntax error, not an inclusive bound: accepting it
        // would silently change what a typo'd query returns.
        assert!(matches!(
            run(&e, &mut s, &["ZRANGE", "z", "b", "c", "BYLEX"]),
            Value::Error(m) if m.contains("not valid string range item")
        ));
        // WITHSCORES makes no sense here and Redis refuses it.
        assert!(matches!(
            run(&e, &mut s, &["ZRANGE", "z", "-", "+", "BYLEX", "WITHSCORES"]),
            Value::Error(m) if m.contains("WITHSCORES not supported")
        ));
    }

    #[test]
    fn zrange_limit_windows_the_result() {
        let (e, _d, mut s) = open();
        run(
            &e,
            &mut s,
            &["ZADD", "z", "1", "a", "2", "b", "3", "c", "4", "d"],
        );
        assert_eq!(
            bulks(&run(
                &e,
                &mut s,
                &["ZRANGE", "z", "-inf", "+inf", "BYSCORE", "LIMIT", "1", "2"]
            )),
            ["b", "c"]
        );
        // A negative count is "everything from the offset".
        assert_eq!(
            bulks(&run(
                &e,
                &mut s,
                &["ZRANGE", "z", "-inf", "+inf", "BYSCORE", "LIMIT", "2", "-1"]
            )),
            ["c", "d"]
        );
        // LIMIT without BYSCORE/BYLEX is refused rather than ignored: with ranks
        // the bounds already are the window.
        assert!(matches!(
            run(&e, &mut s, &["ZRANGE", "z", "0", "-1", "LIMIT", "0", "1"]),
            Value::Error(m) if m.contains("LIMIT is only supported")
        ));
    }

    #[test]
    fn zrange_rejects_an_option_it_does_not_know() {
        // A silently ignored option returns the wrong rows and looks like it
        // worked.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a"]);
        assert!(matches!(
            run(&e, &mut s, &["ZRANGE", "z", "0", "-1", "SIDEWAYS"]),
            Value::Error(m) if m.starts_with("ERR syntax error")
        ));
    }

    // ── LPUSHX / RPUSHX ──────────────────────────────────────────────────────

    #[test]
    fn pushx_does_not_create_the_key() {
        let (e, _d, mut s) = open();
        assert_eq!(run(&e, &mut s, &["LPUSHX", "gone", "v"]), Value::Integer(0));
        // And it really did not create it — a 0 with a created empty list would
        // pass the reply check but break the contract.
        assert_eq!(run(&e, &mut s, &["EXISTS", "gone"]), Value::Integer(0));

        run(&e, &mut s, &["RPUSH", "l", "a"]);
        assert_eq!(run(&e, &mut s, &["LPUSHX", "l", "z"]), Value::Integer(2));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            ["z", "a"]
        );
        assert_eq!(run(&e, &mut s, &["RPUSHX", "l", "y"]), Value::Integer(3));
        assert_eq!(
            bulks(&run(&e, &mut s, &["LRANGE", "l", "0", "-1"])),
            ["z", "a", "y"]
        );
    }

    #[test]
    fn pushx_reports_wrongtype_rather_than_zero() {
        // A 0 would say "the key is not there" when it is there and is a set.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SADD", "s", "m"]);
        assert!(matches!(
            run(&e, &mut s, &["LPUSHX", "s", "v"]),
            Value::Error(m) if m.starts_with("WRONGTYPE")
        ));
    }

    #[test]
    fn zrevrange_is_zrange_rev() {
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["ZADD", "z", "1", "a", "2", "b", "3", "c"]);
        assert_eq!(
            bulks(&run(&e, &mut s, &["ZREVRANGE", "z", "0", "-1"])),
            ["c", "b", "a"]
        );
        assert_eq!(
            bulks(&run(
                &e,
                &mut s,
                &["ZREVRANGE", "z", "0", "1", "WITHSCORES"]
            )),
            ["c", "3", "b", "2"]
        );
    }
}
