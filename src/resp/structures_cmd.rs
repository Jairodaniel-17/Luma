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
        StructureError::NotANumber => err("ERR value is not a valid float"),
        StructureError::MemberTooLong => err("ERR value is too large"),
        StructureError::TooManyEntries => err("ERR structure is at its configured entry limit"),
    }
}

fn scoped(session: &Session, key: &[u8]) -> String {
    let key = String::from_utf8_lossy(key);
    match &session.tenant {
        Some(tenant) => format!("{tenant}:{key}"),
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
        "LPOP" | "RPOP" => pop(&structures, session, args, name == "LPOP", name),
        "LLEN" => cardinality(&structures, session, args, "llen"),
        "LRANGE" => lrange(&structures, session, args),
        "LREM" => lrem(&structures, session, args),

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

        // ── sets ─────────────────────────────────────────────────────────────
        "SADD" => sadd(&structures, session, args),
        "SREM" => srem(&structures, session, args),
        "SMEMBERS" => smembers(&structures, session, args),
        "SISMEMBER" => sismember(&structures, session, args),
        "SCARD" => cardinality(&structures, session, args, "scard"),

        // ── sorted sets ──────────────────────────────────────────────────────
        "ZADD" => zadd(&structures, session, args),
        "ZREM" => zrem(&structures, session, args),
        "ZSCORE" => zscore(&structures, session, args),
        "ZCARD" => cardinality(&structures, session, args, "zcard"),
        "ZRANGE" => zrange(&structures, session, args),
        "ZRANGEBYSCORE" => zrangebyscore(&structures, session, args),
        "ZRANK" => zrank(&structures, session, args),

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

    let result = structures.mutate(&scoped(session, &args[0]), Structure::empty_list, |s| {
        if left {
            s.lpop(count.unwrap_or(1))
        } else {
            s.rpop(count.unwrap_or(1))
        }
    });

    match result {
        Ok(applied) => match count {
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
    // A missing key is 0, not an error: `LLEN` of nothing is nothing.
    match with_structure(structures, &scoped(session, &args[0]), 0, |s| Ok(s.len())) {
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
    if args.len() < 3 || !(args.len() - 1).is_multiple_of(2) {
        return wrong_args("zadd");
    }
    let mut pairs = Vec::new();
    for chunk in args[1..].chunks_exact(2) {
        let Some(score) = parse_f64(&chunk[0]) else {
            return err("ERR value is not a valid float");
        };
        pairs.push((score, chunk[1].clone()));
    }
    match structures.mutate(&scoped(session, &args[0]), Structure::empty_zset, |s| {
        let zset = s.as_zset_mut()?;
        let mut added = 0;
        for (score, member) in &pairs {
            if zset.add(member.clone(), *score)? {
                added += 1;
            }
        }
        Ok(added)
    }) {
        // Counts members *added*, not updated — which is what a client uses to
        // tell a new job from a rescheduled one.
        Ok(applied) => Value::Integer(applied.value),
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
    let (Some(start), Some(stop)) = (parse_i64(&args[1]), parse_i64(&args[2])) else {
        return err("ERR value is not an integer or out of range");
    };
    let with_scores = args
        .iter()
        .skip(3)
        .any(|a| a.eq_ignore_ascii_case(b"WITHSCORES"));

    match with_structure(structures, &scoped(session, &args[0]), Vec::new(), |s| {
        Ok(s.as_zset()?
            .range()
            .map(|(m, score)| (m.clone(), score))
            .collect::<Vec<_>>())
    }) {
        Ok(all) => {
            let len = all.len() as i64;
            let (from, to) = normalize_range(start, stop, len);
            if from > to {
                return Value::Array(Some(Vec::new()));
            }
            let mut out = Vec::new();
            for (member, score) in all
                .into_iter()
                .skip(from as usize)
                .take((to - from + 1) as usize)
            {
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
fn format_score(score: f64) -> String {
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
        match command_dispatch(engine, session, &args, |_, _| Some(None), true) {
            Dispatch::Reply(value) => value,
            Dispatch::Quit => panic!("unexpected quit"),
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
        // With COUNT the shape changes to an array, even when empty — a client
        // that asked for a count expects to iterate the reply.
        assert_eq!(
            run(&e, &mut s, &["LPOP", "nope", "2"]),
            Value::Array(Some(Vec::new()))
        );
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
        let Dispatch::Reply(_) = command_dispatch(&e, &mut s, &args, |_, _| Some(None), true)
        else {
            panic!()
        };
        let Value::Array(Some(items)) = run(&e, &mut s, &["LRANGE", "q", "0", "-1"]) else {
            panic!()
        };
        assert_eq!(items, vec![Value::Bulk(Some(payload))]);
    }

    #[test]
    fn a_structure_key_does_not_collide_with_a_plain_key() {
        // `SET q v` and `RPUSH q x` address different namespaces, so neither
        // silently overwrites the other.
        let (e, _d, mut s) = open();
        run(&e, &mut s, &["SET", "q", "plain"]);
        run(&e, &mut s, &["RPUSH", "q", "item"]);
        assert_eq!(run(&e, &mut s, &["GET", "q"]), Value::bulk("plain"));
        assert_eq!(run(&e, &mut s, &["LLEN", "q"]), Value::Integer(1));
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
}
