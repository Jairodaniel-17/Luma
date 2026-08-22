//! `EVAL` / `EVALSHA` / `SCRIPT` — for three known scripts, without a Lua
//! interpreter.
//!
//! B-R.3 of `docs/SPEC-resp.md` said Lua only if a target client's pinned
//! version demanded it insurmountably. The E2E job showed exactly that: the
//! Celery worker died on `unknown command 'EVALSHA'`. But the demand turned out
//! to be much narrower than "Lua".
//!
//! It is not Celery, and it is not arq. It is **redis-py's `Lock`**, which
//! kombu's `Mutex` uses. `Lock.acquire` is a plain `SET NX PX`; only
//! `release`, `extend` and `reacquire` use Lua, and each is a fixed six-line
//! script that does a compare-and-act on one key. Three known scripts, not a
//! language.
//!
//! So this recognises those three by their text and runs them natively. It
//! embeds no interpreter: `mlua` would mean a C toolchain, a sandbox to get
//! right, and a much larger binary — for three `if` statements.
//!
//! **What that costs.** An unrecognised script is refused with a message that
//! says so. That is the honest failure: a client using its own Lua gets a clear
//! "not supported" instead of a wrong answer. If a target client ever needs
//! arbitrary scripts, this is the seam where a real interpreter goes.
//!
//! Every script here executes by calling the ordinary command helpers, so the
//! per-organization scoping, the TTL handling and the storage semantics are the
//! same ones `GET`, `DEL`, `PTTL` and `PEXPIRE` already have. Reimplementing
//! them here is how the two would drift.

use std::collections::HashMap;

use parking_lot::Mutex;
use sha1::{Digest, Sha1};

use super::protocol::Value;

/// The scripts we can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Known {
    /// Release a lock: delete the key only if it still holds our token.
    Release,
    /// Extend a held lock's TTL, optionally adding to what is left.
    Extend,
    /// Reset a held lock's TTL to a fixed value.
    Reacquire,
}

/// `KEYS[1]` lock name, `ARGV[1]` token.
const RELEASE: &str = "local token = redis.call('get', KEYS[1]) \
                       if not token or token ~= ARGV[1] then return 0 end \
                       redis.call('del', KEYS[1]) return 1";

/// `KEYS[1]` lock, `ARGV[1]` token, `ARGV[2]` ms, `ARGV[3]` \"0\" adds to the
/// remaining ttl, \"1\" replaces it.
const EXTEND: &str = "local token = redis.call('get', KEYS[1]) \
                      if not token or token ~= ARGV[1] then return 0 end \
                      local expiration = redis.call('pttl', KEYS[1]) \
                      if not expiration then expiration = 0 end \
                      if expiration < 0 then return 0 end \
                      local newttl = ARGV[2] \
                      if ARGV[3] == \"0\" then newttl = ARGV[2] + expiration end \
                      redis.call('pexpire', KEYS[1], newttl) return 1";

/// `KEYS[1]` lock, `ARGV[1]` token, `ARGV[2]` ms.
const REACQUIRE: &str = "local token = redis.call('get', KEYS[1]) \
                         if not token or token ~= ARGV[1] then return 0 end \
                         redis.call('pexpire', KEYS[1], ARGV[2]) return 1";

/// Collapse whitespace so indentation and line breaks do not decide identity.
///
/// Matching on normalised text rather than on a SHA-1 constant: the same script
/// reaches us differently indented from different client versions, and a
/// hardcoded digest would break on a reformat that changes nothing.
fn normalise(script: &str) -> String {
    script.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Which script this is, if we know it.
pub fn recognise(script: &str) -> Option<Known> {
    let text = normalise(script);
    for (candidate, known) in [
        (RELEASE, Known::Release),
        (EXTEND, Known::Extend),
        (REACQUIRE, Known::Reacquire),
    ] {
        if text == normalise(candidate) {
            return Some(known);
        }
    }
    None
}

/// The SHA-1 Redis would report for this script.
pub fn sha1_hex(script: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(script);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Digest to script, for `EVALSHA`.
///
/// Process-wide because a script is content-addressed: it carries no tenant
/// data, and two organizations loading the same lock script mean the same
/// thing. Bounded by construction — only recognised scripts are ever stored, so
/// a client cannot grow it.
fn registry() -> &'static Mutex<HashMap<String, Known>> {
    static REGISTRY: std::sync::OnceLock<Mutex<HashMap<String, Known>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Remember a recognised script under its digest. Returns the digest.
pub fn register(script: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(script);
    let known = recognise(&text)?;
    let sha = sha1_hex(script);
    registry().lock().insert(sha.clone(), known);
    Some(sha)
}

/// Look up a previously registered digest.
pub fn lookup(sha: &str) -> Option<Known> {
    registry().lock().get(&sha.to_ascii_lowercase()).copied()
}

/// The error Redis returns for a digest it does not hold.
///
/// The exact prefix matters: redis-py matches on `NOSCRIPT` to decide whether
/// to fall back to `SCRIPT LOAD`. Any other wording turns a recoverable miss
/// into a fatal error, which is what `unknown command 'EVALSHA'` already did.
pub fn no_script() -> Value {
    Value::Error("NOSCRIPT No matching script. Please use EVAL.".to_string())
}

/// One script's plan, given its keys and arguments.
///
/// `Err` carries the reply to send: an argument mistake is the client's, and
/// answering it precisely is more useful than a generic failure.
pub struct Plan {
    pub known: Known,
    pub key: Vec<u8>,
    pub token: Vec<u8>,
    /// Milliseconds, for `Extend` and `Reacquire`.
    pub millis: i64,
    /// `Extend` only: whether the new ttl replaces what is left rather than
    /// adding to it.
    pub replace_ttl: bool,
}

/// Check the arguments a script was called with.
pub fn plan(known: Known, keys: &[Vec<u8>], argv: &[Vec<u8>]) -> Result<Plan, Value> {
    let wrong = |what: &str| Err(Value::Error(format!("ERR {what}")));
    if keys.len() != 1 {
        return wrong("this script takes exactly one key");
    }
    let token = match argv.first() {
        Some(t) => t.clone(),
        None => return wrong("this script needs a token argument"),
    };

    let millis_at = |i: usize| -> Result<i64, Value> {
        argv.get(i)
            .and_then(|v| String::from_utf8_lossy(v).parse::<i64>().ok())
            .ok_or_else(|| Value::Error("ERR value is not an integer or out of range".into()))
    };

    let (millis, replace_ttl) = match known {
        Known::Release => (0, false),
        Known::Reacquire => (millis_at(1)?, true),
        Known::Extend => {
            let millis = millis_at(1)?;
            // "0" adds to what is left, "1" replaces. Reading it the wrong way
            // round makes a lock live for roughly twice as long as asked, which
            // nothing reports and a duplicate job eventually reveals.
            let replace = argv.get(2).map(|v| v.as_slice()) == Some(b"1");
            (millis, replace)
        }
    };

    Ok(Plan {
        known,
        key: keys[0].clone(),
        token,
        millis,
        replace_ttl,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim from redis-py's `redis/lock.py`, indentation included. If a
    /// future version reformats them, this is what fails — loudly, here —
    /// rather than a lock quietly never releasing in production.
    const REDIS_PY_RELEASE: &str = "
        local token = redis.call('get', KEYS[1])
        if not token or token ~= ARGV[1] then
            return 0
        end
        redis.call('del', KEYS[1])
        return 1
    ";

    const REDIS_PY_EXTEND: &str = "
        local token = redis.call('get', KEYS[1])
        if not token or token ~= ARGV[1] then
            return 0
        end
        local expiration = redis.call('pttl', KEYS[1])
        if not expiration then
            expiration = 0
        end
        if expiration < 0 then
            return 0
        end

        local newttl = ARGV[2]
        if ARGV[3] == \"0\" then
            newttl = ARGV[2] + expiration
        end
        redis.call('pexpire', KEYS[1], newttl)
        return 1
    ";

    const REDIS_PY_REACQUIRE: &str = "
        local token = redis.call('get', KEYS[1])
        if not token or token ~= ARGV[1] then
            return 0
        end
        redis.call('pexpire', KEYS[1], ARGV[2])
        return 1
    ";

    #[test]
    fn the_scripts_redis_py_actually_sends_are_recognised() {
        assert_eq!(recognise(REDIS_PY_RELEASE), Some(Known::Release));
        assert_eq!(recognise(REDIS_PY_EXTEND), Some(Known::Extend));
        assert_eq!(recognise(REDIS_PY_REACQUIRE), Some(Known::Reacquire));
    }

    #[test]
    fn indentation_does_not_decide_identity() {
        // The reason this matches on normalised text and not on a hardcoded
        // digest: the same script arrives formatted differently from different
        // client versions, and a reformat that changes nothing would otherwise
        // break every lock release.
        let squashed = "local token = redis.call('get', KEYS[1]) if not token or token ~= ARGV[1] \
                        then return 0 end redis.call('del', KEYS[1]) return 1";
        assert_eq!(recognise(squashed), Some(Known::Release));
    }

    #[test]
    fn an_unknown_script_is_not_guessed_at() {
        // Silently treating an unrecognised script as one of ours would run the
        // wrong operation on the client's keys. Refusing is the only safe
        // answer.
        assert_eq!(recognise("return redis.call('flushall')"), None);
        assert_eq!(recognise(""), None);
        // And one that only looks like ours.
        assert_eq!(
            recognise("local token = redis.call('get', KEYS[1]) return 1"),
            None
        );
    }

    #[test]
    fn the_digest_is_the_one_redis_would_report() {
        // Known SHA-1 answers, so a broken hash cannot pass by agreeing with
        // itself.
        assert_eq!(sha1_hex(b""), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn a_registered_script_is_found_by_its_digest() {
        let sha = register(REDIS_PY_RELEASE.as_bytes()).expect("recognised");
        assert_eq!(sha, sha1_hex(REDIS_PY_RELEASE.as_bytes()));
        assert_eq!(lookup(&sha), Some(Known::Release));
        // Digests are hex, and clients may send them uppercase.
        assert_eq!(lookup(&sha.to_uppercase()), Some(Known::Release));
        assert_eq!(lookup("0000000000000000000000000000000000000000"), None);
    }

    #[test]
    fn an_unknown_script_is_never_registered() {
        // The registry only ever holds scripts we can run, which is also what
        // keeps a client from growing it without bound.
        assert_eq!(register(b"return 1"), None);
        assert_eq!(lookup(&sha1_hex(b"return 1")), None);
    }

    #[test]
    fn extend_reads_the_replace_flag_the_right_way_round() {
        // "0" adds to what is left, "1" replaces it. Backwards, a lock lives
        // for about twice as long as asked — nothing reports it, and it
        // surfaces as a duplicate job much later.
        let keys = vec![b"lock".to_vec()];
        let add = plan(
            Known::Extend,
            &keys,
            &[b"tok".to_vec(), b"1000".to_vec(), b"0".to_vec()],
        )
        .unwrap_or_else(|_| panic!("valid arguments"));
        assert!(!add.replace_ttl);
        assert_eq!(add.millis, 1000);

        let replace = plan(
            Known::Extend,
            &keys,
            &[b"tok".to_vec(), b"1000".to_vec(), b"1".to_vec()],
        )
        .unwrap_or_else(|_| panic!("valid arguments"));
        assert!(replace.replace_ttl);
    }

    #[test]
    fn bad_arguments_are_refused_rather_than_defaulted() {
        let keys = vec![b"lock".to_vec()];
        assert!(plan(Known::Release, &[], &[b"tok".to_vec()]).is_err());
        assert!(plan(Known::Release, &keys, &[]).is_err());
        // A non-numeric ttl must not become zero, which would expire the lock
        // immediately and let a second worker take it.
        assert!(plan(
            Known::Reacquire,
            &keys,
            &[b"tok".to_vec(), b"soon".to_vec()]
        )
        .is_err());
        assert!(plan(Known::Reacquire, &keys, &[b"tok".to_vec()]).is_err());
    }

    #[test]
    fn the_noscript_prefix_is_the_one_clients_match_on() {
        // redis-py matches on this token to decide whether to fall back to
        // SCRIPT LOAD. Any other wording turns a recoverable miss into the
        // fatal error we started with.
        let Value::Error(text) = no_script() else {
            panic!("an error");
        };
        assert!(text.starts_with("NOSCRIPT "), "{text}");
    }
}
