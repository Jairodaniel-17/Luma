//! Redis-shaped data structures: list, hash, set and sorted set.
//!
//! F0.2 of `docs/PLAN-MAESTRO.md`. These are what the RESP command surface in
//! blocks 5 and 6 is built on, so the semantics are pinned here — by tests —
//! before any protocol code depends on them.
//!
//! ## Where they live, and the trade-off that implies
//!
//! A structure is stored as a value in the existing key-value store, under a
//! namespaced key. That is a deliberate choice: durability, WAL append, crash
//! replay, per-key TTL and the revision counter that `WATCH` needs all already
//! work there and are covered by the crash-recovery matrix. Inventing a parallel
//! persistence path would mean re-earning every one of those guarantees.
//!
//! The cost is that a mutation is a read-modify-write of the whole structure, so
//! one push into an *n*-element list is O(n). [`MAX_STRUCTURE_ENTRIES`] is what
//! keeps that bounded, and it is why the product note in `SPEC-resp.md` says
//! Luma is not for queues of ten million resident messages. The upgrade path,
//! if a profile ever demands it, is incremental WAL records per operation —
//! which is a change to how these are *stored*, not to the semantics below.
//!
//! ## Semantics that are contract, not implementation detail
//!
//! - **One key, one type.** Operating on the wrong type is
//!   [`StructureError::WrongType`], which the protocol layer renders as
//!   `-WRONGTYPE`. Clients match on that string.
//! - **Sorted set ordering.** Members sort by score, and ties break
//!   **lexicographically by member**. Redis clients rely on it — `arq` pages
//!   through a zset by score — so it is tested rather than assumed.
//! - **Members are bytes.** Not strings: Redis members are binary, and a
//!   pickled Celery payload is not valid UTF-8.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Maximum entries in one structure. Bounds both memory per key and the cost of
/// the read-modify-write described above.
pub const MAX_STRUCTURE_ENTRIES: usize = 1_000_000;

/// Maximum length of a single member or field, in bytes.
pub const MAX_MEMBER_LEN: usize = 512 * 1024;

/// Bytes, carried through JSON as base64 so a non-UTF-8 member survives the
/// JSON-lines WAL.
pub type Bytes = Vec<u8>;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StructureError {
    /// The key holds a different type. Rendered as `-WRONGTYPE`.
    #[error("WRONGTYPE Operation against a key holding the wrong kind of value")]
    WrongType,
    #[error("structure would exceed {MAX_STRUCTURE_ENTRIES} entries")]
    TooManyEntries,
    #[error("member exceeds {MAX_MEMBER_LEN} bytes")]
    MemberTooLong,
    /// A score that is NaN. Redis refuses it, and allowing it would make the
    /// ordered index inconsistent because NaN compares false with everything.
    #[error("score is not a number")]
    NotANumber,
    #[error("value is not an integer or out of range")]
    NotAnInteger,
    /// A hash field that `HINCRBY` cannot parse. Redis words this one
    /// differently from the generic case, and clients surface the message.
    #[error("hash value is not an integer")]
    HashNotAnInteger,
    /// `LSET` past the end. A read out of range is a nil, but a *write* out of
    /// range is an error: the client asked to overwrite something that is not
    /// there, and silently appending would corrupt its idea of the list.
    #[error("index out of range")]
    IndexOutOfRange,
    /// The stored value is a structure that will not deserialize.
    ///
    /// Distinct from [`StructureError::WrongType`] on purpose. Reporting this as
    /// `WRONGTYPE` says "you used the wrong command", when the truth is "what is
    /// stored cannot be read" — and that mislabel is what hid the infinite-score
    /// bug: the symptom looked like a client mistake.
    #[error("stored structure could not be read: {0}")]
    Corrupt(String),
}

/// Serde helpers that carry byte strings through JSON as base64.
///
/// Two reasons this is needed rather than nice to have:
///
/// 1. **JSON object keys must be strings.** A `BTreeMap<Vec<u8>, _>` simply
///    cannot be serialized as a JSON object, so the byte-keyed maps are written
///    as sequences of `[key, value]` pairs.
/// 2. **Compactness in the WAL.** serde's default for `Vec<u8>` is an array of
///    integers, which spends four to five characters per byte. These structures
///    ride the JSON-lines WAL on every mutation, so base64 is roughly a
///    threefold saving on every record.
mod b64 {
    use super::{Bytes, OrderedScore};
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    fn encode(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn decode<E: serde::de::Error>(text: &str) -> Result<Bytes, E> {
        base64::engine::general_purpose::STANDARD
            .decode(text)
            .map_err(|e| E::custom(format!("member is not base64: {e}")))
    }

    pub mod deque {
        use super::*;

        pub fn serialize<S: Serializer>(
            value: &VecDeque<Bytes>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|item| encode(item))
                .collect::<Vec<_>>()
                .serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<VecDeque<Bytes>, D::Error> {
            let raw = Vec::<String>::deserialize(deserializer)?;
            raw.iter()
                .map(|text| decode::<D::Error>(text))
                .collect::<Result<VecDeque<_>, _>>()
        }
    }

    pub mod set {
        use super::*;

        pub fn serialize<S: Serializer>(
            value: &BTreeSet<Bytes>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|item| encode(item))
                .collect::<Vec<_>>()
                .serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<BTreeSet<Bytes>, D::Error> {
            let raw = Vec::<String>::deserialize(deserializer)?;
            raw.iter()
                .map(|text| decode::<D::Error>(text))
                .collect::<Result<BTreeSet<_>, _>>()
        }
    }

    pub mod bytes_map {
        use super::*;

        pub fn serialize<S: Serializer>(
            value: &BTreeMap<Bytes, Bytes>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|(k, v)| (encode(k), encode(v)))
                .collect::<Vec<_>>()
                .serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<BTreeMap<Bytes, Bytes>, D::Error> {
            let raw = Vec::<(String, String)>::deserialize(deserializer)?;
            raw.iter()
                .map(|(k, v)| Ok((decode::<D::Error>(k)?, decode::<D::Error>(v)?)))
                .collect()
        }
    }

    pub mod score_map {
        use super::*;

        // Both directions go through `OrderedScore`'s own impls rather than a
        // raw `f64`. Using the number directly is what let an infinite score be
        // written as JSON `null` and make the whole sorted set unreadable: the
        // type that knows NaN is illegal is also the type that knows how to
        // survive a format with no infinity.
        pub fn serialize<S: Serializer>(
            value: &BTreeMap<Bytes, OrderedScore>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|(k, v)| (encode(k), *v))
                .collect::<Vec<_>>()
                .serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<BTreeMap<Bytes, OrderedScore>, D::Error> {
            let raw = Vec::<(String, OrderedScore)>::deserialize(deserializer)?;
            raw.into_iter()
                .map(|(k, score)| Ok((decode::<D::Error>(&k)?, score)))
                .collect()
        }
    }

    pub mod ordered_set {
        use super::*;

        pub fn serialize<S: Serializer>(
            value: &BTreeSet<(OrderedScore, Bytes)>,
            serializer: S,
        ) -> Result<S::Ok, S::Error> {
            value
                .iter()
                .map(|(score, member)| (*score, encode(member)))
                .collect::<Vec<_>>()
                .serialize(serializer)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(
            deserializer: D,
        ) -> Result<BTreeSet<(OrderedScore, Bytes)>, D::Error> {
            let raw = Vec::<(OrderedScore, String)>::deserialize(deserializer)?;
            raw.into_iter()
                .map(|(score, member)| Ok((score, decode::<D::Error>(&member)?)))
                .collect()
        }
    }
}

/// A sorted set: members with float scores, ordered by (score, member).
///
/// Two indexes are kept in step — a map for O(1) score lookup and an ordered set
/// for range queries. Scores are held as `OrderedScore` so they can live in a
/// `BTreeSet` at all; `f64` is not `Ord` because of NaN, which is exactly why
/// NaN is rejected on the way in.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ZSet {
    #[serde(with = "b64::score_map")]
    scores: BTreeMap<Bytes, OrderedScore>,
    #[serde(with = "b64::ordered_set")]
    ordered: BTreeSet<(OrderedScore, Bytes)>,
}

/// A score that is totally ordered. Constructed only through
/// [`OrderedScore::new`], which rejects NaN.
///
/// ## Why this has a hand-written `Serialize`
///
/// JSON has no infinity, and `serde_json` writes `f64::INFINITY` as `null`.
/// With the derived impl, `ZADD key +inf member` reported success and the whole
/// sorted set became unreadable on the next command — the deserialize failed and
/// the load path reported it as `WRONGTYPE`, which sent the reader looking for a
/// type mixup that was not there. `+inf` is an ordinary Redis idiom, so silently
/// destroying those keys is not an option.
///
/// Finite scores are still written as a bare JSON number, byte-identical to the
/// derived form, so structures already on disk are unaffected. Only the two
/// infinities are written as strings, and both spellings are accepted on the way
/// back in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrderedScore(f64);

impl Serialize for OrderedScore {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.0.is_finite() {
            return serializer.serialize_f64(self.0);
        }
        serializer.serialize_str(if self.0 > 0.0 { "inf" } else { "-inf" })
    }
}

impl<'de> Deserialize<'de> for OrderedScore {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        // An untagged value: a number from the common path, a string for an
        // infinity, and `null` from data written before this was fixed — those
        // scores are unrecoverable, but reading them as an infinity at least
        // keeps the rest of the structure loadable instead of failing the whole
        // key.
        let raw = serde_json::Value::deserialize(deserializer)?;
        let score = match &raw {
            serde_json::Value::Number(n) => n.as_f64().ok_or_else(|| {
                D::Error::custom(format!("score {n} is not representable as f64"))
            })?,
            serde_json::Value::String(text) => match text.as_str() {
                "inf" | "+inf" | "Infinity" => f64::INFINITY,
                "-inf" | "-Infinity" => f64::NEG_INFINITY,
                other => other
                    .parse::<f64>()
                    .map_err(|e| D::Error::custom(format!("score {other:?}: {e}")))?,
            },
            serde_json::Value::Null => f64::INFINITY,
            other => return Err(D::Error::custom(format!("score is not a number: {other}"))),
        };
        if score.is_nan() {
            return Err(D::Error::custom("score is NaN"));
        }
        Ok(OrderedScore(score))
    }
}

impl OrderedScore {
    pub fn new(score: f64) -> Result<Self, StructureError> {
        if score.is_nan() {
            return Err(StructureError::NotANumber);
        }
        Ok(OrderedScore(score))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl Eq for OrderedScore {}

impl Ord for OrderedScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Safe because NaN never gets in; `total_cmp` also gives -0.0 < 0.0 a
        // definite answer instead of leaving it to chance.
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ZSet {
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    pub fn score(&self, member: &[u8]) -> Option<f64> {
        self.scores.get(member).map(|s| s.get())
    }

    /// Insert or update a member. Returns true when the member is new, which is
    /// what `ZADD` reports.
    pub fn add(&mut self, member: Bytes, score: f64) -> Result<bool, StructureError> {
        check_member(&member)?;
        let score = OrderedScore::new(score)?;
        match self.scores.insert(member.clone(), score) {
            Some(previous) => {
                // Remove the old ordered entry, not just insert the new one:
                // leaving it would make the member appear twice in a range scan.
                self.ordered.remove(&(previous, member.clone()));
                self.ordered.insert((score, member));
                Ok(false)
            }
            None => {
                if self.scores.len() > MAX_STRUCTURE_ENTRIES {
                    self.scores.remove(&member);
                    return Err(StructureError::TooManyEntries);
                }
                self.ordered.insert((score, member));
                Ok(true)
            }
        }
    }

    /// Remove a member. Returns true when it was present.
    pub fn remove(&mut self, member: &[u8]) -> bool {
        match self.scores.remove(member) {
            Some(score) => {
                self.ordered.remove(&(score, member.to_vec()));
                true
            }
            None => false,
        }
    }

    /// Members in rank order, lowest score first, ties by member bytes.
    pub fn range(&self) -> impl Iterator<Item = (&Bytes, f64)> {
        self.ordered
            .iter()
            .map(|(score, member)| (member, score.get()))
    }

    /// Members whose score is within `[min, max]`, inclusive — the shape `arq`
    /// uses to poll for due jobs.
    pub fn range_by_score(&self, min: f64, max: f64) -> Vec<(Bytes, f64)> {
        self.ordered
            .iter()
            .filter(|(score, _)| score.get() >= min && score.get() <= max)
            .map(|(score, member)| (member.clone(), score.get()))
            .collect()
    }

    /// Zero-based rank of a member in ascending order.
    pub fn rank(&self, member: &[u8]) -> Option<usize> {
        let score = *self.scores.get(member)?;
        Some(self.ordered.range(..(score, member.to_vec())).count())
    }

    /// Zero-based rank in *descending* order, which is what `ZREVRANK` reports.
    pub fn rev_rank(&self, member: &[u8]) -> Option<usize> {
        let ascending = self.rank(member)?;
        Some(self.len() - 1 - ascending)
    }

    /// How many members score within `[min, max]`, inclusive.
    pub fn count_by_score(&self, min: f64, max: f64) -> usize {
        self.ordered
            .iter()
            .filter(|(score, _)| score.get() >= min && score.get() <= max)
            .count()
    }

    /// Add `delta` to a member's score, creating it at `delta` if absent.
    ///
    /// Returns the new score. A resulting NaN is an error rather than a stored
    /// value: `+inf` plus `-inf` is the case Redis rejects too, and a NaN score
    /// would break the total ordering the whole structure depends on.
    pub fn incr_by(&mut self, member: Bytes, delta: f64) -> Result<f64, StructureError> {
        let current = self.score(&member).unwrap_or(0.0);
        let next = current + delta;
        if next.is_nan() {
            return Err(StructureError::NotANumber);
        }
        self.add(member, next)?;
        Ok(next)
    }

    /// Remove every member scoring within `[min, max]`. Returns how many went.
    pub fn remove_range_by_score(&mut self, min: f64, max: f64) -> usize {
        let doomed: Vec<Bytes> = self
            .ordered
            .iter()
            .filter(|(score, _)| score.get() >= min && score.get() <= max)
            .map(|(_, member)| member.clone())
            .collect();
        doomed.iter().filter(|m| self.remove(m)).count()
    }

    /// Remove members by ascending rank range, inclusive, with Redis's negative
    /// indexing. Returns how many went.
    pub fn remove_range_by_rank(&mut self, start: i64, stop: i64) -> usize {
        let len = self.len() as i64;
        if len == 0 {
            return 0;
        }
        let start = normalize_index(start, len).max(0);
        let stop = normalize_index(stop, len).min(len - 1);
        if start > stop {
            return 0;
        }
        let doomed: Vec<Bytes> = self
            .ordered
            .iter()
            .skip(start as usize)
            .take((stop - start + 1) as usize)
            .map(|(_, member)| member.clone())
            .collect();
        doomed.iter().filter(|m| self.remove(m)).count()
    }

    /// Pop up to `count` members from the low-score end (`ZPOPMIN`) or the
    /// high-score end (`ZPOPMAX`).
    pub fn pop(&mut self, count: usize, min: bool) -> Vec<(Bytes, f64)> {
        let taken: Vec<(Bytes, f64)> = if min {
            self.ordered
                .iter()
                .take(count)
                .map(|(s, m)| (m.clone(), s.get()))
                .collect()
        } else {
            self.ordered
                .iter()
                .rev()
                .take(count)
                .map(|(s, m)| (m.clone(), s.get()))
                .collect()
        };
        for (member, _) in &taken {
            self.remove(member);
        }
        taken
    }
}

/// The value stored under a structure key.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Structure {
    List {
        #[serde(with = "b64::deque")]
        items: VecDeque<Bytes>,
    },
    Hash {
        #[serde(with = "b64::bytes_map")]
        fields: BTreeMap<Bytes, Bytes>,
    },
    Set {
        #[serde(with = "b64::set")]
        members: BTreeSet<Bytes>,
    },
    ZSet {
        zset: ZSet,
    },
}

impl Structure {
    pub fn empty_list() -> Self {
        Structure::List {
            items: VecDeque::new(),
        }
    }
    pub fn empty_hash() -> Self {
        Structure::Hash {
            fields: BTreeMap::new(),
        }
    }
    pub fn empty_set() -> Self {
        Structure::Set {
            members: BTreeSet::new(),
        }
    }
    pub fn empty_zset() -> Self {
        Structure::ZSet {
            zset: ZSet::default(),
        }
    }

    /// The Redis type name, as `TYPE` reports it.
    pub fn type_name(&self) -> &'static str {
        match self {
            Structure::List { .. } => "list",
            Structure::Hash { .. } => "hash",
            Structure::Set { .. } => "set",
            Structure::ZSet { .. } => "zset",
        }
    }

    /// Element count, as `LLEN` / `HLEN` / `SCARD` / `ZCARD` report it.
    pub fn len(&self) -> usize {
        match self {
            Structure::List { items } => items.len(),
            Structure::Hash { fields } => fields.len(),
            Structure::Set { members } => members.len(),
            Structure::ZSet { zset } => zset.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_list_mut(&mut self) -> Result<&mut VecDeque<Bytes>, StructureError> {
        match self {
            Structure::List { items } => Ok(items),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_list(&self) -> Result<&VecDeque<Bytes>, StructureError> {
        match self {
            Structure::List { items } => Ok(items),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_hash_mut(&mut self) -> Result<&mut BTreeMap<Bytes, Bytes>, StructureError> {
        match self {
            Structure::Hash { fields } => Ok(fields),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_hash(&self) -> Result<&BTreeMap<Bytes, Bytes>, StructureError> {
        match self {
            Structure::Hash { fields } => Ok(fields),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_set_mut(&mut self) -> Result<&mut BTreeSet<Bytes>, StructureError> {
        match self {
            Structure::Set { members } => Ok(members),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_set(&self) -> Result<&BTreeSet<Bytes>, StructureError> {
        match self {
            Structure::Set { members } => Ok(members),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_zset_mut(&mut self) -> Result<&mut ZSet, StructureError> {
        match self {
            Structure::ZSet { zset } => Ok(zset),
            _ => Err(StructureError::WrongType),
        }
    }

    pub fn as_zset(&self) -> Result<&ZSet, StructureError> {
        match self {
            Structure::ZSet { zset } => Ok(zset),
            _ => Err(StructureError::WrongType),
        }
    }

    // ── list operations ──────────────────────────────────────────────────────

    /// `LPUSH`: prepend, returning the new length.
    pub fn lpush(&mut self, values: Vec<Bytes>) -> Result<usize, StructureError> {
        let items = self.as_list_mut()?;
        check_growth(items.len(), values.len())?;
        for value in values {
            check_member(&value)?;
            items.push_front(value);
        }
        Ok(items.len())
    }

    /// `RPUSH`: append, returning the new length.
    pub fn rpush(&mut self, values: Vec<Bytes>) -> Result<usize, StructureError> {
        let items = self.as_list_mut()?;
        check_growth(items.len(), values.len())?;
        for value in values {
            check_member(&value)?;
            items.push_back(value);
        }
        Ok(items.len())
    }

    /// `LPOP count`.
    pub fn lpop(&mut self, count: usize) -> Result<Vec<Bytes>, StructureError> {
        let items = self.as_list_mut()?;
        Ok((0..count).filter_map(|_| items.pop_front()).collect())
    }

    /// `RPOP count`.
    pub fn rpop(&mut self, count: usize) -> Result<Vec<Bytes>, StructureError> {
        let items = self.as_list_mut()?;
        Ok((0..count).filter_map(|_| items.pop_back()).collect())
    }

    /// `LRANGE start stop`, inclusive, with Redis's negative indexing where -1
    /// is the last element. Out-of-range bounds clamp instead of erroring.
    pub fn lrange(&self, start: i64, stop: i64) -> Result<Vec<Bytes>, StructureError> {
        let items = self.as_list()?;
        let len = items.len() as i64;
        if len == 0 {
            return Ok(Vec::new());
        }
        let start = normalize_index(start, len).max(0);
        let stop = normalize_index(stop, len).min(len - 1);
        if start > stop {
            return Ok(Vec::new());
        }
        Ok(items
            .iter()
            .skip(start as usize)
            .take((stop - start + 1) as usize)
            .cloned()
            .collect())
    }

    /// `LREM count value`. `count > 0` removes from the head, `count < 0` from
    /// the tail, `0` removes every occurrence. Returns how many were removed.
    pub fn lrem(&mut self, count: i64, value: &[u8]) -> Result<usize, StructureError> {
        let items = self.as_list_mut()?;
        let limit = if count == 0 {
            usize::MAX
        } else {
            count.unsigned_abs() as usize
        };
        let mut removed = 0;
        if count >= 0 {
            let mut kept = VecDeque::with_capacity(items.len());
            for item in items.drain(..) {
                if removed < limit && item == value {
                    removed += 1;
                } else {
                    kept.push_back(item);
                }
            }
            *items = kept;
        } else {
            let mut kept = VecDeque::with_capacity(items.len());
            // Walk from the tail so `count < 0` removes the *last* occurrences.
            for item in items.drain(..).rev() {
                if removed < limit && item == value {
                    removed += 1;
                } else {
                    kept.push_front(item);
                }
            }
            *items = kept;
        }
        Ok(removed)
    }

    /// `LINDEX index`, with Redis's negative indexing. Out of range is a nil,
    /// not an error.
    pub fn lindex(&self, index: i64) -> Result<Option<Bytes>, StructureError> {
        let items = self.as_list()?;
        let len = items.len() as i64;
        let at = normalize_index(index, len);
        if at < 0 || at >= len {
            return Ok(None);
        }
        Ok(items.get(at as usize).cloned())
    }

    /// `LSET index value`. Out of range is an error, unlike `LINDEX` — the
    /// client asked to write somewhere that does not exist.
    pub fn lset(&mut self, index: i64, value: Bytes) -> Result<(), StructureError> {
        let items = self.as_list_mut()?;
        let len = items.len() as i64;
        let at = normalize_index(index, len);
        if at < 0 || at >= len {
            return Err(StructureError::IndexOutOfRange);
        }
        items[at as usize] = value;
        Ok(())
    }

    /// `LTRIM start stop`: keep only that inclusive range, dropping the rest.
    ///
    /// An empty result is not an error — Redis deletes the key, and the caller
    /// sees an empty list, which `Structures::mutate` prunes the same way it
    /// prunes a list emptied by `LPOP`.
    pub fn ltrim(&mut self, start: i64, stop: i64) -> Result<(), StructureError> {
        let kept = self.lrange(start, stop)?;
        let items = self.as_list_mut()?;
        *items = kept.into();
        Ok(())
    }

    /// `HSETNX field value`: set only when the field is absent. Returns true
    /// when it was written.
    pub fn hsetnx(&mut self, field: Bytes, value: Bytes) -> Result<bool, StructureError> {
        let map = self.as_hash_mut()?;
        if map.contains_key(&field) {
            return Ok(false);
        }
        if map.len() >= MAX_STRUCTURE_ENTRIES {
            return Err(StructureError::TooManyEntries);
        }
        map.insert(field, value);
        Ok(true)
    }

    /// `SPOP count`: remove and return up to `count` members.
    ///
    /// Redis picks at random; this takes them in the set's stored order. See
    /// `docs/integrar/RESP.md` — the divergence is deliberate and documented, because
    /// every real use of `SPOP` is "give me any member" and a deterministic
    /// answer is testable.
    pub fn spop(&mut self, count: usize) -> Result<Vec<Bytes>, StructureError> {
        let members = self.as_set_mut()?;
        let taken: Vec<Bytes> = members.iter().take(count).cloned().collect();
        for member in &taken {
            members.remove(member);
        }
        Ok(taken)
    }

    /// `SRANDMEMBER count`: members *without* removing them.
    ///
    /// A negative count in Redis means "allow repeats, return exactly that
    /// many"; a positive one means "distinct, at most that many". Both are
    /// honoured; the ordering caveat of `spop` applies.
    pub fn srandmember(&self, count: i64) -> Result<Vec<Bytes>, StructureError> {
        let members = self.as_set()?;
        if members.is_empty() {
            return Ok(Vec::new());
        }
        if count >= 0 {
            return Ok(members.iter().take(count as usize).cloned().collect());
        }
        let wanted = count.unsigned_abs() as usize;
        let pool: Vec<&Bytes> = members.iter().collect();
        Ok((0..wanted).map(|i| pool[i % pool.len()].clone()).collect())
    }

    // ── hash operations ──────────────────────────────────────────────────────

    /// `HSET`: returns how many fields were newly created (not updated).
    pub fn hset(&mut self, pairs: Vec<(Bytes, Bytes)>) -> Result<usize, StructureError> {
        let fields = self.as_hash_mut()?;
        let mut created = 0;
        for (field, value) in pairs {
            check_member(&field)?;
            check_member(&value)?;
            if fields.len() >= MAX_STRUCTURE_ENTRIES && !fields.contains_key(&field) {
                return Err(StructureError::TooManyEntries);
            }
            if fields.insert(field, value).is_none() {
                created += 1;
            }
        }
        Ok(created)
    }

    /// `HINCRBY`. Redis stores hash values as strings, so the current value is
    /// parsed and a non-integer is an error rather than a silent reset.
    pub fn hincrby(&mut self, field: Bytes, delta: i64) -> Result<i64, StructureError> {
        check_member(&field)?;
        let fields = self.as_hash_mut()?;
        let current = match fields.get(&field) {
            Some(raw) => std::str::from_utf8(raw)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .ok_or(StructureError::HashNotAnInteger)?,
            None => 0,
        };
        let next = current
            .checked_add(delta)
            .ok_or(StructureError::NotAnInteger)?;
        fields.insert(field, next.to_string().into_bytes());
        Ok(next)
    }

    // ── set operations ───────────────────────────────────────────────────────

    /// `SADD`: returns how many members were newly added.
    pub fn sadd(&mut self, members: Vec<Bytes>) -> Result<usize, StructureError> {
        let set = self.as_set_mut()?;
        let mut added = 0;
        for member in members {
            check_member(&member)?;
            if set.len() >= MAX_STRUCTURE_ENTRIES && !set.contains(&member) {
                return Err(StructureError::TooManyEntries);
            }
            if set.insert(member) {
                added += 1;
            }
        }
        Ok(added)
    }

    /// `SREM`: returns how many members were removed.
    pub fn srem(&mut self, members: &[Bytes]) -> Result<usize, StructureError> {
        let set = self.as_set_mut()?;
        Ok(members.iter().filter(|m| set.remove(*m)).count())
    }
}

fn check_member(member: &[u8]) -> Result<(), StructureError> {
    if member.len() > MAX_MEMBER_LEN {
        return Err(StructureError::MemberTooLong);
    }
    Ok(())
}

fn check_growth(current: usize, adding: usize) -> Result<(), StructureError> {
    if current.saturating_add(adding) > MAX_STRUCTURE_ENTRIES {
        return Err(StructureError::TooManyEntries);
    }
    Ok(())
}

/// Redis index semantics: negative counts back from the end, and an index past
/// either end clamps rather than erroring.
fn normalize_index(index: i64, len: i64) -> i64 {
    if index < 0 {
        (len + index).max(-1)
    } else {
        index
    }
}

// ─── persistence ─────────────────────────────────────────────────────────────

/// Key prefix that keeps structure keys from colliding with plain KV keys.
///
/// A separate namespace rather than a flag on the value: it means `GET` on a
/// list key cannot accidentally return the serialized structure, and `KEYS` on
/// the plain keyspace does not enumerate structures.
pub const STRUCTURE_PREFIX: &str = "struct:";

/// A compare-and-swap that lost; the caller re-reads and tries again.
struct Retry;

/// Serialize a structure for storage.
fn encode(structure: &Structure) -> Result<serde_json::Value, StructureError> {
    serde_json::to_value(structure).map_err(|_| StructureError::WrongType)
}
pub fn structure_key(key: &str) -> String {
    format!("{STRUCTURE_PREFIX}{key}")
}

/// Read-modify-write access to persisted structures.
///
/// Every mutation goes through [`Self::mutate`], which reads the current value,
/// applies the caller's change and writes it back with `if_revision` set to what
/// it read. A concurrent writer therefore loses the compare-and-swap and the
/// operation is retried, instead of one of the two updates being silently
/// dropped — the classic lost-update bug for read-modify-write on shared state.
pub struct Structures<'a> {
    engine: &'a crate::engine::Engine,
}

/// What a mutation did, plus the revision it produced.
///
/// The revision is what `WATCH` compares against: it comes from the KV store's
/// existing per-key counter, so no separate versioning had to be invented.
#[derive(Debug)]
pub struct Applied<T> {
    pub value: T,
    pub revision: u64,
}

impl<'a> Structures<'a> {
    pub fn new(engine: &'a crate::engine::Engine) -> Self {
        Self { engine }
    }

    /// Load a structure, or `None` when the key is unset.
    ///
    /// A key holding something that is not a structure reads as
    /// [`StructureError::WrongType`] rather than as absent: reporting "no such
    /// list" for a key that plainly exists would send a caller in the wrong
    /// direction entirely.
    pub fn load(&self, key: &str) -> Result<Option<(Structure, u64)>, StructureError> {
        let Some(item) = self.engine.get_state(&structure_key(key)) else {
            // Nothing in the structure slot. If the plain slot is taken, the
            // name belongs to a string and every structure command on it is a
            // type error — the two slots are one keyspace to a client.
            //
            // The check lives here rather than in front of the dispatcher so it
            // runs *after* each command has validated its own arity. Redis
            // reports a malformed command before a type conflict, and a client
            // debugging `LPUSH k` with no value must be told about the missing
            // value, not about the type.
            if self.engine.get_state(key).is_some() {
                return Err(StructureError::WrongType);
            }
            return Ok(None);
        };
        let Some(json) = item.value.as_json() else {
            return Err(StructureError::WrongType);
        };
        // A value that is not a structure at all is a type error; a value that
        // *is* one but will not parse is corruption, and saying so is the
        // difference between a client fixing its command and an operator
        // looking at the data.
        let structure = match serde_json::from_value::<Structure>(json.clone()) {
            Ok(structure) => structure,
            Err(e) if json.get("kind").is_some() => {
                return Err(StructureError::Corrupt(e.to_string()))
            }
            Err(_) => return Err(StructureError::WrongType),
        };
        Ok(Some((structure, item.revision)))
    }

    /// Apply `change` to the structure at `key`, creating it with `empty` when
    /// absent, and persist the result.
    ///
    /// Retries on a revision conflict. The bound is deliberate: an unbounded
    /// retry loop under heavy contention is an outage that looks like a hang, so
    /// exhausting it surfaces as an error the caller can report.
    pub fn mutate<T, F>(
        &self,
        key: &str,
        empty: fn() -> Structure,
        mut change: F,
    ) -> Result<Applied<T>, StructureError>
    where
        F: FnMut(&mut Structure) -> Result<T, StructureError>,
    {
        const MAX_ATTEMPTS: usize = 16;
        for _ in 0..MAX_ATTEMPTS {
            let existing = self.load(key)?;
            let (mut structure, expected) = match existing {
                Some((structure, revision)) => (structure, Some(revision)),
                None => (empty(), None),
            };

            let outcome = change(&mut structure)?;

            // An emptied structure is deleted, matching Redis: a list with no
            // elements does not exist, and leaving an empty husk behind would
            // make EXISTS and TYPE disagree with LLEN.
            if structure.is_empty() {
                if expected.is_some() {
                    let _ = self.engine.delete_state(&structure_key(key));
                }
                return Ok(Applied {
                    value: outcome,
                    revision: 0,
                });
            }

            let encoded =
                serde_json::to_value(&structure).map_err(|_| StructureError::WrongType)?;
            match self
                .engine
                .put_state(structure_key(key), encoded, None, expected)
            {
                Ok(item) => {
                    return Ok(Applied {
                        value: outcome,
                        revision: item.revision,
                    })
                }
                // Someone else wrote between our read and our write: start over
                // from their value rather than clobbering it.
                Err(crate::engine::EngineError::State(
                    crate::engine::StateError::RevisionMismatch,
                )) => continue,
                Err(_) => return Err(StructureError::WrongType),
            }
        }
        Err(StructureError::TooManyEntries)
    }

    /// Delete a structure. Returns whether it existed.
    /// Move one element between two list keys as a single durable unit.
    ///
    /// This is `RPOPLPUSH`/`LMOVE` and their blocking forms. Redis makes the
    /// move atomic, and clients buy exactly that guarantee: kombu's unacked
    /// queue exists so a task is not lost when a worker dies mid-delivery.
    /// Implemented as a pop followed by a push, a process death in between
    /// dropped the element — the one failure the primitive is meant to
    /// prevent.
    ///
    /// Both keys are read, the change is computed, and both are written in one
    /// WAL record guarded by a compare-and-swap on each. A concurrent writer on
    /// either key restarts the whole attempt rather than clobbering it.
    ///
    /// Returns the moved element, or `None` when the source is empty.
    ///
    /// `source == destination` is handled by the same path: the rotation idiom
    /// `RPOPLPUSH mylist mylist` reads one structure, rotates it, and writes it
    /// once. Treating it as two keys would prepare two revisions for the same
    /// key and the second write would fail its own compare-and-swap.
    pub fn move_element(
        &self,
        source: &str,
        destination: &str,
        from_left: bool,
        to_left: bool,
    ) -> Result<Option<Bytes>, StructureError> {
        use crate::engine::StateOp;

        const MAX_ATTEMPTS: usize = 16;
        for _ in 0..MAX_ATTEMPTS {
            let same_key = source == destination;

            let (mut src, src_revision) = match self.load(source)? {
                Some((structure, revision)) => (structure, Some(revision)),
                // Nothing to move. Not an error: `RPOPLPUSH` on an empty
                // source is a nil.
                None => return Ok(None),
            };

            if same_key {
                let popped = if from_left {
                    src.lpop(1)?
                } else {
                    src.rpop(1)?
                };
                let Some(element) = popped.into_iter().next() else {
                    return Ok(None);
                };
                let values = vec![element.clone()];
                if to_left {
                    src.lpush(values)?;
                } else {
                    src.rpush(values)?;
                }
                match self.write_one(source, &src, src_revision) {
                    Ok(()) => return Ok(Some(element)),
                    Err(Retry) => continue,
                }
            }

            let (mut dst, dst_revision) = match self.load(destination)? {
                Some((structure, revision)) => (structure, Some(revision)),
                None => (Structure::empty_list(), None),
            };

            // The type check happens before anything is popped: a destination
            // holding the wrong type must not cost the source its element.
            dst.as_list()?;

            let popped = if from_left {
                src.lpop(1)?
            } else {
                src.rpop(1)?
            };
            let Some(element) = popped.into_iter().next() else {
                return Ok(None);
            };
            let values = vec![element.clone()];
            if to_left {
                dst.lpush(values)?;
            } else {
                dst.rpush(values)?;
            }

            let mut ops = Vec::with_capacity(2);
            // An emptied source is deleted, matching what `mutate` does and
            // what Redis does: a list with no elements does not exist.
            if src.is_empty() {
                ops.push(StateOp::Delete {
                    key: structure_key(source),
                });
            } else {
                ops.push(StateOp::Put {
                    key: structure_key(source),
                    value: crate::engine::stored::StoredVal::Json(encode(&src)?),
                    ttl_ms: None,
                    if_revision: src_revision,
                });
            }
            ops.push(StateOp::Put {
                key: structure_key(destination),
                value: crate::engine::stored::StoredVal::Json(encode(&dst)?),
                ttl_ms: None,
                if_revision: dst_revision,
            });

            match self.engine.put_state_batch(ops) {
                Ok(()) => return Ok(Some(element)),
                // Someone wrote to either key between our read and our write.
                // Nothing was applied, so start over from their value.
                Err(crate::engine::EngineError::State(
                    crate::engine::StateError::RevisionMismatch,
                )) => continue,
                Err(_) => return Err(StructureError::WrongType),
            }
        }
        Err(StructureError::TooManyEntries)
    }

    /// Write one structure back under a compare-and-swap, deleting it when it
    /// came out empty.
    fn write_one(
        &self,
        key: &str,
        structure: &Structure,
        expected: Option<u64>,
    ) -> Result<(), Retry> {
        if structure.is_empty() {
            let _ = self.engine.delete_state(&structure_key(key));
            return Ok(());
        }
        let Ok(encoded) = encode(structure) else {
            return Err(Retry);
        };
        match self
            .engine
            .put_state(structure_key(key), encoded, None, expected)
        {
            Ok(_) => Ok(()),
            Err(_) => Err(Retry),
        }
    }
    pub fn delete(&self, key: &str) -> bool {
        self.engine
            .delete_state(&structure_key(key))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> Bytes {
        s.as_bytes().to_vec()
    }

    // ── type discipline ──────────────────────────────────────────────────────

    #[test]
    fn operating_on_the_wrong_type_is_wrongtype() {
        // The protocol layer renders this as `-WRONGTYPE`, and clients match on
        // that exact string, so it must be a distinct error and not a generic
        // failure.
        let mut list = Structure::empty_list();
        assert_eq!(
            list.hset(vec![(b("f"), b("v"))]),
            Err(StructureError::WrongType)
        );
        assert_eq!(list.sadd(vec![b("m")]), Err(StructureError::WrongType));
        assert_eq!(list.as_zset().err(), Some(StructureError::WrongType));

        let mut hash = Structure::empty_hash();
        assert_eq!(hash.lpush(vec![b("x")]), Err(StructureError::WrongType));
    }

    #[test]
    fn type_names_match_redis() {
        assert_eq!(Structure::empty_list().type_name(), "list");
        assert_eq!(Structure::empty_hash().type_name(), "hash");
        assert_eq!(Structure::empty_set().type_name(), "set");
        assert_eq!(Structure::empty_zset().type_name(), "zset");
    }

    // ── lists ────────────────────────────────────────────────────────────────

    #[test]
    fn list_push_and_pop_respect_ends() {
        let mut list = Structure::empty_list();
        list.rpush(vec![b("a"), b("b")]).unwrap();
        list.lpush(vec![b("z")]).unwrap();
        // lpush prepends, so the list is z, a, b
        assert_eq!(list.lrange(0, -1).unwrap(), vec![b("z"), b("a"), b("b")]);
        assert_eq!(list.lpop(1).unwrap(), vec![b("z")]);
        assert_eq!(list.rpop(1).unwrap(), vec![b("b")]);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn lpush_of_several_reverses_them() {
        // Redis pushes arguments one at a time, so LPUSH k a b c leaves c,b,a.
        // Getting this backwards silently reverses every queue built on it.
        let mut list = Structure::empty_list();
        list.lpush(vec![b("a"), b("b"), b("c")]).unwrap();
        assert_eq!(list.lrange(0, -1).unwrap(), vec![b("c"), b("b"), b("a")]);
    }

    #[test]
    fn lrange_handles_negative_and_out_of_range_bounds() {
        let mut list = Structure::empty_list();
        list.rpush(vec![b("a"), b("b"), b("c"), b("d")]).unwrap();
        assert_eq!(list.lrange(0, -1).unwrap().len(), 4);
        assert_eq!(list.lrange(-2, -1).unwrap(), vec![b("c"), b("d")]);
        assert_eq!(list.lrange(1, 2).unwrap(), vec![b("b"), b("c")]);
        // Past the end clamps rather than erroring.
        assert_eq!(list.lrange(2, 99).unwrap(), vec![b("c"), b("d")]);
        // Inverted range is empty, not an error.
        assert!(list.lrange(3, 1).unwrap().is_empty());
        // An empty list yields nothing for any range.
        assert!(Structure::empty_list().lrange(0, -1).unwrap().is_empty());
    }

    #[test]
    fn lrem_direction_follows_the_sign_of_count() {
        let items = || {
            let mut list = Structure::empty_list();
            list.rpush(vec![b("x"), b("a"), b("x"), b("b"), b("x")])
                .unwrap();
            list
        };

        let mut from_head = items();
        assert_eq!(from_head.lrem(1, b"x").as_ref().unwrap(), &1);
        assert_eq!(
            from_head.lrange(0, -1).unwrap(),
            vec![b("a"), b("x"), b("b"), b("x")],
            "count > 0 must remove from the head"
        );

        let mut from_tail = items();
        assert_eq!(from_tail.lrem(-1, b"x").as_ref().unwrap(), &1);
        assert_eq!(
            from_tail.lrange(0, -1).unwrap(),
            vec![b("x"), b("a"), b("x"), b("b")],
            "count < 0 must remove from the tail"
        );

        let mut all = items();
        assert_eq!(all.lrem(0, b"x").as_ref().unwrap(), &3);
        assert_eq!(all.lrange(0, -1).unwrap(), vec![b("a"), b("b")]);
    }

    // ── hashes ───────────────────────────────────────────────────────────────

    #[test]
    fn hset_counts_only_new_fields() {
        let mut hash = Structure::empty_hash();
        assert_eq!(
            hash.hset(vec![(b("a"), b("1")), (b("b"), b("2"))]).unwrap(),
            2
        );
        // Updating an existing field reports zero created, which is what HSET
        // returns and what kombu's unacked bookkeeping reads.
        assert_eq!(hash.hset(vec![(b("a"), b("9"))]).unwrap(), 0);
        assert_eq!(hash.as_hash().unwrap().get(&b("a")), Some(&b("9")));
    }

    #[test]
    fn hgetall_order_is_stable() {
        // A BTreeMap keeps field order deterministic across processes. Clients
        // should not depend on it, but a stable order makes differential testing
        // against real Redis tractable.
        let mut hash = Structure::empty_hash();
        hash.hset(vec![(b("c"), b("3")), (b("a"), b("1")), (b("b"), b("2"))])
            .unwrap();
        let keys: Vec<_> = hash.as_hash().unwrap().keys().cloned().collect();
        assert_eq!(keys, vec![b("a"), b("b"), b("c")]);
    }

    #[test]
    fn hincrby_starts_at_zero_and_rejects_non_integers() {
        let mut hash = Structure::empty_hash();
        assert_eq!(hash.hincrby(b("n"), 5).unwrap(), 5);
        assert_eq!(hash.hincrby(b("n"), -2).unwrap(), 3);

        hash.hset(vec![(b("word"), b("not a number"))]).unwrap();
        assert_eq!(
            hash.hincrby(b("word"), 1),
            Err(StructureError::HashNotAnInteger),
            "incrementing a non-integer must fail rather than silently reset it"
        );
    }

    #[test]
    fn hincrby_overflow_is_an_error_not_a_wrap() {
        let mut hash = Structure::empty_hash();
        hash.hset(vec![(b("n"), i64::MAX.to_string().into_bytes())])
            .unwrap();
        assert_eq!(hash.hincrby(b("n"), 1), Err(StructureError::NotAnInteger));
    }

    // ── sets ─────────────────────────────────────────────────────────────────

    #[test]
    fn sadd_and_srem_count_actual_changes() {
        let mut set = Structure::empty_set();
        assert_eq!(set.sadd(vec![b("a"), b("b"), b("a")]).unwrap(), 2);
        assert_eq!(set.len(), 2);
        assert_eq!(set.srem(&[b("a"), b("missing")]).unwrap(), 1);
        assert_eq!(set.len(), 1);
    }

    // ── sorted sets ──────────────────────────────────────────────────────────

    #[test]
    fn zset_ties_break_lexicographically_by_member() {
        // Contract, not accident: `arq` pages through a zset by score, and an
        // unstable tie order would let it skip or repeat jobs.
        let mut z = Structure::empty_zset();
        let zs = z.as_zset_mut().unwrap();
        for member in ["delta", "alpha", "charlie", "bravo"] {
            zs.add(b(member), 1.0).unwrap();
        }
        let order: Vec<_> = zs.range().map(|(m, _)| m.clone()).collect();
        assert_eq!(
            order,
            vec![b("alpha"), b("bravo"), b("charlie"), b("delta")]
        );
    }

    #[test]
    fn zset_orders_by_score_first() {
        let mut z = Structure::empty_zset();
        let zs = z.as_zset_mut().unwrap();
        zs.add(b("zzz"), 1.0).unwrap();
        zs.add(b("aaa"), 2.0).unwrap();
        let order: Vec<_> = zs.range().map(|(m, _)| m.clone()).collect();
        assert_eq!(order, vec![b("zzz"), b("aaa")], "score beats member order");
    }

    #[test]
    fn zadd_updates_without_duplicating_the_member() {
        // The bug this guards: updating a score by inserting the new ordered
        // entry without removing the old one makes the member appear twice in
        // every range scan.
        let mut z = Structure::empty_zset();
        let zs = z.as_zset_mut().unwrap();
        assert!(zs.add(b("m"), 1.0).unwrap(), "first add is new");
        assert!(!zs.add(b("m"), 5.0).unwrap(), "second add is an update");
        assert_eq!(zs.len(), 1);
        assert_eq!(zs.range().count(), 1);
        assert_eq!(zs.score(b"m"), Some(5.0));
    }

    #[test]
    fn zset_remove_and_rank() {
        let mut z = Structure::empty_zset();
        let zs = z.as_zset_mut().unwrap();
        for (member, score) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            zs.add(b(member), score).unwrap();
        }
        assert_eq!(zs.rank(b"a"), Some(0));
        assert_eq!(zs.rank(b"c"), Some(2));
        assert_eq!(zs.rank(b"missing"), None);
        assert!(zs.remove(b"b"));
        assert!(!zs.remove(b"b"), "removing twice reports absent");
        assert_eq!(zs.rank(b"c"), Some(1));
    }

    #[test]
    fn range_by_score_is_inclusive_at_both_ends() {
        // arq polls for due jobs with an inclusive upper bound; an exclusive one
        // would leave a job that is due at exactly `now` sitting forever.
        let mut z = Structure::empty_zset();
        let zs = z.as_zset_mut().unwrap();
        for (member, score) in [("a", 1.0), ("b", 2.0), ("c", 3.0)] {
            zs.add(b(member), score).unwrap();
        }
        let members: Vec<_> = zs
            .range_by_score(1.0, 2.0)
            .into_iter()
            .map(|(m, _)| m)
            .collect();
        assert_eq!(members, vec![b("a"), b("b")]);
        assert!(zs.range_by_score(9.0, 10.0).is_empty());
    }

    #[test]
    fn nan_scores_are_refused() {
        // A NaN score compares false with everything, which would corrupt the
        // ordered index rather than merely sorting oddly.
        let mut z = Structure::empty_zset();
        assert_eq!(
            z.as_zset_mut().unwrap().add(b("m"), f64::NAN),
            Err(StructureError::NotANumber)
        );
    }

    #[test]
    fn infinite_scores_are_allowed_and_ordered() {
        // Redis accepts +inf/-inf, and they are the natural sentinels for
        // "always last" and "always first".
        let mut z = Structure::empty_zset();
        let zs = z.as_zset_mut().unwrap();
        zs.add(b("last"), f64::INFINITY).unwrap();
        zs.add(b("first"), f64::NEG_INFINITY).unwrap();
        zs.add(b("mid"), 0.0).unwrap();
        let order: Vec<_> = zs.range().map(|(m, _)| m.clone()).collect();
        assert_eq!(order, vec![b("first"), b("mid"), b("last")]);
    }

    // ── limits and binary safety ─────────────────────────────────────────────

    #[test]
    fn an_oversized_member_is_refused() {
        let mut list = Structure::empty_list();
        assert_eq!(
            list.lpush(vec![vec![0u8; MAX_MEMBER_LEN + 1]]),
            Err(StructureError::MemberTooLong)
        );
    }

    #[test]
    fn members_are_binary_safe() {
        // Not strings: a pickled Celery payload is not valid UTF-8, and the
        // whole point of byte members is that it survives anyway.
        let payload = vec![0x80, 0x04, 0x95, 0x00, 0xFF];
        let mut list = Structure::empty_list();
        list.rpush(vec![payload.clone()]).unwrap();
        assert_eq!(list.lrange(0, -1).unwrap(), vec![payload]);
    }

    #[test]
    fn structures_roundtrip_through_json() {
        // They are persisted as a value in the KV store, so the serialized form
        // has to survive the JSON-lines WAL — binary members included.
        let mut original = Structure::empty_zset();
        let zs = original.as_zset_mut().unwrap();
        zs.add(vec![0xFF, 0x00], 1.5).unwrap();
        zs.add(b("plain"), -2.0).unwrap();

        let text = serde_json::to_string(&original).unwrap();
        let back: Structure = serde_json::from_str(&text).unwrap();
        assert_eq!(back, original);
        assert_eq!(back.as_zset().unwrap().score(&[0xFF, 0x00]), Some(1.5));
    }

    #[test]
    fn every_variant_roundtrips_and_keeps_its_type() {
        let mut list = Structure::empty_list();
        list.rpush(vec![b("a")]).unwrap();
        let mut hash = Structure::empty_hash();
        hash.hset(vec![(b("f"), b("v"))]).unwrap();
        let mut set = Structure::empty_set();
        set.sadd(vec![b("m")]).unwrap();

        for original in [list, hash, set, Structure::empty_zset()] {
            let text = serde_json::to_string(&original).unwrap();
            let back: Structure = serde_json::from_str(&text).unwrap();
            assert_eq!(back.type_name(), original.type_name());
            assert_eq!(back, original);
        }
    }

    // ── persistence through the state store ──────────────────────────────────

    fn test_engine() -> (crate::engine::Engine, tempfile::TempDir) {
        use crate::config::Config;
        use crate::engine::Engine;
        use tokio_util::sync::CancellationToken;

        let dir = tempfile::tempdir().unwrap();
        let config = Config {
            data_dir: Some(dir.path().to_str().unwrap().to_string()),
            ..Config::default()
        };
        let engine = Engine::new(config, CancellationToken::new()).unwrap();
        (engine, dir)
    }

    #[test]
    fn a_structure_persists_and_reloads() {
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);

        let applied = structures
            .mutate("jobs", Structure::empty_list, |s| {
                s.rpush(vec![b("first"), b("second")])
            })
            .unwrap();
        assert_eq!(applied.value, 2);
        assert!(applied.revision > 0, "a write must report a revision");

        let (loaded, revision) = structures.load("jobs").unwrap().unwrap();
        assert_eq!(loaded.lrange(0, -1).unwrap(), vec![b("first"), b("second")]);
        assert_eq!(revision, applied.revision);
    }

    #[test]
    fn a_missing_structure_is_none_not_an_error() {
        let (engine, _dir) = test_engine();
        assert!(Structures::new(&engine).load("nothing").unwrap().is_none());
    }

    #[test]
    fn revisions_advance_with_each_mutation() {
        // This counter is what WATCH compares against, so it has to move on
        // every write or an optimistic transaction would never detect a change.
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);

        let first = structures
            .mutate("k", Structure::empty_list, |s| s.rpush(vec![b("a")]))
            .unwrap();
        let second = structures
            .mutate("k", Structure::empty_list, |s| s.rpush(vec![b("b")]))
            .unwrap();
        assert!(
            second.revision > first.revision,
            "revision must advance: {} then {}",
            first.revision,
            second.revision
        );
    }

    #[test]
    fn emptying_a_structure_deletes_the_key() {
        // Redis has no empty list: EXISTS and TYPE must agree with LLEN, and an
        // empty husk left behind would make them disagree.
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);

        structures
            .mutate("k", Structure::empty_list, |s| s.rpush(vec![b("only")]))
            .unwrap();
        assert!(structures.load("k").unwrap().is_some());

        structures
            .mutate("k", Structure::empty_list, |s| s.lpop(1))
            .unwrap();
        assert!(
            structures.load("k").unwrap().is_none(),
            "an emptied structure must not linger as an empty value"
        );
    }

    #[test]
    fn a_plain_kv_value_under_a_structure_key_is_wrongtype() {
        // Defence in depth: the prefix keeps the keyspaces apart, but if
        // something does land there it must report WRONGTYPE rather than being
        // read as an absent structure.
        let (engine, _dir) = test_engine();
        engine
            .put_state(structure_key("intruder"), serde_json::json!(42), None, None)
            .unwrap();
        assert_eq!(
            Structures::new(&engine).load("intruder"),
            Err(StructureError::WrongType)
        );
    }

    /// A name taken by a plain value is not available to a structure.
    ///
    /// This test used to assert the opposite — that the two "live in separate
    /// namespaces" and neither disturbs the other. That was the design
    /// assumption written down as a guarantee, and the differential suite
    /// against a real Redis 7 showed it was wrong: Redis has one keyspace with
    /// one type per key, and clients depend on `SET lock 1` making `LPUSH lock`
    /// fail. Two slots underneath is an implementation detail; two keyspaces is
    /// a different database.
    #[test]
    fn a_plain_key_blocks_the_structure_slot_of_the_same_name() {
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);

        engine
            .put_state("shared".to_string(), serde_json::json!("plain"), None, None)
            .unwrap();

        assert_eq!(
            structures.load("shared").err(),
            Some(StructureError::WrongType),
            "a string under this name makes every structure command a type error"
        );
        assert_eq!(
            structures
                .mutate("shared", Structure::empty_set, |s| s.sadd(vec![b("m")]))
                .err(),
            Some(StructureError::WrongType),
            "and it must refuse the write, not create a shadow structure"
        );
        // The plain value is untouched by the refusal.
        assert_eq!(
            engine.get_state("shared").unwrap().value.as_json(),
            Some(&serde_json::json!("plain"))
        );
    }

    #[test]
    fn a_wrong_type_operation_does_not_write() {
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);

        structures
            .mutate("k", Structure::empty_list, |s| s.rpush(vec![b("a")]))
            .unwrap();
        // Attempting a hash operation on a list must fail *and* leave the list
        // as it was — a partial write here would corrupt the value.
        let err = structures.mutate("k", Structure::empty_list, |s| {
            s.hset(vec![(b("f"), b("v"))])
        });
        assert_eq!(err.err(), Some(StructureError::WrongType));
        assert_eq!(
            structures
                .load("k")
                .unwrap()
                .unwrap()
                .0
                .lrange(0, -1)
                .unwrap(),
            vec![b("a")]
        );
    }

    #[test]
    fn every_variant_survives_a_persistence_roundtrip() {
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);

        structures
            .mutate("l", Structure::empty_list, |s| {
                s.rpush(vec![vec![0xFF, 0x00]])
            })
            .unwrap();
        structures
            .mutate("h", Structure::empty_hash, |s| {
                s.hset(vec![(vec![0x80], vec![0x01, 0xFE])])
            })
            .unwrap();
        structures
            .mutate("s", Structure::empty_set, |s| {
                s.sadd(vec![vec![0x00, 0xFF]])
            })
            .unwrap();
        structures
            .mutate("z", Structure::empty_zset, |s| {
                s.as_zset_mut()?.add(vec![0xFE], -1.5).map(|_| ())
            })
            .unwrap();

        assert_eq!(structures.load("l").unwrap().unwrap().0.type_name(), "list");
        assert_eq!(structures.load("h").unwrap().unwrap().0.type_name(), "hash");
        assert_eq!(structures.load("s").unwrap().unwrap().0.type_name(), "set");
        let (zset, _) = structures.load("z").unwrap().unwrap();
        assert_eq!(zset.as_zset().unwrap().score(&[0xFE]), Some(-1.5));
    }

    /// The bug this pins: `serde_json` writes `f64::INFINITY` as `null`, so with
    /// a derived `Serialize` a sorted set holding `+inf` was written
    /// successfully and then failed to load — and the load path called that
    /// `WRONGTYPE`, which points at the client instead of the data.
    #[test]
    fn an_infinite_score_survives_a_json_round_trip() {
        let mut zset = ZSet::default();
        zset.add(b"never".to_vec(), f64::INFINITY).unwrap();
        zset.add(b"always".to_vec(), f64::NEG_INFINITY).unwrap();
        zset.add(b"middle".to_vec(), 1.5).unwrap();

        let json = serde_json::to_string(&zset).unwrap();
        assert!(
            !json.contains("null"),
            "an infinity must not be written as null: {json}"
        );
        let back: ZSet = serde_json::from_str(&json).unwrap();

        assert_eq!(back.score(b"never"), Some(f64::INFINITY));
        assert_eq!(back.score(b"always"), Some(f64::NEG_INFINITY));
        assert_eq!(back.score(b"middle"), Some(1.5));
        // And the ordering survived, which is the part the `ordered` index
        // depends on.
        let order: Vec<&[u8]> = back.range().map(|(m, _)| m.as_slice()).collect();
        assert_eq!(order, [b"always".as_slice(), b"middle", b"never"]);
    }

    /// A finite score must still be written as a bare JSON number, so sorted
    /// sets already on disk load unchanged.
    #[test]
    fn a_finite_score_is_still_written_as_a_number() {
        let mut zset = ZSet::default();
        zset.add(b"m".to_vec(), 2.5).unwrap();
        let json = serde_json::to_string(&zset).unwrap();
        assert!(
            json.contains("2.5") && !json.contains("\"2.5\""),
            "a finite score must stay a JSON number, not become a string: {json}"
        );
    }

    #[test]
    fn a_structure_that_will_not_parse_is_reported_as_corrupt_not_wrongtype() {
        // The mislabel is what hid the infinite-score bug for as long as it
        // existed, so the distinction is pinned rather than left to a comment.
        let (engine, _dir) = test_engine();
        let structures = Structures::new(&engine);
        // A value that *is* tagged as a structure but whose body is nonsense.
        engine
            .put_state(
                structure_key("broken"),
                serde_json::json!({ "kind": "z_set", "scores": "not a list" }),
                None,
                None,
            )
            .unwrap();
        match structures.load("broken") {
            Err(StructureError::Corrupt(detail)) => {
                assert!(
                    !detail.is_empty(),
                    "the reason must be carried, not dropped"
                )
            }
            other => panic!("expected Corrupt, got {other:?}"),
        }

        // While a value that is not a structure at all is still a type error:
        // that really is a client using the wrong command on a plain string.
        engine
            .put_state(
                structure_key("plain"),
                serde_json::json!("just a string"),
                None,
                None,
            )
            .unwrap();
        assert!(matches!(
            structures.load("plain"),
            Err(StructureError::WrongType)
        ));
    }
}
