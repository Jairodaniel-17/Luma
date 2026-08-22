//! Decoder for Postgres's `pgoutput` logical replication format.
//!
//! Written rather than depended upon, and the spike (W4.1 of
//! `docs/PLAN-MAESTRO.md`) is what settled that. The survey it produced:
//!
//! - `postgres-replication`, the crate in the rust-postgres workspace that does
//!   exactly this, **is not published**. It exists only on the git master
//!   branch.
//! - `tokio-postgres` 0.7.18 — the released version — has neither
//!   `ReplicationMode` nor `copy_both_simple`. Both are master-only too. Its
//!   connection-string parser rejects `replication=database` as an unknown key,
//!   so it cannot even open the right kind of connection.
//! - `pg_replicate` is at 0.1.0.
//! - `rustcdc` pulls `wasmtime`, `mysql_async` and `tiberius` through its
//!   default features. For a database that ships as one binary and gates every
//!   dependency through `cargo deny`, that is a supply-chain surface many times
//!   the size of the feature being bought.
//!
//! So the choice was never "crate versus own code" — it was "git dependency on
//! an unreleased branch versus own code". A pinned git revision is a dependency
//! nobody can audit a version of, and `cargo deny` has nothing to check it
//! against.
//!
//! What is *not* hand-rolled: SCRAM-SHA-256, which comes from
//! `postgres-protocol` (published, and authentication is the one part of this
//! where a subtle mistake is a security bug rather than a parse error).
//!
//! The format itself is small and stable — this file is the whole of it for
//! protocol version 1. Reference: Postgres "Logical Replication Message
//! Formats" in the protocol chapter.

use std::collections::HashMap;

/// A decoding failure. Always a bug or a corrupt stream — never something to
/// retry, so it carries a reason rather than a code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pgoutput: {}", self.0)
    }
}

impl std::error::Error for DecodeError {}

type Result<T> = std::result::Result<T, DecodeError>;

/// A cursor that refuses to read past the end.
///
/// Every field below is length-prefixed or fixed-width by a *remote* party. A
/// slice index would panic on a truncated message, and a truncated message is
/// something the network can produce at any time.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, at: 0 }
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| DecodeError(format!("length overflow reading {what}")))?;
        if end > self.bytes.len() {
            return Err(DecodeError(format!(
                "truncated reading {what}: wanted {n} bytes at offset {}, {} remain",
                self.at,
                self.bytes.len() - self.at
            )));
        }
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn i16(&mut self, what: &str) -> Result<i16> {
        let b = self.take(2, what)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    fn i32(&mut self, what: &str) -> Result<i32> {
        let b = self.take(4, what)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(self.i32(what)? as u32)
    }

    fn i64(&mut self, what: &str) -> Result<i64> {
        let b = self.take(8, what)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(self.i64(what)? as u64)
    }

    /// A NUL-terminated string.
    ///
    /// Lossy rather than strict: a non-UTF-8 identifier or value must not stop
    /// a replication stream, because there is no way to skip one record and
    /// resume — the next LSN is only reachable through this one.
    fn cstr(&mut self, what: &str) -> Result<String> {
        let start = self.at;
        if start > self.bytes.len() {
            return Err(DecodeError(format!("truncated reading {what}")));
        }
        let end = self.bytes[start..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| DecodeError(format!("unterminated string reading {what}")))?;
        self.at = start + end + 1;
        Ok(String::from_utf8_lossy(&self.bytes[start..start + end]).into_owned())
    }
}

/// One column's value inside a tuple.
///
/// `Unchanged` is not `Null`, and conflating them is the classic pgoutput bug:
/// a large column that did not change is omitted from an UPDATE, and writing it
/// through as NULL silently destroys data Postgres still holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Null,
    /// The column was not part of the update; its stored value still stands.
    Unchanged,
    Text(String),
    Binary(Vec<u8>),
}

/// The columns of one row, positionally, as the relation declared them.
pub type Tuple = Vec<Value>;

/// A column as announced in a `Relation` message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    /// Postgres OID. Kept rather than resolved: mapping is the connector's job.
    pub type_id: u32,
    pub type_modifier: i32,
    /// Whether the column takes part in the replica identity (the key).
    pub is_key: bool,
}

/// A table's shape, announced before any row that refers to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    pub id: u32,
    pub namespace: String,
    pub name: String,
    pub replica_identity: u8,
    pub columns: Vec<Column>,
}

impl Relation {
    /// `schema.table`, which is what a mapping configuration names.
    pub fn qualified(&self) -> String {
        format!("{}.{}", self.namespace, self.name)
    }

    /// The columns that make up the replica identity, in declaration order.
    ///
    /// Empty means `REPLICA IDENTITY NOTHING`, and a table configured that way
    /// produces INSERTs that can never be updated or deleted downstream — worth
    /// reporting at connect time rather than discovering on the first UPDATE.
    pub fn key_columns(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|c| c.is_key)
            .map(|c| c.name.as_str())
            .collect()
    }
}

/// What the old-tuple byte of an UPDATE or DELETE said the tuple is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OldTuple {
    /// `REPLICA IDENTITY DEFAULT`: only the key columns are present.
    Key,
    /// `REPLICA IDENTITY FULL`: the whole previous row.
    Full,
}

/// One decoded logical replication message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    Begin {
        final_lsn: u64,
        commit_time: i64,
        xid: u32,
    },
    Commit {
        commit_lsn: u64,
        end_lsn: u64,
        commit_time: i64,
    },
    Relation(Relation),
    /// A user-defined type's name. Kept so an unmappable column can be reported
    /// by type name rather than by a bare OID nobody can look up afterwards.
    Type {
        id: u32,
        namespace: String,
        name: String,
    },
    Insert {
        relation_id: u32,
        tuple: Tuple,
    },
    Update {
        relation_id: u32,
        /// Absent when the replica identity is DEFAULT and the key did not
        /// change — Postgres omits it because the new tuple already carries it.
        old: Option<(OldTuple, Tuple)>,
        tuple: Tuple,
    },
    Delete {
        relation_id: u32,
        /// Always present. Which one it is depends on the replica identity, and
        /// with `NOTHING` Postgres refuses the DELETE rather than sending one
        /// without a key.
        old: (OldTuple, Tuple),
    },
    Truncate {
        relation_ids: Vec<u32>,
        cascade: bool,
        restart_identity: bool,
    },
    Origin {
        commit_lsn: u64,
        name: String,
    },
    /// `pg_logical_emit_message`. Not used by Luma; decoded so it cannot
    /// desynchronize the stream if something else on the database emits one.
    Message {
        transactional: bool,
        lsn: u64,
        prefix: String,
        content: Vec<u8>,
    },
}

/// Decode one logical replication message.
pub fn decode(bytes: &[u8]) -> Result<Change> {
    let mut r = Reader::new(bytes);
    let tag = r.u8("message tag")?;
    match tag {
        b'B' => Ok(Change::Begin {
            final_lsn: r.u64("begin final_lsn")?,
            commit_time: r.i64("begin commit_time")?,
            xid: r.u32("begin xid")?,
        }),
        b'C' => {
            let _flags = r.u8("commit flags")?;
            Ok(Change::Commit {
                commit_lsn: r.u64("commit lsn")?,
                end_lsn: r.u64("commit end_lsn")?,
                commit_time: r.i64("commit time")?,
            })
        }
        b'O' => Ok(Change::Origin {
            commit_lsn: r.u64("origin lsn")?,
            name: r.cstr("origin name")?,
        }),
        b'R' => {
            let id = r.u32("relation id")?;
            let namespace = r.cstr("relation namespace")?;
            let name = r.cstr("relation name")?;
            let replica_identity = r.u8("replica identity")?;
            let count = r.i16("relation column count")?;
            if count < 0 {
                return Err(DecodeError(format!("negative column count {count}")));
            }
            let mut columns = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let flags = r.u8("column flags")?;
                columns.push(Column {
                    name: r.cstr("column name")?,
                    type_id: r.u32("column type")?,
                    type_modifier: r.i32("column type modifier")?,
                    is_key: flags & 1 == 1,
                });
            }
            Ok(Change::Relation(Relation {
                id,
                namespace,
                name,
                replica_identity,
                columns,
            }))
        }
        b'Y' => Ok(Change::Type {
            id: r.u32("type id")?,
            namespace: r.cstr("type namespace")?,
            name: r.cstr("type name")?,
        }),
        b'I' => {
            let relation_id = r.u32("insert relation")?;
            let kind = r.u8("insert tuple kind")?;
            if kind != b'N' {
                return Err(DecodeError(format!(
                    "insert carried tuple kind {:?}, expected N",
                    kind as char
                )));
            }
            Ok(Change::Insert {
                relation_id,
                tuple: tuple(&mut r)?,
            })
        }
        b'U' => {
            let relation_id = r.u32("update relation")?;
            // The old tuple is optional and its marker is what says so, which
            // is why this reads the marker before deciding: taking an 'N' for a
            // key marker would shift every following field by one byte.
            let marker = r.u8("update tuple marker")?;
            let (old, marker) = match marker {
                b'K' => {
                    let t = tuple(&mut r)?;
                    (Some((OldTuple::Key, t)), r.u8("update new marker")?)
                }
                b'O' => {
                    let t = tuple(&mut r)?;
                    (Some((OldTuple::Full, t)), r.u8("update new marker")?)
                }
                other => (None, other),
            };
            if marker != b'N' {
                return Err(DecodeError(format!(
                    "update had no new tuple: marker {:?}",
                    marker as char
                )));
            }
            Ok(Change::Update {
                relation_id,
                old,
                tuple: tuple(&mut r)?,
            })
        }
        b'D' => {
            let relation_id = r.u32("delete relation")?;
            let marker = r.u8("delete tuple marker")?;
            let which = match marker {
                b'K' => OldTuple::Key,
                b'O' => OldTuple::Full,
                other => {
                    return Err(DecodeError(format!(
                        "delete carried tuple kind {:?}, expected K or O",
                        other as char
                    )))
                }
            };
            Ok(Change::Delete {
                relation_id,
                old: (which, tuple(&mut r)?),
            })
        }
        b'T' => {
            let count = r.i32("truncate relation count")?;
            if count < 0 {
                return Err(DecodeError(format!("negative truncate count {count}")));
            }
            let flags = r.u8("truncate flags")?;
            let mut relation_ids = Vec::with_capacity(count as usize);
            for _ in 0..count {
                relation_ids.push(r.u32("truncate relation id")?);
            }
            Ok(Change::Truncate {
                relation_ids,
                cascade: flags & 1 == 1,
                restart_identity: flags & 2 == 2,
            })
        }
        b'M' => {
            let flags = r.u8("message flags")?;
            let lsn = r.u64("message lsn")?;
            let prefix = r.cstr("message prefix")?;
            let len = r.i32("message length")?;
            if len < 0 {
                return Err(DecodeError(format!("negative message length {len}")));
            }
            Ok(Change::Message {
                transactional: flags & 1 == 1,
                lsn,
                prefix,
                content: r.take(len as usize, "message content")?.to_vec(),
            })
        }
        other => Err(DecodeError(format!(
            "unknown message tag {:?} ({other:#04x})",
            other as char
        ))),
    }
}

fn tuple(r: &mut Reader<'_>) -> Result<Tuple> {
    let count = r.i16("tuple column count")?;
    if count < 0 {
        return Err(DecodeError(format!("negative tuple column count {count}")));
    }
    let mut values = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let kind = r.u8("column kind")?;
        values.push(match kind {
            b'n' => Value::Null,
            b'u' => Value::Unchanged,
            b't' | b'b' => {
                let len = r.i32("column length")?;
                if len < 0 {
                    return Err(DecodeError(format!("negative column length {len}")));
                }
                let raw = r.take(len as usize, "column value")?;
                if kind == b't' {
                    Value::Text(String::from_utf8_lossy(raw).into_owned())
                } else {
                    Value::Binary(raw.to_vec())
                }
            }
            other => {
                return Err(DecodeError(format!(
                    "unknown column kind {:?}",
                    other as char
                )))
            }
        });
    }
    Ok(values)
}

/// Relations seen so far on this stream, by id.
///
/// A row message names its table by an integer that means nothing without the
/// `Relation` that introduced it. Postgres re-sends the relation after a
/// reconnect but *not* within a session, so this has to live exactly as long as
/// the stream and be dropped with it.
#[derive(Debug, Default)]
pub struct Relations(HashMap<u32, Relation>);

impl Relations {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a relation. Returns the previous shape if the table changed.
    pub fn insert(&mut self, relation: Relation) -> Option<Relation> {
        self.0.insert(relation.id, relation)
    }

    pub fn get(&self, id: u32) -> Option<&Relation> {
        self.0.get(&id)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Pair a tuple's values with the column names of its relation.
    ///
    /// `None` when the relation was never announced, which is a stream bug
    /// rather than a data condition — quietly producing a row with positional
    /// keys would hide it.
    pub fn name_values<'a>(
        &'a self,
        relation_id: u32,
        tuple: &'a [Value],
    ) -> Option<Vec<(&'a str, &'a Value)>> {
        let relation = self.0.get(&relation_id)?;
        Some(
            relation
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .zip(tuple.iter())
                .collect(),
        )
    }
}

/// Postgres counts microseconds from 2000-01-01, not 1970-01-01.
pub const PG_EPOCH_OFFSET_SECS: i64 = 946_684_800;

/// A pgoutput timestamp as milliseconds since the Unix epoch.
pub fn pg_time_to_unix_ms(pg_micros: i64) -> i64 {
    PG_EPOCH_OFFSET_SECS * 1000 + pg_micros / 1000
}

/// Format an LSN the way Postgres does, `XXXXXXXX/XXXXXXXX`.
///
/// Used in `START_REPLICATION` and in every operator-facing message: an LSN
/// printed as a decimal integer cannot be compared against anything
/// `pg_stat_replication` shows, which is where an operator looks.
pub fn format_lsn(lsn: u64) -> String {
    format!("{:X}/{:X}", lsn >> 32, lsn & 0xFFFF_FFFF)
}

/// Parse `XXXXXXXX/XXXXXXXX` back into an LSN.
pub fn parse_lsn(text: &str) -> Option<u64> {
    let (high, low) = text.split_once('/')?;
    let high = u64::from_str_radix(high.trim(), 16).ok()?;
    let low = u64::from_str_radix(low.trim(), 16).ok()?;
    if low > 0xFFFF_FFFF {
        return None;
    }
    Some((high << 32) | low)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a message the way Postgres does, so the tests exercise the decoder
    /// rather than a mirror of it.
    struct Build(Vec<u8>);

    impl Build {
        fn new(tag: u8) -> Self {
            Build(vec![tag])
        }
        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        fn i16(mut self, v: i16) -> Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        fn u32(self, v: u32) -> Self {
            self.i32(v as i32)
        }
        fn i64(mut self, v: i64) -> Self {
            self.0.extend_from_slice(&v.to_be_bytes());
            self
        }
        fn u64(self, v: u64) -> Self {
            self.i64(v as i64)
        }
        fn cstr(mut self, v: &str) -> Self {
            self.0.extend_from_slice(v.as_bytes());
            self.0.push(0);
            self
        }
        fn text(mut self, v: &str) -> Self {
            self.0.push(b't');
            self.0.extend_from_slice(&(v.len() as i32).to_be_bytes());
            self.0.extend_from_slice(v.as_bytes());
            self
        }
        fn null(mut self) -> Self {
            self.0.push(b'n');
            self
        }
        fn unchanged(mut self) -> Self {
            self.0.push(b'u');
            self
        }
        fn done(self) -> Vec<u8> {
            self.0
        }
    }

    fn relation_message() -> Vec<u8> {
        Build::new(b'R')
            .u32(16_384)
            .cstr("public")
            .cstr("orders")
            .u8(b'd')
            .i16(3)
            .u8(1) // id: part of the key
            .cstr("id")
            .u32(23)
            .i32(-1)
            .u8(0)
            .cstr("customer")
            .u32(25)
            .i32(-1)
            .u8(0)
            .cstr("total")
            .u32(1700)
            .i32(-1)
            .done()
    }

    #[test]
    fn a_relation_announces_the_shape_and_the_key() {
        let Change::Relation(r) = decode(&relation_message()).unwrap() else {
            panic!("expected a relation");
        };
        assert_eq!(r.qualified(), "public.orders");
        assert_eq!(r.columns.len(), 3);
        assert_eq!(r.columns[0].name, "id");
        assert_eq!(r.key_columns(), vec!["id"]);
        assert!(!r.columns[1].is_key, "only id is the replica identity");
        assert_eq!(r.columns[2].type_id, 1700);
    }

    #[test]
    fn an_insert_carries_every_column_in_declaration_order() {
        let bytes = Build::new(b'I')
            .u32(16_384)
            .u8(b'N')
            .i16(3)
            .text("7")
            .text("acme")
            .null()
            .done();
        let Change::Insert { relation_id, tuple } = decode(&bytes).unwrap() else {
            panic!("expected an insert");
        };
        assert_eq!(relation_id, 16_384);
        assert_eq!(
            tuple,
            vec![
                Value::Text("7".into()),
                Value::Text("acme".into()),
                Value::Null
            ]
        );
    }

    #[test]
    fn an_unchanged_toast_column_is_not_a_null() {
        // The classic pgoutput data-loss bug. A large column that did not
        // change is omitted from the UPDATE; writing it through as NULL
        // destroys a value Postgres still holds and never mentions again.
        let bytes = Build::new(b'U')
            .u32(16_384)
            .u8(b'N')
            .i16(3)
            .text("7")
            .unchanged()
            .text("42.00")
            .done();
        let Change::Update { tuple, old, .. } = decode(&bytes).unwrap() else {
            panic!("expected an update");
        };
        assert!(old.is_none(), "DEFAULT identity with an unchanged key");
        assert_eq!(tuple[1], Value::Unchanged);
        assert_ne!(tuple[1], Value::Null);
    }

    #[test]
    fn an_update_that_changes_the_key_carries_the_old_one() {
        let bytes = Build::new(b'U')
            .u32(16_384)
            .u8(b'K')
            .i16(3)
            .text("7")
            .null()
            .null()
            .u8(b'N')
            .i16(3)
            .text("8")
            .text("acme")
            .text("42.00")
            .done();
        let Change::Update { old, tuple, .. } = decode(&bytes).unwrap() else {
            panic!("expected an update");
        };
        let (which, key) = old.expect("a changed key must arrive");
        assert_eq!(which, OldTuple::Key);
        assert_eq!(key[0], Value::Text("7".into()));
        assert_eq!(tuple[0], Value::Text("8".into()));
    }

    #[test]
    fn replica_identity_full_says_so_rather_than_looking_like_a_key() {
        // The difference decides whether a downstream row can be reconstructed
        // or only located. Reading 'O' as 'K' loses that distinction silently.
        let bytes = Build::new(b'D')
            .u32(16_384)
            .u8(b'O')
            .i16(3)
            .text("7")
            .text("acme")
            .text("42.00")
            .done();
        let Change::Delete {
            old: (which, row), ..
        } = decode(&bytes).unwrap()
        else {
            panic!("expected a delete");
        };
        assert_eq!(which, OldTuple::Full);
        assert_eq!(row.len(), 3);
    }

    #[test]
    fn a_delete_without_a_key_marker_is_refused() {
        // Rather than decoded as an empty tuple: a delete whose key cannot be
        // read must not reach a downstream store as "delete nothing" or, worse,
        // "delete everything".
        let bytes = Build::new(b'D').u32(16_384).u8(b'X').i16(0).done();
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn begin_and_commit_bracket_a_transaction() {
        let begin = Build::new(b'B')
            .u64(0x1_0000_0000)
            .i64(700)
            .u32(1234)
            .done();
        let Change::Begin { final_lsn, xid, .. } = decode(&begin).unwrap() else {
            panic!("expected a begin");
        };
        assert_eq!(final_lsn, 0x1_0000_0000);
        assert_eq!(xid, 1234);

        let commit = Build::new(b'C')
            .u8(0)
            .u64(0x1_0000_0000)
            .u64(0x1_0000_0100)
            .i64(700)
            .done();
        let Change::Commit { end_lsn, .. } = decode(&commit).unwrap() else {
            panic!("expected a commit");
        };
        assert_eq!(end_lsn, 0x1_0000_0100);
    }

    #[test]
    fn a_truncate_reports_its_flags_apart() {
        let bytes = Build::new(b'T').i32(2).u8(3).u32(16_384).u32(16_385).done();
        let Change::Truncate {
            relation_ids,
            cascade,
            restart_identity,
        } = decode(&bytes).unwrap()
        else {
            panic!("expected a truncate");
        };
        assert_eq!(relation_ids, vec![16_384, 16_385]);
        assert!(cascade && restart_identity);
    }

    #[test]
    fn a_truncated_message_is_an_error_not_a_panic() {
        // Every field here is sized by a remote party. Slice indexing would
        // turn a short read into a process abort reachable from the network.
        let full = relation_message();
        for cut in 0..full.len() {
            let _ = decode(&full[..cut]);
        }
    }

    #[test]
    fn an_unknown_tag_is_reported_with_the_tag() {
        let err = decode(b"Z\x00\x00\x00\x01").unwrap_err();
        assert!(err.to_string().contains('Z'), "{err}");
    }

    #[test]
    fn an_empty_message_does_not_panic() {
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn relations_pair_values_with_their_column_names() {
        let mut relations = Relations::new();
        let Change::Relation(r) = decode(&relation_message()).unwrap() else {
            unreachable!()
        };
        relations.insert(r);
        let tuple = vec![
            Value::Text("7".into()),
            Value::Text("acme".into()),
            Value::Null,
        ];
        let named = relations.name_values(16_384, &tuple).unwrap();
        assert_eq!(named[0].0, "id");
        assert_eq!(named[2].0, "total");
        // An id nobody announced is a stream bug, reported as absent rather
        // than answered with positional keys.
        assert!(relations.name_values(99, &tuple).is_none());
    }

    #[test]
    fn an_lsn_round_trips_through_the_postgres_format() {
        assert_eq!(format_lsn(0), "0/0");
        assert_eq!(format_lsn(0x1_0000_0000), "1/0");
        assert_eq!(format_lsn(0x16B3_74A8), "0/16B374A8");
        for text in ["0/0", "1/0", "0/16B374A8", "2A/FFFFFFFF"] {
            assert_eq!(format_lsn(parse_lsn(text).unwrap()), text);
        }
        assert_eq!(parse_lsn("nonsense"), None);
        assert_eq!(parse_lsn("0/"), None);
        // A low half wider than 32 bits would silently carry into the high
        // half and name a completely different position in the WAL.
        assert_eq!(parse_lsn("0/1FFFFFFFF"), None);
    }

    #[test]
    fn a_postgres_timestamp_lands_in_the_right_millennium() {
        // Off by this offset and every event is dated 1970, which no dashboard
        // would question and no test would catch without the constant.
        assert_eq!(pg_time_to_unix_ms(0), 946_684_800_000);
        assert_eq!(pg_time_to_unix_ms(1_000_000), 946_684_801_000);
    }
}
