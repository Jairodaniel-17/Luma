//! The setup a logical replication stream needs, and the checks that make its
//! failure modes visible before they become outages.
//!
//! Two objects, with very different lifetimes:
//!
//! - A **publication** says which tables are in scope. It is cheap and holds
//!   nothing.
//! - A **replication slot** is Postgres's promise to keep WAL until a consumer
//!   has read it. It is the opposite of cheap: an abandoned slot pins WAL
//!   forever and fills the primary's disk. That is the failure mode replication
//!   slots are known for, and every warning below exists because of it.

use anyhow::{anyhow, bail, Context, Result};

use super::conn::PgConnection;
use super::pgoutput::parse_lsn;

/// What `IDENTIFY_SYSTEM` reports.
#[derive(Debug, Clone)]
pub struct SystemIdentity {
    pub system_id: String,
    pub timeline: i64,
    pub current_lsn: u64,
    pub database: Option<String>,
}

/// Ask the server who it is and where its WAL currently ends.
///
/// The system id is the one thing that distinguishes a server from a restored
/// copy of itself. Resuming from a saved LSN against a *different* system means
/// resuming at a position that refers to somebody else's WAL, and the numbers
/// will look entirely plausible.
pub async fn identify_system(conn: &mut PgConnection) -> Result<SystemIdentity> {
    let rows = conn
        .simple_query("IDENTIFY_SYSTEM")
        .await
        .context("IDENTIFY_SYSTEM failed — is this a replication connection?")?;
    let row = rows
        .first()
        .ok_or_else(|| anyhow!("IDENTIFY_SYSTEM returned no rows"))?;

    let text = |i: usize| row.get(i).cloned().flatten();
    let lsn_text = text(2).ok_or_else(|| anyhow!("IDENTIFY_SYSTEM returned no LSN"))?;

    Ok(SystemIdentity {
        system_id: text(0).unwrap_or_default(),
        timeline: text(1).and_then(|t| t.parse().ok()).unwrap_or(0),
        current_lsn: parse_lsn(&lsn_text).ok_or_else(|| {
            anyhow!("IDENTIFY_SYSTEM returned an LSN we cannot parse: {lsn_text}")
        })?,
        database: text(3),
    })
}

/// A slot, as `pg_replication_slots` describes it.
#[derive(Debug, Clone)]
pub struct SlotStatus {
    pub name: String,
    pub active: bool,
    pub confirmed_flush_lsn: Option<u64>,
    /// WAL the primary is holding on this slot's behalf, in bytes.
    pub retained_bytes: Option<i64>,
}

/// Look up a slot. `None` when it does not exist.
///
/// Runs on an ordinary connection — a replication connection cannot query the
/// catalog.
pub async fn slot_status(conn: &mut PgConnection, slot: &str) -> Result<Option<SlotStatus>> {
    let sql = format!(
        "SELECT slot_name, active, confirmed_flush_lsn, \
         pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn)::bigint \
         FROM pg_replication_slots WHERE slot_name = '{}'",
        slot.replace('\'', "''")
    );
    let rows = conn.simple_query(&sql).await?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let text = |i: usize| row.get(i).cloned().flatten();
    Ok(Some(SlotStatus {
        name: text(0).unwrap_or_else(|| slot.to_string()),
        active: text(1).as_deref() == Some("t"),
        confirmed_flush_lsn: text(2).as_deref().and_then(parse_lsn),
        retained_bytes: text(3).and_then(|v| v.parse().ok()),
    }))
}

/// Create a logical slot if it is not there, and report where it starts.
///
/// The returned LSN is the *consistent point*: everything committed before it
/// is already in the tables, everything after arrives on the stream. That is
/// the boundary a backfill has to use — a `COPY` taken at any other moment
/// either misses rows or double-counts them, and both look like the connector
/// working.
///
/// Returns `(start_lsn, created)`. `created` false means the slot already
/// existed and `start_lsn` is where it left off.
pub async fn ensure_slot(conn: &mut PgConnection, slot: &str) -> Result<(u64, bool)> {
    let quoted = slot.replace('"', "\"\"");
    let sql = format!("CREATE_REPLICATION_SLOT \"{quoted}\" LOGICAL pgoutput NOEXPORT_SNAPSHOT");

    match conn.simple_query(&sql).await {
        Ok(rows) => {
            let lsn = rows
                .first()
                .and_then(|r| r.get(1).cloned().flatten())
                .and_then(|t| parse_lsn(&t))
                .ok_or_else(|| anyhow!("CREATE_REPLICATION_SLOT returned no consistent point"))?;
            Ok((lsn, true))
        }
        Err(e) => {
            let text = e.to_string();
            // 42710, duplicate_object. Matched on the wording because the
            // replication protocol's error is what we have here, and creating
            // a slot that exists is the ordinary restart path, not a failure.
            if text.contains("already exists") {
                let existing = slot_status_via_new_query(conn, slot).await?;
                Ok((existing, false))
            } else {
                Err(e)
            }
        }
    }
}

/// The confirmed flush LSN of an existing slot, read over the replication
/// connection.
///
/// `READ_REPLICATION_SLOT` rather than the catalog, because by the time this is
/// called the connection is already in replication mode and cannot run a
/// `SELECT`.
async fn slot_status_via_new_query(conn: &mut PgConnection, slot: &str) -> Result<u64> {
    let quoted = slot.replace('"', "\"\"");
    let rows = conn
        .simple_query(&format!("READ_REPLICATION_SLOT \"{quoted}\""))
        .await
        .context("could not read back an existing replication slot")?;
    // Postgres 15+ answers (slot_type, restart_lsn, restart_tli). A NULL
    // restart_lsn means the slot exists but has never been read from, and 0 —
    // "wherever the slot left off" — is exactly right for that.
    Ok(rows
        .first()
        .and_then(|r| r.get(1).cloned().flatten())
        .and_then(|t| parse_lsn(&t))
        .unwrap_or(0))
}

/// Drop a slot, releasing the WAL it pins.
///
/// The only way to undo `ensure_slot`, and the thing an operator has to do
/// after removing a connector. Leaving it is how a disk fills up weeks later
/// with nothing in the logs connecting the two events.
pub async fn drop_slot(conn: &mut PgConnection, slot: &str) -> Result<()> {
    let quoted = slot.replace('"', "\"\"");
    conn.simple_query(&format!("DROP_REPLICATION_SLOT \"{quoted}\" WAIT"))
        .await
        .with_context(|| format!("could not drop replication slot {slot:?}"))?;
    Ok(())
}

/// Create a publication over the named tables if it is not there.
///
/// Tables are named explicitly rather than with `FOR ALL TABLES`: a publication
/// that follows every table means every future table too, including ones
/// nobody meant to expose to a search index.
pub async fn ensure_publication(
    conn: &mut PgConnection,
    publication: &str,
    tables: &[String],
) -> Result<bool> {
    if tables.is_empty() {
        bail!("a publication with no tables would produce a stream with no changes");
    }
    let exists = conn
        .simple_query(&format!(
            "SELECT 1 FROM pg_publication WHERE pubname = '{}'",
            publication.replace('\'', "''")
        ))
        .await?;
    if !exists.is_empty() {
        return Ok(false);
    }

    let list = tables
        .iter()
        .map(|t| qualified_identifier(t))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    conn.simple_query(&format!(
        "CREATE PUBLICATION \"{}\" FOR TABLE {list}",
        publication.replace('"', "\"\"")
    ))
    .await
    .with_context(|| format!("could not create publication {publication:?}"))?;
    Ok(true)
}

/// A table's replica identity, which decides what an UPDATE or DELETE can say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicaIdentity {
    pub table: String,
    /// `d` default (primary key), `n` nothing, `f` full, `i` a named index.
    pub kind: char,
    pub has_primary_key: bool,
}

impl ReplicaIdentity {
    /// Whether rows from this table can be updated and deleted downstream.
    ///
    /// `d` without a primary key is the trap: the table replicates INSERTs
    /// perfectly and then Postgres refuses every UPDATE and DELETE on it with
    /// an error that names the publication, not the missing key.
    pub fn is_usable(&self) -> bool {
        match self.kind {
            'd' => self.has_primary_key,
            'f' | 'i' => true,
            _ => false,
        }
    }

    /// What to tell an operator when it is not usable.
    pub fn advice(&self) -> String {
        match self.kind {
            'd' => format!(
                "{} has REPLICA IDENTITY DEFAULT but no primary key: INSERTs will replicate and \
                 Postgres will refuse every UPDATE and DELETE on it. Add a primary key, or set \
                 REPLICA IDENTITY FULL.",
                self.table
            ),
            'n' => format!(
                "{} has REPLICA IDENTITY NOTHING: UPDATEs and DELETEs carry no key and cannot be \
                 applied downstream. Set REPLICA IDENTITY FULL or add a primary key.",
                self.table
            ),
            other => format!(
                "{} has an unrecognised replica identity {other:?}",
                self.table
            ),
        }
    }
}

/// Check the replica identity of every table before streaming starts.
///
/// At connect time rather than on the first UPDATE. The difference is a message
/// during setup versus a stream that has been quietly dropping updates in
/// production, which is only noticed when somebody compares a count.
pub async fn check_replica_identities(
    conn: &mut PgConnection,
    tables: &[String],
) -> Result<Vec<ReplicaIdentity>> {
    let mut out = Vec::with_capacity(tables.len());
    for table in tables {
        let (schema, name) = split_qualified(table);
        let rows = conn
            .simple_query(&format!(
                "SELECT c.relreplident, \
                 EXISTS (SELECT 1 FROM pg_index i WHERE i.indrelid = c.oid AND i.indisprimary) \
                 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace \
                 WHERE n.nspname = '{}' AND c.relname = '{}'",
                schema.replace('\'', "''"),
                name.replace('\'', "''")
            ))
            .await?;
        let row = rows
            .first()
            .ok_or_else(|| anyhow!("no table named {table:?} in this database"))?;
        out.push(ReplicaIdentity {
            table: table.clone(),
            kind: row
                .first()
                .cloned()
                .flatten()
                .and_then(|v| v.chars().next())
                .unwrap_or('?'),
            has_primary_key: row.get(1).cloned().flatten().as_deref() == Some("t"),
        });
    }
    Ok(out)
}

/// `schema.table` split, defaulting the schema to `public`.
pub fn split_qualified(table: &str) -> (String, String) {
    match table.split_once('.') {
        Some((s, t)) => (s.to_string(), t.to_string()),
        None => ("public".to_string(), table.to_string()),
    }
}

/// Quote `schema.table` for use in DDL.
fn qualified_identifier(table: &str) -> Result<String> {
    let (schema, name) = split_qualified(table);
    if name.is_empty() {
        bail!("{table:?} names no table");
    }
    Ok(format!(
        "\"{}\".\"{}\"",
        schema.replace('"', "\"\""),
        name.replace('"', "\"\"")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_table_name_lands_in_public() {
        assert_eq!(
            split_qualified("orders"),
            ("public".into(), "orders".into())
        );
        assert_eq!(
            split_qualified("sales.orders"),
            ("sales".into(), "orders".into())
        );
    }

    #[test]
    fn a_qualified_identifier_is_quoted_on_both_halves() {
        assert_eq!(
            qualified_identifier("sales.orders").unwrap(),
            "\"sales\".\"orders\""
        );
        // Quoting only the table half is the mistake that lets a schema name
        // close the quoting and continue the statement.
        assert_eq!(
            qualified_identifier("we\"ird.table").unwrap(),
            "\"we\"\"ird\".\"table\""
        );
        assert!(qualified_identifier("schema.").is_err());
    }

    #[test]
    fn default_identity_without_a_primary_key_is_reported_as_unusable() {
        // The trap: the table replicates INSERTs perfectly, and Postgres then
        // refuses every UPDATE and DELETE with an error naming the publication
        // rather than the missing key.
        let identity = ReplicaIdentity {
            table: "public.events".into(),
            kind: 'd',
            has_primary_key: false,
        };
        assert!(!identity.is_usable());
        assert!(
            identity.advice().contains("primary key"),
            "{}",
            identity.advice()
        );

        let with_key = ReplicaIdentity {
            has_primary_key: true,
            ..identity.clone()
        };
        assert!(with_key.is_usable());
    }

    #[test]
    fn replica_identity_nothing_is_never_usable() {
        let identity = ReplicaIdentity {
            table: "public.log".into(),
            kind: 'n',
            has_primary_key: true,
        };
        assert!(
            !identity.is_usable(),
            "a primary key does not help when the identity is NOTHING"
        );
        assert!(identity.advice().contains("NOTHING"));
    }

    #[test]
    fn full_and_index_identities_are_usable() {
        for kind in ['f', 'i'] {
            let identity = ReplicaIdentity {
                table: "public.t".into(),
                kind,
                has_primary_key: false,
            };
            assert!(identity.is_usable(), "identity {kind} should be usable");
        }
    }
}
