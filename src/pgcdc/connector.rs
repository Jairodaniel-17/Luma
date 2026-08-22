//! `luma connect postgres` — the connector itself (W4.2).
//!
//! Turns a Postgres table into a searchable collection and keeps it that way.
//! What it does **not** do is write back: Postgres stays the transactional
//! source of truth, and every document produced here carries the source table
//! and primary key so an application can go read the canonical row (W4.3).
//!
//! Three phases, in this order and for a reason:
//!
//! 1. **Setup.** Publication, slot, and a replica-identity check on every mapped
//!    table. The check is here rather than on the first UPDATE because a table
//!    with no usable identity replicates INSERTs perfectly and then silently
//!    stops accepting changes.
//! 2. **Backfill.** `COPY` of everything already in the table, taken at the
//!    slot's consistent point. The slot is created *first* so that changes made
//!    during the backfill are queued rather than lost — the ordering is the
//!    whole correctness argument, not an implementation detail.
//! 3. **Stream.** Changes from the slot, applied as upserts and deletes.
//!
//! Resumption uses the same shape as `state_db`'s `applied_offset`: a position
//! is persisted, and anything at or before it is skipped on replay. Here the
//! position is an LSN, stored next to the system id of the server that issued
//! it — an LSN from a different server names a perfectly plausible place in
//! somebody else's WAL.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::conn::{PgConfig, PgConnection, StreamMessage};
use super::pgoutput::{decode, format_lsn, pg_time_to_unix_ms, Change, Relations, Value};
use super::slots;
use crate::engine::hub::LumaDatabase;

/// How one table becomes one collection.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TableMapping {
    /// `schema.table`; a bare name means `public`.
    pub table: String,
    /// The Luma namespace the rows land in.
    pub namespace: String,
    /// Columns that make the document id. Empty means "use the replica
    /// identity", which is what the stream itself reports and therefore the
    /// only choice that cannot disagree with Postgres.
    #[serde(default)]
    pub id_columns: Vec<String>,
    /// Columns concatenated into the text that gets embedded. Empty means every
    /// textual column, which is a reasonable default for a table nobody has
    /// described yet and a poor one for a table with a large blob in it.
    #[serde(default)]
    pub text_columns: Vec<String>,
    /// Columns to leave out of the document entirely. For secrets, and for the
    /// columns whose only purpose is upstream bookkeeping.
    #[serde(default)]
    pub skip_columns: Vec<String>,
}

/// A connector's whole configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConnectorConfig {
    /// Names the connector's own state. Changing it starts over from scratch.
    pub name: String,
    /// `postgres://user:pass@host:port/db?sslmode=require`
    pub url: String,
    #[serde(default = "default_slot")]
    pub slot: String,
    #[serde(default = "default_publication")]
    pub publication: String,
    pub tables: Vec<TableMapping>,
    /// Copy what is already in the tables before streaming. On by default:
    /// starting a connector and getting only future changes is almost never
    /// what somebody meant, and the mistake is invisible until a search comes
    /// back short.
    #[serde(default = "default_true")]
    pub backfill: bool,
    /// How often to tell Postgres how far we have got. This is what lets it
    /// release WAL.
    #[serde(default = "default_flush_secs")]
    pub flush_interval_secs: u64,
}

fn default_slot() -> String {
    "luma_cdc".to_string()
}
fn default_publication() -> String {
    "luma_cdc".to_string()
}
fn default_true() -> bool {
    true
}
fn default_flush_secs() -> u64 {
    10
}

impl ConnectorConfig {
    /// Read a connector definition from a TOML file.
    pub fn from_toml(text: &str) -> Result<ConnectorConfig> {
        let config: ConnectorConfig =
            toml::from_str(text).context("the connector configuration does not parse")?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("a connector needs a name: it is what its saved position is filed under");
        }
        if self.tables.is_empty() {
            bail!("a connector with no tables would follow a stream with no changes in it");
        }
        let mut seen = std::collections::HashSet::new();
        for mapping in &self.tables {
            let (schema, name) = slots::split_qualified(&mapping.table);
            if name.is_empty() {
                bail!("{:?} names no table", mapping.table);
            }
            if mapping.namespace.trim().is_empty() {
                bail!("{}.{} has no namespace to land in", schema, name);
            }
            if !seen.insert(format!("{schema}.{name}")) {
                bail!(
                    "{schema}.{name} is mapped twice; the second mapping would overwrite the \
                     first and nothing would report it"
                );
            }
        }
        Ok(())
    }

    /// The tables, fully qualified, in configuration order.
    pub fn qualified_tables(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|m| {
                let (schema, name) = slots::split_qualified(&m.table);
                format!("{schema}.{name}")
            })
            .collect()
    }
}

/// What the connector has persisted about where it is.
///
/// The system id travels with the LSN on purpose. Restoring a Postgres backup
/// and pointing the connector at the copy would otherwise resume at a position
/// that is arithmetically fine and refers to a different history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub lsn: u64,
    pub system_id: String,
    /// Set when the upstream table was truncated. See `Change::Truncate` below:
    /// the derived collection is stale and only an operator should decide what
    /// to do about it.
    #[serde(default)]
    pub stale: bool,
    pub updated_at_ms: i64,
}

/// Where a connector's checkpoint lives in the KV store.
///
/// In the KV store rather than a file: it then goes through the WAL and the
/// snapshots like every other piece of state, which means a restore brings the
/// connector's position back with the data it corresponds to.
pub fn checkpoint_key(name: &str) -> String {
    format!("pgcdc:{name}:checkpoint")
}

/// One connector, running.
pub struct Connector {
    config: ConnectorConfig,
    hub: Arc<LumaDatabase>,
    /// Mapping by qualified table name, for the lookup a row message needs.
    by_table: HashMap<String, TableMapping>,
}

/// What one pass of the connector did. Returned rather than only logged so a
/// test — and `luma connect postgres --once` — can assert on it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunReport {
    pub backfilled: u64,
    pub inserted: u64,
    pub updated: u64,
    pub deleted: u64,
    /// Changes skipped because they were at or before the saved LSN.
    pub already_applied: u64,
    /// Rows a mapping could not turn into a document, with the reason logged
    /// once per table rather than once per row.
    pub skipped: u64,
    pub last_lsn: u64,
    pub truncated_tables: Vec<String>,
}

impl Connector {
    pub fn new(config: ConnectorConfig, hub: Arc<LumaDatabase>) -> Result<Connector> {
        config.validate()?;
        let by_table = config
            .tables
            .iter()
            .map(|m| {
                let (schema, name) = slots::split_qualified(&m.table);
                (format!("{schema}.{name}"), m.clone())
            })
            .collect();
        Ok(Connector {
            config,
            hub,
            by_table,
        })
    }

    /// The saved position, or `None` for a connector that has never run.
    pub fn checkpoint(&self) -> Option<Checkpoint> {
        let item = self
            .hub
            .engine
            .get_state(&checkpoint_key(&self.config.name))?;
        // A raw payload under this key is not a checkpoint. `as_json` returning
        // None is the honest answer, and treating it as "no checkpoint" means a
        // connector starts over rather than resuming from garbage.
        serde_json::from_value(item.value.as_json()?.clone()).ok()
    }

    fn save_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        let value = serde_json::to_value(checkpoint)?;
        self.hub
            .engine
            .put_state(checkpoint_key(&self.config.name), value, None, None)
            .map_err(|e| anyhow!("could not persist the connector checkpoint: {e}"))?;
        Ok(())
    }

    /// Set up the publication and slot, and refuse to go on if a mapped table
    /// cannot produce usable updates.
    ///
    /// Returns the consistent point of a freshly created slot, or `None` when
    /// the slot already existed — which is what says whether a backfill is due.
    pub async fn prepare(&self) -> Result<Option<u64>> {
        let mut conn = self.connect(true).await?;
        let identity = slots::identify_system(&mut conn).await?;

        if let Some(saved) = self.checkpoint() {
            if saved.system_id != identity.system_id {
                bail!(
                    "this connector last ran against Postgres system {} and is now pointed at {}. \
                     Resuming would continue from LSN {} in a different server's history, which \
                     would look entirely plausible. Delete the checkpoint at {} to start over.",
                    saved.system_id,
                    identity.system_id,
                    format_lsn(saved.lsn),
                    checkpoint_key(&self.config.name)
                );
            }
        }

        let tables = self.config.qualified_tables();
        let identities = slots::check_replica_identities(&mut conn, &tables).await?;
        let unusable: Vec<String> = identities
            .iter()
            .filter(|i| !i.is_usable())
            .map(|i| i.advice())
            .collect();
        if !unusable.is_empty() {
            bail!(
                "these tables cannot produce usable updates:\n  {}",
                unusable.join("\n  ")
            );
        }

        slots::ensure_publication(&mut conn, &self.config.publication, &tables).await?;
        let (lsn, created) = slots::ensure_slot(&mut conn, &self.config.slot).await?;
        Ok(if created { Some(lsn) } else { None })
    }

    async fn connect(&self, replication: bool) -> Result<PgConnection> {
        let mut config = PgConfig::from_url(&self.config.url)?;
        config.replication = replication;
        PgConnection::connect(&config).await
    }

    /// Copy what is already in the mapped tables.
    ///
    /// Called after the slot exists, never before. A backfill taken first would
    /// miss every change made between the copy and the slot's creation, and
    /// those rows are gone with no trace that they were ever expected.
    pub async fn backfill(&self) -> Result<u64> {
        let mut conn = self.connect(false).await?;
        let mut total = 0u64;

        for mapping in &self.config.tables {
            let (schema, name) = slots::split_qualified(&mapping.table);
            // A COPY is not part of the replication stream, so there is no
            // `Relation` message to read the key from. Reading the same
            // definition Postgres will use for the stream is what keeps a
            // backfilled document and a later update on the *same* id — with a
            // different key the update creates a second document and the first
            // one never goes away.
            let key_columns = if mapping.id_columns.is_empty() {
                slots::identity_columns(&mut conn, &schema, &name).await?
            } else {
                mapping.id_columns.clone()
            };
            if key_columns.is_empty() {
                bail!(
                    "{schema}.{name} has no primary key and no id_columns, so a backfilled row                      has nothing to be identified by"
                );
            }
            let columns = self.columns_of(&mut conn, &schema, &name).await?;
            let selected: Vec<&String> = columns
                .iter()
                .filter(|c| !mapping.skip_columns.contains(c))
                .collect();
            if selected.is_empty() {
                bail!(
                    "every column of {schema}.{name} is in skip_columns; there would be nothing \
                     left to index"
                );
            }
            let list = selected
                .iter()
                .map(|c| format!("\"{}\"", c.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "COPY (SELECT {list} FROM \"{}\".\"{}\") TO STDOUT",
                schema.replace('"', "\"\""),
                name.replace('"', "\"\"")
            );

            let names: Vec<String> = selected.iter().map(|c| (*c).clone()).collect();
            // Collected rather than ingested inside the callback: ingestion is
            // async and the callback is not, and holding the COPY open across
            // an embedding call would keep a transaction alive for as long as
            // the provider takes.
            let mut batch: Vec<Vec<Option<String>>> = Vec::new();
            let copied = conn
                .copy_out(&sql, |row| {
                    batch.push(row);
                    Ok(())
                })
                .await
                .with_context(|| format!("backfilling {schema}.{name}"))?;

            for row in batch {
                let values: Vec<(String, Value)> = names
                    .iter()
                    .zip(row)
                    .map(|(n, v)| {
                        (
                            n.clone(),
                            match v {
                                Some(text) => Value::Text(text),
                                None => Value::Null,
                            },
                        )
                    })
                    .collect();
                self.ingest_with_key(mapping, &values, &key_columns, "backfill")
                    .await?;
            }
            tracing::info!(
                table = %format!("{schema}.{name}"),
                namespace = %mapping.namespace,
                rows = copied,
                "backfilled"
            );
            total += copied;
        }
        Ok(total)
    }

    /// The column names of a table, in declaration order.
    async fn columns_of(
        &self,
        conn: &mut PgConnection,
        schema: &str,
        name: &str,
    ) -> Result<Vec<String>> {
        let rows = conn
            .simple_query(&format!(
                "SELECT a.attname FROM pg_attribute a \
                 JOIN pg_class c ON c.oid = a.attrelid \
                 JOIN pg_namespace n ON n.oid = c.relnamespace \
                 WHERE n.nspname = '{}' AND c.relname = '{}' \
                 AND a.attnum > 0 AND NOT a.attisdropped \
                 ORDER BY a.attnum",
                schema.replace('\'', "''"),
                name.replace('\'', "''")
            ))
            .await?;
        let columns: Vec<String> = rows
            .into_iter()
            .filter_map(|r| r.into_iter().next().flatten())
            .collect();
        if columns.is_empty() {
            bail!("no table named {schema}.{name} in this database");
        }
        Ok(columns)
    }

    /// Follow the stream until `budget` has passed or `max_changes` applied.
    ///
    /// Bounded rather than endless so the same code serves the long-running
    /// task and a single pass. An unbounded loop would need a second
    /// implementation to be testable, and a second implementation is a second
    /// set of bugs.
    pub async fn stream_once(&self, budget: Duration, max_changes: u64) -> Result<RunReport> {
        let mut report = RunReport::default();
        let mut conn = self.connect(true).await?;
        let identity = slots::identify_system(&mut conn).await?;
        let saved = self.checkpoint();
        let start_lsn = saved.as_ref().map(|c| c.lsn).unwrap_or(0);
        let applied_through = start_lsn;

        conn.start_replication(
            &self.config.slot,
            std::slice::from_ref(&self.config.publication),
            start_lsn,
        )
        .await?;

        let mut relations = Relations::new();
        // Whether the transaction currently being read is one we have already
        // stored. Postgres restarts a slot from its own confirmed position,
        // which can be behind our checkpoint, so a resumed stream ordinarily
        // begins by resending work.
        let mut replaying = false;
        let mut last_flush = Instant::now();
        let flush_every = Duration::from_secs(self.config.flush_interval_secs.max(1));
        let deadline = Instant::now() + budget;
        // The only position worth saving. A transaction is applied or it is
        // not, so a checkpoint taken between two of its rows names a place that
        // is not a resume point in either direction: telling Postgres it may
        // release WAL up to there would let it discard the rest of a
        // transaction we had not finished storing.
        //
        // This was found by a test, not by reasoning: bounding a pass by change
        // count stopped it mid-transaction, and the next pass resumed at a
        // position *inside* the transaction it had half-applied.
        let mut committed = start_lsn;
        let mut in_transaction = false;

        loop {
            // Both stopping conditions are checked at a transaction boundary.
            // Leaving a pass in the middle of one is what produced the bug
            // above; the budget is a target, not a deadline to abandon work at.
            if !in_transaction && (Instant::now() >= deadline || report.applied() >= max_changes) {
                break;
            }
            let remaining = if in_transaction {
                // Finishing the transaction we are inside takes priority, but
                // not forever: a server that dies mid-transaction must not hang
                // the pass.
                Duration::from_secs(30)
            } else {
                deadline.saturating_duration_since(Instant::now())
            };
            let message = match tokio::time::timeout(remaining, conn.next_stream_message()).await {
                Ok(Ok(m)) => m,
                Ok(Err(e)) => return Err(e),
                // Nothing arrived inside the budget. Not an error: a quiet
                // database is the normal case.
                Err(_) => break,
            };

            match message {
                StreamMessage::Keepalive {
                    reply_requested, ..
                } => {
                    // A keepalive carries the server's WAL position, not ours.
                    // Answering it with that number would claim we had stored
                    // changes we have not seen, so what goes back is the last
                    // commit we actually applied.
                    if reply_requested {
                        conn.send_standby_status(committed, committed, committed, false)
                            .await?;
                        last_flush = Instant::now();
                    }
                }
                StreamMessage::XLogData { data, .. } => {
                    let change = decode(&data)?;
                    match &change {
                        Change::Begin { .. } => in_transaction = true,
                        Change::Commit { .. } => in_transaction = false,
                        _ => {}
                    }
                    self.apply(
                        &change,
                        &mut relations,
                        applied_through,
                        &mut replaying,
                        &mut report,
                    )
                    .await?;
                    if let Change::Commit { end_lsn, .. } = change {
                        committed = committed.max(end_lsn);
                        report.last_lsn = committed;
                    }
                }
            }

            if !in_transaction && last_flush.elapsed() >= flush_every && committed > 0 {
                self.flush(&mut conn, committed, &identity.system_id, &report)
                    .await?;
                last_flush = Instant::now();
            }
        }

        if committed > start_lsn {
            self.flush(&mut conn, committed, &identity.system_id, &report)
                .await?;
        }
        report.last_lsn = committed;
        Ok(report)
    }

    /// Persist the position and tell Postgres about it, in that order.
    ///
    /// Ours first. If the process dies between the two, Postgres resends
    /// changes we have already applied and the LSN guard skips them — every
    /// apply here is an upsert or a delete by key, so a repeat is harmless. The
    /// other order risks Postgres releasing WAL for changes we never stored.
    async fn flush(
        &self,
        conn: &mut PgConnection,
        lsn: u64,
        system_id: &str,
        report: &RunReport,
    ) -> Result<()> {
        self.save_checkpoint(&Checkpoint {
            lsn,
            system_id: system_id.to_string(),
            stale: !report.truncated_tables.is_empty(),
            updated_at_ms: now_ms(),
        })?;
        conn.send_standby_status(lsn, lsn, lsn, false).await
    }

    async fn apply(
        &self,
        change: &Change,
        relations: &mut Relations,
        applied_through: u64,
        replaying: &mut bool,
        report: &mut RunReport,
    ) -> Result<()> {
        // A `Relation` is bookkeeping, not a change: it has to be recorded even
        // while replaying, because the rows *after* the replayed part refer to
        // it and it is announced only once per stream.
        if let Change::Relation(relation) = change {
            relations.insert(relation.clone());
            return Ok(());
        }

        match change {
            Change::Begin { final_lsn, .. } => {
                // The whole transaction is at or before what we have stored, so
                // none of it needs applying. Judged as a unit rather than per
                // row: a transaction is the granularity Postgres commits at,
                // and its `final_lsn` is known from its first message.
                *replaying = applied_through > 0 && *final_lsn <= applied_through;
            }
            Change::Commit { .. } => *replaying = false,
            _ if *replaying => {
                report.already_applied += 1;
                return Ok(());
            }
            _ => {}
        }

        match change {
            // Handled above; listed so adding a variant is a compile error here
            // rather than a change that silently does nothing.
            Change::Relation(_) => {}
            Change::Insert { relation_id, tuple } => {
                if let Some((mapping, values)) = self.resolve(relations, *relation_id, tuple) {
                    self.ingest(mapping, &values, Some(relations), *relation_id, "insert")
                        .await?;
                    report.inserted += 1;
                } else {
                    report.skipped += 1;
                }
            }
            Change::Update {
                relation_id, tuple, ..
            } => {
                if let Some((mapping, values)) = self.resolve(relations, *relation_id, tuple) {
                    self.ingest(mapping, &values, Some(relations), *relation_id, "update")
                        .await?;
                    report.updated += 1;
                } else {
                    report.skipped += 1;
                }
            }
            Change::Delete {
                relation_id,
                old: (_, key),
            } => {
                if let Some((mapping, values)) = self.resolve(relations, *relation_id, key) {
                    let id = document_id(mapping, relations, *relation_id, &values)?;
                    let doc_key = format!("doc:{}:{}", mapping.namespace, id);
                    let _ = self.hub.engine.delete_state(&doc_key);
                    report.deleted += 1;
                } else {
                    report.skipped += 1;
                }
            }
            Change::Truncate { relation_ids, .. } => {
                // Not applied. A TRUNCATE upstream means the table is empty, and
                // acting on it would mean deleting a whole derived collection
                // from a single WAL message. That is an operator's decision, so
                // the checkpoint is marked stale and this says so loudly rather
                // than either destroying data or pretending nothing happened.
                for id in relation_ids {
                    let name = relations
                        .get(*id)
                        .map(|r| r.qualified())
                        .unwrap_or_else(|| format!("relation {id}"));
                    tracing::warn!(
                        table = %name,
                        "TRUNCATE arrived on the replication stream. The derived collection is \
                         now stale and Luma will not empty it automatically. Re-run the \
                         connector with backfill after deciding what should happen."
                    );
                    report.truncated_tables.push(name);
                }
            }
            Change::Begin { .. } | Change::Commit { .. } | Change::Origin { .. } => {}
            Change::Type { .. } => {}
            Change::Message { prefix, .. } => {
                tracing::debug!(%prefix, "a logical decoding message passed through");
            }
        }
        Ok(())
    }

    /// Pair a tuple with its mapping and its column names.
    ///
    /// `None` when the relation is not one we map — a publication can carry
    /// more than a connector cares about — or when it was never announced,
    /// which is a stream bug.
    fn resolve<'a>(
        &'a self,
        relations: &Relations,
        relation_id: u32,
        tuple: &[Value],
    ) -> Option<(&'a TableMapping, Vec<(String, Value)>)> {
        let relation = relations.get(relation_id)?;
        let mapping = self.by_table.get(&relation.qualified())?;
        let values = relation
            .columns
            .iter()
            .map(|c| c.name.clone())
            .zip(tuple.iter().cloned())
            .filter(|(name, _)| !mapping.skip_columns.contains(name))
            .collect();
        Some((mapping, values))
    }

    /// Turn a row into a document and hand it to the hub.
    async fn ingest(
        &self,
        mapping: &TableMapping,
        values: &[(String, Value)],
        relations: Option<&Relations>,
        relation_id: u32,
        cause: &str,
    ) -> Result<()> {
        let id = match relations {
            Some(r) => document_id(mapping, r, relation_id, values)?,
            None => document_id_from(mapping, values, &mapping.id_columns)?,
        };
        self.store(mapping, values, id, cause).await
    }

    /// The same, with the key columns named outright.
    ///
    /// The backfill's path: there is no `Relation` message to consult, so the
    /// caller has already read the identity from the catalog.
    async fn ingest_with_key(
        &self,
        mapping: &TableMapping,
        values: &[(String, Value)],
        key_columns: &[String],
        cause: &str,
    ) -> Result<()> {
        let id = document_id_from(mapping, values, key_columns)?;
        self.store(mapping, values, id, cause).await
    }

    async fn store(
        &self,
        mapping: &TableMapping,
        values: &[(String, Value)],
        id: String,
        cause: &str,
    ) -> Result<()> {
        let doc_key = format!("doc:{}:{}", mapping.namespace, id);

        // An UPDATE omits a large column that did not change. Merging with what
        // is stored is what keeps that column; writing the row as it arrived
        // would replace the value with nothing and Postgres would never mention
        // it again.
        let stored = self
            .hub
            .engine
            .get_state(&doc_key)
            .and_then(|item| item.value.as_json().and_then(|v| v.as_object()).cloned());

        let mut row = serde_json::Map::new();
        let mut unmappable: Vec<&str> = Vec::new();
        for (name, value) in values {
            match value {
                Value::Text(text) => {
                    row.insert(name.clone(), serde_json::Value::String(text.clone()));
                }
                Value::Null => {
                    row.insert(name.clone(), serde_json::Value::Null);
                }
                Value::Unchanged => {
                    if let Some(previous) = stored.as_ref().and_then(|s| s.get(name)) {
                        row.insert(name.clone(), previous.clone());
                    }
                }
                // Protocol version 1 does not send binary values, so this is
                // reachable only from a future protocol version. Skipped by
                // column rather than by row: losing one field is better than
                // losing the row, and it is reported either way.
                Value::Binary(_) => unmappable.push(name),
            }
        }
        if !unmappable.is_empty() {
            tracing::warn!(
                table = %mapping.table,
                columns = ?unmappable,
                "columns arrived in a form Luma cannot map; the rest of the row was indexed"
            );
        }

        let text = document_text(&row, &mapping.text_columns);
        // The source reference. This is W4.3's contribution and the reason the
        // connector is worth having: a hit tells the application where the
        // canonical row is, so it reads Postgres rather than trusting a copy.
        let (schema, table) = slots::split_qualified(&mapping.table);
        let source = serde_json::json!({
            "system": "postgres",
            "schema": schema,
            "table": table,
            "primary_key": id,
        });
        let mut document = row.clone();
        document.insert("_source".to_string(), source.clone());

        self.hub
            .ingest_document(
                &mapping.namespace,
                &id,
                &text,
                serde_json::Value::Object(document),
                Some(serde_json::json!({ "_source": source })),
            )
            .await
            .with_context(|| format!("indexing {} from a {cause}", mapping.table))?;
        Ok(())
    }
}

impl RunReport {
    pub fn applied(&self) -> u64 {
        self.inserted + self.updated + self.deleted
    }
}

/// The document id for a row, using the replica identity when the mapping does
/// not name its own columns.
fn document_id(
    mapping: &TableMapping,
    relations: &Relations,
    relation_id: u32,
    values: &[(String, Value)],
) -> Result<String> {
    if !mapping.id_columns.is_empty() {
        return document_id_from(mapping, values, &mapping.id_columns);
    }
    let key_columns: Vec<String> = relations
        .get(relation_id)
        .map(|r| r.key_columns().into_iter().map(str::to_string).collect())
        .unwrap_or_default();
    document_id_from(mapping, values, &key_columns)
}

/// Join the named columns into an id.
///
/// Separated by a unit separator rather than a colon or a dash: a composite key
/// whose parts contain the separator would otherwise produce the same id as a
/// different key, and two rows sharing an id means one of them disappears.
fn document_id_from(
    mapping: &TableMapping,
    values: &[(String, Value)],
    columns: &[String],
) -> Result<String> {
    if columns.is_empty() {
        bail!(
            "{} has no id columns and its replica identity named none, so there is nothing to \
             identify a row by",
            mapping.table
        );
    }
    let mut parts = Vec::with_capacity(columns.len());
    for column in columns {
        let value = values
            .iter()
            .find(|(name, _)| name == column)
            .map(|(_, v)| v);
        match value {
            Some(Value::Text(text)) => parts.push(text.clone()),
            Some(Value::Null) => bail!("{} has a NULL in its key column {column}", mapping.table),
            Some(Value::Unchanged) => bail!(
                "{} sent an unchanged marker for key column {column}, which cannot be resolved \
                 without the old tuple",
                mapping.table
            ),
            Some(Value::Binary(_)) | None => bail!(
                "{} has no usable value for key column {column}; is it in skip_columns?",
                mapping.table
            ),
        }
    }
    Ok(parts.join("\u{1f}"))
}

/// The text that gets embedded.
///
/// Named columns in the order the mapping listed them, or every string-valued
/// column when it named none. `_source` is excluded: indexing the table name
/// into every document would make every query match every row a little.
fn document_text(row: &serde_json::Map<String, serde_json::Value>, columns: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();
    if columns.is_empty() {
        for (name, value) in row {
            if name == "_source" {
                continue;
            }
            if let serde_json::Value::String(text) = value {
                parts.push(text.clone());
            }
        }
    } else {
        for column in columns {
            match row.get(column) {
                Some(serde_json::Value::String(text)) => parts.push(text.clone()),
                Some(serde_json::Value::Null) | None => {}
                Some(other) => parts.push(other.to_string()),
            }
        }
    }
    parts.join("\n")
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A commit timestamp from the stream, as Unix milliseconds.
pub fn commit_time_ms(change: &Change) -> Option<i64> {
    match change {
        Change::Commit { commit_time, .. } | Change::Begin { commit_time, .. } => {
            Some(pg_time_to_unix_ms(*commit_time))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping() -> TableMapping {
        TableMapping {
            table: "public.orders".into(),
            namespace: "orders".into(),
            id_columns: vec![],
            text_columns: vec![],
            skip_columns: vec![],
        }
    }

    #[test]
    fn a_connector_definition_parses_from_toml() {
        let config = ConnectorConfig::from_toml(
            r#"
            name = "erp"
            url = "postgres://luma:secret@db/erp?sslmode=require"

            [[tables]]
            table = "sales.orders"
            namespace = "orders"
            text_columns = ["customer", "notes"]
            skip_columns = ["internal_token"]
            "#,
        )
        .unwrap();
        assert_eq!(config.name, "erp");
        // Defaults that matter: a connector nobody configured still backfills,
        // because starting one and getting only future changes is almost never
        // what somebody meant.
        assert!(config.backfill);
        assert_eq!(config.slot, "luma_cdc");
        assert_eq!(config.qualified_tables(), vec!["sales.orders"]);
    }

    #[test]
    fn a_table_mapped_twice_is_refused() {
        // The second mapping would overwrite the first and nothing would report
        // it: same relation id, same lookup, one winner.
        let err = ConnectorConfig::from_toml(
            r#"
            name = "erp"
            url = "postgres://luma@db/erp"

            [[tables]]
            table = "orders"
            namespace = "a"

            [[tables]]
            table = "public.orders"
            namespace = "b"
            "#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("mapped twice"), "{err}");
    }

    #[test]
    fn a_connector_without_tables_is_refused() {
        let err = ConnectorConfig::from_toml(
            "name = \"erp\"\nurl = \"postgres://luma@db/erp\"\ntables = []\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no tables"), "{err}");
    }

    #[test]
    fn a_composite_id_cannot_collide_with_a_different_key() {
        // Joined on a separator that cannot appear in the parts. With ':' the
        // keys ("a:b", "c") and ("a", "b:c") produce the same id, and two rows
        // sharing an id means one of them disappears.
        let m = TableMapping {
            id_columns: vec!["one".into(), "two".into()],
            ..mapping()
        };
        let left = document_id_from(
            &m,
            &[
                ("one".into(), Value::Text("a:b".into())),
                ("two".into(), Value::Text("c".into())),
            ],
            &m.id_columns,
        )
        .unwrap();
        let right = document_id_from(
            &m,
            &[
                ("one".into(), Value::Text("a".into())),
                ("two".into(), Value::Text("b:c".into())),
            ],
            &m.id_columns,
        )
        .unwrap();
        assert_ne!(left, right);
    }

    #[test]
    fn a_null_in_a_key_column_is_refused_rather_than_stringified() {
        // "null" is a perfectly good id and a completely wrong one: every row
        // with a NULL key would collapse into the same document.
        let m = TableMapping {
            id_columns: vec!["id".into()],
            ..mapping()
        };
        let err = document_id_from(&m, &[("id".into(), Value::Null)], &m.id_columns)
            .unwrap_err()
            .to_string();
        assert!(err.contains("NULL"), "{err}");
    }

    #[test]
    fn a_row_with_no_key_at_all_is_refused_with_a_reason() {
        let err = document_id_from(&mapping(), &[], &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("nothing to identify"), "{err}");
    }

    #[test]
    fn the_replica_identity_supplies_the_id_when_the_mapping_does_not() {
        use super::super::pgoutput::{Column, Relation};
        let mut relations = Relations::new();
        relations.insert(Relation {
            id: 1,
            namespace: "public".into(),
            name: "orders".into(),
            replica_identity: b'd',
            columns: vec![
                Column {
                    name: "id".into(),
                    type_id: 23,
                    type_modifier: -1,
                    is_key: true,
                },
                Column {
                    name: "customer".into(),
                    type_id: 25,
                    type_modifier: -1,
                    is_key: false,
                },
            ],
        });
        let id = document_id(
            &mapping(),
            &relations,
            1,
            &[
                ("id".into(), Value::Text("7".into())),
                ("customer".into(), Value::Text("acme".into())),
            ],
        )
        .unwrap();
        assert_eq!(
            id, "7",
            "the stream's own key is the only choice that cannot disagree with Postgres"
        );
    }

    #[test]
    fn the_embedded_text_follows_the_configured_order() {
        let mut row = serde_json::Map::new();
        row.insert("a".into(), serde_json::json!("first"));
        row.insert("b".into(), serde_json::json!("second"));
        row.insert("n".into(), serde_json::json!(42));
        assert_eq!(
            document_text(&row, &["b".into(), "a".into()]),
            "second\nfirst"
        );
        // A non-string named explicitly is included as its JSON form; a NULL is
        // skipped rather than becoming the word "null".
        assert_eq!(document_text(&row, &["n".into()]), "42");
    }

    #[test]
    fn the_source_reference_is_not_indexed_as_content() {
        // Indexing the table name into every document would make every query
        // match every row a little.
        let mut row = serde_json::Map::new();
        row.insert("body".into(), serde_json::json!("real content"));
        row.insert("_source".into(), serde_json::json!({"table": "orders"}));
        assert_eq!(document_text(&row, &[]), "real content");
    }

    #[test]
    fn a_checkpoint_round_trips_through_json() {
        let checkpoint = Checkpoint {
            lsn: 0x1_0000_00FF,
            system_id: "7300000000000000000".into(),
            stale: false,
            updated_at_ms: 1_700_000_000_000,
        };
        let value = serde_json::to_value(&checkpoint).unwrap();
        let back: Checkpoint = serde_json::from_value(value).unwrap();
        assert_eq!(back.lsn, checkpoint.lsn);
        assert_eq!(back.system_id, checkpoint.system_id);
        // An older checkpoint written before `stale` existed must still load.
        let older: Checkpoint = serde_json::from_value(serde_json::json!({
            "lsn": 1,
            "system_id": "x",
            "updated_at_ms": 0
        }))
        .unwrap();
        assert!(!older.stale);
    }

    #[test]
    fn the_checkpoint_key_is_scoped_to_the_connector() {
        assert_eq!(checkpoint_key("erp"), "pgcdc:erp:checkpoint");
        assert_ne!(checkpoint_key("erp"), checkpoint_key("crm"));
    }
}
