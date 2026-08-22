//! Logical replication against a real Postgres.
//!
//! W4.1 of `docs/PLAN-MAESTRO.md` — the spike whose exit criterion was a
//! decision report before designing anything. This is the evidence behind it:
//! the decision was "own subset", and a survey is only worth something if the
//! subset it recommends actually consumes a stream from a real server.
//!
//! What no unit test can establish, and what this checks:
//!
//! - That the startup message with `replication=database` is accepted, and that
//!   SCRAM-SHA-256 completes against a Postgres 16 default configuration.
//! - That `START_REPLICATION` produces a `CopyBothResponse` — the tag the
//!   released `tokio-postgres` cannot parse, and the concrete reason the
//!   connection is hand-rolled.
//! - That INSERT, UPDATE and DELETE arrive, in transaction order, with the
//!   values and the key Postgres actually sent, rather than the ones a
//!   round-trip through our own encoder would produce.
//! - That a standby status update is accepted and moves `confirmed_flush_lsn`,
//!   which is the mechanism that lets Postgres release WAL. A connector that
//!   never moves it pins every segment since the slot was created and fills the
//!   primary's disk.
//!
//! ## Running it
//!
//! ```text
//! docker run -d --name luma-cdc-pg \
//!   -e POSTGRES_PASSWORD=luma -e POSTGRES_USER=luma -e POSTGRES_DB=luma \
//!   -p 15432:5432 postgres:16-alpine \
//!   -c wal_level=logical -c max_replication_slots=8 -c max_wal_senders=8
//!
//! LUMA_PG_URL="postgres://luma:luma@127.0.0.1:15432/luma?sslmode=disable" \
//!   cargo test --test pgcdc_stream -- --ignored --test-threads=1
//! ```
//!
//! Without `LUMA_PG_URL` it refuses to run rather than passing vacuously: a
//! suite that goes green when its subject is absent reports coverage it does
//! not have.

use luma::pgcdc::conn::{PgConfig, PgConnection, StreamMessage};
use luma::pgcdc::pgoutput::{decode, Change, OldTuple, Relations, Value};
use luma::pgcdc::slots;
use std::time::Duration;

fn url() -> String {
    let url = std::env::var("LUMA_PG_URL").unwrap_or_default();
    assert!(
        !url.is_empty(),
        "LUMA_PG_URL is unset. This suite must not pass without a real Postgres — \
         a green run against nothing is a claim of coverage that does not exist.\n\n\
         docker run -d --name luma-cdc-pg -e POSTGRES_PASSWORD=luma -e POSTGRES_USER=luma \\\n\
         \x20 -e POSTGRES_DB=luma -p 15432:5432 postgres:16-alpine \\\n\
         \x20 -c wal_level=logical -c max_replication_slots=8 -c max_wal_senders=8\n\n\
         LUMA_PG_URL=\"postgres://luma:luma@127.0.0.1:15432/luma?sslmode=disable\" \\\n\
         \x20 cargo test --test pgcdc_stream -- --ignored --test-threads=1"
    );
    url
}

async fn ordinary() -> PgConnection {
    let config = PgConfig::from_url(&url()).expect("LUMA_PG_URL does not parse");
    PgConnection::connect(&config)
        .await
        .expect("could not open an ordinary connection")
}

async fn replicating() -> PgConnection {
    let mut config = PgConfig::from_url(&url()).expect("LUMA_PG_URL does not parse");
    config.replication = true;
    PgConnection::connect(&config)
        .await
        .expect("could not open a replication connection")
}

/// Tear down whatever a previous run left, then build the fixture fresh.
///
/// Slots first: dropping a table a publication names while a slot is still
/// holding WAL leaves the slot alive and the WAL pinned.
async fn reset(slot: &str, publication: &str, table: &str) {
    let mut sql = ordinary().await;
    let _ = sql
        .simple_query(&format!(
            "SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots \
             WHERE slot_name = '{slot}'"
        ))
        .await;
    let _ = sql
        .simple_query(&format!("DROP PUBLICATION IF EXISTS \"{publication}\""))
        .await;
    sql.simple_query(&format!("DROP TABLE IF EXISTS {table}"))
        .await
        .unwrap();
    sql.simple_query(&format!(
        "CREATE TABLE {table} (id int PRIMARY KEY, customer text, total numeric, note text)"
    ))
    .await
    .unwrap();
}

/// Drop a slot, after letting go of the connection that holds it.
///
/// `DROP_REPLICATION_SLOT ... WAIT` waits for the slot to become inactive, and
/// the streaming connection is what keeps it active. Dropping from a second
/// connection while the first is still open blocks until `wal_sender_timeout`
/// kills the walsender — a minute per test, for no reason, which is what the
/// first run of this suite spent.
async fn release(stream: PgConnection, slot: &str) {
    drop(stream);
    slots::drop_slot(&mut replicating().await, slot)
        .await
        .unwrap();
}

/// Read messages until a Commit arrives, or time out.
///
/// Bounded because the alternative is a hang: if a change never shows up, the
/// useful output is "these are the messages that did arrive", not a test that
/// never returns.
async fn drain_until_commit(conn: &mut PgConnection, relations: &mut Relations) -> Vec<Change> {
    let mut changes = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "no Commit arrived within 20s; got {changes:?}"
        );
        let message = match tokio::time::timeout(remaining, conn.next_stream_message()).await {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => panic!("the stream failed: {e}"),
            Err(_) => panic!("no Commit arrived within 20s; got {changes:?}"),
        };
        match message {
            StreamMessage::Keepalive {
                end_lsn,
                reply_requested,
                ..
            } => {
                // Not optional. Ignoring a requested reply lets the server
                // decide the standby is gone, which reads downstream as a
                // periodic disconnect with no cause in the logs.
                if reply_requested {
                    conn.send_standby_status(end_lsn, end_lsn, end_lsn, false)
                        .await
                        .unwrap();
                }
            }
            StreamMessage::XLogData { data, .. } => {
                let change = decode(&data).expect("a message from a real server must decode");
                if let Change::Relation(r) = &change {
                    relations.insert(r.clone());
                }
                let done = matches!(change, Change::Commit { .. });
                changes.push(change);
                if done {
                    return changes;
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_replication_connection_authenticates_and_identifies_the_server() {
    let mut conn = replicating().await;
    let identity = slots::identify_system(&mut conn).await.unwrap();

    assert!(
        !identity.system_id.is_empty(),
        "the system id is the only thing that distinguishes a server from a restored copy of \
         itself; resuming from a saved LSN against the wrong one lands at a plausible-looking \
         position in somebody else's WAL"
    );
    assert!(identity.current_lsn > 0);
    assert_eq!(identity.database.as_deref(), Some("luma"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn the_two_connection_modes_differ_where_the_protocol_says_they_do() {
    // Written first asserting that a replication connection cannot run SQL.
    // It can, and the real server said so. `replication=database` — the
    // *logical* mode — opens a walsender attached to a database, and that
    // connection accepts ordinary SQL as well as replication commands. It is
    // `replication=true`, the physical mode, that refuses queries.
    //
    // Worth keeping as a test rather than deleting, because the corrected
    // version records something W4.2 depends on: one connection can do the
    // catalog checks and then stream, and the line it cannot cross is
    // START_REPLICATION, not the connection mode.
    let mut sql = ordinary().await;
    assert!(
        slots::identify_system(&mut sql).await.is_err(),
        "IDENTIFY_SYSTEM is a replication command and is not available without the mode"
    );

    let mut streaming = replicating().await;
    streaming
        .simple_query("SELECT 1 FROM pg_class LIMIT 1")
        .await
        .expect("a logical replication connection does accept ordinary SQL");
    slots::identify_system(&mut streaming)
        .await
        .expect("and replication commands too");

    // The actual boundary: once COPY-BOTH is open, a query would be interleaved
    // into the stream and desynchronize both sides, so it is refused locally
    // rather than sent.
    let (slot, publication, table) = (
        "luma_spike_modes",
        "luma_spike_modes_pub",
        "public.spike_modes",
    );
    reset(slot, publication, table).await;
    slots::ensure_publication(&mut sql, publication, &[table.to_string()])
        .await
        .unwrap();
    slots::ensure_slot(&mut streaming, slot).await.unwrap();
    streaming
        .start_replication(slot, &[publication.to_string()], 0)
        .await
        .unwrap();
    let err = streaming
        .simple_query("SELECT 1")
        .await
        .expect_err("a streaming connection must refuse queries")
        .to_string();
    assert!(err.contains("streaming"), "{err}");

    release(streaming, slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn insert_update_and_delete_arrive_in_transaction_order() {
    let (slot, publication, table) = ("luma_spike_dml", "luma_spike_dml_pub", "public.spike_dml");
    reset(slot, publication, table).await;

    let mut sql = ordinary().await;
    slots::ensure_publication(&mut sql, publication, &[table.to_string()])
        .await
        .unwrap();

    let identities = slots::check_replica_identities(&mut sql, &[table.to_string()])
        .await
        .unwrap();
    assert!(identities[0].is_usable(), "{}", identities[0].advice());

    let mut stream = replicating().await;
    let (start_lsn, created) = slots::ensure_slot(&mut stream, slot).await.unwrap();
    assert!(created, "the fixture dropped any previous slot");
    assert!(start_lsn > 0, "a fresh slot reports its consistent point");

    stream
        .start_replication(slot, &[publication.to_string()], 0)
        .await
        .expect("START_REPLICATION must produce a CopyBothResponse");

    // One transaction with all three verbs, so ordering within a transaction is
    // part of what is checked rather than an accident of timing.
    sql.simple_query(&format!(
        "BEGIN; \
         INSERT INTO {table} VALUES (1, 'acme', 42.00, 'first'); \
         UPDATE {table} SET customer = 'acme corp' WHERE id = 1; \
         DELETE FROM {table} WHERE id = 1; \
         COMMIT;"
    ))
    .await
    .unwrap();

    let mut relations = Relations::new();
    let changes = drain_until_commit(&mut stream, &mut relations).await;

    assert!(
        matches!(changes.first(), Some(Change::Begin { .. })),
        "a transaction opens with Begin: {changes:?}"
    );
    assert!(
        matches!(changes.last(), Some(Change::Commit { .. })),
        "a transaction closes with Commit: {changes:?}"
    );

    let relation = changes
        .iter()
        .find_map(|c| match c {
            Change::Relation(r) => Some(r),
            _ => None,
        })
        .expect("the relation must be announced before any row that names it");
    assert_eq!(relation.qualified(), "public.spike_dml");
    assert_eq!(relation.key_columns(), vec!["id"]);

    let verbs: Vec<&str> = changes
        .iter()
        .filter_map(|c| match c {
            Change::Insert { .. } => Some("insert"),
            Change::Update { .. } => Some("update"),
            Change::Delete { .. } => Some("delete"),
            _ => None,
        })
        .collect();
    assert_eq!(verbs, vec!["insert", "update", "delete"], "{changes:?}");

    let insert = changes
        .iter()
        .find_map(|c| match c {
            Change::Insert { tuple, .. } => Some(tuple),
            _ => None,
        })
        .unwrap();
    let named = relations.name_values(relation.id, insert).unwrap();
    assert_eq!(named[0], ("id", &Value::Text("1".into())));
    assert_eq!(named[1], ("customer", &Value::Text("acme".into())));
    // numeric arrives as the text Postgres would print, not as a float. That is
    // the point of taking it as text: a round-trip through f64 would change
    // 42.00 into something that is nearly it.
    assert_eq!(named[2], ("total", &Value::Text("42.00".into())));

    let deleted = changes
        .iter()
        .find_map(|c| match c {
            Change::Delete { old, .. } => Some(old),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        deleted.0,
        OldTuple::Key,
        "REPLICA IDENTITY DEFAULT sends only the key"
    );
    assert_eq!(deleted.1[0], Value::Text("1".into()));

    release(stream, slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn an_unchanged_toast_column_arrives_as_unchanged_not_as_null() {
    // The failure this guards against is data loss, not a wrong field: a large
    // column that did not change is omitted from the UPDATE, and a consumer
    // that reads the omission as NULL destroys a value Postgres still holds
    // and never mentions again.
    //
    // It needs a real server because it depends on the TOAST threshold — no
    // hand-built message would prove Postgres actually elides the column.
    let (slot, publication, table) = (
        "luma_spike_toast",
        "luma_spike_toast_pub",
        "public.spike_toast",
    );
    reset(slot, publication, table).await;

    let mut sql = ordinary().await;
    slots::ensure_publication(&mut sql, publication, &[table.to_string()])
        .await
        .unwrap();

    // STORAGE EXTERNAL, not just a long value. The first attempt used
    // `repeat('x', 12000)` and the column arrived in full: 12 kB of one
    // repeated character compresses to almost nothing, so Postgres kept it
    // inline and had no reason to elide it. Only a value stored *out of line*
    // is omitted from the WAL tuple, and EXTERNAL is what makes that
    // deterministic rather than a property of the sample data.
    sql.simple_query(&format!(
        "ALTER TABLE {table} ALTER COLUMN note SET STORAGE EXTERNAL"
    ))
    .await
    .unwrap();
    sql.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'acme', 1.00, repeat('x', 12000))"
    ))
    .await
    .unwrap();

    let mut stream = replicating().await;
    slots::ensure_slot(&mut stream, slot).await.unwrap();
    stream
        .start_replication(slot, &[publication.to_string()], 0)
        .await
        .unwrap();

    sql.simple_query(&format!(
        "UPDATE {table} SET customer = 'acme corp' WHERE id = 1"
    ))
    .await
    .unwrap();

    let mut relations = Relations::new();
    let changes = drain_until_commit(&mut stream, &mut relations).await;
    let (relation_id, tuple) = changes
        .iter()
        .find_map(|c| match c {
            Change::Update {
                relation_id, tuple, ..
            } => Some((*relation_id, tuple)),
            _ => None,
        })
        .expect("the update must arrive");

    let named = relations.name_values(relation_id, tuple).unwrap();
    let note = named.iter().find(|(n, _)| *n == "note").unwrap().1;
    assert_eq!(
        note,
        &Value::Unchanged,
        "Postgres elided the TOASTed column; reading that as NULL would delete it downstream"
    );
    assert_ne!(note, &Value::Null);

    release(stream, slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_standby_status_update_lets_postgres_release_wal() {
    // The mechanism that keeps a slot from filling the primary's disk. Nothing
    // downstream reports its absence: the stream works perfectly while
    // `confirmed_flush_lsn` stays where it started and WAL accumulates.
    let (slot, publication, table) = (
        "luma_spike_flush",
        "luma_spike_flush_pub",
        "public.spike_flush",
    );
    reset(slot, publication, table).await;

    let mut sql = ordinary().await;
    slots::ensure_publication(&mut sql, publication, &[table.to_string()])
        .await
        .unwrap();

    let mut stream = replicating().await;
    slots::ensure_slot(&mut stream, slot).await.unwrap();
    stream
        .start_replication(slot, &[publication.to_string()], 0)
        .await
        .unwrap();

    let before = slots::slot_status(&mut sql, slot)
        .await
        .unwrap()
        .expect("the slot exists");
    assert!(before.active, "a streaming slot reports as active");

    sql.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'acme', 1.00, 'x')"
    ))
    .await
    .unwrap();

    let mut relations = Relations::new();
    let changes = drain_until_commit(&mut stream, &mut relations).await;
    let end_lsn = changes
        .iter()
        .find_map(|c| match c {
            Change::Commit { end_lsn, .. } => Some(*end_lsn),
            _ => None,
        })
        .expect("a commit must arrive");

    stream
        .send_standby_status(end_lsn, end_lsn, end_lsn, true)
        .await
        .unwrap();

    // The server records it asynchronously, so poll rather than assume.
    let mut confirmed = None;
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(250)).await;
        let status = slots::slot_status(&mut sql, slot).await.unwrap().unwrap();
        if status.confirmed_flush_lsn.unwrap_or(0) >= end_lsn {
            confirmed = status.confirmed_flush_lsn;
            break;
        }
    }
    assert!(
        confirmed.is_some(),
        "confirmed_flush_lsn never reached {end_lsn:X}; without this the slot pins every WAL \
         segment since it was created"
    );

    release(stream, slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_table_without_a_usable_replica_identity_is_reported_before_streaming() {
    // At setup, not on the first UPDATE. The difference is a message during
    // configuration versus a stream that has been quietly dropping updates in
    // production until somebody compares a count.
    let mut sql = ordinary().await;
    sql.simple_query("DROP TABLE IF EXISTS public.spike_nokey")
        .await
        .unwrap();
    sql.simple_query("CREATE TABLE public.spike_nokey (id int, payload text)")
        .await
        .unwrap();

    let checked = slots::check_replica_identities(&mut sql, &["public.spike_nokey".to_string()])
        .await
        .unwrap();
    assert_eq!(checked[0].kind, 'd');
    assert!(!checked[0].has_primary_key);
    assert!(!checked[0].is_usable());
    assert!(checked[0].advice().contains("primary key"));

    // And FULL makes the same table usable, which is the advice we give.
    sql.simple_query("ALTER TABLE public.spike_nokey REPLICA IDENTITY FULL")
        .await
        .unwrap();
    let checked = slots::check_replica_identities(&mut sql, &["public.spike_nokey".to_string()])
        .await
        .unwrap();
    assert!(checked[0].is_usable());

    sql.simple_query("DROP TABLE public.spike_nokey")
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_wrong_password_is_refused_by_scram_rather_than_accepted() {
    let mut config = PgConfig::from_url(&url()).unwrap();
    config.password = format!("{}-wrong", config.password);
    let err = match PgConnection::connect(&config).await {
        Ok(_) => panic!("a wrong password must not authenticate"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("authentication") || err.contains("password"),
        "the failure should name authentication, got: {err}"
    );
}
