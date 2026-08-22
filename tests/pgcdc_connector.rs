//! The connector end to end (W4.2, W4.3), against a real Postgres.
//!
//! `pgcdc_stream.rs` checks that the protocol works. This checks that the thing
//! built on top of it does what the product promises: a Postgres table becomes
//! a searchable collection, keeps up with changes, survives a restart without
//! reprocessing what it already applied, and every document it produces says
//! where the canonical row lives.
//!
//! The mock embedding provider throughout — the question here is whether rows
//! become documents, not whether a model produces good vectors.
//!
//! ## Running it
//!
//! ```text
//! LUMA_PG_URL="postgres://luma:luma@127.0.0.1:15432/luma?sslmode=disable" \
//!   cargo test --test pgcdc_connector -- --ignored --test-threads=1
//! ```

use luma::config::Config;
use luma::engine::embeddings::{EmbeddingClient, EmbeddingHandle, EmbeddingProvider};
use luma::engine::hub::LumaDatabase;
use luma::engine::Engine;
use luma::pgcdc::conn::{PgConfig, PgConnection};
use luma::pgcdc::connector::{checkpoint_key, Checkpoint};
use luma::pgcdc::{Connector, ConnectorConfig};
use luma::sqlite::SqliteService;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

fn url() -> String {
    let url = std::env::var("LUMA_PG_URL").unwrap_or_default();
    assert!(
        !url.is_empty(),
        "LUMA_PG_URL is unset. This suite must not pass without a real Postgres.\n\n\
         LUMA_PG_URL=\"postgres://luma:luma@127.0.0.1:15432/luma?sslmode=disable\" \\\n\
         \x20 cargo test --test pgcdc_connector -- --ignored --test-threads=1"
    );
    url
}

/// A Luma instance on its own temporary data directory.
///
/// The directory is returned so it outlives the hub: dropping it first would
/// pull the data out from under an engine that is still writing.
fn luma() -> (Arc<LumaDatabase>, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("cdc.db");
    let config = Config {
        port: 0,
        data_dir: Some(dir.path().to_string_lossy().to_string()),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };
    let engine = Arc::new(Engine::new(config.clone(), CancellationToken::new()).unwrap());
    let sqlite = Arc::new(SqliteService::new(&db_path).unwrap());
    let hub = Arc::new(LumaDatabase::new(
        engine,
        Some(sqlite),
        EmbeddingHandle::new(EmbeddingClient::new(EmbeddingProvider::Mock { dim: 384 })),
        luma::engine::chunking::ChunkingEngine::default(),
        config,
    ));
    (hub, dir)
}

async fn sql() -> PgConnection {
    let config = PgConfig::from_url(&url()).expect("LUMA_PG_URL does not parse");
    PgConnection::connect(&config).await.unwrap()
}

async fn replicating() -> PgConnection {
    let mut config = PgConfig::from_url(&url()).unwrap();
    config.replication = true;
    PgConnection::connect(&config).await.unwrap()
}

/// Build the fixture from scratch, removing whatever a previous run left.
async fn reset(slot: &str, publication: &str, table: &str) {
    let mut conn = sql().await;
    let _ = conn
        .simple_query(&format!(
            "SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots \
             WHERE slot_name = '{slot}'"
        ))
        .await;
    let _ = conn
        .simple_query(&format!("DROP PUBLICATION IF EXISTS \"{publication}\""))
        .await;
    conn.simple_query(&format!("DROP TABLE IF EXISTS {table}"))
        .await
        .unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE {table} (\
         id int PRIMARY KEY, customer text, notes text, secret text, amount numeric)"
    ))
    .await
    .unwrap();
}

fn config_for(name: &str, slot: &str, publication: &str, table: &str, namespace: &str) -> String {
    format!(
        r#"
        name = "{name}"
        url = "{}"
        slot = "{slot}"
        publication = "{publication}"

        [[tables]]
        table = "{table}"
        namespace = "{namespace}"
        text_columns = ["customer", "notes"]
        skip_columns = ["secret"]
        "#,
        url()
    )
}

/// The document the connector should have produced for a row.
fn document(hub: &LumaDatabase, namespace: &str, id: &str) -> Option<serde_json::Value> {
    hub.engine
        .get_state(&format!("doc:{namespace}:{id}"))
        .and_then(|item| item.value.as_json().cloned())
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_backfill_turns_existing_rows_into_documents() {
    let (slot, publication, table, ns) = (
        "luma_conn_backfill",
        "luma_conn_backfill_pub",
        "public.conn_backfill",
        "conn_backfill",
    );
    reset(slot, publication, table).await;

    let mut conn = sql().await;
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES \
         (1, 'acme', 'first order', 'do-not-index', 10.50), \
         (2, 'globex', 'second order', 'also-secret', 20.00)"
    ))
    .await
    .unwrap();

    let (hub, _dir) = luma();
    let connector = Connector::new(
        ConnectorConfig::from_toml(&config_for("backfill", slot, publication, table, ns)).unwrap(),
        hub.clone(),
    )
    .unwrap();

    assert!(
        connector.prepare().await.unwrap().is_some(),
        "a freshly created slot reports its consistent point"
    );
    assert_eq!(connector.backfill().await.unwrap(), 2);

    let doc = document(&hub, ns, "1").expect("row 1 became a document");
    assert_eq!(doc["customer"], "acme");
    assert_eq!(doc["notes"], "first order");
    // numeric arrives as the text Postgres prints, not a float that is nearly it.
    assert_eq!(doc["amount"], "10.50");
    assert!(
        doc.get("secret").is_none(),
        "skip_columns must keep the column out of the document entirely, not just out of the \
         embedded text: {doc}"
    );

    // W4.3: the source reference. This is what makes the copy safe to search —
    // a hit says where the canonical row is, so the application reads Postgres
    // rather than trusting what is stored here.
    let source = &doc["_source"];
    assert_eq!(source["system"], "postgres");
    assert_eq!(source["schema"], "public");
    assert_eq!(source["table"], "conn_backfill");
    assert_eq!(source["primary_key"], "1");

    assert!(document(&hub, ns, "2").is_some());
    release(slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn changes_made_during_a_backfill_are_not_lost() {
    // The correctness argument for creating the slot before copying. A backfill
    // taken first would miss everything written between the copy and the slot,
    // and those rows are gone with nothing to say they were ever expected.
    let (slot, publication, table, ns) = (
        "luma_conn_window",
        "luma_conn_window_pub",
        "public.conn_window",
        "conn_window",
    );
    reset(slot, publication, table).await;

    let mut conn = sql().await;
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'before', 'existing row', 's', 1.00)"
    ))
    .await
    .unwrap();

    let (hub, _dir) = luma();
    let connector = Connector::new(
        ConnectorConfig::from_toml(&config_for("window", slot, publication, table, ns)).unwrap(),
        hub.clone(),
    )
    .unwrap();

    // Slot first.
    connector.prepare().await.unwrap();
    // A row written after the slot exists but before the copy: it will appear
    // in both, and the second application has to be harmless.
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (2, 'during', 'written in the window', 's', 2.00)"
    ))
    .await
    .unwrap();
    connector.backfill().await.unwrap();
    // And one strictly after the copy, which only the stream can carry.
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (3, 'after', 'streamed only', 's', 3.00)"
    ))
    .await
    .unwrap();

    let report = connector
        .stream_once(Duration::from_secs(15), 2)
        .await
        .unwrap();
    assert!(report.applied() > 0, "the stream must carry something");

    for (id, customer) in [("1", "before"), ("2", "during"), ("3", "after")] {
        let doc = document(&hub, ns, id).unwrap_or_else(|| panic!("row {id} is missing"));
        assert_eq!(doc["customer"], customer, "row {id}");
    }
    release(slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn an_update_and_a_delete_reach_the_collection() {
    let (slot, publication, table, ns) = (
        "luma_conn_dml",
        "luma_conn_dml_pub",
        "public.conn_dml",
        "conn_dml",
    );
    reset(slot, publication, table).await;

    let (hub, _dir) = luma();
    let connector = Connector::new(
        ConnectorConfig::from_toml(&config_for("dml", slot, publication, table, ns)).unwrap(),
        hub.clone(),
    )
    .unwrap();
    connector.prepare().await.unwrap();

    let mut conn = sql().await;
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'acme', 'original', 's', 1.00)"
    ))
    .await
    .unwrap();
    connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    assert_eq!(document(&hub, ns, "1").unwrap()["notes"], "original");

    conn.simple_query(&format!(
        "UPDATE {table} SET notes = 'revised' WHERE id = 1"
    ))
    .await
    .unwrap();
    let report = connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    assert_eq!(report.updated, 1, "{report:?}");
    assert_eq!(document(&hub, ns, "1").unwrap()["notes"], "revised");

    conn.simple_query(&format!("DELETE FROM {table} WHERE id = 1"))
        .await
        .unwrap();
    let report = connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    assert_eq!(report.deleted, 1, "{report:?}");
    assert!(
        document(&hub, ns, "1").is_none(),
        "a deleted row must not stay searchable"
    );
    release(slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn an_unchanged_column_keeps_its_stored_value() {
    // The data-loss case, at the level the connector is responsible for.
    // `pgcdc_stream.rs` proves Postgres elides the column; this proves the
    // connector does not then overwrite it with nothing.
    let (slot, publication, table, ns) = (
        "luma_conn_toast",
        "luma_conn_toast_pub",
        "public.conn_toast",
        "conn_toast",
    );
    reset(slot, publication, table).await;

    let mut conn = sql().await;
    conn.simple_query(&format!(
        "ALTER TABLE {table} ALTER COLUMN notes SET STORAGE EXTERNAL"
    ))
    .await
    .unwrap();

    let (hub, _dir) = luma();
    let connector = Connector::new(
        ConnectorConfig::from_toml(&config_for("toast", slot, publication, table, ns)).unwrap(),
        hub.clone(),
    )
    .unwrap();
    connector.prepare().await.unwrap();

    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'acme', repeat('x', 12000), 's', 1.00)"
    ))
    .await
    .unwrap();
    connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    let stored = document(&hub, ns, "1").unwrap();
    assert_eq!(stored["notes"].as_str().unwrap().len(), 12_000);

    // Update a different column. Postgres omits `notes` from the message.
    conn.simple_query(&format!(
        "UPDATE {table} SET customer = 'acme corp' WHERE id = 1"
    ))
    .await
    .unwrap();
    connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();

    let after = document(&hub, ns, "1").unwrap();
    assert_eq!(after["customer"], "acme corp");
    assert_eq!(
        after["notes"].as_str().map(str::len),
        Some(12_000),
        "the elided column was replaced instead of kept; Postgres still holds that value and \
         will never mention it again"
    );
    release(slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_connector_resumes_from_its_checkpoint_rather_than_the_beginning() {
    let (slot, publication, table, ns) = (
        "luma_conn_resume",
        "luma_conn_resume_pub",
        "public.conn_resume",
        "conn_resume",
    );
    reset(slot, publication, table).await;

    let (hub, _dir) = luma();
    let config =
        ConnectorConfig::from_toml(&config_for("resume", slot, publication, table, ns)).unwrap();
    let connector = Connector::new(config.clone(), hub.clone()).unwrap();
    connector.prepare().await.unwrap();

    let mut conn = sql().await;
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'acme', 'one', 's', 1.00)"
    ))
    .await
    .unwrap();
    let first = connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    assert_eq!(first.inserted, 1);

    let checkpoint = connector
        .checkpoint()
        .expect("a pass must persist a position");
    assert!(checkpoint.lsn > 0);
    assert!(!checkpoint.system_id.is_empty());
    assert!(!checkpoint.stale);

    // A second connector over the same data directory: what a restart looks
    // like. The already-applied insert must not come back as a new one.
    let restarted = Connector::new(config, hub.clone()).unwrap();
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (2, 'globex', 'two', 's', 2.00)"
    ))
    .await
    .unwrap();
    let second = restarted
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    assert_eq!(
        second.inserted, 1,
        "only the new row should arrive; got {second:?}"
    );
    assert!(second.last_lsn >= checkpoint.lsn);
    assert!(document(&hub, ns, "2").is_some());
    release(slot).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_checkpoint_from_a_different_server_is_refused() {
    // Restoring a Postgres backup and pointing the connector at the copy would
    // otherwise resume at a position that is arithmetically fine and refers to
    // a different history. Nothing downstream would look wrong.
    let (slot, publication, table, ns) = (
        "luma_conn_sysid",
        "luma_conn_sysid_pub",
        "public.conn_sysid",
        "conn_sysid",
    );
    reset(slot, publication, table).await;

    let (hub, _dir) = luma();
    let config =
        ConnectorConfig::from_toml(&config_for("sysid", slot, publication, table, ns)).unwrap();

    hub.engine
        .put_state(
            checkpoint_key("sysid"),
            serde_json::to_value(Checkpoint {
                lsn: 0x1_0000,
                system_id: "7000000000000000001".into(),
                stale: false,
                updated_at_ms: 0,
            })
            .unwrap(),
            None,
            None,
        )
        .unwrap();

    let connector = Connector::new(config, hub.clone()).unwrap();
    let err = connector
        .prepare()
        .await
        .expect_err("a checkpoint from another server must not be resumed")
        .to_string();
    assert!(err.contains("system"), "{err}");
    assert!(err.contains(&checkpoint_key("sysid")), "{err}");
    // No release: `prepare` refused before creating the slot, which is the
    // point — the check happens before anything is claimed upstream.
    let _ = slot;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_table_without_a_usable_key_is_refused_at_setup() {
    // Not on the first UPDATE. The difference is a message during configuration
    // versus a collection that has been quietly missing changes in production.
    let mut conn = sql().await;
    conn.simple_query("DROP TABLE IF EXISTS public.conn_nokey")
        .await
        .unwrap();
    conn.simple_query("CREATE TABLE public.conn_nokey (id int, payload text)")
        .await
        .unwrap();

    let (hub, _dir) = luma();
    let connector = Connector::new(
        ConnectorConfig::from_toml(&config_for(
            "nokey",
            "luma_conn_nokey",
            "luma_conn_nokey_pub",
            "public.conn_nokey",
            "conn_nokey",
        ))
        .unwrap(),
        hub,
    )
    .unwrap();

    let err = connector
        .prepare()
        .await
        .expect_err("a table with no usable replica identity must be refused")
        .to_string();
    assert!(err.contains("primary key"), "{err}");

    conn.simple_query("DROP TABLE public.conn_nokey")
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real Postgres 16: set LUMA_PG_URL and run with --ignored"]
async fn a_truncate_is_reported_and_not_acted_on() {
    // Emptying a whole derived collection because of one WAL message is an
    // operator's decision. The connector says so loudly and marks the
    // checkpoint stale instead of either destroying data or pretending nothing
    // happened.
    let (slot, publication, table, ns) = (
        "luma_conn_trunc",
        "luma_conn_trunc_pub",
        "public.conn_trunc",
        "conn_trunc",
    );
    reset(slot, publication, table).await;

    let (hub, _dir) = luma();
    let connector = Connector::new(
        ConnectorConfig::from_toml(&config_for("trunc", slot, publication, table, ns)).unwrap(),
        hub.clone(),
    )
    .unwrap();
    connector.prepare().await.unwrap();

    let mut conn = sql().await;
    conn.simple_query(&format!(
        "INSERT INTO {table} VALUES (1, 'acme', 'one', 's', 1.00)"
    ))
    .await
    .unwrap();
    connector
        .stream_once(Duration::from_secs(15), 1)
        .await
        .unwrap();
    assert!(document(&hub, ns, "1").is_some());

    conn.simple_query(&format!("TRUNCATE {table}"))
        .await
        .unwrap();
    // A TRUNCATE applies no rows, so this pass can only end on its budget —
    // there is no change count for it to reach.
    let report = connector
        .stream_once(Duration::from_secs(6), 1)
        .await
        .unwrap();
    assert_eq!(
        report.truncated_tables,
        vec!["public.conn_trunc".to_string()]
    );
    assert!(
        document(&hub, ns, "1").is_some(),
        "the document is stale, not deleted — and the checkpoint says so"
    );
    assert!(
        connector.checkpoint().unwrap().stale,
        "an operator has to be able to see this after the fact, not only in a log line"
    );
    release(slot).await;
}

async fn release(slot: &str) {
    luma::pgcdc::slots::drop_slot(&mut replicating().await, slot)
        .await
        .unwrap();
}
