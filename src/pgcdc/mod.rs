//! Change data capture from Postgres, over logical replication.
//!
//! The product decision this implements: **Luma does not replace Postgres, it
//! connects to it.** Postgres stays the transactional source of truth; what
//! flows here is a derived copy, shaped for search rather than for writing.
//! That is why nothing in this module writes back, and why a federated hit
//! carries the source table and primary key — so an application can go read the
//! canonical row where it actually lives.
//!
//! - `pgoutput` — the wire format, decoded here because no published crate does
//!   it. That module's header carries the survey.
//! - `conn` — a connection that can be put into replication mode.
//! - `slots` — the setup a stream needs: a publication and a replication slot.

pub mod conn;
pub mod connector;
pub mod pgoutput;
pub mod slots;

pub use conn::{PgConfig, PgConnection, SslMode, StreamMessage};
pub use connector::{Connector, ConnectorConfig, RunReport, TableMapping};
pub use pgoutput::{Change, Relation, Relations, Value};
