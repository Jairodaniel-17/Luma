//! Redis protocol (RESP) compatibility.
//!
//! The point of this module, stated once so it is not lost in the details: a
//! client that today speaks to `redis://host:6379` should speak to Luma by
//! changing one environment variable and nothing else. See `docs/SPEC-resp.md`.
//!
//! RESP is **an interface** to the platform, not the product — the positioning
//! note from the SPEC's risk table, which belongs next to the code so it stays
//! true.

pub mod commands;
pub mod listener;
pub mod protocol;
pub mod pubsub;
pub mod structures_cmd;
