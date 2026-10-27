//! A failed fsync must not be retried into a lie (SPEC-friochap F2, CA-4).
//!
//! Driven by `scripts/fsync_eio.sh`, which puts the data directory on a
//! device-mapper device it can switch to the `error` target mid-run; without
//! that setup (`LUMA_EIO_PHASE` unset) the test skips.
//!
//! Phase 1: write and confirm, switch the disk to return EIO, and check that
//! the failing write is reported as failed, that Luma goes read-only, and that
//! it stays read-only after the disk comes back. After a failed fsync Linux may
//! have dropped the dirty pages and marked them clean, so a retried fsync can
//! succeed for data that never reached the disk; accepting writes again would
//! confirm them on top of that hole.
//!
//! Phase 2, after the script repaired and remounted the filesystem: a fresh
//! start holds every write confirmed in phase 1 and takes writes again.

use luma::config::Config;
use luma::engine::Engine;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio_util::sync::CancellationToken;

fn engine(dir: &Path) -> Engine {
    let config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(dir.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };
    Engine::new(config, CancellationToken::new()).unwrap()
}

fn dmsetup_table(name: &str, table: &str) {
    for args in [
        // No filesystem freeze: dirty pages stay unwritten, so the switch
        // lands between a write and its fsync, as a failing disk would.
        vec!["suspend", "--nolockfs", name],
        vec!["load", name, "--table", table],
        vec!["resume", name],
    ] {
        let out = Command::new("dmsetup").args(&args).output().unwrap();
        assert!(
            out.status.success(),
            "dmsetup {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn a_failed_fsync_makes_luma_read_only_and_loses_nothing_confirmed() {
    let Ok(phase) = std::env::var("LUMA_EIO_PHASE") else {
        eprintln!("LUMA_EIO_PHASE not set; skipping");
        return;
    };
    let data = PathBuf::from(std::env::var("LUMA_EIO_MNT").unwrap()).join("data");
    let record = PathBuf::from(std::env::var("LUMA_EIO_RECORD").unwrap());

    if phase == "1" {
        let dm = std::env::var("LUMA_EIO_DM").unwrap();
        let ok_table = std::env::var("LUMA_EIO_TABLE_OK").unwrap();
        let err_table = std::env::var("LUMA_EIO_TABLE_ERR").unwrap();

        let e = engine(&data);
        let mut confirmed = Vec::new();
        for i in 0..200 {
            e.put_state(format!("k{i}"), serde_json::json!(i), None, None)
                .unwrap();
            confirmed.push(i);
        }

        dmsetup_table(&dm, &err_table);
        let mut failed = 0;
        for i in 200..400 {
            match e.put_state(format!("k{i}"), serde_json::json!(i), None, None) {
                Ok(_) => confirmed.push(i),
                Err(_) => failed += 1,
            }
        }
        assert!(failed > 0, "writes on a failing disk must not be confirmed");
        assert_eq!(
            e.health(),
            "read_only",
            "a failed fsync must make the WAL read-only"
        );

        // The disk is healthy again; Luma must still refuse until a restart.
        dmsetup_table(&dm, &ok_table);
        assert!(
            e.put_state("after-recovery".into(), serde_json::json!(0), None, None)
                .is_err(),
            "a write after a failed fsync must not be accepted without a restart"
        );
        std::fs::write(
            &record,
            confirmed
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        eprintln!("phase 1: {} confirmed, {failed} refused", confirmed.len());
        // Leak the engine: dropping it would try to write a final snapshot,
        // which is not what a process killed at this point would do.
        std::mem::forget(e);
    } else {
        let confirmed: Vec<usize> = std::fs::read_to_string(&record)
            .unwrap()
            .lines()
            .map(|l| l.parse().unwrap())
            .collect();
        let e = engine(&data);
        let missing: Vec<_> = confirmed
            .iter()
            .filter(|i| e.get_state(&format!("k{i}")).is_none())
            .collect();
        assert!(missing.is_empty(), "confirmed writes lost: {missing:?}");
        assert_eq!(e.health(), "ok");
        e.put_state("after-restart".into(), serde_json::json!(1), None, None)
            .unwrap();
        eprintln!("phase 2: all {} confirmed writes present", confirmed.len());
    }
}
