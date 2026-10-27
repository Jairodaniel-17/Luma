//! A full disk must cost writes, never data (SPEC-friochap F2, CA-3).
//!
//! Needs a small, otherwise empty filesystem of its own, named by
//! `LUMA_SMALL_FS` (a 64 MB ext4 loop mount, say); without it the test skips.
//! `scripts/disk_full.sh` sets one up in a privileged container.
//!
//! The engine writes until the disk refuses, which drives every write path
//! into ENOSPC at whatever point it happens to be: the WAL mid-record, the
//! LSM projection, a segment rotation. Then space comes back and writing
//! resumes, and a restart must replay every write that was confirmed, before
//! and after, with no corrupt record stopping the replay early. Before the WAL
//! cut a failed write back off, the partial record it left made replay stop
//! there and everything confirmed after it was lost.

use luma::config::Config;
use luma::engine::Engine;
use std::path::Path;
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

#[test]
fn a_full_disk_refuses_writes_and_loses_nothing_confirmed() {
    let Ok(fs) = std::env::var("LUMA_SMALL_FS") else {
        eprintln!("LUMA_SMALL_FS not set; skipping");
        return;
    };
    let data = Path::new(&fs).join("data");
    let filler = Path::new(&fs).join("filler");
    let value = |i: usize| serde_json::json!({ "i": i, "pad": "x".repeat(16 * 1024) });

    let e = engine(&data);
    let mut confirmed = Vec::new();
    let mut refused = 0;
    // Most of the space goes to a filler file, so the engine reaches ENOSPC
    // quickly and with its files in every state of growth.
    let free = fs2_free(&fs);
    std::fs::write(&filler, vec![0u8; free.saturating_sub(8 << 20) as usize]).unwrap();

    for i in 0..4000 {
        match e.put_state(format!("k{i}"), value(i), None, None) {
            Ok(_) => confirmed.push(i),
            Err(_) => {
                refused += 1;
                if refused > 50 {
                    break;
                }
            }
        }
    }
    assert!(
        refused > 0,
        "the disk never filled up; make LUMA_SMALL_FS smaller"
    );
    // Reads keep working on a full disk.
    for &i in &confirmed {
        assert!(
            e.get_state(&format!("k{i}")).is_some(),
            "k{i} unreadable while full"
        );
    }

    // Space comes back: writes resume without a restart, unless a failed fsync
    // made the WAL read-only, which is the one state that needs a restart.
    std::fs::remove_file(&filler).unwrap();
    let read_only = e.health() == "read_only";
    let mut after = Vec::new();
    for i in 10_000..10_050 {
        if e.put_state(format!("k{i}"), value(i), None, None).is_ok() {
            after.push(i);
        }
    }
    if !read_only {
        assert_eq!(
            after.len(),
            50,
            "writes did not resume after space was freed"
        );
    }
    drop(e);

    // Every confirmed write survives the restart, before and after the full
    // disk: no partial record left mid-WAL to stop the replay.
    let e = engine(&data);
    for &i in confirmed.iter().chain(after.iter()) {
        let item = e.get_state(&format!("k{i}"));
        assert!(
            item.is_some(),
            "confirmed write k{i} lost after the disk filled up"
        );
    }
    assert_eq!(e.health(), "ok", "a restart must clear the read-only state");
    assert!(
        e.put_state("after-restart".into(), value(0), None, None)
            .is_ok(),
        "the restarted engine must take writes again"
    );
    eprintln!(
        "confirmed {} before, {} refused, {} after (read_only={read_only})",
        confirmed.len(),
        refused,
        after.len()
    );
}

/// Free bytes on the filesystem at `path`, from `df`, to stay free of a
/// dependency just for the test.
fn fs2_free(path: &str) -> u64 {
    let out = std::process::Command::new("df")
        .args(["-B1", "--output=avail", path])
        .output()
        .expect("df");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)
        .and_then(|l| l.trim().parse().ok())
        .expect("df output")
}

/// The HTTP surface on a nearly full disk: writes answer 507 before they can
/// hit ENOSPC, reads keep working, `/v1/health/ready` says why, and writes come
/// back by themselves once space is freed.
#[tokio::test]
async fn http_writes_get_507_reads_keep_working_and_it_recovers() {
    let Ok(fs) = std::env::var("LUMA_SMALL_FS") else {
        eprintln!("LUMA_SMALL_FS not set; skipping");
        return;
    };
    let data = Path::new(&fs).join("http");
    let filler = Path::new(&fs).join("filler-http");
    let mut config = Config {
        port: 0,
        bind_addr: "127.0.0.1".parse().unwrap(),
        api_key: "test".to_string(),
        data_dir: Some(data.to_string_lossy().to_string()),
        snapshot_interval_secs: 3600,
        ..Config::default()
    };
    config.rate_limit_rps = 0;
    let engine = Engine::new(config.clone(), CancellationToken::new()).unwrap();
    engine
        .put_state("kept".into(), serde_json::json!(1), None, None)
        .unwrap();
    let search = std::sync::Arc::new(
        luma::search::engine::SearchEngine::new(data.join("search-root")).unwrap(),
    );
    let app = luma::api::router(luma::api::RouterDeps {
        engine,
        config,
        sqlite: None,
        search_engine: search,
        auth_store: None,
        embeddings: luma::engine::embeddings::EmbeddingHandle::new(
            luma::engine::embeddings::EmbeddingClient::new(
                luma::engine::embeddings::EmbeddingProvider::Mock { dim: 4 },
            ),
        ),
        resp_metrics: None,
        audit_log: None,
        rbac: None,
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    let client = reqwest::Client::new();
    let put = |key: &str| {
        client
            .put(format!("{base}/v1/state/{key}"))
            .bearer_auth("test")
            .json(&serde_json::json!({ "value": 1 }))
            .send()
    };
    // The free-space reading is cached for two seconds.
    let settle = || tokio::time::sleep(std::time::Duration::from_millis(2200));

    assert_eq!(put("before").await.unwrap().status(), 200);

    let free = fs2_free(&fs);
    std::fs::write(&filler, vec![0u8; free.saturating_sub(1 << 20) as usize]).unwrap();
    settle().await;
    assert_eq!(put("while-full").await.unwrap().status(), 507);
    let read = client
        .get(format!("{base}/v1/state/kept"))
        .bearer_auth("test")
        .send()
        .await
        .unwrap();
    assert_eq!(read.status(), 200, "reads must keep working on a full disk");
    let ready = client
        .get(format!("{base}/v1/health/ready"))
        .send()
        .await
        .unwrap();
    assert_eq!(ready.status(), 503);
    let body: serde_json::Value = ready.json().await.unwrap();
    assert_eq!(body["status"], "disk_full");

    std::fs::remove_file(&filler).unwrap();
    settle().await;
    assert_eq!(
        put("after").await.unwrap().status(),
        200,
        "writes must resume once space is freed, without a restart"
    );
}
