use luma::config::Config;
use luma::engine::Engine;
use std::time::Duration;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_engine_state_ops() {
    let dir = tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        ..Config::default()
    };
    let token = CancellationToken::new();
    let engine = Engine::new(config, token).unwrap();

    // 1. Put
    let key = "test_key".to_string();
    let value = serde_json::json!({"foo": "bar"});
    let item = engine
        .put_state(key.clone(), value.clone(), None, None)
        .unwrap();
    assert_eq!(item.key, key);
    assert_eq!(item.value, value);

    // 2. Get
    let got = engine.get_state(&key).unwrap();
    assert_eq!(got.value, value);

    // 3. Delete
    let deleted = engine.delete_state(&key).unwrap();
    assert!(deleted);
    let got_after = engine.get_state(&key);
    assert!(got_after.is_none());
}

#[tokio::test]
async fn test_engine_ttl_expiration() {
    let dir = tempdir().unwrap();
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        ..Config::default()
    };
    let token = CancellationToken::new();
    let engine = Engine::new(config, token).unwrap();

    let key = "ttl_key".to_string();
    let value = serde_json::json!("expired");
    // Set small TTL
    engine
        .put_state(key.clone(), value, Some(100), None)
        .unwrap();

    // Verify it exists immediately
    assert!(engine.get_state(&key).is_some());

    // Wait for expiration
    tokio::time::sleep(Duration::from_millis(1500)).await; // Wait > 1s because TTL task runs every 1s

    // Verify it's gone (task runs in background in Engine::new)
    assert!(engine.get_state(&key).is_none());
}
