use luma::config::Config;
use luma::engine::meta::MetaEngine;
use luma::engine::Engine;
use luma::search::engine::SearchEngine;
use luma::sqlite::SqliteService;
use luma::vector::{Metric, VectorItem};
use serde_json::json;
use std::sync::Arc;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_meta_engine_hybrid() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        sqlite_enabled: true,
        sqlite_path: Some(db_path.to_str().unwrap().to_string()),
        ..Config::default()
    };

    let token = CancellationToken::new();
    // Use Arc from start
    let core = Arc::new(Engine::new(config.clone(), token).unwrap());
    let sqlite = Arc::new(SqliteService::new(&db_path).unwrap());
    let search = Arc::new(SearchEngine::new(dir.path().to_path_buf()).unwrap());

    let meta = MetaEngine::new(core.clone(), Some(sqlite.clone()), search);

    // 1. Setup Vector Data
    core.create_vector_collection("test_col", 4, Metric::Cosine)
        .unwrap();
    core.vector_add(
        "test_col",
        "doc1",
        VectorItem {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            meta: json!({"id": "doc1"}), // Metadata with ID for SQL match
        },
    )
    .unwrap();

    // 2. Setup SQL Data
    sqlite
        .execute(
            "CREATE TABLE items (id TEXT, name TEXT)".to_string(),
            vec![],
        )
        .await
        .unwrap();
    sqlite
        .execute(
            "INSERT INTO items (id, name) VALUES (?, ?)".to_string(),
            vec![json!("doc1"), json!("foo")],
        )
        .await
        .unwrap();

    // 3. Execute Hybrid Query
    let query = json!({
        "type": "hybrid",
        "vector_query": {
            "vector": [1.0, 0.0, 0.0, 0.0],
            "k": 10
        },
        "sql_query": {
            "sql": "SELECT * FROM items WHERE name = ?",
            "params": ["foo"]
        },
        "rrf_k": 60
    });

    let results = meta.execute("test_col", query).await.unwrap();

    assert_eq!(results.len(), 1);
    let doc = &results[0];

    // Verify RRF merging
    // doc should have data from both?
    // My implementation keeps "docs" from map.
    // If ID matches, it overwrites or keeps first?
    // `docs.entry(id).or_insert(doc)` keeps the FIRST one encountered.
    // Vector results processed first.
    // So `doc` should be the Vector hit (containing "meta").
    // AND it should have "rrf_score".

    assert_eq!(doc["id"], "doc1");
    assert!(doc.get("rrf_score").is_some());
    // SQL result would have "name": "foo" but Vector result "meta" is {"id": "doc1"}.
    // If Vector result is kept, it won't have "name" unless I merge.
    // My implementation: `docs.entry(id).or_insert(doc.clone());`
    // It does NOT merge fields.
    // That's acceptable for "Combine results" (ranking fusion).
}
