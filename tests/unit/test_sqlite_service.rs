use luma::sqlite::SqliteService;
use serde_json::json;
use tempfile::tempdir;

#[tokio::test]
async fn test_sqlite_ops() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let svc = SqliteService::new(&db_path).unwrap();

    // 1. Create Table
    let sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)".to_string();
    svc.execute(sql, vec![]).await.unwrap();

    // 2. Insert
    let sql = "INSERT INTO users (name, age) VALUES (?, ?)".to_string();
    let params = vec![json!("Alice"), json!(30)];
    let affected = svc.execute(sql, params).await.unwrap();
    assert_eq!(affected, 1);

    // 3. Query
    let sql = "SELECT * FROM users WHERE name = ?".to_string();
    let params = vec![json!("Alice")];
    let rows = svc.query(sql, params).await.unwrap();
    assert_eq!(rows.len(), 1);

    let row = &rows[0];
    assert_eq!(row["name"], "Alice");
    assert_eq!(row["age"], 30);
}
