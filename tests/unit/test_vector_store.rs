use luma::vector::{Metric, SearchRequest, VectorItem, VectorSettings, VectorStore};
use serde_json::json;
use tempfile::tempdir;

#[test]
fn test_vector_collection_ops() {
    let dir = tempdir().unwrap();
    let settings = VectorSettings::default();
    let store = VectorStore::open_with_settings(dir.path(), settings).unwrap();

    let collection_name = "test_col";
    let dim = 4;

    // 1. Create Collection
    store
        .create_collection(collection_name, dim, Metric::Cosine)
        .unwrap();
    let info = store.get_collection_info(collection_name).unwrap();
    assert_eq!(info.dim, dim);
    assert_eq!(info.metric, Metric::Cosine);

    // 2. Add Vector
    let id = "vec1";
    let vector = vec![1.0, 0.0, 0.0, 0.0];
    let meta = json!({"tag": "A"});
    let item = VectorItem {
        vector: vector.clone(),
        meta: meta.clone(),
    };
    store.add(collection_name, id, item).unwrap();

    // 3. Search
    let req = SearchRequest {
        vector: vec![1.0, 0.0, 0.0, 0.0],
        k: 1,
        filters: None,
        include_meta: Some(true),
    };
    let hits = store.search(collection_name, req).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, id);
    assert!((hits[0].score - 1.0).abs() < 0.0001); // Cosine(v, v) = 1.0

    // 4. Delete
    store.delete(collection_name, id).unwrap();
    let hits_after = store
        .search(
            collection_name,
            SearchRequest {
                vector,
                k: 1,
                filters: None,
                include_meta: None,
            },
        )
        .unwrap();
    assert!(hits_after.is_empty());
}
