use luma::search::engine::SearchEngine;
use luma::search::types::{Document, DocumentMetadata, SearchRequest};
use tempfile::tempdir;

fn create_engine(dir: &tempfile::TempDir) -> SearchEngine {
    SearchEngine::new(dir.path().to_path_buf()).unwrap()
}

// Helper to inject vector via content so we can control similarity
// engine.embed checks for "TEST_VEC:v1,v2,..."
// But we inject the vector directly into the document struct, which is stored in the log.
// The content is used by engine.embed if we were embedding the document content,
// but here we provide the vector explicitly in Document struct.
// The search function embeds the QUERY.
fn create_doc_with_vec(
    id: u32,
    vec: &[f32],
    group_id: Option<u32>,
    doc_id: Option<&str>,
) -> Document {
    // Content doesn't matter for vector retrieval, but we can set it to TEST_VEC for consistency if we were embedding it
    let vec_str = vec
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let content = format!("TEST_VEC:{}", vec_str);
    Document {
        id,
        vector: vec.to_vec(),
        content,
        metadata: DocumentMetadata {
            filename: None,
            processed_at: None,
            category: None,
            language: None,
            status: None,
            version: None,
            group_id,
            document_id: doc_id.map(|s| s.to_string()),
        },
    }
}

#[test]
fn test_grouping_basic_collapse() {
    let dir = tempdir().unwrap();
    let engine = create_engine(&dir);

    // 1 document, 5 chunks.
    // Doc 1 (id="doc1")
    // Chunks have similar vectors.
    // Query: [1.0, 0.0]
    // Chunks:
    // 1: [0.99, 0.01] (Score high)
    // 2: [0.98, 0.02]
    // 3: [0.97, 0.03]
    // 4: [0.96, 0.04]
    // 5: [0.95, 0.05]

    for i in 0..5 {
        let val = 0.99 - (i as f32 * 0.01);
        let vec = vec![val, 1.0 - val];
        let doc = create_doc_with_vec(i, &vec, None, Some("doc1"));
        engine.ingest(doc).unwrap();
    }

    let req = SearchRequest {
        query: "TEST_VEC:1.0,0.0".to_string(),
        top_k: 10,
        filters: None,
        group_by: Some("document_id".to_string()),
        group_limit: 1,
    };

    let res = engine.search(req).unwrap();
    assert_eq!(res.results.len(), 1);
    assert_eq!(res.results[0].document.id, 0); // Best score is first one
}

#[test]
fn test_grouping_multiple_documents() {
    let dir = tempdir().unwrap();
    let engine = create_engine(&dir);

    // 3 documents, 3 chunks each.
    // Doc A: [0.9, ...] (Score ~0.9)
    // Doc B: [0.8, ...] (Score ~0.8)
    // Doc C: [0.7, ...] (Score ~0.7)

    // Doc A
    for i in 0..3 {
        let vec = vec![0.9, 0.1];
        let doc = create_doc_with_vec(10 + i, &vec, None, Some("docA"));
        engine.ingest(doc).unwrap();
    }
    // Doc B
    for i in 0..3 {
        let vec = vec![0.8, 0.2];
        let doc = create_doc_with_vec(20 + i, &vec, None, Some("docB"));
        engine.ingest(doc).unwrap();
    }
    // Doc C
    for i in 0..3 {
        let vec = vec![0.7, 0.3];
        let doc = create_doc_with_vec(30 + i, &vec, None, Some("docC"));
        engine.ingest(doc).unwrap();
    }

    let req = SearchRequest {
        query: "TEST_VEC:1.0,0.0".to_string(),
        top_k: 10,
        filters: None,
        group_by: Some("document_id".to_string()),
        group_limit: 1,
    };

    let res = engine.search(req).unwrap();
    assert_eq!(res.results.len(), 3);

    // Check order (should be A, B, C)
    assert_eq!(
        res.results[0].document.metadata.document_id.as_deref(),
        Some("docA")
    );
    assert_eq!(
        res.results[1].document.metadata.document_id.as_deref(),
        Some("docB")
    );
    assert_eq!(
        res.results[2].document.metadata.document_id.as_deref(),
        Some("docC")
    );
}

#[test]
fn test_no_grouping() {
    let dir = tempdir().unwrap();
    let engine = create_engine(&dir);

    // 2 docs, 2 chunks each. All distinct scores.
    let docs = vec![
        ("doc1", vec![0.9, 0.1]),
        ("doc1", vec![0.85, 0.15]),
        ("doc2", vec![0.8, 0.2]),
        ("doc2", vec![0.75, 0.25]),
    ];

    for (i, (d, v)) in docs.iter().enumerate() {
        let doc = create_doc_with_vec(i as u32, v, None, Some(d));
        engine.ingest(doc).unwrap();
    }

    let req = SearchRequest {
        query: "TEST_VEC:1.0,0.0".to_string(),
        top_k: 10,
        filters: None,
        group_by: None, // No grouping
        group_limit: 1,
    };

    let res = engine.search(req).unwrap();
    assert_eq!(res.results.len(), 4);
    // Ordered by score
    assert_eq!(res.results[0].score > res.results[1].score, true);
    assert_eq!(res.results[1].score > res.results[2].score, true);
    assert_eq!(res.results[2].score > res.results[3].score, true);
}

#[test]
fn test_determinism() {
    let dir = tempdir().unwrap();
    let engine = create_engine(&dir);

    // Add multiple chunks with same score/content to see if order is stable or at least deterministic
    for i in 0..10 {
        let vec = vec![0.5, 0.5];
        let doc = create_doc_with_vec(i, &vec, None, Some("doc1"));
        engine.ingest(doc).unwrap();
    }

    let req = SearchRequest {
        query: "TEST_VEC:1.0,0.0".to_string(),
        top_k: 10,
        filters: None,
        group_by: Some("document_id".to_string()),
        group_limit: 5,
    };

    let res1 = engine.search(req.clone()).unwrap();
    let res2 = engine.search(req).unwrap();

    assert_eq!(res1.results.len(), res2.results.len());
    for (r1, r2) in res1.results.iter().zip(res2.results.iter()) {
        assert_eq!(r1.document.id, r2.document.id);
        assert_eq!(r1.score, r2.score);
    }
}
