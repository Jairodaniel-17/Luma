use luma::engine::chunking::ChunkingEngine;

#[test]
fn test_long_document_chunking() {
    let engine = ChunkingEngine::new(100, 20);
    // Create a mock document with several paragraphs
    let mut text = String::new();
    for i in 0..10 {
        text.push_str(&format!("This is paragraph number {}. It contains some text that will be used to test the chunking engine and its overlap capabilities.

", i));
    }
    
    let chunks = engine.split_text(&text);
    
    // We expect multiple chunks, not just 1, and no chunk should be significantly larger than chunk_size
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert!(chunk.len() <= 100 + 50); // slight buffer for word completion if needed
    }
    
    // Check overlap: The end of chunk 0 should share some words with the beginning of chunk 1
    let last_words_0: Vec<&str> = chunks[0].split_whitespace().rev().take(5).collect();
    let first_words_1: Vec<&str> = chunks[1].split_whitespace().take(10).collect();
    
    // We reverse last_words_0 back to normal order for comparison, or just check if any word intersects
    let mut intersection = false;
    for w in last_words_0 {
        if first_words_1.contains(&w) {
            intersection = true;
            break;
        }
    }
    assert!(intersection, "There should be overlap between consecutive chunks");
}
