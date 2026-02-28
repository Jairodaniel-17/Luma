use luma::engine::parser::DocumentParser;

#[test]
fn test_plain_text() {
    let content = b"Hello world! This is a test.";
    let extracted = DocumentParser::extract_text("test.txt", content).unwrap();
    assert_eq!(extracted, "Hello world! This is a test.");
}
