use luma::engine::parser::DocumentParser;

#[test]
fn test_plain_text() {
    let content = b"Hello world! This is a test.";
    let extracted = DocumentParser::extract_text("test.txt", content).unwrap();
    assert_eq!(extracted, "Hello world! This is a test.");
}

/// A zip in memory with the given entries, deflated as Word and EPUB writers do.
fn zipped(entries: &[(&str, &str)]) -> Vec<u8> {
    use std::io::Write;
    let mut out = std::io::Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(&mut out);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, body) in entries {
        zip.start_file(*name, options).unwrap();
        zip.write_all(body.as_bytes()).unwrap();
    }
    zip.finish().unwrap();
    out.into_inner()
}

#[test]
fn docx_and_epub_text_comes_out() {
    let docx = zipped(&[(
        "word/document.xml",
        "<w:document><w:body><w:p><w:r><w:t>Orden de compra</w:t></w:r></w:p>\
         <w:p><w:r><w:t>Friochap &amp; Cia</w:t></w:r></w:p></w:body></w:document>",
    )]);
    let text = DocumentParser::extract_text("orden.docx", &docx).unwrap();
    assert_eq!(text, "Orden de compra Friochap & Cia");

    let epub = zipped(&[
        ("mimetype", "application/epub+zip"),
        (
            "OEBPS/cap1.xhtml",
            "<html><body><p>Capitulo uno</p></body></html>",
        ),
    ]);
    let text = DocumentParser::extract_text("libro.epub", &epub).unwrap();
    assert!(text.contains("Capitulo uno"), "got: {text}");
    assert!(!text.contains("epub+zip"), "only (x)html entries are text");
}
