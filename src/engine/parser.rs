use std::io::{Read, Cursor};
use anyhow::Result;
use quick_xml::events::Event;
use quick_xml::Reader;

pub struct DocumentParser;

impl DocumentParser {
    pub fn extract_text(filename: &str, content: &[u8]) -> Result<String> {
        let ext = filename.split('.').last().unwrap_or("").to_lowercase();
        match ext.as_str() {
            "txt" | "md" | "json" | "csv" => Self::extract_text_plain(content),
            "pdf" => Self::extract_text_pdf(content),
            "docx" => Self::extract_text_docx(content),
            "epub" => Self::extract_text_epub(content),
            _ => Err(anyhow::anyhow!("Unsupported file extension: {}", ext)),
        }
    }

    fn extract_text_plain(content: &[u8]) -> Result<String> {
        Ok(String::from_utf8_lossy(content).into_owned())
    }

    fn extract_text_pdf(content: &[u8]) -> Result<String> {
        let text = pdf_extract::extract_text_from_mem(content)
            .map_err(|e| anyhow::anyhow!("Failed to parse PDF: {}", e))?;
        Ok(text)
    }

    fn extract_text_docx(content: &[u8]) -> Result<String> {
        let cursor = Cursor::new(content);
        let mut archive = zip::ZipArchive::new(cursor)?;
        
        let mut text = String::new();
        let mut document_xml = archive.by_name("word/document.xml")
            .map_err(|_| anyhow::anyhow!("Invalid DOCX format"))?;
            
        let mut xml_content = String::new();
        document_xml.read_to_string(&mut xml_content)?;
        
        let mut reader = Reader::from_str(&xml_content);
        reader.trim_text(true);
        let mut buf = Vec::new();
        
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Text(e)) => {
                    let unescaped = e.unescape().map_err(|e| anyhow::anyhow!("Error unescaping DOCX: {}", e))?;
                    text.push_str(&unescaped);
                    text.push(' ');
                },
                Ok(Event::Eof) => break,
                Err(e) => return Err(anyhow::anyhow!("Error parsing DOCX XML: {}", e)),
                _ => (),
            }
            buf.clear();
        }
        
        Ok(text.replace("  ", " ").trim().to_string())
    }

    fn extract_text_epub(content: &[u8]) -> Result<String> {
        let cursor = Cursor::new(content);
        let mut archive = zip::ZipArchive::new(cursor)?;
        
        let mut text = String::new();
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            if file.name().ends_with(".html") || file.name().ends_with(".xhtml") {
                let mut html_content = String::new();
                file.read_to_string(&mut html_content)?;
                
                // Simple HTML tag stripper
                let mut in_tag = false;
                for c in html_content.chars() {
                    if c == '<' {
                        in_tag = true;
                    } else if c == '>' {
                        in_tag = false;
                        text.push(' ');
                    } else if !in_tag {
                        text.push(c);
                    }
                }
                text.push('\n');
            }
        }
        
        Ok(text.replace("  ", " ").trim().to_string())
    }
}
