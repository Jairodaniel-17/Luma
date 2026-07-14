use crate::search::types::{Document, DocumentMetadata};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Upper bound on any single length field read from the log. Lengths are
/// untrusted u32 values; a torn or malicious record could otherwise request a
/// ~4GiB allocation (OOM) or, via unchecked subtraction, underflow and wrap.
const MAX_RECORD_BYTES: u32 = 256 * 1024 * 1024;

/// Compute `content_len = total_len - 4 - meta_len - 4 - vector_len` without
/// underflowing (which panics in debug and wraps to a huge value in release).
/// Returns an `InvalidData` error for a truncated/torn record instead.
fn content_len_checked(total_len: u32, meta_len: u32, vector_len: u32) -> io::Result<u32> {
    total_len
        .checked_sub(4)
        .and_then(|v| v.checked_sub(meta_len))
        .and_then(|v| v.checked_sub(4))
        .and_then(|v| v.checked_sub(vector_len))
        .filter(|&len| len <= MAX_RECORD_BYTES)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt record: inconsistent length fields",
            )
        })
}

pub struct AppendLog {
    path: PathBuf,
}

impl AppendLog {
    pub fn new(path: PathBuf) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self { path })
    }

    pub fn append(&self, doc: &Document) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        let meta_data = (doc.id, &doc.metadata);
        let meta_bytes = serde_json::to_vec(&meta_data)?;
        let vector_bytes = bincode::serialize(&doc.vector)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let content_bytes = doc.content.as_bytes();

        let meta_len = meta_bytes.len() as u32;
        let vector_len = vector_bytes.len() as u32;
        let content_len = content_bytes.len() as u32;

        // Total len excluding the TotalLen field itself
        // Structure: [TotalLen:4][MetaLen:4][MetaBytes][VectorLen:4][VectorBytes][ContentBytes]
        let total_len = 4 + meta_len + 4 + vector_len + content_len;

        file.write_all(&total_len.to_le_bytes())?;
        file.write_all(&meta_len.to_le_bytes())?;
        file.write_all(&meta_bytes)?;
        file.write_all(&vector_len.to_le_bytes())?;
        file.write_all(&vector_bytes)?;
        file.write_all(content_bytes)?;

        Ok(())
    }

    pub fn scan_metadata(&self) -> io::Result<MetadataIterator> {
        // If file doesn't exist, return empty iterator logic (handle in open)
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(MetadataIterator {
                    reader: None,
                    offset: 0,
                });
            }
            Err(e) => return Err(e),
        };
        Ok(MetadataIterator {
            reader: Some(BufReader::new(file)),
            offset: 0,
        })
    }

    pub fn read_vector(&self, offset: u64) -> io::Result<Vec<f32>> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;

        // At offset, we are at [TotalLen].
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        // Skip MetaLen (4)
        file.read_exact(&mut len_buf)?;
        let meta_len = u32::from_le_bytes(len_buf);

        file.seek(SeekFrom::Current(meta_len as i64))?;

        file.read_exact(&mut len_buf)?;
        let vector_len = u32::from_le_bytes(len_buf);

        let mut vec_buf = vec![0u8; vector_len as usize];
        file.read_exact(&mut vec_buf)?;

        bincode::deserialize(&vec_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn read_content(&self, offset: u64) -> io::Result<String> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;

        // At offset, we are at [TotalLen].
        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let total_len = u32::from_le_bytes(len_buf);

        // Skip MetaLen (4)
        file.read_exact(&mut len_buf)?;
        let meta_len = u32::from_le_bytes(len_buf);

        file.seek(SeekFrom::Current(meta_len as i64))?;

        file.read_exact(&mut len_buf)?;
        let vector_len = u32::from_le_bytes(len_buf);

        file.seek(SeekFrom::Current(vector_len as i64))?;

        // Remaining is content. Validate the arithmetic to avoid underflow.
        let content_len = content_len_checked(total_len, meta_len, vector_len)?;
        let mut content_buf = vec![0u8; content_len as usize];
        file.read_exact(&mut content_buf)?;

        String::from_utf8(content_buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn read_document(&self, offset: u64) -> io::Result<Document> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut len_buf = [0u8; 4];
        file.read_exact(&mut len_buf)?;
        let total_len = u32::from_le_bytes(len_buf);

        file.read_exact(&mut len_buf)?;
        let meta_len = u32::from_le_bytes(len_buf);
        if meta_len > MAX_RECORD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt record: meta length too large",
            ));
        }
        let mut meta_buf = vec![0u8; meta_len as usize];
        file.read_exact(&mut meta_buf)?;

        file.read_exact(&mut len_buf)?;
        let vector_len = u32::from_le_bytes(len_buf);
        if vector_len > MAX_RECORD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt record: vector length too large",
            ));
        }
        let mut vector_buf = vec![0u8; vector_len as usize];
        file.read_exact(&mut vector_buf)?;

        let content_len = content_len_checked(total_len, meta_len, vector_len)?;
        let mut content_buf = vec![0u8; content_len as usize];
        file.read_exact(&mut content_buf)?;

        let (id, metadata): (u32, DocumentMetadata) = serde_json::from_slice(&meta_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let vector: Vec<f32> = bincode::deserialize(&vector_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let content = String::from_utf8(content_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(Document {
            id,
            vector,
            content,
            metadata,
        })
    }
}

pub struct MetadataIterator {
    reader: Option<BufReader<File>>,
    offset: u64,
}

impl Iterator for MetadataIterator {
    type Item = io::Result<(u64, u32, DocumentMetadata)>;

    fn next(&mut self) -> Option<Self::Item> {
        let reader = self.reader.as_mut()?;

        let start_offset = self.offset;
        let mut len_buf = [0u8; 4];

        // Read TotalLen
        if let Err(e) = reader.read_exact(&mut len_buf) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                return None;
            }
            return Some(Err(e));
        }
        let total_len = u32::from_le_bytes(len_buf);
        self.offset += 4;

        // Read MetaLen
        if let Err(e) = reader.read_exact(&mut len_buf) {
            return Some(Err(e));
        }
        let meta_len = u32::from_le_bytes(len_buf);
        self.offset += 4;

        // Bound the untrusted meta length before allocating to avoid OOM.
        if meta_len > MAX_RECORD_BYTES {
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt record: meta length too large",
            )));
        }
        // Read Meta
        let mut meta_buf = vec![0u8; meta_len as usize];
        if let Err(e) = reader.read_exact(&mut meta_buf) {
            return Some(Err(e));
        }
        self.offset += meta_len as u64;

        // Skip Vector + Content. Use checked_sub so a torn record with
        // total_len < 4 + meta_len does not underflow (panic/huge seek).
        let Some(remaining) = total_len
            .checked_sub(4)
            .and_then(|v| v.checked_sub(meta_len))
        else {
            return Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt record: inconsistent length fields",
            )));
        };
        if let Err(e) = reader.seek(SeekFrom::Current(remaining as i64)) {
            return Some(Err(e));
        }
        self.offset += remaining as u64;

        let (id, metadata): (u32, DocumentMetadata) = match serde_json::from_slice(&meta_buf) {
            Ok(v) => v,
            Err(e) => return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e))),
        };

        Some(Ok((start_offset, id, metadata)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_len_underflow_returns_error() {
        // total_len smaller than the sum of the length fields must not wrap.
        assert!(content_len_checked(8, 0, 4).is_err());
        assert!(content_len_checked(0, 0, 0).is_err());
        assert!(content_len_checked(4, 100, 0).is_err());
    }

    #[test]
    fn content_len_valid_case() {
        // total_len = 4 + meta_len + 4 + vector_len + content_len(2)
        assert_eq!(content_len_checked(4 + 3 + 4 + 5 + 2, 3, 5).unwrap(), 2);
    }

    #[test]
    fn content_len_rejects_oversized() {
        // Even without underflow, an absurd content_len is rejected.
        // Layout: 4 (total hdr) + meta_len(0) + 4 (vec hdr) + vector_len(0) + content_len.
        let total = 4 + 4 + MAX_RECORD_BYTES.saturating_add(1);
        assert!(content_len_checked(total, 0, 0).is_err());
    }

    #[test]
    fn read_document_torn_record_errors_not_panics() {
        let mut p = std::env::temp_dir();
        p.push(format!("luma_storage_test_{}_torn", std::process::id()));
        // total_len=8, meta_len=0, vector_len=4 + 4 vector bytes.
        // content_len = 8 - 4 - 0 - 4 - 4 would underflow.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(&p, &bytes).unwrap();
        let log = AppendLog::new(p.clone()).unwrap();
        let res = log.read_document(0);
        assert!(res.is_err(), "torn record must error, not panic");
        let _ = std::fs::remove_file(&p);
    }
}
