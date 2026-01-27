use crate::search::types::{Document, DocumentMetadata};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

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
        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;

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

        // Remaining is content
        let content_len = total_len - 4 - meta_len - 4 - vector_len;
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
        let mut meta_buf = vec![0u8; meta_len as usize];
        file.read_exact(&mut meta_buf)?;
        
        file.read_exact(&mut len_buf)?;
        let vector_len = u32::from_le_bytes(len_buf);
        let mut vector_buf = vec![0u8; vector_len as usize];
        file.read_exact(&mut vector_buf)?;

        let content_len = total_len - 4 - meta_len - 4 - vector_len;
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
        let reader = match &mut self.reader {
            Some(r) => r,
            None => return None,
        };

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

        // Read Meta
        let mut meta_buf = vec![0u8; meta_len as usize];
        if let Err(e) = reader.read_exact(&mut meta_buf) {
            return Some(Err(e));
        }
        self.offset += meta_len as u64;

        // Skip Vector + Content
        let remaining = total_len - 4 - meta_len;
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
