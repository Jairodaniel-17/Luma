//! Disk-backed store for q8-quantized vectors, so the quantized codes stop
//! living in a `HashMap<String, QuantizedVec>` on the heap (~`dim` bytes each)
//! and instead sit in a memory-mapped file where the OS keeps only the hot
//! pages resident. This is what lets a collection's RAM stop growing linearly
//! with the vector count — the foundation for scaling to hundreds of millions
//! of vectors on a small box.
//!
//! Layout: [Q8Header][record 0][record 1]... where each record is a little
//! `scale: f32` followed by `dim` `i8` codes. Records are appended in lockstep
//! with the raw-vector mmap, so an id's index into one is its index into the
//! other. The whole file is a derived cache: it can always be rebuilt from the
//! raw vectors (which are durable in the run WAL), so it needs no critical
//! fsync of its own.

use super::q8::QuantizedVec;
use memmap2::MmapMut;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

pub const Q8_MAGIC: [u8; 4] = *b"LQ8V";
pub const Q8_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Q8Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub dim: u32,
    pub count: u32,
    pub capacity: u32,
}

unsafe impl bytemuck::Zeroable for Q8Header {}
unsafe impl bytemuck::Pod for Q8Header {}

pub struct Q8Mmap {
    file: File,
    mmap: MmapMut,
    dim: usize,
}

impl Q8Mmap {
    /// Bytes per record: a leading f32 scale followed by `dim` i8 codes.
    fn record_bytes(dim: usize) -> usize {
        std::mem::size_of::<f32>() + dim
    }

    pub fn create_or_open(
        path: impl AsRef<Path>,
        dim: usize,
        initial_capacity: usize,
    ) -> io::Result<Self> {
        let path = path.as_ref();
        let exists = path.exists();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let header_size = std::mem::size_of::<Q8Header>();
        let record_bytes = Self::record_bytes(dim);

        if !exists || file.metadata()?.len() == 0 {
            let initial_size = header_size + (initial_capacity * record_bytes);
            file.set_len(initial_size as u64)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            let header = Q8Header {
                magic: Q8_MAGIC,
                version: Q8_VERSION,
                dim: dim as u32,
                count: 0,
                capacity: initial_capacity as u32,
            };
            mmap[..header_size].copy_from_slice(bytemuck::bytes_of(&header));
            mmap.flush()?;
            return Ok(Self { file, mmap, dim });
        }

        if file.metadata()?.len() < header_size as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "q8 mmap smaller than header",
            ));
        }
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        if mmap.len() < header_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "q8 mmap smaller than header",
            ));
        }
        let header: &Q8Header = bytemuck::from_bytes(&mmap[..header_size]);
        if header.magic != Q8_MAGIC || header.version != Q8_VERSION || header.dim as usize != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid q8 mmap header or dimension mismatch",
            ));
        }
        // Reject a corrupt/inflated count that would drive out-of-range reads.
        let needed = (header.count as u64)
            .checked_mul(record_bytes as u64)
            .and_then(|body| body.checked_add(header_size as u64));
        match needed {
            Some(needed) if needed <= mmap.len() as u64 => {}
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "q8 mmap header count exceeds file length",
                ))
            }
        }
        Ok(Self { file, mmap, dim })
    }

    fn header(&self) -> &Q8Header {
        bytemuck::from_bytes(&self.mmap[..std::mem::size_of::<Q8Header>()])
    }

    fn header_mut(&mut self) -> &mut Q8Header {
        bytemuck::from_bytes_mut(&mut self.mmap[..std::mem::size_of::<Q8Header>()])
    }

    pub fn count(&self) -> usize {
        self.header().count as usize
    }

    /// Append a quantized vector, returning its index (== the raw-vector index
    /// when appended in lockstep).
    pub fn append(&mut self, q: &QuantizedVec) -> io::Result<usize> {
        if q.data.len() != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "q8 dimension mismatch",
            ));
        }
        let mut count = self.header().count as usize;
        if count >= self.header().capacity as usize {
            self.grow()?;
            count = self.header().count as usize;
        }
        let header_size = std::mem::size_of::<Q8Header>();
        let record_bytes = Self::record_bytes(self.dim);
        let offset = header_size + (count * record_bytes);
        self.mmap[offset..offset + 4].copy_from_slice(&q.scale.to_le_bytes());
        // i8 -> u8 byte view for the code slice.
        let codes: &[u8] = bytemuck::cast_slice(&q.data);
        self.mmap[offset + 4..offset + 4 + self.dim].copy_from_slice(codes);
        self.header_mut().count += 1;
        Ok(count)
    }

    fn grow(&mut self) -> io::Result<()> {
        self.mmap.flush()?;
        let current = self.header().capacity as usize;
        let new_capacity = (current * 2).max(1024);
        let header_size = std::mem::size_of::<Q8Header>();
        let new_size = header_size + (new_capacity * Self::record_bytes(self.dim));
        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.header_mut().capacity = new_capacity as u32;
        Ok(())
    }

    /// Read the q8 record at `index`, returning `(scale, codes)`. Returns `None`
    /// on an out-of-range index rather than panicking.
    pub fn get(&self, index: usize) -> Option<(f32, &[i8])> {
        if index >= self.header().count as usize {
            return None;
        }
        let header_size = std::mem::size_of::<Q8Header>();
        let record_bytes = Self::record_bytes(self.dim);
        let offset = header_size + (index * record_bytes);
        let end = offset.checked_add(record_bytes)?;
        if end > self.mmap.len() {
            return None;
        }
        let mut scale_bytes = [0u8; 4];
        scale_bytes.copy_from_slice(&self.mmap[offset..offset + 4]);
        let scale = f32::from_le_bytes(scale_bytes);
        let codes: &[i8] = bytemuck::cast_slice(&self.mmap[offset + 4..end]);
        Some((scale, codes))
    }

    /// Materialize the record at `index` as an owned `QuantizedVec` (for the
    /// few paths that still need ownership, e.g. building the disk graph).
    pub fn get_owned(&self, index: usize) -> Option<QuantizedVec> {
        let (scale, codes) = self.get(index)?;
        Some(QuantizedVec::new(scale, codes.to_vec()))
    }

    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("luma_q8mmap_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn append_get_roundtrip() {
        let path = tmp("roundtrip");
        let _ = std::fs::remove_file(&path);
        let mut s = Q8Mmap::create_or_open(&path, 4, 2).unwrap();
        let a = QuantizedVec::new(0.5, vec![1, -2, 3, -4]);
        let b = QuantizedVec::new(1.25, vec![-5, 6, -7, 8]);
        assert_eq!(s.append(&a).unwrap(), 0);
        assert_eq!(s.append(&b).unwrap(), 1);
        // Force a grow (capacity was 2).
        let c = QuantizedVec::new(2.0, vec![9, 9, 9, 9]);
        assert_eq!(s.append(&c).unwrap(), 2);
        let (sc, codes) = s.get(1).unwrap();
        assert_eq!(sc, 1.25);
        assert_eq!(codes, &[-5i8, 6, -7, 8]);
        assert_eq!(s.get_owned(2).unwrap().data, vec![9i8, 9, 9, 9]);
        assert!(s.get(3).is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reopen_preserves_records() {
        let path = tmp("reopen");
        let _ = std::fs::remove_file(&path);
        {
            let mut s = Q8Mmap::create_or_open(&path, 3, 16).unwrap();
            s.append(&QuantizedVec::new(0.1, vec![1, 2, 3])).unwrap();
            s.flush().unwrap();
        }
        let s = Q8Mmap::create_or_open(&path, 3, 16).unwrap();
        assert_eq!(s.count(), 1);
        assert_eq!(s.get(0).unwrap().1, &[1i8, 2, 3]);
        let _ = std::fs::remove_file(&path);
    }
}
