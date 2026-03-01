use std::fs::{File, OpenOptions};
use std::io::{self};
use std::path::Path;
use memmap2::MmapMut;

pub const MAGIC_BYTES: [u8; 4] = *b"LUMA";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MmapHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub dim: u32,
    pub count: u32,
    pub capacity: u32,
}

// Ensure header is standard layout and safe for bytemuck
unsafe impl bytemuck::Zeroable for MmapHeader {}
unsafe impl bytemuck::Pod for MmapHeader {}

pub struct VectorMmap {
    file: File,
    mmap: MmapMut,
    dim: usize,
}

impl VectorMmap {
    pub fn create_or_open(path: impl AsRef<Path>, dim: usize, initial_capacity: usize) -> io::Result<Self> {
        let path = path.as_ref();
        let exists = path.exists();
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let header_size = std::mem::size_of::<MmapHeader>();
        let vector_bytes = dim * std::mem::size_of::<f32>();
        
        if !exists || file.metadata()?.len() == 0 {
            let initial_size = header_size + (initial_capacity * vector_bytes);
            file.set_len(initial_size as u64)?;
            
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            let header = MmapHeader {
                magic: MAGIC_BYTES,
                version: VERSION,
                dim: dim as u32,
                count: 0,
                capacity: initial_capacity as u32,
            };
            let header_bytes = bytemuck::bytes_of(&header);
            mmap[..header_size].copy_from_slice(header_bytes);
            mmap.flush()?;
            
            Ok(Self { file, mmap, dim })
        } else {
            let mmap = unsafe { MmapMut::map_mut(&file)? };
            // Verify header
            let header: &MmapHeader = bytemuck::from_bytes(&mmap[..header_size]);
            if header.magic != MAGIC_BYTES || header.version != VERSION || header.dim as usize != dim {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid mmap header or dimension mismatch"));
            }
            Ok(Self { file, mmap, dim })
        }
    }

    pub fn header(&self) -> &MmapHeader {
        let header_size = std::mem::size_of::<MmapHeader>();
        bytemuck::from_bytes(&self.mmap[..header_size])
    }

    fn header_mut(&mut self) -> &mut MmapHeader {
        let header_size = std::mem::size_of::<MmapHeader>();
        bytemuck::from_bytes_mut(&mut self.mmap[..header_size])
    }

    pub fn append(&mut self, vector: &[f32]) -> io::Result<usize> {
        if vector.len() != self.dim {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Dimension mismatch"));
        }

        let mut count = self.header().count as usize;
        let capacity = self.header().capacity as usize;

        if count >= capacity {
            self.grow()?;
            count = self.header().count as usize;
        }

        let header_size = std::mem::size_of::<MmapHeader>();
        let vector_bytes = self.dim * std::mem::size_of::<f32>();
        let offset = header_size + (count * vector_bytes);
        
        let slice = bytemuck::cast_slice(vector);
        self.mmap[offset..offset + vector_bytes].copy_from_slice(slice);
        
        self.header_mut().count += 1;
        
        Ok(count)
    }

    fn grow(&mut self) -> io::Result<()> {
        self.mmap.flush()?;
        let current_capacity = self.header().capacity as usize;
        let new_capacity = (current_capacity * 2).max(1024); // Double or at least 1024
        
        let header_size = std::mem::size_of::<MmapHeader>();
        let vector_bytes = self.dim * std::mem::size_of::<f32>();
        let new_size = header_size + (new_capacity * vector_bytes);
        
        self.file.set_len(new_size as u64)?;
        
        // Remap
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.header_mut().capacity = new_capacity as u32;
        
        Ok(())
    }

    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        if index >= self.header().count as usize {
            return None;
        }
        let header_size = std::mem::size_of::<MmapHeader>();
        let vector_bytes = self.dim * std::mem::size_of::<f32>();
        let offset = header_size + (index * vector_bytes);
        
        Some(bytemuck::cast_slice(&self.mmap[offset..offset + vector_bytes]))
    }
    
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}
