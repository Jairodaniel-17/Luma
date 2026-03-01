pub struct ChunkingEngine {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl Default for ChunkingEngine {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 100,
        }
    }
}

impl ChunkingEngine {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    pub fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();

        let mut i = 0;
        while i < len {
            let mut end = i + self.chunk_size;
            if end >= len {
                end = len;
                let chunk: String = chars[i..end].iter().collect();
                chunks.push(chunk);
                break;
            }

            // Fallback: Just cut strictly if no good separator is found.
            // But let's try to find a natural break (\n\n, \n, ., or space)
            let mut break_point = end;

            // Look back for a natural break within the last 10% of the chunk
            let search_start = if end > (self.chunk_size / 10) {
                end - (self.chunk_size / 10)
            } else {
                i
            };

            for j in (search_start..end).rev() {
                if chars[j] == '\n' || chars[j] == '.' || chars[j] == ' ' {
                    break_point = j + 1; // Include the space/newline
                    break;
                }
            }

            let chunk: String = chars[i..break_point].iter().collect();
            chunks.push(chunk.trim().to_string());

            // Advance, but move back by overlap
            if break_point >= len {
                break;
            }

            // Overlap step
            let next_start = if break_point > self.chunk_overlap {
                break_point - self.chunk_overlap
            } else {
                break_point
            };

            // Ensure we at least advance
            i = std::cmp::max(next_start, i + 1);
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking() {
        let engine = ChunkingEngine::new(20, 5);
        let text = "This is a long text that needs to be split into chunks.";
        let chunks = engine.split_text(text);

        assert_eq!(chunks.len(), 4);
    }
}
