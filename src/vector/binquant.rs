//! Binary quantization: 1 bit per dimension (sign bit), packed into u64 words.
//! For dim-768 a vector shrinks from 3072 B (f32) or 768 B (q8) to **96 B** — a
//! 32x reduction vs f32. That makes it cheap to keep the *entire* candidate
//! representation resident in RAM even at hundreds of millions of vectors, while
//! the full/q8 vectors stay on disk (mmap) for an exact rescore of the top
//! candidates. This is the tiered-memory model that lets an ANN engine serve
//! very large collections in bounded RAM.
//!
//! Candidate scoring is XOR + popcount (Hamming): the number of dimensions whose
//! signs agree approximates cosine similarity ordering well enough to shortlist
//! candidates, which are then re-ranked exactly.
//!
//! Technique adopted from Qdrant's `quantization` crate (Apache-2.0,
//! github.com/qdrant/qdrant) — reimplemented minimally here (no SIMD/transpose
//! generics) to keep it self-contained. SIMD is the upgrade path if popcount
//! ever shows up in a profile.

/// A binary-quantized vector: one bit per dimension, sign-packed into u64 words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryVec {
    /// Number of source dimensions (bits used in the last word may be padding).
    pub dim: usize,
    pub words: Vec<u64>,
}

impl BinaryVec {
    pub fn word_count(dim: usize) -> usize {
        dim.div_ceil(64)
    }

    pub fn byte_len(&self) -> usize {
        self.words.len() * 8
    }
}

/// Encode a vector: bit set when the component is >= 0. Padding bits (beyond
/// `dim`) stay 0 in both stored and query vectors, so they never affect the XOR
/// count.
pub fn encode(v: &[f32]) -> BinaryVec {
    let dim = v.len();
    let mut words = vec![0u64; BinaryVec::word_count(dim)];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    BinaryVec { dim, words }
}

/// Number of dimensions whose signs disagree (Hamming distance). Lower = more
/// similar.
pub fn hamming(a: &BinaryVec, b: &BinaryVec) -> u32 {
    debug_assert_eq!(a.words.len(), b.words.len());
    a.words
        .iter()
        .zip(b.words.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Similarity score for shortlisting: agreeing dimensions minus disagreeing
/// ones, in [-dim, dim]. Higher = more similar. Equivalent ordering to
/// `dim - 2*hamming`; kept as an f32 so it slots into the existing score sort.
pub fn score(a: &BinaryVec, b: &BinaryVec) -> f32 {
    let dim = a.dim.min(b.dim) as i64;
    let dist = hamming(a, b) as i64;
    (dim - 2 * dist) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_hamming() {
        let v = vec![0.3, -0.2, 0.9, -1.0, 0.0, 0.1, -0.4];
        let a = encode(&v);
        let b = encode(&v);
        assert_eq!(hamming(&a, &b), 0);
        assert_eq!(score(&a, &b), a.dim as f32);
    }

    #[test]
    fn opposite_signs_max_hamming() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        let w = vec![-0.1, -0.2, -0.3, -0.4];
        let a = encode(&v);
        let b = encode(&w);
        assert_eq!(hamming(&a, &b), 4);
        assert_eq!(score(&a, &b), -4.0);
    }

    #[test]
    fn ordering_tracks_cosine_for_a_simple_case() {
        // query closer (in sign pattern) to `near` than to `far`.
        let query = encode(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let near = encode(&[0.9, 0.2, 2.0, 0.1, -0.3, -1.0, -0.5, -0.2]); // all signs match
        let far = encode(&[-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]); // several flipped
        assert!(score(&query, &near) > score(&query, &far));
    }

    #[test]
    fn packs_768_dims_into_96_bytes() {
        let v = vec![0.5f32; 768];
        let b = encode(&v);
        assert_eq!(b.words.len(), 12);
        assert_eq!(b.byte_len(), 96);
        // Every component >= 0 -> all 768 bits set, padding bits (768..768) none.
        assert_eq!(hamming(&b, &encode(&vec![-1.0f32; 768])), 768);
    }
}
