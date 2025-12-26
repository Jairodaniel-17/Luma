use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedVec {
    pub scale: f32,
    pub data: Vec<i8>,
}

impl QuantizedVec {
    pub fn new(scale: f32, data: Vec<i8>) -> Self {
        Self { scale, data }
    }

    pub fn dims(&self) -> usize {
        self.data.len()
    }
}

pub fn quantize_per_vector(vec: &[f32]) -> QuantizedVec {
    let mut max_abs = 0.0f32;
    for &x in vec {
        let ax = x.abs();
        if ax > max_abs {
            max_abs = ax;
        }
    }
    let scale = if max_abs <= f32::EPSILON {
        1.0
    } else {
        max_abs / 127.0
    };
    let mut data = Vec::with_capacity(vec.len());
    for &x in vec {
        let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
        data.push(q);
    }
    QuantizedVec::new(scale, data)
}

pub fn dot(a: &QuantizedVec, b: &QuantizedVec, simd_enabled: bool) -> f32 {
    debug_assert_eq!(a.data.len(), b.data.len());
    let raw = dot_i8_inner(&a.data, &b.data, simd_enabled) as f32;
    raw * (a.scale * b.scale)
}

fn dot_i8_inner(a: &[i8], b: &[i8], simd_enabled: bool) -> i32 {
    if simd_enabled {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && a.len() >= 32 {
                unsafe {
                    return dot_i8_avx2(a, b);
                }
            }
        }
    }
    dot_i8_scalar(a, b)
}

pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32) * (*y as i32))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= a.len() {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        let prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        let prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);
        let ones = _mm256_set1_epi16(1);
        let acc_lo = _mm256_madd_epi16(prod_lo, ones);
        let acc_hi = _mm256_madd_epi16(prod_hi, ones);
        sum = _mm256_add_epi32(sum, acc_lo);
        sum = _mm256_add_epi32(sum, acc_hi);
        i += 32;
    }
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum);
    let mut acc = tmp.iter().sum::<i32>();
    while i < a.len() {
        acc += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn quantize_preserves_length() {
        let vec = vec![0.1, -0.5, 1.2];
        let q = quantize_per_vector(&vec);
        assert_eq!(q.data.len(), vec.len());
        assert!(q.scale > 0.0);
    }

    #[test]
    fn dot_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(99);
        for dim in [8usize, 64, 257] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let qa = quantize_per_vector(&a);
            let qb = quantize_per_vector(&b);
            let approx = dot(&qa, &qb, false);
            let exact: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let diff = (approx - exact).abs();
            assert!(
                diff <= exact.abs() * 0.05 + 0.01,
                "dim={dim} approx={approx} exact={exact} diff={diff}"
            );
        }
    }

    #[test]
    fn dot_simd_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(777);
        for dim in [32usize, 96, 384, 1024] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let qa = quantize_per_vector(&a);
            let qb = quantize_per_vector(&b);
            let simd = dot(&qa, &qb, true);
            let scalar = dot(&qa, &qb, false);
            let diff = (simd - scalar).abs();
            assert!(
                diff <= scalar.abs() * 1e-5 + 1e-3,
                "dim={dim} simd={simd} scalar={scalar} diff={diff}"
            );
        }
    }
}
