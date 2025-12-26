#[inline]
pub fn dot(a: &[f32], b: &[f32], simd_enabled: bool) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if simd_enabled {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && a.len() >= 8 {
                unsafe {
                    return dot_avx2(a, b);
                }
            }
        }
    }
    dot_scalar(a, b)
}

#[inline]
pub fn dot_and_norms(a: &[f32], b: &[f32], simd_enabled: bool) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    if simd_enabled {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && a.len() >= 8 {
                unsafe {
                    return accumulate_avx2(a, b);
                }
            }
        }
    }
    accumulate_scalar(a, b)
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn accumulate_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    (dot, norm_a, norm_b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= a.len() {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
    let mut sum = tmp.iter().sum::<f32>();
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    use std::arch::x86_64::*;

    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= a.len() {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        dot = _mm256_add_ps(dot, _mm256_mul_ps(va, vb));
        norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(va, va));
        norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(vb, vb));
        i += 8;
    }

    let mut dot_tmp = [0f32; 8];
    let mut norm_a_tmp = [0f32; 8];
    let mut norm_b_tmp = [0f32; 8];
    _mm256_storeu_ps(dot_tmp.as_mut_ptr(), dot);
    _mm256_storeu_ps(norm_a_tmp.as_mut_ptr(), norm_a);
    _mm256_storeu_ps(norm_b_tmp.as_mut_ptr(), norm_b);

    let mut dot_sum = dot_tmp.iter().sum::<f32>();
    let mut norm_a_sum = norm_a_tmp.iter().sum::<f32>();
    let mut norm_b_sum = norm_b_tmp.iter().sum::<f32>();

    while i < a.len() {
        let x = a[i];
        let y = b[i];
        dot_sum += x * y;
        norm_a_sum += x * x;
        norm_b_sum += y * y;
        i += 1;
    }

    (dot_sum, norm_a_sum, norm_b_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn dot_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(42);
        for dim in [8usize, 17, 384, 768, 1024] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            let scalar = dot_scalar(&a, &b);
            let simd = dot(&a, &b, true);
            assert!(
                approx_close(scalar, simd, 1e-4),
                "dim={dim} scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn dot_and_norms_match_scalar() {
        let mut rng = StdRng::seed_from_u64(7);
        for dim in [8usize, 33, 512, 1280] {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
            let scalar = accumulate_scalar(&a, &b);
            let simd = dot_and_norms(&a, &b, true);
            assert!(approx_close(scalar.0, simd.0, 1e-4));
            assert!(approx_close(scalar.1, simd.1, 1e-3));
            assert!(approx_close(scalar.2, simd.2, 1e-3));
        }
    }

    fn approx_close(expected: f32, actual: f32, eps: f32) -> bool {
        let allowance = eps.max(expected.abs() * 1e-5);
        (expected - actual).abs() <= allowance
    }
}
