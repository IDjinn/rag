use std::time::Instant;
use wide::f32x8;

/// Plain scalar dot product; reference implementation for tests and remainder work.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// SIMD dot product over f32x8 lanes with two independent accumulators
/// (breaks the FMA dependency chain). Requires a.len() == b.len().
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let (mut acc0, mut acc1) = (f32x8::ZERO, f32x8::ZERO);
    let mut i = 0;
    while i + 16 <= n {
        let mut la = [0.0f32; 8];
        let mut lb = [0.0f32; 8];
        la.copy_from_slice(&a[i..i + 8]);
        lb.copy_from_slice(&b[i..i + 8]);
        acc0 += f32x8::new(la) * f32x8::new(lb);

        let mut la2 = [0.0f32; 8];
        let mut lb2 = [0.0f32; 8];
        la2.copy_from_slice(&a[i + 8..i + 16]);
        lb2.copy_from_slice(&b[i + 8..i + 16]);
        acc1 += f32x8::new(la2) * f32x8::new(lb2);
        i += 16;
    }
    let mut sum = (acc0 + acc1).reduce_add();
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// In-place L2 normalization (SIMD-friendly: squares then scales).
pub fn l2_normalize(row: &mut [f32]) {
    let n = row.len();
    let (mut acc0, mut acc1) = (f32x8::ZERO, f32x8::ZERO);
    let mut i = 0;
    while i + 16 <= n {
        let mut la = [0.0f32; 8];
        la.copy_from_slice(&row[i..i + 8]);
        let va = f32x8::new(la);
        acc0 += va * va;
        let mut lb = [0.0f32; 8];
        lb.copy_from_slice(&row[i + 8..i + 16]);
        let vb = f32x8::new(lb);
        acc1 += vb * vb;
        i += 16;
    }
    let mut norm = (acc0 + acc1).reduce_add();
    while i < n {
        norm += row[i] * row[i];
        i += 1;
    }
    let norm = norm.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Deterministic pseudo-random vector for tests (xorshift).
#[cfg(test)]
fn pseudo_vector(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 20_000) as f32 / 10_000.0) - 1.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn simd_matches_scalar() {
        for len in [1usize, 7, 8, 16, 100, 767, 768, 769, 4096] {
            let a = pseudo_vector(len as u64 + 1, len);
            let b = pseudo_vector(len as u64 + 42, len);
            let scalar = dot_scalar(&a, &b);
            let simd = dot(&a, &b);
            let denom = scalar.abs().max(1.0);
            assert!(
                (simd - scalar).abs() / denom < 1e-4,
                "len {len}: simd {simd} vs scalar {scalar}"
            );
        }
    }

    #[test]
    fn normalize_makes_unit_length() {
        let mut v = pseudo_vector(7, 768);
        l2_normalize(&mut v);
        let norm = dot_scalar(&v, &v).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
    }

    /// Report SIMD+rayon search throughput for a realistic index size.
    /// No hard timing assertion (CI variance) — run with `cargo test -- --nocapture`.
    #[test]
    fn throughput_report() {
        const ROWS: usize = 50_000;
        const DIM: usize = 768;
        let query = pseudo_vector(999, DIM);
        let table: Vec<Vec<f32>> = (0..ROWS).map(|r| pseudo_vector(r as u64 + 11, DIM)).collect();

        let t0 = Instant::now();
        let best = table
            .par_iter()
            .enumerate()
            .map(|(r, row)| (dot(row, &query), r))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, r)| r)
            .unwrap_or(0);
        let elapsed = t0.elapsed();
        let mrps = ROWS as f64 / elapsed.as_secs_f64() / 1e6;
        eprintln!(
            "SIMD search: {ROWS} x {DIM} rows in {:.1?} ({mrps:.1}M rows/s), best row {best}",
            elapsed
        );
        assert!(best < ROWS);
    }
}
