//! `DstKind::Upper` / `DstKind::Lower` must leave every entry outside the triangle alone.
//!
//! The store epilogue selects a shifted mask from `mask.data`; when a vector row lies
//! entirely past the diagonal the index into that table goes negative and, before the fix,
//! was only clamped at the top. It read a mask from before the table and stored a full
//! vector below the diagonal. Both instruction sets are exercised by passing `InstrSet`
//! explicitly, so an AVX-512 host still covers the 256-bit path and vice versa.

#![cfg(target_arch = "x86_64")]

use private_gemm_x86::{gemm, Accum, DType, DstKind, IType, InstrSet};

type C64 = [f64; 2];

unsafe fn violations(instr: InstrSet, dtype: DType, kind: DstKind, m: usize, n: usize, k: usize)
    -> usize
{
    let sentinel = 7.0_f64;
    let mut bad = 0;
    match dtype {
        DType::F64 => {
            let a: Vec<f64> = (0..m * k).map(|i| 0.5 + (i % 7) as f64).collect();
            let b: Vec<f64> = (0..k * n).map(|i| 1.0 + (i % 5) as f64).collect();
            let d: Vec<f64> = (0..k).map(|i| 1.25 + i as f64).collect();
            let alpha = 0.75_f64;
            let mut c = vec![sentinel; m * n];
            gemm(dtype, IType::U64, instr, m, n, k,
                 c.as_mut_ptr() as *mut (), 1, m as isize,
                 core::ptr::null(), core::ptr::null(), kind, Accum::Replace,
                 a.as_ptr() as *const (), 1, m as isize, true,
                 d.as_ptr() as *const (), 1isize,
                 b.as_ptr() as *const (), 1, k as isize, false,
                 &alpha as *const f64 as *const (), 1);
            for j in 0..n {
                for i in 0..m {
                    let stored = match kind {
                        DstKind::Upper => i <= j,
                        DstKind::Lower => i >= j,
                        DstKind::Full => true,
                    };
                    if !stored && c[i + j * m] != sentinel {
                        bad += 1;
                    }
                }
            }
        }
        _ => {
            let a: Vec<C64> = (0..m * k).map(|i| [0.5 + (i % 7) as f64, 0.25]).collect();
            let b: Vec<C64> = (0..k * n).map(|i| [1.0 + (i % 5) as f64, -0.5]).collect();
            let d: Vec<f64> = (0..k).map(|i| 1.25 + i as f64).collect();
            let alpha: C64 = [0.75, -0.25];
            let mut c = vec![[sentinel, 2.0]; m * n];
            gemm(dtype, IType::U64, instr, m, n, k,
                 c.as_mut_ptr() as *mut (), 1, m as isize,
                 core::ptr::null(), core::ptr::null(), kind, Accum::Replace,
                 a.as_ptr() as *const (), 1, m as isize, true,
                 d.as_ptr() as *const (), 1isize,
                 b.as_ptr() as *const (), 1, k as isize, false,
                 &alpha as *const C64 as *const (), 1);
            for j in 0..n {
                for i in 0..m {
                    let stored = match kind {
                        DstKind::Upper => i <= j,
                        DstKind::Lower => i >= j,
                        DstKind::Full => true,
                    };
                    if !stored && c[i + j * m][0] != sentinel {
                        bad += 1;
                    }
                }
            }
        }
    }
    bad
}

fn sweep(instr: InstrSet) -> usize {
    let mut total = 0;
    unsafe {
        for dtype in [DType::F64, DType::C64] {
            for kind in [DstKind::Upper, DstKind::Lower] {
                for m in 1..=9 {
                    for n in 1..=9 {
                        for k in 1..=3 {
                            total += violations(instr, dtype, kind, m, n, k);
                        }
                    }
                }
            }
        }
    }
    total
}

#[test]
fn avx256_leaves_entries_outside_the_triangle_alone() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
        eprintln!("avx2/fma unavailable; the 256-bit path cannot run here");
        return;
    }
    assert_eq!(sweep(InstrSet::Avx256), 0);
}

#[test]
fn avx512_leaves_entries_outside_the_triangle_alone() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
        eprintln!("avx512f unavailable; the 512-bit path cannot run here");
        return;
    }
    assert_eq!(sweep(InstrSet::Avx512), 0);
}
