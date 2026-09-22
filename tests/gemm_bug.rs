#![cfg(target_arch = "x86_64")]

use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet, gemm};
use std::ptr::null;

// Both tests compute the lower triangle of C += A * diag(d) * B with identical
// inputs: C = 0 and every entry of A, d and B = 1. Each requested output must
// equal k exactly. The ONLY difference between the tests is the thread count.

// Passes on both the original and fixed implementations.
#[test]
fn sequential_gemm_lower_ones() {
	check_gemm_lower_ones(1);
}

// Fails on the original implementation on our GitHub runner; passes with the fix.
// This makes one public gemm() call with four threads.
#[test]
fn parallel_gemm_lower_ones() {
	check_gemm_lower_ones(4);
}

fn check_gemm_lower_ones(threads: usize) {
	assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"));
	let (m, n, k) = (9361, 4682, 385);
	let a = vec![1.0_f64; m * k];
	let b = vec![1.0_f64; k * n];
	let d = vec![1.0_f64; k];
	let alpha = 1.0_f64;
	let mut c = vec![0.0_f64; m * n];
	// SAFETY: CPU features checked above. All matrices are column-major,
	// with valid dimensions/strides and disjoint storage.
	unsafe {
		gemm(
			DType::F64,
			IType::U64,
			InstrSet::Avx256,
			m,
			n,
			k,
			c.as_mut_ptr().cast(),
			1,
			m as isize,
			null(),
			null(),
			DstKind::Lower,
			Accum::Add,
			a.as_ptr().cast(),
			1,
			m as isize,
			false,
			d.as_ptr().cast(),
			1,
			b.as_ptr().cast(),
			1,
			k as isize,
			false,
			(&alpha as *const f64).cast(),
			threads,
		);
	}
	for j in 0..n {
		for i in j..m {
			assert_eq!(c[i + m * j], k as f64, "threads={threads}: C[{i},{j}]");
		}
	}
}
