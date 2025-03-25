use core::ptr::null_mut;

use aligned_vec::avec;
use diol::prelude::*;
use gemm::gemm;
use gemm_x86::*;
use rand::prelude::*;

fn bench_gemm(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
    let rng = &mut StdRng::seed_from_u64(0);
    let cs = Ord::max(4096, m.next_multiple_of(8));
    let params = gemm_common::cache::kernel_params(m, n, k, 4 * 8, 6, 8);

    let _ = params;

    let lhs = &mut *avec![[4096]| 0.0; cs * k];
    let rhs = &mut *avec![[4096]| 0.0; k * n];
    let dst = &mut *avec![[4096]| 0.0; cs * n];

    rng.fill(lhs);
    rng.fill(rhs);

    bencher.bench(|| unsafe {
        gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            cs as isize,
            1,
            true,
            lhs.as_ptr(),
            cs as isize,
            1,
            rhs.as_ptr(),
            k as isize,
            1,
            1.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    });
}

// low level problem: optimizing the matmul microkernel(s)
// we need an efficient impl of A * B where A: (48 × k), B: (k, 4)
// we need an efficient impl of A * B where A: (48 × k), B: (k, 3)
// we need an efficient impl of A * B where A: (48 × k), B: (k, 2)
// we need an efficient impl of A * B where A: (48 × k), B: (k, 1)
//
// we need an efficient impl of A * B where A: (40 × k), B: (k, 4)
// we need an efficient impl of A * B where A: (40 × k), B: (k, 3)
// we need an efficient impl of A * B where A: (40 × k), B: (k, 2)
// we need an efficient impl of A * B where A: (40 × k), B: (k, 1)
//
// we need an efficient impl of A * B where A: (32 × k), B: (k, 4)
// we need an efficient impl of A * B where A: (32 × k), B: (k, 3)
// we need an efficient impl of A * B where A: (32 × k), B: (k, 2)
// we need an efficient impl of A * B where A: (32 × k), B: (k, 1)
//
// ...

// mid level problem: optimizing the data layout
// if A or B have a layout that the microkernels don't like,
// it can be more efficient to copy them into optimized storage first,
// then do the matmul

// mid level problem #2: multithreading

// high level problem: optimizing the splitting hierarchy
// C += A * B
//
// row split:
//       C0
//   C = C1
//
//       A0
//   A = A1
//
//   C0 += A0 * B
//   C1 += A1 * B
//
// column split:
//   C = C0 C1
//
//   B = B0 B1
//
//   C0 += A * B0
//   C1 += A * B1
//
// depth split:
//   A = A0 A1
//
//       B0
//   B = B1
//
//   C += A0 * B0
//   C += A1 * B1

fn bench_asm<const PACK_LHS: bool, const PACK_RHS: bool>(
    bencher: Bencher,
    (m, n, k): (usize, usize, usize),
) {
    let rng = &mut StdRng::seed_from_u64(0);
    let sizeof = size_of::<f64>() as isize;
    let len = 64 / size_of::<f64>();
    let cs = Ord::max(4096, m.next_multiple_of(8));

    let packed_lhs = &mut *avec![[4096]| 0.0; m.next_multiple_of(8) * k];
    let packed_rhs = &mut *avec![[4096]| 0.0; n.next_multiple_of(48) * k];

    let lhs = &mut *avec![[4096]| 0.0; cs * k];
    let rhs = &mut *avec![[4096]| 0.0; k * n];
    let dst = &mut *avec![[4096]| 0.0; cs * n];

    rng.fill(lhs);
    rng.fill(rhs);

    bencher.bench(|| unsafe {
        let l = 1024 / 48 * 48;

        let mut row_chunk = [8 * l, 4 * l, 2 * l, l, 96, 48usize];
        let mut col_chunk = [8 * 512, 4 * 512, 2 * 512, 512, 4, 4, 4usize];
        if n >= 2 * m {
            row_chunk = [8 * l, 4 * l, 2 * l, l, 96, 48];
            col_chunk = [8 * 768, 4 * 768, 2 * 768, 768, 128, 128, 4];
        }

        // A and B in this benchmark are column major
        // row stride is    sizeof(T) * row_distance
        // column stride is sizeof(T) * row_dim * col_distance
        let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
        let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

        let mut packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
        let mut packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

        packed_rhs_cs[0] = 0;
        for i in 0..col_chunk.len() - 1 {
            if col_chunk[i] >= n {
                packed_lhs_rs[i] = 0;
            }
        }
        for i in 0..row_chunk.len() {
            if row_chunk[i] >= m {
                packed_rhs_cs[i + 1] = 0;
            }
        }

        kernel(
            &F64_SIMD512,
            len,
            sizeof as usize,
            lhs.as_ptr() as _,
            if PACK_LHS {
                packed_lhs.as_mut_ptr() as _
            } else {
                lhs.as_ptr() as _
            },
            rhs.as_ptr() as _,
            if PACK_RHS {
                packed_rhs.as_mut_ptr() as _
            } else {
                rhs.as_ptr() as _
            },
            m,
            n,
            &row_chunk,
            &col_chunk,
            &lhs_rs,
            &rhs_cs,
            &if PACK_LHS { packed_lhs_rs } else { lhs_rs },
            &if PACK_RHS { packed_rhs_cs } else { rhs_cs },
            0,
            0,
            &Dst {
                ptr: dst.as_mut_ptr() as _,
                rs: sizeof,
                cs: cs as isize * sizeof,
                row_idx: null_mut(),
                col_idx: null_mut(),
            },
            &mut MicrokernelInfo {
                flags: 0,
                depth: k,
                lhs_rs: sizeof,
                lhs_cs: cs as isize * sizeof,
                rhs_rs: sizeof,
                rhs_cs: k as isize * sizeof,
                __pad_0__: 0,
                __pad_1__: 0,
                alpha: &raw const *&1.0 as _,
            },
        )
    });
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);

    let k = 512;

    bench.register_many(
        list![
            (bench_asm::<true, false>),
            (bench_asm::<true, true>),
            bench_gemm,
        ],
        [
            (128, 1024, k),
            (256, 1024, k),
            (512, 1024, k),
            (1024, 128, k),
            (1024, 256, k),
            (1024, 512, k),
            (48 * 2, 48 * 2, k),
            (48 * 4, 48 * 4, k),
            (48 * 2, 48 * 64, k),
            (48 * 64, 192, k),
            (192, 48 * 64, k),
            (1 * 1024, 1 * 1024, k),
            (2 * 1024, 2 * 1024, k),
            (3 * 1024, 3 * 1024, k),
            (4 * 1024, 4 * 1024, k),
            (6 * 1024, 6 * 1024, k),
            (7 * 1024, 7 * 1024, k),
            (8 * 1024, 8 * 1024, k),
            (16 * 1024, 16 * 1024, k),
            (8 * 1024, 1024, k),
            (1024, 8 * 1024, k),
            (2048, 8 * 1024, k),
            (4 * 1024, 1024, k),
            (1024, 4 * 1024, k),
        ],
    );

    bench.run()?;
    Ok(())
}
