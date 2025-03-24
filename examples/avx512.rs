use aligned_vec::avec;
use diol::prelude::*;
use gemm::gemm;
use gemm_x86::*;
use rand::prelude::*;

fn bench_gemm(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
    let rng = &mut StdRng::seed_from_u64(0);
    let cs = Ord::max(4096, m.next_multiple_of(8));
    let _ = gemm_common::cache::kernel_params(m, n, k, 4 * 8, 6, 8);

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

        // let mut row_chunk = [96, 48];
        // let mut col_chunk = [4, 4, 4];

        // if m >= 2 * n {
        //     row_chunk[0] *= m.div_ceil(n);
        // }

        // if m + n >= 16384 {
        //     col_chunk[1] = 1024;
        //     if n < 2 * m {
        //         col_chunk[0] = col_chunk[1];
        //     }
        // } else if m + n >= 8192 {
        //     col_chunk[1] = 512;
        //     if n < 2 * m {
        //         col_chunk[0] = col_chunk[1];
        //     }
        // } else if m + n >= 6144 {
        //     col_chunk[1] = 256;
        //     if n < 2 * m {
        //         col_chunk[0] = col_chunk[1];
        //     }
        // } else if m + n >= 4096 {
        //     col_chunk[1] = 128;
        //     if n < 2 * m {
        //         col_chunk[0] = col_chunk[1];
        //     }
        // }

        // if m >= 4 * n {
        //     col_chunk[0] = n;
        // } else if n >= 4 * m {
        //     row_chunk[0] = m;
        //     col_chunk[0] = n;
        //     col_chunk[1] = n;
        // }

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
            &SIMD512,
            len,
            sizeof as usize,
            lhs.as_ptr(),
            if PACK_LHS {
                packed_lhs.as_mut_ptr()
            } else {
                lhs.as_ptr() as _
            },
            rhs.as_ptr(),
            if PACK_RHS {
                packed_rhs.as_mut_ptr()
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
                ptr: dst.as_mut_ptr(),
                rs: sizeof,
                cs: cs as isize * sizeof,
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
                alpha: 1.0,
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
