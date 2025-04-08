use core::ptr::{null, null_mut};

use aligned_vec::avec;
use diol::{Picoseconds, config::*, prelude::*};
use gemm::gemm;
use gemm_x86::*;
use rand::prelude::*;

use gemm_common::cache::CacheInfo;
use std::fs;

fn bench_gemm(bencher: Bencher, (m, n, k): (usize, usize, usize)) {
    let rng = &mut StdRng::seed_from_u64(0);
    let mut cs = Ord::min(m.next_power_of_two(), m.next_multiple_of(8));
    if m > 48 {
        cs = Ord::max(4096, cs);
    }

    let params = gemm_common::cache::kernel_params(m, n, k, 4 * 8, 6, 8);
    // dbg!(params);

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
            false,
            lhs.as_ptr(),
            cs as isize,
            1,
            rhs.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
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

    let mr = if m <= 1 {
        1usize
    } else if m <= 2 {
        2
    } else if m <= 4 {
        4
    } else if m <= 8 {
        8
    } else if m <= 24 {
        24
    } else {
        48
    };
    let nr = if m <= 24 { 8 } else { 4 };

    let mut cs = Ord::min(m.next_power_of_two(), m.next_multiple_of(mr));
    if m > 48 {
        cs = Ord::max(4096, cs);
    }

    let packed_lhs = &mut *avec![[4096]| 0.0; m.next_multiple_of(mr) * k];
    let packed_rhs = &mut *avec![[4096]| 0.0; n.next_multiple_of(8) * k];
    // dbg!(packed_lhs.as_ptr());

    let lhs = &mut *avec![[4096]| 0.0; cs * k];
    let rhs = &mut *avec![[4096]| 0.0; k * n];
    let dst = &mut *avec![[4096]| 0.0; cs * n];

    rng.fill(lhs);
    rng.fill(rhs);

    let f = Ord::min(8, k.div_ceil(64));

    let l1 = 64 / f;
    let l2 = 2048 / f;
    let l3 = 32768 / f;

    let tall = m >= l3;

    // dbg!(l2, l3);

    if m <= 48 && PACK_RHS {
        return bencher.skip();
    }
    let pack_lhs = (n > 6 * nr && tall) || (n > 6 * nr * rayon::current_num_threads());

    if pack_lhs != PACK_LHS {
        return bencher.skip();
    }

    let (row_chunk, col_chunk, rowmajor) = if false && tall {
        ([m, l3, l3 / 2, l1, mr], [n, l3, l3 / 2, l2, nr], true)
    } else {
        (
            [m, m, m, m, mr],
            [n, n, n, l3, nr],
            false,
            // [2 * l3, l3, l3 / 2, l2, mr],
            // [2 * l3, l3, l3 / 2, l1, nr],
            // false,
        )
    };
    let mut row_chunk = row_chunk.map(|r| r.next_multiple_of(mr));
    let mut col_chunk = col_chunk.map(|c| c.next_multiple_of(nr));

    let q = row_chunk.len();

    {
        for i in (1..q).rev() {
            row_chunk[i - 1] = Ord::max(row_chunk[i - 1], row_chunk[i]);
        }
        for i in (1..q).rev() {
            col_chunk[i - 1] = Ord::max(col_chunk[i - 1], col_chunk[i]);
        }
    }

    let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
    let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

    let mut packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
    let mut packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

    _ = &mut packed_lhs_rs;
    _ = &mut packed_rhs_cs;

    // for i in 0..q - rowmajor as usize {
    //     if col_chunk[i] >= n {
    //         packed_lhs_rs[i + rowmajor as usize] = 0;
    //     }
    // }
    // for i in 0..q - (!rowmajor) as usize {
    //     if row_chunk[i] >= m {
    //         packed_rhs_cs[i + (!rowmajor) as usize] = 0;
    //     }
    // }
    // if rowmajor {
    //     packed_lhs_rs[0] = 0;
    // } else {
    //     packed_rhs_cs[0] = 0;
    // }
    // dbg!(row_chunk, col_chunk);

    // packed_lhs_rs[2] = 0;
    // packed_lhs_rs[3] = 0;
    // packed_lhs_rs[4] = 0;
    // packed_rhs_cs[3] = 0;
    // packed_rhs_cs[4] = 0;
    // dbg!(packed_rhs_cs);
    // dbg!(packed_lhs_rs);

    {
        let lhs = Cell(lhs.as_ptr());
        let rhs = Cell(rhs.as_ptr());
        let packed_lhs = Cell(packed_lhs.as_mut_ptr());
        let packed_rhs = Cell(packed_rhs.as_mut_ptr());
        let bencher = Cell(bencher);
        spindle::with_lock(rayon::current_num_threads(), || {
            { bencher }.0.bench(|| unsafe {
                // A and B in this benchmark are column major
                // row stride is    sizeof(T) * row_distance
                // column stride is sizeof(T) * row_dim * col_distance

                kernel_rayon(
                    rayon::current_num_threads(),
                    if m <= 1 {
                        &F64_SIMD64
                    } else if m <= 2 {
                        &F64_SIMD128
                    } else if m <= 4 {
                        &F64_SIMD256[8..]
                    } else if m <= 24 {
                        &F64_SIMD512x8
                    } else {
                        &F64_SIMD512x4[..24]
                    },
                    &F64_SIMDpack_512,
                    mr,
                    nr,
                    size_of::<f64>(),
                    { lhs }.0 as _,
                    if PACK_LHS {
                        { packed_lhs }.0 as _
                    } else {
                        { lhs }.0 as _
                    },
                    { rhs }.0 as _,
                    if PACK_RHS {
                        { packed_rhs }.0 as _
                    } else {
                        { rhs }.0 as _
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
                    Position { row: 0, col: 0 },
                    &MicrokernelInfo {
                        flags: ((rowmajor as usize) << 63),
                        depth: k,
                        lhs_rs: sizeof,
                        lhs_cs: cs as isize * sizeof,
                        rhs_rs: sizeof,
                        rhs_cs: k as isize * sizeof,
                        alpha: &raw const *&1.0 as _,
                        ptr: dst.as_mut_ptr() as _,
                        rs: sizeof,
                        cs: cs as isize * sizeof,
                        row_idx: null_mut(),
                        col_idx: null_mut(),
                        diag_ptr: null(),
                        diag_stride: 0,
                    },
                )
            });
        });
    }
    if true {
        let target = &mut *avec![0.0f64; n * cs];

        unsafe {
            gemm(
                m,
                n,
                k,
                target.as_mut_ptr(),
                cs as isize,
                1,
                false,
                lhs.as_ptr(),
                cs as isize,
                1,
                rhs.as_ptr(),
                k as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        };
        let mut i = 0;
        for (target, dst) in std::iter::zip(&*target, &*dst) {
            if !((target - dst).abs() < 1e-6) {
                dbg!(i / cs, i % cs, target, dst);
                panic!();
            }
            i += 1;
        }
    }
}

fn bench_asm_rowmajor<const PACK_LHS: bool, const PACK_RHS: bool>(
    bencher: Bencher,
    (m, n, k): (usize, usize, usize),
) {
    let rng = &mut StdRng::seed_from_u64(0);
    let sizeof = size_of::<f64>() as isize;

    let mr = if m <= 1 {
        1usize
    } else if m <= 2 {
        2
    } else if m <= 4 {
        4
    } else if m <= 8 {
        8
    } else if m <= 24 {
        24
    } else {
        48
    };
    let nr = if m <= 24 { 8 } else { 4 };

    let mut cs = Ord::min(m.next_power_of_two(), m.next_multiple_of(mr));
    let rs = Ord::min(k.next_power_of_two(), k.next_multiple_of(mr));
    if m > 48 {
        cs = Ord::max(4096, cs);
    }

    let packed_lhs = &mut *avec![[4096]| 0.0; m.next_multiple_of(mr) * k];
    let packed_rhs = &mut *avec![[4096]| 0.0; n.next_multiple_of(8) * k];
    // dbg!(packed_lhs.as_ptr());

    let lhs = &mut *avec![[4096]| 0.0; m * rs];
    let rhs = &mut *avec![[4096]| 0.0; k * n];
    let dst = &mut *avec![[4096]| 0.0; cs * n];

    rng.fill(lhs);
    rng.fill(rhs);

    let f = Ord::min(8, k.div_ceil(64));

    let l1 = 64 / f;
    let l2 = 2048 / f;
    let l3 = 32768 / f;

    let tall = m >= l3;

    // dbg!(l2, l3);

    if m <= 48 && PACK_RHS {
        return bencher.skip();
    }
    let pack_lhs = (n > 6 * nr && tall) || (n > 6 * nr * rayon::current_num_threads());

    if pack_lhs != PACK_LHS {
        return bencher.skip();
    }

    let (row_chunk, col_chunk, rowmajor) = if false && tall {
        ([m, l3, l3 / 2, l1, mr], [n, l3, l3 / 2, l2, nr], true)
    } else {
        (
            [m, m, m, m, mr],
            [n, n, n, l3, nr],
            false,
            // [2 * l3, l3, l3 / 2, l2, mr],
            // [2 * l3, l3, l3 / 2, l1, nr],
            // false,
        )
    };
    let mut row_chunk = row_chunk.map(|r| r.next_multiple_of(mr));
    let mut col_chunk = col_chunk.map(|c| c.next_multiple_of(nr));

    let q = row_chunk.len();

    {
        for i in (1..q).rev() {
            row_chunk[i - 1] = Ord::max(row_chunk[i - 1], row_chunk[i]);
        }
        for i in (1..q).rev() {
            col_chunk[i - 1] = Ord::max(col_chunk[i - 1], col_chunk[i]);
        }
    }

    let lhs_rs = row_chunk.map(|m| (m * rs) as isize * sizeof);
    let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

    let mut packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
    let mut packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

    _ = &mut packed_lhs_rs;
    _ = &mut packed_rhs_cs;

    // for i in 0..q - rowmajor as usize {
    //     if col_chunk[i] >= n {
    //         packed_lhs_rs[i + rowmajor as usize] = 0;
    //     }
    // }
    // for i in 0..q - (!rowmajor) as usize {
    //     if row_chunk[i] >= m {
    //         packed_rhs_cs[i + (!rowmajor) as usize] = 0;
    //     }
    // }
    // if rowmajor {
    //     packed_lhs_rs[0] = 0;
    // } else {
    //     packed_rhs_cs[0] = 0;
    // }
    // dbg!(row_chunk, col_chunk);

    // packed_lhs_rs[2] = 0;
    // packed_lhs_rs[3] = 0;
    // packed_lhs_rs[4] = 0;
    // packed_rhs_cs[3] = 0;
    // packed_rhs_cs[4] = 0;
    // dbg!(packed_rhs_cs);
    // dbg!(packed_lhs_rs);

    {
        let lhs = Cell(lhs.as_ptr());
        let rhs = Cell(rhs.as_ptr());
        let packed_lhs = Cell(packed_lhs.as_mut_ptr());
        let packed_rhs = Cell(packed_rhs.as_mut_ptr());
        let bencher = Cell(bencher);
        spindle::with_lock(rayon::current_num_threads(), || {
            { bencher }.0.bench(|| unsafe {
                // A and B in this benchmark are column major
                // row stride is    sizeof(T) * row_distance
                // column stride is sizeof(T) * row_dim * col_distance

                kernel_rayon(
                    rayon::current_num_threads(),
                    if m <= 1 {
                        &F64_SIMD64
                    } else if m <= 2 {
                        &F64_SIMD128
                    } else if m <= 4 {
                        &F64_SIMD256[8..]
                    } else if m <= 24 {
                        &F64_SIMD512x8
                    } else {
                        &F64_SIMD512x4[..24]
                    },
                    &F64_SIMDpack_512,
                    mr,
                    nr,
                    size_of::<f64>(),
                    { lhs }.0 as _,
                    if PACK_LHS {
                        { packed_lhs }.0 as _
                    } else {
                        { lhs }.0 as _
                    },
                    { rhs }.0 as _,
                    if PACK_RHS {
                        { packed_rhs }.0 as _
                    } else {
                        { rhs }.0 as _
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
                    Position { row: 0, col: 0 },
                    &MicrokernelInfo {
                        flags: ((rowmajor as usize) << 63),
                        depth: k,
                        lhs_rs: rs as isize * sizeof,
                        lhs_cs: sizeof,
                        rhs_rs: sizeof,
                        rhs_cs: k as isize * sizeof,
                        alpha: &raw const *&1.0 as _,
                        ptr: dst.as_mut_ptr() as _,
                        rs: sizeof,
                        cs: cs as isize * sizeof,
                        row_idx: null_mut(),
                        col_idx: null_mut(),
                        diag_ptr: null(),
                        diag_stride: 0,
                    },
                )
            });
        });
    }
    if true {
        let target = &mut *avec![0.0f64; n * cs];

        unsafe {
            gemm(
                m,
                n,
                k,
                target.as_mut_ptr(),
                cs as isize,
                1,
                false,
                lhs.as_ptr(),
                1 as isize,
                rs as isize,
                rhs.as_ptr(),
                k as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        };
        let mut i = 0;
        for (target, dst) in std::iter::zip(&*target, &*dst) {
            if !((target - dst).abs() < 1e-6) {
                dbg!(i / cs, i % cs, target, dst);
                panic!();
            }
            i += 1;
        }
    }
}

fn main() -> eyre::Result<()> {
    let config = &mut Config::from_args()?;
    let plot_dir = &config.plot_dir.0.take();

    if true {
        let cache = try_cache_info_linux().unwrap();
        dbg!(cache[0].cache_bytes);
        dbg!(cache[1].cache_bytes);
        dbg!(cache[2].cache_bytes);

        let k = 64;
        dbg!(cache[0].cache_bytes / (k * size_of::<f64>()));
        dbg!(cache[1].cache_bytes / (k * size_of::<f64>()));
        dbg!(cache[2].cache_bytes / (k * size_of::<f64>()));
    }

    for k in [64, 128, 256, 512] {
        let mut args_small: [_; 16] = core::array::from_fn(|i| {
            let i = i as u32;
            if i % 2 == 0 {
                2usize.pow(1 + i / 2 as u32)
            } else {
                3 * 2usize.pow(i / 2 as u32)
            }
        });
        args_small.sort_unstable();
        let args_small = args_small.map(PlotArg);

        let mut args_big: [_; 11] = core::array::from_fn(|i| {
            let i = i as u32;
            if i % 2 == 0 {
                2usize.pow(9 + i / 2 as u32)
            } else {
                3 * 2usize.pow(8 + i / 2 as u32)
            }
        });
        args_big.sort_unstable();

        let args_big = args_big.map(PlotArg);

        let f = [
            bench_asm::<false, false>,
            bench_asm::<false, true>,
            bench_asm::<true, false>,
            bench_asm::<true, true>,
            bench_gemm,
            bench_asm_rowmajor::<false, false>,
            bench_asm_rowmajor::<false, true>,
            bench_asm_rowmajor::<true, false>,
            bench_asm_rowmajor::<true, true>,
        ];

        if true {
            config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| {
                (n * n * k) as f64 / time.to_secs()
            })
            .with_name("flops");
            let bench = Bench::new(&config);

            let f = f.map(move |f| {
                move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (n, n, k))
            });
            bench.register_many(
                &format!("k={k} m=n"),
                list![
                    f[0].with_name(&format!("asm")),
                    f[1].with_name(&format!("asm pack B")),
                    f[2].with_name(&format!("asm pack A")),
                    f[3].with_name(&format!("asm pack A pack B")),
                    f[5].with_name(&format!("rowmajor asm")),
                    f[6].with_name(&format!("rowmajor asm pack B")),
                    f[7].with_name(&format!("rowmajor asm pack A")),
                    f[8].with_name(&format!("rowmajor asm pack A pack B")),
                    f[4].with_name(&format!("gemm"))
                ],
                args_big,
            );
            let results = bench
                .run()?
                .combine(&serde_json::from_str(&std::fs::read_to_string(&format!(
                    "{}/openblas {}.json",
                    concat!(env!("CARGO_MANIFEST_DIR")),
                    bench.groups.borrow().keys().next().unwrap()
                ))?)?);
            std::fs::write(
                format!(
                    "{}/all {}.json",
                    concat!(env!("CARGO_MANIFEST_DIR")),
                    bench.groups.borrow().keys().next().unwrap()
                ),
                serde_json::to_string(&results)?,
            )?;

            if let Some(plot_dir) = plot_dir {
                results.plot(config, plot_dir)?;
            }
        }

        for PlotArg(m) in args_small {
            if false {
                config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| {
                    (n * m * k) as f64 / time.to_secs()
                })
                .with_name("flops");
                let bench = Bench::new(&config);
                let f = f.map(move |f| {
                    move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (n, m, k))
                });
                bench.register_many(
                    &format!("k={k} n={m}"),
                    list![
                        f[0].with_name(&format!("asm")),
                        f[1].with_name(&format!("asm pack B")),
                        f[2].with_name(&format!("asm packA")),
                        f[3].with_name(&format!("asm packA pack B")),
                        f[4].with_name(&format!("gemm")),
                    ],
                    args_big,
                );
                let results =
                    bench
                        .run()?
                        .combine(&serde_json::from_str(&std::fs::read_to_string(&format!(
                            "{}/openblas {}.json",
                            concat!(env!("CARGO_MANIFEST_DIR")),
                            bench.groups.borrow().keys().next().unwrap()
                        ))?)?);
                std::fs::write(
                    format!(
                        "{}/all {}.json",
                        concat!(env!("CARGO_MANIFEST_DIR")),
                        bench.groups.borrow().keys().next().unwrap()
                    ),
                    serde_json::to_string(&results)?,
                )?;
                if let Some(plot_dir) = plot_dir {
                    results.plot(config, plot_dir)?;
                }
            }

            if false {
                config.plot_metric = PlotMetric::new(move |PlotArg(n), time: Picoseconds| {
                    (n * m * k) as f64 / time.to_secs()
                })
                .with_name("flops");
                let bench = Bench::new(&config);
                let f = f.map(move |f| {
                    move |bencher: Bencher<'_>, PlotArg(n): PlotArg| f(bencher, (m, n, k))
                });
                bench.register_many(
                    &format!("k={k} m={m}"),
                    list![
                        f[0].with_name(&format!("asm")),
                        f[1].with_name(&format!("asm pack B")),
                        f[2].with_name(&format!("asm packA")),
                        f[3].with_name(&format!("asm packA pack B")),
                        f[4].with_name(&format!("gemm")),
                    ],
                    args_big,
                );
                let results =
                    bench
                        .run()?
                        .combine(&serde_json::from_str(&std::fs::read_to_string(&format!(
                            "{}/openblas {}.json",
                            concat!(env!("CARGO_MANIFEST_DIR")),
                            bench.groups.borrow().keys().next().unwrap()
                        ))?)?);
                std::fs::write(
                    format!(
                        "{}/all {}.json",
                        concat!(env!("CARGO_MANIFEST_DIR")),
                        bench.groups.borrow().keys().next().unwrap()
                    ),
                    serde_json::to_string(&results)?,
                )?;
                if let Some(plot_dir) = plot_dir {
                    results.plot(config, plot_dir)?;
                }
            }
        }
    }

    Ok(())
}

fn try_cache_info_linux() -> Result<[CacheInfo; 3], std::io::Error> {
    let mut all_info = [CacheInfo {
        associativity: 8,
        cache_bytes: 0,
        cache_line_bytes: 64,
    }; 3];

    let mut l1_shared_count = 1;
    for cpu_x in fs::read_dir("/sys/devices/system/cpu")? {
        let cpu_x = cpu_x?.path();
        let Some(cpu_x_name) = cpu_x.file_name().and_then(|f| f.to_str()) else {
            continue;
        };
        if !cpu_x_name.starts_with("cpu") {
            continue;
        }
        let cache = cpu_x.join("cache");
        if !cache.is_dir() {
            continue;
        }
        'index: for index_y in fs::read_dir(cache)? {
            let index_y = index_y?.path();
            if !index_y.is_dir() {
                continue;
            }
            let Some(index_y_name) = index_y.file_name().and_then(|f| f.to_str()) else {
                continue;
            };
            if !index_y_name.starts_with("index") {
                continue;
            }

            let mut cache_info = CacheInfo {
                associativity: 8,
                cache_bytes: 0,
                cache_line_bytes: 64,
            };
            let mut level: usize = 0;
            let mut shared_count: usize = 0;

            for entry in fs::read_dir(index_y)? {
                let entry = entry?.path();
                if let Some(name) = entry.file_name() {
                    let contents = fs::read_to_string(&entry)?;
                    let contents = contents.trim();
                    if name == "type" && !matches!(contents, "Data" | "Unified") {
                        continue 'index;
                    }
                    if name == "shared_cpu_list" {
                        for item in contents.split(',') {
                            if item.contains('-') {
                                let mut item = item.split('-');
                                let Some(start) = item.next() else {
                                    continue 'index;
                                };
                                let Some(end) = item.next() else {
                                    continue 'index;
                                };

                                let Ok(start) = start.parse::<usize>() else {
                                    continue 'index;
                                };
                                let Ok(end) = end.parse::<usize>() else {
                                    continue 'index;
                                };

                                shared_count += end + 1 - start;
                            } else {
                                shared_count += 1;
                            }
                        }
                    }

                    if name == "level" {
                        let Ok(contents) = contents.parse::<usize>() else {
                            continue 'index;
                        };
                        level = contents;
                    }

                    if name == "coherency_line_size" {
                        let Ok(contents) = contents.parse::<usize>() else {
                            continue 'index;
                        };
                        cache_info.cache_line_bytes = contents;
                    }
                    if name == "ways_of_associativity" {
                        let Ok(contents) = contents.parse::<usize>() else {
                            continue 'index;
                        };
                        cache_info.associativity = contents;
                    }
                    if name == "size" {
                        if contents.ends_with("G") {
                            let Ok(contents) = contents.trim_end_matches('G').parse::<usize>()
                            else {
                                continue 'index;
                            };
                            cache_info.cache_bytes = contents * 1024 * 1024 * 1024;
                        } else if contents.ends_with("M") {
                            let Ok(contents) = contents.trim_end_matches('M').parse::<usize>()
                            else {
                                continue 'index;
                            };
                            cache_info.cache_bytes = contents * 1024 * 1024;
                        } else if contents.ends_with("K") {
                            let Ok(contents) = contents.trim_end_matches('K').parse::<usize>()
                            else {
                                continue 'index;
                            };
                            cache_info.cache_bytes = contents * 1024;
                        } else {
                            let Ok(contents) = contents.parse::<usize>() else {
                                continue 'index;
                            };
                            cache_info.cache_bytes = contents;
                        }
                    }
                }
            }
            if level == 1 {
                l1_shared_count = shared_count;
            }
            if level > 0 {
                if cache_info.cache_line_bytes >= all_info[level - 1].cache_line_bytes {
                    all_info[level - 1].associativity = cache_info.associativity;
                    all_info[level - 1].cache_line_bytes = cache_info.cache_line_bytes;
                    all_info[level - 1].cache_bytes = cache_info.cache_bytes / shared_count;
                }
            }
        }
    }
    for info in &mut all_info {
        info.cache_bytes *= l1_shared_count;
    }

    Ok(all_info)
}
