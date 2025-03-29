#![cfg_attr(all(not(test), any()), no_std)]
#![allow(non_upper_case_globals)]

use core::ptr::{null, null_mut};

include!(concat!(env!("OUT_DIR"), "/asm.rs"));

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MicrokernelInfo {
    pub flags: usize,
    pub depth: usize,
    pub lhs_rs: isize,
    pub lhs_cs: isize,
    pub rhs_rs: isize,
    pub rhs_cs: isize,
    pub row: usize,
    pub col: usize,
    pub alpha: *const (),
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MillikernelInfo {
    pub lhs_rs: isize,
    pub packed_lhs_rs: isize,
    pub rhs_cs: isize,
    pub packed_rhs_cs: isize,
    pub micro: MicrokernelInfo,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Dst {
    pub ptr: *mut (),
    pub rs: isize,
    pub cs: isize,
    pub row_idx: *const (),
    pub col_idx: *const (),
}

#[inline(always)]
pub unsafe fn call_microkernel(
    microkernel: unsafe extern "C" fn(),
    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    mut nrows: usize,
    mut ncols: usize,

    micro: &mut MicrokernelInfo,

    dst: &Dst,
) -> (usize, usize) {
    unsafe {
        core::arch::asm! {
            "call r10",

            in("rax") lhs,
            in("r15") packed_lhs,
            in("rcx") rhs,
            in("rdx") packed_rhs,
            in("rdi") dst,
            in("rsi") micro,
            inout("r8") nrows,
            inout("r9") ncols,
            in("r10") microkernel,

            out("zmm0") _,
            out("zmm1") _,
            out("zmm2") _,
            out("zmm3") _,
            out("zmm4") _,
            out("zmm5") _,
            out("zmm6") _,
            out("zmm7") _,
            out("zmm8") _,
            out("zmm9") _,
            out("zmm10") _,
            out("zmm11") _,
            out("zmm12") _,
            out("zmm13") _,
            out("zmm14") _,
            out("zmm15") _,
            out("zmm16") _,
            out("zmm17") _,
            out("zmm18") _,
            out("zmm19") _,
            out("zmm20") _,
            out("zmm21") _,
            out("zmm22") _,
            out("zmm23") _,
            out("zmm24") _,
            out("zmm25") _,
            out("zmm26") _,
            out("zmm27") _,
            out("zmm28") _,
            out("zmm29") _,
            out("zmm30") _,
            out("zmm31") _,
            out("k1") _,
        }
    }
    (nrows, ncols)
}

pub unsafe fn millikernel_rowmajor(
    microkernel: unsafe extern "C" fn(),

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    milli: &mut MillikernelInfo,

    dst: &Dst,
) {
    let mut rhs = rhs;
    let mut nrows = nrows;
    let mut lhs = lhs;
    let mut packed_lhs = packed_lhs;

    let cs0 = milli.rhs_cs;
    let cs1 = milli.micro.rhs_cs;
    loop {
        let rs = milli.micro.lhs_rs;
        unsafe {
            let mut rhs = rhs;
            let mut packed_rhs = packed_rhs;
            let mut ncols = ncols;
            let mut lhs = lhs;
            let col = milli.micro.col;

            macro_rules! iter {
                ($($lhs: ident)?) => {{
                    (nrows, ncols) = call_microkernel(
                        microkernel,
                        lhs,
                        packed_lhs,
                        rhs,
                        packed_rhs,
                        nrows,
                        ncols,
                        &mut milli.micro,
                        dst,
                    );

                    rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
                    packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);

                    $(if $lhs != packed_lhs {
                        milli.micro.lhs_rs = 0;
                        lhs = packed_lhs;
                    })?
                }};
            }
            iter!(lhs);
            while ncols > 0 {
                iter!();
            }
            milli.micro.col = col;
        }
        milli.micro.lhs_rs = rs;

        lhs = lhs.wrapping_byte_offset(milli.lhs_rs);
        packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs);
        if rhs != packed_rhs {
            rhs = packed_rhs;
            milli.micro.rhs_cs = 0;
            milli.rhs_cs = milli.packed_rhs_cs;
        }

        if nrows == 0 {
            break;
        }
    }
    milli.micro.rhs_cs = cs1;
    milli.rhs_cs = cs0;
}

pub unsafe fn millikernel_colmajor(
    microkernel: unsafe extern "C" fn(),

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    milli: &mut MillikernelInfo,

    dst: &Dst,
) {
    let mut lhs = lhs;
    let mut ncols = ncols;
    let mut rhs = rhs;
    let mut packed_rhs = packed_rhs;

    let rs0 = milli.lhs_rs;
    let rs1 = milli.micro.lhs_rs;
    loop {
        let cs = milli.micro.rhs_cs;
        unsafe {
            let mut lhs = lhs;
            let mut packed_lhs = packed_lhs;
            let mut nrows = nrows;
            let mut rhs = rhs;
            let row = milli.micro.row;

            macro_rules! iter {
                ($($rhs: ident)?) => {{
                    (nrows, ncols) = call_microkernel(
                        microkernel,
                        lhs,
                        packed_lhs,
                        rhs,
                        packed_rhs,
                        nrows,
                        ncols,
                        &mut milli.micro,
                        dst,
                    );

                    lhs = lhs.wrapping_byte_offset(milli.lhs_rs);
                    packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs);

                    $(if $rhs != packed_rhs {
                        milli.micro.rhs_cs = 0;
                        rhs = packed_rhs;
                    })?
                }};
            }
            iter!(rhs);
            while nrows > 0 {
                iter!();
            }
            milli.micro.row = row;
        }
        milli.micro.rhs_cs = cs;

        rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
        packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);
        if lhs != packed_lhs {
            lhs = packed_lhs;
            milli.micro.lhs_rs = 0;
            milli.lhs_rs = milli.packed_lhs_rs;
        }

        if ncols == 0 {
            break;
        }
    }
    milli.micro.lhs_rs = rs1;
    milli.lhs_rs = rs0;
}

#[inline(never)]
pub unsafe fn kernel<'a>(
    microkernel: &'a [unsafe extern "C" fn()],
    nr: usize,

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    row_chunk: &'a [usize],
    col_chunk: &'a [usize],
    lhs_rs: &'a [isize],
    rhs_cs: &'a [isize],
    packed_lhs_rs: &'a [isize],
    packed_rhs_cs: &'a [isize],

    row: usize,
    col: usize,

    dst: &'a Dst,
    info: &'a mut MicrokernelInfo,
) {
    let mut stack: [(
        *const (),
        *mut (),
        *const (),
        *mut (),
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        bool,
        bool,
    ); 16] = const {
        [(
            null(),
            null_mut(),
            null(),
            null_mut(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            false,
            false,
        ); 16]
    };

    stack[0] = (
        lhs, packed_lhs, rhs, packed_rhs, row, col, nrows, ncols, 0, 0, 0, 0, false, false,
    );

    let mut depth = 0;
    let max_depth = row_chunk.len();

    let milli_rs = *lhs_rs.last().unwrap();
    let milli_cs = *rhs_cs.last().unwrap();

    let micro_rs = info.lhs_rs;
    let micro_cs = info.rhs_cs;

    let mut milli = MillikernelInfo {
        lhs_rs: milli_rs,
        packed_lhs_rs: *packed_lhs_rs.last().unwrap(),
        rhs_cs: milli_cs,
        packed_rhs_cs: *packed_rhs_cs.last().unwrap(),
        micro: *info,
    };
    let millikernel = if milli.micro.flags >> 63 == 0 {
        millikernel_colmajor
    } else {
        millikernel_rowmajor
    };
    let microkernel = microkernel[nr - 1];

    let q = row_chunk.len();
    let row_chunk = &row_chunk[..q - 1];
    let col_chunk = &col_chunk[..q - 1];
    let lhs_rs = &lhs_rs[..q];
    let packed_lhs_rs = &packed_lhs_rs[..q];
    let rhs_cs = &rhs_cs[..q];
    let packed_rhs_cs = &packed_rhs_cs[..q];

    loop {
        let (
            lhs,
            packed_lhs,
            rhs,
            packed_rhs,
            row,
            col,
            nrows,
            ncols,
            i,
            j,
            ii,
            jj,
            is_packed_lhs,
            is_packed_rhs,
        ) = stack[depth];

        if depth + 1 == max_depth {
            let mut lhs = lhs;
            let mut rhs = rhs;

            milli.micro.row = row;
            milli.micro.col = col;

            if is_packed_lhs && lhs != packed_lhs {
                lhs = packed_lhs;
                milli.micro.lhs_rs = 0;
                milli.lhs_rs = milli.packed_lhs_rs;
            }
            if is_packed_rhs && rhs != packed_rhs {
                rhs = packed_rhs;
                milli.micro.rhs_cs = 0;
                milli.rhs_cs = milli.packed_rhs_cs;
            }

            unsafe {
                millikernel(
                    microkernel,
                    lhs,
                    packed_lhs,
                    rhs,
                    packed_rhs,
                    nrows,
                    ncols,
                    &mut milli,
                    dst,
                );
            }
            milli.lhs_rs = milli_rs;
            milli.rhs_cs = milli_cs;
            milli.micro.lhs_rs = micro_rs;
            milli.micro.rhs_cs = micro_cs;

            while depth > 0 {
                depth -= 1;

                let (_, _, _, _, _, _, nrows, ncols, i, j, ii, jj, _, _) = &mut stack[depth];

                let col_chunk = col_chunk[depth];
                let row_chunk = row_chunk[depth];

                let j_chunk = Ord::min(col_chunk, *ncols - *j);
                let i_chunk = Ord::min(row_chunk, *nrows - *i);

                if milli.micro.flags >> 63 == 0 {
                    *i += i_chunk;
                    *ii += 1;
                    if *i == *nrows {
                        *i = 0;
                        *ii = 0;
                        *j += j_chunk;
                        *jj += 1;

                        if *j == *ncols {
                            if depth == 0 {
                                return;
                            }

                            *j = 0;
                            *jj = 0;
                            continue;
                        }
                    }
                } else {
                    *j += j_chunk;
                    *jj += 1;
                    if *j == *ncols {
                        *j = 0;
                        *jj = 0;
                        *i += i_chunk;
                        *ii += 1;

                        if *i == *nrows {
                            *i = 0;
                            *ii = 0;
                            if depth == 0 {
                                return;
                            }
                            continue;
                        }
                    }
                }
                break;
            }
        } else {
            let col_chunk = col_chunk[depth];
            let row_chunk = row_chunk[depth];
            let rhs_cs = rhs_cs[depth];
            let lhs_rs = lhs_rs[depth];
            let prhs_cs = packed_rhs_cs[depth];
            let plhs_rs = packed_lhs_rs[depth];

            let j_chunk = Ord::min(col_chunk, ncols - j);
            let i_chunk = Ord::min(row_chunk, nrows - i);

            depth += 1;
            stack[depth] = (
                lhs.wrapping_byte_offset(lhs_rs * ii as isize),
                packed_lhs.wrapping_byte_offset(plhs_rs * ii as isize),
                rhs.wrapping_byte_offset(rhs_cs * jj as isize),
                packed_rhs.wrapping_byte_offset(prhs_cs * jj as isize),
                row + i,
                col + j,
                i_chunk,
                j_chunk,
                0,
                0,
                0,
                0,
                is_packed_lhs || (j > 0 && packed_lhs_rs[depth] != 0),
                is_packed_rhs || (i > 0 && packed_rhs_cs[depth] != 0),
            );
            continue;
        }
    }
}

#[cfg(test)]
mod tests_f64 {
    use core::ptr::null_mut;

    use super::*;

    use aligned_vec::*;
    use rand::prelude::*;

    #[test]
    fn test_avx512_microkernel() {
        let rng = &mut StdRng::seed_from_u64(0);

        let sizeof = size_of::<f64>() as isize;
        let len = 64 / size_of::<f64>();

        for pack_lhs in [false, true] {
            for pack_rhs in [false, true] {
                for alpha in [1.0.into(), 0.0.into(), 2.5.into()] {
                    let alpha: f64 = alpha;
                    for m in 1..=48usize {
                        for n in (1..=4usize).into_iter().chain([8]) {
                            for cs in [m.next_multiple_of(len), m] {
                                let acs = m.next_multiple_of(len);
                                let k = 1usize;

                                let packed_lhs: &mut [f64] = &mut *avec![0.0.into(); acs * k];
                                let packed_rhs: &mut [f64] =
                                    &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
                                let lhs: &mut [f64] = &mut *avec![0.0.into(); cs * k];
                                let rhs: &mut [f64] = &mut *avec![0.0.into(); n * k];
                                let dst: &mut [f64] = &mut *avec![0.0.into(); cs * n];
                                let target = &mut *avec![0.0.into(); cs * n];

                                rng.fill(lhs);
                                rng.fill(rhs);

                                for i in 0..m {
                                    for j in 0..n {
                                        let target = &mut target[i + cs * j];
                                        let mut acc = 0.0.into();
                                        for depth in 0..k {
                                            acc = f64::mul_add(
                                                lhs[i + cs * depth],
                                                rhs[depth + k * j],
                                                acc,
                                            );
                                        }
                                        *target = f64::mul_add(acc, alpha, *target);
                                    }
                                }

                                unsafe {
                                    millikernel_colmajor(
                                        F64_SIMD512x4[3],
                                        lhs.as_ptr() as _,
                                        if pack_lhs {
                                            packed_lhs.as_mut_ptr() as _
                                        } else {
                                            lhs.as_ptr() as _
                                        },
                                        rhs.as_ptr() as _,
                                        if pack_rhs {
                                            packed_rhs.as_mut_ptr() as _
                                        } else {
                                            rhs.as_ptr() as _
                                        },
                                        m,
                                        n,
                                        &mut MillikernelInfo {
                                            lhs_rs: 48 * sizeof,
                                            packed_lhs_rs: 48 * sizeof * k as isize,
                                            rhs_cs: 4 * sizeof * k as isize,
                                            packed_rhs_cs: 4 * sizeof * k as isize,
                                            micro: MicrokernelInfo {
                                                flags: 0,
                                                depth: k,
                                                lhs_rs: 1 * sizeof,
                                                lhs_cs: cs as isize * sizeof,
                                                rhs_rs: 1 * sizeof,
                                                rhs_cs: k as isize * sizeof,
                                                row: 0,
                                                col: 0,
                                                alpha: &raw const alpha as _,
                                            },
                                        },
                                        &Dst {
                                            ptr: dst.as_mut_ptr() as _,
                                            rs: 1 * sizeof,
                                            cs: cs as isize * sizeof,
                                            row_idx: null_mut(),
                                            col_idx: null_mut(),
                                        },
                                    )
                                };
                                assert_eq!(dst, target);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx512_kernel() {
        let m = 1023usize;
        let n = 1023usize;
        let k = 5usize;

        let rng = &mut StdRng::seed_from_u64(0);
        let sizeof = size_of::<f64>() as isize;
        let cs = m.next_multiple_of(8);
        let cs = Ord::max(4096, cs);

        let lhs: &mut [f64] = &mut *avec![0.0; cs * k];
        let rhs: &mut [f64] = &mut *avec![0.0; k * n];
        let target: &mut [f64] = &mut *avec![0.0; cs * n];

        rng.fill(lhs);
        rng.fill(rhs);

        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                target.as_mut_ptr(),
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
        }

        for pack_lhs in [false, true] {
            for pack_rhs in [false, true] {
                let dst = &mut *avec![0.0; cs * n];
                let packed_lhs = &mut *avec![0.0f64; m.next_multiple_of(8) * k];
                let packed_rhs =
                    &mut *avec![0.0; if pack_rhs { n.next_multiple_of(4) * k } else { 0 }];

                unsafe {
                    let row_chunk = [48 * 32, 48 * 16, 48];
                    let col_chunk = [48 * 64, 48 * 32, 48, 4];

                    let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
                    let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
                    let packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
                    let packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

                    kernel(
                        &F64_SIMD512x4[..24],
                        4,
                        lhs.as_ptr() as _,
                        if pack_lhs {
                            packed_lhs.as_mut_ptr() as _
                        } else {
                            lhs.as_ptr() as _
                        },
                        rhs.as_ptr() as _,
                        if pack_rhs {
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
                        &if pack_lhs { packed_lhs_rs } else { lhs_rs },
                        &if pack_rhs { packed_rhs_cs } else { rhs_cs },
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
                            row: 0,
                            col: 0,
                            alpha: &raw const *&1.0f64 as _,
                        },
                    )
                }
                let mut i = 0;
                for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                    if !((target - dst).abs() < 1e-6) {
                        dbg!(i / cs, i % cs, target, dst);
                        panic!();
                    }
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_c64 {
    use super::*;

    use aligned_vec::*;
    use bytemuck::*;
    use core::ptr::null_mut;
    use gemm::c64;
    use rand::prelude::*;

    #[test]
    fn test_avx512_microkernel() {
        let rng = &mut StdRng::seed_from_u64(0);

        let sizeof = size_of::<c64>() as isize;
        let len = 64 / size_of::<c64>();

        for pack_lhs in [false, true] {
            for pack_rhs in [false, true] {
                for alpha in [
                    1.0.into(),
                    0.0.into(),
                    c64::new(0.0, 3.5),
                    c64::new(2.5, 3.5),
                ] {
                    let alpha: c64 = alpha;
                    for m in 1..=24usize {
                        for n in (1..=4usize).into_iter().chain([8]) {
                            for cs in [m.next_multiple_of(len), m] {
                                for conj_lhs in [false, true] {
                                    for conj_rhs in [false, true] {
                                        let conj_different = conj_lhs != conj_rhs;

                                        let acs = m.next_multiple_of(len);
                                        let k = 1usize;

                                        let packed_lhs: &mut [c64] =
                                            &mut *avec![0.0.into(); acs * k];
                                        let packed_rhs: &mut [c64] =
                                            &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
                                        let lhs: &mut [c64] = &mut *avec![0.0.into(); cs * k];
                                        let rhs: &mut [c64] = &mut *avec![0.0.into(); n * k];
                                        let dst: &mut [c64] = &mut *avec![0.0.into(); cs * n];
                                        let target: &mut [c64] = &mut *avec![0.0.into(); cs * n];

                                        rng.fill(cast_slice_mut::<c64, f64>(lhs));
                                        rng.fill(cast_slice_mut::<c64, f64>(rhs));

                                        for i in 0..m {
                                            for j in 0..n {
                                                let target = &mut target[i + cs * j];
                                                let mut acc: c64 = 0.0.into();
                                                for depth in 0..k {
                                                    let mut l = lhs[i + cs * depth];
                                                    let mut r = rhs[depth + k * j];
                                                    if conj_lhs {
                                                        l = l.conj();
                                                    }
                                                    if conj_rhs {
                                                        r = r.conj();
                                                    }

                                                    acc = l * r + acc;
                                                }
                                                *target = acc * alpha + *target;
                                            }
                                        }

                                        unsafe {
                                            millikernel_colmajor(
                                                C64_SIMD512x4[3],
                                                lhs.as_ptr() as _,
                                                if pack_lhs {
                                                    packed_lhs.as_mut_ptr() as _
                                                } else {
                                                    lhs.as_ptr() as _
                                                },
                                                rhs.as_ptr() as _,
                                                if pack_rhs {
                                                    packed_rhs.as_mut_ptr() as _
                                                } else {
                                                    rhs.as_ptr() as _
                                                },
                                                m,
                                                n,
                                                &mut MillikernelInfo {
                                                    lhs_rs: 24 * sizeof,
                                                    packed_lhs_rs: 24 * sizeof * k as isize,
                                                    rhs_cs: 4 * sizeof * k as isize,
                                                    packed_rhs_cs: 4 * sizeof * k as isize,
                                                    micro: MicrokernelInfo {
                                                        flags: ((conj_lhs as usize) << 1)
                                                            | ((conj_different as usize) << 2),
                                                        depth: k,
                                                        lhs_rs: 1 * sizeof,
                                                        lhs_cs: cs as isize * sizeof,
                                                        rhs_rs: 1 * sizeof,
                                                        rhs_cs: k as isize * sizeof,
                                                        row: 0,
                                                        col: 0,
                                                        alpha: &raw const alpha as _,
                                                    },
                                                },
                                                &Dst {
                                                    ptr: dst.as_mut_ptr() as _,
                                                    rs: 1 * sizeof,
                                                    cs: cs as isize * sizeof,
                                                    row_idx: null_mut(),
                                                    col_idx: null_mut(),
                                                },
                                            )
                                        };
                                        let mut i = 0;
                                        for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                                            if !((target - dst).norm_sqr().sqrt() < 1e-6) {
                                                dbg!(i / cs, i % cs, target, dst);
                                                panic!();
                                            }
                                            i += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_f32 {
    use core::ptr::null_mut;

    use super::*;

    use aligned_vec::*;
    use rand::prelude::*;

    #[test]
    fn test_avx512_microkernel() {
        let rng = &mut StdRng::seed_from_u64(0);

        let sizeof = size_of::<f32>() as isize;
        let len = 64 / size_of::<f32>();

        for pack_lhs in [false, true] {
            for pack_rhs in [false, true] {
                for alpha in [1.0.into(), 0.0.into(), 2.5.into()] {
                    let alpha: f32 = alpha;
                    for m in 1..=96usize {
                        for n in (1..=4usize).into_iter().chain([8]) {
                            for cs in [m.next_multiple_of(len), m] {
                                let acs = m.next_multiple_of(len);
                                let k = 1usize;

                                let packed_lhs: &mut [f32] = &mut *avec![0.0.into(); acs * k];
                                let packed_rhs: &mut [f32] =
                                    &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
                                let lhs: &mut [f32] = &mut *avec![0.0.into(); cs * k];
                                let rhs: &mut [f32] = &mut *avec![0.0.into(); n * k];
                                let dst: &mut [f32] = &mut *avec![0.0.into(); cs * n];
                                let target = &mut *avec![0.0.into(); cs * n];

                                rng.fill(lhs);
                                rng.fill(rhs);

                                for i in 0..m {
                                    for j in 0..n {
                                        let target = &mut target[i + cs * j];
                                        let mut acc = 0.0.into();
                                        for depth in 0..k {
                                            acc = f32::mul_add(
                                                lhs[i + cs * depth],
                                                rhs[depth + k * j],
                                                acc,
                                            );
                                        }
                                        *target = f32::mul_add(acc, alpha, *target);
                                    }
                                }

                                unsafe {
                                    millikernel_rowmajor(
                                        F32_SIMD512x4[3],
                                        lhs.as_ptr() as _,
                                        if pack_lhs {
                                            packed_lhs.as_mut_ptr() as _
                                        } else {
                                            lhs.as_ptr() as _
                                        },
                                        rhs.as_ptr() as _,
                                        if pack_rhs {
                                            packed_rhs.as_mut_ptr() as _
                                        } else {
                                            rhs.as_ptr() as _
                                        },
                                        m,
                                        n,
                                        &mut MillikernelInfo {
                                            lhs_rs: 96 * sizeof,
                                            packed_lhs_rs: 96 * sizeof * k as isize,
                                            rhs_cs: 4 * sizeof * k as isize,
                                            packed_rhs_cs: 4 * sizeof * k as isize,
                                            micro: MicrokernelInfo {
                                                flags: (1 << 63),
                                                depth: k,
                                                lhs_rs: 1 * sizeof,
                                                lhs_cs: cs as isize * sizeof,
                                                rhs_rs: 1 * sizeof,
                                                rhs_cs: k as isize * sizeof,
                                                row: 0,
                                                col: 0,
                                                alpha: &raw const alpha as _,
                                            },
                                        },
                                        &Dst {
                                            ptr: dst.as_mut_ptr() as _,
                                            rs: 1 * sizeof,
                                            cs: cs as isize * sizeof,
                                            row_idx: null_mut(),
                                            col_idx: null_mut(),
                                        },
                                    )
                                };
                                assert_eq!(dst, target);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_avx512_kernel() {
        let m = 6000usize;
        let n = 2000usize;
        let k = 5usize;

        let rng = &mut StdRng::seed_from_u64(0);
        let sizeof = size_of::<f32>() as isize;
        let cs = m.next_multiple_of(16);
        let cs = Ord::max(4096, cs);

        let lhs: &mut [f32] = &mut *avec![0.0; cs * k];
        let rhs: &mut [f32] = &mut *avec![0.0; k * n];
        let target: &mut [f32] = &mut *avec![0.0; cs * n];

        rng.fill(lhs);
        rng.fill(rhs);

        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                target.as_mut_ptr(),
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
        }

        for pack_lhs in [false, true] {
            for pack_rhs in [false, true] {
                let dst = &mut *avec![0.0; cs * n];
                let packed_lhs = &mut *avec![0.0f32; m.next_multiple_of(16) * k];
                let packed_rhs =
                    &mut *avec![0.0; if pack_rhs { n.next_multiple_of(4) * k } else { 0 }];

                unsafe {
                    let row_chunk = [96 * 32, 96 * 16, 96 * 4, 96];
                    let col_chunk = [1024, 256, 64, 16, 4];

                    let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
                    let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
                    let packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
                    let packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

                    kernel(
                        &F32_SIMD512x4[..24],
                        4,
                        lhs.as_ptr() as _,
                        if pack_lhs {
                            packed_lhs.as_mut_ptr() as _
                        } else {
                            lhs.as_ptr() as _
                        },
                        rhs.as_ptr() as _,
                        if pack_rhs {
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
                        &if pack_lhs { packed_lhs_rs } else { lhs_rs },
                        &if pack_rhs { packed_rhs_cs } else { rhs_cs },
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
                            row: 0,
                            col: 0,
                            alpha: &raw const *&1.0f32 as _,
                        },
                    )
                }
                let mut i = 0;
                for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                    if !((target - dst).abs() < 1e-6) {
                        dbg!(i / cs, i % cs, target, dst);
                        panic!();
                    }
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_c32 {
    use super::*;

    use aligned_vec::*;
    use bytemuck::*;
    use core::ptr::null_mut;
    use gemm::c32;
    use rand::prelude::*;

    #[test]
    fn test_avx512_microkernel() {
        let rng = &mut StdRng::seed_from_u64(0);

        let sizeof = size_of::<c32>() as isize;
        let len = 64 / size_of::<c32>();

        for pack_lhs in [false, true] {
            for pack_rhs in [false, true] {
                for alpha in [
                    1.0.into(),
                    0.0.into(),
                    c32::new(0.0, 3.5),
                    c32::new(2.5, 3.5),
                ] {
                    let alpha: c32 = alpha;
                    for m in 1..=48usize {
                        for n in (1..=4usize).into_iter().chain([8]) {
                            for cs in [m.next_multiple_of(len), m] {
                                for conj_lhs in [false, true] {
                                    for conj_rhs in [false, true] {
                                        let conj_different = conj_lhs != conj_rhs;

                                        let acs = m.next_multiple_of(len);
                                        let k = 1usize;

                                        let packed_lhs: &mut [c32] =
                                            &mut *avec![0.0.into(); acs * k];
                                        let packed_rhs: &mut [c32] =
                                            &mut *avec![0.0.into(); n.next_multiple_of(4) * k];
                                        let lhs: &mut [c32] = &mut *avec![0.0.into(); cs * k];
                                        let rhs: &mut [c32] = &mut *avec![0.0.into(); n * k];
                                        let dst: &mut [c32] = &mut *avec![0.0.into(); cs * n];
                                        let target: &mut [c32] = &mut *avec![0.0.into(); cs * n];

                                        rng.fill(cast_slice_mut::<c32, f32>(lhs));
                                        rng.fill(cast_slice_mut::<c32, f32>(rhs));

                                        for i in 0..m {
                                            for j in 0..n {
                                                let target = &mut target[i + cs * j];
                                                let mut acc: c32 = 0.0.into();
                                                for depth in 0..k {
                                                    let mut l = lhs[i + cs * depth];
                                                    let mut r = rhs[depth + k * j];
                                                    if conj_lhs {
                                                        l = l.conj();
                                                    }
                                                    if conj_rhs {
                                                        r = r.conj();
                                                    }

                                                    acc = l * r + acc;
                                                }
                                                *target = acc * alpha + *target;
                                            }
                                        }

                                        unsafe {
                                            millikernel_colmajor(
                                                C32_SIMD512x4[3],
                                                lhs.as_ptr() as _,
                                                if pack_lhs {
                                                    packed_lhs.as_mut_ptr() as _
                                                } else {
                                                    lhs.as_ptr() as _
                                                },
                                                rhs.as_ptr() as _,
                                                if pack_rhs {
                                                    packed_rhs.as_mut_ptr() as _
                                                } else {
                                                    rhs.as_ptr() as _
                                                },
                                                m,
                                                n,
                                                &mut MillikernelInfo {
                                                    lhs_rs: 48 * sizeof,
                                                    packed_lhs_rs: 48 * sizeof * k as isize,
                                                    rhs_cs: 4 * sizeof * k as isize,
                                                    packed_rhs_cs: 4 * sizeof * k as isize,
                                                    micro: MicrokernelInfo {
                                                        flags: ((conj_lhs as usize) << 1)
                                                            | ((conj_different as usize) << 2),
                                                        depth: k,
                                                        lhs_rs: 1 * sizeof,
                                                        lhs_cs: cs as isize * sizeof,
                                                        rhs_rs: 1 * sizeof,
                                                        rhs_cs: k as isize * sizeof,
                                                        row: 0,
                                                        col: 0,
                                                        alpha: &raw const alpha as _,
                                                    },
                                                },
                                                &Dst {
                                                    ptr: dst.as_mut_ptr() as _,
                                                    rs: 1 * sizeof,
                                                    cs: cs as isize * sizeof,
                                                    row_idx: null_mut(),
                                                    col_idx: null_mut(),
                                                },
                                            )
                                        };
                                        let mut i = 0;
                                        for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                                            if !((target - dst).norm_sqr().sqrt() < 1e-4) {
                                                dbg!(i / cs, i % cs, target, dst);
                                                panic!();
                                            }
                                            i += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
