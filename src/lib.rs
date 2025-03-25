#![no_std]
extern crate std;

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
    pub __pad_0__: usize,
    pub __pad_1__: usize,
    pub alpha: *const (),
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

pub unsafe fn millikernel(
    microkernel: unsafe extern "C" fn(),

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    ncols: usize,
    rhs_cs: isize,
    packed_rhs_cs: isize,

    dst: &Dst,
    row: usize,
    col: usize,

    nrows: usize,
    info: &mut MicrokernelInfo,
) {
    unsafe {
        core::arch::asm! {
            "push rbx",
            "push rbp",
            "mov rbx, r8",
            "mov rbp, r9",

            "test r12, r12",
            "jz 5f",

            "cmp rax, rbx",
            "jz 2f",

            "call r10",
            "add rcx, r11",
            "add rdx, r13",
            "add r15, 4",
            "mov rax, rbx",
            "dec r12",
            "jz 5f",

            ".align 16",
            "2:",
            "call r10",
            "add rcx, r11",
            "add rdx, r13",
            "add r15, 4",

            "dec r12",
            "jnz 2b",

            "5:",
            "pop rbp",
            "pop rbx",

            in("r10") microkernel,
            inout("rax") lhs => _,
            inout("r8") packed_lhs => _,
            inout("rcx") rhs => _,
            inout("rdx") packed_rhs => _,
            in("rdi") dst,
            in("rsi") nrows,
            in("r9") info,
            in("r11") rhs_cs,
            inout("r12") ncols => _,
            in("r13") packed_rhs_cs,
            in("r14") row,
            inout("r15") col => _,

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
    };
}

#[inline(never)]
pub unsafe fn kernel<'a>(
    microkernel: &'a [unsafe extern "C" fn()],
    len: usize,
    sizeof: usize,

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
    let mut lhs = lhs;
    let mut rhs = rhs;
    let mut packed_rhs = packed_rhs;
    let mut lhs_rs = lhs_rs;

    assert!(row_chunk.len() <= col_chunk.len());

    if row_chunk.len() == 0 {
        assert!(col_chunk.len() == 1);
        assert_eq!(col_chunk[0], 4);

        unsafe {
            let old = info.lhs_cs;

            if old == isize::MIN {
                info.lhs_cs = (nrows.next_multiple_of(len) * sizeof) as isize;
            }

            if ncols < 4 {
                millikernel(
                    microkernel[nrows.div_ceil(len) - 1 + (ncols - 1) * (microkernel.len() / 4)],
                    lhs,
                    packed_lhs,
                    rhs,
                    packed_rhs,
                    1,
                    rhs_cs[0],
                    packed_rhs_cs[0],
                    dst,
                    row,
                    col,
                    nrows,
                    info,
                );
            } else {
                millikernel(
                    microkernel[nrows.div_ceil(len) - 1 + (4 - 1) * (microkernel.len() / 4)],
                    lhs,
                    packed_lhs,
                    rhs,
                    packed_rhs,
                    ncols / 4,
                    rhs_cs[0],
                    packed_rhs_cs[0],
                    dst,
                    row,
                    col,
                    nrows,
                    info,
                );
                if ncols % 4 != 0 {
                    let rhs = rhs.wrapping_byte_offset((ncols / 4) as isize * rhs_cs[0]);
                    let packed_rhs =
                        packed_rhs.wrapping_byte_offset((ncols / 4) as isize * packed_rhs_cs[0]);
                    millikernel(
                        microkernel
                            [nrows.div_ceil(len) - 1 + (ncols % 4 - 1) * (microkernel.len() / 4)],
                        packed_lhs,
                        packed_lhs,
                        rhs,
                        packed_rhs,
                        1,
                        rhs_cs[0],
                        packed_rhs_cs[0],
                        dst,
                        row,
                        col + ncols / 4 * 4,
                        nrows,
                        info,
                    );
                }
            }
            info.lhs_cs = old;

            if packed_rhs != rhs as _ && packed_rhs_cs[0] != 0 {
                info.rhs_rs = 4 * sizeof as isize;
                info.rhs_cs = sizeof as isize;
            }
        }
    } else {
        assert!(row_chunk.len() + 1 == col_chunk.len());

        let (&first_col_chunk, col_chunk) = col_chunk.split_first().unwrap();
        let (&first_rhs_cs, rhs_cs) = rhs_cs.split_first().unwrap();
        let (&first_packed_rhs_cs, packed_rhs_cs) = packed_rhs_cs.split_first().unwrap();

        let mut j = 0;
        loop {
            let c_chunk = Ord::min(first_col_chunk, ncols - j);
            let old = (info.rhs_rs, info.rhs_cs);

            unsafe {
                let mut lhs = lhs;
                let mut packed_lhs = packed_lhs;
                let mut rhs = rhs;
                let mut rhs_cs = rhs_cs;

                let (&first_row_chunk, row_chunk) = row_chunk.split_first().unwrap();
                let (&first_lhs_rs, lhs_rs) = lhs_rs.split_first().unwrap();
                let (&first_packed_lhs_rs, packed_lhs_rs) = packed_lhs_rs.split_first().unwrap();

                let mut i = 0;
                loop {
                    let r_chunk = Ord::min(first_row_chunk, nrows - i);

                    let old = (info.lhs_rs, info.lhs_cs);

                    kernel(
                        microkernel,
                        len,
                        sizeof,
                        lhs,
                        packed_lhs,
                        rhs,
                        packed_rhs,
                        r_chunk,
                        c_chunk,
                        row_chunk,
                        col_chunk,
                        lhs_rs,
                        rhs_cs,
                        packed_lhs_rs,
                        packed_rhs_cs,
                        row + i,
                        col + j,
                        dst,
                        info,
                    );
                    lhs = lhs.wrapping_byte_offset(first_lhs_rs);
                    packed_lhs = packed_lhs.wrapping_byte_offset(first_packed_lhs_rs);
                    (info.lhs_rs, info.lhs_cs) = old;

                    i += r_chunk;
                    if i == nrows {
                        break;
                    }

                    if packed_rhs_cs[0] != 0 && rhs != packed_rhs {
                        rhs = packed_rhs;
                        rhs_cs = packed_rhs_cs;
                    }
                }
                if lhs != packed_lhs && first_packed_lhs_rs != 0 {
                    info.lhs_rs = sizeof as isize;
                    info.lhs_cs = isize::MIN;
                }
            }

            rhs = rhs.wrapping_byte_offset(first_rhs_cs);
            packed_rhs = packed_rhs.wrapping_byte_offset(first_packed_rhs_cs);
            (info.rhs_rs, info.rhs_cs) = old;

            j += c_chunk;
            if j == ncols {
                break;
            }

            if packed_lhs_rs[0] != 0 && lhs != packed_lhs {
                lhs = packed_lhs;
                lhs_rs = packed_lhs_rs;
            }
        }
        if rhs != packed_rhs && first_packed_rhs_cs != 0 {
            info.rhs_rs = 4 * sizeof as isize;
            info.rhs_cs = sizeof as isize;
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
                                    millikernel(
                                        F64_SIMD512[m.div_ceil(len) - 1 + ((n - 1) % 4) * 6],
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
                                        n.div_ceil(4),
                                        4 * k as isize * sizeof,
                                        4 * k as isize * sizeof,
                                        &Dst {
                                            ptr: dst.as_mut_ptr() as _,
                                            rs: 1 * sizeof,
                                            cs: cs as isize * sizeof,
                                            row_idx: null_mut(),
                                            col_idx: null_mut(),
                                        },
                                        0,
                                        0,
                                        m,
                                        &mut MicrokernelInfo {
                                            flags: 0,
                                            depth: k,
                                            lhs_rs: 1 * sizeof,
                                            lhs_cs: cs as isize * sizeof,
                                            rhs_rs: 1 * sizeof,
                                            rhs_cs: k as isize * sizeof,
                                            __pad_0__: 0,
                                            __pad_1__: 0,
                                            alpha: &raw const alpha as _,
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
        let len = 64 / size_of::<f64>();
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
                        &F64_SIMD512,
                        len,
                        sizeof as usize,
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
                            __pad_0__: 0,
                            __pad_1__: 0,
                            alpha: &raw const *&1.0f64 as _,
                        },
                    )
                }
                let mut i = 0;
                for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                    if !((target - dst).abs() < 1e-6) {
                        std::dbg!(i / cs, i % cs, target, dst);
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
                                            millikernel(
                                                C64_SIMD512
                                                    [m.div_ceil(len) - 1 + ((n - 1) % 4) * 6],
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
                                                n.div_ceil(4),
                                                4 * k as isize * sizeof,
                                                4 * k as isize * sizeof,
                                                &Dst {
                                                    ptr: dst.as_mut_ptr() as _,
                                                    rs: 1 * sizeof,
                                                    cs: cs as isize * sizeof,
                                                    row_idx: null_mut(),
                                                    col_idx: null_mut(),
                                                },
                                                0,
                                                0,
                                                m,
                                                &mut MicrokernelInfo {
                                                    flags: ((conj_lhs as usize) << 1)
                                                        | ((conj_different as usize) << 2),
                                                    depth: k,
                                                    lhs_rs: 1 * sizeof,
                                                    lhs_cs: cs as isize * sizeof,
                                                    rhs_rs: 1 * sizeof,
                                                    rhs_cs: k as isize * sizeof,
                                                    __pad_0__: 0,
                                                    __pad_1__: 0,
                                                    alpha: &raw const alpha as _,
                                                },
                                            )
                                        };
                                        let mut i = 0;
                                        for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                                            if !((target - dst).norm_sqr().sqrt() < 1e-6) {
                                                std::dbg!(i / cs, i % cs, target, dst);
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
                                    millikernel(
                                        F32_SIMD512[m.div_ceil(len) - 1 + ((n - 1) % 4) * 6],
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
                                        n.div_ceil(4),
                                        4 * k as isize * sizeof,
                                        4 * k as isize * sizeof,
                                        &Dst {
                                            ptr: dst.as_mut_ptr() as _,
                                            rs: 1 * sizeof,
                                            cs: cs as isize * sizeof,
                                            row_idx: null_mut(),
                                            col_idx: null_mut(),
                                        },
                                        0,
                                        0,
                                        m,
                                        &mut MicrokernelInfo {
                                            flags: 0,
                                            depth: k,
                                            lhs_rs: 1 * sizeof,
                                            lhs_cs: cs as isize * sizeof,
                                            rhs_rs: 1 * sizeof,
                                            rhs_cs: k as isize * sizeof,
                                            __pad_0__: 0,
                                            __pad_1__: 0,
                                            alpha: &raw const alpha as _,
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
        let sizeof = size_of::<f32>() as isize;
        let len = 64 / size_of::<f32>();
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
                    let row_chunk = [96 * 32, 96 * 16, 96];
                    let col_chunk = [96 * 64, 96 * 32, 96, 4];

                    let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
                    let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
                    let packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
                    let packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

                    kernel(
                        &F32_SIMD512,
                        len,
                        sizeof as usize,
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
                            __pad_0__: 0,
                            __pad_1__: 0,
                            alpha: &raw const *&1.0f32 as _,
                        },
                    )
                }
                let mut i = 0;
                for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                    if !((target - dst).abs() < 1e-6) {
                        std::dbg!(i / cs, i % cs, target, dst);
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
                                            millikernel(
                                                C32_SIMD512
                                                    [m.div_ceil(len) - 1 + ((n - 1) % 4) * 6],
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
                                                n.div_ceil(4),
                                                4 * k as isize * sizeof,
                                                4 * k as isize * sizeof,
                                                &Dst {
                                                    ptr: dst.as_mut_ptr() as _,
                                                    rs: 1 * sizeof,
                                                    cs: cs as isize * sizeof,
                                                    row_idx: null_mut(),
                                                    col_idx: null_mut(),
                                                },
                                                0,
                                                0,
                                                m,
                                                &mut MicrokernelInfo {
                                                    flags: ((conj_lhs as usize) << 1)
                                                        | ((conj_different as usize) << 2),
                                                    depth: k,
                                                    lhs_rs: 1 * sizeof,
                                                    lhs_cs: cs as isize * sizeof,
                                                    rhs_rs: 1 * sizeof,
                                                    rhs_cs: k as isize * sizeof,
                                                    __pad_0__: 0,
                                                    __pad_1__: 0,
                                                    alpha: &raw const alpha as _,
                                                },
                                            )
                                        };
                                        let mut i = 0;
                                        for (&target, &dst) in core::iter::zip(&*target, &*dst) {
                                            if !((target - dst).norm_sqr().sqrt() < 1e-4) {
                                                std::dbg!(i / cs, i % cs, target, dst);
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
