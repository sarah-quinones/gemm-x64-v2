#![no_std]
#![allow(non_upper_case_globals)]

use core::ptr::{null, null_mut};

extern crate std;
use std::dbg;

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

pub unsafe fn millikernel2(
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
    unsafe {
        core::arch::asm! {
            "push rbx",
            "push r8",
            "push r9",
            "mov rbx, r15",

            "test r8, r8",
            "jz 22f",
            "test r9, r9",
            "jz 22f",

            "bt qword ptr [rsi], 63",
            "jc 21f",

            // for j in .. { for i in .. { } }
            "push rcx",
            "push rdx",
            "push rax",
            "push qword ptr [rsi - 32]",
            "push qword ptr [rsi + 16]",

                ".align 16",
                "200:",
                "push rax",
                "push rbx",
                "push rcx",
                "push r8",
                "push qword ptr [rsi + 32]",
                "push qword ptr [rsi + 40]",
                "push qword ptr [rsi + 48]",

                    "cmp rcx, rdx",
                    "jz 2000f",

                    "call r10",
                    "add rax, [rsi - 32]",
                    "add rbx, [rsi - 24]",
                    "mov rcx, rdx",
                    "mov qword ptr [rsi + 40], 0",
                    "test r8, r8",
                    "jz 201f",

                    ".align 16",
                    "2000:",
                    "call r10",
                    "add rax, [rsi - 32]",
                    "add rbx, [rsi - 24]",
                    "test r8, r8",
                    "jnz 2000b",

                "201:",
                "pop qword ptr [rsi + 48]",
                "pop qword ptr [rsi + 40]",
                "pop qword ptr [rsi + 32]",
                "pop r8",
                "pop rcx",
                "pop rbx",
                "pop rax",

                "test r9, r9",
                "jz 20f",

                "add rcx, [rsi - 16]",
                "add rdx, [rsi - 8]",

                "cmp rax, rbx",
                "jz 200b",
                "cmp qword ptr [rsi - 24], 0",
                "jz 200b",

                "mov rax, rbx",
                "mov r15, [rsi - 24]",
                "mov [rsi - 32], r15",
                "xor r15, r15",
                "mov [rsi + 16], r15",
                "jmp 200b",

            ".align 16",
            "20:",
            "pop qword ptr [rsi + 16]",
            "pop qword ptr [rsi - 32]",
            "pop rax",
            "pop rdx",
            "pop rcx",

            "jmp 22f",

            ".align 16",
            "21:",
            // for i in .. { for j in .. { } }
            "push rax",
            "push rbx",
            "push rcx",
            "push qword ptr [rsi - 16]",
            "push qword ptr [rsi + 40]",

                ".align 16",
                "200:",
                "push rcx",
                "push rdx",
                "push rax",
                "push r9",
                "push qword ptr [rsi + 16]",
                "push qword ptr [rsi + 24]",
                "push qword ptr [rsi + 56]",

                    "cmp rax, rbx",
                    "jz 2000f",

                    "call r10",
                    "add rcx, [rsi - 16]",
                    "add rdx, [rsi - 8]",
                    "mov rax, rbx",
                    "mov qword ptr [rsi + 16], 0",
                    "test r9, r9",
                    "jz 201f",

                    ".align 16",
                    "2000:",
                    "call r10",
                    "add rcx, [rsi - 16]",
                    "add rdx, [rsi - 8]",
                    "test r9, r9",
                    "jnz 2000b",

                "201:",
                "pop qword ptr [rsi + 56]",
                "pop qword ptr [rsi + 24]",
                "pop qword ptr [rsi + 16]",
                "pop r9",
                "pop rax",
                "pop rdx",
                "pop rcx",

                "test r8, r8",
                "jz 20f",

                "add rax, [rsi - 32]",
                "add rbx, [rsi - 24]",

                "cmp rcx, rdx",
                "jz 200b",
                "cmp qword ptr [rsi - 8], 0",
                "jz 200b",

                "mov rcx, rdx",
                "mov r15, [rsi - 8]",
                "mov [rsi - 16], r15",
                "xor r15, r15",
                "mov [rsi + 40], r15",
                "jmp 200b",

            ".align 16",
            "20:",
            "pop qword ptr [rsi + 40]",
            "pop qword ptr [rsi - 16]",
            "pop rcx",
            "pop rbx",
            "pop rax",

            ".align 16",
            "22:",
            "mov r15, rbx",
            "pop r9",
            "pop r8",
            "pop rbx",

            in("rax") lhs,
            in("r15") packed_lhs, // rbx
            in("rcx") rhs,
            in("rdx") packed_rhs,
            in("rdi") dst,
            in("rsi") &raw mut milli.micro,
            in("r8") nrows,
            in("r9") ncols,

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
    };
}

pub unsafe fn millikernel<const NR: usize>(
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
            "push rsi",
            "mov rbx, r8",
            "mov rbp, r9",

            "test r12, r12",
            "jz 5f",

            "cmp rax, rbx",
            "jz 2f",

            "call r10",
            "add rcx, r11",
            "add rdx, r13",
            "mov rsi, [rsp]",
            "add r15, {NR}",
            "mov rax, rbx",
            "mov qword ptr [r9 + 16], 0",
            "mov qword ptr [r9 + 24], 0",
            "dec r12",
            "jz 5f",

            ".align 16",
            "2:",
            "call r10",
            "add rcx, r11",
            "add rdx, r13",
            "mov rsi, [rsp]",
            "add r15, {NR}",

            "dec r12",
            "jnz 2b",

            "5:",
            "pop rsi",
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
            NR = const NR,

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
pub unsafe fn kernel2<'a>(
    microkernel: &'a [unsafe extern "C" fn()],
    len: usize,
    sizeof: usize,
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
        bool,
        bool,
    ); 16] = [(
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
        false,
        false,
    ); 16];

    stack[0] = (
        lhs, packed_lhs, rhs, packed_rhs, row, col, nrows, ncols, 0, 0, false, false,
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
                millikernel2(
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

                let (_, _, _, _, _, _, nrows, ncols, i, j, _, _) = &mut stack[depth];

                let col_chunk = col_chunk[depth];
                let row_chunk = row_chunk[depth];

                let j_chunk = Ord::min(col_chunk, *ncols - *j);
                let i_chunk = Ord::min(row_chunk, *nrows - *i);

                if milli.micro.flags >> 63 == 0 {
                    *i += i_chunk;
                    if *i == *nrows {
                        *i = 0;
                        *j += j_chunk;

                        if *j == *ncols {
                            *j = 0;
                            continue;
                        }
                    }
                } else {
                    *j += j_chunk;
                    if *j == *ncols {
                        *j = 0;
                        *i += i_chunk;

                        if *i == *nrows {
                            *i = 0;
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
                lhs.wrapping_byte_offset(lhs_rs * (i / row_chunk) as isize),
                packed_lhs.wrapping_byte_offset(plhs_rs * (i / row_chunk) as isize),
                rhs.wrapping_byte_offset(rhs_cs * (j / col_chunk) as isize),
                packed_rhs.wrapping_byte_offset(prhs_cs * (j / col_chunk) as isize),
                row + i,
                col + j,
                i_chunk,
                j_chunk,
                0,
                0,
                is_packed_lhs || (j > 0 && packed_lhs_rs[depth] != 0),
                is_packed_rhs || (i > 0 && packed_rhs_cs[depth] != 0),
            );
            continue;
        }

        if depth == 0 {
            break;
        }
    }
}

#[inline(never)]
pub unsafe fn kernel<'a>(
    microkernel: &'a [unsafe extern "C" fn()],
    len: usize,
    sizeof: usize,
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
    rev: &mut [bool],
) {
    let mut lhs = lhs;
    let mut lhs_rs = lhs_rs;

    assert!(row_chunk.len() <= col_chunk.len());

    if row_chunk.len() == 0 {
        assert!(col_chunk.len() == 1);
        assert_eq!(col_chunk[0], nr);
        let millikernel = if nr == 8 {
            millikernel::<8>
        } else {
            millikernel::<4>
        };

        unsafe {
            let old = info.lhs_cs;

            if old == isize::MIN {
                info.lhs_cs = (nrows.next_multiple_of(len) * sizeof) as isize;
            }

            let mr = microkernel.len() / nr;
            let i = mr - nrows.div_ceil(len);

            if ncols >= nr {
                millikernel(
                    microkernel[nr * i + nr - 1],
                    lhs,
                    packed_lhs,
                    rhs,
                    packed_rhs,
                    ncols / nr,
                    rhs_cs[0],
                    packed_rhs_cs[0],
                    dst,
                    row,
                    col,
                    nrows,
                    info,
                );
            }
            if ncols % nr != 0 {
                let rhs = rhs.wrapping_byte_offset((ncols / nr) as isize * rhs_cs[0]);
                let packed_rhs =
                    packed_rhs.wrapping_byte_offset((ncols / nr) as isize * packed_rhs_cs[0]);
                millikernel(
                    microkernel[nr * i + ncols % nr - 1],
                    packed_lhs,
                    packed_lhs,
                    rhs,
                    packed_rhs,
                    1,
                    rhs_cs[0],
                    packed_rhs_cs[0],
                    dst,
                    row,
                    col + ncols / nr * nr,
                    nrows,
                    info,
                );
            }

            if packed_rhs != rhs as _ && packed_rhs_cs[0] != 0 {
                info.rhs_rs = (nr * sizeof) as isize;
                info.rhs_cs = sizeof as isize;
            }
        }
    } else {
        assert!(row_chunk.len() + 1 == col_chunk.len());

        let (&first_col_chunk, col_chunk) = col_chunk.split_first().unwrap();
        let (&first_rhs_cs, rhs_cs) = rhs_cs.split_first().unwrap();
        let (&first_packed_rhs_cs, packed_rhs_cs) = packed_rhs_cs.split_first().unwrap();

        let mut j = if rev[0] { ncols } else { 0 };
        loop {
            if rev[0] {
                j = (j - 1) / first_col_chunk * first_col_chunk;
            }

            let c_chunk = Ord::min(first_col_chunk, ncols - j);
            let old = (info.rhs_rs, info.rhs_cs);

            unsafe {
                let mut rhs_cs = rhs_cs;
                let mut rhs =
                    rhs.wrapping_byte_offset(first_rhs_cs * (j / first_col_chunk) as isize);
                let packed_rhs = packed_rhs
                    .wrapping_byte_offset(first_packed_rhs_cs * (j / first_col_chunk) as isize);

                let (&first_row_chunk, row_chunk) = row_chunk.split_first().unwrap();
                let (&first_lhs_rs, lhs_rs) = lhs_rs.split_first().unwrap();
                let (&first_packed_lhs_rs, packed_lhs_rs) = packed_lhs_rs.split_first().unwrap();

                let mut i = if rev[1] { nrows } else { 0 };
                loop {
                    if rev[1] {
                        i = (i - 1) / first_row_chunk * first_row_chunk;
                    }

                    let r_chunk = Ord::min(first_row_chunk, nrows - i);

                    let old = (info.lhs_rs, info.lhs_cs);

                    kernel(
                        microkernel,
                        len,
                        sizeof,
                        nr,
                        lhs.wrapping_byte_offset(first_lhs_rs * (i / first_row_chunk) as isize),
                        packed_lhs.wrapping_byte_offset(
                            first_packed_lhs_rs * (i / first_row_chunk) as isize,
                        ),
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
                        &mut rev[2..],
                    );
                    (info.lhs_rs, info.lhs_cs) = old;

                    if rev[1] {
                        if i == 0 {
                            break;
                        }
                    } else {
                        i += r_chunk;
                        if i == nrows {
                            break;
                        }
                    }

                    if packed_rhs_cs[0] != 0 && rhs != packed_rhs {
                        rhs = packed_rhs;
                        rhs_cs = packed_rhs_cs;
                    }
                }
                if nrows >= usize::MAX {
                    rev[1] = !rev[1];
                }
                if lhs != packed_lhs && first_packed_lhs_rs != 0 {
                    info.lhs_rs = sizeof as isize;
                    info.lhs_cs = isize::MIN;
                }
            }

            (info.rhs_rs, info.rhs_cs) = old;

            if rev[0] {
                if j == 0 {
                    break;
                }
            } else {
                j += c_chunk;
                if j == ncols {
                    break;
                }
            }

            if packed_lhs_rs[0] != 0 && lhs != packed_lhs {
                lhs = packed_lhs;
                lhs_rs = packed_lhs_rs;
            }
        }
        if rhs != packed_rhs && first_packed_rhs_cs != 0 {
            info.rhs_rs = (nr * sizeof) as isize;
            info.rhs_cs = sizeof as isize;
        }
    }
    if ncols >= usize::MAX {
        rev[0] = !rev[0];
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
                                    millikernel::<4>(
                                        F64_SIMD512x4[(6 - m.div_ceil(len)) * 4 + ((n - 1) % 4)],
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
                                            row: 0,
                                            col: 0,
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
                        &F64_SIMD512x4[..24],
                        len,
                        sizeof as usize,
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
                        &mut [false; 32],
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
                                            millikernel::<4>(
                                                C64_SIMD512x4
                                                    [(6 - m.div_ceil(len)) * 4 + ((n - 1) % 4)],
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
                                                    row: 0,
                                                    col: 0,
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
                                    millikernel::<4>(
                                        F32_SIMD512x4[(6 - m.div_ceil(len)) * 4 + ((n - 1) % 4)],
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
                                            row: 0,
                                            col: 0,
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
        let m = 6000usize;
        let n = 2000usize;
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
                std::dbg!(pack_lhs, pack_rhs);

                let dst = &mut *avec![0.0; cs * n];
                let packed_lhs = &mut *avec![0.0f32; m.next_multiple_of(16) * k];
                let packed_rhs =
                    &mut *avec![0.0; if pack_rhs { n.next_multiple_of(4) * k } else { 0 }];

                unsafe {
                    let row_chunk = [m, 96 * 32, 96 * 16, 96 * 4, 96];
                    let col_chunk = [n, 1024, 256, 64, 16, 4];

                    let lhs_rs = row_chunk.map(|m| m as isize * sizeof);
                    let rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
                    let packed_lhs_rs = row_chunk.map(|m| (m * k) as isize * sizeof);
                    let packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);

                    kernel2(
                        &F32_SIMD512x4[..24],
                        len,
                        sizeof as usize,
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
                        // &mut [false; 32],
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
                                            millikernel::<4>(
                                                C32_SIMD512x4
                                                    [(6 - m.div_ceil(len)) * 4 + ((n - 1) % 4)],
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
                                                    row: 0,
                                                    col: 0,
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
