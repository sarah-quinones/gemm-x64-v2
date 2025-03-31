#![cfg_attr(all(not(test), any()), no_std)]
#![allow(non_upper_case_globals)]
#![allow(dead_code, unused_variables)]

use core::{
    ptr::{null, null_mut},
    sync::atomic::{AtomicU8, AtomicUsize, Ordering},
};

include!(concat!(env!("OUT_DIR"), "/asm.rs"));

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Position {
    pub row: usize,
    pub col: usize,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MicrokernelInfo {
    pub flags: usize,
    pub depth: usize,
    pub lhs_rs: isize,
    pub lhs_cs: isize,
    pub rhs_rs: isize,
    pub rhs_cs: isize,
    pub alpha: *const (),

    // dst
    pub ptr: *mut (),
    pub rs: isize,
    pub cs: isize,
    pub row_idx: *const (),
    pub col_idx: *const (),
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

#[inline(always)]
pub unsafe fn call_microkernel(
    microkernel: unsafe extern "C" fn(),
    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    mut nrows: usize,
    mut ncols: usize,

    micro: &MicrokernelInfo,
    position: &mut Position,
) -> (usize, usize) {
    unsafe {
        core::arch::asm! {
            "call r10",

            in("rax") lhs,
            in("r15") packed_lhs,
            in("rcx") rhs,
            in("rdx") packed_rhs,
            in("rdi") position,
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

    milli: &MillikernelInfo,

    pos: &mut Position,
) {
    let mut rhs = rhs;
    let mut nrows = nrows;
    let mut lhs = lhs;
    let mut packed_lhs = packed_lhs;

    loop {
        let rs = milli.micro.lhs_rs;
        unsafe {
            let mut rhs = rhs;
            let mut packed_rhs = packed_rhs;
            let mut ncols = ncols;
            let mut lhs = lhs;
            let col = pos.col;

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
                        &milli.micro,
                        pos,
                    );

                    if !rhs.is_null() {
                        rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
                    }
                    packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);

                    $(if lhs != packed_lhs {
                        $lhs = null();
                    })?
                }};
            }
            iter!(lhs);
            while ncols > 0 {
                iter!();
            }
            pos.col = col;
        }

        if !lhs.is_null() {
            lhs = lhs.wrapping_byte_offset(milli.lhs_rs);
        }
        packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs);
        if rhs != packed_rhs {
            rhs = null();
        }

        if nrows == 0 {
            break;
        }
    }
}

pub unsafe fn millikernel_colmajor(
    microkernel: unsafe extern "C" fn(),

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    milli: &MillikernelInfo,

    pos: &mut Position,
) {
    let mut lhs = lhs;
    let mut ncols = ncols;
    let mut rhs = rhs;
    let mut packed_rhs = packed_rhs;

    loop {
        let cs = milli.micro.rhs_cs;
        unsafe {
            let mut lhs = lhs;
            let mut packed_lhs = packed_lhs;
            let mut nrows = nrows;
            let mut rhs = rhs;
            let row = pos.row;

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
                        &milli.micro,
                        pos,
                    );

                    if !lhs.is_null() {
                        lhs = lhs.wrapping_byte_offset(milli.lhs_rs);
                    }
                    packed_lhs = packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs);

                    $(if rhs != packed_rhs {
                        $rhs = null();
                    })?
                }};
            }
            iter!(rhs);
            while nrows > 0 {
                iter!();
            }
            pos.row = row;
        }

        if !rhs.is_null() {
            rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
        }
        packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);
        if lhs != packed_lhs {
            lhs = null();
        }

        if ncols == 0 {
            break;
        }
    }
}

pub unsafe fn millikernel_par_colmajor(
    thd_id: usize,
    n_threads: usize,

    microkernel_job: &[AtomicU8],
    pack_lhs_job: &[AtomicU8],
    pack_rhs_job: &[AtomicU8],
    finished: &AtomicUsize,
    hyper: usize,

    mr: usize,
    nr: usize,

    microkernel: unsafe extern "C" fn(),

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    milli: &MillikernelInfo,

    pos: Position,
) {
    let max_j = ncols.div_ceil(nr);
    let max_i = nrows.div_ceil(mr);

    let mut hyper = hyper;
    if n_threads % hyper != 0 {
        hyper = 1;
    }

    let n_threads0 = n_threads / hyper;
    let n_threads1 = hyper;

    let thd_id0 = thd_id % (n_threads / hyper);
    let thd_id1 = thd_id / (n_threads / hyper);

    let l2 = 96 / mr * mr;
    let l1 = 4;

    {
        // let max_jobs = max_i * max_j;
        let (div, rem) = (max_i / n_threads0, max_i % n_threads0);

        let i_start = thd_id0 * div + Ord::min(thd_id0, rem);
        let mut i_end = i_start + div;

        if thd_id0 < rem {
            i_end += 1;
        }

        let row = i_start * mr;
        let pos = Position {
            row: pos.row + row,
            col: pos.col,
        };
        let nrows = Ord::min(nrows - row, mr * (i_end - i_start));

        'main: for i in 0..nrows.div_ceil(mr).div_ceil(l2 / mr) {
            let row = i * l2;

            let pos = Position {
                row: pos.row + row,
                col: pos.col,
            };

            let nrows = Ord::min(nrows - row, l2);
            let i_start = i_start + l2 / mr * i;

            let packed_lhs =
                packed_lhs.wrapping_byte_offset(milli.packed_lhs_rs * i_start as isize);

            let mut lhs = lhs;
            if !lhs.is_null() {
                lhs = lhs.wrapping_byte_offset(milli.lhs_rs * i_start as isize);
            }
            let lhs = lhs;

            let pack_lhs_job = &pack_lhs_job[i_start..];

            let mut rhs = rhs;
            let mut packed_rhs = packed_rhs;

            let max_i = nrows.div_ceil(mr);
            let c = max_i.div_ceil(8);

            let (div, rem) = (max_j / n_threads1, max_j % n_threads1);

            let mut j = thd_id1 * div + Ord::min(thd_id1, rem);
            let mut j_end = j + div;
            if thd_id1 < rem {
                j_end += 1;
            }

            let mut col = j * nr;

            if !rhs.is_null() {
                rhs = rhs.wrapping_byte_offset(j as isize * milli.rhs_cs);
            }
            packed_rhs = packed_rhs.wrapping_byte_offset(j as isize * milli.packed_rhs_cs);

            loop {
                let col_chunk = Ord::min(ncols - col, nr);

                unsafe {
                    let mut rhs = rhs;
                    let mut row = 0;
                    let mut i = 0;

                    while row < nrows {
                        let row_chunk = Ord::min(nrows - row, mr);
                        {
                            let mut rhs = rhs;
                            let mut packed_rhs = packed_rhs;

                            for j in j..Ord::min(j_end, j + l1) {
                                let packed_lhs = packed_lhs
                                    .wrapping_byte_offset(milli.packed_lhs_rs * i as isize);
                                let mut lhs = lhs;

                                if !lhs.is_null() {
                                    lhs = lhs.wrapping_byte_offset(milli.lhs_rs * i as isize);

                                    if lhs != packed_lhs {
                                        let val = pack_lhs_job[i].load(Ordering::Acquire);

                                        if val == 2 {
                                            lhs = null();
                                        }
                                    }
                                }

                                {
                                    if !rhs.is_null() && rhs != packed_rhs {
                                        let val = pack_rhs_job[j].load(Ordering::Acquire);

                                        if val == 2 {
                                            rhs = null();
                                        }
                                    }

                                    call_microkernel(
                                        microkernel,
                                        lhs,
                                        packed_lhs,
                                        rhs,
                                        packed_rhs,
                                        row_chunk,
                                        col_chunk,
                                        &milli.micro,
                                        &mut Position {
                                            row: row + pos.row,
                                            col: j * nr + pos.col,
                                        },
                                    );

                                    if !lhs.is_null() && lhs != packed_lhs {
                                        pack_lhs_job[i].store(2, Ordering::Release);
                                    }
                                    if !rhs.is_null() && rhs != packed_rhs {
                                        pack_rhs_job[j].store(2, Ordering::Release);
                                    }
                                }
                                if !rhs.is_null() {
                                    rhs = rhs.wrapping_byte_offset(milli.rhs_cs);
                                }
                                packed_rhs = packed_rhs.wrapping_byte_offset(milli.packed_rhs_cs);
                            }
                        }
                        i += 1;
                        row += row_chunk;
                        if rhs != packed_rhs {
                            rhs = null();
                        }
                    }
                }

                col += col_chunk;
                j += Ord::min(l1, j_end - j);
                if j == j_end {
                    continue 'main;
                }

                if !rhs.is_null() {
                    rhs = rhs.wrapping_byte_offset(l1 as isize * milli.rhs_cs);
                }
                packed_rhs = packed_rhs.wrapping_byte_offset(l1 as isize * milli.packed_rhs_cs);
            }
        }
    }
}

pub unsafe trait Millikernel {
    unsafe fn call(
        &mut self,

        microkernel: unsafe extern "C" fn(),

        lhs: *const (),
        packed_lhs: *mut (),

        rhs: *const (),
        packed_rhs: *mut (),

        nrows: usize,
        ncols: usize,

        milli: &MillikernelInfo,

        pos: Position,
    );
}

struct Milli;
struct MilliPar<'a, 'b> {
    mr: usize,
    nr: usize,
    hyper: usize,
    microkernel_job: Box<[AtomicU8]>,
    pack_lhs_job: Box<[AtomicU8]>,
    pack_rhs_job: Box<[AtomicU8]>,
    finished: AtomicUsize,
    n_threads: usize,
    __: &'a &'b (),
}

unsafe impl Millikernel for Milli {
    unsafe fn call(
        &mut self,

        microkernel: unsafe extern "C" fn(),

        lhs: *const (),
        packed_lhs: *mut (),

        rhs: *const (),
        packed_rhs: *mut (),

        nrows: usize,
        ncols: usize,

        milli: &MillikernelInfo,
        pos: Position,
    ) {
        unsafe {
            (if milli.micro.flags >> 63 == 1 {
                millikernel_rowmajor
            } else {
                millikernel_colmajor
            })(
                microkernel,
                lhs,
                packed_lhs,
                rhs,
                packed_rhs,
                nrows,
                ncols,
                milli,
                &mut { pos },
            )
        }
    }
}

#[derive(Copy, Clone)]
struct Cell<T>(T);
unsafe impl<T> Sync for Cell<T> {}
unsafe impl<T> Send for Cell<T> {}

unsafe impl Millikernel for MilliPar<'_, '_> {
    unsafe fn call(
        &mut self,

        microkernel: unsafe extern "C" fn(),

        lhs: *const (),
        packed_lhs: *mut (),

        rhs: *const (),
        packed_rhs: *mut (),

        nrows: usize,
        ncols: usize,

        milli: &MillikernelInfo,
        pos: Position,
    ) {
        let lhs = Cell(lhs);
        let rhs = Cell(rhs);
        let packed_lhs = Cell(packed_lhs);
        let packed_rhs = Cell(packed_rhs);
        let milli = Cell(milli);

        self.microkernel_job.fill_with(|| AtomicU8::new(0));
        self.pack_lhs_job.fill_with(|| AtomicU8::new(0));
        self.pack_rhs_job.fill_with(|| AtomicU8::new(0));
        self.finished = AtomicUsize::new(0);

        syncthreads::for_each_raw(self.n_threads, |tid| unsafe {
            let milli = { milli }.0;

            (if milli.micro.flags >> 63 == 1 {
                todo!()
            } else {
                millikernel_par_colmajor
            })(
                tid,
                self.n_threads,
                &self.microkernel_job,
                &self.pack_lhs_job,
                &self.pack_rhs_job,
                &self.finished,
                self.hyper,
                self.mr,
                self.nr,
                microkernel,
                { lhs }.0,
                { packed_lhs }.0,
                { rhs }.0,
                { packed_rhs }.0,
                nrows,
                ncols,
                milli,
                pos,
            )
        });
    }
}

#[inline(never)]
unsafe fn kernel_imp(
    millikernel: &mut dyn Millikernel,

    microkernel: &[unsafe extern "C" fn()],
    mr: usize,
    nr: usize,

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    row_chunk: &[usize],
    col_chunk: &[usize],
    lhs_rs: &[isize],
    rhs_cs: &[isize],
    packed_lhs_rs: &[isize],
    packed_rhs_cs: &[isize],

    row: usize,
    col: usize,

    pos: Position,
    info: &MicrokernelInfo,
) {
    let _ = mr;

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

    let mut pos = pos;
    let mut depth = 0;
    let max_depth = row_chunk.len();

    let milli_rs = *lhs_rs.last().unwrap();
    let milli_cs = *rhs_cs.last().unwrap();

    let micro_rs = info.lhs_rs;
    let micro_cs = info.rhs_cs;

    let milli = MillikernelInfo {
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
            ii,
            jj,
            is_packed_lhs,
            is_packed_rhs,
        ) = stack[depth];

        if depth + 1 == max_depth {
            let mut lhs = lhs;
            let mut rhs = rhs;

            pos.row = row;
            pos.col = col;

            if is_packed_lhs && lhs != packed_lhs {
                lhs = null();
            }
            if is_packed_rhs && rhs != packed_rhs {
                rhs = null();
            }

            unsafe {
                millikernel.call(
                    microkernel,
                    lhs,
                    packed_lhs,
                    rhs,
                    packed_rhs,
                    nrows,
                    ncols,
                    &milli,
                    pos,
                );
            }

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

pub unsafe fn kernel_rayon(
    n_threads: usize,
    microkernel: &[unsafe extern "C" fn()],
    mr: usize,
    nr: usize,

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    row_chunk: &[usize],
    col_chunk: &[usize],
    lhs_rs: &[isize],
    rhs_cs: &[isize],
    packed_lhs_rs: &[isize],
    packed_rhs_cs: &[isize],

    row: usize,
    col: usize,

    pos: Position,
    info: &MicrokernelInfo,
) {
    let max_i = nrows.div_ceil(mr);
    let max_j = ncols.div_ceil(nr);
    let max_jobs = max_i * max_j;
    let c = max_i;

    let lhs = Cell(lhs);
    let rhs = Cell(rhs);
    let packed_lhs = Cell(packed_lhs);
    let packed_rhs = Cell(packed_rhs);
    let info = Cell(info);

    syncthreads::with_lock(n_threads, || {
        unsafe {
            kernel_imp(
                &mut MilliPar {
                    mr,
                    nr,
                    hyper: 8,
                    microkernel_job: (0..c * max_j).map(|_| AtomicU8::new(0)).collect(),
                    pack_lhs_job: (0..max_i).map(|_| AtomicU8::new(0)).collect(),
                    pack_rhs_job: (0..max_j).map(|_| AtomicU8::new(0)).collect(),
                    finished: AtomicUsize::new(0),
                    n_threads,
                    __: &&(),
                },
                microkernel,
                mr,
                nr,
                { lhs }.0,
                { packed_lhs }.0,
                { rhs }.0,
                { packed_rhs }.0,
                nrows,
                ncols,
                row_chunk,
                col_chunk,
                lhs_rs,
                rhs_cs,
                packed_lhs_rs,
                packed_rhs_cs,
                row,
                col,
                pos,
                { info }.0,
            )
        };
    });
}

pub unsafe fn kernel(
    microkernel: &[unsafe extern "C" fn()],
    mr: usize,
    nr: usize,

    lhs: *const (),
    packed_lhs: *mut (),

    rhs: *const (),
    packed_rhs: *mut (),

    nrows: usize,
    ncols: usize,

    row_chunk: &[usize],
    col_chunk: &[usize],
    lhs_rs: &[isize],
    rhs_cs: &[isize],
    packed_lhs_rs: &[isize],
    packed_rhs_cs: &[isize],

    row: usize,
    col: usize,

    pos: Position,
    info: &MicrokernelInfo,
) {
    unsafe {
        kernel_imp(
            &mut Milli,
            microkernel,
            mr,
            nr,
            lhs,
            packed_lhs,
            rhs,
            packed_rhs,
            nrows,
            ncols,
            row_chunk,
            col_chunk,
            lhs_rs,
            rhs_cs,
            packed_lhs_rs,
            packed_rhs_cs,
            row,
            col,
            pos,
            info,
        )
    };
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
                                                alpha: &raw const alpha as _,
                                                ptr: dst.as_mut_ptr() as _,
                                                rs: 1 * sizeof,
                                                cs: cs as isize * sizeof,
                                                row_idx: null_mut(),
                                                col_idx: null_mut(),
                                            },
                                        },
                                        &mut Position { row: 0, col: 0 },
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
                        48,
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
                        Position { row: 0, col: 0 },
                        &MicrokernelInfo {
                            flags: 0,
                            depth: k,
                            lhs_rs: sizeof,
                            lhs_cs: cs as isize * sizeof,
                            rhs_rs: sizeof,
                            rhs_cs: k as isize * sizeof,
                            alpha: &raw const *&1.0f64 as _,
                            ptr: dst.as_mut_ptr() as _,
                            rs: sizeof,
                            cs: cs as isize * sizeof,
                            row_idx: null_mut(),
                            col_idx: null_mut(),
                        },
                    );
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
                                                        alpha: &raw const alpha as _,
                                                        ptr: dst.as_mut_ptr() as _,
                                                        rs: 1 * sizeof,
                                                        cs: cs as isize * sizeof,
                                                        row_idx: null_mut(),
                                                        col_idx: null_mut(),
                                                    },
                                                },
                                                &mut Position { row: 0, col: 0 },
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
                                                alpha: &raw const alpha as _,
                                                ptr: dst.as_mut_ptr() as _,
                                                rs: 1 * sizeof,
                                                cs: cs as isize * sizeof,
                                                row_idx: null_mut(),
                                                col_idx: null_mut(),
                                            },
                                        },
                                        &mut Position { row: 0, col: 0 },
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
                    let mut packed_rhs_cs = col_chunk.map(|n| (n * k) as isize * sizeof);
                    packed_rhs_cs[0] = 0;

                    kernel(
                        &F32_SIMD512x4[..24],
                        96,
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
                        Position { row: 0, col: 0 },
                        &MicrokernelInfo {
                            flags: 0,
                            depth: k,
                            lhs_rs: sizeof,
                            lhs_cs: cs as isize * sizeof,
                            rhs_rs: sizeof,
                            rhs_cs: k as isize * sizeof,
                            alpha: &raw const *&1.0f32 as _,
                            ptr: dst.as_mut_ptr() as _,
                            rs: sizeof,
                            cs: cs as isize * sizeof,
                            row_idx: null_mut(),
                            col_idx: null_mut(),
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
                                                        alpha: &raw const alpha as _,
                                                        ptr: dst.as_mut_ptr() as _,
                                                        rs: 1 * sizeof,
                                                        cs: cs as isize * sizeof,
                                                        row_idx: null_mut(),
                                                        col_idx: null_mut(),
                                                    },
                                                },
                                                &mut Position { row: 0, col: 0 },
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
