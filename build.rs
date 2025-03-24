#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_labels,
    unused_macros
)]

use std::env;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

use defer::defer;
use interpol::{format, println};
use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Index;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! setup {
    ($ctx: ident) => {
        macro_rules! asm {
            ($code: tt) => {{
                asm!($code, "");
            }};

            ($code: tt, $comment: tt) => {{
                use std::fmt::Write;

                let code = &mut *$ctx.code.borrow_mut();

                let old = code.len();
                ::interpol::write!(code, "    ").unwrap();
                ::interpol::write!(code, $code).unwrap();
                let new = code.len();

                let mut len = new - old;

                while len < 40 {
                    *code += " ";
                    len += 1;
                }
                *code += "// ";

                ::interpol::writeln!(code, $comment).unwrap();
            }};
        }

        macro_rules! reg {
            ($name: ident) => {
                let $name = $ctx.reg(::std::stringify!($name));
                ::defer::defer!($ctx.reg_drop($name, ::std::stringify!($name)));
            };

            (&$name: ident) => {
                $name = $ctx.reg(::std::stringify!($name));
                ::defer::defer!($ctx.reg_drop($name, ::std::stringify!($name)));
            };
        }

        macro_rules! label {
            ($name: ident) => {
                let $name = $ctx.label(::std::stringify!($name));
                ::defer::defer!($ctx.label_drop($name, ::std::stringify!($name)));
            };
        }
    };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Reg {
    rax = 0,
    rbx = 1,
    rcx = 2,
    rdx = 3,
    rdi = 4,
    rsi = 5,
    rbp = 6,
    rsp = 7,
    r8 = 8,
    r9 = 9,
    r10 = 10,
    r11 = 11,
    r12 = 12,
    r13 = 13,
    r14 = 14,
    r15 = 15,
}
pub use Reg::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Float {
    F32,
    C32,
    F64,
    C64,
}

impl Reg {
    pub const ALL: &[Self] = &[
        Self::rax,
        Self::rbx,
        Self::rcx,
        Self::rdx,
        Self::rdi,
        Self::rsi,
        Self::rbp,
        Self::rsp,
        Self::r8,
        Self::r9,
        Self::r10,
        Self::r11,
        Self::r12,
        Self::r13,
        Self::r14,
        Self::r15,
    ];
}

impl Float {
    pub fn sizeof(self) -> usize {
        match self {
            Float::F32 => 4,
            Float::C32 => 8,
            Float::F64 => 8,
            Float::C64 => 16,
        }
    }
}

impl std::fmt::Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Index<Reg> for Ctx {
    type Output = Cell<bool>;

    fn index(&self, index: Reg) -> &Self::Output {
        &self.reg_busy[index as usize]
    }
}

pub type Code = RefCell<String>;

pub struct Ctx {
    pub reg_busy: [Cell<bool>; 16],
    pub label: Cell<usize>,
    pub code: Code,
}

impl Ctx {
    pub fn new() -> Self {
        Self {
            reg_busy: [const { Cell::new(false) }; 16],
            label: Cell::new(2),
            code: RefCell::new(String::new()),
        }
    }

    pub fn reg(&self, name: &str) -> Reg {
        setup!(self);

        for &reg in Reg::ALL {
            if !self[reg].get() {
                asm!("push {reg}", "save before reg alloc `{name}`");
                self[reg].set(true);
                return reg;
            }
        }

        panic!();
    }

    pub fn reg_drop(&self, reg: Reg, name: &str) {
        setup!(self);

        self[reg].set(false);
        asm!("pop {reg}", "restore after reg dealloc `{name}`");
    }

    pub fn label(&self, name: &str) -> usize {
        let _ = name;
        let label = self.label.get();
        assert!(label < 10);
        self.label.set(label + 1);
        label
    }

    pub fn label_drop(&self, label: usize, name: &str) {
        let _ = name;
        self.label.set(label);
    }
}

const VERSION_MAJOR: usize = 0;
const VERSION_MINOR: usize = 22;
const PREFIX: LazyLock<String> =
    LazyLock::new(|| format!("[faer v{VERSION_MAJOR}.{VERSION_MINOR}]"));

fn avx512_real(ty: &str, m: usize, n: usize) -> Result<(String, String)> {
    let len;
    let sizeof;

    let smov;
    let pmov;
    let pmul;
    let pfma213;
    let pfma231;
    let pxor;
    let broadcast;
    let pgather;

    match ty {
        "f32" => {
            len = 16;
            sizeof = 4;

            pgather = "vgatherdps";
            smov = "vmovss";
            pmov = "vmovups";
            pmul = "vmulps";
            pfma213 = "vfmadd213ps";
            pfma231 = "vfmadd231ps";
            pxor = "vxorps";
            broadcast = "vbroadcastss";
        }
        "f64" => {
            len = 8;
            sizeof = 8;

            pgather = "vgatherqpd";
            smov = "vmovsd";
            pmov = "vmovupd";
            pmul = "vmulpd";
            pfma213 = "vfmadd213pd";
            pfma231 = "vfmadd231pd";
            pxor = "vxorpd";
            broadcast = "vbroadcastsd";
        }
        _ => panic!(),
    }

    let ctx = Ctx::new();
    setup!(ctx);

    let suffix = format!("[with m = {m * len}, n = {n}]");

    let lhs = rax;
    let packed_lhs = rbx;
    let rhs = rcx;
    let packed_rhs = rdx;
    let dst = rdi;
    let nrows = rsi;
    let info = rbp;
    let row = r14;
    let col = r15;

    let word = 8usize;
    let info_flags = 0 * word;
    let info_depth = 1 * word;
    let info_lhs_rs = 2 * word;
    let info_lhs_cs = 3 * word;
    let info_rhs_rs = 4 * word;
    let info_rhs_cs = 5 * word;
    let info_alpha = 8 * word;

    let dst_ptr = 0 * word;
    let dst_rs = 1 * word;
    let dst_cs = 2 * word;

    ctx[rsp].set(true);
    ctx[lhs].set(true);
    ctx[packed_lhs].set(true);
    ctx[rhs].set(true);
    ctx[packed_rhs].set(true);
    ctx[dst].set(true);
    ctx[nrows].set(true);
    ctx[info].set(true);
    ctx[row].set(true);
    ctx[col].set(true);

    let name;
    {
        let depth;
        let lhs_rs;
        let lhs_cs;
        let rhs_rs;
        let rhs_cs;

        {
            name = format!("\"{*PREFIX} gemm.microkernel.{ty}.simd512 {suffix}\"");

            asm!(".globl {name}");

            asm!(".align 16");
            asm!("{name}:");

            defer!({
                // {
                //     let scatter = 0;
                //     let add = true;
                //     defer!(asm!("ret"));

                //     reg!(ptr);
                //     reg!(rs);
                //     reg!(cs);

                //     asm!("mov {ptr}, [{dst} + {dst_ptr}]");

                //     asm!("mov {rs}, [{dst} + {dst_rs}]");
                //     asm!("imul {rs}, {row}");
                //     asm!("add {ptr}, {rs}");
                //     asm!("mov {rs}, [{dst} + {dst_rs}]");

                //     asm!("mov {cs}, [{dst} + {dst_cs}]");
                //     asm!("imul {cs}, {col}");
                //     asm!("add {ptr}, {cs}");
                //     asm!("mov {cs}, [{dst} + {dst_cs}]");

                //     asm!("{broadcast} zmm{m * n + m}, [{info} + {info_alpha}]");
                //     let alpha = format!("zmm{m * n + m}");
                //     for j in 0..n {
                //         for i in 0..m {
                //             let acc = format!("zmm{i + m * j}");

                //             if scatter == 0 || scatter == 1 && i + 1 < m {
                //                 if add {
                //                     asm!("{pfma213} {acc}, {alpha}, [{ptr} + {64 * i}]");
                //                 } else {
                //                     asm!("{pmul} {acc}, {acc}, {alpha}");
                //                 }
                //                 asm!("{pmov} [{ptr} + {64 * i}], {acc}");
                //             } else if scatter == 1 {
                //                 assert!(i + 1 == m);
                //                 if add {
                //                     asm!("{pmov} zmm31 {{{{k1}}}}{{{{z}}}}, [{ptr} + {64 * i}]");
                //                     asm!("{pfma213} {acc}, {alpha}, zmm31");
                //                 } else {
                //                     asm!("{pmul} {acc}, {acc}, {alpha}");
                //                 }
                //                 asm!("{pmov} [{ptr} + {64 * i}] {{{{k1}}}}, {acc}");
                //             } else {
                //                 panic!();
                //             }
                //         }
                //         asm!("add {ptr}, {cs}");
                //     }
                // }

                asm!("cmp {nrows}, {m * len}");
                asm!("jz 2f");
                asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.epilogue.mask.add {suffix}\"");

                asm!(".align 16");
                asm!("2:");
                asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.epilogue.store.add {suffix}\"");
                asm!("ud2");
            });

            asm!("push {lhs}");
            defer!(asm!("pop {lhs}"));

            asm!("push {rhs}");
            defer!(asm!("pop {rhs}"));

            asm!("push {packed_lhs}");
            defer!(asm!("pop {packed_lhs}"));

            asm!("push {packed_rhs}");
            defer!(asm!("pop {packed_rhs}"));

            reg!(&depth);
            reg!(&lhs_rs);
            reg!(&lhs_cs);
            reg!(&rhs_rs);
            reg!(&rhs_cs);

            asm!("mov {depth}, [{info} + {info_depth}]");
            asm!("mov {lhs_rs}, [{info} + {info_lhs_rs}]");
            asm!("mov {lhs_cs}, [{info} + {info_lhs_cs}]");
            asm!("mov {rhs_rs}, [{info} + {info_rhs_rs}]");
            asm!("mov {rhs_cs}, [{info} + {info_rhs_cs}]");

            for i in 0..m * n {
                asm!("{pxor} xmm{i}, xmm{i}, xmm{i}");
            }

            if m > 1 {
                asm!("sub {nrows}, {(m - 1) * len}");
            }
            {
                reg!(tmp);
                asm!("lea {tmp}, [rip + \"{*PREFIX} gemm.microkernel.{ty}.simd512.mask.data\"]");
                if ty == "f64" {
                    asm!("kmovb k1, [{tmp} + {nrows}]");
                } else {
                    asm!("kmovw k1, [{tmp} + {nrows} * 2]");
                }
            }
            if m > 1 {
                asm!("add {nrows}, {(m - 1) * len}");
            }

            asm!("cmp {lhs_rs}, {sizeof}");
            asm!("jz 2f");
            asm!("cmp {packed_lhs}, {lhs}");
            asm!("jz 3f");
            asm!("cmp {packed_rhs}, {rhs}");
            asm!("jz  \"{*PREFIX} gemm.microkernel.{ty}.simd512.gather.packA {suffix}\"");
            asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.gather.packA.packB {suffix}\"");

            asm!("3:");
            asm!("cmp {packed_rhs}, {rhs}");
            asm!("jz  \"{*PREFIX} gemm.microkernel.{ty}.simd512.gather {suffix}\"");
            asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.gather.packB {suffix}\"");

            asm!("2:");
            asm!("cmp {nrows}, {m * len}");
            asm!("jz 4f");

            asm!("test {lhs}, 63");
            asm!("jnz 5f");
            asm!("test {lhs_cs}, 63");
            asm!("jnz 5f");

            asm!("4:");
            asm!("cmp {packed_lhs}, {lhs}");
            asm!("jz 3f");
            asm!("cmp {packed_rhs}, {rhs}");
            asm!("jz  \"{*PREFIX} gemm.microkernel.{ty}.simd512.load.packA {suffix}\"");
            asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.load.packA.packB {suffix}\"");

            asm!("3:");
            asm!("cmp {packed_rhs}, {rhs}");
            asm!("jz  \"{*PREFIX} gemm.microkernel.{ty}.simd512.load {suffix}\"");
            asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.load.packB {suffix}\"");

            asm!("5:");

            asm!("cmp {packed_lhs}, {lhs}");

            asm!("jz 3f");
            asm!("cmp {packed_rhs}, {rhs}");
            asm!("jz  \"{*PREFIX} gemm.microkernel.{ty}.simd512.mask.packA {suffix}\"");
            asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.mask.packA.packB {suffix}\"");

            asm!("3:");
            asm!("cmp {packed_rhs}, {rhs}");
            asm!("jz  \"{*PREFIX} gemm.microkernel.{ty}.simd512.mask {suffix}\"");
            asm!("jmp \"{*PREFIX} gemm.microkernel.{ty}.simd512.mask.packB {suffix}\"");

            asm!("2314567:");
        }

        for gather_lhs in [0, 1, 2] {
            let __gather_lhs__ = if gather_lhs == 0 {
                ".gather"
            } else if gather_lhs == 1 {
                ".mask"
            } else {
                ".load"
            };

            for pack_lhs in [false, true] {
                let __pack_lhs__ = if pack_lhs { ".packA" } else { "" };

                for pack_rhs in [false, true] {
                    let __pack_rhs__ = if pack_rhs { ".packB" } else { "" };

                    let name = format!(
                        "\"{*PREFIX} gemm.microkernel.{ty}.simd512{__gather_lhs__}{__pack_lhs__}{__pack_rhs__} {suffix}\""
                    );
                    asm!(".globl {name}");
                    asm!(".align 16");
                    asm!("{name}:");
                    defer!(asm!("ret"));

                    label!(nanokernel);

                    if pack_lhs {
                        asm!("mov qword ptr [{info} + {info_lhs_rs}], {sizeof}");
                        asm!("mov qword ptr [{info} + {info_lhs_cs}], {64 * m}");
                    }

                    if gather_lhs == 0 {
                        asm!("vmovq xmm31, {lhs_rs}");
                        asm!("{broadcast} zmm31, xmm31");
                        let pmul = if ty == "f64" { "vpmuldq" } else { "vpmulld" };

                        asm!(
                            "{pmul} zmm31, zmm31, [rip + \"{*PREFIX} gemm.microkernel.{ty}.simd512.gather.data\"]"
                        );
                        asm!("imul {lhs_rs}, {len}");
                    }

                    for _ in 0..(n - 1) / 3 * 3 {
                        asm!("sub {rhs_rs}, {rhs_cs}");
                    }

                    asm!("test {depth}, {depth}");
                    asm!("jnz {nanokernel}f");
                    asm!("ud2");

                    asm!(".align 16");
                    asm!("{nanokernel}:");

                    let unroll = 1;
                    for _ in 0..unroll {
                        for j in 0..n {
                            if j % 3 == 0 {
                                asm!("{broadcast} zmm{m * n + m}, [{rhs}]");
                            } else {
                                asm!("{broadcast} zmm{m * n + m}, [{rhs} + {j % 3} * {rhs_cs}]");
                            }
                            if j + 1 < n && (j + 1) % 3 == 0 {
                                asm!("lea {rhs}, [{rhs} + 2 * {rhs_cs}]");
                                asm!("add {rhs}, {rhs_cs}");
                            }
                            if j + 1 == n {
                                asm!("add {rhs}, {rhs_rs}");
                            }

                            if pack_rhs {
                                asm!("{smov} [{packed_rhs} + {j * sizeof}], xmm{m * n + m}");
                                if j + 1 == n {
                                    asm!("add {packed_rhs}, {4 * sizeof}");
                                }
                            }

                            for i in 0..m {
                                if j == 0 {
                                    if gather_lhs == 0 {
                                        let x = if ty == "f64" { "b" } else { "w" };

                                        if i + 1 < m {
                                            asm!("knot{x} k2, k0");
                                            asm!(
                                                "{pgather} zmm{m * n + i} {{{{k2}}}}, [{lhs} + zmm31]"
                                            );
                                            asm!("add {lhs}, {lhs_rs}");
                                        } else {
                                            asm!("kmov{x} k2, k1");
                                            asm!(
                                                "{pgather} zmm{m * n + i} {{{{k2}}}}, [{lhs} + zmm31]"
                                            );
                                            for _ in 0..m - 1 {
                                                asm!("sub {lhs}, {lhs_rs}");
                                            }
                                        }
                                    } else if gather_lhs == 2 || i + 1 < m {
                                        asm!("{pmov} zmm{m * n + i}, [{lhs} + {64 * i}]");
                                    } else {
                                        asm!(
                                            "{pmov} zmm{m * n + i} {{{{k1}}}}{{{{z}}}}, [{lhs} + {64 * i}]"
                                        );
                                    }
                                }

                                asm!("{pfma231} zmm{i + m * j}, zmm{m * n + i}, zmm{m * n + m}");
                                if j == 0 {
                                    if pack_lhs {
                                        asm!("{pmov} [{packed_lhs} + {64 * i}], zmm{m * n + i}");
                                    }
                                }
                            }
                        }
                        asm!("add {lhs}, {lhs_cs}");
                        if pack_lhs {
                            asm!("add {packed_lhs}, {64 * m}");
                        }
                    }

                    // asm!("dec {depth}");
                    asm!("sub {depth}, {unroll}");
                    asm!("jnz {nanokernel}b");

                    asm!("jmp 2314567b");
                    asm!("ud2");
                }
            }
        }
    }

    for scatter in [0, 1] {
        let __scatter__ = if scatter == 0 {
            ".store"
        } else if scatter == 1 {
            ".mask"
        } else {
            ".scatter"
        };

        for add in [false, true] {
            let __add__ = if add { ".add" } else { ".overwrite" };

            asm!(
                ".globl \"{*PREFIX} gemm.microkernel.{ty}.simd512.epilogue{__scatter__}{__add__} {suffix}\""
            );
            asm!(".align 16");
            asm!(
                "\"{*PREFIX} gemm.microkernel.{ty}.simd512.epilogue{__scatter__}{__add__} {suffix}\":"
            );
            defer!(asm!("ret"));

            reg!(ptr);
            reg!(rs);
            reg!(cs);

            asm!("mov {ptr}, [{dst} + {dst_ptr}]");

            asm!("mov {rs}, [{dst} + {dst_rs}]");
            asm!("imul {rs}, {row}");
            asm!("add {ptr}, {rs}");
            asm!("mov {rs}, [{dst} + {dst_rs}]");

            asm!("mov {cs}, [{dst} + {dst_cs}]");
            asm!("imul {cs}, {col}");
            asm!("add {ptr}, {cs}");
            asm!("mov {cs}, [{dst} + {dst_cs}]");

            asm!("{broadcast} zmm{m * n + m}, [{info} + {info_alpha}]");
            let alpha = format!("zmm{m * n + m}");
            for j in 0..n {
                for i in 0..m {
                    let acc = format!("zmm{i + m * j}");

                    if scatter == 0 || scatter == 1 && i + 1 < m {
                        if add {
                            asm!("{pfma213} {acc}, {alpha}, [{ptr} + {64 * i}]");
                        } else {
                            asm!("{pmul} {acc}, {acc}, {alpha}");
                        }
                        asm!("{pmov} [{ptr} + {64 * i}], {acc}");
                    } else if scatter == 1 {
                        assert!(i + 1 == m);
                        if add {
                            asm!("{pmov} zmm31 {{{{k1}}}}{{{{z}}}}, [{ptr} + {64 * i}]");
                            asm!("{pfma213} {acc}, {alpha}, zmm31");
                        } else {
                            asm!("{pmul} {acc}, {acc}, {alpha}");
                        }
                        asm!("{pmov} [{ptr} + {64 * i}] {{{{k1}}}}, {acc}");
                    } else {
                        panic!();
                    }
                }
                asm!("add {ptr}, {cs}");
            }
        }
    }

    Ok((name, ctx.code.into_inner()))
}

fn main() -> Result {
    let mut names = vec![];
    let mut code = String::new();

    for n in 1..=4 {
        for m in 1..=6 {
            let (name, f) = avx512_real("f64", m, n)?;
            code += &*f;
            names.push(name);
        }
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("asm.s");
    fs::write(&dest_path, &code)?;

    {
        let dest_path = Path::new(&out_dir).join("asm.rs");

        let mut code = format!(
            "::core::arch::global_asm!{{ include_str!(concat!(env!(\"OUT_DIR\"), \"/asm.s\")) }}"
        );
        for (i, name) in names.iter().enumerate() {
            code += &format!(
                "
                unsafe extern \"C\" {{
                    #[link_name = {name}]
                    unsafe fn __decl_avx512_{i}__();
                }}
                "
            );
        }

        code += &format!("pub static SIMD512: [unsafe extern \"C\" fn(); {names.len()}] = [");
        for i in 0..names.len() {
            code += &format!("__decl_avx512_{i}__,");
        }
        code += "];";

        code += &format!(
            "
                #[unsafe(export_name = \"{*PREFIX} gemm.microkernel.f64.simd512.gather.data\")]
                pub static __GATHER_F64_512__: ::core::arch::x86_64::__m512i = unsafe {{
                    ::core::mem::transmute([0, 1, 2, 3, 4, 5, 6, 7u64])
                }};

                #[unsafe(export_name = \"{*PREFIX} gemm.microkernel.f32.simd512.gather.data\")]
                pub static __GATHER_F32_512__: ::core::arch::x86_64::__m512i = unsafe {{
                    ::core::mem::transmute([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15u32])
                }};

                #[unsafe(export_name = \"{*PREFIX} gemm.microkernel.f64.simd512.mask.data\")]
                pub static __MASK_F64_512__: [u8; 9] = [
                    0b00000000,
                    0b00000001,
                    0b00000011,
                    0b00000111,
                    0b00001111,
                    0b00011111,
                    0b00111111,
                    0b01111111,
                    0b11111111,
                ];

                #[unsafe(export_name = \"{*PREFIX} gemm.microkernel.f32.simd512.mask.data\")]
                pub static __MASK_F32_512__: [u16; 17] = [
                    0b0000000000000000,
                    0b0000000000000001,
                    0b0000000000000011,
                    0b0000000000000111,
                    0b0000000000001111,
                    0b0000000000011111,
                    0b0000000000111111,
                    0b0000000001111111,
                    0b0000000011111111,
                    0b0000000111111111,
                    0b0000001111111111,
                    0b0000011111111111,
                    0b0000111111111111,
                    0b0001111111111111,
                    0b0011111111111111,
                    0b0111111111111111,
                    0b1111111111111111,
                ];
            "
        );

        fs::write(&dest_path, &code)?;
    }

    println!("cargo::rerun-if-changed=build.rs");

    Ok(())
}
