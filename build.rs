#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_labels,
    unused_macros
)]

use std::env;
use std::fmt::Display;
use std::fs;
use std::ops::*;
use std::path::Path;
use std::sync::LazyLock;

use defer::defer;
use interpol::{format, println};
use std::cell::Cell;
use std::cell::RefCell;
use std::ops::Index;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! setup {
    ($ctx: ident $(,)?) => {
        macro_rules! ctx {
            () => {
                $ctx
            };
        }
    };

    ($ctx: ident, $target: ident $(,)?) => {
        macro_rules! ctx {
            () => {
                $ctx
            };
        }
        macro_rules! target {
            () => {
                $target
            };
        }
    };
}

macro_rules! align {
    () => {
        asm!(".align 16")
    };
}

macro_rules! func {
    ($name: tt) => {
        let __name__ = &format!($name);

        asm!(".globl {QUOTE}{__name__}{QUOTE}");
        align!();
        asm!("{QUOTE}{__name__}{QUOTE}:");
        defer!(asm!("ret"));

        macro_rules! name {
            () => {
                __name__
            };
        }
    };
}

macro_rules! asm {
    ($code: tt) => {{
        asm!($code, "");
    }};

    ($code: tt, $comment: tt) => {{
        use std::fmt::Write;

        let code = &mut *ctx!().code.borrow_mut();

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
        let $name = ctx!().reg(::std::stringify!($name));
        ::defer::defer!(ctx!().reg_drop($name, ::std::stringify!($name)));
    };

    (&$name: ident) => {
        $name = ctx!().reg(::std::stringify!($name));
        ::defer::defer!(ctx!().reg_drop($name, ::std::stringify!($name)));
    };
}

macro_rules! label {
    ($label: ident) => {
        const $label: &'static str = ::std::stringify!($label);
        {
            let name = ::std::stringify!($label);
            let func = name!();
            align!();
            asm!("{QUOTE}{name} @ {func}{QUOTE}:");
        }
    };
}

macro_rules! vxor {
    (zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {{
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vxor(dst, lhs, rhs)}"),
        }
    }};
}

macro_rules! vadd {
    (zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {{
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vadd_mem(dst, lhs, Addr::from(rhs))}"),
        }
    }};

    (zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {{
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vadd(dst, lhs, rhs)}"),
        }
    }};
}

macro_rules! vmov {
    ([$dst: expr][$mask: expr $(,)?], zmm($src: expr) $(,)?) => {{
        match ($dst, $mask, $src) {
            (dst, mask, src) => asm!("{target!().vstoremask(mask, Addr::from(dst), src)}"),
        }
    }};
    ([$dst: expr], zmm($src: expr) $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().vstore(Addr::from(dst), src)}"),
        }
    }};
    (zmm($dst: expr)[$mask: expr $(,)?], [$src: expr] $(,)?) => {{
        match ($dst, $mask, $src) {
            (dst, mask, src) => asm!("{target!().vloadmask(mask, dst, Addr::from(src))}"),
        }
    }};
    (zmm($dst: expr), [$src: expr] $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().vload(dst, Addr::from(src))}"),
        }
    }};

    (zmm($dst: expr), zmm($src: expr) $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().vmov(dst, src)}"),
        }
    }};
}

macro_rules! vswap {
    (zmm($src: expr) $(,)?) => {{
        match $src {
            src => asm!("{target!().vswap(src)}"),
        }
    }};
}

macro_rules! kmov {
    (k($dst: expr), [$src: expr] $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{ target!().kload(dst, Addr::from(src)) }"),
        }
    }};
}

macro_rules! vmul {
    (zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vmul(dst, lhs, rhs)}"),
        }
    };
}

macro_rules! vfma231 {
    (zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vfma231_mem(dst, lhs, Addr::from(rhs))}"),
        }
    };

    (zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vfma231(dst, lhs, rhs)}"),
        }
    };
}

macro_rules! vfma231_conj {
    (zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vfma231_conj_mem(dst, lhs, Addr::from(rhs))}"),
        }
    };

    (zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vfma231_conj(dst, lhs, rhs)}"),
        }
    };
}

macro_rules! vfma213 {
    (zmm($dst: expr), zmm($lhs: expr), zmm($rhs: expr) $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vfma213(dst, lhs, rhs)}"),
        }
    };
    (zmm($dst: expr), zmm($lhs: expr), [$rhs: expr] $(,)?) => {
        match ($dst, $lhs, $rhs) {
            (dst, lhs, rhs) => asm!("{target!().vfma213_mem(dst, lhs, Addr::from(rhs))}"),
        }
    };
}

macro_rules! vbroadcast {
    (zmm($dst: expr), [$src: expr] $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().real().vbroadcast(dst, Addr::from(src))}"),
        }
    }};
}

macro_rules! vmovs {
    ([$dst: expr], xmm($src: expr) $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().scalar().vstore(Addr::from(dst), src)}"),
        }
    }};
    (xmm($dst: expr), [$src: expr] $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().scalar().vload(dst, Addr::from(src))}"),
        }
    }};
}

macro_rules! vmovsr {
    ([$dst: expr], xmm($src: expr) $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().real().scalar().vstore(Addr::from(dst), src)}"),
        }
    }};
    (xmm($dst: expr), [$src: expr] $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("{target!().real().scalar().vload(dst, Addr::from(src))}"),
        }
    }};
}

macro_rules! alloca {
    ($reg: expr) => {
        let __reg__ = $reg;
        asm!("push {__reg__}");
        defer!(asm!("pop {__reg__}"));
    };
}

macro_rules! cmovz {
    ($dst: expr, $src: expr $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("cmovz {dst}, {src}"),
        }
    }};
}

macro_rules! mov {
    ([$dst: expr], $src: expr $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("mov qword ptr { Addr::from(dst) }, {src}"),
        }
    }};

    ($dst: expr, [$src: expr] $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("mov {dst}, { Addr::from(src) }"),
        }
    }};

    ($dst: expr, $src: expr $(,)?) => {{
        match ($dst, $src) {
            (dst, src) => asm!("mov {dst}, {src}"),
        }
    }};
}

macro_rules! cmp {
    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("cmp {lhs}, {rhs}"),
        }
    }};
}

macro_rules! test {
    ([$lhs: expr], $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("test qword ptr {Addr::from(lhs)}, {rhs}"),
        }
    }};

    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("test {lhs}, {rhs}"),
        }
    }};
}

macro_rules! bt {
    ([$lhs: expr], $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("bt qword ptr {Addr::from(lhs)}, {rhs}"),
        }
    }};

    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("bt {lhs}, {rhs}"),
        }
    }};
}

macro_rules! add {
    ([$lhs: expr], $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("add qword ptr {Addr::from(lhs)}, {rhs}"),
        }
    }};

    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("add {lhs}, {rhs}"),
        }
    }};
}

macro_rules! shl {
    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("shl {lhs}, {rhs}"),
        }
    }};
}

macro_rules! shr {
    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("shr {lhs}, {rhs}"),
        }
    }};
}

macro_rules! imul {
    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("imul {lhs}, {rhs}"),
        }
    }};
}

macro_rules! neg {
    ($inout: expr $(,)?) => {{
        match $inout {
            inout => asm!("neg {inout}"),
        }
    }};
}

macro_rules! dec {
    ($inout: expr $(,)?) => {{
        match $inout {
            inout => asm!("dec {inout}"),
        }
    }};
}

macro_rules! sub {
    ($lhs: expr, $rhs: expr $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("sub {lhs}, {rhs}"),
        }
    }};
}

macro_rules! lea {
    ($lhs: expr, [$rhs: expr] $(,)?) => {{
        match ($lhs, $rhs) {
            (lhs, rhs) => asm!("lea {lhs}, { Addr::from(rhs) }"),
        }
    }};
}

macro_rules! jmp {
    ($label: ident) => {
        let name = $label;
        let func = name!();
        asm!("jmp {QUOTE}{name} @ {func}{QUOTE}");
    };
    ($($label: tt)*) => {
        match format!($($label)*) {
            label => asm!("jmp {QUOTE}{label}{QUOTE}"),
        }
    };
}

macro_rules! call {
    ($($label: tt)*) => {
        match format!($($label)*) {
            label => asm!("call {QUOTE}{label}{QUOTE}"),
        }
    };
}

macro_rules! jnz {
    ($label: ident) => {
        let name = $label;
        let func = name!();
        asm!("jnz {QUOTE}{name} @ {func}{QUOTE}");
    };
}

macro_rules! jz {
    ($label: ident) => {
        let name = $label;
        let func = name!();
        asm!("jz {QUOTE}{name} @ {func}{QUOTE}");
    };
}

macro_rules! jnc {
    ($label: ident) => {
        let name = $label;
        let func = name!();
        asm!("jnc {QUOTE}{name} @ {func}{QUOTE}");
    };
}

macro_rules! jc {
    ($label: ident) => {
        let name = $label;
        let func = name!();
        asm!("jc {QUOTE}{name} @ {func}{QUOTE}");
    };
}

macro_rules! brk {
    () => {
        asm!("int3");
    };
}

macro_rules! abort {
    () => {
        asm!("ud2");
    };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Reg {
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
    rip = 16,
}

#[derive(Copy, Clone, Debug)]
struct IndexScale {
    index: Reg,
    scale: isize,
}

#[derive(Copy, Clone, Debug)]
struct PtrIndexScale {
    ptr: Reg,
    index: Reg,
    scale: isize,
}

#[derive(Copy, Clone, Debug)]
struct PtrOffset {
    ptr: Reg,
    offset: isize,
}

#[derive(Copy, Clone, Debug)]
struct PtrStatic<'a> {
    ptr: Reg,
    offset: &'a str,
}

#[derive(Copy, Clone, Debug)]
struct Addr<'a> {
    ptr: Reg,
    index: Reg,
    scale: isize,
    offset: isize,
    static_offset: Option<&'a str>,
}

impl Mul<isize> for Reg {
    type Output = IndexScale;

    fn mul(self, rhs: isize) -> Self::Output {
        IndexScale {
            index: self,
            scale: rhs,
        }
    }
}

impl Mul<Reg> for isize {
    type Output = IndexScale;

    fn mul(self, rhs: Reg) -> Self::Output {
        rhs * self
    }
}

impl Add<isize> for Reg {
    type Output = PtrOffset;

    fn add(self, rhs: isize) -> Self::Output {
        PtrOffset {
            ptr: self,
            offset: rhs,
        }
    }
}

impl<'a> Add<&'a str> for Reg {
    type Output = PtrStatic<'a>;

    fn add(self, rhs: &'a str) -> Self::Output {
        PtrStatic {
            ptr: self,
            offset: rhs,
        }
    }
}

impl<'a> Add<&'a String> for Reg {
    type Output = PtrStatic<'a>;

    fn add(self, rhs: &'a String) -> Self::Output {
        self + &**rhs
    }
}

impl Add<IndexScale> for Reg {
    type Output = PtrIndexScale;

    fn add(self, rhs: IndexScale) -> Self::Output {
        PtrIndexScale {
            ptr: self,
            index: rhs.index,
            scale: rhs.scale,
        }
    }
}

impl Add<isize> for PtrIndexScale {
    type Output = Addr<'static>;

    fn add(self, rhs: isize) -> Self::Output {
        Addr {
            ptr: self.ptr,
            index: self.index,
            scale: self.scale,
            offset: rhs,
            static_offset: None,
        }
    }
}

impl From<Reg> for Addr<'_> {
    fn from(value: Reg) -> Self {
        Self {
            ptr: value,
            index: rsp,
            scale: 0,
            offset: 0,
            static_offset: None,
        }
    }
}

impl From<PtrIndexScale> for Addr<'_> {
    fn from(value: PtrIndexScale) -> Self {
        Self {
            ptr: value.ptr,
            index: value.index,
            scale: value.scale,
            offset: 0,
            static_offset: None,
        }
    }
}

impl From<PtrOffset> for Addr<'_> {
    fn from(value: PtrOffset) -> Self {
        Self {
            ptr: value.ptr,
            index: rsp,
            scale: 0,
            offset: value.offset,
            static_offset: None,
        }
    }
}

impl<'a> From<PtrStatic<'a>> for Addr<'a> {
    fn from(value: PtrStatic<'a>) -> Self {
        Self {
            ptr: value.ptr,
            index: rsp,
            scale: 0,
            offset: 0,
            static_offset: Some(value.offset),
        }
    }
}

use Reg::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Float {
    F32,
    C32,
    F64,
    C64,
}

impl Reg {
    const ALL: &[Self] = &[
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
    fn sizeof(self) -> isize {
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

type Code = RefCell<String>;

struct Ctx {
    reg_busy: [Cell<bool>; 16],
    label: Cell<usize>,
    code: Code,
}

impl Ctx {
    fn new() -> Self {
        Self {
            reg_busy: [const { Cell::new(false) }; 16],
            label: Cell::new(2),
            code: RefCell::new(String::new()),
        }
    }

    #[track_caller]
    fn reg(&self, name: &str) -> Reg {
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

    fn reg_drop(&self, reg: Reg, name: &str) {
        setup!(self);

        self[reg].set(false);
        asm!("pop {reg}", "restore after reg dealloc `{name}`");
    }

    fn label(&self, name: &str) -> usize {
        let _ = name;
        let label = self.label.get();
        assert!(label < 10);
        self.label.set(label + 1);
        label
    }

    fn label_drop(&self, label: usize, name: &str) {
        let _ = name;
        self.label.set(label);
    }
}

const VERSION_MAJOR: usize = 0;
const VERSION_MINOR: usize = 22;
const PREFIX: LazyLock<String> =
    LazyLock::new(|| format!("[faer v{VERSION_MAJOR}.{VERSION_MINOR}]"));
const WORD: isize = 8;
const QUOTE: char = '"';

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Simd {
    _512,
    _256,
    _128,
    _64,
    _32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Ty {
    F32,
    F64,
    C32,
    C64,
}

#[derive(Copy, Clone, Debug)]
struct Target {
    ty: Ty,
    simd: Simd,
}

impl Simd {
    fn dedicated_mask(self) -> bool {
        matches!(self, Simd::_512)
    }

    fn reg(self) -> String {
        match self {
            Simd::_512 => "zmm",
            Simd::_256 => "ymm",
            _ => "xmm",
        }
        .to_string()
    }

    fn sizeof(self) -> isize {
        match self {
            Simd::_512 => 512 / 8,
            Simd::_256 => 256 / 8,
            Simd::_128 => 128 / 8,
            Simd::_64 => 64 / 8,
            Simd::_32 => 32 / 8,
        }
    }
}

impl Ty {
    fn sizeof(self) -> isize {
        match self {
            Ty::F32 => 4,
            Ty::F64 => 8,
            Ty::C32 => 2 * 4,
            Ty::C64 => 2 * 8,
        }
    }

    fn suffix(self) -> String {
        match self {
            Ty::F32 | Ty::C32 => "s",
            Ty::F64 | Ty::C64 => "d",
        }
        .to_string()
    }
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match *self {
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::C32 => "c32",
            Ty::C64 => "c64",
        })
    }
}

impl Display for Addr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let &Self {
            ptr,
            index,
            scale,
            offset,
            static_offset,
        } = self;

        let mut out = format!("{ptr}");

        if scale > 0 {
            assert!((scale as usize).is_power_of_two());
            assert!(scale <= 8);
            if scale == 1 {
                out = format!("{out} + {index}");
            } else {
                out = format!("{out} + {scale} * {index}");
            }
        }

        if offset > 0 {
            out = format!("{out} + {offset}");
        }
        if offset < 0 {
            out = format!("{out} - {-offset}");
        }

        if let Some(offset) = static_offset {
            assert_eq!(ptr, rip);
            out = format!("{out} + {QUOTE}{offset}{QUOTE}");
        }

        write!(f, "[{out}]")
    }
}

impl Target {
    fn load_imp(self, mask: Option<isize>, dst: isize, src: Addr) -> String {
        let Self { ty, simd } = self;

        let reg = simd.reg();

        let instr = match simd {
            Simd::_32 | Simd::_64 => format!("vmovs{ty.suffix()}"),
            _ => {
                if mask.is_none() || simd.dedicated_mask() {
                    format!("vmovup{self.ty.suffix()}")
                } else {
                    format!("vmaskmovp{self.ty.suffix()}")
                }
            }
        };

        match (mask, simd) {
            (None, _) => format!("{instr} {reg}{dst}, {src}"),
            (Some(mask), Simd::_512) => {
                format!("{instr} {reg}{dst} {{{{k{mask}}}}}{{{{z}}}}, {src}")
            }
            (Some(mask), _) => {
                format!("{instr} {reg}{dst}, {reg}{mask}, {src}")
            }
        }
    }

    fn store_imp(self, mask: Option<isize>, dst: Addr, src: isize) -> String {
        let Self { ty, simd } = self;

        let reg = simd.reg();

        let instr = match simd {
            Simd::_32 | Simd::_64 => format!("vmovs{ty.suffix()}"),
            _ => {
                if mask.is_none() || simd.dedicated_mask() {
                    format!("vmovup{self.ty.suffix()}")
                } else {
                    format!("vmaskmovp{self.ty.suffix()}")
                }
            }
        };

        match (mask, simd) {
            (None, _) => format!("{instr} {dst}, {reg}{src}"),
            (Some(mask), Simd::_512) => {
                format!("{instr} {dst} {{{{k{mask}}}}}, {reg}{src}")
            }
            (Some(mask), _) => {
                format!("{instr} {dst}, {reg}{mask}, {reg}{src}")
            }
        }
    }

    fn is_scalar(self) -> bool {
        matches!(
            (self.ty, self.simd),
            (Ty::F32, Simd::_32) | (Ty::F64, Simd::_64),
        )
    }

    fn is_cplx(self) -> bool {
        matches!(self.ty, Ty::C32 | Ty::C64)
    }

    fn is_real(self) -> bool {
        !self.is_cplx()
    }

    fn real(self) -> Target {
        let ty = match self.ty {
            Ty::F32 | Ty::C32 => Ty::F32,
            Ty::F64 | Ty::C64 => Ty::F64,
        };
        Target {
            ty,
            simd: self.simd,
        }
    }

    fn mask_sizeof(self) -> isize {
        if self.simd.dedicated_mask() {
            if self.is_cplx() {
                2 * self.len() / 8
            } else {
                self.len() / 8
            }
        } else {
            self.simd.sizeof()
        }
    }

    fn len(self) -> isize {
        self.simd.sizeof() / self.ty.sizeof()
    }

    fn scalar_suffix(self) -> String {
        if !self.is_cplx() && self.is_scalar() {
            "s"
        } else {
            "p"
        }
        .to_string()
    }

    fn vswap(self, reg: isize) -> String {
        let bits = match self.ty {
            Ty::C64 => 0b01010101,
            Ty::C32 => 0b10110001,
            _ => panic!(),
        };
        format!("vpermilp{self.ty.suffix()} {self.simd.reg()}{reg}, {self.simd.reg()}{reg}, {bits}")
    }
    fn vmov(self, dst: isize, src: isize) -> String {
        format!("vmovaps {self.simd.reg()}{dst}, {self.simd.reg()}{src}")
    }

    fn vload(self, dst: isize, src: Addr) -> String {
        self.load_imp(None, dst, src)
    }
    fn vstore(self, dst: Addr, src: isize) -> String {
        self.store_imp(None, dst, src)
    }

    fn kload(self, dst: isize, src: Addr) -> String {
        if self.simd.dedicated_mask() {
            match self.ty {
                Ty::F32 | Ty::C32 => format!("kmovw k{dst}, {src}"),
                Ty::F64 | Ty::C64 => format!("kmovb k{dst}, {src}"),
            }
        } else {
            self.vload(dst, src)
        }
    }

    fn scalar(self) -> Self {
        Self {
            ty: self.ty,
            simd: match self.ty {
                Ty::F32 => Simd::_32,
                Ty::F64 => Simd::_64,
                Ty::C32 => Simd::_64,
                Ty::C64 => Simd::_128,
            },
        }
    }

    fn vloadmask(self, mask: isize, dst: isize, src: Addr) -> String {
        self.load_imp(Some(mask), dst, src)
    }

    fn vstoremask(self, mask: isize, dst: Addr, src: isize) -> String {
        self.store_imp(Some(mask), dst, src)
    }

    fn vxor(self, dst: isize, lhs: isize, rhs: isize) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        format!("vxorp{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
    }

    fn vadd(self, dst: isize, lhs: isize, rhs: isize) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        format!("vadd{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
    }

    fn vadd_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        format!("vadd{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs}")
    }

    fn vmul(self, dst: isize, lhs: isize, rhs: isize) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        format!("vmul{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
    }

    fn vfma231(self, dst: isize, lhs: isize, rhs: isize) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        if self.is_cplx() {
            format!(
                "vfmaddsub231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}"
            )
        } else {
            format!(
                "vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}"
            )
        }
    }

    fn vfma231_conj(self, dst: isize, lhs: isize, rhs: isize) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        if self.is_cplx() {
            format!(
                "vfmsubadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}"
            )
        } else {
            format!(
                "vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}"
            )
        }
    }

    fn vfma231_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        if self.is_cplx() {
            format!(
                "vfmaddsub231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}"
            )
        } else {
            format!(
                "vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}"
            )
        }
    }

    fn vfma231_conj_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();

        if self.is_cplx() {
            format!(
                "vfmsubadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}"
            )
        } else {
            format!(
                "vfmadd231{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs} {{{{1to{self.real().len()}}}}}"
            )
        }
    }

    fn vfma213(self, dst: isize, lhs: isize, rhs: isize) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        format!("vfmadd213{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {reg}{rhs}")
    }

    fn vfma213_mem(self, dst: isize, lhs: isize, rhs: Addr) -> String {
        let Self { ty, simd } = self;
        let reg = simd.reg();
        format!("vfmadd213{self.scalar_suffix()}{ty.suffix()} {reg}{dst}, {reg}{lhs}, {rhs}")
    }

    fn vbroadcast(self, dst: isize, src: Addr) -> String {
        let instr = if self.is_scalar() {
            return self.vload(dst, src);
        } else if (self.ty, self.simd) == (Ty::F64, Simd::_128) {
            format!("vmovddup")
        } else {
            format!("vbroadcasts{self.ty.suffix()}")
        };

        format!("{instr} {self.simd.reg()}{dst}, {src}")
    }

    fn microkernel(self, m: isize, n: isize) -> (String, String) {
        let Self { ty, simd } = self;
        let bits = simd.sizeof() * 8;
        let need_mask = m * self.len() > 2;

        let ctx = Ctx::new();
        setup!(ctx, self);

        let suffix = format!("[with m = {m * self.len()}, n = {n}]");
        let lhs = rax;
        let packed_lhs = rbx;
        let rhs = rcx;
        let packed_rhs = rdx;
        let dst = rdi;
        let info = rsi;
        let nrows = r8;
        let ncols = r9;

        let info_flags = 0 * WORD;
        let info_depth = 1 * WORD;
        let info_lhs_rs = 2 * WORD;
        let info_lhs_cs = 3 * WORD;
        let info_rhs_rs = 4 * WORD;
        let info_rhs_cs = 5 * WORD;
        let info_row = 6 * WORD;
        let info_col = 7 * WORD;
        let info_alpha = 8 * WORD;

        let dst_ptr = 0 * WORD;
        let dst_rs = 1 * WORD;
        let dst_cs = 2 * WORD;

        ctx[rsp].set(true);
        ctx[lhs].set(true);
        ctx[packed_lhs].set(true);
        ctx[rhs].set(true);
        ctx[packed_rhs].set(true);
        ctx[dst].set(true);
        ctx[nrows].set(true);
        ctx[ncols].set(true);
        ctx[info].set(true);

        let prefix = format!("{*PREFIX} gemm.microkernel.{ty}.simd{bits}");

        let mask_ptr;
        let main = {
            let depth;
            let lhs_rs;
            let lhs_cs;
            let rhs_rs;
            let rhs_cs;

            let main = {
                func!("{prefix} {suffix}");
                label!(row_check);
                if m > 1 {
                    cmp!(nrows, (m - 1) * self.len() + 1);
                    jnc!(col_check);
                    jmp!("{prefix} [with m = {(m - 1) * self.len()}, n = {n}]");
                }
                label!(col_check);
                if n > 1 {
                    cmp!(ncols, n);
                    jnc!(prologue);
                    jmp!("{prefix} [with m = {m * self.len()}, n = {n - 1}]");
                }
                label!(prologue);

                alloca!(lhs);
                alloca!(rhs);
                alloca!(packed_lhs);
                alloca!(packed_rhs);

                reg!(&lhs_rs);
                reg!(&lhs_cs);
                reg!(&rhs_rs);
                reg!(&rhs_cs);
                reg!(&mask_ptr);

                mov!(lhs_rs, [info + info_lhs_rs]);
                mov!(lhs_cs, [info + info_lhs_cs]);
                mov!(rhs_rs, [info + info_rhs_rs]);
                mov!(rhs_cs, [info + info_rhs_cs]);

                {
                    reg!(tmp);

                    test!(lhs_rs, lhs_rs);

                    mov!(tmp, m * simd.sizeof());
                    cmovz!(lhs_cs, tmp);
                    mov!(tmp, ty.sizeof());
                    cmovz!(lhs_rs, tmp);

                    test!(rhs_cs, rhs_cs);
                    cmovz!(rhs_cs, tmp);
                    mov!(tmp, n * ty.sizeof());
                    cmovz!(rhs_rs, tmp);
                }
                for i in 0..m * n {
                    vxor!(zmm(i), zmm(i), zmm(i));
                }

                if need_mask {
                    sub!(nrows, m * self.len());

                    jnc!(no_mask);

                    {
                        reg!(tmp);
                        lea!(tmp, [rip + &format!("{prefix}.mask.data")]);

                        if self.mask_sizeof() <= 8 {
                            lea!(
                                mask_ptr,
                                [tmp + nrows * self.mask_sizeof()
                                    + self.len() * self.mask_sizeof()]
                            );
                        } else {
                            shl!(nrows, self.mask_sizeof().ilog2());
                            lea!(
                                mask_ptr,
                                [tmp + nrows * 1 + self.len() * self.mask_sizeof()]
                            );
                            shr!(nrows, self.mask_sizeof().ilog2());
                        }
                    }
                    label!(no_mask);

                    add!(nrows, m * self.len());
                }

                reg!(&depth);
                mov!(depth, [info + info_depth]);

                cmp!(lhs_rs, ty.sizeof());
                jz!(colmajor);
                cmp!(lhs_cs, ty.sizeof());
                jz!(rowmajor);

                label!(strided);
                {
                    abort!();
                }

                label!(rowmajor);
                {
                    abort!();
                }
                label!(colmajor);
                {
                    cmp!(nrows, m * self.len());
                    jnc!(load);

                    if need_mask {
                        test!(lhs, simd.sizeof() - 1);
                        jnz!(mask);

                        test!(lhs_cs, simd.sizeof() - 1);
                        jnz!(mask);
                    } else {
                        abort!();
                    }
                }

                label!(load);
                {
                    cmp!(packed_lhs, lhs);
                    jnz!(load_A);
                    cmp!(packed_rhs, rhs);
                    jnz!(load_noA_B);

                    label!(load_noA_noB);
                    {
                        call!("{prefix}.load {suffix}");
                        jmp!(epilogue);
                    }
                    label!(load_A);
                    {
                        cmp!(packed_rhs, rhs);
                        jnz!(load_A_B);
                    }

                    label!(load_A_noB);
                    {
                        call!("{prefix}.load.packA {suffix}");
                        jmp!(epilogue);
                    }
                    label!(load_noA_B);
                    {
                        call!("{prefix}.load.packB {suffix}");
                        jmp!(epilogue);
                    }
                    label!(load_A_B);
                    {
                        call!("{prefix}.load.packA.packB {suffix}");
                        jmp!(epilogue);
                    }
                }

                label!(mask);
                {
                    if need_mask {
                        cmp!(packed_lhs, lhs);
                        jnz!(mask_A);
                        cmp!(packed_rhs, rhs);
                        jnz!(mask_noA_B);

                        label!(mask_noA_noB);
                        {
                            call!("{prefix}.mask {suffix}");
                            jmp!(epilogue);
                        }
                        label!(mask_A);
                        {
                            cmp!(packed_rhs, rhs);
                            jnz!(mask_A_B);
                        }

                        label!(mask_A_noB);
                        {
                            call!("{prefix}.mask.packA {suffix}");
                            jmp!(epilogue);
                        }
                        label!(mask_noA_B);
                        {
                            call!("{prefix}.mask.packB {suffix}");
                            jmp!(epilogue);
                        }
                        label!(mask_A_B);
                        {
                            call!("{prefix}.mask.packA.packB {suffix}");
                            jmp!(epilogue);
                        }
                    }
                }
                label!(epilogue);

                cmp!(nrows, m * self.len());
                jnc!(epilogue_load);

                label!(epilogue_mask);
                {
                    if need_mask {
                        test!([info + info_flags], 1);
                        jz!(epilogue_mask_overwrite);

                        label!(epilogue_mask_add);
                        call!("{prefix}.epilogue.mask.add {suffix}");
                        jmp!(end);

                        label!(epilogue_mask_overwrite);
                        call!("{prefix}.epilogue.mask.overwrite {suffix}");
                        jmp!(end);
                    } else {
                        abort!();
                    }
                }

                label!(epilogue_load);
                {
                    test!([info + info_flags], 1);
                    jz!(epilogue_store_overwrite);

                    label!(epilogue_store_add);
                    call!("{prefix}.epilogue.store.add {suffix}");
                    jmp!(end);

                    label!(epilogue_store_overwrite);
                    call!("{prefix}.epilogue.store.overwrite {suffix}");
                    jmp!(end);
                }

                label!(end);

                name!().clone()
            };

            ctx[depth].set(true);
            ctx[rhs_cs].set(true);
            ctx[rhs_rs].set(true);
            ctx[lhs_cs].set(true);
            ctx[lhs_rs].set(true);
            ctx[mask_ptr].set(true);

            for mask in if need_mask {
                vec![false, true]
            } else {
                vec![false]
            } {
                let __mask__ = if mask { ".mask" } else { ".load" };

                for pack_lhs in [false, true] {
                    let __pack_lhs__ = if pack_lhs { ".packA" } else { "" };

                    for pack_rhs in [false, true] {
                        let __pack_rhs__ = if pack_rhs { ".packB" } else { "" };

                        for conj in if self.is_cplx() {
                            vec![false, true]
                        } else {
                            vec![false]
                        } {
                            let __conj__ = if conj { ".conj" } else { "" };

                            func!(
                                "{prefix}{__conj__}{__mask__}{__pack_lhs__}{__pack_rhs__} {suffix}"
                            );
                            if self.is_cplx() && !conj {
                                bt!([info + info_flags], 2);
                                jnc!(start);
                                jmp!(
                                    "{prefix}.conj{__mask__}{__pack_lhs__}{__pack_rhs__} {suffix}"
                                );
                            }
                            label!(start);

                            // if pack_lhs {
                            //     mov!([info + info_lhs_rs], ty.sizeof());
                            //     mov!([info + info_lhs_cs], simd.sizeof() * m);
                            // }

                            let rhs_neg_cs = lhs_rs;

                            mov!(rhs_neg_cs, rhs_cs);
                            neg!(rhs_neg_cs);
                            add!(rhs, rhs_cs);

                            if mask && simd.dedicated_mask() {
                                kmov!(k(1), [mask_ptr]);
                            }

                            label!(nanokernel);
                            let unroll = 1;
                            let bcst = bits == 512 && m == 1 && !pack_rhs;

                            for _ in 0..unroll {
                                for j in 0..n {
                                    let rhs_addr = if j % 4 == 0 {
                                        rhs + 1 * rhs_neg_cs
                                    } else {
                                        rhs + (j % 4 - 1) * rhs_cs
                                    };

                                    if !bcst {
                                        vbroadcast!(zmm(m * n + m), [rhs_addr]);
                                        if self.is_real() && n > 4 {
                                            if j + 1 == 4 {
                                                lea!(rhs, [rhs + rhs_cs * 4]);
                                            }
                                        }

                                        if self.is_real() && j + 1 == n {
                                            if n > 4 {
                                                lea!(rhs, [rhs + rhs_neg_cs * 4]);
                                            }
                                            add!(rhs, rhs_rs);
                                        }
                                    }

                                    if pack_rhs {
                                        vmovsr!([packed_rhs + j * ty.sizeof()], xmm(m * n + m));
                                        if self.is_real() && j + 1 == n {
                                            add!(packed_rhs, n * ty.sizeof());
                                        }
                                    }

                                    for i in 0..m {
                                        if j == 0 {
                                            if !mask || i + 1 < m {
                                                vmov!(zmm(m * n + i), [lhs + simd.sizeof() * i]);
                                            } else {
                                                if simd.dedicated_mask() {
                                                    vmov!(
                                                        zmm(m * n + i)[1],
                                                        [lhs + simd.sizeof() * i]
                                                    );
                                                } else {
                                                    vmov!(zmm(m * n + i), [mask_ptr]);
                                                    vmov!(
                                                        zmm(m * n + i)[m * n + i],
                                                        [lhs + simd.sizeof() * i]
                                                    );
                                                }
                                            }
                                        }
                                        if bcst {
                                            if conj {
                                                vfma231_conj!(
                                                    zmm(m * j + i),
                                                    zmm(m * n + i),
                                                    [rhs_addr],
                                                );
                                            } else {
                                                vfma231!(
                                                    zmm(m * j + i),
                                                    zmm(m * n + i),
                                                    [rhs_addr],
                                                );
                                            }

                                            if self.is_real() && n > 4 {
                                                if j + 1 == 4 {
                                                    lea!(rhs, [rhs + rhs_cs * 4]);
                                                }
                                            }

                                            if self.is_real() && j + 1 == n {
                                                if n > 4 {
                                                    lea!(rhs, [rhs + rhs_neg_cs * 4]);
                                                }
                                                add!(rhs, rhs_rs);
                                            }
                                        } else {
                                            if conj {
                                                vfma231_conj!(
                                                    zmm(m * j + i),
                                                    zmm(m * n + i),
                                                    zmm(m * n + m),
                                                );
                                            } else {
                                                vfma231!(
                                                    zmm(m * j + i),
                                                    zmm(m * n + i),
                                                    zmm(m * n + m),
                                                );
                                            }
                                        }

                                        if j == 0 && pack_lhs {
                                            vmov!([packed_lhs + simd.sizeof() * i], zmm(m * n + i));
                                        }
                                    }

                                    if j == 0 {
                                        add!(lhs, lhs_cs);
                                        if pack_lhs {
                                            add!(packed_lhs, simd.sizeof() * m);
                                        }
                                    }
                                }

                                if self.is_cplx() {
                                    for j in 0..n {
                                        let rhs_addr = if j % 4 == 0 {
                                            rhs + 1 * rhs_neg_cs + ty.sizeof() / 2
                                        } else {
                                            rhs + (j % 4 - 1) * rhs_cs + ty.sizeof() / 2
                                        };

                                        if !bcst {
                                            vbroadcast!(zmm(m * n + m), [rhs_addr]);
                                            if n > 4 {
                                                if (j + 1) % 4 == 0 {
                                                    lea!(rhs, [rhs + rhs_cs * 4]);
                                                }
                                            }

                                            if j + 1 == n {
                                                if n > 4 {
                                                    lea!(rhs, [rhs + rhs_neg_cs * 4]);
                                                }
                                                add!(rhs, rhs_rs);
                                            }
                                        }

                                        if pack_rhs {
                                            vmovsr!(
                                                [packed_rhs + (j * ty.sizeof() + ty.sizeof() / 2)],
                                                xmm(m * n + m)
                                            );
                                            if j + 1 == n {
                                                add!(packed_rhs, n * ty.sizeof());
                                            }
                                        }

                                        for i in 0..m {
                                            if j == 0 {
                                                vswap!(zmm(m * n + i));
                                            }
                                            if bcst {
                                                if conj {
                                                    vfma231_conj!(
                                                        zmm(m * j + i),
                                                        zmm(m * n + i),
                                                        [rhs_addr],
                                                    );
                                                } else {
                                                    vfma231!(
                                                        zmm(m * j + i),
                                                        zmm(m * n + i),
                                                        [rhs_addr],
                                                    );
                                                }

                                                if n > 4 {
                                                    if j + 1 == 4 {
                                                        lea!(rhs, [rhs + rhs_cs * 4]);
                                                    }
                                                }

                                                if j + 1 == n {
                                                    if n > 4 {
                                                        lea!(rhs, [rhs + rhs_neg_cs * 4]);
                                                    }
                                                    add!(rhs, rhs_rs);
                                                }
                                            } else {
                                                if conj {
                                                    vfma231_conj!(
                                                        zmm(m * j + i),
                                                        zmm(m * n + i),
                                                        zmm(m * n + m),
                                                    );
                                                } else {
                                                    vfma231!(
                                                        zmm(m * j + i),
                                                        zmm(m * n + i),
                                                        zmm(m * n + m),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if unroll == 1 {
                                dec!(depth);
                            } else {
                                sub!(depth, unroll);
                            }
                            jnz!(nanokernel);
                        }
                    }
                }
            }

            ctx[depth].set(false);
            ctx[rhs_cs].set(false);
            ctx[rhs_rs].set(false);
            ctx[lhs_cs].set(false);
            ctx[lhs_rs].set(false);

            main
        };

        for mask in if need_mask {
            vec![false, true]
        } else {
            vec![false]
        } {
            let __mask__ = if mask { ".mask" } else { ".store" };

            for add in [false, true] {
                let __add__ = if add { ".add" } else { ".overwrite" };

                func!("{prefix}.epilogue{__mask__}{__add__} {suffix}");
                reg!(ptr);
                reg!(rs);
                reg!(cs);
                mov!(ptr, [dst + dst_ptr]);
                mov!(rs, [dst + dst_rs]);
                mov!(cs, [dst + dst_cs]);

                {
                    reg!(row);
                    reg!(col);
                    mov!(row, [info + info_row]);
                    mov!(col, [info + info_col]);

                    imul!(col, cs);
                    add!(ptr, col);

                    imul!(row, rs);
                    add!(ptr, row);

                    {
                        let alpha_ptr = row;
                        let alpha_re = m * n;
                        let alpha_im = m * n + 1;
                        let mask_ = if simd.dedicated_mask() { 1 } else { m * n + 2 };
                        let tmp = m * n + 3;

                        if self.is_cplx() {
                            vmov!(
                                zmm(alpha_re),
                                [rip + &format!("{*PREFIX} gemm.microkernel.{ty}.flip.re.data")]
                            );
                            vmov!(
                                zmm(alpha_im),
                                [rip + &format!("{*PREFIX} gemm.microkernel.{ty}.flip.im.data")]
                            );
                            bt!([info + info_flags], 1);
                            jnc!(no_conj_lhs);

                            label!(conj_lhs);
                            {
                                bt!([info + info_flags], 2);
                                jc!(conj_lhs_no_conj_rhs);

                                label!(conj_lhs_conj_rhs);
                                {
                                    vxor!(zmm(alpha_re), zmm(alpha_re), zmm(alpha_im));
                                    jmp!(xor);
                                }
                                label!(conj_lhs_no_conj_rhs);
                                {
                                    vxor!(zmm(alpha_re), zmm(alpha_re), zmm(alpha_re));
                                    jmp!(xor);
                                }
                            }

                            label!(no_conj_lhs);
                            {
                                bt!([info + info_flags], 2);
                                jnc!(no_conj_lhs_no_conj_rhs);

                                label!(no_conj_lhs_conj_rhs);
                                {
                                    vmov!(zmm(alpha_re), zmm(alpha_im));
                                    jmp!(xor);
                                }
                                label!(no_conj_lhs_no_conj_rhs);
                                {}
                            }
                            label!(xor);
                            for i in 0..m * n {
                                vxor!(zmm(i), zmm(i), zmm(alpha_re));
                            }
                        }

                        mov!(alpha_ptr, [info + info_alpha]);
                        vbroadcast!(zmm(alpha_re), [alpha_ptr]);
                        if self.is_cplx() {
                            vbroadcast!(zmm(alpha_im), [alpha_ptr + ty.sizeof() / 2]);
                        }

                        if mask {
                            kmov!(k(mask_), [mask_ptr]);
                        }

                        for j in 0..n {
                            for i in 0..m {
                                let src = m * j + i;
                                let ptr = ptr + simd.sizeof() * i;

                                if self.is_cplx() || !add {
                                    if self.is_cplx() {
                                        vswap!(zmm(src));
                                        vmul!(zmm(tmp), zmm(src), zmm(alpha_im));
                                        vswap!(zmm(src));
                                        vfma231!(zmm(tmp), zmm(src), zmm(alpha_re));
                                        vmov!(zmm(src), zmm(tmp));
                                    } else {
                                        // vmul!(zmm(src), zmm(src), zmm(alpha_re));
                                    }
                                };

                                if !mask || i + 1 < m {
                                    if add {
                                        if self.is_cplx() {
                                            vadd!(zmm(src), zmm(alpha_re), [ptr]);
                                        } else {
                                            vfma213!(zmm(src), zmm(alpha_re), [ptr]);
                                        }
                                    }
                                    vmov!([ptr], zmm(src));
                                } else {
                                    if add {
                                        vmov!(zmm(tmp)[mask_], [ptr]);

                                        if self.is_cplx() {
                                            vadd!(zmm(src), zmm(src), zmm(tmp));
                                        } else {
                                            vfma213!(zmm(src), zmm(alpha_re), zmm(tmp));
                                        }
                                    }
                                    vmov!([ptr][mask_], zmm(src));
                                }
                            }
                            add!(ptr, cs);
                        }
                    }
                }
                bt!([rsi], 63);
                jc!(rowmajor);

                label!(colmajor);
                {
                    if mask {
                        add!([info + info_row], nrows);
                        add!([info + info_col], n);

                        mov!(nrows, 0);
                        sub!(ncols, n);
                    } else {
                        add!([info + info_row], m * self.len());
                        sub!(nrows, m * self.len());
                        jnz!(end);

                        sub!(ncols, n);
                        add!([info + info_col], n);
                    }
                }
                jmp!(end);

                label!(rowmajor);
                {
                    add!([info + info_col], n);
                    sub!(ncols, n);

                    jnz!(end);

                    if mask {
                        add!([info + info_row], nrows);
                        mov!(nrows, 0);
                    } else {
                        add!([info + info_row], m * self.len());
                        sub!(nrows, m * self.len());
                    }
                }

                label!(end);
            }
        }

        (main, ctx.code.into_inner())
    }
}

fn main() -> Result {
    let mut code = String::new();

    let mut f32_simd512 = vec![];
    let mut c32_simd512 = vec![];
    let mut f64_simd512 = vec![];
    let mut c64_simd512 = vec![];

    let mut f32_simd512x8 = vec![];
    let mut c32_simd512x8 = vec![];
    let mut f64_simd512x8 = vec![];
    let mut c64_simd512x8 = vec![];

    let mut f32_simd256 = vec![];
    let mut c32_simd256 = vec![];
    let mut f64_simd256 = vec![];
    let mut c64_simd256 = vec![];

    let mut f32_simd128 = vec![];
    let mut c32_simd128 = vec![];
    let mut f64_simd128 = vec![];
    let mut c64_simd128 = vec![];

    let mut f32_simd64 = vec![];
    let mut c32_simd64 = vec![];
    let mut f64_simd64 = vec![];

    let mut f32_simd32 = vec![];

    for (out, ty) in [
        (&mut f32_simd512, Ty::F32),
        (&mut c32_simd512, Ty::C32),
        (&mut f64_simd512, Ty::F64),
        (&mut c64_simd512, Ty::C64),
    ] {
        for m in (1..=6).rev() {
            let last = if m == 1 { 8 } else { 4 };

            for n in 1..=last {
                let target = Target {
                    ty,
                    simd: Simd::_512,
                };

                let (name, f) = target.microkernel(m, n);
                code += &f;
                out.push(name);
            }
        }
    }

    for (out, ty) in [
        (&mut f32_simd512x8, Ty::F32),
        (&mut c32_simd512x8, Ty::C32),
        (&mut f64_simd512x8, Ty::F64),
        (&mut c64_simd512x8, Ty::C64),
    ] {
        for m in (1..=3).rev() {
            for n in 1..=8 {
                let target = Target {
                    ty,
                    simd: Simd::_512,
                };

                let (name, f) = target.microkernel(m, n);
                if m > 1 && n > 4 {
                    code += &f;
                }
                out.push(name);
            }
        }
    }

    for (out, ty) in [
        (&mut f32_simd256, Ty::F32),
        (&mut c32_simd256, Ty::C32),
        (&mut f64_simd256, Ty::F64),
        (&mut c64_simd256, Ty::C64),
    ] {
        for m in (1..=3).rev() {
            let last = if m == 1 { 8 } else { 4 };

            for n in 1..=last {
                let target = Target {
                    ty,
                    simd: Simd::_256,
                };

                let (name, f) = target.microkernel(m, n);
                code += &f;
                out.push(name);
            }
        }
    }

    for (out, ty, simd) in [
        (&mut f32_simd128, Ty::F32, Simd::_128),
        (&mut c32_simd128, Ty::C32, Simd::_128),
        (&mut f64_simd128, Ty::F64, Simd::_128),
        (&mut c64_simd128, Ty::C64, Simd::_128),
        (&mut f32_simd64, Ty::F32, Simd::_64),
        (&mut c32_simd64, Ty::C32, Simd::_64),
        (&mut f64_simd64, Ty::F64, Simd::_64),
        (&mut f32_simd32, Ty::F32, Simd::_32),
    ] {
        for m in (1..=1).rev() {
            for n in 1..=8 {
                let target = Target { ty, simd };

                let (name, f) = target.microkernel(m, n);
                code += &f;
                out.push(name);
            }
        }
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("asm.s");
    fs::write(&dest_path, &code)?;

    {
        let dest_path = Path::new(&out_dir).join("asm.rs");

        let mut code = format!(
            "::core::arch::global_asm!{{ include_str!(concat!(env!({QUOTE}OUT_DIR{QUOTE}), {QUOTE}/asm.s{QUOTE})) }}"
        );

        for (names, ty, bits) in [
            (&f32_simd512x8, Ty::F32, "512x8"),
            (&c32_simd512x8, Ty::C32, "512x8"),
            (&f64_simd512x8, Ty::F64, "512x8"),
            (&c64_simd512x8, Ty::C64, "512x8"),
            (&f32_simd512, Ty::F32, "512x4"),
            (&c32_simd512, Ty::C32, "512x4"),
            (&f64_simd512, Ty::F64, "512x4"),
            (&c64_simd512, Ty::C64, "512x4"),
            (&f32_simd256, Ty::F32, "256"),
            (&c32_simd256, Ty::C32, "256"),
            (&f64_simd256, Ty::F64, "256"),
            (&c64_simd256, Ty::C64, "256"),
            (&f32_simd128, Ty::F32, "128"),
            (&c32_simd128, Ty::C32, "128"),
            (&f64_simd128, Ty::F64, "128"),
            (&c64_simd128, Ty::C64, "128"),
            (&f32_simd64, Ty::F32, "64"),
            (&c32_simd64, Ty::C32, "64"),
            (&f64_simd64, Ty::F64, "64"),
        ] {
            for (i, name) in names.iter().enumerate() {
                code += &format!(
                    "
                unsafe extern {QUOTE}C{QUOTE} {{
                    #[link_name = {QUOTE}{name}{QUOTE}]
                    unsafe fn __decl_{ty}_simd{bits}_{i}__();
                }}
                "
                );
            }

            let upper = format!("{ty}").to_uppercase();
            code += &format!(
                "pub static {upper}_SIMD{bits}: [unsafe extern {QUOTE}C{QUOTE} fn(); {names.len()}] = ["
            );
            for i in 0..names.len() {
                code += &format!("__decl_{ty}_simd{bits}_{i}__,");
            }
            code += "];";
        }

        code += &format!(
            "
                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.f32.simd128.mask.data{QUOTE})]
                 static __MASK_F32_128__: [::core::arch::x86_64::__m128i; 5] = unsafe {{::core::mem::transmute([
                    [0, 0, 0, 0i32],
                    [0, 0, 0, -1],
                    [0, 0, -1, -1],
                    [0, -1, -1, -1],
                    [-1, -1, -1, -1],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.f64.simd256.mask.data{QUOTE})]
                 static __MASK_F64_256__: [::core::arch::x86_64::__m256i; 5] = unsafe {{::core::mem::transmute([
                    [0, 0, 0, 0i64],
                    [0, 0, 0, -1],
                    [0, 0, -1, -1],
                    [0, -1, -1, -1],
                    [-1, -1, -1, -1],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.f64.simd512.mask.data{QUOTE})]
                 static __MASK_F64_512__: [u8; 9] = [
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

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c64.simd256.mask.data{QUOTE})]
                 static __MASK_C64_256__: [::core::arch::x86_64::__m256i; 3] = unsafe {{::core::mem::transmute([
                    [0, 0, 0, 0i64],
                    [0, 0, -1, -1],
                    [-1, -1, -1, -1],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c64.simd512.mask.data{QUOTE})]
                 static __MASK_C64_512__: [u8; 5] = [
                    0b00000000,
                    0b00000011,
                    0b00001111,
                    0b00111111,
                    0b11111111,
                ];

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.f32.simd256.mask.data{QUOTE})]
                 static __MASK_F32_256__: [::core::arch::x86_64::__m256i; 9] = unsafe {{::core::mem::transmute([
                    [0, 0, 0, 0, 0, 0, 0, 0i32],
                    [0, 0, 0, 0,0, 0, 0, -1],
                    [0, 0, 0, 0,0, 0, -1, -1],
                    [0, 0, 0, 0,0, -1, -1, -1],
                    [0, 0, 0, 0,-1, -1, -1, -1],
                    [0, 0, 0, -1, -1, -1, -1, -1],
                    [0, 0, -1, -1, -1, -1, -1, -1],
                    [0, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.f32.simd512.mask.data{QUOTE})]
                 static __MASK_F32_512__: [u16; 17] = [
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

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c32.simd256.mask.data{QUOTE})]
                 static __MASK_C32_256__: [::core::arch::x86_64::__m256i; 5] = unsafe {{::core::mem::transmute([
                    [0, 0, 0, 0, 0, 0, 0, 0i32],
                    [0, 0, 0, 0,0, 0, -1, -1],
                    [0, 0, 0, 0,-1, -1, -1, -1],
                    [0, 0, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c32.simd512.mask.data{QUOTE})]
                 static __MASK_C32_512__: [u16; 9] = [
                    0b0000000000000000,
                    0b0000000000000011,
                    0b0000000000001111,
                    0b0000000000111111,
                    0b0000000011111111,
                    0b0000001111111111,
                    0b0000111111111111,
                    0b0011111111111111,
                    0b1111111111111111,
                ];

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c32.flip.re.data{QUOTE})]
                 static __FLIP_RE_C32__: ::core::arch::x86_64::__m512i = unsafe {{::core::mem::transmute([
                    [i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c64.flip.re.data{QUOTE})]
                 static __FLIP_RE_C64__: ::core::arch::x86_64::__m512i = unsafe {{::core::mem::transmute([
                    [i64::MIN, 0, i64::MIN, 0, i64::MIN, 0, i64::MIN, 0],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c32.flip.im.data{QUOTE})]
                 static __FLIP_IM_C32__: ::core::arch::x86_64::__m512i = unsafe {{::core::mem::transmute([
                    [0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN, 0, i32::MIN],
                ])}};

                #[unsafe(export_name = {QUOTE}{*PREFIX} gemm.microkernel.c64.flip.im.data{QUOTE})]
                 static __FLIP_IM_C64__: ::core::arch::x86_64::__m512i = unsafe {{::core::mem::transmute([
                    [0, i64::MIN, 0, i64::MIN, 0, i64::MIN, 0, i64::MIN],
                ])}};
            "
        );

        fs::write(&dest_path, &code)?;
    }

    println!("cargo::rerun-if-changed=build.rs");

    Ok(())
}
