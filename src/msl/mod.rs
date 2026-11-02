// Metal backend (probe): compiles MIR into a Metal Shading Language kernel, one GPU thread per
// evaluation point. Apple GPUs have no f64 (MSL has no `double`): values are f32, or, with
// mixed precision (`MslOptions::mixed`), df64 pairs of f32 where the denominators need them.
//
// The source is meant to be compiled by the Metal framework at run time
// (`newLibraryWithSource`, with `MTLMathModeSafe`: no fast math, no contraction), which needs
// no Xcode. `sjb2msl` writes it from a saved symjit application; `run.m` is a host program
// that runs a list of kernels on one arena. See docs/METAL.md.
//
// Kernel interface (SoA or AoS as in `crate::ptx`):
//   kernel void <name>(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
//                      device const float *params [[buffer(2)]], constant uint &n [[buffer(3)]],
//                      constant uint &ld_in [[buffer(4)]], constant uint &ld_out [[buffer(5)]],
//                      device const float *in_lo [[buffer(6)]], uint i [[thread_position_in_grid]])
// With a plane mapping (`MslOptions::mapping`, pacman's arenas), one arena instead:
//   kernel void <name>(device float *arena [[buffer(0)]], constant uint &n [[buffer(1)]],
//                      constant uint &ld [[buffer(2)]], device const float *lo [[buffer(3)]],
//                      uint i [[thread_position_in_grid]])
// slot k of the inputs at plane map[k], output j at plane map[count_in + j]. `in_lo` and `lo`
// (same layout as `in` and the arena) hold the low parts of the df64 inputs of mixed precision:
// such an input x is (in[..], in_lo[..]) = (f32(x), f32(x - f32(x))). Without mixed precision
// they are not read. `MslOptions::lo_first` = f: `lo` holds the planes from f on only (plane p
// at lo[(p - f) * ld + i]; the df64 inputs must be planes >= f).
//
// The probe covers straight-line code without math-library calls: what pyAmpliCol's plane
// kernels need. Their complex square roots and divisions are lowered to funclets that are
// straight-line themselves (selects, no jumps): each call is replaced by the funclet's body
// (`inline_funclets`); any other label or jump is an error. The generator records operations
// (`Op`), not text: after inlining, the body is in SSA form once each definition is a value
// (`typing`), and mixed precision types the values. Every value is a local (the Metal compiler
// allocates registers): a `float`, or for df64 a `float2` (hi, lo) of the same name with a `d`
// prefix; masks are floats of all bits set.
//
// f32 arithmetic follows the PTX backend's f32 mode: one rounding per operation, `fma` only for
// the MIR's fused operations, sign-bit `neg`/`abs`/`sign`, unordered `gt`/`geq`/`neq`. Unlike
// the CPU, the GPU flushes f32 denormals to zero. Mixed precision is `ptx::mixed`'s `need` rule
// with df64 in place of f64: divisions and reciprocals are df64, and so is everything a divisor
// or a reciprocal's operand depends on (its backward slice); the inputs `f64_inputs` marks are
// df64 (momenta, parameters), every other operation f32. df64 has 48 bits of significand and
// f32's exponent range (`DF64`: the formulas of `recurrence_runner.cu`).

mod typing;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::fmt::Write;

use anyhow::{anyhow, Result};

use symjit::backend::{Func, Generator, GeneratorType, Loc, Mir, Reg};
use symjit::Program;

use crate::ptx::{Layout, Shape};
use crate::sjb::SavedModel;

use typing::Ty;

/// The host program that runs kernels of `sjb2msl` (`cc -fobjc-arc run.m -framework Metal
/// -framework Foundation`; usage at its top).
pub const RUN_M: &str = include_str!("run.m");

#[derive(Clone, Debug)]
pub struct MslOptions {
    /// the kernel's name (a C identifier)
    pub name: String,
    pub layout: Layout,
    /// planes of one arena, as `PtxOptions::mapping`
    pub mapping: Option<Vec<u32>>,
    /// mixed precision: per input slot, whether it is a df64 input (missing: f32)
    pub mixed: Option<Vec<bool>>,
    /// with a mapping: the first plane of `lo` (the df64 inputs' planes are `lo_first` on)
    pub lo_first: u32,
}

impl MslOptions {
    pub fn new(name: &str) -> MslOptions {
        MslOptions {
            name: name.to_string(),
            layout: Layout::Soa,
            mapping: None,
            mixed: None,
            lo_first: 0,
        }
    }
}

/// What the mixed-precision typing did: value operations by type (moves excluded).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MslCounts {
    pub df64_ops: usize,
    pub f32_ops: usize,
}

/// A Metal kernel's source and the counts of its interface.
pub struct Msl {
    /// the kernel alone (`kernel void ...`), to put after `prelude()` (several may share it)
    pub kernel: String,
    pub shape: Shape,
    pub counts: MslCounts,
}

impl Msl {
    /// A complete source file of this kernel.
    pub fn source(&self) -> String {
        format!("{}\n{}", prelude(), self.kernel)
    }
}

/// df64 arithmetic on (hi, lo) pairs of f32 (Dekker, Knuth; `recurrence_runner.cu`'s `add` and
/// `mul`). Needs IEEE f32 without contraction or reassociation (`MTLMathModeSafe`) and a real
/// `fma`.
const DF64: &str = "\
static inline float2 two_sum(float a, float b) {
\tfloat s = a + b, bb = s - a;
\treturn float2(s, (a - (s - bb)) + (b - bb));
}
static inline float2 fast_two_sum(float a, float b) {
\tfloat s = a + b;
\treturn float2(s, b - (s - a));
}
static inline float2 dd_add(float2 x, float2 y) {
\tfloat2 s = two_sum(x.x, y.x), t = two_sum(x.y, y.y);
\tfloat2 v = fast_two_sum(s.x, s.y + t.x);
\treturn fast_two_sum(v.x, t.y + v.y);
}
static inline float2 dd_mul(float2 x, float2 y) {
\tfloat p = x.x * y.x;
\tfloat e = fma(x.x, y.x, -p);
\te = fma(x.x, y.y, fma(x.y, y.x, e));
\treturn fast_two_sum(p, e);
}
static inline float2 dd_div(float2 x, float2 y) {
\tfloat q1 = x.x / y.x;
\tfloat2 r = dd_add(x, -dd_mul(float2(q1, 0.0f), y));
\tfloat q2 = r.x / y.x;
\tr = dd_add(r, -dd_mul(float2(q2, 0.0f), y));
\tfloat q3 = r.x / y.x;
\treturn dd_add(fast_two_sum(q1, q2), float2(q3, 0.0f));
}
static inline float2 dd_sqrt(float2 x) {
\tif (!(x.x > 0.0f)) return float2(sqrt(x.x), 0.0f);
\tfloat s = sqrt(x.x);
\tfloat2 r = dd_add(x, -dd_mul(float2(s, 0.0f), float2(s, 0.0f)));
\treturn fast_two_sum(s, r.x / (2.0f * s));
}
static inline bool dd_lt(float2 a, float2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
static inline bool dd_le(float2 a, float2 b) { return a.x < b.x || (a.x == b.x && a.y <= b.y); }
static inline bool dd_eq(float2 a, float2 b) { return a.x == b.x && a.y == b.y; }
static inline float2 dd_bits(uint2 u) { return as_type<float2>(u); }
static inline uint2 dd_ubits(float2 x) { return as_type<uint2>(x); }
";

/// The start of a source file: the header and the helpers kernels use.
pub fn prelude() -> String {
    format!(
        "// Generated by sjb2msl {} from saved symjit applications.\n\
         #include <metal_stdlib>\n\
         using namespace metal;\n\
         \n\
         static inline float bits(uint u) {{ return as_type<float>(u); }}\n\
         static inline uint ubits(float x) {{ return as_type<uint>(x); }}\n\
         static inline float mask(bool c) {{ return bits(c ? 0xFFFFFFFFu : 0u); }}\n\
         {}",
        env!("CARGO_PKG_VERSION"),
        DF64
    )
}

impl SavedModel {
    /// The Metal kernel of the model (see the top of this file).
    pub fn msl(&mut self, options: &MslOptions) -> Result<Msl> {
        msl_of(&mut self.prog, &self.mir, options)
    }
}

pub(crate) fn msl_of(prog: &mut Program, mir: &Mir, options: &MslOptions) -> Result<Msl> {
    if crate::ptx::libm::is_library_name(&options.name) {
        return Err(anyhow!("the kernel name `{}` is reserved", options.name));
    }
    let row_first = prog.config().symbolica() && !prog.config().direct_arena();
    let shape = Shape {
        count_in: if row_first {
            prog.count_params
        } else {
            prog.count_states
        },
        count_out: prog.count_obs + prog.count_diffs,
        count_params: if row_first { 0 } else { prog.count_params },
        row_first,
    };
    if let Some(map) = &options.mapping {
        if map.len() != shape.count_in + shape.count_out {
            return Err(anyhow!(
                "the plane mapping has {} planes, the kernel {} inputs and {} outputs",
                map.len(),
                shape.count_in,
                shape.count_out
            ));
        }
        if options.layout != Layout::Soa || row_first || shape.count_params != 0 {
            return Err(anyhow!(
                "a plane mapping needs the SoA layout and a plane kernel without shared parameters"
            ));
        }
    }
    let mut generator = MslGenerator::new(options.clone(), shape);
    let (count_states, count_obs, count_params) =
        (prog.count_states, prog.count_obs, prog.count_params);
    prog.builder.compile_from_mir(
        mir,
        &mut generator,
        count_states,
        count_obs,
        count_params,
    )?;
    generator.check()?;
    Ok(Msl {
        kernel: generator.kernel,
        shape,
        counts: generator.counts,
    })
}

/// x86's predicates: `Gtu`/`Geu`/`Neu` are true if unordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmp {
    Gtu,
    Geu,
    Lt,
    Le,
    Eq,
    Neu,
}

/// An operation `dst := op(args)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Mov,
    Zero,
    Neg,
    Abs,
    /// the sign bit (a mask-like value: -0 or +0)
    Sign,
    Sqrt,
    Recip,
    Half,
    Rint,
    Floor,
    Ceil,
    Trunc,
    Frac,
    Add,
    Sub,
    Mul,
    Div,
    /// ±(a b) ± c, one rounding
    Fma {
        neg_product: bool,
        neg_add: bool,
    },
    Cmp(Cmp),
    And,
    Or,
    Xor,
    /// !a & b
    AndNot,
    Not,
    /// args: true value, false value, mask
    Select,
}

#[derive(Clone, Debug)]
enum Item {
    Op(String, Op, Vec<String>),
    /// `<var> = consts[idx]`; the constants arrive after the code (`add_consts`)
    Const(String, u32),
    /// `<var> = parameter idx` (an input of a Symbolica model, or a shared parameter)
    LoadParam(String, u32),
    LoadMem(String, u32),
    SaveMem(String, u32),
    LoadStack(String, u32),
    SaveStack(String, u32),
    Label(String),
    Branch(String),
    CallFunclet(String),
    Ret,
}

pub struct MslGenerator {
    options: MslOptions,
    shape: Shape,
    body: Vec<Item>,
    consts: Vec<f64>,
    count_gen: u32,
    count_static: u32,
    count_stack: u32,
    count_mem: u32,
    count_states: u32,
    unsupported: Vec<String>,
    kernel: String,
    counts: MslCounts,
}

/// Where the memory slots live, as `ptx::Placement` for straight-line code: inputs never
/// written are read where used, outputs never read are written where saved (or where the
/// stack slot they copy is written: `forward`); the others are locals, gathered and scattered.
struct Placement {
    direct_in: Vec<bool>,
    direct_out: Vec<bool>,
    forward: HashMap<usize, Vec<usize>>,
    forwarded: Vec<bool>,
}

/// The scratch variables (`x0`..`x3`) and the registers.
const SCRATCH: usize = 4;

impl MslGenerator {
    pub fn new(options: MslOptions, shape: Shape) -> MslGenerator {
        MslGenerator {
            options,
            shape,
            body: Vec::new(),
            consts: Vec::new(),
            count_gen: 0,
            count_static: 0,
            count_stack: 0,
            count_mem: 0,
            count_states: 0,
            unsupported: Vec::new(),
            kernel: String::new(),
            counts: MslCounts::default(),
        }
    }

    pub fn check(&self) -> Result<()> {
        if self.unsupported.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(
                "the Metal backend does not support {} (yet)",
                self.unsupported.join(", ")
            ))
        }
    }

    fn unsupported(&mut self, what: &str) {
        if !self.unsupported.iter().any(|s| s == what) {
            self.unsupported.push(what.to_string());
        }
    }

    fn reg(&mut self, r: Reg) -> String {
        match r {
            Reg::Ret | Reg::Left => "r".into(),
            Reg::Temp | Reg::Right => "t".into(),
            Reg::Gen(n) => {
                self.count_gen = self.count_gen.max(n as u32 + 1);
                format!("g{}", n)
            }
            Reg::Static(n) => {
                self.count_static = self.count_static.max(n + 1);
                format!("v{}", n)
            }
        }
    }

    fn op(&mut self, dst: &str, op: Op, args: &[&str]) {
        self.body.push(Item::Op(
            dst.to_string(),
            op,
            args.iter().map(|a| a.to_string()).collect(),
        ));
    }

    fn unary(&mut self, op: Op, dst: Reg, s1: Reg) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        self.op(&d, op, &[&a]);
    }

    fn binary(&mut self, op: Op, dst: Reg, s1: Reg, s2: Reg) {
        let (d, a, b) = (self.reg(dst), self.reg(s1), self.reg(s2));
        self.op(&d, op, &[&a, &b]);
    }

    fn fused(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg, neg_product: bool, neg_add: bool) {
        let (d, a, b, c) = (self.reg(dst), self.reg(s1), self.reg(s2), self.reg(s3));
        self.op(
            &d,
            Op::Fma {
                neg_product,
                neg_add,
            },
            &[&a, &b, &c],
        );
    }

    fn load_loc(&mut self, dst: Reg, loc: Loc) {
        let d = self.reg(dst);
        match loc {
            Loc::Mem(idx) => {
                self.count_mem = self.count_mem.max(idx + 1);
                self.body.push(Item::LoadMem(d, idx));
            }
            Loc::Stack(idx) => {
                self.count_stack = self.count_stack.max(idx + 1);
                self.body.push(Item::LoadStack(d, idx));
            }
            Loc::Param(idx) => self.body.push(Item::LoadParam(d, idx)),
        }
    }

    fn save_loc(&mut self, src: Reg, loc: Loc) {
        let s = self.reg(src);
        match loc {
            Loc::Mem(idx) => {
                self.count_mem = self.count_mem.max(idx + 1);
                self.body.push(Item::SaveMem(s, idx));
            }
            Loc::Stack(idx) => {
                self.count_stack = self.count_stack.max(idx + 1);
                self.body.push(Item::SaveStack(s, idx));
            }
            Loc::Param(_) => self.unsupported("stores to parameters"),
        }
    }

    fn first_out(&self) -> usize {
        if self.shape.row_first {
            0
        } else {
            self.count_states as usize
        }
    }

    /// The input slots (memory slots of the states, or parameters of a Symbolica model).
    fn count_inputs(&self) -> usize {
        if self.shape.row_first {
            self.shape.count_in
        } else {
            self.count_states as usize
        }
    }

    /// Input slot `k` is a df64 input.
    fn f64_input(&self, k: usize) -> bool {
        self.options
            .mixed
            .as_ref()
            .is_some_and(|m| m.get(k).copied().unwrap_or(false))
    }

    /// Slot `k` of the point in `pin` (an input), `pout` (an output) or `plo` (the low part of
    /// an input).
    fn slot(&self, base: &str, k: usize) -> String {
        let k = match &self.options.mapping {
            Some(map) if base == "pout" => map[self.shape.count_in + k] as usize,
            Some(map) if base == "plo" => (map[k] - self.options.lo_first) as usize,
            Some(map) => map[k] as usize,
            None => k,
        };
        let ld = if base == "pout" { "ld_out" } else { "ld_in" };
        match self.options.layout {
            Layout::Soa if k == 0 => format!("{}[0]", base),
            Layout::Soa => format!("{}[{}u * {}]", base, k, ld),
            Layout::Aos => format!("{}[{}]", base, k),
        }
    }

    /// Input slot `k` as a value of type `ty`.
    fn input(&self, k: usize, ty: Ty) -> String {
        if ty == Ty::D64 && self.f64_input(k) {
            format!("float2({}, {})", self.slot("pin", k), self.slot("plo", k))
        } else if ty == Ty::D64 {
            // an f32 input in a df64 slice: exact, no low part
            format!("float2({}, 0.0f)", self.slot("pin", k))
        } else {
            self.slot("pin", k)
        }
    }

    /// Replaces each funclet call by the funclet's body, if every funclet is straight-line,
    /// and drops the jumps over the funclets; what is left must be straight-line.
    fn inline_funclets(&mut self) {
        let targets: Vec<String> = self
            .body
            .iter()
            .filter_map(|item| match item {
                Item::CallFunclet(l) => Some(l.clone()),
                _ => None,
            })
            .collect();
        let mut funclets: HashMap<String, Vec<Item>> = HashMap::new();
        let mut main: Vec<Item> = Vec::new();
        let mut items = std::mem::take(&mut self.body).into_iter();
        while let Some(item) = items.next() {
            match item {
                Item::Label(l) if targets.contains(&l) => {
                    let mut body = Vec::new();
                    loop {
                        match items.next() {
                            Some(Item::Ret) => break,
                            Some(Item::Label(_) | Item::Branch(_) | Item::CallFunclet(_)) => {
                                self.unsupported("funclets with control flow");
                                return;
                            }
                            Some(other) => body.push(other),
                            None => {
                                self.unsupported("a funclet without a return");
                                return;
                            }
                        }
                    }
                    funclets.insert(l, body);
                }
                other => main.push(other),
            }
        }
        let mut body = Vec::new();
        let mut k = 0;
        while k < main.len() {
            match &main[k] {
                // a jump over funclets, which are gone
                Item::Branch(l) if matches!(main.get(k + 1), Some(Item::Label(m)) if m == l) => {
                    k += 2;
                    continue;
                }
                Item::CallFunclet(l) => match funclets.get(l) {
                    Some(f) => body.extend(f.iter().cloned()),
                    None => self.unsupported("a call of a missing funclet"),
                },
                Item::Label(_) | Item::Branch(_) => {
                    self.unsupported("control flow (labels, jumps)")
                }
                Item::Ret => self.unsupported("a return outside a funclet"),
                other => body.push(other.clone()),
            }
            k += 1;
        }
        self.body = body;
    }

    fn placement(&self) -> Placement {
        let n = self.count_mem as usize;
        let (mut loaded, mut saved) = (vec![false; n], vec![false; n]);
        for item in self.body.iter() {
            match item {
                Item::LoadMem(_, idx) => loaded[*idx as usize] = true,
                Item::SaveMem(_, idx) => saved[*idx as usize] = true,
                _ => {}
            }
        }
        let states = if self.shape.row_first {
            0
        } else {
            self.count_states as usize
        };
        let first = self.first_out();
        let outs = first..first + self.shape.count_out;
        let direct_out: Vec<bool> = (0..n)
            .map(|k| outs.contains(&k) && saved[k] && !loaded[k])
            .collect();

        // forwarding (see `ptx::Placement::forward`): an output saved once from a register
        // just loaded from a stack slot is written where that slot is last written
        let mut saves = vec![0usize; n];
        for item in self.body.iter() {
            if let Item::SaveMem(_, idx) = item {
                saves[*idx as usize] += 1;
            }
        }
        let mut forward: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut forwarded = vec![false; self.body.len()];
        for (at, item) in self.body.iter().enumerate() {
            let Item::SaveMem(src, idx) = item else {
                continue;
            };
            let k = *idx as usize;
            if !direct_out[k] || saves[k] != 1 {
                continue;
            }
            let mut slot = None;
            for (back, prev) in self.body[..at].iter().enumerate().rev() {
                match prev {
                    Item::LoadStack(dst, s) if dst == src => {
                        slot = Some((*s, back));
                        break;
                    }
                    Item::LoadMem(dst, _) if dst == src => break,
                    Item::LoadStack(..) | Item::SaveMem(..) | Item::LoadMem(..) => {}
                    _ => break,
                }
            }
            let Some((slot, load)) = slot else {
                continue;
            };
            let write = self.body[..load]
                .iter()
                .rposition(|item| matches!(item, Item::SaveStack(_, s) if *s == slot));
            if let Some(write) = write {
                forward.entry(write).or_default().push(k);
                forwarded[at] = true;
            }
        }
        Placement {
            direct_in: (0..n).map(|k| k < states && !saved[k]).collect(),
            direct_out,
            forward,
            forwarded,
        }
    }

    /// The variable `name` as a value of type `want`, holding a value of type `have`.
    fn value(name: &str, have: Ty, want: Ty) -> String {
        match (have, want) {
            (Ty::D64, Ty::D64) => format!("d{}", name),
            (Ty::D64, _) => format!("d{}.x", name),
            (_, Ty::D64) => format!("float2({}, 0.0f)", name),
            _ => name.to_string(),
        }
    }

    /// The variable a value of type `ty` named `name` is written to.
    fn var(name: &str, ty: Ty) -> String {
        if ty == Ty::D64 {
            format!("d{}", name)
        } else {
            name.to_string()
        }
    }

    /// The statement of `dst := op(args)` with `dst` of type `ty` and the arguments of types
    /// `tys`; None if df64 has no such operation.
    fn statement(dst: &str, ty: Ty, op: Op, args: &[String], tys: &[Ty]) -> Option<String> {
        let d = Self::var(dst, ty);
        // the arguments in the operation's type (masks are 32-bit in every type)
        let at = |k: usize, want: Ty| Self::value(&args[k], tys[k], want);
        let float = if ty == Ty::D64 { Ty::D64 } else { Ty::F32 };
        let a = |k: usize| at(k, float);
        if ty != Ty::D64 {
            let s = match op {
                Op::Mov => a(0),
                Op::Zero => "0.0f".into(),
                Op::Neg => format!("bits(ubits({}) ^ 0x80000000u)", a(0)),
                Op::Abs => format!("bits(ubits({}) & 0x7FFFFFFFu)", a(0)),
                Op::Sign => format!("bits(ubits({}) & 0x80000000u)", at(0, Ty::F32)),
                Op::Sqrt => format!("sqrt({})", a(0)),
                Op::Recip => format!("1.0f / {}", a(0)),
                Op::Half => format!("{} * 0.5f", a(0)),
                Op::Rint => format!("rint({})", a(0)),
                Op::Floor => format!("floor({})", a(0)),
                Op::Ceil => format!("ceil({})", a(0)),
                Op::Trunc => format!("trunc({})", a(0)),
                Op::Frac => format!("{} - floor({})", a(0), a(0)),
                Op::Add => format!("{} + {}", a(0), a(1)),
                Op::Sub => format!("{} - {}", a(0), a(1)),
                Op::Mul => format!("{} * {}", a(0), a(1)),
                Op::Div => format!("{} / {}", a(0), a(1)),
                Op::Fma {
                    neg_product,
                    neg_add,
                } => format!(
                    "fma({}{}, {}, {}{})",
                    if neg_product { "-" } else { "" },
                    a(0),
                    a(1),
                    if neg_add { "-" } else { "" },
                    a(2)
                ),
                Op::Cmp(c) => {
                    // a comparison of df64 values is made in df64
                    if tys[0] == Ty::D64 || tys[1] == Ty::D64 {
                        let (x, y) = (at(0, Ty::D64), at(1, Ty::D64));
                        let e = match c {
                            Cmp::Gtu => format!("!dd_le({}, {})", x, y),
                            Cmp::Geu => format!("!dd_lt({}, {})", x, y),
                            Cmp::Lt => format!("dd_lt({}, {})", x, y),
                            Cmp::Le => format!("dd_le({}, {})", x, y),
                            Cmp::Eq => format!("dd_eq({}, {})", x, y),
                            Cmp::Neu => format!("!dd_eq({}, {})", x, y),
                        };
                        format!("mask({})", e)
                    } else {
                        let (x, y) = (a(0), a(1));
                        let e = match c {
                            Cmp::Gtu => format!("!({} <= {})", x, y),
                            Cmp::Geu => format!("!({} < {})", x, y),
                            Cmp::Lt => format!("{} < {}", x, y),
                            Cmp::Le => format!("{} <= {}", x, y),
                            Cmp::Eq => format!("{} == {}", x, y),
                            Cmp::Neu => format!("{} != {}", x, y),
                        };
                        format!("mask({})", e)
                    }
                }
                Op::And => format!("bits(ubits({}) & ubits({}))", a(0), a(1)),
                Op::Or => format!("bits(ubits({}) | ubits({}))", a(0), a(1)),
                Op::Xor => format!("bits(ubits({}) ^ ubits({}))", a(0), a(1)),
                Op::AndNot => format!("bits(~ubits({}) & ubits({}))", a(0), a(1)),
                Op::Not => format!("bits(~ubits({}))", a(0)),
                Op::Select => format!(
                    "bits((ubits({}) & ubits({})) | (ubits({}) & ~ubits({})))",
                    a(0),
                    a(2),
                    a(1),
                    a(2)
                ),
            };
            return Some(format!("{} = {};", d, s));
        }
        // df64: masks (32-bit) apply to both halves
        let both = |k: usize| {
            if tys[k] == Ty::Mask {
                format!("dd_ubits(float2({}))", args[k])
            } else {
                format!("dd_ubits({})", a(k))
            }
        };
        let s = match op {
            Op::Mov => a(0),
            Op::Zero => "float2(0.0f)".into(),
            Op::Neg => format!("-{}", a(0)),
            Op::Abs => format!("({0}.x < 0.0f ? -{0} : {0})", a(0)),
            Op::Sqrt => format!("dd_sqrt({})", a(0)),
            Op::Recip => format!("dd_div(float2(1.0f, 0.0f), {})", a(0)),
            Op::Half => format!("{} * 0.5f", a(0)),
            Op::Add => format!("dd_add({}, {})", a(0), a(1)),
            Op::Sub => format!("dd_add({}, -{})", a(0), a(1)),
            Op::Mul => format!("dd_mul({}, {})", a(0), a(1)),
            Op::Div => format!("dd_div({}, {})", a(0), a(1)),
            Op::Fma {
                neg_product,
                neg_add,
            } => format!(
                "dd_add(dd_mul({}{}, {}), {}{})",
                if neg_product { "-" } else { "" },
                a(0),
                a(1),
                if neg_add { "-" } else { "" },
                a(2)
            ),
            Op::And => format!("dd_bits({} & {})", both(0), both(1)),
            Op::Or => format!("dd_bits({} | {})", both(0), both(1)),
            Op::Xor => format!("dd_bits({} ^ {})", both(0), both(1)),
            Op::AndNot => format!("dd_bits(~{} & {})", both(0), both(1)),
            Op::Not => format!("dd_bits(~{})", both(0)),
            Op::Select => format!(
                "dd_bits(({} & {}) | ({} & ~{}))",
                both(0),
                both(2),
                both(1),
                both(2)
            ),
            // masks and roundings are not df64
            Op::Sign | Op::Cmp(_) | Op::Rint | Op::Floor | Op::Ceil | Op::Trunc | Op::Frac => {
                return None
            }
        };
        Some(format!("{} = {};", d, s))
    }

    fn build_kernel(&mut self) -> String {
        self.inline_funclets();
        let place = self.placement();
        let (name, shape) = (self.options.name.clone(), self.shape);
        let count_states = if shape.row_first {
            0
        } else {
            self.count_states as usize
        };
        let f64_inputs: Vec<bool> = (0..self.count_inputs()).map(|k| self.f64_input(k)).collect();
        let types = typing::types(
            &self.body,
            &place,
            count_states,
            self.count_mem as usize,
            self.options.mixed.as_ref().map(|_| &f64_inputs[..]),
            shape.row_first,
        );
        self.counts = types.counts;

        let mut o = String::new();
        let _ = writeln!(
            o,
            "// {}: {} inputs, {} outputs, {} shared parameters, layout {}{}",
            name,
            shape.count_in,
            shape.count_out,
            shape.count_params,
            self.options.layout.name(),
            if self.options.mapping.is_some() {
                ", arena planes"
            } else {
                ""
            }
        );
        if self.options.mixed.is_some() {
            let _ = writeln!(
                o,
                "// mixed precision: {} df64 and {} f32 operations",
                self.counts.df64_ops, self.counts.f32_ops
            );
        }
        if self.options.mapping.is_some() {
            let _ = writeln!(
                o,
                "kernel void {}(device float *arena [[buffer(0)]], constant uint &n [[buffer(1)]],\n\
                 \x20   constant uint &ld [[buffer(2)]], device const float *lo [[buffer(3)]],\n\
                 \x20   uint i [[thread_position_in_grid]])\n{{",
                name
            );
            let _ = writeln!(o, "\tif (i >= n) return;");
            let _ = writeln!(o, "\tdevice const float *pin = arena + i;");
            let _ = writeln!(o, "\tdevice const float *plo = lo + i;");
            let _ = writeln!(o, "\tdevice float *pout = arena + i;");
            let _ = writeln!(o, "\tconst uint ld_in = ld, ld_out = ld;");
        } else {
            let _ = writeln!(
                o,
                "kernel void {}(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],\n\
                 \x20   device const float *params [[buffer(2)]], constant uint &n [[buffer(3)]],\n\
                 \x20   constant uint &ld_in [[buffer(4)]], constant uint &ld_out [[buffer(5)]],\n\
                 \x20   device const float *in_lo [[buffer(6)]], uint i [[thread_position_in_grid]])\n{{",
                name
            );
            let _ = writeln!(o, "\tif (i >= n) return;");
            match self.options.layout {
                Layout::Soa => {
                    let _ = writeln!(o, "\tdevice const float *pin = in + i;");
                    let _ = writeln!(o, "\tdevice const float *plo = in_lo + i;");
                    let _ = writeln!(o, "\tdevice float *pout = out + i;");
                }
                Layout::Aos => {
                    let _ = writeln!(o, "\tdevice const float *pin = in + i * ld_in;");
                    let _ = writeln!(o, "\tdevice const float *plo = in_lo + i * ld_in;");
                    let _ = writeln!(o, "\tdevice float *pout = out + i * ld_out;");
                }
            }
        }
        // the variables: registers, scratch, stack slots and the memory slots that are locals
        let mut names: Vec<String> = vec!["r".into(), "t".into()];
        names.extend((0..SCRATCH).map(|k| format!("x{}", k)));
        for (prefix, count) in [
            ("g", self.count_gen),
            ("v", self.count_static),
            ("s", self.count_stack),
        ] {
            names.extend((0..count).map(|k| format!("{}{}", prefix, k)));
        }
        for chunk in names.chunks(16) {
            let vars: Vec<String> = chunk.iter().map(|v| format!("{} = 0.0f", v)).collect();
            let _ = writeln!(o, "\tfloat {};", vars.join(", "));
        }
        if self.options.mixed.is_some() {
            for chunk in names.chunks(16) {
                let vars: Vec<String> = chunk
                    .iter()
                    .map(|v| format!("d{} = float2(0.0f)", v))
                    .collect();
                let _ = writeln!(o, "\tfloat2 {};", vars.join(", "));
            }
        }
        for k in 0..self.count_mem as usize {
            if place.direct_in[k] || place.direct_out[k] {
                continue;
            }
            if k < count_states {
                if self.f64_input(k) {
                    let _ = writeln!(o, "\tfloat2 dm{} = {};", k, self.input(k, Ty::D64));
                } else {
                    let _ = writeln!(o, "\tfloat m{} = {};", k, self.input(k, Ty::F32));
                }
            } else {
                let _ = writeln!(o, "\tfloat m{} = 0.0f;", k);
                if self.options.mixed.is_some() {
                    let _ = writeln!(o, "\tfloat2 dm{} = float2(0.0f);", k);
                }
            }
        }

        let first = self.first_out();
        let mut failed: Vec<String> = Vec::new();
        for (at, item) in self.body.iter().enumerate() {
            let (ty, tys) = &types.items[at];
            let (ty, tys) = (*ty, &tys[..]);
            match item {
                Item::Op(dst, op, args) => match Self::statement(dst, ty, *op, args, tys) {
                    Some(s) => {
                        let _ = writeln!(o, "\t{}", s);
                    }
                    None => failed.push(format!("{:?} in df64", op)),
                },
                Item::Const(var, idx) => match self.consts.get(*idx as usize) {
                    Some(&v) if ty == Ty::D64 => {
                        let hi = v as f32;
                        let lo = (v - hi as f64) as f32;
                        let _ = writeln!(
                            o,
                            "\td{} = float2(bits(0x{:08X}u), bits(0x{:08X}u));",
                            var,
                            hi.to_bits(),
                            lo.to_bits()
                        );
                    }
                    Some(&v) => {
                        let _ = writeln!(o, "\t{} = bits(0x{:08X}u);", var, (v as f32).to_bits());
                    }
                    None => {
                        let _ = writeln!(o, "\t{} = NAN; // missing constant {}", var, idx);
                    }
                },
                Item::LoadParam(dst, idx) => {
                    let v = if self.shape.row_first {
                        self.input(*idx as usize, ty)
                    } else {
                        format!("params[{}]", idx)
                    };
                    let _ = writeln!(o, "\t{} = {};", Self::var(dst, ty), v);
                }
                Item::LoadMem(dst, idx) => {
                    let k = *idx as usize;
                    if place.direct_in[k] {
                        let _ = writeln!(o, "\t{} = {};", Self::var(dst, ty), self.input(k, ty));
                    } else {
                        let _ = writeln!(o, "\t{} = {};", Self::var(dst, ty), Self::var(&format!("m{}", k), ty));
                    }
                }
                Item::SaveMem(src, idx) => {
                    let k = *idx as usize;
                    if place.direct_out[k] {
                        if !place.forwarded[at] {
                            let v = Self::value(src, tys[0], Ty::F32);
                            let _ = writeln!(o, "\t{} = {};", self.slot("pout", k - first), v);
                        }
                    } else {
                        let _ = writeln!(o, "\t{} = {};", Self::var(&format!("m{}", k), ty), Self::var(src, ty));
                    }
                }
                Item::LoadStack(dst, idx) => {
                    let _ = writeln!(o, "\t{} = {};", Self::var(dst, ty), Self::var(&format!("s{}", idx), ty));
                }
                Item::SaveStack(src, idx) => {
                    let _ = writeln!(o, "\t{} = {};", Self::var(&format!("s{}", idx), ty), Self::var(src, ty));
                    for &k in place.forward.get(&at).into_iter().flatten() {
                        let v = Self::value(src, ty, Ty::F32);
                        let _ = writeln!(o, "\t{} = {};", self.slot("pout", k - first), v);
                    }
                }
                Item::Label(_) | Item::Branch(_) | Item::CallFunclet(_) | Item::Ret => {
                    unreachable!("inline_funclets leaves straight-line code")
                }
            }
        }
        for j in 0..shape.count_out {
            let k = first + j;
            if !place.direct_out[k] {
                let v = Self::value(&format!("m{}", k), types.final_mem[k], Ty::F32);
                let _ = writeln!(o, "\t{} = {};", self.slot("pout", j), v);
            }
        }
        let _ = writeln!(o, "}}");
        for f in failed {
            self.unsupported(&f);
        }
        o
    }
}

impl Generator for MslGenerator {
    fn count_shadows(&self) -> u8 {
        0
    }

    fn three_address(&self) -> bool {
        true
    }

    fn bytes(&mut self) -> Vec<u8> {
        self.kernel.as_bytes().to_vec()
    }

    fn what(&self) -> GeneratorType {
        GeneratorType::Unknown
    }

    fn seal(&mut self) {
        let first_out = if self.shape.row_first {
            0
        } else {
            self.count_states
        };
        let needed = (first_out as usize + self.shape.count_out) as u32;
        self.count_mem = self.count_mem.max(needed);
        if !self.shape.row_first {
            self.count_mem = self.count_mem.max(self.count_states);
        }
        self.kernel = self.build_kernel();
    }

    fn align(&mut self) {}

    fn set_label(&mut self, label: &str) {
        self.body.push(Item::Label(label.to_string()));
    }

    fn branch(&mut self, label: &str) {
        self.body.push(Item::Branch(label.to_string()));
    }

    fn branch_if(&mut self, _cond: Reg, _label: &str, _is_else: bool) {
        self.unsupported("control flow (conditional jumps)");
    }

    fn fmov(&mut self, dst: Reg, s1: Reg) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        if d != a {
            self.op(&d, Op::Mov, &[&a]);
        }
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        let (a, b) = (self.reg(s1), self.reg(s2));
        if a != b {
            self.op("x0", Op::Mov, &[&a]);
            self.op(&a, Op::Mov, &[&b]);
            self.op(&b, Op::Mov, &["x0"]);
        }
    }

    fn load_const(&mut self, dst: Reg, idx: u32) {
        let d = self.reg(dst);
        self.body.push(Item::Const(d, idx));
    }

    fn load_mem(&mut self, dst: Reg, idx: u32) {
        self.load_loc(dst, Loc::Mem(idx));
    }

    fn save_mem(&mut self, dst: Reg, idx: u32) {
        self.save_loc(dst, Loc::Mem(idx));
    }

    fn load_param(&mut self, dst: Reg, idx: u32) {
        self.load_loc(dst, Loc::Param(idx));
    }

    fn load_stack(&mut self, dst: Reg, idx: u32) {
        self.load_loc(dst, Loc::Stack(idx));
    }

    fn save_stack(&mut self, dst: Reg, idx: u32) {
        self.save_loc(dst, Loc::Stack(idx));
    }

    fn load_mem_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_loc(xd, Loc::Mem(idx));
        self.load_loc(yd, Loc::Mem(idx + 1));
    }

    fn save_mem_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save_loc(xs, Loc::Mem(idx));
        self.save_loc(ys, Loc::Mem(idx + 1));
    }

    fn load_param_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_loc(xd, Loc::Param(idx));
        self.load_loc(yd, Loc::Param(idx + 1));
    }

    fn load_stack_complex(&mut self, xd: Reg, yd: Reg, idx: u32) {
        self.load_loc(xd, Loc::Stack(idx));
        self.load_loc(yd, Loc::Stack(idx + 1));
    }

    fn save_stack_complex(&mut self, xs: Reg, ys: Reg, idx: u32) {
        self.save_loc(xs, Loc::Stack(idx));
        self.save_loc(ys, Loc::Stack(idx + 1));
    }

    fn load_args(&mut self, _locs: Vec<Loc>, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn save_args(&mut self, _num_args: u8, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn load_args_complex(&mut self, _locs: Vec<Loc>, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn save_args_complex(&mut self, _num_args: u8, _ultra: bool) {
        self.unsupported("compression mode");
    }

    fn save_mem_result(&mut self, idx: u32) {
        self.save_loc(Reg::Ret, Loc::Mem(idx));
    }

    fn save_stack_result(&mut self, idx: u32) {
        self.save_loc(Reg::Ret, Loc::Stack(idx));
    }

    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Neg, dst, s1);
    }

    fn sign(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Sign, dst, s1);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Abs, dst, s1);
    }

    fn abs2(&mut self, dst: Reg, s1: Reg) {
        self.binary(Op::Mul, dst, s1, s1);
    }

    // only generated for fast complex, which lowered MIR does not use (as the PTX backend)
    fn times_i(&mut self, dst: Reg, _s1: Reg) {
        let d = self.reg(dst);
        self.op(&d, Op::Zero, &[]);
    }

    fn times_neg_i(&mut self, dst: Reg, _s1: Reg) {
        let d = self.reg(dst);
        self.op(&d, Op::Zero, &[]);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Sqrt, dst, s1);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Sqrt, dst, s1);
    }

    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Recip, dst, s1);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Half, dst, s1);
    }

    // half to even, as x86's `roundsd` mode 0
    fn round(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Rint, dst, s1);
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Floor, dst, s1);
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Ceil, dst, s1);
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Trunc, dst, s1);
    }

    fn frac(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Frac, dst, s1);
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Add, dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Sub, dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Mul, dst, s1, s2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Div, dst, s1, s2);
    }

    // the formulas of the PTX backend (x86's scalar generator): x1 x2 - y1 y2 and
    // x2 y1 + x1 y2, each with one fused operation
    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let (xd, yd) = (self.reg(xd), self.reg(yd));
        let (x1, y1, x2, y2) = (self.reg(x1), self.reg(y1), self.reg(x2), self.reg(y2));
        let fma = |neg_add| Op::Fma {
            neg_product: false,
            neg_add,
        };
        self.op("x2", Op::Mul, &[&y1, &y2]);
        self.op("x2", fma(true), &[&x1, &x2, "x2"]);
        self.op("x3", Op::Mul, &[&x1, &y2]);
        self.op(&yd, fma(false), &[&x2, &y1, "x3"]);
        self.op(&xd, Op::Mov, &["x2"]);
        true
    }

    // (x1 x2 + y1 y2) and (x2 y1 - x1 y2) over x2^2 + y2^2, each with one fused operation
    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let (xd, yd) = (self.reg(xd), self.reg(yd));
        let (x1, y1, x2, y2) = (self.reg(x1), self.reg(y1), self.reg(x2), self.reg(y2));
        let fma = |neg_add| Op::Fma {
            neg_product: false,
            neg_add,
        };
        self.op("x2", Op::Mul, &[&y1, &y2]);
        self.op("x2", fma(false), &[&x1, &x2, "x2"]);
        self.op("x3", Op::Mul, &[&x1, &y2]);
        self.op("x3", fma(true), &[&x2, &y1, "x3"]);
        self.op("x1", Op::Mul, &[&x2, &x2]);
        self.op("x1", fma(false), &[&y2, &y2, "x1"]);
        self.op(&xd, Op::Div, &["x2", "x1"]);
        self.op(&yd, Op::Div, &["x3", "x1"]);
        true
    }

    fn fuse_load_math(&mut self) {}

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!("support_times2 is false");
    }

    fn real(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn imaginary(&mut self, dst: Reg, _s1: Reg) {
        let d = self.reg(dst);
        self.op(&d, Op::Zero, &[]);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, _s2: Reg) {
        self.fmov(dst, s1);
    }

    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Cmp(Cmp::Gtu), dst, s1, s2);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Cmp(Cmp::Geu), dst, s1, s2);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Cmp(Cmp::Lt), dst, s1, s2);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Cmp(Cmp::Le), dst, s1, s2);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Cmp(Cmp::Eq), dst, s1, s2);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Cmp(Cmp::Neu), dst, s1, s2);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::And, dst, s1, s2);
    }

    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::AndNot, dst, s1, s2);
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Or, dst, s1, s2);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary(Op::Xor, dst, s1, s2);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.unary(Op::Not, dst, s1);
    }

    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, false, false);
    }

    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, false, true);
    }

    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, true, false);
    }

    fn fused_neg_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, true, true);
    }

    fn add_consts(&mut self, consts: &[f64]) {
        self.consts = consts.to_vec();
    }

    fn add_func(&mut self, op: &str, f: Func) {
        match f {
            Func::Unary(_)
            | Func::Binary(_)
            | Func::UnaryCplx(_)
            | Func::BinaryCplx(_)
            | Func::Recursive => {}
            _ => self.unsupported(&format!("the user-defined function `{}`", op)),
        }
    }

    fn call(&mut self, op: &str, _num_args: usize) -> Result<()> {
        self.unsupported(&format!("math functions (`{}`)", op));
        Ok(())
    }

    fn call_complex(&mut self, op: &str, _num_args: usize) -> Result<()> {
        self.unsupported(&format!("complex math functions (`{}`)", op));
        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        self.body.push(Item::CallFunclet(label.to_string()));
    }

    fn ret(&mut self) {
        self.body.push(Item::Ret);
    }

    fn prologue_fast(&mut self, _cap: usize, _count_states: usize, _count_obs: usize) {
        self.unsupported("the fast kernel");
    }

    fn epilogue_fast(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _idx_ret: i32,
    ) {
        self.unsupported("the fast kernel");
    }

    fn prologue_indirect(
        &mut self,
        _cap: usize,
        count_states: usize,
        _count_obs: usize,
        _count_params: usize,
    ) {
        self.count_states = count_states as u32;
    }

    fn epilogue_indirect(
        &mut self,
        _cap: usize,
        _count_states: usize,
        _count_obs: usize,
        _count_params: usize,
    ) {
    }

    fn save_used_registers(&mut self, _used: &[Reg]) {}

    fn load_used_registers(&mut self, _used: &[Reg]) {}

    // (true_val & mask) | (false_val & !mask), mask = Stack[idx]
    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        let (d, t, f) = (self.reg(dst), self.reg(true_val), self.reg(false_val));
        if t == f {
            if d != t {
                self.op(&d, Op::Mov, &[&t]);
            }
            return;
        }
        self.count_stack = self.count_stack.max(idx + 1);
        // the mask is read from its stack slot: a load into scratch keeps the stack slots'
        // values flowing through `LoadStack`
        self.body.push(Item::LoadStack("x1".into(), idx));
        self.op(&d, Op::Select, &[&t, &f, "x1"]);
    }
}
