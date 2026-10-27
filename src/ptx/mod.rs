// PTX backend: compiles MIR into a CUDA PTX module, one GPU thread per evaluation point.
//
// The module is meant to be loaded by the CUDA driver (`cuModuleLoadData`, which compiles it
// for the GPU at hand), not run in-process. `sjb2ptx` writes it from a saved symjit application
// (`.sjb`, see `crate::sjb`). See docs/PTX.md and plans/sjb2ptx/DESIGN.md.
//
// Kernel interface:
//   .entry <name>(.param .u64 in, .param .u64 out, .param .u64 params, .param .u64 n,
//                 .param .u64 ld_in, .param .u64 ld_out)
// Thread i < n evaluates point i: slot k of its inputs is in[k * ld_in + i] (SoA) or
// in[i * ld_in + k] (AoS), and likewise for `out`. The inputs are the states, or the
// parameters of a Symbolica model (`row_first`); the outputs are the observables followed by
// the diffs; `params` holds the shared parameters of other models.
//
// Every value is a `.b64` virtual register (ptxas allocates the machine registers): the MIR's
// registers, its stack slots (%s<k>) and its memory slots (%m<k>). An input the MIR never writes
// is read from `in` where it is used, and an output it never reads is written to `out` where it
// is saved, if every save is unconditional (`Placement`), so that neither stays live across the
// kernel; the other memory slots are gathered from `in` at the start and scattered to `out` at
// the end. The arithmetic is that of the x86-64 scalar
// generator (`amd/scalar.rs`): correctly rounded `.rn` operations, `fma.rn` only for the MIR's
// fused operations (so ptxas contracts nothing), unordered `gt`/`geq`, masks of all bits.
// Funclets run on the kernel's registers: a call sets %site and jumps, `ret` is a `brx.idx`
// through the table of return sites.

mod driver;
mod header;
pub mod libm;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::fmt::Write;

use anyhow::{anyhow, Result};

use symjit::backend::{Func, Generator, GeneratorType, Loc, Reg};

use crate::sjb::SavedModel;

pub use driver::{driver_header, driver_source};
pub use header::header;

/// The PTX ISA version and the default target: clang 18 emits PTX 8.3 (the math library's
/// functions are compiled by it), and CUDA 13 compiles for sm_70 and newer.
pub const PTX_VERSION: &str = "8.3";
pub const DEFAULT_SM: u32 = 70;
/// The block size the header suggests.
pub const BLOCK_SIZE: u32 = 128;

const SIGN_BIT: &str = "0x8000000000000000";
const NOT_SIGN_BIT: &str = "0x7FFFFFFFFFFFFFFF";
const ZERO: &str = "0d0000000000000000";

/// Where slot `k` of point `i` is in `in` and `out`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `a[k * ld + i]`: neighbouring threads read neighbouring addresses (coalesced)
    Soa,
    /// `a[i * ld + k]`: the row-major matrix of `evaluate_matrix`
    Aos,
}

impl Layout {
    pub fn name(self) -> &'static str {
        match self {
            Layout::Soa => "soa",
            Layout::Aos => "aos",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PtxOptions {
    /// the kernel's name (a C identifier)
    pub name: String,
    pub layout: Layout,
    /// `.target sm_<sm>`
    pub sm: u32,
    /// planes of one arena (a pacman runner's): slot `k` of the inputs at plane `mapping[k]`
    /// and output `j` at plane `mapping[count_in + j]`, with `in` = `out` the arena and
    /// `ld_in` = `ld_out` its leading dimension (SoA plane kernels only; outputs distinct and
    /// disjoint from the inputs, which keeps `ld.global.nc` legal)
    pub mapping: Option<Vec<u32>>,
}

impl PtxOptions {
    pub fn new(name: &str) -> PtxOptions {
        PtxOptions {
            name: name.to_string(),
            layout: Layout::Soa,
            sm: DEFAULT_SM,
            mapping: None,
        }
    }
}

/// The counts of the kernel's interface, in f64 slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Shape {
    pub count_in: usize,
    pub count_out: usize,
    /// shared parameters (0 for Symbolica models, whose parameters are the inputs)
    pub count_params: usize,
    /// a Symbolica model: inputs in the parameter slots, outputs in memory from slot 0
    pub row_first: bool,
}

/// A PTX module and the counts of its kernel.
pub struct Ptx {
    pub text: String,
    pub shape: Shape,
}

impl SavedModel {
    /// The PTX module of the model (see the top of this file for its interface).
    pub fn ptx(&mut self, options: &PtxOptions) -> Result<Ptx> {
        if libm::is_library_name(&options.name) {
            return Err(anyhow!(
                "the kernel name `{}` is reserved: names starting with `__sj` belong to the math library",
                options.name
            ));
        }
        let prog = &mut self.prog;
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
            check_mapping(map, &shape, options.layout)?;
        }
        let mut generator = PtxGenerator::new(options.clone(), shape);
        let (count_states, count_obs, count_params) =
            (prog.count_states, prog.count_obs, prog.count_params);
        prog.builder.compile_from_mir(
            &self.mir,
            &mut generator,
            count_states,
            count_obs,
            count_params,
        )?;
        generator.check()?;
        Ok(Ptx {
            text: generator.text(),
            shape,
        })
    }
}

/// A plane mapping fits `shape`: one plane per slot, SoA, no shared parameters, a model whose
/// inputs are states (a plane kernel), and outputs distinct and disjoint from the inputs.
fn check_mapping(map: &[u32], shape: &Shape, layout: Layout) -> Result<()> {
    if layout != Layout::Soa || shape.row_first || shape.count_params != 0 {
        return Err(anyhow!(
            "a plane mapping needs the SoA layout and a plane (direct-arena) kernel without shared parameters"
        ));
    }
    if map.len() != shape.count_in + shape.count_out {
        return Err(anyhow!(
            "the plane mapping has {} planes, the kernel {} inputs and {} outputs",
            map.len(),
            shape.count_in,
            shape.count_out
        ));
    }
    let (inputs, outputs) = map.split_at(shape.count_in);
    let mut written = std::collections::HashSet::new();
    for &p in outputs {
        if !written.insert(p) {
            return Err(anyhow!("the plane mapping writes plane {} twice", p));
        }
    }
    if let Some(p) = inputs.iter().find(|p| written.contains(p)) {
        return Err(anyhow!("the plane mapping reads and writes plane {}", p));
    }
    Ok(())
}

enum Item {
    Line(String),
    /// `mov.b64 <reg>, consts[idx]`; the constants arrive after the code (`add_consts`)
    Const(String, u32),
    /// `<reg> := memory slot idx`, placed in `seal` (see `Placement`)
    LoadMem(String, u32),
    /// `memory slot idx := <reg>`, placed in `seal`
    SaveMem(String, u32),
    /// `<reg> := stack slot idx`
    LoadStack(String, u32),
    /// `stack slot idx := <reg>`; may also write an output (`Placement::forward`)
    SaveStack(String, u32),
    Label(String),
    Branch(String),
    /// jump if the mask `cond` is true (all bits set: a NaN) when `if_true`, else if false
    BranchIf {
        cond: String,
        label: String,
        if_true: bool,
    },
    /// a call of the funclet at `label` returning to site `site`
    CallFunclet {
        label: String,
        site: usize,
    },
    Ret,
}

pub struct PtxGenerator {
    options: PtxOptions,
    shape: Shape,
    body: Vec<Item>,
    consts: Vec<f64>,
    count_gen: u32,
    count_stack: u32,
    count_mem: u32,
    count_states: u32,
    count_sites: usize,
    /// the math library's entry points the kernel calls (`sin`, `cplx_sin`, ...)
    calls: Vec<String>,
    unsupported: Vec<String>,
    module: String,
}

/// Where the memory slots live. A slot that is not direct is a register: gathered from `in`
/// (states) or zeroed in the prologue, scattered to `out` (outputs) in the epilogue.
struct Placement {
    /// the input is read from `in` at each use (it is never written)
    direct_in: Vec<bool>,
    /// the output is written to `out` at each save (it is never read, and saved only
    /// unconditionally: outside funclets, in a body without conditional jumps)
    direct_out: Vec<bool>,
    /// `SaveStack` item -> the direct outputs that are copies of the value it writes: the MIR
    /// keeps an output in a stack slot until a copy at the end; the output is written where
    /// the slot is (by its last write before the copy, at the top level, the slot written by
    /// no funclet), and the copy is dropped, so that the value is not live until the end
    forward: HashMap<usize, Vec<usize>>,
    /// the `SaveMem` items replaced by a forwarded write
    forwarded: Vec<bool>,
}

impl PtxGenerator {
    pub fn new(options: PtxOptions, shape: Shape) -> PtxGenerator {
        PtxGenerator {
            options,
            shape,
            body: Vec::new(),
            consts: Vec::new(),
            count_gen: 0,
            count_stack: 0,
            count_mem: 0,
            count_states: 0,
            count_sites: 0,
            calls: Vec::new(),
            unsupported: Vec::new(),
            module: String::new(),
        }
    }

    /// Fails if the compiled MIR used an operation this backend cannot lower (yet).
    pub fn check(&self) -> Result<()> {
        if self.unsupported.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(
                "the PTX backend does not support {}",
                self.unsupported.join(", ")
            ))
        }
    }

    /// The module (after `seal`).
    pub fn text(&self) -> String {
        self.module.clone()
    }

    fn unsupported(&mut self, what: &str) {
        if !self.unsupported.iter().any(|s| s == what) {
            self.unsupported.push(what.to_string());
        }
        self.emit("trap;".to_string());
    }

    fn emit(&mut self, line: String) {
        self.body.push(Item::Line(line));
    }

    fn reg(&mut self, r: Reg) -> String {
        match r {
            Reg::Ret | Reg::Left => "%ret".into(),
            Reg::Temp | Reg::Right => "%tmp".into(),
            Reg::Gen(n) => {
                self.count_gen = self.count_gen.max(n as u32 + 1);
                format!("%g{}", n)
            }
            Reg::Static(n) => {
                self.unsupported(&format!("unallocated registers (Static({}))", n));
                "%x0".into()
            }
        }
    }

    fn stack(&mut self, idx: u32) -> String {
        self.count_stack = self.count_stack.max(idx + 1);
        format!("%s{}", idx)
    }

    fn mem(&mut self, idx: u32) -> String {
        self.count_mem = self.count_mem.max(idx + 1);
        format!("%m{}", idx)
    }

    fn mov(&mut self, dst: String, src: String) {
        if dst != src {
            self.emit(format!("mov.b64 {}, {};", dst, src));
        }
    }

    fn unary(&mut self, op: &str, dst: Reg, s1: Reg) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        self.emit(format!("{} {}, {};", op, d, a));
    }

    fn binary(&mut self, op: &str, dst: Reg, s1: Reg, s2: Reg) {
        let (d, a, b) = (self.reg(dst), self.reg(s1), self.reg(s2));
        self.emit(format!("{} {}, {}, {};", op, d, a, b));
    }

    /// `dst := s1 <op> imm` on the bits
    fn bits_imm(&mut self, op: &str, dst: Reg, s1: Reg, imm: &str) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        self.emit(format!("{}.b64 {}, {}, {};", op, d, a, imm));
    }

    /// A call of the math library's `__sj_<op>`: the arguments are `args`, the result goes to
    /// %ret (and %tmp, the imaginary part, for a complex function).
    fn call_library(&mut self, op: &str, args: &[String], complex: bool) {
        if !self.calls.iter().any(|c| c == op) {
            self.calls.push(op.to_string());
        }
        self.emit("{".into());
        let mut params = Vec::new();
        for (k, a) in args.iter().enumerate() {
            self.emit(format!("\t.param .b64 arg{};", k));
            self.emit(format!("\tst.param.b64 [arg{}], {};", k, a));
            params.push(format!("arg{}", k));
        }
        if complex {
            self.emit("\t.param .align 8 .b8 ret0[16];".into());
        } else {
            self.emit("\t.param .b64 ret0;".into());
        }
        self.emit(format!(
            "\tcall.uni (ret0), __sj_{}, ({});",
            op,
            params.join(", ")
        ));
        self.emit("\tld.param.b64 %ret, [ret0];".into());
        if complex {
            self.emit("\tld.param.b64 %tmp, [ret0+8];".into());
        }
        self.emit("}".into());
    }

    fn set_zero(&mut self, dst: Reg) {
        let d = self.reg(dst);
        self.emit(format!("mov.b64 {}, {};", d, ZERO));
    }

    // comparisons produce masks: all bits set (true) or zero (false)
    fn compare(&mut self, cmp: &str, dst: Reg, s1: Reg, s2: Reg) {
        let (d, a, b) = (self.reg(dst), self.reg(s1), self.reg(s2));
        self.emit(format!("setp.{}.f64 %p0, {}, {};", cmp, a, b));
        self.emit(format!("selp.b64 {}, -1, 0, %p0;", d));
    }

    /// `dst := ±(s1 * s2) ± s3` with one rounding (`fma.rn`), as x86's vfmadd/vfmsub/vfnmadd/
    /// vfnmsub; negating an operand is exact.
    fn fused(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg, neg_product: bool, neg_add: bool) {
        let (d, a, b, c) = (self.reg(dst), self.reg(s1), self.reg(s2), self.reg(s3));
        let a = if neg_product {
            self.emit(format!("neg.f64 %x0, {};", a));
            "%x0".to_string()
        } else {
            a
        };
        let c = if neg_add {
            self.emit(format!("neg.f64 %x1, {};", c));
            "%x1".to_string()
        } else {
            c
        };
        self.emit(format!("fma.rn.f64 {}, {}, {}, {};", d, a, b, c));
    }

    fn load_loc(&mut self, dst: Reg, loc: Loc) {
        let d = self.reg(dst);
        match loc {
            Loc::Mem(idx) => {
                self.mem(idx);
                self.body.push(Item::LoadMem(d, idx));
            }
            Loc::Stack(idx) => {
                self.stack(idx);
                self.body.push(Item::LoadStack(d, idx));
            }
            Loc::Param(idx) => self.param(d, idx),
        }
    }

    fn save_loc(&mut self, src: Reg, loc: Loc) {
        let s = self.reg(src);
        match loc {
            Loc::Mem(idx) => {
                self.mem(idx);
                self.body.push(Item::SaveMem(s, idx));
            }
            Loc::Stack(idx) => {
                self.stack(idx);
                self.body.push(Item::SaveStack(s, idx));
            }
            Loc::Param(_) => self.unsupported("stores to parameters"),
        }
    }

    /// `dst := parameter idx`: an input of the point (Symbolica models) or a shared parameter.
    /// Parameters are never written, so they are read where they are used.
    fn param(&mut self, dst: String, idx: u32) {
        if self.shape.row_first {
            match self.options.layout {
                Layout::Soa => {
                    self.emit(format!("mul.lo.u64 %a1, %ldi, {};", idx));
                    self.emit("add.u64 %a1, %a1, %pin;".into());
                    self.emit(format!("ld.global.nc.b64 {}, [%a1];", dst));
                }
                Layout::Aos => self.emit(format!("ld.global.nc.b64 {}, [%pin+{}];", dst, 8 * idx)),
            }
        } else {
            self.emit(format!("ld.global.nc.b64 {}, [%par+{}];", dst, 8 * idx));
        }
    }

    /// The first memory slot of the outputs.
    fn first_out(&self) -> usize {
        if self.shape.row_first {
            0
        } else {
            self.count_states as usize
        }
    }

    fn placement(&self) -> Placement {
        let n = self.count_mem as usize;
        let (mut loaded, mut saved, mut saved_in_funclet) =
            (vec![false; n], vec![false; n], vec![false; n]);
        let targets: Vec<&String> = self
            .body
            .iter()
            .filter_map(|item| match item {
                Item::CallFunclet { label, .. } => Some(label),
                _ => None,
            })
            .collect();
        let (mut in_funclet, mut conditional) = (false, false);
        // stack slots some funclet writes
        let mut funclet_stack: Vec<u32> = Vec::new();
        for item in self.body.iter() {
            match item {
                Item::SaveStack(_, idx) if in_funclet => funclet_stack.push(*idx),
                Item::Label(l) if targets.contains(&l) => in_funclet = true,
                Item::Ret => in_funclet = false,
                Item::BranchIf { .. } if !in_funclet => conditional = true,
                Item::LoadMem(_, idx) => loaded[*idx as usize] = true,
                Item::SaveMem(_, idx) => {
                    saved[*idx as usize] = true;
                    saved_in_funclet[*idx as usize] |= in_funclet;
                }
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
            .map(|k| {
                outs.contains(&k) && saved[k] && !loaded[k] && !saved_in_funclet[k] && !conditional
            })
            .collect();

        // forwarding: a direct output saved once (live saves only), from a register loaded
        // from a stack slot among the memory and stack moves just before it
        let mut live_saves = vec![0usize; n];
        for (at, item) in self.body.iter().enumerate() {
            if let Item::SaveMem(_, idx) = item {
                if direct_out[*idx as usize] && !self.overwritten(at) {
                    live_saves[*idx as usize] += 1;
                }
            }
        }
        let mut forward: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut forwarded = vec![false; self.body.len()];
        for (at, item) in self.body.iter().enumerate() {
            let Item::SaveMem(src, idx) = item else {
                continue;
            };
            let k = *idx as usize;
            if !direct_out[k] || live_saves[k] != 1 || self.overwritten(at) {
                continue;
            }
            // the stack slot the saved register was loaded from, just before
            let mut slot = None;
            for (back, prev) in self.body[..at].iter().enumerate().rev() {
                match prev {
                    Item::LoadStack(dst, s) if dst == src => {
                        slot = Some((*s, back));
                        break;
                    }
                    // the register is not a copy of a stack slot
                    Item::LoadMem(dst, _) if dst == src => break,
                    Item::LoadStack(..) | Item::SaveMem(..) | Item::LoadMem(..) => {}
                    _ => break,
                }
            }
            let Some((slot, load)) = slot else {
                continue;
            };
            if funclet_stack.contains(&slot) {
                continue;
            }
            // its last write before the load (straight-line: no conditional jumps)
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

    /// The save of a direct output at `at` is dead: a later save to its slot comes before any
    /// jump or label (direct outputs are never read, and other items do not touch memory slots).
    fn overwritten(&self, at: usize) -> bool {
        let Some(Item::SaveMem(_, idx)) = self.body.get(at) else {
            return false;
        };
        for item in self.body[at + 1..].iter().take(64) {
            match item {
                Item::SaveMem(_, next) if next == idx => return true,
                Item::Line(_) | Item::Const(..) | Item::LoadMem(..) | Item::SaveMem(..) => {}
                _ => return false,
            }
        }
        false
    }

    /// The address of slot `k` of the point in `%a1` (SoA), or the operand addressing it; with a
    /// plane mapping, slot `k` of `%pin` (an input) or `%pout` (an output) is its plane.
    fn slot(&self, o: &mut String, base: &str, ld: &str, k: usize) -> String {
        let k = match &self.options.mapping {
            Some(map) if base == "%pout" => map[self.shape.count_in + k] as usize,
            Some(map) => map[k] as usize,
            None => k,
        };
        match self.options.layout {
            Layout::Soa if k == 0 => format!("[{}]", base),
            Layout::Soa => {
                let _ = writeln!(o, "\tmul.lo.u64 %a1, {}, {};", ld, k);
                let _ = writeln!(o, "\tadd.u64 %a1, %a1, {};", base);
                "[%a1]".to_string()
            }
            Layout::Aos => format!("[{}+{}]", base, 8 * k),
        }
    }

    /// The label of a MIR label: MIR labels contain `@` and other characters PTX does not
    /// allow, so each gets a number.
    fn label(labels: &mut HashMap<String, usize>, label: &str) -> String {
        let n = labels.len();
        format!("$L{}", labels.entry(label.to_string()).or_insert(n))
    }

    /// The thread's index, the bounds test and the input pointers; the states that are not
    /// direct are gathered into the memory registers, the other registers are zeroed.
    fn prologue(&self, o: &mut String, place: &Placement) {
        let (layout, shape) = (self.options.layout, self.shape);
        let _ = writeln!(o, "\tmov.u32 %t0, %ctaid.x;");
        let _ = writeln!(o, "\tmov.u32 %t1, %ntid.x;");
        let _ = writeln!(o, "\tmov.u32 %t2, %tid.x;");
        let _ = writeln!(o, "\tmul.wide.u32 %i, %t0, %t1;");
        let _ = writeln!(o, "\tcvt.u64.u32 %a0, %t2;");
        let _ = writeln!(o, "\tadd.u64 %i, %i, %a0;");
        let _ = writeln!(o, "\tld.param.u64 %a0, [{}_n];", self.options.name);
        let _ = writeln!(o, "\tsetp.ge.u64 %p0, %i, %a0;");
        let _ = writeln!(o, "\t@%p0 bra $Lexit;");
        for (reg, param) in [("%pin", "in"), ("%pout", "out"), ("%par", "params")] {
            let _ = writeln!(
                o,
                "\tld.param.u64 {}, [{}_{}];",
                reg, self.options.name, param
            );
            if reg != "%par" {
                let _ = writeln!(o, "\tcvta.to.global.u64 {}, {};", reg, reg);
            }
        }
        let _ = writeln!(o, "\tcvta.to.global.u64 %par, %par;");
        for (reg, param) in [("%ldi", "ld_in"), ("%ldo", "ld_out")] {
            let _ = writeln!(
                o,
                "\tld.param.u64 {}, [{}_{}];",
                reg, self.options.name, param
            );
            // in bytes
            let _ = writeln!(o, "\tshl.b64 {}, {}, 3;", reg, reg);
        }
        // %pin and %pout point to the point's first slot
        for (reg, ld) in [("%pin", "%ldi"), ("%pout", "%ldo")] {
            match layout {
                Layout::Soa => {
                    let _ = writeln!(o, "\tshl.b64 %a0, %i, 3;");
                }
                Layout::Aos => {
                    let _ = writeln!(o, "\tmul.lo.u64 %a0, %i, {};", ld);
                }
            }
            let _ = writeln!(o, "\tadd.u64 {}, {}, %a0;", reg, reg);
        }

        let count_states = if shape.row_first {
            0
        } else {
            self.count_states as usize
        };
        for k in 0..self.count_mem as usize {
            if place.direct_in[k] || place.direct_out[k] {
                continue;
            }
            if k < count_states {
                let a = self.slot(o, "%pin", "%ldi", k);
                let _ = writeln!(o, "\tld.global.nc.b64 %m{}, {};", k, a);
            } else {
                let _ = writeln!(o, "\tmov.b64 %m{}, {};", k, ZERO);
            }
        }
    }

    /// The outputs that are not direct are scattered from the memory registers.
    fn epilogue(&self, o: &mut String, place: &Placement) {
        let first = self.first_out();
        for j in 0..self.shape.count_out {
            let k = first + j;
            if !place.direct_out[k] {
                let a = self.slot(o, "%pout", "%ldo", j);
                let _ = writeln!(o, "\tst.global.b64 {}, %m{};", a, k);
            }
        }
        let _ = writeln!(o, "$Lexit:");
        let _ = writeln!(o, "\tret;");
    }

    /// The body with its labels, jumps, constants and funclet calls resolved.
    fn lower_body(&mut self, place: &Placement) -> String {
        let mut labels: HashMap<String, usize> = HashMap::new();
        let mut sites: Vec<String> = Vec::new();
        let mut o = String::new();
        // the funclet whose body is being emitted (from its label to its `ret`), to find
        // nested calls, which would overwrite %site
        let targets: Vec<String> = self
            .body
            .iter()
            .filter_map(|item| match item {
                Item::CallFunclet { label, .. } => Some(label.clone()),
                _ => None,
            })
            .collect();
        let mut in_funclet = false;
        let mut nested = false;
        let mut stray_ret = false;

        let first = self.first_out();
        for (at, item) in self.body.iter().enumerate() {
            match item {
                Item::Line(line) => {
                    let _ = writeln!(o, "\t{}", line);
                }
                Item::LoadMem(dst, idx) => {
                    let k = *idx as usize;
                    if place.direct_in[k] {
                        let a = self.slot(&mut o, "%pin", "%ldi", k);
                        let _ = writeln!(o, "\tld.global.nc.b64 {}, {};", dst, a);
                    } else if *dst != format!("%m{}", k) {
                        let _ = writeln!(o, "\tmov.b64 {}, %m{};", dst, k);
                    }
                }
                Item::LoadStack(dst, idx) => {
                    let _ = writeln!(o, "\tmov.b64 {}, %s{};", dst, idx);
                }
                Item::SaveStack(src, idx) => {
                    let _ = writeln!(o, "\tmov.b64 %s{}, {};", idx, src);
                    for &k in place.forward.get(&at).into_iter().flatten() {
                        let a = self.slot(&mut o, "%pout", "%ldo", k - first);
                        let _ = writeln!(o, "\tst.global.b64 {}, {};", a, src);
                    }
                }
                Item::SaveMem(src, idx) => {
                    let k = *idx as usize;
                    if place.direct_out[k] {
                        if self.overwritten(at) || place.forwarded[at] {
                            continue;
                        }
                        let a = self.slot(&mut o, "%pout", "%ldo", k - first);
                        let _ = writeln!(o, "\tst.global.b64 {}, {};", a, src);
                    } else if *src != format!("%m{}", k) {
                        let _ = writeln!(o, "\tmov.b64 %m{}, {};", k, src);
                    }
                }
                Item::Const(reg, idx) => match self.consts.get(*idx as usize) {
                    Some(v) => {
                        let _ = writeln!(o, "\tmov.b64 {}, 0d{:016X};", reg, v.to_bits());
                    }
                    None => {
                        let _ = writeln!(o, "\ttrap; // missing constant {}", idx);
                    }
                },
                Item::Label(l) => {
                    if targets.contains(l) {
                        in_funclet = true;
                    }
                    let _ = writeln!(o, "{}:", Self::label(&mut labels, l));
                }
                Item::Branch(l) => {
                    let _ = writeln!(o, "\tbra.uni {};", Self::label(&mut labels, l));
                }
                Item::BranchIf {
                    cond,
                    label,
                    if_true,
                } => {
                    // a true mask (all bits set) is a NaN, as x86's `vucomisd` + `jpe`/`jpo`
                    let _ = writeln!(o, "\tsetp.nan.f64 %p0, {}, {};", cond, cond);
                    let _ = writeln!(
                        o,
                        "\t@{}%p0 bra {};",
                        if *if_true { "" } else { "!" },
                        Self::label(&mut labels, label)
                    );
                }
                Item::CallFunclet { label, site } => {
                    if in_funclet {
                        nested = true;
                    }
                    let ret = format!("$R{}", site);
                    let _ = writeln!(o, "\tmov.u32 %site, {};", site);
                    let _ = writeln!(o, "\tbra.uni {};", Self::label(&mut labels, label));
                    let _ = writeln!(o, "{}:", ret);
                    sites.push(ret);
                }
                Item::Ret => {
                    if !in_funclet {
                        stray_ret = true;
                    }
                    in_funclet = false;
                    let _ = writeln!(o, "\tbrx.idx %site, $SITES;");
                }
            }
        }

        if nested {
            self.unsupported("funclets calling funclets");
        }
        if stray_ret {
            self.unsupported("a return outside a funclet");
        }

        // ptxas wants the table of return sites before its `brx.idx`
        let mut table = String::new();
        if !sites.is_empty() {
            let _ = writeln!(table, "$SITES: .branchtargets {};", sites.join(", "));
        }
        table + &o
    }

    fn build_module(&mut self) -> String {
        let place = self.placement();
        let body = self.lower_body(&place);
        let name = self.options.name.clone();
        let mut o = String::new();

        let _ = writeln!(o, "//");
        let _ = writeln!(
            o,
            "// Generated by sjb2ptx {} from a saved symjit application.",
            env!("CARGO_PKG_VERSION")
        );
        let _ = writeln!(
            o,
            "// {}: {} inputs, {} outputs, {} shared parameters (f64 slots), layout {}",
            name,
            self.shape.count_in,
            self.shape.count_out,
            self.shape.count_params,
            self.options.layout.name()
        );
        let _ = writeln!(o, "//");
        let _ = writeln!(o);
        let _ = writeln!(o, ".version {}", PTX_VERSION);
        let _ = writeln!(o, ".target sm_{}", self.options.sm);
        let _ = writeln!(o, ".address_size 64");
        let _ = writeln!(o);
        if !self.calls.is_empty() {
            let _ = writeln!(
                o,
                "// the math library (src/ptx/libm): {}",
                self.calls.join(", ")
            );
            o += &libm::functions(&self.calls);
        }
        let _ = writeln!(o, ".visible .entry {}(", name);
        let params: Vec<String> = ["in", "out", "params", "n", "ld_in", "ld_out"]
            .iter()
            .map(|p| format!("\t.param .u64 {}_{}", name, p))
            .collect();
        let _ = writeln!(o, "{})", params.join(",\n"));
        let _ = writeln!(o, "{{");
        let _ = writeln!(o, "\t.reg .pred %p<1>;");
        let _ = writeln!(o, "\t.reg .u32 %t<3>;");
        let _ = writeln!(o, "\t.reg .u32 %site;");
        let _ = writeln!(o, "\t.reg .u64 %i, %pin, %pout, %par, %ldi, %ldo, %a<2>;");
        let _ = writeln!(o, "\t.reg .b64 %ret, %tmp, %x<4>;");
        for (prefix, count) in [
            ("g", self.count_gen),
            ("s", self.count_stack),
            ("m", self.count_mem),
        ] {
            if count > 0 {
                let _ = writeln!(o, "\t.reg .b64 %{}<{}>;", prefix, count);
            }
        }
        let _ = writeln!(o);
        self.prologue(&mut o, &place);
        o += &body;
        self.epilogue(&mut o, &place);
        let _ = writeln!(o, "}}");
        o
    }
}

impl Generator for PtxGenerator {
    fn count_shadows(&self) -> u8 {
        0
    }

    fn three_address(&self) -> bool {
        true
    }

    fn bytes(&mut self) -> Vec<u8> {
        self.module.as_bytes().to_vec()
    }

    fn what(&self) -> GeneratorType {
        // symjit has no PTX generator type (and does not branch on `what`)
        GeneratorType::Unknown
    }

    fn seal(&mut self) {
        // the memory slots the interface needs, even if the MIR does not touch them
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
        self.module = self.build_module();
    }

    fn align(&mut self) {}

    fn set_label(&mut self, label: &str) {
        self.body.push(Item::Label(label.to_string()));
    }

    fn branch(&mut self, label: &str) {
        self.body.push(Item::Branch(label.to_string()));
    }

    // jumps when `cond` is a true mask if `is_else`, when it is false otherwise (the MIR's
    // `branch_if`, see CLAUDE.md)
    fn branch_if(&mut self, cond: Reg, label: &str, is_else: bool) {
        let cond = self.reg(cond);
        self.body.push(Item::BranchIf {
            cond,
            label: label.to_string(),
            if_true: is_else,
        });
    }

    /***********************************/
    fn fmov(&mut self, dst: Reg, s1: Reg) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        self.mov(d, a);
    }

    fn fxchg(&mut self, s1: Reg, s2: Reg) {
        let (a, b) = (self.reg(s1), self.reg(s2));
        if a != b {
            self.mov("%x0".into(), a.clone());
            self.mov(a, b.clone());
            self.mov(b, "%x0".into());
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

    // a complex value occupies two consecutive slots (re, im)
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

    // compression mode: the `_ultra` entry addresses stack slots at run time
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

    // neg, sign and abs on the sign bit, as x86 (`_minus_zero_` masks)
    fn neg(&mut self, dst: Reg, s1: Reg) {
        self.bits_imm("xor", dst, s1, SIGN_BIT);
    }

    fn sign(&mut self, dst: Reg, s1: Reg) {
        self.bits_imm("and", dst, s1, SIGN_BIT);
    }

    fn abs(&mut self, dst: Reg, s1: Reg) {
        self.bits_imm("and", dst, s1, NOT_SIGN_BIT);
    }

    fn abs2(&mut self, dst: Reg, s1: Reg) {
        self.binary("mul.rn.f64", dst, s1, s1);
    }

    // only generated for fast complex, which lowered MIR does not use; the interpreter's value
    fn times_i(&mut self, dst: Reg, _s1: Reg) {
        self.set_zero(dst);
    }

    fn times_neg_i(&mut self, dst: Reg, _s1: Reg) {
        self.set_zero(dst);
    }

    fn root(&mut self, dst: Reg, s1: Reg) {
        self.unary("sqrt.rn.f64", dst, s1);
    }

    fn real_root(&mut self, dst: Reg, s1: Reg) {
        self.unary("sqrt.rn.f64", dst, s1);
    }

    // correctly rounded, as x86's 1 / s1
    fn recip(&mut self, dst: Reg, s1: Reg) {
        self.unary("rcp.rn.f64", dst, s1);
    }

    fn half(&mut self, dst: Reg, s1: Reg) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        self.emit(format!("mul.rn.f64 {}, {}, 0d3FE0000000000000;", d, a));
    }

    // half to even, as x86's `roundsd` mode 0 (the interpreter rounds halves away from zero)
    fn round(&mut self, dst: Reg, s1: Reg) {
        self.unary("cvt.rni.f64.f64", dst, s1);
    }

    fn floor(&mut self, dst: Reg, s1: Reg) {
        self.unary("cvt.rmi.f64.f64", dst, s1);
    }

    fn ceiling(&mut self, dst: Reg, s1: Reg) {
        self.unary("cvt.rpi.f64.f64", dst, s1);
    }

    fn trunc(&mut self, dst: Reg, s1: Reg) {
        self.unary("cvt.rzi.f64.f64", dst, s1);
    }

    // s1 - floor(s1)
    fn frac(&mut self, dst: Reg, s1: Reg) {
        let (d, a) = (self.reg(dst), self.reg(s1));
        self.emit(format!("cvt.rmi.f64.f64 %x0, {};", a));
        self.emit(format!("sub.rn.f64 {}, {}, %x0;", d, a));
    }

    fn plus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("add.rn.f64", dst, s1, s2);
    }

    fn minus(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("sub.rn.f64", dst, s1, s2);
    }

    fn times(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("mul.rn.f64", dst, s1, s2);
    }

    fn divide(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("div.rn.f64", dst, s1, s2);
    }

    // (x1 + i y1)(x2 + i y2) with the formula of x86's scalar generator: x1 x2 - y1 y2 and
    // x2 y1 + x1 y2, each with one fused operation; its scratch registers are ours
    fn times_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let (xd, yd) = (self.reg(xd), self.reg(yd));
        let (x1, y1, x2, y2) = (self.reg(x1), self.reg(y1), self.reg(x2), self.reg(y2));
        self.emit(format!("mul.rn.f64 %x2, {}, {};", y1, y2));
        self.emit("neg.f64 %x2, %x2;".into());
        self.emit(format!("fma.rn.f64 %x2, {}, {}, %x2;", x1, x2));
        self.emit(format!("mul.rn.f64 %x3, {}, {};", x1, y2));
        self.emit(format!("fma.rn.f64 {}, {}, {}, %x3;", yd, x2, y1));
        self.mov(xd, "%x2".into());
        true
    }

    // (x1 + i y1) / (x2 + i y2) with the formula of x86's scalar generator: (x1 x2 + y1 y2)
    // and (x2 y1 - x1 y2) over x2^2 + y2^2, each with one fused operation
    fn divide_complex(&mut self, xd: Reg, yd: Reg, x1: Reg, y1: Reg, x2: Reg, y2: Reg) -> bool {
        let (xd, yd) = (self.reg(xd), self.reg(yd));
        let (x1, y1, x2, y2) = (self.reg(x1), self.reg(y1), self.reg(x2), self.reg(y2));
        self.emit(format!("mul.rn.f64 %x2, {}, {};", y1, y2));
        self.emit(format!("fma.rn.f64 %x2, {}, {}, %x2;", x1, x2));
        self.emit(format!("mul.rn.f64 %x3, {}, {};", x1, y2));
        self.emit("neg.f64 %x3, %x3;".into());
        self.emit(format!("fma.rn.f64 %x3, {}, {}, %x3;", x2, y1));
        self.emit(format!("mul.rn.f64 %x1, {}, {};", x2, x2));
        self.emit(format!("fma.rn.f64 %x1, {}, {}, %x1;", y2, y2));
        self.emit(format!("div.rn.f64 {}, %x2, %x1;", xd));
        self.emit(format!("div.rn.f64 {}, %x3, %x1;", yd));
        true
    }

    fn fuse_load_math(&mut self) {}

    fn support_times2(&self) -> bool {
        false
    }

    fn times2_loc(&mut self, _d1: Reg, _s1: Reg, _l1: Loc, _d2: Reg, _s2: Reg, _l2: Loc) {
        unreachable!("support_times2 is false");
    }

    // real-valued MIR (complex values are lowered by complexify)
    fn real(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn imaginary(&mut self, dst: Reg, _s1: Reg) {
        self.set_zero(dst);
    }

    fn conjugate(&mut self, dst: Reg, s1: Reg) {
        self.fmov(dst, s1);
    }

    fn complex(&mut self, dst: Reg, s1: Reg, _s2: Reg) {
        self.fmov(dst, s1);
    }

    // x86's predicates: `gt`/`geq` are `cmpnle`/`cmpnlt` (true if unordered), `neq` is
    // `cmpneq` (true if unordered), the others are ordered
    fn gt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare("gtu", dst, s1, s2);
    }

    fn geq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare("geu", dst, s1, s2);
    }

    fn lt(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare("lt", dst, s1, s2);
    }

    fn leq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare("le", dst, s1, s2);
    }

    fn eq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare("eq", dst, s1, s2);
    }

    fn neq(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.compare("neu", dst, s1, s2);
    }

    fn and(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("and.b64", dst, s1, s2);
    }

    // !s1 & s2
    fn andnot(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        let (d, a, b) = (self.reg(dst), self.reg(s1), self.reg(s2));
        self.emit(format!("not.b64 %x0, {};", a));
        self.emit(format!("and.b64 {}, %x0, {};", d, b));
    }

    fn or(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("or.b64", dst, s1, s2);
    }

    fn xor(&mut self, dst: Reg, s1: Reg, s2: Reg) {
        self.binary("xor.b64", dst, s1, s2);
    }

    fn not(&mut self, dst: Reg, s1: Reg) {
        self.unary("not.b64", dst, s1);
    }

    // s1 * s2 + s3
    fn fused_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, false, false);
    }

    // s1 * s2 - s3
    fn fused_mul_sub(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, false, true);
    }

    // -(s1 * s2) + s3
    fn fused_neg_mul_add(&mut self, dst: Reg, s1: Reg, s2: Reg, s3: Reg) {
        self.fused(dst, s1, s2, s3, true, false);
    }

    // -(s1 * s2) - s3
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

    // real functions: the arguments are Ret (and Temp), the result Ret
    fn call(&mut self, op: &str, num_args: usize) -> Result<()> {
        if op == "@self" {
            self.unsupported("recursive calls (`@self`)");
            return Ok(());
        }
        match libm::signature(op) {
            Some(sig) if !sig.complex && sig.count_params == num_args => {
                let args: Vec<String> = ["%ret", "%tmp"][..num_args]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                self.call_library(op, &args, false);
            }
            _ => self.unsupported(&format!("the function `{}`", op)),
        }
        Ok(())
    }

    // complex functions: the argument is (Ret, Temp), the second one (Gen(0), Gen(1)); the
    // result goes to (Ret, Temp)
    fn call_complex(&mut self, op: &str, num_args: usize) -> Result<()> {
        let name = if op.starts_with("cplx_") {
            op.to_string()
        } else {
            format!("cplx_{}", op)
        };
        match libm::signature(&name) {
            Some(sig) if sig.complex && sig.count_params == 2 * num_args => {
                let mut args = vec!["%ret".to_string(), "%tmp".to_string()];
                if num_args == 2 {
                    args.push(self.reg(Reg::Gen(0)));
                    args.push(self.reg(Reg::Gen(1)));
                }
                self.call_library(&name, &args, true);
            }
            _ => {
                let plain = name.strip_prefix("cplx_").unwrap_or(&name).to_string();
                self.unsupported(&format!("the complex function `{}`", plain));
            }
        }
        Ok(())
    }

    fn call_funclet(&mut self, label: &str) {
        let site = self.count_sites;
        self.count_sites += 1;
        self.body.push(Item::CallFunclet {
            label: label.to_string(),
            site,
        });
    }

    fn ret(&mut self) {
        self.body.push(Item::Ret);
    }

    // the fast kernel has no GPU counterpart and is not generated
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

    // the prologue and the epilogue are written in `seal`, when the registers are known
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

    // registers are virtual; nothing to preserve
    fn save_used_registers(&mut self, _used: &[Reg]) {}

    fn load_used_registers(&mut self, _used: &[Reg]) {}

    // (true_val & mask) | (false_val & !mask), mask = Stack[idx] (a comparison result)
    fn ifelse(&mut self, dst: Reg, true_val: Reg, false_val: Reg, idx: u32) {
        let (d, t, f) = (self.reg(dst), self.reg(true_val), self.reg(false_val));
        if t == f {
            self.mov(d, t);
            return;
        }
        let m = self.stack(idx);
        self.emit(format!("and.b64 %x0, {}, {};", t, m));
        self.emit(format!("not.b64 %x1, {};", m));
        self.emit(format!("and.b64 %x1, {}, %x1;", f));
        self.emit(format!("or.b64 {}, %x0, %x1;", d));
    }
}
