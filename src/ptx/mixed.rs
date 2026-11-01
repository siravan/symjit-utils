// Mixed precision (`PtxOptions::mixed`): each value is an f32 or an f64.
//
// Rational kernels (QFT currents and amplitudes) lose accuracy near their poles, where a
// denominator such as p^2 - m^2 cancels. The pass keeps the denominators in f64 and computes the
// rest in f32. It retypes the body the f64 generator wrote (`Item`s whose lines are f64
// instructions), before the placement of the memory slots:
//
// - Each definition gets a type. Divisions and reciprocals are f64. With `MixedRule::Need`, so
//   is every value a divisor, a reciprocal's operand or a math-library argument depends on
//   (the backward slice of the denominators); every other operation is f32. With
//   `MixedRule::Forward`, an operation is f64 if one of its operands is f64, else f32.
// - Inputs: the slots `f64_inputs` marks are f64 (momenta, parameters); the others are rounded
//   to f32 where they are loaded. Constants exist in both types.
// - An operand of the other type is converted where it is used: `cvt.f64.f32` (exact) or
//   `cvt.rn.f32.f64`, into the register of the other type of the same name (`%g3` is the f64
//   register of `%fg3`). Masks (comparison results) are sign-extended or truncated instead.
// - The arena stays f64. Outputs are rounded to f32 when they are stored (`round_outputs`), as
//   an arena of f32 planes would hold them.
// - Labels, jumps, funclet calls and returns: every value is converted to f64 before them, and
//   is f64 after them (the analysis does not follow control flow). The SSA values of ssa2ptx
//   (`%v<k>`) are not: none is read across a label or call.
//
// `retype` makes three passes. The generator reuses register names (the PTX is not in SSA form),
// so types belong to definitions, not to names:
// 1. forward: parse the ops; each definition becomes a node (an SSA value), and each op records
//    its seeds (the operands it needs in f64);
// 2. backward (`need` only): one reverse sweep marks each op's seeds and, for a marked definition,
//    its operands: the backward slices of the denominators;
// 3. forward: the types of the nodes each op adds, then the op, with conversions where an operand
//    is not of the type it is used in.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};

use super::Item;

/// How the types of the operations are chosen (see the top of this file).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedRule {
    /// f64 only in the backward slices of the denominators
    Need,
    /// f64 if an operand is f64 (the outputs of divisions, reciprocals and the math library are)
    Forward,
}

impl MixedRule {
    pub fn name(self) -> &'static str {
        match self {
            MixedRule::Need => "need",
            MixedRule::Forward => "forward",
        }
    }

    pub fn parse(s: &str) -> Option<MixedRule> {
        match s {
            "need" => Some(MixedRule::Need),
            "forward" => Some(MixedRule::Forward),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mixed {
    pub rule: MixedRule,
    /// per input slot, whether it is f64 (missing: f32)
    pub f64_inputs: Vec<bool>,
    /// round the outputs to f32 (f64 arena)
    pub round_outputs: bool,
    /// a mixed arena (plane kernels): the f32 inputs and the outputs are f32 slots of `in` and
    /// `out` (4 bytes), the f64 inputs f64 slots of a second arena, `params` (8 bytes), at the
    /// same planes and leading dimension
    pub arena: bool,
}

impl Mixed {
    pub fn new(rule: MixedRule, f64_inputs: Vec<bool>) -> Mixed {
        Mixed {
            rule,
            f64_inputs,
            round_outputs: true,
            arena: false,
        }
    }
}

/// What the pass did: value operations by type (moves excluded) and the conversions it added.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MixedCounts {
    pub f64_ops: usize,
    pub f32_ops: usize,
    pub conversions: usize,
}

/// The f32 register of the f64 register `name` (`%g3` -> `%fg3`).
pub fn f32_name(name: &str) -> String {
    format!("%f{}", &name[1..])
}

/// A `Static` register of an SSA-form MIR (`%v<k>`, `crate::ssa`).
fn ssa_value(name: &str) -> bool {
    name.strip_prefix("%v")
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

/// A register the generator allocates values in (`%v<k>`: the `Static` registers of ssa2ptx).
fn tracked(name: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    name == "%ret"
        || name == "%tmp"
        || ["%g", "%v", "%s", "%m", "%x"]
            .iter()
            .any(|p| name.strip_prefix(p).is_some_and(digits))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Float,
    Mask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    F32,
    F64,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Arith,
    Div,
    Mov,
    Bits,
}

#[derive(Clone, Debug)]
enum Opnd {
    Reg(String),
    F(f64),
    Imm(String),
}

#[derive(Clone, Debug)]
enum Op {
    Line(String),
    Value {
        mnem: String,
        dst: String,
        srcs: Vec<Opnd>,
        class: Class,
    },
    /// `dst := 0` in both types (`xor d, a, a`)
    Zero(String),
    Setp {
        mnem: String,
        pred: String,
        srcs: Vec<Opnd>,
    },
    Selp {
        dst: String,
        rest: String,
    },
    StParam {
        addr: String,
        src: String,
    },
    /// an f64 definition the pass does not retype (`ld.param`, `ld.global`)
    Load64 {
        line: String,
        dst: String,
    },
    Const(String, u32),
    LoadMem(String, u32),
    SaveMem(String, u32),
    LoadStack(String, u32),
    SaveStack(String, u32),
    Control(Item),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    Fixed64,
    Input32,
    Flexible,
    Store64,
    /// an output stored into an f32 slot (mixed arena)
    Store32,
    Mov,
    Div,
    Other,
}

struct Node {
    srcs: Vec<usize>,
    kind: Kind,
    base: Base,
}

fn parse_operand(s: &str) -> Opnd {
    if s.starts_with('%') {
        Opnd::Reg(s.to_string())
    } else if let Some(hex) = s.strip_prefix("0d") {
        Opnd::F(f64::from_bits(u64::from_str_radix(hex, 16).unwrap_or(0)))
    } else {
        Opnd::Imm(s.to_string())
    }
}

fn parse_line(line: &str) -> Result<Op> {
    let text = line.trim();
    let mentions = text
        .split(|c: char| !(c == '%' || c.is_ascii_alphanumeric()))
        .any(tracked);
    let Some(body) = text.strip_suffix(';') else {
        // braces and declarations of the library calls
        if mentions {
            bail!("mixed precision cannot type `{}`", text);
        }
        return Ok(Op::Line(line.to_string()));
    };
    let (mnem, rest) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
    let operands: Vec<&str> = rest.split(',').map(str::trim).collect();
    let mnem = mnem.to_string();
    if mnem.starts_with("st.param") {
        return Ok(Op::StParam {
            addr: operands[0].to_string(),
            src: operands[1].to_string(),
        });
    }
    if mnem.starts_with("ld.param") || mnem.starts_with("ld.global") {
        if tracked(operands[0]) {
            return Ok(Op::Load64 {
                line: line.to_string(),
                dst: operands[0].to_string(),
            });
        }
        return Ok(Op::Line(line.to_string()));
    }
    if mnem.starts_with("setp.") && mnem.ends_with(".f64") {
        return Ok(Op::Setp {
            mnem,
            pred: operands[0].to_string(),
            srcs: operands[1..].iter().map(|s| parse_operand(s)).collect(),
        });
    }
    if mnem == "selp.b64" {
        return Ok(Op::Selp {
            dst: operands[0].to_string(),
            rest: operands[1..].join(", "),
        });
    }
    let value = (mnem.ends_with(".f64") || mnem.ends_with(".b64")) && tracked(operands[0]);
    if !value {
        if mentions {
            bail!("mixed precision cannot type `{}`", text);
        }
        return Ok(Op::Line(line.to_string()));
    }
    let dst = operands[0].to_string();
    let srcs: Vec<Opnd> = operands[1..].iter().map(|s| parse_operand(s)).collect();
    if mnem == "xor.b64" {
        if let [Opnd::Reg(a), Opnd::Reg(b)] = &srcs[..] {
            if a == b {
                return Ok(Op::Zero(dst));
            }
        }
    }
    let class = if mnem.starts_with("div.") || mnem.starts_with("rcp.") {
        Class::Div
    } else if mnem.starts_with("mov.") {
        Class::Mov
    } else if ["and.", "or.", "xor.", "not."]
        .iter()
        .any(|p| mnem.starts_with(p))
    {
        Class::Bits
    } else {
        Class::Arith
    };
    for s in &srcs {
        if let Opnd::Reg(r) = s {
            if !tracked(r) {
                bail!("mixed precision cannot type `{}`", text);
            }
        }
    }
    Ok(Op::Value {
        mnem,
        dst,
        srcs,
        class,
    })
}

fn parse(body: &[Item]) -> Result<Vec<Op>> {
    body.iter()
        .map(|item| {
            Ok(match item {
                Item::Line(l) => parse_line(l)?,
                Item::Const(r, i) => Op::Const(r.clone(), *i),
                Item::LoadMem(r, i) => Op::LoadMem(r.clone(), *i),
                Item::SaveMem(r, i) => Op::SaveMem(r.clone(), *i),
                Item::LoadStack(r, i) => Op::LoadStack(r.clone(), *i),
                Item::SaveStack(r, i) => Op::SaveStack(r.clone(), *i),
                other => Op::Control(other.clone()),
            })
        })
        .collect()
}

/// The pass's view of the values: definitions as nodes, names bound to their last definition.
struct Graph {
    nodes: Vec<Node>,
    cur: HashMap<String, usize>,
    /// the kind of each name's last definition (kept across control items)
    last_kind: HashMap<String, Kind>,
}

impl Graph {
    fn def(&mut self, name: &str, node: Node) -> usize {
        self.last_kind.insert(name.to_string(), node.kind);
        self.nodes.push(node);
        let n = self.nodes.len() - 1;
        self.cur.insert(name.to_string(), n);
        n
    }

    /// The definition `name` holds; an f64 value of unknown origin after a control item.
    fn get(&mut self, name: &str) -> usize {
        if let Some(&n) = self.cur.get(name) {
            return n;
        }
        let kind = *self.last_kind.get(name).unwrap_or(&Kind::Float);
        self.def(
            name,
            Node {
                srcs: vec![],
                kind,
                base: Base::Fixed64,
            },
        )
    }
}

/// Per op: the nodes of its register operands (in order) and its definition.
struct Rec {
    srcs: Vec<usize>,
    def: Option<usize>,
}

/// Per op, for the two sweeps: its `Rec`, the operands it makes f64 (a divisor, a reciprocal's
/// operand, a library argument) and the number of nodes once it is parsed.
struct Step {
    rec: Rec,
    seeds: Vec<usize>,
    end: usize,
}

/// Retypes `body` (see the top of this file). `count_in`: the input slots (memory slots
/// `0 .. count_in`).
pub fn retype(body: &[Item], mixed: &Mixed, count_in: usize) -> Result<(Vec<Item>, MixedCounts)> {
    let ops = parse(body)?;
    let mut g = Graph {
        nodes: Vec::new(),
        cur: HashMap::new(),
        last_kind: HashMap::new(),
    };
    // 1. forward: the nodes (one per definition, i.e. SSA values) and each op's seeds
    let mut steps = Vec::with_capacity(ops.len());
    for op in &ops {
        let mut seeds = Vec::new();
        let rec = match op {
            Op::Line(_) => Rec {
                srcs: vec![],
                def: None,
            },
            Op::Value {
                mnem,
                dst,
                srcs,
                class,
            } => {
                let s: Vec<usize> = srcs
                    .iter()
                    .filter_map(|o| match o {
                        Opnd::Reg(r) => Some(g.get(r)),
                        _ => None,
                    })
                    .collect();
                let has_imm = srcs.iter().any(|o| !matches!(o, Opnd::Reg(_)));
                let (base, kind) = match class {
                    Class::Mov if s.is_empty() => (Base::Flexible, Kind::Float),
                    Class::Mov => (Base::Mov, g.nodes[s[0]].kind),
                    Class::Div => (Base::Div, Kind::Float),
                    Class::Bits if !has_imm && s.iter().all(|&n| g.nodes[n].kind == Kind::Mask) => {
                        (Base::Other, Kind::Mask)
                    }
                    _ => (Base::Other, Kind::Float),
                };
                if *class == Class::Div {
                    // the divisor, or the reciprocal's operand
                    let divisor = if mnem.starts_with("rcp.") { 0 } else { 1 };
                    if let Some(&d) = s.get(divisor) {
                        seeds.push(d);
                    }
                }
                let def = g.def(
                    dst,
                    Node {
                        srcs: s.clone(),
                        kind,
                        base,
                    },
                );
                Rec {
                    srcs: s,
                    def: Some(def),
                }
            }
            Op::Zero(dst) => {
                let def = g.def(
                    dst,
                    Node {
                        srcs: vec![],
                        kind: Kind::Float,
                        base: Base::Flexible,
                    },
                );
                Rec {
                    srcs: vec![],
                    def: Some(def),
                }
            }
            Op::Setp { srcs, .. } => Rec {
                srcs: srcs
                    .iter()
                    .filter_map(|o| match o {
                        Opnd::Reg(r) => Some(g.get(r)),
                        _ => None,
                    })
                    .collect(),
                def: None,
            },
            Op::Selp { dst, .. } => {
                let def = g.def(
                    dst,
                    Node {
                        srcs: vec![],
                        kind: Kind::Mask,
                        base: Base::Other,
                    },
                );
                Rec {
                    srcs: vec![],
                    def: Some(def),
                }
            }
            Op::StParam { src, .. } => {
                let s = g.get(src);
                seeds.push(s);
                Rec {
                    srcs: vec![s],
                    def: None,
                }
            }
            Op::Load64 { dst, .. } => {
                let def = g.def(
                    dst,
                    Node {
                        srcs: vec![],
                        kind: Kind::Float,
                        base: Base::Fixed64,
                    },
                );
                Rec {
                    srcs: vec![],
                    def: Some(def),
                }
            }
            Op::Const(dst, _) => {
                let def = g.def(
                    dst,
                    Node {
                        srcs: vec![],
                        kind: Kind::Float,
                        base: Base::Flexible,
                    },
                );
                Rec {
                    srcs: vec![],
                    def: Some(def),
                }
            }
            Op::LoadMem(dst, idx) => {
                let k = *idx as usize;
                let (srcs, base, kind) = if k < count_in {
                    let f64_input = mixed.f64_inputs.get(k).copied().unwrap_or(false);
                    let base = if f64_input {
                        Base::Fixed64
                    } else {
                        Base::Input32
                    };
                    (vec![], base, Kind::Float)
                } else {
                    let s = g.get(&format!("%m{}", idx));
                    (vec![s], Base::Mov, g.nodes[s].kind)
                };
                let def = g.def(
                    dst,
                    Node {
                        srcs: srcs.clone(),
                        kind,
                        base,
                    },
                );
                Rec {
                    srcs,
                    def: Some(def),
                }
            }
            Op::SaveMem(src, idx) => {
                let s = g.get(src);
                let kind = g.nodes[s].kind;
                let def = g.def(
                    &format!("%m{}", idx),
                    Node {
                        srcs: vec![s],
                        kind,
                        base: if mixed.arena {
                            Base::Store32
                        } else {
                            Base::Store64
                        },
                    },
                );
                Rec {
                    srcs: vec![s],
                    def: Some(def),
                }
            }
            Op::LoadStack(dst, idx) => {
                let s = g.get(&format!("%s{}", idx));
                let kind = g.nodes[s].kind;
                let def = g.def(
                    dst,
                    Node {
                        srcs: vec![s],
                        kind,
                        base: Base::Mov,
                    },
                );
                Rec {
                    srcs: vec![s],
                    def: Some(def),
                }
            }
            Op::SaveStack(src, idx) => {
                let s = g.get(src);
                let kind = g.nodes[s].kind;
                let def = g.def(
                    &format!("%s{}", idx),
                    Node {
                        srcs: vec![s],
                        kind,
                        base: Base::Mov,
                    },
                );
                Rec {
                    srcs: vec![s],
                    def: Some(def),
                }
            }
            Op::Control(item) => {
                if !matches!(item, Item::BranchIf { .. }) {
                    g.cur.clear();
                } else if let Item::BranchIf { cond, .. } = item {
                    g.get(cond);
                }
                Rec {
                    srcs: vec![],
                    def: None,
                }
            }
        };
        steps.push(Step {
            rec,
            seeds,
            end: g.nodes.len(),
        });
    }

    // 2. backward: the slices of the seeds. An op comes after the definitions of its operands,
    // so one reverse sweep both records its seeds and passes a definition's flag on to its operands.
    let mut need = vec![false; g.nodes.len()];
    if mixed.rule == MixedRule::Need {
        for step in steps.iter().rev() {
            for &s in &step.seeds {
                need[s] = true;
            }
            if let Some(d) = step.rec.def {
                if need[d] {
                    for &s in &g.nodes[d].srcs {
                        need[s] = true;
                    }
                }
            }
        }
    }

    // 3. forward: the types of the nodes an op adds (its operands are typed already), then the op
    let mut ty = Vec::with_capacity(g.nodes.len());
    let mut e = Emitter {
        out: Vec::with_capacity(body.len() * 5 / 4),
        valid: HashMap::new(),
        counts: MixedCounts::default(),
        kinds: HashMap::new(),
    };
    for (op, step) in ops.into_iter().zip(&steps) {
        while ty.len() < step.end {
            let n = ty.len();
            let node = &g.nodes[n];
            let t = match node.base {
                Base::Fixed64 | Base::Store64 | Base::Div => Ty::F64,
                Base::Input32 | Base::Store32 => Ty::F32,
                Base::Flexible => Ty::Both,
                Base::Mov => ty[node.srcs[0]],
                Base::Other => match mixed.rule {
                    MixedRule::Need if need[n] => Ty::F64,
                    MixedRule::Need => Ty::F32,
                    MixedRule::Forward if node.srcs.iter().any(|&s| ty[s] == Ty::F64) => Ty::F64,
                    MixedRule::Forward => Ty::F32,
                },
            };
            ty.push(t);
        }
        e.op(op, &step.rec, &g, &ty, mixed)?;
    }
    Ok((e.out, e.counts))
}

struct Emitter {
    out: Vec<Item>,
    /// per name, whether its f32 and its f64 register hold its value
    valid: HashMap<String, (bool, bool)>,
    /// per name, the kind of its value
    kinds: HashMap<String, Kind>,
    counts: MixedCounts,
}

fn variant(name: &str, t: Ty) -> String {
    if t == Ty::F32 {
        f32_name(name)
    } else {
        name.to_string()
    }
}

fn retype_mnemonic(mnem: &str, t: Ty) -> String {
    if t == Ty::F32 {
        mnem.replace(".f64", ".f32").replace(".b64", ".b32")
    } else {
        mnem.to_string()
    }
}

fn immediate(o: &Opnd, t: Ty) -> String {
    match o {
        Opnd::F(v) if t == Ty::F32 => format!("0f{:08X}", (*v as f32).to_bits()),
        Opnd::F(v) => format!("0d{:016X}", v.to_bits()),
        Opnd::Imm(s) if t == Ty::F32 => match s.as_str() {
            "0x8000000000000000" => "0x80000000".to_string(),
            "0x7FFFFFFFFFFFFFFF" => "0x7FFFFFFF".to_string(),
            s => s.to_string(),
        },
        Opnd::Imm(s) => s.clone(),
        Opnd::Reg(r) => r.clone(),
    }
}

impl Emitter {
    fn line(&mut self, s: String) {
        self.out.push(Item::Line(s));
    }

    fn set(&mut self, name: &str, t: Ty, kind: Kind) {
        let flags = match t {
            Ty::F32 => (true, false),
            Ty::F64 => (false, true),
            Ty::Both => (true, true),
        };
        self.valid.insert(name.to_string(), flags);
        self.kinds.insert(name.to_string(), kind);
    }

    /// The register of `name` of type `t` (F32 or F64), converted into if it does not hold the
    /// value yet.
    fn want(&mut self, name: &str, t: Ty, kind: Kind) -> String {
        let (has32, has64) = *self.valid.entry(name.to_string()).or_insert((false, true));
        let f = f32_name(name);
        match t {
            Ty::F32 if !has32 => {
                let cvt = match kind {
                    Kind::Float => "cvt.rn.f32.f64",
                    Kind::Mask => "cvt.u32.u64",
                };
                self.line(format!("{} {}, {};", cvt, f, name));
                self.counts.conversions += 1;
                self.valid.insert(name.to_string(), (true, has64));
            }
            Ty::F64 if !has64 => {
                let cvt = match kind {
                    Kind::Float => "cvt.f64.f32",
                    Kind::Mask => "cvt.s64.s32",
                };
                self.line(format!("{} {}, {};", cvt, name, f));
                self.counts.conversions += 1;
                self.valid.insert(name.to_string(), (has32, true));
            }
            _ => {}
        }
        variant(name, t)
    }

    fn op(&mut self, op: Op, rec: &Rec, g: &Graph, ty: &[Ty], mixed: &Mixed) -> Result<()> {
        let kind_of = |n: usize| g.nodes[n].kind;
        match op {
            Op::Line(l) => self.out.push(Item::Line(l)),
            Op::Value {
                mnem,
                dst,
                srcs,
                class,
            } => {
                let def = rec
                    .def
                    .ok_or_else(|| anyhow!("mixed: a value without a definition"))?;
                let (t, kind) = (ty[def], kind_of(def));
                if t == Ty::Both {
                    // a constant or a copy of one
                    let (a64, a32) = match &srcs[0] {
                        Opnd::Reg(r) => (r.clone(), f32_name(r)),
                        o => (immediate(o, Ty::F64), immediate(o, Ty::F32)),
                    };
                    self.line(format!("mov.b64 {}, {};", dst, a64));
                    self.line(format!("mov.b32 {}, {};", f32_name(&dst), a32));
                    self.set(&dst, Ty::Both, kind);
                    return Ok(());
                }
                let mut operands = Vec::new();
                let mut reg = rec.srcs.iter();
                for o in &srcs {
                    match o {
                        Opnd::Reg(r) => {
                            let n = *reg.next().unwrap();
                            operands.push(self.want(r, t, kind_of(n)));
                        }
                        o => operands.push(immediate(o, t)),
                    }
                }
                self.line(format!(
                    "{} {}, {};",
                    retype_mnemonic(&mnem, t),
                    variant(&dst, t),
                    operands.join(", ")
                ));
                if class != Class::Mov {
                    match t {
                        Ty::F64 => self.counts.f64_ops += 1,
                        _ => self.counts.f32_ops += 1,
                    }
                }
                self.set(&dst, t, kind);
            }
            Op::Zero(dst) => {
                self.line(format!("mov.b64 {}, 0d0000000000000000;", dst));
                self.line(format!("mov.b32 {}, 0f00000000;", f32_name(&dst)));
                self.set(&dst, Ty::Both, Kind::Float);
            }
            Op::Setp { mnem, pred, srcs } => {
                let t = if rec.srcs.iter().any(|&n| ty[n] == Ty::F64) {
                    Ty::F64
                } else {
                    Ty::F32
                };
                let mut operands = Vec::new();
                let mut reg = rec.srcs.iter();
                for o in &srcs {
                    match o {
                        Opnd::Reg(r) => {
                            let n = *reg.next().unwrap();
                            operands.push(self.want(r, t, kind_of(n)));
                        }
                        o => operands.push(immediate(o, t)),
                    }
                }
                self.line(format!(
                    "{} {}, {};",
                    retype_mnemonic(&mnem, t),
                    pred,
                    operands.join(", ")
                ));
                match t {
                    Ty::F64 => self.counts.f64_ops += 1,
                    _ => self.counts.f32_ops += 1,
                }
            }
            Op::Selp { dst, rest } => {
                let t = ty[rec.def.unwrap()];
                self.line(format!(
                    "{} {}, {};",
                    retype_mnemonic("selp.b64", t),
                    variant(&dst, t),
                    rest
                ));
                self.set(&dst, t, Kind::Mask);
            }
            Op::StParam { addr, src } => {
                let s = self.want(&src, Ty::F64, kind_of(rec.srcs[0]));
                self.line(format!("st.param.b64 {}, {};", addr, s));
            }
            Op::Load64 { line, dst } => {
                self.out.push(Item::Line(line));
                self.set(&dst, Ty::F64, Kind::Float);
            }
            Op::Const(dst, idx) => {
                self.out.push(Item::Const(dst.clone(), idx));
                self.out.push(Item::Const(f32_name(&dst), idx));
                self.set(&dst, Ty::Both, Kind::Float);
            }
            Op::LoadMem(dst, idx) if mixed.arena => {
                // f32 slots load into the f32 register; so does an output stored before
                let def = rec.def.unwrap();
                let slot = format!("%m{}", idx);
                let stored32 = self.valid.get(&slot) == Some(&(true, false));
                if g.nodes[def].base == Base::Input32 || (rec.srcs.len() == 1 && stored32) {
                    self.out.push(Item::LoadMem(f32_name(&dst), idx));
                    self.set(&dst, Ty::F32, kind_of(def));
                } else {
                    self.out.push(Item::LoadMem(dst.clone(), idx));
                    self.set(&dst, Ty::F64, kind_of(def));
                }
            }
            Op::SaveMem(src, idx) if mixed.arena => {
                let n = rec.srcs[0];
                let s = self.want(&src, Ty::F32, kind_of(n));
                self.out.push(Item::SaveMem(s, idx));
                self.set(&format!("%m{}", idx), Ty::F32, kind_of(n));
            }
            Op::LoadMem(dst, idx) => {
                let def = rec.def.unwrap();
                self.out.push(Item::LoadMem(dst.clone(), idx));
                self.set(&dst, Ty::F64, kind_of(def));
                if g.nodes[def].base == Base::Input32 {
                    self.line(format!("cvt.rn.f32.f64 {}, {};", f32_name(&dst), dst));
                    self.counts.conversions += 1;
                    self.set(&dst, Ty::F32, Kind::Float);
                }
            }
            Op::SaveMem(src, idx) => {
                let n = rec.srcs[0];
                let s = self.want(&src, Ty::F64, kind_of(n));
                if mixed.round_outputs && ty[n] != Ty::F32 && kind_of(n) == Kind::Float {
                    let f = f32_name(&src);
                    self.line(format!("cvt.rn.f32.f64 {}, {};", f, src));
                    self.line(format!("cvt.f64.f32 {}, {};", src, f));
                    self.counts.conversions += 2;
                    self.valid.insert(src.clone(), (true, true));
                }
                self.out.push(Item::SaveMem(s, idx));
                self.set(&format!("%m{}", idx), Ty::F64, kind_of(n));
            }
            Op::LoadStack(dst, idx) => {
                let slot = format!("%s{}", idx);
                let (has32, has64) = *self.valid.entry(slot).or_insert((false, true));
                if has64 {
                    self.out.push(Item::LoadStack(dst.clone(), idx));
                }
                if has32 {
                    self.out.push(Item::LoadStack(f32_name(&dst), idx));
                }
                let kind = kind_of(rec.def.unwrap());
                self.valid.insert(dst.clone(), (has32, has64));
                self.kinds.insert(dst, kind);
            }
            Op::SaveStack(src, idx) => {
                let n = rec.srcs[0];
                let t = ty[n];
                let (has32, has64) = *self.valid.entry(src.clone()).or_insert((false, true));
                let (w32, w64) = match t {
                    Ty::Both => (has32, has64),
                    Ty::F32 => (true, false),
                    Ty::F64 => (false, true),
                };
                if w64 {
                    let s = self.want(&src, Ty::F64, kind_of(n));
                    self.out.push(Item::SaveStack(s, idx));
                }
                if w32 {
                    let s = self.want(&src, Ty::F32, kind_of(n));
                    self.out.push(Item::SaveStack(s, idx));
                }
                let slot = format!("%s{}", idx);
                self.valid.insert(slot.clone(), (w32, w64));
                self.kinds.insert(slot, kind_of(n));
            }
            Op::Control(item) => {
                // every value in its f64 register, but the SSA values (`%v<k>`, ssa2ptx): the
                // SSA form reads none across a label or call (a value that crosses one stays in
                // a general register, `crate::ssa`), so they need no conversion and keep theirs
                let mut names: Vec<String> = self
                    .valid
                    .iter()
                    .filter(|(n, &(_, has64))| !has64 && !ssa_value(n))
                    .map(|(n, _)| n.clone())
                    .collect();
                names.sort();
                for name in names {
                    let kind = *self.kinds.get(&name).unwrap_or(&Kind::Float);
                    self.want(&name, Ty::F64, kind);
                }
                let clear = !matches!(item, Item::BranchIf { .. });
                self.out.push(item);
                if clear {
                    self.valid.retain(|n, _| ssa_value(n));
                }
            }
        }
        Ok(())
    }
}
