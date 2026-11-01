// SSA form of a saved model's MIR (`sjb2ssa`, `ssa2ptx`) and the `.ssa` file that holds it.
//
// `to_ssa` is symjit's `GreedyAllocator::create` (the allocator's first pass, allocator.rs):
// each definition of a general register `Gen(r)` gets a new `Static(k)` register, and each use
// the `Static` its register holds. `Ret`, `Temp`, `Left` and `Right` stay as they are (library
// calls and funclets pass values in them), and so do the stack and memory slots. As in
// `create`, a label or a call forgets the registers: in builder MIR no general register lives
// across them. The MIR of a lowered complex model does pass values across them (the funclets
// of `Complexifier`, e.g. `@complex_root`, read `Gen` registers set before the call), so a
// register read before any definition since the last label or call is *pinned*. The
// definition of a pinned register that is current where control may leave the straight-line
// code (a label, call, jump or the end) is *kept*: it writes `Gen(r)`, and reads see `Gen(r)`
// until the next definition, so the register holds the original MIR's value at every jump and
// label. Its other definitions (overwritten before control leaves) get a `Static`.
//
// Complex models are lowered (`Complexifier`) before the conversion: it cannot read `Static`
// registers. Compression (`LoadArgs`, `SaveArgs`) is rejected, as by sjb2ptx.
//
// symjit's code generators and interpreter cannot run `Static` registers, so an `.ssa` file
// cannot be loaded as an application: it has its own magic number. Its layout is that of a
// `.sjb` (`Application::save`) otherwise:
//   SSA_MAGIC, version (of the program and MIR formats, 4), flags (bit 0: the MIR is the
//   lowered complex MIR), the program (`Program::save`), the MIR (`Mir::save`), the real
//   locations of a complex model (as `.sjb` files hold them).

use std::collections::HashSet;
use std::io::{Read, Write};

use anyhow::{anyhow, bail, Result};
use symjit::backend::{Loc, Mir, MirInstruction as Instruction, Reg};
use symjit::{CompilerType, Config, Program, Storage};

use crate::ptx::{Ptx, PtxOptions};
use crate::sjb::SavedModel;

/// The first word of an `.ssa` file (a `.sjb` starts with symjit's `Application` magic).
pub const SSA_MAGIC: usize = 0x5353_4132_7469_6a73;
/// The version of the program and MIR formats written (symjit's `Application::save`).
const VERSION: usize = 4;
/// `flags`: the MIR is the lowered MIR of a complex model.
const FLAG_LOWERED: usize = 1;

/// What the conversion did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SsaStats {
    pub instructions: usize,
    /// `Static` registers created (definitions of unpinned registers)
    pub statics: usize,
    /// the general registers that hold values across labels or calls (pinned)
    pub pinned: Vec<u8>,
    /// the definitions of pinned registers kept as they are (the last before a label, call,
    /// jump or end); the others are `Static`
    pub kept: usize,
}

/// The register operands an instruction reads and writes, as `create` consumes and produces
/// them (reads first). `None` for instructions without registers; an error for compression.
pub(crate) fn operands(ins: &Instruction) -> Result<(Vec<Reg>, Vec<Reg>)> {
    use Instruction::*;
    Ok(match ins {
        Uni { dst, s1, .. } | Mov { dst, s1 } => (vec![*s1], vec![*dst]),
        Bi { dst, s1, s2, .. } => (vec![*s1, *s2], vec![*dst]),
        LoadConst { dst, .. } | Load { dst, .. } => (vec![], vec![*dst]),
        Save { src, .. } => (vec![*src], vec![]),
        LoadComplex { xd, yd, .. } => (vec![], vec![*xd, *yd]),
        SaveComplex { xs, ys, .. } => (vec![*xs, *ys], vec![]),
        Fused { dst, a, b, c, .. } => (vec![*a, *b, *c], vec![*dst]),
        IfElse {
            dst,
            true_val,
            false_val,
            ..
        } => (vec![*true_val, *false_val], vec![*dst]),
        BranchIf { cond, .. } => (vec![*cond], vec![]),
        LoadMath { dst, s1, .. } | LoadConstMath { dst, s1, .. } => (vec![*s1], vec![*dst]),
        ComplexBi {
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
            ..
        } => (vec![*x1, *y1, *x2, *y2], vec![*xd, *yd]),
        Nop | End | Call { .. } | Label { .. } | Branch { .. } => (vec![], vec![]),
        LoadArgs { .. } | SaveArgs { .. } => {
            bail!("compression mode (LoadArgs, SaveArgs) has no SSA form")
        }
    })
}

/// Whether control may leave the straight-line code at the instruction (or after it).
fn leaves(ins: &Instruction) -> bool {
    matches!(
        ins,
        Instruction::Call { .. }
            | Instruction::Label { .. }
            | Instruction::Branch { .. }
            | Instruction::BranchIf { .. }
            | Instruction::End
    )
}

/// Whether the instruction forgets the registers (`create`'s `reset_regs`).
fn resets(ins: &Instruction) -> bool {
    matches!(ins, Instruction::Call { .. } | Instruction::Label { .. })
}

/// `ins` with its registers mapped by `f(reg, is_write)`, reads before writes (as `create`).
fn rewrite(ins: Instruction, mut f: impl FnMut(Reg, bool) -> Reg) -> Instruction {
    use Instruction::*;
    match ins {
        Uni { op, dst, s1 } => {
            let s1 = f(s1, false);
            Uni {
                op,
                dst: f(dst, true),
                s1,
            }
        }
        Mov { dst, s1 } => {
            let s1 = f(s1, false);
            Mov {
                dst: f(dst, true),
                s1,
            }
        }
        Bi { op, dst, s1, s2 } => {
            let (s1, s2) = (f(s1, false), f(s2, false));
            Bi {
                op,
                dst: f(dst, true),
                s1,
                s2,
            }
        }
        LoadConst { dst, idx } => LoadConst {
            dst: f(dst, true),
            idx,
        },
        Load { dst, loc } => Load {
            dst: f(dst, true),
            loc,
        },
        Save { src, loc } => Save {
            src: f(src, false),
            loc,
        },
        LoadComplex { xd, yd, loc } => {
            let xd = f(xd, true);
            LoadComplex {
                xd,
                yd: f(yd, true),
                loc,
            }
        }
        SaveComplex { xs, ys, loc } => {
            let xs = f(xs, false);
            SaveComplex {
                xs,
                ys: f(ys, false),
                loc,
            }
        }
        Fused { op, dst, a, b, c } => {
            let (a, b, c) = (f(a, false), f(b, false), f(c, false));
            Fused {
                op,
                dst: f(dst, true),
                a,
                b,
                c,
            }
        }
        IfElse {
            dst,
            true_val,
            false_val,
            cond,
        } => {
            let (true_val, false_val) = (f(true_val, false), f(false_val, false));
            IfElse {
                dst: f(dst, true),
                true_val,
                false_val,
                cond,
            }
        }
        BranchIf {
            cond,
            label,
            is_else,
        } => BranchIf {
            cond: f(cond, false),
            label,
            is_else,
        },
        LoadMath { op, dst, s1, loc } => {
            let s1 = f(s1, false);
            LoadMath {
                op,
                dst: f(dst, true),
                s1,
                loc,
            }
        }
        LoadConstMath { op, dst, s1, idx } => {
            let s1 = f(s1, false);
            LoadConstMath {
                op,
                dst: f(dst, true),
                s1,
                idx,
            }
        }
        ComplexBi {
            op,
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
        } => {
            let (x1, y1, x2, y2) = (f(x1, false), f(y1, false), f(x2, false), f(y2, false));
            let xd = f(xd, true);
            ComplexBi {
                op,
                xd,
                yd: f(yd, true),
                x1,
                y1,
                x2,
                y2,
            }
        }
        other => other,
    }
}

/// The SSA form of `mir` (see the top of this file).
pub fn to_ssa(mir: &Mir) -> Result<(Mir, SsaStats)> {
    // 1. the pinned registers: read before any definition since the last label or call
    let mut defined = [false; 64];
    let mut pinned = [false; 64];
    let mut instructions = 0;
    for ins in mir.code.iter() {
        instructions += 1;
        let (reads, writes) = operands(&ins)?;
        for r in reads.iter().chain(&writes) {
            if let Reg::Static(_) = r {
                bail!("the MIR is in SSA form already");
            }
        }
        for r in reads {
            if let Reg::Gen(r) = r {
                if !defined[r as usize] {
                    pinned[r as usize] = true;
                }
            }
        }
        for r in writes {
            if let Reg::Gen(r) = r {
                defined[r as usize] = true;
            }
        }
        if resets(&ins) {
            defined = [false; 64];
        }
    }

    // 2. backward: the definitions of pinned registers that must write them, the last before
    // each label, call, jump or end (control may leave the straight-line code there, with the
    // value in its register); the others get a `Static` as well
    let code: Vec<Instruction> = mir.code.iter().collect();
    let mut keep = vec![0u64; code.len()];
    let mut later = [false; 64];
    for (i, ins) in code.iter().enumerate().rev() {
        if leaves(ins) {
            later = [false; 64];
            continue;
        }
        for r in operands(ins)?.1 {
            if let Reg::Gen(g) = r {
                if pinned[g as usize] && !later[g as usize] {
                    keep[i] |= 1 << g;
                    later[g as usize] = true;
                }
            }
        }
    }

    // 3. the rewrite: a new `Static` per definition, but the kept ones
    let mut out = Mir::new(mir.config.clone());
    out.consts = mir.consts.clone();
    let mut current: [Option<u32>; 64] = [None; 64];
    let mut statics: u32 = 0;
    let mut kept = 0;
    for (i, ins) in code.into_iter().enumerate() {
        let (reset, leave) = (resets(&ins), leaves(&ins));
        let mut missing = None;
        let ins = rewrite(ins, |r, write| match r {
            Reg::Gen(g) if write && keep[i] & (1 << g) != 0 => {
                current[g as usize] = None;
                kept += 1;
                r
            }
            Reg::Gen(g) if write => {
                current[g as usize] = Some(statics);
                statics += 1;
                Reg::Static(statics - 1)
            }
            Reg::Gen(g) => match current[g as usize] {
                Some(s) => Reg::Static(s),
                // a pinned register holds its value
                None if pinned[g as usize] => r,
                None => {
                    missing = Some(g);
                    r
                }
            },
            r => r,
        });
        if let Some(g) = missing {
            // pass 1 pins every register read before a definition
            return Err(anyhow!(
                "internal error: Gen({}) read before it is defined",
                g
            ));
        }
        out.code.push(&ins);
        if reset {
            current = [None; 64];
        } else if leave {
            for g in 0..64 {
                if pinned[g] {
                    current[g] = None;
                }
            }
        }
    }
    out.populate_labels();
    let stats = SsaStats {
        instructions,
        statics: statics as usize,
        pinned: (0..64u8).filter(|&r| pinned[r as usize]).collect(),
        kept,
    };
    Ok((out, stats))
}

/// A model in SSA form: its program, its MIR (lowered if complex) and its real locations.
pub struct SsaModel {
    pub prog: Program,
    pub mir: Mir,
    pub reals: HashSet<Loc>,
    pub stats: SsaStats,
}

impl SsaModel {
    /// The SSA form of a saved model (`SavedModel::read`: complex models lowered).
    pub fn from_saved(model: SavedModel) -> Result<SsaModel> {
        let (mir, stats) = to_ssa(&model.mir)?;
        Ok(SsaModel {
            prog: model.prog,
            mir,
            reals: model.reals,
            stats,
        })
    }

    pub fn config(&self) -> &Config {
        self.prog.config()
    }

    /// The PTX module (`ssa2ptx`): sjb2ptx's generator on the SSA MIR, `Static(k)` as `%v<k>`.
    /// Mixed precision is not supported yet.
    pub fn ptx(&mut self, options: &PtxOptions) -> Result<Ptx> {
        if options.mixed.is_some() {
            bail!("ssa2ptx does not support mixed precision yet");
        }
        crate::ptx::ptx_of(&mut self.prog, &self.mir, options)
    }

    /// Writes the `.ssa` file (see the top of this file).
    pub fn write(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_all(&SSA_MAGIC.to_le_bytes())?;
        stream.write_all(&VERSION.to_le_bytes())?;
        let flags = if self.config().is_complex() {
            FLAG_LOWERED
        } else {
            0
        };
        stream.write_all(&flags.to_le_bytes())?;
        self.prog.save(stream)?;
        self.mir.save(stream)?;
        write_reals(stream, &self.reals)
    }

    /// Reads an `.ssa` file.
    pub fn read(stream: &mut impl Read) -> Result<SsaModel> {
        let mut word = || -> Result<usize> {
            let mut bytes = [0u8; 8];
            stream.read_exact(&mut bytes)?;
            Ok(usize::from_le_bytes(bytes))
        };
        let magic = word()?;
        if magic != SSA_MAGIC {
            bail!("not an .ssa file (made by sjb2ssa): wrong magic number");
        }
        let version = word()?;
        let flags = word()?;
        if version < 3 {
            bail!("unknown .ssa version {}", version);
        }
        // the saved configuration replaces this one, except for its (empty) functions
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let prog = Program::load_with_version(stream, &config, version)?;
        let mir = Mir::load_with_version(stream, prog.config(), version)?;
        let reals = read_reals(stream)?;
        if prog.config().is_complex() != (flags & FLAG_LOWERED != 0) {
            bail!("the .ssa file's flags do not match its configuration");
        }
        let mut instructions = 0;
        let mut statics = 0;
        let mut pinned = HashSet::new();
        let mut kept = 0;
        for ins in mir.code.iter() {
            instructions += 1;
            let (reads, writes) = operands(&ins)?;
            for r in writes {
                match r {
                    Reg::Static(_) => statics += 1,
                    Reg::Gen(g) => {
                        kept += 1;
                        pinned.insert(g);
                    }
                    _ => {}
                }
            }
            for r in reads {
                if let Reg::Gen(g) = r {
                    pinned.insert(g);
                }
            }
        }
        let mut pinned: Vec<u8> = pinned.into_iter().collect();
        pinned.sort_unstable();
        Ok(SsaModel {
            prog,
            mir,
            reals,
            stats: SsaStats {
                instructions,
                statics,
                pinned,
                kept,
            },
        })
    }
}

// as `.sjb` files hold them (symjit's `save_reals`): the count, then sorted location codes
fn write_reals(stream: &mut impl Write, reals: &HashSet<Loc>) -> Result<()> {
    stream.write_all(&reals.len().to_le_bytes())?;
    let mut codes: Vec<usize> = reals
        .iter()
        .map(|r| match r {
            Loc::Mem(idx) => 0x1_0000_0000 | (*idx as usize),
            Loc::Stack(idx) => 0x2_0000_0000 | (*idx as usize),
            Loc::Param(idx) => 0x3_0000_0000 | (*idx as usize),
        })
        .collect();
    codes.sort_unstable();
    for c in codes {
        stream.write_all(&c.to_le_bytes())?;
    }
    Ok(())
}

fn read_reals(stream: &mut impl Read) -> Result<HashSet<Loc>> {
    let mut bytes = [0u8; 8];
    stream.read_exact(&mut bytes)?;
    let count = usize::from_le_bytes(bytes);
    let mut reals = HashSet::new();
    for _ in 0..count {
        stream.read_exact(&mut bytes)?;
        let c = usize::from_le_bytes(bytes);
        let idx = (c & 0xffff_ffff) as u32;
        reals.insert(match c >> 32 {
            1 => Loc::Mem(idx),
            2 => Loc::Stack(idx),
            3 => Loc::Param(idx),
            _ => bail!("invalid real location in the .ssa file"),
        });
    }
    Ok(reals)
}
