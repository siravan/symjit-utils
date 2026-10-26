//! sjbinfo: a report of what a saved application (`.sjb`) holds — its configuration, the model's
//! sizes, its MIR (as symjit's `dump(.., "stats")`, `Mir::print_stats`), the size of the machine
//! code symjit compiles from it and the math functions it calls.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;
use std::panic::{self, AssertUnwindSafe};

use anyhow::{anyhow, Result};
use symjit::backend::{Complexifier, Func, Loc, Mir, MirInstruction, Reg};
use symjit::{Application, CompilerType, Config};

use crate::obj::generator::{c_name, complex_c_name};

pub(crate) const MAGIC: u64 = 0x40568795410d08e9;

/// The report of the saved application in `bytes` (the contents of the file `name`). With
/// `options.compile`, the application is compiled as symjit loads it (`Application::load`), for
/// the machine it was saved for, to measure its kernels; the code is never run. With
/// `options.constants`, the constant pool is listed.
pub fn describe(name: &str, bytes: &[u8], options: &Options) -> Result<String> {
    if bytes.len() < 16 || u64::from_le_bytes(bytes[..8].try_into().unwrap()) != MAGIC {
        return Err(anyhow!(
            "not a saved symjit application (wrong magic number)"
        ));
    }
    let version = u64::from_le_bytes(bytes[8..16].try_into().unwrap());

    // the saved configuration replaces this one, except for its (empty) functions
    let empty = Config::new(CompilerType::ByteCode, 0)?;
    let (prog, reals, saved) = Application::read_sjb(&mut &bytes[..], &empty)?;
    let config = prog.config().clone();
    let lowered = if config.is_complex() {
        Some(Complexifier::new(&reals, config.clone()).complexify(&saved))
    } else {
        None
    };

    let mut out = String::new();
    let o = &mut out;

    writeln!(o, "{}", name)?;
    row(
        o,
        "file",
        &format!("{} bytes, sjb version {}", thousands(bytes.len()), version),
    );
    row(o, "read by", &format!("symjit {}", symjit_version()));

    section(o, "Model");
    let kind = match (config.is_complex(), reals.len()) {
        (false, _) => "real".to_string(),
        (true, 0) => "complex (the counts below are of f64 values, two per number)".to_string(),
        (true, n) => format!(
            "complex, {} real locations (the counts below are of f64 values)",
            n
        ),
    };
    row(o, "numbers", &kind);
    row(o, "states", &prog.count_states.to_string());
    row(o, "parameters", &prog.count_params.to_string());
    row(o, "observables", &prog.count_obs.to_string());
    row(o, "derivatives", &prog.count_diffs.to_string());
    row(o, "loops", &prog.count_loops.to_string());
    row(
        o,
        "memory",
        &format!("{} values per point", thousands(prog.mem_size())),
    );
    row(
        o,
        "stack",
        &format!("{} slots", thousands(prog.builder.stack_size())),
    );
    row(o, "constants", &thousands(prog.builder.consts.len()));

    section(o, "Configuration");
    row(o, "target", &target(&config));
    row(o, "opt level", &config.opt_level().to_string());
    let (on, off): (Vec<_>, Vec<_>) = option_list(&config).into_iter().partition(|(_, v)| *v);
    let names = |v: Vec<(&'static str, bool)>| v.into_iter().map(|(k, _)| k).collect::<Vec<_>>();
    list(o, "enabled", &names(on));
    list(o, "disabled", &names(off));
    row(o, "stack limit", &thousands(config.stack_limit()));
    row(o, "num_args", &config.num_args().to_string());
    if config.direct_arena_operation() != 0 {
        row(o, "arena op", &config.direct_arena_operation().to_string());
    }
    row(o, "option bits", &format!("{:#010x}", config.opt));
    if unknown_bits(&config) != 0 {
        row(
            o,
            "unknown bits",
            &format!("{:#010x}", unknown_bits(&config)),
        );
    }

    let stats = MirStats::new(&saved);
    section(o, "IR (MIR)");
    row(
        o,
        "instructions",
        &format!(
            "{} ({} bytes)",
            thousands(stats.count),
            thousands(saved.code.buf.len())
        ),
    );
    match &lowered {
        Some(Ok(mir)) => row(
            o,
            "lowered",
            &format!(
                "{} instructions after the complex operations are lowered to real ones{}",
                thousands(mir.code.ip),
                if config.fast_complex() {
                    " (bytecode interpreter)"
                } else {
                    " (bytecode interpreter and kernels)"
                }
            ),
        ),
        Some(Err(err)) => row(o, "lowered", &format!("failed: {}", err)),
        None => {}
    }
    stats.write(o);

    section(o, "Compiled code");
    if options.compile {
        match kernels(bytes) {
            Ok(k) => k.write(o, &config, &prog),
            Err(err) => row(o, "failed", &err),
        }
    } else {
        row(o, "kernels", "not compiled (--no-compile)");
    }

    section(o, "Math functions");
    functions(o, &prog.builder.ft, &stats.calls, &saved, &config);

    if options.constants {
        // the uses are counted in the MIR the kernels are compiled from: the lowered one of a
        // complex model without `fast_complex` (which also holds any constants the lowering
        // appends, `stable_math`), the saved one otherwise
        match &lowered {
            Some(Ok(mir)) if !config.fast_complex() => constants(o, &mir.consts, mir),
            _ => constants(o, &prog.builder.consts, &saved),
        }
    }

    Ok(out)
}

/// The version of the symjit crate (`measure("version")` of a trivial application).
pub(crate) fn symjit_version() -> String {
    use symjit::{var, Compiler};
    let x = var("x");
    let n = Compiler::with_config(Config::new(CompilerType::ByteCode, 0).unwrap())
        .compile(std::slice::from_ref(&x), std::slice::from_ref(&x))
        .map(|app| app.measure("version"))
        .unwrap_or(0);
    format!("{}.{}.{}", n / 10000, n / 100 % 100, n % 100)
}

/// What `describe` reports besides the basics.
#[derive(Clone, Debug)]
pub struct Options {
    /// compile the application to measure its kernels
    pub compile: bool,
    /// list the constant pool
    pub constants: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            compile: true,
            constants: false,
        }
    }
}

/****************************** formatting ******************************/

const WIDTH: usize = 14;

fn section(o: &mut String, title: &str) {
    let _ = writeln!(o, "\n{}", title);
}

fn row(o: &mut String, key: &str, val: &str) {
    let _ = writeln!(o, "  {:<WIDTH$} {}", key, val);
}

/// `key` and the comma-separated `items`, wrapped at 80 columns.
fn list(o: &mut String, key: &str, items: &[&str]) {
    if items.is_empty() {
        row(o, key, "-");
        return;
    }
    let indent = WIDTH + 3;
    let mut lines: Vec<String> = vec![String::new()];
    for (i, item) in items.iter().enumerate() {
        let item = if i + 1 < items.len() {
            format!("{},", item)
        } else {
            item.to_string()
        };
        let line = lines.last_mut().unwrap();
        if !line.is_empty() && indent + line.len() + 1 + item.len() > 80 {
            lines.push(item);
        } else {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&item);
        }
    }
    row(o, key, &lines[0]);
    for line in &lines[1..] {
        let _ = writeln!(o, "{:indent$}{}", "", line);
    }
}

/// 1234567 as "1,234,567".
pub(crate) fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// "LittleThanEqual" as "little_than_equal".
fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/****************************** configuration ******************************/

/// The machine the code is generated for.
fn target(config: &Config) -> String {
    match config.ty {
        CompilerType::Native => format!("native: the machine that loads it (here {})", host()),
        CompilerType::Amd => "x86-64 (AVX, or SSE without it)".into(),
        CompilerType::AmdAVX => "x86-64, AVX".into(),
        CompilerType::AmdSSE => "x86-64, SSE (deprecated)".into(),
        CompilerType::Arm => "ARM64".into(),
        CompilerType::RiscV => "RISC-V 64".into(),
        CompilerType::ByteCode => "bytecode: symjit's interpreter, no machine code".into(),
        CompilerType::Debug => "debug (deprecated; the interpreter)".into(),
    }
}

fn host() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            "x86-64 with AVX-512"
        } else if is_x86_feature_detected!("avx2") {
            "x86-64 with AVX2"
        } else if is_x86_feature_detected!("avx") {
            "x86-64 with AVX"
        } else {
            "x86-64 with SSE only"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "ARM64"
    }
    #[cfg(target_arch = "riscv64")]
    {
        "RISC-V 64"
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    {
        "a machine symjit compiles no code for"
    }
}

/// The option bits of `Config::opt`, as stable's `rust/config.rs` defines them (its module is
/// private), by their `Config::set_option` names (`direct_arena_identity_output` has none).
/// The getters (`Config::use_simd`, `fastmath`, ...) also test the machine; these are the bits
/// the file holds. The opt level (0x0f00) and the arena operation (0x0600_0000) are fields.
const OPTION_BITS: &[(&str, u32)] = &[
    ("use_simd", 0x0000_0001),
    ("use_threads", 0x0000_0002),
    ("cse", 0x0000_0004),
    ("fastmath", 0x0000_0008),
    ("enable_simd512", 0x0000_0010),
    ("complex", 0x0000_0020),
    ("symbolica", 0x0000_0040),
    ("simd_branch", 0x0000_0080),
    ("compact", 0x0000_1000),
    ("compress", 0x0000_2000),
    ("direct", 0x0000_4000),
    ("fast_complex", 0x0000_8000),
    ("debug_bytecode", 0x0001_0000),
    ("debug_scalar", 0x0002_0000),
    ("debug_simd", 0x0004_0000),
    ("debug_stats", 0x0008_0000),
    ("huge", 0x0010_0000),
    ("parallel_mul", 0x0020_0000),
    ("direct_arena", 0x0040_0000),
    ("direct_arena_identity_output", 0x0080_0000),
    ("debug_lock", 0x0100_0000),
    ("stable_math", 0x0800_0000),
    ("debug_topology", 0x1000_0000),
    ("debug_instructions", 0x2000_0000),
    ("coroutines", 0x4000_0000),
];
const FIELD_BITS: u32 = 0x0000_0f00 | 0x0600_0000;

fn has(config: &Config, name: &str) -> bool {
    OPTION_BITS
        .iter()
        .any(|(n, bit)| *n == name && config.opt & bit != 0)
}

/// The options and whether the file sets them; the debug options only when set.
pub(crate) fn option_list(config: &Config) -> Vec<(&'static str, bool)> {
    OPTION_BITS
        .iter()
        .map(|(name, bit)| (*name, config.opt & bit != 0))
        .filter(|(name, on)| *on || !name.starts_with("debug_"))
        .collect()
}

/// The bits of `Config::opt` that are neither options nor fields.
fn unknown_bits(config: &Config) -> u32 {
    let known = OPTION_BITS.iter().fold(FIELD_BITS, |m, (_, bit)| m | bit);
    config.opt & !known
}

/****************************** MIR ******************************/

/// What `Mir::print_stats` counts: the instructions by kind, the registers they use, the
/// highest stack slot written and the pairs of multiplications by memory operands (the
/// candidates of `times2_loc`); and the calls.
struct MirStats {
    count: usize,
    kinds: HashMap<String, usize>,
    regs: BTreeMap<u8, usize>,
    statics: BTreeSet<u32>,
    max_stack: Option<u32>,
    times2: usize,
    labels: usize,
    calls: BTreeMap<String, usize>,
}

impl MirStats {
    fn new(mir: &Mir) -> MirStats {
        let mut stats = MirStats {
            count: mir.code.ip,
            kinds: HashMap::new(),
            regs: BTreeMap::new(),
            statics: BTreeSet::new(),
            max_stack: None,
            times2: 0,
            labels: 0,
            calls: BTreeMap::new(),
        };

        let mut prev_times = false;
        for ins in mir.code.iter() {
            let times = matches!(
                &ins,
                MirInstruction::LoadMath { op, .. } if format!("{:?}", op) == "Times"
            );
            if times && prev_times {
                stats.times2 += 1;
            }
            prev_times = times;

            match &ins {
                MirInstruction::Save {
                    loc: Loc::Stack(idx),
                    ..
                }
                | MirInstruction::SaveComplex {
                    loc: Loc::Stack(idx),
                    ..
                } => {
                    stats.max_stack = Some(stats.max_stack.map_or(*idx, |m| m.max(*idx)));
                }
                MirInstruction::Call { label, num_args } if *num_args > 0 => {
                    *stats.calls.entry(label.clone()).or_default() += 1;
                }
                MirInstruction::Label { .. } => stats.labels += 1,
                _ => {}
            }

            for r in regs(&ins) {
                match r {
                    Reg::Gen(n) => *stats.regs.entry(n).or_default() += 1,
                    Reg::Static(n) => {
                        stats.statics.insert(n);
                    }
                    _ => {}
                }
            }

            *stats.kinds.entry(kind(&ins)).or_default() += 1;
        }

        stats
    }

    fn write(&self, o: &mut String) {
        let mut regs = match (self.regs.keys().next(), self.regs.keys().last()) {
            (Some(first), Some(last)) => format!(
                "{} general (Gen({})..Gen({}))",
                self.regs.len(),
                first,
                last
            ),
            _ => "no general registers".to_string(),
        };
        if !self.statics.is_empty() {
            regs += &format!(", {} not allocated (Static)", thousands(self.statics.len()));
        }
        row(o, "registers", &regs);
        if !self.regs.is_empty() {
            let uses: Vec<String> = self
                .regs
                .iter()
                .map(|(r, n)| format!("{}:{}", r, thousands(*n)))
                .collect();
            let uses: Vec<&str> = uses.iter().map(|s| s.as_str()).collect();
            list(o, "uses", &uses);
        }
        row(
            o,
            "stack slots",
            &match self.max_stack {
                Some(m) => format!("written up to Stack[{}]", m),
                None => "none written".into(),
            },
        );
        row(o, "labels", &thousands(self.labels));
        row(o, "times2 pairs", &thousands(self.times2));

        let _ = writeln!(o, "\n  {:>10}  {:>6}  instruction", "count", "%");
        let mut kinds: Vec<(&String, &usize)> = self.kinds.iter().collect();
        kinds.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (k, n) in kinds {
            let _ = writeln!(
                o,
                "  {:>10}  {:>5.1}%  {}",
                thousands(*n),
                100.0 * *n as f64 / self.count.max(1) as f64,
                k
            );
        }
    }
}

/// The instruction's kind, as `Instruction::desc` (calls by name).
fn kind(ins: &MirInstruction) -> String {
    match ins {
        MirInstruction::Nop => "nop".into(),
        MirInstruction::End => "end".into(),
        MirInstruction::Uni { op, .. } => snake(&format!("{:?}", op)),
        MirInstruction::Bi { op, .. } => snake(&format!("{:?}", op)),
        MirInstruction::Call { label, .. } => format!("call {}", label),
        MirInstruction::Mov { .. } => "mov".into(),
        MirInstruction::Load { .. } => "load".into(),
        MirInstruction::Save { .. } => "save".into(),
        MirInstruction::LoadComplex { .. } => "load_complex".into(),
        MirInstruction::SaveComplex { .. } => "save_complex".into(),
        MirInstruction::LoadConst { .. } => "load_const".into(),
        MirInstruction::LoadArgs { complex: false, .. } => "load_args".into(),
        MirInstruction::LoadArgs { complex: true, .. } => "load_args_complex".into(),
        MirInstruction::SaveArgs { complex: false, .. } => "save_args".into(),
        MirInstruction::SaveArgs { complex: true, .. } => "save_args_complex".into(),
        MirInstruction::Fused { op, .. } => format!("fused {}", snake(&format!("{:?}", op))),
        MirInstruction::IfElse { .. } => "if_else".into(),
        MirInstruction::Label { .. } => "label".into(),
        MirInstruction::Branch { .. } => "branch".into(),
        MirInstruction::BranchIf { .. } => "branch_if".into(),
        MirInstruction::LoadMath { op, .. } => {
            format!("{} by memory", snake(&format!("{:?}", op)))
        }
        MirInstruction::LoadConstMath { op, .. } => {
            format!("{} by constant", snake(&format!("{:?}", op)))
        }
        MirInstruction::ComplexBi { op, .. } => {
            format!("complex {}", snake(&format!("{:?}", op)))
        }
    }
}

/// The registers the instruction reads or writes (`Instruction::regs`).
fn regs(ins: &MirInstruction) -> Vec<Reg> {
    match ins {
        MirInstruction::Bi { dst, s1, s2, .. } => vec![*dst, *s1, *s2],
        MirInstruction::Uni { dst, s1, .. } => vec![*dst, *s1],
        MirInstruction::Mov { dst, s1 } => vec![*dst, *s1],
        MirInstruction::BranchIf { cond, .. } => vec![*cond],
        MirInstruction::Load { dst, .. } | MirInstruction::LoadConst { dst, .. } => vec![*dst],
        MirInstruction::LoadComplex { xd, yd, .. } => vec![*xd, *yd],
        MirInstruction::LoadMath { dst, s1, .. } => vec![*dst, *s1],
        MirInstruction::LoadConstMath { dst, s1, .. } => vec![*dst, *s1],
        MirInstruction::Save { src, .. } => vec![*src],
        MirInstruction::SaveComplex { xs, ys, .. } => vec![*xs, *ys],
        MirInstruction::Fused { dst, a, b, c, .. } => vec![*dst, *a, *b, *c],
        MirInstruction::ComplexBi {
            xd,
            yd,
            x1,
            y1,
            x2,
            y2,
            ..
        } => vec![*xd, *yd, *x1, *y1, *x2, *y2],
        MirInstruction::IfElse {
            dst,
            true_val,
            false_val,
            ..
        } => vec![*dst, *true_val, *false_val],
        _ => vec![],
    }
}

/****************************** kernels ******************************/

/// The sizes of the kernels symjit compiles from the file (0: none).
struct Kernels {
    scalar: usize,
    simd: usize,
    fast: usize,
    saved_simd: bool,
    saved_fast: bool,
    can_fast: bool,
}

/// Loads the application as symjit does (compiling its scalar kernel and the kernels it was
/// saved with), then compiles the other kernels. A panic of the compiler is an error.
fn kernels(bytes: &[u8]) -> Result<Kernels, String> {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<Kernels> {
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let mut app = Application::load(&mut &bytes[..], &config)?;
        let saved_simd = app.measure("ker-simd-size") > 0;
        let saved_fast = app.measure("ker-fast-size") > 0;
        app.prepare_simd();
        let _ = app.get_fast();
        Ok(Kernels {
            scalar: app.measure("ker-scalar-size"),
            simd: app.measure("ker-simd-size"),
            fast: app.measure("ker-fast-size"),
            saved_simd,
            saved_fast,
            can_fast: app.can_fast,
        })
    }));
    panic::set_hook(hook);

    match result {
        Ok(Ok(k)) => Ok(k),
        Ok(Err(err)) => Err(format!("{:#}", err)),
        Err(payload) => Err(format!(
            "symjit panicked: {}",
            payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("(no message)")
        )),
    }
}

impl Kernels {
    fn write(&self, o: &mut String, config: &Config, prog: &symjit::Program) {
        if config.is_bytecode() || config.is_debug() {
            row(o, "kernels", "none: run by the bytecode interpreter");
            return;
        }

        let bytes = |n: usize| format!("{} bytes", thousands(n));
        let total = self.scalar + self.simd + self.fast;
        row(o, "scalar", &bytes(self.scalar));

        let simd = if self.simd > 0 {
            bytes(self.simd)
        } else if !has(config, "use_simd") {
            "none (use_simd is off)".into()
        } else if prog.count_loops > 0 {
            "none (the model has loops)".into()
        } else if !(config.has_avx() || config.is_arm64()) {
            "none (only for AVX and ARM64)".into()
        } else {
            "none (not compiled: see `stack_limit`)".into()
        };
        row(o, "SIMD", &simd);

        let fast = if self.fast > 0 {
            bytes(self.fast)
        } else if !self.can_fast {
            "none (for at most 8 states, 1 observable, no parameters)".into()
        } else {
            "none (not compiled)".into()
        };
        row(o, "fast", &fast);
        row(o, "total", &bytes(total));

        let mut saved = vec!["scalar"];
        if self.saved_simd {
            saved.push("SIMD");
        }
        if self.saved_fast {
            saved.push("fast");
        }
        list(o, "saved with", &saved);
    }
}

/****************************** functions ******************************/

/// The functions in the program's function table and the calls of the MIR: the symjit
/// function each is linked to and the C library function an object file (`sjb2obj`) calls.
fn functions(
    o: &mut String,
    table: &std::collections::HashSet<String>,
    calls: &BTreeMap<String, usize>,
    mir: &Mir,
    config: &Config,
) {
    let names: BTreeSet<&String> = table.iter().chain(calls.keys()).collect();
    if names.is_empty() {
        row(o, "functions", "none");
        return;
    }

    let _ = writeln!(
        o,
        "  {:<WIDTH$} {:>5}  {:<20} C library (sjb2obj)",
        "function", "calls", "symjit"
    );
    for name in names {
        let n = calls.get(name).copied().unwrap_or(0);
        let symjit = match mir.find_op(name) {
            Ok(Func::Unary(_)) => "real, 1 argument",
            Ok(Func::Binary(_)) => "real, 2 arguments",
            Ok(Func::UnaryCplx(_)) => "complex, 1 argument",
            Ok(Func::BinaryCplx(_)) => "complex, 2 arguments",
            Ok(Func::PairedUnary(_)) => "paired",
            Ok(_) => "user-defined",
            Err(_) => "not found",
        };
        let c = if config.is_complex() {
            complex_c(name)
        } else {
            real_c(name)
        };
        let _ = writeln!(
            o,
            "  {:<WIDTH$} {:>5}  {:<20} {}",
            name,
            thousands(n),
            symjit,
            c
        );
    }
}

fn real_c(name: &str) -> String {
    if let Some(c) = c_name(name) {
        return c.into();
    }
    match name {
        "csc" => "1/sin".into(),
        "sec" => "1/cos".into(),
        "cot" => "1/tan".into(),
        "csch" => "1/sinh".into(),
        "sech" => "1/cosh".into(),
        "coth" => "1/tanh".into(),
        "sinc" => "sin(x)/x".into(),
        "arg" => "none needed (0)".into(),
        _ => "- (not in object files)".into(),
    }
}

fn complex_c(name: &str) -> String {
    if let Some(c) = complex_c_name(name) {
        return c.into();
    }
    match name {
        "csc" => "1/csin".into(),
        "sec" => "1/ccos".into(),
        "cot" => "1/ctan".into(),
        "csch" => "1/csinh".into(),
        "sech" => "1/ccosh".into(),
        "coth" => "1/ctanh".into(),
        "log10" => "clog(z)/ln 10".into(),
        "cbrt" => "cpow(z, 1/3)".into(),
        "sinc" => "csin(z)/z".into(),
        _ => "- (not in object files)".into(),
    }
}

/****************************** constants ******************************/

/// The constant pool: each f64 slot (a complex constant of the direct translator takes two,
/// re and im), the instructions that load it (`LoadConst`, `LoadConstMath`), its value (the
/// shortest decimal that reads back to it) and its bits.
fn constants(o: &mut String, consts: &[f64], mir: &Mir) {
    section(o, &format!("Constants ({})", thousands(consts.len())));
    if consts.is_empty() {
        row(o, "constants", "none");
        return;
    }

    let mut uses = vec![0usize; consts.len()];
    for ins in mir.code.iter() {
        if let MirInstruction::LoadConst { idx, .. } | MirInstruction::LoadConstMath { idx, .. } =
            &ins
        {
            if let Some(n) = uses.get_mut(*idx as usize) {
                *n += 1;
            }
        }
    }

    let _ = writeln!(o, "  {:>7}  {:>5}  {:<24}  bits", "index", "uses", "value");
    for (i, (x, n)) in consts.iter().zip(uses.iter()).enumerate() {
        let _ = writeln!(
            o,
            "  {:>7}  {:>5}  {:<24}  {:#018x}",
            i,
            thousands(*n),
            format!("{:?}", x),
            x.to_bits()
        );
    }
}
