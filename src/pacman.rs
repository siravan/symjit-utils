//! pacman: what a pyAmpliCol evaluator container (`evaluators.pacbin`, `pacbin`) holds, and
//! its members extracted. Its kernels are the members of kind `symjit-application`: saved symjit
//! applications (sjb version 3, written by the symjit 2.26.4 pyAmpliCol 1.0.1 links), a plain
//! one (`.symjit`, `evaluate_matrix`) and a plane one (`.plane.symjit`, the direct-arena
//! kernels pyAmpliCol calls) per evaluator chunk. See `docs/PACMAN.md`.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::panic::{self, AssertUnwindSafe};

use anyhow::{anyhow, Result};
use symjit::{Application, CompilerType, Config};

use crate::info::{option_list, symjit_version, thousands, MAGIC};
use crate::pacbin::{Kind, Member, Pacbin};

/// What `describe` reports besides the basics.
#[derive(Clone, Debug)]
pub struct Options {
    /// compile every kernel (for this machine, never run) to measure its machine code
    pub compile: bool,
    /// the number of kernels listed, the largest first (`usize::MAX`: all)
    pub top: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            compile: false,
            top: 10,
        }
    }
}

/// One kernel (a `symjit-application` member), as `Application::read_sjb` reads it.
#[derive(Clone, Debug, Default)]
pub struct Kernel {
    pub path: String,
    pub bytes: usize,
    /// `.plane.symjit`
    pub plane: bool,
    pub sjb_version: u64,
    pub target: &'static str,
    pub opt_level: u32,
    /// the options the file sets (`Config::opt` bits), by name
    pub options: Vec<&'static str>,
    pub complex: bool,
    pub states: usize,
    pub params: usize,
    pub obs: usize,
    pub mir: usize,
    pub constants: usize,
    pub stack: usize,
    /// the machine code compiled from it (`Options::compile`): scalar and SIMD kernels
    pub code: Option<Result<(usize, usize), String>>,
    /// why it could not be read
    pub error: Option<String>,
}

impl Kernel {
    /// The values per point (f64; complex models count two per number): plane kernels take
    /// their inputs as states, plain ones as parameters.
    pub fn inputs(&self) -> usize {
        self.states + self.params
    }

    pub fn read(path: &str, bytes: &[u8]) -> Kernel {
        let mut k = Kernel {
            path: path.to_string(),
            bytes: bytes.len(),
            plane: path.ends_with(".plane.symjit"),
            ..Default::default()
        };
        if let Err(err) = k.fill(bytes) {
            k.error = Some(format!("{:#}", err));
        }
        k
    }

    fn fill(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() < 16 || u64::from_le_bytes(bytes[..8].try_into().unwrap()) != MAGIC {
            return Err(anyhow!(
                "not a saved symjit application (wrong magic number)"
            ));
        }
        self.sjb_version = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        // the saved configuration replaces this one, except for its (empty) functions
        let empty = Config::new(CompilerType::ByteCode, 0)?;
        let (prog, _reals, mir) = Application::read_sjb(&mut &bytes[..], &empty)?;
        let config = prog.config();
        self.target = target(config.ty);
        self.opt_level = config.opt_level() as u32;
        self.options = option_list(config)
            .into_iter()
            .filter(|(_, on)| *on)
            .map(|(name, _)| name)
            .collect();
        self.complex = config.is_complex();
        self.states = prog.count_states;
        self.params = prog.count_params;
        self.obs = prog.count_obs;
        self.mir = mir.code.ip;
        self.constants = prog.builder.consts.len();
        self.stack = prog.builder.stack_size();
        Ok(())
    }
}

/// The report on the container `name` (its contents `pacbin`): the members by kind, and the
/// kernels: their configurations, sizes and the largest of them.
pub fn describe(name: &str, pacbin: &Pacbin, options: &Options) -> Result<String> {
    let mut kernels: Vec<Kernel> = pacbin
        .members
        .iter()
        .filter(|m| m.kind == Kind::SymjitApplication)
        .map(|m| Kernel::read(&m.path, pacbin.payload(m)))
        .collect();
    if options.compile {
        compile_all(pacbin, &mut kernels);
    }

    let mut out = String::new();
    let o = &mut out;
    writeln!(o, "{}", name)?;
    row(
        o,
        "container",
        &format!(
            "pacbin-v1, {} bytes, {} members (digests verified)",
            thousands(pacbin.len()),
            thousands(pacbin.members.len())
        ),
    );
    row(o, "read by", &format!("symjit {}", symjit_version()));

    section(o, "Members");
    let mut kinds: BTreeMap<Kind, (usize, usize)> = BTreeMap::new();
    for m in pacbin.members.iter() {
        let e = kinds.entry(m.kind).or_default();
        e.0 += 1;
        e.1 += m.length as usize;
    }
    for (kind, (n, bytes)) in kinds.iter() {
        let label = if *kind == Kind::SymjitApplication {
            format!("{} (kernels)", kind.name())
        } else {
            kind.name().to_string()
        };
        writeln!(
            o,
            "  {:<40} {:>6}  {:>14} bytes",
            label,
            thousands(*n),
            thousands(*bytes)
        )?;
    }

    section(o, "Kernels");
    if kernels.is_empty() {
        row(o, "kernels", "none");
        return Ok(out);
    }
    let plane = kernels.iter().filter(|k| k.plane).count();
    row(
        o,
        "kernels",
        &format!(
            "{}: {} plain (.symjit), {} plane (.plane.symjit)",
            thousands(kernels.len()),
            thousands(kernels.len() - plane),
            thousands(plane)
        ),
    );
    // the same payload under several paths (pyAmpliCol stores the kernels of a color selector
    // sector apart from those of the helicity sum, even when they are equal)
    let mut digests: BTreeMap<&[u8; 32], usize> = BTreeMap::new();
    for m in pacbin.members.iter() {
        if m.kind == Kind::SymjitApplication {
            digests.insert(&m.sha256, m.length as usize);
        }
    }
    let distinct_bytes: usize = digests.values().sum();
    let all_bytes: usize = kernels.iter().map(|k| k.bytes).sum();
    row(
        o,
        "distinct",
        &format!(
            "{} ({} bytes; the others repeat a payload, {} bytes)",
            thousands(digests.len()),
            thousands(distinct_bytes),
            thousands(all_bytes - distinct_bytes)
        ),
    );
    let bad: Vec<&Kernel> = kernels.iter().filter(|k| k.error.is_some()).collect();
    if !bad.is_empty() {
        row(
            o,
            "unreadable",
            &format!("{} (listed at the end)", thousands(bad.len())),
        );
    }
    let good: Vec<&Kernel> = kernels.iter().filter(|k| k.error.is_none()).collect();

    count_by(o, "sjb version", &good, |k| format!("{}", k.sjb_version));
    count_by(o, "target", &good, |k| k.target.to_string());
    count_by(o, "numbers", &good, |k| {
        if k.complex { "complex" } else { "real" }.to_string()
    });
    count_by(o, "opt level", &good, |k| format!("{}", k.opt_level));

    section(o, "Options (kernels per combination)");
    let mut combos: BTreeMap<(bool, Vec<&str>), usize> = BTreeMap::new();
    for k in good.iter() {
        *combos.entry((k.plane, k.options.clone())).or_default() += 1;
    }
    for ((plane, opts), n) in combos.iter() {
        let label = format!(
            "{} {}",
            thousands(*n),
            if *plane { "plane" } else { "plain" }
        );
        wrapped(o, &label, &opts.join(", "));
    }

    section(
        o,
        "Sizes                      min        median           max         total",
    );
    let stat = |o: &mut String, label: &str, values: Vec<usize>| {
        if let Some(s) = Stats::new(values) {
            let _ = writeln!(
                o,
                "  {:<16} {:>12}  {:>12}  {:>12}  {:>12}",
                label,
                thousands(s.min),
                thousands(s.median),
                thousands(s.max),
                thousands(s.total)
            );
        }
    };
    stat(o, "file bytes", good.iter().map(|k| k.bytes).collect());
    stat(o, "inputs", good.iter().map(|k| k.inputs()).collect());
    stat(o, "outputs", good.iter().map(|k| k.obs).collect());
    stat(o, "MIR", good.iter().map(|k| k.mir).collect());
    stat(o, "constants", good.iter().map(|k| k.constants).collect());
    stat(o, "stack slots", good.iter().map(|k| k.stack).collect());
    if options.compile {
        let compiled: Vec<(usize, usize)> = good
            .iter()
            .filter_map(|k| k.code.as_ref().and_then(|c| c.as_ref().ok()).copied())
            .collect();
        stat(o, "scalar code", compiled.iter().map(|c| c.0).collect());
        stat(o, "SIMD code", compiled.iter().map(|c| c.1).collect());
        let failed: Vec<&&Kernel> = good
            .iter()
            .filter(|k| matches!(k.code, Some(Err(_))))
            .collect();
        for k in failed.iter() {
            if let Some(Err(err)) = &k.code {
                row(o, "not compiled", &format!("{}: {}", k.path, err));
            }
        }
    }
    let _ = writeln!(
        o,
        "  (values per point; a complex number counts as two; code in bytes, for {})",
        if options.compile {
            "this machine"
        } else {
            "--compile"
        }
    );

    if options.top > 0 {
        let mut largest = good.clone();
        largest.sort_by(|a, b| b.mir.cmp(&a.mir).then(a.path.cmp(&b.path)));
        let shown = largest.len().min(options.top);
        let title = if shown == largest.len() {
            "Kernels by MIR instructions".to_string()
        } else {
            format!(
                "Largest kernels ({} of {}, by MIR instructions)",
                shown,
                largest.len()
            )
        };
        section(o, &title);
        let code_header = if options.compile {
            "     scalar       SIMD"
        } else {
            ""
        };
        writeln!(
            o,
            "  {:>9} {:>7} {:>7} {:>10}{}  path",
            "MIR", "inputs", "outputs", "bytes", code_header
        )?;
        for k in largest.iter().take(shown) {
            let code = match (&k.code, options.compile) {
                (Some(Ok((s, v))), _) => format!(" {:>10} {:>10}", thousands(*s), thousands(*v)),
                (_, true) => format!(" {:>10} {:>10}", "-", "-"),
                (_, false) => String::new(),
            };
            writeln!(
                o,
                "  {:>9} {:>7} {:>7} {:>10}{}  {}",
                thousands(k.mir),
                thousands(k.inputs()),
                thousands(k.obs),
                thousands(k.bytes),
                code,
                k.path
            )?;
        }
    }

    if !bad.is_empty() {
        section(o, "Unreadable kernels");
        for k in bad.iter() {
            writeln!(o, "  {}: {}", k.path, k.error.as_deref().unwrap_or(""))?;
        }
    }

    Ok(out)
}

/// The members of `pacbin` whose paths match one of `patterns` (`*` any characters, `/`
/// included; `?` one character), in index order; every pattern must match.
pub fn select<'a>(pacbin: &'a Pacbin, patterns: &[String]) -> Result<Vec<&'a Member>> {
    let mut chosen = vec![false; pacbin.members.len()];
    for p in patterns {
        let mut any = false;
        for (i, m) in pacbin.members.iter().enumerate() {
            if glob(p.as_bytes(), m.path.as_bytes()) {
                chosen[i] = true;
                any = true;
            }
        }
        if !any {
            return Err(anyhow!("no member matches `{}`", p));
        }
    }
    Ok(pacbin
        .members
        .iter()
        .zip(chosen)
        .filter(|(_, c)| *c)
        .map(|(m, _)| m)
        .collect())
}

/// `*` (any run of characters) and `?` (one character) against `text`.
pub fn glob(pattern: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == b'?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some((p, t));
            p += 1;
        } else if let Some((sp, st)) = star {
            p = sp + 1;
            t = st + 1;
            star = Some((sp, st + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|c| *c == b'*')
}

/// Compiles the kernels, on all cores, as pyAmpliCol loads them (`Application::load`, which
/// compiles the scalar kernel and those the file was saved with) plus the SIMD kernel, and
/// measures them. The code is never run. A panic of the compiler is reported, not raised.
fn compile_all(pacbin: &Pacbin, kernels: &mut [Kernel]) {
    let payloads: Vec<&[u8]> = pacbin
        .members
        .iter()
        .filter(|m| m.kind == Kind::SymjitApplication)
        .map(|m| pacbin.payload(m))
        .collect();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(kernels.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<Option<Result<(usize, usize), String>>>> =
        (0..kernels.len()).map(|_| Default::default()).collect();

    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= kernels.len() {
                    break;
                }
                if kernels[i].error.is_some() {
                    continue;
                }
                *results[i].lock().unwrap() = Some(code_sizes(payloads[i]));
            });
        }
    });
    panic::set_hook(hook);

    for (k, r) in kernels.iter_mut().zip(results) {
        k.code = r.into_inner().unwrap();
    }
}

fn code_sizes(bytes: &[u8]) -> Result<(usize, usize), String> {
    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, usize)> {
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let mut app = Application::load(&mut &bytes[..], &config)?;
        app.prepare_simd();
        Ok((app.measure("ker-scalar-size"), app.measure("ker-simd-size")))
    }));
    match result {
        Ok(Ok(sizes)) => Ok(sizes),
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

fn target(ty: CompilerType) -> &'static str {
    match ty {
        CompilerType::Native => "native",
        CompilerType::Amd => "x86-64",
        CompilerType::AmdAVX => "x86-64 AVX",
        CompilerType::AmdSSE => "x86-64 SSE",
        CompilerType::Arm => "ARM64",
        CompilerType::RiscV => "RISC-V 64",
        CompilerType::ByteCode => "bytecode",
        CompilerType::Debug => "debug",
    }
}

struct Stats {
    min: usize,
    median: usize,
    max: usize,
    total: usize,
}

impl Stats {
    fn new(mut values: Vec<usize>) -> Option<Stats> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        Some(Stats {
            min: values[0],
            median: values[values.len() / 2],
            max: values[values.len() - 1],
            total: values.iter().sum(),
        })
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

/// The kernels counted by `key`, as "value: count" pairs.
fn count_by(o: &mut String, label: &str, kernels: &[&Kernel], key: impl Fn(&Kernel) -> String) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for k in kernels {
        *counts.entry(key(k)).or_default() += 1;
    }
    let text: Vec<String> = counts
        .iter()
        .map(|(v, n)| format!("{} ({})", v, thousands(*n)))
        .collect();
    row(o, label, &text.join(", "));
}

/// `label` and `text`, wrapped at 80 columns after commas.
fn wrapped(o: &mut String, label: &str, text: &str) {
    let indent = WIDTH + 3;
    let mut line = String::new();
    let mut first = true;
    for word in text.split(' ') {
        if !line.is_empty() && indent + line.len() + 1 + word.len() > 80 {
            if first {
                row(o, label, &line);
                first = false;
            } else {
                let _ = writeln!(o, "{:indent$}{}", "", line);
            }
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if first {
        row(o, label, &line);
    } else {
        let _ = writeln!(o, "{:indent$}{}", "", line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        let yes = [
            ("*", "a/b/c"),
            ("a/*", "a/b/c"),
            ("*.plane.symjit", "x/y.plane.symjit"),
            (
                "*stage_5*chunk_0*",
                "p/generic_stage_5_subset_6_chunk_0_35.symjit",
            ),
            ("a?c", "abc"),
            ("abc", "abc"),
            ("**", ""),
        ];
        for (p, t) in yes {
            assert!(glob(p.as_bytes(), t.as_bytes()), "{} {}", p, t);
        }
        let no = [
            ("a/*", "b/a"),
            ("*.symjit", "x.symjit.bak"),
            ("a?c", "ac"),
            ("abc", "ab"),
        ];
        for (p, t) in no {
            assert!(!glob(p.as_bytes(), t.as_bytes()), "{} {}", p, t);
        }
    }

    #[test]
    fn stats() {
        let s = Stats::new(vec![5, 1, 9, 3]).unwrap();
        assert_eq!((s.min, s.median, s.max, s.total), (1, 5, 9, 18));
        assert!(Stats::new(vec![]).is_none());
    }
}
