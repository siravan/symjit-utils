//! pacman: reports on and extracts the members of a pyAmpliCol evaluator container
//! (`evaluators.pacbin`), whose kernels are saved symjit applications, and reports on the
//! recurrence plans that call them and the plane tables of its compiled-mode kernels, and
//! replays RustiCol's arena recordings with those kernels. See `docs/PACMAN.md`.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use symjit_utils::bindings;
use symjit_utils::pacbin::{Kind, Pacbin};
use symjit_utils::pacman::{self, Options};
use symjit_utils::plan;
use symjit_utils::recording::{self, Recording};
use symjit_utils::replay::{self, CpuKernel};
use symjit_utils::sha256::hex;

const USAGE: &str = "\
Usage: pacman info [OPTIONS] <FILE.pacbin>...
       pacman list [--kind KIND] [--sha256] <FILE.pacbin>
       pacman extract [OPTIONS] <FILE.pacbin> [MEMBER]...
       pacman plan [--groups N|all] [--pack FILE] <ARTIFACT|PLAN>
       pacman bindings [OPTIONS] <ARTIFACT|execution.json>
       pacman replay [OPTIONS] <ARTIFACT> <RECORDING.json|DIR>...
       pacman runner [OPTIONS] <ARTIFACT> [RECORDING.json]

Reads the evaluator containers of pyAmpliCol artifacts (`evaluators.pacbin`,
pacbin-v1) with the checks pyAmpliCol makes, digests included. Their kernels
are the members of kind `symjit-application`: saved symjit applications, one
plain (`.symjit`) and one plane (`.plane.symjit`) per evaluator chunk.

Commands:
  info      the members by kind; the kernels: their configurations, sizes
            (inputs, outputs, MIR, constants) and the largest of them
  list      one line per member: kind, length and path
  extract   writes members as files, at their paths under the output
            directory; MEMBER is a path or a pattern (`*` matches any
            characters, `/` included, `?` one); without MEMBER, the kernels.
            In a compiled-mode artifact each plane kernel also gets a
            `.mapping` file (lines `<states index> <arena plane>`) and each
            process an `arena.txt` (the arena's regions)
  plan      a recurrence-mode artifact's plan: the stages, the executors run
            (Rusticol intrinsics or symjit kernels) and the row groups in
            execution order; PLAN is a `recurrence-runtime.pacbin` or the
            plan extracted from it
  bindings  a compiled-mode artifact's kernel calls: per execution, the
            leaves (one plane kernel each) in call order and the planes of
            their descriptor tables (arena currents, momenta, the zero plane,
            model-parameter broadcasts; then the outputs), with the arena
            planes pyAmpliCol assigns when it loads the artifact
  replay    runs the kernels of arena recordings of a `pacman` build of
            RustiCol (PACMAN_SAVE_ARENA; a DIR means its *.json) on their
            `.pre` snapshots, as RustiCol runs them, and compares the result
            with their `.post` snapshots bit for bit; planes the schedule
            writes before it reads them are first filled with NaNs. Exit
            status 1 if a recording differs
  runner    writes one execution as C code and object files (the scalar
            kernels) and builds them into `runner`, which runs recordings and
            writes their arenas (`runner OUTDIR RECORDING.json...`, then
            `replay --scalar --against OUTDIR`), and `librunner.so`, which
            exports the interface of `pacman_runner.h` (pacman-runner-v1);
            the execution is a recording's, or `--execution`

Options:
      --compile        info: compile every kernel for this machine (never run)
                       and report the sizes of its machine code
      --top N|all      info: list the N largest kernels (default 10)
  -k, --kind KIND      list, extract: only members of this kind (a name as
                       `list` prints it, or its number)
  -a, --all            extract: every member, not only the kernels
      --no-mapping     extract: write no `.mapping` and `arena.txt` files
  -o, --output DIR     extract: the output directory (default: the container's
                       name without `.pacbin`); `-` writes the one selected
                       member to standard output
      --sha256         list: also print the SHA-256 of each member
      --groups N|all   plan: list the first N row groups (default 50)
      --pack FILE      plan: the kernel pack naming the executors (default:
                       the artifact's `model/eager-kernel-pack.json`)
      --execution PAT  bindings: only the executions matching PAT (`top`,
                       `helicity_sum`, `helicity_sum/color_selector[0]`, ...)
      --leaf PAT       bindings, replay: only the leaves matching PAT, by
                       label (`stage5:0`, `amplitude:0`) or kernel path
      --table          bindings: print each leaf's descriptor table
      --time           replay: print each leaf's time (the best of the runs)
      --repeat N       replay: run each recording N times (default 1)
      --whole          replay: all points as one tile, not the recorded tiles
      --scalar         replay: the scalar kernels only, no SIMD kernel
                       (with --whole: diagnostics, a difference is no error)
      --no-poison      replay: run on `.pre` as it is
  -v, --verbose        replay: one line per leaf, and every differing plane
      --dump DIR       replay: write the replayed arena as DIR/NAME.replay
      --against DIR    replay: compare with DIR/NAME.arena (a runner's) instead
                       of .post; a difference is an error, also with --whole
                       or --scalar
  -o, --output DIR     runner: the directory (default: runner-<process>)
      --execution PAT  runner: the execution, if no recording names it
      --cc CC          runner: the C compiler (default: cc)
      --no-build       runner: write the files only (`make` builds them)
  -h, --help           print this help
  -V, --version        print the version";

enum Command {
    Help,
    Version,
    Info(Vec<PathBuf>, Options),
    List(PathBuf, Option<Kind>, bool),
    Extract(Extract),
    Plan(PathBuf, Option<PathBuf>, plan::Options),
    Bindings(PathBuf, bindings::Options),
    Replay(Replay),
    Runner(Runner),
}

struct Runner {
    artifact: PathBuf,
    recording: Option<PathBuf>,
    output: Option<PathBuf>,
    execution: Option<String>,
    cc: String,
    build: bool,
}

struct Replay {
    artifact: PathBuf,
    recordings: Vec<PathBuf>,
    options: replay::Options,
    time: bool,
    verbose: bool,
    leaf: Option<String>,
    dump: Option<PathBuf>,
    against: Option<PathBuf>,
}

struct Extract {
    input: PathBuf,
    patterns: Vec<String>,
    kind: Option<Kind>,
    all: bool,
    output: Option<String>,
    mapping: bool,
}

fn parse(args: Vec<String>) -> Result<Command, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        return Ok(Command::Help);
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        return Ok(Command::Version);
    }
    let Some(command) = args.first() else {
        return Err("no command (info, list, extract, plan, bindings, replay or runner)".into());
    };

    let mut inputs: Vec<String> = Vec::new();
    let mut options = Options::default();
    let mut kind: Option<Kind> = None;
    let mut all = false;
    let mut mapping = true;
    let mut sha = false;
    let mut output: Option<String> = None;
    let mut pack: Option<PathBuf> = None;
    let mut plan_options = plan::Options::default();
    let mut bindings_options = bindings::Options::default();
    let mut replay_options = replay::Options::default();
    let (mut time, mut verbose) = (false, false);
    let mut replay_leaf: Option<String> = None;
    let mut dump: Option<PathBuf> = None;
    let mut against: Option<PathBuf> = None;
    let mut execution: Option<String> = None;
    let mut cc = "cc".to_string();
    let mut build = true;

    let mut it = args[1..].iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("`{}` needs a value", name))
        };
        match (command.as_str(), arg.as_str()) {
            ("info", "--compile") => options.compile = true,
            ("info", "--top") => {
                let v = value(arg)?;
                options.top = if v == "all" {
                    usize::MAX
                } else {
                    v.parse()
                        .map_err(|_| format!("`--top` takes a number or `all`, not `{}`", v))?
                };
            }
            ("list" | "extract", "-k" | "--kind") => {
                let v = value(arg)?;
                kind = Some(Kind::parse(&v).ok_or_else(|| format!("unknown kind `{}`", v))?);
            }
            ("list", "--sha256") => sha = true,
            ("plan", "--pack") => pack = Some(value(arg)?.into()),
            ("plan", "--groups") => {
                let v = value(arg)?;
                plan_options.groups = if v == "all" {
                    usize::MAX
                } else {
                    v.parse()
                        .map_err(|_| format!("`--groups` takes a number or `all`, not `{}`", v))?
                };
            }
            ("bindings", "--execution") => bindings_options.execution = Some(value(arg)?),
            ("bindings", "--leaf") => bindings_options.leaf = Some(value(arg)?),
            ("bindings", "--table") => bindings_options.table = true,
            ("replay", "--time") => time = true,
            ("replay", "--repeat") => {
                let v = value(arg)?;
                replay_options.repeat =
                    v.parse().ok().filter(|&n| n > 0).ok_or_else(|| {
                        format!("`--repeat` takes a positive number, not `{}`", v)
                    })?;
            }
            ("replay", "--whole") => replay_options.whole = true,
            ("replay", "--scalar") => replay_options.scalar = true,
            ("replay", "--no-poison") => replay_options.poison = false,
            ("replay", "--leaf") => replay_leaf = Some(value(arg)?),
            ("replay", "-v" | "--verbose") => verbose = true,
            ("replay", "--dump") => dump = Some(value(arg)?.into()),
            ("replay", "--against") => against = Some(value(arg)?.into()),
            ("runner", "-o" | "--output") => output = Some(value(arg)?),
            ("runner", "--execution") => execution = Some(value(arg)?),
            ("runner", "--cc") => cc = value(arg)?,
            ("runner", "--no-build") => build = false,
            ("extract", "-a" | "--all") => all = true,
            ("extract", "--no-mapping") => mapping = false,
            ("extract", "-o" | "--output") => output = Some(value(arg)?),
            (_, a) if a.starts_with('-') && a.len() > 1 => {
                return Err(format!("unknown option `{}` for `{}`", a, command))
            }
            _ => inputs.push(arg.clone()),
        }
    }

    match command.as_str() {
        "info" if inputs.is_empty() => Err("no input file".into()),
        "info" => Ok(Command::Info(
            inputs.into_iter().map(PathBuf::from).collect(),
            options,
        )),
        "list" if inputs.len() != 1 => Err("`list` takes one input file".into()),
        "list" => Ok(Command::List(inputs[0].clone().into(), kind, sha)),
        "extract" if inputs.is_empty() => Err("no input file".into()),
        "extract" if all && (kind.is_some() || inputs.len() > 1) => {
            Err("`--all` takes neither `--kind` nor members".into())
        }
        "extract" => Ok(Command::Extract(Extract {
            input: inputs[0].clone().into(),
            patterns: inputs[1..].to_vec(),
            kind,
            all,
            output,
            mapping,
        })),
        "plan" if inputs.len() != 1 => Err("`plan` takes one artifact or plan".into()),
        "plan" => Ok(Command::Plan(inputs[0].clone().into(), pack, plan_options)),
        "bindings" if inputs.len() != 1 => {
            Err("`bindings` takes one artifact or execution.json".into())
        }
        "bindings" => Ok(Command::Bindings(
            inputs[0].clone().into(),
            bindings_options,
        )),
        "replay" if inputs.len() < 2 => {
            Err("`replay` takes an artifact and one recording at least".into())
        }
        "replay" => Ok(Command::Replay(Replay {
            artifact: inputs[0].clone().into(),
            recordings: inputs[1..].iter().map(PathBuf::from).collect(),
            options: replay_options,
            time,
            verbose,
            leaf: replay_leaf,
            dump,
            against,
        })),
        "runner" if inputs.is_empty() || inputs.len() > 2 => {
            Err("`runner` takes an artifact and at most one recording".into())
        }
        "runner" if inputs.len() == 1 && execution.is_none() => {
            Err("`runner` needs a recording or `--execution`".into())
        }
        "runner" => Ok(Command::Runner(Runner {
            artifact: inputs[0].clone().into(),
            recording: inputs.get(1).map(PathBuf::from),
            output: output.map(PathBuf::from),
            execution,
            cc,
            build,
        })),
        _ => Err(format!(
            "unknown command `{}` (info, list, extract, plan, bindings, replay or runner)",
            command
        )),
    }
}

fn read(input: &Path) -> Result<Pacbin> {
    let bytes =
        std::fs::read(input).with_context(|| format!("cannot open `{}`", input.display()))?;
    Pacbin::read(bytes, true).with_context(|| format!("cannot read `{}`", input.display()))
}

fn list(input: &Path, kind: Option<Kind>, sha: bool) -> Result<()> {
    let pacbin = read(input)?;
    let mut out = std::io::stdout().lock();
    for m in pacbin.members.iter() {
        if kind.is_some_and(|k| k != m.kind) {
            continue;
        }
        if sha {
            writeln!(
                out,
                "{:<22} {:>10}  {}  {}",
                m.kind.name(),
                m.length,
                hex(&m.sha256),
                m.path
            )?;
        } else {
            writeln!(out, "{:<22} {:>10}  {}", m.kind.name(), m.length, m.path)?;
        }
    }
    Ok(())
}

fn extract(args: &Extract) -> Result<()> {
    let pacbin = read(&args.input)?;
    let mut members = if args.patterns.is_empty() {
        pacbin.members.iter().collect()
    } else {
        pacman::select(&pacbin, &args.patterns)?
    };
    // without members, the kernels (or `--kind`); `--all`, everything
    let kind = match (args.kind, args.patterns.is_empty() && !args.all) {
        (Some(k), _) => Some(k),
        (None, true) => Some(Kind::SymjitApplication),
        (None, false) => None,
    };
    if let Some(k) = kind {
        members.retain(|m| m.kind == k);
    }
    if members.is_empty() {
        return Err(match kind {
            Some(k) if args.patterns.is_empty() => anyhow!(
                "no member of kind `{}`{}",
                k.name(),
                if args.kind.is_none() {
                    " (no kernels; `--all` extracts every member)"
                } else {
                    ""
                }
            ),
            _ => anyhow!("no member selected"),
        });
    }

    if args.output.as_deref() == Some("-") {
        if members.len() != 1 {
            return Err(anyhow!(
                "`-o -` writes one member, but {} are selected",
                members.len()
            ));
        }
        let mut out = std::io::stdout().lock();
        out.write_all(pacbin.payload(members[0]))?;
        out.flush()?;
        return Ok(());
    }

    let dir = match &args.output {
        Some(d) => PathBuf::from(d),
        None => {
            let stem = args
                .input
                .file_stem()
                .ok_or_else(|| anyhow!("cannot name the output directory: use `-o`"))?;
            PathBuf::from(stem)
        }
    };
    // the plane tables of a compiled-mode artifact: the container's directory
    let (mappings, layouts) = match args.input.parent() {
        Some(artifact) if args.mapping => {
            let artifact = if artifact.as_os_str().is_empty() {
                Path::new(".")
            } else {
                artifact
            };
            if artifact.join("processes").is_dir() {
                let (processes, _) = bindings::load(artifact)?;
                bindings::mappings(&processes)?
            } else {
                Default::default()
            }
        }
        _ => Default::default(),
    };

    let write = |path: &Path, bytes: &[u8]| -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create `{}`", parent.display()))?;
        }
        std::fs::write(path, bytes).with_context(|| format!("cannot write `{}`", path.display()))
    };
    let mut total = 0;
    let mut mapped = std::collections::BTreeSet::new();
    for m in members.iter() {
        // the reader accepts only relative paths without `..`
        let path = dir.join(&m.path);
        write(&path, pacbin.payload(m))?;
        total += m.length;
        if let Some(text) = mappings.get(&m.path) {
            write(&path.with_extension("mapping"), text.as_bytes())?;
            mapped.insert(m.path.as_str());
        }
    }
    let mut processes = 0;
    for (path, text) in layouts.iter() {
        let prefix = path.strip_suffix("arena.txt").unwrap_or(path);
        if mapped.iter().any(|m| m.starts_with(prefix)) {
            write(&dir.join(path), text.as_bytes())?;
            processes += 1;
        }
    }
    eprintln!(
        "pacman: wrote {} member{} ({} bytes) to {}",
        members.len(),
        if members.len() == 1 { "" } else { "s" },
        total,
        dir.display()
    );
    if !mapped.is_empty() {
        eprintln!(
            "pacman: wrote {} plane mapping{} and the arena layout{} of {} process{}",
            mapped.len(),
            if mapped.len() == 1 { "" } else { "s" },
            if processes == 1 { "" } else { "s" },
            processes,
            if processes == 1 { "" } else { "es" }
        );
    }
    Ok(())
}

fn plan(input: &Path, pack: Option<&Path>, options: &plan::Options) -> Result<()> {
    for (i, loaded) in plan::load(input, pack)?.iter().enumerate() {
        if i > 0 {
            println!();
        }
        print!(
            "{}",
            plan::describe(
                &loaded.name,
                &loaded.plan,
                loaded.executors.as_ref(),
                loaded.kernels.as_ref(),
                options
            )?
        );
    }
    Ok(())
}

fn bindings(input: &Path, options: &bindings::Options) -> Result<()> {
    let (processes, pacbin) = bindings::load(input)?;
    let shapes = pacbin.as_deref().map(bindings::kernel_shapes).transpose()?;
    let mut out = std::io::stdout().lock();
    for (i, p) in processes.iter().enumerate() {
        if i > 0 {
            writeln!(out)?;
        }
        write!(out, "{}", bindings::describe(p, shapes.as_ref(), options)?)?;
    }
    Ok(())
}

/// The recordings named: `.json` files, and the `*.json` of directories (sorted).
fn recordings(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(input)
                .with_context(|| format!("cannot read `{}`", input.display()))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect();
            if found.is_empty() {
                return Err(anyhow!("`{}` holds no recording (*.json)", input.display()));
            }
            found.sort();
            out.extend(found);
        } else {
            out.push(input.clone());
        }
    }
    Ok(out)
}

fn short(member: &str) -> &str {
    member.rsplit('/').next().unwrap_or(member)
}

/// `pacman replay`: Ok(false) if a recording could not be replayed or differs.
fn replay_all(args: &Replay) -> Result<bool> {
    let (processes, pacbin) = bindings::load(&args.artifact)?;
    let pacbin_path = pacbin.ok_or_else(|| {
        anyhow!(
            "`{}` has no evaluators.pacbin: `replay` needs the artifact directory",
            args.artifact.display()
        )
    })?;
    let container = read(&pacbin_path)?;
    let members: HashMap<&str, &[u8]> = container
        .members
        .iter()
        .filter(|m| m.kind == Kind::SymjitApplication)
        .map(|m| (m.path.as_str(), container.payload(m)))
        .collect();
    let diagnostic = (args.options.whole || args.options.scalar) && args.against.is_none();
    let mut kernels: HashMap<String, CpuKernel> = HashMap::new();
    let mut ok = true;
    let mut out = std::io::stdout().lock();
    for (i, path) in recordings(&args.recordings)?.iter().enumerate() {
        if i > 0 {
            writeln!(out)?;
        }
        match replay_one(args, path, &processes, &members, &mut kernels, &mut out) {
            Ok(equal) => ok &= equal || diagnostic,
            Err(err) => {
                out.flush()?;
                eprintln!("pacman: error: {}: {:#}", path.display(), err);
                ok = false;
            }
        }
    }
    Ok(ok)
}

fn replay_one(
    args: &Replay,
    path: &Path,
    processes: &[bindings::Process],
    members: &HashMap<&str, &[u8]>,
    kernels: &mut HashMap<String, CpuKernel>,
    out: &mut impl Write,
) -> Result<bool> {
    let rec = Recording::read(path)?;
    let resolved = replay::leaf_calls(&rec, processes)?;
    let calls = &resolved.calls;

    // the kernels not loaded for an earlier recording
    let mut missing: Vec<(&str, &[u8])> = Vec::new();
    for c in calls.iter() {
        if kernels.contains_key(&c.member) || missing.iter().any(|(m, _)| *m == c.member) {
            continue;
        }
        let bytes = members
            .get(c.member.as_str())
            .ok_or_else(|| anyhow!("evaluators.pacbin has no kernel `{}`", c.member))?;
        missing.push((c.member.as_str(), bytes));
    }
    if !missing.is_empty() {
        let start = std::time::Instant::now();
        for ((member, _), k) in missing.iter().zip(replay::load_all(&missing)) {
            kernels.insert(member.to_string(), k?);
        }
        eprintln!(
            "pacman: loaded {} kernel{} in {:.2} s",
            missing.len(),
            if missing.len() == 1 { "" } else { "s" },
            start.elapsed().as_secs_f64()
        );
    }
    let ks: Vec<&CpuKernel> = calls.iter().map(|c| &kernels[&c.member]).collect();
    let mut outcome = replay::replay(&rec, calls, &ks, &args.options)?;
    // the reference: .post, or a runner's arena
    let reference = match &args.against {
        Some(dir) => {
            let file = dir.join(format!(
                "{}.arena",
                path.file_stem().unwrap_or_default().to_string_lossy()
            ));
            let arena = recording::read_f64s(&file, rec.num_planes * rec.num_points)?;
            outcome.diffs = recording::compare(&outcome.arena, &arena, rec.num_points);
            file.display().to_string()
        }
        None => ".post".to_string(),
    };
    let diagnostic = (args.options.whole || args.options.scalar) && args.against.is_none();

    let lanes: std::collections::BTreeSet<usize> = ks
        .iter()
        .filter(|k| k.has_simd())
        .map(|k| k.lanes)
        .collect();
    writeln!(
        out,
        "{}: {}, execution {}, {} points in {} tile{}{}, {} leaves, SIMD {}",
        rec.name,
        rec.method,
        resolved.execution,
        rec.num_points,
        rec.tiles.len(),
        if rec.tiles.len() == 1 { "" } else { "s" },
        if args.options.whole {
            " (replayed as one)"
        } else {
            ""
        },
        calls.len(),
        if args.options.scalar {
            "off".to_string()
        } else if lanes.is_empty() {
            "none".to_string()
        } else {
            format!(
                "{} lanes",
                lanes
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join("/")
            )
        }
    )?;
    for w in resolved.warnings.iter() {
        writeln!(out, "  warning: {}", w)?;
    }

    // the planes read before they are written must be inputs: sources, momenta, the zero
    // plane, parameters (check_arena.py's rule; anything else would be stale data)
    let written_first = recording::first_written(
        calls.iter().map(|c| (&c.inputs[..], &c.outputs[..])),
        rec.num_planes,
    );
    let mut touched = vec![false; rec.num_planes];
    for c in calls.iter() {
        for &p in c.inputs.iter().chain(&c.outputs) {
            touched[p as usize] = true;
        }
    }
    for &p in &written_first {
        touched[p as usize] = false;
    }
    let read_first: Vec<u32> = (0..rec.num_planes as u32)
        .filter(|&p| touched[p as usize])
        .collect();
    let stale: Vec<u32> = read_first
        .iter()
        .copied()
        .filter(|&p| match rec.region_of(p) {
            ("current_re" | "current_im", c) => !rec.source_planes.contains(&c),
            ("amplitude_re" | "amplitude_im", _) => true,
            _ => false,
        })
        .collect();
    write!(
        out,
        "  {} planes read first, {} written first{}",
        read_first.len(),
        written_first.len(),
        if args.options.poison {
            " (poisoned)"
        } else {
            ""
        }
    )?;
    if !stale.is_empty() {
        let (region, index) = rec.region_of(stale[0]);
        write!(
            out,
            "; warning: {} planes read before written are not inputs (first: {} {}[{}])",
            stale.len(),
            stale[0],
            region,
            index
        )?;
    }
    writeln!(out)?;

    let selected = |c: &replay::LeafCall| match &args.leaf {
        None => true,
        Some(p) => [c.label.as_str(), c.member.as_str()]
            .iter()
            .any(|t| pacman::glob(p.as_bytes(), t.as_bytes())),
    };
    if args.verbose || args.time {
        for (c, s) in calls.iter().zip(&outcome.stats) {
            if !selected(c) {
                continue;
            }
            write!(out, "  {:<14} {}", c.label, short(&c.member))?;
            if args.verbose {
                write!(
                    out,
                    "  in {} out {}  simd {} scalar {}{}{}",
                    c.inputs.len(),
                    c.outputs.len(),
                    s.simd_calls,
                    s.scalar_calls,
                    if s.redone > 0 {
                        format!(" (redone {})", s.redone)
                    } else {
                        String::new()
                    },
                    if c.simd_safe { "" } else { " (aliased)" }
                )?;
            }
            if args.time {
                let us = s.time.as_secs_f64() * 1e6;
                write!(
                    out,
                    "  {:.1} µs, {:.3} µs/point",
                    us,
                    us / rec.num_points as f64
                )?;
            }
            writeln!(out)?;
        }
    }
    if args.time {
        let total: f64 = outcome
            .stats
            .iter()
            .map(|s| s.time.as_secs_f64())
            .sum::<f64>()
            * 1e6;
        writeln!(
            out,
            "  kernels: {:.1} µs, {:.3} µs/point{}",
            total,
            total / rec.num_points as f64,
            if args.options.repeat > 1 {
                format!(" (per leaf the best of {} runs)", args.options.repeat)
            } else {
                String::new()
            }
        )?;
    }

    if let Some(dir) = &args.dump {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("cannot create `{}`", dir.display()))?;
        let file = dir.join(format!("{}.replay", rec.name));
        let bytes: Vec<u8> = outcome.arena.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&file, bytes)
            .with_context(|| format!("cannot write `{}`", file.display()))?;
    }

    if outcome.diffs.is_empty() {
        writeln!(
            out,
            "  equal to {}: {} planes × {} points, bit for bit",
            reference, rec.num_planes, rec.num_points
        )?;
        return Ok(true);
    }
    let values: usize = outcome.diffs.iter().map(|d| d.points).sum();
    writeln!(
        out,
        "  DIFFERENT from {}: {} planes, {} values{}",
        reference,
        outcome.diffs.len(),
        values,
        if diagnostic {
            " (diagnostic run: not an error)"
        } else {
            ""
        }
    )?;
    // the amplitudes, which are what RustiCol returns (the currents are intermediate)
    let amplitudes: Vec<&recording::PlaneDiff> = outcome
        .diffs
        .iter()
        .filter(|d| rec.region_of(d.plane).0.starts_with("amplitude"))
        .collect();
    if !amplitudes.is_empty() {
        // relative to the largest amplitude: many amplitudes vanish up to rounding, and their
        // differences are as large as they are
        let (_, first, n) = rec.arena().regions()[2];
        let scale = rec.post
            [first as usize * rec.num_points..(first + 2 * n) as usize * rec.num_points]
            .iter()
            .fold(0.0f64, |m, x| m.max(x.abs()));
        let worst = amplitudes.iter().map(|d| d.max_abs).fold(0.0, f64::max);
        writeln!(
            out,
            "    amplitudes: {} of {} planes differ, by at most {:.1e}, {} of the largest amplitude",
            amplitudes.len(),
            2 * rec.amplitude_planes,
            worst,
            recording::show_relative(if scale > 0.0 && scale.is_finite() {
                worst / scale
            } else {
                f64::INFINITY
            })
        )?;
    }
    // blamed on the leaf that writes the plane last
    let writers = replay::last_writers(calls, rec.num_planes);
    let mut by_writer: Vec<(Option<usize>, Vec<&recording::PlaneDiff>)> = Vec::new();
    for d in outcome.diffs.iter() {
        let w = writers[d.plane as usize];
        match by_writer.iter_mut().find(|(x, _)| *x == w) {
            Some((_, list)) => list.push(d),
            None => by_writer.push((w, vec![d])),
        }
    }
    by_writer.sort_by_key(|(w, _)| w.unwrap_or(usize::MAX));
    for (w, list) in by_writer.iter() {
        let name = match w {
            Some(i) if !selected(&calls[*i]) => continue,
            Some(i) => format!("{} ({})", calls[*i].label, short(&calls[*i].member)),
            None => "written by no leaf".to_string(),
        };
        writeln!(
            out,
            "    {}: {} plane{}, {} values",
            name,
            list.len(),
            if list.len() == 1 { "" } else { "s" },
            list.iter().map(|d| d.points).sum::<usize>()
        )?;
        let shown = if args.verbose { list.len() } else { 1 };
        for d in list.iter().take(shown) {
            writeln!(out, "      {}", recording::describe_diff(&rec, d))?;
        }
    }
    Ok(false)
}

/// `pacman runner`.
fn runner(args: &Runner) -> Result<()> {
    let (processes, pacbin) = bindings::load(&args.artifact)?;
    let pacbin_path = pacbin.ok_or_else(|| {
        anyhow!(
            "`{}` has no evaluators.pacbin: `runner` needs the artifact directory",
            args.artifact.display()
        )
    })?;
    // the execution: the recording's (with its map, which must be bindings' own), or named
    let (process, execution) = match &args.recording {
        Some(json) => {
            let rec = Recording::read(json)?;
            let resolved = replay::leaf_calls(&rec, &processes)?;
            let name = resolved.execution.split(" = ").next().unwrap_or_default();
            let process = processes
                .iter()
                .find(|p| p.prefix == format!("processes/{}", rec.process))
                .expect("leaf_calls found it");
            let e = process
                .executions
                .iter()
                .find(|e| e.name == name)
                .expect("leaf_calls found it");
            if e.canonical_to_physical != rec.canonical_to_physical {
                return Err(anyhow!(
                    "RustiCol's current map (the recording's) differs from pacman's for `{}`: \
                     a runner compiled with pacman's would not fit RustiCol's arena",
                    e.name
                ));
            }
            (process, e)
        }
        None => {
            let pattern = args.execution.as_deref().unwrap_or_default();
            let found: Vec<_> = processes
                .iter()
                .flat_map(|p| p.executions.iter().map(move |e| (p, e)))
                .filter(|(_, e)| pacman::glob(pattern.as_bytes(), e.name.as_bytes()))
                .collect();
            match found.as_slice() {
                [one] => *one,
                [] => return Err(anyhow!("no execution matches `{}`", pattern)),
                many => {
                    return Err(anyhow!(
                        "`{}` matches {} executions: {}",
                        pattern,
                        many.len(),
                        many.iter()
                            .map(|(p, e)| format!("{} of {}", e.name, p.prefix))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }
            }
        }
    };
    let spec = symjit_utils::runner::Spec::new(process, execution)?;

    let container = read(&pacbin_path)?;
    let members: HashMap<&str, &[u8]> = container
        .members
        .iter()
        .filter(|m| m.kind == Kind::SymjitApplication)
        .map(|m| (m.path.as_str(), container.payload(m)))
        .collect();
    let start = std::time::Instant::now();
    let (objects, of_leaf) = symjit_utils::runner::objects(&spec, &members)?;
    let dir = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("runner-{}", spec.process)));
    symjit_utils::runner::write(&dir, &spec, &objects, &of_leaf)?;
    let code: usize = objects.iter().map(Vec::len).sum();
    eprintln!(
        "pacman: wrote the runner of {} of {} to {}: {} leaves, {} kernel object{} ({} bytes) in {:.2} s",
        spec.execution,
        spec.process,
        dir.display(),
        spec.calls.len(),
        objects.len(),
        if objects.len() == 1 { "" } else { "s" },
        code,
        start.elapsed().as_secs_f64()
    );
    if args.build {
        let start = std::time::Instant::now();
        symjit_utils::runner::build(&dir, &args.cc, objects.len())?;
        eprintln!(
            "pacman: built {0}/runner and {0}/librunner.so in {1:.1} s",
            dir.display(),
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let command = match parse(std::env::args().skip(1).collect()) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("pacman: error: {}", msg);
            eprintln!("Try `pacman --help`.");
            return ExitCode::from(2);
        }
    };
    let result = match command {
        Command::Help => {
            println!("{}", USAGE);
            Ok(())
        }
        Command::Version => {
            println!("pacman {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Info(inputs, options) => {
            let mut status = Ok(());
            for (i, input) in inputs.iter().enumerate() {
                let report = read(input)
                    .and_then(|p| pacman::describe(&input.display().to_string(), &p, &options));
                match report {
                    Ok(text) => {
                        if i > 0 {
                            println!();
                        }
                        print!("{}", text);
                    }
                    Err(err) => {
                        eprintln!("pacman: error: {:#}", err);
                        status = Err(());
                    }
                }
            }
            return if status.is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            };
        }
        // a closed pipe (`pacman list ... | head`) ends the listing quietly
        Command::List(input, kind, sha) => {
            list(&input, kind, sha).or_else(|err| match err.downcast_ref::<std::io::Error>() {
                Some(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                _ => Err(err),
            })
        }
        Command::Extract(args) => extract(&args),
        Command::Plan(input, pack, options) => plan(&input, pack.as_deref(), &options),
        Command::Replay(args) => {
            return match replay_all(&args) {
                Ok(true) => ExitCode::SUCCESS,
                Ok(false) => ExitCode::from(1),
                Err(err) => {
                    eprintln!("pacman: error: {:#}", err);
                    ExitCode::from(1)
                }
            };
        }
        Command::Runner(args) => runner(&args),
        Command::Bindings(input, options) => {
            bindings(&input, &options).or_else(|err| match err.downcast_ref::<std::io::Error>() {
                Some(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                _ => Err(err),
            })
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("pacman: error: {:#}", err);
            ExitCode::from(1)
        }
    }
}
