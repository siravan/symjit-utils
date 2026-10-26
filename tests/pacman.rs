//! pacman end to end: containers written by `pacbin::write` (byte for byte pyAmpliCol's writer,
//! checked on real artifacts with `PACMAN_SAMPLES`) holding applications compiled with symjit's
//! Rust API; the reports are compared with what symjit reads and measures.
//!
//! `PACMAN_SAMPLES=DIR` (e.g. a directory of pyAmpliCol artifacts) also reads every `*.pacbin`
//! under DIR: written back, it must give the same bytes, and the digests the artifact's
//! `artifact.json` declares for it must match.
//!
//! `PACMAN_REPLAY_SAMPLES=DIR` replays RustiCol arena recordings (`pacman replay`): every
//! `DIR/<case>/` that holds an `artifact` (a link to the pyAmpliCol artifact the recordings were
//! made with) and recordings (`*.json`, `.pre`, `.post`), and builds and runs a runner per
//! recording. The tile runner and the runner's generated C are tested on kernels compiled from
//! Symbolica instructions (`replay::tests`, `runner::tests`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use symjit::{
    var, Application, Compiler, CompilerType, Composer, Config, Slot, Storage, Translator,
};
use symjit_utils::bindings;
use symjit_utils::pacbin::{self, Kind, Pacbin};
use symjit_utils::plan;
use symjit_utils::sha256::{hex, Sha256};

const PACMAN: &str = env!("CARGO_BIN_EXE_pacman");

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("pacman")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> Output {
    Command::new(PACMAN).args(args).output().unwrap()
}

fn stdout(out: &Output) -> String {
    assert!(
        out.status.success(),
        "pacman failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

fn saved(app: &Application) -> Vec<u8> {
    let mut bytes = Vec::new();
    app.save(&mut bytes).unwrap();
    bytes
}

/// x*y + sin(x), x - y/3 (real, parameters)
fn real() -> Vec<u8> {
    let mut config = Config::new(CompilerType::Native, 0).unwrap();
    config.set_opt_level(2);
    config.set_simd(true);
    let (x, y) = (var("x"), var("y"));
    let mut app = Compiler::with_config(config)
        .compile_params(
            &[],
            &[&(&x * &y) + &x.sin(), &x - &(&y / &symjit::Expr::from(3.0))],
            &[x.clone(), y.clone()],
        )
        .unwrap();
    app.prepare_simd();
    saved(&app)
}

/// z / w + sin(z) * w, complex, Symbolica-style (`Translator`), as pyAmpliCol's kernels
fn complex() -> Vec<u8> {
    let mut config = Config::new(CompilerType::Native, 0).unwrap();
    config.set_opt_level(3);
    config.set_simd(true);
    config.set_complex(true);
    config.set_symbolica(true);
    let mut t = Translator::new(config);
    t.set_num_params(2);
    t.append_div(
        &Slot::Temp(0),
        &Slot::Param(0),
        &Slot::Param(1),
        false,
        false,
    )
    .unwrap();
    t.append_fun(&Slot::Temp(1), "sin", &[Slot::Param(0)], false)
        .unwrap();
    t.append_mul(&Slot::Temp(1), &[Slot::Temp(1), Slot::Param(1)], 0)
        .unwrap();
    t.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Temp(1)], 0)
        .unwrap();
    let mut app = t.compile().unwrap();
    app.prepare_simd();
    saved(&app)
}

struct Sample {
    file: PathBuf,
    real: Vec<u8>,
    complex: Vec<u8>,
    state: Vec<u8>,
}

/// A container as pyAmpliCol lays one out: per chunk a plain and a plane application and a
/// Symbolica state; one kernel stored twice.
fn sample(name: &str) -> Sample {
    let d = dir(name);
    let (real, complex) = (real(), complex());
    let state = b"not really a Symbolica evaluator".to_vec();
    let bytes = pacbin::write(&[
        (
            "p/evaluators/a_1.symjit".into(),
            Kind::SymjitApplication,
            &real[..],
        ),
        (
            "p/evaluators/a_1.plane.symjit".into(),
            Kind::SymjitApplication,
            &complex[..],
        ),
        (
            "p/evaluators/a_1.evaluator.bin".into(),
            Kind::SymbolicaExactState,
            &state[..],
        ),
        (
            "p/sector-0/evaluators/a_2.plane.symjit".into(),
            Kind::SymjitApplication,
            &complex[..],
        ),
    ])
    .unwrap();
    let file = d.join("evaluators.pacbin");
    std::fs::write(&file, bytes).unwrap();
    Sample {
        file,
        real,
        complex,
        state,
    }
}

fn read_sjb(bytes: &[u8]) -> (symjit::Program, usize) {
    let empty = Config::new(CompilerType::ByteCode, 0).unwrap();
    let (prog, _, mir) = Application::read_sjb(&mut &bytes[..], &empty).unwrap();
    (prog, mir.code.ip)
}

fn field<'a>(report: &'a str, key: &str) -> &'a str {
    report
        .lines()
        .find(|l| l.trim_start().starts_with(key))
        .unwrap_or_else(|| panic!("no `{}` in\n{}", key, report))
        .trim_start()[key.len()..]
        .trim()
}

#[test]
fn info_reports_the_kernels() {
    let s = sample("info");
    let f = s.file.to_str().unwrap();
    let report = stdout(&run(&["info", f]));

    assert!(
        report.contains("4 members (digests verified)"),
        "{}",
        report
    );
    assert!(report.contains("symjit-application (kernels)"));
    assert_eq!(
        field(&report, "kernels"),
        "3: 1 plain (.symjit), 2 plane (.plane.symjit)"
    );
    assert!(field(&report, "distinct").starts_with("2 ("), "{}", report);
    assert_eq!(field(&report, "sjb version"), "4 (3)");
    assert_eq!(field(&report, "numbers"), "complex (2), real (1)");
    assert_eq!(field(&report, "opt level"), "2 (1), 3 (2)");

    // the largest kernels, with the sizes symjit reads
    let (prog, mir) = read_sjb(&s.complex);
    let line = report
        .lines()
        .find(|l| l.ends_with("p/evaluators/a_1.plane.symjit"))
        .unwrap();
    let cols: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(cols[0], mir.to_string());
    assert_eq!(cols[1], (prog.count_states + prog.count_params).to_string());
    assert_eq!(cols[2], prog.count_obs.to_string());
    assert_eq!(cols[3], s.complex.len().to_string());

    // --top limits the list
    let report = stdout(&run(&["info", "--top", "1", f]));
    assert!(report.contains("Largest kernels (1 of 3, by MIR instructions)"));
    let report = stdout(&run(&["info", "--top", "0", f]));
    assert!(!report.contains("by MIR instructions"));
}

#[test]
fn info_compiles_on_request() {
    let s = sample("compile");
    let f = s.file.to_str().unwrap();
    let report = stdout(&run(&["info", "--compile", "--top", "all", f]));
    let config = Config::new(CompilerType::ByteCode, 0).unwrap();
    for (path, bytes) in [
        ("p/evaluators/a_1.symjit", &s.real),
        ("p/evaluators/a_1.plane.symjit", &s.complex),
    ] {
        let mut app = Application::load(&mut &bytes[..], &config).unwrap();
        app.prepare_simd();
        let line = report.lines().find(|l| l.ends_with(path)).unwrap();
        let cols: Vec<String> = line
            .split_whitespace()
            .map(|c| c.replace(',', ""))
            .collect();
        assert_eq!(
            cols[4],
            app.measure("ker-scalar-size").to_string(),
            "{}",
            line
        );
        assert_eq!(
            cols[5],
            app.measure("ker-simd-size").to_string(),
            "{}",
            line
        );
    }
    assert!(report.contains("scalar code"));
}

#[test]
fn list_prints_every_member() {
    let s = sample("list");
    let f = s.file.to_str().unwrap();
    let text = stdout(&run(&["list", f]));
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4);
    // sorted by path, as stored
    assert!(lines[0].starts_with("symbolica-exact-state"));
    assert!(lines[0].ends_with("p/evaluators/a_1.evaluator.bin"));

    let text = stdout(&run(&[
        "list",
        "--kind",
        "symjit-application",
        "--sha256",
        f,
    ]));
    assert_eq!(text.lines().count(), 3);
    let want = hex(&Sha256::digest(&s.real));
    assert!(
        text.lines()
            .any(|l| l.contains(&want) && l.ends_with("a_1.symjit")),
        "{}",
        text
    );
    assert_eq!(stdout(&run(&["list", "-k", "2", f])).lines().count(), 1);
}

#[test]
fn extract_writes_members() {
    let s = sample("extract");
    let f = s.file.to_str().unwrap();
    let out = s.file.parent().unwrap();

    // the kernels by default
    let d = out.join("kernels");
    stdout(&run(&["extract", f, "-o", d.to_str().unwrap()]));
    assert_eq!(
        std::fs::read(d.join("p/evaluators/a_1.symjit")).unwrap(),
        s.real
    );
    assert_eq!(
        std::fs::read(d.join("p/sector-0/evaluators/a_2.plane.symjit")).unwrap(),
        s.complex
    );
    assert!(!d.join("p/evaluators/a_1.evaluator.bin").exists());

    // everything
    let d = out.join("all");
    stdout(&run(&["extract", "--all", f, "-o", d.to_str().unwrap()]));
    assert_eq!(
        std::fs::read(d.join("p/evaluators/a_1.evaluator.bin")).unwrap(),
        s.state
    );

    // by pattern (any kind), and one member to standard output
    let d = out.join("some");
    stdout(&run(&["extract", f, "*/a_1.*", "-o", d.to_str().unwrap()]));
    let mut n = 0;
    for e in std::fs::read_dir(d.join("p/evaluators")).unwrap() {
        assert!(e.unwrap().file_name().to_str().unwrap().starts_with("a_1."));
        n += 1;
    }
    assert_eq!(n, 3);
    let one = run(&["extract", f, "p/evaluators/a_1.plane.symjit", "-o", "-"]);
    assert!(one.status.success());
    assert_eq!(one.stdout, s.complex);

    // the default directory: the container's name without `.pacbin`, here
    let cwd = out.join("cwd");
    std::fs::create_dir_all(&cwd).unwrap();
    let status = Command::new(PACMAN)
        .args(["extract", f])
        .current_dir(&cwd)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(cwd.join("evaluators/p/evaluators/a_1.symjit").exists());
}

#[test]
fn errors() {
    let s = sample("errors");
    let f = s.file.to_str().unwrap();

    let code = |out: Output| {
        (
            out.status.code().unwrap(),
            String::from_utf8(out.stderr).unwrap(),
        )
    };

    let (c, e) = code(run(&["extract", f, "nothing*"]));
    assert_eq!(c, 1);
    assert!(e.contains("no member matches `nothing*`"), "{}", e);
    let (c, e) = code(run(&["extract", f, "*", "-o", "-"]));
    assert_eq!(c, 1);
    assert!(e.contains("writes one member, but 4 are selected"), "{}", e);
    let (c, _) = code(run(&["bogus", f]));
    assert_eq!(c, 2);
    let (c, _) = code(run(&["info", "--kind", "1", f]));
    assert_eq!(c, 2);
    let (c, _) = code(run(&["extract", "--all", "--kind", "1", f]));
    assert_eq!(c, 2);
    let (c, _) = code(run(&["info", "--top", "many", f]));
    assert_eq!(c, 2);

    // a damaged payload: the digest check fails
    let mut bytes = std::fs::read(&s.file).unwrap();
    let p = Pacbin::read(bytes.clone(), true).unwrap();
    let m = p.member("p/evaluators/a_1.symjit").unwrap();
    bytes[m.offset as usize + 20] ^= 0x40;
    let bad = s.file.with_file_name("bad.pacbin");
    std::fs::write(&bad, bytes).unwrap();
    let (c, e) = code(run(&["info", bad.to_str().unwrap()]));
    assert_eq!(c, 1);
    assert!(
        e.contains("member digest mismatch: p/evaluators/a_1.symjit"),
        "{}",
        e
    );

    // an unreadable kernel is reported, not fatal
    let junk = b"junk".to_vec();
    let bytes = pacbin::write(&[("k.symjit".into(), Kind::SymjitApplication, &junk[..])]).unwrap();
    let junk_file = s.file.with_file_name("junk.pacbin");
    std::fs::write(&junk_file, bytes).unwrap();
    let report = stdout(&run(&["info", junk_file.to_str().unwrap()]));
    assert!(report.contains("Unreadable kernels"), "{}", report);
    assert!(
        report.contains("k.symjit: not a saved symjit application"),
        "{}",
        report
    );
}

/// Real containers (`PACMAN_SAMPLES`): read with every check, written back byte for byte, and
/// authenticated by their artifact's manifest where there is one.
#[test]
fn real_containers() {
    let Ok(root) = std::env::var("PACMAN_SAMPLES") else {
        eprintln!("PACMAN_SAMPLES is not set: no real containers checked");
        return;
    };
    let mut files = Vec::new();
    find(Path::new(&root), &mut files);
    assert!(!files.is_empty(), "no *.pacbin under {}", root);

    for file in files.iter() {
        let bytes = std::fs::read(file).unwrap();
        let p = Pacbin::read(bytes.clone(), true)
            .unwrap_or_else(|e| panic!("{}: {:#}", file.display(), e));
        assert!(
            pacbin::write(&p.contents()).unwrap() == bytes,
            "{}: written back differently",
            file.display()
        );

        // `artifact.json` beside `evaluators.pacbin`: the payload digest and size, and the
        // container extension's index digest, count and unpacked size
        let manifest = file.with_file_name("artifact.json");
        if file.file_name().unwrap() == "evaluators.pacbin" && manifest.exists() {
            // compact or indented JSON alike
            let text: String = std::fs::read_to_string(&manifest)
                .unwrap()
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            let file_sha = hex(&Sha256::digest(&bytes));
            assert!(
                text.contains(&format!("\"sha256\":\"{}\"", file_sha)),
                "{}",
                file.display()
            );
            assert!(text.contains(&format!("\"index_sha256\":\"{}\"", hex(&p.index_sha256))));
            assert!(text.contains(&format!("\"member_count\":{}", p.members.len())));
            let unpacked: u64 = p.members.iter().map(|m| m.length).sum();
            assert!(text.contains(&format!("\"unpacked_size_bytes\":{}", unpacked)));
            assert!(text.contains(&format!("\"size_bytes\":{}", bytes.len())));
        }
        eprintln!("{}: {} members", file.display(), p.members.len());

        // a recurrence plan: its row groups cover each row table once, in order, and every
        // executor they name is in the kernel pack; `pacman plan` reports it
        if file.file_name().unwrap() == "recurrence-runtime.pacbin" {
            let loaded = plan::load(file, None).unwrap();
            assert_eq!(loaded.len(), 1);
            let l = &loaded[0];
            let executors = l.executors.as_ref().expect("the artifact's kernel pack");
            for (role, table) in [
                (plan::Role::Contribution, "contributions"),
                (plan::Role::Finalization, "finalizations"),
                (plan::Role::Closure, "closures"),
            ] {
                let mut next = 0;
                for g in l.plan.row_groups.iter().filter(|g| g.role == role) {
                    assert_eq!(g.row_start, next, "{} rows out of order", table);
                    next += g.rows as u64;
                }
                assert_eq!(next, l.plan.count(table), "{} not covered", table);
            }
            for g in l.plan.row_groups.iter() {
                assert!(g.executor == plan::NONE || executors.contains_key(&g.executor));
            }
            let artifact = file.ancestors().nth(4).unwrap();
            let report = stdout(&run(&["plan", artifact.to_str().unwrap()]));
            assert!(report.contains("Executors run"), "{}", report);
            assert!(report.contains("symjit kernels run"), "{}", report);
        }

        // a compiled-mode artifact: every leaf's table fits its kernel, points into the arena,
        // and no stage writes a plane it reads or writes twice (pyAmpliCol's no-aliasing
        // contract); `pacman bindings` flags no kernel
        if file.file_name().unwrap() == "evaluators.pacbin" {
            let artifact = file.parent().unwrap();
            let (processes, pacbin) = bindings::load(artifact).unwrap();
            let shapes = bindings::kernel_shapes(&pacbin.unwrap()).unwrap();
            for p in processes.iter() {
                for e in p.executions.iter() {
                    for st in e.stages.iter() {
                        let mut read = std::collections::HashSet::new();
                        let mut written = std::collections::HashSet::new();
                        for l in st.leaves.iter() {
                            let shape = shapes[&format!("{}/{}", p.prefix, l.application)]
                                .as_ref()
                                .unwrap();
                            assert_eq!(*shape, (l.inputs.len(), l.outputs.len()));
                            for d in l.inputs.iter().chain(l.outputs.iter()) {
                                if let bindings::Plane::Current { physical, .. } = d {
                                    assert!(*physical < e.physical_planes);
                                }
                            }
                            read.extend(l.inputs.iter().map(|d| (d.buffer(), d.index())));
                            for d in l.outputs.iter() {
                                assert!(written.insert((d.buffer(), d.index())), "{}", st.label);
                            }
                        }
                        assert!(written.is_disjoint(&read), "{}", st.label);
                    }
                }
            }
            let report = stdout(&run(&["bindings", artifact.to_str().unwrap()]));
            assert!(!report.contains("  ["), "{}", report);

            // `extract` writes each leaf's mapping: one line per descriptor, in the arena
            let d = dir(&format!(
                "mapping-{}",
                files.iter().position(|f| f == file).unwrap()
            ));
            stdout(&run(&[
                "extract",
                file.to_str().unwrap(),
                "-o",
                d.to_str().unwrap(),
            ]));
            for p in processes.iter().filter(|p| !p.executions.is_empty()) {
                assert!(d.join(&p.prefix).join("arena.txt").is_file());
                for e in p.executions.iter() {
                    let arena = e.arena();
                    for (_, _, l) in e.leaves() {
                        let f = d
                            .join(&p.prefix)
                            .join(&l.application)
                            .with_extension("mapping");
                        let text = std::fs::read_to_string(&f).unwrap();
                        assert_eq!(text, arena.mapping(l));
                        let mut n = 0;
                        for (i, line) in text.lines().enumerate() {
                            let (slot, plane) = line.split_once(' ').unwrap();
                            assert_eq!(slot.parse::<usize>().unwrap(), i);
                            assert!(plane.parse::<u32>().unwrap() < arena.len());
                            n += 1;
                        }
                        assert_eq!(n, l.inputs.len() + l.outputs.len());
                    }
                }
            }
        }
    }
}

#[test]
fn bindings_errors() {
    let s = sample("bindings");
    let d = s.file.parent().unwrap();
    let out = run(&["bindings", d.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("not a pyAmpliCol artifact"), "{}", e);
    let out = run(&["bindings", s.file.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let out = run(&["bindings", "--leaf"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn plan_errors() {
    let s = sample("plan");
    let d = s.file.parent().unwrap();
    // not a recurrence artifact
    let out = run(&["plan", d.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("not a recurrence-mode artifact"), "{}", e);
    // a container without a plan
    let out = run(&["plan", s.file.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("has no recurrence plan"), "{}", e);
    let out = run(&["plan", "--groups", "x", s.file.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn runner_errors() {
    let s = sample("runner");
    let d = s.file.parent().unwrap().to_str().unwrap();
    // a recording or an execution
    let out = run(&["runner", d]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&["runner", "--cc"]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&["runner", "--execution", "top", d]);
    assert_eq!(out.status.code(), Some(1));
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("not a pyAmpliCol artifact"), "{}", e);
}

#[test]
fn replay_errors() {
    let s = sample("replay");
    let d = s.file.parent().unwrap().to_str().unwrap();
    // an artifact and a recording at least
    let out = run(&["replay", d]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&["replay", "--repeat", "0", d, d]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&["replay", d, d]);
    assert_eq!(out.status.code(), Some(1));
    let e = String::from_utf8(out.stderr).unwrap();
    assert!(e.contains("not a pyAmpliCol artifact"), "{}", e);
}

/// Real recordings (`PACMAN_REPLAY_SAMPLES`): each reproduces its `.post` bit for bit, with the
/// recorded tiles and as one tile (`--whole`), and `--dump` writes the arena replayed; the
/// runner of its execution (`pacman runner`, built with `cc`) gives the scalar replay's arena.
#[test]
fn real_recordings() {
    let Ok(root) = std::env::var("PACMAN_REPLAY_SAMPLES") else {
        eprintln!("PACMAN_REPLAY_SAMPLES is not set: no recordings replayed");
        return;
    };
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("artifact").is_dir())
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no DIR/<case>/artifact under {}", root);
    for case in cases.iter() {
        // recordings of two cases can have the same names
        let dump = dir(&format!(
            "replay-{}",
            case.file_name().unwrap().to_string_lossy()
        ));
        let recordings = std::fs::read_dir(case)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .count();
        assert!(recordings > 0, "{}: no recordings", case.display());
        let artifact = case.join("artifact");
        let (a, c) = (artifact.to_str().unwrap(), case.to_str().unwrap());
        for extra in [&[][..], &["--whole"][..]] {
            let mut args = vec!["replay"];
            args.extend(extra);
            args.extend(["--dump", dump.to_str().unwrap(), a, c]);
            let report = stdout(&run(&args));
            assert_eq!(
                report.matches("bit for bit").count(),
                recordings,
                "{} {:?}:\n{}",
                case.display(),
                extra,
                report
            );
            assert!(!report.contains("warning"), "{}", report);
            eprintln!("{} {:?}: {} recordings", case.display(), extra, recordings);
        }
        for e in std::fs::read_dir(case).unwrap().flatten() {
            let json = e.path();
            if json.extension().is_some_and(|x| x == "json") {
                let name = json.file_stem().unwrap().to_str().unwrap();
                let replayed = std::fs::read(dump.join(format!("{}.replay", name))).unwrap();
                assert!(replayed == std::fs::read(json.with_extension("post")).unwrap());

                // the recording's execution as a runner: its arena equals the scalar replay's
                let runner = dump.join(format!("runner-{}", name));
                let j = json.to_str().unwrap();
                stdout(&run(&["runner", "-o", runner.to_str().unwrap(), a, j]));
                let out = Command::new(runner.join("runner"))
                    .arg(&dump)
                    .arg(&json)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "{}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        let report = stdout(&run(&[
            "replay",
            "--scalar",
            "--against",
            dump.to_str().unwrap(),
            a,
            c,
        ]));
        assert_eq!(
            report.matches("bit for bit").count(),
            recordings,
            "{}: runner:\n{}",
            case.display(),
            report
        );
        eprintln!(
            "{}: {} recordings through runners",
            case.display(),
            recordings
        );
    }
}

fn find(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.is_dir() {
            find(&path, out);
        } else if path.extension().is_some_and(|x| x == "pacbin") {
            out.push(path);
        }
    }
}
