//! sjb2wasm end to end: models compiled with symjit's Rust API are saved (`.sjb`), compiled by
//! the binary, validated with wabt's `wasm-validate` and run in Node (`tests/wasm_runner.mjs`);
//! their outputs are compared with the JIT. The steps that need `node` or `wasm-validate` are
//! skipped when the tool is not on PATH.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use symjit::{var, Application, Compiler, CompilerType, Config, Expr, Storage};

const SJB2WASM: &str = env!("CARGO_BIN_EXE_sjb2wasm");

const POINTS: [[f64; 3]; 4] = [
    [0.5, 2.0, 0.25],
    [-1.25, 0.75, 1.5],
    [1.5, -0.5, 0.125],
    [0.0, 3.0, -0.75],
];

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sjb2wasm")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn have(tool: &str) -> bool {
    let ok = Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipped: `{}` is not available", tool);
    }
    ok
}

fn config(ty: CompilerType, opt_level: u8, fastmath: bool) -> Config {
    let mut config = Config::new(ty, 0).unwrap();
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config.set_fastmath(fastmath);
    config
}

type Model = (Vec<Expr>, Vec<Expr>);

/// sin(x)*y + exp(-x*x), cos(y)/(1 + x*x), x*y - z
fn trig() -> Model {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&x.sin() * &y) + &(-&(&x * &x)).exp();
    let b = &y.cos() / &(&one + &(&x * &x));
    let c = &(&x * &y) - &z;
    (vec![x, y, z], vec![a, b, c])
}

/// rational functions and square roots only: exact in WebAssembly without fused operations
fn rational() -> Model {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&(&x * &x) - &(&y * &z)) / &(&one + &(&z * &z));
    let b = &(&(&x * &y).sqrt() + &z) * &(&x - &Expr::from(0.75));
    let c = &(&one / &(&(&(&x * &x) + &(&y * &y)) + &one)) - &z;
    (vec![x, y, z], vec![a, b, c])
}

/// sum of (k x + y)^2 / (1 + z^2 k), k = 1..39: register pressure
fn big() -> Model {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let mut s = Expr::from(0.0);
    let mut t = Expr::from(1.0);
    for k in 1..40 {
        let k = Expr::from(k as f64);
        let u = &(&k * &x) + &y;
        s = &s + &(&(&u * &u) / &(&one + &(&(&z * &z) * &k)));
        t = &t * &(&one + &(&u / &(&k * &k)));
    }
    (vec![x, y, z], vec![s, t])
}

fn compile(model: &Model, config: Config) -> Application {
    Compiler::with_config(config)
        .compile(&model.0, &model.1)
        .unwrap()
}

fn save(app: &Application, path: &Path) {
    let mut f = std::fs::File::create(path).unwrap();
    app.save(&mut f).unwrap();
}

fn sjb2wasm(d: &Path, args: &[&str]) -> Output {
    Command::new(SJB2WASM)
        .args(args)
        .current_dir(d)
        .output()
        .unwrap()
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "sjb2wasm failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn fails(out: &Output, status: i32, message: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(status), "{}", stderr);
    assert!(stderr.contains(message), "{}", stderr);
}

/// The JIT's outputs (this machine's native kernel) at `POINTS`.
fn reference(model: &Model, opt_level: u8, fastmath: bool) -> Vec<Vec<f64>> {
    let mut app = compile(model, config(CompilerType::Native, opt_level, fastmath));
    POINTS.iter().map(|p| app.call(p)).collect()
}

fn validate(module: &Path) {
    if !have("wasm-validate") {
        return;
    }
    let out = Command::new("wasm-validate").arg(module).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Runs `module` in Node at `POINTS` (mode `direct`, `indirect` or `fast`); None without node.
fn run(module: &Path, mode: &str, count_states: usize, count_obs: usize) -> Option<Vec<Vec<f64>>> {
    if !have("node") {
        return None;
    }
    let runner = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/wasm_runner.mjs");
    let points: Vec<String> = POINTS
        .iter()
        .map(|p| format!("[{:?}, {:?}, {:?}]", p[0], p[1], p[2]))
        .collect();
    let req = format!(
        "{{\"module\": {:?}, \"mode\": {:?}, \"count_states\": {}, \"count_obs\": {}, \
         \"mem_size\": {}, \"points\": [{}], \"params\": []}}",
        module.to_str().unwrap(),
        mode,
        count_states,
        count_obs,
        count_states + count_obs,
        points.join(", ")
    );
    let mut child = Command::new("node")
        .arg(runner)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(req.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("\"status\":[0,0,0,0]"), "{}", text);
    Some(parse_obs(&text))
}

// the `obs` of the runner's answer, {"obs": [[y0, y1, ...], ...], ...}, with the special values
// that JSON cannot hold as strings ("NaN", "Infinity", "-Infinity", "-0")
fn parse_obs(text: &str) -> Vec<Vec<f64>> {
    let obs = text.split("\"obs\":").nth(1).unwrap();
    let obs = &obs[..obs.find("]]").unwrap() + 2];
    obs.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split("],")
        .map(|row| {
            row.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .map(|v| match v.trim().trim_matches('"') {
                    "NaN" => f64::NAN,
                    "Infinity" => f64::INFINITY,
                    "-Infinity" => f64::NEG_INFINITY,
                    "-0" => -0.0,
                    v => v.parse().unwrap(),
                })
                .collect()
        })
        .collect()
}

fn close(a: &[Vec<f64>], b: &[Vec<f64>], rtol: f64) {
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b) {
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(rb) {
            assert!(
                (x - y).abs() <= rtol * y.abs().max(1e-300),
                "{:?} != {:?}",
                a,
                b
            );
        }
    }
}

// the bits of the outputs, with every NaN as the same one: WebAssembly does not specify the
// sign and payload of a NaN result (x86's `sqrtsd` of a negative number sets the sign bit, V8's
// module does not)
fn bits(a: &[Vec<f64>]) -> Vec<Vec<u64>> {
    a.iter()
        .map(|r| {
            r.iter()
                .map(|x| if x.is_nan() { f64::NAN } else { *x }.to_bits())
                .collect()
        })
        .collect()
}

#[test]
fn usage() {
    let d = dir("usage");
    let out = sjb2wasm(&d, &["--help"]);
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("Usage: sjb2wasm"));
    let out = sjb2wasm(&d, &["--version"]);
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("sjb2wasm "));

    fails(&sjb2wasm(&d, &[]), 2, "no input file");
    fails(
        &sjb2wasm(&d, &["--emit", "a.sjb"]),
        2,
        "unknown option `--emit`",
    );
    fails(
        &sjb2wasm(&d, &["a.sjb", "b.sjb"]),
        2,
        "unexpected argument `b.sjb`",
    );
    fails(&sjb2wasm(&d, &["a.sjb", "-o"]), 2, "`-o` needs a file name");
    fails(
        &sjb2wasm(&d, &["missing.sjb"]),
        1,
        "cannot open `missing.sjb`",
    );

    std::fs::write(d.join("junk.sjb"), b"not a saved application").unwrap();
    fails(&sjb2wasm(&d, &["junk.sjb"]), 1, "cannot read `junk.sjb`");
}

#[test]
fn output_names() {
    let d = dir("names");
    save(
        &compile(&trig(), config(CompilerType::Native, 1, true)),
        &d.join("trig.sjb"),
    );
    ok(&sjb2wasm(&d, &["trig.sjb"]));
    let module = std::fs::read(d.join("trig.wasm")).unwrap();
    assert_eq!(&module[..8], b"\0asm\x01\0\0\0");

    ok(&sjb2wasm(&d, &["-o", "other.wasm", "trig.sjb"]));
    assert_eq!(std::fs::read(d.join("other.wasm")).unwrap(), module);
}

#[test]
fn math_functions_match_the_jit() {
    let d = dir("trig");
    for opt_level in [0, 1, 2, 3] {
        let name = format!("trig{}", opt_level);
        save(
            &compile(&trig(), config(CompilerType::Native, opt_level, true)),
            &d.join(format!("{}.sjb", name)),
        );
        ok(&sjb2wasm(&d, &[&format!("{}.sjb", name)]));
        let module = d.join(format!("{}.wasm", name));
        validate(&module);
        for mode in ["direct", "indirect"] {
            if let Some(obs) = run(&module, mode, 3, 3) {
                // JavaScript's Math functions are not glibc's, and the JIT fuses
                // multiply-adds
                close(&obs, &reference(&trig(), opt_level, true), 1e-14);
            }
        }
    }
}

#[test]
fn arithmetic_is_bit_exact() {
    // without fused operations (`fastmath` off) and math library calls, the module computes
    // the JIT's bits
    let d = dir("rational");
    for opt_level in [0, 1, 2, 3] {
        let name = format!("rational{}", opt_level);
        save(
            &compile(&rational(), config(CompilerType::Native, opt_level, false)),
            &d.join(format!("{}.sjb", name)),
        );
        ok(&sjb2wasm(&d, &[&format!("{}.sjb", name)]));
        let module = d.join(format!("{}.wasm", name));
        validate(&module);
        for mode in ["direct", "indirect"] {
            if let Some(obs) = run(&module, mode, 3, 3) {
                assert_eq!(
                    bits(&obs),
                    bits(&reference(&rational(), opt_level, false)),
                    "opt level {} ({})",
                    opt_level,
                    mode
                );
            }
        }
    }
}

#[test]
fn files_saved_for_other_machines() {
    // nothing is compiled for the saved target: an ARM64 or RISC-V model at opt level 3 has
    // more registers than x86-64 (27 against 14), which the module keeps in locals
    let d = dir("targets");
    let reference = reference(&big(), 3, false);
    for (name, ty) in [
        ("arm", CompilerType::Arm),
        ("riscv", CompilerType::RiscV),
        ("amd", CompilerType::Amd),
        ("bytecode", CompilerType::ByteCode),
    ] {
        save(
            &compile(&big(), config(ty, 3, false)),
            &d.join(format!("{}.sjb", name)),
        );
        ok(&sjb2wasm(&d, &[&format!("{}.sjb", name)]));
        let module = d.join(format!("{}.wasm", name));
        validate(&module);
        if let Some(obs) = run(&module, "direct", 3, 2) {
            // the register allocation (and so the order of the CSE'd sums) is the same
            assert_eq!(bits(&obs), bits(&reference), "{}", name);
        }
    }
}

#[test]
fn fast_export() {
    // one observable and no parameters: the module also exports `fast(x, y, z)`
    let d = dir("fast");
    let (states, obs) = rational();
    let model = (states, vec![obs[0].clone()]);
    save(
        &compile(&model, config(CompilerType::Native, 2, false)),
        &d.join("one.sjb"),
    );
    ok(&sjb2wasm(&d, &["one.sjb"]));
    let module = d.join("one.wasm");
    validate(&module);
    if let Some(obs) = run(&module, "fast", 3, 1) {
        assert_eq!(bits(&obs), bits(&reference(&model, 2, false)));
    }
}

#[test]
fn compressed_models_are_rejected() {
    // v[i] v[i+1] + v[i+2] / (v[i+3]^2 + 1): a shape with four variables that occurs more than
    // twice becomes a funclet, which the WebAssembly backend does not support
    let d = dir("compress");
    let v: Vec<Expr> = (0..8).map(|i| var(&format!("v{}", i))).collect();
    let one = Expr::from(1.0);
    let obs: Vec<Expr> = (0..8)
        .map(|i| {
            let w = &v[(i + 3) % 8];
            &(&v[i] * &v[(i + 1) % 8]) + &(&v[(i + 2) % 8] / &(&(w * w) + &one))
        })
        .collect();
    let model = (v, obs);
    for compress in [false, true] {
        let mut config = config(CompilerType::Native, 1, true);
        config.set_compress(compress);
        let name = format!("model{}.sjb", compress as u8);
        save(&compile(&model, config), &d.join(&name));
        let out = sjb2wasm(&d, &[&name]);
        if compress {
            fails(&out, 1, "does not support compression mode");
        } else {
            ok(&out);
        }
    }
}
