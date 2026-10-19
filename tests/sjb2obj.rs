//! sjb2obj end to end: models compiled with symjit's Rust API are saved (`.sjb`), compiled by
//! the binary into objects and C headers, linked into C programs with `cc ... -lm` and run; their
//! outputs are compared with the JIT bit for bit (the objects' kernels come from copies of the
//! JIT's generators). AArch64 objects are linked with `aarch64-linux-gnu-gcc -static` and run
//! under `qemu-aarch64`. Steps whose tools are missing are skipped.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use symjit::{var, Application, Compiler, CompilerType, Config, Expr, Storage};

const SJB2OBJ: &str = env!("CARGO_BIN_EXE_sjb2obj");

const POINTS: [[f64; 3]; 5] = [
    [0.5, 2.0, 0.25],
    [-1.25, 0.75, 1.5],
    [1.5, -0.5, 0.125],
    [0.0, 3.0, -0.75],
    [100.0, 1e-300, 7.0],
];

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sjb2obj")
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

fn host_ok() -> bool {
    (cfg!(target_os = "linux") || cfg!(target_os = "macos"))
        && (cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"))
        && have("cc")
}

fn config(ty: CompilerType, opt_level: u8, fastmath: bool) -> Config {
    let mut config = Config::new(ty, 0).unwrap();
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config.set_fastmath(fastmath);
    config
}

type Model = (Vec<Expr>, Vec<Expr>);

/// math functions (`log10` is symjit's `log`, C's `log10`), the ones C lacks (`csc`, `sinc`),
/// and arithmetic
fn functions() -> Model {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&x.sin() * &y) + &(-&(&x * &x)).exp();
    let b = &(&one + &(&x * &x)).ln() - &(&one + &(&y * &y)).log10();
    let c = &(&(&x * &y) - &z) + &(&(&z + &one).csc() * &x.sinc());
    let d = &(&x * &x).log10() + &(&one + &(&z * &z)).pow(&y);
    (vec![x, y, z], vec![a, b, c, d])
}

/// rational functions and square roots only
fn rational() -> Model {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&(&x * &x) - &(&y * &z)) / &(&one + &(&z * &z));
    let b = &(&(&x * &y).abs().sqrt() + &z) * &(&x - &Expr::from(0.75));
    let c = &(&one / &(&(&(&x * &x) + &(&y * &y)) + &one)) - &z;
    (vec![x, y, z], vec![a, b, c])
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

fn sjb2obj(d: &Path, args: &[&str]) -> Output {
    Command::new(SJB2OBJ)
        .args(args)
        .current_dir(d)
        .output()
        .unwrap()
}

fn ok(out: &Output) {
    assert!(
        out.status.success(),
        "{}",
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

/// A C program that evaluates kernel `name` (header `name.h`) at `POINTS`, one line of %a
/// outputs per point, followed by the fast kernel's output if `fast`.
fn driver(d: &Path, name: &str, count_obs: usize, fast: bool) {
    let up = name.to_uppercase();
    let points: Vec<String> = POINTS
        .iter()
        .map(|p| format!("{{{:?}, {:?}, {:?}}}", p[0], p[1], p[2]))
        .collect();
    let fast = if fast {
        format!("printf(\"%a \", {name}_fast(P[k][0], P[k][1], P[k][2]));")
    } else {
        String::new()
    };
    let c = format!(
        "#include <stdio.h>\n#include \"{name}.h\"\n\
         static const double P[][3] = {{{}}};\n\
         int main(void) {{\n\
         \x20 for (int k = 0; k < {}; k++) {{\n\
         \x20   double mem[{up}_MEM_SIZE] = {{0}};\n\
         \x20   for (int i = 0; i < 3; i++) mem[i] = P[k][i];\n\
         \x20   if ({name}(mem, NULL, 0, NULL) != 0) return 1;\n\
         \x20   for (int j = 0; j < {count_obs}; j++) printf(\"%a \", mem[{up}_COUNT_STATES + j]);\n\
         \x20   {fast}\n\
         \x20   printf(\"\\n\");\n\
         \x20 }}\n\
         \x20 return 0;\n}}\n",
        points.join(", "),
        POINTS.len()
    );
    std::fs::write(d.join(format!("{}.c", name)), c).unwrap();
}

/// Links `name.c` with `name.o` (with `cc`, or for AArch64 statically with the cross compiler)
/// and runs it (under qemu for AArch64); the outputs at `POINTS`.
fn run(d: &Path, name: &str, arm: bool) -> Vec<Vec<f64>> {
    let exe = d.join(name);
    let mut cc = if arm {
        let mut c = Command::new("aarch64-linux-gnu-gcc");
        c.arg("-static");
        c
    } else {
        Command::new("cc")
    };
    let out = cc
        .args(["-Wall", "-Werror"])
        .arg(d.join(format!("{}.c", name)))
        .arg(d.join(format!("{}.o", name)))
        .args(["-lm", "-o"])
        .arg(&exe)
        .output()
        .unwrap();
    ok(&out);
    let out = if arm {
        Command::new("qemu-aarch64").arg(&exe).output().unwrap()
    } else {
        Command::new(&exe).output().unwrap()
    };
    ok(&out);
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| {
            l.split_whitespace()
                .map(|v| hexf(v).unwrap_or_else(|| panic!("bad output `{}`", v)))
                .collect()
        })
        .collect()
}

/// Parses C's %a output (e.g. 0x1.8p+1, -0x0p+0, inf, nan).
fn hexf(s: &str) -> Option<f64> {
    match s {
        "inf" => return Some(f64::INFINITY),
        "-inf" => return Some(f64::NEG_INFINITY),
        "nan" | "-nan" => return Some(f64::NAN),
        _ => {}
    }
    let (neg, s) = match s.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, s),
    };
    let s = s.strip_prefix("0x")?;
    let (mant, exp) = s.split_once('p')?;
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut v = u64::from_str_radix(int, 16).ok()? as f64;
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        v += c.to_digit(16)? as f64 * scale;
        scale /= 16.0;
    }
    let v = v * 2f64.powi(exp.parse().ok()?);
    Some(if neg { -v } else { v })
}

// the bits of the outputs, with every NaN as the same one
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
    let out = sjb2obj(&d, &["--help"]);
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("Usage: sjb2obj"));
    let out = sjb2obj(&d, &["--version"]);
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("sjb2obj "));

    fails(&sjb2obj(&d, &[]), 2, "no input file");
    fails(
        &sjb2obj(&d, &["--emit", "a.sjb"]),
        2,
        "unknown option `--emit`",
    );
    fails(
        &sjb2obj(&d, &["a.sjb", "--name"]),
        2,
        "`--name` needs a value",
    );
    fails(
        &sjb2obj(&d, &["a.sjb", "--no-header", "--header", "a.h"]),
        2,
        "exclude each other",
    );
    fails(
        &sjb2obj(&d, &["missing.sjb"]),
        1,
        "cannot open `missing.sjb`",
    );
    std::fs::write(d.join("junk.sjb"), b"not a saved application").unwrap();
    fails(&sjb2obj(&d, &["junk.sjb"]), 1, "cannot read `junk.sjb`");
}

#[test]
fn output_names_and_formats() {
    let d = dir("names");
    let app = compile(&rational(), config(CompilerType::Native, 1, true));
    save(&app, &d.join("model.sjb"));

    // model.o and model.h, kernel `model`
    ok(&sjb2obj(&d, &["model.sjb"]));
    let header = std::fs::read_to_string(d.join("model.h")).unwrap();
    assert!(header.contains("int model(double *mem"), "{}", header);
    assert!(
        header.contains("#define MODEL_COUNT_STATES 3"),
        "{}",
        header
    );

    // -o, --name, --header
    ok(&sjb2obj(
        &d,
        &[
            "model.sjb",
            "-o",
            "k.o",
            "--name",
            "kern",
            "--header",
            "kh.h",
        ],
    ));
    assert!(std::fs::read_to_string(d.join("kh.h"))
        .unwrap()
        .contains("int kern(double *mem"));
    let _ = std::fs::remove_file(d.join("n.h"));
    ok(&sjb2obj(&d, &["model.sjb", "-o", "n.o", "--no-header"]));
    assert!(d.join("n.o").exists() && !d.join("n.h").exists());

    // formats
    ok(&sjb2obj(
        &d,
        &["model.sjb", "-o", "elf.o", "--format", "elf"],
    ));
    assert_eq!(&std::fs::read(d.join("elf.o")).unwrap()[..4], b"\x7fELF");
    ok(&sjb2obj(
        &d,
        &["model.sjb", "-o", "mac.o", "--format", "macho"],
    ));
    assert_eq!(
        &std::fs::read(d.join("mac.o")).unwrap()[..4],
        &[0xcf, 0xfa, 0xed, 0xfe]
    );
    fails(
        &sjb2obj(&d, &["model.sjb", "--format", "coff"]),
        1,
        "unknown object format `coff`",
    );

    // the kernel is named after the file: it must be a C identifier
    save(&app, &d.join("my-model.sjb"));
    fails(&sjb2obj(&d, &["my-model.sjb"]), 1, "not a C identifier");
    ok(&sjb2obj(&d, &["my-model.sjb", "--name", "my_model"]));
}

#[test]
fn kernels_match_the_jit() {
    if !host_ok() {
        return;
    }
    let d = dir("jit");
    for (label, model, fastmath) in [
        ("functions", functions(), true),
        ("rational", rational(), false),
    ] {
        for opt_level in [0, 1, 2, 3] {
            let name = format!("{}{}", label, opt_level);
            save(
                &compile(&model, config(CompilerType::Native, opt_level, fastmath)),
                &d.join(format!("{}.sjb", name)),
            );
            ok(&sjb2obj(&d, &[&format!("{}.sjb", name)]));
            driver(&d, &name, model.1.len(), false);
            assert_eq!(
                bits(&run(&d, &name, false)),
                bits(&reference(&model, opt_level, fastmath)),
                "{}",
                name
            );
        }
    }
}

#[test]
fn logarithms() {
    // symjit's `ln` is C's `log`, its `log` (`Expr::log10`) C's `log10`
    if !host_ok() {
        return;
    }
    let d = dir("log");
    let x = var("x");
    let model = (vec![x.clone(), var("y"), var("z")], vec![x.ln(), x.log10()]);
    save(
        &compile(&model, config(CompilerType::Native, 1, true)),
        &d.join("logs.sjb"),
    );
    ok(&sjb2obj(&d, &["logs.sjb"]));
    driver(&d, "logs", 2, false);
    let out = run(&d, "logs", false);
    assert_eq!(out[4], vec![100f64.ln(), 2.0]);
}

#[test]
fn fast_kernel() {
    // one observable and no parameters: `<name>_fast(x, y, z)`
    if !host_ok() {
        return;
    }
    let d = dir("fast");
    let (states, obs) = functions();
    let model = (states, vec![obs[0].clone()]);
    save(
        &compile(&model, config(CompilerType::Native, 2, true)),
        &d.join("one.sjb"),
    );
    ok(&sjb2obj(&d, &["one.sjb"]));
    assert!(std::fs::read_to_string(d.join("one.h"))
        .unwrap()
        .contains("double one_fast(double x0, double x1, double x2)"));
    driver(&d, "one", 1, true);
    let want = reference(&model, 2, true);
    for (row, w) in run(&d, "one", false).iter().zip(&want) {
        assert_eq!(
            bits(&[vec![row[0]], vec![row[1]]]),
            bits(&[w.clone(), w.clone()])
        );
    }
}

#[test]
fn arm64_objects() {
    // AArch64 objects of models compiled with `ty = "arm"` on any host; run under qemu
    let d = dir("arm");
    for opt_level in [0, 2, 3] {
        let name = format!("arm{}", opt_level);
        save(
            &compile(&rational(), config(CompilerType::Arm, opt_level, false)),
            &d.join(format!("{}.sjb", name)),
        );
        ok(&sjb2obj(&d, &[&format!("{}.sjb", name), "--format", "elf"]));
        let obj = std::fs::read(d.join(format!("{}.o", name))).unwrap();
        assert_eq!(&obj[..4], b"\x7fELF");
        assert_eq!(u16::from_le_bytes([obj[18], obj[19]]), 183); // EM_AARCH64
        assert!(std::fs::read_to_string(d.join(format!("{}.h", name)))
            .unwrap()
            .contains("AArch64 ELF"));

        if have("qemu-aarch64") && have("aarch64-linux-gnu-gcc") {
            driver(&d, &name, 3, false);
            // without fused operations, ARM64 computes x86-64's bits
            assert_eq!(
                bits(&run(&d, &name, true)),
                bits(&reference(&rational(), opt_level, false)),
                "{}",
                name
            );
        }
    }
}

#[test]
fn other_targets_are_rejected() {
    let d = dir("targets");
    for (name, ty) in [
        ("riscv", CompilerType::RiscV),
        ("bytecode", CompilerType::ByteCode),
    ] {
        save(
            &compile(&rational(), config(ty, 1, true)),
            &d.join(format!("{}.sjb", name)),
        );
        fails(
            &sjb2obj(&d, &[&format!("{}.sjb", name)]),
            1,
            "object files hold x86-64 or AArch64 kernels",
        );
    }
}
