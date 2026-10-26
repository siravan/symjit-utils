//! The utilities together: `#[kernel]` functions of a Rust file are saved by rust2sjb, and the
//! saved applications are compiled by sjb2obj (linked into a C program with `cc` and run) and
//! sjb2wasm (run in Node); all compute what symjit computes running the saved application.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use common::{have, parse_hex, run_wasm};
use symjit::{Application, CompilerType, Complex, Config};

const KERNELS: &str = "\
use symjit::Complex;

/// Newton's iteration for the square root: a kernel loop
#[kernel]
pub fn newton(a: f64) -> (f64, f64) {
    let mut x = a;
    let mut i = 0.0;
    while i < 25.0 {
        x = 0.5 * (x + a / x);
        i += 1.0;
    }
    (x, x * x - a)
}

/// The Mandelbrot iteration a fixed number of times: a complex kernel loop
#[kernel]
pub fn orbit(c: Complex<f64>, steps: f64) -> (Complex<f64>, f64) {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < steps {
        z = z * z + c;
        n += 1.0;
    }
    (z, z.re * z.re + z.im * z.im)
}

/// The escape time of the Mandelbrot iteration: a complex kernel loop whose condition depends
/// on the complex values (see `complex_loop_condition`)
#[kernel]
pub fn escape(c: Complex<f64>, limit: f64) -> f64 {
    let mut z = Complex::new(0.0, 0.0);
    let mut n = 0.0;
    while n < limit && z.re * z.re + z.im * z.im < 4.0 {
        z = z * z + c;
        n += 1.0;
    }
    n
}

/// Branches and math functions
#[kernel]
pub fn smooth(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    if r < 1.0 {
        (1.0 - r * r).sqrt() * x.sin()
    } else {
        y / r
    }
}
";

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("pipeline")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn tool(name: &str, d: &Path, args: &[&str]) -> Output {
    let exe = match name {
        "rust2sjb" => env!("CARGO_BIN_EXE_rust2sjb"),
        "sjb2obj" => env!("CARGO_BIN_EXE_sjb2obj"),
        _ => env!("CARGO_BIN_EXE_sjb2wasm"),
    };
    let out = Command::new(exe)
        .args(args)
        .current_dir(d)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{} {:?}: {}",
        name,
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// The points (the inputs of the kernel, in doubles: complex inputs as (re, im) pairs).
fn points(name: &str) -> Vec<Vec<f64>> {
    match name {
        "newton" => vec![vec![2.0], vec![0.25], vec![1e6], vec![3.0]],
        "orbit" => vec![
            vec![-0.5, 0.5, 20.0, 0.0],
            vec![0.3, 0.1, 8.0, 0.0],
            vec![-1.75, 0.0, 30.0, 0.0],
            vec![0.25, -0.5, 1.0, 0.0],
        ],
        "escape" => vec![
            vec![-0.5, 0.5, 100.0, 0.0],
            vec![0.3, 0.6, 100.0, 0.0],
            vec![-1.75, 0.0, 50.0, 0.0],
            vec![1.0, 1.0, 100.0, 0.0],
        ],
        _ => vec![
            vec![0.25, 0.5],
            vec![-0.75, 0.1],
            vec![2.0, -1.5],
            vec![0.0, 0.0],
        ],
    }
}

/// What symjit computes running the saved application (outputs in doubles).
fn reference(sjb: &Path, points: &[Vec<f64>]) -> (Vec<Vec<f64>>, usize, usize) {
    let mut f = std::fs::File::open(sjb).unwrap();
    let app = Application::load(&mut f, &Config::new(CompilerType::Native, 0).unwrap()).unwrap();
    let complex = app.config.is_complex();
    let (n_in, n_out) = (app.count_params, app.count_obs);
    let applet = app.seal().unwrap();
    let out = points
        .iter()
        .map(|p| {
            if complex {
                let args: Vec<Complex<f64>> =
                    p.chunks(2).map(|c| Complex::new(c[0], c[1])).collect();
                let mut out = vec![Complex::default(); n_out / 2];
                applet.evaluate(&args, &mut out);
                out.iter().flat_map(|z| [z.re, z.im]).collect()
            } else {
                let mut out = vec![0.0; n_out];
                applet.evaluate(p, &mut out);
                out
            }
        })
        .collect();
    (out, n_in, n_out)
}

/// The object's kernel at `points` (Symbolica convention: `name(outs, NULL, 0, inputs)`).
fn run_object(d: &Path, name: &str, points: &[Vec<f64>], n_out: usize) -> Vec<Vec<f64>> {
    let rows: Vec<String> = points
        .iter()
        .map(|p| {
            format!(
                "{{{}}}",
                p.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    let c = format!(
        "#include <stdio.h>\n#include \"{name}.h\"\n\
         static const double P[][{}] = {{{}}};\n\
         int main(void) {{\n\
         \x20 for (int k = 0; k < {}; k++) {{\n\
         \x20   double out[{n_out}] = {{0}};\n\
         \x20   if ({name}(out, NULL, 0, P[k]) != 0) return 1;\n\
         \x20   for (int j = 0; j < {n_out}; j++) printf(\"%a \", out[j]);\n\
         \x20   printf(\"\\n\");\n\
         \x20 }}\n\
         \x20 return 0;\n}}\n",
        points[0].len(),
        rows.join(", "),
        points.len()
    );
    std::fs::write(d.join("main.c"), c).unwrap();
    let exe = d.join("main");
    let out = Command::new("cc")
        .args(["-Wall", "-Werror"])
        .arg(d.join("main.c"))
        .arg(d.join(format!("{}.o", name)))
        .args(["-lm", "-o"])
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new(&exe).output().unwrap();
    assert!(out.status.success());
    parse_hex(&out.stdout)
}

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
fn rust_kernels_through_objects_and_modules() {
    let d = dir("kernels");
    std::fs::write(d.join("k.rs"), KERNELS).unwrap();
    let objects = (cfg!(target_os = "linux") || cfg!(target_os = "macos"))
        && (cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64"))
        && have("cc");

    for name in ["newton", "orbit", "smooth"] {
        tool(
            "rust2sjb",
            &d,
            &["k.rs", "--kernel", name, "-fno-fast-math"],
        );
        let sjb = d.join(format!("{}.sjb", name));
        let pts = points(name);
        let (want, n_in, n_out) = reference(&sjb, &pts);
        assert_eq!(n_in, pts[0].len(), "{}", name);

        if objects {
            tool("sjb2obj", &d, &[&format!("{}.sjb", name)]);
            assert_eq!(
                bits(&run_object(&d, name, &pts, n_out)),
                bits(&want),
                "{} (object)",
                name
            );
        }

        tool("sjb2wasm", &d, &[&format!("{}.sjb", name)]);
        if let Some(got) = run_wasm(&d.join(format!("{}.wasm", name)), "params", 0, n_out, &pts) {
            if name == "smooth" {
                // JavaScript's Math.sin is not the C library's
                for (g, w) in got.iter().zip(&want) {
                    for (a, b) in g.iter().zip(w) {
                        assert!(
                            (a - b).abs() <= 1e-15 * b.abs().max(1.0),
                            "{name}: {g:?} {w:?}"
                        );
                    }
                }
            } else {
                assert_eq!(bits(&got), bits(&want), "{} (wasm)", name);
            }
        }
    }
}

/// A complex kernel loop whose condition depends on the complex values (an escape-time loop)
/// computes the same in the packed-complex native kernel, in the kernel symjit lowers with
/// `Complexifier` (`fast_complex=false`) and in the WebAssembly module (also lowered). Symjit
/// 2.27's complexifier negated the (zero) imaginary part of a mask kept in a complex slot, which
/// the test of the loop then found non-zero (fixed in `stable`, `tests/complex_mask_not.rs`).
#[test]
fn complex_loop_condition() {
    let d = dir("escape");
    std::fs::write(d.join("k.rs"), KERNELS).unwrap();
    let pts = points("escape");
    tool(
        "rust2sjb",
        &d,
        &["k.rs", "--kernel", "escape", "-o", "fast.sjb"],
    );
    let (want, _, n_out) = reference(&d.join("fast.sjb"), &pts);
    assert_eq!(want[1], vec![15.0, 0.0]);
    tool(
        "rust2sjb",
        &d,
        &[
            "k.rs",
            "--kernel",
            "escape",
            "--set",
            "fast_complex=false",
            "-o",
            "lowered.sjb",
        ],
    );
    assert_eq!(
        reference(&d.join("lowered.sjb"), &pts).0,
        want,
        "fast_complex=false"
    );
    tool("sjb2wasm", &d, &["fast.sjb"]);
    if let Some(got) = run_wasm(&d.join("fast.wasm"), "params", 0, n_out, &pts) {
        assert_eq!(got, want, "wasm");
    }
}
