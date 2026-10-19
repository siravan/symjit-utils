//! rust2sjb: the kernels of `kernels/corpus.rs` (sjc's corpus of the `#[kernel]` language) are
//! compiled by the interpreter, saved and loaded again by symjit, and run; they must compute what
//! the same functions compute as plain Rust (`kernels/corpus_plain.rs`, the corpus without the
//! attributes), and what the kernel computes before it is saved, bit for bit. Kernels that call
//! other kernels or callbacks cannot be saved. Then the command line and the error messages.

#[allow(clippy::all, dead_code, unused)]
mod corpus {
    include!("kernels/corpus_plain.rs");
}

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::Result;
use symjit::{Application, CompilerType, Complex, Config, Storage};
use symjit_utils::source::{builtin_config, Source};

const CORPUS: &str = include_str!("kernels/corpus.rs");
const RUST2SJB: &str = env!("CARGO_BIN_EXE_rust2sjb");

/// The kernels that call other kernels or callbacks, which a saved application cannot hold.
const CALLERS: [&str; 10] = [
    "iterate",
    "orbit",
    "around_circle",
    "midpoint_integral",
    "ramped",
    "norm2",
    "fib_path",
    "spiral",
    "callbacks_in_loop",
    "callback_in_complex",
];

// arguments and results of the plain functions, as flat lists of complex numbers (the
// parameters and outputs of the kernel, in order; real ones in complex kernels take and give
// the real part)

trait Arg: Sized {
    fn take(it: &mut dyn Iterator<Item = Complex<f64>>) -> Self;
}

impl Arg for f64 {
    fn take(it: &mut dyn Iterator<Item = Complex<f64>>) -> Self {
        it.next().unwrap().re
    }
}

impl Arg for Complex<f64> {
    fn take(it: &mut dyn Iterator<Item = Complex<f64>>) -> Self {
        it.next().unwrap()
    }
}

impl<const N: usize> Arg for [f64; N] {
    fn take(it: &mut dyn Iterator<Item = Complex<f64>>) -> Self {
        std::array::from_fn(|_| f64::take(it))
    }
}

impl<A: Arg, B: Arg> Arg for (A, B) {
    fn take(it: &mut dyn Iterator<Item = Complex<f64>>) -> Self {
        let a = A::take(it);
        (a, B::take(it))
    }
}

trait Out {
    fn put(&self, out: &mut Vec<Complex<f64>>);
}

impl Out for f64 {
    fn put(&self, out: &mut Vec<Complex<f64>>) {
        out.push(Complex::new(*self, 0.0));
    }
}

impl Out for bool {
    // a `bool` output of a kernel is its mask: all bits set (a NaN) for true, zero for false
    fn put(&self, out: &mut Vec<Complex<f64>>) {
        let bits = if *self { u64::MAX } else { 0 };
        out.push(Complex::new(f64::from_bits(bits), 0.0));
    }
}

impl Out for Complex<f64> {
    fn put(&self, out: &mut Vec<Complex<f64>>) {
        out.push(*self);
    }
}

impl<T: Out, const N: usize> Out for [T; N] {
    fn put(&self, out: &mut Vec<Complex<f64>>) {
        for v in self.iter() {
            v.put(out);
        }
    }
}

macro_rules! out_tuple {
    ($($t:ident $i:tt),*) => {
        impl<$($t: Out),*> Out for ($($t,)*) {
            fn put(&self, out: &mut Vec<Complex<f64>>) {
                $(self.$i.put(out);)*
            }
        }
    };
}

out_tuple!(A 0, B 1);
out_tuple!(A 0, B 1, C 2);
out_tuple!(A 0, B 1, C 2, D 3);
out_tuple!(A 0, B 1, C 2, D 3, E 4);

type Reference = fn(&[Complex<f64>]) -> Vec<Complex<f64>>;

macro_rules! cases {
    ($($name:ident ( $($t:ty),* )),* $(,)?) => {
        vec![$((
            stringify!($name),
            (|args: &[Complex<f64>]| {
                let mut it = args.iter().copied();
                let r = corpus::$name($(<$t as Arg>::take(&mut it)),*);
                let mut out = Vec::new();
                r.put(&mut out);
                out
            }) as Reference,
        )),*]
    };
}

type C = Complex<f64>;

/// The kernels of the corpus that a saved application can hold, with their plain functions.
fn kernels() -> Vec<(&'static str, Reference)> {
    cases!(
        arith(f64, f64, f64),
        elementary(f64, f64),
        piecewise(f64, f64),
        tree(f64, f64),
        complex_ops(C, C),
        area(C, C, C, C),
        fib(f64),
        several(f64, f64),
        rotate(C, f64),
        count_inside(C, C, C),
        newton_sqrt(f64),
        mandelbrot(C),
        mandelbrot_untyped(C),
        fib_iter(f64),
        collatz(f64),
        escape(C, f64),
        midpoint_rule(f64, f64, f64),
        masks(f64, f64),
        unrolled(f64, [f64; 4]),
        mixed(C, f64),
        bisect(f64, f64, f64),
        shapes(f64, (f64, f64)),
        real_api(f64, f64),
        orthant(C, C, C, C, f64),
        step(f64, f64),
        complex_step(C, C),
        integrand(f64),
        pole(C, C),
        ramp(f64),
        square(f64),
    )
}

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self) -> f64 {
        4.0 * self.next() - 2.0
    }
}

/// Arguments for which the kernel's loops end (as sjc's tests choose them).
fn arguments(name: &str, n: usize, rng: &mut Rng) -> Vec<Complex<f64>> {
    let mut re: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    let im: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    match name {
        "newton_sqrt" => re[0] = re[0].abs() + 0.1,
        "collatz" => re[0] = (re[0].abs() * 10.0).floor() + 1.0,
        "fib" => re[0] = (re[0].abs() * 2.0).floor(),
        "tree" => re[1] = re[1].abs(),
        _ => {}
    }
    let mut args: Vec<Complex<f64>> = re
        .iter()
        .zip(&im)
        .map(|(&r, &i)| Complex::new(r, i))
        .collect();
    if name == "orthant" {
        // the unit cube, down to boxes of side 1/4 .. 1/2
        let side = 0.25 + 0.25 * rng.next();
        args = vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(1.0, 1.0),
            Complex::new(side, 0.0),
        ];
    }
    args
}

fn configs() -> Vec<(String, Config)> {
    let mut configs = Vec::new();
    for opt in 0..=3 {
        for fastmath in [false, true] {
            let mut config = builtin_config(CompilerType::Native).unwrap();
            config.set_opt_level(opt);
            config.set_fastmath(fastmath);
            configs.push((format!("opt level {opt}, fastmath {fastmath}"), config));
        }
    }
    configs
}

/// The kernel's outputs at `args`, as complex numbers.
fn evaluate(app: Application, args: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let complex = app.config.is_complex();
    let n_out = if complex {
        app.count_obs / 2
    } else {
        app.count_obs
    };
    let applet = app.seal().unwrap();
    if complex {
        let mut out = vec![Complex::default(); n_out];
        applet.evaluate(args, &mut out);
        out
    } else {
        let args: Vec<f64> = args.iter().map(|z| z.re).collect();
        let mut out = vec![0.0; n_out];
        applet.evaluate(&args, &mut out);
        out.iter().map(|&x| Complex::new(x, 0.0)).collect()
    }
}

/// The application saved and loaded again, as `rust2sjb` writes it and symjit reads it.
fn saved_and_loaded(app: &Application) -> Application {
    let mut bytes = Vec::new();
    app.save(&mut bytes).unwrap();
    Application::load(
        &mut bytes.as_slice(),
        &Config::new(CompilerType::Native, 0).unwrap(),
    )
    .unwrap()
}

fn same(a: &[Complex<f64>], b: &[Complex<f64>]) -> bool {
    let eq = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| eq(x.re, y.re) && eq(x.im, y.im))
}

fn close(a: &[Complex<f64>], b: &[Complex<f64>], rtol: f64) -> bool {
    let near = |x: f64, y: f64| {
        (x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= rtol * y.abs().max(1.0)
    };
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| near(x.re, y.re) && near(x.im, y.im))
}

#[test]
fn the_plain_corpus_is_the_corpus_without_attributes() {
    let plain = include_str!("kernels/corpus_plain.rs");
    let stripped: Vec<&str> = CORPUS
        .lines()
        .filter(|l| {
            let t = l.trim();
            !(t.starts_with("#[kernel") && t.ends_with(']')) && t != "use symjit::kernel::kernel;"
        })
        .collect();
    // after the two lines of comment and an empty line at its top
    let body: Vec<&str> = plain.lines().skip(3).collect();
    assert!(
        body == stripped,
        "tests/kernels/corpus_plain.rs is out of date: regenerate it from corpus.rs (remove the \
         `#[kernel...]` lines and `use symjit::kernel::kernel;`)"
    );
}

#[test]
fn saved_kernels_compute_the_plain_functions() -> Result<()> {
    let src = Source::parse("corpus.rs", CORPUS)?;
    let kernels = kernels();

    // every kernel of the file is checked or is a caller
    let mut listed: Vec<&str> = kernels.iter().map(|(n, _)| *n).chain(CALLERS).collect();
    let mut found = src.kernels();
    listed.sort();
    found.sort();
    assert_eq!(listed, found);

    for (name, reference) in kernels {
        for (label, config) in configs() {
            let app = src.compile(name, config.clone())?;
            if name == "area" {
                // random rectangles make it recurse millions of times; compiled only
                saved_and_loaded(&app);
                continue;
            }
            let width = if app.config.is_complex() { 2 } else { 1 };
            let n_in = app.count_params / width;
            let mut rng = Rng(0x5eed ^ name.len() as u64);
            for _ in 0..20 {
                let args = arguments(name, n_in, &mut rng);
                let loaded = evaluate(saved_and_loaded(&app), &args);
                let direct = evaluate(src.compile(name, config.clone())?, &args);
                assert!(
                    same(&loaded, &direct),
                    "{name} ({label}) at {args:?}: saved {loaded:?}, compiled {direct:?}"
                );
                if !config.fastmath() {
                    // without fused multiply-adds, within rounding of Rust (whose powi and
                    // complex functions are not symjit's)
                    let want = reference(&args);
                    assert!(
                        close(&loaded, &want, 1e-12),
                        "{name} ({label}) at {args:?}: kernel {loaded:?}, Rust {want:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn kernels_that_call_kernels_or_callbacks_are_rejected() -> Result<()> {
    let src = Source::parse("corpus.rs", CORPUS)?;
    for name in CALLERS {
        let config = builtin_config(CompilerType::Native)?;
        match src.compile(name, config) {
            Ok(_) => panic!("`{name}` compiled"),
            Err(e) => {
                let e = format!("{e:#}");
                assert!(e.contains("a saved application cannot call"), "{name}: {e}");
            }
        }
    }
    Ok(())
}

/// The kernel `name` of `text`, or the error.
fn compile(text: &str, name: &str) -> Result<Application> {
    Source::parse("t.rs", text)?.compile(name, builtin_config(CompilerType::Native)?)
}

fn error(text: &str, name: &str) -> String {
    match compile(text, name) {
        Ok(_) => panic!("`{name}` compiled"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn errors_name_the_construct_and_its_position() {
    let e = error("#[kernel] fn f(x: f64) -> f64 { return x; }", "f");
    assert!(
        e.contains("t.rs:1:") && e.contains("cannot `return`"),
        "{e}"
    );

    let e = error(
        "#[kernel(callback)] fn cb(x: f64) -> f64 { x }\n#[kernel] fn f(x: f64) -> f64 { cb(x) }",
        "f",
    );
    assert!(e.contains("cannot call the Rust callback `cb`"), "{e}");

    let e = error(
        "#[kernel] fn g(x: f64) -> f64 { x }\n#[kernel] fn f(x: f64) -> f64 { g(x) }",
        "f",
    );
    assert!(e.contains("cannot call the kernel `g`"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { if x > 0.0 { x; } x }", "f");
    assert!(e.contains("without `else`"), "{e}");

    let e = error(
        "#[kernel] fn f(x: f64) -> f64 { let mut k = 0; while x > 1.0 { k += 1; } x }",
        "f",
    );
    assert!(e.contains("not a kernel value"), "{e}");

    let e = error(
        "#[kernel] fn f(x: f64) -> f64 { println!(\"{}\", 1); x }",
        "f",
    );
    assert!(e.contains("macros"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x % 2.0 }", "f");
    assert!(e.contains("no `%`"), "{e}");

    let e = error("#[kernel] fn f(x: i32) -> f64 { 1.0 }", "f");
    assert!(e.contains("parameters of a kernel"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { loop {} }", "f");
    assert!(e.contains("did not end"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x.mul_add(2.0, 1.0) }", "f");
    assert!(e.contains("no method `mul_add`"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x.pow2i() }", "f");
    assert!(e.contains("no method `pow2i`"), "{e}");

    let e = error(
        "#[kernel(inline)] fn h(x: f64) -> f64 { h(x) }\n#[kernel] fn f(x: f64) -> f64 { h(x) }",
        "f",
    );
    assert!(e.contains("cannot call itself"), "{e}");

    let e = error("#[kernel] fn f(x: f64) -> f64 { x }", "nope");
    assert!(
        e.contains("no `#[kernel]` named `nope`") && e.contains("the kernels are: f"),
        "{e}"
    );

    assert!(Source::parse("t.rs", "fn (").is_err());
}

#[test]
fn if_without_else_that_assigns_is_a_kernel_branch() -> Result<()> {
    // `if c { y = .. }` on a kernel value: the else side keeps `y`
    let app = compile(
        "#[kernel] fn f(x: f64) -> f64 { let mut y = 0.5; if x > 0.0 { y = x * 2.0; } y }",
        "f",
    )?
    .seal()?;
    assert_eq!(app.evaluate_single(&[3.0]), 6.0);
    assert_eq!(app.evaluate_single(&[-3.0]), 0.5);
    Ok(())
}

// the command line

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("rust2sjb")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rust2sjb(d: &Path, args: &[&str]) -> Output {
    Command::new(RUST2SJB)
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

fn load(path: &Path) -> Application {
    let mut f = std::fs::File::open(path).unwrap();
    Application::load(&mut f, &Config::new(CompilerType::Native, 0).unwrap()).unwrap()
}

const KERNELS: &str = "\
#[kernel]
fn hypot2(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

#[kernel]
fn halve(x: f64) -> f64 {
    let mut y = x;
    while y > 1.0 {
        y = y / 2.0;
    }
    y
}
";

#[test]
fn usage() {
    let d = dir("usage");
    let out = rust2sjb(&d, &["--help"]);
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("Usage: rust2sjb"));
    let out = rust2sjb(&d, &["--version"]);
    ok(&out);
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("rust2sjb "));

    fails(&rust2sjb(&d, &[]), 2, "no input file");
    fails(&rust2sjb(&d, &["k.rs", "-O4"]), 2, "unknown option `-O4`");
    fails(&rust2sjb(&d, &["k.rs", "--set", "cse"]), 2, "KEY=VALUE");
    fails(
        &rust2sjb(&d, &["missing.rs"]),
        1,
        "cannot read `missing.rs`",
    );

    std::fs::write(d.join("k.rs"), KERNELS).unwrap();
    fails(
        &rust2sjb(&d, &["k.rs"]),
        1,
        "several kernels (hypot2, halve)",
    );
    fails(
        &rust2sjb(&d, &["k.rs", "--kernel", "nope"]),
        1,
        "no `#[kernel]` named `nope`",
    );
    fails(
        &rust2sjb(&d, &["k.rs", "--kernel", "halve", "--target", "sparc"]),
        1,
        "unknown target `sparc`",
    );
    fails(
        &rust2sjb(&d, &["k.rs", "--kernel", "halve", "--set", "nonsense=1"]),
        1,
        "--set nonsense=1",
    );
}

#[test]
fn saved_kernels_and_options() {
    let d = dir("options");
    std::fs::write(d.join("k.rs"), KERNELS).unwrap();

    // <kernel>.sjb by default
    ok(&rust2sjb(&d, &["k.rs", "--kernel", "hypot2"]));
    let app = load(&d.join("hypot2.sjb"));
    assert_eq!(app.config.opt_level(), 2);
    assert!(app.config.fastmath());
    assert_eq!(app.seal().unwrap().evaluate_single(&[3.0, 4.0]), 5.0);
    ok(&rust2sjb(
        &d,
        &["k.rs", "--kernel", "hypot2", "-O3", "-o", "h3.sjb"],
    ));
    assert_eq!(load(&d.join("h3.sjb")).config.opt_level(), 3);

    // options: output, fastmath, target, further options; a kernel with a loop is compiled by
    // the direct translator, which always uses opt level 2
    ok(&rust2sjb(
        &d,
        &[
            "k.rs",
            "--kernel",
            "halve",
            "-o",
            "h.sjb",
            "-O3",
            "-fno-fast-math",
            "--set",
            "stable_math=true",
        ],
    ));
    let app = load(&d.join("h.sjb"));
    assert_eq!(app.config.opt_level(), 2);
    assert!(!app.config.fastmath());
    assert!(app.config.stable_math());
    assert_eq!(app.seal().unwrap().evaluate_single(&[10.0]), 0.625);

    ok(&rust2sjb(
        &d,
        &[
            "k.rs", "--kernel", "halve", "-o", "arm.sjb", "--target", "aarch64",
        ],
    ));
    assert!(load(&d.join("arm.sjb")).config.is_arm64());
    ok(&rust2sjb(
        &d,
        &[
            "k.rs", "--kernel", "halve", "-o", "bc.sjb", "--target", "bytecode",
        ],
    ));
    assert!(load(&d.join("bc.sjb")).config.is_bytecode());

    // a single kernel needs no --kernel
    std::fs::write(
        d.join("one.rs"),
        "#[kernel]\nfn twice(x: f64) -> f64 { 2.0 * x }\n",
    )
    .unwrap();
    ok(&rust2sjb(&d, &["one.rs"]));
    assert_eq!(
        load(&d.join("twice.sjb"))
            .seal()
            .unwrap()
            .evaluate_single(&[1.5]),
        3.0
    );
}
