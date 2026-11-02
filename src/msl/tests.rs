// The Metal kernels of models compiled with symjit's Rust API: the source, the errors, and,
// on a Mac with a Metal device, the GPU's results against the JIT's (within f32 rounding).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use symjit::{var, Application, Compiler, CompilerType, Config, Expr, Storage};

use super::*;
use crate::sjb::SavedModel;

fn config(opt_level: u8, fastmath: bool) -> Config {
    let mut config = Config::new(CompilerType::Native, 0).unwrap();
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config.set_fastmath(fastmath);
    config
}

fn saved(app: &Application) -> SavedModel {
    let mut bytes: Vec<u8> = Vec::new();
    app.save(&mut bytes).unwrap();
    SavedModel::read(&mut &bytes[..]).unwrap()
}

/// arithmetic, comparisons, a select, sqrt, abs and roundings of x, y, z (as the PTX tests)
fn arithmetic(config: Config) -> Application {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&(&x * &y) + &z) / &(&one + &(&z * &z));
    let b = &(&(&x * &x) + &(&y * &y)).sqrt() - &(&x - &Expr::from(0.75)).abs();
    let c = x.lt(&y).ifelse(&(&x * &z), &(&y - &z));
    let d = &(&x.floor() + &y.ceil()) + &(&z.round() * &x.trunc());
    let e = &x.max(&y) - &z.min(&x);
    Compiler::with_config(config)
        .compile(&[x, y, z], &[a, b, c, d, e])
        .unwrap()
}

#[test]
fn straight_line_source() {
    let app = arithmetic(config(2, false));
    let m = saved(&app).msl(&MslOptions::new("model")).unwrap();
    assert_eq!((m.shape.count_in, m.shape.count_out), (3, 5));
    let src = m.source();
    assert!(src.contains("kernel void model(device const float *in [[buffer(0)]]"));
    assert!(src.contains("sqrt("), "{}", src);
    // no fused operations without fastmath: nothing the Metal compiler could contract either
    assert!(!src.contains("fma("), "{}", src);
    // outputs are written where saved, inputs read where used
    assert!(src.contains("pout["), "{}", src);
    assert!(src.contains("pin["), "{}", src);

    let fused = saved(&arithmetic(config(2, true)))
        .msl(&MslOptions::new("model"))
        .unwrap();
    assert!(fused.source().contains("fma("));
}

#[test]
fn unsupported_operations_are_errors() {
    let x = var("x");
    let app = Compiler::with_config(config(2, false))
        .compile(std::slice::from_ref(&x), &[x.sin()])
        .unwrap();
    let err = saved(&app).msl(&MslOptions::new("model")).err().unwrap();
    assert!(
        err.to_string().contains("math functions (`sin`)"),
        "{}",
        err
    );
    let err = saved(&app).msl(&MslOptions::new("__sj_x")).err().unwrap();
    assert!(err.to_string().contains("reserved"), "{}", err);
}

/// `run.m` built once into the test directory; None (skipped) without a Metal device or cc.
fn runner() -> Option<&'static Path> {
    static RUN: OnceLock<Option<PathBuf>> = OnceLock::new();
    RUN.get_or_init(|| {
        if !cfg!(target_os = "macos") {
            eprintln!("skipped: Metal needs macOS");
            return None;
        }
        let dir = std::env::temp_dir().join(format!("sjb2msl-tests-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("run.m");
        std::fs::write(&src, RUN_M).unwrap();
        let exe = dir.join("run");
        let out = Command::new("cc")
            .args(["-O2", "-fobjc-arc"])
            .arg(&src)
            .args(["-framework", "Metal", "-framework", "Foundation", "-o"])
            .arg(&exe)
            .output();
        match out {
            Ok(o) if o.status.success() => Some(exe),
            Ok(o) => panic!("run.m: {}", String::from_utf8_lossy(&o.stderr)),
            Err(e) => {
                eprintln!("skipped: no cc ({})", e);
                None
            }
        }
    })
    .as_deref()
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn write_floats(path: &Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

/// Runs `cmd` (a `run` command line); None if the machine has no Metal device (exit 77).
fn run(run: &Path, args: &[String]) -> Option<String> {
    let out = Command::new(run).args(args).output().unwrap();
    if out.status.code() == Some(77) {
        eprintln!("skipped: no Metal device");
        return None;
    }
    assert!(
        out.status.success(),
        "run {:?}: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `got` (f32 on the GPU) is `want` (f64 on the CPU) to f32 rounding: within `tol` of it,
/// relative to max(|want|, 1); NaNs where `want` is NaN.
fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len());
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{} [{}]: {} instead of NaN", what, k, g);
            continue;
        }
        let err = (g as f64 - w).abs() / w.abs().max(1.0);
        assert!(
            err <= tol,
            "{} [{}]: {} instead of {} ({:.1e})",
            what,
            k,
            g,
            w,
            err
        );
    }
}

#[test]
fn gpu_matches_the_jit_to_f32_rounding() {
    let Some(exe) = runner() else {
        return;
    };
    let dir = exe.parent().unwrap();
    let n = 1000;
    // points away from the roundings' jumps, where an f32 input could round to the other side
    let pts: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = i as f64 + 0.5;
            [
                (t * 0.37).sin() * 3.3 + 0.13,
                (t * 0.11).cos() * 2.9 - 0.07,
                (t * 0.23).sin() * 1.7 + 0.21,
            ]
        })
        .filter(|p| {
            p.iter()
                .all(|x| (x - x.round()).abs() > 1e-3 && (x.abs() - 0.5).abs() > 1e-3)
        })
        .collect();
    let n = pts.len();
    for fastmath in [false, true] {
        let mut app = arithmetic(config(2, fastmath));
        let want: Vec<Vec<f64>> = pts.iter().map(|p| app.call(p)).collect();
        let m = saved(&app).msl(&MslOptions::new("model")).unwrap();
        let src = dir.join(format!("arith{}.metal", fastmath as u8));
        std::fs::write(&src, m.source()).unwrap();
        // SoA inputs of f32-rounded points
        let input: Vec<f32> = (0..3)
            .flat_map(|k| pts.iter().map(move |p| p[k] as f32))
            .collect();
        let inf = dir.join("in.f32");
        write_floats(&inf, &input);
        let par = dir.join("params.f32");
        write_floats(&par, &[]);
        let outf = dir.join(format!("out{}.f32", fastmath as u8));
        let args: Vec<String> = [
            "plain".into(),
            src.display().to_string(),
            "model".into(),
            inf.display().to_string(),
            "3".into(),
            "5".into(),
            par.display().to_string(),
            n.to_string(),
            outf.display().to_string(),
        ]
        .to_vec();
        if run(exe, &args).is_none() {
            return;
        }
        let got = floats(&outf);
        for j in 0..5 {
            // the JIT on the f32-rounded inputs, so that only the GPU's arithmetic differs
            let want: Vec<f64> = pts
                .iter()
                .map(|p| app.call(&p.map(|x| x as f32 as f64))[j])
                .collect();
            assert_close(
                &got[j * n..(j + 1) * n],
                &want,
                1e-5,
                &format!("output {} (fastmath {})", j, fastmath),
            );
        }
        drop(want);
    }
}

/// A complex plane kernel as pyAmpliCol compiles one, on arena planes.
#[test]
fn plane_kernel_on_an_arena() {
    let Some(exe) = runner() else {
        return;
    };
    let dir = exe.parent().unwrap();
    // w = z0 z1 + 1; planes: z0 re/im 3, 0; z1 re/im 1, 5; w re/im 2, 4
    let bytes = crate::replay::plane_application(crate::replay::tests::PRODUCT_PLUS_ONE);
    let mut model = SavedModel::read(&mut &bytes[..]).unwrap();
    let mut options = MslOptions::new("leaf0");
    options.mapping = Some(vec![3, 0, 1, 5, 2, 4]);
    let m = model.msl(&options).unwrap();
    assert_eq!((m.shape.count_in, m.shape.count_out), (4, 2));
    let src = dir.join("plane.metal");
    std::fs::write(&src, m.source()).unwrap();
    std::fs::write(dir.join("plane.txt"), "leaf0\n").unwrap();
    let (ld, n) = (64usize, 50usize);
    let mut arena = vec![f32::NAN; 6 * ld];
    for i in 0..ld {
        let t = i as f32;
        arena[3 * ld + i] = 0.5 + 0.01 * t;
        arena[i] = -0.25 + 0.02 * t;
        arena[ld + i] = 1.5 - 0.03 * t;
        arena[5 * ld + i] = 0.75 + 0.005 * t;
    }
    write_floats(&dir.join("plane.f32"), &arena);
    let args: Vec<String> = [
        "arena".into(),
        src.display().to_string(),
        dir.join("plane.txt").display().to_string(),
        dir.join("plane.f32").display().to_string(),
        ld.to_string(),
        n.to_string(),
        "2".into(),
        dir.join("plane.out").display().to_string(),
    ]
    .to_vec();
    let Some(stdout) = run(exe, &args) else {
        return;
    };
    assert!(stdout.contains("1 pipelines"), "{}", stdout);
    let got = floats(&dir.join("plane.out"));
    for i in 0..ld {
        let (a, b) = (arena[3 * ld + i] as f64, arena[i] as f64);
        let (c, d) = (arena[ld + i] as f64, arena[5 * ld + i] as f64);
        let (re, im) = (got[2 * ld + i], got[4 * ld + i]);
        if i < n {
            assert_close(&[re, im], &[a * c - b * d + 1.0, a * d + b * c], 1e-6, "w");
        } else {
            // points past n are not touched
            assert!(re.is_nan() && im.is_nan());
        }
    }
    // the inputs are unchanged
    for p in [0, 1, 3, 5] {
        assert_eq!(&got[p * ld..(p + 1) * ld], &arena[p * ld..(p + 1) * ld]);
    }
}
