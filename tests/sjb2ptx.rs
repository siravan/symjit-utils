//! sjb2ptx end to end: models compiled with symjit's Rust API are saved (`.sjb`), compiled by the
//! binary with `--driver`, linked with a C harness by `cc` and run on the GPU; their outputs are
//! compared bit for bit with the JIT's scalar kernel (the generator follows x86's
//! `amd/scalar.rs`). Without a usable GPU (no driver, or `/dev/nvidia*` unreachable, e.g. in
//! a sandbox) the GPU steps are skipped; `SJB2PTX_REQUIRE_GPU=1` makes that a failure.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use symjit::{
    var, Application, Compiler, CompilerType, Complex, Composer, Config, Expr, Slot, Storage,
    Translator,
};

const SJB2PTX: &str = env!("CARGO_BIN_EXE_sjb2ptx");

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sjb2ptx")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(ty: CompilerType, opt_level: u8, fastmath: bool) -> Config {
    let mut config = Config::new(ty, 0).unwrap();
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config.set_fastmath(fastmath);
    config
}

fn save(app: &Application, path: &Path) {
    let mut f = std::fs::File::create(path).unwrap();
    app.save(&mut f).unwrap();
}

fn sjb2ptx(d: &Path, args: &[&str]) -> Output {
    Command::new(SJB2PTX)
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

/// The points: random values and the special ones (each column of `SPECIAL` is a point).
fn points(count_in: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    const SPECIAL: [f64; 12] = [
        0.0,
        -0.0,
        1e-310,
        -1e-310,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        2.5,
        -2.5,
        1e300,
        -1e-300,
        1.0,
    ];
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    (0..n)
        .map(|i| {
            (0..count_in)
                .map(|k| {
                    if i < SPECIAL.len() {
                        SPECIAL[(i + 5 * k) % SPECIAL.len()]
                    } else {
                        next()
                    }
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Host,
    Device,
}

/// How a test runs the kernel.
#[derive(Clone, Copy)]
struct Run {
    aos: bool,
    /// the leading dimensions are this much larger than needed
    pad: usize,
    /// `<name>_cuda_set_batch` (0: the default)
    batch: usize,
    mode: Mode,
}

const RUN: Run = Run {
    aos: false,
    pad: 0,
    batch: 0,
    mode: Mode::Host,
};

/// A C program that runs kernel `name` with its driver: it reads the inputs (and parameters)
/// from files in the kernel's layout, evaluates them on the host or device path, and writes
/// the outputs. Exit status 77: no usable GPU.
fn harness(d: &Path, name: &str) {
    let up = name.to_uppercase();
    let c = format!(
        r#"#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "{name}_cuda.h"

typedef int CUresult;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;

static void *slurp(const char *path, size_t *len) {{
    FILE *f = fopen(path, "rb");
    if (!f) {{ perror(path); exit(2); }}
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(*len ? *len : 1);
    if (fread(p, 1, *len, f) != *len) {{ perror(path); exit(2); }}
    fclose(f);
    return p;
}}

typedef CUresult (*retain_t)(CUcontext *, int);
typedef CUresult (*push_t)(CUcontext);
typedef CUresult (*pop_t)(CUcontext *);
typedef CUresult (*alloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*free_t)(CUdeviceptr);
typedef CUresult (*htod_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*dtoh_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*sync_t)(void);

/* the device path: copies to device memory, launches, copies back */
static int device(const char *name_unused, {name}_cuda *ctx, size_t n, const double *in,
                  size_t in_bytes, size_t ld_in, double *out, size_t out_bytes, size_t ld_out,
                  const double *params, size_t params_bytes) {{
    (void)name_unused;
    /* the symbols cuda.h maps these names to (CUDA 13's cuCtxSynchronize_v2 takes a context) */
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) return -1;
    retain_t cuDevicePrimaryCtxRetain = (retain_t)dlsym(lib, "cuDevicePrimaryCtxRetain");
    push_t cuCtxPushCurrent = (push_t)dlsym(lib, "cuCtxPushCurrent_v2");
    pop_t cuCtxPopCurrent = (pop_t)dlsym(lib, "cuCtxPopCurrent_v2");
    alloc_t cuMemAlloc = (alloc_t)dlsym(lib, "cuMemAlloc_v2");
    free_t cuMemFree = (free_t)dlsym(lib, "cuMemFree_v2");
    htod_t cuMemcpyHtoD = (htod_t)dlsym(lib, "cuMemcpyHtoD_v2");
    dtoh_t cuMemcpyDtoH = (dtoh_t)dlsym(lib, "cuMemcpyDtoH_v2");
    sync_t cuCtxSynchronize = (sync_t)dlsym(lib, "cuCtxSynchronize");
    CUcontext c, old;
    CUdeviceptr din = 0, dout = 0, dpar = 0;
    if (cuDevicePrimaryCtxRetain(&c, 0) || cuCtxPushCurrent(c)) return -1;
    if (cuMemAlloc(&din, in_bytes + 8) || cuMemAlloc(&dout, out_bytes + 8)) return -1;
    if (params && cuMemAlloc(&dpar, params_bytes + 8)) return -1;
    if (cuMemcpyHtoD(din, in, in_bytes) || cuMemcpyHtoD(dout, out, out_bytes)) return -1;
    if (params && cuMemcpyHtoD(dpar, params, params_bytes)) return -1;
    int r = {name}_cuda_evaluate_device(ctx, n, din, ld_in, dout, ld_out, dpar, NULL);
    if (r == 0 && (cuCtxSynchronize() || cuMemcpyDtoH(out, dout, out_bytes))) r = -1;
    cuMemFree(din);
    cuMemFree(dout);
    if (dpar) cuMemFree(dpar);
    cuCtxPopCurrent(&old);
    return r;
}}

/* {name} n ld_in ld_out batch host|device in.bin params.bin|- out.bin */
int main(int argc, char **argv) {{
    if (argc != 9) return 2;
    size_t n = strtoull(argv[1], NULL, 10), ld_in = strtoull(argv[2], NULL, 10);
    size_t ld_out = strtoull(argv[3], NULL, 10), batch = strtoull(argv[4], NULL, 10);
    size_t in_bytes, params_bytes = 0;
    double *in = slurp(argv[6], &in_bytes);
    double *params = strcmp(argv[7], "-") ? slurp(argv[7], &params_bytes) : NULL;
    size_t out_count = {up}_CUDA_LAYOUT_SOA ? {up}_CUDA_COUNT_OUT * ld_out : n * ld_out;
    double *out = calloc(out_count ? out_count : 1, sizeof(double));

    {name}_cuda *ctx;
    if ({name}_cuda_init(&ctx, 0) != 0) {{
        const char *e = {name}_cuda_error(ctx);
        fprintf(stderr, "%s\n", e);
        return strstr(e, "cuInit") || strstr(e, "libcuda") || strstr(e, "cuDeviceGet") ? 77 : 1;
    }}
    if (batch) {name}_cuda_set_batch(ctx, batch);
    int r = strcmp(argv[5], "host") == 0
        ? {name}_cuda_evaluate(ctx, n, in, ld_in, out, ld_out, params)
        : device(argv[0], ctx, n, in, in_bytes, ld_in, out, 8 * out_count, ld_out, params,
                 params_bytes);
    if (r != 0) {{
        fprintf(stderr, "%s\n", {name}_cuda_error(ctx));
        return 1;
    }}
    FILE *f = fopen(argv[8], "wb");
    fwrite(out, sizeof(double), out_count, f);
    fclose(f);
    {name}_cuda_free(ctx);
    free(in);
    free(params);
    free(out);
    return 0;
}}
"#
    );
    std::fs::write(d.join(format!("{}_main.c", name)), c).unwrap();
    let out = Command::new("cc")
        .args(["-Wall", "-Wextra", "-Werror", "-O1"])
        .arg(d.join(format!("{}_main.c", name)))
        .arg(d.join(format!("{}_cuda.c", name)))
        .args(["-ldl", "-o"])
        .arg(d.join(name))
        .output()
        .unwrap();
    ok(&out);
}

fn write_f64(path: &Path, values: &[f64]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

/// Compiles `name.sjb` with `--driver` (and `--layout aos` if `run.aos`), builds the harness
/// and evaluates `points` (with the shared `params`); the outputs per point, or None without
/// a GPU.
fn gpu(
    d: &Path,
    name: &str,
    run: Run,
    points: &[Vec<f64>],
    params: &[f64],
    count_out: usize,
) -> Option<Vec<Vec<f64>>> {
    let sjb = format!("{}.sjb", name);
    let ptx = format!("{}.ptx", name);
    let mut args = vec![sjb.as_str(), "-o", ptx.as_str(), "--driver"];
    if run.aos {
        args.extend(["--layout", "aos"]);
    }
    ok(&sjb2ptx(d, &args));
    harness(d, name);

    let n = points.len();
    let count_in = points.first().map_or(0, |p| p.len());
    let (ld_in, ld_out) = if run.aos {
        (count_in + run.pad, count_out + run.pad)
    } else {
        (n + run.pad, n + run.pad)
    };
    // the inputs in the kernel's layout, NaN in the padding
    let mut input = vec![f64::NAN; if run.aos { n * ld_in } else { count_in * ld_in }];
    for (i, p) in points.iter().enumerate() {
        for (k, v) in p.iter().enumerate() {
            input[if run.aos {
                i * ld_in + k
            } else {
                k * ld_in + i
            }] = *v;
        }
    }
    write_f64(&d.join("in.bin"), &input);
    let params_file = if params.is_empty() {
        "-".to_string()
    } else {
        write_f64(&d.join("params.bin"), params);
        "params.bin".to_string()
    };

    let out = Command::new(d.join(name))
        .args([
            n.to_string(),
            ld_in.to_string(),
            ld_out.to_string(),
            run.batch.to_string(),
            if run.mode == Mode::Host {
                "host"
            } else {
                "device"
            }
            .to_string(),
            "in.bin".to_string(),
            params_file,
            "out.bin".to_string(),
        ])
        .current_dir(d)
        .output()
        .unwrap();
    if out.status.code() == Some(77) {
        let msg = String::from_utf8_lossy(&out.stderr);
        assert!(
            std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none(),
            "no usable GPU: {}",
            msg
        );
        eprintln!("skipped: no usable GPU ({})", msg.trim());
        return None;
    }
    ok(&out);

    let bytes = std::fs::read(d.join("out.bin")).unwrap();
    let all: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    Some(
        (0..n)
            .map(|i| {
                (0..count_out)
                    .map(|j| {
                        all[if run.aos {
                            i * ld_out + j
                        } else {
                            j * ld_out + i
                        }]
                    })
                    .collect()
            })
            .collect(),
    )
}

/// The same bits (NaNs of any payload are equal: the GPU's NaNs are canonical).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn assert_same(got: &[Vec<f64>], expected: &[Vec<f64>], points: &[Vec<f64>], what: &str) {
    assert_eq!(got.len(), expected.len(), "{}", what);
    let mut bad = Vec::new();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        for (j, (a, b)) in g.iter().zip(e.iter()).enumerate() {
            if !same(*a, *b) {
                bad.push(format!(
                    "point {} {:?}, output {}: GPU {:?}, JIT {:?}",
                    i, points[i], j, a, b
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{}: {} differences, e.g.\n{}",
        what,
        bad.len(),
        bad[..bad.len().min(8)].join("\n")
    );
}

type Model = (Vec<Expr>, Vec<Expr>);

/// The six comparisons as the bits of a number (a sum of masks would hide them: a true mask is
/// a NaN), with x86's NaN rules (`>` and `>=` are true for NaN operands, `!=` too).
fn comparisons(x: &Expr, y: &Expr, z: &Expr) -> Expr {
    let bit = |c: Expr, v: f64| c.ifelse(&Expr::from(v), &Expr::from(0.0));
    let terms = [
        bit(x.gt(y), 1.0),
        bit(x.ge(z), 2.0),
        bit(x.lt(y), 4.0),
        bit(y.le(z), 8.0),
        bit(y.eq(z), 16.0),
        bit(x.ne(y), 32.0),
    ];
    terms[1..].iter().fold(terms[0].clone(), |a, b| &a + b)
}

/// arithmetic, square roots, comparisons, selects, roundings, min/max and powers of x, y, z
fn arithmetic() -> Model {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let obs = vec![
        &(&(&x * &y) + &z) / &(&one + &(&z * &z)),
        &(&(&x * &x) + &(&y * &y)).sqrt() - &(&x - &Expr::from(0.75)).abs(),
        x.lt(&y).ifelse(&(&x * &z), &(&y - &z)),
        &(&x.floor() + &y.ceil()) + &(&z.round() * &x.trunc()),
        &x.max(&y) - &z.min(&x),
        &(&x.pow(&Expr::from(3.0)) - &(&Expr::from(2.0) * &y.pow(&Expr::from(-2.0)))) + &z.fract(),
        comparisons(&x, &y, &z),
        &(-&x).recip() + &y.abs(),
    ];
    (vec![x, y, z], obs)
}

fn compile(model: &Model, config: Config) -> Application {
    Compiler::with_config(config)
        .compile(&model.0, &model.1)
        .unwrap()
}

/// Equal but for the rounding of a few operations: the MIR of another machine fuses different
/// multiplications and additions into FMAs.
fn assert_close(got: &[Vec<f64>], expected: &[Vec<f64>], points: &[Vec<f64>], what: &str) {
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        for (j, (a, b)) in g.iter().zip(e.iter()).enumerate() {
            let close = same(*a, *b) || (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0);
            assert!(
                close,
                "{}: point {} {:?}, output {}: GPU {:?}, JIT {:?}",
                what, i, points[i], j, a, b
            );
        }
    }
}

/// The JIT's outputs for the MIR of `app` (compiled for any machine), run on this machine:
/// the application saved for `native` and loaded again (MIR of opt level <= 2 fits x86-64's
/// registers).
fn jit_of(app: &mut Application, points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let ty = app.prog.builder.config.ty;
    app.prog.builder.config.ty = CompilerType::Native;
    let mut bytes: Vec<u8> = Vec::new();
    app.save(&mut bytes).unwrap();
    app.prog.builder.config.ty = ty;
    let mut jit = Application::load(
        &mut &bytes[..],
        &Config::new(CompilerType::ByteCode, 0).unwrap(),
    )
    .unwrap();
    points.iter().map(|p| jit.call(p)).collect()
}

/// Real models, bit for bit with the x86 JIT running the same MIR, for every opt level, with
/// and without fused multiply-adds, and for MIR saved for ARM64 and RISC-V and the bytecode
/// interpreter. ARM64 and RISC-V MIR of opt level 3 (up to 27 registers) does not fit x86-64:
/// it is compared with this machine's own compilation, whose FMAs may differ.
#[test]
fn real_models_match_the_jit() {
    let d = dir("real");
    let model = arithmetic();
    let pts = points(3, 2_003, 1);
    for (ty, saved_ty) in [
        (CompilerType::Native, CompilerType::Native),
        (CompilerType::Arm, CompilerType::Arm),
        (CompilerType::RiscV, CompilerType::RiscV),
        (CompilerType::Native, CompilerType::ByteCode),
    ] {
        for opt_level in 0..=3 {
            for fastmath in [false, true] {
                let what = format!(
                    "{:?} MIR saved as {:?}, opt level {}, fastmath {}",
                    ty, saved_ty, opt_level, fastmath
                );
                let mut app = compile(&model, config(ty, opt_level, fastmath));
                app.prog.builder.config.ty = saved_ty;
                save(&app, &d.join("real.sjb"));

                let foreign = ty != CompilerType::Native && opt_level == 3;
                let expected = if foreign {
                    let mut jit =
                        compile(&model, config(CompilerType::Native, opt_level, fastmath));
                    pts.iter().map(|p| jit.call(p)).collect()
                } else {
                    jit_of(&mut app, &pts)
                };

                let Some(got) = gpu(&d, "real", RUN, &pts, &[], 8) else {
                    return;
                };
                if foreign {
                    assert_close(&got, &expected, &pts, &what);
                } else {
                    assert_same(&got, &expected, &pts, &what);
                }
            }
        }
    }
}

/// Both layouts, leading dimensions larger than needed, small batches, the device path.
#[test]
fn layouts_batches_and_device_pointers() {
    let d = dir("layouts");
    let model = arithmetic();
    let mut app = compile(&model, config(CompilerType::Native, 2, true));
    save(&app, &d.join("lay.sjb"));
    let pts = points(3, 10_007, 2);
    let expected: Vec<Vec<f64>> = pts.iter().map(|p| app.call(p)).collect();

    for aos in [false, true] {
        for (pad, batch, mode) in [
            (0, 0, Mode::Host),
            (3, 0, Mode::Host),
            (0, 1_000, Mode::Host),
            (5, 777, Mode::Host),
            (0, 0, Mode::Device),
            (2, 0, Mode::Device),
        ] {
            let run = Run {
                aos,
                pad,
                batch,
                mode,
            };
            let what = format!(
                "aos {}, padding {}, batch {}, device {}",
                aos,
                pad,
                batch,
                mode == Mode::Device
            );
            let Some(got) = gpu(&d, "lay", run, &pts, &[], 8) else {
                return;
            };
            assert_same(&got, &expected, &pts, &what);
        }
    }
}

#[test]
fn shared_parameters() {
    let d = dir("params");
    let (x, y, a, b) = (var("x"), var("y"), var("a"), var("b"));
    let obs = vec![&(&a * &x) + &(&b * &y), &(&x - &a) / &(&y + &b)];
    let mut app = Compiler::with_config(config(CompilerType::Native, 2, true))
        .compile_params(&[x, y], &obs, &[a, b])
        .unwrap();
    save(&app, &d.join("par.sjb"));
    let pts = points(2, 1_001, 3);
    let params = [1.25, -0.5];
    let expected: Vec<Vec<f64>> = pts.iter().map(|p| app.call_params(p, &params)).collect();
    for aos in [false, true] {
        let run = Run { aos, ..RUN };
        let Some(got) = gpu(&d, "par", run, &pts, &params, 2) else {
            return;
        };
        assert_same(&got, &expected, &pts, &format!("aos {}", aos));
    }
}

/// A Symbolica-convention model with a kernel loop: out0 = x + k for the least k >= 0 with
/// x + k >= 10 (at most 64 steps), out1 = the number of steps; the inputs are the parameters.
#[test]
fn symbolica_loops() {
    let d = dir("loops");
    let mut c = config(CompilerType::Native, 2, true);
    c.set_symbolica(true);
    // kernel loops need the direct translator (the indirect one does not reassign `Temp`s)
    c.set_direct(true);
    let mut t = Translator::new(c);
    t.set_num_params(1);
    let one = t.append_constant(Complex::new(1.0, 0.0)).unwrap();
    let ten = t.append_constant(Complex::new(10.0, 0.0)).unwrap();
    let zero = t.append_constant(Complex::new(0.0, 0.0)).unwrap();
    let limit = t.append_constant(Complex::new(64.0, 0.0)).unwrap();
    t.append_assign(&Slot::Temp(0), &Slot::Param(0)).unwrap();
    t.append_assign(&Slot::Temp(2), &Slot::Const(zero)).unwrap();
    t.append_label(1).unwrap();
    t.append_fun(
        &Slot::Temp(1),
        "lt",
        &[Slot::Temp(0), Slot::Const(ten)],
        true,
    )
    .unwrap();
    t.append_fun(
        &Slot::Temp(3),
        "lt",
        &[Slot::Temp(2), Slot::Const(limit)],
        true,
    )
    .unwrap();
    t.append_fun(&Slot::Temp(1), "and", &[Slot::Temp(1), Slot::Temp(3)], true)
        .unwrap();
    t.append_if_else(&Slot::Temp(1), 2).unwrap();
    t.append_add(&Slot::Temp(0), &[Slot::Temp(0), Slot::Const(one)], 0)
        .unwrap();
    t.append_add(&Slot::Temp(2), &[Slot::Temp(2), Slot::Const(one)], 0)
        .unwrap();
    t.append_goto(1).unwrap();
    t.append_label(2).unwrap();
    t.append_assign(&Slot::Out(0), &Slot::Temp(0)).unwrap();
    t.append_assign(&Slot::Out(1), &Slot::Temp(2)).unwrap();
    let app = t.compile().unwrap();
    save(&app, &d.join("loop.sjb"));
    let applet = app.seal().unwrap();

    let pts = points(1, 1_000, 4);
    let expected: Vec<Vec<f64>> = pts
        .iter()
        .map(|p| {
            let mut outs = [0.0; 2];
            applet.evaluate(p, &mut outs);
            outs.to_vec()
        })
        .collect();
    let Some(got) = gpu(&d, "loop", RUN, &pts, &[], 2) else {
        return;
    };
    assert_same(&got, &expected, &pts, "loops");
    // the threads ran different numbers of iterations
    let steps: std::collections::HashSet<u64> = got.iter().map(|o| o[1].to_bits()).collect();
    assert!(steps.len() >= 5, "{:?}", steps);
}

/// z / w + z * w, 1 / (z + w), z^2 and sqrt(w) of complex z, w, in the Symbolica convention
fn complex(config: Config) -> Application {
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let (z, w) = (Slot::Param(0), Slot::Param(1));
    t.append_div(&Slot::Temp(0), &z, &w, false, false).unwrap();
    t.append_mul(&Slot::Temp(1), &[z, w], 0).unwrap();
    t.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Temp(1)], 0)
        .unwrap();
    t.append_add(&Slot::Temp(2), &[z, w], 0).unwrap();
    t.append_fun(&Slot::Out(1), "recip", &[Slot::Temp(2)], false)
        .unwrap();
    t.append_pow(&Slot::Out(2), &z, 2, false).unwrap();
    t.append_fun(&Slot::Out(3), "root", &[w], false).unwrap();
    t.compile().unwrap()
}

/// Complex models are lowered to real operations, as the JIT's kernels without fast complex:
/// bit for bit with that scalar kernel, with and without `stable_math` (huge and tiny values).
#[test]
fn complex_models_match_the_jit() {
    let d = dir("complex");
    let pts = points(4, 2_001, 5);
    for stable in [false, true] {
        for opt_level in [0, 2, 3] {
            for saved_fast_complex in [false, true] {
                let what = format!(
                    "stable {}, opt level {}, saved with fast_complex {}",
                    stable, opt_level, saved_fast_complex
                );
                let make = |fast_complex: bool| {
                    let mut c = config(CompilerType::Native, opt_level, true);
                    c.set_complex(true);
                    c.set_symbolica(true);
                    c.set_stable_math(stable);
                    c.set_fast_complex(fast_complex);
                    complex(c)
                };
                save(&make(saved_fast_complex), &d.join("cplx.sjb"));
                let jit = make(false).seal().unwrap();
                let expected: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|p| {
                        let args = [Complex::new(p[0], p[1]), Complex::new(p[2], p[3])];
                        let mut outs = [Complex::default(); 4];
                        jit.evaluate(&args, &mut outs);
                        outs.iter().flat_map(|z| [z.re, z.im]).collect()
                    })
                    .collect();
                let Some(got) = gpu(&d, "cplx", RUN, &pts, &[], 8) else {
                    return;
                };
                assert_same(&got, &expected, &pts, &what);
            }
        }
    }
}

/// The math library's entry points (src/ptx/libm): stable's functions but `random`.
const REAL_FUNCTIONS: [&str; 36] = [
    "sin", "cos", "tan", "csc", "sec", "cot", "sinc", "sinh", "cosh", "tanh", "csch", "sech",
    "coth", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "cbrt", "exp", "ln",
    "log", "expm1", "log1p", "exp2", "log2", "erf", "erfc", "gamma", "loggamma", "Si", "Ci", "Shi",
    "Chi", "arg",
];
const BINARY_FUNCTIONS: [&str; 2] = ["power", "atan2"];
const COMPLEX_FUNCTIONS: [&str; 24] = [
    "sin", "cos", "tan", "csc", "sec", "cot", "sinc", "sinh", "cosh", "tanh", "csch", "sech",
    "coth", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "cbrt", "exp", "ln",
    "log", "arg",
];

/// Points for the math functions: all pairs of special and interesting values, then random
/// values of random magnitudes (2^-40 to 2^40, some up to 2^1000) and signs.
fn function_points(count_in: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
    const SPECIAL: [f64; 40] = [
        0.0,
        -0.0,
        1e-310,
        -1e-310,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        3.0,
        -3.5,
        1e-8,
        -1e-8,
        1e5,
        -1e5,
        1e22,
        -1e22,
        1e300,
        -1e300,
        709.9,
        -745.2,
        0.1,
        22.5,
        -170.5,
        171.7,
        1.0000001,
        0.9999999,
        std::f64::consts::FRAC_PI_2,
        -std::f64::consts::PI,
        1e9 + 0.5,
        88.5,
        999.0,
        12.0,
        -4.000001,
        0.7,
        -23.25,
    ];
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 11
    };
    let len = SPECIAL.len();
    (0..n)
        .map(|i| {
            (0..count_in)
                .map(|k| {
                    if i < len * len {
                        // pairs of consecutive inputs run through all combinations
                        SPECIAL[if k % 2 == 0 { i % len } else { i / len }]
                    } else {
                        let r = next();
                        let mantissa = 1.0 + (r & 0xFFFFF) as f64 / (1 << 20) as f64;
                        let e = if r >> 40 & 15 == 0 {
                            ((r >> 21) % 2001) as i32 - 1000
                        } else {
                            ((r >> 21) % 81) as i32 - 40
                        };
                        let sign = if r >> 50 & 1 == 0 { 1.0 } else { -1.0 };
                        sign * mantissa * 2f64.powi(e)
                    }
                })
                .collect()
        })
        .collect()
}

/// A real model with every real function of x (and power and atan2 of x and y), and a complex
/// one (Symbolica convention) with every complex function of z (and z^w).
fn function_models() -> (Application, Application) {
    let (x, y) = (var("x"), var("y"));
    let mut obs: Vec<Expr> = REAL_FUNCTIONS.iter().map(|f| Expr::unary(f, &x)).collect();
    obs.extend(BINARY_FUNCTIONS.iter().map(|f| Expr::binary(f, &x, &y)));
    let real = Compiler::with_config(config(CompilerType::Native, 2, true))
        .compile(&[x, y], &obs)
        .unwrap();

    let mut c = config(CompilerType::Native, 2, true);
    c.set_complex(true);
    c.set_symbolica(true);
    c.set_fast_complex(false);
    let mut t = Translator::new(c);
    t.set_num_params(2);
    let (z, w) = (Slot::Param(0), Slot::Param(1));
    for (k, f) in COMPLEX_FUNCTIONS.iter().enumerate() {
        t.append_fun(&Slot::Out(k), f, &[z], false).unwrap();
    }
    t.append_fun(&Slot::Out(COMPLEX_FUNCTIONS.len()), "power", &[z, w], false)
        .unwrap();
    let complex = t.compile().unwrap();
    (real, complex)
}

/// The library's C compiled for this machine as it is for the GPU (src/ptx/libm/gen.sh --host),
/// linked into a program that evaluates the functions of `function_models` on the points of
/// `in.bin` (one row per point) and writes their values to `ref.bin`. None without clang 18.
fn host_reference(d: &Path, points: &[Vec<f64>], complex: bool) -> Option<Vec<Vec<f64>>> {
    let gen = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ptx/libm/gen.sh");
    let lib = d.join("libsj.o");
    if !lib.exists() {
        match Command::new("sh").arg(gen).arg("--host").arg(&lib).output() {
            Ok(out) if out.status.success() => {}
            _ => {
                eprintln!("skipped: clang 18 or LLVM 18's tools are not installed (gen.sh --host)");
                return None;
            }
        }
    }

    let mut decls = String::new();
    let mut body = String::new();
    let count_out = if complex {
        for (j, f) in COMPLEX_FUNCTIONS.iter().enumerate() {
            decls += &format!("cplx __sj_cplx_{}(double, double);\n", f);
            body += &format!(
                "        r = __sj_cplx_{}(p[0], p[1]); o[{}] = r.re; o[{}] = r.im;\n",
                f,
                2 * j,
                2 * j + 1
            );
        }
        let j = COMPLEX_FUNCTIONS.len();
        decls += "cplx __sj_cplx_power(double, double, double, double);\n";
        body += &format!(
            "        r = __sj_cplx_power(p[0], p[1], p[2], p[3]); o[{}] = r.re; o[{}] = r.im;\n",
            2 * j,
            2 * j + 1
        );
        2 * (j + 1)
    } else {
        for (j, f) in REAL_FUNCTIONS.iter().enumerate() {
            decls += &format!("double __sj_{}(double);\n", f);
            body += &format!("        o[{}] = __sj_{}(p[0]);\n", j, f);
        }
        for (j, f) in BINARY_FUNCTIONS.iter().enumerate() {
            decls += &format!("double __sj_{}(double, double);\n", f);
            body += &format!(
                "        o[{}] = __sj_{}(p[0], p[1]);\n",
                REAL_FUNCTIONS.len() + j,
                f
            );
        }
        REAL_FUNCTIONS.len() + BINARY_FUNCTIONS.len()
    };
    let count_in = if complex { 4 } else { 2 };
    let c = format!(
        r#"#include <stdio.h>
#include <stdlib.h>
typedef struct {{ double re, im; }} cplx;
{decls}
int main(int argc, char **argv) {{
    size_t n = strtoull(argv[1], NULL, 10);
    double *in = malloc(8 * n * {count_in}), *out = malloc(8 * n * {count_out});
    FILE *f = fopen(argv[2], "rb");
    if (fread(in, 8, n * {count_in}, f) != n * {count_in}) return 1;
    fclose(f);
    for (size_t i = 0; i < n; i++) {{
        const double *p = in + i * {count_in};
        double *o = out + i * {count_out};
        cplx r;
        (void)r;
{body}    }}
    f = fopen(argv[3], "wb");
    fwrite(out, 8, n * {count_out}, f);
    fclose(f);
    (void)argc;
    return 0;
}}
"#
    );
    std::fs::write(d.join("ref.c"), c).unwrap();
    ok(&Command::new("cc")
        .args(["-O1", "ref.c", "libsj.o", "-o", "ref"])
        .current_dir(d)
        .output()
        .unwrap());
    let flat: Vec<f64> = points.iter().flatten().copied().collect();
    write_f64(&d.join("ref_in.bin"), &flat);
    ok(&Command::new(d.join("ref"))
        .args([
            points.len().to_string(),
            "ref_in.bin".into(),
            "ref.bin".into(),
        ])
        .current_dir(d)
        .output()
        .unwrap());
    let bytes = std::fs::read(d.join("ref.bin")).unwrap();
    let all: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    Some(all.chunks(count_out).map(|c| c.to_vec()).collect())
}

/// The kernels call each function of the library (none is composed or inlined by symjit), and
/// the GPU computes the bits the same C computes on this machine: the library's functions are
/// compiled without FMA contraction, so their operations are IEEE's.
#[test]
fn math_library_matches_its_host_build() {
    let d = dir("libm");
    let (real, complex) = function_models();

    let pts = function_points(2, 6_000, 11);
    save(&real, &d.join("real.sjb"));
    ok(&sjb2ptx(&d, &["real.sjb", "-o", "real_check.ptx"]));
    let text = std::fs::read_to_string(d.join("real_check.ptx")).unwrap();
    for f in REAL_FUNCTIONS.iter().chain(&BINARY_FUNCTIONS) {
        assert!(
            text.contains(&format!("), __sj_{}, (", f)),
            "no call of `{}`",
            f
        );
    }
    let Some(expected) = host_reference(&d, &pts, false) else {
        return;
    };
    let count_out = REAL_FUNCTIONS.len() + BINARY_FUNCTIONS.len();
    let Some(got) = gpu(&d, "real", RUN, &pts, &[], count_out) else {
        return;
    };
    assert_same(&got, &expected, &pts, "real functions, GPU vs host build");

    let pts = function_points(4, 6_000, 12);
    save(&complex, &d.join("cplx.sjb"));
    ok(&sjb2ptx(&d, &["cplx.sjb", "-o", "cplx_check.ptx"]));
    let text = std::fs::read_to_string(d.join("cplx_check.ptx")).unwrap();
    for f in COMPLEX_FUNCTIONS.iter().chain(&["power"]) {
        assert!(
            text.contains(&format!("), __sj_cplx_{}, (", f)),
            "no call of complex `{}`",
            f
        );
    }
    let expected = host_reference(&d, &pts, true).unwrap();
    let count_out = 2 * (COMPLEX_FUNCTIONS.len() + 1);
    let got = gpu(&d, "cplx", RUN, &pts, &[], count_out).unwrap();
    assert_same(
        &got,
        &expected,
        &pts,
        "complex functions, GPU vs host build",
    );
}

/// The distance of a and b in units in the last place (0 for the same value, NaNs equal; the
/// maximum for a NaN against a number or values of different signs).
fn ulps(a: f64, b: f64) -> u64 {
    if same(a, b) || a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// The relative distance of two complex values (0 for the same bits; components that are the
/// same, e.g. both infinite, do not count).
fn relative(g: (f64, f64), e: (f64, f64)) -> f64 {
    if same(g.0, e.0) && same(g.1, e.1) {
        return 0.0;
    }
    let d = |a: f64, b: f64| if same(a, b) { 0.0 } else { a - b };
    let rel = d(g.0, e.0).hypot(d(g.1, e.1)) / e.0.hypot(e.1);
    if rel.is_nan() {
        f64::INFINITY
    } else {
        rel
    }
}

/// The library against the JIT (glibc's functions, Cephes' and num_complex's on glibc's):
/// real functions within 8 ulps, complex ones within 1e-12 (relative) for arguments of moderate
/// size. Known deviations: stable's Cephes gamma(±0) is NaN (musl: ±inf, as C's tgamma), it
/// flushes subnormal results to 0, and its lgamma is +inf for subnormal arguments, -inf at -inf and off by up to 3e-9 near 1 and 2
/// (musl's is right, checked with mpmath); num_complex's arcsin, arccos and arcsinh cancel for
/// large |z|, and their results depend on the last bits of the real functions there.
#[test]
fn math_functions_match_the_jit_within_ulps() {
    let d = dir("libm_jit");
    let (mut real, _) = function_models();
    let pts = function_points(2, 6_000, 21);
    save(&real, &d.join("real.sjb"));
    let expected = jit_of(&mut real, &pts);
    let count_out = REAL_FUNCTIONS.len() + BINARY_FUNCTIONS.len();
    let Some(got) = gpu(&d, "real", RUN, &pts, &[], count_out) else {
        return;
    };
    let names: Vec<&str> = REAL_FUNCTIONS
        .iter()
        .chain(&BINARY_FUNCTIONS)
        .copied()
        .collect();
    let mut bad = Vec::new();
    for (j, name) in names.iter().enumerate() {
        let mut worst = 0;
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            let (x, a, b) = (pts[i][0], g[j], e[j]);
            let u = ulps(a, b);
            let known = match *name {
                "gamma" => {
                    (x == 0.0 && b.is_nan() && a == 1.0 / x)
                        || (b == 0.0 && a.abs() < f64::MIN_POSITIVE)
                }
                "loggamma" => {
                    (x.abs() < f64::MIN_POSITIVE && b == f64::INFINITY && a.is_finite())
                        || (x == f64::NEG_INFINITY && a == f64::INFINITY)
                        || (a - b).abs() <= 1e-8 * b.abs()
                }
                _ => false,
            };
            if u <= 8 {
                worst = worst.max(u);
            } else if !known {
                bad.push(format!("{} {:?}: GPU {:?}, JIT {:?}", name, pts[i], a, b));
            }
        }
        eprintln!("{:>10}: at most {} ulps", name, worst);
    }
    assert!(
        bad.is_empty(),
        "real functions, GPU vs JIT: {} differences, e.g.\n{}",
        bad.len(),
        bad[..bad.len().min(8)].join("\n")
    );

    let (_, complex) = function_models();
    let pts = function_points(4, 6_000, 22);
    save(&complex, &d.join("cplx.sjb"));
    let (_, complex) = function_models();
    let jit = complex.seal().unwrap();
    let count = COMPLEX_FUNCTIONS.len() + 1;
    let expected: Vec<Vec<f64>> = pts
        .iter()
        .map(|p| {
            let args = [Complex::new(p[0], p[1]), Complex::new(p[2], p[3])];
            let mut outs = vec![Complex::default(); count];
            jit.evaluate(&args, &mut outs);
            outs.iter().flat_map(|z| [z.re, z.im]).collect()
        })
        .collect();
    let got = gpu(&d, "cplx", RUN, &pts, &[], 2 * count).unwrap();
    let names: Vec<&str> = COMPLEX_FUNCTIONS
        .iter()
        .chain(&["power"])
        .copied()
        .collect();
    let mut bad = Vec::new();
    for (k, name) in names.iter().enumerate() {
        let limit = if ["arcsin", "arccos", "arcsinh"].contains(name) {
            16.0
        } else {
            1048576.0
        };
        let moderate = |v: f64| v == 0.0 || (v.abs() >= 1.0 / 1048576.0 && v.abs() <= limit);
        let mut worst = 0.0f64;
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            if !pts[i].iter().all(|v| moderate(*v)) {
                continue;
            }
            let rel = relative((g[2 * k], g[2 * k + 1]), (e[2 * k], e[2 * k + 1]));
            worst = worst.max(rel);
            if rel > 1e-12 {
                bad.push(format!(
                    "{} {:?}: GPU {:?} {:?}, JIT {:?} {:?}",
                    name,
                    pts[i],
                    g[2 * k],
                    g[2 * k + 1],
                    e[2 * k],
                    e[2 * k + 1]
                ));
            }
        }
        eprintln!("{:>10}: at most {:e} (complex, relative)", name, worst);
    }
    assert!(
        bad.is_empty(),
        "complex functions, GPU vs JIT: {} differences, e.g.\n{}",
        bad.len(),
        bad[..bad.len().min(8)].join("\n")
    );
}

/// Real and complex models that combine calls of math functions with arithmetic, for every
/// opt level, with and without FMAs, MIR saved for other machines and `stable_math`: the calls
/// keep the values around them (the GPU has no caller-saved registers to spill), and the
/// results are the JIT's but for the functions' last bits (relative 1e-12).
#[test]
fn math_models_match_the_jit() {
    let d = dir("libm_models");
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let model: Model = (
        vec![x.clone(), y.clone(), z.clone()],
        vec![
            &(&x.sin() * &(-&(&y * &y)).exp()) + &(&one + &(&x * &x)).ln(),
            &Expr::binary("atan2", &y, &x) * &(&x / &Expr::from(4.0)).cosh(),
            (&x.abs() + &one).pow(&y),
            &(&x.tanh() + &y.erf()) - &(&z * &x.sinc()),
            &(&z.cbrt() * &x.cos()) + &(&y.abs() + &Expr::from(0.5)).lgam(),
            x.lt(&y).ifelse(&z.exp_m1(), &(&x.asinh() + &y.atan())),
        ],
    );
    let pts = points(3, 2_003, 31);
    for (ty, saved_ty) in [
        (CompilerType::Native, CompilerType::Native),
        (CompilerType::Arm, CompilerType::Arm),
        (CompilerType::RiscV, CompilerType::RiscV),
        (CompilerType::Native, CompilerType::ByteCode),
    ] {
        for opt_level in 0..=3 {
            for fastmath in [false, true] {
                let what = format!(
                    "{:?} MIR saved as {:?}, opt level {}, fastmath {}",
                    ty, saved_ty, opt_level, fastmath
                );
                let mut app = compile(&model, config(ty, opt_level, fastmath));
                app.prog.builder.config.ty = saved_ty;
                save(&app, &d.join("real.sjb"));
                let mut jit = compile(&model, config(CompilerType::Native, opt_level, fastmath));
                let expected: Vec<Vec<f64>> = pts.iter().map(|p| jit.call(p)).collect();
                let Some(got) = gpu(&d, "real", RUN, &pts, &[], 6) else {
                    return;
                };
                assert_close(&got, &expected, &pts, &what);
            }
        }
    }

    // complex: z, w of moderate size (the random points)
    let pts: Vec<Vec<f64>> = points(4, 2_012, 32).split_off(12);
    let build = |c: Config| {
        let mut t = Translator::new(c);
        t.set_num_params(2);
        let (z, w) = (Slot::Param(0), Slot::Param(1));
        // exp(z) w + sin(w) / z, ln(z) + z^w, tan(z w), sqrt(cosh(w)) and acosh(z)
        t.append_fun(&Slot::Temp(0), "exp", &[z], false).unwrap();
        t.append_mul(&Slot::Temp(1), &[Slot::Temp(0), w], 0)
            .unwrap();
        t.append_fun(&Slot::Temp(2), "sin", &[w], false).unwrap();
        t.append_div(&Slot::Temp(3), &Slot::Temp(2), &z, false, false)
            .unwrap();
        t.append_add(&Slot::Out(0), &[Slot::Temp(1), Slot::Temp(3)], 0)
            .unwrap();
        t.append_fun(&Slot::Temp(4), "ln", &[z], false).unwrap();
        t.append_fun(&Slot::Temp(5), "power", &[z, w], false)
            .unwrap();
        t.append_add(&Slot::Out(1), &[Slot::Temp(4), Slot::Temp(5)], 0)
            .unwrap();
        t.append_mul(&Slot::Temp(6), &[z, w], 0).unwrap();
        t.append_fun(&Slot::Out(2), "tan", &[Slot::Temp(6)], false)
            .unwrap();
        t.append_fun(&Slot::Temp(7), "cosh", &[w], false).unwrap();
        t.append_fun(&Slot::Out(3), "root", &[Slot::Temp(7)], false)
            .unwrap();
        t.append_fun(&Slot::Out(4), "arccosh", &[z], false).unwrap();
        t.compile().unwrap()
    };
    for stable in [false, true] {
        for opt_level in [0, 2, 3] {
            let what = format!("complex, stable {}, opt level {}", stable, opt_level);
            let make = || {
                let mut c = config(CompilerType::Native, opt_level, true);
                c.set_complex(true);
                c.set_symbolica(true);
                c.set_stable_math(stable);
                c.set_fast_complex(false);
                build(c)
            };
            save(&make(), &d.join("cplx.sjb"));
            let jit = make().seal().unwrap();
            let expected: Vec<Vec<f64>> = pts
                .iter()
                .map(|p| {
                    let args = [Complex::new(p[0], p[1]), Complex::new(p[2], p[3])];
                    let mut outs = [Complex::default(); 5];
                    jit.evaluate(&args, &mut outs);
                    outs.iter().flat_map(|z| [z.re, z.im]).collect()
                })
                .collect();
            let got = gpu(&d, "cplx", RUN, &pts, &[], 10).unwrap();
            for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                for k in 0..5 {
                    let rel = relative((g[2 * k], g[2 * k + 1]), (e[2 * k], e[2 * k + 1]));
                    assert!(
                        rel <= 1e-12,
                        "{}: point {:?}, output {}: GPU {:?} {:?}, JIT {:?} {:?}",
                        what,
                        pts[i],
                        k,
                        g[2 * k],
                        g[2 * k + 1],
                        e[2 * k],
                        e[2 * k + 1]
                    );
                }
            }
        }
    }
}

#[test]
fn files_and_errors() {
    let d = dir("cli");
    let model = arithmetic();
    save(
        &compile(&model, config(CompilerType::Native, 2, true)),
        &d.join("m.sjb"),
    );

    // the module and header by default; the driver on request, named after the kernel
    ok(&sjb2ptx(&d, &["m.sjb"]));
    assert!(d.join("m.ptx").exists() && d.join("m.h").exists());
    assert!(!d.join("m_cuda.c").exists());
    ok(&sjb2ptx(
        &d,
        &[
            "m.sjb",
            "-o",
            "k.ptx",
            "--name",
            "kern",
            "--no-header",
            "--driver",
        ],
    ));
    assert!(d.join("k.ptx").exists() && !d.join("k.h").exists());
    let driver = std::fs::read_to_string(d.join("kern_cuda.c")).unwrap();
    assert!(driver.contains(".visible .entry kern("), "{}", driver);
    assert!(std::fs::read_to_string(d.join("kern_cuda.h"))
        .unwrap()
        .contains("int kern_cuda_init(kern_cuda **ctx, int device);"));
    // the driver builds without warnings
    let out = Command::new("cc")
        .args([
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            "kern_cuda.c",
            "-o",
            "kern_cuda.o",
        ])
        .current_dir(&d)
        .output()
        .unwrap();
    ok(&out);

    // random numbers are not supported, nor names of the math library
    let x = var("x");
    let app = Compiler::with_config(config(CompilerType::Native, 2, true))
        .compile(std::slice::from_ref(&x), &[Expr::unary("random", &x)])
        .unwrap();
    save(&app, &d.join("random.sjb"));
    fails(
        &sjb2ptx(&d, &["random.sjb"]),
        1,
        "does not support the function `random`",
    );
    fails(
        &sjb2ptx(&d, &["m.sjb", "--name", "__sjm_exp"]),
        1,
        "is reserved",
    );
    fails(
        &sjb2ptx(&d, &["missing.sjb"]),
        1,
        "cannot open `missing.sjb`",
    );
    fails(
        &sjb2ptx(&d, &["m.sjb", "--name", "1bad"]),
        1,
        "not a C identifier",
    );

    for (args, msg) in [
        (&[][..], "no input file"),
        (&["m.sjb", "--layout", "rows"][..], "unknown layout `rows`"),
        (&["m.sjb", "--sm", "x"][..], "expected a number"),
        (
            &["m.sjb", "--no-header", "--header", "h.h"][..],
            "exclude each other",
        ),
        (&["m.sjb", "--bogus"][..], "unknown option `--bogus`"),
    ] {
        fails(&sjb2ptx(&d, args), 2, msg);
    }
}

/// The driver reports bad arguments and keeps working.
#[test]
fn driver_errors() {
    let d = dir("driver_errors");
    let model = arithmetic();
    save(
        &compile(&model, config(CompilerType::Native, 2, true)),
        &d.join("e.sjb"),
    );
    ok(&sjb2ptx(&d, &["e.sjb", "--driver"]));
    let c = r#"#include <stdio.h>
#include <string.h>
#include "e_cuda.h"
int main(void) {
    double in[3] = {1, 2, 3}, out[8];
    e_cuda *ctx;
    if (e_cuda_init(&ctx, 0) != 0) {
        const char *e = e_cuda_error(ctx);
        fprintf(stderr, "%s\n", e);
        return strstr(e, "cuInit") || strstr(e, "libcuda") || strstr(e, "cuDeviceGet") ? 77 : 1;
    }
    /* SoA with ld_in < n */
    if (e_cuda_evaluate(ctx, 2, in, 1, out, 2, NULL) != -1) return 3;
    if (!strstr(e_cuda_error(ctx), "too small")) return 4;
    /* then a good call, which clears the error */
    if (e_cuda_evaluate(ctx, 1, in, 1, out, 1, NULL) != 0) return 5;
    if (e_cuda_error(ctx)[0] != 0) return 6;
    if (e_cuda_evaluate(ctx, 0, NULL, 0, NULL, 0, NULL) != 0) return 7;
    e_cuda_free(ctx);
    e_cuda_free(NULL);
    printf("%.17g\n", out[0]);
    return 0;
}
"#;
    std::fs::write(d.join("main.c"), c).unwrap();
    ok(&Command::new("cc")
        .args([
            "-Wall", "-Werror", "main.c", "e_cuda.c", "-ldl", "-o", "main",
        ])
        .current_dir(&d)
        .output()
        .unwrap());
    let out = Command::new(d.join("main")).output().unwrap();
    if out.status.code() == Some(77) {
        assert!(std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none());
        eprintln!("skipped: no usable GPU");
        return;
    }
    ok(&out);
    // (1 * 2 + 3) / (1 + 9)
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "0.5");
}
