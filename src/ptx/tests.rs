// The PTX of models compiled with symjit's Rust API, saved and read back: the instructions
// each MIR operation becomes, the interface (layouts, Symbolica models), the errors, and
// `ptxas` accepting every module (when the CUDA toolkit is installed; see `ptxas`).

use std::path::PathBuf;
use std::process::Command;

use symjit::{var, Application, Compiler, CompilerType, Complex, Composer, Config, Expr};
use symjit::{Slot, Storage, Translator};

use super::*;
use crate::sjb::SavedModel;

/// `ptxas` of the CUDA toolkit: `$PTXAS`, or the default install location.
fn ptxas_path() -> Option<PathBuf> {
    let path = std::env::var_os("PTXAS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda/bin/ptxas"));
    path.exists().then_some(path)
}

/// Assembles the module for sm_120 with ptxas, if it is installed (skipped otherwise).
fn ptxas(text: &str, what: &str) {
    let Some(ptxas) = ptxas_path() else {
        eprintln!("skipped: ptxas is not installed (set $PTXAS)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("sjb2ptx-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // tests run in parallel and may check modules of the same name
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let file = dir.join(format!(
        "{}-{}.ptx",
        what.replace(|c: char| !c.is_alphanumeric(), "_"),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&file, text).unwrap();
    let out = Command::new(ptxas)
        .args(["-arch=sm_120", "-o", "/dev/null"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "ptxas rejects the module of {}:\n{}\n{}",
        what,
        String::from_utf8_lossy(&out.stderr),
        text
    );
    let _ = std::fs::remove_file(&file);
}

fn config(ty: CompilerType, opt_level: u8, fastmath: bool) -> Config {
    let mut config = Config::new(ty, 0).unwrap();
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

fn ptx_with(app: &Application, layout: Layout) -> anyhow::Result<Ptx> {
    let mut options = PtxOptions::new("model");
    options.layout = layout;
    saved(app).ptx(&options)
}

fn ptx(app: &Application) -> Ptx {
    let ptx = ptx_with(app, Layout::Soa).unwrap();
    ptxas(&ptx.text, "model");
    ptx
}

/// arithmetic, comparisons, a select, sqrt, abs and roundings of x, y, z
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

fn count(text: &str, pat: &str) -> usize {
    text.matches(pat).count()
}

#[test]
fn arithmetic_for_every_target_and_option() {
    for ty in [
        CompilerType::Native,
        CompilerType::Arm,
        CompilerType::RiscV,
        CompilerType::ByteCode,
    ] {
        for opt_level in 0..=3 {
            for fastmath in [false, true] {
                let what = format!("{:?}, opt level {}, fastmath {}", ty, opt_level, fastmath);
                // compiled for the host, then saved as compiled for `ty`: the MIR is the
                // host's, register-allocated for its opt level
                let mut app = arithmetic(config(CompilerType::ByteCode, opt_level, fastmath));
                app.prog.builder.config.ty = ty;
                let p = ptx_with(&app, Layout::Soa).unwrap();
                ptxas(&p.text, &what);
                assert_eq!(
                    p.shape,
                    Shape {
                        count_in: 3,
                        count_out: 5,
                        count_params: 0,
                        row_first: false
                    }
                );
                // no contraction by ptxas: every add, sub and mul is `.rn`, FMAs come from
                // the MIR only
                for op in ["add", "sub", "mul"] {
                    assert_eq!(
                        count(&p.text, &format!("{}.f64", op)),
                        0,
                        "{}: unrounded {}",
                        what,
                        op
                    );
                }
                if !fastmath {
                    assert_eq!(count(&p.text, "fma.rn.f64"), 0, "{}", what);
                }
                assert!(p.text.contains("div.rn.f64"), "{}", what);
                assert!(p.text.contains("sqrt.rn.f64"), "{}", what);
            }
        }
    }
}

#[test]
fn fused_operations() {
    let app = arithmetic(config(CompilerType::Native, 2, true));
    let p = ptx(&app);
    assert!(p.text.contains("fma.rn.f64"), "{}", p.text);
}

/// x86's semantics: unordered gt/geq, sign/abs/neg on the sign bit, round half to even
#[test]
fn operation_semantics() {
    let (x, y) = (var("x"), var("y"));
    let exprs = [
        x.gt(&y),
        x.ge(&y),
        x.lt(&y),
        x.le(&y),
        x.eq(&y),
        x.ne(&y),
        -&x,
        x.abs(),
        x.round(),
        x.fract(),
        y.recip(),
    ];
    let app = Compiler::with_config(config(CompilerType::Native, 0, false))
        .compile(&[x, y], &exprs)
        .unwrap();
    let p = ptx(&app);
    for pat in [
        "setp.gtu.f64",
        "setp.geu.f64",
        "setp.lt.f64",
        "setp.le.f64",
        "setp.eq.f64",
        "setp.neu.f64",
        "selp.b64",
        "xor.b64",
        "0x8000000000000000",
        "0x7FFFFFFFFFFFFFFF",
        "cvt.rni.f64.f64",
        "cvt.rmi.f64.f64",
    ] {
        assert!(p.text.contains(pat), "no `{}` in\n{}", pat, p.text);
    }
}

#[test]
fn layouts() {
    let app = arithmetic(config(CompilerType::Native, 2, true));
    let soa = ptx_with(&app, Layout::Soa).unwrap();
    let aos = ptx_with(&app, Layout::Aos).unwrap();
    ptxas(&aos.text, "aos");

    // SoA: a pointer that steps by ld_in per slot; AoS: offsets from the point's row
    assert!(
        soa.text.contains("ld.global.nc.b64 %m2, [%a1];"),
        "{}",
        soa.text
    );
    assert!(soa.text.contains("add.u64 %a1, %a1, %ldi;"));
    assert!(
        aos.text.contains("ld.global.nc.b64 %m2, [%pin+16];"),
        "{}",
        aos.text
    );
    assert!(aos.text.contains("mul.lo.u64 %a0, %i, %ldi;"));
    // five outputs after the three states
    assert_eq!(count(&soa.text, "st.global.b64 [%a1], %m"), 5);
    assert!(
        aos.text.contains("st.global.b64 [%pout+32], %m7;"),
        "{}",
        aos.text
    );
}

#[test]
fn shared_parameters() {
    let (x, a) = (var("x"), var("a"));
    let app = Compiler::with_config(config(CompilerType::Native, 2, true))
        .compile_params(
            std::slice::from_ref(&x),
            &[&(&a * &x) + &a],
            std::slice::from_ref(&a),
        )
        .unwrap();
    let p = ptx(&app);
    assert_eq!(
        p.shape,
        Shape {
            count_in: 1,
            count_out: 1,
            count_params: 1,
            row_first: false
        }
    );
    assert!(p.text.contains("[%par+0]"), "{}", p.text);
}

/// z / w + z * w, z^2 and a complex sqrt of complex inputs (Symbolica convention: inputs are
/// the parameters, outputs `mem`)
fn complex(stable: bool, fast_complex: bool, opt_level: u8) -> Application {
    let mut config = config(CompilerType::Native, opt_level, true);
    config.set_complex(true);
    config.set_symbolica(true);
    config.set_stable_math(stable);
    config.set_fast_complex(fast_complex);
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let (z, w) = (Slot::Param(0), Slot::Param(1));
    t.append_div(&Slot::Temp(0), &z, &w, false, false).unwrap();
    t.append_mul(&Slot::Temp(1), &[z, w], 0).unwrap();
    t.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Temp(1)], 0)
        .unwrap();
    t.append_pow(&Slot::Out(1), &z, 2, false).unwrap();
    t.append_fun(&Slot::Out(2), "root", &[w], false).unwrap();
    t.compile().unwrap()
}

#[test]
fn complex_models() {
    for stable in [false, true] {
        for fast_complex in [false, true] {
            for opt_level in [0, 2] {
                let what = format!(
                    "complex, stable {}, fast_complex {}, opt level {}",
                    stable, fast_complex, opt_level
                );
                let app = complex(stable, fast_complex, opt_level);
                let p = ptx_with(&app, Layout::Soa).unwrap();
                ptxas(&p.text, &what);
                // two complex inputs and three complex outputs, in f64 slots
                assert_eq!(
                    p.shape,
                    Shape {
                        count_in: 4,
                        count_out: 6,
                        count_params: 0,
                        row_first: true
                    },
                    "{}",
                    what
                );
                // inputs are read from `in`, there are no shared parameters
                assert!(!p.text.contains("[%par"), "{}", what);
                assert!(p.text.contains("add.u64 %a1, %a1, %pin;"), "{}", what);
                // the complex square root is a funclet: called through %site, returning
                // through the table, which comes first
                let table = p.text.find("$SITES: .branchtargets").expect(&what);
                let jump = p.text.find("brx.idx %site, $SITES;").expect(&what);
                assert!(table < jump, "{}", what);
            }
        }
    }
}

#[test]
fn kernel_loops() {
    let mut config = config(CompilerType::Native, 2, true);
    config.set_symbolica(true);
    // kernel loops need the direct translator
    config.set_direct(true);
    let mut t = Translator::new(config);
    t.set_num_params(1);
    let one = t.append_constant(Complex::new(1.0, 0.0)).unwrap();
    let ten = t.append_constant(Complex::new(10.0, 0.0)).unwrap();
    // out = x; while out < 10 { out = out + 1 }
    t.append_assign(&Slot::Temp(0), &Slot::Param(0)).unwrap();
    t.append_label(1).unwrap();
    t.append_fun(
        &Slot::Temp(1),
        "lt",
        &[Slot::Temp(0), Slot::Const(ten)],
        true,
    )
    .unwrap();
    t.append_if_else(&Slot::Temp(1), 2).unwrap();
    t.append_add(&Slot::Temp(0), &[Slot::Temp(0), Slot::Const(one)], 0)
        .unwrap();
    t.append_goto(1).unwrap();
    t.append_label(2).unwrap();
    t.append_assign(&Slot::Out(0), &Slot::Temp(0)).unwrap();
    let app = t.compile().unwrap();
    let p = ptx(&app);
    // the loop test (a mask, NaN when true) and the jump back
    assert!(p.text.contains("setp.nan.f64 %p0,"), "{}", p.text);
    assert!(p.text.contains("bra.uni $L"), "{}", p.text);
    // the constants as immediates (10 = 0x4024000000000000)
    assert!(p.text.contains("0d4024000000000000"), "{}", p.text);
}

/// The entry points of the math library: every function of v227's `VirtualTable` except
/// `random` (and `sin_cos`, which the MIR does not use).
const REAL_FUNCTIONS: [&str; 37] = [
    "sin", "cos", "tan", "csc", "sec", "cot", "sinc", "sinh", "cosh", "tanh", "csch", "sech",
    "coth", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "cbrt", "exp", "ln",
    "log", "expm1", "log1p", "exp2", "log2", "erf", "erfc", "gamma", "loggamma", "Si", "Ci", "Shi",
    "Chi", "arg", "power",
];
const COMPLEX_FUNCTIONS: [&str; 25] = [
    "sin", "cos", "tan", "csc", "sec", "cot", "sinc", "sinh", "cosh", "tanh", "csch", "sech",
    "coth", "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh", "cbrt", "exp", "ln",
    "log", "arg", "power",
];

#[test]
fn library_entry_points() {
    for op in REAL_FUNCTIONS.iter().chain(&["atan2"]) {
        let sig = libm::signature(op).unwrap_or_else(|| panic!("no `{}`", op));
        let binary = *op == "power" || *op == "atan2";
        assert_eq!(sig.count_params, if binary { 2 } else { 1 }, "{}", op);
        assert!(!sig.complex, "{}", op);
    }
    for op in COMPLEX_FUNCTIONS {
        let sig = libm::signature(&format!("cplx_{}", op)).unwrap_or_else(|| panic!("no `{}`", op));
        assert_eq!(
            sig.count_params,
            if op == "power" { 4 } else { 2 },
            "{}",
            op
        );
        assert!(sig.complex, "{}", op);
    }
    assert!(libm::signature("random").is_none());
    assert!(libm::signature("cplx_random").is_none());

    // a module with all of them assembles, and each function appears once
    let mut ops: Vec<String> = REAL_FUNCTIONS.iter().map(|s| s.to_string()).collect();
    ops.push("atan2".into());
    ops.extend(COMPLEX_FUNCTIONS.iter().map(|s| format!("cplx_{}", s)));
    let text = libm::functions(&ops);
    let module = format!(
        ".version {}\n.target sm_70\n.address_size 64\n\n{}",
        PTX_VERSION, text
    );
    ptxas(&module, "the whole library");
    for op in &ops {
        let def = format!(") __sj_{}(\n", op);
        assert_eq!(count(&text, &def), 1, "{}", op);
    }

    // only what is needed: sinh needs exp, expm1 and __expo2, but nothing trigonometric
    let text = libm::functions(&["sinh".to_string()]);
    assert!(text.contains(") __sjm_expm1(\n"), "{}", text);
    assert!(
        !text.contains("__sjm_sin(") && !text.contains("__sjm_cos("),
        "{}",
        text
    );
    assert!(libm::functions(&[]).is_empty());
}

/// libm.ptx is what libm/gen.sh makes of the sources (needs clang 18 and LLVM 18's tools).
#[test]
fn library_is_current() {
    let tools = ["clang-18", "llvm-link-18", "opt-18", "llc-18", "python3"];
    if tools
        .iter()
        .any(|t| Command::new(t).arg("--version").output().is_err())
    {
        eprintln!("skipped: clang 18 or LLVM 18's tools are not installed");
        return;
    }
    let dir = std::env::temp_dir().join(format!("sjb2ptx-libm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("libm.ptx");
    let gen = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ptx/libm/gen.sh");
    let status = Command::new("sh")
        .arg(gen)
        .arg("--ptx")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let fresh = std::fs::read_to_string(&out).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        fresh == include_str!("libm/libm.ptx"),
        "src/ptx/libm/libm.ptx is stale: run src/ptx/libm/gen.sh"
    );
}

#[test]
fn math_functions() {
    let (x, y) = (var("x"), var("y"));
    let app = Compiler::with_config(config(CompilerType::Native, 2, true))
        .compile(
            &[x.clone(), y.clone()],
            &[
                &x.sin() + &x.exp(),
                x.pow(&y),
                Expr::binary("atan2", &x, &y),
                Expr::unary("Si", &x),
            ],
        )
        .unwrap();
    let p = ptx(&app);
    for pat in [
        "call.uni (ret0), __sj_sin, (arg0);",
        "call.uni (ret0), __sj_power, (arg0, arg1);",
        "call.uni (ret0), __sj_atan2, (arg0, arg1);",
        "call.uni (ret0), __sj_Si, (arg0);",
        ".visible .func  (.param .b64 func_retval0) __sj_exp(\n",
        // sin's argument reduction, also needed by Si
        ".func  (.param .b32 func_retval0) __sjm___rem_pio2(\n",
    ] {
        assert!(p.text.contains(pat), "no `{}` in\n{}", pat, p.text);
    }
    assert_eq!(count(&p.text, ") __sjm___rem_pio2(\n"), 1);
    assert!(!p.text.contains("__sj_cplx_"), "{}", p.text);

    // complex functions: one argument in (Ret, Temp), powers also in (Gen(0), Gen(1))
    let mut config = config(CompilerType::Native, 2, true);
    config.set_complex(true);
    config.set_symbolica(true);
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let (z, w) = (Slot::Param(0), Slot::Param(1));
    t.append_fun(&Slot::Out(0), "sin", &[z], false).unwrap();
    t.append_fun(&Slot::Out(1), "power", &[z, w], false)
        .unwrap();
    let app = t.compile().unwrap();
    let p = ptx(&app);
    for pat in [
        "call.uni (ret0), __sj_cplx_sin, (arg0, arg1);",
        "call.uni (ret0), __sj_cplx_power, (arg0, arg1, arg2, arg3);",
        "ld.param.b64 %tmp, [ret0+8];",
    ] {
        assert!(p.text.contains(pat), "no `{}` in\n{}", pat, p.text);
    }
}

#[test]
fn unsupported_functions() {
    let x = var("x");
    let app = Compiler::with_config(config(CompilerType::Native, 2, true))
        .compile(std::slice::from_ref(&x), &[Expr::unary("random", &x)])
        .unwrap();
    let err = ptx_with(&app, Layout::Soa).err().unwrap().to_string();
    assert!(
        err.contains("does not support the function `random`"),
        "{}",
        err
    );

    // the library's names are reserved
    let app = arithmetic(config(CompilerType::Native, 2, true));
    let err = saved(&app)
        .ptx(&PtxOptions::new("__sjm_exp"))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("reserved"), "{}", err);
}

#[test]
fn header_describes_the_interface() {
    let app = arithmetic(config(CompilerType::Native, 2, true));
    let p = ptx(&app);
    let mut options = PtxOptions::new("model");
    let h = header(&options, &p.shape, false, "model.ptx", "model.h");
    for pat in [
        "#define MODEL_KERNEL \"model\"",
        "#define MODEL_COUNT_IN 3",
        "#define MODEL_COUNT_OUT 5",
        "#define MODEL_COUNT_PARAMS 0",
        "#define MODEL_LAYOUT_SOA 1",
        "in[k * ld_in + i]",
    ] {
        assert!(h.contains(pat), "no `{}` in\n{}", pat, h);
    }
    options.layout = Layout::Aos;
    let h = header(&options, &p.shape, true, "model.ptx", "model.h");
    assert!(h.contains("#define MODEL_LAYOUT_SOA 0"));
    assert!(h.contains("in[i * ld_in + k]"));
    assert!(h.contains("#define MODEL_COMPLEX 1"));
}
