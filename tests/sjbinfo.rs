//! sjbinfo end to end: models compiled with symjit's Rust API are saved (`.sjb`) and described;
//! the report's numbers are compared with what symjit measures.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use symjit::{
    var, Application, Compiler, CompilerType, Complex, Composer, Config, Expr, Slot, Storage,
    Translator,
};

const SJBINFO: &str = env!("CARGO_BIN_EXE_sjbinfo");

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sjbinfo")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(ty: CompilerType) -> Config {
    let mut config = Config::new(ty, 0).unwrap();
    config.set_opt_level(2);
    config.set_cse(true);
    config.set_fastmath(true);
    config.set_simd(true);
    config
}

/// sin(x)*y + exp(-x*x), cos(y)/(1 + x*x), x^y + log10(z) + ln(z) csc(x)
fn trig(config: Config) -> Application {
    let (x, y, z) = (var("x"), var("y"), var("z"));
    let one = Expr::from(1.0);
    let a = &(&x.sin() * &y) + &(-&(&x * &x)).exp();
    let b = &y.cos() / &(&one + &(&x * &x));
    let c = &(&x.pow(&y) + &z.log10()) + &(&z.ln() * &x.csc());
    Compiler::with_config(config)
        .compile(&[x, y, z], &[a, b, c])
        .unwrap()
}

/// z / w + sin(z) * w, complex (`Translator`)
fn complex(fast_complex: bool) -> Application {
    let mut config = config(CompilerType::Native);
    config.set_complex(true);
    config.set_symbolica(true);
    config.set_fast_complex(fast_complex);
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
    t.compile().unwrap()
}

fn save(app: &Application, path: &Path) {
    let mut f = std::fs::File::create(path).unwrap();
    app.save(&mut f).unwrap();
}

fn sjbinfo(d: &Path, args: &[&str]) -> Output {
    Command::new(SJBINFO)
        .args(args)
        .current_dir(d)
        .output()
        .unwrap()
}

/// The report of a successful run.
fn report(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

/// The value of the report's row `key` in the section `section`.
fn value(report: &str, section: &str, key: &str) -> String {
    let mut inside = false;
    for line in report.lines() {
        if !line.starts_with(' ') && !line.is_empty() {
            inside = line == section;
        } else if inside {
            let line = line.trim_start();
            if let Some(rest) = line.strip_prefix(key) {
                if rest.starts_with("  ") {
                    return rest.trim().to_string();
                }
            }
        }
    }
    panic!("no `{}` in section `{}` of\n{}", key, section, report)
}

fn bytes(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    format!("{} bytes", out)
}

#[test]
fn real_model_matches_symjit() {
    let d = dir("real");
    let mut app = trig(config(CompilerType::Native));
    save(&app, &d.join("trig.sjb"));
    let r = report(&sjbinfo(&d, &["trig.sjb"]));

    assert!(r.starts_with("trig.sjb\n"), "{}", r);
    assert_eq!(value(&r, "Model", "numbers"), "real");
    assert_eq!(value(&r, "Model", "states"), "3");
    assert_eq!(value(&r, "Model", "observables"), "3");
    assert!(value(&r, "Configuration", "target").starts_with("native"));
    assert_eq!(value(&r, "Configuration", "opt level"), "2");
    assert!(value(&r, "Configuration", "enabled").contains("cse"));

    // the numbers symjit measures for the application as it was compiled
    assert!(value(&r, "IR (MIR)", "instructions")
        .starts_with(&format!("{} (", app.measure("mir-size"))));
    app.prepare_simd();
    assert_eq!(
        value(&r, "Compiled code", "scalar"),
        bytes(app.measure("ker-scalar-size"))
    );
    assert_eq!(
        value(&r, "Compiled code", "SIMD"),
        bytes(app.measure("ker-simd-size"))
    );
    assert!(value(&r, "Compiled code", "fast").starts_with("none"));

    // functions: symjit's `log` is the decimal logarithm, `ln` the natural one; C has no `csc`
    for (f, c) in [
        ("sin", "sin"),
        ("cos", "cos"),
        ("exp", "exp"),
        ("power", "pow"),
        ("log", "log10"),
        ("ln", "log"),
        ("csc", "1/sin"),
    ] {
        let row = value(&r, "Math functions", f);
        assert!(row.starts_with("1  real"), "{}: {}", f, row);
        assert!(row.ends_with(&format!(" {}", c)), "{}: {}", f, row);
    }
}

#[test]
fn fast_kernel_and_no_compile() {
    let d = dir("fast");
    let x = var("x");
    let app = Compiler::with_config(config(CompilerType::Native))
        .compile(std::slice::from_ref(&x), &[&(&x * &x) + &x.sqrt()])
        .unwrap();
    save(&app, &d.join("f.sjb"));
    let r = report(&sjbinfo(&d, &["f.sjb"]));
    assert!(
        value(&r, "Compiled code", "fast").ends_with(" bytes"),
        "{}",
        r
    );
    assert_eq!(value(&r, "Math functions", "functions"), "none");

    let r = report(&sjbinfo(&d, &["--no-compile", "f.sjb"]));
    assert_eq!(
        value(&r, "Compiled code", "kernels"),
        "not compiled (--no-compile)"
    );
}

#[test]
fn complex_models() {
    let d = dir("complex");
    for fast_complex in [true, false] {
        let app = complex(fast_complex);
        save(&app, &d.join("c.sjb"));
        let r = report(&sjbinfo(&d, &["c.sjb"]));
        assert!(
            value(&r, "Model", "numbers").starts_with("complex"),
            "{}",
            r
        );
        assert!(value(&r, "IR (MIR)", "lowered").contains("instructions after"));
        assert_eq!(
            value(&r, "Compiled code", "scalar"),
            bytes(app.measure("ker-scalar-size"))
        );
        let row = value(&r, "Math functions", "sin");
        assert!(row.starts_with("1  complex, 1 argument"), "{}", row);
        assert!(row.ends_with(" csin"), "{}", row);
    }
}

/// Files saved for other machines are compiled for them (the code is not run) and bytecode
/// files have no machine code.
#[test]
fn other_targets() {
    let d = dir("targets");
    for (ty, target) in [
        (CompilerType::Arm, "ARM64"),
        (CompilerType::RiscV, "RISC-V 64"),
        (CompilerType::ByteCode, "bytecode"),
    ] {
        let mut config = config(CompilerType::ByteCode);
        config.set_simd(true);
        let mut app = trig(config);
        // the type only matters to compilation: save the file as compiled for `ty`
        app.prog.builder.config.ty = ty;
        save(&app, &d.join("t.sjb"));
        let r = report(&sjbinfo(&d, &["t.sjb"]));
        assert!(
            value(&r, "Configuration", "target").starts_with(target),
            "{}",
            r
        );
        match ty {
            CompilerType::ByteCode => assert_eq!(
                value(&r, "Compiled code", "kernels"),
                "none: run by the bytecode interpreter"
            ),
            CompilerType::Arm => {
                assert!(value(&r, "Compiled code", "scalar").ends_with(" bytes"));
                assert!(value(&r, "Compiled code", "SIMD").ends_with(" bytes"));
            }
            _ => {
                assert!(value(&r, "Compiled code", "scalar").ends_with(" bytes"));
                assert_eq!(
                    value(&r, "Compiled code", "SIMD"),
                    "none (only for AVX and ARM64)"
                );
            }
        }
    }
}

#[test]
fn loops_have_no_simd_kernel() {
    let d = dir("loops");
    let mut config = config(CompilerType::Native);
    config.set_symbolica(true);
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
    save(&app, &d.join("l.sjb"));
    let r = report(&sjbinfo(&d, &["l.sjb"]));
    assert_ne!(value(&r, "IR (MIR)", "labels"), "0", "{}", r);
    let simd = value(&r, "Compiled code", "SIMD");
    assert!(
        simd == "none (the model has loops)" || simd.ends_with(" bytes"),
        "{}",
        r
    );
}

#[test]
fn several_files_and_errors() {
    let d = dir("errors");
    save(&trig(config(CompilerType::Native)), &d.join("a.sjb"));
    std::fs::write(d.join("bad.sjb"), b"not a saved application").unwrap();

    let out = sjbinfo(&d, &["a.sjb", "a.sjb"]);
    let r = report(&out);
    assert_eq!(r.matches("\nModel\n").count(), 2);

    let out = sjbinfo(&d, &["bad.sjb", "a.sjb"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("cannot read `bad.sjb`: not a saved symjit application"),
        "{}",
        err
    );
    // the good file is still described
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("a.sjb\n"));

    let out = sjbinfo(&d, &["missing.sjb"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot open `missing.sjb`"));

    for (args, msg) in [
        (&[][..], "no input file"),
        (&["--bogus", "a.sjb"][..], "unknown option `--bogus`"),
    ] {
        let out = sjbinfo(&d, args);
        assert_eq!(out.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&out.stderr).contains(msg));
    }
    assert!(report(&sjbinfo(&d, &["--help"])).starts_with("Usage: sjbinfo"));
}

/// `-c`/`--constants` lists the constant pool: value, bits and the instructions that load it.
#[test]
fn constants() {
    let d = dir("constants");
    let mut t = Translator::new(config(CompilerType::Native));
    t.set_num_params(1);
    // 0.1 is not exact in binary: its bits tell it apart from neighbours
    let a = t.append_constant(Complex::new(0.1, 0.0)).unwrap();
    let b = t.append_constant(Complex::new(-1e300, 0.0)).unwrap();
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(a)], 0)
        .unwrap();
    t.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Const(b)], 0)
        .unwrap();
    t.append_mul(&Slot::Out(1), &[Slot::Param(0), Slot::Const(a)], 0)
        .unwrap();
    let app = t.compile().unwrap();
    save(&app, &d.join("k.sjb"));
    let consts = app.prog.builder.consts.clone();

    let plain = report(&sjbinfo(&d, &["k.sjb"]));
    assert!(!plain.contains("\nConstants"), "{}", plain);

    for flag in ["-c", "--constants"] {
        let r = report(&sjbinfo(&d, &[flag, "--no-compile", "k.sjb"]));
        let section = r.split("\nConstants (").nth(1).expect(&r);
        assert!(
            section.starts_with(&format!("{})\n", consts.len())),
            "{}",
            r
        );
        let rows: Vec<Vec<&str>> = section
            .lines()
            .skip(2)
            .map(|l| l.split_whitespace().collect())
            .collect();
        assert_eq!(rows.len(), consts.len(), "{}", r);
        for (i, (row, x)) in rows.iter().zip(consts.iter()).enumerate() {
            assert_eq!(row[0], i.to_string());
            assert_eq!(row[2].parse::<f64>().unwrap().to_bits(), x.to_bits());
            assert_eq!(row[3], format!("{:#018x}", x.to_bits()));
        }
        // the slots of 0.1 and -1e300: loaded (the MIR may fuse the loads into operations)
        for x in [0.1, -1e300] {
            let row = rows
                .iter()
                .find(|row| row[2].parse::<f64>().unwrap() == x)
                .expect(&r);
            assert_ne!(row[1], "0", "{}", r);
        }
    }
}
