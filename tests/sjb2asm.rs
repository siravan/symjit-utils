//! sjb2asm end to end (moved from symjit's `asm` branch, `tests/disasm_listing.rs`): models built
//! with symjit's Rust API are compiled for RISC-V, AArch64 and x86-64 on any host (the code is
//! never run; x86-64 kernels use the CPU's features: AVX and AVX-512 where the host has them, SSE
//! elsewhere), and their listings checked: every instruction before the data area decodes, labels
//! and label references are named, and where the binutils are installed (`riscv64-linux-gnu-as`,
//! `aarch64-linux-gnu-as`, `x86_64-linux-gnu-as`, or the host's own, with `ld` and `objcopy`) the
//! listing assembles and links at address 0 to the kernel's bytes. Kernels without labels (raw
//! bytes) must find the same data area as `@data`. `SJB2ASM_SAMPLES=DIR` also checks every
//! `*.sjb` and `kernel.json` under DIR (e.g. pacrunner's `lib/pacrunner-simd`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use symjit::{Application, Compiled, Complex, Composer, Config, Slot, Storage, Translator};
use symjit_utils::asm::{self, Kernel};
use symjit_utils::disasm::{self, Mode};

const SJB2ASM: &str = env!("CARGO_BIN_EXE_sjb2asm");

fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("sjb2asm")
        .join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// symjit's defaults (`tests/symjit.toml` of symjit) for the target `ty`
fn config_for(ty: &str, complex: bool) -> Config {
    let mut config = Config::from_name(ty, 0).unwrap();
    config.set_simd(true);
    config.set_cse(true);
    config.set_fastmath(true);
    config.set_opt_level(2);
    config.set_complex(complex);
    config
}

/// out0 = p0 ? p0 p1 : p1 + 7.5;  out1 = exp(p0) + 1e6 p1
fn compile_real(config: Config) -> Application {
    let mut t = Translator::new(config);
    t.set_num_params(2);
    let (p0, p1) = (Slot::Param(0), Slot::Param(1));
    let c75 = t.append_constant(Complex::new(7.5, 0.0)).unwrap();
    let c1e6 = t.append_constant(Complex::new(1e6, 0.0)).unwrap();

    t.append_if_else(&p0, 0).unwrap();
    t.append_mul(&Slot::Temp(1), &[p0, p1], 0).unwrap();
    t.append_goto(1).unwrap();
    t.append_label(0).unwrap();
    t.append_add(&Slot::Temp(2), &[p1, Slot::Const(c75)], 0)
        .unwrap();
    t.append_label(1).unwrap();
    t.append_join(&Slot::Out(0), &p0, &Slot::Temp(1), &Slot::Temp(2))
        .unwrap();

    t.append_fun(&Slot::Temp(3), "exp", &[p0], false).unwrap();
    t.append_mul(&Slot::Temp(4), &[p1, Slot::Const(c1e6)], 0)
        .unwrap();
    t.append_add(&Slot::Out(1), &[Slot::Temp(3), Slot::Temp(4)], 0)
        .unwrap();
    t.compile().unwrap()
}

/// exp(s0) + 7.5 s1 of two states: one output and no parameters, so it has a fast kernel
fn compile_one(config: Config) -> Application {
    let mut config = config;
    config.set_symbolica(false);
    let (x, y) = (symjit::var("x"), symjit::var("y"));
    let f = &x.exp() + &(&symjit::Expr::from(7.5) * &y);
    symjit::Compiler::with_config(config)
        .compile(&[x, y], &[f])
        .unwrap()
}

/// out = exp(p0 (0.2 + 0.3i)) + p1^-3 + p199 (complex; a parameter far from the base)
fn compile_complex(config: Config) -> Application {
    let mut t = Translator::new(config);
    t.set_num_params(200);
    let ca = t.append_constant(Complex::new(0.2, 0.3)).unwrap();
    t.append_mul(&Slot::Temp(0), &[Slot::Param(0), Slot::Const(ca)], 0)
        .unwrap();
    t.append_fun(&Slot::Temp(1), "exp", &[Slot::Temp(0)], false)
        .unwrap();
    t.append_pow(&Slot::Temp(2), &Slot::Param(1), -3, false)
        .unwrap();
    t.append_add(
        &Slot::Out(0),
        &[Slot::Temp(1), Slot::Temp(2), Slot::Param(199)],
        0,
    )
    .unwrap();
    t.compile().unwrap()
}

/// the kinds of kernel the decoders' architecture has
fn kinds(arch: &str) -> &'static [Kernel] {
    if arch == "riscv64" {
        &[Kernel::Scalar, Kernel::Fast]
    } else {
        &[Kernel::Scalar, Kernel::Simd, Kernel::Fast]
    }
}

/// the instruction lines of the assembly form before the data area
fn code_lines(listing: &str) -> Vec<&str> {
    listing
        .lines()
        .take_while(|l| !l.starts_with(".L_data:"))
        .filter(|l| l.starts_with('\t') && !l.trim_start().starts_with('.'))
        .collect()
}

/// every instruction before the data area decodes (no raw words, no addresses)
fn check_decoded(listing: &str, ctx: &str) {
    for line in listing.lines().take_while(|l| !l.starts_with(".L_data:")) {
        assert!(
            !line.contains(".word") && !line.contains(".inst") && !line.contains(".byte"),
            "undecoded: {line}\n{ctx}"
        );
        assert!(!line.contains(" <"), "an address: {line}\n{ctx}");
    }
}

fn check_asm(listing: &str, arch: &str, kind: &str, consts: bool) {
    let ctx = format!("{arch} {kind}:\n{listing}");
    assert!(listing.contains(&format!("kernel, {arch},")), "{ctx}");
    let mut lines = vec![
        format!("\t.globl symjit_{kind}"),
        format!("symjit_{kind}:"),
        ".L_data:".to_string(),
    ];
    // the fast kernel does not call itself
    if kind != "fast" {
        lines.push(".L_self:".to_string());
    }
    for line in lines {
        assert!(listing.lines().any(|l| *l == line), "no {line}: {ctx}");
    }
    let code = code_lines(listing);
    assert!(code.len() > 10, "{ctx}");
    check_decoded(listing, &ctx);
    assert!(code.iter().any(|l| l.trim() == "ret"), "{ctx}");
    if consts {
        // the constant pool as .quads under their labels
        assert!(listing.contains("\n.L_const_0_:\n\t.quad 0x"), "{ctx}");
    }
}

/// the binutils of `arch`, if installed: as, ld, objcopy
fn binutils(arch: &str) -> Option<[String; 3]> {
    let triple = match arch {
        "riscv64" => "riscv64-linux-gnu",
        "aarch64" => "aarch64-linux-gnu",
        "x86_64" => "x86_64-linux-gnu",
        _ => return None,
    };
    let native = cfg!(target_os = "linux") && std::env::consts::ARCH == arch;
    for prefix in [format!("{}-", triple), String::new()] {
        if prefix.is_empty() && !native {
            continue;
        }
        let tools = [
            format!("{}as", prefix),
            format!("{}ld", prefix),
            format!("{}objcopy", prefix),
        ];
        let ok = tools.iter().all(|t| {
            Command::new(t)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        });
        if ok {
            return Some(tools);
        }
    }
    None
}

fn run(cmd: &str, args: &[&str], dir: &Path) {
    let out = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{} {:?}:\n{}",
        cmd,
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// assembles and links the listing at address 0 and compares its bytes with `expected`;
/// false if the binutils are not installed
fn assembles_to(listing: &str, arch: &str, kind: &str, expected: &[u8], dir: &Path) -> bool {
    let Some([asm, ld, objcopy]) = binutils(arch) else {
        eprintln!("no binutils for {arch}: the listing is not assembled");
        return false;
    };
    std::fs::write(dir.join("k.s"), listing).unwrap();
    run(&asm, &["-o", "k.o", "k.s"], dir);
    let entry = format!("symjit_{}", kind);
    run(&ld, &["-Ttext=0", "-e", &entry, "-o", "k.elf", "k.o"], dir);
    run(
        &objcopy,
        &["-O", "binary", "-j", ".text", "k.elf", "k.bin"],
        dir,
    );
    let mut got = std::fs::read(dir.join("k.bin")).unwrap();

    // the assembler may pad the section to its alignment (RISC-V with nops) after the kernel
    if got.len() > expected.len() && got.len() - expected.len() < 16 {
        got.truncate(expected.len());
    }

    if got != expected {
        let first = got
            .iter()
            .zip(expected)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(expected.len()));
        panic!(
            "{arch} {kind}: {} bytes assembled, {} expected; first difference at 0x{:x}\n{listing}",
            got.len(),
            expected.len(),
            first
        );
    }
    true
}

/// the offset of the label `@data` of a kernel
fn data_label(labels: &[(usize, String)]) -> Option<usize> {
    labels.iter().find(|(_, n)| n == "@data").map(|(o, _)| *o)
}

/// every kernel of `app` that `arch` has, in process: the assembly form decodes and assembles
/// back, the hex form covers the kernel, and the raw bytes find `@data` again
fn check_kernels(mut app: Application, arch: &str, name: &str, consts: bool) -> usize {
    let mut checked = 0;
    for kernel in kinds(arch) {
        let kind = kernel.name();
        let Ok(code) = asm::kernel(&mut app, *kernel) else {
            assert!(
                *kernel != Kernel::Scalar,
                "{name}: no scalar kernel for {arch}"
            );
            continue;
        };
        let (bytes, labels) = (code.dumps(), code.labels().to_vec());
        let listing = disasm::listing(arch, kind, &bytes, &labels, Mode::Asm).unwrap();
        check_asm(&listing, arch, kind, consts);
        let d = dir(&format!("{name}-{arch}-{kind}"));
        assembles_to(&listing, arch, kind, &bytes, &d);

        let hex = disasm::listing(arch, kind, &bytes, &labels, Mode::Hex).unwrap();
        assert!(hex.contains("\n@data:\n"), "{hex}");
        assert!(hex.contains(&format!("{} bytes", bytes.len())), "{hex}");
        assert!(hex.lines().any(|l| l.starts_with("       0:  ")), "{hex}");

        // without labels: the data area found from the code is `@data`
        let data = data_label(&labels).unwrap();
        assert_eq!(
            disasm::data_start(arch, &bytes).unwrap(),
            data,
            "{name} {arch} {kind}"
        );
        let raw = asm::disassemble_raw(&bytes, arch, kind, None, Mode::Asm).unwrap();
        check_decoded(&raw, &format!("{name} {arch} {kind} raw:\n{raw}"));
        assembles_to(&raw, arch, kind, &bytes, &d);
        checked += 1;
    }
    checked
}

/// the application as its JSON, compiled again for `ty` as sjb2asm compiles it
fn reloaded(app: &Application, ty: &str) -> Application {
    asm::load(app.to_json().unwrap().as_bytes(), Some(ty)).unwrap()
}

#[test]
fn riscv_kernels() {
    let app = compile_real(config_for("riscv", false));
    let listing = asm::disassemble(
        app.to_json().unwrap().as_bytes(),
        &asm::Options {
            mode: Mode::Asm,
            ..Default::default()
        },
    )
    .unwrap();
    // branches name their labels, label references show the constant they load
    assert!(listing.contains(", .L_main\n"), "{listing}");
    assert!(listing.contains("# .L_const_0_ = 7.5"), "{listing}");
    assert!(listing.contains("# .L_func_exp_\n"), "{listing}");
    assert_eq!(check_kernels(app, "riscv64", "real", true), 1);
    assert_eq!(
        check_kernels(
            compile_one(config_for("riscv", false)),
            "riscv64",
            "one",
            true
        ),
        2
    );
    check_kernels(
        compile_complex(config_for("riscv", true)),
        "riscv64",
        "complex",
        true,
    );
}

#[test]
fn arm_kernels() {
    let app = compile_real(config_for("arm", false));
    let listing = asm::disassemble(
        app.to_json().unwrap().as_bytes(),
        &asm::Options {
            mode: Mode::Asm,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(listing.contains("\tadrp x0, .L_const_0_\n"), "{listing}");
    assert!(listing.contains(", #:lo12:.L_const_0_]"), "{listing}");
    assert!(listing.contains("// .L_const_0_ = 7.5"), "{listing}");
    assert_eq!(check_kernels(app, "aarch64", "real", true), 2);
    assert_eq!(
        check_kernels(
            compile_one(config_for("arm", false)),
            "aarch64",
            "one",
            true
        ),
        3
    );
    check_kernels(
        compile_complex(config_for("arm", true)),
        "aarch64",
        "complex",
        true,
    );
    // complex numbers lowered to real pairs (the real generators)
    let mut config = config_for("arm", true);
    config.set_fast_complex(false);
    check_kernels(compile_complex(config), "aarch64", "complexified", true);
}

#[test]
fn x86_kernels() {
    let app = compile_real(config_for("amd", false));
    let listing = asm::disassemble(
        app.to_json().unwrap().as_bytes(),
        &asm::Options {
            mode: Mode::Asm,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(listing.contains("\t.intel_syntax noprefix\n"), "{listing}");
    assert!(listing.contains("[rip+.L_const_0_]"), "{listing}");
    assert!(listing.contains("# .L_const_0_ = 7.5"), "{listing}");
    check_kernels(app, "x86_64", "real", true);
    check_kernels(compile_one(config_for("amd", false)), "x86_64", "one", true);
    check_kernels(
        compile_complex(config_for("amd", true)),
        "x86_64",
        "complex",
        true,
    );
    let mut config = config_for("amd", true);
    config.set_fast_complex(false);
    check_kernels(compile_complex(config), "x86_64", "complexified", true);
}

#[test]
fn x86_sse_and_avx512_kernels() {
    check_kernels(
        compile_real(config_for("amd-sse", false)),
        "x86_64",
        "sse-real",
        true,
    );
    check_kernels(
        compile_complex(config_for("amd-sse", true)),
        "x86_64",
        "sse-complex",
        true,
    );
    // AVX-512 kernels where the host has AVX-512 (else the AVX or SSE ones)
    let mut config = config_for("amd", false);
    config.enable_simd512(true);
    check_kernels(compile_real(config), "x86_64", "avx512-real", true);
    let mut config = config_for("amd", true);
    config.enable_simd512(true);
    check_kernels(compile_complex(config), "x86_64", "avx512-complex", true);
}

#[test]
fn the_target_is_chosen_by_ty() {
    // one application, saved for the host, compiled for each architecture
    let app = compile_real(config_for("native", false));
    for (ty, arch) in [("riscv", "riscv64"), ("arm", "aarch64"), ("amd", "x86_64")] {
        let other = reloaded(&app, ty);
        assert_eq!(asm::app_arch(&other).unwrap(), arch);
        check_kernels(other, arch, &format!("reloaded-{ty}"), true);
    }
}

/// runs sjb2asm with `args`
fn sjb2asm(args: &[&str]) -> Output {
    Command::new(SJB2ASM).args(args).output().unwrap()
}

fn stdout(out: &Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout.clone()).unwrap()
}

/// the listing without the function table's addresses, which are the compiling process's
fn without_addresses(listing: &str) -> String {
    listing
        .lines()
        .filter(|l| !l.contains("address in the process"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn command_line() {
    let d = dir("cli");
    let app = compile_real(config_for("native", false));
    let json = d.join("real.json");
    std::fs::write(&json, app.to_json().unwrap()).unwrap();
    let sjb = d.join("real.sjb");
    app.save(&mut std::fs::File::create(&sjb).unwrap()).unwrap();
    let (json, sjb) = (json.to_str().unwrap(), sjb.to_str().unwrap());

    for (arch, ty) in [("riscv64", "riscv"), ("aarch64", "arm"), ("x86_64", "amd")] {
        let mut expected = reloaded(&app, ty);
        let code = asm::kernel(&mut expected, Kernel::Scalar).unwrap();
        let (bytes, labels) = (code.dumps(), code.labels().to_vec());
        let want = disasm::listing(arch, "scalar", &bytes, &labels, Mode::Hex).unwrap();
        let want_asm = disasm::listing(arch, "scalar", &bytes, &labels, Mode::Asm).unwrap();

        // .sjb and JSON give the same listings as in process: by default (= --hex) the one with
        // addresses and bytes, with --asm the assembler input
        for input in [json, sjb] {
            for args in [&["--arch", arch, input][..], &["-x", "--arch", arch, input]] {
                let got = stdout(&sjb2asm(args));
                assert_eq!(
                    without_addresses(&got),
                    without_addresses(&want),
                    "{args:?}"
                );
            }
            let got = stdout(&sjb2asm(&["--asm", "--arch", arch, input]));
            assert_eq!(
                without_addresses(&got),
                without_addresses(&want_asm),
                "{input}"
            );
        }
        assert!(want.contains("\n@data:\n"), "{want}");
        assert!(want.lines().any(|l| l.starts_with("       0:  ")), "{want}");

        // -o
        let out = d.join(format!("{arch}.s"));
        stdout(&sjb2asm(&[
            "-a",
            ty,
            "-s",
            "-o",
            out.to_str().unwrap(),
            json,
        ]));
        let text = std::fs::read_to_string(&out).unwrap();
        assert_eq!(without_addresses(&text), without_addresses(&want_asm));

        // the raw bytes: the same code, the data area found from it
        let bin = d.join(format!("{arch}.bin"));
        std::fs::write(&bin, &bytes).unwrap();
        let raw = stdout(&sjb2asm(&["--raw", "--arch", arch, bin.to_str().unwrap()]));
        assert!(raw.lines().any(|l| l.starts_with("       0:  ")), "{raw}");
        let data = data_label(&labels).unwrap();
        assert!(
            raw.lines()
                .next()
                .unwrap()
                .contains(&format!("data 0x{:x}..", data)),
            "{raw}"
        );
        let given = stdout(&sjb2asm(&[
            "--raw",
            "--arch",
            arch,
            "--data",
            &format!("0x{:x}", data),
            bin.to_str().unwrap(),
        ]));
        assert_eq!(raw, given);
    }

    // a kernel the target does not have, usage errors
    let out = sjb2asm(&["--arch", "riscv64", "--kernel", "simd", json]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no simd kernel"));
    for args in [
        &["--raw", json][..],
        &["--arch", "mips", json],
        &["--kernel", "vector", json],
        &["--data", "8", json],
        &[],
    ] {
        assert_eq!(sjb2asm(args).status.code(), Some(2), "{args:?}");
    }
    assert!(stdout(&sjb2asm(&["--help"])).contains("Usage: sjb2asm"));
}

/// the `*.sjb` and `kernel.json` files under `dir`
fn samples(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            samples(&p, out);
        } else if p.extension().is_some_and(|x| x == "sjb")
            || p.file_name().is_some_and(|n| n == "kernel.json")
        {
            out.push(p);
        }
    }
}

#[test]
fn sample_kernels() {
    let Ok(root) = std::env::var("SJB2ASM_SAMPLES") else {
        eprintln!("SJB2ASM_SAMPLES is not set: no sample kernels checked");
        return;
    };
    let mut files = Vec::new();
    samples(Path::new(&root), &mut files);
    files.sort();
    assert!(!files.is_empty(), "no samples under {root}");
    let mut checked = 0;
    for f in &files {
        let bytes = std::fs::read(f).unwrap();
        for (ty, arch) in [("riscv", "riscv64"), ("arm", "aarch64"), ("amd", "x86_64")] {
            let app = asm::load(&bytes, Some(ty)).unwrap();
            let name = f.display().to_string().replace('/', "_");
            checked += check_kernels(app, arch, &name, false);
        }
    }
    eprintln!("{} files, {} kernels", files.len(), checked);
}
