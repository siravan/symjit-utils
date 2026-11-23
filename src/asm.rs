//! `sjb2asm`: a saved symjit application (`.sjb`, or its "symjit-application" JSON) compiled by
//! symjit for a chosen architecture, and its kernel disassembled (`disasm`) with the assembler's
//! labels (`MachineCode::labels`, symjit 2.29.2); or the raw bytes of a kernel (a
//! `dump(.., "scalar")` file) disassembled without labels. See `docs/ASM.md`.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use symjit::backend::MachineCode;
use symjit::{Application, Compiled, Config};

use crate::disasm::{self, Mode};

/// the kernels of an application
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kernel {
    Scalar,
    Simd,
    Fast,
}

impl Kernel {
    pub fn parse(s: &str) -> Result<Kernel> {
        match s {
            "scalar" => Ok(Kernel::Scalar),
            "simd" => Ok(Kernel::Simd),
            "fast" => Ok(Kernel::Fast),
            _ => Err(anyhow!("unknown kernel `{}` (scalar, simd or fast)", s)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            Kernel::Simd => "simd",
            Kernel::Fast => "fast",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    /// the target as symjit's `ty` (`amd`, `amd-avx`, `amd-sse`, `arm`, `riscv`, `native`);
    /// None keeps the file's own
    pub ty: Option<String>,
    pub kernel: Kernel,
    pub mode: Mode,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            ty: None,
            kernel: Kernel::Scalar,
            mode: Mode::Hex,
        }
    }
}

/// symjit's `ty` of an `--arch` value: an architecture (`x86_64`, `aarch64`, `riscv64`) or a `ty`
pub fn ty_of_arch(arch: &str) -> Result<String> {
    Ok(match arch {
        "x86_64" | "x86-64" | "amd64" | "amd" => "amd",
        "aarch64" | "arm64" | "arm" => "arm",
        "riscv64" | "riscv" => "riscv",
        "amd-avx" | "amd-sse" | "native" => arch,
        _ => {
            return Err(anyhow!(
                "unknown architecture `{}` (x86_64, aarch64, riscv64; or amd-avx, amd-sse, native)",
                arch
            ))
        }
    }
    .to_string())
}

/// the decoder's name of an architecture (`--arch` of a raw input)
pub fn arch_name(arch: &str) -> Result<&'static str> {
    Ok(match ty_of_arch(arch)?.as_str() {
        "amd" | "amd-avx" | "amd-sse" => "x86_64",
        "arm" => "aarch64",
        "riscv" => "riscv64",
        _ => host_arch()?,
    })
}

fn host_arch() -> Result<&'static str> {
    let arch = std::env::consts::ARCH;
    disasm::ARCHS
        .iter()
        .find(|a| **a == arch)
        .copied()
        .ok_or_else(|| anyhow!("no disassembler for the host ({})", arch))
}

/// whether `bytes` are a "symjit-application" JSON file (else a `.sjb`)
fn is_json(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| *b == b'{')
}

/// the application of a `.sjb` or JSON file, compiled for `ty` (None: the file's own target).
/// A loaded application is compiled for the target its file names (`from_json` ignores the
/// config passed in), so `ty` is written into the JSON.
pub fn load(bytes: &[u8], ty: Option<&str>) -> Result<Application> {
    // only the defined functions of this config would be used: a .sjb holds none
    let config = Config::from_name("native", 0)?;
    let text = if is_json(bytes) {
        String::from_utf8(bytes.to_vec()).context("the JSON is not UTF-8")?
    } else {
        Application::sjb_mir_json(&mut &bytes[..], &config, "json")?
    };

    let text = match ty {
        None => text,
        Some(ty) => {
            let mut value: Value = serde_json::from_str(&text)?;
            let config = value
                .get_mut("program")
                .and_then(|p| p.get_mut("config"))
                .and_then(|c| c.as_object_mut())
                .ok_or_else(|| anyhow!("not a symjit-application file (no program.config)"))?;
            config.insert("ty".to_string(), Value::String(ty.to_string()));
            value.to_string()
        }
    };

    Application::from_json(&text, &config)
}

/// the decoder's name of the application's target
pub fn app_arch(app: &Application) -> Result<&'static str> {
    if app.config.is_amd64() {
        Ok("x86_64")
    } else if app.config.is_arm64() {
        Ok("aarch64")
    } else if app.config.is_riscv64() {
        Ok("riscv64")
    } else {
        Err(anyhow!("the application is not compiled to machine code"))
    }
}

/// the machine code of a kernel, compiled if it is lazy (simd, fast)
pub fn kernel(app: &mut Application, kernel: Kernel) -> Result<&MachineCode<f64>> {
    let code = match kernel {
        Kernel::Scalar => app.compiled.as_ref(),
        Kernel::Simd => {
            app.prepare_simd();
            app.compiled_simd.as_ref()
        }
        Kernel::Fast => {
            app.prepare_fast();
            app.compiled_fast.as_ref()
        }
    };
    let why = match kernel {
        Kernel::Scalar => "",
        Kernel::Simd => " (x86-64 with AVX and AArch64 only)",
        Kernel::Fast => " (it needs one output, no parameters and few states)",
    };
    code.filter(|c| c.size > 0)
        .ok_or_else(|| anyhow!("the application has no {} kernel{}", kernel.name(), why))
}

/// the listing of a kernel of the application in `bytes` (`.sjb` or JSON)
pub fn disassemble(bytes: &[u8], options: &Options) -> Result<String> {
    // symjit's generators panic on what they cannot compile (the SSE generator, used for x86-64
    // on a host without AVX, has no compress-mode subroutines): an error, not a crash
    let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut app = load(bytes, options.ty.as_deref())?;
        let arch = app_arch(&app)?;
        let code = kernel(&mut app, options.kernel)?;
        Ok((arch, code.dumps(), code.labels().to_vec()))
    }));
    let (arch, bytes, labels) = match compiled {
        Ok(result) => result?,
        Err(payload) => return Err(compile_panic(&payload, bytes, options)),
    };
    disasm::listing(arch, options.kernel.name(), &bytes, &labels, options.mode)
}

/// the error of a panic while symjit compiled the kernel
fn compile_panic(
    payload: &Box<dyn std::any::Any + Send>,
    bytes: &[u8],
    options: &Options,
) -> anyhow::Error {
    let msg = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic".to_string());
    // the target the kernel was compiled for, without compiling again
    let ty = options.ty.clone().or_else(|| {
        let text = if is_json(bytes) {
            String::from_utf8(bytes.to_vec()).ok()?
        } else {
            let config = Config::from_name("native", 0).ok()?;
            Application::sjb_mir_json(&mut &bytes[..], &config, "json").ok()?
        };
        let value: Value = serde_json::from_str(&text).ok()?;
        Some(value["program"]["config"]["ty"].as_str()?.to_string())
    });
    let config = ty.as_deref().and_then(|t| Config::from_name(t, 0).ok());
    match config {
        Some(c) if c.is_amd64() && !c.has_avx() => anyhow!(
            "symjit cannot compile this {} kernel for x86-64 without AVX (its SSE generator: {}); \
             x86-64 kernels need a host with AVX",
            options.kernel.name(),
            msg
        ),
        _ => anyhow!(
            "symjit failed to compile the {} kernel: {}",
            options.kernel.name(),
            msg
        ),
    }
}

/// the listing of raw kernel bytes for `arch` (a decoder name): the data area starts at `data`,
/// or where `disasm::data_start` finds it
pub fn disassemble_raw(
    code: &[u8],
    arch: &str,
    kind: &str,
    data: Option<usize>,
    mode: Mode,
) -> Result<String> {
    let data = match data {
        Some(d) if d > code.len() => {
            return Err(anyhow!(
                "--data 0x{:x} is past the end (0x{:x})",
                d,
                code.len()
            ))
        }
        Some(d) => d,
        None => disasm::data_start(arch, code)?,
    };
    let labels = if data < code.len() {
        vec![(data, "@data".to_string())]
    } else {
        Vec::new()
    };
    disasm::listing(arch, kind, code, &labels, mode)
}
