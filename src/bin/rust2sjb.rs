//! rust2sjb: compiles a `#[kernel]` function of a Rust file into a saved symjit application
//! (`.sjb`), which symjit loads (`load_func` in Python, `Application::load` in Rust) and the
//! other utilities compile (sjb2obj, sjb2wasm). See `docs/RUST.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use symjit::{CompilerType, Storage};
use symjit_utils::source::{builtin_config, Source};

const USAGE: &str = "\
Usage: rust2sjb [OPTIONS] <INPUT.rs>

Compiles a `#[kernel]` function of a Rust file into a saved symjit application
(.sjb), without a Rust compiler.

Options:
  -o, --output <FILE>   the saved application (default: <kernel>.sjb)
  --kernel <NAME>       the `#[kernel]` function to compile (default: the only one)
  --target <TARGET>     the machine the kernel is compiled for: native (default),
                        x86_64, aarch64, riscv64 or bytecode
  -O0, -O1, -O2, -O3    the opt level (default 2)
  -ffast-math, -fno-fast-math
                        fuse multiplications and additions, or not (default: fuse)
  --set <KEY=VALUE>     a further option of the configuration (symjit's `set_option`,
                        e.g. stable_math=true, cse=false); may be repeated
  -h, --help            print this help
  -V, --version         print the version";

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    kernel: Option<String>,
    target: Option<String>,
    opt_level: Option<u8>,
    fastmath: Option<bool>,
    options: Vec<(String, String)>,
}

enum Command {
    Help,
    Version,
    Run(Args),
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut a = Args {
        input: PathBuf::new(),
        output: None,
        kernel: None,
        target: None,
        opt_level: None,
        fastmath: None,
        options: Vec::new(),
    };
    let mut input: Option<PathBuf> = None;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        let mut value = |what: &str| -> Result<String, String> {
            args.next()
                .ok_or_else(|| format!("`{}` needs a value", what))
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-o" | "--output" => a.output = Some(value(&arg)?.into()),
            "--kernel" => a.kernel = Some(value(&arg)?),
            "--target" => a.target = Some(value(&arg)?),
            "-O0" | "-O1" | "-O2" | "-O3" => a.opt_level = Some(arg.as_bytes()[2] - b'0'),
            "-ffast-math" => a.fastmath = Some(true),
            "-fno-fast-math" => a.fastmath = Some(false),
            "--set" => {
                let kv = value(&arg)?;
                match kv.split_once('=') {
                    Some((k, v)) if !k.is_empty() => a.options.push((k.to_string(), v.to_string())),
                    _ => return Err(format!("`--set` takes KEY=VALUE, not `{}`", kv)),
                }
            }
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option `{}`", arg))
            }
            _ if input.is_none() => input = Some(arg.into()),
            _ => return Err(format!("unexpected argument `{}`", arg)),
        }
    }

    match input {
        Some(input) => {
            a.input = input;
            Ok(Command::Run(a))
        }
        None => Err("no input file".into()),
    }
}

fn target(name: &str) -> Result<CompilerType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "native" => CompilerType::Native,
        "x86_64" | "x86-64" | "amd64" | "x64" => CompilerType::Amd,
        "aarch64" | "arm64" => CompilerType::Arm,
        "riscv64" | "riscv" => CompilerType::RiscV,
        "bytecode" => CompilerType::ByteCode,
        _ => bail!(
            "unknown target `{}` (use native, x86_64, aarch64, riscv64 or bytecode)",
            name
        ),
    })
}

fn run(a: &Args) -> Result<()> {
    let text = std::fs::read_to_string(&a.input)
        .with_context(|| format!("cannot read `{}`", a.input.display()))?;
    let src = Source::parse(&a.input.display().to_string(), &text)?;
    let kernel = match &a.kernel {
        Some(k) => k.clone(),
        None => match src.kernels().as_slice() {
            [k] => k.to_string(),
            [] => bail!("`{}` has no `#[kernel]` function", a.input.display()),
            ks => bail!(
                "`{}` has several kernels ({}); choose one with --kernel",
                a.input.display(),
                ks.join(", ")
            ),
        },
    };

    let ty = match &a.target {
        Some(t) => target(t)?,
        None => CompilerType::Native,
    };
    let mut config = builtin_config(ty)?;
    if let Some(level) = a.opt_level {
        config.set_opt_level(level);
    }
    if let Some(fastmath) = a.fastmath {
        config.set_fastmath(fastmath);
    }
    for (key, value) in a.options.iter() {
        config
            .set_option(key, value)
            .map_err(|e| anyhow!("--set {}={}: {}", key, value, e))?;
    }

    let app = src.compile(&kernel, config)?;
    let mut bytes = Vec::new();
    app.save(&mut bytes)?;

    let output = match &a.output {
        Some(output) => output.clone(),
        None => PathBuf::from(format!("{}.sjb", kernel)),
    };
    std::fs::write(&output, bytes)
        .with_context(|| format!("cannot write `{}`", output.display()))?;
    Ok(())
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("rust2sjb: error: {}", msg);
            eprintln!("Try `rust2sjb --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("rust2sjb {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("rust2sjb: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}
