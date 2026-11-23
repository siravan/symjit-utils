//! sjb2asm: a kernel of a saved symjit application (`.sjb` or "symjit-application" JSON), compiled
//! by symjit for a chosen architecture, as a listing with the address and the bytes of each
//! instruction, or as assembly text that GNU `as` assembles back to the same bytes. See
//! `docs/ASM.md`.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use symjit_utils::asm::{self, Kernel, Options};
use symjit_utils::disasm::Mode;

const USAGE: &str = "\
Usage: sjb2asm [OPTIONS] <INPUT.sjb | INPUT.json>
       sjb2asm --raw --arch ARCH [OPTIONS] <KERNEL.bin>

Compiles a saved symjit application (`save`, or `dump(.., \"json\")` of the
Python package) with symjit for ARCH and lists one of its kernels: the address,
the bytes and the instruction of each line, the data area as hex values; the
code is never run. With --asm, the kernel is written as assembly text that GNU
`as` assembles back to the same bytes.
With --raw, the input is the bytes of a kernel (`dump(.., \"scalar\")`), which
have no labels: where the data area starts is found from the code.

Options:
  -a, --arch ARCH     x86_64, aarch64, riscv64 (or symjit's ty: amd-avx,
                      amd-sse, native); default: the file's own target
  -k, --kernel KIND   scalar (default), simd or fast
  -s, --asm           assembler input (symbolic labels, no addresses or
                      bytes) instead of the listing
  -x, --hex           the listing (the default)
  -o, --output FILE   write to FILE instead of standard output
      --raw           the input is a kernel's bytes (needs --arch)
      --data OFFSET   with --raw: where the data area starts (decimal or 0x..)
  -h, --help          print this help
  -V, --version       print the version

x86-64 kernels use the features of the host CPU (AVX, AVX-512) where it has
them; amd-sse gives the SSE kernels anywhere.";

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    arch: Option<String>,
    options: Options,
    raw: bool,
    data: Option<usize>,
}

enum Command {
    Help,
    Version,
    Run(Args),
}

fn number(s: &str) -> Result<usize, String> {
    let n = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => usize::from_str_radix(h, 16),
        None => s.parse(),
    };
    n.map_err(|_| format!("`{}` is not an offset", s))
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut input: Option<PathBuf> = None;
    let mut output = None;
    let mut arch = None;
    let mut options = Options::default();
    let mut raw = false;
    let mut data = None;

    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("`{}` needs a value", name))
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-a" | "--arch" => arch = Some(value(&arg)?),
            "-k" | "--kernel" => {
                options.kernel = Kernel::parse(&value(&arg)?).map_err(|e| e.to_string())?
            }
            "-s" | "--asm" => options.mode = Mode::Asm,
            "-x" | "--hex" => options.mode = Mode::Hex,
            "-o" | "--output" => output = Some(PathBuf::from(value(&arg)?)),
            "--raw" => raw = true,
            "--data" => data = Some(number(&value(&arg)?)?),
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option `{}`", arg))
            }
            _ if input.is_some() => return Err("more than one input file".into()),
            _ => input = Some(arg.into()),
        }
    }

    let input = input.ok_or("no input file")?;
    if raw && arch.is_none() {
        return Err("--raw needs --arch".into());
    }
    if data.is_some() && !raw {
        return Err("--data is for --raw input".into());
    }
    if let Some(a) = &arch {
        options.ty = Some(asm::ty_of_arch(a).map_err(|e| e.to_string())?);
    }

    Ok(Command::Run(Args {
        input,
        output,
        arch,
        options,
        raw,
        data,
    }))
}

fn run(args: &Args) -> Result<()> {
    let bytes = std::fs::read(&args.input)
        .with_context(|| format!("cannot open `{}`", args.input.display()))?;

    let text = if args.raw {
        let arch = asm::arch_name(args.arch.as_deref().ok_or_else(|| anyhow!("no --arch"))?)?;
        asm::disassemble_raw(
            &bytes,
            arch,
            args.options.kernel.name(),
            args.data,
            args.options.mode,
        )?
    } else {
        asm::disassemble(&bytes, &args.options)
            .with_context(|| format!("cannot disassemble `{}`", args.input.display()))?
    };

    match &args.output {
        Some(path) => std::fs::write(path, text)
            .with_context(|| format!("cannot write `{}`", path.display()))?,
        // a closed pipe (`| head`) is not an error
        None => match std::io::stdout().write_all(text.as_bytes()) {
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {}
            r => r.context("cannot write the listing")?,
        },
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjb2asm: error: {}", msg);
            eprintln!("Try `sjb2asm --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjb2asm {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(args)) => match run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("sjb2asm: error: {:#}", e);
                ExitCode::FAILURE
            }
        },
    }
}
