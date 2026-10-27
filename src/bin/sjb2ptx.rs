//! sjb2ptx: compiles a saved symjit application (`.sjb`) into a CUDA PTX module (one GPU
//! thread per point) and its C header. See `docs/PTX.md`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use symjit_utils::ptx::{driver_header, driver_source, header, Layout, PtxOptions, DEFAULT_SM};
use symjit_utils::SavedModel;

const USAGE: &str = "\
Usage: sjb2ptx [OPTIONS] <INPUT.sjb>

Compiles a saved symjit application (`save` of the Python package) into a CUDA
PTX module, one GPU thread per evaluation point, and its C header. The module
needs only the NVIDIA driver, which compiles it for the GPU at hand. With
--driver, also writes NAME_cuda.c and NAME_cuda.h (next to the module): C code
that embeds the module and runs it (`cc prog.c NAME_cuda.c -ldl`).

Options:
  -o, --output <FILE>  the PTX module (default: INPUT with the extension .ptx)
  --name <IDENT>       the kernel's name (default: the module's stem; a C identifier)
  --layout <LAYOUT>    soa | aos: slot k of point i at in[k*ld + i] (soa, default)
                       or in[i*ld + k] (aos)
  --sm <N>             the PTX target sm_N (default: 70)
  --header <FILE>      the C header (default: the module with the extension .h)
  --no-header          do not write the C header
  --driver             also write the C driver NAME_cuda.c and NAME_cuda.h
  -h, --help           print this help
  -V, --version        print the version";

/// Above this many instructions the CUDA driver takes long to compile a kernel (the probes of
/// plans/cuda-2.29: 0.4 s for 10k, 28 s for 100k); splitting into stages is planned.
const LARGE_KERNEL: usize = 30_000;

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    name: Option<String>,
    layout: Layout,
    sm: u32,
    header: Option<PathBuf>,
    no_header: bool,
    driver: bool,
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
        name: None,
        layout: Layout::Soa,
        sm: DEFAULT_SM,
        header: None,
        no_header: false,
        driver: false,
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
            "--name" => a.name = Some(value(&arg)?),
            "--layout" => {
                a.layout = match value(&arg)?.as_str() {
                    "soa" => Layout::Soa,
                    "aos" => Layout::Aos,
                    other => {
                        return Err(format!("unknown layout `{}` (expected soa or aos)", other))
                    }
                }
            }
            "--sm" => {
                let v = value(&arg)?;
                a.sm = match v.parse::<u32>() {
                    Ok(sm) if sm >= 30 => sm,
                    _ => return Err(format!("`--sm {}`: expected a number such as 70", v)),
                }
            }
            "--header" => a.header = Some(value(&arg)?.into()),
            "--no-header" => a.no_header = true,
            "--driver" => a.driver = true,
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option `{}`", arg))
            }
            _ if input.is_none() => input = Some(arg.into()),
            _ => return Err(format!("unexpected argument `{}`", arg)),
        }
    }

    if a.no_header && a.header.is_some() {
        return Err("--no-header and --header exclude each other".into());
    }
    match input {
        Some(input) => {
            a.input = input;
            Ok(Command::Run(a))
        }
        None => Err("no input file".into()),
    }
}

fn file_name(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("`{}` does not end in a file name", path.display()))
}

fn stem(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("`{}` does not end in a file name", path.display()))
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The instructions of a module (the lines inside the kernel and the math library's functions,
/// without register declarations and labels).
fn count_instructions(text: &str) -> usize {
    text.lines()
        .filter(|l| l.starts_with('\t') && l.ends_with(';') && !l.starts_with("\t.reg"))
        .count()
}

fn run(a: &Args) -> Result<()> {
    let file =
        File::open(&a.input).with_context(|| format!("cannot open `{}`", a.input.display()))?;
    let mut model = SavedModel::read(&mut BufReader::new(file))
        .with_context(|| format!("cannot read `{}`", a.input.display()))?;

    let output = match &a.output {
        Some(output) => output.clone(),
        None => a.input.with_extension("ptx"),
    };
    let name = match &a.name {
        Some(name) => name.clone(),
        None => stem(&output)?,
    };
    if !is_identifier(&name) {
        bail!(
            "the kernel name `{}` is not a C identifier (use --name)",
            name
        );
    }
    let header_path = match &a.header {
        Some(header) => Some(header.clone()),
        None if a.no_header => None,
        None => Some(output.with_extension("h")),
    };

    let options = PtxOptions {
        name,
        layout: a.layout,
        sm: a.sm,
        mapping: None,
    };
    let ptx = model.ptx(&options)?;

    let count = count_instructions(&ptx.text);
    if count > LARGE_KERNEL {
        eprintln!(
            "sjb2ptx: warning: the kernel has {} PTX instructions; the CUDA driver may take \
             minutes to compile it",
            count
        );
    }

    std::fs::write(&output, &ptx.text)
        .with_context(|| format!("cannot write `{}`", output.display()))?;
    if a.driver {
        let dir = output.parent().unwrap_or(Path::new(""));
        let complex = model.config().is_complex();
        for (file, text) in [
            (
                format!("{}_cuda.c", options.name),
                driver_source(&options, &ptx.shape, &ptx.text),
            ),
            (
                format!("{}_cuda.h", options.name),
                driver_header(&options, &ptx.shape, complex),
            ),
        ] {
            let path = dir.join(file);
            std::fs::write(&path, text)
                .with_context(|| format!("cannot write `{}`", path.display()))?;
        }
    }
    if let Some(path) = header_path {
        let text = header(
            &options,
            &ptx.shape,
            model.config().is_complex(),
            &file_name(&output)?,
            &file_name(&path)?,
        );
        std::fs::write(&path, text)
            .with_context(|| format!("cannot write `{}`", path.display()))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjb2ptx: error: {}", msg);
            eprintln!("Try `sjb2ptx --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjb2ptx {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("sjb2ptx: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}
