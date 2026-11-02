//! sjb2msl: compiles a saved symjit application (`.sjb`) into a Metal Shading Language kernel
//! (one GPU thread per point, f32). A probe: straight-line kernels without math functions. See
//! `docs/METAL.md`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use symjit_utils::msl::{MslOptions, RUN_M};
use symjit_utils::ptx::Layout;
use symjit_utils::SavedModel;

const USAGE: &str = "\
Usage: sjb2msl [OPTIONS] <INPUT.sjb>

Compiles a saved symjit application into a Metal Shading Language kernel, one
GPU thread per evaluation point, in single precision (Apple GPUs have no f64).
The Metal framework compiles the source at run time; Xcode is not needed. A
probe: kernels of straight-line arithmetic (pyAmpliCol's plane kernels), no
control flow or math functions.

Options:
  -o, --output <FILE>  the source (default: INPUT with the extension .metal)
  --name <IDENT>       the kernel's name (default: the source's stem; a C identifier)
  --layout <LAYOUT>    soa | aos: slot k of point i at in[k*ld + i] (soa, default)
                       or in[i*ld + k] (aos)
  --host               also write run.m next to the source: a host program that
                       compiles and runs kernels (usage at its top)
  -h, --help           print this help
  -V, --version        print the version";

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    name: Option<String>,
    layout: Layout,
    host: bool,
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
        host: false,
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
            "--host" => a.host = true,
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

fn run(a: &Args) -> Result<()> {
    let file =
        File::open(&a.input).with_context(|| format!("cannot open `{}`", a.input.display()))?;
    let mut model = SavedModel::read(&mut BufReader::new(file))
        .with_context(|| format!("cannot read `{}`", a.input.display()))?;
    let output = match &a.output {
        Some(output) => output.clone(),
        None => a.input.with_extension("metal"),
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
    let mut options = MslOptions::new(&name);
    options.layout = a.layout;
    let msl = model.msl(&options)?;
    std::fs::write(&output, msl.source())
        .with_context(|| format!("cannot write `{}`", output.display()))?;
    if a.host {
        let path = output.parent().unwrap_or(Path::new("")).join("run.m");
        std::fs::write(&path, RUN_M)
            .with_context(|| format!("cannot write `{}`", path.display()))?;
    }
    println!(
        "sjb2msl: {}: kernel {}, {} inputs, {} outputs, {} parameters",
        output.display(),
        name,
        msl.shape.count_in,
        msl.shape.count_out,
        msl.shape.count_params
    );
    Ok(())
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjb2msl: error: {}", msg);
            eprintln!("Try `sjb2msl --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjb2msl {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("sjb2msl: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}
