//! sjb2wasm: compiles a saved symjit application (`.sjb`) into a WebAssembly module.
//! See `docs/WASM.md` for the module's interface.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use symjit_utils::SavedModel;

const USAGE: &str = "\
Usage: sjb2wasm [OPTIONS] <INPUT.sjb>

Compiles a saved symjit application (`save` of the Python package) into a
WebAssembly module.

Options:
  -o, --output <FILE>  the module to write (default: INPUT with the extension .wasm)
  -h, --help           print this help
  -V, --version        print the version";

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
}

enum Command {
    Help,
    Version,
    Run(Args),
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut args = args.peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-o" | "--output" => match args.next() {
                Some(file) if output.is_none() => output = Some(file.into()),
                Some(_) => return Err("`-o` is given more than once".into()),
                None => return Err(format!("`{}` needs a file name", arg)),
            },
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option `{}`", arg))
            }
            _ if input.is_none() => input = Some(arg.into()),
            _ => return Err(format!("unexpected argument `{}`", arg)),
        }
    }

    match input {
        Some(input) => Ok(Command::Run(Args { input, output })),
        None => Err("no input file".into()),
    }
}

fn run(args: &Args) -> Result<()> {
    let file = File::open(&args.input)
        .with_context(|| format!("cannot open `{}`", args.input.display()))?;
    let mut model = SavedModel::read(&mut BufReader::new(file))
        .with_context(|| format!("cannot read `{}`", args.input.display()))?;
    let module = model.wasm()?;

    let output = match &args.output {
        Some(output) => output.clone(),
        None => default_output(&args.input),
    };
    std::fs::write(&output, module)
        .with_context(|| format!("cannot write `{}`", output.display()))?;
    Ok(())
}

fn default_output(input: &Path) -> PathBuf {
    input.with_extension("wasm")
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjb2wasm: error: {}", msg);
            eprintln!("Try `sjb2wasm --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjb2wasm {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(args)) => match run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("sjb2wasm: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}
