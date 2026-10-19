//! sjb2obj: compiles a saved symjit application (`.sjb`) into a relocatable object file and
//! its C header, to be linked into C/C++ programs. See `docs/OBJ.md`.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use symjit_utils::SavedModel;

const USAGE: &str = "\
Usage: sjb2obj [OPTIONS] <INPUT.sjb>

Compiles a saved symjit application (`save` of the Python package) into a
relocatable object file and its C header, for the architecture it was compiled
for (x86-64 or AArch64; `ty=\"native\"` is this machine).

Options:
  -o, --output <FILE>  the object file (default: INPUT with the extension .o)
  --name <IDENT>       the name of the kernel functions (default: the object file's
                       stem; a C identifier)
  --format <FORMAT>    elf | macho (default: this system's)
  --header <FILE>      the C header (default: the object file with the extension .h)
  --no-header          do not write the C header
  -h, --help           print this help
  -V, --version        print the version";

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    name: Option<String>,
    format: Option<String>,
    header: Option<PathBuf>,
    no_header: bool,
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
        format: None,
        header: None,
        no_header: false,
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
            "--format" => a.format = Some(value(&arg)?),
            "--header" => a.header = Some(value(&arg)?.into()),
            "--no-header" => a.no_header = true,
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

fn run(a: &Args) -> Result<()> {
    let file =
        File::open(&a.input).with_context(|| format!("cannot open `{}`", a.input.display()))?;
    let model = SavedModel::read(&mut BufReader::new(file))
        .with_context(|| format!("cannot read `{}`", a.input.display()))?;

    let output = match &a.output {
        Some(output) => output.clone(),
        None => a.input.with_extension("o"),
    };
    let symbol = match &a.name {
        Some(name) => name.clone(),
        None => stem(&output)?,
    };
    if !is_identifier(&symbol) {
        bail!(
            "the kernel name `{}` is not a C identifier (use --name)",
            symbol
        );
    }
    let header = match &a.header {
        Some(header) => Some(header.clone()),
        None if a.no_header => None,
        None => Some(output.with_extension("h")),
    };

    let object_file = file_name(&output)?;
    let header_file = match &header {
        Some(h) => file_name(h)?,
        None => format!("{}.h", symbol),
    };
    let format = a.format.as_deref().unwrap_or("");
    let (bytes, text) = model.object_files(&symbol, format, &object_file, &header_file)?;

    std::fs::write(&output, bytes)
        .with_context(|| format!("cannot write `{}`", output.display()))?;
    if let Some(header) = header {
        std::fs::write(&header, text)
            .with_context(|| format!("cannot write `{}`", header.display()))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjb2obj: error: {}", msg);
            eprintln!("Try `sjb2obj --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjb2obj {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("sjb2obj: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}
