//! sjbinfo: prints what a saved symjit application (`.sjb`) holds: its configuration, the
//! model's sizes, its MIR, the size of the machine code compiled from it and the math functions
//! it calls. See `docs/INFO.md`.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use symjit_utils::info::Options;

const USAGE: &str = "\
Usage: sjbinfo [OPTIONS] <INPUT.sjb>...

Prints what a saved symjit application (`save` of the Python package) holds:
its configuration, the model's sizes, its MIR, the size of the machine code
symjit compiles from it (for the machine it was saved for; never run) and the
math functions it calls.

Options:
  -c, --constants   also list the constant pool (value and bits of each constant)
      --no-compile  do not compile the application (no code sizes)
  -h, --help        print this help
  -V, --version     print the version";

struct Args {
    inputs: Vec<PathBuf>,
    options: Options,
}

enum Command {
    Help,
    Version,
    Run(Args),
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut options = Options::default();

    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-c" | "--constants" => options.constants = true,
            "--no-compile" => options.compile = false,
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option `{}`", arg))
            }
            _ => inputs.push(arg.into()),
        }
    }

    if inputs.is_empty() {
        Err("no input file".into())
    } else {
        Ok(Command::Run(Args { inputs, options }))
    }
}

fn info(input: &PathBuf, options: &Options) -> Result<String> {
    let bytes =
        std::fs::read(input).with_context(|| format!("cannot open `{}`", input.display()))?;
    symjit_utils::info::describe(&input.display().to_string(), &bytes, options)
        .with_context(|| format!("cannot read `{}`", input.display()))
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjbinfo: error: {}", msg);
            eprintln!("Try `sjbinfo --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjbinfo {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(args)) => {
            let mut status = ExitCode::SUCCESS;
            let mut first = true;
            for input in args.inputs.iter() {
                match info(input, &args.options) {
                    Ok(text) => {
                        // a blank line between the reports
                        if !first {
                            println!();
                        }
                        first = false;
                        print!("{}", text);
                    }
                    Err(err) => {
                        eprintln!("sjbinfo: error: {:#}", err);
                        status = ExitCode::from(1);
                    }
                }
            }
            status
        }
    }
}
