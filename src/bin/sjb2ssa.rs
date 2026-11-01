//! sjb2ssa: the SSA form of a saved symjit application (`.sjb`), as an `.ssa` file for ssa2ptx.
//! See `docs/SSA.md`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{anyhow, Context, Result};
use symjit_utils::ssa::SsaModel;
use symjit_utils::SavedModel;

const USAGE: &str = "\
Usage: sjb2ssa [OPTIONS] <INPUT.sjb>

Converts the MIR of a saved symjit application into SSA form: each definition
of a general register gets a register of its own (symjit's Static registers,
as the first pass of its register allocator makes them). A complex model is
lowered to real operations first. The result is written as an .ssa file, for
ssa2ptx; symjit cannot run it (its code generators take allocated registers).

Options:
  -o, --output <FILE>  the .ssa file (default: INPUT with the extension .ssa)
  --set <KEY=VALUE>    an option of the saved configuration (symjit's
                       `set_option`, e.g. stable_math=true), set before the
                       complex operations are lowered; may be repeated
  -q, --quiet          do not print what the conversion did
  -h, --help           print this help
  -V, --version        print the version";

struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    settings: Vec<(String, String)>,
    quiet: bool,
}

enum Command {
    Help,
    Version,
    Run(Args),
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut input: Option<PathBuf> = None;
    let mut output = None;
    let mut settings = Vec::new();
    let mut quiet = false;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("`{}` needs a value", name))
        };
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-o" | "--output" => output = Some(PathBuf::from(value(&arg)?)),
            "--set" => {
                let v = value(&arg)?;
                let (key, val) = v
                    .split_once('=')
                    .ok_or_else(|| format!("`--set {}`: expected KEY=VALUE", v))?;
                settings.push((key.to_string(), val.to_string()));
            }
            "-q" | "--quiet" => quiet = true,
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option `{}`", arg))
            }
            _ if input.is_none() => input = Some(arg.into()),
            _ => return Err(format!("unexpected argument `{}`", arg)),
        }
    }
    let input = input.ok_or("no input file")?;
    Ok(Command::Run(Args {
        input,
        output,
        settings,
        quiet,
    }))
}

fn run(a: &Args) -> Result<()> {
    let file =
        File::open(&a.input).with_context(|| format!("cannot open `{}`", a.input.display()))?;
    let model = SavedModel::read_with(&mut BufReader::new(file), &a.settings)
        .with_context(|| format!("cannot read `{}`", a.input.display()))?;
    let ssa = SsaModel::from_saved(model)
        .with_context(|| format!("cannot convert `{}`", a.input.display()))?;

    let output = a
        .output
        .clone()
        .unwrap_or_else(|| a.input.with_extension("ssa"));
    if output == a.input {
        return Err(anyhow!("the output would overwrite the input (use -o)"));
    }
    let file =
        File::create(&output).with_context(|| format!("cannot write `{}`", output.display()))?;
    let mut w = BufWriter::new(file);
    ssa.write(&mut w)?;
    w.flush()
        .with_context(|| format!("cannot write `{}`", output.display()))?;
    if !a.quiet {
        let s = &ssa.stats;
        let pinned = if s.pinned.is_empty() {
            "none".to_string()
        } else {
            s.pinned
                .iter()
                .map(|r| format!("%{}", r))
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!(
            "sjb2ssa: {}: {} instructions, {} static registers; {} definitions kept in the registers that cross labels or calls ({}){}",
            output.display(),
            s.instructions,
            s.statics,
            s.kept,
            pinned,
            if ssa.config().is_complex() {
                "; complex model lowered"
            } else {
                ""
            }
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)) {
        Err(msg) => {
            eprintln!("sjb2ssa: error: {}", msg);
            eprintln!("Try `sjb2ssa --help`.");
            ExitCode::from(2)
        }
        Ok(Command::Help) => {
            println!("{}", USAGE);
            ExitCode::SUCCESS
        }
        Ok(Command::Version) => {
            println!("sjb2ssa {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Command::Run(a)) => match run(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("sjb2ssa: error: {:#}", err);
                ExitCode::from(1)
            }
        },
    }
}
