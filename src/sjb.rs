//! Saved symjit applications (`.sjb`, written by `save` of the Python package or `Storage::save`
//! of the Rust crate), read for the code generators of this crate.

use std::collections::HashSet;
use std::io::Read;

use anyhow::{anyhow, Result};
use symjit::backend::{Complexifier, Generator, Loc, Mir, MirInstruction};
use symjit::{Application, CompilerType, Config, Program};

use crate::wasm::WasmGenerator;

/// A saved application: its program and its real MIR (lowered by `Complexifier` for a complex
/// model, as the JIT runs it on its real generators and the bytecode interpreter).
pub struct SavedModel {
    pub prog: Program,
    pub mir: Mir,
    /// the MIR as saved (complex operations not lowered yet)
    pub saved: Mir,
    pub reals: HashSet<Loc>,
}

impl SavedModel {
    /// Reads a saved application. Nothing is compiled, so the file may come from any machine:
    /// its MIR is register-allocated for the machine it was compiled for, which matters only
    /// to native code generators (a WebAssembly module keeps registers in locals).
    pub fn read(stream: &mut impl Read) -> Result<SavedModel> {
        // the saved configuration replaces this one, except for its (empty) functions
        let config = Config::new(CompilerType::ByteCode, 0)?;
        let (mut prog, reals, saved) = Application::read_sjb(stream, &config)?;
        check_calls(&saved)?;

        let mir = if prog.config().is_complex() {
            let mir = Complexifier::new(&reals, prog.config().clone()).complexify(&saved)?;
            // `stable_math` appends constants (as `Application::with_mir` copies them)
            if mir.consts.len() > prog.builder.consts.len() {
                prog.builder.consts = mir.consts.clone();
            }
            mir
        } else {
            saved.clone()
        };

        Ok(SavedModel {
            prog,
            mir,
            saved,
            reals,
        })
    }

    pub fn config(&self) -> &Config {
        self.prog.config()
    }

    /// The WebAssembly module of the model (see `crate::wasm` for its interface).
    pub fn wasm(&mut self) -> Result<Vec<u8>> {
        let prog = &mut self.prog;
        let mut generator = WasmGenerator::new(prog.config().clone());
        generator.set_mem_size(prog.mem_size());
        if prog.count_params == 0 && prog.count_obs == 1 && prog.count_diffs == 0 {
            generator.export_fast(prog.mem_size());
        }
        let (count_states, count_obs, count_params) =
            (prog.count_states, prog.count_obs, prog.count_params);
        prog.builder.compile_from_mir(
            &self.mir,
            &mut generator,
            count_states,
            count_obs,
            count_params,
        )?;
        generator.check()?;
        Ok(generator.bytes())
    }
}

// the functions a saved model calls are math functions: a `.sjb` file does not hold the
// user-defined (`defuns`) and recursive functions of the model it was saved from
fn check_calls(mir: &Mir) -> Result<()> {
    for ins in mir.code.iter() {
        if let MirInstruction::Call { label, num_args } = &ins {
            if *num_args > 0 && mir.find_op(label).is_err() {
                return Err(anyhow!(
                    "the model calls `{}`, a user-defined or recursive function, which a \
                     saved application does not hold",
                    label
                ));
            }
        }
    }
    Ok(())
}
