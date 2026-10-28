//! Object-file kernels: copies of symjit's x86-64 and AArch64 scalar and packed-complex
//! generators and of its complexifier, in object mode. Calls of math functions are direct
//! calls of the C math library, recorded as relocations (`relocations`), instead of calls
//! through a table of the addresses of symjit's own functions; AArch64 constants are loaded
//! PC-relative (`ldr d, label`, the JIT's `adrp` assumes page-aligned code); complex functions
//! are C99's (`<complex.h>`), and the ones C does not have are composed of C functions
//! (`complexify`, `lower`).
//!
//! The files are stable's (`rust/amd`, `rust/arm`, `rust/assembler.rs`, `rust/complexify.rs`) with
//! the object mode of dev's `v228-0004`/`0028` patches, so that the kernels compute the bits of
//! the JIT's kernels. The modules below stand in for the paths of stable's crate they use.

// copies of stable's files, kept close to them: the helpers they share with the SIMD
// generators (not copied) stay
#[allow(dead_code)]
pub(crate) mod amd;
#[allow(dead_code)]
pub(crate) mod arm;
#[allow(dead_code)]
pub(crate) mod assembler;
#[allow(dead_code)]
pub(crate) mod complexify;
pub(crate) mod lower;

use anyhow::{anyhow, Result};

use symjit::backend::{Generator, Mir, MirInstruction};
use symjit::{Config, Program};

use crate::linker::{header, Arch, Format, HeaderInfo, ObjectBuilder, Relocation, Target};
use crate::sjb::SavedModel;
use amd::{AmdComplexGenerator, AmdScalarGenerator, AmdVectorF64x4Generator};
use arm::{ArmComplexGenerator, ArmGenerator};
use complexify::Complexifier;
use lower::lower_for_object;

// the reach of AArch64 PC-relative literal loads (`ldr d, label`, +-1 MB)
const ARM_OBJECT_LIMIT: usize = 1 << 20;

impl SavedModel {
    /// The object format named `format` ("" for the host's, "elf" or "macho") and the
    /// architecture the model was compiled for (x86-64, or AArch64, e.g. with `ty = "arm"`;
    /// "native" is the machine that saved it, which is taken to be this one).
    pub fn object_target(&self, format: &str) -> Result<Target> {
        let format = match format {
            "" if cfg!(target_os = "macos") => Format::MachO,
            "" if cfg!(target_os = "linux") => Format::Elf,
            "" => {
                return Err(anyhow!(
                    "name the object format (\"elf\" or \"macho\") on this system"
                ))
            }
            "elf" => Format::Elf,
            "macho" => Format::MachO,
            _ => {
                return Err(anyhow!(
                    "unknown object format `{}` (use \"elf\" or \"macho\")",
                    format
                ))
            }
        };
        let config = self.config();
        let arch = if config.is_amd64() {
            Arch::X86_64
        } else if config.is_arm64() {
            Arch::Aarch64
        } else {
            return Err(anyhow!(
                "object files hold x86-64 or AArch64 kernels (the model is compiled for {:?})",
                config.compiler_type()
            ));
        };
        Ok(Target::new(format, arch))
    }

    /// Whether the model has a fast kernel (`<name>_fast(x0, ...)`), as the JIT decides.
    pub fn can_fast(&self) -> bool {
        let (config, prog) = (self.config(), &self.prog);
        config.may_fast()
            && (prog.count_states <= 4 || (prog.count_states <= 8 && !config.is_window()))
            && prog.count_params == 0
            && prog.count_obs == 1
            && prog.count_diffs == 0
    }

    /// The kernels of the model as functions of a relocatable object file: `name`, the
    /// scalar kernel (with the native kernels' calling convention,
    /// `int name(double *mem, const void *states, size_t idx, const double *params)`), and
    /// `<name>_fast` (`double name_fast(double x0, ...)`) for models that have a fast kernel.
    /// Calls of math functions become relocations against the C math library.
    pub fn object(&self, name: &str, target: Target) -> Result<ObjectBuilder> {
        let mut config = self.config().clone();
        let idx_ret = self.can_fast().then_some(self.prog.count_states as i32);
        let mut obj = ObjectBuilder::new(target);

        // the code of the object, which calls only the C library: a complex model that calls
        // complex functions is compiled through `complexify` (as with `fast_complex=false`),
        // whose calls go to C99's complex functions, and the real functions C does not have
        // are composed of C functions (`lower_for_object`); the program gets their constants
        // and stack slots
        let mut prog = self.prog.clone();
        let original = &self.saved;
        let complex_calls = config.is_complex() && calls_complex_functions(original, &config);
        if complex_calls {
            config.set_fast_complex(false);
        }
        let complex_fast = config.is_complex() && config.fast_complex();
        let mir = if complex_calls {
            let scratch = prog.builder.stack_size();
            prog.builder.count_stack = Some(scratch + 2);
            let mir = Complexifier::for_object(&self.reals, config.clone(), scratch as u32)
                .complexify(original)?;
            if mir.consts.len() > prog.builder.consts.len() {
                prog.builder.consts = mir.consts.clone();
            }
            mir
        } else if config.is_complex() {
            self.mir.clone()
        } else {
            lower_for_object(&self.mir, &mut prog)?
        };
        let prog = &mut prog;

        // the kernels, compiled as the JIT's `compile_ty` and `compile_*_fast` do
        let (scalar, fast) = match target.arch {
            Arch::X86_64 => {
                let scalar = if complex_fast {
                    let mut g = AmdComplexGenerator::new(config.clone());
                    compile(original, prog, &mut g, None)?;
                    (g.bytes(), relocations(g.relocations()))
                } else {
                    let mut g = AmdScalarGenerator::new(config.clone());
                    compile(&mir, prog, &mut g, None)?;
                    (g.bytes(), relocations(g.relocations()))
                };
                let fast = match idx_ret {
                    Some(_) => {
                        let mut g = AmdScalarGenerator::new(config.clone());
                        compile(&mir, prog, &mut g, idx_ret)?;
                        Some((g.bytes(), relocations(g.relocations())))
                    }
                    None => None,
                };
                (scalar, fast)
            }
            Arch::Aarch64 => {
                let scalar = if complex_fast {
                    let mut g = ArmComplexGenerator::new(config.clone());
                    compile(original, prog, &mut g, None)?;
                    (g.bytes(), relocations(g.relocations()))
                } else {
                    let mut g = ArmGenerator::new(config.clone());
                    compile(&mir, prog, &mut g, None)?;
                    (g.bytes(), relocations(g.relocations()))
                };
                let fast = match idx_ret {
                    Some(_) => {
                        let mut g = ArmGenerator::new(config.clone());
                        compile(&mir, prog, &mut g, idx_ret)?;
                        Some((g.bytes(), relocations(g.relocations())))
                    }
                    None => None,
                };
                // constants are loaded with PC-relative literal loads, which reach 1 MB
                let size = scalar.0.len().max(fast.as_ref().map_or(0, |f| f.0.len()));
                if size >= ARM_OBJECT_LIMIT {
                    return Err(anyhow!(
                        "AArch64 object files hold kernels up to {} bytes (this one is {})",
                        ARM_OBJECT_LIMIT,
                        size
                    ));
                }
                (scalar, fast)
            }
        };

        obj.add_function(name, &scalar.0, &scalar.1)?;
        if let Some((code, relocs)) = fast {
            obj.add_function(&format!("{}_fast", name), &code, &relocs)?;
        }
        Ok(obj)
    }

    /// The AVX2 SIMD kernel of the model (4 lanes) as the function `name` of a relocatable
    /// object file, with the scalar kernel's calling convention: a call computes the points
    /// `idx .. idx + 4`. `simd_branch` is on, so lanes that take different branches are merged
    /// and the kernel never asks for the scalar kernel. x86-64 only. As the JIT does, a complex
    /// model is compiled from its complexified (real) MIR; one that calls complex functions is
    /// an error.
    pub fn object_simd(&self, name: &str, target: Target) -> Result<ObjectBuilder> {
        if target.arch != Arch::X86_64 {
            return Err(anyhow!("SIMD object kernels are x86-64 (AVX2) only"));
        }
        let mut config = self.config().clone();
        config.set_simd_branch(true);
        let mut prog = self.prog.clone();
        let mir = if config.is_complex() {
            self.mir.clone()
        } else {
            lower_for_object(&self.mir, &mut prog)?
        };
        let mut g = AmdVectorF64x4Generator::new(config);
        compile(&mir, &mut prog, &mut g, None)?;
        let mut obj = ObjectBuilder::new(target);
        obj.add_function(name, &g.bytes(), &relocations(g.relocations()))?;
        Ok(obj)
    }

    /// The object file (in `format`, see `object_target`) and its C header: the kernels are
    /// named `symbol`, and the header names its files `object_file` and `header_file` (e.g.
    /// "model.o", "model.h").
    pub fn object_files(
        &self,
        symbol: &str,
        format: &str,
        object_file: &str,
        header_file: &str,
    ) -> Result<(Vec<u8>, String)> {
        let target = self.object_target(format)?;
        let obj = self.object(symbol, target)?;
        let fast = obj.functions.len() > 1;
        let imports = obj.undefined_symbols();
        let bytes = obj.build()?;

        let config = self.config();
        let cpu = match target.arch {
            Arch::X86_64 => "AVX and FMA (x86-64-v3)",
            // packed complex products use fcmla where the compiling CPU has FCMA
            // (not for models compiled through `complexify`, see `object`)
            Arch::Aarch64
                if config.is_complex()
                    && config.fast_complex()
                    && !calls_complex_functions(&self.saved, config)
                    && host_has_fcma() =>
            {
                "ARMv8.3-A (with FCMA)"
            }
            Arch::Aarch64 => "ARMv8-A",
        };
        let prog = &self.prog;
        let text = header(&HeaderInfo {
            name: symbol,
            target,
            count_states: prog.count_states,
            count_obs: prog.count_obs,
            count_params: prog.count_params,
            count_diffs: prog.count_diffs,
            mem_size: prog.mem_size(),
            complex: config.is_complex(),
            fast,
            symbolica: config.symbolica(),
            direct_arena: config.direct_arena(),
            cpu,
            imports: &imports,
            functions: &[],
            external_kernels: &[],
            included_kernels: &[],
            version: env!("CARGO_PKG_VERSION"),
            object_file,
            header_file,
        });

        Ok((bytes, text))
    }
}

/// Whether the complex model `mir` calls complex functions (not the model itself).
fn calls_complex_functions(mir: &Mir, config: &Config) -> bool {
    mir.code.iter().any(|ins| {
        matches!(&ins, MirInstruction::Call { label, num_args } if *num_args > 0
            && !label.starts_with('@')
            && !config.is_external_func(label))
    })
}

// compiles `mir` into `generator`, the fast kernel if `idx_ret` is given
fn compile(
    mir: &Mir,
    prog: &mut Program,
    generator: &mut impl Generator,
    idx_ret: Option<i32>,
) -> Result<()> {
    match idx_ret {
        None => prog.builder.compile_from_mir(
            mir,
            generator,
            prog.count_states,
            prog.count_obs,
            prog.count_params,
        ),
        Some(idx) => prog.builder.compile_fast_from_mir(
            mir,
            generator,
            prog.count_states,
            prog.count_obs,
            idx,
        ),
    }
}

fn relocations(relocs: Vec<(usize, String)>) -> Vec<Relocation> {
    relocs
        .iter()
        .map(|(at, symbol)| Relocation::call(*at, symbol))
        .collect()
}

// whether the compiling CPU has FCMA (used by the packed-complex ARM generator)
fn host_has_fcma() -> bool {
    #[cfg(target_arch = "aarch64")]
    return std::arch::is_aarch64_feature_detected!("fcma");
    #[cfg(not(target_arch = "aarch64"))]
    return false;
}

pub(crate) mod config {
    pub use symjit::backend::ABI_AREA;
    pub use symjit::Config;

    /// `Config::kernel_type() == KernelType::RowFirst` (the Symbolica calling convention:
    /// inputs in `params`, outputs in `mem`), which symjit does not export.
    pub fn row_first(config: &Config) -> bool {
        config.symbolica() && !config.direct_arena()
    }
}

pub(crate) mod code {
    pub use symjit::backend::Func;
}

pub(crate) mod symbol {
    pub use symjit::backend::Loc;
}

pub(crate) mod mir {
    pub use symjit::backend::{Mir, MirInstruction as Instruction};
}

pub(crate) mod utils {
    pub use symjit::backend::{align_stack, Reg};

    #[derive(PartialEq)]
    #[allow(unused)]
    pub enum DataType {
        F32,
        F64,
    }
}

pub(crate) mod generator {
    use anyhow::{anyhow, Result};

    pub use symjit::backend::{Generator, GeneratorType, StackRegions};

    /// The C math library function computing `op` (symjit's `log` is the decimal logarithm and
    /// `ln` the natural one), as `VirtualTable::c_name` of stable.
    pub fn c_name(op: &str) -> Option<&'static str> {
        let name = match op {
            "sin" => "sin",
            "cos" => "cos",
            "tan" => "tan",
            "sinh" => "sinh",
            "cosh" => "cosh",
            "tanh" => "tanh",
            "arcsin" => "asin",
            "arccos" => "acos",
            "arctan" => "atan",
            "arcsinh" => "asinh",
            "arccosh" => "acosh",
            "arctanh" => "atanh",
            "exp" => "exp",
            "expm1" => "expm1",
            "exp2" => "exp2",
            "ln" => "log",
            "log" => "log10",
            "log1p" => "log1p",
            "log2" => "log2",
            "cbrt" => "cbrt",
            "erf" => "erf",
            "erfc" => "erfc",
            "gamma" => "tgamma",
            "loggamma" => "lgamma",
            "power" => "pow",
            "atan2" => "atan2",
            _ => return None,
        };
        Some(name)
    }

    /// The C library symbol a call of the real function `op` goes to, or the error to report.
    pub fn object_symbol(op: &str) -> Result<&'static str> {
        c_name(op).ok_or_else(|| {
            anyhow!(
                "`{}` is not a C math library function, so it cannot be called from an object file",
                op
            )
        })
    }

    /// The C99 complex function (`<complex.h>`, in the C math library) a call of the complex
    /// `op` goes to. A `double complex` argument and result are in the first two
    /// floating-point registers (x86-64 System V and AArch64), which are `Ret` and `Temp`, and
    /// the second argument of `cpow` in the next two, `Gen(0)` and `Gen(1)` (x86-64: xmm2 and
    /// xmm3, AArch64: d2 and d3); `carg` returns a `double` in `Ret`. The complex functions
    /// C99 does not have (`csc`, ..., `log10`, `cbrt`, `sinc`) are composed of these by
    /// `complexify`. symjit's complex `log` is the natural logarithm, as `ln`.
    pub fn complex_c_name(op: &str) -> Option<&'static str> {
        Some(match op {
            "sin" => "csin",
            "cos" => "ccos",
            "tan" => "ctan",
            "sinh" => "csinh",
            "cosh" => "ccosh",
            "tanh" => "ctanh",
            "arcsin" => "casin",
            "arccos" => "cacos",
            "arctan" => "catan",
            "arcsinh" => "casinh",
            "arccosh" => "cacosh",
            "arctanh" => "catanh",
            "exp" => "cexp",
            "ln" | "log" => "clog",
            "root" => "csqrt",
            "power" => "cpow",
            "arg" => "carg",
            _ => return None,
        })
    }

    /// The C99 symbol of the complex function `op`, or the error to report.
    pub fn object_complex_symbol(op: &str) -> Result<&'static str> {
        complex_c_name(op).ok_or_else(|| object_complex_error(op))
    }

    /// Calls of user-defined functions and recursive calls (a saved application holds neither).
    pub fn object_external_error(op: &str) -> anyhow::Error {
        anyhow!(
            "the user-defined or recursive function `{}` cannot be called from an object file",
            op
        )
    }

    pub fn object_complex_error(op: &str) -> anyhow::Error {
        anyhow!(
            "the complex function `{}` cannot be called from an object file",
            op
        )
    }
}
