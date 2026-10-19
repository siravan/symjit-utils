# sjb2obj: object files from saved functions

`sjb2obj` compiles a function saved by symjit (`f.save(...)` of the Python package, or
`Storage::save` of the Rust crate) into a relocatable object file and a C header, so that it can be
linked into C and C++ programs:

```python
from sympy import symbols, sin, exp
from symjit import compile_func

x, y = symbols("x y")
f = compile_func([x, y], [sin(x) * y + exp(-x * x)])
f.save("model.sjb")
```

```bash
sjb2obj model.sjb              # writes model.o and model.h
```

```c
#include <stdio.h>
#include "model.h"

int main(void) {
    double mem[MODEL_MEM_SIZE] = {0.5, 2.0};               /* the states, then the outputs */
    model(mem, NULL, 0, NULL);
    printf("%g %g\n", mem[MODEL_COUNT_STATES], model_fast(0.5, 2.0));
    return 0;
}
```

```bash
cc main.c model.o -lm -o main     # or c++; the header has extern "C"
```

The kernels are compiled by copies of symjit's own code generators, in object mode, so they give
the JIT's results bit for bit (except for complex functions, see below).

## Options

```text
sjb2obj [OPTIONS] <INPUT.sjb>
  -o, --output <FILE>  the object file (default: INPUT with the extension .o)
  --name <IDENT>       the name of the kernel functions (default: the object file's stem)
  --format <FORMAT>    elf | macho (default: this system's, ELF on Linux, Mach-O on macOS)
  --header <FILE>      the C header (default: the object file with the extension .h)
  --no-header          do not write the C header
```

The kernel name must be a C identifier. The architecture is the one the function was compiled for:
x86-64 or AArch64 (`ty="arm"`, so ARM64 objects for an Apple silicon Mac can be written on an
x86-64 Linux machine); `ty="native"` is taken to be the machine running sjb2obj. Functions compiled
for RISC-V or the bytecode interpreter are rejected. A saved function holds no user-defined
functions (`defuns`), so functions that call them are rejected. Exit status: 1 on errors, 2 on
usage errors.

`f.save(path)` is available on the functions returned by `compile_func`, `compile_composer` and
`compile_evaluator` (for the last two, `dtype="float64"` or `"complex128"` selects the kernel);
the functions of `compile_ode` save with `f.compiler.save(path)`.

## The object and the header

The object holds:

* `int name(double *mem, const symjit_slice *states, size_t idx, const double *params)`,
  the kernel, with two calling modes:
  * direct: `name(mem, NULL, 0, params)` evaluates one point; `mem` holds the states
    (`NAME_COUNT_STATES` values), then the outputs (`NAME_COUNT_OBS`), then the
    derivatives of an ODE (`NAME_COUNT_DIFFS`); `NAME_MEM_SIZE` doubles in all.
  * indirect: `name(NULL, states, i, params)`; `states` points to one `symjit_slice`
    (`{double *ptr; size_t len;}`) per state and then per output; the kernel reads element
    `i` of each state array and writes element `i` of each output array.

  `params` points to the parameters (`NAME_COUNT_PARAMS`), or is NULL if there are
  none. It returns 0. For `compile_composer`/`compile_evaluator` functions the inputs
  are the parameters and the outputs are written to `mem`: `name(outs, NULL, 0, inputs)`.
  Complex values are (re, im) pairs, and the counts are in doubles.
* `double name_fast(double x0, ...)`, for functions with a single output, no parameters
  and at most a few states: the output at one point.

The header defines the sizes, `symjit_slice`, and the prototypes, and says which C math
library functions the kernels call and which instructions they use.

## Limitations

* x86-64 and AArch64 (ARM64), as ELF (Linux) and Mach-O (macOS); the code is
  position-independent and links into executables (PIE or not) and shared libraries.
  Math functions are called with `call`/`bl` relocations; on AArch64 constants are
  loaded PC-relative, which limits a kernel to 1 MB.
* The kernels use AVX and FMA (x86-64-v3) on x86-64, and ARMv8-A on AArch64 (ARMv8.3-A with
  FCMA for complex kernels compiled on CPUs that have it). The header states it.
* The kernels call the math functions of the C math library (link with `-lm`): `sin`,
  `cos`, `tan`, their inverses and hyperbolic versions, `exp`, `expm1`, `exp2`, `log`
  (symjit's `ln`), `log10` (symjit's `log`, `sympy.log(x, 10)`), `log1p`, `log2`, `cbrt`,
  `erf`, `erfc`, `tgamma` (`gamma`), `lgamma` (`loggamma`), `pow` and `atan2`.
  - symjit's functions that C does not have are composed of C functions as symjit
    computes them, so that the results are the JIT's bit for bit: `csc`, `sec`, `cot` and
    their hyperbolic versions are `1.0 / sin(x)`, ..., `sinc(x)` is `sin(x) / x` (1 at 0),
    and the real `arg` is 0 (`src/obj/lower.rs`).
  - Complex functions (`dtype="complex128"`) call the C library's C99 functions: `csin`,
    `ccos`, `ctan`, `csinh`, `ccosh`, `ctanh`, `casin`, `cacos`, `catan`, `casinh`, `cacosh`,
    `catanh`, `cexp`, `clog` (symjit's complex `ln` and `log`), `cpow`, `carg` (and `csqrt`);
    `csc`, ..., `coth` are their reciprocals, `log10` is `clog(z) / ln 10`, `cbrt` is
    `cpow(z, 1/3)` and `sinc` is `csin(z) / z`. A `double complex` argument and result are in
    two floating-point registers on x86-64 and AArch64, where symjit keeps them, so the calls
    need no moves. The results can differ from the JIT's (which computes complex functions with
    the Rust crate `num_complex`, or C99's with its `libm` feature) in the last bits. Such
    models are compiled as with `fast_complex=False`; complex models without function calls
    keep the packed-complex code, bit for bit the JIT's (also with `stable_math=True`).
  - `random` is rejected.
* Models compressed by `compress=True` are supported.
* Only the scalar and fast kernels are written, not the SIMD kernels.

## Implementation

`src/obj/` holds copies of symjit's (v227's) `rust/amd/{mod,asm,f64x8,fused,scalar,complex}.rs`,
`rust/arm/{mod,macros,scalar,complex}.rs`, `rust/assembler.rs` and `rust/complexify.rs`, with the
object mode of the development line (`patches/v228-0004-obj.patch`, `v228-0028-object-math.patch`)
built in: no table of function addresses (`add_func` does nothing), direct calls of the C library
recorded in `Assembler::relocations` (`amd::object_call`, `arm::object_call`), PC-relative constant
loads on AArch64, and C99 calls for complex functions (`Complexifier::call_complex_object`, used
for every complex call). The modules at the top of `src/obj/mod.rs` stand in for the v227 paths the
copies use (`symjit::backend`); `KernelType::RowFirst` is `config::row_first`. Keep the copies close
to v227's files, so that a fix there can be carried over (`diff -u ../v227/rust/amd/scalar.rs
src/obj/amd/scalar.rs` shows the object mode). `SavedModel::object` chooses the kernels as the JIT
does; `src/linker/` (from sjc) writes ELF and Mach-O objects (`ObjectBuilder`) and the header.
