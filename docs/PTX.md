# sjb2ptx: CUDA PTX modules from saved functions

`sjb2ptx` compiles a function saved by symjit (`f.save(...)` of the Python package, `Storage::save`
of the Rust crate, or `rust2sjb`) into a CUDA PTX module. Each GPU thread evaluates the function at
one point. The module needs only the NVIDIA driver, which compiles PTX for the GPU at hand
(`cuModuleLoadData`); neither the CUDA toolkit nor symjit is needed to run it.

```bash
sjb2ptx model.sjb                       # model.ptx and model.h
sjb2ptx model.sjb -o k.ptx --name k --layout aos
sjb2ptx model.sjb --driver               # also model_cuda.c and model_cuda.h
cc prog.c model_cuda.c -ldl              # a C program running it: no CUDA toolkit needed
```

| option | meaning |
|---|---|
| `-o <FILE>` | The PTX module (default: the input with the extension `.ptx`). |
| `--name <IDENT>` | The kernel's name (default: the module's stem; a C identifier). |
| `--layout soa\|aos` | Where the inputs and outputs of a point are (below; default `soa`). |
| `--sm <N>` | The module's `.target sm_N` (default 70; the driver compiles it for newer GPUs). |
| `--header <FILE>` | The C header (default: the module with the extension `.h`). |
| `--no-header` | Do not write the header. |
| `--driver` | Also write the C driver `<name>_cuda.c` and `<name>_cuda.h` next to the module (below). |
| `--f32` | Single precision (below, "Numbers"); not with `--driver`. |

The function may have been compiled for any machine (`ty="native"`, `"arm"`, `"riscv"`,
`"bytecode"`) and opt level: sjb2ptx compiles its saved MIR and lets ptxas allocate the registers.
The exit status is 1 on errors and 2 on usage errors.

**Status (phases 1 to 3 of `plans/sjb2ptx/DESIGN.md`).**

What works:
* arithmetic, `sqrt`, `Abs`, the roundings, powers;
* the math functions, real and complex (section "Math functions");
* comparisons, `Piecewise`, `Min`/`Max`, masks;
* `Sum`/`Product` loops and the loops and if/else blocks of Symbolica evaluators and `Composer`
  programs;
* complex models, including `stable_math`.

What is rejected, with an error (exit status 1):
* compression mode (`compress=True`);
* recursive functions;
* `random`;
* kernel names starting with `__sj` (the math library's).

## The kernel

```
.visible .entry <name>(.param .u64 in, .param .u64 out, .param .u64 params,
                       .param .u64 n, .param .u64 ld_in, .param .u64 ld_out)
```

Launch it with a one-dimensional grid of at least `n` threads (`<NAME>_BLOCK_SIZE` per block).
Thread `i` evaluates point `i` and does nothing if `i >= n`. All pointers are device pointers, and
all values are doubles.

* **Inputs** (`<NAME>_COUNT_IN` per point): the states. For a Symbolica model (`compile_evaluator`,
  `compile_composer`), they are the model's parameters, which differ from point to point.
* **Outputs** (`<NAME>_COUNT_OUT` per point): the observables, followed by the derivatives of an ODE
  model.
* **`params`**: the shared parameters of a model compiled with `params=[...]`
  (`<NAME>_COUNT_PARAMS`), the same for every point; 0 if there are none.
* **Complex models** count slots: a complex value takes two consecutive slots, re and im.
* **Layout:**
  * SoA (default): slot `k` of point `i` is at `in[k * ld_in + i]` (`ld_in >= n`), and likewise for
    `out`. Neighbouring threads read neighbouring addresses, which is fast.
  * AoS (`--layout aos`): it is at `in[i * ld_in + k]` (`ld_in >= COUNT_IN`), the row-major matrix
    of `evaluate_matrix`.
* `in` and `out` must not overlap.

The header `<name>.h` defines `<NAME>_KERNEL`, `<NAME>_COUNT_IN`, `<NAME>_COUNT_OUT`,
`<NAME>_COUNT_PARAMS`, `<NAME>_LAYOUT_SOA`, `<NAME>_COMPLEX` and `<NAME>_BLOCK_SIZE`. Its comment
shows the launch.

## The C driver (`--driver`)

`<name>_cuda.c` embeds the module and runs it through the CUDA driver API. It loads
`libcuda.so.1` with `dlopen` and declares the few driver types and functions it uses itself, so
neither the CUDA toolkit nor `-lcuda` is needed to build it: `cc prog.c <name>_cuda.c -ldl`.

```c
#include "model_cuda.h"

model_cuda *ctx;
if (model_cuda_init(&ctx, 0) != 0) {            /* device 0: its primary context */
    fprintf(stderr, "%s\n", model_cuda_error(ctx));
    model_cuda_free(ctx);
    return 1;
}
/* n points of host arrays in the kernel's layout (SoA here: ld_in = ld_out = n) */
if (model_cuda_evaluate(ctx, n, in, n, out, n, NULL) != 0)
    fprintf(stderr, "%s\n", model_cuda_error(ctx));
model_cuda_free(ctx);
```

| function | |
|---|---|
| `int <name>_cuda_init(<name>_cuda **ctx, int device)` | Loads libcuda, takes the device's primary context and compiles the module (the driver's JIT; its cache, `~/.nv/ComputeCache`, makes later starts fast). `*ctx` is set even on failure, for the error message. |
| `int <name>_cuda_evaluate(ctx, n, in, ld_in, out, ld_out, params)` | Host arrays: copies the points to the GPU in batches, runs the kernel, copies the outputs back. `params` points to the shared parameters (NULL if none). |
| `int <name>_cuda_evaluate_device(ctx, n, in, ld_in, out, ld_out, params, stream)` | Device arrays (`CUdeviceptr` values): launches the kernel on `stream` (a `CUstream`, or NULL) and returns without waiting. |
| `void <name>_cuda_set_batch(ctx, max_points)` | The most points `evaluate` sends at a time (default 2^20; fewer if the GPU's memory is short). |
| `const char *<name>_cuda_error(ctx)` | The last error, with the CUDA error name or the PTX compiler's log; `""` after a successful call. |
| `void <name>_cuda_free(ctx)` | Frees the buffers, the module and the context reference. |

The functions return 0 on success and -1 on errors. A context is used by one thread at a time.
`<name>_cuda.h` also defines `<NAME>_CUDA_COUNT_IN`, `_COUNT_OUT`, `_COUNT_PARAMS` and
`_LAYOUT_SOA`. The driver is for Linux (`libcuda.so.1`); it uses the primary context, so it works
next to the CUDA runtime API in the same program.

## Numbers

The kernel performs the IEEE operations of symjit's x86-64 scalar kernel:
* additions, subtractions, multiplications, divisions and square roots are correctly rounded (the
  `.rn` instructions, which ptxas does not contract);
* fused multiply-adds appear only where symjit's MIR has them (`fastmath=True`, the default), with
  one rounding as on x86 with FMA;
* comparisons give masks (all bits set for true), with x86's NaN rules: `>` and `>=` are true for
  NaN operands, `!=` too;
* `round` rounds halves to even;
* `sign`, `Abs` and negation act on the sign bit;
* subnormals are kept.

So for real models the GPU returns the bits of the CPU kernels (tested with ±0, subnormals, ±inf
and NaN). The only exception is the payload of NaN results.

For complex models the kernel runs the complex arithmetic as real operations, with symjit's
complexifier (as `fast_complex=False` does on the CPU), and returns the bits of the CPU's scalar
kernel with `fast_complex=False`. Results differ in the last bit from the `fast_complex=True`
kernels, which use a different formula, and from the CPU's SIMD kernels, whose complex formulas
also differ from the scalar kernel's.

With `--f32` (`PtxOptions::f32`) every value is an f32 instead: `in`, `out` and `params` are
arrays of floats (slot `k` at `4 k` bytes), the registers are `.b32`, the operations the same
correctly rounded `.rn` ones in single precision, masks and sign-bit operations on 32 bits, and
constants rounded to the nearest f32. Math functions are computed in f64 by the same library: the
arguments are widened and the result rounded to f32. The C header says "floats"; `--driver` is
f64 only. There is no CPU f32 kernel to compare with: f32 results are compared with the f64 ones
within single-precision rounding.

## Math functions

The module includes the functions its kernel calls, compiled to PTX from C: musl 1.2.5's real
functions, and symjit's functions on top of them (`src/ptx/libm/`, see its README). They are
`.func`s called with `call.uni`. A module contains each function it needs once, with its
dependencies (e.g. `sin`'s argument reduction).

* **Real functions:** all of symjit's, with symjit's definitions. `log` is the decimal logarithm,
  `ln` the natural one, `csc` is `1/sin`, `sinc(0) = 1`, `arg` of a real is 0. Si, Ci, Shi and Chi are
  Cephes' (as in symjit).
* **Complex functions:** num_complex's formulas, as the JIT's, on musl's real functions. A complex
  `log` is the natural logarithm, as in symjit.
* **Not supported:** `random`.

The GPU computes the same bits as this C does on x86-64. It differs from the JIT, which uses glibc
and Cephes, in the last bits:
* real functions within 8 ulps (most within 1 or 2);
* complex functions within 1e-12 relative for arguments of moderate size.

Exceptions:
* musl's `gamma` and `loggamma` follow C at special values: `gamma(±0)` is ±inf where the JIT gives
  NaN, and `loggamma(-inf)` is +inf.
* They are also more accurate. The JIT's `loggamma` is off by up to 3e-9 near 1 and 2 and infinite
  for subnormal arguments, and its `gamma` flushes subnormal results to zero.

## How

`src/ptx/mod.rs` (`PtxGenerator`) implements symjit's `Generator` trait and writes PTX text.
* Every MIR register, stack slot and memory slot becomes a `.b64` virtual register. An input the
  kernel never writes is read from `in` where it is used, and an output it never reads is stored
  to `out` where it is computed, when all its stores are unconditional (no conditional jumps
  outside funclets). If the MIR keeps such an output in a stack slot until a copy at the end,
  the store moves to the slot's last write. The other memory slots are loaded at the start and
  stored at the end. This keeps inputs and outputs out of the live registers: in pyAmpliCol's
  large plane kernels, loading every input at the start made ptxas spill most of the kernel.
* Funclets, such as the complex square root, run on the kernel's registers. A call sets `%site`
  and jumps; the return is a `brx.idx` through the table of return sites.
* Labels are numbered, and constants are immediates.

Tests:
* **`src/ptx/tests.rs`** checks the instructions of each operation, the interface, the layouts
  and the errors. It assembles every module with `ptxas` when the CUDA toolkit is installed
  (`$PTXAS`, default `/usr/local/cuda/bin/ptxas`).
* **`tests/sjb2ptx.rs`** runs the kernels on the GPU through the driver and a C harness.
  * Every math function on 6,000 points (special values, pairs of them, random magnitudes up to
    2^±1000): the GPU must give the bits of `src/ptx/libm/gen.sh --host`, the same C compiled for
    x86-64, and stay within the bounds above of the JIT. Models mixing calls with arithmetic
    (every opt level, FMAs on and off, MIR of other machines, `stable_math`) are compared with the
    JIT within 1e-12.
  * Models without math functions must match the JIT's scalar kernel bit for bit:
    * every opt level, with and without FMAs;
    * MIR saved for x86-64, ARM64, RISC-V and the bytecode interpreter (ARM64 and RISC-V MIR of
      opt level 3 does not fit x86-64's registers; it is compared within 1e-12);
    * both layouts, padded leading dimensions, small batches, the device path;
    * shared parameters, kernel loops, complex models (`stable_math` too);
    * ±0, subnormals, ±inf and NaN.
* **`src/ptx/tests.rs` `library_is_current`** regenerates the library with `gen.sh` (when clang
  18 is installed) and requires `libm.ptx` to be the same.
* **`tests/python/test_ptx.py`** compiles functions with symjit's Python package, builds their
  drivers into a shared library and calls them with ctypes, in both layouts:
  * arithmetic, comparisons, loops, parameters and complex models against numpy and the JIT;
  * special values bit for bit with the native scalar kernel;
  * the math functions within 1e-13 of the JIT;
  * the random expressions of symjit's `test_fuzz.py`, the random `Composer` programs of
    `test_composer_fuzz.py`, and Symbolica evaluators.

  Without a usable GPU (no driver, or `/dev/nvidia*` unreachable, as in a sandbox) the GPU steps
  are skipped; `SJB2PTX_REQUIRE_GPU=1` makes that an error.
