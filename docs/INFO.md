# sjbinfo: what a saved application holds

`sjbinfo` prints a report on saved symjit applications (`.sjb` files, written by `save` of the
Python package or `Storage::save` of the Rust crate). It covers the configuration they were compiled
with, the model's sizes, the MIR (symjit's intermediate representation, with the statistics of
`dump(path, "stats")`), the size of the machine code symjit compiles from them and the math
functions they call:

```bash
sjbinfo model.sjb                 # one report
sjbinfo a.sjb b.sjb               # one after the other, separated by a blank line
sjbinfo --no-compile model.sjb    # without compiling (no code sizes)
sjbinfo -c model.sjb              # with the constant pool
```

| option | meaning |
|---|---|
| `-c`, `--constants` | Also list the constant pool (a last section, "Constants"). |
| `--no-compile` | Do not compile the application: the "Compiled code" section says so. |
| `-h`, `--help` | Print the usage. |
| `-V`, `--version` | Print the version. |

The exit status is 0, 1 if a file cannot be read (the other files are still reported; the error
goes to stderr), and 2 for a usage error.

## The report

```
pi.sjb
  file           1,307 bytes, sjb version 4
  read by        symjit 2.27.4

Model
  numbers        real
  states         1
  parameters     0
  observables    2
  derivatives    0
  loops          0
  memory         4 values per point
  stack          90 slots
  constants      4

Configuration
  target         native: the machine that loads it (here x86-64 with AVX-512)
  opt level      2
  enabled        use_simd, use_threads, cse, fastmath, compact, fast_complex,
                 parallel_mul
  disabled       enable_simd512, complex, symbolica, simd_branch, compress,
                 direct, huge, direct_arena, direct_arena_identity_output,
                 stable_math, coroutines
  stack limit    1,048,576
  num_args       64
  option bits    0x0020920f

IR (MIR)
  instructions   236 (1,061 bytes)
  registers      6 general (Gen(0)..Gen(5))
  uses           0:52, 1:219, 2:12, 3:14, 4:4, 5:12
  stack slots    written up to Stack[89]
  labels         0
  times2 pairs   6

       count       %  instruction
          52   22.0%  times by memory
          39   16.5%  plus by memory
          31   13.1%  root
          ...

Compiled code
  scalar         2,096 bytes
  SIMD           3,808 bytes
  fast           none (for at most 8 states, 1 observable, no parameters)
  total          5,904 bytes
  saved with     scalar

Math functions
  function       calls  symjit               C library (sjb2obj)
  power             19  real, 2 arguments    pow
```

**Model.** These are the counts of the memory layout (`Program`): states (the inputs), parameters,
observables (the outputs), derivatives (ODE models), the `Sum`/`Product` loops of the model, the
values per evaluation point, the stack slots and the constants. A complex model's counts are of f64
values, two per complex number.

**Configuration.**
* `target` is the machine the code is generated for. `native` means the machine that loads the
  file, so the report names the one running `sjbinfo`.
* The options are listed by their `set_option` names (`--set` of `rust2sjb`, `ty="native;key=value"`
  of the Python package). They are the bits the file holds: symjit ignores some of them on some
  machines (`use_simd` and `fast_complex` apply to AVX and ARM64 only, `fastmath` not to SSE). The
  `debug_*` options are listed only when set.
* `option bits` is the raw option word, `Config::opt`.

**IR (MIR).** This section describes the MIR as saved: register-allocated for the target at its
opt level, and complex for a complex model.
* `instructions` is the instruction count, with the size of the MIR's serialized form.
* `lowered` (complex models only) is the count after the complexifier has lowered the complex
  operations to real ones. The bytecode interpreter runs that lowered MIR. So do the kernels,
  unless `fast_complex` is on.
* `registers` lists the general registers used and how often each is read or written.
  Unallocated (`Static`) registers appear at opt levels below 2.
* `stack slots` is the highest stack slot the MIR writes.
* `labels` counts jump targets: kernel loops, `if`/`else`, funclets.
* `times2 pairs` counts consecutive multiplications by a memory operand, which an x86 generator
  can pair.
* The table gives each instruction kind with its share. `x by memory` and `x by constant` take
  their second operand from memory or the constant pool. `fused` instructions are multiply-adds.

**Compiled code.** `sjbinfo` compiles the application as symjit's `load_func` does, for the target
in the file, and also compiles the kernels symjit compiles only when they are first used. Code for
another machine (an ARM64 or RISC-V file on x86-64) is compiled but never run. The section lists
three kernels:
* `scalar`: one point per call.
* `SIMD`: vectorized calls. These need `use_simd` and AVX or ARM64, and are absent for models with
  loops.
* `fast`: register arguments, for at most 8 states, one observable and no parameters.

`saved with` lists the kernels that had been compiled when the file was saved; `load_func` compiles
these at once. A bytecode file has no machine code. If compilation fails, the section shows
symjit's error.

**Math functions.** The table lists the functions of the program's function table and the calls in
the MIR:
* `symjit` is how symjit links the function: real or complex, and its number of arguments.
* `C library (sjb2obj)` is the function of the C math library an object file written by `sjb2obj`
  calls. In symjit 2.27 `log` is the decimal logarithm (C's `log10`) and `ln` the natural one. Real
  models get C99's `<complex.h>` functions only when complex (`csin`, `cpow`, ...). Functions C
  lacks are composed of others (`1/sin` for `csc`, `cpow(z, 1/3)` for a complex `cbrt`).
* A function marked `- (not in object files)` makes `sjb2obj` reject the file.
* A user-defined function (`defuns`) is listed as `user-defined` or `not found`: a `.sjb` file
  does not hold it.

**Constants** (with `-c`). This section lists the constant pool the kernels and the interpreter
read, one f64 per slot:

```
Constants (4)
    index   uses  value                     bits
        0      2  2.0                       0x4000000000000000
        1     19  -0.5                      0xbfe0000000000000
        2      0  1.5                       0x3ff8000000000000
        3      0  0.5                       0x3fe0000000000000
```

* `index` is the slot number that the MIR's `load_const` and `x by constant` instructions use.
* `uses` counts those instructions in the MIR the kernels are compiled from. That is the lowered
  MIR for a complex model without `fast_complex`, and the saved MIR otherwise.
* `value` is the shortest decimal that reads back to the same number.
* `bits` is its IEEE 754 encoding.

A constant with no uses is left over from a rewrite of the builder: `x**1.5` becomes `x*sqrt(x)`
and leaves 1.5 in the pool. The direct translator (`Composer`) stores a complex constant in two
slots, its real part and then its imaginary part. For complex models with `stable_math`, the pool
also holds the constants of the stable operations (`2^-1022`, `inf`, `0.25`): symjit appends them
when it compiles the model, before it is saved.

## How

`src/info.rs` reads the file with `symjit::Application::read_sjb`. Unlike the other utilities, it
does not reject calls of user-defined functions. The MIR statistics are those of symjit's
`Mir::print_stats`, re-implemented because that writes a file and its helpers are private. The code
sizes come from `Application::load`, `prepare_simd` and `get_fast`, measured with
`Application::measure` (`ker-scalar-size`, `ker-simd-size`, `ker-fast-size`). A panic of the
compiler is caught and reported. The tests (`tests/sjbinfo.rs`) compare the reports with symjit's
own measurements of the same applications.
