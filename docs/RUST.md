# rust2sjb: saved applications from Rust source

`rust2sjb` compiles a `#[kernel]` function of a Rust file (the language of the development line's
`#[kernel]` macro) into a saved symjit application (`.sjb`), without a Rust compiler. symjit loads
it (`load_func` in Python, `Application::load` in Rust), and `sjb2obj` and `sjb2wasm` compile it:

```rust
// kernels.rs
use symjit::Complex;

#[kernel]
fn newton(a: f64) -> (f64, f64) {
    let mut x = a;
    let mut i = 0.0;
    while i < 25.0 {
        x = 0.5 * (x + a / x);
        i += 1.0;
    }
    (x, x * x - a)
}

#[kernel]
fn rotate(z: Complex<f64>, t: f64) -> Complex<f64> {
    z * Complex::new(t.cos(), t.sin())
}
```

```bash
rust2sjb kernels.rs --kernel newton              # newton.sjb
rust2sjb kernels.rs --kernel rotate -O3 -fno-fast-math --target aarch64 -o rot.sjb
sjb2obj newton.sjb                               # newton.o + newton.h
```

```python
from symjit import load_func
f = load_func("newton.sjb")
f.evaluate([[2.0]])                              # [[1.41421356..., 0.0]]
```

## Options

| option | meaning |
|---|---|
| `-o <FILE>` | The saved application (default: `<kernel>.sjb`). |
| `--kernel <NAME>` | The `#[kernel]` function to compile. Default: the only one of the file (an error lists them if there are several). |
| `--target <TARGET>` | The machine the kernel is compiled for: `native` (default), `x86_64`, `aarch64`, `riscv64` or `bytecode` (the saved type; `aarch64` for an object file for ARM64 written by `sjb2obj`). |
| `-O0` ... `-O3` | The opt level (default 2). A kernel with a `while` loop is compiled by symjit's direct translator, which always uses opt level 2. |
| `-ffast-math`, `-fno-fast-math` | Fuse multiplications and additions (the default), or not. |
| `--set <KEY=VALUE>` | A further option of the configuration (`Config::set_option`: `stable_math=true`, `cse=false`, `fast_complex=false`, ...); may be repeated. |

The other options are symjit's built-in defaults (those of `Config::default()` without a
`symjit.toml`, which is not read). The kernel's inputs are its parameters and its outputs go to
`mem`, as for `compile_evaluator`; from Python, call the loaded function with `evaluate` (or
`evaluate_complex` for a complex kernel). A `bool` output is its mask: all bits set (a NaN) for
true, zero for false.

## How

rust2sjb parses the file with `syn` and interprets the kernel's body (`src/source/`, from sjc)
against a kernel builder (`src/kernel.rs`, the development line's `KernelBuilder` on symjit's
`Translator`), doing what the code the macro generates does: numbers, `for` loops, conditions on
numbers and closures run while the kernel is built, and kernel values (`f64`, `Complex<f64>` and
`bool` values derived from the parameters) add instructions. Where the macro's names and symjit
2.27 differ, the builder follows Rust: `x.log10()` is symjit's `log`, `x.fract()` is
`x - x.trunc()` (symjit's `frac` is `x - floor(x)`), `a - b` and `a / b` are a subtraction and a
division as in Rust.

The tests (`tests/rust2sjb.rs`) compile the kernels of `tests/kernels/corpus.rs` (sjc's corpus),
save and load them, and compare them with the same functions compiled as plain Rust
(`tests/kernels/corpus_plain.rs`, the corpus without the attributes): within 1e-12 without fused
multiply-adds, and bit for bit with the kernel before it was saved.

**What works.** Everything the macro accepts in kernels: arithmetic and the methods of `f64` and
`Complex<f64>` on kernel values; comparisons, `&&`, `||`, `!`, `^` (both sides are always
evaluated); kernel `if`/`else` (also without `else` when it assigns variables) and kernel `while`
loops, with the variables they assign passed through; recursion (the kernel calling itself);
`#[kernel(inline)]` helpers with any parameters (`i32`, arrays); `let` with patterns and types;
tuples, arrays (`[x; N]`, indexing, `iter`, `enumerate`, `zip`, `rev`, `map` with a closure),
ranges, `for` loops (unrolled), `loop`/`break`/`continue` in loops run while building; closures
that do not assign outer variables; `const` items of the file and `std::f64::consts`; numbers'
`f64` methods; `as` casts between numbers.

**What does not.**
* Calls of other `#[kernel]` kernels and of `#[kernel(callback)]` functions: a saved application
  holds one kernel and no Rust functions (make a called kernel `#[kernel(inline)]`).
* The exponent-field operations `pow2i`, `getexp`, `getmant` (symjit 2.27 does not have them).
* `match`, `if let`, macros (`println!`), structs, traits and generics, labeled loops. They are
  errors with the file, line and column:

  ```
  rust2sjb: error: kernels.rs:12:9: macros (`println!`) are not supported by rust2sjb
  ```

**Fixed in v227 (2026-10-18):** a complex kernel loop whose condition depends on the complex
values (e.g. an escape-time loop) was wrong when symjit lowered the complex code (`Complexifier`:
the bytecode interpreter, `fast_complex=false`, `sjb2wasm` modules, objects of complex models
that call functions): its `not` negated the zero imaginary part of a mask
(`patches/v227-complex-mask-not.patch`; `complex_loop_condition` in `tests/pipeline.rs`).
