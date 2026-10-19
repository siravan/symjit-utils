# sjb2wasm: WebAssembly modules from saved functions

`sjb2wasm` compiles a function saved by symjit (`f.save(...)` of the Python package, or
`Storage::save` of the Rust crate) into a self-contained WebAssembly module that can be shipped to
and run by any WebAssembly host (a browser, Node, wasmtime, ...):

```python
from sympy import symbols, sin, cos
from symjit import compile_func

x, y = symbols("x y")
f = compile_func([x, y], [sin(x), cos(y)])
f.save("model.sjb")
```

```bash
sjb2wasm model.sjb                  # writes model.wasm
sjb2wasm model.sjb -o other.wasm
```

The function may have been compiled for any machine (`ty="native"`, `"arm"`, `"riscv"`,
`"bytecode"`, any `opt_level`): nothing is compiled for the saved target, and the module keeps
the MIR's registers in locals, of which it has any number. symjit's library itself does not
generate or execute modules.

## What is supported

* loading and storing states, observables, parameters and stack slots,
* constants and calls to imported math functions, real and complex,
* arithmetic (`+ - * /`, powers, `sqrt`, `Abs`, `floor`, `ceiling`, ...),
* comparisons, `And`/`Or`/`Not`, `Min`/`Max` and `Piecewise` (as branch-free
  selects),
* complex arithmetic and functions (`dtype="complex128"`), including complex `sqrt` and the
  `stable_math` option,
* control flow: `Sum`/`Product` loops and the if/else blocks, gotos and loops of
  Symbolica evaluators (`compile_evaluator`) and `Composer` programs
  (`compile_composer`),
* the `fast` export for models without parameters and with one output.

Rejected with an error (exit status 1):

* models compressed by `compress=True` (`the WebAssembly backend does not support compression
  mode`); `compress=True` changes only models with a repeated expression shape of three or
  more variables,
* models that call user-defined functions (`defuns`) or call themselves recursively: a saved
  function does not hold them (`a user-defined or recursive function, which a saved
  application does not hold`),
* models whose code exceeds the JavaScript API's limit of 7,654,321 bytes per function (e.g.
  the largest Symbolica evaluators).

Numerically, the module computes what symjit's bytecode interpreter computes, with these
differences from native code: fused multiply-adds (`fastmath=True`, the default) are evaluated
as a multiply followed by an add (core WebAssembly has no FMA instruction), so results can
differ from the native kernels in the last bit; `round` rounds halves to even (like the aarch64
and riscv64 backends; the interpreter rounds them away from zero); and the sign and payload of
a NaN result are the host's (WebAssembly does not specify them; x86 sets the sign bit). With
`fastmath=False` and without math functions, the module computes the native kernels' bits.
Imported functions are as accurate as the host's implementation (e.g. JavaScript's `Math`).

## Module interface

```wat
(import "symjit" "<name>" (func (param f64) (result f64)))      ;; unary functions
(import "symjit" "<name>" (func (param f64 f64) (result f64)))  ;; binary functions
(import "symjit" "cplx_<name>" (func (param f64 f64) (result f64 f64)))          ;; complex
(import "symjit" "cplx_<name>" (func (param f64 f64 f64 f64) (result f64 f64)))  ;; complex
(memory (export "memory") n)
(global (export "__heap_base") i32)
(func (export "run") (param $mem i32) (param $states i32) (param $idx i32)
                     (param $params i32) (result i32))
(func (export "fast") (param $x0 f64) ... (result f64))   ;; some models, see below
```

* **Imports.** Only the functions the model uses are imported, all from module
  `"symjit"` under symjit's function names, e.g. `sin`, `cos`, `exp`, `ln`, `log`,
  `power` (binary), `atan2` (binary), `arcsin`, `arcsinh`, `sec`, `cbrt`, `erf`,
  `gamma`. Complex functions are named `cplx_<name>` (`cplx_sin`, `cplx_exp`,
  `cplx_power`, ...) and take and return complex numbers as (re, im) pairs, using
  multiple return values (a JavaScript import returns `[re, im]`).
  `WebAssembly.Module.imports(module)` lists them.
* **Memory.** Addresses below `__heap_base` hold the stack used by `run`; the host
  places its data at or above `__heap_base`. The memory starts with one free 64 KiB
  page above `__heap_base`; a host that needs more calls `memory.grow`.
* **`run`** returns 0 on success and 1 if the stack region is exhausted. All
  pointers are byte offsets into the exported memory; values are `f64`.

`run` has the same two calling modes as the native kernels:

* **Direct:** `run(mem, 0, 0, params)`. `mem` points to the model memory: the
  states (`count_states` values) followed by the observables (`count_obs` values;
  then the diffs for ODE models). The host writes the states, calls `run` and
  reads the observables.
* **Indirect:** `run(0, states, idx, params)`. `states` points to a table of
  `count_states + count_obs` entries of 8 bytes, each an `(i32 ptr, i32 len)` pair
  pointing to an `f64` array (the wasm32 layout of `&mut [f64]`): first the state
  arrays, then the observable arrays. `run` reads element `idx` of every state
  array and writes element `idx` of every observable array, so calling it for
  `idx = 0..n-1` evaluates the model over `n` points.

`params` points to the parameter values (`params=[...]` of `compile_func`), in order.

Modules from the Symbolica bridge (`compile_evaluator`, `compile_composer`) take their
inputs as parameters and write their outputs to `mem`: call `run(outs, 0, 0, inputs)`
with `outs` pointing to room for the outputs (complex values as (re, im) pairs).

**`fast`** is exported for models without parameters, with a single output and at
most 1000 states (the JavaScript API's limit on function parameters):
`fast(x0, x1, ...)` takes the states as arguments and returns the output (NaN if the
stack is exhausted). It allocates the model memory on the module's stack and calls
`run` in direct mode, so it needs no memory management by the host:

```js
const { fast } = new WebAssembly.Instance(module, { symjit: { sin: Math.sin } }).exports;
fast(0.5, 2.0);
```

## Minimal JavaScript host

```js
import { readFileSync } from "node:fs";

const module = new WebAssembly.Module(readFileSync("model.wasm"));
const { run, memory, __heap_base } = new WebAssembly.Instance(module, {
    symjit: { sin: Math.sin, cos: Math.cos },   // one entry per import
}).exports;

const f64 = new Float64Array(memory.buffer);
const mem = __heap_base.value;           // states, then observables
const params = mem + 8 * 16;

f64.set([0.5, 1.0], mem / 8);            // x, y
run(mem, 0, 0, params);
console.log(f64[mem / 8 + 2], f64[mem / 8 + 3]);   // sin(x), cos(y)
```

`tests/wasm_runner.mjs` is a fuller host: it maps symjit's function names to
JavaScript (`Math` lacks `erf`, `gamma` and a few others; a host must supply them),
implements the complex functions, lays out the memory and calls `run` in all modes.
Mind the differences between JavaScript and C math: `Math.pow(1, Infinity)` and
`Math.pow(-1, Infinity)` are NaN where C's `pow` (and symjit's native code) return 1,
so a host should wrap `power` as the runner does.

Hosts must respect the JavaScript API's limits: modules whose `run` function is
larger than 7,654,321 bytes are rejected by sjb2wasm, and `fast` is
only exported for at most 1000 states.

## Inspecting a module

The modules carry a `name` section, so wabt's tools show readable output:

```bash
wasm-validate model.wasm
wasm2wat model.wasm          # text form, with local names (ret, temp, r0, ...)
wasm-objdump -x model.wasm   # sections, imports and exports
```

## Implementation

`src/sjb.rs` (`SavedModel::read`) reads the saved application with
`symjit::Application::read_sjb` (which compiles nothing), lowers a complex model's MIR with
`symjit::backend::Complexifier`, and replays the MIR into the generator with
`Builder::compile_from_mir`. `src/wasm/encoder.rs` is the binary encoder (instructions, LEB128, sections);
its encodings are pinned against `wat2wasm` by `src/wasm/tests.rs`, generated with
`python3 src/wasm/gen_tests.py > src/wasm/tests.rs`. `src/wasm/mod.rs`
(`WasmGenerator`) implements symjit's `Generator` trait (`symjit::backend`): MIR registers become locals
(`Ret`, `Temp`, `r0`...), stack slots live in linear memory below `__heap_base`,
constants become `f64.const` immediates, and the body is buffered until `seal`,
when constants and import indices are resolved and the module is assembled.
Labels and jumps are buffered as markers and structured by `src/wasm/flow.rs`: the
code is split into basic blocks placed in nested `block`s inside a `loop`; forward
jumps are plain `br`s, backward jumps and subroutine returns go through a `br_table`
dispatcher on a block-number local. `tests/sjb2wasm.rs` and `tests/python/test_wasm.py`
validate the modules with `wasm-validate` and run them in Node against the JIT, numpy,
Symbolica and the fuzz programs of `test_fuzz.py` and `test_composer_fuzz.py` (skipped if
`node` or wabt is missing).
