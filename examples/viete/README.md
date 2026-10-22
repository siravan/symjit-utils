# pi by Viète's formula, through all five utilities

`viete.rs` is a `#[kernel]` computing Viète's product (as `../../../stable/examples/pi2.py`, with
N = 10 factors instead of 21): `viete(0.5)` is pi within 1.2e-6. `run.sh` takes it through every
utility and checks each result:

| step | program | prints |
|---|---|---|
| plain Rust | `viete.rs` without the attribute, plus a `main` (`rustc`) | the reference value |
| `rust2sjb viete.rs` | | `viete.sjb` |
| `sjbinfo viete.sjb` | | the report (model, configuration, MIR, code sizes) |
| `sjb2wasm viete.sjb` | `wasm_main.mjs` (Node) | `viete(0.5)` |
| `sjb2obj viete.sjb` | `obj_main.c` linked with `viete.o` | `viete(0.5)` |
| `sjb2ptx viete.sjb --driver` | `ptx_main.c` with `viete_cuda.c`: 4,096 points from 0.25 to 0.75 on the GPU | `viete(0.5)` |

```bash
examples/viete/run.sh                 # outputs in examples/viete/out
examples/viete/run.sh /tmp/viete      # or elsewhere
```

The kernel's input is a parameter and its output goes to `mem` (as for `compile_evaluator`), so
the WebAssembly module is called as `run(outs, 0, 0, inputs)` and the object as
`viete(outs, NULL, 0, inputs)`. Steps whose tools are missing (`rustc`, `node`, `cc`) are skipped,
as is the GPU run without a usable GPU (in the Bash sandbox, for instance). On hosts other than
x86-64 and AArch64 (RISC-V), `sjb2obj` cannot write a native object: the script writes an AArch64
object from a second `.sjb` saved with `rust2sjb --target aarch64` and does not run it.

On x86-64 with an RTX 5060 (2026-10-21) all five give 3.1415914215111997, bit for bit the same.
