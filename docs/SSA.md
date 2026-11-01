# sjb2ssa and ssa2ptx: the SSA form of saved functions

`sjb2ssa` converts the MIR of a saved symjit application (`.sjb`) into SSA form, where each
definition of a general register gets a register of its own, and writes it as an `.ssa` file.
`ssa2ptx` compiles an `.ssa` file into a CUDA PTX module, as `sjb2ptx` does for the `.sjb` file.

The SSA form is a base for passes that need one name per value: mixed precision with one type
per value (planned; `docs/PTX.md`, "Numbers"), liveness for splitting large kernels into
stages, and other MIR passes. By itself it does not change the GPU code: ptxas builds its own SSA
form and allocates the registers. On the 1,718 plane kernels of pyAmpliCol's u u~ > Z + 5g
artifact (2026-11-01), ssa2ptx's modules gave the same SASS as sjb2ptx's for 1,695 kernels and,
over all of them, 2,100,576 instructions against 2,100,568, with the same register counts.

```
sjb2ssa model.sjb                  # -> model.ssa, and a report on stderr
ssa2ptx model.ssa                  # -> model.ptx, model.h
ssa2ptx model.ssa --driver         # also model_cuda.c, model_cuda.h (as sjb2ptx --driver)
```

## sjb2ssa

| Option | |
|---|---|
| `-o, --output <FILE>` | The `.ssa` file (default: the input with the extension `.ssa`). |
| `--set <KEY=VALUE>` | An option of the saved configuration (symjit's `set_option`, e.g. `stable_math=true`), set before the complex operations are lowered. May be repeated. |
| `-q, --quiet` | No report. |

The report gives the instructions, the `Static` registers made, and the definitions kept in
general registers with the registers they are in.

**How.** The conversion (`src/ssa.rs`, `to_ssa`) is the first pass of symjit's register allocator
(`GreedyAllocator::create`): a definition of `Gen(r)` gets a new `Static(k)`, and a use the
`Static` its register holds. `Ret`, `Temp`, `Left` and `Right` stay as they are: library calls
and funclets pass values in them. So do the stack and memory slots (`Load`, `Save`).
* A complex model is lowered (symjit's `Complexifier`) first, since it cannot read `Static`
  registers; the `.ssa` file holds the lowered MIR.
* As in `create`, a label or a call forgets the registers. In the MIR of symjit's builder no
  general register is live across them, but the funclets of lowered complex models (such as
  `@complex_root`) read general registers set before the call. Such a register, read before any
  definition since the last label or call, is *pinned*. Of its definitions, the one current where
  control may leave the straight-line code (a label, a call, a jump or the end) is *kept*: it
  writes `Gen(r)`, so the register holds the original MIR's value at every jump and label. The
  other definitions of a pinned register, overwritten before control leaves, get a `Static` as
  well. In the u u~ > Z + 5g artifact, 19,336 definitions were kept and 1,039,902 became `Static`
  (98.2%).
* Compression mode (`LoadArgs`, `SaveArgs`) is rejected, as sjb2ptx rejects it.

## The `.ssa` file

symjit's code generators and its interpreter cannot run `Static` registers, so an `.ssa` file
must not load as an application. It has its own magic number (`ssa::SSA_MAGIC`); symjit and
sjb2ptx refuse it. Otherwise its layout is that of a `.sjb` file (symjit's `Application::save`):

| | |
|---|---|
| `SSA_MAGIC` | 8 bytes |
| version | 8 bytes: of the program and MIR formats (4) |
| flags | 8 bytes: bit 0, the MIR is a lowered complex model's |
| the program | `Program::save`: the configuration, the counts, the constants, the functions |
| the MIR | `Mir::save`: symjit's MIR encoding, which has `Static` registers (`REG_STATIC`) |
| the real locations | a complex model's real slots, as in a `.sjb` file |

There is no mask of prepared kernels: nothing in an `.ssa` file is compiled.

## ssa2ptx

The options of sjb2ptx (`docs/PTX.md`) without `--mixed` and `--f64-inputs`: `-o`, `--name`,
`--layout`, `--sm`, `--header`, `--no-header`, `--driver`, `--f32`. The module comes from
sjb2ptx's generator: symjit's `Mir::rerun` passes the registers through, and `Static(k)` becomes
the `.b64` register `%v<k>` (`.b32` with `--f32`). So the kernel's interface, layouts, header and
driver are sjb2ptx's, and the results are the same bits.

## Tests

* `src/ptx/tests.rs` `ssa_form_of_saved_models`: every `Static` written once, before its reads;
  no second conversion; the file read back as written; symjit's reader refuses an `.ssa` file and
  the `.ssa` reader a `.sjb`; straight-line models have no kept definitions and no general
  registers; complex models keep some; ptxas accepts the modules (f64 and f32, funclets, math
  functions).
* `tests/sjb2ptx.rs`: the real, loop, complex and math-library models also run through
  `sjb2ssa` and `ssa2ptx` on the GPU, which must give the same bits as `sjb2ptx`;
  `ssa_files_and_errors` checks the file names and errors of both programs.
