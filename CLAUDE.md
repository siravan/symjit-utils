# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`symjit-utils` (0.3.0; this directory is a git checkout of github.com/siravan/symjit-utils, branch
`main`, since 2026-10-26; this file is in the repo since PR #18 and shared by the x86-64
development machine and the macOS arm64 checkout: keep machine-specific notes marked as such):
command-line utilities for saved symjit applications (`.sjb`) that together are
to provide what `sjc` does: `sjb2wasm`, `sjb2obj`, `rust2sjb`, `sjbinfo` (done), and
`sjb2ptx` (the most important one; designed in `../plans/sjb2ptx/DESIGN.md`, phases 0–4 done, with
math functions from vendored musl compiled to PTX; phase 5, stage splitting, is planned in
`../plans/sjb2ptx/PHASE5.md`), `sjb2ssa`/`ssa2ptx` (the SSA form, `docs/SSA.md`), `sjb2msl` (Metal probe, `docs/METAL.md`,
plan in `plans/metal/PLAN.md`, in this repo, PR #18), and `pacman` (pyAmpliCol's evaluator containers, plans and plane tables, `docs/PACMAN.md`).
(`../plans/` is `symjit-playground/plans` on the x86-64 machine; the macOS checkout has no
`symjit-playground` around it, so those plans are only there.)
It is its own
Cargo workspace inside `symjit-playground/`, built on symjit **from crates.io** (2.28.3 since 2026-10-31, 2.28.2 since 2026-10-28, 2.28.1 since 0.2.0,
`default-features = false`; before, the path `../stable`; to test against `../stable`, add a
`[patch.crates-io] symjit = { path = "../stable" }` locally and do not commit it), whose source is
changed as little as possible: `symjit::backend` re-exports the internals a code
generator needs (`Generator`, `GeneratorType`, `StackRegions`, `Mir`, `MirInstruction`, `Builder`,
`Complexifier`, `Func`, `Loc`, `Reg`, `align_stack`, `ABI_AREA`), and `Application::read_sjb` reads a
`.sjb` file without compiling it (`patches/v227-backend-exports.patch`). The wasm backend and the
linker are copied from `../sjc` (built on `../core`) and adapted to stable's `Generator` trait, which
lacks dev's `begin_funclet`, `inlines_math`, `relocations`, `pow2i`, `getexp`, `getmant`. sjb2obj's
generators are copies of stable's own (not core's: dev changed semantics, e.g. real `log` is `ln` there
since v228-0021, `log10` in stable), with dev's object mode built in (decided 2026-10-18: option A,
stable stays without object support).

`../stable` was named `../v227` until 2026-10-19 (renamed so that its version can move from 2.27.4 to
2.28.0). Its patches written before the rename are `../patches/v227-*.patch`; new ones are
`../patches/stable-*.patch`.

## Build, test, run

```bash
cargo build --release                    # -> target/release/sjb2wasm, sjb2obj, rust2sjb, sjbinfo, sjb2ptx,
                                         #    sjb2msl, sjb2ssa, ssa2ptx, pacman
cargo test --release                     # unit tests (wasm, linker, ptx: ptxas checks every module
                                         # if /usr/local/cuda/bin/ptxas or $PTXAS) + tests/sjb2wasm.rs (Node,
                                         # wasm-validate) + tests/sjb2obj.rs (cc; qemu for AArch64)
                                         # + tests/rust2sjb.rs (corpus vs plain Rust) + pipeline.rs
                                         # + tests/sjbinfo.rs (reports vs `measure`)
                                         # + tests/sjb2ptx.rs (GPU vs JIT; needs the sandbox
                                         # disabled, skips otherwise; SJB2PTX_REQUIRE_GPU=1)
                                         # + tests/pacman.rs (PACMAN_SAMPLES=DIR also checks the
                                         # real *.pacbin under DIR, e.g. ~/pyamplicol-examples/artifacts)
                                         # (PACMAN_REPLAY_SAMPLES=DIR replays the arena recordings of
                                         # DIR/<case>/ against DIR/<case>/artifact, e.g.
                                         # ~/pyamplicol-examples/pacman)
cd tests/python && python -m unittest test_wasm test_obj test_ptx   # needs target/release/sjb2wasm, sjb2obj, sjb2ptx
python3 src/wasm/gen_tests.py > src/wasm/tests.rs # regenerate the encoder tests (wat2wasm)
```

The Python tests (`tests/python/sjb_util.py`) use `../stable/python` (or `$SYMJIT_PYTHON_PATH`; its
`_lib*.so` must be current) and its `test_fuzz`/`test_composer_fuzz`; tests using `symbolica` must run
outside the sandbox (if Symbolica runs unlicensed, it locks its thread permit in `$XDG_RUNTIME_DIR`,
which the shell inherits as another uid's `/run/user/1000`: "Could not acquire Symbolica thread permit:
Permission denied"; then set `XDG_RUNTIME_DIR=$HOME/.cache/xdg-runtime`). `test_ptx.py` needs the GPU (sandbox
disabled; skips otherwise, `SJB2PTX_REQUIRE_GPU=1`): its `Batch` builds the drivers of many kernels
into one shared library per batch (own directory: dlopen reuses a loaded path) and calls them with
ctypes; the module holds the primary context (`keep_primary_context`), or every `free` destroys it
and the fuzz takes 10x longer. `tests/wasm_runner.mjs` is the Node host of both test suites (and the reference
host of `docs/WASM.md`).

The suite passes on x86-64 (150 with a GPU), AArch64 (Raspberry Pi 400, macOS arm64) and RISC-V
(147, 2 ignored: no object files). Without CUDA (macOS) every GPU and `ptxas` check skips and still
reports `ok`, so a green run there says nothing about the PTX backend.

## Architecture

- `src/sjb.rs`, `SavedModel::read`: `Application::read_sjb` with an empty bytecode `Config` (the saved
  configuration replaces it; only its `Defuns` would be used), rejects calls of functions a `.sjb` does
  not hold (user-defined, recursive), lowers a complex model with `Complexifier` and copies its extra
  constants to the program (as `Application::with_mir` does). `SavedModel::wasm` replays the MIR into
  `WasmGenerator` (`Builder::compile_from_mir`).
- `src/wasm/` (from sjc): `WasmGenerator` (registers become locals, so MIR saved for any machine and opt
  level fits), `flow.rs` (structured control flow), `encoder.rs` (pinned against `wat2wasm` by the
  generated `tests.rs`). `what()` returns `GeneratorType::Unknown` (stable has no wasm type and never
  branches on it). It implements `divide_complex` itself (the interpreter's `num_complex` formula):
  `Mir::rerun`'s fallback, `Complexifier::generic_complex_divide`, writes `re(Temp)` = `Gen(0)` before it
  reads `x1`, where `stable_math`'s division puts the scaled numerator.
- `src/obj/` (sjb2obj): copies of stable's `amd/{mod,asm,f64x8,fused,scalar,complex}.rs`,
  `arm/{mod,macros,scalar,complex}.rs`, `assembler.rs`, `complexify.rs`, always in object mode
  (`add_func` does nothing; `object_call` emits `call rel32`/`bl` + `Assembler::relocations`; ARM
  constants `ldr d, label`; complex calls go to C99 (`generator::complex_c_name`), packed-complex
  generators reject them, so such models use `Complexifier::for_object` with fast_complex off;
  `relocations()` is an inherent method). `obj/mod.rs` has shim modules for the stable paths
  (`crate::obj::config` etc.; `KernelType::RowFirst` → `config::row_first`) and
  `SavedModel::object`/`object_files` (from sjc's `app.rs`, without callees or user functions);
  `obj/lower.rs` composes the real functions C lacks. `c_name` is stable's (`log` → `log10`). The
  copies are rustfmt'd leaf files except `amd/mod.rs` (as stable's). Port stable generator fixes here.
- `src/linker/` (from sjc): `ObjectBuilder` (ELF, Mach-O), `header`; its `tests.rs` uses
  `obj::arm::object_samples`.
- `src/kernel.rs` (rust2sjb): core's `KernelBuilder` on stable's public `Translator`/`Composer`;
  `compile()` returns stable's `Application` (core: `finish()` → `Compilation`). Removed: calls of
  other kernels and callbacks (`Defuns` has no `parents` in stable, and a `.sjb` cannot hold them),
  `debug_salt`, `pow2i`/`getexp`/`getmant`, the `math` feature's `MathOps`. Adapted to stable:
  `a - b` is `a + neg(b)` and `a / b` is `append_div` (stable's direct translator panics on
  "minus"/"divide" functions: `Operation::new`), `log10` is stable's `log`, `fract` is
  `x - x.trunc()` (symjit's `frac` is `x - floor(x)`; dev's builder has this mismatch too).
- `src/source.rs`, `src/source/{interp,syntax,value}.rs` (from sjc): no `Backend`/`Naming`/
  `register`; `Source::compile(name, config) -> Application`; kernel calls and callbacks are
  interpreter errors (`kernel_call`, `callback_call`); `builtin_config` = `Config::default()`
  without a toml. Tests: `tests/kernels/corpus.rs` (sjc's) and `corpus_plain.rs` (attributes
  removed; a test checks it is in sync) as the plain-Rust reference.
- `src/info.rs` (sjbinfo): `describe(name, bytes, compile)` → the report. Reads with `read_sjb`
  (not `SavedModel::read`, which rejects user-function calls), MIR statistics re-implemented from
  stable's private `Mir::print_stats`/`Instruction::desc`/`regs`; code sizes by `Application::load`
  + `prepare_simd` + `get_fast` and `measure` (cross-machine files are compiled, never run; panics
  caught); option bits from `Config::opt` with constants copied from stable's private `config.rs`
  (`OPTION_BITS`: the getters `use_simd`/`fastmath`/`fast_complex`/`simd_branch` also test the
  machine); C names from `obj::generator::{c_name, complex_c_name}` plus the composed ones;
  `info::Options { compile, constants }` (`--no-compile`, `-c`/`--constants`: the pool with uses
  counted in the MIR the kernels compile from).
- `src/ptx/` (sjb2ptx, `plans/sjb2ptx/DESIGN.md`, `docs/PTX.md`): `PtxGenerator` writes PTX text; every
  MIR register, `Stack[k]` (`%s<k>`) and `Mem[k]` (`%m<k>`) is a `.b64` virtual register (ptxas allocates);
  memory and stack accesses are `Item::{Load,Save}{Mem,Stack}`, placed at `seal` (`Placement`): inputs
  never written are read from `in` at each use (ptxas re-merges repeated loads), outputs never read and
  saved only unconditionally are stored where saved, forwarded to the last top-level write of the stack
  slot the final copy reads (scan back must stop at a `LoadMem` of the same register: fuzz seeds 73/94
  at O0, test `copied_inputs_match_the_jit`); other slots gathered/scattered as before;
  `Param` slots read where used (Symbolica inputs from `in`, else `params`). `PtxOptions::f32`
  (`--f32`, 2026-10-29): `emit` retypes value lines (`.f64`→`.f32`, `.b64`→`.b32`; address ops are
  `.u64` and untouched), so value instructions must go through `emit`; constants via `fconst`,
  offsets via `size`/`shift`; library calls bypass `emit` and convert through `%d<4>` (the library
  stays f64); the f64 output must stay byte-identical. `PtxOptions::mixed` (`mixed.rs`, `--mixed
  need|forward`, `--f64-inputs LIST`, 2026-10-31): divisions/reciprocals f64, plus (`need`) the
  backward slice of divisors and math arguments or (`forward`) any op with an f64 operand; the rest
  f32 through f32 twins (`%fg3`); every value f64 at labels, jumps, funclet calls and returns; the
  interface stays f64; `Mixed::arena` (plane kernels with a mapping): f32 slots in `in`/`out`, f64
  inputs in a second arena passed as `params` (`%p64`, `%ld64`); test `ptx::tests::mixed_precision`.
  `PtxOptions::mapping`
  (pacman runners): slot k at plane `map[k]` (outputs `map[count_in + j]`) in `slot()`, checked by
  `check_mapping` (SoA, plane kernel, outputs distinct and disjoint from inputs). Semantics are **x86
  `amd/scalar.rs`'s**, not the interpreter's: `.rn` ops, `fma.rn` only for MIR `Fused`, `gtu`/`geu`/
  `neu`, sign-bit `neg`/`abs`/`sign`, `cvt.rni` round, `branch_if` = `setp.nan` (a true mask is NaN),
  x86's FMA complex formulas. Funclets: `%site` + `bra`, `ret` = `brx.idx` through `$SITES`
  (`.branchtargets` must precede it). GPU-checked bit for bit vs the x86 scalar kernel (real models,
  and complex with fast_complex off). Compression, `@self`, `random` → `check` errors.
  `driver.rs` (`--driver`): `<name>_cuda.c` embeds the PTX, dlopens `libcuda.so.1` with hand-declared
  types/functions checked against CUDA 13's cuda.h (use exactly cuda.h's mapped symbols:
  `cuMemAlloc_v2`, `cuCtxPushCurrent_v2`, ... but plain `cuCtxSynchronize`: CUDA 13's
  `cuCtxSynchronize_v2` takes a context), primary context pushed/popped per call, batched
  `cuMemcpy2DUnaligned_v2` copies in either layout. `tests/sjb2ptx.rs`: per-model C harness (exit 77
  = no GPU → skip), bit for bit vs the JIT running the same MIR (`jit_of`: saved as native and
  loaded; ARM/RISC-V opt 3 MIR does not fit x86 → within 1e-12).
  Math functions (`libm.rs`, `libm/`, its README): `libm/libm.ptx` is generated by `libm/gen.sh`
  (clang-18 → bitcode, `llvm-link`, `opt` internalize + O2 with inline threshold 0, `llc`, `post.py`
  prefixes internal names `__sjm_`) from vendored musl 1.2.5 + `libm/sj/*.c` (symjit's names
  `__sj_<op>`/`__sj_cplx_<op>`; num_complex 0.4.6 port; Cephes Si/Ci/Shi/Chi port; acosh fixed for
  x < 1); `libm.rs` splits it at run time into `.func`/`.global` chunks with their `__sj*`
  dependencies; `call`/`call_complex` emit `call.uni` (args Ret[/Temp], complex Ret/Temp[/Gen0/Gen1],
  struct return in `.b8 ret0[16]`). Tests: the GPU vs `gen.sh --host` object bit for bit, vs the JIT
  within 8 ulps (real) / 1e-12 (complex, moderate |z|), with documented Cephes deviations.
- `src/pacbin.rs`, `src/pacman.rs`, `src/sha256.rs` (pacman): `Pacbin::read` ports every check of
  pyAmpliCol's reader (`rusticol-core/src/pacbin.rs`, 0BSD) incl. SHA-256 digests (own `sha256.rs`:
  no `sha2` in the offline registry); `pacbin::write` reproduces pyAmpliCol's containers byte for
  byte. `pacman::describe` reads each `symjit-application` member with `read_sjb` (as sjbinfo;
  `info::{option_list, thousands, symjit_version, MAGIC}` are `pub(crate)` for it); `--compile`
  loads them on all cores with the panic hook silenced once around the scope (not per kernel: the
  hook is global). Plane kernels count inputs as states, plain ones as parameters. The
  recompilation feasibility study is the last section of `docs/PACMAN.md`. `src/plan.rs` (`pacman
  plan`): decodes pyAmpliCol's recurrence plan (direct plan v2, `direct_codec.rs`; record sizes in
  `TABLES` must match its `reader.count(.., size)`; the header has 5 template counts before the
  executor count) up to the momentum forms, joins executors with `eager-kernel-pack.json`
  (`serde_json`), checks the symjit kernels used against `evaluators.pacbin`. `src/bindings.rs` (`pacman
  bindings`): compiled mode's `PlaneDescriptor` tables from `processes/*/execution.json`
  (`compiled_plane_arena` per stage) plus the load-time current-plane map (stage liveness, not
  saved; must track `derive_stage_current_layout`), kernel shapes checked against `evaluators.pacbin`; `Arena`
  numbers an execution's buffers as one arena for the `.mapping` files (`<states index> <arena plane>`) and
  `arena.txt` that `extract` writes beside compiled-mode plane kernels (`--no-mapping`). `docs/PACMAN.md` also
  traces how recurrence and compiled modes call the kernels. `src/recording.rs` + `src/replay.rs` (`pacman replay`): RustiCol arena
  recordings (`PACMAN_SAVE_ARENA` of a `pacman` RustiCol build, `../plans/pacman-ptx/CAPTURE.md`)
  rerun from `.pre` and compared with `.post` bit for bit; `recording.rs` is executor-independent
  (for the GPU driver), `replay_with` takes any tile executor. Ports of rusticol-core:
  `explicit_loader_config` + `load_bytes` (`CpuKernel`), `invoke_range` (`invoke`),
  `validate_raw_aliases` (`simd_safe`); current planes from the recording's `canonical_to_physical`
  (`Arena::index_with`), not `bindings`' rebuilt map; planes written before read are poisoned
  (`.pre` can already hold outputs: tiles are rerun); each tile runs in a 32-point-stride copy as
  in RustiCol. The scalar and SIMD kernels differ in the last bits (`--scalar`). `src/runner.rs` (`pacman runner`): one
  execution → `kernels/k<k>.o` (`SavedModel::object` in process, scalar kernels only), generated
  `runner.c` (per-leaf plane tables = `.mapping` rows, metadata, `pacman_runner_v1_run`),
  `pacman_runner.h` (`RUNNER_H`, the `pacman-runner-v1` interface: one pointer per arena plane,
  a leaf list, a point range), `main.c` (`MAIN_C`: recordings → `OUTDIR/NAME.arena`, checked with
  `replay --scalar --against OUTDIR`), `Makefile`; built with `cc` into `runner` and
  `librunner.so`. The current map compiled in is `bindings`' (`Execution::canonical_to_physical`);
  with a recording it must equal RustiCol's. Plans: `../plans/pacman-ptx/RUNNER.md`. `--cuda`
  (`--sm`, `--jobs`): `ptx_modules` (one PTX per leaf, `PtxOptions::mapping` = its planes; kernel
  gets the arena as `in` and `out`), `ptxas` per SM in threads + `fatbinary` → `kernels/leaf<k>.fatbin`,
  embedded by `.incbin` in `runner_cuda_c` (`RUN_CUDA_C`: libcuda by dlopen, primary context of
  device 0, device arena grown to fit, copies in read-first planes and back all written ones as 2D
  copies over equally spaced host planes, `PACMAN_RUNNER_STATS=1`, status 3 = device error);
  `tables_c` is shared with the CPU runner; test `cuda_runner_reproduces_the_scalar_replay`.
  `--f32` (`#define F32 1`): f32 leaves on a second device arena; `CONVERT_PTX` (`pacman_to_f32`/
  `_to_f64`, plane lists in `G.lists`) converts after the copies in and before the copies back,
  which stay f64; test `f32_cuda_runner_is_close_to_the_scalar_replay` (within 1e-6). `--mixed RULE`
  (`runner::Precision::{F64, F32, Mixed { rule, arena }}`): `--arena mixed` (default, `#define
  MIXED_ARENA 1`: currents/amplitudes in f32 planes, momenta, zero plane and parameters f64 from
  `FIRST_F64` on) or `--arena f64` (same values bit for bit, test
  `mixed_arena_runner_equals_the_f64_arena_one`). `--fallback PCT` (with `--cuda --f32`, `#define
  HYBRID`, `PACMAN_RUNNER_FALLBACK`): reductions rerun the PCT largest and all non-finite points on
  the CPU in f64. `--set KEY=VALUE`: kernel options before compiling. CUDA 13's
  ptxas has no sm_70: assemble test PTX for sm_80. `--simd`/`--simd512` (CPU): `obj/amd/vector.rs` and
  `vector_f64x8.rs` (symjit 2.28.1's AVX2/AVX-512 generators in object mode: per-lane math calls as
  relocations, no complex calls) through `SavedModel::object_simd(.., lanes)` (`simd_branch` forced
  on; complex models from their complexified MIR); `objects`/`runner_c`/`write` take `lanes` (1, 4,
  8) and emit `#define LANES`: SIMD kernels only, point ranges of multiples of 8, AVX2+FMA or
  AVX512F+DQ checked at run time, leaves run per block of `PACMAN_RUNNER_BLOCK` points (64), the points of a call split over
  `PACMAN_RUNNER_THREADS` pthreads (default 1, created per call, `-pthread`; worker stacks
  `STACK_NEEDED` = `KERNEL_STACK` (`Kernels::stack`, from `Amd::reserved`) + 256 KiB, at least 1 MiB;
  the caller runs a range only with that much stack left; setup by `pthread_once`, stats under a
  mutex, so concurrent callers are safe); equals
  `replay` (SIMD, not `--scalar`) bit for bit; test `simd_runner_reproduces_the_replay`. SIMD object
  kernels are x86-64 only; the scalar runner (the only one on AArch64) takes the same
  `PACMAN_RUNNER_BLOCK`/`_THREADS`/`_STATS` (ranges of any points; `KERNEL_STACK` = the scalar frame
  `SavedModel::object` records; on macOS the caller's stack from `pthread_get_stackaddr_np`/
  `_stacksize_np`, `pthread_getattr_np` is Linux-only; test `scalar_runner_runs_on_threads`, PR #17).
- `src/recurrence_runner.cu` (`pacman recurrence-runner [-o DIR] [--sm LIST]`, nvcc → `librecurrence.so`,
  C ABI `pacman_recurrence_v1_run`): one CUDA kernel runs a RustiCol **recurrence**-mode plan (not eager
  mode: eager artifacts use RustiCol's eager runtime and never reach it) given as 64-bit row words, one
  thread per point; loaded by a `pacman` RustiCol through `PACMAN_RECURRENCE_RUNNER` (patches 3 and 5:
  `RUSTICOL_TILE_POINTS` sets the recurrence tile). The host rewrites the words into 32-byte
  instructions per words key (contribution runs stably sorted by destination, register
  accumulator); `PACMAN_RECURRENCE_KERNEL` = `df64` (default, f32-pair arithmetic on a df64 arena,
  converted in place at upload/download) | `f64` (bit-identical to RustiCol). `v1` (the words
  interpreted directly), `acc` (= today's `f64`), `cache` and `df64cache` (shared-memory slots) were
  removed on 2026-10-30; they are in git history before that commit. The arithmetic is
  generic over double/df64 (`Cx<R>`), templated per op (keeps operands in registers). Test
  `recurrence_runner_variants_agree` (synthetic plan, C harness via dlopen). Bench outside the crate:
  `../plans/pacman-ptx/probes/eager/recbench.cu`. Plan: `../plans/pacman-ptx/RECURRENCE.md` §10–11.
- `src/ssa.rs` (`sjb2ssa`, `ssa2ptx`, `docs/SSA.md`, 2026-11-01): `to_ssa` = symjit's
  `GreedyAllocator::create` (Gen defs → `Static(k)`; Ret/Temp/slots unchanged; reset at Label/Call)
  plus *pinned* registers (read before a def since the last Label/Call: lowered complex funclets) whose
  def current at a Label/Call/Branch/BranchIf/End is *kept* in `Gen(r)`. Complex models are lowered
  first (Complexifier panics on Static). `.ssa` file = `.sjb` layout with its own `SSA_MAGIC` + version
  + flags (bit 0 lowered), no kernel mask; `SsaModel::{read,write,ptx}`. `ptx::ptx_of` is shared with
  `SavedModel::ptx`; `PtxGenerator::reg(Static(k))` = `%v<k>` (sjb2ptx output unchanged). Mixed precision:
  `mixed.rs` types `%v` as generator registers; `%v` values are not converted at control items (none
  crosses a label/call). `pacman runner --cuda --ssa`. Merged in PR #16 (1d9fa85).
- `src/msl/` (sjb2msl, `docs/METAL.md`, 2026-11-01, a probe): `MslGenerator` records `Item::Op`s (not
  text) and writes MSL, f32 (MSL has no `double`) or mixed (`MslOptions::mixed` = df64 inputs,
  `--df64-inputs`): `typing.rs` types the SSA values (straight-line after inlining) with `ptx::mixed`'s
  `need` rule, df64 = `float2` `d<name>` locals (`DF64` helpers in the prelude), low parts of df64
  inputs from `in_lo`/`lo` (buffer 6/3); straight-line code only: funclets whose bodies
  are straight-line (lowered complex sqrt/division) are inlined at each call (`inline_funclets`),
  other labels/jumps, math calls, compression are errors; roundings in df64 are errors. `Placement` is a straight-line copy of
  ptx's (keep in sync); `MslOptions::mapping` as `PtxOptions::mapping` (kernel takes one `arena`).
  `MslOptions::lo_first` (the runner's `lo` holds planes FIRST_F64.. only); an f32 input typed
  df64 (in a divisor slice) gets a zero low part (`float2(x, 0)`), not `lo`'s.
  `run.m` (`RUN_M`): Objective-C host, compiles the source at run time (`newLibraryWithSource`,
  `MTLMathModeSafe`; no Xcode), `run arena|plain`. `examples/metal_probe.rs`: a recorded execution on
  the GPU vs `.post`, plus CPU f64 runs on f32-rounded inputs; `examples/metal_sweep.sh ZNG [N...]` records
  and probes the zng sweep (results n = 1..8 in `docs/METAL.md`). Unit tests run on the GPU on macOS.
- `src/runner/metal.rs` + `src/runner/run_metal.m` (`pacman runner --metal [--f32]`, 2026-11-02,
  branch `metal-runner`): `metal_source` (prelude + `pacman_leaf<k>` per leaf, mixed = inputs
  ≥ `first_f64_plane`), `runner_metal_m` (`tables_c` + `REDUCE_C` + `RUN_METAL_M`; the source
  embedded by Mach-O `.incbin` of `kernels/leaves.metal`), `write_metal`/`build_metal` (`cc
  -fobjc-arc`, Metal + Foundation frameworks; `librunner.so`, `runner` from `MAIN_C`). Host: f64
  planes → shared f32 arena on the CPU (unified memory; low parts of planes ≥ FIRST_F64 split
  off), arena in blocks of `PACMAN_RUNNER_BLOCK` (4096) points (block-major, kernel `ld` = the
  block), every leaf per block, one command buffer, copy back widened (amplitudes, or all written
  planes with `PACMAN_RUNNER_COPY_BACK=all`/`PACMAN_SAVE_ARENA`), `run_reduce` from the f32 arena;
  one call at a time (mutex). Test `metal_runner_is_close_to_the_scalar_replay`.
  `examples/metal_runner_zng.sh`: the zng sweep in RustiCol (results in `docs/METAL.md`).
- `src/bin/{sjb2wasm,sjb2obj,rust2sjb,sjbinfo,sjb2ptx,sjb2msl,sjb2ssa,ssa2ptx,pacman}.rs`: the CLIs (sjb2obj: `-o`, `--name`, `--format`,
  `--header`, `--no-header`; the architecture is the saved one, `native` = host; rust2sjb:
  `--kernel`, `--target`, `-O0..3`, `-f(no-)fast-math`, `--set KEY=VALUE`; sjbinfo: several
  inputs, `--no-compile`, `-c`/`--constants`; sjb2ptx: `-o`, `--name`, `--layout soa|aos`, `--sm`,
  `--header`, `--no-header`, `--driver`, `--f32`, `--mixed RULE`, `--f64-inputs LIST`; sjb2ssa: `-o`,
  `--set KEY=VALUE`, `-q`; ssa2ptx: sjb2ptx's options (`--f32` not with `--driver`); pacman: `info [--compile] [--top N|all]`, `list [--kind]
  [--sha256]`, `extract [-a] [--kind] [--no-mapping] [-o DIR|-] [MEMBER|GLOB...]`, `plan [--groups N|all] [--pack F]
  ARTIFACT|PLAN`, `bindings [--execution PAT] [--leaf PAT] [--table] ARTIFACT|execution.json`, `replay [--time]
  [--repeat N] [--whole] [--scalar] [--no-poison] [--leaf PAT] [-v] [--dump DIR] ARTIFACT
  RECORDING.json|DIR...` (`--against DIR`), `runner [-o DIR] [--execution PAT] [--cc CC] [--no-build]
  [--cuda] [--sm LIST] [--jobs N] [--f32] [--mixed RULE] [--arena mixed|f64] [--ssa] [--fallback PCT]
  [--simd|--simd512] [--metal] [--set KEY=VALUE] ARTIFACT [RECORDING.json]`, `recurrence-runner [-o DIR]
  [--sm LIST]`); exit status 1 on
  errors, 2 on usage errors.

## Things that bite

- Compressed models (`compress=True` with a repeated shape of ≥ 3 variables, `topology.rs`) are rejected
  by sjb2wasm (`does not support compression mode`) and supported by sjb2obj; the `Expr` API rarely
  produces them.
- stable's `compile_ode` functions (`OdeFunc`) have no `save`: `f.compiler.save(path)` (`sjb_util.save`).
- Comparing complex results as `re + 1j*im` turns an infinite `im` into a NaN `re`: compare components.
- WebAssembly does not specify the sign/payload of NaN results: compare NaNs as equal.
- Complexifier `not` of a complex mask negated its zero imaginary part (complex loops with
  complex-dependent conditions wrong when lowered): fixed in stable and in `src/obj/complexify.rs`
  (`patches/v227-complex-mask-not.patch`).
- Kernels with `while` use stable's direct translator, which forces opt level 2.
- Kernel names starting with `__sj` are rejected (the math library's namespace). After changing
  `src/ptx/libm/sj/*.c` or `gen.sh`, run `src/ptx/libm/gen.sh` (the `library_is_current` test fails
  otherwise); the C must stay FMA-free (`-ffp-contract=off`) for the GPU to match the host build.
- stable's `Expr::si()`/`ci()`/`shi()`/`chi()` emitted lowercase names its function table does not know
  (`Si`, `Ci`, ...); fixed 2026-10-18 (`patches/v227-expr-integral-names.patch`).
- GPU runs (libcuda) need the Bash sandbox disabled; `ptxas` runs inside it. The CUDA toolkit is in
  `/usr/local/cuda` (13.4, not on PATH), the driver needs only `libcuda.so.1`. This checkout is
  also used on macOS arm64 (no CUDA toolkit, no `qemu-aarch64`; `node`, `wasm-validate`, `cc` present;
  Apple M4 GPU, Metal 4: runs need the sandbox disabled). On macOS arm64, `pacman replay` (SIMD and
  scalar) differs from RustiCol's `.post` in the last bits (~1e-15), unlike x86-64 (bit for bit).
- Metal: f32 denormals are flushed to zero; Metal caches compiled sources and pipelines (the n = 8
  execution: 22 s + 100 s the first time, 0.04 s after). pyAmpliCol amplitudes in f32 lose up to
  5e-3 (n = 8), almost all from rounding the momenta; mixed f32/df64 (momenta, zero plane,
  parameters df64) gives 1.3e-6 at the same speed (`docs/METAL.md`). Single helicities in RustiCol
  (the `top` execution) reach 1e-4..2.4e-3 at their worst points: cancellation in the currents
  (not fixable by the typing rule; `plans/metal/PLAN.md` step 2). Metal runner block size matters
  (n = 8: 16384 → 33.6, 4096 → 26.3 µs/point GPU); its CPU conversions are a quarter of the
  time for short kernels.
- `test_wasm.test_stable_math` fails on macOS by 1 ulp in `sin(x/y)`: Node's `Math.sinh` differs from
  the system libm (the wasm module calls the host's math).
- stable's own complex SIMD kernels differ from its scalar kernel in the last bit (vector generator
  formulas): compare PTX output with the scalar kernel (`use_simd=False`) or within an ulp.
- Kernel `bool` outputs are masks (all bits set for true), not 1.0. A sum of masks is NaN whenever one
  is true: tests encode comparisons as bits (`ifelse(1, 0)`) to see each one.
- `Translator` kernel loops need `config.set_direct(true)`: the indirect translator does not reassign
  `Temp`s, and the JIT (and the GPU) then loop forever.
- Edition 2021, as stable; never run rustfmt on `src/lib.rs` or a `mod.rs` (it recurses), format leaf files.
  `src/wasm/tests.rs` is generated and not rustfmt-clean.
