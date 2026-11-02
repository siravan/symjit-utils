# Metal backend: status and plan (2026-11-02)

This is the working plan for `sjb2msl` (`src/msl/`, `docs/METAL.md`). It lives in the repo so
that every machine's Claude Code sees it. The other plans (`sjb2ptx`, `pacman-ptx`) are in
`../plans/` of the x86-64 development machine's `symjit-playground`, outside this repo.

## Where it stands

- **Done** (PR #18, branch `metal`):
  - `sjb2msl` compiles straight-line `.sjb` kernels to MSL, in f32 or mixed f32/df64.
  - `run.m` is a run-time-compiling Objective-C host.
  - `examples/metal_probe.rs` runs a recorded RustiCol execution on the GPU and compares
    against `.post`.
  - `examples/metal_sweep.sh` runs the zng sweep.
  - Unit tests run on the GPU on macOS and skip elsewhere, so the x86-64 suite is unaffected.
- **Measured** on an Apple M4 (10 GPU cores):
  - mixed precision runs at 7.4× (n = 4) down to 4.5× (n = 8) stock RustiCol's helicity sum,
    with Σ|A|² within 2.4e-6 at every point;
  - plain f32 loses up to 6.6e-3.
  - Details are in `docs/METAL.md`.
- **The finding that matters beyond Metal:** the f32 error is almost entirely the momenta's
  rounding. Nearly collinear partons cancel in the propagator denominators. f64 (or df64)
  momenta plus f64 for the backward slices of divisors (`ptx::mixed`'s `need` rule) bring f32
  currents to about 1e-6. That is the floor of f32 currents (1.6e-6 in the CPU experiment).
  The same holds for `pacman runner --cuda --mixed need --arena mixed`.
- **Only on a Mac:** nothing Metal runs on x86-64. Work that needs the GPU must be done on the
  Mac checkout.
- **Can be done on x86-64:** code generation, typing and `Placement` changes can be written and
  unit-tested there (`src/msl/tests.rs`: `straight_line_source` and
  `unsupported_operations_are_errors` need no GPU).

## Next steps, in order

### 1. `pacman runner --metal` (the step that makes it usable)

The aim is RustiCol calling the GPU through the `pacman-runner-v1` interface, as `--cuda` does.

- **Generation:** `runner.rs` gets a `--metal` mode next to `--cuda`. It emits one `.metal`
  source for all leaves of an execution (`MslOptions::mapping` = the leaf's planes;
  `metal_probe.rs` already builds exactly this) and a `runner_metal.m` in place of
  `runner_cuda.c`, built with `cc -fobjc-arc -framework Metal -framework Foundation` into
  `librunner.dylib`. `tables_c` is shared.
- **Arena:** RustiCol's arena is host f64. Unified memory means no copies. A shared
  `MTLBuffer` (`newBufferWithBytesNoCopy`, page-aligned; otherwise a copy) wraps the f64
  arena, and a small **conversion kernel** splits it on the GPU into
  - the f32 arena: round to nearest from the f64 bits with integer operations, since MSL has
    no `double`; mind subnormals, Inf and NaN;
  - the lo planes of the df64 inputs (momenta, zero plane, parameters: `FIRST_F64` as in the
    CUDA mixed arena);
  - and back for the written planes: f32 → f64 bits, exact.

  Write the conversion as plain integer code and test it bit for bit against
  `(x as f32, (x - (x as f32) as f64) as f32)` on random and edge-case f64s.
- **Compile cache:** compile the source once per process (`newLibraryWithSource`) and keep the
  pipelines. The first n = 8 compile takes about 2 minutes, later ones 0.1 s from Metal's cache.
  Consider `MTLBinaryArchive` or precompiling with `xcrun metal` where Xcode exists.
- **Dispatch:** one command buffer per runner call, one compute encoder, leaves in order with
  `memoryBarrierWithScope:MTLBarrierScopeBuffers` between dependent leaves, or one encoder per
  stage.
- **Test:** `metal_runner_is_close_to_the_scalar_replay`, as
  `f32_cuda_runner_is_close_to_the_scalar_replay` (within about 3e-6 for mixed), skipped
  without Metal.

### 2. Locality

- 4,096 points per call is the sweet spot. Larger calls get slower from n = 5 on: the arena is
  planes × points × 4 bytes, 181 MB at n = 8.
- In the runner, split a RustiCol tile into 4,096-point sub-tiles per call, or reorder:
  run all leaves on sub-tile 1, then sub-tile 2, ...
- Later, consider smaller kernels (stage splitting, `../plans/sjb2ptx/PHASE5.md`), since time
  follows code size, and per-leaf threadgroup memory for hot planes.

### 3. Coverage and tests

- **Other processes** (e+e- > jets, gg > ng) need math functions or control flow that the
  backend rejects today: math-library calls are errors. Check with
  `sjb2msl` on each plane kernel of an artifact first. If needed, port a small set (sqrt is
  native; `exp`/`log` from musl in f32 or df64).
- **A tolerance test** against a small checked-in recording (n = 3 or 4: a few MB), mixed
  within 3e-6, f32 expected worse than 1e-4 (documents why mixed is needed).

### 4. Possibly later

- **df64 currents** where 1e-6 is not enough. That is roughly 10× the cost per operation, so
  only the slices that need it; measure the error per leaf first.
- **df64 on CUDA consumer GPUs** (f64 at 1/64 rate): `recurrence_runner.cu` already has df64.
  The plane-kernel PTX path could take the same typing (`--mixed need` with df64 in place of
  f64).

## Reproducing the numbers

```bash
# on the Mac, with zng-mac set up (gen.sh, check.sh) and a pacman-feature pyamplicol on PATH
OUT=/tmp/metal-sweep examples/metal_sweep.sh ~/rust/MadGraph7/pyamplicol/zng-mac 1 2 3 4 5 6 7 8
```
