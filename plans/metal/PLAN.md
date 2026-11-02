# Metal backend: status and plan (2026-11-02)

This is the working plan for `sjb2msl` (`src/msl/`) and `pacman runner --metal`
(`src/runner/metal.rs`); the documentation is `docs/METAL.md`. It lives in the repo so
that every machine's Claude Code sees it. The other plans (`sjb2ptx`, `pacman-ptx`) are in
`../plans/` of the x86-64 development machine's `symjit-playground`, outside this repo.

## Where it stands

- **Done:**
  - PR #18 (branch `metal`):
    - `sjb2msl` compiles straight-line `.sjb` kernels to MSL, in f32 or mixed f32/df64.
    - `run.m` is a run-time-compiling Objective-C host.
    - `examples/metal_probe.rs` runs a recorded RustiCol execution on the GPU and compares
      against `.post`.
    - `examples/metal_sweep.sh` runs the zng sweep with the probe.
  - Branch `metal-runner`, on top of `metal`:
    - `pacman runner --metal [--f32]`: a `pacman-runner-v1` library that RustiCol loads
      through `PACMAN_RUNNER`. It has one embedded Metal source, CPU conversions into a shared
      f32 arena (plus df64 low parts) in blocks of `PACMAN_RUNNER_BLOCK` points, and
      `run_reduce`.
    - `examples/metal_runner_zng.sh` runs the zng sweep inside RustiCol.
  - Unit tests run on the GPU on macOS and skip elsewhere, so the x86-64 suite is unaffected.
- **Measured** on an Apple M4 (10 GPU cores):
  - Inside RustiCol (`pyamplicol profile`), the helicity sum takes 28.5 µs/point at n = 8
    against stock's 94.2 (3.3×), and 3.4–3.8× for n = 5..8. |M|² is within 2.4e-6 at every
    point for n = 1..8.
  - One helicity: 2.4× at n = 8. It is less accurate: 99% of points within 2e-5, but the
    worst reach 1e-4 to 2.4e-3.
  - The probe alone (kernel time, best of 5) gives 7.4× (n = 4) down to 4.5× (n = 8).
  - Plain f32 loses up to 6.6e-3 on the helicity sum.
  - Details are in `docs/METAL.md`.
- **Findings that matter beyond Metal:**
  - **Momenta rounding:** the f32 error of the helicity sum is almost entirely the momenta's
    rounding. Nearly collinear partons cancel in the propagator denominators. f64 (or df64)
    momenta plus f64 for the backward slices of divisors (`ptx::mixed`'s `need` rule) bring
    f32 currents to about 1e-6. That is the floor of f32 currents (1.6e-6 in the CPU
    experiment). The same should hold for `pacman runner --cuda --mixed need --arena mixed`.
  - **Single helicities:** the worst points cancel strongly in the currents (condition number
    about 1e5 with respect to the momenta). The `need` rule cannot fix that, and neither can
    df64 for all momentum-only algebra (tried: no gain). Only df64 currents, or an f64 CPU
    rerun of such points, would. Expect the same on CUDA `--mixed need`; worth measuring
    there with `--fallback`.
- **Only on a Mac:** nothing Metal runs on x86-64. Work that needs the GPU must be done on the
  Mac checkout.
- **Can be done on x86-64:** code generation, typing and `Placement` changes can be written and
  unit-tested there (`src/msl/tests.rs`: `straight_line_source` and
  `unsupported_operations_are_errors` need no GPU).

## Next steps, in order

### 1. Faster conversions

The single-threaded CPU conversions take 3.4% of the runner's time for the n = 8 sum, but
about a quarter for short kernels (n ≤ 4, one helicity). The options, cheapest first:

- convert on `PACMAN_RUNNER_THREADS` threads (the planes split over them);
- overlap: convert block b + 1 while the GPU runs block b (a command buffer per block, two
  arenas);
- a GPU conversion kernel: f64 bits → f32 + lo with integer operations (MSL has no `double`;
  mind subnormals, Inf and NaN), tested bit for bit against `(x as f32, (x - (x as f32) as
  f64) as f32)`, with a shared buffer wrapping RustiCol's planes
  (`newBufferWithBytesNoCopy`, page-aligned).

### 2. Accuracy of single helicities: a fallback

- As CUDA's `--fallback PCT`: link the CPU runner's f64 kernels (`hybrid_cpu_c`) and rerun the
  largest PCT of points of each reduction, and every non-finite one, on the CPU. The worst
  points are large values (100–5,000× the median), so "largest" catches them.
- Alternatively, df64 currents only in the leaves whose error grows most. Measure the error per
  leaf first (`metal_probe` + per-plane comparison).

### 3. Coverage and tests

- **Other processes** (e+e- > jets, gg > ng) may need math functions or control flow that the
  backend rejects today: math-library calls are errors. Check with `sjb2msl` on each plane
  kernel of an artifact first. If needed, port a small set (sqrt is native; `exp`/`log` from
  musl in f32 or df64).
- **A tolerance test** against a small checked-in recording (n = 3 or 4: a few MB), mixed
  within 3e-6, f32 expected worse than 1e-4 (documents why mixed is needed).

### 4. Possibly later

- **Smaller kernels** (stage splitting, `../plans/sjb2ptx/PHASE5.md`), since GPU time follows
  code size; per-leaf threadgroup memory for hot planes.
- **`MTLBinaryArchive`**, or precompiling with `xcrun metal` where Xcode exists: the first
  n = 8 compile takes 2 minutes (cached by Metal after that).
- **df64 on CUDA consumer GPUs** (f64 at 1/64 rate): `recurrence_runner.cu` already has df64.
  The plane-kernel PTX path could take the same typing (`--mixed need` with df64 in place of
  f64).

## Reproducing the numbers

```bash
# on the Mac, with zng-mac set up (gen.sh, check.sh) and a pacman-feature pyamplicol on PATH
OUT=/tmp/metal-sweep examples/metal_sweep.sh ~/rust/MadGraph7/pyamplicol/zng-mac 1 2 3 4 5 6 7 8
OUT=/tmp/metal-runners examples/metal_runner_zng.sh ~/rust/MadGraph7/pyamplicol/zng-mac 1 2 3 4 5 6 7 8
```
