# sjb2msl: a Metal probe

Apple GPUs have no f64: the Metal Shading Language has no `double` ("'double' is not supported
in Metal"). `sjb2msl` is a probe of what an f32 Metal backend would give pyAmpliCol's
compiled-mode kernels on a Mac: it compiles a saved symjit application into an MSL kernel, one
GPU thread per point, and `examples/metal_probe.rs` runs a whole recorded execution with it.

## What it supports

- **Straight-line arithmetic only.** It handles add, subtract, multiply, divide, FMA, sqrt,
  reciprocal, comparisons as masks, bit operations, selects and roundings, with the PTX
  backend's f32 semantics: one rounding per operation, and `fma` only where the MIR fuses.
- **Funclets, if they are straight-line.** Lowering complex square roots and divisions to real
  operations produces funclets, but their bodies are branch-free (selects only), so each call
  is replaced by the funclet's body.
- **Errors for the rest.** Any other label or jump, math-library calls (`sin`, ...),
  compression, and the fast kernel are errors.
- **Enough for pyAmpliCol.** All 306 plane kernels of the n = 8 artifact (u u~ > Z + 7g)
  compile.
- **Interfaces:** plain kernels (`in`, `out`, `params`; SoA or AoS) and arena-plane kernels
  (`MslOptions::mapping`, as `PtxOptions::mapping`).
- **No Xcode needed.** `src/msl/run.m` compiles the source at run time (`newLibraryWithSource`,
  `MTLMathModeSafe`). The Command Line Tools' `cc` and the Metal framework are enough.
- **Measured on the M4:** f32 division, `fma` and `sqrt` are bit-identical to the CPU, but f32
  denormals are flushed to zero.

```bash
sjb2msl KERNEL.plane.symjit -o k.metal --host     # k.metal and run.m
cargo run --release --example metal_probe -- ARTIFACT RECORDING.json -o DIR --points 4096
```

The recording comes from `PACMAN_SAVE_ARENA` of a `pacman`-feature pyAmpliCol.

## Results (Apple M4, 10 GPU cores; 2026-11-01)

These are for n = 8, the `helicity_sum/color_selector[0]` execution: 141 leaves and 11,057
planes. The 1,024 recorded points come from `zng-mac/check/momenta8.json`.

| | µs/point |
|---|---|
| GPU, f32, 2,048 / 4,096 / 8,192 / 16,384 / 65,536 points per call | 22.0 / **20.3** / 26.0 / 28.7 / 33.5 |
| stock RustiCol, 1 thread (`zng-mac` run.sh) | 94.2 |
| pacman scalar CPU runner, 10 threads (`zng-mac`) | 93.6 |
| `pacman replay` (SIMD JIT, 1 thread, 1,024 points) | 124 |

- **Speed:** about 4.5× stock RustiCol and the 10-thread CPU runner.
- **Batch size:** smaller arenas are faster per point; more points make the plane stride
  longer.
- **Not compute-bound:** per-kernel time is proportional to code size (about 0.02 ns per source
  line per point), about 2.5% of the GPU's f32 peak. Every kernel gets the full
  1,024-thread occupancy, and the threadgroup size (64–256) makes no difference.
- **Compile time:** the first compile of the 41 MB source takes 22 s, and building its 141
  pipelines another 100 s. Metal caches both: 0.03 s and 0.01 s on later runs.

**Accuracy**, against RustiCol (f64). Errors are per point, as median / 99% / max:

| | Σ\|A\|² over the amplitudes, relative | \|A − A₀\| / max\|A\| |
|---|---|---|
| GPU, f32 | 1.4e-6 / 1.1e-4 / **5.0e-3** | 3.1e-8 / 3.9e-6 / 2.4e-3 |
| CPU f64, all inputs rounded to f32 | 1.1e-6 / 9.5e-5 / 4.2e-3 | 1.9e-8 / 3.3e-6 / 2.0e-3 |
| CPU f64, only the momenta rounded to f32 | 1.1e-6 / 9.5e-5 / 4.2e-3 | 1.9e-8 / 3.3e-6 / 2.0e-3 |
| CPU f64, all inputs but the momenta rounded | 9.4e-8 / 3.7e-7 / 1.6e-6 | 2.5e-9 / 4.3e-8 / 7.3e-7 |
| CPU f64 (`pacman replay`, SIMD) | 1.6e-16 / 6.5e-16 / 8.2e-16 | |

**The error is the momenta's f32 rounding, not f32 arithmetic.** Nearly collinear partons make
the propagator denominators cancel. f64 arithmetic on f32 momenta is as wrong as the GPU, and
f32 currents cost at most 1.6e-6.

**What a Metal backend would need:** momenta in two f32 (df64, from the f64 arena), and df64 for
what the denominators depend on. That is `ptx::mixed`'s `need` rule, with df64 in place of f64.
The rest can stay f32. The CPU replay on this Mac also differs from `.post` in the last bits
(about 1e-15), unlike on x86, where it matches bit for bit.

## Next steps

1. **Mixed precision with df64:** `mixed.rs`'s slice analysis with df64 arithmetic
   (`recurrence_runner.cu`'s `two_sum`/`fast_two_sum` formulas), and momenta read as
   (hi, lo) pairs. Then measure how much of the 4.5× is left.
2. **Locality:** run the arena in tiles of about 4,096 points. Later, possibly smaller kernels
   (stage splitting, `plans/sjb2ptx/PHASE5.md`), since time follows code size.
3. **`pacman runner --metal`:** unified memory means no copies, since a shared buffer can wrap
   RustiCol's arena.
