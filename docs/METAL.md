# sjb2msl: a Metal probe

Apple GPUs have no f64: the Metal Shading Language has no `double` ("'double' is not supported
in Metal"). `sjb2msl` is a probe of what a Metal backend would give pyAmpliCol's compiled-mode
kernels on a Mac: it compiles a saved symjit application into an MSL kernel, one GPU thread per
point, in f32 or in mixed f32/df64 precision, and `examples/metal_probe.rs` runs a whole
recorded execution with it.

**Result:** mixed precision runs the n = 8 helicity sum on an M4's GPU at 20.2 µs/point,
4.6× stock RustiCol, with Σ|A|² within 1.3e-6 of RustiCol's f64 at every recorded point. Over the
whole sweep, n = 4..8, it is 7.4× down to 4.5×, within 2.4e-6 (see the sweep below).

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
- **Mixed precision** (`MslOptions::mixed`, `sjb2msl --df64-inputs LIST`, `metal_probe
  --mixed`): `ptx::mixed`'s `need` rule with df64 (a `float2` of f32 hi and lo) in place of
  f64.
  - divisions and reciprocals are df64, and so is everything a divisor depends on;
  - the marked inputs (momenta, the zero plane, parameters) are df64, their low parts
    `f32(x - f32(x))` read from a second buffer (`in_lo`, or `lo` beside the arena);
  - everything else is f32, and outputs are stored as f32.

  The generator records structured operations (`Op`), so after inlining the body is in SSA
  form and the typing (`src/msl/typing.rs`) is exact. df64 arithmetic uses the Dekker/Knuth
  formulas of `recurrence_runner.cu` (`DF64` in `mod.rs`).
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
| GPU, mixed f32/df64, 4,096 points (0.4% of the operations df64) | **20.2** |
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
| GPU, mixed f32/df64 | 3.0e-7 / 9.6e-7 / **1.3e-6** | 1.4e-8 / 2.1e-7 / 4.8e-6 |
| CPU f64, all inputs rounded to f32 | 1.1e-6 / 9.5e-5 / 4.2e-3 | 1.9e-8 / 3.3e-6 / 2.0e-3 |
| CPU f64, only the momenta rounded to f32 | 1.1e-6 / 9.5e-5 / 4.2e-3 | 1.9e-8 / 3.3e-6 / 2.0e-3 |
| CPU f64, all inputs but the momenta rounded | 9.4e-8 / 3.7e-7 / 1.6e-6 | 2.5e-9 / 4.3e-8 / 7.3e-7 |
| CPU f64 (`pacman replay`, SIMD) | 1.6e-16 / 6.5e-16 / 8.2e-16 | |

**The error is the momenta's f32 rounding, not f32 arithmetic.** Nearly collinear partons make
the propagator denominators cancel. f64 arithmetic on f32 momenta is as wrong as the GPU, and
f32 currents cost at most 1.6e-6.

**Mixed precision removes it.** Momenta in df64, and df64 for what the denominators depend on
(4,675 of 1.17 million operations), bring the worst point to 1.3e-6. That is the floor of the
f32 currents (1.6e-6 in the CPU experiment), and it costs nothing in speed. The CPU replay on
this Mac also differs from `.post` in the last bits (about 1e-15), unlike on x86, where it
matches bit for bit.

## The sweep, n = 1..8 (2026-11-02)

The same comparison for u u~ > Z + (n−1)g, n = 1..8 (the `zng` sweep, `examples/metal_sweep.sh`):
each artifact's helicity-sum execution recorded with `PACMAN_SAVE_ARENA_POINTS=1024` from
`zng-mac/check/momenta<n>.json` (n = 1 has a single phase-space point), then `metal_probe` with
and without `--mixed`, at 4,096, 16,384 and 65,536 points per call. Times are the best of 5.

| n | leaves | planes | df64 ops | GPU mixed, µs/point | GPU f32 (4,096) | stock RustiCol, 1 thread | CPU runner, 10 threads | mixed vs stock |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 117 | 3.1% | 0.0024 (16,384) | 0.0054 | 0.094 | 0.113 | (39×) |
| 2 | 3 | 213 | 4.2% | 0.014 (4,096) | 0.0093 | 0.235 | 0.295 | (17×) |
| 3 | 4 | 397 | 2.6% | 0.039 (4,096) | 0.035 | 0.556 | 0.778 | (14×) |
| 4 | 9 | 753 | 1.9% | 0.224 (65,536) | 0.254 | 1.66 | 1.93 | 7.4× |
| 5 | 18 | 1,449 | 1.2% | 0.750 (4,096) | 0.744 | 5.47 | 5.04 | 7.3× |
| 6 | 34 | 2,821 | 0.8% | 2.44 (4,096) | 2.40 | 14.6 | 12.4 | 6.0× |
| 7 | 69 | 5,541 | 0.5% | 7.20 (4,096) | 7.15 | 37.3 | 34.0 | 5.2× |
| 8 | 141 | 11,057 | 0.4% | 21.0 (4,096) | 21.0 | 94.2 | 93.6 | 4.5× |

Σ|A|² over the amplitudes, relative to RustiCol, per point (median / max):

| n | GPU mixed | GPU f32 | first compile + pipelines |
|---|---|---|---|
| 1 | 2.2e-7 / 2.2e-7 | 2.2e-7 / 2.2e-7 | 0.05 s |
| 2 | 1.3e-7 / 6.5e-7 | 1.4e-7 / 7.0e-7 | 0.2 s |
| 3 | 1.6e-7 / 2.4e-6 | 2.7e-7 / **4.2e-4** | 0.5 s |
| 4 | 2.1e-7 / 2.3e-6 | 4.3e-7 / **5.7e-3** | 1.6 s |
| 5 | 2.2e-7 / 1.5e-6 | 6.6e-7 / **2.5e-3** | 5 s |
| 6 | 2.4e-7 / 1.3e-6 | 8.3e-7 / **5.8e-3** | 15 s |
| 7 | 2.7e-7 / 1.9e-6 | 1.2e-6 / **6.6e-3** | 46 s |
| 8 | 3.0e-7 / 1.3e-6 | 1.4e-6 / **5.0e-3** | 122 s |

- **Mixed precision holds at every n:** the worst point is within 2.4e-6, while f32 loses
  up to 6.6e-3 from n = 3 on (n ≤ 2 has no nearly collinear pair to cancel). Mixed precision costs
  0–5% over f32. No value is non-finite.
- **Caveat:** the GPU column is kernel time only (`run.m`, the arena already on the GPU), while
  RustiCol's is `pyamplicol profile` wall time, which includes the phase-space and host work.
  For n ≤ 3 the GPU time is a few launches (about 5–20 µs per call whatever the size), so those
  ratios mean little.
- **The gain shrinks with n,** from 7.4× at n = 4 to 4.5× at n = 8: time follows code size, and
  the arena of a call (planes × points × 4 bytes, 181 MB at n = 8 and 4,096 points) outgrows
  the caches. Larger calls are slower from n = 5 on.
- **Compile time** grows with the source (45 MB at n = 8) and is paid once: Metal caches it.

## Next steps

1. **Mixed precision with df64: done** (see above).
2. **Locality:** run the arena in tiles of about 4,096 points. Later, possibly smaller kernels
   (stage splitting, `plans/sjb2ptx/PHASE5.md`), since time follows code size.
3. **`pacman runner --metal`:** unified memory means no copies, since a shared buffer can wrap
   RustiCol's arena. RustiCol's arena is f64: a conversion kernel would split it into the f32
   arena and the lo planes (integer operations, since there is no `double`), and round the
   written planes back.
4. **More points:** the n = 1..8 sweep is done (above). Still to do: other processes, and a
   tolerance test in the suite against a checked-in recording.

The plan for these is `plans/metal/PLAN.md`.
