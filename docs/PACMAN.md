# pacman: the kernels of pyAmpliCol artifacts

[pyAmpliCol](https://github.com/MadGraphTeam/pyamplicol) (QFT amplitudes) compiles its evaluators
with symjit and stores them in one container per artifact, `evaluators.pacbin` (format
`pacbin-v1`). `pacman` reads such containers with the checks pyAmpliCol itself makes, digests
included, reports on the kernels they hold, and extracts members as files:

```bash
pacman info evaluators.pacbin                 # members by kind, kernel statistics, the 10 largest
pacman info --compile --top all evaluators.pacbin   # + machine-code sizes; every kernel listed
pacman list evaluators.pacbin                 # kind, length and path of every member
pacman list --kind symjit-application --sha256 evaluators.pacbin
pacman extract evaluators.pacbin              # the kernels, into ./evaluators/<member path>
pacman extract -a evaluators.pacbin -o out    # every member, into out/
pacman extract evaluators.pacbin '*stage_5*' -o out     # members whose paths match
pacman extract evaluators.pacbin path/to/x.plane.symjit -o - > x.sjb   # one member to stdout
pacman plan artifacts/qq_z6g_recurrence_jit_o2   # a recurrence artifact's plan: stages, executors
pacman plan --groups all recurrence-runtime.pacbin   # every row group, in execution order
pacman bindings artifacts/qq_z6g_compiled_jit_o3     # a compiled artifact's kernel calls and their planes
pacman bindings --execution helicity_sum --leaf stage5:0 --table artifacts/qq_z6g_compiled_jit_o3
pacman replay artifacts/qq_z6g_compiled_jit_o3 arena/       # RustiCol arena recordings, rerun
pacman replay --time --repeat 5 artifacts/qq_z6g_compiled_jit_o3 arena/x.e0.r0.json   # + per-leaf times
pacman runner -o r artifacts/qq_z6g_compiled_jit_o3 arena/x.e0.r0.json   # its execution as r/runner, r/librunner.so
r/runner out arena/*.json && pacman replay --scalar --against out artifacts/qq_z6g_compiled_jit_o3 arena/
```

| option | command | meaning |
|---|---|---|
| `--compile` | info | Compile every kernel for this machine (as pyAmpliCol loads it, plus the SIMD kernel; never run, on all cores) and report the code sizes. |
| `--top N\|all` | info | List the N largest kernels by MIR instructions (default 10; 0 lists none). |
| `-k`, `--kind KIND` | list, extract | Only members of this kind: a name as `list` prints it, or its number. |
| `--sha256` | list | Also print each member's SHA-256 (as the index stores it). |
| `-a`, `--all` | extract | Every member, not only the kernels. |
| `--no-mapping` | extract | Write no `.mapping` and `arena.txt` files (see [Plane mappings](#plane-mappings-pacman-extract)). |
| `-o`, `--output DIR` | extract | The output directory (default: the container's file name without `.pacbin`, in the current directory). `-` writes the one selected member to standard output. |
| `--groups N\|all` | plan | List the first N row groups (default 50). |
| `--pack FILE` | plan | The kernel pack that names the executors (default: the artifact's `model/eager-kernel-pack.json`). |
| `--execution PAT` | bindings | Only the executions whose names match (`top`, `helicity_sum`, `helicity_sum/color_selector[0]`, `helicity_selector[1]`, ...). |
| `--leaf PAT` | bindings | Only the leaves whose label (`stage5:0`, `amplitude:0`) or kernel path matches. |
| `--table` | bindings | Print each selected leaf's descriptor table. |
| `--time` | replay | Print each leaf's kernel time and µs per point (the best of the runs), and the total. |
| `--repeat N` | replay | Run each recording N times (default 1); every run must give the same bits. |
| `--whole` | replay | Run all points as one tile instead of the recorded tiles (a diagnostic: differences are reported, not errors). |
| `--scalar` | replay | Use only the scalar kernels (a diagnostic, as `--whole`). |
| `--no-poison` | replay | Run on `.pre` as it is, without filling the planes written first with NaNs. |
| `--leaf PAT` | replay | Report only the leaves whose label or kernel path matches (all leaves still run). |
| `-v`, `--verbose` | replay | One line per leaf (descriptors, SIMD and scalar calls) and every differing plane. |
| `--dump DIR` | replay | Write the replayed arena as `DIR/NAME.replay` (the layout of `.post`). |
| `--against DIR` | replay | Compare with `DIR/NAME.arena` (a runner's output) instead of `.post`; differences are errors, also with `--whole` or `--scalar`. |
| `-o`, `--output DIR` | runner | The runner's directory (default `runner-<process>`). |
| `--execution PAT` | runner | The execution (it must match one), when no recording names it. |
| `--cc CC` | runner | The C compiler (default `cc`). |
| `--no-build` | runner | Write the files only; `make` in the directory builds them. |

`extract` selects the members whose paths match one of its MEMBER arguments (a path, or a
pattern: `*` matches any characters, `/` included, `?` one); without them it extracts the kernels.
Each member is written at its path under the output directory (the reader admits only relative
paths without `..`). The exit status is 0, 1 on errors (a container that fails a check, a pattern
that matches nothing) and 2 on usage errors. Each extracted kernel is an ordinary `.sjb` file:
`sjbinfo`, `sjb2wasm`, `sjb2obj` and `sjb2ptx` take it, and symjit 2.28 loads it. From a
compiled-mode artifact, `extract` also writes each plane kernel's `.mapping` file and each
process's `arena.txt` (see [Plane mappings](#plane-mappings-pacman-extract)).

## The report

```
evaluators.pacbin
  container      pacbin-v1, 10,592,680 bytes, 141 members (digests verified)
  read by        symjit 2.28.0

Members
  symjit-application (kernels)                 94       4,820,585 bytes
  symbolica-exact-state                        47       5,740,670 bytes

Kernels
  kernels        94: 47 plain (.symjit), 47 plane (.plane.symjit)
  distinct       86 (4,340,067 bytes; the others repeat a payload, 480,518 bytes)
  sjb version    3 (94)
  target         native (94)
  numbers        complex (94)
  opt level      3 (94)

Options (kernels per combination)
  4 plain        use_simd, cse, fastmath, complex, symbolica, compact,
                 fast_complex, parallel_mul, direct_arena_identity_output
  43 plain       use_simd, cse, fastmath, complex, symbolica, compact,
                 parallel_mul, direct_arena_identity_output
  47 plane       use_simd, cse, fastmath, complex, symbolica, compact,
                 parallel_mul, direct_arena, direct_arena_identity_output

Sizes                      min        median           max         total
  file bytes                186        13,442       240,430     4,820,585
  inputs                      8           372         3,320        86,200
  outputs                     2           128         1,024        35,256
  MIR                         6         2,681        47,923       971,434
  constants                   0             5             6           450
  stack slots                80         1,070         2,532        96,316
  scalar code               272        58,736     1,096,240    21,552,288
  SIMD code                 608        87,744     1,165,760    24,351,168
  (values per point; a complex number counts as two; code in bytes, for this machine)

Largest kernels (10 of 94, by MIR instructions)
        MIR  inputs outputs      bytes     scalar       SIMD  path
     47,923   3,320   1,024    240,430    994,912  1,135,360  processes/.../generic_stage_5_subset_6_chunk_0_3a95....symjit
...
```

(the `u u~ > Z g g g g g g` artifact compiled at O3, with `--compile`, on an AVX2 machine; paths
shortened). What it says:

- **Members**: the container's members by kind. The kernels are the members of kind
  `symjit-application`; `symbolica-exact-state` members (`.evaluator.bin`) are Symbolica's saved
  evaluators (pyAmpliCol uses them for exact evaluation; each embeds a byte-for-byte copy of its
  chunk's plain kernel, 482 of 482 in `pp_zjj`).
  Other containers of an artifact (`eager-runtime.pacbin`, `recurrence-runtime.pacbin`) hold
  tables and plans, no kernels.
- **kernels**: pyAmpliCol saves two applications per evaluator chunk: a *plain* one
  (`.symjit`; inputs as parameters, called with `evaluate_matrix`) and a *plane* one
  (`.plane.symjit`; inputs as states, `direct_arena`; its raw scalar and SIMD kernels are what
  pyAmpliCol calls). **distinct** counts the different payloads: pyAmpliCol stores the kernels of
  a color-selector sector apart from those of the helicity sum, even when they are equal.
- **sjb version**: 3 for everything pyAmpliCol 1.0.1 writes (it links symjit 2.26.4; stable reads
  version 3 since 2026-10-21, `patches/stable-sjb-version3-load.patch`).
- **Options**: the option bits of the kernels' configurations, per combination and flavor (the
  names of `Config::set_option`, as `sjbinfo` prints them).
- **Sizes**: minimum, median, maximum and total over the kernels: inputs and outputs per point (f64
  values; a complex number counts as two), MIR instructions (complex ones not lowered), the constant
  pool, the stack slots; with `--compile`, the machine code of the scalar and SIMD kernels as this
  machine's symjit compiles it (what pyAmpliCol gets would differ only by symjit version: the files
  hold MIR, and the machine code is compiled when they are loaded).
- **Largest kernels**: by MIR instructions; `inputs` and `outputs` per point, `bytes` of the file.
- **Unreadable kernels** (only if any): members of kind 1 that are not symjit applications or that
  symjit cannot read, with the reason. They do not stop the report.

## Recurrence plans (`pacman plan`)

A recurrence-mode artifact keeps its plan in `recurrence/schedules/<digest>/recurrence-runtime.pacbin`
(member `schedule/recurrence-direct-schedule-v2.bin`, "direct plan v2", magic `PACRDAP2`, fixed
little-endian records; `rusticol-core/src/recurrence/direct_codec.rs`). `src/plan.rs` decodes it up
to the momentum forms (the selector, replay, helicity-resolution and proof tables are counted) and
joins it with the executor templates of `model/eager-kernel-pack.json`
(`recurrence_direct_template.templates`, by `direct_executor_id`). `pacman plan` takes the
artifact directory, the container or the extracted member.

How pyAmpliCol runs it (`recurrence/direct_runtime.rs`, `direct_backend.rs`): for each tile of points
(`point tile`, 1,024 here), the row groups run in plan order; a group calls its executor once per row,
over the whole tile. Values live in a split-complex arena (real and imaginary planes, one value per
point); a current's components sit at its `component_base`, and currents whose uses do not overlap
share arena space. Stage 0 fills the external sources from momenta and helicities; each later stage
builds the currents of one more external leg: contributions add a vertex of two parent currents (with
their momentum forms, signed sums of external momenta) into the destination current (the first row
initializes it), finalizations apply the propagator in place; the last stage's closures contract
currents into the amplitudes. Topology replay then maps the computed colour sector to the others.

An executor is either a Rusticol intrinsic (Rust code for a standard vertex, propagator, source fill
or closure) or a prepared call: a symjit plane kernel (`kernels/NNNNNN/application-1.plane.symjit`,
member `model/eager-kernels/...` of `evaluators.pacbin`, pinned by the template's
`source_application_sha256`; the report checks it). pyAmpliCol binds a certified intrinsic in place
of an equivalent prepared kernel, and only instantiates the executors the plan uses. In the
`u u~ > Z + 6g` recurrence artifact (2026-10-22) all 9,599 rows of its 69 groups are intrinsics:
no symjit kernel runs, although 374 of the catalog's 597 executors are symjit kernels.

```
Executors run (one call per row, over a tile of points)
      rows  groups  role          executor
     2,949      14  contribution  intrinsic rusticol.recurrence-intrinsic.weyl-vector-to-weyl-a.v1
     ...
  symjit kernels run 0 of 9,599 rows (0 distinct executors; catalog: 223 intrinsic, 374 symjit)
```

## Compiled mode: how the kernels are called

Compiled mode is the same current recursion, with each stage fused into Symbolica-built symjit
kernels instead of per-vertex calls (traced in pyamplicol 1.0.1, `engine/compiled_direct_prototype.rs`,
`evaluator/symjit_compiled_direct.rs`, `engine/evaluation.rs`; descriptor
`processes/<id>/execution.json`):

- **Stages.** Stage k (`generic_stage_k_subset_{k+1}`) computes every current of k+1 external legs
  from the currents of earlier stages, the momenta and the model parameters; the amplitude stage
  (`generic_amplitude_stage`) contracts currents into amplitudes. Each stage's outputs are split into
  *leaves* (chunks) of at most `evaluator.output_chunk_size` complex outputs (512 in the example card:
  1,024 f64 planes), one symjit kernel each. So "one large kernel" is one chunk of one stage; its size
  is set by the chunk size.
- **Three executions** in one artifact, each with its own kernels (`execution.json`): the top level
  (`compiled`, a quotient over helicities: 66 currents, used for a selected single helicity), the
  helicity sum (`helicity_sum_execution.compiled`: 1,425 currents, 384 amplitudes; used for the
  unselected total) and per colour sector (`color_selector_executions`, the same kernels again: the
  duplicates `pacman info` counts).
- **Calls.** At load each leaf's plane application is loaded (`Application::load` + `prepare_simd`)
  and bound to a persistent split-complex, component-major arena: inputs are current planes of
  earlier stages, momentum planes (real), and broadcast planes of model parameters; outputs overwrite
  the stage's current (or amplitude) planes; no aliasing. A runtime call fills sources and momenta,
  then for each tile of points (at most 32, `COMPILED_DIRECT_LOCALITY_POINT_CAP`) runs the stages in
  order and, within a stage, its leaves in order: one plane-kernel call per leaf per tile, which runs
  the SIMD kernel once per 4 points. The amplitudes are then reduced (helicity and colour sums).
- **Cost** (`u u~ > Z + 6g`, helicity sum, AVX2, 2026-10-22): 13 leaves per tile; pyAmpliCol's profile
  counts 26 arena calls per 64-point call and 34.1 µs/point, all but ~0.05 µs in the kernels. Timed
  alone (64-point batches), the leaves add up to 30.8 µs/point; stage 5's first chunk (3,320 inputs,
  1,024 outputs, 1.17 MB of SIMD code) takes 7.9, the stage 4, 6 and 7 first chunks 4.1, 4.7 and 4.5.
- **Chunk size** (same process, regenerated with each `output_chunk_size`, `pyamplicol profile`,
  2026-10-22): 1024 → 37.8 µs/point (8 leaves, largest SIMD kernel 1.40 MB), 512 → 31.6–34.2 (13,
  1.17 MB), 256 → 27.8 (21, 0.85 MB), 128 → 26.1 (36, 0.52 MB), 64 → 25.7–25.9 (69, 0.27 MB), 32 →
  27.0. Total code grows 9% from 512 to 64, but once a leaf's code fits in L2 (1 MB here) its 8
  SIMD passes per tile stop refetching it from L3. Results agree (totals bit-identical, resolved
  values within 4e-13).

## Plane tables (`pacman bindings`)

`pacman bindings ARTIFACT|execution.json` lists, per process and execution, the leaves in call
order and the descriptor table each kernel is called with, as `rusticol-core` builds it at load time
(`src/bindings.rs`, after `load_stage`, `append_component_bindings`,
`derive_stage_current_layout` and `bind_with_current_map`). With an artifact (or an
`execution.json` inside one) it also checks each kernel's input and output planes against
`evaluators.pacbin` and flags a leaf whose kernel is missing or of another shape.

What is saved. Each stage of `execution.json` has a `compiled_plane_arena` block: `input_bindings`
(per stage input: `kind` `value`, `momentum` or `model_parameter`, `global_component`,
`source_id`), `output_bindings` (per output: the current or amplitude component) and `leaves` (per
kernel: `input_indices` into the stage inputs, `output_start..output_stop`). The table is then
fixed: one descriptor per f64 plane, the inputs then the outputs, two per complex value:

| binding | descriptors | points at |
|---|---|---|
| current `g` | real, imaginary | `current_re/im + phys[g] × stride` |
| current never written (structural zero) | zero, zero | the shared zero plane |
| momentum `g` | real, zero | `momenta + (g − value components) × stride` (form × 4 + Lorentz component) |
| model parameter `p` | broadcast, broadcast | the kernel's own planes filled with `p`'s real and imaginary parts (refilled when it changes) |
| output `c` | real, imaginary | `current_re/im + phys[c] × stride`, or `amplitude_re/im + c × stride` |

`stride` is the tile capacity, chosen at load time (at most 32 points). What is *not* saved is
`phys[]`: pyAmpliCol maps the canonical current components onto fewer arena planes by their
lifetimes (sources defined at event 0, stage k's outputs at event k after its inputs are read, the
amplitude stage reads last; planes assigned by first event, then component, each taking the lowest
plane whose current was last read strictly earlier). The map is deterministic, and `pacman` rebuilds
it: in the `u u~ > Z + 6g` helicity sum, 3,992 components share 2,300 planes (pyAmpliCol's profile
reports the same arena), and 15,228 of the 27,028 current descriptors point at another plane than
their saved component.

```
  execution helicity_sum: 13 kernels in 8 stages (the last computes the amplitudes)
    currents     3,992 components (68 of sources, 0 structural zeros, 3,992 live) in 2,300 arena planes
    remapped     15,228 of 27,028 current descriptors point at another plane than execution.json's component
    momenta      148 planes (37 forms × 4), model parameters 12, amplitudes 384

    leaf             in    out  current  momenta   zero bcast  kernel
    stage1:0        252    440      128       60     60     4  generic_stage_1_subset_2_8e85....plane.symjit
    ...
    stage5:0       3320   1024     3212       52     52     4  generic_stage_5_subset_6_chunk_0_3562....plane.symjit
    ...
    amplitude:0    1544    768     1544        0      0     0  generic_amplitude_stage_505b....plane.symjit
```

`--table` prints one row per descriptor: slot, `in`/`out`, buffer (`current_re`, `current_im`,
`amplitude_re`, `amplitude_im`, `momenta`, `zero`, `broadcast`), index in it, and the canonical
component (currents), the momentum form and component, or the broadcast parameter. The tests check
on the real artifacts that every table fits its kernel and that no stage writes a plane it reads or
writes twice (pyAmpliCol's no-aliasing contract). Recurrence and eager artifacts have no plane-arena
stages; `bindings` says so.

## Plane mappings (`pacman extract`)

To run a compiled-mode artifact's kernels without descriptor tables (e.g. as PTX with the plane
offsets compiled in), `extract` writes, next to each extracted plane kernel `X.plane.symjit` that a
stage calls, `X.plane.mapping`: one line per descriptor, `<index in states> <arena plane>`, the
inputs then the outputs, as `bindings --table` lists them. It reads the `processes/*/execution.json`
of the container's directory (so the container must be in its artifact); a container without them
gets no mappings, and `--no-mapping` turns them off.

pyAmpliCol keeps the planes in separate buffers, so the mappings number them as one arena per
execution, in this order: the current planes' real parts (`phys[]`, as above), their imaginary
parts, the amplitudes' real parts, their imaginary parts, the momenta, the zero plane, and two
planes per model parameter (real, imaginary) in place of pyAmpliCol's per-kernel broadcast planes.
Plane `i` is at `arena + i × stride`. Each process's `arena.txt` lists the regions of each of its
executions (`execution region first planes`); the executions have separate arenas, each numbered
from 0:

```
# execution region first planes
helicity_sum current_re 0 2300
helicity_sum current_im 2300 2300
helicity_sum amplitude_re 4600 384
helicity_sum amplitude_im 4984 384
helicity_sum momenta 5368 148
helicity_sum zero 5516 1
helicity_sum parameters 5517 24
```

Every stage kernel belongs to one execution (true of all the sample artifacts; `extract` stops
with an error if a kernel were called with two different tables). Kernels no stage calls (the
`generic_model_parameter_derivation` kernels of `pp_zjj`) and the plain `.symjit` copies get no
mapping. The call order of the kernels is `bindings`' leaf order. The tests compare every mapping of
the sample artifacts with the tables; the 47 mappings of `qq_z6g_compiled_jit_o3` (60,728 lines,
three executions) also agree with an independent Python rebuild of the tables.

## Replaying arena recordings (`pacman replay`)

A `pacman` build of RustiCol (pyAmpliCol's Rust runtime with a patch adding the cargo feature
`pacman`; `PACMAN_SAVE_ARENA=DIR`) saves the arena of a compiled-mode
execution right before its first kernel (`NAME.pre`) and after its last (`NAME.post`), with
`NAME.json`: the regions (the order of [Plane mappings](#plane-mappings-pacman-extract)),
RustiCol's `canonical_to_physical` current map, the source planes, the tiles (points per tile, in
order), the method (`all`, `materialized_helicity`, `correlated`, `validated`) and the schedule
(per stage the leaves run, by `application_path`). `pacman replay ARTIFACT RECORDING.json|DIR...`
(a directory means its `*.json`) runs those kernels on `.pre` and compares the result with
`.post` bit for bit (NaNs equal only with the same bits); the exit status is 1 if a recording
differs or cannot be replayed. It is the control for running the kernels elsewhere (the GPU):
`src/recording.rs` (reading, poisoning, comparing) and `replay::replay_with` (any executor of a
tile) do not depend on how a leaf runs.

How it runs the kernels (`src/replay.rs`), as `rusticol-core` does:
- **The execution**: the one of `processes/<process>/execution.json` whose leaves include every
  recorded leaf; the leaf tables come from `bindings`, but the current planes from the recording's
  `canonical_to_physical` (a disagreement with `bindings`' own map is a warning).
- **Loading** (`evaluator/symjit_plane.rs`, `load_bytes`): `Application::load` with RustiCol's
  explicit loader configuration (the saved one replaces it), `prepare_simd`, `seal`, the raw plane
  kernels; the SIMD kernel only if it supports indirect calls, with 2 or 4 lanes. Kernels load on
  all cores, once per run of `pacman`.
- **Calls** (`invoke_range`): per tile, each leaf in schedule order runs its SIMD kernel on whole
  lane blocks (a nonzero status redoes the block with the scalar kernel) and its scalar kernel on
  the rest. SIMD only if no input plane is also an output plane (`validate_raw_aliases`). Each
  tile runs in an arena of its own whose planes are the largest tile rounded up to 8 points apart,
  as RustiCol's (with `--whole`: the recording's arena, one tile).
- **Poison**: RustiCol can run a tile more than once (the helicity lane, the spin assignments of
  correlations), so `.pre` may already hold the outputs, and a replay writing nothing would pass.
  The planes whose first access in schedule order is a write are filled with
  `0x7ff8_dead_beef_0000 | plane` (shown as `poison(plane)`). Not every output plane: physical
  planes are reused, and some hold a source an early leaf reads (20 of the 330 written planes of
  a single-helicity recording). The planes read first must be sources, momenta, the zero plane or
  parameters; others are reported as a warning (stale data).

```
uubar_Z_6g_compiled.e0.r0: all, execution helicity_sum/color_selector[0], 256 points in 8 tiles, 13 leaves, SIMD 4 lanes
  281 planes read first, 5232 written first (poisoned)
  equal to .post: 5541 planes × 256 points, bit for bit
```

A difference is blamed on the leaf that writes the plane last (with the first differing point,
both values, the distance in ulps and relative to the plane's largest value); for the amplitudes,
which RustiCol returns, a line gives the largest difference relative to the largest amplitude
(many amplitudes vanish up to rounding, and so do their differences, which are then large relative
to themselves).

Results (2026-10-24, Zen 5, nine recordings):
- All 9 recordings (u u~ > Z + 6g compiled O3: the profile's colour-sector execution, 5,541
  planes × 256 points; a colour selection; four single helicities on the `top` execution; and
  g g > g g with a colour and a spin correlation, method `correlated`) reproduce `.post` bit for
  bit, with the recorded tiles and with `--whole`: the split into tiles does not change the bits.
- `--scalar` does not: the scalar and SIMD kernels round differently (cause not yet traced), by a few
  ulps on most planes, more on currents that cancel to ~1e-28. The amplitudes differ by at most
  6.7e-16 of the largest amplitude in the helicity sum, 5e-14 for single helicities (4 planes),
  5e-17 in g g > g g. A GPU kernel that matches the x86 scalar kernel (as `sjb2ptx`'s do) is
  therefore to be held to the `--scalar` replay bit for bit (`--dump`), and to `.post` within
  such bounds.
- Time (`--time --repeat 5`, one thread): 37.1–37.6 µs per point for the Z + 6g helicity sum,
  against 33–34 µs per point in RustiCol's profile; laid out as one 256-point arena instead of
  32-point tiles, 47.6.

`tests/pacman.rs` replays every `DIR/<case>/*.json` with `DIR/<case>/artifact` (a link to the
artifact) when `PACMAN_REPLAY_SAMPLES=DIR` is set; the tile runner
is tested on kernels compiled from Symbolica instructions in `replay::tests` (tiles, SIMD and
scalar counts, aliasing, poison against an executor that skips a leaf).

## Runners (`pacman runner`)

`pacman runner [OPTIONS] ARTIFACT [RECORDING.json]` writes one execution (a recording's, as
`replay` resolves it, or `--execution`) as C code and object files and builds two things from them
with `cc`:
- `runner`, an executable that runs recordings;
- `librunner.so`, a shared library with the execution-level interface `pacman-runner-v1`
  (`pacman_runner.h`), for RustiCol to load in place of its leaf loop. A CUDA runner is to export
  the same interface.

Performance is not the point: the CPU runner is the pathfinder for the CUDA one.

The files (`src/runner.rs`):

| file | what it holds |
|---|---|
| `kernels/k<k>.o` | each distinct leaf kernel through sjb2obj's code generator (`SavedModel::object`, in process): the **scalar** kernel `int pacman_kernel_<k>(double *mem, const symjit_slice *planes, size_t i, const double *params)`; pyAmpliCol's kernels call no library function, so the objects have no relocations |
| `runner.c` | per leaf its arena planes (the `.mapping` rows: inputs, then outputs), the leaf order, the leaves' kernel paths, the current map compiled in (`bindings`' map; with a recording, it must equal RustiCol's `canonical_to_physical`), the metadata and `pacman_runner_v1_run` |
| `pacman_runner.h` | the interface |
| `main.c` | `runner [--whole] [--no-poison] OUTDIR RECORDING.json...` |
| `Makefile` | the build `pacman` runs (`cc -O1`; `-shared` for the library) |

The interface:
- `pacman_runner_v1_metadata()`: version, size, plane and leaf counts, the regions, process,
  execution, per leaf its kernel path and planes, and the current map, so a caller can check that
  the library fits its arena.
- `pacman_runner_v1_run(planes, num_planes, leaves, num_leaves, point_start, point_count)`:
  - `planes`: one pointer per arena plane, numbered as the recordings and `.mapping` files are;
    model parameters are planes too, broadcast;
  - `leaves`: the leaves to run, in order (`NULL`: all), since the schedule varies with the
    method;
  - points `point_start..+point_count` of each plane; each leaf runs its scalar kernel once per
    point;
  - status 0, 1 for bad arguments, 2 if a kernel returns nonzero.

`runner` checks that a recording belongs to its arena (plane count and current map), takes the
schedule from the recording's `application_path`s, and poisons the planes written first in that
schedule as `replay` does. It runs per recorded tile (or once, `--whole`) and writes
`OUTDIR/NAME.arena`. `pacman replay --scalar --against OUTDIR` then compares it with the scalar
replay. Against `.post` it can only agree within the `--scalar` bounds above, since RustiCol runs
the SIMD kernels.

Results (2026-10-24): all 9 recordings, through runners of their executions, equal the scalar
replay bit for bit:
- the Z+6g colour sector: 13 leaves, 5.6 MB of kernel objects, 149 µs/point;
- Z+6g `top`: 21 leaves, of which the helicity schedules run 13, 2.4 µs/point;
- g g > g g: 4 leaves, 17 µs/point.

Generating takes 0.06 s and building 0.1 s. The `.so` is tested through `dlopen` (metadata,
argument errors, a point range split in two) in `runner::tests`, and the real recordings in
`tests/pacman.rs` (`PACMAN_REPLAY_SAMPLES`).

A `pacman` RustiCol build loads runners with `PACMAN_RUNNER=LIB[:LIB...]`.
Each compiled engine takes the library whose metadata matches it and calls it in place of its leaf
loop, in all four evaluate methods. Recordings made with runners
active equal `replay --scalar` bit for bit, for every method.

pyAmpliCol's own native compiled mode (`backend = "cpp"`/`"asm"`) loads one library per leaf
with the interface `pyamplicol-native-compiled-direct-application-v1` and keeps the leaf loop and
the 32-point tiles in RustiCol. The runner's interface is per execution instead, so that a GPU
runner can take the whole arena and any number of points (`RUNNER.md` §1).

## The container format

`pacbin-v1` is pyAmpliCol's (`rusticol-core/src/pacbin.rs`, 0BSD). Little endian: a 64-byte
header (`PACBIN\0\0`, version 1, payload alignment 64, index offset, member count), the payloads in
the order of their paths, each at a 64-byte boundary after zero padding, the index (a 32-byte
header `PACIDX\0\0`, then per member a 56-byte entry — path length, kind, flags, offset, length,
SHA-256 of the payload — its path and zero padding to 8 bytes; paths canonical, ASCII, sorted
bytewise and unique ignoring ASCII case), and a 64-byte footer (`PACEND\0\0`, the index offset and
count again, the SHA-256 of the index). The layout is canonical: no gaps, no slack, so a container
is determined by its members. `src/pacbin.rs` reads it with every check of pyAmpliCol's reader and
writes it as pyAmpliCol's writer does: written back, each of the seven containers of four example
artifacts and of the wheel's self-test gives the same bytes (`tests/pacman.rs`, `PACMAN_SAMPLES`).

The member kinds: 1 `symjit-application`, 2 `symbolica-exact-state`, 3 `native-library`,
4 `eager-runtime-metadata`, 5 `eager-runtime-table`, 7 `recurrence-direct-plan`,
8 `recurrence-color-projection-certificate`, 9 `on-the-fly-process-seed`, 10 `color-contraction`.

The artifact's manifest, `artifact.json`, authenticates `evaluators.pacbin`: its payload record
holds the file's SHA-256 and size, its `evaluator_payload_container` extension the index SHA-256,
the member count and the unpacked size, and the `artifact_id` is the SHA-256 of the canonical JSON
(`sort_keys`, `separators=(",", ":")`, a final newline) of
`{"kind": "pyamplicol-runtime-payload-identity", "schema_version": 1, "payloads": [...]}`, the
payload records of the roles `compiled-model`, `evaluator-manifest`, `evaluator-state` (the
container), `model-parameters` and `runtime-physics`, sorted by path. pyAmpliCol refuses an
artifact whose `artifact_id` does not match.

## Recompiling kernels: feasibility (2026-10-22)

The question was whether `pacman` can also recompile the kernels in a container. The short
answer: rewriting kernels and repacking the artifact is mechanical and verified; what limits the
gain is what pyAmpliCol accepts, and where the machine code comes from.

**What a kernel file holds.** A saved symjit application holds the configuration, the program
(sizes, constants, function table) and the MIR *after* optimization: peephole fusions and, at
opt levels 2 and 3, register allocation and stack-slot compaction are already applied. It holds
no machine code: `Application::load` compiles the scalar kernel (and the SIMD kernel, which the
mask in the file asks for) when pyAmpliCol loads it. Hence:

1. **Code generation is the runtime's, not the file's.** Every improvement in symjit's code
   generators (folding loads into arithmetic, fewer reloads, smaller encodings: the levers of the
   profile of 2026-10-21) reaches pyAmpliCol by linking a newer symjit into `rusticol-core`,
   without touching any artifact. Recompiling the files would not change it.
2. **Options read only by the code generators can be changed in place.** The option word sits at
   byte 0x20 of a version-3 file. Setting `enable_simd512` this way gave 8-lane kernels with
   bit-identical results and 1.56× faster plane kernels for the largest kernel (2026-10-21). But
   pyAmpliCol's loader refuses such kernels (`validate_application`: "enables unsupported SIMD512
   code generation") and checks the SIMD lane width (4 on x86), and it pins the other relevant bits
   (`symbolica`, `direct_arena`, `direct`, the arena operation, identity output, `complex`). Options
   that change the MIR (opt level, `cse`, `fastmath`, `compact`, `fast_complex`) cannot be changed
   this way: the MIR in the file was built with them.
3. **Re-optimizing needs a new MIR pass.** The expression the MIR came from is not in the file,
   and the saved MIR is register-allocated. A re-optimizer would rebuild the data flow from the
   linear MIR (feasible: these kernels have no loops; they are straight-line code except for the
   funclets of compression mode, in 212 of the 964 kernels of the `compress` artifact `pp_zjj`), then
   rerun allocation and the peephole passes of a newer symjit, or new ones (e.g. fewer spills,
   using all 16 registers). That is work in `stable`, not in pacman; pacman would only apply it.
4. **Re-translating from source is pyAmpliCol's job.** The `.evaluator.bin` members are
   Symbolica's own serialized evaluators (Symbolica 3.0.0's format, with the instructions symjit
   translated, and a copy of the plain kernel), and the plane kernels come from pyAmpliCol's own
   plane transformation of them. Rebuilding kernels from source means running pyAmpliCol's
   generator (with a different symjit or different options), not a container tool.

**Writing kernels back.** Verified on 2,236 kernels (all of the four example artifacts): loading a
version-3 kernel with stable and saving it gives version 4; deleting the stack limit and argument
count (bytes 40..56, Config) and setting the version to 3 gives the original file, except that
the block of real locations at its end (saved from a `HashSet`, in no fixed order, by 2.26.4 as
well) can come out in another order. So stable can write kernels pyAmpliCol's symjit 2.26.4 reads,
as long as the MIR uses no operation 2.26.4 lacks.

**Repacking.** `pacbin::write` reproduces pyAmpliCol's containers byte for byte. A changed
container must then be re-authenticated in `artifact.json`: the payload's `sha256` and
`size_bytes`, the extension's `index_sha256` and `unpacked_size_bytes`, and the recomputed
`artifact_id` (all derivable; `serde_json` is in the offline registry). Kernels referenced by
digest elsewhere (eager `DirectTable` kernels check their `evaluator_state_sha256`; plane
applications carry a `source_digest`, checked only for its form) need care if those members change,
and a rewritten plain kernel leaves the copy inside its `.evaluator.bin` as it was (rewriting only
the plane kernels, the ones pyAmpliCol runs, avoids both).

**What a `pacman recompile` could do now**, then: (a) set or clear code-generation-only option bits
(a pyAmpliCol build that accepts 8 lanes would run the AVX-512 kernels directly: the faster kernels
measured above); (b) apply a future MIR re-optimization pass; (c) repack and re-authenticate the
artifact. (a) and (c) are small; (b) is the substantial part and belongs in stable first.
