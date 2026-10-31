//! pacman runner: one compiled-mode execution of a pyAmpliCol artifact as C code and object
//! files, built into an executable that replays arena recordings and a shared library with the
//! execution-level interface `pacman-runner-v1` (`RUNNER_H`), which RustiCol can load in place
//! of its own leaf loop and which a CUDA runner is to export as well. See
//! `docs/PACMAN.md`.
//!
//! - The kernels are the leaves' plane kernels as object files (`SavedModel::object`, sjb2obj's
//!   code), scalar kernels only: `int k(double *mem, const symjit_slice *planes, size_t i,
//!   const double *params)`, called per point.
//! - `runner.c` holds the execution: per leaf the arena planes of its descriptor table (the
//!   `.mapping` rows: inputs, then outputs, in the numbering of `bindings::Arena`), the leaf
//!   order, the current map compiled in, and `pacman_runner_v1_run`.
//! - `--simd`, `--simd512`: the kernels are AVX2 or AVX-512 SIMD kernels instead
//!   (`SavedModel::object_simd`, 4 or 8 points per call, `simd_branch` on, no scalar kernel); the runner takes point ranges of multiples of 8 and
//!   runs every leaf on blocks of `PACMAN_RUNNER_BLOCK` points (default 64) at a time, the
//!   points of a call split over `PACMAN_RUNNER_THREADS` threads (default 1).
//! - `main.c` reads recordings (`NAME.json`, `NAME.pre`), runs their schedules per recorded tile
//!   and writes the arena as `NAME.arena` (the layout of `.post`), for `pacman replay
//!   --against`.
//! - The CUDA runner (`--cuda`) has the same interface, tables and `main.c`. Its kernels are one
//!   PTX module per leaf (`sjb2ptx` with the leaf's planes compiled in, `PtxOptions::mapping`),
//!   compiled by `ptxas` for the requested GPUs and packed by `fatbinary` into
//!   `kernels/leaf<k>.fatbin`, which `runner.c` embeds (`.incbin`). At run time it opens libcuda
//!   with `dlopen` (no CUDA toolkit needed), keeps one device arena of `NUM_PLANES` planes, copies
//!   in the planes the leaves read before writing them, runs the leaves in order and copies back
//!   the planes they write.

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

use crate::bindings::{leaf_label, Arena, Execution, Plane, Process};
use crate::replay::LeafCall;
use crate::SavedModel;

/// The interface version (`pacman_runner_metadata::abi_version`).
pub const ABI_VERSION: u32 = 1;

/// One execution, as the runner compiles it in.
#[derive(Clone, Debug)]
pub struct Spec {
    pub process: String,
    pub execution: String,
    pub arena: Arena,
    /// canonical current component → arena current plane (`u32::MAX`: none)
    pub canonical_to_physical: Vec<u32>,
    /// every leaf in execution order (`member`: the path in `evaluators.pacbin`)
    pub calls: Vec<LeafCall>,
    /// the leaves' kernels relative to the process directory (`application_path`, as the
    /// recordings' schedules name them)
    pub leaves: Vec<String>,
}

impl Spec {
    /// The execution `e` of `process`, with `bindings`' current map.
    pub fn new(process: &Process, e: &Execution) -> Result<Spec> {
        let arena = e.arena();
        let c2p = &e.canonical_to_physical;
        let mut calls = Vec::new();
        let mut leaves = Vec::new();
        for (stage, j, leaf) in e.leaves() {
            let label = leaf_label(&e.stages, stage, j);
            let planes = |list: &[Plane]| -> Result<Vec<u32>> {
                list.iter()
                    .map(|&p| {
                        arena
                            .index_with(p, leaf, c2p)
                            .ok_or_else(|| anyhow!("{}: {:?} has no arena plane", label, p))
                    })
                    .collect()
            };
            let inputs = planes(&leaf.inputs)?;
            let outputs = planes(&leaf.outputs)?;
            calls.push(LeafCall::new(
                format!("{}/{}", process.prefix, leaf.application),
                label,
                inputs,
                outputs,
            ));
            leaves.push(leaf.application.clone());
        }
        let name = process
            .prefix
            .strip_prefix("processes/")
            .unwrap_or(&process.prefix);
        Ok(Spec {
            process: name.to_string(),
            execution: e.name.clone(),
            arena,
            canonical_to_physical: c2p.clone(),
            calls,
            leaves,
        })
    }
}

/// The compiled kernels of a CPU runner (`objects`).
#[derive(Debug)]
pub struct Kernels {
    /// one object file per distinct kernel (`pacman_kernel_<k>`)
    pub objects: Vec<Vec<u8>>,
    /// for each leaf the index of its kernel
    pub of_leaf: Vec<usize>,
    /// points per kernel call: 1 (scalar), 4 (AVX2) or 8 (AVX-512)
    pub lanes: u32,
    /// the largest stack a kernel reserves, in bytes (SIMD kernels; 0 for scalar ones)
    pub stack: usize,
}

/// The kernels of `spec` as object files: one per distinct kernel, and for each leaf the index
/// of its kernel. `members` maps member paths to their bytes. `lanes`: 1 for scalar kernels, 4
/// or 8 for SIMD kernels (`SavedModel::object_simd`: AVX2, AVX-512) of that many points per
/// call. `settings`: configuration options for the kernels (`SavedModel::read_with`).
pub fn objects(
    spec: &Spec,
    members: &HashMap<&str, &[u8]>,
    lanes: u32,
    settings: &[(String, String)],
) -> Result<Kernels> {
    let mut objects = Vec::new();
    let mut stack = 0;
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut of_leaf = Vec::new();
    for call in spec.calls.iter() {
        if let Some(&k) = index.get(call.member.as_str()) {
            of_leaf.push(k);
            continue;
        }
        let bytes = members
            .get(call.member.as_str())
            .ok_or_else(|| anyhow!("evaluators.pacbin has no kernel `{}`", call.member))?;
        let model = SavedModel::read_with(&mut &bytes[..], settings)
            .with_context(|| format!("cannot read `{}`", call.member))?;
        let config = model.config();
        let prog = &model.prog;
        if !config.direct_arena() {
            bail!("`{}` is not a plane (direct-arena) kernel", call.member);
        }
        if (prog.count_states, prog.count_obs, prog.count_params)
            != (call.inputs.len(), call.outputs.len(), 0)
        {
            bail!(
                "{}: a table of {} input and {} output planes for `{}`, a kernel of {}, {} \
                 and {} parameters",
                call.label,
                call.inputs.len(),
                call.outputs.len(),
                call.member,
                prog.count_states,
                prog.count_obs,
                prog.count_params
            );
        }
        let k = objects.len();
        let target = model.object_target("")?;
        let name = format!("pacman_kernel_{}", k);
        let object = if lanes > 1 {
            model.object_simd(&name, target, lanes as usize)
        } else {
            model.object(&name, target)
        }
        .and_then(|o| {
            stack = stack.max(o.stack);
            o.build()
        })
        .with_context(|| format!("cannot compile `{}` into an object", call.member))?;
        objects.push(object);
        index.insert(&call.member, k);
        of_leaf.push(k);
    }
    Ok(Kernels {
        objects,
        of_leaf,
        lanes,
        stack,
    })
}

/// The kernels of `spec` as PTX modules for the CUDA runner: one per leaf (`pacman_leaf<k>`),
/// with its planes compiled in (inputs then outputs, as the `.mapping` rows), for `.target
/// sm_<sm>`; with `f32`, single-precision kernels on an arena of floats. `settings` as for
/// `objects`.
pub fn ptx_modules(
    spec: &Spec,
    members: &HashMap<&str, &[u8]>,
    sm: u32,
    f32: bool,
    settings: &[(String, String)],
) -> Result<Vec<String>> {
    let mut modules = Vec::new();
    for (k, call) in spec.calls.iter().enumerate() {
        let bytes = members
            .get(call.member.as_str())
            .ok_or_else(|| anyhow!("evaluators.pacbin has no kernel `{}`", call.member))?;
        let mut model = SavedModel::read_with(&mut &bytes[..], settings)
            .with_context(|| format!("cannot read `{}`", call.member))?;
        if !model.config().direct_arena() {
            bail!("`{}` is not a plane (direct-arena) kernel", call.member);
        }
        let mut options = crate::ptx::PtxOptions::new(&format!("pacman_leaf{}", k));
        options.sm = sm;
        options.f32 = f32;
        options.mapping = Some(call.inputs.iter().chain(&call.outputs).copied().collect());
        let ptx = model.ptx(&options).with_context(|| {
            format!("{}: cannot compile `{}` into PTX", call.label, call.member)
        })?;
        modules.push(ptx.text);
    }
    Ok(modules)
}

/// A C string literal.
fn c_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_ascii_graphic() || c == ' ' => out.push(c),
            c => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    let _ = write!(out, "\\{:03o}", b);
                }
            }
        }
    }
    out.push('"');
    out
}

/// `{a, b, ...}` in lines of 16.
fn c_array(values: &[u32]) -> String {
    let mut out = String::from("{");
    for (i, v) in values.iter().enumerate() {
        if i % 16 == 0 {
            out.push_str("\n    ");
        } else {
            out.push(' ');
        }
        let _ = write!(out, "{}u,", v);
    }
    out.push_str("\n}");
    out
}

/// The tables and metadata both runners compile in: the defines `NUM_PLANES`, `NUM_LEAVES`,
/// `MAX_TABLE`, per leaf its planes, input and output counts, kernel path, the current map and
/// `METADATA`.
fn tables_c(spec: &Spec) -> String {
    let mut s = String::new();
    let regions = spec.arena.regions();
    let max_table = spec
        .calls
        .iter()
        .map(|c| c.inputs.len() + c.outputs.len())
        .max()
        .unwrap_or(0);
    let _ = writeln!(
        s,
        "\n#define NUM_PLANES {}u\n#define NUM_LEAVES {}u\n#define MAX_TABLE {}\n",
        spec.arena.len(),
        spec.calls.len(),
        max_table
    );
    for (i, c) in spec.calls.iter().enumerate() {
        let planes: Vec<u32> = c.inputs.iter().chain(&c.outputs).copied().collect();
        let _ = writeln!(
            s,
            "/* {}: {} */\nstatic const uint32_t LEAF{}_PLANES[] = {};\n",
            c.label,
            c.member,
            i,
            c_array(&planes)
        );
    }
    s.push_str("static const uint32_t *const LEAF_PLANES[] = {\n");
    for i in 0..spec.calls.len() {
        let _ = writeln!(s, "    LEAF{}_PLANES,", i);
    }
    s.push_str("};\n\n");
    let inputs: Vec<u32> = spec.calls.iter().map(|c| c.inputs.len() as u32).collect();
    let outputs: Vec<u32> = spec.calls.iter().map(|c| c.outputs.len() as u32).collect();
    let _ = writeln!(
        s,
        "static const uint32_t LEAF_INPUTS[] = {};\n\nstatic const uint32_t LEAF_OUTPUTS[] = {};\n",
        c_array(&inputs),
        c_array(&outputs)
    );
    s.push_str("static const char *const LEAF_PATHS[] = {\n");
    for path in spec.leaves.iter() {
        let _ = writeln!(s, "    {},", c_string(path));
    }
    s.push_str("};\n\n");
    let c2p = if spec.canonical_to_physical.is_empty() {
        "{0u}".to_string()
    } else {
        c_array(&spec.canonical_to_physical)
    };
    let _ = writeln!(
        s,
        "static const uint32_t CANONICAL_TO_PHYSICAL[] = {};\n",
        c2p
    );
    let regions_c: Vec<String> = regions
        .iter()
        .map(|(_, first, n)| format!("{{{}u, {}u}}", first, n))
        .collect();
    let _ = writeln!(
        s,
        "static const pacman_runner_metadata METADATA = {{\n\
         \x20   PACMAN_RUNNER_ABI_VERSION,\n\
         \x20   (uint32_t)sizeof(pacman_runner_metadata),\n\
         \x20   NUM_PLANES,\n\
         \x20   NUM_LEAVES,\n\
         \x20   {{{}}},\n\
         \x20   {},\n\
         \x20   {},\n\
         \x20   LEAF_PATHS,\n\
         \x20   LEAF_PLANES,\n\
         \x20   LEAF_INPUTS,\n\
         \x20   LEAF_OUTPUTS,\n\
         \x20   CANONICAL_TO_PHYSICAL,\n\
         \x20   {}u,\n\
         \x20   0u,\n\
         }};\n",
        regions_c.join(", "),
        c_string(&spec.process),
        c_string(&spec.execution),
        spec.canonical_to_physical.len()
    );
    s
}

/// `runner.c`: the execution `spec` with the kernels `of_leaf` (from `objects`) of `lanes`
/// points per call.
pub fn runner_c(spec: &Spec, kernels: &Kernels) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "/* runner.c: the execution {} of {} ({} leaves, {} arena planes), generated by pacman {}.\n\
         \x20  The interface is in pacman_runner.h. */\n\n\
         #define _GNU_SOURCE /* pthread_getattr_np */\n\
         #include <stddef.h>\n\
         #include <stdint.h>\n\n\
         #include \"pacman_runner.h\"\n\n\
         typedef struct {{\n    double *ptr;\n    size_t len;\n}} symjit_slice;\n\n\
         typedef int (*kernel_fn)(double *, const symjit_slice *, size_t, const double *);\n",
        spec.execution,
        spec.process,
        spec.calls.len(),
        spec.arena.len(),
        env!("CARGO_PKG_VERSION")
    );
    for k in 0..kernels.objects.len() {
        let _ = writeln!(
            s,
            "int pacman_kernel_{}(double *, const symjit_slice *, size_t, const double *);",
            k
        );
    }
    s.push_str(&tables_c(spec));
    s.push_str("static const kernel_fn LEAF_KERNELS[] = {\n");
    for k in &kernels.of_leaf {
        let _ = writeln!(s, "    pacman_kernel_{},", k);
    }
    s.push_str("};\n\n");
    let _ = writeln!(
        s,
        "#define LANES {}\n#define KERNEL_STACK {}u /* the largest stack a kernel reserves */\n",
        kernels.lanes, kernels.stack
    );
    s.push_str(
        "/* SIMD kernels: PACMAN_RUNNER_BLOCK and PACMAN_RUNNER_THREADS when unset (a hybrid\n\
         \x20  runner's cpu.c sets its own) */\n\
         #ifndef DEFAULT_BLOCK\n#define DEFAULT_BLOCK 64u\n#endif\n\
         #ifndef DEFAULT_THREADS\n#define DEFAULT_THREADS 1\n#endif\n\n",
    );
    s.push_str(REDUCE_C);
    s.push_str(RUN_C);
    s
}

/// The fixed part of `runner.c`.
const RUN_C: &str = r#"const pacman_runner_metadata *pacman_runner_v1_metadata(void) {
    return &METADATA;
}

#if LANES > 1
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* PACMAN_RUNNER_STATS=1: calls, points and time, printed when the library is unloaded; the
   counters are updated under STATS_LOCK */
static struct {
    int on;
    unsigned long long calls, points;
    double seconds;
} STATS;
static pthread_mutex_t STATS_LOCK = PTHREAD_MUTEX_INITIALIZER;

#define MAX_THREADS 256

/* the settings, read once by `setup` (pthread_once), so that calls may come from several
   threads at once: PACMAN_RUNNER_BLOCK, PACMAN_RUNNER_THREADS, whether the CPU runs the
   kernels */
static uint32_t BLOCK, THREADS;
static int CPU_OK;
static pthread_once_t SETUP = PTHREAD_ONCE_INIT;

/* the stack a thread needs to run the kernels: the largest kernel's and 256 KiB for the C
   library's math functions and the runner */
#define STACK_NEEDED ((size_t)KERNEL_STACK + (256u << 10))
/* the stack of every worker thread: STACK_NEEDED, at least 1 MiB */
#define WORKER_STACK (STACK_NEEDED < (1u << 20) ? (size_t)(1u << 20) : STACK_NEEDED)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

__attribute__((destructor)) static void print_stats(void) {
    if (STATS.on && STATS.calls > 0) {
        fprintf(stderr,
                "pacman runner (%s of %s): %llu calls, %llu points, %u threads, %.3f s (%.2f "
                "us/point)\n",
                METADATA.execution, METADATA.process, STATS.calls, STATS.points, THREADS,
                STATS.seconds, 1e6 * STATS.seconds / STATS.points);
    }
}

static void setup(void) {
    /* whether the CPU runs the SIMD kernels (AVX2 and FMA; AVX-512: AVX512F and AVX512DQ) */
    __builtin_cpu_init();
#if LANES == 8
    CPU_OK = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
    const char *needs = "AVX512F and AVX512DQ";
#else
    CPU_OK = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    const char *needs = "AVX2 and FMA";
#endif
    if (!CPU_OK) {
        fprintf(stderr, "pacman runner (%s of %s): this CPU has no %s\n", METADATA.execution,
                METADATA.process, needs);
    }
    /* every leaf on a block of points at a time, so that the block's planes stay in the
       cache (PACMAN_RUNNER_BLOCK points, a multiple of 8, default DEFAULT_BLOCK, at most 2^20) */
    const char *env = getenv("PACMAN_RUNNER_BLOCK");
    uint32_t b = env != NULL ? (uint32_t)strtoul(env, NULL, 10) : DEFAULT_BLOCK;
    BLOCK = b < 8 ? 8 : b > (1u << 20) ? (1u << 20) : b / 8 * 8;
    /* the points of a call split over PACMAN_RUNNER_THREADS threads (default DEFAULT_THREADS;
       0: the online CPUs) */
    env = getenv("PACMAN_RUNNER_THREADS");
    long t = env != NULL ? (long)strtoul(env, NULL, 10) : DEFAULT_THREADS;
    if (t == 0) t = sysconf(_SC_NPROCESSORS_ONLN);
    THREADS = t < 1 ? 1 : t > MAX_THREADS ? MAX_THREADS : (uint32_t)t;
    const char *stats = getenv("PACMAN_RUNNER_STATS");
    STATS.on = stats != NULL && stats[0] != '\0' && strcmp(stats, "0") != 0;
}

/* the lowest address of the calling thread's stack (NULL: unknown), found once per thread */
static __thread char *STACK_LOW;
static __thread int STACK_KNOWN;

/* whether the calling thread has STACK_NEEDED bytes of stack left below this frame */
static int caller_has_room(void) {
    if (!STACK_KNOWN) {
        STACK_KNOWN = 1;
        pthread_attr_t attr;
        if (pthread_getattr_np(pthread_self(), &attr) == 0) {
            void *low;
            size_t size;
            if (pthread_attr_getstack(&attr, &low, &size) == 0) {
                STACK_LOW = low;
            }
            pthread_attr_destroy(&attr);
        }
    }
    char here;
    return STACK_LOW != NULL && STACK_LOW < &here && (size_t)(&here - STACK_LOW) > STACK_NEEDED;
}
#endif

/* the kernel of `leaf` on the points start..stop, LANES points per call */
static int run_leaf(uint32_t leaf, double *const *planes, uint32_t start, uint32_t stop) {
    symjit_slice table[MAX_TABLE];
    const uint32_t *map = LEAF_PLANES[leaf];
    uint32_t n = LEAF_INPUTS[leaf] + LEAF_OUTPUTS[leaf];
    for (uint32_t i = 0; i < n; i++) {
        table[i].ptr = planes[map[i]];
        table[i].len = stop;
    }
    kernel_fn kernel = LEAF_KERNELS[leaf];
    for (uint32_t point = start; point < stop; point += LANES) {
        if (kernel(NULL, table, point, NULL) != 0) {
            return PACMAN_RUNNER_KERNEL_STATUS;
        }
    }
    return PACMAN_RUNNER_OK;
}

/* every leaf of `leaves` (count of them) on the points start..stop, a block of points at a time */
static int run_points(double *const *planes, const uint32_t *leaves, uint32_t count,
                      uint32_t start, uint32_t stop, uint32_t block) {
    /* `first = last`, not `first += block`: a block may end past UINT32_MAX */
    for (uint32_t first = start, last; first < stop; first = last) {
        last = stop - first < block ? stop : first + block;
        for (uint32_t k = 0; k < count; k++) {
            int status = run_leaf(leaves == NULL ? k : leaves[k], planes, first, last);
            if (status != PACMAN_RUNNER_OK) {
                return status;
            }
        }
    }
    return PACMAN_RUNNER_OK;
}

#if LANES > 1
/* the points of one thread */
typedef struct {
    double *const *planes;
    const uint32_t *leaves;
    uint32_t count, start, stop;
    int status;
} share_t;

static void *run_share(void *arg) {
    share_t *share = arg;
    share->status =
        run_points(share->planes, share->leaves, share->count, share->start, share->stop, BLOCK);
    return NULL;
}

/* the points start..start+point_count on up to THREADS threads, contiguous ranges of whole
   8-point groups and at least BLOCK points each. The calling thread runs the first range if
   its stack has STACK_NEEDED bytes left; otherwise every range runs on a worker thread (of
   WORKER_STACK bytes), one even for a single range. */
static int run_threads(double *const *planes, const uint32_t *leaves, uint32_t count,
                       uint32_t start, uint32_t point_count) {
    uint32_t threads = point_count / BLOCK;
    threads = threads < 1 ? 1 : threads > THREADS ? THREADS : threads;
    int here = caller_has_room();
    if (threads == 1 && here) {
        return run_points(planes, leaves, count, start, start + point_count, BLOCK);
    }
    share_t shares[MAX_THREADS];
    pthread_t ids[MAX_THREADS];
    int started[MAX_THREADS];
    uint32_t groups = point_count / 8, point = start;
    for (uint32_t t = 0; t < threads; t++) {
        uint32_t n = groups / threads + (t < groups % threads);
        shares[t] = (share_t){planes, leaves, count, point, point + 8 * n, PACMAN_RUNNER_OK};
        point += 8 * n;
    }
    pthread_attr_t attr;
    int attr_ok = pthread_attr_init(&attr) == 0;
    if (attr_ok) {
        pthread_attr_setstacksize(&attr, WORKER_STACK);
    }
    uint32_t first = here ? 1 : 0; /* the first range on a worker thread */
    for (uint32_t t = first; t < threads; t++) {
        started[t] = pthread_create(&ids[t], attr_ok ? &attr : NULL, run_share, &shares[t]) == 0;
    }
    if (attr_ok) {
        pthread_attr_destroy(&attr);
    }
    if (here) {
        run_share(&shares[0]);
    }
    int status = PACMAN_RUNNER_OK;
    for (uint32_t t = 0; t < threads; t++) {
        if (t >= first) {
            if (started[t]) {
                pthread_join(ids[t], NULL);
            } else {
                run_share(&shares[t]); /* no thread: its points on this one */
            }
        }
        if (status == PACMAN_RUNNER_OK) {
            status = shares[t].status;
        }
    }
    return status;
}
#endif

int pacman_runner_v1_run(double *const *planes, uint32_t num_planes, const uint32_t *leaves,
                         uint32_t num_leaves, uint32_t point_start, uint32_t point_count) {
    if (planes == NULL || num_planes != NUM_PLANES || point_count == 0 ||
        point_start > UINT32_MAX - point_count) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        if (planes[p] == NULL) {
            return PACMAN_RUNNER_BAD_ARGUMENTS;
        }
    }
    uint32_t count = leaves == NULL ? NUM_LEAVES : num_leaves;
    for (uint32_t k = 0; k < count; k++) {
        if (leaves != NULL && leaves[k] >= NUM_LEAVES) {
            return PACMAN_RUNNER_BAD_ARGUMENTS;
        }
    }
#if LANES > 1
    /* SIMD kernels only: whole blocks of 8 points */
    if (point_start % 8 != 0 || point_count % 8 != 0) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    pthread_once(&SETUP, setup);
    if (!CPU_OK) {
        return PACMAN_RUNNER_DEVICE_ERROR;
    }
    double started = STATS.on ? now() : 0.0;
    int status = run_threads(planes, leaves, count, point_start, point_count);
    if (STATS.on && status == PACMAN_RUNNER_OK) {
        double seconds = now() - started;
        pthread_mutex_lock(&STATS_LOCK);
        STATS.calls++;
        STATS.points += point_count;
        STATS.seconds += seconds;
        pthread_mutex_unlock(&STATS_LOCK);
    }
    return status;
#else
    return run_points(planes, leaves, count, point_start, point_start + point_count, point_count);
#endif
}

int pacman_runner_v1_run_reduce(double *const *planes, uint32_t num_planes,
                                const uint32_t *leaves, uint32_t num_leaves, uint32_t point_start,
                                uint32_t point_count, const pacman_runner_reduction *reduction,
                                double *out) {
    if (!reduction_ok(reduction, out)) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    int status = pacman_runner_v1_run(planes, num_planes, leaves, num_leaves, point_start,
                                      point_count);
    if (status == PACMAN_RUNNER_OK) {
        reduce_host(planes, reduction, point_start, point_count, out);
    }
    return status;
}
"#;

/// The reduction's checks and the reduction on the host (`pacman_runner_v1_run_reduce`), in
/// both runners' `runner.c`; compiled with `-ffp-contract=off`.
const REDUCE_C: &str = r#"/* whether `r` is a well-formed reduction of this arena's amplitudes */
static int reduction_ok(const pacman_runner_reduction *r, const double *out) {
    if (r == NULL || out == NULL || r->struct_size != sizeof *r ||
        (r->num_groups > 0 && (r->offsets == NULL || r->weights == NULL))) {
        return 0;
    }
    if (r->num_groups == 0) return 1;
    if (r->offsets[0] != 0) return 0;
    for (uint32_t g = 0; g < r->num_groups; g++) {
        if (r->offsets[g + 1] < r->offsets[g]) return 0;
    }
    uint32_t terms = r->offsets[r->num_groups];
    if (terms > 0 && r->indices == NULL) return 0;
    for (uint32_t k = 0; k < terms; k++) {
        if (r->indices[k] >= METADATA.regions[PACMAN_AMPLITUDE_RE][1]) return 0;
    }
    return 1;
}

/* the reduction of the host planes, points start .. start + count, into out[0 .. count) */
static void reduce_host(double *const *planes, const pacman_runner_reduction *r, uint32_t start,
                        uint32_t count, double *out) {
    double *const *re_planes = planes + METADATA.regions[PACMAN_AMPLITUDE_RE][0];
    double *const *im_planes = planes + METADATA.regions[PACMAN_AMPLITUDE_IM][0];
    for (uint32_t i = 0; i < count; i++) out[i] = 0.0;
    for (uint32_t g = 0; g < r->num_groups; g++) {
        double weight = r->weights[g];
        for (uint32_t i = 0; i < count; i++) {
            double re = 0.0, im = 0.0;
            for (uint32_t k = r->offsets[g]; k < r->offsets[g + 1]; k++) {
                re += re_planes[r->indices[k]][start + i];
                im += im_planes[r->indices[k]][start + i];
            }
            out[i] += weight * (re * re + im * im);
        }
    }
}

"#;

/// `pacman_runner.h`: the interface.
pub const RUNNER_H: &str = r#"/* pacman_runner.h: the interface of a pacman runner (pacman-runner-v1): one compiled-mode
   execution of a pyAmpliCol artifact, run on an arena of planes.

   The arena is numbered as pacman's `.mapping` files and RustiCol's arena recordings: the
   current planes' real parts, their imaginary parts, the amplitudes' real parts, their
   imaginary parts, the momenta (form x 4 + Lorentz component), the zero plane, and two planes
   per model parameter (real, imaginary), each filled with the parameter's value. A plane holds
   one double per point; plane p of point i is planes[p][i]. */

#ifndef PACMAN_RUNNER_H
#define PACMAN_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PACMAN_RUNNER_ABI_VERSION 1u

/* status codes of pacman_runner_v1_run */
#define PACMAN_RUNNER_OK 0
#define PACMAN_RUNNER_BAD_ARGUMENTS 1   /* counts, null pointers, leaf indices, point range (a
                                           SIMD runner: start and count multiples of 8) */
#define PACMAN_RUNNER_KERNEL_STATUS 2   /* a kernel returned a nonzero status */
#define PACMAN_RUNNER_DEVICE_ERROR 3    /* the GPU (a CUDA runner) or a CPU without AVX2 (a SIMD
                                           runner): see standard error */

/* the arena's regions, in order (the indices of `regions`) */
enum {
    PACMAN_CURRENT_RE, PACMAN_CURRENT_IM, PACMAN_AMPLITUDE_RE, PACMAN_AMPLITUDE_IM,
    PACMAN_MOMENTA, PACMAN_ZERO, PACMAN_PARAMETERS, PACMAN_REGIONS
};

typedef struct {
    uint32_t abi_version;               /* PACMAN_RUNNER_ABI_VERSION */
    uint32_t struct_size;               /* sizeof(pacman_runner_metadata) */
    uint32_t num_planes;
    uint32_t num_leaves;
    uint32_t regions[PACMAN_REGIONS][2];  /* first plane, planes */
    const char *process;
    const char *execution;              /* e.g. "helicity_sum/color_selector[0]" */
    const char *const *leaves;          /* per leaf, its kernel (`application_path`) */
    const uint32_t *const *leaf_planes; /* per leaf, its arena planes: inputs, then outputs */
    const uint32_t *leaf_inputs;        /* per leaf, its input planes */
    const uint32_t *leaf_outputs;       /* per leaf, its output planes */
    const uint32_t *canonical_to_physical;  /* current component -> current plane (UINT32_MAX: none) */
    uint32_t num_components;
    uint32_t reserved;                  /* 0 */
} pacman_runner_metadata;

const pacman_runner_metadata *pacman_runner_v1_metadata(void);

/* Runs the leaves leaves[0 .. num_leaves) (indices into metadata->leaves, in this order; NULL
   for all leaves in execution order) on points point_start .. point_start + point_count of
   `planes` (num_planes pointers, one per arena plane). Returns a PACMAN_RUNNER_* status. */
int pacman_runner_v1_run(double *const *planes, uint32_t num_planes, const uint32_t *leaves,
                         uint32_t num_leaves, uint32_t point_start, uint32_t point_count);

/* The reduction of the amplitudes into one value per point that RustiCol applies after the
   leaves (`reduce_planes_f64_into_selected_slice` with coherent groups, or without groups: a
   group per amplitude). For point i:

       value = 0.0
       for each group g in order:
           re = 0.0, im = 0.0
           for k in offsets[g] .. offsets[g + 1]: re += Re A[indices[k]], im += Im A[indices[k]]
           value += weights[g] * (re * re + im * im)

   with the operations in this order and rounded one by one (no fused multiply-add), so that the
   value equals RustiCol's bit for bit. A is the amplitude region: Re A[j] is arena plane
   regions[PACMAN_AMPLITUDE_RE][0] + j, Im A[j] plane regions[PACMAN_AMPLITUDE_IM][0] + j. */
typedef struct {
    uint32_t struct_size;     /* sizeof(pacman_runner_reduction) */
    uint32_t num_groups;
    const uint32_t *offsets;  /* num_groups + 1, offsets[0] = 0, nondecreasing */
    const uint32_t *indices;  /* offsets[num_groups] amplitude indices */
    const double *weights;    /* num_groups */
} pacman_runner_reduction;

/* Runs the leaves as pacman_runner_v1_run, then the reduction: out[i] for point point_start + i
   (point_count values). The amplitude planes are left as they are in `planes` (a CUDA runner
   does not copy them back, unless PACMAN_RUNNER_COPY_BACK=all or PACMAN_SAVE_ARENA is set).
   Optional: runners of pacman 0.3.0 lack it (look it up with dlsym). */
int pacman_runner_v1_run_reduce(double *const *planes, uint32_t num_planes,
                                const uint32_t *leaves, uint32_t num_leaves, uint32_t point_start,
                                uint32_t point_count, const pacman_runner_reduction *reduction,
                                double *out);

#ifdef __cplusplus
}
#endif

#endif
"#;

/// `main.c`: the executable that replays recordings.
pub const MAIN_C: &str = r#"/* main.c: runs arena recordings of RustiCol (NAME.json, NAME.pre) through the runner and
   writes the arena after the run as OUTDIR/NAME.arena (the layout of NAME.post), for
   `pacman replay --against OUTDIR`. Generated by pacman.

       runner [--whole] [--no-poison] [--reduce FILE] OUTDIR RECORDING.json...

   The schedule (the recording's leaves) runs per recorded tile, or on all points at once
   (--whole). Planes the schedule writes before it reads them are first filled with NaNs
   (0x7ff8deadbeef0000 | plane), unless --no-poison. With --reduce, the calls are
   pacman_runner_v1_run_reduce with the reduction of FILE (a line per group: its weight, then its
   amplitude indices), and the values (one double per point) go to OUTDIR/NAME.reduced too. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pacman_runner.h"

#define POISON 0x7ff8deadbeef0000ull

static char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "runner: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = malloc(n + 1);
    if (data == NULL || fread(data, 1, n, f) != (size_t)n) {
        fprintf(stderr, "runner: cannot read %s\n", path);
        fclose(f);
        free(data);
        return NULL;
    }
    fclose(f);
    data[n] = 0;
    *size = n;
    return data;
}

/* the value of the first "key" of the JSON text (the recordings' keys are unique) */
static const char *value_of(const char *json, const char *key) {
    char quoted[64];
    snprintf(quoted, sizeof quoted, "\"%s\"", key);
    const char *p = strstr(json, quoted);
    if (p == NULL) {
        return NULL;
    }
    p += strlen(quoted);
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    if (*p != ':') {
        return NULL;
    }
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

static int number(const char *json, const char *key, uint64_t *out) {
    const char *p = value_of(json, key);
    char *end;
    if (p == NULL || *p < '0' || *p > '9') {
        return 0;
    }
    *out = strtoull(p, &end, 10);
    return 1;
}

/* a JSON array of numbers; *count values, malloc'd */
static uint32_t *numbers(const char *json, const char *key, size_t *count) {
    const char *p = value_of(json, key);
    if (p == NULL || *p != '[') {
        return NULL;
    }
    size_t cap = 16, n = 0;
    uint32_t *v = malloc(cap * sizeof *v);
    p++;
    for (;;) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (*p == ']') {
            break;
        }
        char *end;
        unsigned long long x = strtoull(p, &end, 10);
        if (end == p) {
            free(v);
            return NULL;
        }
        if (n == cap) {
            cap *= 2;
            v = realloc(v, cap * sizeof *v);
        }
        v[n++] = (uint32_t)x;
        p = end;
    }
    *count = n;
    return v;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* --reduce FILE: the reduction */
static pacman_runner_reduction REDUCTION;

static int read_reduction(const char *path) {
    size_t size;
    char *text = read_file(path, &size);
    if (text == NULL) {
        return 0;
    }
    size_t groups = 0, terms = 0, cap = 16;
    uint32_t *offsets = malloc((cap + 1) * sizeof *offsets), *indices = malloc(cap * sizeof *indices);
    double *weights = malloc(cap * sizeof *weights);
    offsets[0] = 0;
    for (char *line = strtok(text, "\n"); line != NULL; line = strtok(NULL, "\n")) {
        char *p = line, *end;
        double w = strtod(p, &end);
        if (end == p) continue; /* a blank line */
        if (groups == cap || terms + 64 > cap) {
            cap = 2 * cap + 64;
            offsets = realloc(offsets, (cap + 1) * sizeof *offsets);
            indices = realloc(indices, cap * sizeof *indices);
            weights = realloc(weights, cap * sizeof *weights);
        }
        weights[groups] = w;
        for (p = end;;) {
            unsigned long k = strtoul(p, &end, 10);
            if (end == p) break;
            if (terms == cap) {
                cap *= 2;
                indices = realloc(indices, cap * sizeof *indices);
            }
            indices[terms++] = (uint32_t)k;
            p = end;
        }
        offsets[++groups] = (uint32_t)terms;
    }
    free(text);
    REDUCTION.struct_size = sizeof REDUCTION;
    REDUCTION.num_groups = (uint32_t)groups;
    REDUCTION.offsets = offsets;
    REDUCTION.indices = indices;
    REDUCTION.weights = weights;
    return 1;
}

static int run_recording(const char *json_path, const char *outdir, int whole, int poison,
                         int reduce) {
    const pacman_runner_metadata *md = pacman_runner_v1_metadata();
    size_t size;
    char *json = read_file(json_path, &size);
    if (json == NULL) {
        return 1;
    }
    int status = 1;
    uint64_t planes = 0, points = 0;
    size_t num_tiles = 0, num_c2p = 0, num_leaves = 0;
    uint32_t *tiles = NULL, *c2p = NULL, *schedule = NULL;
    double *arena = NULL, *values = NULL;
    double **table = NULL;
    uint8_t *first = NULL;
    char *pre = NULL;

    if (!number(json, "num_planes", &planes) || !number(json, "num_points", &points) ||
        (tiles = numbers(json, "tiles", &num_tiles)) == NULL ||
        (c2p = numbers(json, "canonical_to_physical", &num_c2p)) == NULL) {
        fprintf(stderr, "runner: %s: not a pacman arena recording\n", json_path);
        goto done;
    }
    if (planes != md->num_planes || num_c2p != md->num_components ||
        memcmp(c2p, md->canonical_to_physical, num_c2p * sizeof *c2p) != 0) {
        fprintf(stderr, "runner: %s: a recording of another arena than %s of %s\n", json_path,
                md->execution, md->process);
        goto done;
    }

    /* the schedule: the leaves by kernel path, in order */
    schedule = malloc((md->num_leaves + 1) * sizeof *schedule);
    for (const char *p = json; (p = strstr(p, "\"application_path\"")) != NULL;) {
        const char *v = value_of(p, "application_path");
        if (v == NULL || *v != '"') {
            fprintf(stderr, "runner: %s: a malformed schedule\n", json_path);
            goto done;
        }
        v++;
        const char *end = strchr(v, '"');
        size_t len = end - v;
        uint32_t leaf = md->num_leaves;
        for (uint32_t i = 0; i < md->num_leaves; i++) {
            if (strlen(md->leaves[i]) == len && strncmp(md->leaves[i], v, len) == 0) {
                leaf = i;
                break;
            }
        }
        if (leaf == md->num_leaves || num_leaves == md->num_leaves) {
            fprintf(stderr, "runner: %s: the schedule runs a leaf %.*s not in %s\n", json_path,
                    (int)len, v, md->execution);
            goto done;
        }
        schedule[num_leaves++] = leaf;
        p = end;
    }

    /* the arena before the run */
    char path[4096];
    size_t stem = strlen(json_path) - (strlen(json_path) > 5 &&
                  strcmp(json_path + strlen(json_path) - 5, ".json") == 0 ? 5 : 0);
    snprintf(path, sizeof path, "%.*s.pre", (int)stem, json_path);
    pre = read_file(path, &size);
    if (pre == NULL) {
        goto done;
    }
    if (size != planes * points * sizeof(double)) {
        fprintf(stderr, "runner: %s holds %zu bytes, expected %llu\n", path, size,
                (unsigned long long)(planes * points * sizeof(double)));
        goto done;
    }
    arena = (double *)pre;

    /* poison the planes whose first access in the schedule is a write */
    first = calloc(planes, 1);
    for (size_t k = 0; k < num_leaves; k++) {
        uint32_t leaf = schedule[k];
        const uint32_t *map = md->leaf_planes[leaf];
        for (uint32_t i = 0; i < md->leaf_inputs[leaf]; i++) {
            if (first[map[i]] == 0) first[map[i]] = 1;
        }
        for (uint32_t i = md->leaf_inputs[leaf]; i < md->leaf_inputs[leaf] + md->leaf_outputs[leaf]; i++) {
            if (first[map[i]] == 0) first[map[i]] = 2;
        }
    }
    uint32_t poisoned = 0;
    for (uint64_t p = 0; poison && p < planes; p++) {
        if (first[p] != 2) continue;
        uint64_t bits = POISON | p;
        double value;
        memcpy(&value, &bits, sizeof value);
        for (uint64_t i = 0; i < points; i++) arena[p * points + i] = value;
        poisoned++;
    }

    /* run, per tile or once */
    table = malloc(planes * sizeof *table);
    values = calloc(points > 0 ? points : 1, sizeof *values);
    double start = seconds();
    uint64_t offset = 0;
    size_t calls = whole ? 1 : num_tiles;
    for (size_t t = 0; t < calls; t++) {
        uint64_t n = whole ? points : tiles[t];
        if (offset + n > points) {
            fprintf(stderr, "runner: %s: the tiles do not fit the points\n", json_path);
            goto done;
        }
        for (uint64_t p = 0; p < planes; p++) table[p] = arena + p * points + offset;
        int s = reduce ? pacman_runner_v1_run_reduce(table, (uint32_t)planes, schedule,
                                                     (uint32_t)num_leaves, 0, (uint32_t)n,
                                                     &REDUCTION, values + offset)
                       : pacman_runner_v1_run(table, (uint32_t)planes, schedule,
                                              (uint32_t)num_leaves, 0, (uint32_t)n);
        if (s != PACMAN_RUNNER_OK) {
            fprintf(stderr, "runner: %s: tile %zu: status %d\n", json_path, t, s);
            goto done;
        }
        offset += n;
    }
    double elapsed = seconds() - start;

    /* OUTDIR/NAME.arena */
    const char *base = strrchr(json_path, '/');
    base = base == NULL ? json_path : base + 1;
    size_t name = stem - (base - json_path);
    snprintf(path, sizeof path, "%s/%.*s.arena", outdir, (int)name, base);
    FILE *f = fopen(path, "wb");
    if (f == NULL || fwrite(arena, sizeof(double), planes * points, f) != planes * points) {
        fprintf(stderr, "runner: cannot write %s\n", path);
        if (f != NULL) fclose(f);
        goto done;
    }
    fclose(f);
    if (reduce) {
        snprintf(path, sizeof path, "%s/%.*s.reduced", outdir, (int)name, base);
        f = fopen(path, "wb");
        if (f == NULL || fwrite(values, sizeof(double), points, f) != points) {
            fprintf(stderr, "runner: cannot write %s\n", path);
            if (f != NULL) fclose(f);
            goto done;
        }
        fclose(f);
    }
    printf("%.*s: %zu leaves on %llu points in %zu call%s, %u planes poisoned: %.1f us, "
           "%.3f us/point -> %s\n",
           (int)name, base, num_leaves, (unsigned long long)points, calls, calls == 1 ? "" : "s",
           poisoned, 1e6 * elapsed, 1e6 * elapsed / points, path);
    status = 0;

done:
    free(json);
    free(pre);
    free(tiles);
    free(c2p);
    free(schedule);
    free(table);
    free(values);
    free(first);
    return status;
}

int main(int argc, char **argv) {
    int whole = 0, poison = 1, reduce = 0, i = 1;
    /* a CUDA runner: the arenas written must be complete, currents included */
    setenv("PACMAN_RUNNER_COPY_BACK", "all", 1);
    for (; i < argc && argv[i][0] == '-'; i++) {
        if (strcmp(argv[i], "--whole") == 0) {
            whole = 1;
        } else if (strcmp(argv[i], "--no-poison") == 0) {
            poison = 0;
        } else if (strcmp(argv[i], "--reduce") == 0 && i + 1 < argc) {
            if (!read_reduction(argv[++i])) return 2;
            reduce = 1;
        } else {
            fprintf(stderr, "runner: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (argc - i < 2) {
        const pacman_runner_metadata *md = pacman_runner_v1_metadata();
        fprintf(stderr,
                "usage: %s [--whole] [--no-poison] [--reduce FILE] OUTDIR RECORDING.json...\n"
                "runs recordings of the execution %s of %s (%u leaves, %u planes)\n",
                argv[0], md->execution, md->process, md->num_leaves, md->num_planes);
        return 2;
    }
    int status = 0;
    for (int k = i + 1; k < argc; k++) {
        status |= run_recording(argv[k], argv[i], whole, poison, reduce);
    }
    return status;
}
"#;

/// The `Makefile` that rebuilds the runner with `make` (the commands `build` runs).
pub fn makefile(kernels: usize) -> String {
    let objects: Vec<String> = (0..kernels).map(|k| format!("kernels/k{}.o", k)).collect();
    format!(
        "# generated by pacman runner\n\
         CC ?= cc\n\
         CFLAGS ?= -O1\n\
         KERNELS = {}\n\n\
         all: runner librunner.so\n\n\
         runner.o: runner.c pacman_runner.h\n\
         \t$(CC) $(CFLAGS) -ffp-contract=off -pthread -fPIC -c runner.c -o runner.o\n\n\
         librunner.so: runner.o $(KERNELS)\n\
         \t$(CC) -pthread -shared -o librunner.so runner.o $(KERNELS)\n\n\
         runner: main.c runner.o $(KERNELS) pacman_runner.h\n\
         \t$(CC) $(CFLAGS) -pthread -o runner main.c runner.o $(KERNELS)\n",
        objects.join(" ")
    )
}

/// Writes the runner of `spec` into `dir`: `kernels/k<k>.o`, `runner.c`, `pacman_runner.h`,
/// `main.c`, `Makefile`.
pub fn write(dir: &Path, spec: &Spec, kernels: &Kernels) -> Result<()> {
    let write = |name: &str, bytes: &[u8]| -> Result<()> {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create `{}`", parent.display()))?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("cannot write `{}`", path.display()))
    };
    for (k, o) in kernels.objects.iter().enumerate() {
        write(&format!("kernels/k{}.o", k), o)?;
    }
    write("runner.c", runner_c(spec, kernels).as_bytes())?;
    write("pacman_runner.h", RUNNER_H.as_bytes())?;
    write("main.c", MAIN_C.as_bytes())?;
    write("Makefile", makefile(kernels.objects.len()).as_bytes())?;
    Ok(())
}

/// Builds `runner` and `librunner.so` in `dir` with the C compiler `cc`, as the `Makefile` does.
pub fn build(dir: &Path, cc: &str, kernels: usize) -> Result<()> {
    let objects: Vec<String> = (0..kernels).map(|k| format!("kernels/k{}.o", k)).collect();
    let steps: Vec<Vec<String>> = vec![
        [
            "-O1",
            "-ffp-contract=off",
            "-pthread",
            "-fPIC",
            "-c",
            "runner.c",
            "-o",
            "runner.o",
        ]
        .map(String::from)
        .to_vec(),
        ["-pthread", "-shared", "-o", "librunner.so", "runner.o"]
            .map(String::from)
            .into_iter()
            .chain(objects.iter().cloned())
            .collect(),
        ["-O1", "-pthread", "-o", "runner", "main.c", "runner.o"]
            .map(String::from)
            .into_iter()
            .chain(objects.iter().cloned())
            .collect(),
    ];
    for args in steps {
        let out = Command::new(cc)
            .args(&args)
            .current_dir(dir)
            .output()
            .with_context(|| format!("cannot run `{}`", cc))?;
        if !out.status.success() {
            bail!(
                "`{} {}` failed:\n{}",
                cc,
                args.join(" "),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    Ok(())
}

/// `runner.c` of the CUDA runner: the execution `spec`, the leaves' fatbinaries
/// (`kernels/leaf<k>.fatbin`, embedded) and `pacman_runner_v1_run` on the GPU. With `f32` the
/// leaves are single precision: the f64 planes RustiCol passes are copied as before into the f64
/// device arena, converted into an f32 one for the leaves, and the planes copied back are
/// converted back to f64 first (`CONVERT_PTX`).
pub fn runner_cuda_c(spec: &Spec, f32: bool, fallback: Option<(f64, u32)>) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "/* runner.c: the execution {} of {} ({} leaves, {} arena planes) on the GPU, generated by\n\
         \x20  pacman {}. The interface is in pacman_runner.h. */\n\n\
         #include <dlfcn.h>\n\
         #include <stddef.h>\n\
         #include <stdint.h>\n\
         #include <stdio.h>\n\
         #include <stdlib.h>\n\
         #include <string.h>\n\
         #include <time.h>\n\n\
         #include \"pacman_runner.h\"\n",
        spec.execution,
        spec.process,
        spec.calls.len(),
        spec.arena.len(),
        env!("CARGO_PKG_VERSION")
    );
    let _ = writeln!(
        s,
        "/* 1: the leaves are f32 kernels on an f32 copy of the arena */\n#define F32 {}\n",
        f32 as u8
    );
    let (fraction, lanes) = fallback.unwrap_or((0.0, 1));
    let _ = writeln!(
        s,
        "/* 1: a hybrid runner, whose reductions rerun their largest points on the CPU (cpu.c),\n\
         \x20  FALLBACK_DEFAULT of them (PACMAN_RUNNER_FALLBACK), with kernels of CPU_LANES points */\n\
         #define HYBRID {}\n#define FALLBACK_DEFAULT {:?}\n#define CPU_LANES {}u\n",
        fallback.is_some() as u8,
        fraction,
        lanes
    );
    s.push_str(&tables_c(spec));
    if f32 {
        s.push_str(
            "/* the conversions between the f64 and the f32 arenas (PTX, compiled by the driver) */\n",
        );
        s.push_str("static const char CONVERT_PTX[] =\n");
        for line in CONVERT_PTX.lines() {
            let _ = writeln!(s, "    \"{}\\n\"", line.replace('\t', "\\t"));
        }
        s.push_str("    ;\n\n");
    }
    s.push_str(
        "/* the reduction (PTX, compiled by the driver) */\nstatic const char REDUCE_PTX[] =\n",
    );
    for line in REDUCE_PTX.lines() {
        let _ = writeln!(s, "    \"{}\\n\"", line.replace('\t', "\\t"));
    }
    s.push_str("    ;\n\n");
    s.push_str("/* the leaves' kernels: fatbinaries of their cubins, embedded */\n__asm__(\n");
    s.push_str("    \".section .rodata\\n\"\n");
    for k in 0..spec.calls.len() {
        let _ = writeln!(
            s,
            "    \".balign 64\\n.globl pacman_leaf{0}_fatbin\\n.hidden pacman_leaf{0}_fatbin\\n\
             pacman_leaf{0}_fatbin:\\n.incbin \\\"kernels/leaf{0}.fatbin\\\"\\n\"",
            k
        );
    }
    s.push_str("    \".previous\\n\");\n\n");
    for k in 0..spec.calls.len() {
        let _ = writeln!(s, "extern const unsigned char pacman_leaf{}_fatbin[];", k);
    }
    s.push_str("\nstatic const unsigned char *const LEAF_FATBINS[] = {\n");
    for k in 0..spec.calls.len() {
        let _ = writeln!(s, "    pacman_leaf{}_fatbin,", k);
    }
    s.push_str("};\n\n");
    s.push_str(REDUCE_C);
    s.push_str(RUN_CUDA_C);
    s
}

/// The f32 runner's conversion kernels, `pacman_to_f32` and `pacman_to_f64`, each
/// `(src, dst, planes, count, n, ld)`: for the `count` planes listed in `planes` (u32; y: one
/// plane per block row, strided by the grid) and the points `0 .. n` (x: one thread per point),
/// `dst[p * ld + i] = src[p * ld + i]`, rounded to nearest into f32 (exact back to f64).
const CONVERT_PTX: &str = "\
.version 8.3
.target sm_70
.address_size 64
.visible .entry pacman_to_f32(.param .u64 src, .param .u64 dst, .param .u64 planes,
    .param .u64 count, .param .u64 n, .param .u64 ld)
{
    .reg .pred %p<2>;
    .reg .u32 %t<4>;
    .reg .u64 %a<10>;
    .reg .f64 %d;
    .reg .f32 %f;
    mov.u32 %t0, %ctaid.x;
    mov.u32 %t1, %ntid.x;
    mov.u32 %t2, %tid.x;
    mul.wide.u32 %a0, %t0, %t1;
    cvt.u64.u32 %a1, %t2;
    add.u64 %a0, %a0, %a1;
    ld.param.u64 %a1, [n];
    setp.ge.u64 %p0, %a0, %a1;
    @%p0 bra $Ldone;
    ld.param.u64 %a2, [src];
    cvta.to.global.u64 %a2, %a2;
    ld.param.u64 %a3, [dst];
    cvta.to.global.u64 %a3, %a3;
    ld.param.u64 %a4, [planes];
    cvta.to.global.u64 %a4, %a4;
    ld.param.u64 %a5, [count];
    ld.param.u64 %a6, [ld];
    mov.u32 %t3, %ctaid.y;
    cvt.u64.u32 %a7, %t3;
    mov.u32 %t3, %nctaid.y;
    cvt.u64.u32 %a8, %t3;
$Lloop:
    setp.ge.u64 %p1, %a7, %a5;
    @%p1 bra $Ldone;
    shl.b64 %a9, %a7, 2;
    add.u64 %a9, %a9, %a4;
    ld.global.u32 %t3, [%a9];
    cvt.u64.u32 %a9, %t3;
    mul.lo.u64 %a9, %a9, %a6;
    add.u64 %a9, %a9, %a0;
    shl.b64 %a1, %a9, 3;
    add.u64 %a1, %a1, %a2;
    ld.global.f64 %d, [%a1];
    cvt.rn.f32.f64 %f, %d;
    shl.b64 %a1, %a9, 2;
    add.u64 %a1, %a1, %a3;
    st.global.f32 [%a1], %f;
    add.u64 %a7, %a7, %a8;
    bra.uni $Lloop;
$Ldone:
    ret;
}
.visible .entry pacman_to_f64(.param .u64 src, .param .u64 dst, .param .u64 planes,
    .param .u64 count, .param .u64 n, .param .u64 ld)
{
    .reg .pred %p<2>;
    .reg .u32 %t<4>;
    .reg .u64 %a<10>;
    .reg .f64 %d;
    .reg .f32 %f;
    mov.u32 %t0, %ctaid.x;
    mov.u32 %t1, %ntid.x;
    mov.u32 %t2, %tid.x;
    mul.wide.u32 %a0, %t0, %t1;
    cvt.u64.u32 %a1, %t2;
    add.u64 %a0, %a0, %a1;
    ld.param.u64 %a1, [n];
    setp.ge.u64 %p0, %a0, %a1;
    @%p0 bra $Ldone;
    ld.param.u64 %a2, [src];
    cvta.to.global.u64 %a2, %a2;
    ld.param.u64 %a3, [dst];
    cvta.to.global.u64 %a3, %a3;
    ld.param.u64 %a4, [planes];
    cvta.to.global.u64 %a4, %a4;
    ld.param.u64 %a5, [count];
    ld.param.u64 %a6, [ld];
    mov.u32 %t3, %ctaid.y;
    cvt.u64.u32 %a7, %t3;
    mov.u32 %t3, %nctaid.y;
    cvt.u64.u32 %a8, %t3;
$Lloop:
    setp.ge.u64 %p1, %a7, %a5;
    @%p1 bra $Ldone;
    shl.b64 %a9, %a7, 2;
    add.u64 %a9, %a9, %a4;
    ld.global.u32 %t3, [%a9];
    cvt.u64.u32 %a9, %t3;
    mul.lo.u64 %a9, %a9, %a6;
    add.u64 %a9, %a9, %a0;
    shl.b64 %a1, %a9, 2;
    add.u64 %a1, %a1, %a2;
    ld.global.f32 %f, [%a1];
    cvt.f64.f32 %d, %f;
    shl.b64 %a1, %a9, 3;
    add.u64 %a1, %a1, %a3;
    st.global.f64 [%a1], %d;
    add.u64 %a7, %a7, %a8;
    bra.uni $Lloop;
$Ldone:
    ret;
}
";

/// The CUDA runner's reduction kernel, `pacman_reduce(arena, ld, n, re0, im0, groups, offsets,
/// indices, weights, out)`: one thread per point `i < n`, the loop of `pacman_runner.h` over the
/// f64 arena (plane `p` of point `i` at `arena[p * ld + i]`; `re0`/`im0` the first real and
/// imaginary amplitude planes), `out[i]` the value. `.rn` on every operation: ptxas fuses none.
const REDUCE_PTX: &str = "\
.version 8.3
.target sm_70
.address_size 64
.visible .entry pacman_reduce(.param .u64 arena, .param .u64 ld, .param .u64 n,
    .param .u64 re0, .param .u64 im0, .param .u64 groups, .param .u64 offsets,
    .param .u64 indices, .param .u64 weights, .param .u64 out)
{
    .reg .pred %p<2>;
    .reg .u32 %t<4>;
    .reg .u64 %a<18>;
    .reg .f64 %d<8>;
    mov.u32 %t0, %ctaid.x;
    mov.u32 %t1, %ntid.x;
    mov.u32 %t2, %tid.x;
    mul.wide.u32 %a0, %t0, %t1;
    cvt.u64.u32 %a1, %t2;
    add.u64 %a0, %a0, %a1;
    ld.param.u64 %a1, [n];
    setp.ge.u64 %p0, %a0, %a1;
    @%p0 bra $Ldone;
    ld.param.u64 %a2, [arena];
    cvta.to.global.u64 %a2, %a2;
    ld.param.u64 %a3, [ld];
    shl.b64 %a3, %a3, 3;
    shl.b64 %a1, %a0, 3;
    add.u64 %a2, %a2, %a1;
    ld.param.u64 %a4, [re0];
    ld.param.u64 %a5, [im0];
    ld.param.u64 %a6, [groups];
    ld.param.u64 %a7, [offsets];
    cvta.to.global.u64 %a7, %a7;
    ld.param.u64 %a8, [indices];
    cvta.to.global.u64 %a8, %a8;
    ld.param.u64 %a9, [weights];
    cvta.to.global.u64 %a9, %a9;
    mov.f64 %d0, 0d0000000000000000;
    mov.u64 %a10, 0;
$Lgroup:
    setp.ge.u64 %p0, %a10, %a6;
    @%p0 bra $Lstore;
    shl.b64 %a11, %a10, 2;
    add.u64 %a11, %a11, %a7;
    ld.global.u32 %t0, [%a11];
    ld.global.u32 %t1, [%a11+4];
    cvt.u64.u32 %a12, %t0;
    cvt.u64.u32 %a13, %t1;
    mov.f64 %d1, 0d0000000000000000;
    mov.f64 %d2, 0d0000000000000000;
$Lterm:
    setp.ge.u64 %p1, %a12, %a13;
    @%p1 bra $Lsquare;
    shl.b64 %a14, %a12, 2;
    add.u64 %a14, %a14, %a8;
    ld.global.u32 %t2, [%a14];
    cvt.u64.u32 %a15, %t2;
    add.u64 %a16, %a15, %a4;
    mul.lo.u64 %a16, %a16, %a3;
    add.u64 %a16, %a16, %a2;
    ld.global.f64 %d3, [%a16];
    add.u64 %a17, %a15, %a5;
    mul.lo.u64 %a17, %a17, %a3;
    add.u64 %a17, %a17, %a2;
    ld.global.f64 %d4, [%a17];
    add.rn.f64 %d1, %d1, %d3;
    add.rn.f64 %d2, %d2, %d4;
    add.u64 %a12, %a12, 1;
    bra.uni $Lterm;
$Lsquare:
    mul.rn.f64 %d5, %d1, %d1;
    mul.rn.f64 %d6, %d2, %d2;
    add.rn.f64 %d5, %d5, %d6;
    shl.b64 %a11, %a10, 3;
    add.u64 %a11, %a11, %a9;
    ld.global.f64 %d7, [%a11];
    mul.rn.f64 %d5, %d7, %d5;
    add.rn.f64 %d0, %d0, %d5;
    add.u64 %a10, %a10, 1;
    bra.uni $Lgroup;
$Lstore:
    ld.param.u64 %a11, [out];
    cvta.to.global.u64 %a11, %a11;
    add.u64 %a11, %a11, %a1;
    st.global.f64 [%a11], %d0;
$Ldone:
    ret;
}
";

/// The fixed part of the CUDA runner's `runner.c`.
const RUN_CUDA_C: &str = r#"/* the CUDA driver API, opened with dlopen (the types and symbols of CUDA 13's cuda.h) */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

typedef struct {
    size_t srcXInBytes, srcY;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    void *srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    void *dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
} CUDA_MEMCPY2D;

enum { CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2 };
enum { CU_STREAM_NON_BLOCKING = 1, CU_EVENT_DISABLE_TIMING = 2 };

#define BLOCK 128u
#define MAX_STREAMS 32
#define READ_FIRST 1u
#define WRITTEN 2u
#define COPY_BACK 4u /* written and copied back */

static struct {
    int state; /* 0: not opened yet, 1: ready, -1: failed */
    void *lib;
    CUdevice dev;
    CUcontext ctx;
    CUfunction functions[NUM_LEAVES];
    CUdeviceptr arena; /* NUM_PLANES planes of `ld` points */
    size_t ld;
#if F32
    CUdeviceptr arena32; /* the f32 arena the leaves run on, NUM_PLANES planes of `ld` points */
    CUdeviceptr lists;   /* the planes converted in, then those converted out (2 x NUM_PLANES) */
    CUfunction to_f32, to_f64;
#endif
    /* the reduction: its kernel, its tables on the device (offsets, indices, weights) as last
       uploaded and their host copy, and the values (`ld` doubles) */
    CUfunction reduce;
    CUdeviceptr tables, values;
    size_t tables_size;
    unsigned char *tables_host;
    int copy_all; /* PACMAN_RUNNER_COPY_BACK=all or PACMAN_SAVE_ARENA: every written plane */
    /* PACMAN_RUNNER_STREAMS=S (default 8): the independent leaves of a group run on S streams */
    unsigned num_streams;
    CUstream streams[MAX_STREAMS];
    CUevent joins[MAX_STREAMS]; /* one per stream, recorded after its part of a group */
    CUevent fork;               /* recorded on the default stream before a group */
    struct {
        CUresult (*Init)(unsigned);
        CUresult (*DeviceGet)(CUdevice *, int);
        CUresult (*DevicePrimaryCtxRetain)(CUcontext *, CUdevice);
        CUresult (*CtxPushCurrent)(CUcontext);
        CUresult (*CtxPopCurrent)(CUcontext *);
        CUresult (*ModuleLoadData)(CUmodule *, const void *);
        CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
        CUresult (*MemAlloc)(CUdeviceptr *, size_t);
        CUresult (*MemFree)(CUdeviceptr);
        CUresult (*Memcpy2DUnaligned)(const CUDA_MEMCPY2D *);
        CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
        CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
        CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                 unsigned, unsigned, CUstream, void **, void **);
        CUresult (*GetErrorString)(CUresult, const char **);
        CUresult (*StreamCreate)(CUstream *, unsigned);
        CUresult (*EventCreate)(CUevent *, unsigned);
        CUresult (*EventRecord)(CUevent, CUstream);
        CUresult (*StreamWaitEvent)(CUstream, CUevent, unsigned);
    } cu;
} G;

static unsigned char FLAGS[NUM_PLANES];
/* the group that last read or wrote each plane (groups of leaves without conflicts) */
static uint32_t READ_IN[NUM_PLANES], WRITTEN_IN[NUM_PLANES];
static uint32_t GROUP;

/* PACMAN_RUNNER_STATS=1: calls, points, copies and time, printed when the library is unloaded */
static struct {
    int on;
    unsigned long long calls, points, copies, bytes, groups, reduced;
    double seconds;
} STATS;

#if HYBRID
/* the CPU reruns: the fraction of points, the selection, the compact arena of `cap` points */
static struct {
    double fraction;
    uint32_t *order, *selected;
    size_t points_cap;
    double *arena, *values;
    double *table[NUM_PLANES];
    size_t cap;
    unsigned long long points;
    double seconds;
} H;
#endif

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

__attribute__((destructor)) static void print_stats(void) {
    if (STATS.on && STATS.calls > 0) {
        fprintf(stderr,
                "pacman runner (%s of %s%s): %llu calls, %llu points (%llu reduced on the GPU), "
                "%llu copies (%.1f MB), %u streams, %llu groups, %.3f s (%.2f us/point)\n",
                METADATA.execution, METADATA.process, F32 ? ", f32" : "", STATS.calls,
                STATS.points, STATS.reduced, STATS.copies,
                STATS.bytes * 1e-6, G.num_streams, STATS.groups, STATS.seconds,
                STATS.points ? 1e6 * STATS.seconds / STATS.points : 0.0);
#if HYBRID
        fprintf(stderr,
                "pacman runner (%s of %s): %llu points rerun on the CPU in f64 (fallback %g%%), "
                "%.3f s of the time above\n",
                METADATA.execution, METADATA.process, H.points, 100.0 * H.fraction, H.seconds);
#endif
    }
}

static void report(const char *step, CUresult r) {
    const char *message = NULL;
    if (G.cu.GetErrorString != NULL) {
        G.cu.GetErrorString(r, &message);
    }
    fprintf(stderr, "pacman runner (%s of %s): %s failed: %s (%d)\n", METADATA.execution,
            METADATA.process, step, message != NULL ? message : "unknown error", r);
}

#define CU(call, step)              \
    do {                            \
        CUresult r_ = (call);       \
        if (r_ != 0) {              \
            report(step, r_);       \
            goto fail;              \
        }                           \
    } while (0)

/* copies the planes with `flag` in `flags` between the host planes (points start .. start +
   count) and the device arena, a run of equally spaced host planes in one 2D copy */
static int copy_planes(double *const *planes, uint32_t start, uint32_t count,
                       const unsigned char *flags, unsigned char flag, int to_device) {
    size_t bytes = (size_t)count * sizeof(double);
    uint32_t p = 0;
    while (p < NUM_PLANES) {
        if (!(flags[p] & flag)) {
            p++;
            continue;
        }
        uint32_t q = p + 1;
        ptrdiff_t pitch = (ptrdiff_t)bytes;
        if (q < NUM_PLANES && (flags[q] & flag)) {
            ptrdiff_t d = (char *)planes[q] - (char *)planes[p];
            if (d >= (ptrdiff_t)bytes) {
                pitch = d;
                while (q < NUM_PLANES && (flags[q] & flag) &&
                       (char *)planes[q] - (char *)planes[q - 1] == pitch) {
                    q++;
                }
            }
        }
        CUDA_MEMCPY2D c;
        memset(&c, 0, sizeof c);
        CUdeviceptr device = G.arena + (CUdeviceptr)p * G.ld * sizeof(double);
        if (to_device) {
            c.srcMemoryType = CU_MEMORYTYPE_HOST;
            c.srcHost = planes[p] + start;
            c.srcPitch = (size_t)pitch;
            c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            c.dstDevice = device;
            c.dstPitch = G.ld * sizeof(double);
        } else {
            c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            c.srcDevice = device;
            c.srcPitch = G.ld * sizeof(double);
            c.dstMemoryType = CU_MEMORYTYPE_HOST;
            c.dstHost = planes[p] + start;
            c.dstPitch = (size_t)pitch;
        }
        c.WidthInBytes = bytes;
        c.Height = q - p;
        CUresult r = G.cu.Memcpy2DUnaligned(&c);
        if (r != 0) {
            report(to_device ? "copying planes to the GPU" : "copying planes from the GPU", r);
            return -1;
        }
        STATS.copies++;
        STATS.bytes += bytes * (q - p);
        p = q;
    }
    return 0;
}

/* opens libcuda, the primary context of device 0 and the leaves' modules, once */
static int cuda_open(void) {
    if (G.state != 0) {
        return G.state;
    }
    G.state = -1;
    const char *stats = getenv("PACMAN_RUNNER_STATS");
    STATS.on = stats != NULL && stats[0] != '\0' && strcmp(stats, "0") != 0;
    const char *copy_back = getenv("PACMAN_RUNNER_COPY_BACK");
    const char *capture = getenv("PACMAN_SAVE_ARENA");
    G.copy_all = (copy_back != NULL && strcmp(copy_back, "all") == 0) ||
                 (capture != NULL && capture[0] != '\0');
    const char *streams = getenv("PACMAN_RUNNER_STREAMS");
    G.num_streams = streams != NULL ? (unsigned)strtoul(streams, NULL, 10) : 8;
    if (G.num_streams < 1) G.num_streams = 1;
    if (G.num_streams > MAX_STREAMS) G.num_streams = MAX_STREAMS;
#if HYBRID
    /* PACMAN_RUNNER_FALLBACK=F: a fraction (0.01) or a percentage (1%) */
    H.fraction = FALLBACK_DEFAULT;
    const char *fallback = getenv("PACMAN_RUNNER_FALLBACK");
    if (fallback != NULL && fallback[0] != '\0') {
        char *end;
        double f = strtod(fallback, &end);
        if (*end == '%') f /= 100.0;
        H.fraction = f < 0.0 ? 0.0 : f > 1.0 ? 1.0 : f;
    }
#endif
    G.lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (G.lib == NULL) {
        fprintf(stderr, "pacman runner: cannot open libcuda.so.1: %s\n", dlerror());
        return -1;
    }
    struct {
        void **fn;
        const char *name;
    } symbols[] = {
        {(void **)&G.cu.Init, "cuInit"},
        {(void **)&G.cu.DeviceGet, "cuDeviceGet"},
        {(void **)&G.cu.DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"},
        {(void **)&G.cu.CtxPushCurrent, "cuCtxPushCurrent_v2"},
        {(void **)&G.cu.CtxPopCurrent, "cuCtxPopCurrent_v2"},
        {(void **)&G.cu.ModuleLoadData, "cuModuleLoadData"},
        {(void **)&G.cu.ModuleGetFunction, "cuModuleGetFunction"},
        {(void **)&G.cu.MemAlloc, "cuMemAlloc_v2"},
        {(void **)&G.cu.MemFree, "cuMemFree_v2"},
        {(void **)&G.cu.Memcpy2DUnaligned, "cuMemcpy2DUnaligned_v2"},
        {(void **)&G.cu.MemcpyHtoD, "cuMemcpyHtoD_v2"},
        {(void **)&G.cu.MemcpyDtoH, "cuMemcpyDtoH_v2"},
        {(void **)&G.cu.LaunchKernel, "cuLaunchKernel"},
        {(void **)&G.cu.GetErrorString, "cuGetErrorString"},
        {(void **)&G.cu.StreamCreate, "cuStreamCreate"},
        {(void **)&G.cu.EventCreate, "cuEventCreate"},
        {(void **)&G.cu.EventRecord, "cuEventRecord"},
        {(void **)&G.cu.StreamWaitEvent, "cuStreamWaitEvent"},
    };
    for (size_t i = 0; i < sizeof symbols / sizeof symbols[0]; i++) {
        *symbols[i].fn = dlsym(G.lib, symbols[i].name);
        if (*symbols[i].fn == NULL) {
            fprintf(stderr, "pacman runner: libcuda.so.1 has no %s\n", symbols[i].name);
            return -1;
        }
    }
    CUcontext previous;
    CU(G.cu.Init(0), "cuInit");
    CU(G.cu.DeviceGet(&G.dev, 0), "cuDeviceGet");
    CU(G.cu.DevicePrimaryCtxRetain(&G.ctx, G.dev), "cuDevicePrimaryCtxRetain");
    CU(G.cu.CtxPushCurrent(G.ctx), "cuCtxPushCurrent");
    for (uint32_t k = 0; k < NUM_LEAVES; k++) {
        CUmodule module;
        char name[32];
        snprintf(name, sizeof name, "pacman_leaf%u", k);
        CU(G.cu.ModuleLoadData(&module, LEAF_FATBINS[k]), "cuModuleLoadData (no kernel for this GPU?)");
        CU(G.cu.ModuleGetFunction(&G.functions[k], module, name), "cuModuleGetFunction");
    }
    {
        CUmodule module;
        CU(G.cu.ModuleLoadData(&module, REDUCE_PTX), "cuModuleLoadData (the reduction)");
        CU(G.cu.ModuleGetFunction(&G.reduce, module, "pacman_reduce"), "cuModuleGetFunction");
    }
#if F32
    {
        CUmodule module;
        CU(G.cu.ModuleLoadData(&module, CONVERT_PTX), "cuModuleLoadData (the conversions)");
        CU(G.cu.ModuleGetFunction(&G.to_f32, module, "pacman_to_f32"), "cuModuleGetFunction");
        CU(G.cu.ModuleGetFunction(&G.to_f64, module, "pacman_to_f64"), "cuModuleGetFunction");
        CU(G.cu.MemAlloc(&G.lists, 2 * NUM_PLANES * sizeof(uint32_t)), "cuMemAlloc (plane lists)");
    }
#endif
    if (G.num_streams > 1) {
        /* non-blocking: ordered against the default stream (the copies) only by the events */
        CU(G.cu.EventCreate(&G.fork, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        for (unsigned s = 0; s < G.num_streams; s++) {
            CU(G.cu.StreamCreate(&G.streams[s], CU_STREAM_NON_BLOCKING), "cuStreamCreate");
            CU(G.cu.EventCreate(&G.joins[s], CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        }
    }
    G.cu.CtxPopCurrent(&previous);
    G.state = 1;
    return 1;
fail:
    return -1;
}

const pacman_runner_metadata *pacman_runner_v1_metadata(void) {
    return &METADATA;
}

#if F32
static uint32_t LISTS[2 * NUM_PLANES];

/* converts the `count` planes listed at `offset` in G.lists from the arena `src` into `dst`
   (`f`: pacman_to_f32 or pacman_to_f64), points 0 .. `points`, on the default stream */
static int convert(CUfunction f, CUdeviceptr src, CUdeviceptr dst, uint32_t offset,
                   uint32_t count, uint32_t points) {
    if (count == 0) return 0;
    CUdeviceptr list = G.lists + (CUdeviceptr)offset * sizeof(uint32_t);
    unsigned long long num = count, n = points, ld = G.ld;
    void *args[] = {&src, &dst, &list, &num, &n, &ld};
    unsigned rows = count < 65535 ? count : 65535;
    CU(G.cu.LaunchKernel(f, (points + BLOCK - 1) / BLOCK, rows, 1, BLOCK, 1, 1, 0, NULL, args, NULL),
       "cuLaunchKernel (a conversion)");
    return 0;
fail:
    return -1;
}
#endif

/* starts a new group of leaves (no plane marked) */
static void new_group(void) {
    if (++GROUP == 0) {
        memset(READ_IN, 0, sizeof READ_IN);
        memset(WRITTEN_IN, 0, sizeof WRITTEN_IN);
        GROUP = 1;
    }
}

/* adds `leaf` to the current group if it does not conflict with it (reads a plane the group
   writes, or writes one it reads or writes); 0 if it conflicts */
static int join_group(uint32_t leaf) {
    const uint32_t *map = LEAF_PLANES[leaf];
    uint32_t inputs = LEAF_INPUTS[leaf], end = inputs + LEAF_OUTPUTS[leaf];
    for (uint32_t i = 0; i < inputs; i++) {
        if (WRITTEN_IN[map[i]] == GROUP) return 0;
    }
    for (uint32_t i = inputs; i < end; i++) {
        if (WRITTEN_IN[map[i]] == GROUP || READ_IN[map[i]] == GROUP) return 0;
    }
    for (uint32_t i = 0; i < inputs; i++) READ_IN[map[i]] = GROUP;
    for (uint32_t i = inputs; i < end; i++) WRITTEN_IN[map[i]] = GROUP;
    return 1;
}

/* the default stream waits for the streams used by the last group (`used` of them) */
static int join_streams(unsigned used) {
    for (unsigned s = 0; s < used; s++) {
        CU(G.cu.EventRecord(G.joins[s], G.streams[s]), "cuEventRecord");
        CU(G.cu.StreamWaitEvent(NULL, G.joins[s], 0), "cuStreamWaitEvent");
    }
    return 0;
fail:
    return -1;
}

/* launches the leaves: on the default stream in order, or, with several streams, each group of
   independent leaves spread over the streams, which wait for the default stream before the group
   (the copies in, the previous group) and which the default stream waits for after it */
static int launch(const uint32_t *leaves, uint32_t count, unsigned grid, void **args) {
    if (G.num_streams == 1) {
        for (uint32_t k = 0; k < count; k++) {
            uint32_t leaf = leaves == NULL ? k : leaves[k];
            CU(G.cu.LaunchKernel(G.functions[leaf], grid, 1, 1, BLOCK, 1, 1, 0, NULL, args, NULL),
               "cuLaunchKernel");
        }
        STATS.groups++;
        return 0;
    }
    unsigned used = 0; /* launches in the current group */
    for (uint32_t k = 0; k < count; k++) {
        uint32_t leaf = leaves == NULL ? k : leaves[k];
        if (k == 0 || !join_group(leaf)) {
            if (join_streams(used < G.num_streams ? used : G.num_streams) != 0) return -1;
            new_group();
            join_group(leaf);
            CU(G.cu.EventRecord(G.fork, NULL), "cuEventRecord");
            for (unsigned s = 0; s < G.num_streams; s++) {
                CU(G.cu.StreamWaitEvent(G.streams[s], G.fork, 0), "cuStreamWaitEvent");
            }
            used = 0;
            STATS.groups++;
        }
        CUstream stream = G.streams[used % G.num_streams];
        CU(G.cu.LaunchKernel(G.functions[leaf], grid, 1, 1, BLOCK, 1, 1, 0, stream, args, NULL),
           "cuLaunchKernel");
        used++;
    }
    return join_streams(used < G.num_streams ? used : G.num_streams);
fail:
    return -1;
}

#if HYBRID
/* the f64 CPU runner (cpu.c: `pacman runner`'s runner.c with its entry points renamed) */
int pacman_cpu_v1_run_reduce(double *const *planes, uint32_t num_planes,
                             const uint32_t *leaves, uint32_t num_leaves, uint32_t point_start,
                             uint32_t point_count, const pacman_runner_reduction *reduction,
                             double *out);

static const double *SORT_VALUES;

/* descending values, then ascending indices */
static int by_value(const void *a, const void *b) {
    uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b;
    double x = SORT_VALUES[i], y = SORT_VALUES[j];
    if (x > y) return -1;
    if (x < y) return 1;
    return (i > j) - (i < j);
}

/* the points of `values` (n) to rerun, into H.selected: every point whose value is not finite,
   then the ceil(fraction x n) largest finite values; their number */
static uint32_t select_points(const double *values, uint32_t n) {
    if (n > H.points_cap) {
        free(H.order);
        free(H.selected);
        H.order = malloc((size_t)n * sizeof *H.order);
        H.selected = malloc((size_t)n * sizeof *H.selected);
        H.points_cap = H.order != NULL && H.selected != NULL ? n : 0;
        if (H.points_cap == 0) return UINT32_MAX;
    }
    uint32_t count = 0, finite = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (values[i] - values[i] != 0.0) {
            H.selected[count++] = i; /* NaN or infinite */
        } else {
            H.order[finite++] = i;
        }
    }
    double want = H.fraction * n;
    uint32_t k = (uint32_t)want;
    if (k < want) k++;
    if (k > finite) k = finite;
    if (k > 0) {
        SORT_VALUES = values;
        qsort(H.order, finite, sizeof *H.order, by_value);
        memcpy(H.selected + count, H.order, (size_t)k * sizeof *H.order);
        count += k;
    }
    return count;
}

/* reruns the selected points of the call on the CPU in f64 and writes their values into `out`:
   their planes read first are gathered from the host planes (still those RustiCol wrote: only
   amplitudes, written by the leaves, are copied back, and after this) into a compact arena */
static int rerun_on_cpu(double *const *planes, const uint32_t *leaves, uint32_t num_leaves,
                        uint32_t point_start, uint32_t point_count,
                        const pacman_runner_reduction *reduction, double *out) {
    uint32_t n = select_points(out, point_count);
    if (n == UINT32_MAX) {
        fprintf(stderr, "pacman runner: out of memory\n");
        return PACMAN_RUNNER_DEVICE_ERROR;
    }
    if (n == 0) return PACMAN_RUNNER_OK;
    double started = now();
    /* SIMD kernels: whole blocks of 8, padded with the last point */
    uint32_t padded = CPU_LANES > 1 ? (n + 7) / 8 * 8 : n;
    if (padded > H.cap) {
        free(H.arena);
        free(H.values);
        H.arena = malloc((size_t)NUM_PLANES * padded * sizeof(double));
        H.values = malloc((size_t)padded * sizeof(double));
        H.cap = H.arena != NULL && H.values != NULL ? padded : 0;
        if (H.cap == 0) {
            fprintf(stderr, "pacman runner: out of memory\n");
            return PACMAN_RUNNER_DEVICE_ERROR;
        }
    }
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        H.table[p] = H.arena + (size_t)p * padded;
        if (!(FLAGS[p] & READ_FIRST)) continue;
        const double *from = planes[p] + point_start;
        for (uint32_t j = 0; j < padded; j++) {
            H.table[p][j] = from[H.selected[j < n ? j : n - 1]];
        }
    }
    int status = pacman_cpu_v1_run_reduce(H.table, NUM_PLANES, leaves, num_leaves, 0, padded,
                                          reduction, H.values);
    if (status != PACMAN_RUNNER_OK) {
        fprintf(stderr, "pacman runner (%s of %s): the CPU rerun returned status %d\n",
                METADATA.execution, METADATA.process, status);
        return status;
    }
    for (uint32_t j = 0; j < n; j++) out[H.selected[j]] = H.values[j];
    H.points += n;
    H.seconds += now() - started;
    return PACMAN_RUNNER_OK;
}
#endif

/* uploads the tables of `r` unless they equal the last ones, then reduces points 0 .. n of the
   f64 arena into G.values, on the default stream */
static int reduce_device(const pacman_runner_reduction *r, uint32_t n) {
    uint32_t groups = r->num_groups, terms = groups > 0 ? r->offsets[groups] : 0;
    size_t offsets_size = (groups + 1) * sizeof(uint32_t);
    size_t indices_at = offsets_size;
    size_t weights_at = (offsets_size + terms * sizeof(uint32_t) + 7) / 8 * 8;
    size_t size = weights_at + groups * sizeof(double);
    unsigned char *host = calloc(size, 1);
    if (host == NULL) {
        fprintf(stderr, "pacman runner: out of memory\n");
        return -1;
    }
    if (groups > 0) {
        memcpy(host, r->offsets, offsets_size);
        memcpy(host + indices_at, r->indices, terms * sizeof(uint32_t));
        memcpy(host + weights_at, r->weights, groups * sizeof(double));
    }
    if (size != G.tables_size || memcmp(host, G.tables_host, size) != 0) {
        if (G.tables != 0) G.cu.MemFree(G.tables);
        G.tables = 0;
        free(G.tables_host);
        G.tables_host = NULL;
        G.tables_size = 0;
        CU(G.cu.MemAlloc(&G.tables, size), "cuMemAlloc (the reduction's tables)");
        CU(G.cu.MemcpyHtoD(G.tables, host, size), "cuMemcpyHtoD (the reduction's tables)");
        G.tables_host = host;
        G.tables_size = size;
        host = NULL;
    }
    free(host);
    host = NULL;
    {
        CUdeviceptr offsets = G.tables, indices = G.tables + indices_at,
                    weights = G.tables + weights_at;
        unsigned long long ld = G.ld, count = n, num_groups = groups,
                           re0 = METADATA.regions[PACMAN_AMPLITUDE_RE][0],
                           im0 = METADATA.regions[PACMAN_AMPLITUDE_IM][0];
        void *args[] = {&G.arena,    &ld,      &count,   &re0,     &im0,
                        &num_groups, &offsets, &indices, &weights, &G.values};
        CU(G.cu.LaunchKernel(G.reduce, (n + BLOCK - 1) / BLOCK, 1, 1, BLOCK, 1, 1, 0, NULL, args,
                             NULL),
           "cuLaunchKernel (the reduction)");
    }
    return 0;
fail:
    free(host);
    return -1;
}

/* pacman_runner_v1_run, and with `reduction` pacman_runner_v1_run_reduce */
static int run(double *const *planes, uint32_t num_planes, const uint32_t *leaves,
               uint32_t num_leaves, uint32_t point_start, uint32_t point_count,
               const pacman_runner_reduction *reduction, double *out) {
    if (planes == NULL || num_planes != NUM_PLANES || point_count == 0 ||
        point_start > UINT32_MAX - point_count) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        if (planes[p] == NULL) {
            return PACMAN_RUNNER_BAD_ARGUMENTS;
        }
    }
    uint32_t count = leaves == NULL ? NUM_LEAVES : num_leaves;
    for (uint32_t k = 0; k < count; k++) {
        if (leaves != NULL && leaves[k] >= NUM_LEAVES) {
            return PACMAN_RUNNER_BAD_ARGUMENTS;
        }
    }
    if (cuda_open() != 1) {
        return PACMAN_RUNNER_DEVICE_ERROR;
    }

    /* the planes to copy in (read before the leaves write them: RustiCol writes them before the
       call) and out (written; by default only the amplitudes, which RustiCol reduces; the
       currents are scratch) */
    memset(FLAGS, 0, sizeof FLAGS);
    for (uint32_t k = 0; k < count; k++) {
        uint32_t leaf = leaves == NULL ? k : leaves[k];
        const uint32_t *map = LEAF_PLANES[leaf];
        for (uint32_t i = 0; i < LEAF_INPUTS[leaf]; i++) {
            if (!(FLAGS[map[i]] & WRITTEN)) FLAGS[map[i]] |= READ_FIRST;
        }
        for (uint32_t i = LEAF_INPUTS[leaf]; i < LEAF_INPUTS[leaf] + LEAF_OUTPUTS[leaf]; i++) {
            FLAGS[map[i]] |= WRITTEN;
        }
    }
    uint32_t amplitudes = METADATA.regions[PACMAN_AMPLITUDE_RE][0];
    uint32_t amplitudes_end =
        METADATA.regions[PACMAN_AMPLITUDE_IM][0] + METADATA.regions[PACMAN_AMPLITUDE_IM][1];
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        if ((FLAGS[p] & WRITTEN) && (G.copy_all || (p >= amplitudes && p < amplitudes_end))) {
            FLAGS[p] |= COPY_BACK;
        }
    }

    double started = now();
    CUcontext previous;
    CU(G.cu.CtxPushCurrent(G.ctx), "cuCtxPushCurrent");
    /* the device arena: room for the points, a multiple of 32 */
    size_t ld = ((size_t)point_count + 31) / 32 * 32;
    if (ld > G.ld) {
        if (G.arena != 0) {
            G.cu.MemFree(G.arena);
            G.cu.MemFree(G.values);
            G.arena = 0;
            G.values = 0;
            G.ld = 0;
        }
#if F32
        if (G.arena32 != 0) {
            G.cu.MemFree(G.arena32);
            G.arena32 = 0;
        }
        CU(G.cu.MemAlloc(&G.arena32, (size_t)NUM_PLANES * ld * sizeof(float)), "cuMemAlloc (the f32 arena)");
#endif
        CU(G.cu.MemAlloc(&G.arena, (size_t)NUM_PLANES * ld * sizeof(double)), "cuMemAlloc (the arena)");
        CU(G.cu.MemAlloc(&G.values, ld * sizeof(double)), "cuMemAlloc (the reduced values)");
        G.ld = ld;
    }
    if (copy_planes(planes, point_start, point_count, FLAGS, READ_FIRST, 1) != 0) {
        goto fail;
    }
    unsigned long long n = point_count, stride = G.ld;
#if F32
    /* the planes copied in, rounded into the f32 arena, and those to copy back */
    uint32_t num_in = 0, num_out = 0;
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        if (FLAGS[p] & READ_FIRST) LISTS[num_in++] = p;
        if (FLAGS[p] & COPY_BACK) LISTS[NUM_PLANES + num_out++] = p;
    }
    CU(G.cu.MemcpyHtoD(G.lists, LISTS, sizeof LISTS), "cuMemcpyHtoD (plane lists)");
    if (convert(G.to_f32, G.arena, G.arena32, 0, num_in, point_count) != 0) {
        goto fail;
    }
    CUdeviceptr arena = G.arena32;
#else
    CUdeviceptr arena = G.arena;
#endif
    void *args[] = {&arena, &arena, &arena, &n, &stride, &stride};
    unsigned grid = (point_count + BLOCK - 1) / BLOCK;
    if (launch(leaves, count, grid, args) != 0) {
        goto fail;
    }
#if F32
    /* on the default stream, which the leaves' streams joined */
    if (convert(G.to_f64, G.arena32, G.arena, NUM_PLANES, num_out, point_count) != 0) {
        goto fail;
    }
#endif
    if (reduction != NULL) {
        if (reduce_device(reduction, point_count) != 0) {
            goto fail;
        }
        /* synchronous: it waits for the reduction */
        CU(G.cu.MemcpyDtoH(out, G.values, (size_t)point_count * sizeof(double)),
           "cuMemcpyDtoH (the reduced values)");
        STATS.copies++;
        STATS.bytes += (size_t)point_count * sizeof(double);
        STATS.reduced += point_count;
#if HYBRID
        int status = rerun_on_cpu(planes, leaves, num_leaves, point_start, point_count,
                                  reduction, out);
        if (status != PACMAN_RUNNER_OK) {
            G.cu.CtxPopCurrent(&previous);
            return status;
        }
#endif
    }
    /* synchronous: it waits for the kernels (the default stream joins the streams); with a
       reduction, only for captures (PACMAN_RUNNER_COPY_BACK=all, PACMAN_SAVE_ARENA) */
    if ((reduction == NULL || G.copy_all) &&
        copy_planes(planes, point_start, point_count, FLAGS, COPY_BACK, 0) != 0) {
        goto fail;
    }
    G.cu.CtxPopCurrent(&previous);
    STATS.calls++;
    STATS.points += point_count;
    STATS.seconds += now() - started;
    return PACMAN_RUNNER_OK;
fail:
    G.cu.CtxPopCurrent(&previous);
    return PACMAN_RUNNER_DEVICE_ERROR;
}

int pacman_runner_v1_run(double *const *planes, uint32_t num_planes, const uint32_t *leaves,
                         uint32_t num_leaves, uint32_t point_start, uint32_t point_count) {
    return run(planes, num_planes, leaves, num_leaves, point_start, point_count, NULL, NULL);
}

int pacman_runner_v1_run_reduce(double *const *planes, uint32_t num_planes,
                                const uint32_t *leaves, uint32_t num_leaves, uint32_t point_start,
                                uint32_t point_count, const pacman_runner_reduction *reduction,
                                double *out) {
    if (!reduction_ok(reduction, out)) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    return run(planes, num_planes, leaves, num_leaves, point_start, point_count, reduction, out);
}
"#;

/// The `Makefile` of the CUDA runner: `ptxas` per leaf and SM, `fatbinary`, `cc` (the commands
/// `build_cuda` runs).
/// `cpu.c` of a hybrid runner: the CPU runner of `spec` (`runner_c`) with its entry points
/// renamed `pacman_cpu_v1_*`, linked into the CUDA runner for its f64 reruns.
pub fn hybrid_cpu_c(spec: &Spec, kernels: &Kernels) -> String {
    format!(
        "/* cpu.c: the f64 CPU kernels of a hybrid runner (runner.c reruns its largest points\n\
         \x20  here), `pacman runner`'s runner.c with its entry points renamed. */\n\
         #define pacman_runner_v1_metadata pacman_cpu_v1_metadata\n\
         #define pacman_runner_v1_run pacman_cpu_v1_run\n\
         #define pacman_runner_v1_run_reduce pacman_cpu_v1_run_reduce\n\
         /* its reruns are a few hundred points: blocks of 8 on every online CPU */\n\
         #define DEFAULT_BLOCK 8u\n#define DEFAULT_THREADS 0\n\n{}",
        runner_c(spec, kernels)
    )
}

/// The `Makefile` of the CUDA runner, with `cpu_kernels` CPU kernel objects (a hybrid runner).
pub fn makefile_cuda(leaves: usize, sms: &[u32], cpu_kernels: Option<usize>) -> String {
    let sms: Vec<String> = sms.iter().map(|sm| sm.to_string()).collect();
    let fatbins: Vec<String> = (0..leaves)
        .map(|k| format!("kernels/leaf{}.fatbin", k))
        .collect();
    let (objects, cpu) = match cpu_kernels {
        None => (String::new(), String::new()),
        Some(n) => {
            let kernels: Vec<String> = (0..n).map(|k| format!("kernels/k{}.o", k)).collect();
            (
                format!(" cpu.o {}", kernels.join(" ")),
                "cpu.o: cpu.c pacman_runner.h\n\
                 \t$(CC) $(CFLAGS) -ffp-contract=off -pthread -fPIC -c cpu.c -o cpu.o\n\n"
                    .to_string(),
            )
        }
    };
    format!(
        "# generated by pacman runner --cuda\n\
         CC ?= cc\n\
         CFLAGS ?= -O1\n\
         PTXAS ?= {}\n\
         FATBINARY ?= {}\n\
         SMS = {}\n\
         FATBINS = {}\n\
         OBJECTS = runner.o{}\n\n\
         all: runner librunner.so\n\n\
         kernels/leaf%.fatbin: kernels/leaf%.ptx\n\
         \tfor sm in $(SMS); do $(PTXAS) -O3 -arch=sm_$$sm $< -o kernels/leaf$*.sm_$$sm.cubin || exit 1; done\n\
         \t$(FATBINARY) --create=$@ -64 $(foreach sm,$(SMS),--image3=kind=elf,sm=$(sm),file=kernels/leaf$*.sm_$(sm).cubin)\n\n\
         runner.o: runner.c pacman_runner.h $(FATBINS)\n\
         \t$(CC) $(CFLAGS) -ffp-contract=off -fPIC -c runner.c -o runner.o\n\n\
         {}\
         librunner.so: $(OBJECTS)\n\
         \t$(CC) -pthread -shared -o librunner.so $(OBJECTS) -ldl\n\n\
         runner: main.c $(OBJECTS) pacman_runner.h\n\
         \t$(CC) $(CFLAGS) -pthread -o runner main.c $(OBJECTS) -ldl\n",
        cuda_tool("PTXAS", "ptxas"),
        cuda_tool("FATBINARY", "fatbinary"),
        sms.join(" "),
        fatbins.join(" "),
        objects,
        cpu
    )
}

/// A CUDA tool: `$VAR`, else `/usr/local/cuda/bin/<name>` if it exists, else `<name>` (the PATH).
pub fn cuda_tool(var: &str, name: &str) -> String {
    if let Some(path) = std::env::var_os(var) {
        return path.to_string_lossy().into_owned();
    }
    let path = Path::new("/usr/local/cuda/bin").join(name);
    if path.exists() {
        path.to_string_lossy().into_owned()
    } else {
        name.to_string()
    }
}

/// Writes the CUDA runner of `spec` into `dir`: `kernels/leaf<k>.ptx`, `runner.c`,
/// `pacman_runner.h`, `main.c`, `Makefile`.
pub fn write_cuda(
    dir: &Path,
    spec: &Spec,
    modules: &[String],
    sms: &[u32],
    f32: bool,
    fallback: Option<(&Kernels, f64)>,
) -> Result<()> {
    let write = |name: &str, bytes: &[u8]| -> Result<()> {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create `{}`", parent.display()))?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("cannot write `{}`", path.display()))
    };
    for (k, m) in modules.iter().enumerate() {
        write(&format!("kernels/leaf{}.ptx", k), m.as_bytes())?;
    }
    let hybrid = fallback.map(|(kernels, fraction)| (fraction, kernels.lanes));
    write("runner.c", runner_cuda_c(spec, f32, hybrid).as_bytes())?;
    write("pacman_runner.h", RUNNER_H.as_bytes())?;
    write("main.c", MAIN_C.as_bytes())?;
    let cpu_kernels = fallback.map(|(kernels, _)| kernels.objects.len());
    write(
        "Makefile",
        makefile_cuda(modules.len(), sms, cpu_kernels).as_bytes(),
    )?;
    if let Some((kernels, _)) = fallback {
        for (k, o) in kernels.objects.iter().enumerate() {
            write(&format!("kernels/k{}.o", k), o)?;
        }
        write("cpu.c", hybrid_cpu_c(spec, kernels).as_bytes())?;
    }
    Ok(())
}

/// Runs `program args` in `dir`; fails with its standard error.
fn run(dir: &Path, program: &str, args: &[String]) -> Result<()> {
    let out = Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .with_context(|| format!("cannot run `{}`", program))?;
    if !out.status.success() {
        bail!(
            "`{} {}` failed:\n{}",
            program,
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// Builds the CUDA runner in `dir` as its `Makefile` does: `ptxas` for every leaf and SM
/// (`jobs` at a time, the largest modules first), `fatbinary` per leaf, then `runner.c` with
/// the C compiler `cc` into `librunner.so` and `runner`.
/// Builds the CUDA runner in `dir` as its `Makefile` does; `cpu_kernels`: a hybrid runner's
/// CPU kernel objects (`cpu.c`, `kernels/k<k>.o`).
pub fn build_cuda(
    dir: &Path,
    cc: &str,
    leaves: usize,
    sms: &[u32],
    jobs: usize,
    cpu_kernels: Option<usize>,
) -> Result<()> {
    let ptxas = cuda_tool("PTXAS", "ptxas");
    let fatbinary = cuda_tool("FATBINARY", "fatbinary");
    let mut work: Vec<(u64, usize, u32)> = Vec::new();
    for k in 0..leaves {
        let size = std::fs::metadata(dir.join(format!("kernels/leaf{}.ptx", k)))
            .map(|m| m.len())
            .unwrap_or(0);
        for &sm in sms {
            work.push((size, k, sm));
        }
    }
    work.sort_by(|a, b| b.cmp(a));
    let work = std::sync::Mutex::new(work.into_iter());
    let failure = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| loop {
                let next = work.lock().unwrap().next();
                let Some((_, k, sm)) = next else {
                    break;
                };
                if failure.lock().unwrap().is_some() {
                    break;
                }
                let args: Vec<String> = vec![
                    "-O3".into(),
                    format!("-arch=sm_{}", sm),
                    format!("kernels/leaf{}.ptx", k),
                    "-o".into(),
                    format!("kernels/leaf{}.sm_{}.cubin", k, sm),
                ];
                if let Err(e) = run(dir, &ptxas, &args) {
                    *failure.lock().unwrap() = Some(e);
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().unwrap() {
        return Err(e);
    }
    for k in 0..leaves {
        let mut args = vec![format!("--create=kernels/leaf{}.fatbin", k), "-64".into()];
        for &sm in sms {
            args.push(format!(
                "--image3=kind=elf,sm={},file=kernels/leaf{}.sm_{}.cubin",
                sm, k, sm
            ));
        }
        run(dir, &fatbinary, &args)?;
    }
    let mut objects = vec!["runner.o".to_string()];
    let mut steps: Vec<Vec<String>> = vec![[
        "-O1",
        "-ffp-contract=off",
        "-fPIC",
        "-c",
        "runner.c",
        "-o",
        "runner.o",
    ]
    .map(String::from)
    .to_vec()];
    if let Some(n) = cpu_kernels {
        steps.push(
            [
                "-O1",
                "-ffp-contract=off",
                "-pthread",
                "-fPIC",
                "-c",
                "cpu.c",
                "-o",
                "cpu.o",
            ]
            .map(String::from)
            .to_vec(),
        );
        objects.push("cpu.o".into());
        objects.extend((0..n).map(|k| format!("kernels/k{}.o", k)));
    }
    let library = ["-pthread", "-shared", "-o", "librunner.so"]
        .map(String::from)
        .into_iter()
        .chain(objects.iter().cloned())
        .chain(["-ldl".to_string()]);
    steps.push(library.collect());
    let program = ["-O1", "-pthread", "-o", "runner", "main.c"]
        .map(String::from)
        .into_iter()
        .chain(objects.iter().cloned())
        .chain(["-ldl".to_string()]);
    steps.push(program.collect());
    for args in steps {
        run(dir, cc, &args)?;
    }
    Ok(())
}

/// The recurrence runner (`pacman recurrence-runner`): one CUDA kernel that runs any RustiCol
/// recurrence plan given as row words.
pub const RECURRENCE_RUNNER_CU: &str = include_str!("recurrence_runner.cu");

/// Writes `recurrence_runner.cu` into `dir` and builds `librecurrence.so` from it with `nvcc`
/// (`NVCC`) for the GPUs `sms`. Without FMA contraction, the kernel's results equal RustiCol's
/// scalar ones bit for bit.
pub fn build_recurrence(dir: &Path, sms: &[u32]) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create `{}`", dir.display()))?;
    std::fs::write(dir.join("recurrence_runner.cu"), RECURRENCE_RUNNER_CU)
        .with_context(|| format!("cannot write into `{}`", dir.display()))?;
    let nvcc = cuda_tool("NVCC", "nvcc");
    let mut args: Vec<String> = ["-O3", "-fmad=false", "-Xcompiler", "-fPIC", "-shared"]
        .map(String::from)
        .to_vec();
    args.extend(["-cudart".into(), "static".into()]);
    for sm in sms {
        args.push(format!("-gencode=arch=compute_{0},code=sm_{0}", sm));
    }
    args.extend(["-o", "librecurrence.so", "recurrence_runner.cu"].map(String::from));
    run(dir, &nvcc, &args)
}

/// The compute capability of the local GPU as an SM number (`nvidia-smi`), if there is one.
pub fn local_sm() -> Option<u32> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let (major, minor) = text.lines().next()?.trim().split_once('.')?;
    Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{self, Recording};
    use crate::replay::tests::{PRODUCT_PLUS_ONE, SQUARE_MINUS};
    use crate::replay::{self, plane_application, CpuKernel};

    /// Loads `librunner.so` with dlopen (as RustiCol will), checks its metadata and argument
    /// errors, and runs all leaves on all points of an arena file at once:
    /// `dltest LIB ARENA POINTS OUT`.
    const DLTEST_C: &str = r#"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "pacman_runner.h"
typedef const pacman_runner_metadata *(*metadata_fn)(void);
typedef int (*run_fn)(double *const *, uint32_t, const uint32_t *, uint32_t, uint32_t, uint32_t);
int main(int argc, char **argv) {
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) { fprintf(stderr, "%s\n", dlerror()); return 3; }
    metadata_fn metadata = (metadata_fn)dlsym(lib, "pacman_runner_v1_metadata");
    run_fn run = (run_fn)dlsym(lib, "pacman_runner_v1_run");
    if (metadata == NULL || run == NULL) return 4;
    const pacman_runner_metadata *md = metadata();
    if (md->abi_version != 1 || md->struct_size != sizeof *md || md->reserved != 0) return 5;
    uint32_t points = (uint32_t)atoi(argv[3]), planes = md->num_planes;
    double *arena = malloc(sizeof(double) * planes * points);
    double **table = malloc(sizeof *table * planes);
    FILE *f = fopen(argv[2], "rb");
    if (fread(arena, sizeof(double), (size_t)planes * points, f) != (size_t)planes * points) return 6;
    fclose(f);
    for (uint32_t p = 0; p < planes; p++) table[p] = arena + (size_t)p * points;
    uint32_t bad_leaf = md->num_leaves;
    if (run(table, planes + 1, NULL, 0, 0, points) != PACMAN_RUNNER_BAD_ARGUMENTS ||
        run(table, planes, &bad_leaf, 1, 0, points) != PACMAN_RUNNER_BAD_ARGUMENTS ||
        run(table, planes, NULL, 0, 0, 0) != PACMAN_RUNNER_BAD_ARGUMENTS ||
        run(NULL, planes, NULL, 0, 0, points) != PACMAN_RUNNER_BAD_ARGUMENTS) return 7;
    /* two calls: the first point, then the rest */
    if (run(table, planes, NULL, 0, 0, 1) != PACMAN_RUNNER_OK ||
        run(table, planes, NULL, 0, 1, points - 1) != PACMAN_RUNNER_OK) return 8;
    f = fopen(argv[4], "wb");
    fwrite(arena, sizeof(double), (size_t)planes * points, f);
    fclose(f);
    printf("%s %s %u leaves\n", md->process, md->execution, md->num_leaves);
    return 0;
}
"#;

    fn have_cc() -> bool {
        Command::new("cc")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// Two leaves on an arena of 12 planes (5 current planes, a momentum plane, the zero plane):
    /// w = z0 z1 + 1 into current plane 2, then v = w² - p (a momentum) into current plane 3.
    fn spec() -> Spec {
        Spec {
            process: "t".into(),
            execution: "top".into(),
            arena: Arena {
                currents: 5,
                amplitudes: 0,
                momenta: 1,
                parameters: 0,
            },
            canonical_to_physical: vec![0, 1, 2, 3, u32::MAX],
            calls: vec![
                LeafCall::new(
                    "processes/t/evaluators/a.plane.symjit".into(),
                    "stage1:0".into(),
                    vec![0, 5, 1, 6],
                    vec![2, 7],
                ),
                LeafCall::new(
                    "processes/t/evaluators/b.plane.symjit".into(),
                    "stage2:0".into(),
                    vec![2, 7, 10, 11],
                    vec![3, 8],
                ),
            ],
            leaves: vec![
                "evaluators/a.plane.symjit".into(),
                "evaluators/b.plane.symjit".into(),
            ],
        }
    }

    const RECORDING: &str = r#"{"format": "pacman-arena-v1", "dtype": "f64-le",
        "layout": "plane-major", "name": "t.e0.r0", "process": "t", "engine": 0, "recording": 0,
        "method": "all", "num_planes": 12, "num_points": 9, "current_planes": 5,
        "amplitude_planes": 0, "momentum_planes": 1, "model_parameters": 0,
        "regions": [
          {"name": "current_re", "first": 0, "count": 5},
          {"name": "current_im", "first": 5, "count": 5},
          {"name": "amplitude_re", "first": 10, "count": 0},
          {"name": "amplitude_im", "first": 10, "count": 0},
          {"name": "momenta", "first": 10, "count": 1},
          {"name": "zero", "first": 11, "count": 1},
          {"name": "parameters", "first": 12, "count": 0}],
        "canonical_to_physical": [0, 1, 2, 3, 4294967295], "unmapped_physical": 4294967295,
        "source_planes": [0, 1], "tiles": [5, 4],
        "schedule": [
          {"stage": 0, "label": "s1", "amplitude": false,
           "leaves": [{"leaf": 0, "application_path": "evaluators/a.plane.symjit"}]},
          {"stage": 1, "label": "s2", "amplitude": false,
           "leaves": [{"leaf": 0, "application_path": "evaluators/b.plane.symjit"}]}]}"#;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_runner_reproduces_the_replay() {
        if !have_cc() || !std::is_x86_feature_detected!("avx2") {
            eprintln!("no C compiler (cc) or no AVX2: the SIMD runner is not built");
            return;
        }
        simd_runner_check(4);
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq") {
            simd_runner_check(8);
        } else {
            eprintln!("no AVX512F and AVX512DQ: the AVX-512 runner is not run");
        }
    }

    // the SIMD runner of `lanes` lanes on a recording of 16 points against the (SIMD) replay
    #[cfg(target_arch = "x86_64")]
    fn simd_runner_check(lanes: u32) {
        let dir = std::env::temp_dir().join(format!(
            "pacman-simd{}-runner-{}",
            lanes,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let kernels = objects(&spec, &members, lanes, &[]).unwrap();
        assert!(kernels.stack > 0);
        write(&dir, &spec, &kernels).unwrap();
        build(&dir, "cc", kernels.objects.len()).unwrap();

        // the recording of `runner_reproduces_the_scalar_replay` on 16 points, in tiles of 8
        let n = 16;
        let json = dir.join("t.e0.r0.json");
        let text = RECORDING
            .replace("\"num_points\": 9", "\"num_points\": 16")
            .replace("\"tiles\": [5, 4]", "\"tiles\": [8, 8]");
        std::fs::write(&json, text).unwrap();
        let mut pre = vec![7.0; 12 * n];
        for p in [0, 1, 5, 6, 10] {
            for i in 0..n {
                pre[p * n + i] = ((p * n + i) as f64 * 0.37).sin();
            }
        }
        pre[11 * n..12 * n].fill(0.0);
        let bytes: Vec<u8> = pre.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();
        std::fs::write(json.with_extension("post"), &bytes).unwrap();
        let rec = Recording::read(&json).unwrap();
        let kernels = [
            CpuKernel::load("a", &a).unwrap(),
            CpuKernel::load("b", &b).unwrap(),
        ];
        let ks: Vec<&CpuKernel> = kernels.iter().collect();
        let want = replay::replay(&rec, &spec.calls, &ks, &replay::Options::default())
            .unwrap()
            .arena;
        assert!(want[3 * n..4 * n].iter().all(|x| x.is_finite()));

        // one thread, then the 16 points of `--whole` split over 2 threads (blocks of 8)
        for threads in ["1", "3"] {
            for args in [&["out"][..], &["--whole", "out"][..]] {
                let out = Command::new(dir.join("runner"))
                    .args(args)
                    .arg(&json)
                    .env("PACMAN_RUNNER_THREADS", threads)
                    .env("PACMAN_RUNNER_BLOCK", "8")
                    .current_dir(&dir)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let got = recording::read_f64s(&dir.join("out/t.e0.r0.arena"), 12 * n).unwrap();
                assert!(
                    recording::compare(&got, &want, n).is_empty(),
                    "{:?} on {} threads",
                    args,
                    threads
                );
            }
        }

        // a point range that is not a multiple of 8 is refused
        let text = RECORDING.replace("\"tiles\": [5, 4]", "\"tiles\": [9]");
        std::fs::write(&json, text).unwrap();
        let mut pre9 = vec![0.0; 12 * 9];
        for p in 0..12 {
            pre9[p * 9..p * 9 + 9].copy_from_slice(&pre[p * n..p * n + 9]);
        }
        let bytes: Vec<u8> = pre9.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();
        let out = Command::new(dir.join("runner"))
            .args(["out"])
            .arg(&json)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(!out.status.success());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn runner_reproduces_the_scalar_replay() {
        if !have_cc() {
            eprintln!("no C compiler (cc): the runner is not built");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pacman-runner-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let kernels = objects(&spec, &members, 1, &[]).unwrap();
        assert_eq!(kernels.of_leaf, vec![0, 1]);
        write(&dir, &spec, &kernels).unwrap();
        build(&dir, "cc", kernels.objects.len()).unwrap();

        // a recording: sources in current planes 0, 1 (re 0, 1, im 5, 6) and the momentum
        // plane 10; the outputs' planes hold stale values (poisoned by the runner)
        let json = dir.join("t.e0.r0.json");
        std::fs::write(&json, RECORDING).unwrap();
        let mut pre = vec![7.0; 12 * 9];
        for p in [0, 1, 5, 6, 10] {
            for i in 0..9 {
                pre[p * 9 + i] = ((p * 9 + i) as f64 * 0.37).sin();
            }
        }
        pre[11 * 9..12 * 9].fill(0.0);
        let bytes: Vec<u8> = pre.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();
        std::fs::write(json.with_extension("post"), &bytes).unwrap();
        let rec = Recording::read(&json).unwrap();
        let kernels = [
            CpuKernel::load("a", &a).unwrap(),
            CpuKernel::load("b", &b).unwrap(),
        ];
        let ks: Vec<&CpuKernel> = kernels.iter().collect();
        let scalar = replay::Options {
            scalar: true,
            ..replay::Options::default()
        };
        let want = replay::replay(&rec, &spec.calls, &ks, &scalar)
            .unwrap()
            .arena;
        assert!(want[3 * 9..4 * 9].iter().all(|x| x.is_finite()));

        // the executable, per tile and on all points at once
        for args in [&["out"][..], &["--whole", "out"][..]] {
            let out = Command::new(dir.join("runner"))
                .args(args)
                .arg(&json)
                .current_dir(&dir)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let got = recording::read_f64s(&dir.join("out/t.e0.r0.arena"), 12 * 9).unwrap();
            assert!(recording::compare(&got, &want, 9).is_empty(), "{:?}", args);
        }

        // the library, loaded with dlopen, on the snapshot without poison
        std::fs::write(dir.join("dltest.c"), DLTEST_C).unwrap();
        let cc = Command::new("cc")
            .args(["-O1", "-o", "dltest", "dltest.c", "-ldl"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            cc.status.success(),
            "{}",
            String::from_utf8_lossy(&cc.stderr)
        );
        let out = Command::new(dir.join("dltest"))
            .args(["./librunner.so", "t.e0.r0.pre", "9", "out/dl.arena"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "dltest: {:?}", out.status.code());
        assert_eq!(String::from_utf8_lossy(&out.stdout), "t top 2 leaves\n");
        let got = recording::read_f64s(&dir.join("out/dl.arena"), 12 * 9).unwrap();
        let plain = replay::Options {
            poison: false,
            ..scalar
        };
        let want = replay::replay(&rec, &spec.calls, &ks, &plain)
            .unwrap()
            .arena;
        assert!(recording::compare(&got, &want, 9).is_empty());

        // a recording of another arena is refused
        std::fs::write(
            &json,
            RECORDING.replace("[0, 1, 2, 3, 4294967295]", "[0, 1, 3, 2, 4294967295]"),
        )
        .unwrap();
        let out = Command::new(dir.join("runner"))
            .args(["out"])
            .arg(&json)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1));
        assert!(String::from_utf8_lossy(&out.stderr).contains("another arena"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The f32 CUDA runner: its conversion kernels pass ptxas, and on the GPU its arena equals
    /// the scalar replay within f32 rounding (the toy leaves add a few roundings to the inputs').
    /// Skipped as `cuda_runner_reproduces_the_scalar_replay`.
    #[test]
    fn f32_cuda_runner_is_close_to_the_scalar_replay() {
        let ptxas = cuda_tool("PTXAS", "ptxas");
        let fatbinary = cuda_tool("FATBINARY", "fatbinary");
        let have = |tool: &str| {
            Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !have_cc() || !have(&ptxas) || !have(&fatbinary) {
            eprintln!("no C compiler or CUDA toolkit: the f32 CUDA runner is not built");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pacman-f32-runner-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        std::fs::write(dir.join("convert.ptx"), CONVERT_PTX).unwrap();
        let out = Command::new(&ptxas)
            .args(["-arch=sm_80", "convert.ptx", "-o", "convert.cubin"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );

        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let modules = ptx_modules(&spec, &members, crate::ptx::DEFAULT_SM, true, &[]).unwrap();
        assert!(modules.iter().all(|m| !m.contains("f64")));
        // f32 planes: the leaf's plane offsets are in floats
        assert!(
            modules[1].contains("shl.b64 %ldo, %ldo, 2;"),
            "{}",
            modules[1]
        );
        let sms = [local_sm().unwrap_or(80)];
        write_cuda(&dir, &spec, &modules, &sms, true, None).unwrap();
        assert!(std::fs::read_to_string(dir.join("runner.c"))
            .unwrap()
            .contains("#define F32 1"));
        build_cuda(&dir, "cc", modules.len(), &sms, 2, None).unwrap();

        let json = dir.join("t.e0.r0.json");
        std::fs::write(&json, RECORDING).unwrap();
        let mut pre = vec![7.0; 12 * 9];
        for p in [0, 1, 5, 6, 10] {
            for i in 0..9 {
                pre[p * 9 + i] = ((p * 9 + i) as f64 * 0.37).sin();
            }
        }
        pre[11 * 9..12 * 9].fill(0.0);
        let bytes: Vec<u8> = pre.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();
        std::fs::write(json.with_extension("post"), &bytes).unwrap();
        let rec = Recording::read(&json).unwrap();
        let kernels = [
            CpuKernel::load("a", &a).unwrap(),
            CpuKernel::load("b", &b).unwrap(),
        ];
        let ks: Vec<&CpuKernel> = kernels.iter().collect();
        let scalar = replay::Options {
            scalar: true,
            ..replay::Options::default()
        };
        let want = replay::replay(&rec, &spec.calls, &ks, &scalar)
            .unwrap()
            .arena;

        let out = Command::new(dir.join("runner"))
            .args(["--whole", "out"])
            .arg(&json)
            .current_dir(&dir)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !out.status.success()
            && (stderr.contains("libcuda")
                || stderr.contains("cuInit")
                || stderr.contains("cuDeviceGet"))
            && std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none()
        {
            eprintln!("skipped: no usable GPU ({})", stderr.trim());
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        }
        assert!(out.status.success(), "{}", stderr);
        let got = recording::read_f64s(&dir.join("out/t.e0.r0.arena"), 12 * 9).unwrap();
        let mut rounded = 0;
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "value {}: {} for NaN", k, g);
                continue;
            }
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                "value {}: {} for {}",
                k,
                g,
                w
            );
            rounded += (g != w) as usize;
        }
        // the leaves' outputs went through f32
        assert!(rounded > 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The CUDA runner of the same execution equals the scalar replay bit for bit, per tile
    /// and on all points; the library refuses bad arguments before it opens the GPU. Skipped
    /// without the CUDA toolkit (ptxas, fatbinary) or a C compiler, and its GPU steps without a
    /// usable GPU (no driver, or in a sandbox) unless `SJB2PTX_REQUIRE_GPU` is set.
    #[test]
    fn cuda_runner_reproduces_the_scalar_replay() {
        let ptxas = cuda_tool("PTXAS", "ptxas");
        let fatbinary = cuda_tool("FATBINARY", "fatbinary");
        let have = |tool: &str| {
            Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !have_cc() || !have(&ptxas) || !have(&fatbinary) {
            eprintln!("no C compiler or CUDA toolkit: the CUDA runner is not built");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pacman-cuda-runner-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let modules = ptx_modules(&spec, &members, crate::ptx::DEFAULT_SM, false, &[]).unwrap();
        assert_eq!(modules.len(), 2);
        // the leaf's planes are compiled in: leaf 1 writes plane 3 (re) and 8 (im)
        assert!(modules[1].contains(".visible .entry pacman_leaf1("));
        assert!(
            modules[1].contains("mul.lo.u64 %a1, %ldo, 8;"),
            "{}",
            modules[1]
        );
        let sms = [local_sm().unwrap_or(80)];
        write_cuda(&dir, &spec, &modules, &sms, false, None).unwrap();
        build_cuda(&dir, "cc", modules.len(), &sms, 2, None).unwrap();

        // the library's argument checks come before the GPU
        std::fs::write(dir.join("dltest.c"), DLTEST_C).unwrap();
        let cc = Command::new("cc")
            .args(["-O1", "-o", "dltest", "dltest.c", "-ldl"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            cc.status.success(),
            "{}",
            String::from_utf8_lossy(&cc.stderr)
        );

        let json = dir.join("t.e0.r0.json");
        std::fs::write(&json, RECORDING).unwrap();
        let mut pre = vec![7.0; 12 * 9];
        for p in [0, 1, 5, 6, 10] {
            for i in 0..9 {
                pre[p * 9 + i] = ((p * 9 + i) as f64 * 0.37).sin();
            }
        }
        pre[11 * 9..12 * 9].fill(0.0);
        let bytes: Vec<u8> = pre.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();
        std::fs::write(json.with_extension("post"), &bytes).unwrap();
        let rec = Recording::read(&json).unwrap();
        let kernels = [
            CpuKernel::load("a", &a).unwrap(),
            CpuKernel::load("b", &b).unwrap(),
        ];
        let ks: Vec<&CpuKernel> = kernels.iter().collect();
        let scalar = replay::Options {
            scalar: true,
            ..replay::Options::default()
        };
        let want = replay::replay(&rec, &spec.calls, &ks, &scalar)
            .unwrap()
            .arena;

        for args in [&["out"][..], &["--whole", "out"][..]] {
            let out = Command::new(dir.join("runner"))
                .args(args)
                .arg(&json)
                .current_dir(&dir)
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !out.status.success()
                && (stderr.contains("libcuda")
                    || stderr.contains("cuInit")
                    || stderr.contains("cuDeviceGet"))
                && std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none()
            {
                eprintln!("skipped: no usable GPU ({})", stderr.trim());
                std::fs::remove_dir_all(&dir).unwrap();
                return;
            }
            assert!(out.status.success(), "{}", stderr);
            let got = recording::read_f64s(&dir.join("out/t.e0.r0.arena"), 12 * 9).unwrap();
            assert!(recording::compare(&got, &want, 9).is_empty(), "{:?}", args);
        }

        // through dlopen, as RustiCol loads it: argument errors, then two calls; with
        // PACMAN_RUNNER_COPY_BACK=all every written plane comes back
        let dltest = |copy_back: Option<&str>| {
            let mut command = Command::new(dir.join("dltest"));
            command
                .args(["./librunner.so", "t.e0.r0.pre", "9", "out/dl.arena"])
                .current_dir(&dir)
                .env_remove("PACMAN_RUNNER_COPY_BACK")
                .env_remove("PACMAN_SAVE_ARENA");
            if let Some(value) = copy_back {
                command.env("PACMAN_RUNNER_COPY_BACK", value);
            }
            let out = command.output().unwrap();
            assert!(out.status.success(), "dltest: {:?}", out.status.code());
            recording::read_f64s(&dir.join("out/dl.arena"), 12 * 9).unwrap()
        };
        let plain = replay::Options {
            poison: false,
            ..scalar
        };
        let want = replay::replay(&rec, &spec.calls, &ks, &plain)
            .unwrap()
            .arena;
        assert!(recording::compare(&dltest(Some("all")), &want, 9).is_empty());
        // by default only amplitude planes come back, and this arena has none: the host's
        // currents keep their values
        assert!(recording::compare(&dltest(None), &pre, 9).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn plane_mappings_must_fit() {
        let a = plane_application(PRODUCT_PLUS_ONE);
        let mut spec = spec();
        spec.calls.truncate(1);
        let member = spec.calls[0].member.clone();
        let members: HashMap<&str, &[u8]> = [(member.as_str(), &a[..])].into_iter().collect();
        assert!(ptx_modules(&spec, &members, 70, false, &[]).is_ok());
        for (outputs, message) in [
            (vec![2, 2], "writes plane 2 twice"),
            (vec![2, 5], "reads and writes plane 5"),
            (vec![2], "the plane mapping has 5 planes"),
        ] {
            spec.calls[0].outputs = outputs;
            let err = format!(
                "{:#}",
                ptx_modules(&spec, &members, 70, false, &[]).unwrap_err()
            );
            assert!(err.contains(message), "{}", err);
        }
    }

    #[test]
    fn kernels_must_fit_their_tables() {
        let a = plane_application(PRODUCT_PLUS_ONE);
        let mut spec = spec();
        spec.calls.truncate(1);
        spec.calls[0].outputs.push(4);
        let member = spec.calls[0].member.clone();
        let members: HashMap<&str, &[u8]> = [(member.as_str(), &a[..])].into_iter().collect();
        let err = format!("{:#}", objects(&spec, &members, 1, &[]).unwrap_err());
        assert!(
            err.contains("a table of 4 input and 3 output planes"),
            "{}",
            err
        );
        spec.calls[0].member = "missing".into();
        assert!(objects(&spec, &members, 1, &[]).is_err());
    }

    #[test]
    fn c_literals() {
        assert_eq!(c_string("a/b \"c\"\\"), "\"a/b \\\"c\\\"\\\\\"");
        assert_eq!(c_string("é"), "\"\\303\\251\"");
        assert_eq!(c_array(&[1, 2]), "{\n    1u, 2u,\n}");
        let long: Vec<u32> = (0..17).collect();
        assert_eq!(c_array(&long).matches('\n').count(), 3);
    }
    /// The toy execution with its outputs in two amplitudes: a writes amplitude 0 (planes 6, 8),
    /// b reads it and writes amplitude 1 (planes 7, 9).
    fn amplitude_spec() -> Spec {
        let mut spec = spec();
        spec.arena = Arena {
            currents: 3,
            amplitudes: 2,
            momenta: 1,
            parameters: 0,
        };
        spec.canonical_to_physical = vec![0, 1, u32::MAX];
        let (a, b) = (spec.calls[0].clone(), spec.calls[1].clone());
        spec.calls = vec![
            LeafCall::new(a.member, a.label, vec![0, 3, 1, 4], vec![6, 8]),
            LeafCall::new(
                b.member,
                "amplitude:0".into(),
                vec![6, 8, 10, 11],
                vec![7, 9],
            ),
        ];
        spec
    }

    const AMPLITUDE_RECORDING: &str = r#"{"format": "pacman-arena-v1", "dtype": "f64-le",
        "layout": "plane-major", "name": "t.e0.r0", "process": "t", "engine": 0, "recording": 0,
        "method": "all", "num_planes": 12, "num_points": 9, "current_planes": 3,
        "amplitude_planes": 2, "momentum_planes": 1, "model_parameters": 0,
        "regions": [
          {"name": "current_re", "first": 0, "count": 3},
          {"name": "current_im", "first": 3, "count": 3},
          {"name": "amplitude_re", "first": 6, "count": 2},
          {"name": "amplitude_im", "first": 8, "count": 2},
          {"name": "momenta", "first": 10, "count": 1},
          {"name": "zero", "first": 11, "count": 1},
          {"name": "parameters", "first": 12, "count": 0}],
        "canonical_to_physical": [0, 1, 4294967295], "unmapped_physical": 4294967295,
        "source_planes": [0, 1], "tiles": [5, 4],
        "schedule": [
          {"stage": 0, "label": "s1", "amplitude": false,
           "leaves": [{"leaf": 0, "application_path": "evaluators/a.plane.symjit"}]},
          {"stage": 1, "label": "s2", "amplitude": true,
           "leaves": [{"leaf": 0, "application_path": "evaluators/b.plane.symjit"}]}]}"#;

    /// The groups of the reduction tests: (weight, amplitude indices); the last one is empty.
    const GROUPS: &[(f64, &[u32])] = &[(0.75, &[0, 1]), (2.0, &[1]), (-1.5, &[0]), (0.5, &[])];

    /// RustiCol's reduction (`reduce_planes_f64_into_selected_slice`, coherent groups) of the
    /// amplitude planes 6, 7 (re) and 8, 9 (im) of a plane-major arena of `points` points.
    fn reduce_reference(arena: &[f64], points: usize) -> Vec<f64> {
        let mut values = vec![0.0; points];
        for &(weight, indices) in GROUPS {
            for (i, value) in values.iter_mut().enumerate() {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for &k in indices {
                    re += arena[(6 + k as usize) * points + i];
                    im += arena[(8 + k as usize) * points + i];
                }
                *value += weight * (re * re + im * im);
            }
        }
        values
    }

    /// Runs the runner in `dir` with `--reduce` on the amplitude recording, per tile and whole:
    /// the values equal `reduce_reference` of the arena it wrote, bit for bit; an amplitude
    /// index out of range is refused. `None` if the runner could not open the GPU.
    fn reduce_check(dir: &Path) -> Option<()> {
        let groups: String = GROUPS
            .iter()
            .map(|(w, ks)| {
                let ks: Vec<String> = ks.iter().map(|k| k.to_string()).collect();
                format!("{} {}\n", w, ks.join(" "))
            })
            .collect();
        std::fs::write(dir.join("groups.txt"), groups).unwrap();
        std::fs::write(dir.join("bad.txt"), "1 0\n1 2\n").unwrap();
        let json = dir.join("t.e0.r0.json");
        std::fs::write(&json, AMPLITUDE_RECORDING).unwrap();
        let mut pre = vec![7.0; 12 * 9];
        for p in [0, 1, 3, 4, 10] {
            for i in 0..9 {
                pre[p * 9 + i] = ((p * 9 + i) as f64 * 0.37).sin();
            }
        }
        pre[11 * 9..12 * 9].fill(0.0);
        let bytes: Vec<u8> = pre.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();
        for whole in [false, true] {
            let mut args = vec!["--reduce", "groups.txt"];
            if whole {
                args.push("--whole");
            }
            let out = Command::new(dir.join("runner"))
                .args(&args)
                .arg("out")
                .arg(&json)
                .current_dir(dir)
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !out.status.success()
                && (stderr.contains("libcuda")
                    || stderr.contains("cuInit")
                    || stderr.contains("cuDeviceGet"))
                && std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none()
            {
                eprintln!("skipped: no usable GPU ({})", stderr.trim());
                return None;
            }
            assert!(out.status.success(), "{}", stderr);
            let arena = recording::read_f64s(&dir.join("out/t.e0.r0.arena"), 12 * 9).unwrap();
            // the leaves ran: the amplitudes are finite
            assert!(arena[6 * 9..10 * 9].iter().all(|x| x.is_finite()));
            let got = recording::read_f64s(&dir.join("out/t.e0.r0.reduced"), 9).unwrap();
            let want = reduce_reference(&arena, 9);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "point {}: {} for {}", i, g, w);
            }
        }
        let out = Command::new(dir.join("runner"))
            .args(["--reduce", "bad.txt", "out"])
            .arg(&json)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("status 1"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(())
    }

    #[test]
    fn runner_reduces_as_rusticol() {
        if !have_cc() {
            eprintln!("no C compiler (cc): the runner is not built");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pacman-reduce-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = amplitude_spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let kernels = objects(&spec, &members, 1, &[]).unwrap();
        write(&dir, &spec, &kernels).unwrap();
        build(&dir, "cc", kernels.objects.len()).unwrap();
        reduce_check(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cuda_runners_reduce_as_rusticol() {
        let ptxas = cuda_tool("PTXAS", "ptxas");
        let fatbinary = cuda_tool("FATBINARY", "fatbinary");
        let have = |tool: &str| {
            Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !have_cc() || !have(&ptxas) || !have(&fatbinary) {
            eprintln!("no C compiler or CUDA toolkit: the CUDA runner is not built");
            return;
        }
        // the reduction kernel assembles for the oldest target the runners support
        let dir = std::env::temp_dir().join(format!("pacman-cuda-reduce-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("reduce.ptx"), REDUCE_PTX).unwrap();
        let out = Command::new(&ptxas)
            .args(["-arch=sm_80", "reduce.ptx", "-o", "reduce.cubin"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = amplitude_spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let sms = [local_sm().unwrap_or(80)];
        for f32 in [false, true] {
            let sub = dir.join(if f32 { "f32" } else { "f64" });
            std::fs::create_dir_all(sub.join("out")).unwrap();
            let modules = ptx_modules(&spec, &members, crate::ptx::DEFAULT_SM, f32, &[]).unwrap();
            write_cuda(&sub, &spec, &modules, &sms, f32, None).unwrap();
            build_cuda(&sub, "cc", modules.len(), &sms, 2, None).unwrap();
            if reduce_check(&sub).is_none() {
                break;
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
    /// A hybrid runner (f32 leaves on the GPU, reruns on the CPU in f64) of the amplitude
    /// execution against the CPU runner and the f32 CUDA runner, with point 2 out of f32's range:
    /// with no fallback only the non-finite point is rerun, with all of it every value is the
    /// CPU runner's bit for bit, and with 34% the 4 largest of the 8 others are rerun too.
    #[test]
    fn hybrid_runner_reruns_on_the_cpu() {
        let ptxas = cuda_tool("PTXAS", "ptxas");
        let fatbinary = cuda_tool("FATBINARY", "fatbinary");
        let have = |tool: &str| {
            Command::new(tool)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        if !have_cc() || !have(&ptxas) || !have(&fatbinary) {
            eprintln!("no C compiler or CUDA toolkit: the hybrid runner is not built");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pacman-hybrid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (a, b) = (
            plane_application(PRODUCT_PLUS_ONE),
            plane_application(SQUARE_MINUS),
        );
        let spec = amplitude_spec();
        let members: HashMap<&str, &[u8]> = [
            (spec.calls[0].member.as_str(), &a[..]),
            (spec.calls[1].member.as_str(), &b[..]),
        ]
        .into_iter()
        .collect();
        let sms = [local_sm().unwrap_or(80)];
        let kernels = objects(&spec, &members, 1, &[]).unwrap();
        let modules = ptx_modules(&spec, &members, crate::ptx::DEFAULT_SM, true, &[]).unwrap();
        for name in ["cpu", "f32", "hybrid"] {
            let sub = dir.join(name);
            std::fs::create_dir_all(sub.join("out")).unwrap();
            match name {
                "cpu" => {
                    write(&sub, &spec, &kernels).unwrap();
                    build(&sub, "cc", kernels.objects.len()).unwrap();
                }
                "f32" => {
                    write_cuda(&sub, &spec, &modules, &sms, true, None).unwrap();
                    build_cuda(&sub, "cc", modules.len(), &sms, 2, None).unwrap();
                }
                _ => {
                    write_cuda(&sub, &spec, &modules, &sms, true, Some((&kernels, 0.34))).unwrap();
                    build_cuda(
                        &sub,
                        "cc",
                        modules.len(),
                        &sms,
                        2,
                        Some(kernels.objects.len()),
                    )
                    .unwrap();
                }
            }
        }

        let groups: String = GROUPS
            .iter()
            .map(|(w, ks)| {
                let ks: Vec<String> = ks.iter().map(|k| k.to_string()).collect();
                format!("{} {}\n", w, ks.join(" "))
            })
            .collect();
        std::fs::write(dir.join("groups.txt"), groups).unwrap();
        let json = dir.join("t.e0.r0.json");
        std::fs::write(&json, AMPLITUDE_RECORDING).unwrap();
        let mut pre = vec![7.0; 12 * 9];
        for p in [0, 1, 3, 4, 10] {
            for i in 0..9 {
                pre[p * 9 + i] = ((p * 9 + i) as f64 * 0.37).sin();
            }
        }
        // point 2: z0 z1 = 1e50 overflows f32, not f64
        pre[2] = 1e25;
        pre[9 + 2] = 1e25;
        pre[11 * 9..12 * 9].fill(0.0);
        let bytes: Vec<u8> = pre.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &bytes).unwrap();

        // the values of a run of `name` (`fallback`: PACMAN_RUNNER_FALLBACK); None: no GPU
        let values = |name: &str, fallback: Option<&str>, whole: bool| -> Option<Vec<f64>> {
            let sub = dir.join(name);
            let mut command = Command::new(sub.join("runner"));
            command.args(["--reduce", "../groups.txt"]);
            if whole {
                command.arg("--whole");
            }
            command.arg("out").arg(&json).current_dir(&sub);
            match fallback {
                Some(f) => command.env("PACMAN_RUNNER_FALLBACK", f),
                None => command.env_remove("PACMAN_RUNNER_FALLBACK"),
            };
            let out = command.output().unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !out.status.success()
                && (stderr.contains("libcuda")
                    || stderr.contains("cuInit")
                    || stderr.contains("cuDeviceGet"))
                && std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none()
            {
                eprintln!("skipped: no usable GPU ({})", stderr.trim());
                return None;
            }
            assert!(out.status.success(), "{}: {}", name, stderr);
            Some(recording::read_f64s(&sub.join("out/t.e0.r0.reduced"), 9).unwrap())
        };
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

        let cpu = values("cpu", None, true).unwrap();
        assert!(cpu.iter().all(|x| x.is_finite()));
        let Some(f32) = values("f32", None, true) else {
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        };
        assert!(!f32[2].is_finite(), "{:?}", f32);
        assert!((0..9).filter(|&i| i != 2).all(|i| f32[i] != cpu[i]));

        // no fallback: only the non-finite point
        let none = values("hybrid", Some("0"), true).unwrap();
        for i in 0..9 {
            let want = if i == 2 { cpu[i] } else { f32[i] };
            assert_eq!(none[i].to_bits(), want.to_bits(), "point {}", i);
        }
        // all points, at once and per tile: the CPU runner's values
        assert_eq!(
            bits(&values("hybrid", Some("100%"), true).unwrap()),
            bits(&cpu)
        );
        assert_eq!(
            bits(&values("hybrid", Some("1"), false).unwrap()),
            bits(&cpu)
        );
        // the default, 34%: ceil(0.34 x 9) = 4 of the finite points, the largest f32 values
        let mut finite: Vec<usize> = (0..9).filter(|&i| i != 2).collect();
        finite.sort_by(|&i, &j| f32[j].partial_cmp(&f32[i]).unwrap());
        let rerun: Vec<usize> = finite[..4].iter().copied().chain([2]).collect();
        let part = values("hybrid", None, true).unwrap();
        for i in 0..9 {
            let want = if rerun.contains(&i) { cpu[i] } else { f32[i] };
            assert_eq!(part[i].to_bits(), want.to_bits(), "point {}", i);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A synthetic recurrence plan on the recurrence runner: every op (clears, the five
    /// contributions with initializing and accumulating rows on interleaved destinations, the
    /// three propagators, identity finalization, closures), once per `PACMAN_RECURRENCE_KERNEL`.
    /// `f64` with the contribution rows unsorted (`PACMAN_RECURRENCE_SORT=0`) equals `f64` bit for
    /// bit; `df64` (the default) agrees with it to 1e-9. Under each kernel, the amplitudes zeroed on the device and kept there reduce
    /// (`pacman_recurrence_v1_reduce`) to the host's reduction of the copied-back ones, bit for
    /// bit; the first calls pass the struct of before the reduction. Skipped without nvcc or a C compiler, and its GPU steps without a usable GPU
    /// unless `SJB2PTX_REQUIRE_GPU` is set.
    const RECTEST_C: &str = r#"
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t abi_version, struct_size;
    const uint64_t *words;
    uint64_t words_key;
    uint32_t word_count;
    uint8_t width[8];
    double *current_re, *current_im, *amplitude_re, *amplitude_im;
    const double *momenta, *scale_re, *scale_im, *factor_re, *factor_im;
    const uint32_t *upload_current, *upload_amplitude, *download_amplitude, *download_current;
    uint32_t current_planes, amplitude_planes, momentum_planes, scale_count, factor_count;
    uint32_t upload_current_runs, upload_amplitude_runs, download_amplitude_runs,
        download_current_runs;
    uint32_t stride, points;
    const uint32_t *zero_amplitude;
    uint32_t zero_amplitude_runs, flags;
} call_t;

typedef struct {
    uint32_t abi_version, struct_size;
    const uint32_t *planes;
    const double *weights;
    double *output;
    uint32_t count, points;
} reduce_t;

enum { P = 12, A = 9, F = 6, X = 3, S = 3, K = 4 };
static uint64_t words[64];
static uint32_t nwords;
static uint64_t w_;
static int at_;
static void put(uint64_t v, int bits) { w_ |= v << at_; at_ += bits; }
static void begin(int op) { w_ = 0; at_ = 0; put(op, 4); }
static void end(void) { words[nwords++] = w_; }
static void clear(int base, int count) { begin(1); put(base, P); put(count, P); end(); }
static void contrib(int op, int init, int slot, int p0, int p1, int dst, int f0, int f1, int x) {
    begin(op); put(init, 1); put(slot, S); put(p0, P); put(p1, P); put(dst, P);
    put(f0, F); put(f1, F); put(x, X); end();
}
static void prop(int op, int base, int form, int x) { begin(op); put(base, P); put(form, F); put(x, X); end(); }
static void identity(int base, int count, int x) { begin(10); put(base, P); put(count, K); put(x, X); end(); }
static void closure(int p0, int p1, int amp, int x, int first, int count) {
    begin(11); put(p0, P); put(p1, P); put(amp, A); put(x, X); put(first, X); put(count, K); end();
}

int main(int argc, char **argv) {
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    int (*run)(const call_t *) = (int (*)(const call_t *))dlsym(lib, "pacman_recurrence_v1_run");
    int (*reduce)(const reduce_t *) = (int (*)(const reduce_t *))dlsym(lib, "pacman_recurrence_v1_reduce");
    if (!run || !reduce) return 1;
    /* sources: Weyl 0, 2; vectors 4, 8, 12; tensor 16 (22 planes); stage 1 at 32..43, stage 2 at 44..49 */
    clear(32, 12);
    contrib(2, 1, 0, 0, 4, 32, 0, 0, 1);
    contrib(4, 1, 1, 16, 4, 34, 0, 0, 2);
    contrib(5, 1, 2, 4, 8, 38, 0, 0, 0);
    contrib(3, 0, 1, 2, 8, 32, 0, 0, 3);
    contrib(6, 0, 0, 4, 8, 34, 0, 1, 4);
    contrib(5, 0, 1, 8, 12, 38, 0, 0, 5);
    contrib(2, 0, 2, 0, 12, 32, 0, 0, 0);
    contrib(4, 0, 0, 16, 12, 34, 0, 0, 1);
    prop(7, 32, 2, 1);
    prop(9, 34, 3, 2);
    identity(38, 6, 3);
    clear(44, 6);
    contrib(3, 1, 0, 32, 34, 44, 0, 0, 1);
    contrib(4, 1, 1, 38, 34, 46, 0, 0, 2);
    contrib(2, 0, 2, 0, 34, 44, 0, 0, 3);
    contrib(6, 0, 1, 34, 4, 46, 1, 2, 4);
    prop(8, 44, 1, 5);
    closure(44, 0, 0, 0, 2, 2);
    closure(46, 8, 1, 1, 3, 2);
    closure(32, 2, 2, 2, 0, 2);
    enum { POINTS = 300, STRIDE = 320, CP = 50, AP = 3, MP = 16 };
    static double cre[CP * STRIDE], cim[CP * STRIDE], are[AP * STRIDE], aim[AP * STRIDE], mom[MP * STRIDE];
    for (int k = 0; k < CP * STRIDE; k++) {
        cre[k] = sin(0.37 * k + 0.1);
        cim[k] = cos(0.53 * k + 0.2);
    }
    for (int f = 0; f < 4; f++)
        for (int i = 0; i < STRIDE; i++)
            for (int mu = 0; mu < 4; mu++)
                mom[(f * 4 + mu) * STRIDE + i] = mu == 0 ? 2.5 + 0.4 * sin(0.7 * i + f) : sin(0.3 * i + mu + 2 * f);
    double sre[3] = {0.5, -1.25, 0.75}, sim[3] = {0.25, 0.5, -0.125};
    double fre[6] = {1.0, -0.5, 0.25, 2.0, -1.0, 0.75}, fim[6] = {0.0, 0.5, -0.75, 0.0, 0.25, -0.5};
    uint32_t up_c[2] = {0, 22}, up_a[2] = {0, AP}, down_a[2] = {0, AP};
    call_t c;
    memset(&c, 0, sizeof c);
    c.abi_version = 1;
    c.struct_size = (uint32_t)offsetof(call_t, zero_amplitude); /* the struct before the reduction */
    c.words = words;
    c.words_key = 7;
    c.word_count = nwords;
    uint8_t width[8] = {P, A, F, X, S, K, 0, 0};
    memcpy(c.width, width, 8);
    c.current_re = cre; c.current_im = cim; c.amplitude_re = are; c.amplitude_im = aim;
    c.momenta = mom; c.scale_re = sre; c.scale_im = sim; c.factor_re = fre; c.factor_im = fim;
    c.upload_current = up_c; c.upload_amplitude = up_a; c.download_amplitude = down_a;
    c.current_planes = CP; c.amplitude_planes = AP; c.momentum_planes = MP; c.scale_count = 3;
    c.factor_count = 6; c.upload_current_runs = 1; c.upload_amplitude_runs = 1;
    c.download_amplitude_runs = 1; c.stride = STRIDE; c.points = POINTS;
    /* twice: the second call reuses the cached program and the device arena */
    for (int r = 0; r < 2; r++) {
        memset(are, 0, sizeof are);
        memset(aim, 0, sizeof aim);
        int status = run(&c);
        if (status == 3) return 77; /* device error: no usable GPU */
        if (status != 0) { fprintf(stderr, "status %d\n", status); return 1; }
    }
    /* the amplitudes cleared on the device and kept there, then reduced: as the host reduces them */
    static double want[POINTS], got[POINTS];
    uint32_t planes[4] = {2, 0, 1, 0};
    double weights[4] = {0.75, 1.5, -0.25, 2.0};
    for (int i = 0; i < POINTS; i++) {
        want[i] = got[i] = 0.125 * i;
        for (int k = 0; k < 4; k++) {
            double re = are[planes[k] * STRIDE + i], im = aim[planes[k] * STRIDE + i];
            want[i] += weights[k] * fma(re, re, im * im);
        }
    }
    static double are2[AP * STRIDE], aim2[AP * STRIDE];
    for (int k = 0; k < AP * STRIDE; k++) are2[k] = aim2[k] = 1e300; /* must not be uploaded */
    call_t k2 = c;
    k2.struct_size = sizeof k2;
    k2.amplitude_re = are2;
    k2.amplitude_im = aim2;
    k2.upload_amplitude_runs = 0;
    k2.zero_amplitude = up_a;
    k2.zero_amplitude_runs = 1;
    k2.flags = 1;
    if (run(&k2) != 0) return 1;
    for (int k = 0; k < AP * STRIDE; k++)
        if (are2[k] != 1e300 || aim2[k] != 1e300) { fprintf(stderr, "kept amplitudes were copied back\n"); return 1; }
    reduce_t r = {1, sizeof r, planes, weights, got, 4, POINTS};
    if (reduce(&r) != 0) { fprintf(stderr, "reduce failed\n"); return 1; }
    for (int i = 0; i < POINTS; i++)
        if (memcmp(&want[i], &got[i], 8)) {
            fprintf(stderr, "reduction, point %d: %.17g for %.17g\n", i, got[i], want[i]);
            return 1;
        }
    for (int p = 0; p < AP; p++)
        for (int i = 0; i < POINTS; i++) printf("%.17g %.17g\n", are[p * STRIDE + i], aim[p * STRIDE + i]);
    return 0;
}
"#;

    #[test]
    fn recurrence_runner_variants_agree() {
        let nvcc = cuda_tool("NVCC", "nvcc");
        let have_nvcc = Command::new(&nvcc)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !have_cc() || !have_nvcc {
            eprintln!("no C compiler or nvcc: the recurrence runner is not built");
            return;
        }
        let dir = std::env::temp_dir().join(format!("pacman-recurrence-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let sm = local_sm().unwrap_or(crate::ptx::DEFAULT_SM.max(80));
        build_recurrence(&dir, &[sm]).unwrap();
        std::fs::write(dir.join("rectest.c"), RECTEST_C).unwrap();
        let cc = Command::new("cc")
            .args(["-O1", "-ffp-contract=off", "-o", "rectest", "rectest.c", "-ldl", "-lm"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(cc.status.success(), "{}", String::from_utf8_lossy(&cc.stderr));
        let lib = dir.join("librecurrence.so");
        let mut results: Vec<(&str, Vec<f64>)> = Vec::new();
        for kernel in ["f64", "f64 unsorted", "df64", ""] {
            let mut cmd = Command::new(dir.join("rectest"));
            cmd.arg(&lib).current_dir(&dir);
            if kernel == "f64 unsorted" {
                cmd.env("PACMAN_RECURRENCE_KERNEL", "f64").env("PACMAN_RECURRENCE_SORT", "0");
            } else if kernel.is_empty() {
                cmd.env_remove("PACMAN_RECURRENCE_KERNEL");
            } else {
                cmd.env("PACMAN_RECURRENCE_KERNEL", kernel);
            }
            let out = cmd.output().unwrap();
            if out.status.code() == Some(77) && std::env::var_os("SJB2PTX_REQUIRE_GPU").is_none() {
                eprintln!("skipped: no usable GPU");
                std::fs::remove_dir_all(&dir).unwrap();
                return;
            }
            assert!(out.status.success(), "{}: {}", kernel, String::from_utf8_lossy(&out.stderr));
            let values: Vec<f64> = String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(values.len(), 3 * 300 * 2, "{}", kernel);
            results.push((if kernel.is_empty() { "default" } else { kernel }, values));
        }
        let want = &results[0].1;
        assert!(want.iter().all(|v| v.is_finite()) && want.iter().any(|v| *v != 0.0));
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (kernel, got) in &results[1..] {
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                if *kernel == "f64 unsorted" {
                    assert_eq!(g.to_bits(), w.to_bits(), "{} value {}: {} for {}", kernel, k, g, w);
                } else {
                    assert!((g - w).abs() <= 1e-9 * scale, "{} value {}: {} for {}", kernel, k, g, w);
                }
            }
        }
        // the default is df64: it equals df64's values exactly
        let df64 = &results.iter().find(|r| r.0 == "df64").unwrap().1;
        let default = &results.iter().find(|r| r.0 == "default").unwrap().1;
        assert!(df64.iter().zip(default).all(|(a, b)| a.to_bits() == b.to_bits()));
        // and it is not the f64 arithmetic
        assert!(df64.iter().zip(want).any(|(a, b)| a.to_bits() != b.to_bits()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
