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
//! - `main.c` reads recordings (`NAME.json`, `NAME.pre`), runs their schedules per recorded tile
//!   and writes the arena as `NAME.arena` (the layout of `.post`), for `pacman replay
//!   --against`.

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

/// The kernels of `spec` as object files: one per distinct kernel (`pacman_kernel_<k>`), and
/// for each leaf the index of its kernel. `members` maps member paths to their bytes.
pub fn objects(spec: &Spec, members: &HashMap<&str, &[u8]>) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let mut objects = Vec::new();
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
        let model = SavedModel::read(&mut &bytes[..])
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
        let object = model
            .object(&format!("pacman_kernel_{}", k), target)
            .and_then(|o| o.build())
            .with_context(|| format!("cannot compile `{}` into an object", call.member))?;
        objects.push(object);
        index.insert(&call.member, k);
        of_leaf.push(k);
    }
    Ok((objects, of_leaf))
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

/// `runner.c`: the execution `spec` with the kernels `of_leaf` (from `objects`).
pub fn runner_c(spec: &Spec, of_leaf: &[usize], kernels: usize) -> String {
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
        "/* runner.c: the execution {} of {} ({} leaves, {} arena planes), generated by pacman {}.\n\
         \x20  The interface is in pacman_runner.h. */\n\n\
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
    for k in 0..kernels {
        let _ = writeln!(
            s,
            "int pacman_kernel_{}(double *, const symjit_slice *, size_t, const double *);",
            k
        );
    }
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
    s.push_str("static const kernel_fn LEAF_KERNELS[] = {\n");
    for k in of_leaf {
        let _ = writeln!(s, "    pacman_kernel_{},", k);
    }
    s.push_str("};\n\nstatic const char *const LEAF_PATHS[] = {\n");
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
    s.push_str(RUN_C);
    s
}

/// The fixed part of `runner.c`.
const RUN_C: &str = r#"const pacman_runner_metadata *pacman_runner_v1_metadata(void) {
    return &METADATA;
}

/* the scalar kernel of `leaf` on each point of start..stop */
static int run_leaf(uint32_t leaf, double *const *planes, uint32_t start, uint32_t stop) {
    symjit_slice table[MAX_TABLE];
    const uint32_t *map = LEAF_PLANES[leaf];
    uint32_t n = LEAF_INPUTS[leaf] + LEAF_OUTPUTS[leaf];
    for (uint32_t i = 0; i < n; i++) {
        table[i].ptr = planes[map[i]];
        table[i].len = stop;
    }
    kernel_fn kernel = LEAF_KERNELS[leaf];
    for (uint32_t point = start; point < stop; point++) {
        if (kernel(NULL, table, point, NULL) != 0) {
            return PACMAN_RUNNER_KERNEL_STATUS;
        }
    }
    return PACMAN_RUNNER_OK;
}

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
    for (uint32_t k = 0; k < count; k++) {
        int status = run_leaf(leaves == NULL ? k : leaves[k], planes, point_start,
                              point_start + point_count);
        if (status != PACMAN_RUNNER_OK) {
            return status;
        }
    }
    return PACMAN_RUNNER_OK;
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
#define PACMAN_RUNNER_BAD_ARGUMENTS 1   /* counts, null pointers, leaf indices, point range */
#define PACMAN_RUNNER_KERNEL_STATUS 2   /* a kernel returned a nonzero status */

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

#ifdef __cplusplus
}
#endif

#endif
"#;

/// `main.c`: the executable that replays recordings.
pub const MAIN_C: &str = r#"/* main.c: runs arena recordings of RustiCol (NAME.json, NAME.pre) through the runner and
   writes the arena after the run as OUTDIR/NAME.arena (the layout of NAME.post), for
   `pacman replay --against OUTDIR`. Generated by pacman.

       runner [--whole] [--no-poison] OUTDIR RECORDING.json...

   The schedule (the recording's leaves) runs per recorded tile, or on all points at once
   (--whole). Planes the schedule writes before it reads them are first filled with NaNs
   (0x7ff8deadbeef0000 | plane), unless --no-poison. */

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

static int run_recording(const char *json_path, const char *outdir, int whole, int poison) {
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
    double *arena = NULL;
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
        int s = pacman_runner_v1_run(table, (uint32_t)planes, schedule, (uint32_t)num_leaves, 0,
                                     (uint32_t)n);
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
    free(first);
    return status;
}

int main(int argc, char **argv) {
    int whole = 0, poison = 1, i = 1;
    for (; i < argc && argv[i][0] == '-'; i++) {
        if (strcmp(argv[i], "--whole") == 0) {
            whole = 1;
        } else if (strcmp(argv[i], "--no-poison") == 0) {
            poison = 0;
        } else {
            fprintf(stderr, "runner: unknown option %s\n", argv[i]);
            return 2;
        }
    }
    if (argc - i < 2) {
        const pacman_runner_metadata *md = pacman_runner_v1_metadata();
        fprintf(stderr,
                "usage: %s [--whole] [--no-poison] OUTDIR RECORDING.json...\n"
                "runs recordings of the execution %s of %s (%u leaves, %u planes)\n",
                argv[0], md->execution, md->process, md->num_leaves, md->num_planes);
        return 2;
    }
    int status = 0;
    for (int k = i + 1; k < argc; k++) {
        status |= run_recording(argv[k], argv[i], whole, poison);
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
         \t$(CC) $(CFLAGS) -fPIC -c runner.c -o runner.o\n\n\
         librunner.so: runner.o $(KERNELS)\n\
         \t$(CC) -shared -o librunner.so runner.o $(KERNELS)\n\n\
         runner: main.c runner.o $(KERNELS) pacman_runner.h\n\
         \t$(CC) $(CFLAGS) -o runner main.c runner.o $(KERNELS)\n",
        objects.join(" ")
    )
}

/// Writes the runner of `spec` into `dir`: `kernels/k<k>.o`, `runner.c`, `pacman_runner.h`,
/// `main.c`, `Makefile`.
pub fn write(dir: &Path, spec: &Spec, objects: &[Vec<u8>], of_leaf: &[usize]) -> Result<()> {
    let write = |name: &str, bytes: &[u8]| -> Result<()> {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create `{}`", parent.display()))?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("cannot write `{}`", path.display()))
    };
    for (k, o) in objects.iter().enumerate() {
        write(&format!("kernels/k{}.o", k), o)?;
    }
    write(
        "runner.c",
        runner_c(spec, of_leaf, objects.len()).as_bytes(),
    )?;
    write("pacman_runner.h", RUNNER_H.as_bytes())?;
    write("main.c", MAIN_C.as_bytes())?;
    write("Makefile", makefile(objects.len()).as_bytes())?;
    Ok(())
}

/// Builds `runner` and `librunner.so` in `dir` with the C compiler `cc`, as the `Makefile` does.
pub fn build(dir: &Path, cc: &str, kernels: usize) -> Result<()> {
    let objects: Vec<String> = (0..kernels).map(|k| format!("kernels/k{}.o", k)).collect();
    let steps: Vec<Vec<String>> = vec![
        ["-O1", "-fPIC", "-c", "runner.c", "-o", "runner.o"]
            .map(String::from)
            .to_vec(),
        ["-shared", "-o", "librunner.so", "runner.o"]
            .map(String::from)
            .into_iter()
            .chain(objects.iter().cloned())
            .collect(),
        ["-O1", "-o", "runner", "main.c", "runner.o"]
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
        let (objects, of_leaf) = objects(&spec, &members).unwrap();
        assert_eq!(of_leaf, vec![0, 1]);
        write(&dir, &spec, &objects, &of_leaf).unwrap();
        build(&dir, "cc", objects.len()).unwrap();

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

    #[test]
    fn kernels_must_fit_their_tables() {
        let a = plane_application(PRODUCT_PLUS_ONE);
        let mut spec = spec();
        spec.calls.truncate(1);
        spec.calls[0].outputs.push(4);
        let member = spec.calls[0].member.clone();
        let members: HashMap<&str, &[u8]> = [(member.as_str(), &a[..])].into_iter().collect();
        let err = format!("{:#}", objects(&spec, &members).unwrap_err());
        assert!(
            err.contains("a table of 4 input and 3 output planes"),
            "{}",
            err
        );
        spec.calls[0].member = "missing".into();
        assert!(objects(&spec, &members).is_err());
    }

    #[test]
    fn c_literals() {
        assert_eq!(c_string("a/b \"c\"\\"), "\"a/b \\\"c\\\"\\\\\"");
        assert_eq!(c_string("é"), "\"\\303\\251\"");
        assert_eq!(c_array(&[1, 2]), "{\n    1u, 2u,\n}");
        let long: Vec<u32> = (0..17).collect();
        assert_eq!(c_array(&long).matches('\n').count(), 3);
    }
}
