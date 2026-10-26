//! pacman replay: runs the kernels of a RustiCol arena recording (`recording`) on its `.pre`
//! snapshot, as RustiCol runs them, to reproduce its `.post` snapshot. See `docs/PACMAN.md`.
//!
//! - The leaves come from `execution.json` (`bindings`), the current planes from the
//!   recording's own `canonical_to_physical` (RustiCol's map, which `bindings` rebuilds).
//! - A kernel is loaded as `rusticol-core` loads it (`evaluator/symjit_plane.rs`,
//!   `load_bytes`): `Application::load` with its explicit loader configuration (the saved one
//!   replaces it), `prepare_simd`, `seal`, the raw plane kernels; the SIMD kernel only if it
//!   supports indirect calls, with its lane count.
//! - A leaf runs on a tile as `invoke_range` runs it: SIMD calls for whole lane blocks (a
//!   nonzero status redoes the block with the scalar kernel), then scalar calls for the rest;
//!   SIMD only if no input plane is also an output plane (`validate_raw_aliases`).

use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use symjit::PlaneDescriptor;
use symjit::{Applet, Application, Compiled, CompiledPlaneFunc, CompilerType, Config, Defuns};

use crate::bindings::{leaf_label, Execution, Plane, Process};
use crate::pacman::Kernel;
use crate::recording::{self, PlaneDiff, Recording};

/// One kernel call of the schedule: the kernel and its descriptor table as arena planes.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafCall {
    /// the kernel's member path in `evaluators.pacbin`
    pub member: String,
    /// `stage<k>:<leaf>` or `amplitude:<leaf>`
    pub label: String,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    /// no input plane is an output plane
    pub simd_safe: bool,
}

impl LeafCall {
    pub fn new(member: String, label: String, inputs: Vec<u32>, outputs: Vec<u32>) -> LeafCall {
        let simd_safe = !inputs.iter().any(|p| outputs.contains(p));
        LeafCall {
            member,
            label,
            inputs,
            outputs,
            simd_safe,
        }
    }
}

/// A recording's leaf calls, and what was noticed on the way.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// the execution of `execution.json` the recording ran (`a = b` if several have its
    /// leaves)
    pub execution: String,
    pub calls: Vec<LeafCall>,
    pub warnings: Vec<String>,
}

/// The leaf calls of `rec`'s schedule in the artifact's `processes`: the execution is the one
/// of `processes/<process>` whose leaves include every recorded leaf.
pub fn leaf_calls(rec: &Recording, processes: &[Process]) -> Result<Resolved> {
    let prefix = format!("processes/{}", rec.process);
    let process = processes
        .iter()
        .find(|p| p.prefix == prefix)
        .ok_or_else(|| anyhow!("the artifact has no process `{}`", rec.process))?;
    let matching: Vec<&Execution> = process
        .executions
        .iter()
        .filter(|e| {
            rec.schedule.iter().all(|s| {
                e.leaves()
                    .any(|(_, _, l)| l.application == s.application_path)
            })
        })
        .collect();
    let execution = matching.first().ok_or_else(|| {
        anyhow!(
            "no execution of `{}` has the recorded leaves (first: `{}`)",
            process.file.display(),
            rec.schedule[0].application_path
        )
    })?;
    let mut resolved = resolve(rec, &prefix, execution)?;
    // e.g. helicity_sum and its colour selector: the same kernels on the same planes
    for e in &matching[1..] {
        resolved.execution.push_str(" = ");
        resolved.execution.push_str(&e.name);
    }
    Ok(resolved)
}

fn resolve(rec: &Recording, prefix: &str, e: &Execution) -> Result<Resolved> {
    let mut warnings = Vec::new();
    let arena = rec.arena();
    if (e.amplitudes, e.momentum_planes, e.model_parameters)
        != (
            rec.amplitude_planes as usize,
            rec.momentum_planes as usize,
            rec.model_parameters as usize,
        )
    {
        bail!(
            "execution `{}` has {} amplitudes, {} momentum planes and {} model parameters, \
             the recording {}, {} and {}",
            e.name,
            e.amplitudes,
            e.momentum_planes,
            e.model_parameters,
            rec.amplitude_planes,
            rec.momentum_planes,
            rec.model_parameters
        );
    }
    if rec.canonical_to_physical.len() != e.value_components {
        bail!(
            "the recording maps {} current components, execution `{}` has {}",
            rec.canonical_to_physical.len(),
            e.name,
            e.value_components
        );
    }
    if e.physical_planes != rec.current_planes {
        warnings.push(format!(
            "pacman derives {} current planes, RustiCol {}",
            e.physical_planes, rec.current_planes
        ));
    }

    let mut differing = 0;
    let mut calls = Vec::new();
    for s in rec.schedule.iter() {
        let (stage, j, leaf) = e
            .leaves()
            .find(|(_, _, l)| l.application == s.application_path)
            .expect("the execution was chosen by its leaves");
        let label = leaf_label(&e.stages, stage, j);
        if stage.label != s.label || j != s.leaf {
            warnings.push(format!(
                "`{}` is leaf {} of `{}` in the recording, {} of `{}` in execution.json",
                s.application_path, s.leaf, s.label, j, stage.label
            ));
        }
        let mut planes = |list: &[Plane]| -> Result<Vec<u32>> {
            list.iter()
                .map(|&p| {
                    if let Plane::Current {
                        physical,
                        canonical,
                        ..
                    } = p
                    {
                        if rec.canonical_to_physical.get(canonical as usize) != Some(&physical) {
                            differing += 1;
                        }
                    }
                    arena
                        .index_with(p, leaf, &rec.canonical_to_physical)
                        .ok_or_else(|| {
                            anyhow!(
                                "{} ({}): {:?} has no plane in the recording's map",
                                label,
                                s.application_path,
                                p
                            )
                        })
                })
                .collect()
        };
        let inputs = planes(&leaf.inputs)?;
        let outputs = planes(&leaf.outputs)?;
        let mut sorted = outputs.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            bail!("{}: two outputs share a plane", label);
        }
        calls.push(LeafCall::new(
            format!("{}/{}", prefix, s.application_path),
            label,
            inputs,
            outputs,
        ));
    }
    if differing > 0 {
        warnings.push(format!(
            "{} current descriptors are on other planes in pacman's map than in RustiCol's \
             (`bindings` must track `derive_stage_current_layout`); RustiCol's are used",
            differing
        ));
    }
    Ok(Resolved {
        execution: e.name.clone(),
        calls,
        warnings,
    })
}

/// A plane kernel loaded for this machine as RustiCol loads it.
pub struct CpuKernel {
    applet: Applet,
    scalar: CompiledPlaneFunc<f64>,
    simd: Option<CompiledPlaneFunc<f64>>,
    /// the SIMD kernel's points per call (1 without one)
    pub lanes: usize,
    /// descriptors: input planes, output planes
    pub inputs: usize,
    pub outputs: usize,
}

/// `explicit_loader_config`: every option off; `Application::load` replaces it with the
/// saved configuration, except for its (empty) functions.
fn loader_config() -> Result<Config> {
    let mut config = Config::new(CompilerType::Native, 0)?;
    config.set_opt_level(0);
    config.set_cse(false);
    config.set_fastmath(false);
    config.set_simd(false);
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_complex(false);
    config.set_fast_complex(false);
    config.set_threads(false);
    config.set_symbolica(false);
    config.set_compact(false);
    config.set_compress(false);
    config.set_direct(false);
    config.set_huge(false);
    config.set_parallel_mul(false);
    config.set_direct_arena(false);
    config.set_defuns(Defuns::new());
    Ok(config)
}

impl CpuKernel {
    /// Loads the plane application `bytes` (`path` names it in messages). A panic of the
    /// compiler is returned as an error.
    pub fn load(path: &str, bytes: &[u8]) -> Result<CpuKernel> {
        let shape = Kernel::read(path, bytes);
        if let Some(err) = shape.error {
            bail!("{}: {}", path, err);
        }
        let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<CpuKernel> {
            let config = loader_config()?;
            let mut input = bytes;
            let mut app = Application::load(&mut input, &config)?;
            if !input.is_empty() {
                bail!("{} trailing bytes", input.len());
            }
            app.prepare_simd();
            let applet = app.seal()?;
            let scalar = applet.scalar_plane_kernel().ok_or_else(|| {
                anyhow!("no scalar plane kernel (not a direct-arena application)")
            })?;
            let (simd, lanes) = match applet.compiled_simd.as_ref() {
                Some(c) if c.support_indirect() => (applet.simd_plane_kernel(), c.count_lanes()),
                _ => (None, 1),
            };
            if simd.is_some() && lanes != 2 && lanes != 4 {
                bail!("a SIMD kernel of {} lanes (RustiCol takes 2 or 4)", lanes);
            }
            Ok(CpuKernel {
                applet,
                scalar,
                simd,
                lanes,
                inputs: shape.inputs(),
                outputs: shape.obs,
            })
        }));
        match result {
            Ok(r) => r.map_err(|e| anyhow!("{}: {:#}", path, e)),
            Err(payload) => Err(anyhow!(
                "{}: symjit panicked: {}",
                path,
                payload
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("(no message)")
            )),
        }
    }

    pub fn has_simd(&self) -> bool {
        self.simd.is_some()
    }
}

/// Loads `members` (path, bytes) on all cores, in order. The panic hook is silenced around the
/// whole scope (it is global), so compiler panics are only returned as errors.
pub fn load_all(members: &[(&str, &[u8])]) -> Vec<Result<CpuKernel>> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(members.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<std::sync::Mutex<Option<Result<CpuKernel>>>> =
        (0..members.len()).map(|_| Default::default()).collect();
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= members.len() {
                    break;
                }
                let (path, bytes) = members[i];
                *results[i].lock().unwrap() = Some(CpuKernel::load(path, bytes));
            });
        }
    });
    panic::set_hook(hook);
    results
        .into_iter()
        .map(|r| r.into_inner().unwrap().expect("every member is loaded"))
        .collect()
}

/// What one leaf did over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CallStats {
    pub simd_calls: u64,
    pub scalar_calls: u64,
    /// SIMD blocks redone with the scalar kernel (a nonzero SIMD status)
    pub redone: u64,
    pub time: Duration,
}

/// `invoke_range` on points `0..n` of `table`.
fn invoke(
    k: &CpuKernel,
    use_simd: bool,
    table: *const PlaneDescriptor<f64>,
    n: usize,
    stats: &mut CallStats,
) -> Result<()> {
    let params = k.applet.params.as_ptr();
    let null = std::ptr::null();
    let mut point = 0;
    if let Some(simd) = k.simd.filter(|_| use_simd && k.lanes > 1) {
        // the scalar head up to the first lane boundary: none, a tile starts at point 0
        while n - point >= k.lanes {
            // SAFETY: every descriptor covers points 0..n of a live, distinct arena plane
            // (`run_tile`), and the block lies inside them
            let status = unsafe { simd(null, table, point, params) };
            stats.simd_calls += 1;
            if status != 0 {
                stats.redone += 1;
                for p in point..point + k.lanes {
                    // SAFETY: as above
                    let status = unsafe { (k.scalar)(null, table, p, params) };
                    stats.scalar_calls += 1;
                    if status != 0 {
                        bail!(
                            "the scalar kernel returned status {} at point {}",
                            status,
                            p
                        );
                    }
                }
            }
            point += k.lanes;
        }
    }
    while point < n {
        // SAFETY: as above
        let status = unsafe { (k.scalar)(null, table, point, params) };
        stats.scalar_calls += 1;
        if status != 0 {
            bail!(
                "the scalar kernel returned status {} at point {}",
                status,
                point
            );
        }
        point += 1;
    }
    Ok(())
}

/// Runs `calls` in order on one tile: points `offset..offset + n` of a plane-major `arena`
/// of `ld` points per plane. `kernels[i]` is the kernel of `calls[i]`; `scalar` uses no SIMD
/// kernel. Adds to `stats[i]`.
#[allow(clippy::too_many_arguments)]
pub fn run_tile(
    arena: &mut [f64],
    ld: usize,
    offset: usize,
    n: usize,
    calls: &[LeafCall],
    kernels: &[&CpuKernel],
    scalar: bool,
    stats: &mut [CallStats],
) -> Result<()> {
    assert_eq!(calls.len(), kernels.len());
    assert_eq!(calls.len(), stats.len());
    if n == 0 || offset + n > ld || !arena.len().is_multiple_of(ld) {
        bail!(
            "tile {}..{} does not fit planes of {} points",
            offset,
            offset + n,
            ld
        );
    }
    let planes = arena.len() / ld;
    let base = arena.as_mut_ptr();
    for ((call, k), s) in calls.iter().zip(kernels).zip(stats.iter_mut()) {
        if (call.inputs.len(), call.outputs.len()) != (k.inputs, k.outputs) {
            bail!(
                "{}: a table of {} input and {} output planes for a kernel of {} and {}",
                call.label,
                call.inputs.len(),
                call.outputs.len(),
                k.inputs,
                k.outputs
            );
        }
        let mut table = Vec::with_capacity(call.inputs.len() + call.outputs.len());
        for &p in call.inputs.iter().chain(&call.outputs) {
            if p as usize >= planes {
                bail!("{}: plane {} of an arena of {}", call.label, p, planes);
            }
            // SAFETY: the plane lies inside `arena`, and so do its points offset..offset + n
            let data = unsafe { base.add(p as usize * ld + offset) };
            table.push(PlaneDescriptor { data, len: n });
        }
        let start = Instant::now();
        invoke(k, call.simd_safe && !scalar, table.as_ptr(), n, s)
            .map_err(|e| anyhow!("{} ({}): {:#}", call.label, call.member, e))?;
        s.time += start.elapsed();
    }
    Ok(())
}

/// How a recording is replayed.
#[derive(Clone, Debug)]
pub struct Options {
    /// all points as one tile instead of the recorded tiles
    pub whole: bool,
    /// the scalar kernels only
    pub scalar: bool,
    /// poison the planes written first (`recording::first_written`)
    pub poison: bool,
    /// runs; the times are the best per leaf, and every run must give the same bits
    pub repeat: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            whole: false,
            scalar: false,
            poison: true,
            repeat: 1,
        }
    }
}

/// The result of a replay.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub arena: Vec<f64>,
    /// the planes poisoned
    pub poisoned: usize,
    /// per call; `time` is the best over the runs
    pub stats: Vec<CallStats>,
    /// the planes that differ from `.post`
    pub diffs: Vec<PlaneDiff>,
}

/// Replays `rec` with `kernels[i]` for `calls[i]` and compares the arena with `.post`.
pub fn replay(
    rec: &Recording,
    calls: &[LeafCall],
    kernels: &[&CpuKernel],
    options: &Options,
) -> Result<Outcome> {
    replay_with(rec, calls, options, |arena, ld, offset, n, stats| {
        run_tile(arena, ld, offset, n, calls, kernels, options.scalar, stats)
    })
}

/// Replays `rec` with an executor of its own: `tile(arena, ld, offset, n, stats)` runs
/// `calls` in order on points `offset..offset + n` of the plane-major `arena` of `ld` points
/// per plane (as `run_tile`), adding to `stats[i]` for `calls[i]`. It is called per recorded
/// tile on a copy of the tile (offset 0, `ld` = the largest tile rounded up to 8, RustiCol's
/// stride), or with `options.whole` once on the whole arena. Poisons, repeats and
/// compares as `options` say (`options.scalar` is the executor's business).
pub fn replay_with(
    rec: &Recording,
    calls: &[LeafCall],
    options: &Options,
    mut tile: impl FnMut(&mut [f64], usize, usize, usize, &mut [CallStats]) -> Result<()>,
) -> Result<Outcome> {
    let ld = rec.num_points;
    let tiles: Vec<(usize, usize)> = rec
        .offsets()
        .into_iter()
        .zip(rec.tiles.iter().copied())
        .collect();
    let poisoned = if options.poison {
        recording::first_written(
            calls.iter().map(|c| (&c.inputs[..], &c.outputs[..])),
            rec.num_planes,
        )
    } else {
        Vec::new()
    };

    let mut best: Vec<CallStats> = Vec::new();
    let mut first: Option<Vec<f64>> = None;
    for run in 0..options.repeat.max(1) {
        let mut arena = rec.pre.clone();
        recording::poison(&mut arena, ld, &poisoned);
        let mut stats = vec![CallStats::default(); calls.len()];
        if options.whole {
            tile(&mut arena, ld, 0, ld, &mut stats)?;
        } else {
            // each tile in an arena of its own, planes `stride` points apart, as RustiCol
            // holds one (the tile capacity rounded up to 8)
            let stride = tiles
                .iter()
                .map(|t| t.1)
                .max()
                .unwrap_or(0)
                .next_multiple_of(8);
            let mut local = vec![0.0; rec.num_planes * stride];
            for &(offset, n) in tiles.iter() {
                for p in 0..rec.num_planes {
                    local[p * stride..p * stride + n]
                        .copy_from_slice(&arena[p * ld + offset..p * ld + offset + n]);
                }
                tile(&mut local, stride, 0, n, &mut stats)?;
                for p in 0..rec.num_planes {
                    arena[p * ld + offset..p * ld + offset + n]
                        .copy_from_slice(&local[p * stride..p * stride + n]);
                }
            }
        }
        match first.as_ref() {
            None => {
                best = stats;
                first = Some(arena);
            }
            Some(f) => {
                if !recording::compare(&arena, f, ld).is_empty() {
                    bail!("run {} gave other bits than run 1", run + 1);
                }
                for (b, s) in best.iter_mut().zip(stats) {
                    b.time = b.time.min(s.time);
                }
            }
        }
    }
    let arena = first.expect("one run at least");
    let diffs = recording::compare(&arena, &rec.post, ld);
    Ok(Outcome {
        arena,
        poisoned: poisoned.len(),
        stats: best,
        diffs,
    })
}

/// The call that writes each plane last (`None`: no call writes it), to blame differences on.
pub fn last_writers(calls: &[LeafCall], num_planes: usize) -> Vec<Option<usize>> {
    let mut out = vec![None; num_planes];
    for (i, c) in calls.iter().enumerate() {
        for &p in &c.outputs {
            out[p as usize] = Some(i);
        }
    }
    out
}

/// A saved complex plane application as pyAmpliCol compiles one (`explicit_plane_config`) from
/// the Symbolica instructions `model` of two complex inputs (tests of `replay` and `runner`).
#[cfg(test)]
pub(crate) fn plane_application(model: &str) -> Vec<u8> {
    use symjit::{Compiler, Storage};
    let mut config = Config::new(CompilerType::Native, 0).unwrap();
    config.set_opt_level(2);
    config.set_cse(true);
    config.set_fastmath(true);
    config.set_simd(true);
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_complex(true);
    config.set_fast_complex(false);
    config.set_threads(false);
    config.set_symbolica(true);
    config.set_compact(true);
    config.set_compress(false);
    config.set_direct(false);
    config.set_huge(false);
    config.set_parallel_mul(true);
    config.set_direct_arena(true);
    config.set_direct_arena_operation(0);
    config.set_direct_arena_identity_output(true);
    let mut compiler = Compiler::with_config(config);
    let mut app = compiler.translate(model.to_string(), 2).unwrap();
    app.prepare_simd();
    let mut bytes = Vec::new();
    app.save(&mut bytes).unwrap();
    bytes
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A complex plane kernel as pyAmpliCol compiles one, loaded.
    fn plane_kernel(model: &str) -> CpuKernel {
        CpuKernel::load("test.plane.symjit", &plane_application(model)).unwrap()
    }

    /// w = z0 z1 + 1
    pub(crate) const PRODUCT_PLUS_ONE: &str =
        "([('mul', ('temp', 0), [('param', 0), ('param', 1)], 0),
        ('add', ('out', 0), [('temp', 0), ('const', 0)], 0)], 1, [1])";
    /// v = z0² - z1
    pub(crate) const SQUARE_MINUS: &str = "([('mul', ('temp', 0), [('param', 0), ('param', 0)], 0),
        ('mul', ('temp', 1), [('param', 1), ('const', 0)], 0),
        ('add', ('out', 0), [('temp', 0), ('temp', 1)], 0)], 2, [-1])";
    const PRODUCT: &str = "([('mul', ('out', 0), [('param', 0), ('param', 1)], 0)], 0, [])";
    const SUM: &str = "([('add', ('out', 0), [('param', 0), ('param', 1)], 0)], 0, [])";

    #[test]
    fn runs_tiles_as_rusticol() {
        // leaf 1: w = z0 z1 + 1, leaf 2: v = w² - z0
        let k1 = plane_kernel(PRODUCT_PLUS_ONE);
        let k2 = plane_kernel(SQUARE_MINUS);
        assert_eq!((k1.inputs, k1.outputs), (4, 2));
        // planes: z0 re/im 0, 1; z1 re/im 2, 3; w 4, 5; v 6, 7
        let calls = vec![
            LeafCall::new("a".into(), "stage1:0".into(), vec![0, 1, 2, 3], vec![4, 5]),
            LeafCall::new("b".into(), "stage2:0".into(), vec![4, 5, 0, 1], vec![6, 7]),
        ];
        let kernels = [&k1, &k2];
        let ld = 11;
        let mut arena = vec![f64::NAN; 8 * ld];
        for i in 0..ld {
            for p in 0..4 {
                arena[p * ld + i] = 0.25 * (i as f64 + 1.0) + 0.5 * p as f64 - 0.3;
            }
        }
        let pre = arena.clone();
        let mut stats = vec![CallStats::default(); 2];
        // tiles of 6 and 5 points
        run_tile(&mut arena, ld, 0, 6, &calls, &kernels, false, &mut stats).unwrap();
        run_tile(&mut arena, ld, 6, 5, &calls, &kernels, false, &mut stats).unwrap();
        for i in 0..ld {
            let z = |p: usize| symjit::Complex::new(pre[2 * p * ld + i], pre[(2 * p + 1) * ld + i]);
            let w = z(0) * z(1) + 1.0;
            let v = w * w - z(0);
            for (plane, want) in [(4, w.re), (5, w.im), (6, v.re), (7, v.im)] {
                let got = arena[plane * ld + i];
                assert!(
                    (got - want).abs() <= 1e-14 * want.abs().max(1.0),
                    "plane {} point {}: {} {}",
                    plane,
                    i,
                    got,
                    want
                );
            }
        }
        let lanes = k1.lanes as u64;
        if k1.has_simd() {
            // per tile: whole blocks by SIMD, the rest scalar
            let simd = 6 / lanes + 5 / lanes;
            assert_eq!(stats[0].simd_calls, simd);
            assert_eq!(stats[0].scalar_calls, 11 - simd * lanes);
        }

        // scalar only: the same values (real arithmetic, so the same bits)
        let mut scalar = pre.clone();
        let mut s = vec![CallStats::default(); 2];
        run_tile(&mut scalar, ld, 0, ld, &calls, &kernels, true, &mut s).unwrap();
        assert_eq!(s[0].simd_calls, 0);
        assert!(recording::compare(&scalar, &arena, ld).is_empty());

        // an aliased table runs scalar only
        let alias = LeafCall::new("a".into(), "x".into(), vec![0, 1, 2, 3], vec![0, 1]);
        assert!(!alias.simd_safe);
        let mut s = vec![CallStats::default()];
        let mut copy = pre.clone();
        run_tile(&mut copy, ld, 0, ld, &[alias], &[&k1], false, &mut s).unwrap();
        assert_eq!((s[0].simd_calls, s[0].scalar_calls), (0, 11));

        // a table of the wrong shape and a tile too large are errors
        let short = LeafCall::new("a".into(), "x".into(), vec![0, 1], vec![4, 5]);
        let mut s = vec![CallStats::default()];
        assert!(run_tile(&mut copy, ld, 0, ld, &[short], &[&k1], false, &mut s).is_err());
        assert!(run_tile(&mut copy, ld, 6, 6, &calls[..1], &[&k1], false, &mut s).is_err());
    }

    #[test]
    fn replay_catches_a_missing_write() {
        let k1 = plane_kernel(PRODUCT);
        let k2 = plane_kernel(SUM);
        let calls = vec![
            LeafCall::new("a".into(), "stage1:0".into(), vec![0, 1, 2, 3], vec![4, 5]),
            LeafCall::new("b".into(), "stage2:0".into(), vec![4, 5, 0, 1], vec![6, 7]),
        ];
        let kernels = [&k1, &k2];
        let ld = 9;
        let mut rec = Recording::parse(&synthetic_meta(ld)).unwrap();
        let mut pre = vec![0.0; rec.num_planes * ld];
        for (i, x) in pre[..4 * ld].iter_mut().enumerate() {
            *x = (i as f64 * 0.37).sin();
        }
        let mut post = pre.clone();
        let mut stats = vec![CallStats::default(); 2];
        run_tile(&mut post, ld, 0, 5, &calls, &kernels, false, &mut stats).unwrap();
        run_tile(&mut post, ld, 5, 4, &calls, &kernels, false, &mut stats).unwrap();
        // the snapshot already holds the outputs (a tile run twice)
        rec.pre = post.clone();
        rec.post = post;
        let options = Options {
            repeat: 2,
            ..Options::default()
        };
        let out = replay(&rec, &calls, &kernels, &options).unwrap();
        assert!(out.diffs.is_empty());
        assert_eq!(out.poisoned, 4);
        if k2.has_simd() {
            // tiles of 5 and 4 points
            let lanes = k2.lanes as u64;
            assert_eq!(out.stats[1].simd_calls, 5 / lanes + 4 / lanes);
        }

        // an executor that skips the second leaf: the snapshot already holds its outputs,
        // so only the poison shows it
        let skip = |arena: &mut [f64], ld, offset, n, stats: &mut [CallStats]| {
            run_tile(
                arena,
                ld,
                offset,
                n,
                &calls[..1],
                &kernels[..1],
                false,
                &mut stats[..1],
            )
        };
        let out = replay_with(&rec, &calls, &options, skip).unwrap();
        assert_eq!(
            out.diffs.iter().map(|d| d.plane).collect::<Vec<_>>(),
            vec![6, 7]
        );
        assert_eq!(recording::show(out.diffs[0].got), "poison(6)");
        let plain = Options {
            poison: false,
            ..Options::default()
        };
        let out = replay_with(&rec, &calls, &plain, skip).unwrap();
        assert!(out.diffs.is_empty());

        // a leaf writing other planes: they differ, and are blamed on it
        let moved = vec![
            calls[0].clone(),
            LeafCall::new("b".into(), "stage2:0".into(), vec![4, 5, 0, 1], vec![8, 9]),
        ];
        let out = replay(&rec, &moved, &kernels, &options).unwrap();
        assert_eq!(
            out.diffs.iter().map(|d| d.plane).collect::<Vec<_>>(),
            vec![8, 9]
        );
        assert_eq!(last_writers(&moved, 12)[8], Some(1));
        assert_eq!(last_writers(&moved, 12)[6], None);
    }

    /// An arena of 12 planes (5 current planes); the calls above use planes 0..10.
    fn synthetic_meta(points: usize) -> String {
        format!(
            r#"{{"format": "pacman-arena-v1", "dtype": "f64-le", "layout": "plane-major",
            "name": "t.e0.r0", "process": "t", "engine": 0, "method": "all",
            "num_planes": 12, "num_points": {}, "current_planes": 5, "amplitude_planes": 0,
            "momentum_planes": 1, "model_parameters": 0,
            "regions": [
              {{"name": "current_re", "first": 0, "count": 5}},
              {{"name": "current_im", "first": 5, "count": 5}},
              {{"name": "amplitude_re", "first": 10, "count": 0}},
              {{"name": "amplitude_im", "first": 10, "count": 0}},
              {{"name": "momenta", "first": 10, "count": 1}},
              {{"name": "zero", "first": 11, "count": 1}},
              {{"name": "parameters", "first": 12, "count": 0}}],
            "canonical_to_physical": [0, 1, 2], "unmapped_physical": 4294967295,
            "source_planes": [0, 1], "tiles": [5, 4],
            "schedule": [{{"stage": 0, "label": "s1", "amplitude": false,
              "leaves": [{{"leaf": 0, "application_path": "a"}}]}}]}}"#,
            points
        )
    }
}
