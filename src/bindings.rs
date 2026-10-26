//! pacman bindings: the plane tables pyAmpliCol builds for the symjit kernels of a compiled-mode
//! artifact (`processes/<process>/execution.json`), as `rusticol-core` (pyamplicol 1.0.1) builds
//! them when it loads the artifact (`engine/compiled_direct_prototype.rs`: `load_production`,
//! `load_stage`, `append_component_bindings`, `derive_stage_current_layout`;
//! `evaluator/symjit_compiled_direct.rs`: `bind_with_current_map`).
//!
//! An execution runs its stages in order and the leaves (one plane kernel each) of a stage in
//! order. A kernel is called with one `PlaneDescriptor` per real plane: its inputs, then its
//! outputs. Each complex value is two planes: a current is its real and imaginary planes in
//! the arena, a momentum component its (real) momentum plane and the shared zero plane, a model
//! parameter two planes of the kernel's own filled with its real and imaginary parts. Outputs
//! overwrite current (or, in the amplitude stage, amplitude) planes. A plane holds one tile of
//! points, so a descriptor points at `buffer + index × stride`, the stride being the tile
//! capacity, chosen at load time.
//!
//! `execution.json` names currents by their canonical components; at load time the components
//! are mapped onto fewer arena planes by their lifetimes (stage liveness): a plane is reused
//! once every stage reading its current has run. That map is not saved; it is rebuilt here.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::fmt::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::info::thousands;
use crate::pacbin::Pacbin;
use crate::pacman::{glob, Kernel};

const UNMAPPED: u32 = u32::MAX;

/// One plane of a kernel's descriptor table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Plane {
    /// a current component: its arena plane and its index in `execution.json`
    Current {
        physical: u32,
        canonical: u32,
        imaginary: bool,
    },
    Amplitude {
        component: u32,
        imaginary: bool,
    },
    /// a real momentum plane: `form × 4 + Lorentz component`
    Momentum(u32),
    /// the shared plane of zeros
    Zero,
    /// a plane of the kernel's own filled with one value (`Leaf::broadcasts`)
    Broadcast(u32),
}

impl Plane {
    pub fn buffer(&self) -> &'static str {
        match self {
            Plane::Current {
                imaginary: false, ..
            } => "current_re",
            Plane::Current {
                imaginary: true, ..
            } => "current_im",
            Plane::Amplitude {
                imaginary: false, ..
            } => "amplitude_re",
            Plane::Amplitude {
                imaginary: true, ..
            } => "amplitude_im",
            Plane::Momentum(_) => "momenta",
            Plane::Zero => "zero",
            Plane::Broadcast(_) => "broadcast",
        }
    }

    /// The plane's index in its buffer (the zero plane: 0).
    pub fn index(&self) -> u32 {
        match *self {
            Plane::Current { physical, .. } => physical,
            Plane::Amplitude { component, .. } => component,
            Plane::Momentum(i) | Plane::Broadcast(i) => i,
            Plane::Zero => 0,
        }
    }
}

/// What a broadcast plane holds: a model parameter's real or imaginary part.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Broadcast {
    pub parameter: u32,
    pub imaginary: bool,
}

/// One kernel call: the plane application and its descriptor table.
#[derive(Clone, Debug)]
pub struct Leaf {
    /// the application, relative to the process directory
    pub application: String,
    pub inputs: Vec<Plane>,
    pub outputs: Vec<Plane>,
    pub broadcasts: Vec<Broadcast>,
    /// canonical current components read (structural zeros excluded) and written
    used: BTreeSet<usize>,
    defined: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Stage {
    pub label: String,
    pub amplitude: bool,
    pub leaves: Vec<Leaf>,
}

/// One execution of a process (the stages of one `compiled.stage_evaluators`).
#[derive(Clone, Debug)]
pub struct Execution {
    /// e.g. `top`, `helicity_sum`, `helicity_sum/color_selector[0]`
    pub name: String,
    pub value_components: usize,
    pub source_components: usize,
    pub structural_zeros: usize,
    pub live_components: usize,
    /// the arena's current planes after the liveness map
    pub physical_planes: u32,
    pub momentum_planes: usize,
    pub model_parameters: usize,
    pub amplitudes: usize,
    pub stages: Vec<Stage>,
    /// canonical current component → arena plane (`u32::MAX`: none), the map derived here
    pub canonical_to_physical: Vec<u32>,
}

impl Execution {
    pub fn leaves(&self) -> impl Iterator<Item = (&Stage, usize, &Leaf)> {
        self.stages
            .iter()
            .flat_map(|s| s.leaves.iter().enumerate().map(move |(j, l)| (s, j, l)))
    }

    pub fn arena(&self) -> Arena {
        Arena {
            currents: self.physical_planes,
            amplitudes: self.amplitudes as u32,
            momenta: self.momentum_planes as u32,
            parameters: self.model_parameters as u32,
        }
    }
}

/// The planes of an execution numbered as one arena (the numbering of the `.mapping` files):
/// the current planes' real parts, their imaginary parts, the amplitudes' real parts, their
/// imaginary parts, the momenta, the zero plane, then two planes per model parameter (real,
/// imaginary) in place of pyAmpliCol's per-kernel broadcast planes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Arena {
    pub currents: u32,
    pub amplitudes: u32,
    pub momenta: u32,
    pub parameters: u32,
}

impl Arena {
    /// (region, first plane, planes) in arena order.
    pub fn regions(&self) -> [(&'static str, u32, u32); 7] {
        let mut out = [("", 0, 0); 7];
        let mut first = 0;
        for (i, (name, n)) in [
            ("current_re", self.currents),
            ("current_im", self.currents),
            ("amplitude_re", self.amplitudes),
            ("amplitude_im", self.amplitudes),
            ("momenta", self.momenta),
            ("zero", 1),
            ("parameters", 2 * self.parameters),
        ]
        .into_iter()
        .enumerate()
        {
            out[i] = (name, first, n);
            first += n;
        }
        out
    }

    pub fn len(&self) -> u32 {
        let (_, first, n) = self.regions()[6];
        first + n
    }

    /// The arena plane of `plane`, a descriptor of `leaf`.
    pub fn index(&self, plane: Plane, leaf: &Leaf) -> u32 {
        let r = self.regions();
        match plane {
            Plane::Current {
                physical,
                imaginary,
                ..
            } => r[imaginary as usize].1 + physical,
            Plane::Amplitude {
                component,
                imaginary,
            } => r[2 + imaginary as usize].1 + component,
            Plane::Momentum(m) => r[4].1 + m,
            Plane::Zero => r[5].1,
            Plane::Broadcast(b) => {
                let b = leaf.broadcasts[b as usize];
                r[6].1 + 2 * b.parameter + b.imaginary as u32
            }
        }
    }

    /// The arena plane of `plane` with current planes taken from `c2p` (canonical component →
    /// physical plane, e.g. a recording's `canonical_to_physical`) instead of the map derived
    /// here; `None` for a current that `c2p` does not map onto a plane of this arena.
    pub fn index_with(&self, plane: Plane, leaf: &Leaf, c2p: &[u32]) -> Option<u32> {
        match plane {
            Plane::Current {
                canonical,
                imaginary,
                ..
            } => {
                let physical = *c2p.get(canonical as usize)?;
                (physical < self.currents).then(|| self.regions()[imaginary as usize].1 + physical)
            }
            _ => Some(self.index(plane, leaf)),
        }
    }

    /// A leaf's `.mapping` file: per descriptor (the kernel's `states` index, inputs then
    /// outputs) a line `<index> <arena plane>`.
    pub fn mapping(&self, leaf: &Leaf) -> String {
        let mut out = String::new();
        for (i, p) in leaf.inputs.iter().chain(&leaf.outputs).enumerate() {
            let _ = writeln!(out, "{} {}", i, self.index(*p, leaf));
        }
        out
    }
}

/// The `.mapping` files of the kernels of `processes` (by member path in `evaluators.pacbin`),
/// and per process its `arena.txt`: the regions of each execution's arena.
pub fn mappings(processes: &[Process]) -> Result<(HashMap<String, String>, Vec<(String, String)>)> {
    let mut files: HashMap<String, String> = HashMap::new();
    let mut layouts = Vec::new();
    for p in processes.iter().filter(|p| !p.executions.is_empty()) {
        let mut layout = format!(
            "# the arena planes of the .mapping files of {} (one arena per execution;\n\
             # a plane is at arena + index × stride)\n\
             # execution region first planes\n",
            p.file.display()
        );
        for e in p.executions.iter() {
            let arena = e.arena();
            for (region, first, n) in arena.regions() {
                writeln!(layout, "{} {} {} {}", e.name, region, first, n)?;
            }
            for (_, _, l) in e.leaves() {
                let member = format!("{}/{}", p.prefix, l.application);
                let text = arena.mapping(l);
                if files.get(&member).is_some_and(|t| *t != text) {
                    bail!(
                        "`{}` is called with different planes in two executions",
                        member
                    );
                }
                files.insert(member, text);
            }
        }
        layouts.push((format!("{}/arena.txt", p.prefix), layout));
    }
    Ok((files, layouts))
}

/// The executions of one `execution.json`.
pub struct Process {
    /// the `execution.json`
    pub file: PathBuf,
    /// the process directory relative to the artifact (`processes/<process>`)
    pub prefix: String,
    pub executions: Vec<Execution>,
}

/// The executions of `json` (an `execution.json`) that have compiled plane-arena stages.
pub fn executions(json: &str) -> Result<Vec<Execution>> {
    let doc: Value = serde_json::from_str(json)?;
    let mut found = Vec::new();
    find(&doc, String::new(), &mut found);
    found
        .into_iter()
        .map(|(name, e)| execution(&name, e).with_context(|| format!("execution `{}`", name)))
        .collect()
}

/// Every object holding `compiled.stage_evaluators` with plane arenas and a `runtime_schema`.
fn find<'a>(v: &'a Value, path: String, out: &mut Vec<(String, &'a Value)>) {
    match v {
        Value::Object(map) => {
            let stages = &v["compiled"]["stage_evaluators"];
            if map.contains_key("runtime_schema")
                && !stages["amplitude_stage"]["compiled_plane_arena"].is_null()
            {
                let name = if path.is_empty() {
                    "top".to_string()
                } else {
                    path.clone()
                };
                out.push((name, v));
            }
            for (k, x) in map {
                if k == "compiled" || k == "runtime_schema" {
                    continue;
                }
                let k = k.strip_suffix("_executions").unwrap_or(k);
                let k = if k == "execution" {
                    ""
                } else {
                    k.strip_suffix("_execution").unwrap_or(k)
                };
                let p = match (path.is_empty(), k) {
                    // `.../color_selector_executions[0]/execution`
                    (_, "") => path.clone(),
                    (true, _) => k.to_string(),
                    (false, _) => format!("{}/{}", path, k),
                };
                find(x, p, out);
            }
        }
        Value::Array(items) => {
            for (i, x) in items.iter().enumerate() {
                find(x, format!("{}[{}]", path, i), out);
            }
        }
        _ => {}
    }
}

fn usize_of(v: &Value, what: &str) -> Result<usize> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| anyhow!("no `{}`", what))
}

fn str_of<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str().ok_or_else(|| anyhow!("no `{}`", what))
}

fn array<'a>(v: &'a Value, what: &str) -> Result<&'a Vec<Value>> {
    v.as_array().ok_or_else(|| anyhow!("no `{}`", what))
}

fn execution(name: &str, e: &Value) -> Result<Execution> {
    let layout = &e["runtime_schema"]["parameter_layout"];
    let n_value = usize_of(&layout["value_component_count"], "value_component_count")?;
    let n_momentum = usize_of(
        &layout["momentum_parameter_count"],
        "momentum_parameter_count",
    )?;
    let n_parameters = usize_of(&layout["model_parameter_count"], "model_parameter_count")?;
    if n_momentum % 4 != 0 {
        bail!("{} momentum components: not a multiple of four", n_momentum);
    }

    // canonical_source_layout: the sources own sorted, disjoint component ranges
    let mut sources = Vec::new();
    for s in array(&e["runtime_schema"]["source_fill"]["sources"], "sources")? {
        let slot = &s["value_slot"];
        let start = usize_of(&slot["component_start"], "component_start")?;
        let stop = usize_of(&slot["component_stop"], "component_stop")?;
        if stop <= start || stop > n_value || sources.last().is_some_and(|&l| l >= start) {
            bail!("the sources' components are not sorted, disjoint ranges of the currents");
        }
        sources.extend(start..stop);
    }

    let se = &e["compiled"]["stage_evaluators"];
    let stages = array(&se["stages"], "stages")?;
    let amplitude = &se["amplitude_stage"];

    // canonical_structural_zero_components: neither a source nor written by a stage
    let source_set: BTreeSet<usize> = sources.iter().copied().collect();
    let mut produced = BTreeSet::new();
    for st in stages {
        for slot in array(&st["output_slots"], "output_slots")? {
            let start = usize_of(&slot["component_start"], "component_start")?;
            let stop = usize_of(&slot["component_stop"], "component_stop")?;
            if stop > n_value || (start..stop).any(|c| source_set.contains(&c)) {
                bail!(
                    "stage {} writes outside the stage components",
                    st["evaluator_label"]
                );
            }
            produced.extend(start..stop);
        }
    }
    let zeros: BTreeSet<usize> = (0..n_value)
        .filter(|c| !source_set.contains(c) && !produced.contains(c))
        .collect();

    let n_amplitude = usize_of(&amplitude["output_length"], "output_length")?;
    let ctx = Context_ {
        n_value,
        n_momentum,
        n_parameters,
        n_amplitude,
        zeros: &zeros,
    };
    let mut out = Vec::new();
    for st in stages {
        out.push(stage(st, false, &ctx)?);
    }
    out.push(stage(amplitude, true, &ctx)?);

    let (map, physical, live) = current_layout(&out, &sources, n_value)?;
    for s in out.iter_mut() {
        for l in s.leaves.iter_mut() {
            for p in l.inputs.iter_mut().chain(l.outputs.iter_mut()) {
                if let Plane::Current {
                    physical,
                    canonical,
                    ..
                } = p
                {
                    *physical = map[*canonical as usize];
                }
            }
        }
    }
    Ok(Execution {
        name: name.to_string(),
        value_components: n_value,
        source_components: sources.len(),
        structural_zeros: zeros.len(),
        live_components: live,
        physical_planes: physical,
        momentum_planes: n_momentum,
        model_parameters: n_parameters,
        amplitudes: n_amplitude,
        stages: out,
        canonical_to_physical: map,
    })
}

struct Context_<'a> {
    n_value: usize,
    n_momentum: usize,
    n_parameters: usize,
    n_amplitude: usize,
    zeros: &'a BTreeSet<usize>,
}

/// load_stage: the leaves' tables, current planes still canonical.
fn stage(st: &Value, is_amplitude: bool, ctx: &Context_) -> Result<Stage> {
    let label = str_of(&st["evaluator_label"], "evaluator_label")?.to_string();
    let arena = &st["compiled_plane_arena"];
    if arena.is_null() {
        bail!("stage `{}` has no plane bindings", label);
    }
    let n_inputs = usize_of(&st["parameter_count"], "parameter_count")?;
    let mut inputs: Vec<Option<&Value>> = vec![None; n_inputs];
    for b in array(&arena["input_bindings"], "input_bindings")? {
        let i = usize_of(&b["parameter_index"], "parameter_index")?;
        if i >= n_inputs || inputs[i].replace(b).is_some() {
            bail!("stage `{}` has invalid input bindings", label);
        }
    }
    if inputs.iter().any(Option::is_none) {
        bail!("stage `{}` has incomplete input bindings", label);
    }

    let n_outputs = usize_of(&st["output_length"], "output_length")?;
    let limit = if is_amplitude {
        ctx.n_amplitude
    } else {
        ctx.n_value
    };
    let expected = if is_amplitude { "amplitude" } else { "current" };
    let mut outputs: Vec<Option<u32>> = vec![None; n_outputs];
    let mut seen = BTreeSet::new();
    for b in array(&arena["output_bindings"], "output_bindings")? {
        let i = usize_of(&b["output_index"], "output_index")?;
        let c = usize_of(&b["component"], "component")?;
        if i >= n_outputs
            || c >= limit
            || b["arena"].as_str() != Some(expected)
            || !seen.insert(c)
            || outputs[i].replace(c as u32).is_some()
        {
            bail!("stage `{}` has invalid output bindings", label);
        }
    }
    if outputs.iter().any(Option::is_none) {
        bail!("stage `{}` does not bind every output", label);
    }

    let mut leaves = Vec::new();
    let mut cursor = 0;
    for leaf in array(&arena["leaves"], "leaves")? {
        let start = usize_of(&leaf["output_start"], "output_start")?;
        let stop = usize_of(&leaf["output_stop"], "output_stop")?;
        if start != cursor || stop < start || stop > n_outputs {
            bail!("stage `{}`: the leaves' outputs are not consecutive", label);
        }
        cursor = stop;
        let mut l = Leaf {
            application: str_of(&leaf["application_path"], "application_path")?.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            broadcasts: Vec::new(),
            used: BTreeSet::new(),
            defined: Vec::new(),
        };
        // append_component_bindings
        for i in array(&leaf["input_indices"], "input_indices")? {
            let i = usize_of(i, "input index")?;
            let b = inputs
                .get(i)
                .copied()
                .flatten()
                .ok_or_else(|| anyhow!("stage `{}` reads an absent input {}", label, i))?;
            let g = usize_of(&b["global_component"], "global_component")?;
            match str_of(&b["kind"], "kind")? {
                "value" => {
                    if g >= ctx.n_value {
                        bail!("stage `{}` reads current {} of {}", label, g, ctx.n_value);
                    }
                    if ctx.zeros.contains(&g) {
                        l.inputs.extend([Plane::Zero, Plane::Zero]);
                    } else {
                        l.used.insert(g);
                        for imaginary in [false, true] {
                            l.inputs.push(Plane::Current {
                                physical: UNMAPPED,
                                canonical: g as u32,
                                imaginary,
                            });
                        }
                    }
                }
                "momentum" => {
                    let local = g
                        .checked_sub(ctx.n_value)
                        .filter(|&m| m < ctx.n_momentum)
                        .ok_or_else(|| anyhow!("stage `{}` reads momentum {}", label, g))?;
                    l.inputs
                        .extend([Plane::Momentum(local as u32), Plane::Zero]);
                }
                "model_parameter" => {
                    let p = usize_of(&b["source_id"], "source_id")?;
                    if p >= ctx.n_parameters {
                        bail!("stage `{}` reads model parameter {}", label, p);
                    }
                    for imaginary in [false, true] {
                        l.inputs.push(Plane::Broadcast(l.broadcasts.len() as u32));
                        l.broadcasts.push(Broadcast {
                            parameter: p as u32,
                            imaginary,
                        });
                    }
                }
                kind => bail!("stage `{}` reads an input of kind `{}`", label, kind),
            }
        }
        if l.inputs.len() != 2 * usize_of(&leaf["input_len"], "input_len")? {
            bail!("stage `{}`: a leaf's input map has the wrong length", label);
        }
        // symjit_output_bindings
        for &c in &outputs[start..stop] {
            let c = c.expect("checked above");
            for imaginary in [false, true] {
                l.outputs.push(if is_amplitude {
                    Plane::Amplitude {
                        component: c,
                        imaginary,
                    }
                } else {
                    Plane::Current {
                        physical: UNMAPPED,
                        canonical: c,
                        imaginary,
                    }
                });
            }
            if !is_amplitude {
                l.defined.push(c as usize);
            }
        }
        leaves.push(l);
    }
    if cursor != n_outputs {
        bail!("stage `{}`: the leaves do not cover the outputs", label);
    }
    Ok(Stage {
        label,
        amplitude: is_amplitude,
        leaves,
    })
}

/// derive_stage_current_layout: the sources are defined at event 0, the outputs of stage k
/// (from 1) at event k, after its inputs are read; the amplitude stage reads at the last
/// event. Planes are assigned by first event (then component), each taking the lowest plane
/// whose current was last read strictly before. Returns the map, the planes and the live
/// components.
fn current_layout(
    stages: &[Stage],
    sources: &[usize],
    n_value: usize,
) -> Result<(Vec<u32>, u32, usize)> {
    let mut first: Vec<Option<u64>> = vec![None; n_value];
    let mut last: Vec<Option<u64>> = vec![None; n_value];
    let define = |first: &mut [Option<u64>], last: &mut [Option<u64>], c: usize, e: u64| {
        if first[c].is_some() {
            bail!("current component {} is defined twice", c);
        }
        first[c] = Some(e);
        last[c] = Some(e);
        Ok(())
    };
    for &c in sources {
        define(&mut first, &mut last, c, 0)?;
    }
    for (k, st) in stages.iter().enumerate() {
        let event = k as u64 + 1;
        for l in &st.leaves {
            for &c in &l.used {
                if first[c].is_none_or(|f| f > event) {
                    bail!("current component {} is read before it is written", c);
                }
                last[c] = last[c].max(Some(event));
            }
        }
        for l in &st.leaves {
            for &c in &l.defined {
                define(&mut first, &mut last, c, event)?;
            }
        }
    }
    let mut lifetimes: Vec<(u64, usize, u64)> = (0..n_value)
        .filter_map(|c| Some((first[c]?, c, last[c]?)))
        .collect();
    if lifetimes.is_empty() {
        bail!("no live current components");
    }
    lifetimes.sort_unstable();

    let mut active = BinaryHeap::<Reverse<(u64, u32)>>::new();
    let mut free = BinaryHeap::<Reverse<u32>>::new();
    let mut next = 0u32;
    let mut map = vec![UNMAPPED; n_value];
    for &(f, c, l) in &lifetimes {
        while let Some(&Reverse((end, p))) = active.peek() {
            if end >= f {
                break;
            }
            active.pop();
            free.push(Reverse(p));
        }
        let p = match free.pop() {
            Some(Reverse(p)) => p,
            None => {
                next += 1;
                next - 1
            }
        };
        map[c] = p;
        active.push(Reverse((l, p)));
    }
    Ok((map, next, lifetimes.len()))
}

/// The processes of `input`: an artifact directory (its `processes/*/execution.json`) or one
/// `execution.json`; and the artifact's `evaluators.pacbin`, if found.
pub fn load(input: &Path) -> Result<(Vec<Process>, Option<PathBuf>)> {
    let (files, artifact) = if input.is_dir() {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(input.join("processes")) {
            for e in entries.flatten() {
                let f = e.path().join("execution.json");
                if f.is_file() {
                    files.push(f);
                }
            }
        }
        if files.is_empty() {
            bail!(
                "`{}` holds no process (processes/*/execution.json): not a pyAmpliCol artifact",
                input.display()
            );
        }
        files.sort();
        (files, Some(input.to_path_buf()))
    } else {
        // <artifact>/processes/<process>/execution.json
        let artifact = input
            .ancestors()
            .nth(3)
            .filter(|a| a.join("evaluators.pacbin").is_file())
            .map(Path::to_path_buf);
        (vec![input.to_path_buf()], artifact)
    };
    let mut out = Vec::new();
    for f in files {
        let json = std::fs::read_to_string(&f)
            .with_context(|| format!("cannot open `{}`", f.display()))?;
        let executions =
            executions(&json).with_context(|| format!("cannot read `{}`", f.display()))?;
        let prefix = match f.parent().and_then(Path::file_name) {
            Some(p) => format!("processes/{}", p.to_string_lossy()),
            None => String::new(),
        };
        out.push(Process {
            file: f,
            prefix,
            executions,
        });
    }
    let pacbin = artifact
        .map(|a| a.join("evaluators.pacbin"))
        .filter(|p| p.is_file());
    Ok((out, pacbin))
}

/// The kernels' shapes in `evaluators.pacbin`: (input planes, output planes) by member path.
pub fn kernel_shapes(file: &Path) -> Result<HashMap<String, Result<(usize, usize), String>>> {
    let bytes = std::fs::read(file).with_context(|| format!("cannot open `{}`", file.display()))?;
    let container =
        Pacbin::read(bytes, true).with_context(|| format!("cannot read `{}`", file.display()))?;
    let mut out = HashMap::new();
    for m in container.members.iter() {
        if !m.path.ends_with(".plane.symjit") {
            continue;
        }
        let k = Kernel::read(&m.path, container.payload(m));
        let shape = match k.error {
            Some(e) => Err(e),
            None => Ok((k.inputs(), k.obs)),
        };
        out.insert(m.path.clone(), shape);
    }
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// only the executions whose names match (glob)
    pub execution: Option<String>,
    /// only the leaves whose label (`stage5:0`, `amplitude:0`) or application matches (glob)
    pub leaf: Option<String>,
    /// also print each selected leaf's descriptor table
    pub table: bool,
}

/// `stage<k>:<leaf>` or `amplitude:<leaf>`.
pub fn leaf_label(stages: &[Stage], stage: &Stage, leaf: usize) -> String {
    if stage.amplitude {
        format!("amplitude:{}", leaf)
    } else {
        let k = stages
            .iter()
            .position(|s| std::ptr::eq(s, stage))
            .unwrap_or(0);
        format!("stage{}:{}", k + 1, leaf)
    }
}

fn matches(pattern: &Option<String>, texts: &[&str]) -> bool {
    match pattern {
        None => true,
        Some(p) => texts.iter().any(|t| glob(p.as_bytes(), t.as_bytes())),
    }
}

/// The report on `process`; with `shapes` (from `kernel_shapes`), each kernel's input and
/// output planes are checked against its table.
pub fn describe(
    process: &Process,
    shapes: Option<&HashMap<String, Result<(usize, usize), String>>>,
    options: &Options,
) -> Result<String> {
    let mut out = String::new();
    let o = &mut out;
    writeln!(o, "{}", process.file.display())?;
    if process.executions.is_empty() {
        writeln!(
            o,
            "  no compiled plane-arena stages (not a compiled-mode process)"
        )?;
        return Ok(out);
    }
    let mut shown = 0;
    for e in process
        .executions
        .iter()
        .filter(|e| matches(&options.execution, &[&e.name]))
    {
        shown += 1;
        let leaves: Vec<_> = e.leaves().collect();
        let (mut currents, mut remapped) = (0usize, 0usize);
        for (_, _, l) in &leaves {
            for p in l.inputs.iter().chain(&l.outputs) {
                if let Plane::Current {
                    physical,
                    canonical,
                    ..
                } = *p
                {
                    currents += 1;
                    remapped += (physical != canonical) as usize;
                }
            }
        }
        writeln!(o)?;
        writeln!(
            o,
            "  execution {}: {} kernels in {} stages (the last computes the amplitudes)",
            e.name,
            leaves.len(),
            e.stages.len()
        )?;
        writeln!(
            o,
            "    currents     {} components ({} of sources, {} structural zeros, {} live) \
             in {} arena planes",
            thousands(e.value_components),
            thousands(e.source_components),
            thousands(e.structural_zeros),
            thousands(e.live_components),
            thousands(e.physical_planes as usize)
        )?;
        writeln!(
            o,
            "    remapped     {} of {} current descriptors point at another plane than \
             execution.json's component",
            thousands(remapped),
            thousands(currents)
        )?;
        writeln!(
            o,
            "    momenta      {} planes ({} forms × 4), model parameters {}, amplitudes {}",
            e.momentum_planes,
            e.momentum_planes / 4,
            e.model_parameters,
            e.amplitudes
        )?;
        writeln!(o)?;
        writeln!(
            o,
            "    {:<12} {:>6} {:>6} {:>8} {:>8} {:>6} {:>5}  {}",
            "leaf", "in", "out", "current", "momenta", "zero", "bcast", "kernel"
        )?;
        for (stage, j, l) in &leaves {
            let label = leaf_label(&e.stages, stage, *j);
            if !matches(&options.leaf, &[&label, &l.application]) {
                continue;
            }
            let count = |f: fn(&Plane) -> bool| l.inputs.iter().filter(|p| f(p)).count();
            let check = match shapes {
                None => "",
                Some(s) => match s.get(&format!("{}/{}", process.prefix, l.application)) {
                    None => "  [missing from evaluators.pacbin]",
                    Some(Err(_)) => "  [unreadable]",
                    Some(Ok((i, n))) if (*i, *n) == (l.inputs.len(), l.outputs.len()) => "",
                    Some(Ok(_)) => "  [the kernel's shape differs]",
                },
            };
            let name = l.application.rsplit('/').next().unwrap_or(&l.application);
            writeln!(
                o,
                "    {:<12} {:>6} {:>6} {:>8} {:>8} {:>6} {:>5}  {}{}",
                label,
                l.inputs.len(),
                l.outputs.len(),
                count(|p| matches!(p, Plane::Current { .. })),
                count(|p| matches!(p, Plane::Momentum(_))),
                count(|p| matches!(p, Plane::Zero)),
                count(|p| matches!(p, Plane::Broadcast(_))),
                name,
                check
            )?;
        }
        if options.table {
            for (stage, j, l) in &leaves {
                let label = leaf_label(&e.stages, stage, *j);
                if !matches(&options.leaf, &[&label, &l.application]) {
                    continue;
                }
                table(o, &label, l)?;
            }
        }
    }
    if shown == 0 {
        writeln!(o, "  no execution matches")?;
    }
    Ok(out)
}

fn table(o: &mut String, label: &str, l: &Leaf) -> Result<()> {
    writeln!(o)?;
    writeln!(o, "    {}  {}", label, l.application)?;
    writeln!(
        o,
        "    {:>6}  {:<4} {:<13} {:>6}  {}",
        "slot", "dir", "buffer", "index", "canonical"
    )?;
    for (i, p) in l.inputs.iter().chain(&l.outputs).enumerate() {
        let dir = if i < l.inputs.len() { "in" } else { "out" };
        let note = match *p {
            Plane::Current { canonical, .. } => canonical.to_string(),
            Plane::Momentum(m) => format!("form {} component {}", m / 4, m % 4),
            Plane::Broadcast(b) => {
                let b = l.broadcasts[b as usize];
                format!(
                    "parameter {} {}",
                    b.parameter,
                    if b.imaginary { "im" } else { "re" }
                )
            }
            _ => String::new(),
        };
        let index = if *p == Plane::Zero {
            "-".to_string()
        } else {
            p.index().to_string()
        };
        writeln!(
            o,
            "    {:>6}  {:<4} {:<13} {:>6}  {}",
            i,
            dir,
            p.buffer(),
            index,
            note
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two stages: stage 1 writes 4 from the sources 0 and 1; stage 2 writes 5 from 4, a
    /// momentum and a model parameter (so 0 and 1 are dead and 5 reuses plane 0); 2 and 3
    /// are never written (structural zeros). The amplitude reads 4 and 5.
    fn sample() -> String {
        let input = |i: usize, kind: &str, g: usize, src: usize| {
            format!(
                r#"{{"parameter_index":{i},"kind":"{kind}","source_id":{src},"component":0,"global_component":{g}}}"#
            )
        };
        let stage = |label: &str, inputs: Vec<String>, arena: &str, outs: &[usize], slots: &str| {
            let n = inputs.len();
            let outputs: Vec<String> = outs
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    format!(r#"{{"output_index":{i},"arena":"{arena}","component":{c}}}"#)
                })
                .collect();
            let indices: Vec<String> = (0..n).map(|i| i.to_string()).collect();
            format!(
                r#"{{"evaluator_label":"{label}","parameter_count":{n},"output_length":{m},
                "output_slots":[{slots}],
                "compiled_plane_arena":{{"input_bindings":[{ins}],"output_bindings":[{outs}],
                "leaves":[{{"application_path":"e/{label}.plane.symjit","input_len":{n},
                "input_indices":[{idx}],"output_start":0,"output_stop":{m}}}]}}}}"#,
                m = outs.len(),
                ins = inputs.join(","),
                outs = outputs.join(","),
                idx = indices.join(",")
            )
        };
        let s1 = stage(
            "s1",
            vec![input(0, "value", 0, 0), input(1, "value", 1, 0)],
            "current",
            &[4],
            r#"{"component_start":4,"component_stop":5}"#,
        );
        let s2 = stage(
            "s2",
            vec![
                input(0, "value", 4, 0),
                input(1, "momentum", 7, 0),
                input(2, "model_parameter", 99, 1),
                input(3, "value", 2, 0),
            ],
            "current",
            &[5],
            r#"{"component_start":5,"component_stop":6}"#,
        );
        let amp = stage(
            "amp",
            vec![input(0, "value", 4, 0), input(1, "value", 5, 0)],
            "amplitude",
            &[0],
            "",
        );
        format!(
            r#"{{"helicity_sum_execution":{{
            "runtime_schema":{{"parameter_layout":{{"value_component_count":6,
              "momentum_parameter_count":4,"model_parameter_count":2}},
              "source_fill":{{"sources":[{{"value_slot":{{"component_start":0,"component_stop":2}}}}]}}}},
            "compiled":{{"stage_evaluators":{{"stages":[{s1},{s2}],"amplitude_stage":{amp}}}}}}}}}"#
        )
    }

    #[test]
    fn tables_follow_the_load_time_layout() {
        let e = executions(&sample()).unwrap();
        assert_eq!(e.len(), 1);
        let e = &e[0];
        assert_eq!(e.name, "helicity_sum");
        assert_eq!(
            (e.value_components, e.source_components, e.structural_zeros),
            (6, 2, 2)
        );
        // 0 and 1 die after stage 1, 4 takes plane 2 and 5 reuses plane 0
        assert_eq!((e.live_components, e.physical_planes), (4, 3));
        let cur = |physical, canonical, imaginary| Plane::Current {
            physical,
            canonical,
            imaginary,
        };
        let s2 = &e.stages[1].leaves[0];
        assert_eq!(
            s2.inputs,
            vec![
                cur(2, 4, false),
                cur(2, 4, true),
                Plane::Momentum(1),
                Plane::Zero,
                Plane::Broadcast(0),
                Plane::Broadcast(1),
                Plane::Zero,
                Plane::Zero,
            ]
        );
        assert_eq!(s2.outputs, vec![cur(0, 5, false), cur(0, 5, true)]);
        assert_eq!(
            s2.broadcasts,
            vec![
                Broadcast {
                    parameter: 1,
                    imaginary: false
                },
                Broadcast {
                    parameter: 1,
                    imaginary: true
                }
            ]
        );
        // one arena: current_re 0..3, current_im 3..6, amplitude 6, 7, momenta 8..12,
        // zero 12, parameters 13..17
        let arena = e.arena();
        assert_eq!(arena.len(), 17);
        assert_eq!(
            arena.mapping(s2),
            "0 2\n1 5\n2 9\n3 12\n4 15\n5 16\n6 12\n7 12\n8 0\n9 3\n"
        );
        let amp = &e.stages[2].leaves[0];
        assert_eq!(arena.mapping(amp), "0 2\n1 5\n2 0\n3 3\n4 6\n5 7\n");
        assert_eq!(amp.inputs[2], cur(0, 5, false));
        assert_eq!(
            amp.outputs,
            vec![
                Plane::Amplitude {
                    component: 0,
                    imaginary: false
                },
                Plane::Amplitude {
                    component: 0,
                    imaginary: true
                }
            ]
        );
    }

    #[test]
    fn invalid_bindings_are_rejected() {
        // stage 1 reading current 5, which stage 2 writes
        let bad = sample().replacen(r#""global_component":0}"#, r#""global_component":5}"#, 1);
        let err = executions(&bad).unwrap_err();
        assert!(
            format!("{:#}", err).contains("read before it is written"),
            "{:#}",
            err
        );
        let bad = sample().replacen(r#""kind":"momentum""#, r#""kind":"other""#, 1);
        assert!(executions(&bad).is_err());
    }
}
