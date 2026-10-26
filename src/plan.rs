//! pacman plan: pyAmpliCol's recurrence plans ("direct plan v2", the member
//! `schedule/recurrence-direct-schedule-v2.bin` of `recurrence/schedules/<digest>/
//! recurrence-runtime.pacbin`), decoded as `rusticol-core/src/recurrence/direct_codec.rs`
//! encodes them (pyamplicol 1.0.1), up to the momentum forms, and joined with the executor
//! templates of the artifact's `model/eager-kernel-pack.json` (`recurrence_direct_template`).
//!
//! A plan is a list of row groups run in order for each tile of points; each group calls one
//! executor (a Rusticol intrinsic or a prepared kernel, e.g. a symjit plane application) once
//! per row. Sources fill the external currents (stage 0); contributions add vertices of two
//! parent currents to a destination current, finalizations apply propagators in place
//! (stages 1..), closures contract currents into amplitudes (the last stage).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::info::thousands;
use crate::pacbin::Pacbin;
use crate::pacman::Kernel;
use crate::sha256::hex;

pub const MEMBER: &str = "schedule/recurrence-direct-schedule-v2.bin";
const MAGIC: &[u8; 8] = b"PACRDAP2";
const VERSION: u32 = 2;
pub const NONE: u32 = u32::MAX;
const FLAG_INITIALIZE: u32 = 1;
const FLAG_CERTIFIED_REUSE: u32 = 2;

/// The tables of the plan, in the order of the file, with the size of one record.
const TABLES: [(&str, usize); 27] = [
    ("currents", 44),
    ("sources", 28),
    ("contributions", 32),
    ("finalizations", 24),
    ("closures", 40),
    ("row groups", 28),
    ("momentum forms", 16),
    ("momentum terms", 8),
    ("selector domains", 16),
    ("selector words", 8),
    ("replay targets", 44),
    ("source permutations", 4),
    ("replay momentum signs", 4),
    ("replay helicity map", 4),
    ("amplitude destinations", 32),
    ("resolved helicities", 44),
    ("source-state assignments", 8),
    ("source dispatch variants", 64),
    ("source embeddings", 12),
    ("source projections", 8),
    ("resolved source selections", 8),
    ("public helicities", 4),
    ("exact factors", 64),
    ("closure proof contributions", 0),
    ("closure proof groups", 0),
    ("reflection certificates", 0),
    ("three-line traversal certificates", 0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    Source,
    Contribution,
    Finalization,
    Closure,
}

impl Role {
    fn from_code(code: u16) -> Result<Role> {
        Ok(match code {
            0 => Role::Source,
            1 => Role::Contribution,
            2 => Role::Finalization,
            3 => Role::Closure,
            _ => bail!("unknown executor role {}", code),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Role::Source => "source",
            Role::Contribution => "contribution",
            Role::Finalization => "finalization",
            Role::Closure => "closure",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Current {
    pub id: u32,
    /// a source (external) current, else built by contributions
    pub source: bool,
    pub stage: u16,
    pub base: u32,
    pub count: u16,
    pub first_use: u32,
    pub last_use: u32,
}

/// parent0, parent1 (or `NONE`), their momentum forms, destination, exact factor, selector
/// domain, flags: the arena component bases of the currents
#[derive(Clone, Copy, Debug)]
pub struct Contribution {
    pub parent0: u32,
    pub parent1: u32,
    pub destination: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct RowGroup {
    pub stage: u16,
    pub role: Role,
    /// 0 initialize, 1 add, 2 finalize in place, 3 closure add
    pub operation: u16,
    pub executor: u32,
    pub row_start: u64,
    pub rows: u32,
}

impl RowGroup {
    pub fn operation_name(&self) -> &'static str {
        match self.operation {
            0 => "initialize",
            1 => "add",
            2 => "finalize-in-place",
            3 => "closure-add",
            _ => "?",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub strategy: u32,
    pub point_tile_size: u32,
    pub workspace_mib: u32,
    pub arena_components: u32,
    pub physical_sectors: u32,
    pub retained_helicities: u64,
    pub amplitude_destinations: u32,
    pub parameters: u32,
    pub external_sources: u32,
    pub executor_count: u32,
    /// the number of records of each table (`TABLES`)
    pub counts: Vec<(&'static str, u64)>,
    pub currents: Vec<Current>,
    pub contributions: Vec<Contribution>,
    pub row_groups: Vec<RowGroup>,
    /// (first term, number of terms) per form; terms are (source slot, coefficient)
    pub momentum_forms: Vec<(u64, u32)>,
    pub momentum_terms: Vec<(u32, i32)>,
}

impl Plan {
    pub fn strategy_name(&self) -> &'static str {
        match self.strategy {
            0 => "topology-replay",
            1 => "all-flow-union",
            2 => "contracted-color-union",
            _ => "?",
        }
    }

    pub fn count(&self, table: &str) -> u64 {
        self.counts
            .iter()
            .find(|(t, _)| *t == table)
            .map(|(_, n)| *n)
            .unwrap_or(0)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| anyhow!("truncated recurrence plan ({})", what))?;
        let b = &self.bytes[self.at..end];
        self.at = end;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2, "u16")?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, "u32")?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4, "i32")?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8, "u64")?.try_into().unwrap()))
    }
    fn rest(&self) -> usize {
        self.bytes.len() - self.at
    }
}

/// Decodes a plan up to its momentum forms (the rest: selectors, replay, helicity resolution,
/// exact factors and closure proofs, is counted, not decoded).
pub fn decode(bytes: &[u8]) -> Result<Plan> {
    let mut r = Reader { bytes, at: 0 };
    if r.take(8, "magic")? != MAGIC {
        bail!("not a recurrence direct plan v2 (wrong magic number)");
    }
    let version = r.u32()?;
    if version != VERSION {
        bail!("unsupported recurrence plan version {}", version);
    }
    if r.u32()? != 0 {
        bail!("unsupported recurrence plan header flags");
    }
    let strategy = r.u32()?;
    let point_tile_size = r.u32()?;
    let workspace_mib = r.u32()?;
    let arena_components = r.u32()?;
    let physical_sectors = r.u32()?;
    let retained_helicities = r.u64()?;
    let amplitude_destinations = r.u32()?;
    let parameters = r.u32()?;
    let external_sources = r.u32()?;
    // state templates, source templates, source templates or dispatch domains, runtime
    // helicity contracts and variants
    for _ in 0..5 {
        r.u32()?;
    }
    let executor_count = r.u32()?;
    // five digests, the closure candidate count and its digest
    r.take(5 * 32, "digests")?;
    r.u64()?;
    r.take(32, "digest")?;

    let mut counts = Vec::with_capacity(TABLES.len());
    for (name, _) in TABLES {
        counts.push((name, r.u64()?));
    }
    // the fixed-size tables must fit in what is left
    let fixed: u128 = counts
        .iter()
        .zip(TABLES)
        .map(|((_, n), (_, size))| *n as u128 * size as u128)
        .sum();
    if fixed > r.rest() as u128 {
        bail!("truncated recurrence plan (its tables do not fit)");
    }
    let n = |i: usize| counts[i].1 as usize;

    let mut currents = Vec::with_capacity(n(0));
    for _ in 0..n(0) {
        let id = r.u32()?;
        let kind = r.u16()?;
        let stage = r.u16()?;
        let _state_template = r.u32()?;
        let base = r.u32()?;
        let count = r.u16()?;
        r.u16()?;
        let _momentum_form = r.u32()?;
        let _selector = r.u32()?;
        let first_use = r.u32()?;
        let last_use = r.u32()?;
        r.u32()?;
        r.u32()?;
        currents.push(Current {
            id,
            source: kind == 0,
            stage,
            base,
            count,
            first_use,
            last_use,
        });
    }
    r.take(n(1) * 28, "sources")?;
    let mut contributions = Vec::with_capacity(n(2));
    for _ in 0..n(2) {
        let parent0 = r.u32()?;
        let parent1 = r.u32()?;
        r.u32()?;
        r.u32()?;
        let destination = r.u32()?;
        r.u32()?;
        r.u32()?;
        let flags = r.u32()?;
        contributions.push(Contribution {
            parent0,
            parent1,
            destination,
            flags,
        });
    }
    r.take(n(3) * 24 + n(4) * 40, "finalizations and closures")?;
    let mut row_groups = Vec::with_capacity(n(5));
    for _ in 0..n(5) {
        let stage = r.u16()?;
        let role = Role::from_code(r.u16()?)?;
        let operation = r.u16()?;
        r.u16()?;
        let executor = r.u32()?;
        let row_start = r.u64()?;
        let rows = r.u32()?;
        r.u32()?;
        let table = match role {
            Role::Source => 1,
            Role::Contribution => 2,
            Role::Finalization => 3,
            Role::Closure => 4,
        };
        if row_start + rows as u64 > counts[table].1 {
            bail!("recurrence row group exceeds its {} table", counts[table].0);
        }
        if executor != NONE && executor >= executor_count {
            bail!(
                "recurrence row group names executor {} of {}",
                executor,
                executor_count
            );
        }
        row_groups.push(RowGroup {
            stage,
            role,
            operation,
            executor,
            row_start,
            rows,
        });
    }
    let mut momentum_forms = Vec::with_capacity(n(6));
    for _ in 0..n(6) {
        let start = r.u64()?;
        let count = r.u32()?;
        r.u32()?;
        momentum_forms.push((start, count));
    }
    let mut momentum_terms = Vec::with_capacity(n(7));
    for _ in 0..n(7) {
        let slot = r.u32()?;
        momentum_terms.push((slot, r.i32()?));
    }

    Ok(Plan {
        strategy,
        point_tile_size,
        workspace_mib,
        arena_components,
        physical_sectors,
        retained_helicities,
        amplitude_destinations,
        parameters,
        external_sources,
        executor_count,
        counts,
        currents,
        contributions,
        row_groups,
        momentum_forms,
        momentum_terms,
    })
}

/// What an executor runs (from a template of `recurrence_direct_template.templates`).
#[derive(Clone, Debug)]
pub struct Executor {
    pub role: String,
    /// `rusticol-intrinsic`, `prepared-direct-call`, ...
    pub kind: String,
    /// the intrinsic (`runtime_template` without its key), or the prepared call's backend
    pub name: String,
    /// a prepared call's application (relative to `model/eager-kernels/`) and its SHA-256
    pub application: Option<(String, String)>,
    pub kernel_id: Option<u64>,
}

impl Executor {
    pub fn label(&self) -> String {
        match &self.application {
            Some((path, _)) => format!("{} {}", self.name, path),
            None => format!("{} {}", self.kind_short(), self.name),
        }
    }

    pub fn is_symjit(&self) -> bool {
        self.kind == "prepared-direct-call" && self.name == "jit"
    }

    fn kind_short(&self) -> &str {
        match self.kind.as_str() {
            "rusticol-intrinsic" => "intrinsic",
            k => k,
        }
    }
}

/// The executors of an `eager-kernel-pack.json`, by `direct_executor_id`.
pub fn executors(pack_json: &str) -> Result<HashMap<u32, Executor>> {
    let pack: serde_json::Value = serde_json::from_str(pack_json)?;
    let templates = pack["recurrence_direct_template"]["templates"]
        .as_array()
        .ok_or_else(|| anyhow!("the kernel pack has no recurrence direct templates"))?;
    let mut out = HashMap::new();
    for t in templates {
        let id = t["direct_executor_id"]
            .as_u64()
            .ok_or_else(|| anyhow!("a direct template has no executor id"))?
            as u32;
        let b = &t["payload_binding"];
        let kind = b["kind"].as_str().unwrap_or("?").to_string();
        let (name, application) = if kind == "rusticol-intrinsic" {
            let name = b["runtime_template"].as_str().unwrap_or("?");
            (name.split(':').next().unwrap_or(name).to_string(), None)
        } else {
            let app = b["source_application_path"].as_str().map(|p| {
                (
                    p.to_string(),
                    b["source_application_sha256"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                )
            });
            (t["backend"].as_str().unwrap_or("?").to_string(), app)
        };
        let executor = Executor {
            role: t["role"].as_str().unwrap_or("?").to_string(),
            kind,
            name,
            application,
            kernel_id: b["prepared_kernel_id"].as_u64(),
        };
        if out.insert(id, executor).is_some() {
            bail!("the kernel pack repeats executor {}", id);
        }
    }
    Ok(out)
}

/// The plans of `input` (an artifact directory, a `recurrence-runtime.pacbin` or a plan member
/// extracted from one), each with its name, its executors (from `pack`, or the artifact's
/// `model/eager-kernel-pack.json`) and the prepared kernels found in the artifact's
/// `evaluators.pacbin`.
pub fn load(input: &Path, pack: Option<&Path>) -> Result<Vec<Loaded>> {
    let mut files = Vec::new();
    let artifact = if input.is_dir() {
        let schedules = input.join("recurrence").join("schedules");
        if let Ok(entries) = std::fs::read_dir(&schedules) {
            for e in entries.flatten() {
                let f = e.path().join("recurrence-runtime.pacbin");
                if f.is_file() {
                    files.push(f);
                }
            }
        }
        files.sort();
        if files.is_empty() {
            bail!(
                "`{}` holds no recurrence plan (recurrence/schedules/*/recurrence-runtime.pacbin): \
                 not a recurrence-mode artifact",
                input.display()
            );
        }
        Some(input.to_path_buf())
    } else {
        files.push(input.to_path_buf());
        // recurrence/schedules/<digest>/recurrence-runtime.pacbin
        input
            .ancestors()
            .nth(4)
            .filter(|a| a.join("model").join("eager-kernel-pack.json").is_file())
            .map(Path::to_path_buf)
    };

    let pack = pack.map(Path::to_path_buf).or_else(|| {
        artifact
            .as_ref()
            .map(|a| a.join("model").join("eager-kernel-pack.json"))
    });
    let executors = match &pack {
        Some(p) => Some(
            executors(
                &std::fs::read_to_string(p)
                    .with_context(|| format!("cannot open `{}`", p.display()))?,
            )
            .with_context(|| format!("cannot read `{}`", p.display()))?,
        ),
        None => None,
    };
    let kernels = match (&artifact, &executors) {
        (Some(a), Some(ex)) => kernel_files(&a.join("evaluators.pacbin"), ex)?,
        _ => None,
    };

    let mut out = Vec::new();
    for f in files {
        let bytes = std::fs::read(&f).with_context(|| format!("cannot open `{}`", f.display()))?;
        let plan = if bytes.starts_with(MAGIC) {
            decode(&bytes)
        } else {
            let container = Pacbin::read(bytes, true)?;
            let member = container
                .member(MEMBER)
                .ok_or_else(|| anyhow!("the container has no recurrence plan (`{}`)", MEMBER))?;
            decode(container.payload(member))
        }
        .with_context(|| format!("cannot read `{}`", f.display()))?;
        out.push(Loaded {
            name: f.display().to_string(),
            plan,
            executors: executors.clone(),
            kernels: kernels.clone(),
        });
    }
    Ok(out)
}

/// A plan, as `load` finds it.
pub struct Loaded {
    pub name: String,
    pub plan: Plan,
    pub executors: Option<HashMap<u32, Executor>>,
    pub kernels: Option<HashMap<String, KernelFile>>,
}

/// The symjit kernels the executors call, looked up in `evaluators.pacbin` (members
/// `model/eager-kernels/<application path>`).
fn kernel_files(
    file: &Path,
    executors: &HashMap<u32, Executor>,
) -> Result<Option<HashMap<String, KernelFile>>> {
    if !file.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(file).with_context(|| format!("cannot open `{}`", file.display()))?;
    let container =
        Pacbin::read(bytes, true).with_context(|| format!("cannot read `{}`", file.display()))?;
    let mut out = HashMap::new();
    for e in executors.values() {
        let Some((path, sha)) = &e.application else {
            continue;
        };
        if out.contains_key(path) {
            continue;
        }
        let member = format!("model/eager-kernels/{}", path);
        if let Some(m) = container.member(&member) {
            let kernel = Kernel::read(&m.path, container.payload(m));
            out.insert(
                path.clone(),
                KernelFile {
                    member,
                    digest_ok: hex(&m.sha256) == *sha,
                    mir: kernel.error.is_none().then_some(kernel.mir),
                },
            );
        }
    }
    Ok(Some(out))
}

/// A prepared kernel an executor calls, as found in the artifact's `evaluators.pacbin`.
#[derive(Clone, Debug)]
pub struct KernelFile {
    pub member: String,
    /// the digest the template pins matches the member's
    pub digest_ok: bool,
    pub mir: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// the number of row groups listed (`usize::MAX`: all)
    pub groups: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { groups: 50 }
    }
}

/// The report on `plan` (from `name`); with the executors of the kernel pack, what each group
/// runs, and with `kernels` (by application path), the prepared kernels found.
pub fn describe(
    name: &str,
    plan: &Plan,
    executors: Option<&HashMap<u32, Executor>>,
    kernels: Option<&HashMap<String, KernelFile>>,
    options: &Options,
) -> Result<String> {
    let mut out = String::new();
    let o = &mut out;
    writeln!(o, "{}", name)?;
    let row = |o: &mut String, k: &str, v: String| {
        let _ = writeln!(o, "  {:<22} {}", k, v);
    };
    row(o, "strategy", plan.strategy_name().to_string());
    row(
        o,
        "point tile",
        format!(
            "{} points (workspace {} MiB)",
            thousands(plan.point_tile_size as usize),
            plan.workspace_mib
        ),
    );
    row(
        o,
        "external sources",
        format!(
            "{}; {} model parameters",
            plan.external_sources, plan.parameters
        ),
    );
    row(
        o,
        "colour sectors",
        format!(
            "{} physical; {} helicities retained",
            thousands(plan.physical_sectors as usize),
            thousands(plan.retained_helicities as usize)
        ),
    );
    row(
        o,
        "amplitudes",
        thousands(plan.amplitude_destinations as usize),
    );
    let components: usize = plan.currents.iter().map(|c| c.count as usize).sum();
    row(
        o,
        "currents",
        format!(
            "{} ({} complex components, in an arena of {}: currents whose uses do not overlap share it)",
            thousands(plan.currents.len()),
            thousands(components),
            thousands(plan.arena_components as usize)
        ),
    );
    let rows: u64 = plan.row_groups.iter().map(|g| g.rows as u64).sum();
    row(
        o,
        "row groups",
        format!(
            "{} ({} rows; {} executors in the catalog)",
            thousands(plan.row_groups.len()),
            thousands(rows as usize),
            thousands(plan.executor_count as usize)
        ),
    );
    let init = plan
        .contributions
        .iter()
        .filter(|c| c.flags & FLAG_INITIALIZE != 0)
        .count();
    let reuse = plan
        .contributions
        .iter()
        .filter(|c| c.flags & FLAG_CERTIFIED_REUSE != 0)
        .count();
    row(
        o,
        "contributions",
        format!(
            "{} rows: {} initialize their destination, the others add; {} certified reuses",
            thousands(plan.contributions.len()),
            thousands(init),
            thousands(reuse)
        ),
    );
    let tables: Vec<String> = plan
        .counts
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(t, n)| format!("{} {}", t, thousands(*n as usize)))
        .collect();
    wrapped(o, "tables", &tables.join(", "));

    // the stages: currents built, and the rows of each role
    let _ = writeln!(
        o,
        "\nStages        currents  components   sources contributions finalizations closures"
    );
    let mut stages: BTreeMap<u16, [usize; 6]> = BTreeMap::new();
    for c in plan.currents.iter() {
        let s = stages.entry(c.stage).or_default();
        s[0] += 1;
        s[1] += c.count as usize;
    }
    for g in plan.row_groups.iter() {
        let s = stages.entry(g.stage).or_default();
        let i = match g.role {
            Role::Source => 2,
            Role::Contribution => 3,
            Role::Finalization => 4,
            Role::Closure => 5,
        };
        s[i] += g.rows as usize;
    }
    for (stage, s) in stages.iter() {
        let cell = |v: usize| {
            if v == 0 {
                "-".to_string()
            } else {
                thousands(v)
            }
        };
        let _ = writeln!(
            o,
            "  {:<9} {:>10} {:>11} {:>9} {:>13} {:>13} {:>8}",
            stage,
            cell(s[0]),
            cell(s[1]),
            cell(s[2]),
            cell(s[3]),
            cell(s[4]),
            cell(s[5])
        );
    }

    let label = |id: u32| -> String {
        if id == NONE {
            return "none (union sources, filled before the schedule)".into();
        }
        match executors.and_then(|e| e.get(&id)) {
            Some(e) => e.label(),
            None if executors.is_some() => "? (not in the kernel pack)".into(),
            None => "-".into(),
        }
    };

    if let Some(executors) = executors {
        // what runs: the executors by rows
        let mut by: BTreeMap<String, (usize, usize, Role)> = BTreeMap::new();
        let mut symjit_rows = 0usize;
        let mut used = BTreeMap::new();
        for g in plan.row_groups.iter() {
            let e = by.entry(label(g.executor)).or_insert((0, 0, g.role));
            e.0 += 1;
            e.1 += g.rows as usize;
            if let Some(x) = executors.get(&g.executor) {
                if x.is_symjit() {
                    symjit_rows += g.rows as usize;
                    used.insert(g.executor, x);
                }
            }
        }
        let mut list: Vec<_> = by.into_iter().collect();
        list.sort_by(|a, b| b.1 .1.cmp(&a.1 .1).then(a.0.cmp(&b.0)));
        let _ = writeln!(
            o,
            "\nExecutors run (one call per row, over a tile of points)"
        );
        let _ = writeln!(
            o,
            "  {:>8} {:>7}  {:<13} executor",
            "rows", "groups", "role"
        );
        for (name, (groups, rows, role)) in list.iter() {
            let _ = writeln!(
                o,
                "  {:>8} {:>7}  {:<13} {}",
                thousands(*rows),
                thousands(*groups),
                role.name(),
                name
            );
        }
        let total: usize = plan.row_groups.iter().map(|g| g.rows as usize).sum();
        let kinds: BTreeMap<&str, usize> = executors.values().fold(BTreeMap::new(), |mut m, e| {
            *m.entry(if e.is_symjit() {
                "symjit"
            } else if e.kind == "rusticol-intrinsic" {
                "intrinsic"
            } else {
                "other"
            })
            .or_default() += 1;
            m
        });
        let catalog: Vec<String> = kinds
            .iter()
            .map(|(k, n)| format!("{} {}", thousands(*n), k))
            .collect();
        let _ = writeln!(
            o,
            "  symjit kernels run {} of {} rows ({} distinct executors; catalog: {})",
            thousands(symjit_rows),
            thousands(total),
            used.len(),
            catalog.join(", ")
        );

        if !used.is_empty() {
            let _ = writeln!(o, "\nsymjit kernels run");
            for (id, x) in used.iter() {
                let (path, _) = x.application.as_ref().unwrap();
                let found = kernels.and_then(|k| k.get(path));
                let status = match (kernels, found) {
                    (None, _) => String::new(),
                    (Some(_), None) => "  NOT in evaluators.pacbin".into(),
                    (Some(_), Some(f)) => format!(
                        "  {}{}",
                        f.mir
                            .map(|m| format!("{} MIR", thousands(m)))
                            .unwrap_or_else(|| "unreadable".into()),
                        if f.digest_ok { "" } else { ", DIGEST MISMATCH" }
                    ),
                };
                let _ = writeln!(o, "  executor {:<5} {}{}", id, path, status);
            }
        }
    }

    if options.groups > 0 {
        let shown = plan.row_groups.len().min(options.groups);
        if shown == plan.row_groups.len() {
            let _ = writeln!(o, "\nRow groups in execution order");
        } else {
            let _ = writeln!(
                o,
                "\nRow groups in execution order ({} of {}; --groups all lists every one)",
                shown,
                plan.row_groups.len()
            );
        }
        let _ = writeln!(
            o,
            "  {:>5} {:>5}  {:<13} {:<17} {:>8} {:>7}  runs",
            "group", "stage", "role", "operation", "executor", "rows"
        );
        for (i, g) in plan.row_groups.iter().take(shown).enumerate() {
            let ex = if g.executor == NONE {
                "-".to_string()
            } else {
                g.executor.to_string()
            };
            let _ = writeln!(
                o,
                "  {:>5} {:>5}  {:<13} {:<17} {:>8} {:>7}  {}",
                i,
                g.stage,
                g.role.name(),
                g.operation_name(),
                ex,
                thousands(g.rows as usize),
                label(g.executor)
            );
        }
    }
    Ok(out)
}

fn wrapped(o: &mut String, label: &str, text: &str) {
    let indent = 25;
    let mut line = String::new();
    let mut first = true;
    let flush = |o: &mut String, line: &str, first: &mut bool| {
        if *first {
            let _ = writeln!(o, "  {:<22} {}", label, line);
            *first = false;
        } else {
            let _ = writeln!(o, "{:indent$}{}", "", line);
        }
    };
    for word in text.split(' ') {
        if !line.is_empty() && indent + line.len() + 1 + word.len() > 100 {
            flush(o, &line, &mut first);
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    flush(o, &line, &mut first);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plan of two sources, one contribution group and one closure group, encoded as
    /// `direct_codec.rs` encodes it.
    pub(crate) fn sample() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        for v in [VERSION, 0, 0, 64, 16, 12, 1] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&4u64.to_le_bytes());
        for v in [1u32, 0, 2, 2, 2, 2, 0, 0, 3] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&[0u8; 5 * 32 + 8 + 32]);
        let mut counts = [0u64; 27];
        counts[0] = 3; // currents
        counts[1] = 2; // sources
        counts[2] = 2; // contributions
        counts[4] = 1; // closures
        counts[5] = 3; // row groups
        counts[6] = 1; // momentum forms
        counts[7] = 2; // momentum terms
        for c in counts {
            b.extend_from_slice(&c.to_le_bytes());
        }
        for (id, kind, stage, base) in [(0u32, 0u16, 0u16, 0u32), (1, 0, 0, 4), (2, 1, 1, 8)] {
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&kind.to_le_bytes());
            b.extend_from_slice(&stage.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&base.to_le_bytes());
            b.extend_from_slice(&4u16.to_le_bytes());
            b.extend_from_slice(&0u16.to_le_bytes());
            for v in [0u32, 0, stage as u32, 2, NONE, NONE] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b.extend_from_slice(&[0u8; 2 * 28]);
        for flags in [FLAG_INITIALIZE, 0] {
            for v in [0u32, 4, 0, 0, 8, 0, 0, flags] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b.extend_from_slice(&[0u8; 40]);
        for (stage, role, op, ex, start, rows) in [
            (0u16, 0u16, 0u16, 0u32, 0u64, 2u32),
            (1, 1, 1, 1, 0, 2),
            (2, 3, 3, 2, 0, 1),
        ] {
            for v in [stage, role, op, 0] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b.extend_from_slice(&ex.to_le_bytes());
            b.extend_from_slice(&start.to_le_bytes());
            b.extend_from_slice(&rows.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
        }
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        for (slot, c) in [(0u32, 1i32), (1, -1)] {
            b.extend_from_slice(&slot.to_le_bytes());
            b.extend_from_slice(&c.to_le_bytes());
        }
        b
    }

    #[test]
    fn decodes() {
        let p = decode(&sample()).unwrap();
        assert_eq!(p.strategy_name(), "topology-replay");
        assert_eq!((p.point_tile_size, p.arena_components), (64, 12));
        assert_eq!(p.currents.len(), 3);
        assert!(p.currents[0].source && !p.currents[2].source);
        assert_eq!(p.contributions[0].destination, 8);
        assert_eq!(p.row_groups[1].role, Role::Contribution);
        assert_eq!(p.row_groups[2].operation_name(), "closure-add");
        assert_eq!(p.momentum_terms, vec![(0, 1), (1, -1)]);
    }

    #[test]
    fn rejects_damage() {
        let good = sample();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(decode(&bad).unwrap_err().to_string().contains("magic"));
        assert!(decode(&good[..good.len() - 20])
            .unwrap_err()
            .to_string()
            .contains("truncated"));
        // a row group beyond its table
        let mut bad = good.clone();
        let rows_at = good.len() - 16 - 16 - 28 + 20; // the last group's row count
        bad[rows_at] = 9;
        assert!(decode(&bad).unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn joins_executors() {
        let pack = r#"{"recurrence_direct_template": {"templates": [
            {"direct_executor_id": 0, "role": "source", "backend": "jit",
             "payload_binding": {"kind": "rusticol-intrinsic",
                                 "runtime_template": "rusticol.source-fill.vector.v1:abc"}},
            {"direct_executor_id": 1, "role": "contribution", "backend": "jit",
             "payload_binding": {"kind": "prepared-direct-call", "prepared_kernel_id": 7,
                                 "source_application_path": "kernels/000007/application-1.plane.symjit",
                                 "source_application_sha256": "00"}},
            {"direct_executor_id": 2, "role": "closure", "backend": "jit",
             "payload_binding": {"kind": "rusticol-intrinsic",
                                 "runtime_template": "rusticol.closure-reduce.v1"}}]}}"#;
        let ex = executors(pack).unwrap();
        assert!(ex[&1].is_symjit() && !ex[&0].is_symjit());
        assert_eq!(ex[&0].label(), "intrinsic rusticol.source-fill.vector.v1");
        let p = decode(&sample()).unwrap();
        let text = describe("p", &p, Some(&ex), None, &Options::default()).unwrap();
        assert!(
            text.contains("symjit kernels run 2 of 5 rows (1 distinct executors"),
            "{}",
            text
        );
        assert!(text.contains("executor 1     kernels/000007/application-1.plane.symjit"));
    }
}
