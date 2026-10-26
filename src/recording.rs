//! Arena recordings of a `pacman` build of RustiCol (`PACMAN_SAVE_ARENA`, see `docs/PACMAN.md`):
//! the arena of one compiled-mode execution right before its first kernel (`NAME.pre`) and after
//! its last (`NAME.post`), with `NAME.json` describing it.
//!
//! The arena is plane-major little-endian f64, `[num_planes][num_points]`, its planes in the
//! order of `bindings::Arena`. The points are the recorded tiles one after the other: tile `t`
//! is points `offset[t]..offset[t] + tiles[t]`, and RustiCol ran the schedule's leaves on each
//! tile in turn.
//!
//! This module only reads, poisons and compares arenas; it does not run kernels, so any
//! executor (`replay`'s CPU kernels, a GPU driver) can use it.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::bindings::Arena;

pub const FORMAT: &str = "pacman-arena-v1";

/// The value written into a poisoned plane (`| plane`): a quiet NaN that names its plane.
pub const POISON: u64 = 0x7ff8_dead_beef_0000;

/// One leaf of the recorded schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct Scheduled {
    /// the stage's index in the execution (the amplitude stage is the last)
    pub stage: usize,
    /// the stage's `evaluator_label`
    pub label: String,
    pub amplitude: bool,
    /// the leaf's index in its stage
    pub leaf: usize,
    /// the plane kernel, relative to the process directory (as in `execution.json`)
    pub application_path: String,
}

/// A recording: its metadata and both snapshots.
#[derive(Clone, Debug)]
pub struct Recording {
    /// the `.json` file
    pub path: PathBuf,
    pub name: String,
    pub process: String,
    pub engine: u64,
    pub method: String,
    pub num_planes: usize,
    pub num_points: usize,
    pub current_planes: u32,
    pub amplitude_planes: u32,
    pub momentum_planes: u32,
    pub model_parameters: u32,
    /// canonical current component → physical current plane (`unmapped_physical`: none)
    pub canonical_to_physical: Vec<u32>,
    pub unmapped_physical: u32,
    /// the physical current planes the sources fill
    pub source_planes: Vec<u32>,
    /// points per tile, in order
    pub tiles: Vec<usize>,
    pub schedule: Vec<Scheduled>,
    pub pre: Vec<f64>,
    pub post: Vec<f64>,
}

fn usize_of(v: &Value, what: &str) -> Result<usize> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| anyhow!("`{}` is not a number", what))
}

fn u32_of(v: &Value, what: &str) -> Result<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| anyhow!("`{}` is not a 32-bit number", what))
}

fn str_of<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str()
        .ok_or_else(|| anyhow!("`{}` is not a string", what))
}

fn array<'a>(v: &'a Value, what: &str) -> Result<&'a Vec<Value>> {
    v.as_array()
        .ok_or_else(|| anyhow!("`{}` is not an array", what))
}

fn u32s(v: &Value, what: &str) -> Result<Vec<u32>> {
    array(v, what)?.iter().map(|x| u32_of(x, what)).collect()
}

/// The little-endian f64 values of a file of exactly `len` of them (an arena: `.pre`, `.post`,
/// a runner's `.arena`).
pub fn read_f64s(path: &Path, len: usize) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot open `{}`", path.display()))?;
    if bytes.len() != len * 8 {
        bail!(
            "`{}` holds {} bytes, expected {} ({} f64)",
            path.display(),
            bytes.len(),
            len * 8,
            len
        );
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect())
}

impl Recording {
    /// Reads `NAME.json` and the `NAME.pre` and `NAME.post` beside it.
    pub fn read(json: &Path) -> Result<Recording> {
        let text = std::fs::read_to_string(json)
            .with_context(|| format!("cannot open `{}`", json.display()))?;
        let mut r =
            Recording::parse(&text).with_context(|| format!("cannot read `{}`", json.display()))?;
        r.path = json.to_path_buf();
        let len = r.num_planes * r.num_points;
        r.pre = read_f64s(&json.with_extension("pre"), len)?;
        r.post = read_f64s(&json.with_extension("post"), len)?;
        Ok(r)
    }

    /// The metadata of a recording (`pre` and `post` empty), checked for consistency.
    pub fn parse(text: &str) -> Result<Recording> {
        let d: Value = serde_json::from_str(text)?;
        let format = str_of(&d["format"], "format")?;
        if format != FORMAT {
            bail!("format `{}`, expected `{}`", format, FORMAT);
        }
        for (key, want) in [("dtype", "f64-le"), ("layout", "plane-major")] {
            let got = str_of(&d[key], key)?;
            if got != want {
                bail!("`{}` is `{}`, expected `{}`", key, got, want);
            }
        }
        let mut schedule = Vec::new();
        for st in array(&d["schedule"], "schedule")? {
            let stage = usize_of(&st["stage"], "stage")?;
            let label = str_of(&st["label"], "label")?.to_string();
            let amplitude = st["amplitude"]
                .as_bool()
                .ok_or_else(|| anyhow!("`amplitude` is not a boolean"))?;
            for l in array(&st["leaves"], "leaves")? {
                schedule.push(Scheduled {
                    stage,
                    label: label.clone(),
                    amplitude,
                    leaf: usize_of(&l["leaf"], "leaf")?,
                    application_path: str_of(&l["application_path"], "application_path")?
                        .to_string(),
                });
            }
        }
        let r = Recording {
            path: PathBuf::new(),
            name: str_of(&d["name"], "name")?.to_string(),
            process: str_of(&d["process"], "process")?.to_string(),
            engine: d["engine"].as_u64().unwrap_or(0),
            method: str_of(&d["method"], "method")?.to_string(),
            num_planes: usize_of(&d["num_planes"], "num_planes")?,
            num_points: usize_of(&d["num_points"], "num_points")?,
            current_planes: u32_of(&d["current_planes"], "current_planes")?,
            amplitude_planes: u32_of(&d["amplitude_planes"], "amplitude_planes")?,
            momentum_planes: u32_of(&d["momentum_planes"], "momentum_planes")?,
            model_parameters: u32_of(&d["model_parameters"], "model_parameters")?,
            canonical_to_physical: u32s(&d["canonical_to_physical"], "canonical_to_physical")?,
            unmapped_physical: u32_of(&d["unmapped_physical"], "unmapped_physical")?,
            source_planes: u32s(&d["source_planes"], "source_planes")?,
            tiles: array(&d["tiles"], "tiles")?
                .iter()
                .map(|t| usize_of(t, "tiles"))
                .collect::<Result<_>>()?,
            schedule,
            pre: Vec::new(),
            post: Vec::new(),
        };

        let arena = r.arena();
        if arena.len() as usize != r.num_planes {
            bail!(
                "{} planes, but the region counts add up to {}",
                r.num_planes,
                arena.len()
            );
        }
        let regions = array(&d["regions"], "regions")?;
        let expected = arena.regions();
        if regions.len() != expected.len() {
            bail!("{} regions, expected {}", regions.len(), expected.len());
        }
        for (got, (name, first, count)) in regions.iter().zip(expected) {
            let g = (
                str_of(&got["name"], "region name")?,
                u32_of(&got["first"], "region first")?,
                u32_of(&got["count"], "region count")?,
            );
            if g != (name, first, count) {
                bail!(
                    "region {:?}, expected `{}` from {} ({} planes)",
                    g,
                    name,
                    first,
                    count
                );
            }
        }
        if r.tiles.contains(&0) || r.tiles.iter().sum::<usize>() != r.num_points {
            bail!(
                "the tiles {:?} do not split the {} points",
                r.tiles,
                r.num_points
            );
        }
        if r.schedule.is_empty() {
            bail!("the schedule runs no leaf");
        }
        if let Some(&p) = r
            .canonical_to_physical
            .iter()
            .chain(&r.source_planes)
            .find(|&&p| p != r.unmapped_physical && p >= r.current_planes)
        {
            bail!("current plane {} of {}", p, r.current_planes);
        }
        Ok(r)
    }

    /// The arena's layout (`bindings::Arena`, whose regions the recording repeats).
    pub fn arena(&self) -> Arena {
        Arena {
            currents: self.current_planes,
            amplitudes: self.amplitude_planes,
            momenta: self.momentum_planes,
            parameters: self.model_parameters,
        }
    }

    /// The first point of each tile.
    pub fn offsets(&self) -> Vec<usize> {
        self.tiles
            .iter()
            .scan(0, |next, &n| {
                let first = *next;
                *next += n;
                Some(first)
            })
            .collect()
    }

    /// The region and the index in it of an arena plane.
    pub fn region_of(&self, plane: u32) -> (&'static str, u32) {
        for (name, first, n) in self.arena().regions() {
            if plane >= first && plane < first + n {
                return (name, plane - first);
            }
        }
        ("outside", plane)
    }
}

/// The planes whose first access, over `calls` (a kernel's (input planes, output planes), in
/// call order; a call reads all its inputs before it writes), is a write. Whatever the snapshot
/// holds in them is never read by a correct execution, so they can be poisoned. The others that
/// the calls touch are read first: the execution's true inputs.
pub fn first_written<'a>(
    calls: impl IntoIterator<Item = (&'a [u32], &'a [u32])>,
    num_planes: usize,
) -> Vec<u32> {
    // 0: untouched, 1: read first, 2: written first
    let mut first = vec![0u8; num_planes];
    for (inputs, outputs) in calls {
        for &p in inputs {
            if first[p as usize] == 0 {
                first[p as usize] = 1;
            }
        }
        for &p in outputs {
            if first[p as usize] == 0 {
                first[p as usize] = 2;
            }
        }
    }
    (0..num_planes as u32)
        .filter(|&p| first[p as usize] == 2)
        .collect()
}

/// Fills `planes` of a plane-major arena of `num_points` points with `POISON | plane`.
pub fn poison(arena: &mut [f64], num_points: usize, planes: &[u32]) {
    for &p in planes {
        let value = f64::from_bits(POISON | p as u64);
        arena[p as usize * num_points..(p as usize + 1) * num_points].fill(value);
    }
}

/// Distance in units in the last place between two finite doubles (`u64::MAX` if either is
/// not finite and they differ).
pub fn ulps(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() {
        return u64::MAX;
    }
    // map the doubles onto a monotonic integer line (-0.0 just below 0.0)
    let key = |x: f64| {
        let magnitude = (x.to_bits() & !(1 << 63)) as i64;
        if x.is_sign_negative() {
            -magnitude - 1
        } else {
            magnitude
        }
    };
    key(a).abs_diff(key(b))
}

/// A plane that differs from the reference.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneDiff {
    pub plane: u32,
    /// the points whose bits differ
    pub points: usize,
    /// the first of them, with the values found and expected
    pub point: usize,
    pub got: f64,
    pub want: f64,
    /// the largest distance in ulps over the plane's differing points
    pub max_ulps: u64,
    /// the largest difference over the plane's points (infinite if a value is not finite)
    pub max_abs: f64,
    /// the largest difference over the plane's points relative to the largest magnitude in the
    /// reference plane (infinite if either is not finite, or the reference is all zeros): near
    /// zero, after cancellation, ulps say little
    pub relative: f64,
}

/// The planes of `got` whose bits differ from `want` (NaNs equal only with the same bits).
pub fn compare(got: &[f64], want: &[f64], num_points: usize) -> Vec<PlaneDiff> {
    assert_eq!(got.len(), want.len());
    let mut out = Vec::new();
    if num_points == 0 {
        return out;
    }
    for (p, (g, w)) in got
        .chunks_exact(num_points)
        .zip(want.chunks_exact(num_points))
        .enumerate()
    {
        let mut diff: Option<PlaneDiff> = None;
        let mut largest = 0.0f64;
        for (i, (&a, &b)) in g.iter().zip(w).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            let u = ulps(a, b);
            // NaN compares as the largest
            let delta = (a - b).abs();
            largest = if delta.is_nan() {
                f64::INFINITY
            } else {
                largest.max(delta)
            };
            match diff.as_mut() {
                Some(d) => {
                    d.points += 1;
                    d.max_ulps = d.max_ulps.max(u);
                }
                None => {
                    diff = Some(PlaneDiff {
                        plane: p as u32,
                        points: 1,
                        point: i,
                        got: a,
                        want: b,
                        max_ulps: u,
                        max_abs: 0.0,
                        relative: 0.0,
                    })
                }
            }
        }
        if let Some(d) = diff.as_mut() {
            d.max_abs = if largest.is_finite() {
                largest
            } else {
                f64::INFINITY
            };
            let scale = w.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            d.relative = if scale.is_finite() && scale > 0.0 && largest.is_finite() {
                largest / scale
            } else {
                f64::INFINITY
            };
        }
        out.extend(diff);
    }
    out
}

/// `1.5e-3` style, with a poison value named as such.
pub fn show(x: f64) -> String {
    let bits = x.to_bits();
    if bits & !0xffff == POISON {
        format!("poison({})", bits & 0xffff)
    } else {
        format!("{:e}", x)
    }
}

/// `n ulp(s)`, or `not finite`.
pub fn show_ulps(u: u64) -> String {
    match u {
        u64::MAX => "not finite".to_string(),
        1 => "1 ulp".to_string(),
        u => format!("{} ulps", u),
    }
}

/// One line for a differing plane.
pub fn describe_diff(r: &Recording, d: &PlaneDiff) -> String {
    let (region, index) = r.region_of(d.plane);
    let mut s = String::new();
    let _ = write!(
        s,
        "plane {} ({}[{}]): {} of {} points differ, first at point {}: {} instead of {} ({}",
        d.plane,
        region,
        index,
        d.points,
        r.num_points,
        d.point,
        show(d.got),
        show(d.want),
        show_ulps(ulps(d.got, d.want))
    );
    if d.points > 1 {
        let _ = write!(s, "; at most {}", show_ulps(d.max_ulps));
    }
    let _ = write!(
        s,
        "; {} of the plane's largest value)",
        show_relative(d.relative)
    );
    s
}

/// A relative difference: `2.3e-16`, or `not finite`.
pub fn show_relative(x: f64) -> String {
    if x.is_finite() {
        format!("{:.1e}", x)
    } else {
        "not finite".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(tiles: &str) -> String {
        format!(
            r#"{{"format": "pacman-arena-v1", "dtype": "f64-le", "layout": "plane-major",
            "name": "p.e0.r0", "process": "p", "engine": 0, "recording": 0, "method": "all",
            "num_planes": 14, "num_points": 3, "current_planes": 3, "amplitude_planes": 1,
            "momentum_planes": 1, "model_parameters": 2,
            "regions": [
              {{"name": "current_re", "first": 0, "count": 3}},
              {{"name": "current_im", "first": 3, "count": 3}},
              {{"name": "amplitude_re", "first": 6, "count": 1}},
              {{"name": "amplitude_im", "first": 7, "count": 1}},
              {{"name": "momenta", "first": 8, "count": 1}},
              {{"name": "zero", "first": 9, "count": 1}},
              {{"name": "parameters", "first": 10, "count": 4}}],
            "canonical_to_physical": [0, 4294967295, 1, 2], "unmapped_physical": 4294967295,
            "source_planes": [0], "tiles": {},
            "schedule": [{{"stage": 0, "label": "s1", "amplitude": false,
              "leaves": [{{"leaf": 0, "application_path": "evaluators/a.plane.symjit"}}]}}]}}"#,
            tiles
        )
    }

    #[test]
    fn parses_metadata() {
        let r = Recording::parse(&meta("[2, 1]")).unwrap();
        assert_eq!(r.num_planes, 14);
        assert_eq!(r.offsets(), vec![0, 2]);
        assert_eq!(r.region_of(9), ("zero", 0));
        assert_eq!(r.region_of(11), ("parameters", 1));
        assert_eq!(
            r.schedule,
            vec![Scheduled {
                stage: 0,
                label: "s1".into(),
                amplitude: false,
                leaf: 0,
                application_path: "evaluators/a.plane.symjit".into()
            }]
        );
    }

    #[test]
    fn rejects_inconsistent_metadata() {
        // tiles not adding up to the points
        assert!(Recording::parse(&meta("[2, 2]")).is_err());
        assert!(Recording::parse(&meta("[3, 0]")).is_err());
        // regions not adding up to the planes
        let wrong = meta("[3]").replace("\"num_planes\": 14", "\"num_planes\": 13");
        assert!(Recording::parse(&wrong).is_err());
        let wrong = meta("[3]").replace("plane-major", "point-major");
        assert!(Recording::parse(&wrong).is_err());
        // a current plane outside the region
        let wrong = meta("[3]").replace("[0, 4294967295, 1, 2]", "[0, 3]");
        assert!(Recording::parse(&wrong).is_err());
    }

    #[test]
    fn reads_snapshots_of_the_right_size() {
        let dir = std::env::temp_dir().join(format!("pacman-recording-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let json = dir.join("p.e0.r0.json");
        std::fs::write(&json, meta("[3]")).unwrap();
        let values: Vec<u8> = (0..42).flat_map(|i| (i as f64).to_le_bytes()).collect();
        std::fs::write(json.with_extension("pre"), &values).unwrap();
        std::fs::write(json.with_extension("post"), &values[..values.len() - 8]).unwrap();
        let err = format!("{:#}", Recording::read(&json).unwrap_err());
        assert!(err.contains("expected 336"), "{}", err);
        std::fs::write(json.with_extension("post"), &values).unwrap();
        let r = Recording::read(&json).unwrap();
        assert_eq!(r.pre[41], 41.0);
        assert_eq!(r.post.len(), 42);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn poisons_only_planes_written_first() {
        // call 1 reads 0 and 9 and writes 1; call 2 reads 1 and writes 0 (a reused source
        // plane, read first) and 2
        let calls: Vec<(Vec<u32>, Vec<u32>)> =
            vec![(vec![0, 9], vec![1]), (vec![1, 9], vec![0, 2])];
        let first = first_written(calls.iter().map(|(i, o)| (&i[..], &o[..])), 13);
        assert_eq!(first, vec![1, 2]);
        // a call reads before it writes: a plane both read and written by it is read first
        let first = first_written([(&[3u32][..], &[3u32][..])], 13);
        assert!(first.is_empty());

        let mut arena = vec![1.0; 13 * 2];
        poison(&mut arena, 2, &[1, 2]);
        assert_eq!(arena[0], 1.0);
        assert_eq!(arena[2].to_bits(), POISON | 1);
        assert_eq!(arena[5].to_bits(), POISON | 2);
        assert!(arena[5].is_nan());
        assert_eq!(arena[6], 1.0);
        assert_eq!(show(arena[5]), "poison(2)");
    }

    #[test]
    fn compares_bits() {
        let want = vec![1.0, 2.0, f64::NAN, 0.0, 5.0, 6.0];
        let mut got = want.clone();
        assert!(compare(&got, &want, 3).is_empty());
        got[1] = f64::from_bits(2.0f64.to_bits() + 3);
        got[3] = -0.0;
        got[5] = f64::from_bits(POISON | 1);
        let d = compare(&got, &want, 3);
        assert_eq!(d.len(), 2);
        assert_eq!(
            (d[0].plane, d[0].points, d[0].point, d[0].max_ulps),
            (0, 1, 1, 3)
        );
        assert_eq!((d[1].plane, d[1].points, d[1].point), (1, 2, 0));
        assert_eq!(d[1].max_ulps, u64::MAX);
        // plane 0: 3 ulps of 2 against a largest value of 2
        assert_eq!(d[0].relative, 3.0 * f64::EPSILON);
        assert!(d[1].relative.is_infinite());
        assert_eq!(d[0].max_abs, 6.0 * f64::EPSILON);
        // NaNs of other bits differ
        let other = f64::from_bits(f64::NAN.to_bits() | 1);
        let mut nan = want.clone();
        nan[2] = other;
        assert_eq!(compare(&nan, &want, 3).len(), 1);

        assert_eq!(ulps(1.0, f64::from_bits(1.0f64.to_bits() + 1)), 1);
        assert_eq!(ulps(-0.0, 0.0), 1);
        assert_eq!(ulps(-f64::from_bits(1), f64::from_bits(1)), 3);
        assert_eq!(ulps(1.0, f64::INFINITY), u64::MAX);

        let r = Recording::parse(&meta("[3]")).unwrap();
        let mut d = d[0].clone();
        d.plane = 4;
        assert_eq!(
            describe_diff(&r, &d),
            "plane 4 (current_im[1]): 1 of 3 points differ, first at point 1: \
             2.0000000000000013e0 instead of 2e0 (3 ulps; 6.7e-16 of the plane's largest value)"
        );
    }
}
