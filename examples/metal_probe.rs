//! metal_probe: runs a recorded compiled-mode execution of pyAmpliCol on the Metal GPU, in
//! f32, and compares its amplitudes with the recording's (RustiCol, f64). See docs/METAL.md.
//!
//!   cargo run --release --example metal_probe -- ARTIFACT RECORDING.json [-o DIR]
//!       [--points N] [--repeat R]
//!
//! Every leaf of the execution becomes one Metal kernel on the recording's arena planes
//! (`MslOptions::mapping`), all in one source; `run.m` runs them in order on N points (the
//! recorded points repeated), R times, and reports the best GPU time. The arena starts as the
//! recording's `.pre` in f32, with the planes the execution writes first poisoned (NaN).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use symjit_utils::bindings;
use symjit_utils::msl::{prelude, MslOptions, RUN_M};
use symjit_utils::pacbin::{Kind, Pacbin};
use symjit_utils::recording::{self, Recording};
use symjit_utils::replay;
use symjit_utils::SavedModel;

struct Args {
    artifact: PathBuf,
    recording: PathBuf,
    dir: PathBuf,
    points: usize,
    repeat: usize,
}

fn parse() -> Result<Args> {
    let mut pos = Vec::new();
    let (mut dir, mut points, mut repeat) = (PathBuf::from("metal-probe"), 16384, 5);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().ok_or_else(|| anyhow!("`{}` needs a value", a));
        match a.as_str() {
            "-o" => dir = value()?.into(),
            "--points" => points = value()?.parse()?,
            "--repeat" => repeat = value()?.parse()?,
            _ => pos.push(PathBuf::from(&a)),
        }
    }
    let [artifact, recording] = <[PathBuf; 2]>::try_from(pos).map_err(|_| {
        anyhow!("usage: metal_probe ARTIFACT RECORDING.json [-o DIR] [--points N] [--repeat R]")
    })?;
    Ok(Args {
        artifact,
        recording,
        dir,
        points,
        repeat,
    })
}

fn write_f32(path: &Path, v: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes).with_context(|| format!("cannot write `{}`", path.display()))
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("cannot read `{}`", path.display()))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn quantiles(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let q = |f: f64| v[((v.len() - 1) as f64 * f) as usize];
    (q(0.5), q(0.99), v[v.len() - 1])
}

fn main() -> Result<()> {
    let a = parse()?;
    std::fs::create_dir_all(&a.dir)?;
    let (processes, pacbin) = bindings::load(&a.artifact)?;
    let pacbin = pacbin.ok_or_else(|| anyhow!("the artifact has no evaluators.pacbin"))?;
    let container = Pacbin::read(std::fs::read(&pacbin)?, false)?;
    let members: HashMap<&str, &[u8]> = container
        .members
        .iter()
        .filter(|m| m.kind == Kind::SymjitApplication)
        .map(|m| (m.path.as_str(), container.payload(m)))
        .collect();
    let rec = Recording::read(&a.recording)?;
    let resolved = replay::leaf_calls(&rec, &processes)?;
    let calls = &resolved.calls;
    println!(
        "{}: execution {}, {} leaves, {} planes, {} recorded points",
        rec.name,
        resolved.execution,
        calls.len(),
        rec.num_planes,
        rec.num_points
    );

    // the kernels
    let start = Instant::now();
    let mut source = prelude();
    let mut names = String::new();
    for (k, c) in calls.iter().enumerate() {
        let bytes = members
            .get(c.member.as_str())
            .ok_or_else(|| anyhow!("evaluators.pacbin has no kernel `{}`", c.member))?;
        let mut model = SavedModel::read(&mut &bytes[..])?;
        let mut options = MslOptions::new(&format!("leaf{}", k));
        options.mapping = Some(c.inputs.iter().chain(&c.outputs).copied().collect());
        let msl = model
            .msl(&options)
            .with_context(|| format!("{} ({})", c.label, c.member))?;
        source.push('\n');
        source.push_str(&msl.kernel);
        names.push_str(&format!("leaf{}\n", k));
    }
    let src = a.dir.join("probe.metal");
    std::fs::write(&src, &source)?;
    std::fs::write(a.dir.join("kernels.txt"), &names)?;
    println!(
        "sjb2msl: {} kernels, {:.1} MB of source, {:.2} s",
        calls.len(),
        source.len() as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );

    // the arena: N points, point i a copy of recorded point i % num_points
    let (np, n) = (rec.num_points, a.points);
    let mut pre = rec.pre.clone();
    let poisoned = recording::first_written(
        calls.iter().map(|c| (&c.inputs[..], &c.outputs[..])),
        rec.num_planes,
    );
    for &p in &poisoned {
        pre[p as usize * np..(p as usize + 1) * np].fill(f64::NAN);
    }
    let mut arena = vec![0f32; rec.num_planes * n];
    for p in 0..rec.num_planes {
        for i in 0..n {
            arena[p * n + i] = pre[p * np + i % np] as f32;
        }
    }
    write_f32(&a.dir.join("arena.f32"), &arena)?;
    drop(arena);

    // run.m
    let run_m = a.dir.join("run.m");
    std::fs::write(&run_m, RUN_M)?;
    let exe = a.dir.join("run");
    let out = Command::new("cc")
        .args(["-O2", "-fobjc-arc", "-Wno-deprecated-declarations"])
        .arg(&run_m)
        .args(["-framework", "Metal", "-framework", "Foundation", "-o"])
        .arg(&exe)
        .output()?;
    if !out.status.success() {
        bail!("cc run.m: {}", String::from_utf8_lossy(&out.stderr));
    }
    let out_path = a.dir.join("arena.out");
    let status = Command::new(&exe)
        .arg("arena")
        .arg(&src)
        .arg(a.dir.join("kernels.txt"))
        .arg(a.dir.join("arena.f32"))
        .arg(n.to_string())
        .arg(n.to_string())
        .arg(a.repeat.to_string())
        .arg(&out_path)
        .status()?;
    if !status.success() {
        bail!("run: {}", status);
    }
    let got = read_f32(&out_path)?;

    // the copies of a point agree bit for bit (the same inputs)
    let mut copies_differ = 0usize;
    for p in 0..rec.num_planes {
        for i in np..n {
            if got[p * n + i].to_bits() != got[p * n + i % np].to_bits() {
                copies_differ += 1;
            }
        }
    }

    // the amplitude planes (re, im)
    let mut re_planes = HashMap::new();
    let mut im_planes = HashMap::new();
    for c in calls.iter().filter(|c| c.label.starts_with("amplitude:")) {
        for &p in &c.outputs {
            match rec.region_of(p) {
                ("amplitude_re", j) => {
                    re_planes.insert(j, p as usize);
                }
                ("amplitude_im", j) => {
                    im_planes.insert(j, p as usize);
                }
                _ => {}
            }
        }
    }
    let mut amps: Vec<(usize, usize)> = re_planes
        .iter()
        .filter_map(|(j, &re)| im_planes.get(j).map(|&im| (re, im)))
        .collect();
    amps.sort();
    let what = if copies_differ > 0 {
        format!(
            ", {} values differ between copies of a point",
            copies_differ
        )
    } else {
        String::new()
    };
    println!("gpu (f32){}:", what);
    report(&rec, &amps, |p, i| got[p * n + i] as f64);

    // the CPU for comparison: pacman replay (RustiCol's SIMD JIT kernels, 1 thread)
    let mut need: Vec<(&str, &[u8])> = Vec::new();
    for c in calls.iter() {
        if !need.iter().any(|(m, _)| *m == c.member) {
            need.push((c.member.as_str(), members[c.member.as_str()]));
        }
    }
    let mut kernels = HashMap::new();
    for ((m, _), k) in need.iter().zip(replay::load_all(&need)) {
        kernels.insert(m.to_string(), k?);
    }
    let ks: Vec<&replay::CpuKernel> = calls.iter().map(|c| &kernels[&c.member]).collect();
    let options = replay::Options {
        whole: true,
        repeat: 3,
        ..Default::default()
    };
    let outcome = replay::replay(&rec, calls, &ks, &options)?;
    let cpu: f64 = outcome.stats.iter().map(|s| s.time.as_secs_f64()).sum();
    println!(
        "cpu: pacman replay (SIMD JIT, 1 thread) {:.6} s for {} points, {:.4} us/point",
        cpu,
        np,
        cpu / np as f64 * 1e6
    );
    println!("cpu (f64) against the recording:");
    report(&rec, &amps, |p, i| outcome.arena[p * np + i]);

    // f64 arithmetic on inputs rounded to f32: what the inputs' rounding alone costs, for all
    // of them, the momenta only and the others only
    let options = replay::Options {
        whole: true,
        ..Default::default()
    };
    for (what, round) in [
        ("all inputs", None),
        ("the momenta", Some(true)),
        ("the inputs but the momenta", Some(false)),
    ] {
        let mut rounded = rec.clone();
        for p in 0..rec.num_planes {
            let momentum = rec.region_of(p as u32).0 == "momenta";
            if round.is_none_or(|m| m == momentum) {
                for x in rounded.pre[p * np..(p + 1) * np].iter_mut() {
                    *x = *x as f32 as f64;
                }
            }
        }
        let outcome = replay::replay(&rounded, calls, &ks, &options)?;
        println!("cpu (f64) on {} rounded to f32:", what);
        report(&rec, &amps, |p, i| outcome.arena[p * np + i]);
    }
    Ok(())
}

/// The amplitudes `got(plane, point)` against the recording's `.post`: per (re, im) value,
/// relative to the largest |A| of the point, and |A|² summed over the amplitudes (a helicity
/// sum's shape) per point.
fn report(rec: &Recording, amps: &[(usize, usize)], got: impl Fn(usize, usize) -> f64) {
    let np = rec.num_points;
    let (mut value_err, mut sum_err) = (Vec::new(), Vec::new());
    let (mut non_finite, mut zero_points) = (0usize, 0usize);
    for i in 0..np {
        let largest = amps
            .iter()
            .map(|&(re, im)| rec.post[re * np + i].hypot(rec.post[im * np + i]))
            .fold(0.0, f64::max);
        let (mut s_got, mut s_want) = (0.0f64, 0.0f64);
        for &(re, im) in amps {
            for p in [re, im] {
                let (g, w) = (got(p, i), rec.post[p * np + i]);
                if !g.is_finite() {
                    non_finite += 1;
                    continue;
                }
                if largest > 0.0 {
                    value_err.push((g - w).abs() / largest);
                }
                s_got += g * g;
                s_want += w * w;
            }
        }
        if s_want > 0.0 {
            sum_err.push((s_got - s_want).abs() / s_want);
        } else {
            zero_points += 1;
        }
    }
    let (v50, v99, vmax) = quantiles(value_err);
    let (s50, s99, smax) = quantiles(sum_err);
    println!(
        "  {} (re, im) amplitude planes, {} points; {} non-finite values",
        amps.len(),
        np,
        non_finite
    );
    println!(
        "  |A - A_rusticol| / max|A| per point: median {:.1e}, 99% {:.1e}, max {:.1e}",
        v50, v99, vmax
    );
    println!(
        "  sum |A|^2 per point, relative: median {:.1e}, 99% {:.1e}, max {:.1e}{}",
        s50,
        s99,
        smax,
        if zero_points > 0 {
            format!(" ({} points with no amplitude)", zero_points)
        } else {
            String::new()
        }
    );
}
