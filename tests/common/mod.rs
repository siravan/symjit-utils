//! Helpers of the end-to-end tests: external tools, C's `%a` output, the Node host of wasm modules.

#![allow(dead_code)]

use std::path::Path;
use std::process::{Command, Stdio};

pub fn have(tool: &str) -> bool {
    let ok = Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipped: `{}` is not available", tool);
    }
    ok
}

/// Parses C's %a output (e.g. 0x1.8p+1, -0x0p+0, inf, nan).
pub fn hexf(s: &str) -> Option<f64> {
    match s {
        "inf" => return Some(f64::INFINITY),
        "-inf" => return Some(f64::NEG_INFINITY),
        "nan" | "-nan" => return Some(f64::NAN),
        _ => {}
    }
    let (neg, s) = match s.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, s),
    };
    let s = s.strip_prefix("0x")?;
    let (mant, exp) = s.split_once('p')?;
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut v = u64::from_str_radix(int, 16).ok()? as f64;
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        v += c.to_digit(16)? as f64 * scale;
        scale /= 16.0;
    }
    let v = v * 2f64.powi(exp.parse().ok()?);
    Some(if neg { -v } else { v })
}

/// The lines of %a numbers a C program printed.
pub fn parse_hex(out: &[u8]) -> Vec<Vec<f64>> {
    String::from_utf8_lossy(out)
        .lines()
        .map(|l| {
            l.split_whitespace()
                .map(|v| hexf(v).unwrap_or_else(|| panic!("bad output `{}`", v)))
                .collect()
        })
        .collect()
}

/// A number for the JSON requests of `wasm_runner.mjs`.
fn json(v: f64) -> String {
    if v.is_nan() {
        "\"NaN\"".into()
    } else if v.is_infinite() {
        if v > 0.0 {
            "\"Infinity\""
        } else {
            "\"-Infinity\""
        }
        .into()
    } else if v == 0.0 && v.is_sign_negative() {
        "\"-0\"".into()
    } else {
        format!("{:?}", v)
    }
}

/// Runs `module` in Node (`tests/wasm_runner.mjs`) in `mode` at `points`; the outputs, or None
/// without node. `mode` "params" is the Symbolica convention (inputs as parameters, outputs in
/// memory).
pub fn run_wasm(
    module: &Path,
    mode: &str,
    count_states: usize,
    count_obs: usize,
    points: &[Vec<f64>],
) -> Option<Vec<Vec<f64>>> {
    if !have("node") {
        return None;
    }
    let runner = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/wasm_runner.mjs");
    let points: Vec<String> = points
        .iter()
        .map(|p| {
            format!(
                "[{}]",
                p.iter().map(|&v| json(v)).collect::<Vec<_>>().join(", ")
            )
        })
        .collect();
    let req = format!(
        "{{\"module\": {:?}, \"mode\": {:?}, \"count_states\": {}, \"count_obs\": {}, \
         \"mem_size\": {}, \"points\": [{}], \"params\": []}}",
        module.to_str().unwrap(),
        mode,
        count_states,
        count_obs,
        (count_states + count_obs).max(64),
        points.join(", ")
    );
    let mut child = Command::new("node")
        .arg(runner)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(req.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    Some(parse_obs(&text))
}

// the `obs` of the runner's answer, {"obs": [[y0, y1, ...], ...], ...}, with the special values
// that JSON cannot hold as strings ("NaN", "Infinity", "-Infinity", "-0")
fn parse_obs(text: &str) -> Vec<Vec<f64>> {
    let obs = text.split("\"obs\":").nth(1).unwrap();
    let obs = &obs[..obs.find("]]").unwrap() + 2];
    obs.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split("],")
        .map(|row| {
            row.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .split(',')
                .map(|v| match v.trim().trim_matches('"') {
                    "NaN" => f64::NAN,
                    "Infinity" => f64::INFINITY,
                    "-Infinity" => f64::NEG_INFINITY,
                    "-0" => -0.0,
                    v => v.parse().unwrap(),
                })
                .collect()
        })
        .collect()
}
