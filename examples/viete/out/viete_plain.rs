// pi by Viète's formula (https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula), as in
// stable/examples/pi2.py:
//
//     2/pi = sqrt(1/2) * sqrt(1/2 + 1/2 sqrt(1/2)) * sqrt(1/2 + 1/2 sqrt(1/2 + 1/2 sqrt(1/2))) * ...
//
// `viete(0.5)` is pi (within 1.2e-6 for N = 10: the error shrinks by 4 per factor; pi2.py takes 21
// factors). As in pi2.py, factor `i` rebuilds its nested square roots from `x`, so the expression
// has N (N + 1) / 2 square roots before symjit shares the common ones.
//
// rust2sjb compiles the `#[kernel]` function into a .sjb; run.sh also compiles this file as plain
// Rust (the attribute removed) for the reference value.

const N: usize = 20;

fn viete(x: f64) -> f64 {
    let mut p = 1.0;
    for i in 0..N {
        let mut t = x;
        for _ in 0..i {
            t = x + x * t.sqrt();
        }
        p *= t.sqrt();
    }
    2.0 / p
}

fn main() {
    let p = viete(0.5);
    println!("rust: viete(0.5) = {}  (pi - viete(0.5) = {:.3e})", p, std::f64::consts::PI - p);
}
