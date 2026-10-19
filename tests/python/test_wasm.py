"""Tests of sjb2wasm: functions compiled by symjit (v227's Python package) are saved and
compiled by sjb2wasm into WebAssembly modules (`sjb_util.write_wasm`).

The generated modules are checked with wabt's `wasm-validate` and executed in Node
(`wasm_runner.mjs`, which also supplies the imported math functions) in the direct,
indirect and (when exported) fast calling modes; the results are compared with
numpy and with symjit's own evaluation of the function.

`WasmFuzz` runs the random expressions of test_fuzz.py through the backend, and
`WasmComposer` the random Composer programs of test_composer_fuzz.py (straight-line
code, if/else blocks and loops); all modules of a family are evaluated by one Node
process. `WasmSymbolica` compiles Symbolica evaluators (skipped without symbolica).
Environment variables SYMJIT_FUZZ_SEEDS and SYMJIT_FUZZ_START work as in the fuzz tests;
SYMJIT_WASM_TIMEOUT (seconds, default 300) bounds each Node run, so that an infinite
loop in generated code fails the test instead of hanging it.

Skipped without the sjb2wasm binary (see sjb_util.py), or when `node` or `wasm-validate` is not
on PATH.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sjb_util import HAVE_SJB2WASM, write_wasm

from symjit import Composer, compile_composer, compile_evaluator, compile_func, load_func

import test_composer_fuzz as cf
import test_fuzz as fz

# imported lazily: the unlicensed symbolica aborts a second process that imports it
HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "..", "wasm_runner.mjs")
HAVE_TOOLS = shutil.which("node") is not None and shutil.which("wasm-validate") is not None
TIMEOUT = float(os.environ.get("SYMJIT_WASM_TIMEOUT", "300"))

x, y, z, p = sp.symbols("x y z p")


HAVE_WASM = HAVE_SJB2WASM
NEEDS = "needs sjb2wasm (cargo build --release in symjit-utils/, or $SJB2WASM), node and wabt (wasm-validate)"
X = sp.symbols("x0:20")



def encode(v):
    """a float for JSON: NaN, infinities and -0.0 as strings (see wasm_runner.mjs)"""
    v = float(v)
    if np.isfinite(v) and not (v == 0 and np.signbit(v)):
        return v
    return "-0" if v == 0 else str(v).replace("inf", "Infinity").replace("nan", "NaN")


def decode(v):
    return float(v.replace("Infinity", "inf")) if isinstance(v, str) else float(v)


def decode_result(res):
    if "obs" in res:
        res["obs"] = [[decode(v) for v in row] for row in res["obs"]]
    return res


def run_node(requests):
    """Evaluates a request (dict) or a batch of requests (list) with wasm_runner.mjs."""
    try:
        res = subprocess.run(["node", RUNNER], input=json.dumps(requests), capture_output=True, text=True,
                             timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"node did not finish in {TIMEOUT:g} s (an infinite loop in the generated code?)")
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    out = json.loads(res.stdout)
    return [decode_result(r) for r in out] if isinstance(out, list) else decode_result(out)


def request(path, f, points, params=(), mode="direct"):
    return {
        "module": path,
        "mode": mode,
        "count_states": f.count_states,
        "count_obs": f.count_obs,
        "mem_size": 4096,
        "points": [[encode(v) for v in pt] for pt in points],
        "params": [encode(v) for v in params],
    }


def has_fast(f):
    return f.count_params == 0 and f.count_obs == 1


def as_reals(points):
    """complex points -> interleaved (re, im) values, the layout of complex states"""
    return [[v for c in pt for v in (c.real, c.imag)] for pt in points]


def as_complex(obs):
    obs = np.asarray(obs)
    return obs[:, 0::2] + 1j * obs[:, 1::2]


@unittest.skipUnless(HAVE_WASM and HAVE_TOOLS, NEEDS)
class Wasm(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def module(self, f, name="f.wasm"):
        path = os.path.join(self.dir.name, name)
        write_wasm(f, path)
        res = subprocess.run(["wasm-validate", path], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        return path

    def run_modes(self, f, points, params=(), complex_=False):
        """{mode: obs} for every calling mode the module supports"""
        path = self.module(f)
        pts = as_reals(points) if complex_ else points
        modes = ["direct", "indirect"] + (["fast"] if has_fast(f) else [])
        out = {}
        for mode, res in zip(modes, run_node([request(path, f, pts, params, m) for m in modes])):
            self.assertNotIn("error", res, f"{mode}: {res.get('error')}")
            self.assertTrue(all(s == 0 for s in res["status"]), res["status"])
            obs = np.array(res["obs"])
            out[mode] = as_complex(obs) if complex_ else obs
        return out

    def check(self, states, eqs, points, params=(), param_syms=None, tol=1e-12, ref_eqs=None, **kw):
        kw = dict(kw)
        if param_syms is not None:
            kw["params"] = param_syms
        complex_ = kw.get("dtype") == "complex128"
        f = compile_func(states, eqs, **kw)

        dtype = complex if complex_ else float
        ref_fn = sp.lambdify([states + list(param_syms or [])], ref_eqs or eqs, "numpy")
        with np.errstate(all="ignore"):
            ref = np.array([ref_fn(list(pt) + list(params)) for pt in points], dtype=dtype)
        py = np.array([f(*pt, *params) for pt in points], dtype=dtype)
        np.testing.assert_allclose(py, ref, rtol=tol, atol=tol, err_msg="python call")

        for mode, got in self.run_modes(f, points, params, complex_).items():
            np.testing.assert_allclose(got, ref, rtol=tol, atol=tol, err_msg=f"node, {mode} mode")
        return f

    POINTS = [(0.5, 2.0), (-1.25, 0.75), (3.0, -0.5)]
    POINTS3 = [(0.5, 2.0, -1.0), (-1.25, 0.75, 0.3), (3.0, -0.5, 2.5), (0.0, 0.0, 0.0)]

    # ------------------------------------------------- data movement and calls (phase 0)
    def test_permutation(self):
        self.check([x, y], [y, x, x], self.POINTS)

    def test_imported_unary(self):
        f = self.check([x, y], [sp.sin(x), sp.cos(y), sp.exp(x)], self.POINTS)
        res = run_node(request(self.module(f), f, self.POINTS))
        self.assertEqual(sorted(res["imports"]), ["cos", "exp", "sin"])

    def test_imported_binary(self):
        pts = [(0.5, 2.0), (1.5, 0.25), (3.0, -0.5)]
        self.check([x, y], [x**y, sp.atan2(y, x)], pts)

    def test_nested_calls(self):
        for opt_level in (0, 3):
            self.check([x], [sp.sin(sp.exp(sp.cos(x)))], [(0.1,), (-2.0,), (4.5,)], opt_level=opt_level)

    def test_constant(self):
        self.check([x], [sp.Float(2.5), x], [(1.0,), (-3.0,)])

    def test_parameters(self):
        self.check([x], [sp.sin(p) * x + p], [(1.0,), (2.0,)], params=[0.75], param_syms=[p])

    def test_many_states(self):
        # more than 16 states/observables: copied with a loop in indirect mode
        pts = [tuple(np.linspace(-1, 1, 20) + k) for k in range(3)]
        self.check(list(X), list(reversed(X)), pts)

    def test_compressed_models_are_rejected(self):
        # a shape with four variables that occurs more than twice becomes a funclet, which the
        # backend does not support (`compress=True` changes nothing for other models)
        v = sp.symbols("v0:8")
        eqs = [v[i] * v[(i + 1) % 8] + v[(i + 2) % 8] / (v[(i + 3) % 8] ** 2 + 1) for i in range(8)]
        f = compile_func(list(v), eqs, compress=True)
        with self.assertRaisesRegex(ValueError, "does not support compression mode"):
            write_wasm(f, os.path.join(self.dir.name, "c.wasm"))
        self.check([x], [sp.sin(x) + x], [(0.5,)], compress=True)

    # ------------------------------------------------- arithmetic and masks (phase 1)
    def test_arithmetic(self):
        eqs = [x + y, x - y, x * y, x / y, -x, x**2, x**3, 1 / x, (x + y) ** 2 / (1 + x * x), x / 2]
        for opt_level in (0, 1, 2, 3):
            self.check([x, y], eqs, self.POINTS, opt_level=opt_level)

    def test_fused_operations(self):
        # products followed by sums are fused into mul-add/mul-sub at opt_level >= 1
        eqs = [x * y + z, x * y - z, z - x * y, -x * y - z, x * y * z + x * z + y]
        for opt_level in (0, 1, 2, 3):
            self.check([x, y, z], eqs, self.POINTS3, opt_level=opt_level)

    def test_sqrt_abs_rounding(self):
        pts = [(0.5, 2.0), (2.25, -1.5), (7.0, 0.75), (4.5, -2.5)]
        eqs = [sp.sqrt(x), sp.Abs(y), sp.floor(y), sp.ceiling(y), x - sp.floor(x)]
        self.check([x, y], eqs, pts)

    def test_min_max(self):
        self.check([x, y, z], [sp.Min(x, y), sp.Max(x, y, z), sp.Min(x, 0) + sp.Max(y, 1)], self.POINTS3)

    def test_piecewise(self):
        eqs = [
            sp.Piecewise((x, x > y), (y, True)),
            sp.Piecewise((x * y, x < 0), (x + y, y >= 1), (z, True)),
            sp.Piecewise((sp.sin(x), sp.And(x > 0, y > 0)), (sp.cos(y), sp.Or(x <= -1, z == 0)), (1.5, True)),
            sp.Piecewise((1.0, sp.Eq(x, y)), (2.0, sp.Ne(z, 0)), (3.0, True)),
        ]
        pts = self.POINTS3 + [(1.0, 1.0, 2.0), (-1.0, -1.0, 0.0)]
        for opt_level in (0, 2, 3):
            self.check([x, y, z], eqs, pts, opt_level=opt_level)

    def test_special_values(self):
        # NaN, infinities, signed zeros and subnormals: bit-identical to the bytecode
        # interpreter, apart from the sign and payload of NaNs (comparisons with NaN are false,
        # as in IEEE 754 and numpy; the native backends deviate there, see
        # test_special_values.py)
        vals = [0.0, -0.0, 1.0, -1.5, np.inf, -np.inf, np.nan, 5e-324, -2.2e-308, 1e308, 0.5, 2.5, -2.5]
        pts = [(a, b) for a in vals for b in vals]
        eqs = [
            x + y, x - y, x * y, x / y, -x, 1 / x, x**2, x**3, x / 2, x * y + 1.0, x * y - x, sp.sqrt(x),
            sp.Abs(x), sp.floor(x), sp.ceiling(x), sp.Min(x, y), sp.Max(x, y), sp.exp(x), x**y,
            sp.Piecewise((x, x > y), (y, True)), sp.Piecewise((x, x <= y), (-y, True)),
            sp.Piecewise((1.0, sp.Eq(x, y)), (2.0, sp.Ne(x, 0)), (3.0, True)),
            sp.Piecewise((1.0, sp.And(x > 0, y < 0)), (2.0, sp.Or(x < y, sp.Not(y >= 1))), (0.0, True)),
        ]
        f = g = compile_func([x, y], eqs, ty="bytecode")
        with np.errstate(all="ignore"):
            want = np.array([g(*pt) for pt in pts])
        path = self.module(f)
        for mode, res in zip(("direct", "indirect"), run_node([request(path, f, pts, (), m) for m in ("direct", "indirect")])):
            got = np.array(res["obs"])
            same = (np.isnan(got) & np.isnan(want)) | ((got == want) & (np.signbit(got) == np.signbit(want)))
            bad = np.argwhere(~same)
            self.assertEqual(len(bad), 0, f"{mode}: e.g. {eqs[bad[0][1]]} at {pts[bad[0][0]]}: "
                             f"{got[tuple(bad[0])]!r} (bytecode {want[tuple(bad[0])]!r})" if len(bad) else "")

    def test_fast_export(self):
        f = self.check([x, y], [sp.sin(x) * y + x / (1 + y * y)], self.POINTS)
        self.assertTrue(has_fast(f))
        # more states than registers: arguments still arrive as wasm parameters
        pts = [tuple(np.linspace(-1, 1, 20) * k) for k in (1, 2)]
        self.check(list(X), [sum(v * (i + 1) for i, v in enumerate(X))], pts)

    def test_no_fast_export_with_parameters_or_several_outputs(self):
        for f in (compile_func([x], [x * p], params=[p]), compile_func([x], [x, 2 * x])):
            [res] = run_node([request(self.module(f), f, [(1.0,)], [2.0], "fast")])
            self.assertIn("no `fast` export", res.get("error", ""))

    def test_complex_arithmetic(self):
        pts = [(0.5 + 0.25j, -1.0 + 2.0j), (1.5 - 0.5j, 0.75 + 0.0j), (-2.0 + 1.0j, 0.3 - 1.2j)]
        eqs = [x + y, x - y, x * y, x / y, x * x * y + 2 * x, 1 / (1 + x * y), sp.conjugate(x) * y, sp.re(x) + sp.im(y)]
        for opt_level in (0, 2):
            self.check([x, y], eqs, pts, dtype="complex128", opt_level=opt_level)

    def test_stable_math(self):
        # range-safe complex division, 1/z, abs and sqrt: bit-identical to the bytecode
        # interpreter. `stable_math` passes the scaled numerator of a division in re(Temp),
        # which the fallback division of `Mir::rerun` used as scratch (the module divides
        # itself, `WasmGenerator::divide_complex`)
        pts = [(1e308 + 5e-324j, 1 + 0j), (3e200 + 4e200j, 1e200 - 2e200j), (3e-320 + 4e-320j, 1e-320 - 2e-320j),
               (3 + 4j, 1 - 2j), (-0.0 - 0.0j, 2 + 0j), (1e-160 + 1e160j, 1e160 + 1e-160j)]
        eqs = [x / y, 1 / y, sp.Abs(x), sp.sqrt(x), sp.sin(x / y)]
        for stable_math in (False, True):
            for opt_level in (0, 2, 3):
                kw = dict(dtype="complex128", stable_math=stable_math, opt_level=opt_level)
                f = compile_func([x, y], eqs, **kw)
                g = compile_func([x, y], eqs, ty="bytecode", fast_complex=False, **kw)
                with np.errstate(all="ignore"):
                    want = np.array([[v for c in g(*pt) for v in (complex(c).real, complex(c).imag)] for pt in pts])
                path = self.module(f)
                for mode, res in zip(("direct", "indirect"), run_node([request(path, f, as_reals(pts), (), m)
                                                                      for m in ("direct", "indirect")])):
                    got = np.array(res["obs"])
                    same = (np.isnan(got) & np.isnan(want)) | (got.view(np.uint64) == want.view(np.uint64))
                    self.assertTrue(same.all(), f"stable_math={stable_math}, opt_level={opt_level}, {mode}:\n"
                                    f"{got}\nbytecode:\n{want}")

    # ------------------------------------------------- not supported yet / save and load
    # ------------------------------------------------- control flow and complex calls (phase 2)
    def test_loops(self):
        k = sp.Symbol("k")
        eqs = [sp.Sum(x**k / (1 + k), (k, 0, 10)), sp.Product(1 + x / k, (k, 1, 6)), sp.Sum(sp.sin(k * x), (k, 1, 5)) + y]
        for opt_level in (0, 2, 3):
            self.check([x, y], eqs, self.POINTS, opt_level=opt_level, tol=1e-11, ref_eqs=[e.doit() for e in eqs])

    def test_complex_functions(self):
        pts = [(0.5 + 0.25j, -1.0 + 2.0j), (1.5 - 0.5j, 0.75 + 0.1j), (-2.0 + 1.0j, 0.3 - 1.2j)]
        eqs = [sp.sin(x), sp.cos(x) * sp.exp(y), sp.sqrt(x * y + 1), x**y, sp.tanh(x) + sp.log(y), sp.sqrt(x) ** 3]
        for opt_level in (0, 2):
            self.check([x, y], eqs, pts, dtype="complex128", opt_level=opt_level, tol=1e-12)

    def test_user_defined_functions_are_rejected(self):
        # a saved function does not hold them
        ext = sp.Function("ext")
        f = compile_func([x, y], [ext(x) + y], defuns={"ext": lambda a: 2 * a + 1})
        with self.assertRaisesRegex(ValueError, "saved application does not hold"):
            write_wasm(f, os.path.join(self.dir.name, "r.wasm"))

    def test_composer(self):
        # the Symbolica calling convention: run(outs, 0, 0, params)
        cp = Composer(2, 2)
        a, b = cp.arg(0), cp.arg(1)
        cp.assign(cp.out(0), cp.fadd(cp.fmul(a, b), cp.constant(1.0)))
        cp.assign(cp.out(1), cp.fdiv(a, cp.fadd(b, cp.constant(2.0))))
        f = compile_composer(cp)
        path = os.path.join(self.dir.name, "c.wasm")
        write_wasm(f, path)
        pts = [[0.5, 2.0], [-1.25, 0.75]]
        req = {"module": path, "mode": "params", "count_states": 0, "count_obs": 2, "mem_size": 64,
               "points": pts, "params": []}
        [res] = run_node([req])
        self.assertNotIn("error", res, res.get("error"))
        # bit for bit as the JIT (which divides by multiplying with the reciprocal here)
        np.testing.assert_array_equal(np.array(res["obs"]), np.asarray(f.evaluate(np.array(pts))))

    def test_save_and_load(self):
        # a loaded function saves the same application
        f = compile_func([x], [sp.sin(x) * x + 1])
        sjb = os.path.join(self.dir.name, "f.sjb")
        f.save(sjb)
        g = load_func(sjb)
        with open(self.module(f, "a.wasm"), "rb") as a, open(self.module(g, "b.wasm"), "rb") as b:
            self.assertEqual(a.read(), b.read())


# the option combinations every fuzz case is compiled with (SIMD, threads and fast
# complex do not apply to wasm)
FUZZ_OPTIONS = [
    dict(opt_level=0),
    dict(opt_level=1),
    dict(opt_level=2),
    dict(opt_level=3, cse=False),
    dict(opt_level=2, fastmath=False),
    dict(opt_level=2, compact=False),
]


@unittest.skipUnless(HAVE_WASM and HAVE_TOOLS, NEEDS)
class WasmFuzz(unittest.TestCase):
    tol = 1e-8

    def run_family(self, family, generator, dtype=float):
        X = fz.inputs(fz.NPOINTS, 7, dtype)
        complex_ = dtype == complex
        points = X.T.tolist() if not complex_ else as_reals(X.T)
        jobs = []

        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(fz.START, fz.START + fz.SEEDS):
                exprs = fz.build_case(family, generator, seed)
                if exprs is None:
                    continue
                try:
                    ref = fz.reference(exprs, X, dtype)
                except Exception:
                    continue
                if not np.all(np.isfinite(ref)):
                    continue

                for k, options in enumerate(FUZZ_OPTIONS):
                    kw = dict(options, dtype="complex128" if complex_ else "float64")
                    try:
                        f = compile_func(fz.V, exprs, **kw)
                    except ValueError as e:
                        raise AssertionError(f"{family} seed {seed}, options {options}: {e}\n{exprs}")

                    path = os.path.join(tmp, f"{seed}-{k}.wasm")
                    write_wasm(f, path)
                    modes = ["indirect", "direct"] + (["fast"] if has_fast(f) else [])
                    for mode in modes:
                        pts = points if mode == "indirect" else points[:9]
                        jobs.append(((seed, options, mode, exprs, ref), request(path, f, pts, (), mode)))

            results = run_node([job for _, job in jobs]) if jobs else []

        self.assertGreater(len(jobs), 0)
        for (seed, options, mode, exprs, ref), res in zip((info for info, _ in jobs), results):
            with self.subTest(seed=seed, options=options, mode=mode):
                self.assertNotIn("error", res, f"{family} seed {seed}: {res.get('error')}\n{exprs}")
                obs = np.array(res["obs"])
                got = (as_complex(obs) if complex_ else obs).T
                want = ref[:, : got.shape[1]]
                err = np.abs(got - want) / (1 + np.abs(want))
                self.assertTrue(
                    np.all(err < self.tol),
                    f"{family} seed {seed}, options {options}, {mode}: relative error {err.max():.3g}\n"
                    f"expressions: {exprs}",
                )

    def test_real(self):
        self.run_family("real", fz.case_real)

    def test_shared(self):
        self.run_family("shared", fz.case_shared)

    def test_piecewise(self):
        self.run_family("piecewise", fz.case_piecewise)

    def test_complex(self):
        self.run_family("complex", fz.case_complex, complex)


# Composer programs with the Symbolica calling convention: run(outs, 0, 0, params)
COMPOSER_OPTIONS = {
    "straight": [dict(), dict(direct=False), dict(fastmath=False), dict(opt_level=0)],
    "branches": [dict(), dict(direct=False), dict(simd_branch=False), dict(fastmath=False), dict(opt_level=0)],
    "loops": [dict(), dict(simd_branch=False), dict(fastmath=False), dict(opt_level=0)],
}


@unittest.skipUnless(HAVE_WASM and HAVE_TOOLS, NEEDS)
class WasmComposer(unittest.TestCase):
    tol = 1e-8

    def run_family(self, family):
        X = np.random.default_rng(7).uniform(-2, 2, (cf.NPARAMS, cf.NPOINTS))
        points = X.T.tolist()
        jobs = []

        with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
            for seed in range(cf.START, cf.START + cf.SEEDS):
                program = cf.Program(seed, family)
                ref = program.reference(X)
                if not np.all(np.isfinite(ref)) or np.max(np.abs(ref)) > 1e12:
                    continue
                for k, options in enumerate(COMPOSER_OPTIONS[family]):
                    try:
                        f = compile_composer(program.compose(), **options)
                    except ValueError as e:
                        raise AssertionError(f"composer {family} seed {seed}, options {options}: {e}")
                    path = os.path.join(tmp, f"{seed}-{k}.wasm")
                    write_wasm(f, path)
                    c = f.compiler
                    req = {
                        "module": path, "mode": "params", "count_states": 0, "count_obs": c.count_obs,
                        "mem_size": max(64, c.count_obs), "points": [[encode(v) for v in p] for p in points],
                        "params": [],
                    }
                    jobs.append(((seed, options, ref), req))

            results = run_node([req for _, req in jobs])

        self.assertGreater(len(jobs), 0)
        for (seed, options, ref), res in zip((info for info, _ in jobs), results):
            with self.subTest(seed=seed, options=options):
                self.assertNotIn("error", res, f"composer {family} seed {seed}: {res.get('error')}")
                self.assertTrue(all(s == 0 for s in res["status"]))
                got = np.array(res["obs"]).T
                err = np.abs(got - ref) / (1 + np.abs(ref))
                self.assertTrue(np.all(err < self.tol),
                                f"composer {family} seed {seed}, options {options}: relative error {err.max():.3g}")

    def test_straight_line(self):
        self.run_family("straight")

    def test_branches(self):
        self.run_family("branches")

    def test_loops(self):
        self.run_family("loops")


@unittest.skipUnless(HAVE_WASM and HAVE_TOOLS, NEEDS)
@unittest.skipUnless(HAVE_SYMBOLICA, "needs symbolica")
class WasmSymbolica(unittest.TestCase):
    EXPRS = [
        "x^2 + 3*x*y - sin(y)/(1 + x^2)",
        "if(y, x + 1, x + 2)",
        "if(x - y, exp(x) * cos(y), sqrt(1 + x^2))",
        "(x + y)^5 - (x - y)^3 + x*y*cos(x*y)",
    ]

    def test_evaluators(self):
        from symbolica import E, S

        X = np.random.default_rng(3).uniform(-1, 1, (40, 2))
        X[::3, 1] = 0.0  # take both branches of if()
        with tempfile.TemporaryDirectory() as tmp:
            for k, text in enumerate(self.EXPRS):
                ev = E(text).evaluator([S("x"), S("y")])
                ref = np.asarray(ev.evaluate(X))
                for simd_branch in (False, True):
                    with self.subTest(expr=text, simd_branch=simd_branch):
                        f = compile_evaluator(ev, simd_branch=simd_branch)
                        path = os.path.join(tmp, f"{k}-{simd_branch}.wasm")
                        write_wasm(f, path)
                        req = {
                            "module": path, "mode": "params", "count_states": 0, "count_obs": f.compiler.count_obs,
                            "mem_size": 64, "points": [[encode(v) for v in p] for p in X.tolist()],
                            "params": [],
                        }
                        [res] = run_node([req])
                        self.assertNotIn("error", res, res.get("error"))
                        np.testing.assert_allclose(np.array(res["obs"]), ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
