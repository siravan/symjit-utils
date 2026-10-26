"""Tests of sjb2ptx: functions compiled by symjit (stable's Python package) are saved and compiled
by sjb2ptx into PTX modules with their C drivers (`sjb_util.write_ptx`), which are built into one
shared library per test (`Batch`) and called with ctypes; the CUDA driver runs the kernels on
the GPU.

The results are compared with numpy and with symjit's own evaluation of the function:
* bit for bit with the native scalar kernel (`use_simd=False`) for arithmetic and special values
  (the generator follows x86's `amd/scalar.rs`);
* within 1e-13 of the JIT for math functions (the GPU's are musl's, see src/ptx/libm);
* within the fuzz tests' tolerance of numpy.

`PtxFuzz` runs the random expressions of test_fuzz.py through the backend in both layouts,
`PtxComposer` the random Composer programs of test_composer_fuzz.py (straight-line code, if/else
blocks and loops), and `PtxSymbolica` compiles Symbolica evaluators. Environment variables
SYMJIT_FUZZ_SEEDS and SYMJIT_FUZZ_START work as in the fuzz tests.

Skipped without the sjb2ptx binary (see sjb_util.py), without `cc`, or without a usable GPU (no
driver, or `/dev/nvidia*` unreachable, as in a sandbox); `SJB2PTX_REQUIRE_GPU=1` makes a missing
GPU an error.
"""

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sjb_util import HAVE_SJB2PTX, write_ptx

from symjit import Composer, compile_composer, compile_evaluator, compile_func, load_func

import test_composer_fuzz as cf
import test_fuzz as fz

# imported lazily: the unlicensed symbolica aborts a second process that imports it
HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None

x, y, z, p = sp.symbols("x y z p")
X = sp.symbols("x0:20")


def gpu_status():
    """(usable, why not): libcuda loads, cuInit succeeds and there is a device"""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        return False, f"no CUDA driver ({e})"
    r = cuda.cuInit(0)
    if r != 0:
        return False, f"cuInit returned {r}"
    count = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return False, "no CUDA device"
    return True, ""


def keep_primary_context():
    """Holds a reference to device 0's primary context for the whole process: each driver
    releases its reference in `<name>_cuda_free`, and the last release would destroy the context
    (and the next `init` create it again, which takes most of the time of a test)."""
    cuda = ctypes.CDLL("libcuda.so.1")
    ctx, dev = ctypes.c_void_p(), ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0:
        cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    return ctx


HAVE_GPU, NO_GPU = gpu_status()
PRIMARY_CONTEXT = keep_primary_context() if HAVE_GPU else None
if not HAVE_GPU and os.environ.get("SJB2PTX_REQUIRE_GPU"):
    raise RuntimeError(f"SJB2PTX_REQUIRE_GPU is set, but there is no usable GPU: {NO_GPU}")
HAVE_CC = shutil.which("cc") is not None
NEEDS = f"needs sjb2ptx (cargo build --release in symjit-utils/, or $SJB2PTX), cc and a GPU ({NO_GPU})"
READY = HAVE_SJB2PTX and HAVE_CC and HAVE_GPU


class Batch:
    """Kernels written by sjb2ptx --driver into a new directory in `tmp`, built into one shared
    library (each driver's symbols start with its kernel's name) and run on host arrays. Each
    batch has its own directory: dlopen returns an already loaded library of the same path."""

    def __init__(self, tmp):
        self.tmp = tempfile.mkdtemp(dir=tmp)
        self.kernels = {}
        self.lib = None

    def add(self, f, layout="soa", **save_args):
        """Compiles `f`; returns the kernel's name."""
        name = f"k{len(self.kernels)}"
        write_ptx(f, os.path.join(self.tmp, name), name, layout, **save_args)
        with open(os.path.join(self.tmp, f"{name}_cuda.h")) as h:
            defs = dict(re.findall(r"#define \w+_CUDA_(\w+) (\d+)", h.read()))
        self.kernels[name] = {
            "count_in": int(defs["COUNT_IN"]),
            "count_out": int(defs["COUNT_OUT"]),
            "count_params": int(defs["COUNT_PARAMS"]),
            "soa": defs["LAYOUT_SOA"] == "1",
        }
        return name

    def build(self):
        def compile_one(name):
            res = subprocess.run(["cc", "-O1", "-fPIC", "-Wall", "-Werror", "-c", f"{name}_cuda.c", "-o",
                                  f"{name}_cuda.o"], cwd=self.tmp, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr)

        with ThreadPoolExecutor(os.cpu_count()) as pool:
            list(pool.map(compile_one, self.kernels))
        lib = os.path.join(self.tmp, "kernels.so")
        objs = [f"{name}_cuda.o" for name in self.kernels]
        res = subprocess.run(["cc", "-shared", "-o", lib, *objs, "-ldl"], cwd=self.tmp, capture_output=True,
                             text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr)
        self.lib = ctypes.CDLL(lib)
        dp = ctypes.POINTER(ctypes.c_double)
        for name in self.kernels:
            f = getattr(self.lib, f"{name}_cuda_init")
            f.argtypes, f.restype = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int], ctypes.c_int
            f = getattr(self.lib, f"{name}_cuda_evaluate")
            f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, dp, ctypes.c_size_t, dp, ctypes.c_size_t, dp]
            f.restype = ctypes.c_int
            f = getattr(self.lib, f"{name}_cuda_error")
            f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_char_p
            f = getattr(self.lib, f"{name}_cuda_free")
            f.argtypes, f.restype = [ctypes.c_void_p], None
        return self

    def run(self, name, points, params=()):
        """The outputs (one row per point) for `points` (one row per point, the kernel's inputs)."""
        k = self.kernels[name]
        n = len(points)
        pts = np.asarray(points, dtype=float).reshape(n, k["count_in"])
        if k["soa"]:
            inp, ld_in = np.ascontiguousarray(pts.T), n
            out, ld_out = np.zeros((k["count_out"], n)), n
        else:
            inp, ld_in = np.ascontiguousarray(pts), k["count_in"]
            out, ld_out = np.zeros((n, k["count_out"])), k["count_out"]
        if inp.size == 0:
            inp = np.zeros(1)
        par = np.ascontiguousarray(params, dtype=float) if len(params) else None
        dp = ctypes.POINTER(ctypes.c_double)
        ctx = ctypes.c_void_p()
        lib = self.lib
        error = getattr(lib, f"{name}_cuda_error")
        try:
            if getattr(lib, f"{name}_cuda_init")(ctypes.byref(ctx), 0) != 0:
                raise RuntimeError(error(ctx).decode())
            r = getattr(lib, f"{name}_cuda_evaluate")(ctx, n, inp.ctypes.data_as(dp), ld_in,
                                                      out.ctypes.data_as(dp), ld_out,
                                                      par.ctypes.data_as(dp) if par is not None else None)
            if r != 0:
                raise RuntimeError(error(ctx).decode())
        finally:
            getattr(lib, f"{name}_cuda_free")(ctx)
        return out.T if k["soa"] else out


def as_reals(points):
    """complex points -> interleaved (re, im) values, the layout of complex states"""
    return [[v for c in pt for v in (complex(c).real, complex(c).imag)] for pt in points]


def as_complex(obs):
    obs = np.asarray(obs)
    return obs[:, 0::2] + 1j * obs[:, 1::2]


def same_bits(got, want):
    """the same values, bit for bit (NaNs of any payload are equal: the GPU's are canonical)"""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    return (np.isnan(got) & np.isnan(want)) | (got.view(np.uint64) == want.view(np.uint64))


@unittest.skipUnless(READY, NEEDS)
class Ptx(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def gpu(self, f, points, params=(), layouts=("soa", "aos"), complex_=False):
        """{layout: outputs} of `f` on the GPU"""
        batch = Batch(self.dir.name)
        names = {layout: batch.add(f, layout) for layout in layouts}
        batch.build()
        pts = as_reals(points) if complex_ else points
        out = {}
        for layout, name in names.items():
            obs = batch.run(name, pts, params)
            out[layout] = as_complex(obs) if complex_ else obs
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

        for layout, got in self.gpu(f, points, params, complex_=complex_).items():
            np.testing.assert_allclose(got, ref, rtol=tol, atol=tol, err_msg=f"GPU, {layout}")
            np.testing.assert_allclose(got, py, rtol=tol, atol=tol, err_msg=f"GPU vs JIT, {layout}")
        return f

    def check_bits(self, states, eqs, points, **kw):
        """bit for bit with the native scalar kernel"""
        f = compile_func(states, eqs, use_simd=False, **kw)
        complex_ = kw.get("dtype") == "complex128"
        with np.errstate(all="ignore"):
            want = np.array([np.atleast_1d(f(*pt)) for pt in points], dtype=complex if complex_ else float)
        if complex_:
            want = np.array(as_reals(want))
        for layout, got in self.gpu(f, as_reals(points) if complex_ else points).items():
            same = same_bits(got, want)
            bad = np.argwhere(~same)
            self.assertEqual(len(bad), 0, f"{layout}, {kw}: e.g. output {bad[0][1]} at {points[bad[0][0]]}: "
                             f"{got[tuple(bad[0])]!r} (JIT {want[tuple(bad[0])]!r})" if len(bad) else "")

    POINTS = [(0.5, 2.0), (-1.25, 0.75), (3.0, -0.5)]
    POINTS3 = [(0.5, 2.0, -1.0), (-1.25, 0.75, 0.3), (3.0, -0.5, 2.5), (0.0, 0.0, 0.0)]

    # ------------------------------------------------- arithmetic, masks, control flow
    def test_arithmetic(self):
        eqs = [x + y, x - y, x * y, x / y, -x, x**2, x**3, 1 / x, (x + y) ** 2 / (1 + x * x), x / 2]
        for opt_level in (0, 1, 2, 3):
            self.check([x, y], eqs, self.POINTS, opt_level=opt_level)

    def test_fused_operations(self):
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
        # NaN, infinities, signed zeros and subnormals: the bits of the native scalar kernel,
        # including x86's comparisons (`>` and `>=` are true for NaN operands)
        vals = [0.0, -0.0, 1.0, -1.5, np.inf, -np.inf, np.nan, 5e-324, -2.2e-308, 1e308, 0.5, 2.5, -2.5]
        pts = [(a, b) for a in vals for b in vals]
        eqs = [
            x + y, x - y, x * y, x / y, -x, 1 / x, x**2, x**3, x / 2, x * y + 1.0, x * y - x, sp.sqrt(x),
            sp.Abs(x), sp.floor(x), sp.ceiling(x), sp.Min(x, y), sp.Max(x, y), x**-2,
            sp.Piecewise((x, x > y), (y, True)), sp.Piecewise((x, x <= y), (-y, True)),
            sp.Piecewise((1.0, sp.Eq(x, y)), (2.0, sp.Ne(x, 0)), (3.0, True)),
            sp.Piecewise((1.0, sp.And(x > 0, y < 0)), (2.0, sp.Or(x < y, sp.Not(y >= 1))), (0.0, True)),
        ]
        for opt_level in (0, 2, 3):
            for fastmath in (False, True):
                self.check_bits([x, y], eqs, pts, opt_level=opt_level, fastmath=fastmath)

    def test_loops(self):
        k = sp.Symbol("k")
        eqs = [sp.Sum(x**k / (1 + k), (k, 0, 10)), sp.Product(1 + x / k, (k, 1, 6)), sp.Sum(sp.sin(k * x), (k, 1, 5)) + y]
        for opt_level in (0, 2, 3):
            self.check([x, y], eqs, self.POINTS, opt_level=opt_level, tol=1e-11, ref_eqs=[e.doit() for e in eqs])

    def test_parameters(self):
        self.check([x], [sp.sin(p) * x + p], [(1.0,), (2.0,)], params=[0.75], param_syms=[p])

    def test_many_states(self):
        pts = [tuple(np.linspace(-1, 1, 20) + k) for k in range(3)]
        self.check(list(X), list(reversed(X)), pts)
        self.check(list(X), [sum(v * (i + 1) for i, v in enumerate(X))], pts)

    # ------------------------------------------------- math functions
    def test_math_functions(self):
        u = sp.Symbol("u")
        eqs = [
            sp.sin(x), sp.cos(x), sp.tan(x), sp.csc(x), sp.sec(x), sp.cot(x), sp.sinh(x), sp.cosh(x), sp.tanh(x),
            sp.csch(x), sp.sech(x), sp.coth(x), sp.asin(u), sp.acos(u), sp.atan(x), sp.asinh(x), sp.acosh(y),
            sp.atanh(u), sp.exp(x), sp.log(y), sp.log(y, 10), sp.cbrt(y), sp.erf(x), sp.erfc(x), sp.gamma(y),
            sp.loggamma(y), sp.Si(x), sp.Ci(y), sp.Shi(x), sp.Chi(y), sp.atan2(x, y), sp.sinc(x), y**x,
            sp.exp(sp.sin(x) * y) / (1 + sp.log(y) ** 2),
        ]
        rng = np.random.default_rng(1)
        pts = [(a, b, c) for a, b, c in zip(rng.uniform(-3, 3, 200), rng.uniform(1.01, 6, 200), rng.uniform(-0.99, 0.99, 200))]
        # stable's Cephes loggamma is off by up to 3e-9 near 1 and 2 (src/ptx/libm/README.md)
        rtol = np.full(len(eqs), 1e-13)
        rtol[eqs.index(sp.loggamma(y))] = 1e-8
        for opt_level in (0, 2, 3):
            f = compile_func([x, y, u], eqs, opt_level=opt_level)
            want = np.array([f(*pt) for pt in pts])
            for layout, got in self.gpu(f, pts).items():
                err = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
                bad = np.argwhere(err > rtol)
                self.assertEqual(len(bad), 0, f"{layout}, opt {opt_level}: e.g. {eqs[bad[0][1]]} at "
                                 f"{pts[bad[0][0]]}: {got[tuple(bad[0])]!r} (JIT {want[tuple(bad[0])]!r})"
                                 if len(bad) else "")

    def test_math_special_values(self):
        # the same values as the JIT, but for gamma(±0), loggamma(-inf) and loggamma of
        # subnormals, where musl follows C (±inf, +inf, -log|x|) and stable's Cephes does not (NaN,
        # -inf, inf): see src/ptx/libm/README.md
        vals = [0.0, -0.0, 1.0, -1.0, 2.0, np.inf, -np.inf, np.nan, 1e-310, 1e300, -1e300, 0.5, 710.0]
        eqs = [sp.sin(x), sp.exp(x), sp.log(x), sp.sqrt(x) * sp.atan(x), sp.tanh(x), sp.cosh(x), sp.erfc(x),
               sp.asinh(x), sp.acosh(x), sp.atanh(x), sp.gamma(x), sp.loggamma(x), x**y, sp.atan2(x, y)]
        pts = [(a, b) for a in vals for b in vals]
        f = compile_func([x, y], eqs)
        with np.errstate(all="ignore"):
            want = np.array([f(*pt) for pt in pts])
        gamma, loggamma = eqs.index(sp.gamma(x)), eqs.index(sp.loggamma(x))
        for i, (a, _) in enumerate(pts):
            if a == 0:
                want[i, gamma] = np.copysign(np.inf, a)
            if a == -np.inf:
                want[i, loggamma] = np.inf
            if 0 < abs(a) < 2.2e-308:
                want[i, loggamma] = -np.log(abs(a))
        for layout, got in self.gpu(f, pts).items():
            classes_same = (np.isnan(got) == np.isnan(want)) & (np.isinf(got) == np.isinf(want))
            classes_same &= np.where(np.isinf(want), got == want, True)
            bad = np.argwhere(~classes_same)
            self.assertEqual(len(bad), 0, f"{layout}: e.g. {eqs[bad[0][1]]} at {pts[bad[0][0]]}: "
                             f"{got[tuple(bad[0])]!r} (JIT {want[tuple(bad[0])]!r})" if len(bad) else "")
            finite = np.isfinite(want) & (np.abs(want) > 1e-300)
            np.testing.assert_allclose(got[finite], want[finite], rtol=2e-15, err_msg=layout)

    # ------------------------------------------------- complex models
    def test_complex_arithmetic(self):
        pts = [(0.5 + 0.25j, -1.0 + 2.0j), (1.5 - 0.5j, 0.75 + 0.0j), (-2.0 + 1.0j, 0.3 - 1.2j)]
        eqs = [x + y, x - y, x * y, x / y, x * x * y + 2 * x, 1 / (1 + x * y), sp.conjugate(x) * y, sp.re(x) + sp.im(y)]
        for opt_level in (0, 2):
            for fast_complex in (False, True):
                self.check([x, y], eqs, pts, dtype="complex128", opt_level=opt_level, fast_complex=fast_complex)
                self.check_bits([x, y], eqs, pts, dtype="complex128", opt_level=opt_level, fast_complex=False)

    def test_complex_functions(self):
        rng = np.random.default_rng(2)
        pts = [(a, b) for a, b in zip(rng.uniform(-2, 2, 50) + 1j * rng.uniform(-2, 2, 50),
                                      rng.uniform(-2, 2, 50) + 1j * rng.uniform(-2, 2, 50))]
        eqs = [sp.sin(x), sp.cos(x) * sp.exp(y), sp.sqrt(x * y + 1), x**y, sp.tanh(x) + sp.log(y), sp.sqrt(x) ** 3,
               sp.tan(x), sp.asin(x), sp.acos(y), sp.atan(x), sp.asinh(x), sp.acosh(y), sp.atanh(x), sp.sinh(y),
               sp.cosh(x), sp.cot(y), sp.sec(x), sp.csc(y), sp.cbrt(x)]
        for opt_level in (0, 2):
            self.check([x, y], eqs, pts, dtype="complex128", opt_level=opt_level, tol=1e-11)

    def test_stable_math(self):
        # range-safe complex division, 1/z, abs and sqrt: the bits of the scalar kernel without
        # fast complex (the same complexified operations)
        pts = [(1e308 + 5e-324j, 1 + 0j), (3e200 + 4e200j, 1e200 - 2e200j), (3e-320 + 4e-320j, 1e-320 - 2e-320j),
               (3 + 4j, 1 - 2j), (-0.0 - 0.0j, 2 + 0j), (1e-160 + 1e160j, 1e160 + 1e-160j)]
        eqs = [x / y, 1 / y, sp.Abs(x), sp.sqrt(x)]
        for stable_math in (False, True):
            for opt_level in (0, 2, 3):
                self.check_bits([x, y], eqs, pts, dtype="complex128", stable_math=stable_math,
                                opt_level=opt_level, fast_complex=False)

    # ------------------------------------------------- rejected models, Composer, save/load
    def test_compressed_models_are_rejected(self):
        v = sp.symbols("v0:8")
        eqs = [v[i] * v[(i + 1) % 8] + v[(i + 2) % 8] / (v[(i + 3) % 8] ** 2 + 1) for i in range(8)]
        f = compile_func(list(v), eqs, compress=True)
        with self.assertRaisesRegex(ValueError, "does not support compression mode"):
            write_ptx(f, os.path.join(self.dir.name, "c"), "c")

    def test_user_defined_functions_are_rejected(self):
        ext = sp.Function("ext")
        f = compile_func([x, y], [ext(x) + y], defuns={"ext": lambda a: 2 * a + 1})
        with self.assertRaisesRegex(ValueError, "saved application does not hold"):
            write_ptx(f, os.path.join(self.dir.name, "r"), "r")

    def test_composer(self):
        # the Symbolica calling convention: the kernel's inputs are the parameters
        cp = Composer(2, 2)
        a, b = cp.arg(0), cp.arg(1)
        cp.assign(cp.out(0), cp.fadd(cp.fmul(a, b), cp.constant(1.0)))
        cp.assign(cp.out(1), cp.fdiv(cp.sin(a), cp.fadd(b, cp.constant(2.0))))
        f = compile_composer(cp)
        pts = np.array([[0.5, 2.0], [-1.25, 0.75], [3.0, -1.5]])
        want = np.asarray(f.evaluate(pts))
        for layout, got in self.gpu(f, pts).items():
            np.testing.assert_allclose(got, want, rtol=1e-15, atol=0, err_msg=layout)

    def test_save_and_load(self):
        # a loaded function gives the same module
        f = compile_func([x], [sp.sin(x) * x + 1])
        sjb = os.path.join(self.dir.name, "f.sjb")
        f.save(sjb)
        g = load_func(sjb)
        write_ptx(f, os.path.join(self.dir.name, "a"), "k")
        write_ptx(g, os.path.join(self.dir.name, "b"), "k")
        with open(os.path.join(self.dir.name, "a.ptx")) as a, open(os.path.join(self.dir.name, "b.ptx")) as b:
            self.assertEqual(a.read(), b.read())


# the option combinations every fuzz case is compiled with (SIMD and threads do not apply)
FUZZ_OPTIONS = [
    dict(opt_level=0),
    dict(opt_level=1),
    dict(opt_level=2),
    dict(opt_level=3, cse=False),
    dict(opt_level=2, fastmath=False),
    dict(opt_level=2, compact=False),
    dict(opt_level=2, fast_complex=False, stable_math=True),
]


@unittest.skipUnless(READY, NEEDS)
class PtxFuzz(unittest.TestCase):
    tol = 1e-8

    def run_family(self, family, generator, dtype=float):
        Xv = fz.inputs(fz.NPOINTS, 7, dtype)
        complex_ = dtype == complex
        points = Xv.T.tolist() if not complex_ else as_reals(Xv.T)
        jobs = []

        with tempfile.TemporaryDirectory() as tmp:
            batch = Batch(tmp)
            for seed in range(fz.START, fz.START + fz.SEEDS):
                exprs = fz.build_case(family, generator, seed)
                if exprs is None:
                    continue
                try:
                    ref = fz.reference(exprs, Xv, dtype)
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
                    layout = "aos" if k % 2 else "soa"
                    jobs.append(((seed, options, layout, exprs, ref), batch.add(f, layout)))

            self.assertGreater(len(jobs), 0)
            batch.build()
            for (seed, options, layout, exprs, ref), name in jobs:
                with self.subTest(seed=seed, options=options, layout=layout):
                    obs = batch.run(name, points)
                    got = (as_complex(obs) if complex_ else obs).T
                    want = ref[:, : got.shape[1]]
                    err = np.abs(got - want) / (1 + np.abs(want))
                    self.assertTrue(
                        np.all(err < self.tol),
                        f"{family} seed {seed}, options {options}, {layout}: relative error {err.max():.3g}\n"
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


# Composer programs with the Symbolica calling convention: the inputs are the parameters
COMPOSER_OPTIONS = {
    "straight": [dict(), dict(direct=False), dict(fastmath=False), dict(opt_level=0)],
    "branches": [dict(), dict(direct=False), dict(simd_branch=False), dict(fastmath=False), dict(opt_level=0)],
    "loops": [dict(), dict(simd_branch=False), dict(fastmath=False), dict(opt_level=0)],
}


@unittest.skipUnless(READY, NEEDS)
class PtxComposer(unittest.TestCase):
    tol = 1e-8

    def run_family(self, family):
        Xv = np.random.default_rng(7).uniform(-2, 2, (cf.NPARAMS, cf.NPOINTS))
        points = Xv.T.tolist()
        jobs = []

        with tempfile.TemporaryDirectory() as tmp, np.errstate(all="ignore"):
            batch = Batch(tmp)
            for seed in range(cf.START, cf.START + cf.SEEDS):
                program = cf.Program(seed, family)
                ref = program.reference(Xv)
                if not np.all(np.isfinite(ref)) or np.max(np.abs(ref)) > 1e12:
                    continue
                for k, options in enumerate(COMPOSER_OPTIONS[family]):
                    try:
                        f = compile_composer(program.compose(), **options)
                    except ValueError as e:
                        raise AssertionError(f"composer {family} seed {seed}, options {options}: {e}")
                    layout = "aos" if k % 2 else "soa"
                    jobs.append(((seed, options, layout, ref), batch.add(f, layout)))

            self.assertGreater(len(jobs), 0)
            batch.build()
            for (seed, options, layout, ref), name in jobs:
                with self.subTest(seed=seed, options=options, layout=layout):
                    got = batch.run(name, points).T
                    err = np.abs(got - ref) / (1 + np.abs(ref))
                    self.assertTrue(np.all(err < self.tol), f"composer {family} seed {seed}, options {options}, "
                                    f"{layout}: relative error {err.max():.3g}")

    def test_straight_line(self):
        self.run_family("straight")

    def test_branches(self):
        self.run_family("branches")

    def test_loops(self):
        self.run_family("loops")


@unittest.skipUnless(READY, NEEDS)
@unittest.skipUnless(HAVE_SYMBOLICA, "needs symbolica")
class PtxSymbolica(unittest.TestCase):
    EXPRS = [
        "x^2 + 3*x*y - sin(y)/(1 + x^2)",
        "if(y, x + 1, x + 2)",
        "if(x - y, exp(x) * cos(y), sqrt(1 + x^2))",
        "(x + y)^5 - (x - y)^3 + x*y*cos(x*y)",
        "log(1 + x^2) * exp(-y^2) + x^y",
    ]

    def test_evaluators(self):
        from symbolica import E, S

        Xv = np.random.default_rng(3).uniform(0.1, 1, (40, 2))
        Xv[::3, 1] = 0.0  # take both branches of if()
        with tempfile.TemporaryDirectory() as tmp:
            batch = Batch(tmp)
            jobs = []
            for text in self.EXPRS:
                ev = E(text).evaluator([S("x"), S("y")])
                ref = np.asarray(ev.evaluate(Xv))
                for simd_branch in (False, True):
                    f = compile_evaluator(ev, simd_branch=simd_branch)
                    for layout in ("soa", "aos"):
                        jobs.append(((text, simd_branch, layout, ref), batch.add(f, layout)))
            batch.build()
            for (text, simd_branch, layout, ref), name in jobs:
                with self.subTest(expr=text, simd_branch=simd_branch, layout=layout):
                    np.testing.assert_allclose(batch.run(name, Xv), ref, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
