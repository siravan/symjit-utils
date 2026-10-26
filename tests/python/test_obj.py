"""Tests of sjb2obj: functions compiled by symjit (stable's Python package) are saved (`f.save`),
compiled by sjb2obj into relocatable objects with C headers (`sjb_util.write_obj`), linked into a
generated C program with `cc ... -lm`, run, and compared with the JIT-compiled functions (the
kernels come from copies of the JIT's generators, so results must be bit-identical, except for
C99's complex functions).

`ObjFuzz` does this for the random expressions of test_fuzz.py (one C program per
family); SYMJIT_FUZZ_SEEDS and SYMJIT_FUZZ_START work as there.

They run on Linux and macOS, x86-64 and ARM64 (the objects are in the host's format and
architecture); `ObjCross` checks AArch64 objects (`ty="arm"`) on any host. Skipped without
the sjb2obj binary (see sjb_util.py) or a C compiler (`cc`). Saved functions cannot hold
user-defined functions or callbacks, so objects do not call them.
"""

import importlib.util
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sjb_util import HAVE_SJB2OBJ, write_obj

from symjit import compile_composer, compile_func, compile_ode

import test_fuzz as fz

x, y, z, p, t = sp.symbols("x y z p t")


HERE = os.path.dirname(os.path.abspath(__file__))
HAVE_OBJ = HAVE_SJB2OBJ
HOST_OK = (
    (sys.platform.startswith("linux") or sys.platform == "darwin")
    and platform.machine() in ("x86_64", "AMD64", "aarch64", "arm64")
    and shutil.which("cc") is not None
)
HOST_ARCH = "AArch64" if platform.machine() in ("aarch64", "arm64") else "x86-64"
HAVE_SYMBOLICA = importlib.util.find_spec("symbolica") is not None
NEEDS = "needs sjb2obj (cargo build --release in symjit-utils/, or $SJB2OBJ), Linux or macOS, and a C compiler"


def c_double(v):
    v = float(v)
    if np.isnan(v):
        return "NAN"
    if np.isinf(v):
        return "INFINITY" if v > 0 else "-INFINITY"
    return repr(v)


class Model:
    """A compiled function written as `<dir>/<name>.o/.h`, with its layout from the header."""

    def __init__(self, f, name, directory, symbolica=False, **save_args):
        self.name = name
        self.symbolica = symbolica
        write_obj(f, os.path.join(directory, name), **save_args)
        with open(os.path.join(directory, name + ".h")) as fd:
            header = fd.read()
        macros = dict(re.findall(rf"#define {name.upper()}_(\w+) (\d+)", header))
        self.cs = int(macros["COUNT_STATES"])
        self.co = int(macros["COUNT_OBS"])
        self.cp = int(macros["COUNT_PARAMS"])
        self.mem = int(macros["MEM_SIZE"])
        self.fast = f"double {name}_fast(" in header
        # calls C99's complex functions (whose results can differ from the JIT's in the last bits)
        self.c99 = re.search(r"\bc(sin|cos|tan|sinh|cosh|tanh|asin|acos|atan|asinh|acosh|atanh|exp|log|pow|arg|sqrt)\b", header) is not None


def run_c(directory, models, points, params=(), arm=False):
    """Links one C program with all `models` and evaluates each at every point (states,
    or inputs for Composer/Symbolica models) in direct mode, and with `_fast` if there is
    one. Returns {name: [(status, outputs, fast output or None), ...]}."""
    width = max(len(pt) for pt in points)
    lines = ["#include <math.h>", "#include <stdio.h>", "#include <string.h>"]
    lines += [f'#include "{m.name}.h"' for m in models]
    rows = ", ".join("{ " + ", ".join(c_double(v) for v in pt) + " }" for pt in points)
    lines.append(f"static const double P[{len(points)}][{width}] = {{ {rows} }};")
    lines.append(f"static const double sj_params_[{len(params) + 1}] = {{ {', '.join(map(c_double, list(params) + [0]))} }};")
    lines.append(f"static double mem[{max(m.mem for m in models) + 64}];")
    lines.append("int main(void) {")
    lines.append("    (void)sj_params_;")
    for m in models:
        lines.append(f"    for (int k = 0; k < {len(points)}; k++) {{")
        lines.append("        memset(mem, 0, sizeof(mem));")
        if m.symbolica:  # inputs are the parameters, outputs go to mem
            lines.append(f'        printf("{m.name} %d", {m.name}(mem, NULL, 0, P[k]));')
            first = 0
        else:
            lines.append(f"        memcpy(mem, P[k], {m.cs} * sizeof(double));")
            lines.append(f'        printf("{m.name} %d", {m.name}(mem, NULL, 0, sj_params_));')
            first = m.cs
        lines.append(f'        for (int i = 0; i < {m.co}; i++) printf(" %a", mem[{first} + i]);')
        if m.fast:
            args = ", ".join(f"P[k][{i}]" for i in range(m.cs))
            lines.append(f'        printf(" %a", {m.name}_fast({args}));')
        lines.append('        printf("\\n");')
        lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")

    src = os.path.join(directory, "main.c")
    with open(src, "w") as fd:
        fd.write("\n".join(lines) + "\n")
    exe = os.path.join(directory, "main")
    objs = [os.path.join(directory, m.name + ".o") for m in models]
    # AArch64 objects (ty="arm") are linked statically and run under qemu
    compiler = ["aarch64-linux-gnu-gcc", "-static"] if arm else ["cc"]
    cc = subprocess.run([*compiler, "-Wall", "-Werror", src, *objs, "-lm", "-o", exe], capture_output=True, text=True)
    if cc.returncode != 0 or cc.stderr:
        raise AssertionError(f"cc failed:\n{cc.stderr[:3000]}")
    run = subprocess.run((["qemu-aarch64", exe] if arm else [exe]), capture_output=True, text=True, timeout=600)
    if run.returncode != 0:
        raise AssertionError(f"the C program failed: {run.stderr}")

    by_model = {m.name: m for m in models}
    out = {m.name: [] for m in models}
    for line in run.stdout.splitlines():
        words = line.split()
        m = by_model[words[0]]
        vals = [float.fromhex(w) for w in words[2:]]
        out[m.name].append((int(words[1]), vals[: m.co], vals[m.co] if m.fast else None))
    return out


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


@unittest.skipUnless(HAVE_OBJ and HOST_OK, NEEDS)
class Obj(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def check(self, f, name, points, params=(), complex_=False):
        m = Model(f, name, self.d)
        res = run_c(self.d, [m], [list(pt) for pt in points], params)[name]
        for pt, (status, obs, fast) in zip(points, res):
            self.assertEqual(status, 0)
            if complex_:
                args = [complex(pt[i], pt[i + 1]) for i in range(0, len(pt), 2)]
                want = np.array(f(*args), dtype=complex)
                want = np.ravel(np.column_stack([want.real, want.imag]))
            else:
                want = np.ravel(f(*pt, *params))
            self.assertTrue(same(obs, want), f"{name} at {pt}: object {obs}, JIT {list(want)}")
            if fast is not None:
                self.assertTrue(same(fast, want[0]), f"{name}_fast at {pt}: {fast} vs {want[0]}")
        return m

    POINTS = [(0.5, 2.0), (-1.25, 0.75), (3.0, -0.5), (0.0, 0.0), (2.0, 1e-300)]

    def test_pi(self):
        # known values rather than the JIT: pi by Machin's formula (arctan as a Taylor
        # series, and as libm's atan) and by the BBP series (a loop calling pow)
        n, k = sp.symbols("n k")

        def arctan_series(a, terms=21):
            return sum((-1) ** i * a ** (2 * i + 1) / (2 * i + 1) for i in range(terms))

        machin = [compile_func([x, y], [4 * (4 * f(x) - f(y))]) for f in (arctan_series, sp.atan)]
        bbp = compile_func(
            [n], [sp.Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k, (k, 0, n))]
        )
        models = [Model(machin[0], "pi_series", self.d), Model(machin[1], "pi_atan", self.d)]
        res = run_c(self.d, models, [[1 / 5, 1 / 239]])
        ulp = math.ulp(math.pi)
        for m in models:
            [(status, [v], fast)] = res[m.name]
            self.assertEqual(status, 0)
            self.assertLessEqual(abs(v - math.pi), ulp, m.name)
            self.assertEqual(fast, v)

        res = run_c(self.d, [Model(bbp, "pi_bbp", self.d)], [[float(t)] for t in range(13)])["pi_bbp"]
        errors = [abs(obs[0] - math.pi) for _, obs, _ in res]
        self.assertTrue(all(e1 <= e0 for e0, e1 in zip(errors, errors[1:])))  # converges
        self.assertEqual(errors[10:], [0.0, 0.0, 0.0])  # exact from 10 terms on
        self.assertGreater(errors[0], 0.008)  # 47/15 with one term

    def test_functions_and_branches(self):
        eqs = [
            sp.sin(x) * y + sp.exp(-x * x), sp.log(1 + x * x), sp.atan2(y, x), sp.Abs(x) ** y,
            sp.Piecewise((sp.cos(x), x < y), (y**3, True)), sp.Min(x, y) + sp.Max(x, sp.tanh(y)),
            sp.sqrt(x * x + y * y), sp.floor(x) + sp.ceiling(y),
        ]
        for opt_level in (0, 1, 2, 3):
            m = self.check(compile_func([x, y], eqs, opt_level=opt_level), f"model{opt_level}", self.POINTS)
            self.assertFalse(m.fast)

    def test_fast_kernel(self):
        m = self.check(compile_func([x, y], [sp.sin(x) * y + x / (1 + y * y)]), "fastmodel", self.POINTS)
        self.assertTrue(m.fast)

    def test_parameters(self):
        f = compile_func([x], [x * p + sp.sin(p), sp.exp(x) - p], params=[p])
        self.check(f, "params", [(0.5,), (-2.0,)], params=[0.75])

    def test_complex(self):
        f = compile_func([x, y], [x * y + x, x / (1 + y * y), sp.sqrt(x) * y], dtype="complex128")
        pts = [(0.5, 0.25, -1.0, 2.0), (1.5, -0.5, 0.75, 0.0), (-2.0, 1.0, 0.3, -1.2)]
        self.check(f, "cplx", pts, complex_=True)

    def test_ode(self):
        f = compile_ode(t, [x, y], [y * sp.cos(t), -sp.sin(x) - 0.1 * y])
        m = Model(f, "pendulum", self.d)
        self.assertEqual((m.cs, m.co), (3, 0))  # t, x, y; the derivatives follow in mem
        lines = [
            '#include <stdio.h>', '#include "pendulum.h"', "int main(void) {",
            "    double mem[PENDULUM_MEM_SIZE] = {0.25, 1.0, -0.5};",
            "    int s = pendulum(mem, NULL, 0, NULL);",
            '    printf("%d %a %a\\n", s, mem[PENDULUM_COUNT_STATES], mem[PENDULUM_COUNT_STATES + 1]);',
            "    return 0;", "}",
        ]
        with open(os.path.join(self.d, "ode.c"), "w") as fd:
            fd.write("\n".join(lines) + "\n")
        exe = os.path.join(self.d, "ode")
        subprocess.run(["cc", "-Wall", "-Werror", os.path.join(self.d, "ode.c"), os.path.join(self.d, "pendulum.o"),
                        "-lm", "-o", exe], check=True)
        words = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(words[0], "0")
        self.assertTrue(same([float.fromhex(w) for w in words[1:]], f(0.25, [1.0, -0.5])))

    def test_indirect_mode(self):
        f = compile_func([x, y], [sp.sin(x) * y, x - y * y])
        Model(f, "vec", self.d)
        n = 101
        X, Y = np.linspace(-2, 2, n), np.cos(np.linspace(0, 5, n))
        lines = [
            "#include <stdio.h>", '#include "vec.h"',
            f"static double xs[{n}], ys[{n}], o0[{n}], o1[{n}];", "int main(void) {",
            f"    for (int i = 0; i < {n}; i++) if (scanf(\"%lf %lf\", &xs[i], &ys[i]) != 2) return 2;",
            f"    symjit_slice arrays[VEC_COUNT_STATES + VEC_COUNT_OBS] = {{ {{xs, {n}}}, {{ys, {n}}}, {{o0, {n}}}, {{o1, {n}}} }};",
            f"    for (size_t i = 0; i < {n}; i++) if (vec(NULL, arrays, i, NULL)) return 1;",
            f'    for (int i = 0; i < {n}; i++) printf("%a %a\\n", o0[i], o1[i]);',
            "    return 0;", "}",
        ]
        with open(os.path.join(self.d, "v.c"), "w") as fd:
            fd.write("\n".join(lines) + "\n")
        exe = os.path.join(self.d, "v")
        subprocess.run(["cc", "-Wall", "-Werror", os.path.join(self.d, "v.c"), os.path.join(self.d, "vec.o"),
                        "-lm", "-o", exe], check=True)
        stdin = "\n".join(f"{float(a)!r} {float(b)!r}" for a, b in zip(X, Y))
        out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
        got = np.array([[float.fromhex(w) for w in line.split()] for line in out.splitlines()]).T
        # the scalar kernel, as in the object (vectorized calls use the SIMD kernel)
        want = np.array([np.ravel(f(a, b)) for a, b in zip(X, Y)]).T
        self.assertTrue(same(got, want))

    def test_composer_with_branches_and_loops(self):
        import test_composer_fuzz as cf

        programs = []
        for family in ("branches", "loops"):
            for seed in range(3):
                programs.append((f"{family}{seed}", cf.Program(seed, family).compose()))
        models, natives = [], {}
        for name, cp in programs:
            models.append(Model(compile_composer(cp), name, self.d, symbolica=True, dtype="float64"))
            natives[name] = compile_composer(cp)
        pts = [[float(k), 0.5 - k / 7, 1.0 + k / 3] for k in range(12)]
        res = run_c(self.d, models, pts)
        for m in models:
            for pt, (status, obs, _) in zip(pts, res[m.name]):
                args = pt[: m.cp]
                want = np.ravel(natives[m.name](*args))
                self.assertEqual(status, 0)
                self.assertTrue(same(obs, want), f"{m.name} at {args}: {obs} vs {list(want)}")

    @unittest.skipUnless(HAVE_SYMBOLICA, "needs symbolica")
    def test_symbolica_evaluator(self):
        from symbolica import E, S
        from symjit import compile_evaluator

        ev = E("if(y, x^2 + sin(x), cos(x) * y) + exp(x*y)").evaluator([S("x"), S("y")])
        f = compile_evaluator(ev)
        m = Model(f, "symb", self.d, symbolica=True, dtype="float64")
        pts = [[0.5, 0.0], [0.5, 2.0], [-1.25, 0.75]]
        res = run_c(self.d, [m], pts)["symb"]
        want = np.asarray(ev.evaluate(np.array(pts)))
        for (status, obs, _), w in zip(res, want):
            self.assertEqual(status, 0)
            np.testing.assert_allclose(obs, np.ravel(w), rtol=1e-15)

    def test_stable_math(self):
        # range-safe complex division, 1/z, abs, sqrt and negative powers: the copies of the
        # generators and of the complexifier compute the JIT's bits, through the packed-complex
        # generator (fast_complex) and through `complexify` (the scalar generator), also when
        # the model calls a complex function (then compiled through the object complexifier)
        pts = [(1e308, 5e-324, 1.0, 0.0), (3e200, 4e200, 1e200, -2e200), (3e-320, 4e-320, 1e-320, -2e-320),
               (3.0, 4.0, 1.0, -2.0), (-0.0, -0.0, 2.0, 0.0), (1e-160, 1e160, 1e160, 1e-160)]
        eqs = [x / y, 1 / y, sp.Abs(x), sp.sqrt(x), y**-2, x * y**-3]
        for k, (fast_complex, extra) in enumerate([(True, []), (False, []), (True, [sp.sin(x / y)])]):
            for opt_level in (0, 2, 3):
                kw = dict(dtype="complex128", stable_math=True, opt_level=opt_level)
                f = compile_func([x, y], eqs + extra, fast_complex=fast_complex, **kw)
                # a model calling complex functions is compiled as with fast_complex=False
                jit = compile_func([x, y], eqs + extra, fast_complex=fast_complex and not extra, **kw)
                m = Model(f, f"stable{k}_{opt_level}", self.d)
                for pt, (status, obs, _) in zip(pts, run_c(self.d, [m], [list(p) for p in pts])[m.name]):
                    self.assertEqual(status, 0)
                    want = np.array(jit(complex(pt[0], pt[1]), complex(pt[2], pt[3])), dtype=complex)
                    if extra:  # C99's csin, not the JIT's
                        got = np.array(obs[0::2]) + 1j * np.array(obs[1::2])
                        self.assertTrue(close_complex(got[-1:], want[-1:], 1e-13))
                        obs, want = obs[:-2], want[:-1]
                    want = np.ravel(np.column_stack([want.real, want.imag]))
                    self.assertTrue(same(obs, want), f"{m.name} at {pt}: object {obs}, JIT {list(want)}")

    def test_compressed(self):
        # compression-mode funclets (a shape of four variables that occurs more than twice)
        v = sp.symbols("v0:8")
        eqs = [v[i] * v[(i + 1) % 8] + v[(i + 2) % 8] / (v[(i + 3) % 8] ** 2 + 1) for i in range(8)]
        pts = [tuple(np.linspace(-1, 1, 8) * k) for k in (1, 2, 3)]
        for opt_level in (0, 3):
            f = compile_func(list(v), [e + sp.sin(e) for e in eqs], compress=True, opt_level=opt_level)
            self.check(f, f"compressed{opt_level}", pts)
            g = compile_func(list(v), eqs, compress=True, opt_level=opt_level, dtype="complex128")
            self.check(g, f"compressed_complex{opt_level}", [tuple(c for a in pt for c in (a, 0.25)) for pt in pts],
                       complex_=True)

    def test_errors(self):
        path = os.path.join(self.d, "e")
        # a saved function does not hold user-defined functions
        f = compile_func([x], [sp.Function("g")(x)], defuns={"g": lambda a: 2 * a})
        with self.assertRaisesRegex(ValueError, "saved application does not hold"):
            write_obj(f, path)
        f = compile_func([x], [x])
        with self.assertRaisesRegex(ValueError, "format"):
            write_obj(f, path, format="coff")
        with self.assertRaisesRegex(ValueError, "C identifier"):
            write_obj(f, os.path.join(self.d, "not-an-identifier"))

    def test_macho_format(self):
        f = compile_func([x], [sp.sin(x)])
        write_obj(f, os.path.join(self.d, "mac"), format="macho")
        with open(os.path.join(self.d, "mac.o"), "rb") as fd:
            self.assertEqual(fd.read(4), bytes.fromhex("cffaedfe"))
        with open(os.path.join(self.d, "mac.h")) as fd:
            self.assertIn(f"{HOST_ARCH} Mach-O", fd.read())


# the real functions the C library does not have (composed of C functions in the object), and
# the complex functions (C99's, from the C library)
REAL_EXTRA = [sp.csc, sp.sec, sp.cot, sp.csch, sp.sech, sp.coth, sp.sinc]
COMPLEX_FUNCS = [
    sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh, sp.asin, sp.acos, sp.atan, sp.asinh, sp.acosh,
    sp.atanh, sp.exp, sp.log, sp.sqrt, sp.csc, sp.sec, sp.cot, sp.csch, sp.sech, sp.coth, sp.cbrt, sp.sinc,
    lambda z: sp.log(z, 10), sp.arg,
]
COMPLEX_POINTS = [(0.5, 0.25), (-1.25, 0.75), (-2.0, 0.0), (3.0, -2.0), (0.1, -1.5)]
HAVE_QEMU = shutil.which("qemu-aarch64") is not None and shutil.which("aarch64-linux-gnu-gcc") is not None


def close_complex(got, want, rtol):
    """Every component within rtol of |want| (both NaN, or the same infinity, also pass)."""
    for g, w in zip(got, want):
        for a, b in ((g.real, w.real), (g.imag, w.imag)):
            if np.isnan(a) or np.isnan(b) or np.isinf(a) or np.isinf(b):
                if not ((np.isnan(a) and np.isnan(b)) or a == b):
                    return False
            elif abs(a - b) > rtol * max(abs(w), 1e-300):
                return False
    return True


@unittest.skipUnless(HAVE_OBJ and HOST_OK, NEEDS)
class ObjMathFunctions(unittest.TestCase):
    """Functions the C math library does not have: symjit's real ones (`csc`, ..., `sinc`) are
    composed of C functions as symjit computes them (bit for bit), and the complex ones call
    C99's (`csin`, ...), which can differ from the JIT's (`num_complex`) in the last bits."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.d = self.dir.name

    def tearDown(self):
        self.dir.cleanup()

    def real(self, arm=False):
        ty = "arm" if arm else "native"
        f = compile_func([x], [g(x) for g in REAL_EXTRA], ty=ty)
        jit = compile_func([x], [g(x) for g in REAL_EXTRA])
        m = Model(f, "real", self.d)
        pts = [[0.5], [-1.3], [0.0], [2.0], [1e-300], [-0.0]]
        for pt, (status, obs, _) in zip(pts, run_c(self.d, [m], pts, arm=arm)["real"]):
            self.assertEqual(status, 0)
            want = np.ravel(jit(*pt))
            if arm:
                np.testing.assert_allclose(obs, want, rtol=1e-14, err_msg=f"at {pt}")
            else:
                self.assertTrue(same(obs, want), f"at {pt}: object {obs}, JIT {list(want)}")

    def complex(self, arm=False):
        ty = "arm" if arm else "native"
        exprs = [g(x) for g in COMPLEX_FUNCS] + [x**y]
        f = compile_func([x, y], exprs, dtype="complex128", ty=ty)
        jit = compile_func([x, y], exprs, dtype="complex128")
        m = Model(f, "cplx", self.d)
        pts = [[a, b, 0.7, -0.2] for a, b in COMPLEX_POINTS]
        for pt, (status, obs, _) in zip(pts, run_c(self.d, [m], pts, arm=arm)["cplx"]):
            self.assertEqual(status, 0)
            got = np.array(obs[0::2]) + 1j * np.array(obs[1::2])
            want = np.array(jit(complex(pt[0], pt[1]), complex(pt[2], pt[3])), dtype=complex)
            self.assertTrue(close_complex(got, want, 1e-13), f"at {pt}: object {list(got)}, JIT {list(want)}")

    def test_real_functions_c_lacks(self):
        self.real()

    def test_complex_functions(self):
        self.complex()
        with open(os.path.join(self.d, "cplx.h")) as fd:
            self.assertIn("csin", fd.read())

    def test_sinc_at_zero(self):
        f = compile_func([x], [sp.sinc(x)], dtype="complex128")
        m = Model(f, "sinc0", self.d)
        (status, obs, _), = run_c(self.d, [m], [[0.0, 0.0]])["sinc0"]
        self.assertEqual((status, obs), (0, [1.0, 0.0]))

    @unittest.skipUnless(HAVE_QEMU, "needs qemu-aarch64 and aarch64-linux-gnu-gcc")
    def test_arm64_objects_run(self):
        self.real(arm=True)
        self.complex(arm=True)

    @unittest.skipUnless(HAVE_QEMU, "needs qemu-aarch64 and aarch64-linux-gnu-gcc")
    def test_arm64_stable_math(self):
        # the AArch64 generators' range-safe division, abs and sqrt (their formulas differ
        # from x86-64's in the last bit: compared with this machine's JIT within rounding)
        pts = [[1e308, 5e-324, 1.0, 0.0], [3e200, 4e200, 1e200, -2e200], [3e-320, 4e-320, 1e-320, -2e-320],
               [3.0, 4.0, 1.0, -2.0], [1e-160, 1e160, 1e160, 1e-160]]
        eqs = [x / y, 1 / y, sp.Abs(x), sp.sqrt(x), y**-2]
        for fast_complex in (True, False):
            kw = dict(dtype="complex128", stable_math=True, fast_complex=fast_complex)
            f = compile_func([x, y], eqs, ty="arm", **kw)
            jit = compile_func([x, y], eqs, **kw)
            m = Model(f, f"stable_arm{int(fast_complex)}", self.d)
            for pt, (status, obs, _) in zip(pts, run_c(self.d, [m], pts, arm=True)[m.name]):
                self.assertEqual(status, 0)
                want = np.array(jit(complex(pt[0], pt[1]), complex(pt[2], pt[3])), dtype=complex)
                want = np.ravel(np.column_stack([want.real, want.imag]))
                # component by component (re + 1j * im would turn an infinite im into a NaN re)
                self.assertTrue(close_complex(np.array(obs, dtype=complex), want.astype(complex), 1e-15),
                                f"at {pt}: object {obs}, JIT {list(want)}")


@unittest.skipUnless(HAVE_OBJ, "needs the sjb2obj binary")
class ObjCross(unittest.TestCase):
    """AArch64 objects compiled with ty="arm" on any host (checked, not run)."""

    def test_arm64_objects(self):
        n, k = sp.symbols("n k")
        f = compile_func([x, y], [4 * (4 * sp.atan(x) - sp.atan(y))], ty="arm")
        bbp = compile_func(
            [n], [sp.Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k, (k, 0, n))],
            ty="arm",
        )
        with tempfile.TemporaryDirectory() as d:
            for g, name in ((f, "pi_atan"), (bbp, "pi_bbp")):
                write_obj(g, os.path.join(d, name), format="macho")
                with open(os.path.join(d, name + ".o"), "rb") as fd:
                    b = fd.read()
                self.assertEqual(b[:4], bytes.fromhex("cffaedfe"))
                self.assertEqual(int.from_bytes(b[4:8], "little"), 0x0100000C)  # CPU_TYPE_ARM64
                with open(os.path.join(d, name + ".h")) as fd:
                    header = fd.read()
                self.assertIn("AArch64 Mach-O", header)
                self.assertIn(f"double {name}_fast(double x0", header)

            write_obj(f, os.path.join(d, "elf"), format="elf")
            with open(os.path.join(d, "elf.o"), "rb") as fd:
                b = fd.read()
            self.assertEqual(b[:4], b"\x7fELF")
            self.assertEqual(int.from_bytes(b[18:20], "little"), 183)  # EM_AARCH64


@unittest.skipUnless(HAVE_OBJ and HAVE_QEMU, "needs the sjb2obj binary, qemu-aarch64 and aarch64-linux-gnu-gcc")
class ObjPiAarch64(unittest.TestCase):
    """The pi kernels as AArch64 ELF objects (ty="arm"), linked with test_pi.c (sjc's
    examples/obj) by the cross compiler and run under qemu."""

    def test_pi(self):
        n, k = sp.symbols("n k")

        def arctan_series(v):  # as examples/obj/pi.py
            s = v
            for i in range(1, 21):
                coef = -(1 + 2 * i) if i & 1 == 1 else 1 + 2 * i
                s += v ** abs(coef) / coef
            return s

        functions = {
            "pi_series": ([x, y], [4 * (4 * arctan_series(x) - arctan_series(y))]),
            "pi_atan": ([x, y], [4 * (4 * sp.atan(x) - sp.atan(y))]),
            "pi_bbp": ([n], [sp.Sum((4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6)) / 16**k,
                                    (k, 0, n))]),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, (states, exprs) in functions.items():
                write_obj(compile_func(states, exprs, ty="arm"), os.path.join(d, name), format="elf")
            exe = os.path.join(d, "test_pi")
            src = os.path.join(HERE, "test_pi.c")
            subprocess.run(["aarch64-linux-gnu-gcc", "-static", "-Wall", "-O2", "-I", d, src,
                            *[os.path.join(d, f"{name}.o") for name in functions], "-lm", "-o", exe], check=True)
            p = subprocess.run(["qemu-aarch64", exe], capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("ok!", p.stdout)


FUZZ_OPTIONS = [dict(opt_level=0), dict(opt_level=2), dict(opt_level=3, cse=False), dict(opt_level=2, fastmath=False)]


@unittest.skipUnless(HAVE_OBJ and HOST_OK, NEEDS)
class ObjFuzz(unittest.TestCase):
    def run_family(self, family, generator, dtype=float):
        X = fz.inputs(fz.NPOINTS, 7, dtype)
        complex_ = dtype == complex
        pts = [list(np.ravel(np.column_stack([c.real, c.imag]))) if complex_ else list(c) for c in X.T]
        models, funcs = [], {}

        with tempfile.TemporaryDirectory() as d, warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            for seed in range(fz.START, fz.START + fz.SEEDS):
                exprs = fz.build_case(family, generator, seed)
                if exprs is None:
                    continue
                for k, options in enumerate(FUZZ_OPTIONS):
                    f = compile_func(fz.V, exprs, dtype="complex128" if complex_ else "float64", **options)
                    name = f"{family}_{seed}_{k}"
                    try:
                        models.append(Model(f, name, d))
                    except ValueError as e:
                        raise AssertionError(f"{family} seed {seed} {options}: {e}\n{exprs}")
                    funcs[name] = (f, seed, options, exprs)

            self.assertGreater(len(models), 0)
            res = run_c(d, models, pts)

            for m in models:
                f, seed, options, exprs = funcs[m.name]
                for c, (status, obs, fast) in zip(X.T, res[m.name]):
                    want = np.array(f(*c), dtype=dtype)
                    if complex_:
                        want = np.column_stack([want.real, want.imag])
                    want = np.ravel(want)
                    self.assertEqual(status, 0)
                    if m.c99:
                        # C99's complex functions, not the JIT's: within rounding
                        ok = close_complex(np.array(obs[0::2]) + 1j * np.array(obs[1::2]),
                                           want[0::2] + 1j * want[1::2], 1e-12)
                        if not ok:
                            self.fail(f"{family} seed {seed} {options} at {c}:\nobject {obs}\n"
                                      f"JIT {list(want)}\nexpressions: {exprs}")
                        continue
                    if not same(obs, want) or (fast is not None and not same(fast, want[0])):
                        self.fail(f"{family} seed {seed} {options} at {c}:\nobject {obs} fast {fast}\n"
                                  f"JIT {list(want)}\nexpressions: {exprs}")
        return len(models)

    def test_real(self):
        self.run_family("real", fz.case_real)

    def test_shared(self):
        self.run_family("shared", fz.case_shared)

    def test_piecewise(self):
        self.run_family("piecewise", fz.case_piecewise)

    def test_complex(self):
        self.run_family("complex", fz.case_complex, complex)


if __name__ == "__main__":
    unittest.main()
