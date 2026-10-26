"""What the Python tests of symjit-utils share: the symjit package (`$SYMJIT_PYTHON_PATH`, else
`stable/python` of this checkout) and its test helpers (`stable/python/tests`), the utility binaries
(`$SJB2WASM`/`$SJB2OBJ`/`$SJB2PTX`, else `target/release/sjb2wasm`/`sjb2obj`/`sjb2ptx`; build them with
`cargo build --release` in `symjit-utils/`), and the outputs of saved functions compiled by them."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
STABLE = os.path.normpath(os.path.join(UTILS_ROOT, "..", "stable"))

sys.path.insert(0, os.environ.get("SYMJIT_PYTHON_PATH", os.path.join(STABLE, "python")))
# test_fuzz and test_composer_fuzz of the symjit package
sys.path.insert(0, os.path.join(STABLE, "python", "tests"))
sys.path.insert(0, HERE)

SJB2WASM = os.environ.get("SJB2WASM", os.path.join(UTILS_ROOT, "target", "release", "sjb2wasm"))
HAVE_SJB2WASM = os.path.exists(SJB2WASM)
SJB2OBJ = os.environ.get("SJB2OBJ", os.path.join(UTILS_ROOT, "target", "release", "sjb2obj"))
HAVE_SJB2OBJ = os.path.exists(SJB2OBJ)
SJB2PTX = os.environ.get("SJB2PTX", os.path.join(UTILS_ROOT, "target", "release", "sjb2ptx"))
HAVE_SJB2PTX = os.path.exists(SJB2PTX)


def save(f, path, **save_args):
    """`f.save(path)`; functions of the Symbolica bridge (Composer, evaluators) save the real
    kernel unless `dtype` says otherwise."""
    if hasattr(f, "complex_compiler") and "dtype" not in save_args:
        save_args["dtype"] = "complex128" if f.is_complex else "float64"
    if not hasattr(f, "save"):  # stable's `compile_ode` functions save through their compiler
        f.compiler.save(path)
        return
    f.save(path, **save_args)


def run_tool(tool, *args):
    res = subprocess.run([tool, *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise ValueError(res.stderr.strip().split("error: ", 1)[-1])


def write_wasm(f, path, **save_args):
    """Writes the WebAssembly module `path` of `f` (saved next to it as `path`.sjb)."""
    sjb = path + ".sjb"
    save(f, sjb, **save_args)
    run_tool(SJB2WASM, sjb, "-o", path)


def write_obj(f, path, format=None, **save_args):
    """Writes the object file `path`.o and its C header `path`.h (from `path`.sjb), with the
    kernels named after the last component of `path`."""
    sjb = path + ".sjb"
    save(f, sjb, **save_args)
    args = ["-o", path + ".o", "--name", os.path.basename(path)]
    if format is not None:
        args += ["--format", format]
    run_tool(SJB2OBJ, sjb, *args)


def write_ptx(f, path, name, layout="soa", **save_args):
    """Writes the PTX module `path`.ptx of `f` (from `path`.sjb) with kernel `name`, and its C
    driver `name`_cuda.c/.h next to it."""
    sjb = path + ".sjb"
    save(f, sjb, **save_args)
    run_tool(SJB2PTX, sjb, "-o", path + ".ptx", "--name", name, "--layout", layout, "--no-header",
             "--driver")
