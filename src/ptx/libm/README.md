# The math library of sjb2ptx

A GPU kernel cannot call the CPU's math library, so sjb2ptx brings its own: C compiled to PTX,
checked in as `libm.ptx`. `src/ptx/libm.rs` splits it into its functions and tables, and a module
includes only those its kernel calls, with their dependencies.

| Path | What |
|---|---|
| `musl/` | Files of musl 1.2.5 (MIT, `musl/COPYRIGHT`), unmodified: the real functions of `src/math/` that symjit needs and their helpers, `src/internal/libm.h`, `include/math.h`, `arch/generic/fp_arch.h`. |
| `compat/` | `features.h`, `endian.h`, `bits/alltypes.h` in place of musl's, for a freestanding build without symbol aliases (PTX has none). |
| `sj/symjit_math.c` | symjit's real functions (`__sj_<op>`, stable's `rust/code.rs` names: `ln` is `log`, `log` is `log10`, `gamma` is `tgamma`, ...) on musl's. |
| `sj/symjit_complex.c` | symjit's complex functions (`__sj_cplx_<op>`): num_complex 0.4.6's formulas, ported operation by operation, and code.rs's `recip`, `tanh_c`, `tan_c`. |
| `sj/cephes.c` | Cephes' Si, Ci, Shi, Chi, ported line by line from the Rust port that stable calls (spec_math 0.1.6, MIT or Apache-2.0); musl has no such functions. |
| `sj/overrides.c` | `sqrt`, `fabs` and (GPU) `floor` as single instructions. |
| `gen.sh`, `post.py` | The build: writes `libm.ptx`, or the same functions as an x86-64 object for the tests. |
| `libm.ptx` | Generated; do not edit. |

Sources: musl from `https://musl.libc.org/releases/musl-1.2.5.tar.gz` (sha256
`a9a118bbe84d8764da0ea0d28b3ab3fae8477fc7e4085d90102b8596fc7c75e4`, downloaded 2026-10-18); num_complex
0.4.6 and spec_math 0.1.6 from the cargo registry (the versions stable builds with).

## Regenerating

```bash
src/ptx/libm/gen.sh                    # rewrites libm.ptx
src/ptx/libm/gen.sh --host libsj.o     # the same C for x86-64 (tests/sjb2ptx.rs links it)
```

It needs clang 18, LLVM 18's `llvm-link`, `opt` and `llc`, and python3. Each C file is compiled to
bitcode with `-O2 -ffp-contract=off -fno-builtin -fno-math-errno`. The files are linked, and only
the entry points stay public. `opt` optimizes the module again with inlining threshold 0, which
inlines only trivial functions and those called once, so that each math function stays one
`.func` and a kernel calling several functions does not get copies of them. `llc` writes PTX
(`sm_70`, PTX 8.3). `post.py` then prefixes the internal names with `__sjm_`, and sjb2ptx rejects
kernel names that start with `__sj`. The unit test `library_is_current` checks that `libm.ptx`
is what `gen.sh` makes of the sources.

## Numbers

Without FMA contraction, the library performs IEEE operations in a fixed order. The GPU therefore
computes the bits that the same C computes on x86-64; the test `math_library_matches_its_host_build`
checks this for every function on 6,000 points each. Against stable's JIT, which uses glibc's
functions, Cephes' and num_complex's (on glibc's), the results differ in the last bits:

* **Real functions:** within 8 ulps on the test points; most within 1 or 2.
* **Complex functions:** within 2.5e-13 relative for arguments of moderate size.
  num_complex's `arcsin`, `arccos` and `arcsinh` cancel for large |z|. There, the JIT is inaccurate
  too, and both results depend on the last bits of the real functions.
* **Special values:** musl's `tgamma` and `lgamma` follow C. `gamma(±0)` is ±inf, where stable's
  Cephes gives NaN. Subnormal results stay subnormal where Cephes flushes them to 0.
  `loggamma(-inf)` is +inf (Cephes: -inf). `loggamma` of a subnormal is finite (Cephes: inf). Near
  1 and 2, `loggamma` is correctly rounded, where Cephes is off by up to 3e-9 relative (checked with
  mpmath).
* **One musl bug is worked around:** musl 1.2.5's `acosh` tests |x| and returns finite values for
  x ≤ -2. `__sj_arccosh` returns NaN below 1.
