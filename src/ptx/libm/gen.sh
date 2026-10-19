#!/bin/sh
# sjb2ptx's math library: compiles musl's functions and symjit's (sj/) to PTX.
#
#   src/ptx/libm/gen.sh                 rewrite libm.ptx (checked in; sjb2ptx embeds it)
#   src/ptx/libm/gen.sh --ptx FILE      write the PTX to FILE instead (tests: is libm.ptx current?)
#   src/ptx/libm/gen.sh --host FILE     an x86-64 object of the same C, compiled the same way
#                                       (tests: the reference the GPU must match bit for bit)
#
# Every C file becomes LLVM bitcode with clang 18 (-O2, no FMA contraction, no builtins, no
# errno); the files are linked into one module, in which only the entry points (`__sj_<op>`,
# `__sj_cplx_<op>`) stay public, and optimized again (inlining across files); llc writes PTX or
# the host object. In the PTX, the internal functions and tables get the prefix `__sjm_`.
# Needs clang-18, llvm-link-18, opt-18, llc-18 and python3 ($CLANG, $LLVM_LINK, $OPT, $LLC).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
CLANG=${CLANG:-clang-18}
LLVM_LINK=${LLVM_LINK:-llvm-link-18}
OPT=${OPT:-opt-18}
LLC=${LLC:-llc-18}

MODE=ptx
OUT=$HERE/libm.ptx
case "${1:-}" in
    "") ;;
    --ptx) OUT=$2 ;;
    --host) MODE=host; OUT=$2 ;;
    *) echo "usage: $0 [--ptx FILE | --host FILE]" >&2; exit 2 ;;
esac

MUSL="acos acosh asin asinh atan atan2 atanh cbrt __cos cos cosh erf exp exp2 exp_data expm1
      __expo2 hypot lgamma_r log log10 log1p log2 log2_data log_data __math_divzero
      __math_invalid __math_oflow __math_uflow __math_xflow pow pow_data __rem_pio2
      __rem_pio2_large scalbn __sin sin sinh __tan tan tanh tgamma"
SJ="overrides symjit_math symjit_complex cephes"

if [ $MODE = ptx ]; then
    TARGET="--target=nvptx64-nvidia-cuda -march=sm_70 -nogpulib"
else
    TARGET="--target=x86_64-unknown-linux-gnu"
    MUSL="$MUSL floor"   # sj/overrides.c: no floor instruction on x86-64
fi

RES=$($CLANG -print-resource-dir)/include
CFLAGS="-std=c11 -O2 -ffp-contract=off -fno-builtin -fno-math-errno -ffreestanding -nostdinc
        -I $HERE/compat -I $RES -I $HERE/musl/arch/generic -I $HERE/musl/src/internal
        -I $HERE/musl/include -Wno-parentheses -Wno-shift-op-parentheses -Wno-unused-parameter"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

for f in $MUSL; do
    $CLANG $TARGET $CFLAGS -emit-llvm -c "$HERE/musl/src/math/$f.c" -o "$TMP/m_$f.bc"
done
for f in $SJ; do
    $CLANG $TARGET $CFLAGS -Wall -Wextra -Werror -Wno-unused-parameter \
        -emit-llvm -c "$HERE/sj/$f.c" -o "$TMP/s_$f.bc"
done

API=$(cat "$HERE/sj/symjit_math.c" "$HERE/sj/symjit_complex.c" |
      sed -n 's/^\(double\|cplx\) \(__sj_[A-Za-z0-9_]*\)(.*/\2/p' | sort -u | paste -sd, -)

$LLVM_LINK "$TMP"/*.bc -o "$TMP/all.bc"
$OPT -passes='internalize,globaldce,default<O2>' -inline-threshold=0 \
    -internalize-public-api-list="$API" \
    "$TMP/all.bc" -o "$TMP/opt.bc"

if [ $MODE = host ]; then
    $LLC -O2 -relocation-model=pic -filetype=obj "$TMP/opt.bc" -o "$OUT"
else
    $LLC -O2 -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx83 "$TMP/opt.bc" -o "$TMP/libm.ptx"
    python3 "$HERE/post.py" "$TMP/libm.ptx" "$OUT"
fi
