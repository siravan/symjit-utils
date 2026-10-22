#!/usr/bin/env bash
# pi by Viète's formula through all five utilities:
#
#   viete.rs --(plain rustc)--> reference value
#   viete.rs --rust2sjb--> viete.sjb --sjbinfo--> report
#                                    --sjb2wasm--> viete.wasm --node wasm_main.mjs--> viete(0.5)
#                                    --sjb2obj---> viete.o    --cc obj_main.c------> viete(0.5)
#                                    --sjb2ptx---> viete.ptx  --cc ptx_main.c------> viete(0.5) on the GPU
#
#     examples/viete/run.sh [OUT_DIR]     (default: examples/viete/out)
#
# Builds the utilities (cargo build --release) and writes everything under OUT_DIR. The steps
# whose tools are missing (rustc, node, cc) or that find no GPU are skipped; the PTX module is
# still written and, if the CUDA toolkit is installed, checked with ptxas (for $PTXAS_ARCH,
# default sm_120: CUDA 13's ptxas no longer accepts sm_70, the module's .target, for which the
# driver compiles it). Exits with status 1 if a step fails or a result is not within 1.3e-6 of pi
# (N = 10 factors in viete.rs).

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=$(mkdir -p "${1:-$here/out}" && cd "${1:-$here/out}" && pwd)
bin="${CARGO_TARGET_DIR:-$root/target}/release"
tol=1.3e-6

have() { command -v "$1" > /dev/null 2>&1; }
step() { printf '\n== %s\n' "$*"; }
check() { # check LABEL VALUE: the value must be pi within $tol
    awk -v v="$2" -v tol="$tol" -v label="$1" 'BEGIN {
        d = v - 3.14159265358979323846; if (d < 0) d = -d
        if (!(d <= tol)) { printf "%s: %s is not pi within %s\n", label, v, tol; exit 1 } }'
}
value() { sed -n 's/^[a-z]*: *viete(0.5) = \([^ ]*\).*/\1/p'; }

step "build the utilities"
(cd "$root" && cargo build --release --quiet)

if have rustc; then
    step "plain Rust (viete.rs without #[kernel], with a main)"
    { sed '/^#\[kernel\]$/d' "$here/viete.rs"
      printf '\nfn main() {\n    let p = viete(0.5);\n    println!("rust: viete(0.5) = {}  (pi - viete(0.5) = {:.3e})", p, std::f64::consts::PI - p);\n}\n'
    } > "$out/viete_plain.rs"
    rustc -O --edition 2021 -o "$out/viete_plain" "$out/viete_plain.rs"
    line=$("$out/viete_plain"); echo "$line"
    check rust "$(value <<< "$line")"
else
    step "plain Rust: skipped (no rustc)"
fi

step "rust2sjb viete.rs -o viete.sjb"
"$bin/rust2sjb" "$here/viete.rs" -o "$out/viete.sjb"

step "sjbinfo viete.sjb"
"$bin/sjbinfo" "$out/viete.sjb"

step "sjb2wasm viete.sjb"
mkdir -p "$out/wasm"
"$bin/sjb2wasm" "$out/viete.sjb" -o "$out/wasm/viete.wasm"
if have node; then
    line=$(node "$here/wasm_main.mjs" "$out/wasm/viete.wasm"); echo "$line"
    check wasm "$(value <<< "$line")"
else
    echo "node: skipped (not installed)"
fi

mkdir -p "$out/obj"
case $(uname -m) in
x86_64 | aarch64 | arm64) native_obj=1 ;;
*) native_obj=0 ;;
esac
if [ "$native_obj" -eq 0 ]; then
    # sjb2obj writes x86-64 and AArch64 objects only: show it on a .sjb saved for aarch64
    step "sjb2obj viete_aarch64.sjb (this host, $(uname -m), has no object support)"
    "$bin/rust2sjb" "$here/viete.rs" --target aarch64 -o "$out/obj/viete_aarch64.sjb"
    "$bin/sjb2obj" "$out/obj/viete_aarch64.sjb" -o "$out/obj/viete.o"
    sed -n 's/^\/\* viete.h: \(.*\)\.$/\1/p' "$out/obj/viete.h"
    echo "obj: not run (an AArch64 object)"
else
    step "sjb2obj viete.sjb"
    "$bin/sjb2obj" "$out/viete.sjb" -o "$out/obj/viete.o"
fi
if [ "$native_obj" -eq 0 ]; then
    :
elif have cc; then
    cc -O2 -I"$out/obj" "$here/obj_main.c" "$out/obj/viete.o" -lm -o "$out/obj/obj_main"
    line=$("$out/obj/obj_main"); echo "$line"
    check obj "$(value <<< "$line")"
else
    echo "cc: skipped (not installed)"
fi

step "sjb2ptx viete.sjb --driver"
mkdir -p "$out/ptx"
"$bin/sjb2ptx" "$out/viete.sjb" -o "$out/ptx/viete.ptx" --driver
ptxas=${PTXAS:-/usr/local/cuda/bin/ptxas}
if [ -x "$ptxas" ]; then
    "$ptxas" -arch="${PTXAS_ARCH:-sm_120}" -o /dev/null "$out/ptx/viete.ptx"
    echo "ptxas: the module assembles for ${PTXAS_ARCH:-sm_120}"
fi
if have cc; then
    cc -O2 -I"$out/ptx" "$here/ptx_main.c" "$out/ptx/viete_cuda.c" -ldl -lm -o "$out/ptx/ptx_main"
    status=0; line=$("$out/ptx/ptx_main") || status=$?
    if [ "$status" -eq 77 ]; then
        echo "ptx: skipped (no usable GPU)"
    elif [ "$status" -ne 0 ]; then
        exit "$status"
    else
        echo "$line"
        check ptx "$(value <<< "$line")"
    fi
else
    echo "cc: skipped (not installed)"
fi

step "done: outputs in $out"
