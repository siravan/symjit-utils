#!/bin/bash
# metal_runner_zng.sh ZNG [N...]: the u u~ > Z + (N-1)g sweep inside RustiCol with Metal runners
# (`pacman runner --metal`, docs/METAL.md). ZNG is the zng-mac directory (artifacts/n<N> from
# its gen.sh; check/momenta<N>.json and the stock results check/n<N>-<sum|hel>-cpu.json from its
# check.sh; logs/<sum|hel><N>-cpu.log from its run.sh). Per N: a runner for every execution, then
# `pyamplicol evaluate` against stock (|M|^2 relative, per point) and `pyamplicol profile` (as
# run.sh) for the helicity sum and one helicity. Output in OUT (default ./metal-runners).
# Env: TARGET (profile --target-runtime, s; default 5), METAL_FLAGS (e.g. --f32). Needs a
# `pacman`-feature pyamplicol on PATH, numpy, and the Bash sandbox disabled (the GPU).
Z=$(cd "${1:?usage: metal_runner_zng.sh ZNG [N...]}" && pwd) || exit 1
shift
OUT=${OUT:-$PWD/metal-runners}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
U=$(cd "$(dirname "$0")/.." && pwd)
(cd "$U" && cargo build -q --release --bin pacman) || exit 1
P=$U/target/release/pacman
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR_BENCH:-$Z/xdg}
cat > "$OUT/compare.py" <<'EOF'
import json, sys, numpy as np
x = np.array([v["real"] for v in json.load(open(sys.argv[1]))])
y = np.array([v["real"] for v in json.load(open(sys.argv[2]))])
if len(x) != len(y) or len(x) == 0:
    sys.exit(f"{len(x)} stock values, {len(y)} Metal values")
rel = np.abs(y - x) / np.abs(x).clip(1e-300)
print(f"{len(x)} points: |M|^2 relative median {np.median(rel):.1e}, 99% "
      f"{np.quantile(rel, .99):.1e}, max {rel.max():.1e}; {(~np.isfinite(y)).sum()} non-finite")
EOF
cd "$Z" || exit 1
for n in ${@:-1 2 3 4 5 6 7 8}; do
  A=artifacts/n$n
  for e in $("$P" bindings $A 2>/dev/null | sed -n 's/^  execution \([^ ]*\):.*/\1/p'); do
    R=$OUT/runners/n$n/$(echo "$e" | tr '/[]' '_._')
    [ -f "$R/librunner.so" ] ||
      "$P" runner --metal $METAL_FLAGS --execution "$e" -o "$R" $A > "$OUT/build$n.log" 2>&1 ||
      echo "n$n $e: build failed: $(tail -1 "$OUT/build$n.log")"
  done
  libs=$(ls -d "$OUT"/runners/n$n/*/librunner.so | tr '\n' ':')
  libs=${libs%:}
  H=h:$(for i in $(seq 1 $((n + 2))); do [ $((i % 2)) = 1 ] && printf -- '-1,' || printf -- '+1,'; done | sed 's/,$//')
  # the gluons 4..n+2 (none for n = 1); not `seq`: BSD seq counts down
  F=flow:2$(for ((i = 4; i <= n + 2; i++)); do printf ",%d" $i; done),1
  for w in sum hel; do
    SEL="--color-flow $F"
    [ $w = hel ] && SEL="--helicity $H $SEL"
    O=$OUT/n$n-$w
    PACMAN_RUNNER=$libs RUSTICOL_TILE_POINTS=16384 PACMAN_RUNNER_STATS=1 \
      pyamplicol evaluate $A --momenta check/momenta$n.json $SEL --json > "$O.json" 2> "$O.err"
    echo "n$n $w check: $(python3 "$OUT/compare.py" check/n$n-$w-cpu.json "$O.json")"
  done
  for w in sum hel; do
    SEL=""
    [ $w = hel ] && SEL="--helicity $H --color-flow $F"
    L=$OUT/$w$n-metal.log
    PACMAN_RUNNER=$libs RUSTICOL_TILE_POINTS=16384 PACMAN_RUNNER_STATS=1 \
      pyamplicol profile $A $SEL --batch-size 16384 --target-runtime ${TARGET:-5} > "$L" 2>&1
    echo "n$n $w: Metal $(grep -oE 'wall [0-9.]+ (ns|us)/point' "$L" | tail -1)," \
      "stock $(grep -oE 'wall [0-9.]+ (ns|us)/point' logs/$w$n-cpu.log 2>/dev/null | tail -1)"
    grep '^pacman runner' "$L" | grep -v ' 0 calls' | sed 's/^/  /'
  done
done
