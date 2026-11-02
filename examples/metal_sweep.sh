#!/bin/bash
# metal_sweep.sh ZNG [N...]: the u u~ > Z + (N-1)g sweep on a Mac's GPU (docs/METAL.md).
# ZNG is the zng-mac directory (artifacts/n<N> from its gen.sh, check/momenta<N>.json from its
# check.sh). Per N: records the helicity-sum arena (1024 points; needs a `pacman`-feature
# pyamplicol on PATH), then examples/metal_probe in mixed precision at 4096, 16384 and 65536
# points per call and in f32 at 4096. Output in OUT (default ./metal-sweep): rec<N>/ and
# n<N>-<mixed|f32>-<points>.log. GPU runs need the Bash sandbox disabled.
Z=$(cd "${1:?usage: metal_sweep.sh ZNG [N...]}" && pwd) || exit 1
shift
OUT=${OUT:-$PWD/metal-sweep}
U=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR_BENCH:-$Z/xdg}
(cd "$U" && cargo build -q --release --example metal_probe) || exit 1
P=$U/target/release/examples/metal_probe
for n in ${@:-1 2 3 4 5 6 7 8}; do
  # the gluons 4..n+2 (none for n = 1); not `seq`: BSD seq counts down
  F=flow:2$(for ((i = 4; i <= n + 2; i++)); do printf ",%d" $i; done),1
  R=$OUT/rec$n
  if [ ! -d "$R" ]; then
    (cd "$Z" && PACMAN_SAVE_ARENA=$R PACMAN_SAVE_ARENA_POINTS=1024 pyamplicol evaluate artifacts/n$n \
      --momenta check/momenta$n.json --color-flow $F --json > /dev/null 2> "$R.err") ||
      { echo "n$n: recording failed (see $R.err)"; continue; }
  fi
  J=$(ls "$R"/*.json | head -1)
  for pts in 4096 16384 65536; do
    L=$OUT/n$n-mixed-$pts.log
    "$P" "$Z/artifacts/n$n" "$J" -o "$OUT/p$n-m" --points $pts --mixed > "$L" 2>&1
    echo "n$n mixed $pts: $(grep -E '^best' "$L")"
  done
  L=$OUT/n$n-f32-4096.log
  "$P" "$Z/artifacts/n$n" "$J" -o "$OUT/p$n-f" --points 4096 > "$L" 2>&1
  echo "n$n f32 4096: $(grep -E '^best' "$L")"
done
