#!/usr/bin/env bash
# CPU baseline for the board numbers: Karpathy's llama2.c (run.c) runs stories260K on this machine's CPU
# with the board's conditions (256-token context, greedy, one thread) -> model/zoo/cpu_baseline.json
#   bash scripts/cpu_baseline.sh            (make cpu-baseline)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/build/cpu_baseline"
MODEL="$ROOT/data/stories260K/stories260K.bin"
T=256
REPS=${REPS:-200}
# macOS: the Command Line Tools SDK can be newer than the linker (tapi "unknown architecture arm64e.x1");
# then link against Xcode's SDK, as in scripts/xc7_toolchain.sh
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
mkdir -p "$B"
[ -f "$B/run.c" ] || curl -sSL -o "$B/run.c" https://raw.githubusercontent.com/karpathy/llama2.c/master/run.c
cp "$ROOT/scripts/cpu_bench.c" "$B/"
CC=${CC:-cc}
cd "$B"
"$CC" -O3 -w -o bench_O3 cpu_bench.c -lm
"$CC" -Ofast -mcpu=native -w -o bench_Ofast cpu_bench.c -lm
o3=$(./bench_O3 "$MODEL" $T "$REPS")
of=$(./bench_Ofast "$MODEL" $T "$REPS")
cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)
sha=$(shasum -a 256 run.c | cut -c1-16)
python3 - "$ROOT/model/zoo/cpu_baseline.json" "$cpu" "$sha" "$o3" "$of" <<'EOF'
import json, sys, platform
out, cpu, sha, o3, of = sys.argv[1:]
r = {"model": "stories260k", "code": "karpathy/llama2.c run.c (master, sha256 " + sha + "...)",
     "cpu": cpu.strip(), "os": platform.platform(), "threads": 1, "context": 256, "decoding": "greedy",
     "O3": json.loads(o3), "Ofast_native": json.loads(of)}
json.dump(r, open(out, "w"), indent=1)
print(json.dumps(r, indent=1))
EOF
