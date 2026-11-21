#!/usr/bin/env bash
# Open-source synthesis (Yosys, Xilinx 7-series) of the decision RTL: resource numbers per policy.
#   scripts/dec_synth.sh guard                 independent safety guard alone (dec_guard)
#   scripts/dec_synth.sh rule|fsm  <model>     decision_top with the rule / FSM policy (no engine)
#   scripts/dec_synth.sh engine    <model>     decision_top with the integer LM/GRU engine of mem/decision/<model>
# Model memories as initialised BRAM (as in a bitstream).  Observation ROM 1024 x 64 bit (test fixture).
# Output: build/decision/synth/<tag>/{stat.txt,summary.json}; merged into model/decision/hw.json
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
KIND="$1"; M="${2:-}"
case "$KIND" in
  guard)  TAG=guard; TOP=dec_guard; PARAMS="" ;;
  rule)   TAG=rule;  TOP=decision_top; PARAMS="chparam -set POLICY 1 -set NWORDS 1024 decision_top;" ;;
  fsm)    TAG=fsm;   TOP=decision_top; PARAMS="chparam -set POLICY 2 -set NWORDS 1024 decision_top;" ;;
  engine) TAG="$M";  TOP=decision_top; PARAMS="chparam -set POLICY 0 -set NWORDS 1024 decision_top;" ;;
  *) echo "usage: $0 guard | rule|fsm|engine <model>"; exit 1 ;;
esac
OUT="build/decision/synth/$TAG"
mkdir -p "$OUT"
SRC=(rtl/decision/dec_pkg.sv rtl/decision/dec_guard.sv)
if [ "$TOP" = decision_top ]; then
  [ -f "mem/decision/$M/zoo_pkg.sv" ] || { echo "missing mem/decision/$M"; exit 1; }
  [ -f build/decision/obs.mem ] || { echo "missing build/decision/obs.mem (run scripts/dec_top.sh first)"; exit 1; }
  SRC=("mem/decision/$M/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv
       rtl/zoo/zoo_core.sv "${SRC[@]}" rtl/decision/dec_policies.sv rtl/decision/decision_top.sv)
fi
sv2v -DSYNTHESIS "${SRC[@]}" -w "$OUT/design.v"
t0=$(date +%s)
yosys -q -l "$OUT/synth.log" -p "read_verilog -defer $OUT/design.v; $PARAMS hierarchy -top $TOP; \
  synth_xilinx -family xc7 -top $TOP -flatten; tee -q -o $OUT/stat.txt stat; \
  tee -q -o $OUT/ltp_$TAG.txt ltp -noff" > /dev/null
cp "$OUT/stat.txt" "$OUT/stat_$TAG.txt"
"$PY" scripts/yosys_summary.py "$OUT/stat_$TAG.txt" "$TAG" --json > "$OUT/summary.json"
"$PY" - "$OUT/summary.json" "$TAG" "$KIND" "$(( $(date +%s) - t0 ))" <<'EOF'
import json, os, sys
s = json.load(open(sys.argv[1])); s["kind"] = sys.argv[3]; s["synth_s"] = int(sys.argv[4])
cap = dict(lut=63400, ff=126800, bram=135, dsp=240)
print(f"{sys.argv[2]:10s} " + "  ".join(f"{k.upper()} {s[k]} ({100 * float(s[k]) / v:.1f}%)" for k, v in cap.items())
      + (f"  depth {s['depth']}" if "depth" in s else ""))
path = os.path.join("model", "decision", "hw.json")
allr = json.load(open(path)) if os.path.exists(path) else {}
allr.setdefault(sys.argv[2], {})["yosys"] = s
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
EOF
