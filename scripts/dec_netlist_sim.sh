#!/usr/bin/env bash
# Post-synthesis (gate-level) check of the stand-alone decision self-test on the Nexys A7-100T.
# The netlist that scripts/dec_pnr.sh synthesised and handed to nextpnr (build/decision/pnr/<tag>/d.json:
# Yosys synth_xilinx -nodsp of rtl/decision/dec_board_top.sv without -DSIM: MMCME2_BASE, BUFG, I/O buffers, the
# model, the telemetry ROM and the expected hashes as initialised block RAM) is written as Verilog and simulated
# pin-only with Yosys' cell models (cells_sim.v), behavioural block RAMs (tb/sim_ramb.v) and the MMCM pass-through
# (tb/sim_mmcme2_base.v). tb/tb_dec_board.sv -DNETLIST presses BTNC (longer than the 10 ms debounce) and requires
# LED16_G with every scenario matched: the trace hash the netlist computes for each episode equals the hash of
# sw/decision/hw_ref.py.
#   scripts/dec_netlist_sim.sh <model> [lm|gru|rule|fsm]      (after scripts/dec_pnr.sh <model> <policy>)
# Output: build/decision/netlist/<tag>/{netlist.v, build.log, sim.log}; RESULT line merged into model/decision/hw.json
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"
export PATH="/opt/homebrew/bin:/usr/bin:/bin:$PATH"
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
M="$1"; POL="${2:-lm}"; TAG="${M}_${POL}"
case "$POL" in lm|gru) PID=0 ;; rule) PID=1 ;; fsm) PID=2 ;; *) echo "bad policy"; exit 1 ;; esac
PNR="build/decision/pnr/$TAG"; OUT="build/decision/netlist/$TAG"
[ -f "$PNR/d.json" ] || { echo "missing $PNR/d.json (run scripts/dec_pnr.sh $M $POL first)"; exit 1; }
mkdir -p "$OUT"
if [ ! -f "$OUT/netlist.v" ] || [ "$PNR/d.json" -nt "$OUT/netlist.v" ]; then
  yosys -q -l "$OUT/yosys.log" -p "read_json $PNR/d.json; setundef -zero -undriven -params; \
    write_verilog -noattr $OUT/netlist.v"
  # the JSON round trip writes real-valued cell parameters (MMCM) as strings: restore them as numbers
  sed -E -i '' 's/(\.(CLK[A-Z0-9_]*_F|CLKIN1_PERIOD|CLK[A-Z0-9_]*_PHASE|CLK[A-Z0-9_]*_DUTY_CYCLE|REF_JITTER1))\("(-?[0-9.]+)"\)/\1(\3)/' \
    "$OUT/netlist.v"
fi
CELLS="$(dirname "$(command -v yosys)")/../share/yosys/xilinx/cells_sim.v"
"$PY" - "$CELLS" "$OUT/cells_sim.v" <<'PYEOF'
import re, sys
src = open(sys.argv[1]).read()
out = re.sub(r"(\(\*[^)]*\*\)\s*)*module (RAMB18E1|RAMB36E1)\b.*?endmodule", "", src, flags=re.S)
assert "module RAMB18E1" not in out and "module RAMB36E1" not in out
open(sys.argv[2], "w").write(out)
PYEOF
sv2v -DNETLIST tb/tb_dec_board.sv -w "$OUT/tb.v"
T0=$(date +%s)
verilator --binary --timing --timescale 1ns/1ps -j "${VJOBS:-4}" -Wno-fatal -Wno-lint -Wno-style \
  -Wno-SPECIFYIGN -Wno-MULTIDRIVEN -Wno-COMBDLY -Wno-INITIALDLY -Wno-UNOPTFLAT -Wno-TIMESCALEMOD \
  -GPOLICY=$PID --top-module tb_dec_board "$OUT/tb.v" "$OUT/netlist.v" tb/sim_mmcme2_base.v tb/sim_ramb.v \
  "$OUT/cells_sim.v" -Mdir "$OUT/obj" -o Vtb > "$OUT/build.log" 2>&1 || { grep -m 20 "%Error" "$OUT/build.log"; exit 1; }
T1=$(date +%s)
"$OUT/obj/Vtb" > "$OUT/sim.log" 2>&1 || true
T2=$(date +%s)
grep -E "RESULT|PASS|FAIL|MMCM|RAMB|unsupported" "$OUT/sim.log" | head -8 | sed "s/^/[$TAG netlist] /"
"$PY" - "$OUT/sim.log" "$TAG" "$((T1 - T0))" "$((T2 - T1))" <<'PYEOF'
import json, os, re, sys
log, tag, tb, ts = open(sys.argv[1]).read(), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
m = re.search(r"RESULT board policy \d+: scenarios (\d+) matched (\d+) green (\d) red (\d) cycles (\d+)", log)
res = dict(passed="PASS:" in log, scenarios=int(m[1]) if m else None, matched=int(m[2]) if m else None,
           cycles=int(m[5]) if m else None, seconds=dict(build=tb, sim=ts))
path = os.path.join("model", "decision", "hw.json")
allr = json.load(open(path)) if os.path.exists(path) else {}
allr.setdefault(tag, {})["netlist_sim"] = res
json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
PYEOF
grep -q "PASS:" "$OUT/sim.log"
