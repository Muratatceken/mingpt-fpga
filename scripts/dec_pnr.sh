#!/usr/bin/env bash
# Open-source implementation of the stand-alone decision self-test (rtl/decision/dec_board_top.sv) on the
# Nexys A7-100T: Yosys (-nodsp) -> nextpnr-himbaechel (place, route, timing) [-> bitstream when timing is met].
#   scripts/dec_pnr.sh <model> [lm|gru|rule|fsm] [MHz]     (mem/decision/<model>; build/decision/{obs,exp_hash}.mem
#                                                           from scripts/dec_top.sh)
# Output: build/decision/pnr/<tag>/{pnr.log, report.json, result.json}; merged into model/decision/hw.json
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
X="${XC7_DIR:-$HOME/tools/xc7}"
M="$1"; POL="${2:-lm}"; MHZ="${3:-25}"; PART=xc7a100tcsg324-1
case "$POL" in lm|gru) PID=0 ;; rule) PID=1 ;; fsm) PID=2 ;; *) echo "bad policy"; exit 1 ;; esac
TAG="${M}_${POL}"; OUT="$ROOT/build/decision/pnr/$TAG"
NEXTPNR="$X/nextpnr/build/nextpnr-himbaechel"
CHIPDB="$(find "$X/nextpnr/build" -name "chipdb-xc7a100t.bin" | head -1)"
DB="$X/prjxray/database/artix7"
[ -x "$NEXTPNR" ] && [ -n "$CHIPDB" ] || { echo "nextpnr not built: run scripts/xc7_toolchain.sh" >&2; exit 1; }
mkdir -p "$OUT" && cd "$ROOT"
NSCEN=$(( $(wc -l < build/decision/exp_hash.mem) ))
sv2v -DSYNTHESIS "mem/decision/$M/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv \
  rtl/zoo/zoo_core.sv rtl/decision/dec_pkg.sv rtl/decision/dec_guard.sv rtl/decision/dec_policies.sv \
  rtl/decision/decision_top.sv rtl/decision/dec_board_top.sv -w "$OUT/rtl.v"
"$PY" - board/nexys_a7/nexys_a7.xdc "$OUT/board.xdc" <<'EOF'
import re, sys
keep = re.compile(r"^(CLK100MHZ|CPU_RESETN|BTNC|LED\[\d+\]|LED16_[RGB])$")
out = []
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip().rstrip(";").strip()
    port = re.search(r"\[get_ports \{?\s*([^ {}]+)\s*\}?\]$", line)
    props = re.search(r"-dict \{(.*?)\}", line)
    if port and not keep.match(port[1]):
        continue
    if props and port:
        out.append(f"set_property -dict {{{' '.join(props[1].replace('PACKAGE_PIN', 'LOC').split())}}} "
                   f"[get_ports {port[1]}]")
    elif line.startswith("create_clock") and port:
        out.append(f"create_clock -period {re.search(r'-period ([0-9.]+)', line)[1]} [get_ports {port[1]}]")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
EOF
t0=$(date +%s)
yosys -q -l "$OUT/synth.log" -p "read_verilog -defer $OUT/rtl.v; \
  chparam -set CLK_MHZ $MHZ -set POLICY $PID -set NSCEN $NSCEN dec_board_top; \
  synth_xilinx -family xc7 -top dec_board_top -flatten -nodsp; tee -q -o $OUT/stat.txt stat; write_json $OUT/d.json"
t1=$(date +%s)
"$NEXTPNR" --device "$PART" --chipdb "$CHIPDB" --json "$OUT/d.json" --top dec_board_top \
  --vopt xdc="$OUT/board.xdc" --vopt fasm="$OUT/d.fasm" --report "$OUT/report.json" --timing-allow-fail \
  --seed "${SEED:-1}" --log "$OUT/pnr.log" -q || { tail -30 "$OUT/pnr.log"; exit 1; }
t2=$(date +%s)
MET=$("$PY" -c "import json; f=json.load(open('$OUT/report.json'))['fmax'].values(); print(int(all(v['achieved']>=v['constraint']-1e-6 for v in f)))")
rm -f "$OUT/dec_$TAG.bit"
if [ "${BIT:-1}" = 1 ] && [ "$MET" = 1 ]; then
  "$X/prjxray/env/bin/python" "$X/prjxray/utils/fasm2frames.py" --part "$PART" --db-root "$DB" "$OUT/d.fasm" \
    "$OUT/d.frames" && "$X/prjxray/build/tools/xc7frames2bit" --part_file "$DB/$PART/part.yaml" \
    --part_name "$PART" --frm_file "$OUT/d.frames" --output_file "$OUT/dec_$TAG.bit" || true
fi
"$PY" - "$OUT" "$TAG" "$MHZ" "$((t1 - t0))" "$((t2 - t1))" <<'EOF'
import json, os, sys
out, tag, mhz, ts, tp = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rep = json.load(open(os.path.join(out, "report.json")))
fmax = rep.get("fmax", {})
util = {k: v for k, v in rep.get("utilization", {}).items() if v.get("used", 0)}
met = all(v["achieved"] >= v["constraint"] - 1e-6 for v in fmax.values())
res = dict(clock_mhz=mhz, met=met, fmax_mhz=round(min(v["achieved"] for v in fmax.values()), 2),
           utilization={k: f"{v['used']}/{v['available']}" for k, v in util.items()},
           seconds=dict(synth=ts, pnr=tp), bit=os.path.exists(os.path.join(out, f"dec_{tag}.bit")))
json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
path = os.path.join("model", "decision", "hw.json")
allr = json.load(open(path)) if os.path.exists(path) else {}
allr.setdefault(tag, {})["pnr"] = res
json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
print(f"{tag}: clock {mhz} MHz, Fmax {res['fmax_mhz']:.1f} MHz {'OK' if met else 'FAIL'}; synth {ts} s, pnr {tp} s")
for k, v in util.items():
    print(f"  {k}: {v['used']}/{v['available']}")
EOF
