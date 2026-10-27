#!/usr/bin/env bash
# Open-source implementation of the Nexys A7-100T design on macOS: Yosys -> nextpnr-himbaechel (place, route, timing)
# -> fasm2frames -> xc7frames2bit. Tools from scripts/xc7_toolchain.sh ($XC7_DIR, default ~/tools/xc7).
#   scripts/nexys_pnr.sh <model> [MHz]      (mem/nexys/<model>/ from sw/zoo/board.py export; MHz = core clock, 50)
#   DSP=1  map multipliers to DSP48E1. Default: LUTs (-nodsp), the netlist that scripts/nexys_netlist_sim.sh proves
#          bit-exact. With DSP48E1, Yosys 0.69 drops the sign extension of the requantiser product (docs/05, sec. 6),
#          so a DSP=1 bitstream computes wrong tokens; use it for timing/resources only.
#   BIT=0  stop after routing (Fmax and utilisation only).
#   As scripts/vivado_nexys_a7.tcl: if timing fails at 50 MHz the design is rebuilt at 25 MHz (FALLBACK=0: no
#   retry); the bitstream is written only when timing is met.
# Output: build/nexys/<model>/pnr[_dsp]/{pnr.log, report.json, result.json, nexys_<model>.bit}; model/zoo/nexys.json
# Program: openFPGALoader -b nexys_a7_100 build/nexys/<model>/pnr/nexys_<model>.bit
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
X="${XC7_DIR:-$HOME/tools/xc7}"
M="$1"; MHZ="${2:-50}"; PART=xc7a100tcsg324-1
if [ "${DSP:-0}" = 1 ]; then SYNOPT=""; TAG=pnr_dsp; else SYNOPT="-nodsp"; TAG=pnr; fi
MEM="mem/nexys/$M"; OUT="$ROOT/build/nexys/$M/$TAG"
NEXTPNR="$X/nextpnr/build/nextpnr-himbaechel"
CHIPDB="$(find "$X/nextpnr/build" -name "chipdb-xc7a100t.bin" | head -1)"
DB="$X/prjxray/database/artix7"
[ -x "$NEXTPNR" ] && [ -n "$CHIPDB" ] || { echo "nextpnr not built: run scripts/xc7_toolchain.sh" >&2; exit 1; }
[ -d "$MEM" ] || { echo "$MEM missing: make nexys MODEL=$M" >&2; exit 1; }
mkdir -p "$OUT" && cd "$ROOT"                      # $readmemh paths (mem/nexys/...) relative to the repo root

sv2v -DSYNTHESIS "$MEM/zoo_pkg.sv" "$MEM/board_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv \
  rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv board/nexys_a7/bin2bcd.sv board/nexys_a7/vga_text.sv \
  board/nexys_a7/seg7.sv board/nexys_a7/lm_gen.sv board/nexys_a7/nexys_a7_top.sv -w "$OUT/rtl.v"

# nextpnr's XDC reader: one port per set_property, no braces around port names, no trailing ';', LOC for the pin.
# set_false_path / [current_design] lines are dropped (nextpnr ignores them; the I/O paths are asynchronous anyway).
python3 - board/nexys_a7/nexys_a7.xdc "$OUT/nexys.xdc" <<'EOF'
import re, sys
out = []
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip().rstrip(";").strip()
    port = re.search(r"\[get_ports \{?\s*([^ {}]+)\s*\}?\]$", line)
    props = re.search(r"-dict \{(.*?)\}", line)
    if props and port:
        # LOC, not PACKAGE_PIN: nextpnr copies PACKAGE_PIN with attrs[LOC] = attrs.at(PACKAGE_PIN), a dangling
        # reference once the insert reallocates (empty pin name on macOS/libc++)
        out.append(f"set_property -dict {{{' '.join(props[1].replace('PACKAGE_PIN', 'LOC').split())}}} "
                   f"[get_ports {port[1]}]")
    elif line.startswith("create_clock") and port:
        out.append(f"create_clock -period {re.search(r'-period ([0-9.]+)', line)[1]} [get_ports {port[1]}]")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(f"{len(out)} XDC constraints")
EOF

SYN_S=0; PNR_S=0; TRIED=""
implement() {     # $1 = core clock in MHz: synthesis, place and route; sets FMAX and MET (worst clock)
  local t0 t1 t2
  t0=$(date +%s)
  yosys -q -l "$OUT/synth.log" -p "read_verilog -defer $OUT/rtl.v; chparam -set CLK_MHZ $1 nexys_a7_top; \
    synth_xilinx -family xc7 -top nexys_a7_top -flatten $SYNOPT; tee -q -o $OUT/stat.txt stat; write_json $OUT/nexys.json"
  t1=$(date +%s)
  "$NEXTPNR" --device "$PART" --chipdb "$CHIPDB" --json "$OUT/nexys.json" --top nexys_a7_top \
    --vopt xdc="$OUT/nexys.xdc" --vopt fasm="$OUT/nexys.fasm" --report "$OUT/report.json" --timing-allow-fail \
    --seed "${SEED:-1}" --log "$OUT/pnr.log" -q || { tail -30 "$OUT/pnr.log"; exit 1; }
  t2=$(date +%s)
  SYN_S=$((SYN_S + t1 - t0)); PNR_S=$((PNR_S + t2 - t1))
  read -r FMAX MET < <(python3 -c "import json; f = json.load(open('$OUT/report.json'))['fmax'].values(); \
    print(f'{min(v[\"achieved\"] for v in f):.2f}', int(all(v['achieved'] >= v['constraint'] - 1e-6 for v in f)))")
  TRIED="$TRIED $1:$FMAX"
}
F="$MHZ"
implement "$F"
if [ "$MET" = 0 ] && [ "${FALLBACK:-1}" = 1 ] && [ "$F" -gt 25 ]; then
  echo "$M: Fmax $FMAX MHz < $F MHz -> rebuilding at 25 MHz"
  mv "$OUT/report.json" "$OUT/report_${F}mhz.json"; mv "$OUT/pnr.log" "$OUT/pnr_${F}mhz.log"
  F=25; implement "$F"
fi

T2=$(date +%s)
rm -f "$OUT/nexys_$M.bit"
if [ "${BIT:-1}" = 1 ] && [ "$MET" = 1 ]; then
  "$X/prjxray/env/bin/python" "$X/prjxray/utils/fasm2frames.py" --part "$PART" --db-root "$DB" \
    "$OUT/nexys.fasm" "$OUT/nexys.frames"
  "$X/prjxray/build/tools/xc7frames2bit" --part_file "$DB/$PART/part.yaml" --part_name "$PART" \
    --frm_file "$OUT/nexys.frames" --output_file "$OUT/nexys_$M.bit"
fi
T3=$(date +%s)

python3 - "$OUT" "$M" "$F" "$TAG" "$SYN_S" "$PNR_S" "$((T3 - T2))" "$TRIED" <<'EOF'
import fcntl, json, os, sys
out, m, mhz, tag = sys.argv[1:5]
t_syn, t_pnr, t_bit = map(int, sys.argv[5:8])
tried = {int(a): float(b) for a, b in (x.split(":") for x in sys.argv[8].split())}
rep = json.load(open(os.path.join(out, "report.json")))
fmax = rep.get("fmax", {})
util = {k: v for k, v in rep.get("utilization", {}).items() if v.get("used", 0)}
met = all(v["achieved"] >= v["constraint"] - 1e-6 for v in fmax.values())
res = dict(model=m, clock_mhz=int(mhz), met=met, fmax_mhz=round(min(v["achieved"] for v in fmax.values()), 2),
           tried_fmax_mhz=tried, flow=tag, utilization={k: f"{v['used']}/{v['available']}" for k, v in util.items()},
           seconds=dict(synth=t_syn, pnr=t_pnr, bitstream=t_bit),
           bit=os.path.exists(os.path.join(out, f"nexys_{m}.bit")))
json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
path = os.path.join(out, "..", "..", "..", "..", "model", "zoo", "nexys.json")
with open(path + ".lock", "w") as lock:                      # parallel runs of this script
    fcntl.flock(lock, fcntl.LOCK_EX)
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr.setdefault(m, {})[tag] = {k: v for k, v in res.items() if k not in ("model", "flow", "seconds")}
    json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
os.remove(path + ".lock")
tries = ", ".join(f"{a} MHz: {b:.1f}" for a, b in tried.items())
print(f"{m} {tag}: clock {mhz} MHz, Fmax {res['fmax_mhz']:.1f} MHz {'OK' if met else 'FAIL'} (tried {tries}); "
      f"bitstream {'yes' if res['bit'] else 'no'}")
for k, v in util.items():
    print(f"  {k}: {v['used']}/{v['available']}")
print(f"  synth {t_syn} s, pnr {t_pnr} s, bitstream {t_bit} s")
EOF
