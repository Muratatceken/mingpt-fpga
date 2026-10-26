#!/usr/bin/env bash
# Open-source synthesis (Yosys, Xilinx 7-series) of the whole Nexys A7-100T design for one model:
# engine + generator + VGA + 7-segment, model memories as initialised BRAM (as in the bitstream).
#   scripts/nexys_synth.sh <model>          (mem/nexys/<model>/ from sw/zoo/board.py export)
# Resource estimate only (no timing). Output: build/nexys/<model>/yosys/{stat.txt, synth.log, summary.json}
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$1"; MEM="mem/nexys/$M"; OUT="$ROOT/build/nexys/$M/yosys"
mkdir -p "$OUT" && cd "$ROOT"                      # $readmemh paths (mem/nexys/...) relative to the repo root
sv2v -DSYNTHESIS "$MEM/zoo_pkg.sv" "$MEM/board_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv \
  rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv board/nexys_a7/bin2bcd.sv board/nexys_a7/vga_text.sv \
  board/nexys_a7/seg7.sv board/nexys_a7/lm_gen.sv board/nexys_a7/nexys_a7_top.sv -w "$OUT/nexys_$M.v"
yosys -q -l "$OUT/synth.log" -p "read_verilog -defer $OUT/nexys_$M.v; hierarchy -top nexys_a7_top; \
  synth_xilinx -family xc7 -top nexys_a7_top -flatten; tee -q -o $OUT/stat.txt stat"
python3 "$ROOT/scripts/yosys_summary.py" "$OUT/stat.txt" "$M" --json > "$OUT/summary.json"
python3 - "$OUT/summary.json" "$M" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
cap = dict(lut=63400, ff=126800, bram=135, dsp=240)          # xc7a100t
print(f"{sys.argv[2]:12s} " + "  ".join(f"{k.upper()} {s[k]} ({100 * float(s[k]) / v:.0f}%)" for k, v in cap.items()))
import os
path = os.path.join(os.path.dirname(sys.argv[1]), "..", "..", "..", "..", "model", "zoo", "nexys.json")
allr = json.load(open(path)) if os.path.exists(path) else {}
allr.setdefault(sys.argv[2], {})["yosys"] = s
json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
EOF
