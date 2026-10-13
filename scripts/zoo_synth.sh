#!/usr/bin/env bash
# Açık kaynak sentez (Yosys, Xilinx 7-serisi) -- mikrokodlu zoo motoru, tek model.
#   scripts/zoo_synth.sh <model>
# Kaynak kullanımı tahmini; zamanlama (Fmax) yok. Çıktı: build/zoo/yosys/stat_<model>.txt
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"     # ARM Verilator/Yosys/sv2v (macOS)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$1"
OUT="$ROOT/build/zoo/yosys"
mkdir -p "$OUT" && cd "$ROOT"            # $readmemh yolları (mem/zoo/...) depo köküne göreli
sv2v "mem/zoo/$M/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/requant.sv rtl/div_seq.sv rtl/isqrt_seq.sv \
  rtl/zoo/zoo_core.sv -w "$OUT/zoo_$M.v"
yosys -q -l "$OUT/synth_$M.log" -p "read_verilog -defer $OUT/zoo_$M.v; hierarchy -top zoo_core; \
  synth_xilinx -family xc7 -top zoo_core -flatten; tee -q -o $OUT/stat_$M.txt stat"
python3 "$ROOT/scripts/yosys_summary.py" "$OUT/stat_$M.txt" "$M"
