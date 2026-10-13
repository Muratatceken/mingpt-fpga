#!/usr/bin/env bash
# Açık kaynak sentez (Yosys, Xilinx 7-serisi) -- mikrokodlu zoo motoru, tek model.
#   scripts/zoo_synth.sh <model> [spec]
#   spec: yalnızca modelin kullandığı birimler (SPECIALIZE=1); yoksa tüm birimlerle genel motor.
#   Her iki durumda da model bellekleri yüklenebilir RAM'dir (LOADABLE=1): Yosys belirli bir
#   modelin programını/ağırlıklarını sabit sayıp mantığı katlayamaz.
# Kaynak kullanımı tahmini; zamanlama (Fmax) yok. Çıktı: build/zoo/yosys/stat_<model>[_spec].txt
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"     # ARM Verilator/Yosys/sv2v (macOS)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$1"; SPEC=0; TAG=""
[ "${2:-}" = spec ] && { SPEC=1; TAG=_spec; }
OUT="$ROOT/build/zoo/yosys"
mkdir -p "$OUT" && cd "$ROOT"            # $readmemh yolları (mem/zoo/...) depo köküne göreli
sv2v "mem/zoo/$M/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv \
  rtl/zoo/zoo_core.sv -w "$OUT/zoo_$M$TAG.v"
yosys -q -l "$OUT/synth_$M$TAG.log" -p "read_verilog -defer $OUT/zoo_$M$TAG.v; \
  chparam -set SPECIALIZE $SPEC -set LOADABLE 1 zoo_core; hierarchy -top zoo_core; \
  synth_xilinx -family xc7 -top zoo_core -flatten; tee -q -o $OUT/stat_$M$TAG.txt stat"
python3 "$ROOT/scripts/yosys_summary.py" "$OUT/stat_$M$TAG.txt" "$M$TAG"
