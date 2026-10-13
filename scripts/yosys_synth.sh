#!/usr/bin/env bash
# Açık kaynak sentez (Yosys, Xilinx 7-serisi): kaynak kullanımı tahmini, zamanlama yok.
#   scripts/yosys_synth.sh [gpt_top|gpt_selftest]
# Gerekenler (macOS): brew install yosys sv2v
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOP="${1:-gpt_top}"
OUT="$ROOT/build/yosys"
mkdir -p "$OUT" && cd "$ROOT"            # $readmemh yolları (mem/...) depo köküne göreli

SRCS=$(cat scripts/files.f)
[ "$TOP" = gpt_top ] && SRCS=$(grep -v selftest scripts/files.f)
sv2v $SRCS -w "$OUT/$TOP.v"
yosys -q -l "$OUT/synth_$TOP.log" -p "read_verilog -defer $OUT/$TOP.v; hierarchy -top $TOP; \
  synth_xilinx -family xc7 -top $TOP -flatten; tee -q -o $OUT/stat_$TOP.txt stat"

python3 - "$OUT/stat_$TOP.txt" <<'PY'
import re, sys
cells = {m.group(2): int(m.group(1)) for m in re.finditer(r"^\s+(\d+)\s+(\w+)\s*$", open(sys.argv[1]).read(), re.M)}
g = lambda *k: sum(cells.get(x, 0) for x in k)
lut_logic = g("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "INV")
lut_ram = 4 * g("RAM64M", "RAM32M") + g("RAM64X1D") * 2 + g("RAM32X1D") * 2 + g("RAM128X1D") * 4 + g("SRL16E", "SRLC32E")
ff = g("FDRE", "FDSE", "FDCE", "FDPE")
bram = g("RAMB36E1") + 0.5 * g("RAMB18E1")
dsp = g("DSP48E1")
cap = dict(LUT=53200, FF=106400, BRAM=140, DSP=220)   # xc7z020
rows = [("LUT (mantık)", lut_logic, None), ("LUT (dağıtık RAM/SRL)", lut_ram, None),
        ("LUT toplam", lut_logic + lut_ram, "LUT"), ("FF", ff, "FF"),
        ("BRAM (RAMB36 eşd.)", bram, "BRAM"), ("DSP48E1", dsp, "DSP")]
print(f"Yosys sentezi ({sys.argv[1].split('stat_')[-1][:-4]}, xc7, zamanlama yok)")
for name, v, c in rows:
    pct = f"  {100 * v / cap[c]:5.1f}% / xc7z020" if c else ""
    print(f"  {name:24s} {v:8g}{pct}")
PY
