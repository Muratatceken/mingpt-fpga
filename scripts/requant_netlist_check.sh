#!/usr/bin/env bash
# Yosys synth_xilinx check of rtl/requant.sv with registered inputs (tb/tb_requant_netlist.sv), with and without
# DSP48E1 mapping, simulated against the RTL with Yosys' cell models (Icarus). Yosys 0.69: the DSP version
# loses the sign of the product (negative accumulators saturate to +max), the -nodsp version is exact.
#   scripts/requant_netlist_check.sh            -> build/requant_netlist/{dsp,nodsp}/
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CELLS="$(dirname "$(command -v yosys)")/../share/yosys/xilinx/cells_sim.v"
yosys -V
for v in dsp nodsp; do
  OUT="$ROOT/build/requant_netlist/$v"; mkdir -p "$OUT"
  OPT=""; if [ "$v" = nodsp ]; then OPT="-nodsp"; fi
  sv2v -DSYNTHESIS rtl/requant.sv tb/tb_requant_netlist.sv -w "$OUT/rtl.v"
  printf 'read_verilog %s/rtl.v\nsynth_xilinx -family xc7 -top requant_reg -flatten -noiopad %s\nrename requant_reg requant_reg_net\nopt_clean\ntee -q -o %s/stat.txt stat\nwrite_verilog -noattr %s/net.v\n' \
    "$OUT" "$OPT" "$OUT" "$OUT" > "$OUT/synth.ys"
  yosys -q -l "$OUT/synth.log" -s "$OUT/synth.ys" > /dev/null
  sv2v rtl/requant.sv tb/tb_requant_netlist.sv -w "$OUT/tb.v"
  iverilog -g2012 -s tb_requant_netlist -o "$OUT/sim.vvp" "$OUT/tb.v" "$OUT/net.v" "$CELLS"
  echo "$v ($(grep -E '^ +[0-9]+ +DSP48E1' "$OUT/stat.txt" | awk '{print $1}' || echo 0) DSP48E1): $(vvp -n "$OUT/sim.vvp" | grep RESULT)"
done
