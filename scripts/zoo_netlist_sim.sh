#!/usr/bin/env bash
# Gate-level check: synthesize the engine for one model with Yosys (Xilinx 7-series cells, the model's
# memories as initialized ROMs) and simulate the NETLIST with Icarus and Yosys' cell models,
# against the same golden vectors.
#   scripts/zoo_netlist_sim.sh <model | memdir> [NSEQ]
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
# macOS: if xcrun picks a Command Line Tools SDK newer than the selected Xcode's linker, use Xcode's SDK
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
M="$1"; NSEQ="${2:-3}"
case "$M" in */*) MEM="${M%/}" ;; *) MEM="mem/zoo/$M" ;; esac
TAG="$(echo "$MEM" | tr '/' '_')"
OUT="$ROOT/build/zoo/netlist/$TAG"
mkdir -p "$OUT"
CELLS="$(dirname "$(command -v yosys)")/../share/yosys/xilinx/cells_sim.v"
sv2v -DSYNTHESIS "$MEM/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv \
  rtl/zoo/zoo_core.sv -w "$OUT/rtl.v"
yosys -q -l "$OUT/synth.log" -p "read_verilog -defer $OUT/rtl.v; chparam -set SPECIALIZE 1 zoo_core; \
  hierarchy -top zoo_core; synth_xilinx -family xc7 -top zoo_core -flatten; opt_clean; setundef -zero -undriven -params; \
  write_verilog -noattr $OUT/netlist.v; tee -q -o $OUT/stat.txt stat"
sv2v -DNETLIST -DXCHECK "$MEM/zoo_pkg.sv" tb/tb_zoo.sv -w "$OUT/tb.v"
iverilog -g2012 -s tb_zoo -o "$OUT/sim.vvp" "$OUT/tb.v" "$OUT/netlist.v" "$CELLS" > "$OUT/iverilog.log" 2>&1 \
  || { tail -20 "$OUT/iverilog.log"; exit 1; }
vvp -n "$OUT/sim.vvp" +NSEQ="$NSEQ"
