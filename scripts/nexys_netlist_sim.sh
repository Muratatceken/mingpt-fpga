#!/usr/bin/env bash
# Post-synthesis check of the whole Nexys A7-100T design (board/nexys_a7/nexys_a7_top.sv) for one model:
# Yosys synthesises the REAL top (no -DSIM: MMCM, BUFG, I/O buffers, model memories as initialised BRAM)
# with Xilinx 7-series cells; the NETLIST is simulated with Yosys' cell models (cells_sim.v: LUT, CARRY4,
# FD*, DSP48E1, ...), behavioural block RAMs (tb/sim_ramb.v: the installed cells_sim.v declares RAMB18E1 /
# RAMB36E1 without behaviour), an MMCME2_BASE pass-through (tb/sim_mmcme2_base.v) and the board testbench
# in pin-only mode (tb/tb_nexys.sv -DNETLIST). sw/zoo/board_sim.py --sim netlist then runs the RTL
# (Verilator, pin-only) with the same keys and requires the netlist to be cycle-identical on every pin (pin
# signature at every idle point, LED token counter time stamps, 7-segment, VGA frame) and the RTL to match
# board.py.
#   scripts/nexys_netlist_sim.sh <model> [board_sim.py options]      (default: --press CUD --sw1 1 --frame)
# Environment: NETSIM=verilator (default, 2-state, fast) | icarus (4-state with X checks on the pins, about
#   150 clocks/s: use with short runs, e.g. MAXCYC); BUILD_ONLY=1: netlist + simulator only;
#   DSP=1: keep the DSP48E1 mapping. By default the multipliers are mapped to LUTs (-nodsp): with DSP48E1,
#   Yosys 0.69 synth_xilinx drops the sign extension of the requantiser product (rtl/requant.sv prod_r) when
#   its inputs come from registers, so negative accumulators saturate to +max and the netlist is wrong
#   (scripts/requant_netlist_check.sh reproduces it on requant alone; the RTL is correct).
# Debounce / typewriter / 7-segment periods are fixed to the testbench values at synthesis (chparam), as
# the netlist has no parameters. Output: build/nexys/<model>/netlist[_dsp]/{netlist.v, stat.txt, synth.log, sim.log}
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"               # the Python with numpy (before the PATH change below)
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
# macOS: if xcrun picks a Command Line Tools SDK newer than the selected Xcode's linker, use Xcode's SDK
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"                                          # $readmemh paths (mem/nexys/...) relative to the repo root
M="$1"; shift
if [ $# -eq 0 ]; then set -- --press CUD --sw1 1 --frame; fi
NETSIM="${NETSIM:-verilator}"
if [ "${DSP:-0}" = 1 ]; then SYNOPT=""; TAG=netlist_dsp; else SYNOPT="-nodsp"; TAG=netlist; fi
MEM="mem/nexys/$M"; OUT="$ROOT/build/nexys/$M/$TAG"
mkdir -p "$OUT"
CELLS="$(dirname "$(command -v yosys)")/../share/yosys/xilinx/cells_sim.v"
SRC=(rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv board/nexys_a7/bin2bcd.sv
     board/nexys_a7/vga_text.sv board/nexys_a7/seg7.sv board/nexys_a7/lm_gen.sv board/nexys_a7/nexys_a7_top.sv)
# synthesis only when a source or memory image is newer than the netlist
if [ ! -f "$OUT/netlist.v" ] || [ -n "$(find "${SRC[@]}" "$MEM" "$0" -newer "$OUT/netlist.v" 2>/dev/null)" ]; then
  echo "synthesising $M (Yosys synth_xilinx) ..."
  T0=$(date +%s)
  sv2v -DSYNTHESIS "$MEM/zoo_pkg.sv" "$MEM/board_pkg.sv" "${SRC[@]}" -w "$OUT/rtl.v"
  yosys -q -l "$OUT/synth.log" -p "read_verilog -defer $OUT/rtl.v; \
    chparam -set DEB_CYC 4 -set SLOW_CYC 64 -set SEG_CYC 8 nexys_a7_top; \
    synth_xilinx -family xc7 -top nexys_a7_top -flatten $SYNOPT; opt_clean; setundef -zero -undriven -params; \
    tee -q -o $OUT/stat.txt stat; write_verilog -noattr $OUT/netlist.v"
  echo "synthesis $(( $(date +%s) - T0 )) s"
fi
grep -E "^ +[0-9]+ +(cells|RAMB|DSP48E1|MMCME2_BASE|BUFG|LUT|FD|CARRY4)" "$OUT/stat.txt" | head -20 || true
# Yosys cell models without the behaviour-less block RAMs (replaced by tb/sim_ramb.v)
"$PY" - "$CELLS" "$OUT/cells_sim.v" <<'EOF'
import os, re, sys
src = open(sys.argv[1]).read()
out = re.sub(r"(\(\*[^)]*\*\)\s*)*module (RAMB18E1|RAMB36E1)\b.*?endmodule", "", src, flags=re.S)
assert "module RAMB18E1" not in out and "module RAMB36E1" not in out
if not os.path.exists(sys.argv[2]) or open(sys.argv[2]).read() != out:
    open(sys.argv[2], "w").write(out)
EOF
SIMSRC=("$OUT/tb.v" "$OUT/netlist.v" tb/sim_mmcme2_base.v tb/sim_ramb.v "$OUT/cells_sim.v")
# the generator's token / hash registers under their RTL names: the testbench logs them as for the RTL
PROBES=0; PDEF=()
if grep -q 'wire \[15:0\] \\u_gen\.nxt ' "$OUT/netlist.v" && grep -q 'wire \[31:0\] \\u_gen\.hash ' "$OUT/netlist.v"; then
  PROBES=1; PDEF=(-DNETLIST_PROBES)
fi
sv2v -DNETLIST -DXCHECK "${PDEF[@]+"${PDEF[@]}"}" tb/tb_nexys.sv -w "$OUT/tb_new.v"
if cmp -s "$OUT/tb_new.v" "$OUT/tb.v"; then rm "$OUT/tb_new.v"; else mv "$OUT/tb_new.v" "$OUT/tb.v"; fi
if [ "$NETSIM" = icarus ]; then
  iverilog -g2012 -s tb_nexys -o "$OUT/sim.vvp" "${SIMSRC[@]}" > "$OUT/build.log" 2>&1 \
    || { tail -20 "$OUT/build.log"; exit 1; }
  CMD="vvp -n $OUT/sim.vvp"
else
  if [ ! -x "$OUT/obj/Vtb_nexys" ] || [ -n "$(find "${SIMSRC[@]}" -newer "$OUT/obj/Vtb_nexys" 2>/dev/null)" ]; then
    echo "building the netlist simulator (Verilator) ..."
    T0=$(date +%s)
    verilator --binary --timing --timescale 1ns/1ps -j "${VJOBS:-4}" -Wno-fatal -Wno-lint -Wno-style \
      -Wno-SPECIFYIGN -Wno-MULTIDRIVEN -Wno-COMBDLY -Wno-INITIALDLY -Wno-UNOPTFLAT -Wno-TIMESCALEMOD \
      --top-module tb_nexys "${SIMSRC[@]}" -Mdir "$OUT/obj" -o Vtb_nexys > "$OUT/build.log" 2>&1 \
      || { grep -m 20 "%Error" "$OUT/build.log"; exit 1; }
    echo "simulator build $(( $(date +%s) - T0 )) s"
  fi
  CMD="$OUT/obj/Vtb_nexys"
fi
if [ "${BUILD_ONLY:-0}" = 1 ]; then exit 0; fi
if [ "$NETSIM" = icarus ] && [ -n "${MAXCYC:-}" ]; then
  # 4-state check of the first MAXCYC clocks (reset, clear, the first steps): no X/Z on any pin
  vvp -n "$OUT/sim.vvp" +PRESS= +MAXCYC="$MAXCYC" > "$OUT/xcheck.log" 2>&1 || true
  grep -E "MMCME2|XERR|XCHECK|TIMEOUT|TBTOK" "$OUT/xcheck.log" | head -20
  grep -q "XCHECK unknown 0" "$OUT/xcheck.log"
  exit $?
fi
cd sw/zoo && NETSIM_CMD="$CMD" NETLIST_DIR="$OUT" NETLIST_PROBES="$PROBES" "$PY" board_sim.py "$M" --sim netlist "$@"
