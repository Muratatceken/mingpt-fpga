#!/usr/bin/env bash
# Bit-exact Verilator simulation of the microcoded zoo engine for one model.
#   scripts/zoo_sim.sh <model> [NSEQ] [spec]
#   spec: build the engine with only the units the model uses (SPECIALIZE=1)
# Needs mem/zoo/<model>/ (cd sw/zoo && python3 export_hw.py <model>).
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"     # ARM Verilator/Yosys/sv2v (macOS)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$1"; NSEQ="${2:-1000000}"; SPEC=0; TAG=""
[ "${3:-}" = spec ] && { SPEC=1; TAG=_spec; }
OBJ="$ROOT/build/zoo/obj_$M$TAG"
cd "$ROOT"
mkdir -p "$OBJ"
verilator --binary --timing --timescale 1ns/1ps -j 0 -Wall -Wno-fatal \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -GSPECIALIZE=$SPEC \
  "mem/zoo/$M/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv \
  rtl/zoo/zoo_core.sv tb/tb_zoo.sv --top-module tb_zoo -Mdir "$OBJ" -o Vtb_zoo > "$OBJ/build.log" 2>&1 \
  || { tail -40 "$OBJ/build.log"; exit 1; }
GEN=""; [ -f "mem/zoo/$M/gen.txt" ] && GEN=+GEN      # closed-loop greedy generation check
"$OBJ/Vtb_zoo" +NSEQ="$NSEQ" $GEN
