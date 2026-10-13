#!/usr/bin/env bash
# Bit-exact Verilator simulation of the microcoded zoo engine for one model.
#   scripts/zoo_sim.sh <model> [NSEQ]
# Needs mem/zoo/<model>/ (cd sw/zoo && python3 export_hw.py <model>).
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"     # ARM Verilator/Yosys/sv2v (macOS)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$1"; NSEQ="${2:-1000000}"
OBJ="$ROOT/build/zoo/obj_$M"
cd "$ROOT"
mkdir -p "$OBJ"
verilator --binary --timing --timescale 1ns/1ps -j 0 -Wall -Wno-fatal \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  "mem/zoo/$M/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/requant.sv rtl/div_seq.sv rtl/isqrt_seq.sv \
  rtl/zoo/zoo_core.sv tb/tb_zoo.sv --top-module tb_zoo -Mdir "$OBJ" -o Vtb_zoo > "$OBJ/build.log" 2>&1 \
  || { tail -40 "$OBJ/build.log"; exit 1; }
"$OBJ/Vtb_zoo" +NSEQ="$NSEQ"
