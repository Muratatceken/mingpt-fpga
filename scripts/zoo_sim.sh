#!/usr/bin/env bash
# Bit-exact simulation of the microcoded zoo engine for one exported model.
#   scripts/zoo_sim.sh <model | memdir> [NSEQ] [spec]
#     model   name under mem/zoo/ (e.g. gpt2), or any directory written by export_hw.py
#     spec    engine with only the units the model uses (SPECIALIZE=1)
# Environment:
#   LOAD=1    memories start empty and are loaded through the ld_* port (LOADABLE=1, PRELOAD=0)
#   COVER=1   Verilator line/toggle coverage -> <objdir>/coverage.dat
#   SIM=icarus  4-state simulation (sv2v + Icarus) with X checks on every logit
#   TV=<file> golden vectors (default <memdir>/tv.txt)
#   MODEL=<dir> model images loaded into the engine (with LOAD=1), e.g. for the universal engine:
#             LOAD=1 MODEL=mem/zoo/universal/gpt2 scripts/zoo_sim.sh universal
# Verilator runs with --assert (address ranges, accumulator overflow, divider precondition).
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"     # ARM Verilator/Yosys/sv2v/Icarus (macOS)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
M="$1"; NSEQ="${2:-1000000}"
case "$M" in */*) MEM="${M%/}" ;; *) MEM="mem/zoo/$M" ;; esac
NAME="$(basename "$MEM")"
SPEC=0; if [ "${3:-}" = spec ]; then SPEC=1; fi
LOAD="${LOAD:-0}"; COVER="${COVER:-0}"; SIM="${SIM:-verilator}"
TAG="$NAME"
if [ "$SPEC" = 1 ]; then TAG="${TAG}_spec"; fi
if [ "$LOAD" = 1 ]; then TAG="${TAG}_load"; fi
if [ "$COVER" = 1 ]; then TAG="${TAG}_cov"; fi
OBJ="$ROOT/build/zoo/obj_${TAG}_${SIM}"
mkdir -p "$OBJ"
SRC=(rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv tb/tb_zoo.sv)
PLUS=(+NSEQ="$NSEQ")
GDIR="$MEM"; if [ -n "${MODEL:-}" ]; then GDIR="${MODEL%/}"; PLUS+=(+MODELDIR="$GDIR"); fi
if [ -f "$GDIR/gen.txt" ]; then PLUS+=(+GEN); fi            # closed-loop greedy generation check
if [ -n "${TV:-}" ]; then PLUS+=(+TV="$TV"); fi

if [ "$SIM" = icarus ]; then
  sv2v -DXCHECK "$MEM/zoo_pkg.sv" "${SRC[@]}" -w "$OBJ/sim.v" > "$OBJ/build.log" 2>&1 || { tail -20 "$OBJ/build.log"; exit 1; }
  iverilog -g2012 -s tb_zoo -P tb_zoo.SPECIALIZE=$SPEC -P tb_zoo.LOADTEST=$LOAD -o "$OBJ/sim.vvp" "$OBJ/sim.v" \
    >> "$OBJ/build.log" 2>&1 || { tail -20 "$OBJ/build.log"; exit 1; }
  vvp -n "$OBJ/sim.vvp" "${PLUS[@]}"
  exit 0
fi

COV=(--x-assign fast)                                      # (bash 3.2: no empty arrays under set -u)
if [ "$COVER" = 1 ]; then COV+=(--coverage-line --coverage-toggle); fi
verilator --binary --timing --assert --timescale 1ns/1ps -j 0 -Wall -Wno-fatal \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM "${COV[@]}" \
  -GSPECIALIZE=$SPEC -GLOADTEST=$LOAD "$MEM/zoo_pkg.sv" "${SRC[@]}" --top-module tb_zoo \
  -Mdir "$OBJ" -o Vtb_zoo > "$OBJ/build.log" 2>&1 || { tail -40 "$OBJ/build.log"; exit 1; }
if [ "$COVER" = 1 ]; then PLUS+=(+verilator+coverage+file+"$OBJ/coverage.dat"); fi
"$OBJ/Vtb_zoo" "${PLUS[@]}"                                   # from the repo root: MEM_PATH is relative
