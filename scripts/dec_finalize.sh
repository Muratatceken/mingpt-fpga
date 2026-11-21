#!/usr/bin/env bash
# After training (make dec-train): promote the seeds, INT8 export, engine and decision_top RTL checks, board
# self-test simulation, synthesis, place & route, gate-level netlist simulation, validation evaluation and the
# Project I report.
#   scripts/dec_finalize.sh                 (all models; log: build/decision/finalize_lm_gru.log)
#   MODELS=gru scripts/dec_finalize.sh      (one family; log: build/decision/finalize_gru.log)
# proje1_report.py collects the RESULT lines of every build/decision/finalize_*.log.
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"
export PYTHON="$PY"
MODELS="${MODELS:-lm gru}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p build/decision
LOG="build/decision/finalize_$(echo $MODELS | tr ' ' _).log"
exec > >(tee "$LOG") 2>&1
tag() { sed "s/^/[$1] /"; }
echo "== promote"
for m in $MODELS; do for s in 0 1 2; do
  src="model/decision/runs/v1/${m}_lr0.001_s${s}"
  cp "$src.pt" "model/decision/${m}_s${s}.pt"; cp "$src.json" "model/decision/${m}_s${s}.json"
done; done
echo "== INT8 export (float <-> INT8 agreement on all 960 validation scenarios)"
for m in $MODELS; do for s in 0 1 2; do
  ( cd sw/decision && "$PY" export_dec.py --run "model/decision/${m}_s${s}" --name "${m}_s${s}" | tail -14 )
done; done
export PATH="/opt/homebrew/bin:$PATH"
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
for m in $MODELS; do
  x="${m}_s0"
  echo "== $x: engine RTL vs integer reference (all 64 logits)"
  scripts/zoo_sim.sh "mem/decision/$x" | grep -E "RESULT|PASS|FAIL" | tag "$x"
  echo "== $x: decision_top vs hw_ref"
  scripts/dec_top.sh "$x" "$m" 24 | tag "$x"                  # also leaves obs.mem + exp_hash.mem for this engine
  echo "== $x: board self-test (simulation)"
  OBJ="build/decision/obj_board_$x"; mkdir -p "$OBJ"
  verilator --binary --timing -j 0 -Wno-fatal -Wno-lint -Wno-style -DSIM "mem/decision/$x/zoo_pkg.sv" \
    rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv rtl/decision/dec_pkg.sv \
    rtl/decision/dec_guard.sv rtl/decision/dec_policies.sv rtl/decision/decision_top.sv rtl/decision/dec_board_top.sv \
    tb/tb_dec_board.sv --top-module tb_dec_board -GPOLICY=0 -Mdir "$OBJ" -o Vtb > "$OBJ/build.log" 2>&1 \
    || { tail -20 "$OBJ/build.log"; exit 1; }
  "$OBJ/Vtb" | grep -E "RESULT|PASS|FAIL" | tag "$x"
  echo "== $x: synthesis and place & route (25 MHz)"
  scripts/dec_synth.sh engine "$x" 2>&1 | grep -v "^Warning" | tail -2
  scripts/dec_pnr.sh "$x" "$m" 25 2>&1 | grep -E "MHz|SLICE_LUTX|SLICE_FFX|RAMB36|DSP" | tag "$x" || true
  echo "== $x: gate-level simulation of the placed netlist"
  scripts/dec_netlist_sim.sh "$x" "$m" || echo "[$x netlist] FAIL"
done
echo "== validation evaluation and report"
( cd sw/decision && "$PY" evaluate_dec.py > /dev/null && "$PY" proje1_report.py )
echo "== done"
