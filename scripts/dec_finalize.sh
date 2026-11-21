#!/usr/bin/env bash
# After training (make dec-train): promote the seeds, INT8 export, engine and decision_top RTL checks, board
# self-test simulation, synthesis, place & route, validation evaluation and the Project I report.
#   scripts/dec_finalize.sh            (log: build/decision/finalize.log)
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"
export PYTHON="$PY"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG=build/decision/finalize.log
mkdir -p build/decision
exec > >(tee "$LOG") 2>&1
echo "== promote"
for m in lm gru; do for s in 0 1 2; do
  src="model/decision/runs/v1/${m}_lr0.001_s${s}"
  cp "$src.pt" "model/decision/${m}_s${s}.pt"; cp "$src.json" "model/decision/${m}_s${s}.json"
done; done
echo "== INT8 export (float <-> INT8 agreement: all 960 validation scenarios for seed 0, 240 for seeds 1-2)"
for m in lm gru; do for s in 0 1 2; do
  N=""; [ "$s" != 0 ] && N="--nagree 240"
  ( cd sw/decision && "$PY" export_dec.py --run "model/decision/${m}_s${s}" --name "${m}_s${s}" $N | tail -9 )
done; done
echo "== engine RTL vs integer reference (all 64 logits)"
for x in lm_s0 gru_s0; do scripts/zoo_sim.sh "mem/decision/$x" | grep -E "RESULT|PASS|FAIL"; done
echo "== decision_top vs hw_ref (trained engines)"
scripts/dec_top.sh lm_s0 lm 24
scripts/dec_top.sh gru_s0 gru 24
echo "== synthesis"
scripts/dec_synth.sh engine lm_s0
scripts/dec_synth.sh engine gru_s0
echo "== board self-test (simulation) and place & route, 25 MHz"
export PATH="/opt/homebrew/bin:$PATH"
for x in lm_s0 gru_s0; do
  pol=lm; [ "$x" = gru_s0 ] && pol=gru
  scripts/dec_top.sh "$x" "$pol" 24 > /dev/null                 # obs.mem + exp_hash.mem for this engine
  OBJ="build/decision/obj_board_$x"; mkdir -p "$OBJ"
  verilator --binary --timing -j 0 -Wno-fatal -Wno-lint -Wno-style -DSIM "mem/decision/$x/zoo_pkg.sv" \
    rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv rtl/decision/dec_pkg.sv \
    rtl/decision/dec_guard.sv rtl/decision/dec_policies.sv rtl/decision/decision_top.sv rtl/decision/dec_board_top.sv \
    tb/tb_dec_board.sv --top-module tb_dec_board -GPOLICY=0 -Mdir "$OBJ" -o Vtb > "$OBJ/build.log" 2>&1
  "$OBJ/Vtb" | grep -E "RESULT|PASS|FAIL"
  scripts/dec_pnr.sh "$x" "$pol" 25 || true
done
echo "== validation evaluation and report"
( cd sw/decision && "$PY" evaluate_dec.py > /dev/null && "$PY" proje1_report.py )
echo "== done"
