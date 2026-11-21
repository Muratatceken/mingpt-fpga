#!/usr/bin/env bash
# decision_top (ROM replay -> tokenizer -> policy -> guard -> trace) against sw/decision/hw_ref.py.
#   scripts/dec_top.sh <model> [lm|rule|fsm|all] [n_scen]
#   <model>: an export under mem/decision/<model> (its zoo_pkg.sv sizes the engine; also needed for rule/fsm builds)
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
M="$1"; WHICH="${2:-all}"; NS="${3:-24}"
MEM="mem/decision/$M"
[ -f "$MEM/zoo_pkg.sv" ] || { echo "missing $MEM (run export_dec.py first)"; exit 1; }
SRC=("$MEM/zoo_pkg.sv" rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv
     rtl/decision/dec_pkg.sv rtl/decision/dec_guard.sv rtl/decision/dec_policies.sv rtl/decision/decision_top.sv
     tb/tb_dec_top.sv)
pols="$WHICH"; [ "$WHICH" = all ] && pols="lm rule fsm"
for p in $pols; do
  case "$p" in lm) PID=0 ;; rule) PID=1 ;; fsm) PID=2 ;; *) echo "bad policy $p"; exit 1 ;; esac
  ( cd sw/decision && "$PY" hw_ref.py "$M" "$p" "$NS" >/dev/null )
  OBJ="build/decision/obj_top_${M}_${p}"
  mkdir -p "$OBJ"
  verilator --binary --timing --assert -j 0 -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN \
    "${SRC[@]}" --top-module tb_dec_top -GPOLICY=$PID -GTRACE="\"build/decision/trace_${p}.txt\"" \
    -Mdir "$OBJ" -o Vtb > "$OBJ/build.log" 2>&1 || { tail -30 "$OBJ/build.log"; exit 1; }
  "$OBJ/Vtb" | grep -E "RESULT|PASS|FAIL|MISMATCH|rror" | head -12
done
