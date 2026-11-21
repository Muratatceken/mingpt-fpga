#!/usr/bin/env bash
# Decision RTL unit tests (guard exhaustive, tokenizer, rule, FSM, guard watchdog) against Python golden vectors.
#   scripts/dec_units.sh            (Verilator)      SIM=icarus scripts/dec_units.sh   (sv2v + Icarus, 4-state)
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"                 # the numpy/torch interpreter, before PATH changes
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
( cd sw/decision && "$PY" gen_rtl_pkg.py >/dev/null && "$PY" gen_rtl_vectors.py )
SRC=(rtl/decision/dec_pkg.sv rtl/decision/dec_guard.sv rtl/decision/dec_policies.sv tb/tb_dec_units.sv)
SIM="${SIM:-verilator}"
OBJ="build/decision/obj_units_${SIM}"
mkdir -p "$OBJ"
if [ "$SIM" = icarus ]; then
  sv2v "${SRC[@]}" -w "$OBJ/sim.v"
  iverilog -g2012 -o "$OBJ/sim.vvp" "$OBJ/sim.v"
  vvp -n "$OBJ/sim.vvp" | grep -E "RESULT|PASS|FAIL|MISMATCH"
else
  verilator --binary --timing --assert -j 0 -Wall -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
    -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-TIMESCALEMOD -Wno-PROCASSINIT "${SRC[@]}" --top-module tb_dec_units \
    -Mdir "$OBJ" -o Vtb > "$OBJ/build.log" 2>&1 || { tail -20 "$OBJ/build.log"; exit 1; }
  "$OBJ/Vtb" | grep -E "RESULT|PASS|FAIL|MISMATCH"
fi
