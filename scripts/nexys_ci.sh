#!/usr/bin/env bash
# Board-level regression without Python: the whole Nexys A7 design (tb/tb_nexys.sv) runs the automatic first
# run after reset (sampling, fresh seed) and a greedy run (SW0 up + BTNU); both runs must end with the
# on-board check OK, i.e. the FPGA's token hash equals the one sw/zoo/board.py computed at export.
#   scripts/nexys_ci.sh <model> [<model> ...]          (mem/nexys/<model>/ must exist)
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" || true
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SRC="rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv board/nexys_a7/bin2bcd.sv
     board/nexys_a7/vga_text.sv board/nexys_a7/seg7.sv board/nexys_a7/lm_gen.sv board/nexys_a7/nexys_a7_top.sv tb/tb_nexys.sv"
fail=0
for M in "$@"; do
  MEM="mem/nexys/$M"; OBJ="build/nexys/$M/ci"
  mkdir -p "$OBJ"
  # shellcheck disable=SC2086
  verilator --binary --timing --assert --timescale 1ns/1ps -j 0 -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -DSIM \
    --x-assign fast "$MEM/zoo_pkg.sv" "$MEM/board_pkg.sv" $SRC --top-module tb_nexys -Mdir "$OBJ" -o V \
    > "$OBJ/build.log" 2>&1 || { tail -30 "$OBJ/build.log"; exit 1; }
  "$OBJ/V" +PRESS=U +SW0=1 +SCREEN="$OBJ/screen.txt" > "$OBJ/sim.log"
  runs=$(grep -c "^RUN [01] check 1 " "$OBJ/screen.txt" || true)
  if [ "$runs" = 2 ]; then
    echo "PASS $M: $(grep -c GENTOK "$OBJ/sim.log") tokens, on-board check OK in the sampled and the greedy run"
  else
    echo "FAIL $M: $(grep '^RUN' "$OBJ/screen.txt" | tr '\n' ' ')"; fail=1
  fi
done
exit $fail
