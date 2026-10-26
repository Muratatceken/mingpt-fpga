#!/usr/bin/env bash
# 4-state simulation of the whole Nexys A7-100T design (board/nexys_a7/nexys_a7_top.sv, -DSIM) for one model:
# sv2v + Icarus with the board testbench (tb/tb_nexys.sv -DXCHECK). After the first reset no output pin
# (VGA_*, LED*, AN, CA..CG, DP) may be X/Z in any cycle, nor the text RAM write port (address, data) while
# its write enable is high; tokens, hashes, on-board check, text area, status row and 7-segment display are
# compared with sw/zoo/board.py as in the Verilator run (sw/zoo/board_sim.py --sim icarus).
#   scripts/nexys_icarus.sh <model> [board_sim.py options]          (default: one automatic run + C)
# Output: build/nexys/<model>_icarus/{sim.log, screen.txt}, simulator in build/nexys/<model>/obj_icarus/
set -euo pipefail
PY="${PYTHON:-$(command -v python3)}"               # the Python with numpy; board_sim.py finds sv2v / Icarus
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$1"; shift
if [ $# -eq 0 ]; then set -- --press C; fi
cd "$ROOT/sw/zoo" && "$PY" board_sim.py "$M" --sim icarus "$@"
