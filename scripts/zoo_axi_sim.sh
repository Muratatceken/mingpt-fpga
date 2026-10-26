#!/usr/bin/env bash
# System simulation of the AXI4-Lite engine (what the PYNQ driver does on the board).
#   scripts/zoo_axi_sim.sh <model image dir> [NSEQ]     e.g. mem/zoo/universal/gpt2
# Uses the universal (one-bitstream) package mem/zoo/universal/zoo_pkg.sv.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:$PATH"
# macOS: if xcrun picks a Command Line Tools SDK newer than the selected Xcode's linker, use Xcode's SDK
XSDK="$(xcode-select -p 2>/dev/null)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
if [ "$(uname)" = Darwin ] && [ -z "${SDKROOT:-}" ] && [ -d "$XSDK" ] && \
   ! xcrun --show-sdk-path 2>/dev/null | grep -q "^$(xcode-select -p)"; then export SDKROOT="$XSDK"; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OBJ="$ROOT/build/zoo/obj_axi"
mkdir -p "$OBJ"
verilator --binary --timing --assert --timescale 1ns/1ps -j 0 -Wall -Wno-fatal \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  mem/zoo/universal/zoo_pkg.sv rtl/zoo/zrom.sv rtl/zoo/zdiv.sv rtl/requant.sv rtl/isqrt_seq.sv rtl/zoo/zoo_core.sv \
  rtl/zoo/zoo_axi.sv tb/tb_zoo_axi.sv --top-module tb_zoo_axi -Mdir "$OBJ" -o Vtb > "$OBJ/build.log" 2>&1 \
  || { tail -40 "$OBJ/build.log"; exit 1; }
"$OBJ/Vtb" +MODELDIR="${1%/}" +NSEQ="${2:-1000000}"
