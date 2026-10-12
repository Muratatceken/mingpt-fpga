#!/usr/bin/env bash
# Non-project xsim run (Linux, Vivado settings64.sh sourced):
#   scripts/xsim_batch.sh [NSEQ]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NSEQ="${1:-20}"
OUT="$ROOT/build/xsim"
mkdir -p "$OUT" && cd "$OUT"

# compile a copy of gpt_pkg.sv whose MEM_PATH is absolute (xsim runs in build/xsim)
sed "s|localparam string MEM_PATH = \"[^\"]*\";|localparam string MEM_PATH = \"$ROOT/mem\";|" \
    "$ROOT/rtl/gpt_pkg.sv" > gpt_pkg.sv
SRCS="gpt_pkg.sv $(grep -v gpt_pkg "$ROOT/scripts/files.f" | sed "s|^|$ROOT/|")"

xvlog -sv $SRCS "$ROOT/tb/tb_gpt_top.sv"
xelab tb_gpt_top -s tb_gpt_top_sim -timescale 1ns/1ps
xsim tb_gpt_top_sim -R -testplusarg "NSEQ=$NSEQ"
