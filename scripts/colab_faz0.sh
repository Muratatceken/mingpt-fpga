#!/usr/bin/env bash
# Phase 0 (feedback-constrained sequence models) on a Colab GPU, one command in a notebook cell:
#
#   !git clone -q --depth 1 https://github.com/Muratatceken/mingpt-fpga.git 2>/dev/null; cd mingpt-fpga && git pull -q && bash scripts/colab_faz0.sh
#
# Runs: classic baselines (gzip/xz/zstd/zstd+dict), the algebraic tasks and the byte LMs on Loghub, three
# seeds each (sw/fbseq/fbseq_gpu.py all), then prints the tables. Extra arguments go to fbseq_gpu.py,
# e.g. `bash scripts/colab_faz0.sh --jobs 6` or `--seeds 0,1,2,3,4`.
# Results: model/fbseq/gpu/*.json and /content/faz0_results.zip (Files panel -> download).
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi -L || echo "no GPU visible: Runtime -> Change runtime type -> GPU"
command -v zstd >/dev/null || { apt-get -qq update >/dev/null && apt-get -qq install -y zstd >/dev/null; }
bash scripts/get_loghub.sh >/dev/null
python3 sw/fbseq/fbseq_gpu.py check
python3 sw/fbseq/fbseq_gpu.py all "$@" | tee build_faz0.log
OUTZIP=/content/faz0_results.zip
[ -d /content ] || OUTZIP="$PWD/faz0_results.zip"
python3 - "$OUTZIP" <<'EOF'
import os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir("model/fbseq/gpu")):
        z.write(os.path.join("model/fbseq/gpu", f))
    z.write("build_faz0.log")
print("results:", sys.argv[1])
EOF
