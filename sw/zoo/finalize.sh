#!/usr/bin/env bash
# Final steps once the training campaigns are done (each step is resumable):
#   1. pipeline.py      INT8 + bit-exact RTL for every trained run (test set)
#   2. hw_final.py      promote the best-LR seed-0 names50k family to the main models; Yosys of the final models
#   3. main models      evaluate.py (INT8 metrics), export_hw.py (mem/zoo/<arch>), universal one-bitstream images
#   4. hw_run.py        RTL (generic + specialized) and Yosys of the main models; AXI system test of all models
#   5. perf_model.py, paper_data.py, report_zoo.py, PDFs
#   sw/zoo/finalize.sh  2>&1 | tee model/zoo/runs/finalize.log
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
export OMP_NUM_THREADS=1 VJOBS=4
echo "== 1 pipeline";              python3 pipeline.py
echo "== 2 promote + synth";       python3 hw_final.py promote && python3 hw_final.py synth names50k stories260k
echo "== 3 main models";           python3 check_ir.py && python3 evaluate.py && python3 stories.py && python3 export_hw.py
python3 export_hw.py --universal bigram mlp rnn gru bow gpt2 microgpt llama nanochat stories260k
echo "== 4 RTL + Yosys";           python3 hw_run.py --synth
(cd "$ROOT" && PRELOAD=0 scripts/zoo_synth.sh mem/zoo/universal)
python3 - <<'PY'
import json, subprocess, sys
p = "../../model/zoo/hw_metrics.json"
d = json.load(open(p))
y = json.loads(subprocess.run([sys.executable, "../../scripts/yosys_summary.py",
                               "../../build/zoo/yosys/stat_mem_zoo_universal.txt", "u", "--json"],
                              capture_output=True, text=True).stdout)
d["universal"] = dict(yosys=y, note="one-bitstream engine, PRELOAD=0, LOADABLE=1, all units")
json.dump(d, open(p, "w"), indent=1)
print("universal:", y)
PY
for m in bigram mlp rnn gru bow gpt2 microgpt llama nanochat stories260k; do
  (cd "$ROOT" && scripts/zoo_axi_sim.sh "mem/zoo/universal/$m" | grep -E "AXI|PASS|FAIL")
done
echo "== 5 data + PDFs";           python3 perf_model.py && python3 paper_data.py && python3 report_zoo.py
(cd "$ROOT" && make report && make paper) > /dev/null 2>&1 || echo "LaTeX build: see report/build and paper/*.log"
echo "== done"
