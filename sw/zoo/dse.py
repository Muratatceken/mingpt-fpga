"""
Design-space exploration of the GEMV lane count P (attention/KV lanes stay at the head size 8):
for P in {8, 16, 32} export selected models, check them bit-exact in Verilator and synthesise the
engine with Yosys (all units, loadable memories).
  python3 dse.py [--P 8,16,32] [model ...]
Results: model/zoo/dse.json   (cycles per token and resources per P and model)
"""
import json
import os
import re
import subprocess
import sys

from data import ROOT

MODELS = ["mlp", "gru", "gpt2", "llama", "nanochat", "stories260k"]


def main():
    args = sys.argv[1:]
    Ps = [8, 16, 32]
    if "--P" in args:
        k = args.index("--P")
        Ps = [int(x) for x in args[k + 1].split(",")]
        args = args[:k] + args[k + 2:]
    models = args or MODELS
    path = os.path.join(ROOT, "model", "zoo", "dse.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for P in Ps:
        root = os.path.join("build", "zoo", "dse", f"P{P}")
        subprocess.run([sys.executable, "export_hw.py", "--out-root", root, "--nseq", "10"] + models,
                       env=dict(os.environ, ZOO_P=str(P)), check=True, capture_output=True)
        for m in models:
            memdir = os.path.join(root, m)
            out = subprocess.run([os.path.join(ROOT, "scripts", "zoo_sim.sh"), memdir], capture_output=True,
                                 text=True, cwd=ROOT).stdout
            r = re.search(r"RESULT \S+ steps (\d+) bad_steps (\d+) errors (\d+) avg_cycles (\d+)", out)
            prof = {k: int(v) for k, v in re.findall(r"^PROF \S+ (\S+) (\d+)", out, re.M)}
            subprocess.run([os.path.join(ROOT, "scripts", "zoo_synth.sh"), memdir], capture_output=True, cwd=ROOT)
            stat = os.path.join(ROOT, "build", "zoo", "yosys", f"stat_{memdir.replace('/', '_')}.txt")
            y = json.loads(subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "yosys_summary.py"), stat,
                                           m, "--json"], capture_output=True, text=True).stdout)
            steps = int(r.group(1))
            e = dict(P=P, model=m, steps=steps, errors=int(r.group(3)), cycles=int(r.group(4)),
                     gemv=round(prof.get("GEMV", 0) / steps, 1), yosys=y)
            res[f"{m}_P{P}"] = e
            json.dump(res, open(path, "w"), indent=1)
            print(f"P={P:2d} {m:12s} errors {e['errors']} cycles {e['cycles']:6d} (GEMV {e['gemv']:7.1f}) "
                  f"LUT {y['lut']} FF {y['ff']} DSP {y['dsp']} BRAM {y['bram']} depth {y.get('depth')}", flush=True)


if __name__ == "__main__":
    main()
