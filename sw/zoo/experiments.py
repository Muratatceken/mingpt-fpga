"""
The names.txt training campaign (resumable; finished runs are skipped).
  python3 experiments.py [phase ...]       phases: lr50k seeds50k lr12k seeds12k lr200k seeds200k
Protocol: learning rate chosen per architecture on the validation set (seed 0, LR in {1e-3, 3e-3, 1e-2}),
then extra seeds at that learning rate; results are reported on the test set (mean +- std over seeds).
"""
import glob
import json
import os
import subprocess
import sys

from configs import FAMILY, LADDER
from data import ROOT

RUNS = os.path.join(ROOT, "model", "zoo", "runs")
LRS = "1e-3,3e-3,1e-2"


def best_lr(tag, arch):
    rs = [json.load(open(f)) for f in glob.glob(os.path.join(RUNS, tag, f"{arch}_lr*_s0.json"))]
    return min(rs, key=lambda r: r["val_loss"])["lr"] if rs else None


def train(tag, archs, budget, lrs, seeds):
    cmd = [sys.executable, "train.py", "--tag", tag, "--arch", ",".join(archs), "--budget", str(budget),
           "--lrs", lrs, "--seeds", seeds, "--jobs", "4", "--threads", "2"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def phase(name):
    budget = dict(lr50k=50000, seeds50k=50000, lr12k=12000, seeds12k=12000, lr200k=200000, seeds200k=200000)[name]
    archs = FAMILY + list(LADDER) if budget == 50000 else FAMILY
    tag = f"names{budget // 1000}k"
    if name.startswith("lr"):
        train(tag, archs, budget, LRS, "0")
    else:
        seeds = "1,2,3,4" if budget == 50000 else "1,2"
        for a in archs:                                   # one call per arch: each has its own best LR
            train(tag, [a], budget, f"{best_lr(tag, a):g}", seeds)


if __name__ == "__main__":
    for p in sys.argv[1:] or ["lr50k", "seeds50k", "lr12k", "seeds12k", "lr200k", "seeds200k"]:
        phase(p)
