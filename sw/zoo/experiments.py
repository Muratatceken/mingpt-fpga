"""
The names.txt training campaign (resumable; finished runs are skipped).
  python3 experiments.py [phase ...]       phases: lr50k seeds50k lr12k seeds12k s200k
Protocol: learning rate chosen per architecture on the validation set (seed 0, LR in {1e-3, 3e-3, 1e-2}),
then seeds 1 and 2 at that learning rate (3 seeds); 200k: the 50k learning rate, seeds 0 and 1.
Results are reported on the test set (mean +- std over seeds).
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
    """lr*: learning-rate sweep (seed 0); seeds*: seeds 1, 2 at the best LR; s200k: seeds 0, 1 at the
    best LR found at 50k (no sweep at 200k: compute budget)."""
    budget = dict(lr50k=50000, seeds50k=50000, lr12k=12000, seeds12k=12000, s200k=200000)[name]
    archs = FAMILY + list(LADDER) if budget == 50000 else FAMILY
    tag = f"names{budget // 1000}k"
    if name.startswith("lr"):
        train(tag, archs, budget, LRS, "0")
    elif name == "s200k":
        for a in archs:
            train(tag, [a], budget, f"{best_lr('names50k', a):g}", "0,1")
    else:
        for a in archs:                                   # one call per arch: each has its own best LR
            train(tag, [a], budget, f"{best_lr(tag, a):g}", "1,2")


def wait_for_running():
    """wait until no names training is running (e.g. a sweep started by an earlier driver)"""
    import time
    while subprocess.run(["pgrep", "-f", "train.py --tag names"], capture_output=True).stdout.strip():
        time.sleep(60)


if __name__ == "__main__":
    wait_for_running()
    for p in sys.argv[1:] or ["lr50k", "seeds50k", "lr12k", "seeds12k", "s200k"]:
        phase(p)
