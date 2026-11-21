"""Integer export of a decision model (WP4.1-4.2): compile -> calibrate (train only) -> INT8 -> engine images.

    python3 export_dec.py --run model/decision/runs/v1/lm_lr0.003_s0 --name lm_s0
    python3 export_dec.py --random lm --name lm_rand            # pipeline smoke test with random weights
Writes mem/decision/<name>/ (engine images, tv.txt golden vectors on validation scenarios, zoo_pkg.sv),
build/decision/<name>.qprog.pkl and mem/decision/<name>/export.json:
  calibration provenance (training episodes only), float-vs-INT8 decision agreement on the validation pairs
  (and the same split into float->INT8 action changes), contract hash.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "zoo"))
from export_hw import export  # noqa: E402
from ir import FloatRunner, compile_model  # noqa: E402
from quant import IntRunner, calibrate, quantize  # noqa: E402

from contract import PAD, ROOT, contract_hash  # noqa: E402
from models_dec import SPECS, DecisionLlama, build, periodic_rope  # noqa: E402
from train_policy import load_val  # noqa: E402

DATA = os.path.join(ROOT, "data", "decision")


def compile_decision(m):
    prog = compile_model("decision", m)
    if isinstance(m, DecisionLlama):
        cos, sin = periodic_rope(m.dh)
        prog.rope = (cos.double().numpy(), sin.double().numpy())
        for o in prog.ops:
            if o["op"] == "ATTN":
                o["win"] = int(m.W)
    else:
        prog.rope = None
    return prog


def decide_np(l4):
    return 3 - int(np.argmax(l4[::-1]))


def agreement(prog, q, seq, acc, meta, limit=None):
    """float vs INT8 masked-argmax decision at the decision step of each validation scenario."""
    n = len(seq) if limit is None else min(limit, len(seq))
    same = okf = oki = 0
    for j in range(n):
        pos = 4 + 5 * meta[j]["t0"]
        fr, ir = FloatRunner(prog), IntRunner(q)
        for t in range(pos + 1):
            lf = fr.step(int(seq[j, t]), t)
            li = ir.step(int(seq[j, t]), t)
        af, ai = decide_np(lf[:4]), decide_np(li[:4])
        same += af == ai
        okf += bool(acc[j, af])
        oki += bool(acc[j, ai])
    return dict(n=n, agree=same / n, float_acc=okf / n, int_acc=oki / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--random", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ncal", type=int, default=300)
    ap.add_argument("--ntv", type=int, default=8)
    ap.add_argument("--nagree", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.run:
        rec = json.load(open(os.path.join(ROOT, a.run) + ".json"))
        spec = rec["spec"]
        m = build(spec)
        m.load_state_dict(torch.load(os.path.join(ROOT, a.run) + ".pt", map_location="cpu"))
    else:
        spec = SPECS[a.random]
        torch.manual_seed(0)
        m = build(spec)
    m.eval()
    prog = compile_decision(m)
    # calibration: training episodes only (never validation / test), truncated to the engine context
    z = np.load(os.path.join(DATA, "train", "train.npz"))
    X = z["X"]
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(X), a.ncal, replace=False)
    cal = []
    for j in idx:
        x = X[j][X[j] != PAD][:prog.T]
        cal.append([int(v) for v in x])
    rec_cal = calibrate(prog, cal)
    q = quantize(prog, rec_cal)
    os.makedirs(os.path.join(ROOT, "build", "decision"), exist_ok=True)
    pickle.dump(q, open(os.path.join(ROOT, "build", "decision", f"{a.name}.qprog.pkl"), "wb"))
    vseq, vy, vacc, vmeta = load_val()
    ag = agreement(prog, q, vseq, vacc, vmeta, limit=a.nagree)
    # golden vectors: validation scenarios (whole prefix up to the decision MODE token, < T = 256)
    tv = []
    for j in range(0, min(len(vseq), 2 * a.ntv), 2):
        pos = 4 + 5 * vmeta[j]["t0"]
        tv.append([int(v) for v in vseq[j, :pos + 1]])
    out = os.path.join(ROOT, "mem", "decision", a.name)
    export(a.name, tv, q=q, out=out, quiet=True)
    info = dict(name=a.name, run=a.run, random=a.random, spec=spec, contract_sha256=contract_hash(),
                calibration=dict(source="train.npz", n=a.ncal, rng=12345,
                                 indices_sha256=hashlib.sha256(np.asarray(idx).tobytes()).hexdigest()),
                agreement_val=ag, tv=dict(n=len(tv), lengths=[len(x) for x in tv]), seconds=round(time.time() - t0, 1))
    json.dump(info, open(os.path.join(out, "export.json"), "w"), indent=1)
    print(json.dumps(dict(name=a.name, agreement=ag, out=out), indent=1))


if __name__ == "__main__":
    main()
