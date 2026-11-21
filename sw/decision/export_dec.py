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
from multiprocessing import Pool

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


_W = {}


def _init(prog, q, seq, meta):
    _W.update(prog=prog, q=q, seq=seq, meta=meta)


def _one(j):
    prog, q, seq, meta = _W["prog"], _W["q"], _W["seq"], _W["meta"]
    pos = 4 + 5 * meta[j]["t0"]
    fr, ir = FloatRunner(prog), IntRunner(q)
    for t in range(pos + 1):
        lf = fr.step(int(seq[j, t]), t)
        li = ir.step(int(seq[j, t]), t)
    lf4 = np.asarray(lf[:4], dtype=float)
    top2 = np.sort(lf4)[::-1]
    return decide_np(lf4), decide_np(np.asarray(li[:4])), float(top2[0] - top2[1])


def agreement(prog, q, seq, acc, meta, limit=None, jobs=1):
    """float vs INT8 masked-argmax decision at the decision step of each validation scenario.
    Also: float top-2 logit margin of every disagreement, and agreement when the float margin is >= 0.2."""
    n = len(seq) if limit is None else min(limit, len(seq))
    if jobs > 1:
        with Pool(jobs, initializer=_init, initargs=(prog, q, seq, meta)) as p:
            res = p.map(_one, range(n), chunksize=8)
    else:
        _init(prog, q, seq, meta)
        res = [_one(j) for j in range(n)]
    af = np.array([r[0] for r in res])
    ai = np.array([r[1] for r in res])
    mg = np.array([r[2] for r in res])
    okf, oki = acc[np.arange(n), af], acc[np.arange(n), ai]
    dis = np.flatnonzero(af != ai)
    wide = mg >= 0.2
    return dict(n=n, agree=float((af == ai).mean()), float_acc=float(okf.mean()), int_acc=float(oki.mean()),
                margin_q05=float(np.quantile(mg, 0.05)), n_disagree=int(len(dis)),
                agree_margin_ge_0p2=float((af[wide] == ai[wide]).mean()) if wide.any() else None,
                n_margin_ge_0p2=int(wide.sum()),
                disagreements=[dict(j=int(j), template=meta[j]["template"], d=int(meta[j]["d"]),
                                    member=meta[j]["member"], float_act=int(af[j]), int_act=int(ai[j]),
                                    float_margin=round(float(mg[j]), 4), float_ok=bool(okf[j]),
                                    int_ok=bool(oki[j])) for j in dis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--random", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ncal", type=int, default=300)
    ap.add_argument("--ntv", type=int, default=8)
    ap.add_argument("--nagree", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=3)
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
    ag = agreement(prog, q, vseq, vacc, vmeta, limit=a.nagree, jobs=a.jobs)
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
    print(json.dumps(dict(name=a.name, agreement={k: v for k, v in ag.items() if k != "disagreements"}, out=out), indent=1))


if __name__ == "__main__":
    main()
