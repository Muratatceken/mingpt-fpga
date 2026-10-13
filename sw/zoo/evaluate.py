"""
Quantise every trained model and compare float vs. integer quality on the test set.
  python3 evaluate.py [model ...]
Outputs: build/zoo/<name>.qprog.pkl (integer program for export) and model/zoo/int_metrics.json
"""
import json
import os
import pickle
import sys
import time

import numpy as np

from check_ir import load_trained
from data import ROOT, load
from ir import FloatRunner, attach_rope, compile_model
from models import ZOO
from quant import IntRunner, calibrate, nll, quantize

BUILD = os.path.join(ROOT, "build", "zoo")
N_CALIB = 2000


def seqs_of(X, Y):
    """teacher-forced (input, target) pairs per name, up to and including the end token."""
    out = []
    for x, y in zip(X.tolist(), Y.tolist()):
        n = sum(1 for v in y if v >= 0)
        out.append((x[:n], y[:n]))
    return out


def evaluate(name, d):
    t0 = time.time()
    m = load_trained(name)
    prog = attach_rope(compile_model(name, m))
    train = seqs_of(*d["train"])[:N_CALIB]
    rec = calibrate(prog, [x for x, _ in train])
    q = quantize(prog, rec)
    tot_f = tot_i = 0.0
    n = agree = 0
    for x, y in seqs_of(*d["test"]):
        fr, ir = FloatRunner(prog), IntRunner(q)
        for t, (tok, tgt) in enumerate(zip(x, y)):
            lf = fr.step(tok, t)
            li = ir.step(tok, t) * q.s_logit
            tot_f += nll(lf, tgt)
            tot_i += nll(li, tgt)
            agree += int(np.argmax(lf) == np.argmax(li))
            n += 1
    os.makedirs(BUILD, exist_ok=True)
    pickle.dump(q, open(os.path.join(BUILD, f"{name}.qprog.pkl"), "wb"))
    res = dict(float_nll=tot_f / n, int_nll=tot_i / n, delta=(tot_i - tot_f) / n, argmax_agree=agree / n,
               tokens=n, FR=q.FR, ops=len(q.ops), seconds=round(time.time() - t0, 1),
               h0_clipped=any(o.get("h0_clipped", False) for o in q.ops))
    print(f"{name:9s} float {res['float_nll']:.4f}  int8 {res['int_nll']:.4f}  "
          f"Δ {res['delta']:+.4f}  argmax uyumu {100 * res['argmax_agree']:.1f}%  ({res['seconds']} s)", flush=True)
    return res


def main():
    d = load()
    names = sys.argv[1:] or [n for n in ZOO if os.path.exists(os.path.join(ROOT, "model", "zoo", f"{n}.pt"))]
    path = os.path.join(ROOT, "model", "zoo", "int_metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        metrics[name] = evaluate(name, d)
        json.dump(metrics, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
