"""Float vs INT8 logit error along the token stream (WP4.2 diagnosis of the S6 agreement gate).

At every MODE position (step k, token 4 + 5k) of each validation scenario, up to its decision step, the four action
logits of the float program (ir.FloatRunner) are compared with the integer program (quant.IntRunner) dequantized by
its logit scale. A recurrent state (GRU) carries its quantization error forward; the banded LM recomputes its context
from INT8 K/V entries that are at most 127 tokens old. The error per step range shows which of the two happens.

    python3 int8_drift.py                       # every promoted model with an export (mem/decision/<m>/export.json)
    python3 int8_drift.py --models gru_s2 --stride 10
Output: docs/decision/sonuclar/int8_drift.json
"""
import argparse
import glob
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
from ir import FloatRunner  # noqa: E402
from quant import IntRunner  # noqa: E402

from contract import ROOT  # noqa: E402
from export_dec import compile_decision  # noqa: E402
from models_dec import build  # noqa: E402
from train_policy import load_val  # noqa: E402

OUT = os.path.join(ROOT, "docs", "decision", "sonuclar", "int8_drift.json")
BINS = ((0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48))
_W = {}


def _init(prog, q, seq, meta):
    _W.update(prog=prog, q=q, seq=seq, meta=meta)


def _one(j):
    prog, q, seq, meta = _W["prog"], _W["q"], _W["seq"], _W["meta"]
    pos = 4 + 5 * meta[j]["t0"]
    fr, ir = FloatRunner(prog), IntRunner(q)
    err = []
    for t in range(pos + 1):
        lf = fr.step(int(seq[j, t]), t)
        li = ir.step(int(seq[j, t]), t)
        if t >= 4 and (t - 4) % 5 == 0:
            a = np.asarray(lf[:4], dtype=float)
            b = np.asarray(li[:4], dtype=float) * q.s_logit
            err.append(float(np.abs(a - b).max()))
    return err


def drift(name, seq, meta, stride, jobs):
    rec = json.load(open(os.path.join(ROOT, "model", "decision", f"{name}.json")))
    m = build(rec["spec"])
    m.load_state_dict(torch.load(os.path.join(ROOT, "model", "decision", f"{name}.pt"), map_location="cpu"))
    m.eval()
    prog = compile_decision(m)
    q = pickle.load(open(os.path.join(ROOT, "build", "decision", f"{name}.qprog.pkl"), "rb"))
    idx = list(range(0, len(seq), stride))
    with Pool(jobs, initializer=_init, initargs=(prog, q, seq, meta)) as p:
        res = p.map(_one, idx, chunksize=8)
    bins = {}
    for lo, hi in BINS:
        e = np.array([x for er in res for k, x in enumerate(er) if lo <= k < hi])
        if len(e):
            bins[f"{lo}-{hi - 1}"] = dict(n=int(len(e)), median=float(np.median(e)), q95=float(np.quantile(e, 0.95)),
                                          max=float(e.max()))
    return dict(scenarios=len(idx), stride=stride, bins=bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    names = a.models.split(",") if a.models else sorted(
        os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(ROOT, "mem", "decision", "*", "export.json"))
        if not json.load(open(f)).get("random"))
    seq, _, _, meta = load_val()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for n in names:
        t0 = time.time()
        out[n] = drift(n, seq, meta, a.stride, a.jobs)
        b = out[n]["bins"]
        print(f"{n}: q95 |dlogit| " + "  ".join(f"{k} {v['q95']:.3f}" for k, v in b.items()) + f"  ({time.time() - t0:.0f} s)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(sorted(out.items())), open(OUT, "w"), indent=1)
    print(OUT)


if __name__ == "__main__":
    main()
