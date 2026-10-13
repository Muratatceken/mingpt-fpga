"""
Compiler check: the float IR program of every trained model must reproduce PyTorch logits.
  python3 check_ir.py [model ...]
"""
import os
import sys

import numpy as np
import torch

from data import ROOT, load
from ir import attach_rope, compile_model, float_logits
from models import ZOO

CKPT = os.path.join(ROOT, "model", "zoo")


def load_trained(name):
    m = ZOO[name]()
    m.load_state_dict(torch.load(os.path.join(CKPT, f"{name}.pt"), map_location="cpu"))
    m.eval()
    return m


def main():
    d = load()
    names = sys.argv[1:] or list(ZOO)
    X = d["test"][0][:20]
    for name in names:
        m = load_trained(name)
        prog = attach_rope(compile_model(name, m))
        with torch.no_grad():
            ref = m(X).double().numpy()
        err = max(np.abs(float_logits(prog, X[i].tolist()) - ref[i]).max() for i in range(len(X)))
        print(f"{name:9s} ops/token={len(prog.ops):3d}  max |IR - torch| = {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
