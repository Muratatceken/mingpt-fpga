"""
Data files for the PDF report's plots (report/data/*.dat):
  gelu_lut_l0.dat   layer-0 GELU LUT as real values (input q*s_h, output LUT*s_g)
  gelu_exact.dat    minGPT's tanh GELU on a fine grid
  logits.dat        float vs integer logits (teacher forced, subsampled)
  exp_lut.dat       2^(-k/256) LUT entries (UQ1.15 -> real)
"""
import os

import numpy as np
import torch

from common import ROOT, load_model, all_sort_inputs, SORT_LENGTH
from intmodel import IntGPT, load_qparams
from quantize import gelu

OUT = os.path.join(ROOT, "report", "data")


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(" ".join(f"{v:.6g}" for v in r) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    q = load_qparams()
    l0 = q["layers"][0]
    hq = np.arange(-128, 128)
    write("gelu_lut_l0.dat", "x y", zip(hq * l0["s_h"], l0["GELU_LUT"] * l0["s_g"]))
    xs = np.linspace(-128 * l0["s_h"], 127 * l0["s_h"], 400)
    write("gelu_exact.dat", "x y", zip(xs, gelu(xs)))
    write("exp_lut.dat", "k y", zip(np.arange(256), q["EXP_LUT"] / 2 ** 15))

    model = load_model()
    inputs = all_sort_inputs()[::5]                      # 146 sequences
    with torch.no_grad():
        full = model.generate(torch.tensor(inputs), SORT_LENGTH, do_sample=False)
        flog = model(full[:, :-1])[0].double().numpy()
    pts = []
    for i in range(len(inputs)):
        m = IntGPT(q)
        for t, tok in enumerate(full[i, :-1].tolist()):
            _, lg, _ = m.step(tok, t)
            for v in range(len(lg)):
                pts.append((flog[i, t, v], lg[v] * q["s_logit"]))
    write("logits.dat", "f i", pts[::2])
    print(f"wrote {OUT} ({len(pts[::2])} logit points)")


if __name__ == "__main__":
    main()
