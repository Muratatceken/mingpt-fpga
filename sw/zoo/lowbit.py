"""
Low-bit weights for the zoo models: Larq-style binary / ternary networks and LittleBit, trained on the
Mac with quantisation-aware training (QAT) and knowledge distillation from the float model. The FPGA only
runs inference; this study decides which weight format the engine should support.

  python3 lowbit.py <model> <method> [--steps N] [--lr X] [--seed S]      one run -> model/zoo/lowbit/<model>/<method>.json
  python3 lowbit.py sweep <model> [...]                                    every method of METHODS
  python3 lowbit.py table                                                  summary of all runs

Methods (weights of every linear layer inside the network; token embedding and output head stay INT8,
as in LittleBit and in most binary-network work; activations stay INT8 in the engine)
  int8      per-output-channel INT8 (the engine today), no training
  int4      per-channel symmetric INT4, QAT with a straight-through estimator (STE)
  tern      ternary {-a, 0, +a} per output channel, threshold 0.7 mean|w| (TWN; larq.quantizers.SteTern with
            ternary_weight_networks=True), STE clipped at |w| <= 1
  bin       binary {-a, +a}, a = mean|w| per output channel (larq.quantizers.MagnitudeAwareSign, Bi-Real Net),
            STE clipped at |w| <= 1 (larq.quantizers.SteSign), latent weights clipped to [-1, 1] (larq weight_clip)
  lb<r>p<k> LittleBit (Lee et al., NeurIPS 2025): W ~ diag(h) sign(U) diag(l) sign(V)^T diag(g), rank r = <r>
            percent of min(d_out, d_in), k paths (k = 2: residual path on W - W_pri); Dual-SVID initialisation
  Every method also reports its post-training value (step 0: rounding / Dual-SVID only, no training).
Training: Adam, cosine learning rate, loss = KL(teacher || student) on the next-token distribution.
BPW counts sign bits and 16-bit scales of the quantised layers (LittleBit: h, g, l per path; binary /
ternary / INT: one scale per output channel).
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import ROOT, load

OUT = os.path.join(ROOT, "model", "zoo", "lowbit")
METHODS = ["int8", "int4", "tern", "bin", "lb50p2", "lb25p2", "lb25p1", "lb12p2"]
SCALE_BITS = 16


# ----------------------------------------------------------------------------- quantisers (Larq, in PyTorch)
class SteSign(torch.autograd.Function):
    """larq.quantizers.SteSign: forward sign (sign(0) = +1), backward grad * 1{|x| <= clip}"""

    @staticmethod
    def forward(ctx, x, clip):
        ctx.save_for_backward(x)
        ctx.clip = clip
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return (g if ctx.clip is None else g * (x.abs() <= ctx.clip).to(g.dtype)), None


def ste_sign(x, clip=1.0):
    return SteSign.apply(x, clip)


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def q_int(w, bits):
    """symmetric per-output-channel integer quantisation (absmax scale), STE"""
    qmax = 2 ** (bits - 1) - 1
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.clamp(ste_round(w / s), -qmax, qmax) * s


def q_tern(w):
    """ternary weight networks: delta = 0.7 mean|w|, alpha = mean |w| over |w| > delta (per output channel)"""
    a = w.detach().abs()
    delta = 0.7 * a.mean(dim=1, keepdim=True)
    mask = (a > delta).to(w.dtype)
    alpha = (a * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    t = ste_sign(w) * mask                            # STE through the sign, the mask is a constant
    return alpha * t


def q_bin(w):
    """magnitude-aware sign: mean|w| per output channel (stop-gradient) times ste_sign(w)"""
    alpha = w.detach().abs().mean(dim=1, keepdim=True)
    return alpha * ste_sign(w)


class QLinear(nn.Module):
    """nn.Linear with fake-quantised weights (latent float weights are trained)"""

    def __init__(self, lin, method):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone())
        self.method = method

    def qweight(self):
        w = self.weight
        return dict(int8=lambda: q_int(w, 8), int4=lambda: q_int(w, 4), tern=lambda: q_tern(w),
                    bin=lambda: q_bin(w))[self.method]()

    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)

    def bits(self):
        o, i = self.weight.shape
        wb = dict(int8=8, int4=4, tern=math.log2(3), bin=1)[self.method]
        return o * i * wb + o * SCALE_BITS

    def clip_latent(self):
        if self.method in ("bin", "tern"):
            with torch.no_grad():
                self.weight.clamp_(-1.0, 1.0)


# ----------------------------------------------------------------------------- LittleBit
def dual_svid(W, r):
    """Dual-SVID: W ~ U S V^T (rank r); U0 = U sqrt(S), V0 = V sqrt(S); |U0| ~ h lu^T, |V0| ~ g lv^T (rank 1)
    -> latent U0, V0 (their signs are the binary factors) and scales h, g, l = lu * lv"""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U0 = U[:, :r] * S[:r].sqrt()
    V0 = Vh[:r].T * S[:r].sqrt()

    def rank1(A):
        u, s, vh = torch.linalg.svd(A.abs(), full_matrices=False)
        a, b = u[:, 0] * s[0].sqrt(), vh[0] * s[0].sqrt()
        if a.sum() < 0:                                # Perron vector of a non-negative matrix: one sign
            a, b = -a, -b
        return a.clamp_min(1e-8), b.clamp_min(1e-8)

    h, lu = rank1(U0)
    g, lv = rank1(V0)
    return U0, V0, h, g, lu * lv


class LBLinear(nn.Module):
    """LittleBit layer: sum over paths of diag(h) sign(U) diag(l) sign(V)^T diag(g)"""

    def __init__(self, lin, rank_pct, paths):
        super().__init__()
        W = lin.weight.detach().double()
        o, i = W.shape
        self.r = max(1, int(round(rank_pct / 100 * min(o, i))))
        self.paths = paths
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone())
        self.U, self.V, self.h, self.g, self.l = (nn.ParameterList() for _ in range(5))
        R = W
        for _ in range(paths):
            U0, V0, h, g, l = dual_svid(R, self.r)
            for lst, t in ((self.U, U0), (self.V, V0), (self.h, h), (self.g, g), (self.l, l)):
                lst.append(nn.Parameter(t.float()))
            R = R - self.path(len(self.U) - 1).detach().double()
        self.shape = (o, i)

    def path(self, k):
        su, sv = ste_sign(self.U[k], None), ste_sign(self.V[k], None)
        return (self.h[k][:, None] * su * self.l[k][None, :]) @ (sv * self.g[k][:, None]).T

    def qweight(self):
        return sum(self.path(k) for k in range(self.paths))

    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)

    def bits(self):
        o, i = self.shape
        return self.paths * (self.r * (o + i) + SCALE_BITS * (o + i + self.r))

    def clip_latent(self):
        pass


def is_head(name, mod, V):
    return name.endswith(("lm_head", "output", "fc2")) and mod.weight.shape[0] == V


def quantise(model, method, V):
    """copy of the model with every linear layer except the output head replaced"""
    m = copy.deepcopy(model)
    names = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, nn.Linear) and not is_head(n, mod, V)]
    for n, mod in names:
        parent = m.get_submodule(n.rsplit(".", 1)[0]) if "." in n else m
        key = n.rsplit(".", 1)[-1]
        if method.startswith("lb"):
            r, p = method[2:].split("p")
            new = LBLinear(mod, int(r), int(p))
        else:
            new = QLinear(mod, method)
        if isinstance(parent, nn.ModuleDict):
            parent[key] = new
        else:
            setattr(parent, key, new)
    return m


def qlayers(m):
    return [mod for mod in m.modules() if isinstance(mod, (QLinear, LBLinear))]


def materialise(m):
    """plain float model whose linear weights are the quantised ones (for the integer pipeline / RTL)"""
    out = copy.deepcopy(m)
    for n, mod in list(out.named_modules()):
        if isinstance(mod, (QLinear, LBLinear)):
            lin = nn.Linear(mod.qweight().shape[1], mod.qweight().shape[0], bias=mod.bias is not None)
            with torch.no_grad():
                lin.weight.copy_(mod.qweight())
                if mod.bias is not None:
                    lin.bias.copy_(mod.bias)
            parent = out.get_submodule(n.rsplit(".", 1)[0]) if "." in n else out
            key = n.rsplit(".", 1)[-1]
            if isinstance(parent, nn.ModuleDict):
                parent[key] = lin
            else:
                setattr(parent, key, lin)
    return out


# ----------------------------------------------------------------------------- data, teacher
def setup(model):
    """float teacher, data access and evaluation for a model of the zoo"""
    if model == "stories260k" or model.startswith("ts_"):
        from tinystories import Batches, val_windows
        if model == "stories260k":
            from stories import load_model
            t = load_model()
        else:
            from configs import build
            run = json.load(open(os.path.join(ROOT, "model", "zoo", "runs", "stories260k", f"{model[3:]}_lr0.003_s0.json")))
            t = build(run["spec"])
            t.load_state_dict(torch.load(run["path"] + ".pt", map_location="cpu"))
        V, T = 512, 256
        gen = Batches("train", T, seed=11)
        Xt, Yt = val_windows(T, "test")
        Xt, Yt = Xt[:100], Yt[:100]
        batch = lambda B: gen.get(B)
        test = (Xt, Yt)
    else:
        from check_ir import load_trained
        d = load()
        t = load_trained(model)
        V = d["V"]
        Xtr, Ytr = d["train"]
        g = torch.Generator().manual_seed(11)

        def batch(B):
            ix = torch.randint(0, len(Xtr), (B,), generator=g)
            return Xtr[ix], Ytr[ix]
        test = d["test"]
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t, V, batch, test


def nll(m, X, Y, dev, bs=50):
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, y = X[i:i + bs].to(dev), Y[i:i + bs].to(dev)
            lg = m(x)
            tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-1, reduction="sum").item()
            n += (y >= 0).sum().item()
    m.train()
    return tot / n


def run(model, method, steps=None, lr=None, seed=0, dev=None):
    torch.manual_seed(seed)
    stories = model == "stories260k" or model.startswith("ts_")
    dev = dev or ("mps" if stories and torch.backends.mps.is_available() else "cpu")
    steps = steps if steps is not None else (1500 if stories else 3000)
    lr = lr or (2e-3 if method.startswith("lb") or method in ("bin", "tern") else 5e-4)
    B = 32 if stories else 64
    teacher, V, batch, (Xt, Yt) = setup(model)
    teacher = teacher.to(dev)
    t0 = time.time()
    res = dict(model=model, method=method, steps=0, lr=lr, seed=seed, device=dev)
    res["float_nll"] = nll(teacher, Xt, Yt, dev)
    s = quantise(teacher.cpu(), method, V).to(dev)
    teacher = teacher.to(dev)
    for p in s.parameters():
        p.requires_grad_(True)
    for mod in s.modules():                         # embeddings and the head stay as they are (INT8 later)
        if isinstance(mod, nn.Embedding) or (isinstance(mod, nn.Linear)):
            for p in mod.parameters():
                p.requires_grad_(False)
    ql = qlayers(s)
    n_w = sum(mod.qweight().numel() for mod in ql)
    res["qbits"] = sum(mod.bits() for mod in ql)
    res["qweights"] = n_w
    res["bpw"] = res["qbits"] / n_w
    res["ptq_nll"] = nll(s, Xt, Yt, dev)
    curve = [(0, res["ptq_nll"])]
    if method != "int8" and steps > 0:
        params = [p for p in s.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * k / steps)))
        for k in range(1, steps + 1):
            x, y = batch(B)
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                pt = F.log_softmax(teacher(x), -1)
            ps = F.log_softmax(s(x), -1)
            mask = (y >= 0).float()
            kl = ((pt.exp() * (pt - ps)).sum(-1) * mask).sum() / mask.sum()
            opt.zero_grad()
            kl.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            for mod in ql:
                mod.clip_latent()
            if k % max(1, steps // 10) == 0 or k == steps:
                curve.append((k, nll(s, Xt, Yt, dev)))
                print(f"  {model} {method} step {k}: kl {kl.item():.4f} test {curve[-1][1]:.4f}", flush=True)
        res["steps"] = steps
    res["qat_nll"] = nll(s, Xt, Yt, dev)
    res["curve"] = curve
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(OUT, model), exist_ok=True)
    json.dump(res, open(os.path.join(OUT, model, f"{method}.json"), "w"), indent=1)
    torch.save(materialise(s).cpu().state_dict(), os.path.join(OUT, model, f"{method}.pt"))
    if method.startswith("lb"):                     # the factors themselves, for the FPGA (ir.factor_littlebit)
        fac = {}
        for n, mod in s.named_modules():
            if isinstance(mod, LBLinear):
                fac[n] = [dict(U=torch.where(mod.U[k] >= 0, 1.0, -1.0).cpu(), V=torch.where(mod.V[k] >= 0, 1.0, -1.0).cpu(),
                               h=mod.h[k].detach().cpu(), g=mod.g[k].detach().cpu(), l=mod.l[k].detach().cpu())
                          for k in range(mod.paths)]
        torch.save(fac, os.path.join(OUT, model, f"{method}.factors.pt"))
    print(f"{model:12s} {method:7s} BPW {res['bpw']:6.3f}  float {res['float_nll']:.4f}  PTQ {res['ptq_nll']:.4f}  "
          f"QAT {res['qat_nll']:.4f}  ({res['seconds']} s)", flush=True)
    return res


def table(md=False):
    rows = []
    for model in sorted(os.listdir(OUT)):
        if not os.path.isdir(os.path.join(OUT, model)):
            continue
        for f in sorted(os.listdir(os.path.join(OUT, model))):
            if f.endswith(".json"):
                rows.append(json.load(open(os.path.join(OUT, model, f))))
    if md:                                           # markdown: one row per model, QAT loss (BPW) per method
        models = sorted({r["model"] for r in rows}, key=lambda m: (m.startswith("stories"), m))
        by = {(r["model"], r["method"]): r for r in rows}
        print("| Model | float | " + " | ".join(METHODS) + " |")
        print("|---|---:|" + "---:|" * len(METHODS))
        for m in models:
            fl = next(r["float_nll"] for r in rows if r["model"] == m)
            cells = []
            for k in METHODS:
                r = by.get((m, k))
                cells.append(f"{r['qat_nll']:.3f} ({r['bpw']:.2f})".replace(".", ",") if r else "")
            print(f"| {m} | {fl:.3f} | ".replace(".", ",") + " | ".join(cells) + " |")
        return
    for r in sorted(rows, key=lambda r: (r["model"], -r["bpw"])):
        print(f"{r['model']:12s} {r['method']:7s} BPW {r['bpw']:6.3f}  float {r['float_nll']:.4f}  "
              f"PTQ {r['ptq_nll']:.4f}  QAT {r['qat_nll']:.4f}  Δ {r['qat_nll'] - r['float_nll']:+.4f}")


def main():
    a = sys.argv[1:]
    opt = lambda k, d, f=int: f(a[a.index(k) + 1]) if k in a else d
    if a[0] == "table":
        table(md="--md" in a)
    elif a[0] == "sweep":
        for model in [x for x in a[1:] if not x.startswith("--")]:
            for m in METHODS:
                run(model, m, steps=opt("--steps", None), lr=opt("--lr", None, float))
    else:
        run(a[0], a[1], steps=opt("--steps", None), lr=opt("--lr", None, float), seed=opt("--seed", 0))


if __name__ == "__main__":
    main()
