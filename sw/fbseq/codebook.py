"""
Transition codebook (docs/10_teori.md §5.4): each block of kb state bits owns m learned transformations of its
2^kb states; per step the input network only chooses one of them (log2 m control bits). In hardware every
next-state bit is then one LUT over kb + log2 m inputs (kb=3, m=8: LUT6), whatever the transformations are.

    python3 codebook.py a5    [--seeds 0,1,2,3] [--m 8]   A5 word problem      -> model/fbseq/codebook_a5.json
    python3 codebook.py bytes [--seeds 0,1]     [--m 8]   Loghub byte LM       -> model/fbseq/codebook_bytes.json

Relaxation: the block keeps a distribution p over its states; the step matrix is the mixture
M_t = sum_k pi_{t,k} M_k of the codebook's column-stochastic matrices. Training is hard-forward
with the hardaware.py schedules: "softhard" (soft mixture for 2/3 of the steps, then hard) or "slope"
(forward argmax choice and 0/1 tables from the start, backward softmax / sigmoid of a*logit, a: 1 -> 5).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fbseq as cpu  # noqa: E402
import hardaware as ha  # noqa: E402

OUT = os.path.join(cpu.ROOT, "model", "fbseq")


class Codebook(cpu.Gated):
    def __init__(self, V, n_out, S=24, L=2, kb=3, m=8, w=1, n_embd=16, n_hidden=64, n_head=64, init="random",
                 gain=1.0):
        super().__init__(V, n_out, S=S, L=L, kb=kb, mode="full", w=w, n_embd=n_embd, n_hidden=n_hidden, n_head=n_head)
        N, Q = S // kb, 2 ** kb
        self.m, self.N = m, N
        n_win = w * n_embd
        self.gate = nn.ModuleList(nn.Sequential(nn.Linear(n_win + (S if l else 0), n_hidden), nn.Tanh(),
                                                nn.Linear(n_hidden, N * m)) for l in range(L))
        bits = ((torch.arange(Q)[None, :] >> torch.arange(kb)[:, None]) & 1).float()      # (kb, Q)
        self.register_buffer("qbits", bits, persistent=False)
        if init == "identity":                                                            # every entry near identity
            cb = (6.0 * bits - 3.0).expand(L, N, m, kb, Q) + 0.5 * torch.randn(L, N, m, kb, Q)
        else:
            cb = 3.0 * (2 * torch.randint(0, 2, (L, N, m, kb, Q)).float() - 1)           # random maps ...
            cb[:, :, 0] = 6.0 * bits - 3.0                                                # ... entry 0: identity
        self.gain = gain                                                                  # table logits = gain * cb:
        self.cb = nn.Parameter(cb / gain)                                                 # gain > 1 = faster tables
        self.layer_idx = 0

    def tables(self, logits):
        """choice logits (B, T, N*m) -> step matrices M_t (B, T, N, Q, Q)"""
        B, T = logits.shape[:2]
        N, m, Q = self.N, self.m, 2 ** self.kb
        z = logits.view(B, T, N, m)
        hard = self.hard if self.method == "softhard" else True          # schedule: hardaware.schedule
        scale = self.slope if self.method == "slope" else 1.0
        pi = torch.softmax(scale * z, -1)
        D = torch.sigmoid(scale * self.gain * self.cb[self.layer_idx])                    # (N, m, kb, Q)
        if hard:
            pi = pi + (F.one_hot(z.argmax(-1), m).float() - pi).detach()
            D = D + ((D > 0.5).float() - D).detach()
        nb = self.qbits.bool()[:, :, None]                                                # (kb, Q', 1)
        Mk = torch.where(nb, D[..., :, None, :], 1 - D[..., :, None, :]).prod(-3)          # (N, m, Q', Q)
        return (torch.einsum("btnk,nkij->btnij", pi, Mk),)

    def run(self, tab, B, T):
        N, Q = self.N, 2 ** self.kb
        Ms = tab[0].unbind(1)
        p = Ms[0].new_zeros(B, N, Q)
        p[..., 0] = 1.0
        out = []
        for t in range(T):
            p = torch.einsum("bnij,bnj->bni", Ms[t], p)
            out.append(torch.einsum("bnq,jq->bnj", p, self.qbits).reshape(B, self.S))
        return torch.stack(out, 1)

    def forward(self, idx):
        B, T = idx.shape
        win = self.window(idx)
        states, below = [], None
        for l in range(self.L):
            self.layer_idx = l
            tab = self.tables(self.gate[l](win if below is None else torch.cat([win, below], -1)))
            below = self.run(tab, B, T)
            states.append(below)
        self.last_states = states
        return self.head(torch.cat([win] + states, -1))


def a5_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    n_in, n_out = cpu.TASKS["a5"]
    model = Codebook(n_in, n_out, S=24, L=2, kb=3, m=job["m"], init=job.get("init", "random"), gain=job.get("gain", 1.0))
    erng = np.random.default_rng(1000 + job["seed"])
    ev = {L: [torch.as_tensor(a) for a in cpu.task("a5", 256, L, erng)] for L in (64, 256)}

    def batch():
        x, y = cpu.task("a5", 64, 64, rng)
        return torch.as_tensor(x), torch.as_tensor(y)

    def acc(model):
        return {L: float((model(x)[..., :n_out].argmax(-1) == y).float().mean()) for L, (x, y) in ev.items()}
    t0 = time.time()
    curve, gn, bad = ha.train(model, job["method"], batch, n_out, job["steps"], 3e-3, acc, 250)
    tag = f"codebook_k3_m{job['m']}_{job.get('init', 'random')}" + (f"_g{job['gain']:g}" if job.get("gain", 1.0) != 1.0 else "")
    return dict(task="a5", model=tag, method=job["method"],
                seed=job["seed"], steps=job["steps"],
                params=cpu.n_params(model), curve=[(s, round(a[64], 3), round(a[256], 3)) for s, a in curve],
                last_acc256=round(curve[-1][1][256], 4), best_acc256=round(max(a[256] for _, a in curve), 4),
                grad_norm_mean=round(float(np.mean(gn)), 4), nonfinite_steps=bad, seconds=round(time.time() - t0, 1))


def bytes_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    tr, va, _ = cpu.loghub()
    Xtr, Ytr = cpu.packets_xy(tr)
    Xva, Yva = cpu.packets_xy(va)
    model = Codebook(257, 256, S=120, L=2, kb=3, m=job["m"], w=4, n_embd=16, n_hidden=job.get("n_hidden", 52),
                     n_head=64)
    g = torch.Generator().manual_seed(job["seed"])
    sub = torch.arange(0, len(Xva), 4)

    def batch():
        ix = torch.randint(0, len(Xtr), (32,), generator=g)
        return Xtr[ix], Ytr[ix]

    def bpb(model, X=Xva[sub], Y=Yva[sub]):
        nll = 0.0
        for i in range(0, len(X), 64):
            lg = model(X[i:i + 64])[..., :256]
            nll += F.cross_entropy(lg.reshape(-1, 256), Y[i:i + 64].reshape(-1), reduction="sum").item()
        return nll / math.log(2) / Y.numel()
    t0 = time.time()
    curve, gn, bad = ha.train(model, job["method"], batch, 256, job["steps"], 3e-3, bpb, 500)
    ha.hard_eval(model)
    with torch.no_grad():
        final = bpb(model, Xva, Yva)
    return dict(model=f"codebook_k3_m{job['m']}", method=job["method"], seed=job["seed"], steps=job["steps"],
                params=cpu.n_params(model),
                bpb_hard=round(final, 4), curve=[(s, round(v, 4)) for s, v in curve],
                grad_norm_mean=round(float(np.mean(gn)), 4), nonfinite_steps=bad, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["a5", "bytes"])
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--methods", default="softhard,slope")
    ap.add_argument("--init", default="random", choices=["random", "identity"])
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [dict(seed=int(s), m=a.m, steps=a.steps, method=me, init=a.init, gain=a.gain)
            for me in a.methods.split(",") for s in a.seeds.split(",")]
    fn = a5_job if a.what == "a5" else bytes_job
    path = os.path.join(OUT, f"codebook_{a.what}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    with Pool(min(a.jobs, len(jobs))) as pool:
        for r in pool.imap_unordered(fn, jobs):
            print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
            k = lambda o: (o["model"], o.get("method", ""), o["seed"])
            out = [o for o in out if k(o) != k(r)] + [r]
            json.dump(sorted(out, key=k), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
