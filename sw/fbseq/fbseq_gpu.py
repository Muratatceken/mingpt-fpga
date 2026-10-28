"""
CUDA version of fbseq.py (Colab, A100): the same candidates, tasks, data and metrics, trained in
parallel form. One command runs everything: scripts/colab_faz0.sh.

    python3 fbseq_gpu.py all|synth|bytes|base|summary [--seeds 0,1,2] [--steps N] [--batch N] [--jobs N]
                                                      [--device cuda|cpu]

Differences from fbseq.py (CPU)
  * B: the loop is evaluated as an associative scan over the sequence (log2 T steps):
      kb = 1  s_t = b_t + c_t * s_{t-1}, composed as (c, b) pairs (keep / set / toggle are all affine);
      kb > 1  the block state is a distribution over its 2^kb states, p_t = M_t p_{t-1}, composed as
              matrix products: the probabilistic relaxation of the automaton, as in fbseq.Gated.run;
              soft and hard outputs equal the CPU loop's (checked by `check`).
  * C: torch.nn.GRU (cuDNN) instead of the makemore cell, width matched to the same budget
  * several jobs share the GPU (--jobs), TF32 matmuls
Defaults keep the CPU protocol (batch, steps) so the two runs are comparable; three seeds.
Results -> model/fbseq/gpu/{synth,bytes,base}.json and tables on stdout (summary).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fbseq as cpu  # noqa: E402
from fbseq import ste, n_params  # noqa: E402

OUT = os.path.join(cpu.ROOT, "model", "fbseq", "gpu")


def linscan(b, c):
    """s_t = b_t + c_t s_{t-1}, s_{-1} = 0, for all t: Hillis-Steele over (c, b) pairs, dim 1 = time"""
    T, k = b.shape[1], 1
    while k < T:
        b = torch.cat([b[:, :k], c[:, k:] * b[:, :-k] + b[:, k:]], 1)
        c = torch.cat([c[:, :k], c[:, k:] * c[:, :-k]], 1)
        k *= 2
    return b


def matscan(M):
    """prefix products M_t ... M_0 for all t, dim 1 = time, matrices in the last two dims"""
    T, k = M.shape[1], 1
    while k < T:
        M = torch.cat([M[:, :k], M[:, k:] @ M[:, :-k]], 1)
        k *= 2
    return M


class GatedScan(cpu.Gated):
    """candidate B with the same parameters as fbseq.Gated, evaluated by scans"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        kb = self.kb
        bits = ((torch.arange(2 ** kb)[None, :] >> torch.arange(kb)[:, None]) & 1).float()   # (kb, Q)
        self.register_buffer("qbits", bits, persistent=False)

    def layer(self, d):
        B, T, S, kb = d.shape[0], d.shape[1], self.S, self.kb
        d = torch.sigmoid(d)
        if self.hard:
            d = ste(d)
        if kb == 1:
            d = d.view(B, T, S, 2)
            if self.mode == "setkeep":
                return linscan(d[..., 0] * d[..., 1], 1 - d[..., 0])
            return linscan(d[..., 1], d[..., 0] - d[..., 1])
        N, Q = S // kb, 2 ** kb
        d = d.view(B, T, N, kb, Q)                                   # d[j, q] = P(next bit j = 1 | state q)
        nb = self.qbits.bool()[:, :, None]                           # (kb, Q', 1): bit j of the next state
        M = torch.where(nb, d[..., :, None, :], 1 - d[..., :, None, :]).prod(-3)   # (B, T, N, Q', Q)
        p = matscan(M)[..., 0]                                       # start in state 0: column 0
        return torch.einsum("btnq,jq->btnj", p, self.qbits).reshape(B, T, S)       # E[bit j]

    def forward(self, idx):
        win = self.window(idx)
        states, below = [], None
        for l in range(self.L):
            below = self.layer(self.gate[l](win if below is None else torch.cat([win, below], -1)))
            states.append(below)
        return self.head(torch.cat([win] + states, -1))


class GRULM(nn.Module):
    def __init__(self, V, n_out, n_embd, n_hidden):
        super().__init__()
        self.wte = nn.Embedding(V, n_embd)
        self.gru = nn.GRU(n_embd, n_hidden, batch_first=True)
        self.head = nn.Linear(n_hidden, n_out)

    def forward(self, idx):
        return self.head(self.gru(self.wte(idx))[0])


def build(kind, V, n_out, T, **kw):
    if kind == "gated":
        return GatedScan(V, n_out, **kw)
    if kind == "gru":
        return GRULM(V, n_out, kw.get("n_embd", 16), kw["n_hidden"])
    return cpu.build(kind, V, n_out, T, **kw)


# the same models as the CPU run; the cuDNN GRU gets its own width for the same budget
SYNTH_MODELS = dict(cpu.SYNTH_MODELS)
BYTE_MODELS = dict(cpu.BYTE_MODELS)
BYTE_MODELS["C_gru"] = dict(kind="gru", n_embd=32, n_hidden=124)


def fit(model, batch, n_out, steps, lr, hard_from=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    soft_loss = None
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        if hard_from is not None and step == hard_from and hasattr(model, "hard"):
            model.hard = True
        x, y = batch()
        loss = F.cross_entropy(model(x)[..., :n_out].reshape(-1, n_out), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if hard_from is not None and step == hard_from - 1:
            soft_loss = loss.item()
    return soft_loss


def setup(job):
    torch.manual_seed(job["seed"])
    if job["device"] == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device(job["device"])


def synth_job(job):
    dev = setup(job)
    rng = np.random.default_rng(job["seed"])
    n_in, n_out = cpu.TASKS[job["task"]]
    spec = dict(SYNTH_MODELS[job["model"]])
    kind = spec.pop("kind")
    model = build(kind, n_in, n_out, T=256, **spec).to(dev)

    def batch():
        x, y = cpu.task(job["task"], job["batch"], 64, rng)
        return torch.as_tensor(x, device=dev), torch.as_tensor(y, device=dev)
    t0 = time.time()
    fit(model, batch, n_out, job["steps"], 3e-3, hard_from=(2 * job["steps"]) // 3 if kind == "gated" else None)
    model.eval()
    res = dict(task=job["task"], model=job["model"], seed=job["seed"], params=n_params(model),
               seconds=round(time.time() - t0, 1), batch=job["batch"], steps=job["steps"])
    erng = np.random.default_rng(1000 + job["seed"])
    with torch.no_grad():
        for L in (64, 256):
            x, y = cpu.task(job["task"], 256, L, erng)
            pred = model(torch.as_tensor(x, device=dev))[..., :n_out].argmax(-1).cpu().numpy()
            res[f"acc{L}"] = float((pred == y).mean())
    return res


def byte_job(job):
    dev = setup(job)
    tr, va, _ = cpu.loghub()
    Xtr, Ytr = (t.to(dev) for t in cpu.packets_xy(tr))
    Xva, Yva = (t.to(dev) for t in cpu.packets_xy(va))
    spec = dict(BYTE_MODELS[job["model"]])
    kind = spec.pop("kind")
    model = build(kind, 257, 256, T=cpu.PACKET, **spec).to(dev)
    g = torch.Generator(device=dev).manual_seed(job["seed"])

    def batch():
        ix = torch.randint(0, len(Xtr), (job["batch"],), generator=g, device=dev)
        return Xtr[ix], Ytr[ix]
    t0 = time.time()
    soft = fit(model, batch, 256, job["steps"], 3e-3, hard_from=(2 * job["steps"]) // 3 if kind == "gated" else None)
    model.eval()
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(Xva), 128):
            lg = model(Xva[i:i + 128])[..., :256]
            nll += F.cross_entropy(lg.reshape(-1, 256), Yva[i:i + 128].reshape(-1), reduction="sum").item()
    return dict(model=job["model"], seed=job["seed"], params=n_params(model),
                bpb=round(nll / math.log(2) / Yva.numel(), 4),
                soft_train_bpb=None if soft is None else round(soft / math.log(2), 4),
                batch=job["batch"], steps=job["steps"], seconds=round(time.time() - t0, 1))


def run_jobs(fn, jobs, path, n_proc):
    old = json.load(open(path)) if os.path.exists(path) else []
    key = lambda x: (x.get("task", ""), x["model"], x["seed"])

    def keep(r):
        nonlocal old
        print(json.dumps(r), flush=True)
        old = [o for o in old if key(o) != key(r)] + [r]
        json.dump(sorted(old, key=lambda x: tuple(str(v) for v in key(x))), open(path, "w"), indent=1)
    if n_proc <= 1:
        for j in jobs:
            keep(fn(j))
        return
    ctx = torch.multiprocessing.get_context("spawn")          # CUDA needs spawned workers
    with ctx.Pool(n_proc) as pool:
        for r in pool.imap_unordered(fn, jobs):
            keep(r)


def summary():
    def stats(v):
        return f"{np.mean(v):.4f} ± {np.std(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
    p = os.path.join(OUT, "synth.json")
    if os.path.exists(p):
        R = json.load(open(p))
        tasks = list(cpu.TASKS)
        print("\n## Algebraic tasks: accuracy at length 256 (trained at 64), mean over seeds")
        print("| model | " + " | ".join(tasks) + " |\n|---|" + "---|" * len(tasks))
        for m in SYNTH_MODELS:
            row = [np.mean([r["acc256"] for r in R if r["model"] == m and r["task"] == t] or [np.nan]) for t in tasks]
            print(f"| {m} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    p = os.path.join(OUT, "bytes.json")
    if os.path.exists(p):
        R = json.load(open(p))
        print("\n## Loghub, independent 256-byte packets: bits per byte (val)")
        print("| model | params | bpb | seeds |\n|---|---|---|---|")
        for m in BYTE_MODELS:
            v = [r["bpb"] for r in R if r["model"] == m]
            if v:
                n = [r["params"] for r in R if r["model"] == m][0]
                print(f"| {m} | {n} | {stats(v)} | {len(v)} |")
    p = os.path.join(OUT, "base.json")
    if os.path.exists(p):
        b = json.load(open(p))
        print("\n## Classic compressors, same packets: bits per byte")
        print("| method | bpb |\n|---|---|")
        for k, v in b.items():
            if k.endswith("_bpb"):
                print(f"| {k[:-4]} | {v} |")


def check():
    """GatedScan (scan) == fbseq.Gated (loop), soft and hard, for every kb, on random weights"""
    torch.manual_seed(0)
    for kb, mode in ((1, "setkeep"), (1, "full"), (2, "full"), (3, "full")):
        a = cpu.Gated(7, 5, S=12, L=2, kb=kb, mode=mode, w=2)
        b = GatedScan(7, 5, S=12, L=2, kb=kb, mode=mode, w=2)
        b.load_state_dict(a.state_dict())
        for m in (a, b):
            for g in m.gate:
                g[-1].bias.data.normal_(0, 3)                        # tables far from the identity
        b.load_state_dict(a.state_dict())
        x = torch.randint(0, 7, (4, 50))
        for hard in (False, True):
            a.hard = b.hard = hard
            ok = torch.allclose(a(x), b(x), atol=1e-4)
            print(f"kb={kb} {mode:7s} scan == loop ({'hard' if hard else 'soft'}): {ok}")
            assert ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "synth", "bytes", "base", "summary", "check"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--models", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "check":
        return check()
    if a.what == "summary":
        return summary()
    t0 = time.time()
    if a.what in ("all", "base"):
        json.dump(cpu.baselines(), open(os.path.join(OUT, "base.json"), "w"), indent=1)
    if a.what in ("all", "synth"):
        models = a.models.split(",") if a.models else list(SYNTH_MODELS)
        jobs = [dict(task=t, model=m, seed=s, steps=a.steps or 2000, batch=a.batch or 64, device=a.device)
                for t in cpu.TASKS for m in models for s in seeds]
        run_jobs(synth_job, jobs, os.path.join(OUT, "synth.json"), a.jobs)
    if a.what in ("all", "bytes"):
        models = a.models.split(",") if a.models else list(BYTE_MODELS)
        jobs = [dict(model=m, seed=s, steps=a.steps or 3000, batch=a.batch or 32, device=a.device)
                for m in models for s in seeds]
        run_jobs(byte_job, jobs, os.path.join(OUT, "bytes.json"), a.jobs)
    print(f"\n{a.what}: {time.time() - t0:.0f} s on {a.device}")
    summary()


if __name__ == "__main__":
    main()
