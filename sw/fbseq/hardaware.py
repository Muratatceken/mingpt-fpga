"""
Hard-aware training of candidate B: the forward pass is a true Boolean machine from the first step,
only the backward pass is a surrogate. Compared with the soft -> hard schedule of fbseq.py.

    python3 hardaware.py bytes [--seeds 0,1]    Loghub byte LM          -> model/fbseq/hardaware_bytes.json
    python3 hardaware.py a5    [--seeds 0,1]    A5 word problem, B_block3 -> model/fbseq/hardaware_a5.json

Methods (all evaluated with the deterministic hard forward)
  softhard   fbseq.py: soft (multilinear) for 2/3 of the steps, then hard tables with straight-through
  gumbel     hard forward of the noisy logit (logistic = binary Gumbel noise), surrogate sigmoid((l+n)/tau),
             tau annealed 1 -> 0.25 (Gumbel-ST, as in "Mind the Gap", NeurIPS 2025, for feed-forward nets)
  hardst     deterministic hard forward, surrogate sigmoid(l) (straight-through from the first step)
  slope      deterministic hard forward, surrogate sigmoid(a * l), a annealed 1 -> 5 (HM-RNN slope annealing)
Logged: hard val bpb (best of the periodic checks and final), gate saturation (|logit| > 3), mean gradient
norm, non-finite losses; A5: accuracy curve, best and last checkpoint.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fbseq as cpu  # noqa: E402

OUT = os.path.join(cpu.ROOT, "model", "fbseq")


class GatedHA(cpu.Gated):
    """fbseq.Gated whose tables are hard in the forward pass for the methods gumbel / hardst / slope"""
    method, tau, slope = "softhard", 1.0, 1.0

    def tables(self, d):
        if self.method == "softhard":
            return super().tables(d)
        logit = d
        if self.method == "gumbel" and self.training:
            u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
            logit = logit + torch.log(u) - torch.log1p(-u)              # logistic noise = Gumbel difference
        scale = 1.0 / self.tau if self.method == "gumbel" else self.slope
        soft = torch.sigmoid(scale * logit)
        d = soft + ((logit > 0).float() - soft).detach()                # Boolean forward, surrogate backward
        return self.split(d)


def schedule(model, method, step, steps):
    p = step / max(1, steps - 1)
    model.method = method
    if method == "softhard":
        model.hard = step >= (2 * steps) // 3
    else:
        model.hard = False                                              # GatedHA.tables is hard itself
        model.tau = 1.0 * (0.25 / 1.0) ** p                             # 1 -> 0.25 geometric
        model.slope = 1.0 + 4.0 * p if method == "slope" else 1.0       # 1 -> 5 (slope only; hardst stays 1)


def hard_eval(model):
    """deterministic Boolean forward"""
    model.eval()
    if model.method == "softhard":
        model.hard = True
    return model


def train(model, method, batch, n_out, steps, lr, evaluate, every):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    curve, gnorms, bad = [], [], 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        schedule(model, method, step, steps)
        model.train()
        x, y = batch()
        loss = F.cross_entropy(model(x)[..., :n_out].reshape(-1, n_out), y.reshape(-1))
        if not torch.isfinite(loss):
            bad += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
        opt.step()
        if (step + 1) % every == 0 or step == steps - 1:
            hard_eval(model)
            with torch.no_grad():
                curve.append((step + 1, evaluate(model)))
    return curve, gnorms, bad


def saturation(model, x):
    """fraction of gate logits with |logit| > 3 (the table entry is decided with margin)"""
    with torch.no_grad():
        win = model.window(x)
        fr, below = [], None
        for l in range(model.L):
            d = model.gate[l](win if below is None else torch.cat([win, below], -1))
            fr.append(float((d.abs() > 3).float().mean()))
            below = model.run(model.tables(d), x.shape[0], x.shape[1])
    return fr


def bytes_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    tr, va, _ = cpu.loghub()
    Xtr, Ytr = cpu.packets_xy(tr)
    Xva, Yva = cpu.packets_xy(va)
    spec = dict(cpu.BYTE_MODELS[job["model"]])
    spec.pop("kind")
    model = GatedHA(257, 256, **spec)
    g = torch.Generator().manual_seed(job["seed"])
    sub = torch.arange(0, len(Xva), 4)                                  # periodic checks on 1/4 of val

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
    curve, gn, bad = train(model, job["method"], batch, 256, job["steps"], 3e-3, bpb, 500)
    hard_eval(model)
    with torch.no_grad():
        final = bpb(model, Xva, Yva)
    if job.get("save"):
        os.makedirs(os.path.dirname(job["save"]), exist_ok=True)
        torch.save(dict(state=model.state_dict(), spec=cpu.BYTE_MODELS[job["model"]], method=job["method"]), job["save"])
    return dict(model=job["model"], method=job["method"], seed=job["seed"], steps=job["steps"],
                bpb_hard=round(final, 4), curve=[(s, round(v, 4)) for s, v in curve],
                best_check=round(min(v for _, v in curve), 4), grad_norm_mean=round(float(np.mean(gn)), 4),
                nonfinite_steps=bad, gate_saturation=[round(f, 4) for f in saturation(model, Xva[:16])],
                seconds=round(time.time() - t0, 1))


def a5_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    n_in, n_out = cpu.TASKS["a5"]
    spec = dict(cpu.SYNTH_MODELS["B_block3"])
    spec.pop("kind")
    model = GatedHA(n_in, n_out, **spec)
    erng = np.random.default_rng(1000 + job["seed"])
    ev = {L: [torch.as_tensor(a) for a in cpu.task("a5", 256, L, erng)] for L in (64, 256)}

    def batch():
        x, y = cpu.task("a5", 64, 64, rng)
        return torch.as_tensor(x), torch.as_tensor(y)

    def acc(model):
        return {L: float((model(x)[..., :n_out].argmax(-1) == y).float().mean()) for L, (x, y) in ev.items()}
    t0 = time.time()
    curve, gn, bad = train(model, job["method"], batch, n_out, job["steps"], 3e-3, acc, 250)
    return dict(task="a5", model="B_block3", method=job["method"], seed=job["seed"], steps=job["steps"],
                curve=[(s, round(a[64], 3), round(a[256], 3)) for s, a in curve],
                last_acc256=round(curve[-1][1][256], 4), best_acc256=round(max(a[256] for _, a in curve), 4),
                best_step=max(curve, key=lambda c: c[1][256])[0], grad_norm_mean=round(float(np.mean(gn)), 4),
                nonfinite_steps=bad, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bytes", "a5"])
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--methods", default="softhard,gumbel,hardst,slope")
    ap.add_argument("--models", default="B_setkeep,B_full")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--save", action="store_true", help="byte models: keep the weights in build/fbseq/ckpt/")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    methods = a.methods.split(",")
    if a.what == "bytes":
        ck = os.path.join(cpu.ROOT, "build", "fbseq", "ckpt")
        jobs = [dict(model=m, method=me, seed=s, steps=a.steps or 3000,
                     save=os.path.join(ck, f"{m}_{me}_s{s}.pt") if a.save else None)
                for m in a.models.split(",") for me in methods for s in seeds]
        fn, path = bytes_job, os.path.join(OUT, "hardaware_bytes.json")
    else:
        jobs = [dict(method=me, seed=s, steps=a.steps or 3000) for me in methods for s in seeds]
        fn, path = a5_job, os.path.join(OUT, "hardaware_a5.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    key = lambda r: (r.get("model"), r["method"], r["seed"])
    with Pool(min(a.jobs, len(jobs))) as pool:
        for r in pool.imap_unordered(fn, jobs):
            print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
            out = [o for o in out if key(o) != key(r)] + [r]
            json.dump(sorted(out, key=lambda r: tuple(str(v) for v in key(r))), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
