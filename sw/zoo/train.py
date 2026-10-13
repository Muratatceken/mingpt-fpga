"""
Train models of the family / feature ladder with a common recipe.
  python3 train.py --tag T [--data names|stories] [--arch family|ladder|a,b,..] [--budget N]
                   [--seeds 0,1,..] [--lrs 1e-3,3e-3,..] [--steps N] [--batch N] [--jobs N] [--device cpu|mps]
One run = (arch, lr, seed) -> model/zoo/runs/<tag>/<arch>_lr<lr>_s<seed>.{pt,json}
  json: spec (class + kwargs, see configs.build), params, train/val/test loss, curve, hyper-parameters.
Recipe: initialisation of the source repository (models.source_init), AdamW (0.9, 0.99), wd 0.01,
warm-up 500 steps, cosine to lr/30, grad clip 1.0.
names: loss over the name tokens incl. the end token; stories: next-token loss on T-token windows.
"""
import argparse
import json
import math
import os
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per process: many runs share 8 cores
import torch  # noqa: E402
import torch.nn.functional as F

from configs import FAMILY, LADDER, build, make
from data import ROOT
from models import n_params, source_init

RUNS = os.path.join(ROOT, "model", "zoo", "runs")
WD, WARMUP, BETAS, CLIP = 0.01, 500, (0.9, 0.99), 1.0


def loss_on(model, X, Y, bs=256, device="cpu"):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, y = X[i:i + bs].to(device), Y[i:i + bs].to(device)
            logits = model(x)
            tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=-1, reduction="sum").item()
            n += (y >= 0).sum().item()
    model.train()
    return tot / n


def causality_check(model, V, T):
    """logits at position t must not depend on tokens after t."""
    model.eval()
    T = min(T, 24)
    x = torch.randint(0, V, (2, T))
    with torch.no_grad():
        a = model(x)
        for t in range(T - 1):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, V, (2, T - t - 1))
            assert torch.allclose(a[:, :t + 1], model(x2)[:, :t + 1], atol=1e-5), f"causality violated at t={t}"
    model.train()


def run_one(job):
    torch.set_num_threads(job["threads"])
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(3407 + job["seed"])
    dev = job["device"]
    model = source_init(build(job["spec"]))
    V, T = job["spec"]["kwargs"]["V"], job["spec"]["kwargs"]["T"]
    causality_check(model, V, T)
    model.to(dev)
    if job["data"] == "names":
        from data import load
        d = load()
        Xtr, Ytr = d["train"]
        sets = dict(val=d["val"], test=d["test"])
        batch = lambda: (lambda ix: (Xtr[ix], Ytr[ix]))(torch.randint(0, len(Xtr), (job["batch"],)))
    else:
        from tinystories import Batches, val_windows
        tr = Batches("train", T, seed=job["seed"])
        sets = dict(val=val_windows(T, "val"), test=val_windows(T, "test"))
        batch = lambda: tr.get(job["batch"])
    lr, steps = job["lr"], job["steps"]
    min_lr = lr / 30

    def lr_at(step):
        if step < WARMUP:
            return lr * (step + 1) / WARMUP
        p = (step - WARMUP) / max(1, steps - WARMUP)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, eps=1e-8, weight_decay=WD)
    t0, curve = time.time(), []
    every = max(1, steps // 10)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch()
        logits = model(x.to(dev))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(dev).reshape(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step % every == 0 or step == steps - 1:
            v = loss_on(model, *sets["val"], device=dev)
            curve.append((step, round(loss.item(), 4), round(v, 4)))
            print(f"  {job['name']:28s} step {step:6d}  train {loss.item():.4f}  val {v:.4f}", flush=True)
    model.cpu()
    res = dict(job, params=n_params(model), val_loss=loss_on(model, *sets["val"]),
               test_loss=loss_on(model, *sets["test"]), seconds=round(time.time() - t0, 1), curve=curve,
               hparams=dict(steps=steps, batch=job["batch"], lr=lr, min_lr=min_lr, wd=WD, warmup=WARMUP,
                            betas=BETAS, clip=CLIP))
    torch.save(model.state_dict(), job["path"] + ".pt")
    json.dump(res, open(job["path"] + ".json", "w"), indent=1)
    print(f"{job['name']}: {res['params']} params, val {res['val_loss']:.4f}, test {res['test_loss']:.4f}, "
          f"{res['seconds']} s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="names", choices=["names", "stories"])
    ap.add_argument("--arch", default="family")
    ap.add_argument("--budget", type=int, default=50000)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lrs", default="3e-3")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    archs = FAMILY if a.arch == "family" else list(LADDER) if a.arch == "ladder" else a.arch.split(",")
    out = os.path.join(RUNS, a.tag)
    os.makedirs(out, exist_ok=True)
    jobs = []
    for arch in archs:
        cls, kw = make(arch, a.data, a.budget)
        for lr in [float(x) for x in a.lrs.split(",")]:
            for seed in [int(x) for x in a.seeds.split(",")]:
                name = f"{arch}_lr{lr:g}_s{seed}"
                path = os.path.join(out, name)
                if os.path.exists(path + ".json"):
                    continue                                   # resumable: finished runs are skipped
                jobs.append(dict(name=name, path=path, arch=arch, data=a.data, budget=a.budget, lr=lr, seed=seed,
                                 spec=dict(cls=cls, kwargs=kw), steps=a.steps, batch=a.batch, device=a.device,
                                 threads=a.threads, tag=a.tag))
    # slow (transformer) runs first so the pool drains evenly
    jobs.sort(key=lambda j: j["spec"]["cls"] in ("Bigram", "BoW", "MLP"))
    print(f"{len(jobs)} runs -> {out}", flush=True)
    if a.jobs <= 1:
        for j in jobs:
            run_one(j)
    else:
        with get_context("spawn").Pool(a.jobs) as pool:
            for _ in pool.imap_unordered(run_one, jobs):
                pass


if __name__ == "__main__":
    main()
