"""
Train every model of the zoo on names.txt with identical settings.
  python3 train.py [model ...]        (default: all)
Outputs: model/zoo/<name>.pt and model/zoo/metrics.json
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from data import ROOT, load
from models import ZOO, n_params

OUT = os.path.join(ROOT, "model", "zoo")
STEPS, BATCH, LR, MIN_LR, WD, WARMUP, SEED = 20000, 64, 3e-3, 1e-4, 0.01, 500, 3407


def loss_on(model, X, Y, bs=500):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits = model(X[i:i + bs])
            y = Y[i:i + bs]
            tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                   ignore_index=-1, reduction="sum").item()
            n += (y >= 0).sum().item()
    model.train()
    return tot / n


def causality_check(model, V, T):
    """logits at position t must not depend on tokens after t."""
    model.eval()
    x = torch.randint(0, V, (4, T))
    with torch.no_grad():
        a = model(x)
        for t in range(T - 1):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, V, (4, T - t - 1))
            b = model(x2)
            assert torch.allclose(a[:, :t + 1], b[:, :t + 1], atol=1e-5), f"causality violated at t={t}"
    model.train()


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * p))


def train(name, d):
    torch.manual_seed(SEED)
    model = ZOO[name]()
    causality_check(model, d["V"], d["T"])
    Xtr, Ytr = d["train"]
    Xte, Yte = d["test"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-8, weight_decay=WD)
    t0 = time.time()
    curve = []
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ix = torch.randint(0, len(Xtr), (BATCH,))
        logits = model(Xtr[ix])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Ytr[ix].reshape(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0 or step == STEPS - 1:
            te = loss_on(model, Xte, Yte)
            curve.append((step, round(loss.item(), 4), round(te, 4)))
            print(f"  {name:9s} step {step:5d}  train {loss.item():.4f}  test {te:.4f}", flush=True)
    m = dict(params=n_params(model), train_loss=loss_on(model, Xtr[:5000], Ytr[:5000]),
             test_loss=loss_on(model, Xte, Yte), seconds=round(time.time() - t0, 1), curve=curve,
             hparams=dict(steps=STEPS, batch=BATCH, lr=LR, min_lr=MIN_LR, wd=WD, warmup=WARMUP, seed=SEED))
    torch.save(model.state_dict(), os.path.join(OUT, f"{name}.pt"))
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)
    d = load()
    names = sys.argv[1:] or list(ZOO)
    path = os.path.join(OUT, "metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        metrics[name] = train(name, d)
        json.dump(metrics, open(path, "w"), indent=1)
        print(f"{name}: {metrics[name]['params']} params, test loss {metrics[name]['test_loss']:.4f}, "
              f"{metrics[name]['seconds']} s")


if __name__ == "__main__":
    main()
