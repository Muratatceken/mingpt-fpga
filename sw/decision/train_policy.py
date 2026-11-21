"""Train the decision models (WP3.4 GRU, WP3.5 LM) on data/decision/train (labels at MODE positions only).

    python3 train_policy.py --model lm  --lrs 1e-3,3e-3 --seeds 0      [--device mps]
    python3 train_policy.py --model gru --lrs 3e-3 --seeds 0,1,2 --jobs 3
Recipe = sw/zoo/train.py (AdamW (0.9, 0.99), wd 0.01, warm-up 500, cosine to lr/30, clip 1.0); the loss is the
4-way action cross-entropy at decision positions (the decoder is a masked argmax over the 4 action tokens).
Validation = matched val pairs: decision correct iff the predicted action is in the acceptable set.
Output: model/decision/runs/<tag>/<model>_lr<lr>_s<seed>.{pt,json}
"""
import argparse
import json
import math
import os
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from contract import PAD, ROOT  # noqa: E402
from models_dec import SPECS, build, n_params  # noqa: E402

RUNS = os.path.join(ROOT, "model", "decision", "runs")
DATA = os.path.join(ROOT, "data", "decision")
WD, WARMUP, BETAS, CLIP = 0.01, 500, (0.9, 0.99), 1.0


def decide(logits4):
    """masked argmax over the 4 actions, ties -> more severe (higher id)."""
    return 3 - torch.argmax(torch.flip(logits4, dims=[-1]), dim=-1)


def load_val(split="val"):
    z = np.load(os.path.join(DATA, split, "pairs.npz"))
    with open(os.path.join(DATA, split, "pairs_meta.json")) as f:
        meta = json.load(f)
    return z["seq"], z["y"], z["acc"], meta


@torch.no_grad()
def eval_pairs(model, seq, acc, meta, device, bs=480):
    model.eval()
    preds = []
    for i in range(0, len(seq), bs):
        s = torch.from_numpy(seq[i:i + bs]).long().to(device)
        t0 = meta[i]["t0"]
        pos = 4 + 5 * t0
        preds.append(decide(model(s[:, :pos + 1])[:, pos, :4]).cpu().numpy())
    model.train()
    pred = np.concatenate(preds)
    ok = acc[np.arange(len(pred)), pred]
    out = dict(acc=float(ok.mean()))
    pair_ok = ok.reshape(-1, 2).all(1)
    out["pair_acc"] = float(pair_ok.mean())
    by = {}
    for j, m in enumerate(meta[::2]):
        k = f"{m['template']}/d{m['d']}"
        by.setdefault(k, []).append(pair_ok[j])
    out["pair_acc_by"] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
    return out, pred


def run_one(job):
    torch.set_num_threads(job["threads"])
    dev = job["device"]
    z = np.load(os.path.join(DATA, "train", "train.npz"))
    X = torch.from_numpy(z["X"].astype(np.int64))
    Y = torch.from_numpy(z["Y"].astype(np.int64))
    lens = (X != PAD).sum(1)
    vseq, vy, vacc, vmeta = load_val()
    torch.manual_seed(3407 + job["seed"])
    rng = np.random.default_rng(3407 + job["seed"])
    spec = SPECS[job["model"]]
    model = build(spec).to(dev)
    lr, steps, bs = job["lr"], job["steps"], job["batch"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WD)
    sched = lambda s: s / WARMUP if s < WARMUP else (1 / 30 + (1 - 1 / 30) * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, steps - WARMUP))))
    curve, t0 = [], time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * sched(step)
        idx = torch.from_numpy(rng.integers(0, len(X), bs))
        L = int(lens[idx].max())
        x, y = X[idx, :L].to(dev), Y[idx, :L].to(dev)
        logits = model(x)[..., :4]
        loss = F.cross_entropy(logits.reshape(-1, 4), y.reshape(-1), ignore_index=-1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step % job["eval_every"] == 0 or step == steps - 1:
            ev, _ = eval_pairs(model, vseq, vacc, vmeta, dev)
            curve.append(dict(step=step, loss=float(loss.item()), val_acc=ev["acc"], val_pair_acc=ev["pair_acc"],
                              sec=round(time.time() - t0, 1)))
            print(f"[{job['model']} lr={lr:g} s={job['seed']}] step {step:5d} loss {loss.item():.4f} "
                  f"val acc {ev['acc']:.3f} pair {ev['pair_acc']:.3f}  {time.time() - t0:.0f}s", flush=True)
    ev, pred = eval_pairs(model, vseq, vacc, vmeta, dev)
    out_dir = os.path.join(RUNS, job["tag"])
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{job['model']}_lr{lr:g}_s{job['seed']}")
    torch.save(model.state_dict(), stem + ".pt")
    with open(os.path.join(DATA, "..", "..", "sw", "decision", "manifests", "train_manifest.json")) as f:
        man = json.load(f)
    rec = dict(model=job["model"], spec=spec, params=n_params(model), lr=lr, seed=job["seed"], steps=steps,
               batch=bs, device=dev, val=ev, curve=curve, seconds=round(time.time() - t0, 1),
               train_X_sha256=man["X_sha256"], recipe=dict(wd=WD, warmup=WARMUP, betas=BETAS, clip=CLIP))
    with open(stem + ".json", "w") as f:
        json.dump(rec, f, indent=1)
    return stem, ev["acc"], ev["pair_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lm")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--lrs", default="1e-3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    jobs = [dict(model=a.model, tag=a.tag, lr=float(lr), seed=int(s), steps=a.steps, batch=a.batch,
                 eval_every=a.eval_every, device=a.device, threads=max(1, 8 // max(1, a.jobs)))
            for lr in a.lrs.split(",") for s in a.seeds.split(",")]
    if a.jobs > 1:
        with get_context("spawn").Pool(a.jobs) as pool:
            res = pool.map(run_one, jobs)
    else:
        res = [run_one(j) for j in jobs]
    for stem, acc, pacc in res:
        print(f"{os.path.basename(stem)}: val acc {acc:.3f}  pair acc {pacc:.3f}")


if __name__ == "__main__":
    main()
