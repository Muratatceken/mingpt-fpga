"""
Feedback-constrained sequence models (phase 0): which state transition does a byte-level LM need,
and how much logic sits in its feedback loop?

    python3 fbseq.py synth [--steps N]   algebraic tasks (expressivity ladder)       -> model/fbseq/synth.json
    python3 fbseq.py bytes [--steps N]   byte LM on Loghub, independent 256-B packets -> model/fbseq/bytes.json
    python3 fbseq.py base                gzip / xz / zstd / zstd+dict, same packets   -> model/fbseq/base.json

Candidates
  A  window    feed-forward over the last W tokens (sw/zoo MLP): no feedback loop
  B  gated     input-gated Boolean recurrence, L layers of S state bits in blocks of kb bits;
               the next block state is an entry of a table d_t chosen by the current block state,
               i.e. a 2^kb:1 mux per bit; d_t comes from the input window and the layer below only
                 kb=1 setkeep  s' = g ? u : s          keep / set (the Krohn-Rhodes flip-flop)
                 kb=1 full     s' = s ? a : b          + toggle: all four maps of one bit
                 kb=2, kb=3    any map of 4 / 8 states (contain Z3 / A5)
  C  gru       sw/zoo RNN(cell="gru"): state-dependent gates, the whole cell inside the loop
  D  microgpt  sw/zoo MicroGPT: attention over the packet (KV cache)
B is trained with soft transitions (kb=1: affine in the state bit; kb>1: a distribution over the block's
states, the probabilistic automaton), then with hard 0/1 tables and straight-through gradients; it is
evaluated hard, i.e. as the finite-state machine the hardware would run. (Up to the commit after 985fbba
kb>1 used a bit-wise multilinear relaxation, whose Jacobian can exceed 1: the byte runs diverged to NaN.)
"""
import argparse
import gzip
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zlib
import lzma
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sw", "zoo"))
from models import MLP, RNN, MicroGPT, n_params  # noqa: E402

OUT = os.path.join(ROOT, "model", "fbseq")
PACKET = 256
BOS = 256


def ste(x):
    """hard 0/1 in the forward pass, identity gradient"""
    return x + ((x > 0.5).float() - x).detach()


# ----------------------------------------------------------------------------- candidate B
class Gated(nn.Module):
    def __init__(self, V, n_out, S=64, L=1, kb=1, mode="full", w=1, n_embd=16, n_hidden=64, n_head=64):
        super().__init__()
        assert S % kb == 0 and (mode != "setkeep" or kb == 1)
        self.V, self.S, self.L, self.kb, self.mode, self.w = V, S, L, kb, mode, w
        self.hard = False
        self.wte = nn.Embedding(V + 1, n_embd)                       # +1: padding before the start
        n_win = w * n_embd
        n_d = 2 * S if kb == 1 else S * 2 ** kb
        self.gate = nn.ModuleList(nn.Sequential(nn.Linear(n_win + (S if l else 0), n_hidden), nn.Tanh(),
                                                nn.Linear(n_hidden, n_d)) for l in range(L))
        self.head = nn.Sequential(nn.Linear(n_win + L * S, n_head), nn.Tanh(), nn.Linear(n_head, n_out))
        bits = ((torch.arange(2 ** kb)[None, :] >> torch.arange(kb)[:, None]) & 1).float()   # bit j of state q
        for g in self.gate:                                          # start at the identity map (keep)
            b = g[-1].bias.data
            if mode == "setkeep":
                b.view(S, 2)[:, 0] = -3.0                            # g -> 0
            elif kb == 1:
                b.view(S, 2)[:, 0], b.view(S, 2)[:, 1] = 3.0, -3.0   # a -> 1, b -> 0
            else:
                b.view(S // kb, kb, 2 ** kb)[:] = 6.0 * bits - 3.0   # d[q][j] = bit j of q

    def window(self, idx):
        embs, x = [], idx
        for _ in range(self.w):
            embs.append(self.wte(x))
            x = torch.roll(x, 1, 1)
            x[:, 0] = self.V
        return torch.cat(embs, -1)

    def tables(self, d):
        """gate output -> what the loop needs: kb=1: s' = b + c*s, returned as (b, c); kb>1: the table
        d[j, q] = P(next bit j = 1 | block state q), shape (B, T, S/kb, kb, 2^kb)"""
        d = torch.sigmoid(d)
        if self.hard:
            d = ste(d)
        return self.split(d)

    def split(self, d):
        B, T, S, kb = d.shape[0], d.shape[1], self.S, self.kb
        if kb == 1:
            d = d.view(B, T, S, 2)
            if self.mode == "setkeep":                   # s' = g ? u : s  ->  b = g*u, c = 1-g
                return d[..., 0] * d[..., 1], 1 - d[..., 0]
            return d[..., 1], d[..., 0] - d[..., 1]      # s' = s ? a : b  ->  b, c = a-b
        return (d.view(B, T, S // kb, kb, 2 ** kb),)

    def run(self, tab, B, T):
        """the loop over the sequence -> states (B, T, S). kb>1: each block keeps a distribution p over its
        2^kb states, p' = M p with M[q', q] = prod_j P(bit j of q' | q) column-stochastic (stable for long
        sequences); the bits passed on are E[bit j]. With hard tables p is one-hot: the Boolean machine."""
        tab = [x.unbind(1) for x in tab]                             # per-step views (cheap backward)
        if self.kb == 1:
            s, out = tab[0][0].new_zeros(B, self.S), []
            for t in range(T):                                       # the only sequential part: the loop
                s = tab[0][t] + tab[1][t] * s
                out.append(s)
            return torch.stack(out, 1)
        kb, N, Q = self.kb, self.S // self.kb, 2 ** self.kb
        bits = ((torch.arange(Q)[None, :] >> torch.arange(kb)[:, None]) & 1).to(tab[0][0])   # (kb, Q)
        nb = bits.bool()[:, :, None]                                  # (kb, Q', 1)
        p = tab[0][0].new_zeros(B, N, Q)
        p[..., 0] = 1.0                                              # every block starts in state 0
        out = []
        for t in range(T):
            d = tab[0][t]                                            # (B, N, kb, Q)
            M = torch.where(nb, d[..., :, None, :], 1 - d[..., :, None, :]).prod(-3)   # (B, N, Q', Q)
            p = torch.einsum("bnij,bnj->bni", M, p)
            out.append(torch.einsum("bnq,jq->bnj", p, bits).reshape(B, self.S))
        return torch.stack(out, 1)

    def forward(self, idx):
        B, T = idx.shape
        win = self.window(idx)
        states, below = [], None
        for l in range(self.L):
            tab = self.tables(self.gate[l](win if below is None else torch.cat([win, below], -1)))   # all t at once
            below = self.run(tab, B, T)
            states.append(below)
        self.last_states = states
        return self.head(torch.cat([win] + states, -1))


def build(kind, V, n_out, T, **kw):
    if kind == "window":
        return MLP(V=max(V, n_out), T=T, n_embd=kw.get("n_embd", 16), n_hidden=kw["n_hidden"], k=kw["w"])
    if kind == "gru":
        return RNN(V=max(V, n_out), T=T, n_embd=kw.get("n_embd", 16), n_hidden=kw["n_hidden"], cell="gru")
    if kind == "microgpt":
        return MicroGPT(V=max(V, n_out), T=T, n_layer=kw["n_layer"], n_embd=kw["n_embd"],
                        n_head=kw["n_embd"] // 8, hidden=kw["hidden"])
    return Gated(V, n_out, **kw)


# ----------------------------------------------------------------------------- algebraic tasks
def _a5():
    def even(p):
        return sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0
    perms = [p for p in itertools.permutations(range(5)) if even(p)]
    index = {p: i for i, p in enumerate(perms)}
    gens = [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4), (0, 1, 2, 3, 4)]       # (0 1 2 3 4), (0 1 2), identity
    table = np.array([[index[tuple(p[g[i]] for i in range(5))] for p in perms] for g in gens])  # g o p
    return table, index[(0, 1, 2, 3, 4)]


A5_TABLE, A5_ID = _a5()


def task(name, B, T, rng):
    """(x, y): input symbols and the hidden state after each symbol, both (B, T)"""
    if name == "flag":                       # 1 sets, 2 resets, 0/3 are noise: star-free (set/keep)
        x = rng.choice(4, size=(B, T), p=[0.4, 0.1, 0.1, 0.4])
        y = np.zeros_like(x)
        f = np.zeros(B, dtype=np.int64)
        for t in range(T):
            f = np.where(x[:, t] == 1, 1, np.where(x[:, t] == 2, 0, f))
            y[:, t] = f
        return x, y
    if name == "parity":                     # Z2
        x = rng.integers(0, 2, size=(B, T))
        return x, np.cumsum(x, 1) % 2
    if name == "mod3":                       # Z3
        x = rng.integers(0, 2, size=(B, T))
        return x, np.cumsum(x, 1) % 3
    if name == "a5":                         # the smallest non-solvable group
        x = rng.integers(0, 3, size=(B, T))
        y = np.zeros_like(x)
        p = np.full(B, A5_ID)
        for t in range(T):
            p = A5_TABLE[x[:, t], p]
            y[:, t] = p
        return x, y
    raise KeyError(name)


TASKS = dict(flag=(4, 2), parity=(2, 2), mod3=(2, 3), a5=(3, 60))        # n_in, n_out
SYNTH_MODELS = {
    "A_window8": dict(kind="window", w=8, n_hidden=64),
    "B_setkeep": dict(kind="gated", S=24, L=2, kb=1, mode="setkeep"),
    "B_full":    dict(kind="gated", S=24, L=2, kb=1, mode="full"),
    "B_block2":  dict(kind="gated", S=24, L=2, kb=2),
    "B_block3":  dict(kind="gated", S=24, L=2, kb=3),
    "C_gru":     dict(kind="gru", n_hidden=32),
    "D_microgpt": dict(kind="microgpt", n_layer=2, n_embd=32, hidden=64),
}


# ----------------------------------------------------------------------------- training
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


def synth_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"])
    n_in, n_out = TASKS[job["task"]]
    spec = dict(SYNTH_MODELS[job["model"]])
    kind = spec.pop("kind")
    model = build(kind, n_in, n_out, T=256, **spec)
    T = 64

    def batch():
        x, y = task(job["task"], 64, T, rng)
        return torch.as_tensor(x), torch.as_tensor(y)
    t0 = time.time()
    fit(model, batch, n_out, job["steps"], 3e-3, hard_from=(2 * job["steps"]) // 3 if kind == "gated" else None)
    model.eval()
    res = dict(task=job["task"], model=job["model"], seed=job["seed"], params=n_params(model),
               seconds=round(time.time() - t0, 1))
    erng = np.random.default_rng(1000 + job["seed"])
    with torch.no_grad():
        for L in (64, 256):
            x, y = task(job["task"], 256, L, erng)
            pred = model(torch.as_tensor(x))[..., :n_out].argmax(-1).numpy()
            res[f"acc{L}"] = float((pred == y).mean())
    return res


# ----------------------------------------------------------------------------- byte data (Loghub)
def loghub():
    """train / val packets: each log file split at the line nearest 80 %, cut into 256-byte packets"""
    d = os.path.join(ROOT, "data", "loghub")
    files = sorted(f for f in os.listdir(d) if f.endswith(".log"))
    if not files:
        sys.exit("data/loghub is empty: run scripts/get_loghub.sh")
    tr, va = [], []
    for f in files:
        b = open(os.path.join(d, f), "rb").read()
        cut = b.rfind(b"\n", 0, int(0.8 * len(b))) + 1
        for part, dst in ((b[:cut], tr), (b[cut:], va)):
            n = len(part) // PACKET
            dst.append(np.frombuffer(part[:n * PACKET], dtype=np.uint8).reshape(n, PACKET))
    return np.concatenate(tr), np.concatenate(va), files


BYTE_MODELS = {                              # widths searched for ~100K parameters each
    "A_window8":  dict(kind="window", w=8, n_embd=16, n_hidden=248),
    "A_window16": dict(kind="window", w=16, n_embd=16, n_hidden=188),
    "A_window32": dict(kind="window", w=32, n_embd=12, n_hidden=152),
    "B_setkeep":  dict(kind="gated", S=120, L=2, kb=1, mode="setkeep", w=4, n_embd=16, n_hidden=80, n_head=64),
    "B_full":     dict(kind="gated", S=120, L=2, kb=1, mode="full", w=4, n_embd=16, n_hidden=80, n_head=64),
    "B_block2":   dict(kind="gated", S=120, L=2, kb=2, w=4, n_embd=16, n_hidden=48, n_head=64),
    "B_block3":   dict(kind="gated", S=120, L=2, kb=3, w=4, n_embd=16, n_hidden=26, n_head=64),
    "C_gru":      dict(kind="gru", n_embd=32, n_hidden=124),
    "D_microgpt": dict(kind="microgpt", n_layer=4, n_embd=40, hidden=136),
}


def packets_xy(p):
    """inputs [BOS, b0 .. b254], targets [b0 .. b255]"""
    p = torch.as_tensor(p.astype(np.int64))
    return torch.cat([torch.full((len(p), 1), BOS), p[:, :-1]], 1), p


def byte_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    tr, va, _ = loghub()
    Xtr, Ytr = packets_xy(tr)
    Xva, Yva = packets_xy(va)
    spec = dict(BYTE_MODELS[job["model"]])
    kind = spec.pop("kind")
    model = build(kind, 257, 256, T=PACKET, **spec)
    g = torch.Generator().manual_seed(job["seed"])

    def batch():
        ix = torch.randint(0, len(Xtr), (32,), generator=g)
        return Xtr[ix], Ytr[ix]
    t0 = time.time()
    soft = fit(model, batch, 256, job["steps"], job.get("lr", 3e-3),
               hard_from=(2 * job["steps"]) // 3 if kind == "gated" else None)
    model.eval()
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(Xva), 64):
            lg = model(Xva[i:i + 64])[..., :256]
            nll += F.cross_entropy(lg.reshape(-1, 256), Yva[i:i + 64].reshape(-1), reduction="sum").item()
    bpb = nll / math.log(2) / Yva.numel()
    return dict(model=job["model"], seed=job["seed"], params=n_params(model), bpb=round(bpb, 4),
                soft_train_bpb=None if soft is None else round(soft / math.log(2), 4),
                steps=job["steps"], seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- classic baselines
def baselines():
    tr, va, files = loghub()
    raw = va.size
    res = dict(packets=len(va), packet_bytes=PACKET, val_bytes=int(raw), files=files)
    res["zlib9_raw"] = sum(len((lambda c: c.compress(bytes(p)) + c.flush())(zlib.compressobj(9, zlib.DEFLATED, -15)))
                           for p in va)
    res["gzip9"] = sum(len(gzip.compress(bytes(p), 9)) for p in va)
    res["xz9e_raw"] = sum(len(lzma.compress(bytes(p), format=lzma.FORMAT_RAW,
                                            filters=[dict(id=lzma.FILTER_LZMA2, preset=9 | lzma.PRESET_EXTREME)]))
                          for p in va)
    with tempfile.TemporaryDirectory() as tmp:
        for name, arr in (("tr", tr), ("va", va)):
            os.makedirs(os.path.join(tmp, name))
            for i, p in enumerate(arr):
                open(os.path.join(tmp, name, f"{i:06d}"), "wb").write(bytes(p))
        runs = [("zstd19", [])]
        for kib in (16, 64, 100):                # dictionary = side information shipped once, like LM weights
            dict_path = os.path.join(tmp, f"dict{kib}")
            subprocess.run(["zstd", "-q", "--train", "-r", os.path.join(tmp, "tr"), "-o", dict_path,
                            f"--maxdict={kib * 1024}"], check=True, capture_output=True)
            res[f"zstd_dict{kib}k_bytes"] = os.path.getsize(dict_path)
            runs.append((f"zstd19_dict{kib}k", ["-D", dict_path, "--no-dictID"]))
        for tag, extra in runs:
            out = os.path.join(tmp, tag)
            os.makedirs(out)
            subprocess.run(["zstd", "-q", "-19", "--no-check", "-r", os.path.join(tmp, "va"),
                            "--output-dir-flat", out] + extra, check=True, capture_output=True)
            res[tag] = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    for k in ("zlib9_raw", "gzip9", "xz9e_raw") + tuple(t for t, _ in runs):
        res[k + "_bpb"] = round(8 * res[k] / raw, 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["synth", "bytes", "base"])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--models", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.what == "base":
        r = baselines()
        json.dump(r, open(os.path.join(OUT, "base.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "files"}, indent=1))
        return
    if a.what == "synth":
        models = a.models.split(",") if a.models else list(SYNTH_MODELS)
        jobs = [dict(task=t, model=m, seed=s, steps=a.steps or 2000) for t in TASKS for m in models for s in seeds]
        fn, path = synth_job, os.path.join(OUT, "synth.json")
    else:
        models = a.models.split(",") if a.models else list(BYTE_MODELS)
        jobs = [dict(model=m, seed=s, steps=a.steps or 3000) for m in models for s in seeds]
        fn, path = byte_job, os.path.join(OUT, "bytes.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    with Pool(min(a.jobs, len(jobs))) as pool:
        for r in pool.imap_unordered(fn, jobs):
            print(json.dumps(r), flush=True)
            key = lambda x: (x.get("task"), x["model"], x["seed"])
            old = [o for o in old if key(o) != key(r)] + [r]
            json.dump(sorted(old, key=lambda x: tuple(str(v) for v in key(x))), open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
