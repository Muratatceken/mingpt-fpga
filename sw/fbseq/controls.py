"""
Phase-0 control experiments for candidate B (see fbseq.py).

    python3 controls.py gap      soft vs hard on validation, state divergence D(t), 4- and 8-byte windows
                                 -> model/fbseq/gap.json
    python3 controls.py a5       hand-built A5 tracker: exhaustive check of every (state, input), random
                                 sequences, and the LUT circuit's Fmax (loop_bench kind a5exact)
                                 -> model/fbseq/a5_exact.json
    python3 controls.py a5train  trainability: B_block3 on A5 with longer training -> model/fbseq/a5_train.json
    python3 controls.py cost     multiply-accumulates per byte and feedback depth of every byte model
                                 -> model/fbseq/cost.json
    python3 controls.py horizon  byte GRU with its state reset every k bytes (in training and/or evaluation)
                                 -> model/fbseq/gru_horizon.json

gap: after the soft phase (2/3 of the steps) the same weights are evaluated soft (multilinear states),
hard without adaptation, and the states compared bit by bit (soft thresholded at 0.5):
D(t) = fraction of state bits that differ at position t; then the hard phase runs and the final hard
model is evaluated as in fbseq.py.
"""
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
GAP_MODELS = {
    "B_setkeep":    cpu.BYTE_MODELS["B_setkeep"],
    "B_full":       cpu.BYTE_MODELS["B_full"],
    "B_setkeep_w8": dict(kind="gated", S=120, L=2, kb=1, mode="setkeep", w=8, n_embd=16, n_hidden=64, n_head=64),
    "B_full_w8":    dict(kind="gated", S=120, L=2, kb=1, mode="full", w=8, n_embd=16, n_hidden=64, n_head=64),
}


class Probe(cpu.Gated):
    """fbseq.Gated keeps the per-layer states of its last forward pass (last_states)"""


def val_bpb(model, X, Y, keep_states=False):
    nll, st = 0.0, []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            lg = model(X[i:i + 64])[..., :256]
            nll += F.cross_entropy(lg.reshape(-1, 256), Y[i:i + 64].reshape(-1), reduction="sum").item()
            if keep_states:
                st.append([s.clone() for s in model.last_states])
    bpb = nll / math.log(2) / Y.numel()
    return (bpb, [torch.cat([b[l] for b in st]) for l in range(model.L)]) if keep_states else bpb


def gap_job(job):
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    tr, va, _ = cpu.loghub()
    Xtr, Ytr = cpu.packets_xy(tr)
    Xva, Yva = cpu.packets_xy(va)
    spec = dict(GAP_MODELS[job["model"]])
    spec.pop("kind")
    model = Probe(257, 256, **spec)
    g = torch.Generator().manual_seed(job["seed"])

    def batch():
        ix = torch.randint(0, len(Xtr), (32,), generator=g)
        return Xtr[ix], Ytr[ix]
    steps, soft_steps = job["steps"], (2 * job["steps"]) // 3
    t0 = time.time()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.01)
    res = dict(model=job["model"], seed=job["seed"], params=cpu.n_params(model), steps=steps)
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1.0, (step + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        if step == soft_steps:                           # end of the soft phase: the three evaluations
            model.eval()
            model.hard = False
            res["bpb_soft"], s_soft = val_bpb(model, Xva, Yva, keep_states=True)
            model.hard = True
            res["bpb_hard_direct"], s_hard = val_bpb(model, Xva, Yva, keep_states=True)
            D = [((ss > 0.5).float() != sh).float().mean(dim=(0, 2)) for ss, sh in zip(s_soft, s_hard)]   # (T,)
            ts = [0, 1, 3, 7, 15, 31, 63, 127, 255]
            res["divergence"] = {f"layer{l}": {str(t): round(float(d[t]), 4) for t in ts} for l, d in enumerate(D)}
            res["divergence_mean"] = [round(float(d.mean()), 4) for d in D]
            # how far from 0/1 the soft states are: mean min(s, 1-s)
            res["soft_state_margin"] = [round(float(torch.minimum(ss, 1 - ss).mean()), 4) for ss in s_soft]
            model.train()
        x, y = batch()
        loss = F.cross_entropy(model(x)[..., :256].reshape(-1, 256), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    res["bpb_hard_ft"] = val_bpb(model, Xva, Yva)
    model.hard = False
    res["bpb_soft_after_ft"] = val_bpb(model, Xva, Yva)
    res["seconds"] = round(time.time() - t0, 1)
    for k in list(res):
        if k.startswith("bpb"):
            res[k] = round(res[k], 4)
    return res


# ----------------------------------------------------------------------------- A5 by hand
def a5_tracker():
    """the A5 task keeps p_t = p_{t-1} o g_t (fbseq.A5_TABLE). The inverse r = p^-1 updates point-wise,
    r_t(j) = g_t^-1(r_{t-1}(j)), so five independent 3-bit blocks, block j holding r(j), track it:
    block next state = ginv[x_t][state], a function of 3 state bits and 2 input bits (one LUT5)"""
    import itertools
    gens = [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4), (0, 1, 2, 3, 4)]
    ginv = [tuple(sorted(range(5), key=lambda i: g[i])) for g in gens]      # inverse permutations
    # truth table of next-state bit k of a block: inputs (s2 s1 s0, x1 x0), 32 entries
    tt = [[0] * 32 for _ in range(3)]
    for s in range(8):
        for x in range(4):
            nxt = ginv[x][s] if (s < 5 and x < 3) else s                    # unused codes: hold
            for k in range(3):
                tt[k][(s << 2) | x] = (nxt >> k) & 1
    # exhaustive check over every reachable state and input
    ok = all(sum(tt[k][(s << 2) | x] << k for k in range(3)) == ginv[x][s] for s in range(5) for x in range(3))
    perms = [p for p in itertools.permutations(range(5))
             if sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0]
    index = {p: i for i, p in enumerate(perms)}
    return tt, ok, index


def a5_exact():
    tt, ok, index = a5_tracker()
    rng = np.random.default_rng(0)
    x, y = cpu.task("a5", 2000, 1024, rng)
    r = np.tile(np.arange(5), (2000, 1))                                   # identity
    correct = 0
    for t in range(x.shape[1]):
        sx = x[:, t]
        idx = (r << 2) | sx[:, None]
        r = sum(np.array(tt[k])[idx] << k for k in range(3))
        p = np.argsort(r, axis=1)                                         # p = r^-1
        correct += sum(index[tuple(row)] == lab for row, lab in zip(map(tuple, p), y[:, t]))
    res = dict(exhaustive_ok=bool(ok), sequences=2000, length=1024, accuracy=correct / x.size,
               loop="per state bit: LUT5(3 state bits of its block, 2 input bits), one level")
    sys.path.insert(0, HERE)
    import loop_bench
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
    res["fmax"] = [loop_bench.run("a5exact", 120, seed) for seed in (1, 2)]
    return res


def a5train_job(job):
    torch.set_num_threads(1)
    return cpu.synth_job(job)


# ----------------------------------------------------------------------------- cost
def cost():
    """MACs per byte (feed-forward layers, recurrent cells, attention at the mean position of a 256-B
    packet) and what sits in the feedback loop"""
    loops = dict(window="none", gated1="one 2:1 mux per state bit", gated2="one LUT6 (4:1 mux)",
                 gated3="LUT6 pair (8:1 mux)", gru="whole GRU cell (arithmetic)", microgpt="none (KV cache in memory)")
    res = {}
    models = dict(cpu.BYTE_MODELS, **GAP_MODELS)
    for name, spec in models.items():
        sp = dict(spec)
        kind = sp.pop("kind")
        m = cpu.build(kind, 257, 256, T=cpu.PACKET, **sp)
        macs = 0
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                macs += mod.in_features * mod.out_features
        if kind == "microgpt":
            C, L = sp["n_embd"], sp["n_layer"]
            macs += L * 2 * C * (cpu.PACKET + 1) / 2                         # q.k and att.v, mean position
        key = kind if kind != "gated" else f"gated{sp['kb']}"
        res[name] = dict(params=cpu.n_params(m), macs_per_byte=int(macs), feedback=loops[key])
    return res


def main():
    what = sys.argv[1]
    if what == "gap":
        seeds = [0]
        jobs = [dict(model=m, seed=s, steps=3000) for m in GAP_MODELS for s in seeds]
        path = os.path.join(OUT, "gap.json")
        with Pool(len(jobs)) as pool:
            out = []
            for r in pool.imap_unordered(gap_job, jobs):
                print(json.dumps(r), flush=True)
                out.append(r)
                json.dump(sorted(out, key=lambda x: x["model"]), open(path, "w"), indent=1)
    elif what == "a5":
        r = a5_exact()
        json.dump(r, open(os.path.join(OUT, "a5_exact.json"), "w"), indent=1)
        print(json.dumps(r, indent=1))
    elif what == "a5train":
        jobs = [dict(task="a5", model="B_block3", seed=s, steps=8000) for s in (0, 1)]
        with Pool(len(jobs)) as pool:
            out = list(pool.imap_unordered(a5train_job, jobs))
        json.dump(out, open(os.path.join(OUT, "a5_train.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
    elif what == "horizon":
        jobs = [dict(train_reset=None, seed=0, steps=3000), dict(train_reset=8, seed=0, steps=3000)]
        with Pool(len(jobs)) as pool:
            out = list(pool.imap_unordered(horizon_job, jobs))
        json.dump(out, open(os.path.join(OUT, "gru_horizon.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
    elif what == "cost":
        r = cost()
        json.dump(r, open(os.path.join(OUT, "cost.json"), "w"), indent=1)
        print(json.dumps(r, indent=1))



# ----------------------------------------------------------------------------- GRU memory horizon
def gru_forward(model, idx, reset=None):
    """sw/zoo RNN(cell="gru") forward with the state reset to its start value every `reset` bytes"""
    B, T = idx.shape
    emb = model.wte(idx)
    h, hs = model.start.expand(B, -1), []
    for t in range(T):
        if reset and t and t % reset == 0:
            h = model.start.expand(B, -1)
        h = model.step(emb[:, t], h)
        hs.append(h)
    return model.lm_head(torch.stack(hs, 1))


def horizon_job(job):
    """train the byte GRU (fbseq C_gru) with an optional state reset every job["train_reset"] bytes, then
    evaluate with resets every 8 .. 128 bytes and without: how much of the gain is memory beyond k bytes?"""
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    tr, va, _ = cpu.loghub()
    Xtr, Ytr = cpu.packets_xy(tr)
    Xva, Yva = cpu.packets_xy(va)
    spec = dict(cpu.BYTE_MODELS["C_gru"])
    spec.pop("kind")
    model = cpu.build("gru", 257, 256, T=cpu.PACKET, **spec)
    g = torch.Generator().manual_seed(job["seed"])
    steps, R = job["steps"], job["train_reset"]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1.0, (step + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        ix = torch.randint(0, len(Xtr), (32,), generator=g)
        loss = F.cross_entropy(gru_forward(model, Xtr[ix], R)[..., :256].reshape(-1, 256), Ytr[ix].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    res = dict(train_reset=R, seed=job["seed"], steps=steps, eval={})
    with torch.no_grad():
        for k in (8, 16, 32, 64, 128, None):
            nll = 0.0
            for i in range(0, len(Xva), 64):
                lg = gru_forward(model, Xva[i:i + 64], k)[..., :256]
                nll += F.cross_entropy(lg.reshape(-1, 256), Yva[i:i + 64].reshape(-1), reduction="sum").item()
            res["eval"][str(k or "none")] = round(nll / math.log(2) / Yva.numel(), 4)
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    main()
