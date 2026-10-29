"""
Controls suggested by the theory review (docs/10_teori.md §7-8).

    python3 probes.py a5embed                  the A5 tracker written into the very model class that is trained
                                               (fbseq.Gated, 3-bit blocks, gate MLPs): exact tables, head trained
                                               alone                          -> model/fbseq/probe_a5embed.json
    python3 probes.py bytes CKPT               on a trained byte model (hardaware.py --save):
                                               (a) same weights, soft vs hard forward;
                                               (b) state frozen, head re-initialised and retrained;
                                               (c) single-bit flips: lifetime of the difference, spread to layer 2,
                                                   effect on the loss;
                                               (d) delayed read-out: linear probe state_t -> byte x_{t-k}
                                                                              -> model/fbseq/probe_<ckpt>.json
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fbseq as cpu  # noqa: E402
import hardaware as ha  # noqa: E402

OUT = os.path.join(cpu.ROOT, "model", "fbseq")


# ----------------------------------------------------------------------------- A5 inside the trained class
def a5embed(steps=1500):
    torch.manual_seed(0)
    n_in, n_out = cpu.TASKS["a5"]
    spec = dict(cpu.SYNTH_MODELS["B_block3"])
    spec.pop("kind")
    m = cpu.Gated(n_in, n_out, **spec)                                   # S=24: 8 blocks of 3 bits, L=2
    kb, Q, N = m.kb, 2 ** m.kb, m.S // m.kb
    gens = [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4), (0, 1, 2, 3, 4)]
    ginv = [tuple(sorted(range(5), key=lambda i: g[i])) for g in gens]
    # block j < 3 holds r(j) = p^-1(j) (three points determine an even permutation), coded as r(j) XOR j so
    # that the start r(j) = j is code 0 (fbseq.Gated starts every block in state 0); other blocks hold
    target = np.zeros((3, N, kb, Q))                                      # per input symbol: next-bit tables
    for x in range(3):
        for n in range(N):
            for q in range(Q):
                nxt = q
                if n < 3 and (q ^ n) < 5:
                    nxt = ginv[x][q ^ n] ^ n
                for b in range(kb):
                    target[x, n, b, q] = (nxt >> b) & 1
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.wte.weight[:3, :3] = torch.eye(3)                              # symbol x -> e_x
        g0 = m.gate[0]
        g0[0].weight[:3, :3] = 10 * torch.eye(3)                         # hidden_k = tanh(10[x=k] - 5) = +-1
        g0[0].bias[:3] = -5
        a = 5 * (2 * torch.tensor(target, dtype=torch.float32).reshape(3, -1) - 1)   # (3, N*kb*Q)
        g0[2].weight[:, :3] = a.T                                        # logit = sum_k a_k h_k + sum_k a_k
        g0[2].bias[:] = a.sum(0)                                         #       = 2 a_x = +-10
        g1 = m.gate[1]                                                   # layer 2: hold (identity tables)
        ident = torch.tensor([[[(q >> b) & 1 for q in range(Q)] for b in range(kb)] for _ in range(N)])
        g1[2].bias[:] = (10 * (2 * ident.float() - 1)).reshape(-1)
    m.hard = True
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.head.parameters():                                        # re-initialised head, trained alone
        p.requires_grad_(True)
    torch.nn.init.xavier_uniform_(m.head[0].weight)
    torch.nn.init.xavier_uniform_(m.head[2].weight)
    rng = np.random.default_rng(0)
    erng = np.random.default_rng(1000)
    ev = {L: [torch.as_tensor(v) for v in cpu.task("a5", 256, L, erng)] for L in (64, 256)}
    # 1) the state alone: decode p from the three blocks, compare with the label
    import itertools
    perms = [p for p in itertools.permutations(range(5))
             if sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0]
    index = {p: i for i, p in enumerate(perms)}
    with torch.no_grad():
        x, y = ev[256]
        m(x)
        st = m.last_states[0].round().long().numpy()                    # (B, T, 24)
    r = np.stack([(st[..., 3 * n] + 2 * st[..., 3 * n + 1] + 4 * st[..., 3 * n + 2]) ^ n for n in range(3)], -1)
    lookup = {}
    for p in perms:
        inv = tuple(sorted(range(5), key=lambda i: p[i]))
        lookup[inv[:3]] = index[p]
    dec = np.vectorize(lambda a, b, c: lookup.get((a, b, c), -1))(r[..., 0], r[..., 1], r[..., 2])
    state_acc = float((dec == y.numpy()).mean())
    # 2) head trained alone on the frozen exact machine
    opt = torch.optim.Adam(m.head.parameters(), lr=3e-3)
    for step in range(steps):
        xb, yb = (torch.as_tensor(v) for v in cpu.task("a5", 64, 64, rng))
        loss = F.cross_entropy(m(xb)[..., :n_out].reshape(-1, n_out), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = {L: float((m(x)[..., :n_out].argmax(-1) == y).float().mean()) for L, (x, y) in ev.items()}
    return dict(state_decode_acc256=state_acc, head_only_acc64=acc[64], head_only_acc256=acc[256], head_steps=steps,
                note="exact tables written into fbseq.Gated(kb=3, S=24, L=2) gate MLPs; blocks 0-2 hold r(j)=p^-1(j)")


# ----------------------------------------------------------------------------- probes on a trained byte model
def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = dict(ck["spec"])
    spec.pop("kind")
    m = ha.GatedHA(257, 256, **spec)
    m.load_state_dict(ck["state"])
    m.method = ck["method"]
    m.eval()
    return m


def bpb_of(m, X, Y, bs=64):
    nll = 0.0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg = m(X[i:i + bs])[..., :256]
            nll += F.cross_entropy(lg.reshape(-1, 256), Y[i:i + bs].reshape(-1), reduction="sum").item()
    return nll / math.log(2) / Y.numel()


def soft_vs_hard(m, X, Y):
    meth = m.method
    m.method = "softhard"
    m.hard = False
    soft = bpb_of(m, X, Y)
    m.method, m.hard = meth, False
    hard = bpb_of(m, X, Y)
    return dict(bpb_soft_same_weights=round(soft, 4), bpb_hard=round(hard, 4))


def head_retrain(m, Xtr, Ytr, Xva, Yva, Xte, Yte, steps=3000):
    torch.manual_seed(1)
    for p in m.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        for layer in (m.head[0], m.head[2]):
            torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(layer.bias)
    for p in m.head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(m.head.parameters(), lr=3e-3, betas=(0.9, 0.99), weight_decay=0.01)
    g = torch.Generator().manual_seed(1)
    for step in range(steps):                                       # the recipe of hardaware.train
        for gr in opt.param_groups:
            gr["lr"] = 3e-3 * min(1.0, (step + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        ix = torch.randint(0, len(Xtr), (32,), generator=g)
        loss = F.cross_entropy(m(Xtr[ix])[..., :256].reshape(-1, 256), Ytr[ix].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.head.parameters(), 1.0)
        opt.step()
    return dict(bpb_val_head_retrained=round(bpb_of(m, Xva, Yva), 4), bpb_test_head_retrained=round(bpb_of(m, Xte, Yte), 4))


def layer_states(m, x, flip=None):
    """hard states of every layer; flip = (layer, t0, bit): invert that bit at t0 and run on"""
    B, T = x.shape
    win = m.window(x)
    states, below = [], None
    for l in range(m.L):
        tab = m.tables(m.gate[l](win if below is None else torch.cat([win, below], -1)))
        b, c = tab                                                       # kb = 1: s' = b + c s
        s, out = win.new_zeros(B, m.S), []
        for t in range(T):
            s = b[:, t] + c[:, t] * s
            if flip is not None and flip[0] == l and t == flip[1]:
                s = s.clone()
                s[:, flip[2]] = 1 - s[:, flip[2]]
            out.append(s)
        below = torch.stack(out, 1)
        states.append(below)
    return win, states


def flips(m, X, Y, n_pkt=128, t0=32, bits=24, seed=0):
    """single-bit flips in layer 0 at t0: how long the difference lives in that bit, how many layer-2 bits it
    disturbs, and the loss change after t0"""
    assert m.kb == 1
    rng = np.random.default_rng(seed)
    x, y = X[:n_pkt], Y[:n_pkt]
    with torch.no_grad():
        win, base = layer_states(m, x)
        lg = m.head(torch.cat([win] + base, -1))[..., :256]
        base_nll = F.cross_entropy(lg[:, t0 + 1:].reshape(-1, 256), y[:, t0 + 1:].reshape(-1), reduction="none")
        life, spread, dloss = [], [], []
        for bit in rng.choice(m.S, bits, replace=False):
            win, st = layer_states(m, x, flip=(0, t0, int(bit)))
            diff0 = (st[0][:, t0:, bit] != base[0][:, t0:, bit])        # (B, T - t0)
            # lifetime: steps until the flipped bit agrees again (first agreement), T - t0 if never
            agree = (~diff0).float()
            first = torch.where(agree.any(1), agree.argmax(1), torch.full((len(x),), diff0.shape[1]))
            life += first.tolist()
            spread.append(float((st[1][:, t0:] != base[1][:, t0:]).float().sum(-1).mean()))
            lg2 = m.head(torch.cat([win] + st, -1))[..., :256]
            nll = F.cross_entropy(lg2[:, t0 + 1:].reshape(-1, 256), y[:, t0 + 1:].reshape(-1), reduction="none")
            dloss.append(float((nll - base_nll).mean() / math.log(2)))
    life = np.array(life)
    horizon = X.shape[1] - t0
    return dict(t0=t0, bits_flipped=bits, packets=n_pkt,
                lifetime_median=float(np.median(life)), lifetime_mean=float(life.mean()),
                survive_frac={str(k): float((life >= k).mean()) for k in (1, 4, 16, 64, horizon)},
                layer2_bits_disturbed_mean=float(np.mean(spread)),
                dbpb_after_flip_mean=float(np.mean(dloss)), dbpb_after_flip_max=float(np.max(dloss)))


def delayed_readout(m, Xtr, Xva, ks=(1, 2, 4, 8, 16, 32, 64, 128), n_tr=600, n_va=300, steps=400):
    """linear probe from the concatenated hard states at t to the byte at t - k (window excluded):
    what the state still carries about the past.  Control: the same probe from the window features alone
    (the last w bytes), i.e. how much of x_{t-k} is predictable from the recent bytes anyway"""
    with torch.no_grad():
        w_tr, st_tr = layer_states(m, Xtr[:n_tr])
        w_va, st_va = layer_states(m, Xva[:n_va])
    S_tr, S_va = torch.cat(st_tr, -1), torch.cat(st_va, -1)
    torch.manual_seed(0)

    def probe(a, ya, b, yb):
        lin = torch.nn.Linear(a.shape[1], 257)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        for _ in range(steps):
            ix = torch.randint(0, len(a), (4096,))
            loss = F.cross_entropy(lin(a[ix]), ya[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float((lin(b).argmax(-1) == yb).float().mean())
    res = {}
    for k in ks:
        def pairs(S, X):
            return S[:, k:].reshape(-1, S.shape[-1]), X[:, 1:X.shape[1] - k + 1].reshape(-1)   # input x_{t-k}
        a, ya = pairs(S_tr, Xtr[:n_tr])
        b, yb = pairs(S_va, Xva[:n_va])
        wa, _ = pairs(w_tr, Xtr[:n_tr])
        wb, _ = pairs(w_va, Xva[:n_va])
        unigram = float((yb == torch.mode(ya).values).float().mean())
        res[str(k)] = dict(acc=round(probe(a, ya, b, yb), 4), window_only=round(probe(wa, ya, wb, yb), 4),
                           majority=round(unigram, 4))
    return res


def bytes_probes(path):
    torch.set_num_threads(4)
    tr, va, te, _ = cpu.loghub(test=True)
    Xtr, Ytr = cpu.packets_xy(tr)
    Xva, Yva = cpu.packets_xy(va)
    Xte, Yte = cpu.packets_xy(te)
    m = load(path)
    res = dict(ckpt=os.path.basename(path))
    t0 = time.time()
    res["bpb_val"] = round(bpb_of(m, Xva, Yva), 4)
    res["bpb_test"] = round(bpb_of(m, Xte, Yte), 4)
    res.update(soft_vs_hard(load(path), Xva, Yva))
    res["flips"] = flips(load(path), Xva, Yva)
    res["delayed_readout"] = delayed_readout(load(path), Xtr, Xva)
    res.update(head_retrain(load(path), Xtr, Ytr, Xva, Yva, Xte, Yte))
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    what = sys.argv[1]
    if what == "a5embed":
        r = a5embed()
        json.dump(r, open(os.path.join(OUT, "probe_a5embed.json"), "w"), indent=1)
    else:
        r = bytes_probes(sys.argv[2])
        name = os.path.splitext(os.path.basename(sys.argv[2]))[0]
        json.dump(r, open(os.path.join(OUT, f"probe_{name}.json"), "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
