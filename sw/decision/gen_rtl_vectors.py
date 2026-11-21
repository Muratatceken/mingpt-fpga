"""Golden vectors for the decision RTL unit tests (tb/tb_dec_units.sv, tb/tb_dec_guard_seq.sv).

    python3 gen_rtl_vectors.py        -> build/decision/vec_{guard,tok,rule,fsm,gseq}.txt
guard : exhaustive over mode (incl. the illegal code 3) x prop_valid x proposal x V/I/T just below/at the limits x
        valid flags x ground command x invalid counter state
tok   : random codes plus every bin edge and edge-1, with random valid flags
rule  : every (V, I, T) token combination including invalid tokens
fsm   : random token streams + the validation pair episodes (designed cases), per-step expected action
gseq  : watchdog edge cases for the registered guard (WDT = 8): proposal at busy cycle WDT-2, WDT-1, never
"""
import itertools
import json
import os

import numpy as np

from contract import EDGES, I_BASE, I_INV, ROOT, T_BASE, T_INV, V_BASE, V_INV
from guard import guard_step
from policies import FSMPolicy, rule_action
from tokenizer import bins_from_tokens, obs_tokens

OUT = os.path.join(ROOT, "build", "decision")


def w(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(f"{len(rows)}\n")
        for r in rows:
            f.write(" ".join(str(int(x)) for x in r) + "\n")
    return len(rows)


def guard_vectors():
    rows = []
    for mode, pv, prop, v, i, t, vv, iv, tv, gc, inv in itertools.product(
            range(4), (0, 1), range(4), (6399, 6400, 7000), (2799, 2800), (449, 450), (0, 1), (0, 1), (0, 1),
            range(3), range(8)):
        a, m, r, n = guard_step(prop, bool(pv), mode, v, i, t, bool(vv), bool(iv), bool(tv), gc, inv)
        rows.append((mode, pv, prop, v, i, t, vv, iv, tv, gc, inv, a, m, r, n))
    return rows


def tok_vectors(rng):
    rows = []
    codes = {"V": [], "I": [], "T": []}
    for ch in codes:
        e = EDGES[ch]
        codes[ch] = list(e) + list(e - 1) + list(rng.integers(int(e[0]) - 500, int(e[-1]) + 500, 300))
    n = max(len(c) for c in codes.values())
    for k in range(n):
        v, i, t = (codes[ch][k % len(codes[ch])] for ch in ("V", "I", "T"))
        vv, iv, tv = rng.random(3) > 0.1
        tV, tI, tT = obs_tokens(np.array([v]), np.array([i]), np.array([t]), np.array([vv]), np.array([iv]),
                                np.array([tv]))
        rows.append((v, i, t, vv, iv, tv, tV[0], tI[0], tT[0]))
    return rows


def rule_vectors():
    rows = []
    for tV in list(range(V_BASE, V_INV + 1)):
        for tI in list(range(I_BASE, I_INV + 1)):
            for tT in list(range(T_BASE, T_INV + 1)):
                vb, ib, tb, inv = bins_from_tokens(np.array([tV]), np.array([tI]), np.array([tT]))
                rows.append((tV, tI, tT, rule_action(vb, ib, tb, inv)[0]))
    return rows


def fsm_vectors(rng, n_rand=300, N=60):
    """rows: episode, step, tV, tI, tT, prev_act, expected act."""
    rows = []
    # random but structured streams: slowly wandering bins, load switches, invalid samples
    vb = rng.integers(3, 16, n_rand)
    ib = rng.integers(0, 14, n_rand)
    tb = rng.integers(0, 12, n_rand)
    fsm = FSMPolicy(n_rand)
    prev = np.zeros(n_rand, dtype=np.int64)
    for k in range(N):
        vb = np.clip(vb + rng.integers(-2, 3, n_rand), 0, 17)
        ib = np.where(rng.random(n_rand) < 0.15, rng.integers(0, 14, n_rand), ib)
        tb = np.clip(tb + rng.integers(-1, 2, n_rand), 0, 11)
        tV = np.where(rng.random(n_rand) < 0.03, V_INV, V_BASE + vb)
        tI = np.where(rng.random(n_rand) < 0.03, I_INV, I_BASE + ib)
        tT = np.where(rng.random(n_rand) < 0.03, T_INV, T_BASE + tb)
        a = fsm.act(tV, tI, tT, 0, prev)
        for b in range(n_rand):
            rows.append((b, k, tV[b], tI[b], tT[b], prev[b], a[b]))
        prev = np.where(rng.random(n_rand) < 0.3, rng.integers(0, 4, n_rand), a)
    # designed cases: the validation pair episodes with their applied actions
    z = np.load(os.path.join(ROOT, "data", "decision", "val", "pairs.npz"))
    seq = z["seq"]
    E = len(seq)
    fsm = FSMPolicy(E)
    steps = (seq.shape[1] - 1) // 5
    prev = np.zeros(E, dtype=np.int64)
    for k in range(steps):
        tV, tI, tT = seq[:, 1 + 5 * k], seq[:, 2 + 5 * k], seq[:, 3 + 5 * k]
        a = fsm.act(tV, tI, tT, 0, prev)
        for b in range(E):
            rows.append((n_rand + b, k, tV[b], tI[b], tT[b], prev[b], a[b]))
        if 5 + 5 * k < seq.shape[1]:
            prev = seq[:, 5 + 5 * k]
    rows.sort(key=lambda r: (r[0], r[1]))                           # consecutive rows per episode
    return rows


def gseq_vectors(rng, WDT=8, n=400):
    """rows: arrive (busy-cycle index of the proposal, -1 none), prop, v, i, t, vv, iv, tv, gc, exp_a, exp_m, exp_r."""
    rows = []
    mode, inv = 0, 0
    for k in range(n):
        arrive = int(rng.choice([0, 1, WDT - 3, WDT - 2, WDT - 1, WDT, -1]))
        prop = int(rng.integers(0, 4))
        v = int(rng.choice([6399, 6400, 7200]))
        i = int(rng.choice([2799, 2800, 1200]))
        t = int(rng.choice([449, 450, 200]))
        vv, iv, tv = (int(x) for x in rng.random(3) > 0.15)
        gc = int(rng.choice([0, 0, 0, 1, 2]))
        pv = 0 <= arrive < WDT - 1                     # at busy index WDT-1 the timeout wins
        a, m, r, inv = guard_step(prop, pv, mode, v, i, t, bool(vv), bool(iv), bool(tv), gc, inv)
        a, m, r, inv = int(a), int(m), int(r), int(inv)
        rows.append((arrive, prop, v, i, t, vv, iv, tv, gc, a, m, r))
        mode = m
    return rows


def main():
    rng = np.random.default_rng(2026)
    n = dict(guard=w("vec_guard.txt", guard_vectors()), tok=w("vec_tok.txt", tok_vectors(rng)),
             rule=w("vec_rule.txt", rule_vectors()), fsm=w("vec_fsm.txt", fsm_vectors(rng)),
             gseq=w("vec_gseq.txt", gseq_vectors(rng)))
    print(json.dumps(n))


if __name__ == "__main__":
    main()
