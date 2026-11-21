"""Bit-exact Python model of rtl/decision/decision_top.sv (WP4.3) and its test vectors.

    python3 hw_ref.py <model_name> [policy] [n_scen]     -> build/decision/obs.mem, build/decision/trace_<policy>.txt
policy: lm (engine with mem/decision/<model_name>), rule, fsm.
Scenarios: freshly generated pair members (split "rtl", never val/test) with directed edits: invalid samples,
ground commands (RESUME / FORCE_SAFE), suppressed proposals (watchdog timeout), hard-limit codes.
The trace (proposal, applied, reason, mode before/after, valid) and the FNV-1a style hash must match the RTL.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "zoo"))
from quant import IntRunner  # noqa: E402

from contract import ACT_BASE, BOS, MODE_BASE, NORMAL, ROOT  # noqa: E402
from guard import guard_step  # noqa: E402
from pairs import generate_cell  # noqa: E402
from policies import FSMPolicy, rule_action  # noqa: E402
from tokenizer import bins_from_tokens, obs_tokens  # noqa: E402

OUT = os.path.join(ROOT, "build", "decision")
FNV0, FNVP = 0x811C9DC5, 0x01000193


def decide(l4):
    return 3 - int(np.argmax(np.asarray(l4)[::-1]))


def scenarios(n_scen, rng, steps=48):
    """list of dicts with per-step arrays v,i,t,vv,iv,tv,gc,sup (length steps)."""
    out = []
    per = max(1, n_scen // 12)
    for tpl in ("BAT_SEN", "SEN", "OC"):
        for d in (2, 8, 16, 32):
            prs, _ = generate_cell("rtl", tpl, "high", "high" if d % 16 == 0 else "low", d, per)
            for p in prs:
                for side in ("A", "B"):
                    m = p[side]
                    out.append(dict(v=m["v"][:steps].copy(), i=m["i"][:steps].copy(), t=m["t"][:steps].copy(),
                                    vv=m["vv"][:steps].copy(), iv=m["iv"][:steps].copy(), tv=m["tv"][:steps].copy(),
                                    gc=np.zeros(steps, dtype=np.int64), sup=np.zeros(steps, dtype=np.int64)))
    # directed edits on a few scenarios
    for j, s in enumerate(out):
        kind = j % 6
        if kind == 1:                                   # invalid samples, incl. a persistent run
            s["vv"][10:14] = False
            s["iv"][30] = False
        elif kind == 2:                                 # FORCE_SAFE then RESUME
            s["gc"][12] = 2
            s["gc"][20] = 1
        elif kind == 3:                                 # suppressed proposals -> timeout fallback
            s["sup"][5] = 1
            s["sup"][40:42] = 1
        elif kind == 4:                                 # hard limits
            s["v"][15] = 6399
            s["i"][25] = 2800
            s["t"][35] = 450
            s["gc"][36] = 1
    return out[:n_scen]


def run_ref(scen, policy, q=None):
    """returns trace words (list of int) and hash for one scenario."""
    N = len(scen["v"])
    ir = IntRunner(q) if policy == "lm" else None
    fsm = FSMPolicy(1) if policy == "fsm" else None
    if ir is not None:
        ir.step(BOS, 0)
    pos, mode, inv, prev = 1, 0, 0, NORMAL
    words, h = [], FNV0
    for k in range(N):
        v, i, t = (int(scen[c][k]) for c in ("v", "i", "t"))
        vv, iv, tv = (bool(scen[c][k]) for c in ("vv", "iv", "tv"))
        gc, sup = int(scen["gc"][k]), int(scen["sup"][k])
        tV, tI, tT = (int(x[0]) for x in obs_tokens(np.array([v]), np.array([i]), np.array([t]), np.array([vv]),
                                                    np.array([iv]), np.array([tv])))
        if policy == "lm":
            ir.step(tV, pos)
            ir.step(tI, pos + 1)
            ir.step(tT, pos + 2)
            lg = ir.step(MODE_BASE + mode, pos + 3)
            prop = decide(lg[:4])
        elif policy == "rule":
            vb, ib, tb, anyinv = bins_from_tokens(np.array([tV]), np.array([tI]), np.array([tT]))
            prop = int(rule_action(vb, ib, tb, anyinv)[0])
        else:
            prop = int(fsm.act(np.array([tV]), np.array([tI]), np.array([tT]), np.array([mode]), np.array([prev]))[0])
        pv = sup == 0
        a, m2, r, inv = (int(x) for x in guard_step(prop, pv, mode, v, i, t, vv, iv, tv, gc, inv))
        if policy == "lm":
            ir.step(ACT_BASE + a, pos + 4)
            pos += 5
        w = (int(pv) << 11) | (m2 << 9) | (mode << 7) | (r << 4) | (a << 2) | prop
        words.append(w)
        h = ((h ^ w) * FNVP) & 0xFFFFFFFF
        prev, mode = a, m2
    return words, h


def write_obs(scens, path):
    with open(path, "w") as f:
        for s in scens:
            for k in range(len(s["v"])):
                w = (int(s["v"][k]) & 0xFFFF) | ((int(s["i"][k]) & 0xFFFF) << 16) | ((int(s["t"][k]) & 0xFFFF) << 32)
                w |= (int(s["vv"][k]) << 48) | (int(s["iv"][k]) << 49) | (int(s["tv"][k]) << 50)
                w |= (int(s["gc"][k]) << 51) | (int(s["sup"][k]) << 53)
                f.write(f"{w:016x}\n")


def main():
    name = sys.argv[1]
    policy = sys.argv[2] if len(sys.argv) > 2 else "lm"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    scens = scenarios(n, rng)
    write_obs(scens, os.path.join(OUT, "obs.mem"))
    q = None
    if policy == "lm":
        q = pickle.load(open(os.path.join(ROOT, "build", "decision", f"{name}.qprog.pkl"), "rb"))
    rows = []
    for j, s in enumerate(scens):
        words, h = run_ref(s, policy, q)
        rows.append(dict(base=j * len(s["v"]), n=len(s["v"]), words=words, hash=h))
    with open(os.path.join(OUT, "exp_hash.mem"), "w") as f:            # dec_board_top self-test table
        f.write("".join(f"{r['hash']:08x}\n" for r in rows))
    with open(os.path.join(OUT, f"trace_{policy}.txt"), "w") as f:
        f.write(f"{len(rows)}\n")
        for r in rows:
            f.write(f"{r['base']} {r['n']} {r['hash']} " + " ".join(str(w) for w in r["words"]) + "\n")
    print(json.dumps(dict(policy=policy, scenarios=len(rows), steps=sum(r["n"] for r in rows))))


if __name__ == "__main__":
    main()
