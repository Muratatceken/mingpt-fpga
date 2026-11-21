"""Closed-loop episode runner: plant + policy + guard, token recording and (optional) rollout labels.

run_episodes(plant, policy, N, label=True) -> dict of per-step arrays (B, N):
    tV tI tT mode act prop reason   (tokens / guard trace)
    v i t vv iv tv                  (raw codes, valid flags)
    a_star acc(B,N,4) costs(B,N,4,3) (labels, when label=True)
    ftype ... (copied hidden parameters, for analysis only)
Behaviour "oracle_eps": with prob eps a random action, else the label a* (needs label=True).
"""
import numpy as np

from contract import M_NORMAL, NORMAL
from guard import guard_step
from labeler import label_step
from tokenizer import episode_tokens, obs_tokens


def run_episodes(plant, policy, N, label=True, H=None, oracle_eps=None, rng=None, gc=None, prop_valid=None,
                 record_hidden=False, label_at=None):
    """label_at: optional set of steps at which to compute labels (default: every step when label=True)."""
    B = plant.B
    plant.reset()
    policy.reset()
    mode = np.full(B, M_NORMAL, dtype=np.int64)
    inv = np.zeros(B, dtype=np.int64)
    prev = np.full(B, NORMAL, dtype=np.int64)                    # initial state: payload on, mode NORMAL
    rec = {k: np.zeros((B, N), dtype=np.int64) for k in ("tV", "tI", "tT", "mode", "act", "prop", "reason",
                                                          "v", "i", "t")}
    for k in ("vv", "iv", "tv"):
        rec[k] = np.zeros((B, N), dtype=bool)
    if label:
        rec["a_star"] = np.zeros((B, N), dtype=np.int64)
        rec["acc"] = np.zeros((B, N, 4), dtype=bool)
        rec["costs"] = np.zeros((B, N, 4, 3), dtype=np.int64)
    if record_hidden:
        rec["v_true"] = np.zeros((B, N))
        rec["i_bus"] = np.zeros((B, N))
    for k in range(N):
        plant.actuate()
        v, i, t, vv, iv, tv = plant.observe()
        tV, tI, tT = obs_tokens(v, i, t, vv, iv, tv)
        for key, val in (("tV", tV), ("tI", tI), ("tT", tT), ("mode", mode), ("v", v), ("i", i), ("t", t),
                         ("vv", vv), ("iv", iv), ("tv", tv)):
            rec[key][:, k] = val
        if record_hidden:
            h = plant.hidden()
            rec["v_true"][:, k], rec["i_bus"][:, k] = h["v_true"], h["i_bus"]
        if label and (label_at is None or k in label_at):
            a_star, acc, costs = label_step(plant, mode, inv, H=H)
            rec["a_star"][:, k], rec["acc"][:, k], rec["costs"][:, k] = a_star, acc, costs
        prop = policy.act(tV, tI, tT, mode, prev)
        if oracle_eps is not None:
            sel = oracle_eps[0]                                    # bool mask of episodes using the oracle
            rnd = rng.random(B) < oracle_eps[1]
            o = np.where(rnd, rng.integers(0, 4, B), rec["a_star"][:, k])
            prop = np.where(sel, o, prop)
        pv = True if prop_valid is None else prop_valid[:, k]
        g = 0 if gc is None else gc[:, k]
        applied, mode_new, reason, inv = guard_step(prop, pv, mode, v, i, t, vv, iv, tv, g, inv)
        rec["prop"][:, k], rec["act"][:, k], rec["reason"][:, k] = prop, applied, reason
        plant.command(applied, mode_new)
        prev, mode = applied, mode_new
        plant.physics()
    rec["seq"] = episode_tokens(rec["tV"], rec["tI"], rec["tT"], rec["mode"], rec["act"])
    for key in ("ftype", "f_onset", "oc_kind", "noise"):
        rec[key] = plant.P[key].copy()
    return rec
