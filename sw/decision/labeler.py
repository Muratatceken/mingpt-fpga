"""Counterfactual rollout labels (WP2.4, 2209-A Yöntem "Doğru Eylem ve Eğitim Etiketleri").

At decision step k the plant state is copied four times; branch b applies first proposal b (through the guard)
and continues for H steps with the monotone rule  proposal = max(rule(obs), b)  (contract.labeler).  All
branches use the episode's noise bank (common random numbers).  Costs from hidden state, per step:
    G  safety: V_true < 6.2 V, I_bus > 3.0 A, T > 50 C, SoC < 5 %
    K  unmet fault requirement: mode below the required mode, or payload on during an OC event
    T  avoidable service loss: payload commanded off although no fault requires it
    R  unnecessary restriction: mode KISITLI or GUVENLI although no fault requires a restricted mode
    S  unnecessary safe mode: mode GUVENLI although no fault requires it
    (the third cost component is T+R+S; it orders NORMAL < YUKU_KAPAT < KISITLI < GUVENLI when nothing is required)
Branches are compared lexicographically on (G, K, T+R+S); acceptable set: equal G and K, T+R+S <= min + tau.
Reference action a* = most severe action among the minimisers (same tie rule as the decoders).
"""
import numpy as np

from contract import C, GUVENLI_MOD, LAB, M_GUVENLI, M_KISITLI, NORMAL
from guard import guard_step
from policies import rule_action
from tokenizer import bins_from_tokens, obs_tokens

SAFE = LAB["safety_true"]
GRACE = C["faults"]["requirement_grace_steps"]


def rollout_costs(plant, mode, inv_count, H=None, continuation="monotone"):
    """plant: state after actuate()+observe() at step k.  Returns costs (B, 4, 3) int: G, K, T+S."""
    H = LAB["horizon"] if H is None else H
    B = plant.B
    cp = plant.clone(4)
    # labels may use the current hidden state but not future random events: faults that have not started by
    # the decision step are removed from the branches (no look-ahead into fault onsets)
    future = cp.P["f_onset"] > cp.t
    cp.P["ftype"] = np.where(future, 0, cp.P["ftype"])
    cp.oc_next = np.where(future, 1 << 30, cp.oc_next)
    a0 = np.tile(np.arange(4), B)
    mode = np.repeat(np.asarray(mode), 4)
    inv = np.repeat(np.asarray(inv_count), 4)
    G = np.zeros(4 * B, dtype=np.int64)
    K = np.zeros_like(G)
    TS = np.zeros_like(G)
    for h in range(H):
        if h > 0:
            cp.actuate()
        v, i, t, vv, iv, tv = cp.observe()
        if h == 0:
            prop = a0
        else:
            vb, ib, tb, anyinv = bins_from_tokens(*obs_tokens(v, i, t, vv, iv, tv))
            r = rule_action(vb, ib, tb, anyinv)
            prop = np.maximum(r, a0) if continuation == "monotone" else r
        applied, mode, _, inv = guard_step(prop, True, mode, v, i, t, vv, iv, tv, 0, inv)
        hid = cp.hidden()
        G += ((hid["v_true"] < SAFE["V_min_mV"] / 1000.0) | (hid["i_bus"] > SAFE["I_max_mA"] / 1000.0)
              | (hid["temp"] > SAFE["T_max_dC"] / 10.0) | (hid["soc"] < SAFE["SoC_min"]))
        req_mode, req_off = cp.requirement(GRACE)
        K += (mode < req_mode) | (hid["oc_in"] & (hid["payload"] == 1))
        TS += (((applied != NORMAL) & ~req_off) + ((mode >= M_KISITLI) & (req_mode < 1))
               + ((mode == M_GUVENLI) & (req_mode < 2)))
        cp.command(applied, mode)
        cp.physics()
    return np.stack([G, K, TS], axis=1).reshape(B, 4, 3)


def labels_from_costs(costs, tau=None):
    """costs (B,4,3) -> a_star (B,), acceptable (B,4) bool."""
    tau = LAB["tolerance_tau"] if tau is None else tau
    G, K, TS = costs[..., 0], costs[..., 1], costs[..., 2]
    gmin = G.min(1, keepdims=True)
    ok = G == gmin
    kmin = np.where(ok, K, 1 << 30).min(1, keepdims=True)
    ok &= K == kmin
    tmin = np.where(ok, TS, 1 << 30).min(1, keepdims=True)
    best = ok & (TS == tmin)
    acc = ok & (TS <= tmin + tau)
    a_star = 3 - np.argmax(best[:, ::-1], axis=1)                # most severe minimiser
    return a_star.astype(np.int64), acc


def label_step(plant, mode, inv_count, H=None, tau=None, continuation="monotone"):
    c = rollout_costs(plant, mode, inv_count, H=H, continuation=continuation)
    a, acc = labels_from_costs(c, tau=tau)
    return a, acc, c


__all__ = ["rollout_costs", "labels_from_costs", "label_step", "GUVENLI_MOD"]
