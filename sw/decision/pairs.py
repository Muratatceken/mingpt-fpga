"""Matched-history pairs (WP2.4).  Two episodes whose last d steps are token-identical but whose earlier
history differs, so that the correct action at decision step t0 differs.

Templates (member A, member B -> intended labels at t0):
  BAT_SEN  A: stuck V sensor        B: battery capacity loss     -> KISITLI_MOD / GUVENLI_MOD   (Şekil 1)
           script: payload on, load shed at t_off and held off.  B's voltage rises, then the dVf ramp brings it
           back to the plateau at which A's sensor is stuck.
  SEN      A: healthy                B: stuck V sensor            -> NORMAL / KISITLI_MOD
           script: load off at t_off for L steps, back on at t_on.  A's reading responds, B's does not.
  OC       A: transient OC at t0-1   B: intermittent OC (events at t_b1 and t0-1) -> NORMAL / KISITLI_MOD
           script: each OC event is answered with YUKU_KAPAT; B re-enables at t_on.
Construction = search: for every pair a grid of event times x noise seeds is simulated in one batch; the
first candidate that verifies is kept.  Verification: (1) realised d == target, where d = t0 - last step at
which any token differs; (2) no guard override before t0 in either member; (3) labels at t0 equal the
intended ones; (4) acceptable sets disjoint.  D3 = an extra identical load cycle earlier in the episode;
D4 = SEN template without the revealing load cycle (histories identical; reported separately).
"""
import numpy as np

from contract import GUVENLI_MOD, KISITLI_MOD, M_NORMAL, NORMAL, SCN, YUKU_KAPAT
from episodes import run_episodes
from policies import RulePolicy, Scripted
from sim import (F_BAT, F_NONE, F_OC, F_SEN, OC_INTERMITTENT, OC_TRANSIENT, NoiseBank, Plant, sample_params,
                 seed_of, soc_for_ocv)
from tokenizer import first_diff_step_from_end

TEMPLATES = ("BAT_SEN", "SEN", "OC")
INTENDED = {"BAT_SEN": (KISITLI_MOD, GUVENLI_MOD), "SEN": (NORMAL, KISITLI_MOD), "OC": (NORMAL, KISITLI_MOD),
            "D4": (NORMAL, KISITLI_MOD)}
SEV = {"low": (0.2, 0.3), "high": (0.4, 0.6)}
OC_SEV = {"low": (0.5, 0.8), "high": (1.0, 1.5)}
NOISE = {"low": 0, "high": 1}
H_LAB = 60


def _scalar_base(rng, noise):
    P = sample_params(rng, 1, noise=noise, ftype=F_NONE, train=False)
    return {k: float(v[0]) if np.asarray(v).dtype.kind == "f" else int(v[0]) for k, v in P.items()}


def _r0eff(r0, tamb, m=1.0):
    return r0 * m * (1.0 + 0.01 * max(0.0, 25.0 - tamb))


def template_params(template, sev, noise, rng):
    """one shared physical parameter set; returns (pA, pB, info)."""
    p = _scalar_base(rng, NOISE[noise])
    info = {}
    if template == "BAT_SEN":
        A = rng.uniform(*SEV[sev])
        p.update(bat_A=A, bat_tau=rng.uniform(2.0, 4.0), bat_qm=rng.uniform(0.2, 0.5),
                 r0=rng.uniform(0.18, 0.30), i_nom=rng.uniform(1.0, 1.3))
        i_lo = p["i_hk"] - p["i_sa"]
        r0b = _r0eff(p["r0"], p["tamb"], 1.3)
        p["soc0"] = float(soc_for_ocv(6.95 + A + i_lo * (r0b + p["rp"])))
        info["delta"] = int(rng.integers(1, 4))                       # BAT onset precedes the load shed
        pA, pB = dict(p, ftype=F_SEN, f_onset=0), dict(p, ftype=F_BAT)
    elif template in ("SEN", "D4"):
        for _ in range(200):                                          # response magnitude = severity
            gap = rng.uniform(*SEV[sev])
            r0 = gap / p["i_nom"] - p["rp"]
            r0 = r0 / (1.0 + 0.01 * max(0.0, 25.0 - p["tamb"]))
            if 0.08 <= r0 <= 0.40:
                break
            p = _scalar_base(rng, NOISE[noise])
        p["r0"] = r0
        ib = p["i_nom"] + p["i_hk"] - p["i_sa"]
        # loaded steady state half a polarisation swing below the bin centre: the post-switch transient
        # (vp relaxing towards rp*I) then stays inside the 7.2-7.3 V bin
        v_ss = 7.25 - 0.5 * p["rp"] * ib
        p["soc0"] = float(soc_for_ocv(v_ss + ib * (_r0eff(r0, p["tamb"]) + p["rp"])))
        info["L"] = int(rng.integers(3, 7))
        info["ds"] = int(rng.integers(2, 6))
        pA, pB = dict(p, ftype=F_NONE), dict(p, ftype=F_SEN)
    elif template == "OC":
        p.update(oc_dI=rng.uniform(*OC_SEV[sev]))
        info["L"] = int(rng.integers(2, 6))
        pA = dict(p, ftype=F_OC, oc_kind=OC_TRANSIENT)
        pB = dict(p, ftype=F_OC, oc_kind=OC_INTERMITTENT)
    else:
        raise ValueError(template)
    return pA, pB, info


def schedules(template, e, info, t0, N, d3_at=None):
    """proposal schedules (A, B) for event parameter e; -1 entries never occur (fully scripted)."""
    sA = np.full(N, NORMAL, dtype=np.int64)
    sB = np.full(N, NORMAL, dtype=np.int64)
    meta = {}
    if template == "BAT_SEN":
        t_off = e
        sA[t_off - 1:] = YUKU_KAPAT
        sB[t_off - 1:] = YUKU_KAPAT
        meta = dict(t_off=t_off, f_onset_B=t_off - info["delta"], seg_start=t_off - info["delta"])
    elif template == "SEN":
        t_on = e
        t_off = t_on - info["L"]
        for s in (sA, sB):
            s[t_off - 1:t_on - 1] = YUKU_KAPAT
        meta = dict(t_on=t_on, t_off=t_off, f_onset_B=max(t_off - info["ds"], 0), seg_start=t_off - info["ds"])
    elif template == "D4":
        meta = dict(f_onset_B=e, seg_start=e)
    elif template == "OC":
        t_on = e
        t_b1 = t_on - 1 - info["L"]
        sB[t_b1:t_on - 1] = YUKU_KAPAT
        sA[t0 - 1] = YUKU_KAPAT
        sB[t0 - 1] = YUKU_KAPAT
        meta = dict(t_on=t_on, t_b1=t_b1, f_onset_A=t0 - 1, f_onset_B=t_b1, interval_B=(t0 - 1) - t_on,
                    seg_start=t_b1)
    if d3_at is not None:                                             # identical earlier load cycle
        for s in (sA, sB):
            s[d3_at:d3_at + 2] = YUKU_KAPAT
    return sA, sB, meta


def e_grid(template, d, t0):
    if template == "BAT_SEN":
        return list(range(2, t0 - d + 2))
    if template == "SEN":
        return list(range(9, t0 - d + 2))
    if template == "OC":
        return list(range(8, t0 - d + 2))
    if template == "D4":
        return list(range(5, 30))
    raise ValueError(template)


def _stack(dicts):
    keys = dicts[0].keys()
    return {k: np.array([d[k] for d in dicts]) for k in keys}


def generate_cell(split, template, sev, noise, d, n_pairs, difficulty="D2", t0=None, seeds_per_pair=3,
                  max_grid=None, base_index=0):
    """returns (pairs, stats).  pairs: list of dicts (A/B records at the selected candidate)."""
    t0 = SCN["decision_step"] if t0 is None else t0
    N = t0 + 1
    tpl = "D4" if difficulty == "D4" else template
    grid = e_grid(tpl, d, t0)
    if max_grid is not None and len(grid) > max_grid:
        grid = grid[-max_grid:] if tpl in ("BAT_SEN",) else grid[:max_grid]
    cand_A, cand_B, sched_A, sched_B, metas, bank_seeds, owner = [], [], [], [], [], [], []
    for p in range(n_pairs):
        pid = base_index + p
        rngp = np.random.default_rng(seed_of(split, tpl, sev, noise, d, difficulty, pid, "params"))
        pA, pB, info = template_params(tpl, sev, noise, rngp)
        d3_at = None
        for s in range(seeds_per_pair):
            for e in grid:
                sA, sB, meta = schedules(tpl, e, info, t0, N)
                if difficulty == "D3":
                    room = meta["seg_start"] - 4
                    if room < 1:
                        continue
                    d3_at = int(np.random.default_rng(seed_of(split, pid, s, e, "d3")).integers(1, room + 1))
                    sA, sB, meta = schedules(tpl, e, info, t0, N, d3_at=d3_at)
                    meta["d3_at"] = d3_at
                a, b = dict(pA), dict(pB)
                if "f_onset_A" in meta:
                    a["f_onset"] = meta["f_onset_A"]
                if "f_onset_B" in meta:
                    b["f_onset"] = meta["f_onset_B"]
                if tpl == "OC":
                    b["oc_interval"] = meta["interval_B"]
                if tpl in ("SEN", "D4"):
                    a["f_onset"] = 1 << 20
                cand_A.append(a)
                cand_B.append(b)
                sched_A.append(sA)
                sched_B.append(sB)
                metas.append(dict(meta, e=e, seed=s, info=info))
                bank_seeds.append(seed_of(split, tpl, sev, noise, d, difficulty, pid, s, "noise"))
                owner.append(p)
    if not cand_A:
        return [], dict(attempted=n_pairs, accepted=0, candidates=0)
    owner = np.array(owner)
    bank = NoiseBank(bank_seeds, N + H_LAB + 2)
    PA, PB = _stack(cand_A), _stack(cand_B)
    SA, SB = np.stack(sched_A), np.stack(sched_B)
    first, second = ("B", "A") if tpl == "BAT_SEN" else ("A", "B")
    P = {"A": PA, "B": PB}
    S = {"A": SA, "B": SB}
    rec = {}
    lab_at = {t0}
    r1 = run_episodes(Plant(P[first], bank), Scripted(S[first], RulePolicy(len(owner))), N, label=True,
                      H=H_LAB, label_at=lab_at, record_hidden=True)
    if tpl in ("BAT_SEN", "SEN", "D4"):                               # stuck value = partner's plateau
        vt = r1["v_true"]
        if tpl == "BAT_SEN":                                          # B's final plateau (last two steps)
            plateau = vt[:, t0 - 1:t0 + 1].mean(1)
        elif tpl == "SEN":                                            # A's loaded value before the load shed
            toff = np.array([m["t_off"] for m in metas])
            idx = np.clip(toff[:, None] + np.arange(-4, -1)[None, :], 0, t0)
            plateau = np.take_along_axis(vt, idx, axis=1).mean(1)
        else:
            plateau = vt[:, t0 - 5:t0].mean(1)
        P[second]["sen_value"] = plateau
    r2 = run_episodes(Plant(P[second], bank), Scripted(S[second], RulePolicy(len(owner))), N, label=True,
                      H=H_LAB, label_at=lab_at, record_hidden=True)
    rec[first], rec[second] = r1, r2
    rA, rB = rec["A"], rec["B"]
    iA, iB = INTENDED[tpl]
    pairs, chosen = [], set()
    reasons = {"d": 0, "guard": 0, "label": 0, "overlap": 0}
    for c in range(len(owner)):
        p = owner[c]
        if p in chosen:
            continue
        rd = first_diff_step_from_end(rA["seq"][c], rB["seq"][c], t0)
        if tpl == "D4":
            ok_d = rd is None
        else:
            ok_d = rd == d
        if not ok_d:
            reasons["d"] += 1
            continue
        if rA["reason"][c, :t0].any() or rB["reason"][c, :t0].any():
            reasons["guard"] += 1
            continue
        if rA["a_star"][c, t0] != iA or rB["a_star"][c, t0] != iB:
            reasons["label"] += 1
            continue
        if (rA["acc"][c, t0] & rB["acc"][c, t0]).any():
            reasons["overlap"] += 1
            continue
        chosen.add(p)
        pairs.append(dict(
            split=split, template=template, difficulty=difficulty, sev=sev, noise=noise, d=d, t0=t0,
            pair_index=int(base_index + p), meta=metas[c], bank_seed=int(bank_seeds[c]),
            A=_member(rA, P["A"], S["A"], c), B=_member(rB, P["B"], S["B"], c)))
    stats = dict(attempted=n_pairs, accepted=len(pairs), candidates=len(owner), rejected=reasons)
    return pairs, stats


def _member(r, P, S, c):
    keep = ("seq", "tV", "tI", "tT", "mode", "act", "prop", "reason", "v", "i", "t", "vv", "iv", "tv", "a_star",
            "acc", "costs", "v_true", "i_bus")
    m = {k: np.asarray(r[k][c]).copy() for k in keep if k in r}
    m["params"] = {k: (float(v[c]) if np.asarray(v).dtype.kind == "f" else int(v[c])) for k, v in P.items()}
    m["schedule"] = S[c].copy()
    return m


__all__ = ["generate_cell", "TEMPLATES", "INTENDED", "template_params", "schedules", "M_NORMAL", "GUVENLI_MOD"]
