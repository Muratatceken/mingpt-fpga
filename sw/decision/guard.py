"""Independent safety guard (golden model of rtl/decision/guard.sv).  Vectorised over episodes.

Inputs per step : proposal (0..3), prop_valid (bool; False = watchdog timeout), mode (0..2),
                  raw codes v,i,t + valid flags, ground command (0 none, 1 RESUME, 2 FORCE_SAFE)
State           : inv_count (consecutive steps with any invalid channel, saturating at 7)
Outputs         : applied action, new mode, reason code, new inv_count
Priority        : FORCE_SAFE > hard limit V/T > RESUME > (timeout fallback, hard limit I, invalid, table)
"""
import numpy as np

from contract import (GC_FORCE_SAFE, GC_RESUME, GUVENLI_MOD, INV_PERSIST, KISITLI_MOD, LIM_I, LIM_T, LIM_V,
                      M_GUVENLI, M_KISITLI, M_NORMAL, NORMAL, R_DENY, R_FORCE, R_INVALID, R_LIM_I, R_LIM_VT, R_OK,
                      R_RESUME, R_TIMEOUT, TT, YUKU_KAPAT)

INV_SAT = 7


def mode_after(a, mode):
    """mode after applying action a (YUKU_KAPAT keeps the mode)."""
    return np.select([a == NORMAL, a == KISITLI_MOD, a == GUVENLI_MOD], [M_NORMAL, M_KISITLI, M_GUVENLI], mode)


def guard_step(prop, prop_valid, mode, v, i, t, vv, iv, tv, gc, inv_count):
    prop, mode, gc = (np.asarray(x, dtype=np.int64) for x in (prop, mode, gc))
    mode = np.where(mode > M_GUVENLI, M_GUVENLI, mode)            # illegal mode code -> treated as GUVENLI
    prop_valid = np.asarray(prop_valid, dtype=bool)
    v, i, t = (np.asarray(x, dtype=np.int64) for x in (v, i, t))
    vv, iv, tv = (np.asarray(x, dtype=bool) for x in (vv, iv, tv))
    any_inv = ~(vv & iv & tv)
    inv_count = np.where(any_inv, np.minimum(np.asarray(inv_count) + 1, INV_SAT), 0)
    inv_persist = inv_count >= INV_PERSIST
    lim_vt = (vv & (v < LIM_V)) | (tv & (t >= LIM_T))
    lim_i = iv & (i >= LIM_I)

    # tier 3: proposal / timeout fallback, then I limit, invalid, transition table
    fallback = np.where(mode == M_GUVENLI, GUVENLI_MOD, KISITLI_MOD)
    a = np.where(prop_valid, prop, fallback)
    reason = np.where(prop_valid, R_OK, R_TIMEOUT)
    c = lim_i & (a < YUKU_KAPAT)
    a, reason = np.where(c, YUKU_KAPAT, a), np.where(c, R_LIM_I, reason)
    c = inv_persist & (a < KISITLI_MOD)
    a, reason = np.where(c, KISITLI_MOD, a), np.where(c, R_INVALID, reason)
    rep = np.full_like(a, -1)
    for (mi, ai), r in TT.items():
        if r is not None:
            rep = np.where((mode == mi) & (a == ai), r, rep)
    c = rep >= 0
    a, reason = np.where(c, rep, a), np.where(c, R_DENY, reason)

    # tier 2: RESUME (only from GUVENLI with no active limit)
    c = (gc == GC_RESUME) & (mode == M_GUVENLI) & ~lim_vt & ~lim_i
    a, reason = np.where(c, KISITLI_MOD, a), np.where(c, R_RESUME, reason)
    # tier 1: hard V/T limit and FORCE_SAFE
    a, reason = np.where(lim_vt, GUVENLI_MOD, a), np.where(lim_vt, R_LIM_VT, reason)
    c = gc == GC_FORCE_SAFE
    a, reason = np.where(c, GUVENLI_MOD, a), np.where(c, R_FORCE, reason)
    return a, mode_after(a, mode), reason, inv_count


def guard_scalar(prop, prop_valid, mode, v, i, t, vv, iv, tv, gc, inv_count):
    a, m, r, n = guard_step(prop, prop_valid, mode, v, i, t, vv, iv, tv, gc, inv_count)
    return int(a), int(m), int(r), int(n)
