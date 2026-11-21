"""Telemetry -> tokens (contract v0).  Per step: V I T MODE ACT_applied.  Sequence: BOS, step0, step1, ...

Decision logits for step k are read at the MODE token of step k, i.e. at sequence position 4 + 5k
(BOS at 0; step k occupies positions 1+5k .. 5+5k).  Binning is "number of edges <= code", identical in RTL.
"""
import numpy as np

from contract import (ACT_BASE, BOS, EDGES, I_BASE, I_INV, MODE_BASE, PAD, T_BASE, T_INV, TOK_PER_STEP, V_BASE,
                      V_INV, bin_code)


def obs_tokens(v, i, t, vv, iv, tv):
    """raw codes + valid flags (arrays, same shape) -> (tokV, tokI, tokT) int arrays."""
    tV = np.where(vv, V_BASE + bin_code(v, "V"), V_INV)
    tI = np.where(iv, I_BASE + bin_code(i, "I"), I_INV)
    tT = np.where(tv, T_BASE + bin_code(t, "T"), T_INV)
    return tV.astype(np.int64), tI.astype(np.int64), tT.astype(np.int64)


def bins_from_tokens(tV, tI, tT):
    """token ids -> (vb, ib, tb, any_invalid).  Invalid channels give bin -1."""
    vb = np.where(tV == V_INV, -1, tV - V_BASE)
    ib = np.where(tI == I_INV, -1, tI - I_BASE)
    tb = np.where(tT == T_INV, -1, tT - T_BASE)
    return vb, ib, tb, (vb < 0) | (ib < 0) | (tb < 0)


def episode_tokens(tV, tI, tT, mode, act):
    """per-step token arrays of shape (B, N) -> sequences (B, 1+5N)."""
    B, N = tV.shape
    seq = np.empty((B, 1 + TOK_PER_STEP * N), dtype=np.int64)
    seq[:, 0] = BOS
    seq[:, 1::5] = tV
    seq[:, 2::5] = tI
    seq[:, 3::5] = tT
    seq[:, 4::5] = MODE_BASE + mode
    seq[:, 5::5] = ACT_BASE + act
    return seq


def decision_pos(k):
    """sequence index of the MODE token of step k (logits there predict the action of step k)."""
    return 4 + TOK_PER_STEP * np.asarray(k)


def first_diff_step_from_end(seqA, seqB, k0):
    """for two token sequences and decision step k0: number of trailing steps (up to and incl. the
    observation part of step k0) that are identical.  d = k0 - last differing step.
    Compares the step tokens V,I,T,MODE of step k0 and V,I,T,MODE,ACT of earlier steps."""
    last = decision_pos(k0)
    a, b = seqA[1:last + 1], seqB[1:last + 1]
    diff = np.nonzero(a != b)[0]
    if len(diff) == 0:
        return None                                    # identical histories (indistinguishable)
    return k0 - int(diff[-1] // TOK_PER_STEP)


__all__ = ["obs_tokens", "bins_from_tokens", "episode_tokens", "decision_pos", "first_diff_step_from_end",
           "PAD", "EDGES"]
