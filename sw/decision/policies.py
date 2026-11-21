"""Decision policies on the token stream (vectorised over episodes).

Every policy sees exactly what the learned models see: per step the tokens V I T MODE and the previously
applied action.  The rule and the FSM are written in "RTL style" (integer bins, explicit registers) so that
rtl/decision/{rule_policy,fsm_policy}.sv can be compared step by step.

  RulePolicy      latest observation only (contract.rule)
  FSMPolicy       history-aware: OC event count, load-switch response tracking, sticky suspect flags
  RandomSafe      random actions held for a geometric dwell (behaviour policy for training data)
  Scripted        per-episode action schedule (template prefixes)
"""
import numpy as np

from contract import (GUVENLI_MOD, KISITLI_MOD, NORMAL, RULE_I_OC, RULE_T_SAFE, RULE_V_RESTR, RULE_V_SAFE,
                      YUKU_KAPAT)
from tokenizer import bins_from_tokens


def rule_action(vb, ib, tb, inv):
    """memoryless threshold rule on bins (vectorised)."""
    a = np.full(np.shape(vb), NORMAL, dtype=np.int64)
    a = np.where(vb <= RULE_V_RESTR, KISITLI_MOD, a)
    a = np.where(ib >= RULE_I_OC, YUKU_KAPAT, a)
    a = np.where((vb <= RULE_V_SAFE) | (tb >= RULE_T_SAFE), GUVENLI_MOD, a)
    a = np.where(inv, KISITLI_MOD, a)
    return a


class RulePolicy:
    name = "rule"

    def __init__(self, B):
        self.B = B

    def reset(self, idx=None):
        pass

    def act(self, tV, tI, tT, mode, prev_act):
        vb, ib, tb, inv = bins_from_tokens(tV, tI, tT)
        return rule_action(vb, ib, tb, inv)


# FSM register widths (documented for RTL): ocn 2b, ocp 1b, swt 2b, swdir 1b, vref 5b, iref 4b, vresp 1b, cresp 1b,
# rose 1b, vpeak 5b, sens 1b, batt 1b, prev_pl 1b, prev_ib 4b, prev_vb 5b, started 1b, vmin_on 5b -> 41 bits
SW_WIN = 3          # steps observed after a payload switch


class FSMPolicy:
    name = "fsm"

    def __init__(self, B):
        self.B = B
        self.reset()

    def reset(self, idx=None):
        z = lambda: np.zeros(self.B, dtype=np.int64)
        if idx is None:
            self.ocn, self.ocp, self.swt, self.swdir = z(), z(), z(), z()
            self.vref, self.iref, self.vresp, self.cresp = z(), z(), z(), z()
            self.rose, self.vpeak, self.sens, self.batt = z(), z(), z(), z()
            self.vmin_on = np.full(self.B, 31, dtype=np.int64)
            self.prev_pl, self.prev_ib, self.prev_vb, self.started = z(), z(), z(), z()
        else:
            for k in ("ocn", "ocp", "swt", "swdir", "vref", "iref", "vresp", "cresp", "rose", "vpeak", "sens",
                      "batt", "prev_pl", "prev_ib", "prev_vb", "started"):
                getattr(self, k)[idx] = 0
            self.vmin_on[idx] = 31

    def state(self):
        return np.stack([self.ocn, self.ocp, self.swt, self.swdir, self.vref, self.iref, self.vresp, self.cresp,
                         self.rose, self.vpeak, self.sens, self.batt, self.prev_pl, self.prev_ib, self.prev_vb, self.started,
                         self.vmin_on],
                        axis=1)

    def act(self, tV, tI, tT, mode, prev_act):
        """prev_act: action applied at the previous step (-1 at step 0)."""
        vb, ib, tb, inv = bins_from_tokens(tV, tI, tT)
        pl = (prev_act == NORMAL).astype(np.int64)                    # commanded payload state now
        started = self.started.copy()
        prev_pl = np.where(started == 1, self.prev_pl, pl)
        prev_ib = np.where(started == 1, self.prev_ib, ib)
        prev_vb = np.where(started == 1, self.prev_vb, vb)

        # ---- OC event counting (rising edge of I >= 2000 mA)
        oc_now = (~inv) & (ib >= RULE_I_OC)
        oc_rise = oc_now & (prev_ib < RULE_I_OC)
        ocn = np.where(oc_rise, np.minimum(self.ocn + 1, 3), self.ocn)
        ocp = np.where(oc_now, 1, self.ocp)                           # OC recovery pending

        # ---- payload switch response tracking (the switch commanded at k-1 is visible from step k)
        sw = (started == 1) & (pl != prev_pl)
        swt = np.where(sw, SW_WIN, self.swt)
        swdir = np.where(sw, 1 - pl, self.swdir)                       # 1 = switched off
        vref = np.where(sw, prev_vb, self.vref)                        # readings before the switch
        iref = np.where(sw, prev_ib, self.iref)
        vresp = np.where(sw, 0, self.vresp)
        cresp = np.where(sw, 0, self.cresp)
        active = (swt > 0) & (~inv)
        cresp = np.where(active & (np.abs(ib - iref) >= 2), 1, cresp)
        vresp = np.where(active & (np.abs(vb - vref) >= 1), 1, vresp)
        new_rose = active & (swdir == 1) & (vb >= vref + 2)
        rose = np.where(new_rose, 1, self.rose)
        # peak V while the load is off (restarted at every switch-off)
        vpeak = np.where(sw & (pl == 0), np.maximum(vb, 0), self.vpeak)
        vpeak = np.where((pl == 0) & (~inv), np.maximum(vpeak, vb), vpeak)
        done = (swt == 1)                                              # last step of the response window
        swt = np.where(swt > 0, swt - 1, swt)
        sens = np.where(done & (cresp == 1) & (vresp == 0), 1, self.sens)
        # battery suspect: after rising on load shed, V falls below its peak at constant (off) load and
        # ends below 7.0 V -- a healthy battery does not lose voltage at constant low load
        batt = np.where((rose == 1) & (pl == 0) & (~inv) & (vb <= RULE_V_RESTR) & (vb < vpeak), 1, self.batt)
        # ... or the loaded voltage once sagged below 6.6 V and V is still below 7.0 V with the load off
        vmin_on = np.where((pl == 1) & (~inv), np.minimum(self.vmin_on, vb), self.vmin_on)
        batt = np.where((vmin_on <= RULE_V_SAFE) & (pl == 0) & (~inv) & (vb <= RULE_V_RESTR), 1, batt)

        # ---- decision
        a = np.full(self.B, NORMAL, dtype=np.int64)
        low_v = (~inv) & (vb <= RULE_V_RESTR)
        probing = (pl == 0) & (swt > 0)
        a = np.where(low_v & (pl == 1), YUKU_KAPAT, a)                  # probe: shed load, watch the response
        a = np.where(low_v & probing, YUKU_KAPAT, a)
        a = np.where(low_v & (pl == 0) & (swt == 0), KISITLI_MOD, a)
        a = np.where((ocp == 1) & (~oc_now), np.where(ocn >= 2, KISITLI_MOD, NORMAL), a)   # OC recovery
        a = np.where(oc_now, np.where(ocn >= 2, KISITLI_MOD, YUKU_KAPAT), a)
        a = np.where(sens == 1, np.maximum(a, KISITLI_MOD), a)
        a = np.where(batt == 1, GUVENLI_MOD, a)
        a = np.where((~inv) & (sens == 0) & (vb <= RULE_V_SAFE), GUVENLI_MOD, a)
        a = np.where((~inv) & (tb >= RULE_T_SAFE), GUVENLI_MOD, a)
        a = np.where(inv, np.maximum(a, KISITLI_MOD), a)
        ocp = np.where((ocp == 1) & (~oc_now), 0, ocp)                # recovery decision taken

        # ---- commit registers
        self.ocn, self.ocp, self.swt, self.swdir = ocn, ocp, swt, swdir
        self.vref, self.iref, self.vresp, self.cresp = vref, iref, vresp, cresp
        self.rose, self.vpeak, self.sens, self.batt = rose, vpeak, sens, batt
        self.vmin_on = vmin_on
        self.prev_pl, self.prev_ib, self.started = pl, np.where(inv, prev_ib, ib), np.ones_like(started)
        self.prev_vb = np.where(inv, prev_vb, vb)
        return a


class RandomSafe:
    """random action held for a geometric dwell (mean ~3 steps); behaviour policy only."""
    name = "random"

    def __init__(self, B, rng, mean_dwell=3.0):
        self.B, self.rng, self.p = B, rng, 1.0 / mean_dwell
        self.cur = np.zeros(B, dtype=np.int64)

    def reset(self, idx=None):
        pass

    def act(self, tV, tI, tT, mode, prev_act):
        ch = self.rng.random(self.B) < self.p
        self.cur = np.where(ch, self.rng.choice(4, size=self.B, p=[0.4, 0.25, 0.25, 0.1]), self.cur)
        return self.cur.copy()


class Scripted:
    """schedule: int array (B, N) of proposals, -1 = defer to `fallback` policy."""
    name = "scripted"

    def __init__(self, schedule, fallback):
        self.s, self.fb, self.k = schedule, fallback, 0

    def reset(self, idx=None):
        self.k = 0
        self.fb.reset(idx)

    def act(self, tV, tI, tT, mode, prev_act):
        a_fb = self.fb.act(tV, tI, tT, mode, prev_act)
        s = self.s[:, self.k] if self.k < self.s.shape[1] else np.full(len(a_fb), -1)
        self.k += 1
        return np.where(s >= 0, s, a_fb)
