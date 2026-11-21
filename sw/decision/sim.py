"""Satellite power-subsystem plant (WP2.1-2.3), vectorised over episodes (struct of arrays, numpy).

Hidden state x: SoC, polarisation voltage vp, battery temperature, physical payload switch, housekeeping
multiplier, fault state.  Observation y = sensor codes (mV, mA, 0.1 C) + valid flags.  Step = 1 s.

Order inside step k:  (1) due actuations   (2) observe y_k   (3) [caller: policy + guard -> command(k)]
                      (4) physics k -> k+1
Battery (2S2P Li-ion, single RC branch):
    I_bus = payload*(I_nom + dI_oc) + I_hk*hk          I_b = I_bus - I_sa
    V     = OCV(SoC) - dVf - I_b*R0(T) - vp            R0(T) = R0*r0m*(1 + 0.01*max(0, 25-T))
    SoC'  = SoC - I_b/(3600*Q_eff)    vp' = a*vp + (1-a)*Rp*I_b, a = exp(-1/tau_p)
    T'    = T + (I_b^2*(R0(T)+Rp) - (T - T_amb)/R_th)/C_th
Faults: OC (transient latch-up or intermittent), BAT (capacity loss ramp dVf, Q_eff, R0), SEN (stuck V sensor).
All randomness comes from a per-episode noise bank indexed by absolute step (common random numbers for
pair members and labeler branches).
"""
import hashlib

import numpy as np

from contract import C, KISITLI_MOD, M_GUVENLI, MODE_HK, NORMAL

OCV_SOC = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
OCV_V = np.array([6.00, 6.50, 6.80, 7.05, 7.25, 7.40, 7.52, 7.64, 7.76, 7.90, 8.08, 8.40])
F_NONE, F_OC, F_BAT, F_SEN = 0, 1, 2, 3
FAULTS = ["NONE", "OC", "BAT", "SEN"]
OC_TRANSIENT, OC_INTERMITTENT = 0, 1
INF = 1 << 30


def ocv(soc):
    return np.interp(soc, OCV_SOC, OCV_V)


def soc_for_ocv(v):
    return np.interp(v, OCV_V, OCV_SOC)


def seed_of(*parts):
    """stable 64-bit seed from strings/ints (sha256; never Python's hash())."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


class NoiseBank:
    """standard normals / uniforms per (episode, absolute step); rows are shared by clones (src index)."""

    def __init__(self, seeds, T):
        self.T = T
        B = len(seeds)
        self.zv, self.zi, self.zt = (np.empty((B, T)) for _ in range(3))
        self.udrop = np.empty((B, T, 3))
        self.uineff = np.empty((B, T))
        for b, s in enumerate(seeds):
            r = np.random.default_rng(s)
            self.zv[b], self.zi[b], self.zt[b] = r.standard_normal((3, T))
            self.udrop[b] = r.random((T, 3))
            self.uineff[b] = r.random(T)

    @staticmethod
    def concat(banks):
        nb = NoiseBank.__new__(NoiseBank)
        nb.T = min(b.T for b in banks)
        for k in ("zv", "zi", "zt", "udrop", "uineff"):
            setattr(nb, k, np.concatenate([getattr(b, k)[:, :nb.T] for b in banks]))
        return nb


PARAM_KEYS = ["soc0", "q_ah", "r0", "rp", "tau_p", "cth", "rth", "tamb", "i_nom", "i_hk", "i_sa", "delay",
              "noise", "p_ineff", "ftype", "f_onset", "oc_dI", "oc_kind", "oc_interval", "bat_A", "bat_tau",
              "bat_qm", "bat_r0m", "sen_value"]


def sample_params(rng, B, noise=None, ftype=None, train=True):
    """nominal parameter ranges (contract/D1.3).  Returns dict of arrays."""
    u = lambda lo, hi: rng.uniform(lo, hi, B)
    P = dict(
        soc0=u(0.55, 0.90), q_ah=np.full(B, 5.2), r0=u(0.12, 0.30), rp=u(0.03, 0.08), tau_p=u(2.0, 6.0),
        cth=u(100.0, 200.0), rth=u(2.0, 4.0), tamb=u(0.0, 30.0), i_nom=u(0.9, 1.3), i_hk=u(0.08, 0.15),
        i_sa=u(0.0, 0.3), delay=(rng.integers(0, 2, B) if train else np.zeros(B, dtype=np.int64)),
        noise=(rng.integers(0, 2, B) if noise is None else np.full(B, noise)),
        p_ineff=np.full(B, 0.05 if train else 0.0),
        ftype=(rng.integers(0, 4, B) if ftype is None else np.full(B, ftype)),
        f_onset=rng.integers(5, 60, B),
        oc_dI=np.where(rng.random(B) < 0.5, u(0.5, 0.8), u(1.0, 1.5)),
        oc_kind=(rng.random(B) < 0.3).astype(np.int64),          # 30% intermittent, 70% transient
        oc_interval=rng.integers(5, 41, B),
        bat_A=np.where(rng.random(B) < 0.5, u(0.2, 0.3), u(0.4, 0.6)), bat_tau=u(2.0, 6.0),
        bat_qm=u(0.2, 0.5), bat_r0m=np.full(B, 1.3),
        sen_value=np.zeros(B),                                    # filled at onset (stuck at reading +- gap)
    )
    gap = np.where(rng.random(B) < 0.5, u(0.2, 0.3), u(0.4, 0.6)) * np.where(rng.random(B) < 0.5, -1.0, 1.0)
    P["sen_gap"] = gap                                            # stuck value = true V at onset + gap
    return P


class Plant:
    def __init__(self, P, bank, src=None):
        self.P = {k: np.asarray(v).copy() for k, v in P.items()}
        self.B = len(self.P["soc0"])
        self.bank = bank
        self.src = np.arange(self.B) if src is None else np.asarray(src)
        sig = C["noise"]
        lv = self.P["noise"].astype(int)
        self.sv = np.array([sig["low"]["V_mV"], sig["high"]["V_mV"]], float)[lv]
        self.si = np.array([sig["low"]["I_mA"], sig["high"]["I_mA"]], float)[lv]
        self.st = np.array([sig["low"]["T_dC"], sig["high"]["T_dC"]], float)[lv]
        self.pd = np.array([sig["low"]["dropout"], sig["high"]["dropout"]], float)[lv]
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self):
        P, B = self.P, self.B
        self.t = 0
        self.soc = P["soc0"].copy()
        self.payload = np.ones(B, dtype=np.int64)
        self.hk = np.ones(B)
        self.pend_pl = np.ones(B, dtype=np.int64)
        self.pend_hk = np.ones(B)
        self.pend_t = np.full(B, -INF, dtype=np.int64)
        self.oc_in = np.zeros(B, dtype=bool)
        self.oc_n = np.zeros(B, dtype=np.int64)
        self.oc_next = np.where(P["ftype"] == F_OC, P["f_onset"], INF).astype(np.int64)
        self.oc_cleared = np.zeros(B, dtype=bool)
        self.sen_set = np.zeros(B, dtype=bool)
        self.sen_val = P["sen_value"].copy()
        Ib = self.i_bus() - P["i_sa"]
        self.vp = P["rp"] * Ib                                    # steady state
        self.temp = P["tamb"] + Ib ** 2 * (P["r0"] + P["rp"]) * P["rth"]
        self._oc_update()

    STATE_KEYS = ["soc", "payload", "hk", "pend_pl", "pend_hk", "pend_t", "oc_in", "oc_n", "oc_next", "oc_cleared",
                  "sen_set", "sen_val", "vp", "temp"]

    def clone(self, rep):
        """repeat every episode `rep` times (branch b of episode e at index e*rep+b)."""
        nP = {k: np.repeat(v, rep, axis=0) for k, v in self.P.items()}
        c = Plant.__new__(Plant)
        c.P, c.B, c.bank = nP, self.B * rep, self.bank
        c.src = np.repeat(self.src, rep)
        for k in ("sv", "si", "st", "pd"):
            setattr(c, k, np.repeat(getattr(self, k), rep))
        c.t = self.t
        for k in self.STATE_KEYS:
            setattr(c, k, np.repeat(getattr(self, k), rep, axis=0))
        return c

    # ------------------------------------------------------------ physics helpers
    def bat_on(self, t=None):
        t = self.t if t is None else t
        return (self.P["ftype"] == F_BAT) & (t >= self.P["f_onset"])

    def dvf(self, t=None):
        t = self.t if t is None else t
        P = self.P
        dt = np.maximum(t - P["f_onset"], 0)
        return np.where(self.bat_on(t), P["bat_A"] * (1.0 - np.exp(-dt / P["bat_tau"])), 0.0)

    def i_bus(self):
        P = self.P
        extra = np.where(self.oc_in & (self.payload == 1), P["oc_dI"], 0.0)
        return self.payload * (P["i_nom"] + extra) + P["i_hk"] * self.hk

    def r0_eff(self):
        P = self.P
        r0m = np.where(self.bat_on(), P["bat_r0m"], 1.0)
        return P["r0"] * r0m * (1.0 + 0.01 * np.maximum(0.0, 25.0 - self.temp))

    def v_true(self):
        Ib = self.i_bus() - self.P["i_sa"]
        return ocv(self.soc) - self.dvf() - Ib * self.r0_eff() - self.vp

    # ------------------------------------------------------------ fault bookkeeping (after actuation)
    def _oc_update(self):
        P, t = self.P, self.t
        is_oc = P["ftype"] == F_OC
        end = self.oc_in & (self.payload == 0)
        self.oc_in = np.where(end, False, self.oc_in)
        self.oc_cleared |= end & (P["oc_kind"] == OC_TRANSIENT)
        start = is_oc & ~self.oc_in & ~self.oc_cleared & (self.payload == 1) & (t >= self.oc_next)
        self.oc_in = self.oc_in | start
        self.oc_n = self.oc_n + start
        self.oc_next = np.where(start, INF, self.oc_next)

    # ------------------------------------------------------------ the step protocol
    def actuate(self):
        due = self.t >= self.pend_t + self.P["delay"]
        was_off = self.payload == 0
        self.payload = np.where(due, self.pend_pl, self.payload)
        self.hk = np.where(due, self.pend_hk, self.hk)
        turned_on = was_off & (self.payload == 1)
        P = self.P
        inter = (P["ftype"] == F_OC) & (P["oc_kind"] == OC_INTERMITTENT) & (self.oc_n >= 1)
        self.oc_next = np.where(turned_on & inter & ~self.oc_in, self.t + P["oc_interval"], self.oc_next)
        self._oc_update()

    def observe(self):
        """sensor codes + valid flags at the current step (after actuation)."""
        P, t, s = self.P, self.t, self.src
        vt = self.v_true()
        # SEN: the sensor freezes at onset (stuck value = reading at onset + gap, or a template-given value)
        sen_now = (P["ftype"] == F_SEN) & (t >= P["f_onset"])
        newly = sen_now & ~self.sen_set
        self.sen_val = np.where(newly & (self.sen_val == 0), vt + P.get("sen_gap", 0.0), self.sen_val)
        self.sen_set |= newly
        v_read = np.where(sen_now, self.sen_val, vt)
        v = np.rint(v_read * 1000.0 + self.sv * self.bank.zv[s, t]).astype(np.int64)
        i = np.rint(self.i_bus() * 1000.0 + self.si * self.bank.zi[s, t]).astype(np.int64)
        tc = np.rint(self.temp * 10.0 + self.st * self.bank.zt[s, t]).astype(np.int64)
        u = self.bank.udrop[s, t]
        vv, iv, tv = u[:, 0] >= self.pd, u[:, 1] >= self.pd, u[:, 2] >= self.pd
        return v, i, tc, vv, iv, tv

    def command(self, applied, mode_new):
        """schedule the applied action (takes effect after `delay` steps; switch commands can fail)."""
        P, t = self.P, self.t
        pl = (np.asarray(applied) == NORMAL).astype(np.int64)
        hk = MODE_HK[np.asarray(mode_new)]
        fail = (pl != self.payload) & (self.bank.uineff[self.src, t] < P["p_ineff"])
        self.pend_pl = np.where(fail, self.payload, pl)
        self.pend_hk = hk
        self.pend_t = np.full(self.B, t + 1, dtype=np.int64)

    def physics(self):
        P = self.P
        Ib = self.i_bus() - P["i_sa"]
        r0 = self.r0_eff()
        qeff = P["q_ah"] * np.where(self.bat_on(), P["bat_qm"], 1.0)
        self.soc = np.clip(self.soc - Ib / (3600.0 * qeff), 0.0, 1.0)
        a = np.exp(-1.0 / P["tau_p"])
        self.vp = a * self.vp + (1.0 - a) * P["rp"] * Ib
        self.temp = self.temp + (Ib ** 2 * (r0 + P["rp"]) - (self.temp - P["tamb"]) / P["rth"]) / P["cth"]
        self.t += 1

    # ------------------------------------------------------------ hidden information for the labeler
    def hidden(self):
        """per-episode hidden quantities used for costs (never given to policies)."""
        return dict(v_true=self.v_true(), i_bus=self.i_bus(), temp=self.temp.copy(), soc=self.soc.copy(),
                    payload=self.payload.copy(), oc_in=self.oc_in.copy(), oc_n=self.oc_n.copy())

    def requirement(self, grace):
        """(req_mode, req_off): minimum mode required and whether the payload must be off, from hidden state."""
        P, t = self.P, self.t
        act = t >= P["f_onset"] + grace
        req_mode = np.zeros(self.B, dtype=np.int64)
        req_mode = np.where(act & (P["ftype"] == F_SEN), 1, req_mode)
        req_mode = np.where((P["ftype"] == F_OC) & (P["oc_kind"] == OC_INTERMITTENT) & (self.oc_n >= 1)
                            & (t >= P["f_onset"] + grace), 1, req_mode)
        req_mode = np.where(act & (P["ftype"] == F_BAT), 2, req_mode)
        req_off = (req_mode >= 1) | self.oc_in
        return req_mode, req_off


__all__ = ["Plant", "NoiseBank", "sample_params", "seed_of", "ocv", "soc_for_ocv", "F_NONE", "F_OC", "F_BAT",
           "F_SEN", "FAULTS", "M_GUVENLI", "KISITLI_MOD"]
