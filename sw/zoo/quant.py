"""
Integer quantisation of an IR Program and the bit-exact integer interpreter (hardware spec).

Formats
  R    residual          INT16, scale 2^-FR (power of two from calibration)
  int8 activations       INT8, per-buffer static scale (calibrated, or fixed by a proven bound)
  NORM output            INT8, scale 2^-ZF with ZF = floor(log2(127/bound)), bound = sqrt(n-1) (LN) / sqrt(n) (RMS)
  tanh / sigmoid output  INT8 Q0.7 (|y| < 1), recurrent state HS likewise
  logits                 INT16, calibrated scale
  GEMV                   INT8 x INT8 -> INT32, input scales folded into the weights, per-output requant m0*2^-sh
  RoPE                   cos/sin in Q1.14; the input is calibrated on the max pair norm, so rotation cannot overflow
"""
import math

import numpy as np

from ir import ACTS, FloatRunner

M0_BITS, NORM_P, NORM_QBITS, ATT_RS, BOW_RS, ROPE_F = 16, 36, 40, 32, 20, 14
EXP_LUT = np.array([int(round(2 ** 15 * 2.0 ** (-k / 256))) for k in range(256)], dtype=np.int64)
Q07 = 2.0 ** -7


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(v, lo, hi)


def qmul(m):
    """positive real m -> (m0, sh) with m0 in [2^15, 2^16)"""
    assert m > 0
    sh = 0
    while m * (1 << sh) < (1 << (M0_BITS - 1)):
        sh += 1
    while m * (1 << sh) >= (1 << M0_BITS):
        sh -= 1
    m0 = int(round(m * (1 << sh)))
    if m0 == 1 << M0_BITS:
        m0, sh = m0 >> 1, sh - 1
    assert 1 <= sh <= 62, (m, sh)
    return m0, sh


def requant(acc, m0, sh, bits):
    """vectorised: (acc*m0 + 2^(sh-1)) >> sh, saturated. acc, m0, sh int64 arrays (or scalars)."""
    acc = np.asarray(acc, dtype=np.int64)
    return sat((acc * m0 + (np.int64(1) << (np.asarray(sh, dtype=np.int64) - 1))) >> sh, bits)


def isqrt(n):
    return math.isqrt(int(n))


# ----------------------------------------------------------------------------- calibration
# Calibration method for the INT8 activation scales (the same for every model):
#   PCT   None -> max |x| over the calibration set; p -> p-th percentile of |x| (second pass, histograms)
#   NORM  "bound" -> NORM output scale from the proven bound (never saturates);
#         "cal"   -> from the calibrated range, never coarser than the bound
CALIB = dict(PCT=99.99, NORM="cal")


def calibrate(prog, seqs, pct=None):
    pct = CALIB["PCT"] if pct is None else pct
    rec = {}
    fr = FloatRunner(prog, rec)
    for s in seqs:
        for t, tok in enumerate(s):
            fr.step(int(tok), t)
    if not pct:
        return rec
    # INT8 quantisation points only; R / L (INT16) and HS (fixed Q0.7) keep their maxima
    keys = {k: v for k, v in rec.items() if len(k) == 2 and k[1] in ("out", "pre", "post", "pair")}
    fr = FloatRunner(prog)
    fr.calhist = (keys, {})
    for s in seqs:
        for t, tok in enumerate(s):
            fr.step(int(tok), t)
    for k, h in fr.calhist[1].items():
        c = np.cumsum(h)
        b = int(np.searchsorted(c, pct / 100.0 * c[-1]))
        rec[k] = min(rec[k], keys[k] * (b + 1) / len(h))
    return rec


# ----------------------------------------------------------------------------- quantisation
class QProgram:
    pass


def quantize(prog, rec):
    q = QProgram()
    q.name, q.V, q.T, q.n_layer = prog.name, prog.V, prog.T, prog.n_layer
    q.ops = [dict(o) for o in prog.ops]
    rmax = rec.get(("R",), 0.0)
    q.FR = int(math.floor(math.log2(32767.0 / rmax))) if rmax > 0 else 12
    sR = 2.0 ** -q.FR
    lmax = rec[("L",)]
    q.s_logit = lmax / 32767.0
    q.lin, q.tab, q.lut, q.tscale = {}, {}, {}, {}
    sc = {"R": sR, "HS": Q07}           # current scale of each buffer
    kvs = {}
    for i, o in enumerate(q.ops):
        op = o["op"]
        if op == "HRESET":
            h0 = prog.tab[o["table"]]
            o["h0_clipped"] = bool(np.any(np.abs(h0) > 1.0))
            q.tab[o["table"]] = sat(np.round(h0 / Q07).astype(np.int64), 8)
            sc["HS"] = Q07
        elif op == "EMB16":
            for key in (o["tok"], o.get("pos")):
                if key:
                    q.tab[key] = sat(np.round(prog.tab[key] / sR).astype(np.int64), 16)
        elif op in ("EMB8", "EMBWIN"):
            tb = prog.tab[o["table"]]
            s = np.abs(tb).max() / 127.0
            q.tab[o["table"]] = sat(np.round(tb / s).astype(np.int64), 8)
            q.tscale[o["table"]] = s
            sc[o["dst"]] = s
        elif op == "NORM":
            n, sx = o["n"], sc[o["src"]]
            bound = math.sqrt(n - 1) if o["kind"] == "ln" else math.sqrt(n)
            c = n ** 3 if o["kind"] == "ln" else n
            o["E"] = max(1, int(round(c * o["eps"] / sx ** 2)))
            o["K"] = int(round(math.sqrt(n) * 2 ** NORM_P))
            if o["dst"] == "R":
                o["shift"], o["bits"] = NORM_P - q.FR, 16
                assert 2.0 ** (15 - q.FR) > bound, "residual format too narrow for a normalised vector"
            else:
                zf = int(math.floor(math.log2(127.0 / bound)))
                if CALIB["NORM"] == "cal" and rec.get((i, "out"), 0) > 0:
                    zf = max(zf, int(math.floor(math.log2(127.0 / rec[(i, "out")]))))
                o["shift"], o["bits"] = NORM_P - zf, 8
                sc[o["dst"]] = 2.0 ** -zf
        elif op == "GEMV":
            W, b = prog.lin[o["lin"]]
            s_in = np.concatenate([np.full(_buflen(prog, s, W, o), sc[s]) for s in o["srcs"]])
            assert len(s_in) == W.shape[1], (o["lin"], len(s_in), W.shape)
            Weff = W * s_in[None, :]
            amax = np.abs(Weff).max(axis=1)
            s_w = np.where(amax > 0, amax / 127.0, 1.0)
            Wq = np.clip(np.round(Weff / s_w[:, None]), -127, 127).astype(np.int64)
            bq = np.round(b / s_w).astype(np.int64)
            mode = o["mode"]
            if mode == "o8":
                pn = rec.get((i, "pair"), 0.0)
                s_out = (pn if pn > 0 else rec[(i, "out")]) / 127.0
                sc[o["dst"]] = s_out
            elif mode == "radd":
                s_out = sR
            elif mode == "lut":
                s_out = rec[(i, "pre")] / 127.0
                act = o["act"]
                s_post = Q07 if act in ("tanh", "sigmoid") else rec[(i, "post")] / 127.0
                hq = np.arange(-128, 128)
                lut = sat(np.round(ACTS[act](hq * s_out) / s_post).astype(np.int64), 8)
                q.lut[i] = lut
                sc[o["dst"]] = s_post
            elif mode == "logit":
                s_out = q.s_logit
            m = [qmul(s_w[k] / s_out) for k in range(W.shape[0])]
            q.lin[o["lin"]] = dict(W=Wq, b=bq, m0=np.array([x[0] for x in m], dtype=np.int64),
                                   sh=np.array([x[1] for x in m], dtype=np.int64))
            assert np.all(np.abs(bq) < 2 ** 31)
        elif op == "ROPE":
            cos, sin = prog.rope
            q.rope = (np.round(cos * 2 ** ROPE_F).astype(np.int64), np.round(sin * 2 ** ROPE_F).astype(np.int64))
        elif op == "KVSTORE":
            kvs[o["layer"]] = (sc[o["k"]], sc[o["v"]])
        elif op == "ATTN":
            sk, sv = kvs[o["layer"]]
            o["CEXP"], o["FEXP"] = qmul(sc[o["q"]] * sk * o["alpha"] * math.log2(math.e))
            assert o["FEXP"] > 8
            sc[o["dst"]] = sv                          # convex combination of V rows
        elif op == "MUL":
            s_out = rec[(i, "out")] / 127.0
            o["m0"], o["sh"] = qmul(sc[o["a"]] * sc[o["b"]] / s_out)
            sc[o["dst"]] = s_out
        elif op == "GRUUPD":
            assert sc[o["z"]] == Q07 and sc[o["hb"]] == Q07 and sc["HS"] == Q07
        elif op == "COPY":
            sc[o["dst"]] = sc[o["src"]]
            if o["dst"] == "HS":
                assert sc["HS"] == Q07
        elif op == "REQ":
            s_out = rec[(i, "out")] / 127.0
            o["m0"], o["sh"] = qmul(sc[o["src"]] / s_out)
            sc[o["dst"]] = s_out
        elif op == "BOW":
            q.rcp = np.array([int(round(2 ** BOW_RS / (t + 1))) for t in range(q.T)], dtype=np.int64)
        elif op == "TABLOGIT":
            q.tab[o["table"]] = sat(np.round(prog.tab[o["table"]] / q.s_logit).astype(np.int64), 16)
    return q


def _buflen(prog, name, W, o):
    """length of a GEMV source buffer (only needed for multi-source GEMVs)."""
    if len(o["srcs"]) == 1:
        return W.shape[1]
    H = len(prog.tab["h0"])                    # [input embedding, hidden-size buffer]
    return W.shape[1] - H if name == o["srcs"][0] else H


# ----------------------------------------------------------------------------- integer interpreter
def norm_int(x, o):
    """Integer LayerNorm / RMSNorm without affine part, per segment of length n."""
    n, K, E, shift, bits = o["n"], o["K"], o["E"], o["shift"], o["bits"]
    out = np.empty(len(x), dtype=np.int64)
    for s0 in range(0, len(x), n):
        seg = [int(v) for v in x[s0:s0 + n]]
        if o["kind"] == "ln":
            s = sum(seg)
            d = [n * v - s for v in seg]
        else:
            d = seg
        Q = sum(v * v for v in d) + E
        k = max(0, (NORM_QBITS - Q.bit_length()) // 2)      # normalise before the square root
        rt = math.isqrt(Q << (2 * k))
        inv = (K << k) // rt
        rnd = 1 << (shift - 1)
        out[s0:s0 + n] = [max(-(1 << (bits - 1)), min((1 << (bits - 1)) - 1, (v * inv + rnd) >> shift)) for v in d]
    return out


def rope_int(x, nh, dh, style, c, s):
    y = x.copy()
    r = 1 << (ROPE_F - 1)
    for h in range(nh):
        base = h * dh
        for i in range(dh // 2):
            if style == "inter":
                i1, i2 = base + 2 * i, base + 2 * i + 1
                a, b = int(x[i1]), int(x[i2])
                y[i1] = (a * int(c[i]) - b * int(s[i]) + r) >> ROPE_F
                y[i2] = (a * int(s[i]) + b * int(c[i]) + r) >> ROPE_F
            else:
                i1, i2 = base + i, base + i + dh // 2
                a, b = int(x[i1]), int(x[i2])
                y[i1] = (a * int(c[i]) + b * int(s[i]) + r) >> ROPE_F
                y[i2] = (-a * int(s[i]) + b * int(c[i]) + r) >> ROPE_F
    return sat(y, 8)


def attn_int(q, K, Vc, o):
    nh, nkv, dh = o["nh"], o["nkv"], o["dh"]
    rep, F, CEXP = nh // nkv, o["FEXP"], o["CEXP"]
    y = np.zeros(nh * dh, dtype=np.int64)
    for h in range(nh):
        kv = h // rep
        qh = q[h * dh:(h + 1) * dh]
        s = K[:, kv * dh:(kv + 1) * dh] @ qh                  # int64, exact
        m = s.max()
        u = (m - s) * CEXP
        n_ = u >> F
        kk = (u >> (F - 8)) & 0xFF
        e = np.where(n_ > 15, 0, EXP_LUT[kk] >> np.minimum(n_, 15))
        E = int(e.sum())
        R = (1 << ATT_RS) // E
        A = e @ Vc[:, kv * dh:(kv + 1) * dh]
        y[h * dh:(h + 1) * dh] = sat((A * R + (1 << (ATT_RS - 1))) >> ATT_RS, 8)
    return y


class IntRunner:
    def __init__(self, q):
        self.q = q
        self.reset()

    def reset(self):
        self.K, self.Vc, self.HS, self.S, self.hist = {}, {}, None, None, []

    def step(self, tok, t):
        q = self.q
        if t == 0:
            self.reset()
        self.hist.append(tok)
        buf = {}
        if self.HS is not None:
            buf["HS"] = self.HS
        for i, o in enumerate(q.ops):
            op = o["op"]
            if op == "HRESET":
                if t == 0:
                    self.HS = q.tab[o["table"]].copy()
                buf["HS"] = self.HS
            elif op == "EMB16":
                x = q.tab[o["tok"]][tok].copy()
                if o.get("pos"):
                    x = sat(x + q.tab[o["pos"]][t], 16)
                buf["R"] = x
            elif op == "EMB8":
                buf[o["dst"]] = q.tab[o["table"]][tok].copy()
            elif op == "EMBWIN":
                tb = q.tab[o["table"]]
                blank = tb.shape[0] - 1
                buf[o["dst"]] = np.concatenate([tb[self.hist[t - j]] if t - j >= 0 else tb[blank] for j in range(o["k"])])
            elif op == "NORM":
                buf[o["dst"]] = norm_int(buf[o["src"]], o)
            elif op == "GEMV":
                L = q.lin[o["lin"]]
                x = np.concatenate([buf[s] for s in o["srcs"]]).astype(np.int64)
                acc = L["W"] @ x + L["b"]
                assert np.all(np.abs(acc) < 2 ** 31)
                mode = o["mode"]
                if mode == "o8":
                    buf[o["dst"]] = requant(acc, L["m0"], L["sh"], 8)
                elif mode == "radd":
                    buf["R"] = sat(buf["R"] + requant(acc, L["m0"], L["sh"], 16), 16)
                elif mode == "lut":
                    buf[o["dst"]] = q.lut[i][requant(acc, L["m0"], L["sh"], 8) + 128]
                elif mode == "logit":
                    buf["L"] = requant(acc, L["m0"], L["sh"], 16)
            elif op == "ROPE":
                buf[o["buf"]] = rope_int(buf[o["buf"]], o["nh"], o["dh"], o["style"], q.rope[0][t], q.rope[1][t])
            elif op == "KVSTORE":
                l = o["layer"]
                self.K.setdefault(l, []).append(buf[o["k"]].copy())
                self.Vc.setdefault(l, []).append(buf[o["v"]].copy())
            elif op == "ATTN":
                l = o["layer"]
                buf[o["dst"]] = attn_int(buf[o["q"]], np.array(self.K[l]), np.array(self.Vc[l]), o)
            elif op == "MUL":
                buf[o["dst"]] = requant(buf[o["a"]] * buf[o["b"]], o["m0"], o["sh"], 8)
            elif op == "GRUUPD":
                z, hb = buf[o["z"]], buf[o["hb"]]
                self.HS = sat(self.HS + ((z * (hb - self.HS) + 64) >> 7), 8)
                buf["HS"] = self.HS
            elif op == "COPY":
                buf[o["dst"]] = buf[o["src"]].copy()
                if o["dst"] == "HS":
                    self.HS = buf["HS"]
            elif op == "REQ":
                buf[o["dst"]] = requant(buf[o["src"]], o["m0"], o["sh"], 8)
            elif op == "BOW":
                if t == 0:
                    self.S = np.zeros(len(buf["R"]), dtype=np.int64)
                self.S = self.S + buf["R"]
                avg = (self.S * int(q.rcp[t]) + (1 << (BOW_RS - 1))) >> BOW_RS
                buf["R"] = sat(buf["R"] + avg, 16)
            elif op == "TABLOGIT":
                buf["L"] = q.tab[o["table"]][tok].copy()
        return buf["L"]


def nll(logits, target):
    m = logits.max()
    return float(m + math.log(np.exp(logits - m).sum()) - logits[target])
