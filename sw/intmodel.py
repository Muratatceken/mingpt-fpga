"""
Bit-exact integer reference model ("golden model") of the FPGA accelerator.

Every operation here uses only integer add / multiply / shift / compare, floor
division and floor square root -- exactly what the RTL implements. Python ints
have arbitrary precision, so any overflow in the RTL shows up as a mismatch.

The model runs one token per call (autoregressive decode with a KV cache),
which is also how the hardware works.
"""
import math
import pickle

import numpy as np

from config import QPARAM_PATH


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return max(lo, min(hi, v))


def requant(acc, m0, sh, bits):
    """y = clamp(round(acc * m0 / 2^sh)), round-half-up via add + arithmetic shift."""
    return sat((acc * m0 + (1 << (sh - 1))) >> sh, bits)


def linear(lin, x, bits):
    """GEMV + bias + per-channel requant. x: list of ints. Returns list (padded rows included)."""
    W, b, m0, sh = lin["W"], lin["b"], lin["m0"], lin["sh"]
    accs = W @ np.asarray(x, dtype=np.int64)  # exact: |acc| << 2^63
    out = []
    for o in range(W.shape[0]):
        acc = int(accs[o])                     # INT32 accumulator in HW
        assert -(1 << 31) <= acc < (1 << 31)
        out.append(requant(acc + int(b[o]), int(m0[o]), int(sh[o]), bits))
    return out


class IntGPT:
    def __init__(self, q):
        self.q = q
        cfg = q["cfg"]
        self.C, self.H, self.L, self.V, self.T = (cfg["n_embd"], cfg["n_head"], cfg["n_layer"],
                                                  cfg["vocab"], cfg["block"])
        self.hd = self.C // self.H
        self.kc = [[[0] * self.C for _ in range(self.T)] for _ in range(self.L)]
        self.vc = [[[0] * self.C for _ in range(self.T)] for _ in range(self.L)]

    # -- LayerNorm without affine (gamma/beta folded into the next Linear) --
    def layernorm(self, x):
        q, N = self.q, len(x)
        s = sum(x)
        d = [N * xi - s for xi in x]                   # d_i = N (x_i - mean)
        Q = sum(di * di for di in d) + q["LN_E"]        # N^3 (var + eps) / s_r^2
        rt = math.isqrt(Q)                              # floor(sqrt(Q))
        inv = q["LN_K"] // rt                           # floor(sqrt(N) 2^P / rt)
        sh = q["LN_P"] - q["ZF"]
        return [sat((di * inv + (1 << (sh - 1))) >> sh, 8) for di in d]

    # -- one attention head over positions 0..t --
    def attention_head(self, l, h, qv, t):
        q, hd = self.q, self.hd
        ql = q["layers"][l]
        base = h * hd
        s = [sum(qv[base + d] * self.kc[l][j][base + d] for d in range(hd)) for j in range(t + 1)]
        m = max(s)
        F = ql["FEXP"]
        e = []
        for sj in s:
            u = (m - sj) * ql["CEXP"]                   # -log2(e^x) in Q.F, >= 0
            n = u >> F                                  # integer part
            k = (u >> (F - 8)) & 0xFF                   # top 8 fractional bits
            e.append(0 if n > 15 else int(q["EXP_LUT"][k]) >> n)
        E = sum(e)                                      # >= 32768 (max element has e = 1.0)
        R = (1 << q["ATT_RS"]) // E
        rs = q["ATT_RS"]
        y = []
        for d in range(hd):
            A = sum(e[j] * self.vc[l][j][base + d] for j in range(t + 1))
            y.append(sat((A * R + (1 << (rs - 1))) >> rs, 8))
        return y

    def step(self, tok, t, trace=None):
        """Process token `tok` at position `t`. Returns (next_token, logits, final_residual)."""
        q, C = self.q, self.C
        res = [sat(int(q["wte"][tok, i]) + int(q["wpe"][t, i]), 16) for i in range(C)]
        for l in range(self.L):
            ql = q["layers"][l]
            z = self.layernorm(res)
            qkv = linear(ql["attn"], z, 8)
            qv = qkv[:C]
            self.kc[l][t] = qkv[C:2 * C]
            self.vc[l][t] = qkv[2 * C:]
            y = []
            for h in range(self.H):
                y += self.attention_head(l, h, qv, t)
            dr = linear(ql["proj"], y, 16)
            res = [sat(a + b, 16) for a, b in zip(res, dr)]
            if trace is not None:
                trace.append(("attn", l, list(res)))
            z = self.layernorm(res)
            hq = linear(ql["fc"], z, 8)
            g = [int(ql["GELU_LUT"][v + 128]) for v in hq]
            dr = linear(ql["mproj"], g, 16)
            res = [sat(a + b, 16) for a, b in zip(res, dr)]
            if trace is not None:
                trace.append(("mlp", l, list(res)))
        z = self.layernorm(res)
        logits = linear(q["head"], z, 16)[: self.V]
        nxt = max(range(self.V), key=lambda i: (logits[i], -i))  # first index wins ties
        return nxt, logits, res

    def generate(self, prompt, n_new):
        """Greedy generation; returns (generated tokens, per-step records)."""
        steps = []
        seq = list(prompt)
        for t in range(len(prompt) + n_new - 1):
            nxt, logits, res = self.step(seq[t], t)
            steps.append(dict(t=t, tok_in=seq[t], tok_out=nxt, logits=logits, res=res))
            if t >= len(prompt) - 1:
                seq.append(nxt)
        return seq[len(prompt):], steps


def load_qparams():
    with open(QPARAM_PATH, "rb") as fh:
        return pickle.load(fh)
