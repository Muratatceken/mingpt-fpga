"""
Step 2: turn the trained float model into an integer-only model.

  1. Fold (mathematically exact, verified numerically):
       - LayerNorm gamma/beta into the following Linear      W' = W diag(g), b' = b + W beta
       - attention scale 1/sqrt(hd) into the query rows of c_attn
  2. Calibrate activation ranges on all 729 sort inputs.
  3. Quantize: weights INT8 per-output-channel, activations INT8 per-tensor,
     residual stream INT16 with a power-of-two scale, biases INT32, and
     per-channel requantization multipliers (m0, sh) so that  y = (acc*m0) >> sh.
  4. Build all integer constants (LayerNorm, exp LUT, GELU LUTs).

Output: build/qparams.pkl (consumed by intmodel.py and export.py)
"""
import math
import os
import pickle

import numpy as np
import torch

from common import QPARAM_PATH, load_model, all_sort_inputs

# ---------------------------------------------------------------------------
# Global fixed-point design constants (mirrored in rtl/gpt_pkg.sv by export.py)
# ---------------------------------------------------------------------------
P_LANES = 8       # parallel MAC lanes in the GEMV engine
ZF = 4            # LayerNorm output: INT8 with scale 2^-ZF (Q3.4). |z| <= sqrt(N-1) < 8, no clipping.
M0_BITS = 16      # requant multiplier m0 in [2^15, 2^16)
EXP_LUT_BITS = 8  # 2^-f LUT addressed by the top 8 fractional bits
EXP_ONE = 1 << 15 # exp values are UQ1.15: exp(0) = 32768
LN_P = 36         # LayerNorm reciprocal precision: inv = round(sqrt(N) 2^LN_P) // isqrt(Q)
ATT_RS = 32       # softmax reciprocal precision: R = 2^ATT_RS // sum(e)


def gelu(x):
    """minGPT's NewGELU (tanh approximation), numpy version."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# 1. Folding
# ---------------------------------------------------------------------------
def fold(model):
    """Return float numpy params with LN affine + attention scale folded in."""
    sd = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    cfg = dict(n_layer=len(model.transformer.h),
               n_head=model.transformer.h[0].attn.n_head,
               n_embd=model.transformer.wte.weight.shape[1],
               vocab=model.transformer.wte.weight.shape[0],
               block=model.block_size)
    C, H = cfg["n_embd"], cfg["n_head"]
    hd = C // H
    f = {"wte": sd["transformer.wte.weight"], "wpe": sd["transformer.wpe.weight"], "layers": []}
    for l in range(cfg["n_layer"]):
        p = f"transformer.h.{l}."
        g1, b1 = sd[p + "ln_1.weight"], sd[p + "ln_1.bias"]
        g2, b2 = sd[p + "ln_2.weight"], sd[p + "ln_2.bias"]
        Wa, ba = sd[p + "attn.c_attn.weight"], sd[p + "attn.c_attn.bias"]
        Wa_f = Wa * g1[None, :]
        ba_f = ba + Wa @ b1
        Wa_f[:C] /= math.sqrt(hd)
        ba_f[:C] /= math.sqrt(hd)
        Wf, bf = sd[p + "mlp.c_fc.weight"], sd[p + "mlp.c_fc.bias"]
        f["layers"].append(dict(
            attn=(Wa_f, ba_f),
            proj=(sd[p + "attn.c_proj.weight"], sd[p + "attn.c_proj.bias"]),
            fc=(Wf * g2[None, :], bf + Wf @ b2),
            mproj=(sd[p + "mlp.c_proj.weight"], sd[p + "mlp.c_proj.bias"]),
        ))
    gf, bfin = sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"]
    Wh = sd["lm_head.weight"]
    f["head"] = (Wh * gf[None, :], Wh @ bfin)
    return cfg, f


def layernorm_plain(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def folded_forward(cfg, f, idx, trace=None):
    """Float64 forward of the folded model on a full sequence idx (T,). Returns logits (T, V).
    If trace is a dict, records activation tensors for calibration."""
    C, H, T = cfg["n_embd"], cfg["n_head"], len(idx)
    hd = C // H
    x = f["wte"][idx] + f["wpe"][:T]
    rec = (lambda k, v: trace.setdefault(k, []).append(v)) if trace is not None else (lambda k, v: None)
    rec("res", x)
    mask = np.tril(np.ones((T, T), dtype=bool))
    for l, L in enumerate(f["layers"]):
        z = layernorm_plain(x)
        qkv = z @ L["attn"][0].T + L["attn"][1]
        q, k, v = qkv[:, :C], qkv[:, C:2 * C], qkv[:, 2 * C:]
        rec(f"q{l}", q); rec(f"k{l}", k); rec(f"v{l}", v)
        y = np.zeros((T, C))
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            att = q[:, sl] @ k[:, sl].T          # 1/sqrt(hd) already folded into q
            att = np.where(mask, att, -np.inf)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y[:, sl] = att @ v[:, sl]
        x = x + y @ L["proj"][0].T + L["proj"][1]
        rec("res", x)
        z = layernorm_plain(x)
        hpre = z @ L["fc"][0].T + L["fc"][1]
        rec(f"h{l}", hpre)
        g = gelu(hpre)
        rec(f"g{l}", g)
        x = x + g @ L["mproj"][0].T + L["mproj"][1]
        rec("res", x)
    z = layernorm_plain(x)
    logits = z @ f["head"][0].T + f["head"][1]
    rec("logit", logits)
    return logits


# ---------------------------------------------------------------------------
# 3. Quantization helpers
# ---------------------------------------------------------------------------
def quant_multiplier(m):
    """Represent a positive real m as m0 * 2^-sh with m0 in [2^15, 2^16)."""
    assert m > 0
    sh = 0
    while m * (1 << sh) < (1 << (M0_BITS - 1)):
        sh += 1
    m0 = int(round(m * (1 << sh)))
    if m0 == (1 << M0_BITS):
        m0 >>= 1
        sh -= 1
    assert (1 << (M0_BITS - 1)) <= m0 < (1 << M0_BITS) and 1 <= sh <= 63, (m, m0, sh)
    return m0, sh


def quant_linear(W, b, s_in, s_out, n_out_pad=None):
    """INT8 per-output-channel weights, INT32 bias at scale s_in*s_w, requant to s_out."""
    n_out, n_in = W.shape
    amax = np.abs(W).max(axis=1)
    s_w = np.where(amax > 0, amax / 127.0, 1.0)
    Wq = np.clip(np.round(W / s_w[:, None]), -127, 127).astype(np.int64)
    bq = np.round(b / (s_in * s_w)).astype(np.int64)
    assert np.all(np.abs(bq) < 2 ** 31)
    s_out = np.broadcast_to(np.asarray(s_out, dtype=np.float64), (n_out,))
    m0, sh = zip(*[quant_multiplier(s_in * s_w[o] / s_out[o]) for o in range(n_out)])
    m0, sh = np.array(m0, dtype=np.int64), np.array(sh, dtype=np.int64)
    if n_out_pad is not None and n_out_pad > n_out:  # padded rows: zero weights / bias
        pad = n_out_pad - n_out
        Wq = np.vstack([Wq, np.zeros((pad, n_in), dtype=np.int64)])
        bq = np.concatenate([bq, np.zeros(pad, dtype=np.int64)])
        m0 = np.concatenate([m0, np.full(pad, 1 << (M0_BITS - 1), dtype=np.int64)])
        sh = np.concatenate([sh, np.full(pad, M0_BITS, dtype=np.int64)])
    return dict(W=Wq, b=bq, m0=m0, sh=sh, s_w=s_w, s_in=s_in, s_out=s_out[:n_out].copy())


def main():
    model = load_model()
    cfg, f = fold(model)
    C, H, L, V, T = cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["vocab"], cfg["block"]
    hd = C // H
    assert hd & (hd - 1) == 0, "power-of-two head dim keeps the attention scale fold exact"
    print("config:", cfg)

    # -- check folding is exact against minGPT itself --
    inputs = all_sort_inputs()
    seqs = []
    with torch.no_grad():
        full = model.generate(torch.tensor(inputs), len(inputs[0]), do_sample=False)[:, :T]
        ref_logits = model(full)[0].double().numpy()
    for i, s in enumerate(full.numpy()):
        lg = folded_forward(cfg, f, s)
        assert np.allclose(lg, ref_logits[i], atol=1e-4), "folding mismatch"
        seqs.append(s)
    print("folding check: folded float64 model == minGPT (atol 1e-4) on all 729 sequences")

    # -- calibration (max-abs over every activation of every sequence) --
    trace = {}
    for s in seqs:
        folded_forward(cfg, f, s, trace)
    amax = {k: max(np.abs(a).max() for a in v) for k, v in trace.items()}

    # residual stream: INT16, scale 2^-FR (power of two -> adds are free)
    FR = int(math.floor(math.log2(32767.0 / amax["res"])))
    s_r = 2.0 ** -FR
    s_z = 2.0 ** -ZF
    print(f"residual max |x| = {amax['res']:.3f}  -> INT16 Q{15 - FR}.{FR}")

    q = dict(cfg=cfg, FR=FR, ZF=ZF, P=P_LANES, LN_P=LN_P, ATT_RS=ATT_RS, layers=[])
    q["wte"] = np.clip(np.round(f["wte"] / s_r), -32768, 32767).astype(np.int64)
    q["wpe"] = np.clip(np.round(f["wpe"] / s_r), -32768, 32767).astype(np.int64)

    # LayerNorm constants: eps term and reciprocal numerator (see docs/01_teori.md)
    q["LN_E"] = int(round(1e-5 * C ** 3 / s_r ** 2))
    q["LN_K"] = int(round(math.sqrt(C) * 2 ** LN_P))

    # exp LUT: 2^(-k/256) in UQ1.15
    q["EXP_LUT"] = np.array([int(round(EXP_ONE * 2.0 ** (-k / 2 ** EXP_LUT_BITS)))
                             for k in range(2 ** EXP_LUT_BITS)], dtype=np.int64)

    for l in range(L):
        Lf = f["layers"][l]
        s_q = amax[f"q{l}"] / 127.0
        s_k = amax[f"k{l}"] / 127.0
        s_v = amax[f"v{l}"] / 127.0
        s_h = amax[f"h{l}"] / 127.0
        s_g = amax[f"g{l}"] / 127.0
        s_qkv = np.concatenate([np.full(C, s_q), np.full(C, s_k), np.full(C, s_v)])
        ql = dict(s_q=s_q, s_k=s_k, s_v=s_v, s_h=s_h, s_g=s_g)
        ql["attn"] = quant_linear(*Lf["attn"], s_in=s_z, s_out=s_qkv)
        # attention output is a convex combination of V rows -> keeps V's scale (no clipping)
        ql["proj"] = quant_linear(*Lf["proj"], s_in=s_v, s_out=s_r)
        ql["fc"] = quant_linear(*Lf["fc"], s_in=s_z, s_out=s_h)
        ql["mproj"] = quant_linear(*Lf["mproj"], s_in=s_g, s_out=s_r)
        # softmax: exp(s * s_q*s_k) = 2^(s * s_q*s_k*log2 e) = 2^(s * CEXP * 2^-FEXP)
        ql["CEXP"], ql["FEXP"] = quant_multiplier(s_q * s_k * math.log2(math.e))
        assert ql["FEXP"] > EXP_LUT_BITS
        # GELU LUT: int8 (scale s_h) -> int8 (scale s_g)
        hq = np.arange(-128, 128)
        ql["GELU_LUT"] = np.clip(np.round(gelu(hq * s_h) / s_g), -128, 127).astype(np.int64)
        q["layers"].append(ql)

    s_logit = amax["logit"] / 32767.0
    n_pad = -(-V // P_LANES) * P_LANES
    q["head"] = quant_linear(*f["head"], s_in=s_z, s_out=s_logit, n_out_pad=n_pad)
    q["s_logit"] = s_logit

    os.makedirs(os.path.dirname(QPARAM_PATH), exist_ok=True)
    with open(QPARAM_PATH, "wb") as fh:
        pickle.dump(q, fh)
    print(f"saved {QPARAM_PATH}")
    for l, ql in enumerate(q["layers"]):
        print(f"  layer {l}: s_q={ql['s_q']:.4g} s_k={ql['s_k']:.4g} s_v={ql['s_v']:.4g} "
              f"s_h={ql['s_h']:.4g} s_g={ql['s_g']:.4g} CEXP={ql['CEXP']} FEXP={ql['FEXP']}")


if __name__ == "__main__":
    main()
