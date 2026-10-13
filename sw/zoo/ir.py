"""
A tiny instruction set that expresses every model of the zoo as a per-token program.

    compile_<model>(torch model) -> Program (float weights, op list)
    run_float(prog, ...)         -> float reference; must match PyTorch (validates the compiler)
    quantize(prog, calib)        -> QProgram (integer tables + constants)          [quant.py]
    run_int(qprog, ...)          -> bit-exact integer spec of the hardware          [quant.py]

Buffers (per token step):  R  residual (C)        A..H  activation vectors
State (across steps):      K[l], V[l] KV cache    HS recurrent state    S BoW sums    TOK token history
Ops (dict with "op" key):
  EMB16  tok, pos?           R = tok_table[tok] (+ pos_table[pos])
  EMB8   table, dst          dst = table[tok]
  EMBWIN table, k, dst       dst = concat(table[tok_{t-j}] for j < k), BLANK row before t=0
  NORM   kind(ln|rms), src, dst, n, eps     per segment of length n, no affine (folded)
  GEMV   lin, srcs, dst, mode(o8|radd|lut|logit), act
  ROPE   buf, nh, dh, style(inter|half)
  KVSTORE layer, k, v
  ATTN   layer, q, dst, nh, nkv, dh, alpha  (scores = alpha * q.k)
  MUL    a, b, dst
  GRUUPD z, hb               HS = (1-z)*HS + z*hb
  COPY   src, dst
  REQ    src, dst            re-quantisation point (identity in float)
  BOW                        S += R ; R += S/(t+1)
  TABLOGIT table             logits = table[tok]
  HRESET table               at t=0: HS = table
"""
import math

import numpy as np
import torch

from models import Bigram, MLP, RNN, BoW, GPT2, MicroGPT, Llama, NanoChat

ACTS = {
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3))),
    "relu": lambda x: np.maximum(x, 0.0),
    "relu2": lambda x: np.maximum(x, 0.0) ** 2,
    "silu": lambda x: x / (1.0 + np.exp(-x)),
}


HIST_BINS = 4096


def npw(t):
    return t.detach().double().cpu().numpy()


class Program:
    def __init__(self, name, V, T):
        self.name, self.V, self.T = name, V, T
        self.ops, self.lin, self.tab = [], {}, {}
        self.n_layer = 0
        self.dims = {}

    def linear(self, key, W, b=None):
        self.lin[key] = (np.asarray(W, dtype=np.float64),
                         np.zeros(W.shape[0]) if b is None else np.asarray(b, dtype=np.float64))
        return key

    def add(self, op, **kw):
        kw["op"] = op
        self.ops.append(kw)


# ----------------------------------------------------------------------------- compilers
def fold_affine(W, b, gamma, beta=None):
    """W (out,in) applied after an affine norm  y = W (g*z + beta) + b."""
    W2 = W * gamma[None, :]
    b2 = (np.zeros(W.shape[0]) if b is None else b) + (W @ beta if beta is not None else 0.0)
    return W2, b2


def compile_bigram(m):
    p = Program("bigram", *m.logits.shape[:1], 16)
    p.V = m.logits.shape[0]
    p.tab["logit"] = npw(m.logits)
    p.add("TABLOGIT", table="logit")
    return p


def compile_mlp(m):
    p = Program("mlp", m.V, m.T)
    p.tab["wte"] = npw(m.wte.weight)
    n_e = m.wte.weight.shape[1]
    p.add("EMBWIN", table="wte", k=m.T, dst="A", n=n_e)
    p.add("GEMV", lin=p.linear("fc1", npw(m.fc1.weight), npw(m.fc1.bias)), srcs=["A"], dst="B", mode="lut", act="tanh")
    p.add("GEMV", lin=p.linear("fc2", npw(m.fc2.weight), npw(m.fc2.bias)), srcs=["B"], dst="L", mode="logit")
    return p


def compile_rnn(m):
    name = m.cell
    p = Program(name, m.lm_head.weight.shape[0], 16)
    p.tab["wte"] = npw(m.wte.weight)
    p.tab["h0"] = npw(m.start)[0]
    p.add("HRESET", table="h0")
    p.add("EMB8", table="wte", dst="A")
    if name == "rnn":
        p.add("GEMV", lin=p.linear("xh", npw(m.xh_to_h.weight), npw(m.xh_to_h.bias)),
              srcs=["A", "HS"], dst="B", mode="lut", act="tanh")
        p.add("COPY", src="B", dst="HS")
    else:
        p.add("GEMV", lin=p.linear("r", npw(m.xh_to_r.weight), npw(m.xh_to_r.bias)),
              srcs=["A", "HS"], dst="B", mode="lut", act="sigmoid")
        p.add("GEMV", lin=p.linear("z", npw(m.xh_to_z.weight), npw(m.xh_to_z.bias)),
              srcs=["A", "HS"], dst="C", mode="lut", act="sigmoid")
        p.add("MUL", a="B", b="HS", dst="D")
        p.add("GEMV", lin=p.linear("hbar", npw(m.xh_to_hbar.weight), npw(m.xh_to_hbar.bias)),
              srcs=["A", "D"], dst="E", mode="lut", act="tanh")
        p.add("GRUUPD", z="C", hb="E")
    p.add("GEMV", lin=p.linear("head", npw(m.lm_head.weight), npw(m.lm_head.bias)), srcs=["HS"], dst="L", mode="logit")
    return p


def compile_bow(m):
    p = Program("bow", m.lm_head.weight.shape[0], m.wpe.weight.shape[0])
    p.tab["wte"], p.tab["wpe"] = npw(m.wte.weight), npw(m.wpe.weight)
    p.add("EMB16", tok="wte", pos="wpe")
    p.add("BOW")
    p.add("REQ", src="R", dst="A")
    p.add("GEMV", lin=p.linear("fc", npw(m.c_fc.weight), npw(m.c_fc.bias)), srcs=["A"], dst="B", mode="lut", act="tanh")
    p.add("GEMV", lin=p.linear("proj", npw(m.c_proj.weight), npw(m.c_proj.bias)), srcs=["B"], dst="R", mode="radd")
    p.add("REQ", src="R", dst="A")
    p.add("GEMV", lin=p.linear("head", npw(m.lm_head.weight), npw(m.lm_head.bias)), srcs=["A"], dst="L", mode="logit")
    return p


def _attn_block(p, l, a, wq, bq, wk, bk, wv, bv, nh, nkv, dh, rope=None, qknorm=False, alpha=1.0, eps=1e-5):
    """q/k/v projections (+RoPE, +QK-norm), KV store and attention into buffer B."""
    p.add("GEMV", lin=p.linear(f"l{l}.q", wq, bq), srcs=[a], dst="Q", mode="o8")
    p.add("GEMV", lin=p.linear(f"l{l}.k", wk, bk), srcs=[a], dst="K", mode="o8")
    p.add("GEMV", lin=p.linear(f"l{l}.v", wv, bv), srcs=[a], dst="Vb", mode="o8")
    if rope:
        p.add("ROPE", buf="Q", nh=nh, dh=dh, style=rope)
        p.add("ROPE", buf="K", nh=nkv, dh=dh, style=rope)
    if qknorm:
        p.add("NORM", kind="rms", src="Q", dst="Q", n=dh, eps=eps)
        p.add("NORM", kind="rms", src="K", dst="K", n=dh, eps=eps)
    p.add("KVSTORE", layer=l, k="K", v="Vb")
    p.add("ATTN", layer=l, q="Q", dst="B", nh=nh, nkv=nkv, dh=dh, alpha=alpha)


def compile_gpt2(m):
    C = m.wte.weight.shape[1]
    nh = m.n_head
    dh = C // nh
    p = Program("gpt2", m.lm_head.weight.shape[0], m.wpe.weight.shape[0])
    p.n_layer = len(m.h)
    p.tab["wte"], p.tab["wpe"] = npw(m.wte.weight), npw(m.wpe.weight)
    p.add("EMB16", tok="wte", pos="wpe")
    for l, b in enumerate(m.h):
        g1, b1 = npw(b["ln_1"].weight), npw(b["ln_1"].bias)
        W, bb = npw(b["c_attn"].weight), npw(b["c_attn"].bias)
        Wq, bq = fold_affine(W[:C], bb[:C], g1, b1)
        Wk, bk = fold_affine(W[C:2 * C], bb[C:2 * C], g1, b1)
        Wv, bv = fold_affine(W[2 * C:], bb[2 * C:], g1, b1)
        Wq, bq = Wq / math.sqrt(dh), bq / math.sqrt(dh)
        p.add("NORM", kind="ln", src="R", dst="A", n=C, eps=1e-5)
        _attn_block(p, l, "A", Wq, bq, Wk, bk, Wv, bv, nh, nh, dh)
        p.add("GEMV", lin=p.linear(f"l{l}.proj", npw(b["c_proj"].weight), npw(b["c_proj"].bias)),
              srcs=["B"], dst="R", mode="radd")
        Wf, bf = fold_affine(npw(b["c_fc"].weight), npw(b["c_fc"].bias), npw(b["ln_2"].weight), npw(b["ln_2"].bias))
        p.add("NORM", kind="ln", src="R", dst="A", n=C, eps=1e-5)
        p.add("GEMV", lin=p.linear(f"l{l}.fc", Wf, bf), srcs=["A"], dst="F", mode="lut", act="gelu")
        p.add("GEMV", lin=p.linear(f"l{l}.mproj", npw(b["mlp_proj"].weight), npw(b["mlp_proj"].bias)),
              srcs=["F"], dst="R", mode="radd")
    Wh, bh = fold_affine(npw(m.lm_head.weight), None, npw(m.ln_f.weight), npw(m.ln_f.bias))
    p.add("NORM", kind="ln", src="R", dst="A", n=C, eps=1e-5)
    p.add("GEMV", lin=p.linear("head", Wh, bh), srcs=["A"], dst="L", mode="logit")
    return p


def compile_microgpt(m):
    C = m.wte.weight.shape[1]
    nh = m.n_head
    dh = C // nh
    p = Program("microgpt", m.lm_head.weight.shape[0], m.wpe.weight.shape[0])
    p.n_layer = len(m.h)
    p.tab["wte"], p.tab["wpe"] = npw(m.wte.weight), npw(m.wpe.weight)
    p.add("EMB16", tok="wte", pos="wpe")
    p.add("NORM", kind="rms", src="R", dst="R", n=C, eps=1e-5)
    for l, b in enumerate(m.h):
        p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=1e-5)
        _attn_block(p, l, "A", npw(b["wq"].weight) / math.sqrt(dh), None, npw(b["wk"].weight), None,
                    npw(b["wv"].weight), None, nh, nh, dh)
        p.add("GEMV", lin=p.linear(f"l{l}.wo", npw(b["wo"].weight)), srcs=["B"], dst="R", mode="radd")
        p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=1e-5)
        p.add("GEMV", lin=p.linear(f"l{l}.fc1", npw(b["fc1"].weight)), srcs=["A"], dst="F", mode="lut", act="relu")
        p.add("GEMV", lin=p.linear(f"l{l}.fc2", npw(b["fc2"].weight)), srcs=["F"], dst="R", mode="radd")
    p.add("REQ", src="R", dst="A")
    p.add("GEMV", lin=p.linear("head", npw(m.lm_head.weight)), srcs=["A"], dst="L", mode="logit")
    return p


def compile_llama(m):
    C = m.tok_embeddings.weight.shape[1]
    nh, nkv, dh = m.n_head, m.n_kv, m.dh
    p = Program("llama", m.tok_embeddings.weight.shape[0], m.cos.shape[0])
    p.n_layer = len(m.layers)
    p.tab["wte"] = npw(m.tok_embeddings.weight)
    p.add("EMB16", tok="wte")
    for l, b in enumerate(m.layers):
        g = npw(b["attention_norm"].weight)
        p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=1e-5)
        _attn_block(p, l, "A", npw(b["wq"].weight) * g[None, :] / math.sqrt(dh), None,
                    npw(b["wk"].weight) * g[None, :], None, npw(b["wv"].weight) * g[None, :], None,
                    nh, nkv, dh, rope="inter")
        p.add("GEMV", lin=p.linear(f"l{l}.wo", npw(b["wo"].weight)), srcs=["B"], dst="R", mode="radd")
        g2 = npw(b["ffn_norm"].weight)
        p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=1e-5)
        p.add("GEMV", lin=p.linear(f"l{l}.w1", npw(b["w1"].weight) * g2[None, :]), srcs=["A"], dst="F", mode="lut", act="silu")
        p.add("GEMV", lin=p.linear(f"l{l}.w3", npw(b["w3"].weight) * g2[None, :]), srcs=["A"], dst="G", mode="o8")
        p.add("MUL", a="F", b="G", dst="F")
        p.add("GEMV", lin=p.linear(f"l{l}.w2", npw(b["w2"].weight)), srcs=["F"], dst="R", mode="radd")
    gn = npw(m.norm.weight)
    p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=1e-5)
    p.add("GEMV", lin=p.linear("head", npw(m.output.weight) * gn[None, :]), srcs=["A"], dst="L", mode="logit")
    return p


def compile_nanochat(m):
    C = m.wte.weight.shape[1]
    nh, nkv, dh, e = m.n_head, m.n_kv, m.dh, m.eps
    p = Program("nanochat", m.lm_head.weight.shape[0], m.cos.shape[0])
    p.n_layer = len(m.h)
    p.tab["wte"] = npw(m.wte.weight)
    p.add("EMB16", tok="wte")
    p.add("NORM", kind="rms", src="R", dst="R", n=C, eps=e)
    for l, b in enumerate(m.h):
        p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=e)
        _attn_block(p, l, "A", npw(b["c_q"].weight), None, npw(b["c_k"].weight), None, npw(b["c_v"].weight), None,
                    nh, nkv, dh, rope="half", qknorm=True, alpha=1.2 * 1.2 / math.sqrt(dh), eps=e)
        p.add("GEMV", lin=p.linear(f"l{l}.proj", npw(b["c_proj"].weight)), srcs=["B"], dst="R", mode="radd")
        p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=e)
        p.add("GEMV", lin=p.linear(f"l{l}.fc", npw(b["c_fc"].weight)), srcs=["A"], dst="F", mode="lut", act="relu2")
        p.add("GEMV", lin=p.linear(f"l{l}.mproj", npw(b["mlp_proj"].weight)), srcs=["F"], dst="R", mode="radd")
    p.add("NORM", kind="rms", src="R", dst="A", n=C, eps=e)
    p.add("GEMV", lin=p.linear("head", npw(m.lm_head.weight)), srcs=["A"], dst="L", mode="logit")
    return p


def compile_model(name, m):
    if isinstance(m, Bigram):
        return compile_bigram(m)
    if isinstance(m, MLP):
        return compile_mlp(m)
    if isinstance(m, RNN):
        return compile_rnn(m)
    if isinstance(m, BoW):
        return compile_bow(m)
    if isinstance(m, GPT2):
        return compile_gpt2(m)
    if isinstance(m, MicroGPT):
        return compile_microgpt(m)
    if isinstance(m, Llama):
        return compile_llama(m)
    if isinstance(m, NanoChat):
        return compile_nanochat(m)
    raise ValueError(name)


# ----------------------------------------------------------------------------- float interpreter
def rope_tables(dh, T, theta=10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dh, 2)[: dh // 2] / dh))
    ang = np.outer(np.arange(T), freqs)
    return np.cos(ang), np.sin(ang)


def rope_apply(x, nh, dh, style, c, s):
    """Rotate a flat vector (nh*dh) in place-semantics; returns new vector. c, s: (dh/2,)"""
    y = x.copy()
    for h in range(nh):
        base = h * dh
        for i in range(dh // 2):
            if style == "inter":
                i1, i2 = base + 2 * i, base + 2 * i + 1
                a, b = x[i1], x[i2]
                y[i1], y[i2] = a * c[i] - b * s[i], a * s[i] + b * c[i]
            else:
                i1, i2 = base + i, base + i + dh // 2
                a, b = x[i1], x[i2]
                y[i1], y[i2] = a * c[i] + b * s[i], -a * s[i] + b * c[i]
    return y


class FloatRunner:
    """Executes a Program token by token in float64. `rec` collects calibration ranges."""

    def __init__(self, prog, rec=None):
        self.p = prog
        self.rec = rec
        self.reset()

    def reset(self):
        self.K, self.Vc = {}, {}
        self.HS = None
        self.S = None
        self.hist = []

    calhist = None                   # (ranges, histograms): second calibration pass for percentiles

    def note(self, key, val):
        if self.rec is not None:
            v = float(np.max(np.abs(val))) if np.size(val) else 0.0
            self.rec[key] = max(self.rec.get(key, 0.0), v)
        if self.calhist is not None and key in self.calhist[0] and self.calhist[0][key] > 0:
            h = self.calhist[1].setdefault(key, np.zeros(HIST_BINS))
            h += np.histogram(np.abs(val), bins=HIST_BINS, range=(0.0, self.calhist[0][key]))[0]

    def step(self, tok, t):
        p = self.p
        if t == 0:
            self.reset()
        self.hist.append(tok)
        buf = {}
        if self.HS is not None:
            buf["HS"] = self.HS
        for i, o in enumerate(p.ops):
            op = o["op"]
            if op == "HRESET":
                if t == 0:
                    self.HS = p.tab[o["table"]].copy()
                buf["HS"] = self.HS
                self.note(("HS",), self.HS)
            elif op == "EMB16":
                x = p.tab[o["tok"]][tok].copy()
                if o.get("pos"):
                    x = x + p.tab[o["pos"]][t]
                buf["R"] = x
                self.note(("R",), x)
            elif op == "EMB8":
                buf[o["dst"]] = p.tab[o["table"]][tok].copy()
            elif op == "EMBWIN":
                tbl, k = p.tab[o["table"]], o["k"]
                blank = tbl.shape[0] - 1
                rows = [tbl[self.hist[t - j]] if t - j >= 0 else tbl[blank] for j in range(k)]
                buf[o["dst"]] = np.concatenate(rows)
            elif op == "NORM":
                x, n = buf[o["src"]], o["n"]
                out = np.empty_like(x)
                for s0 in range(0, len(x), n):
                    seg = x[s0:s0 + n]
                    if o["kind"] == "ln":
                        seg = seg - seg.mean()
                        out[s0:s0 + n] = seg / math.sqrt((seg ** 2).mean() + o["eps"])
                    else:
                        out[s0:s0 + n] = seg / math.sqrt((seg ** 2).mean() + o["eps"])
                buf[o["dst"]] = out
                if o["dst"] == "R":
                    self.note(("R",), out)
                else:
                    self.note((i, "out"), out)
            elif op == "GEMV":
                W, b = p.lin[o["lin"]]
                x = np.concatenate([buf[s] for s in o["srcs"]])
                y = W @ x + b
                mode = o["mode"]
                if mode == "o8":
                    buf[o["dst"]] = y
                    self.note((i, "out"), y)
                    nxt = p.ops[i + 3] if i + 3 < len(p.ops) else None
                    self.note((i, "pair"), pair_norms(y, o, p, i))
                elif mode == "radd":
                    buf["R"] = buf["R"] + y
                    self.note(("R",), buf["R"])
                elif mode == "lut":
                    self.note((i, "pre"), y)
                    a = ACTS[o["act"]](y)
                    self.note((i, "post"), a)
                    buf[o["dst"]] = a
                elif mode == "logit":
                    buf["L"] = y
                    self.note(("L",), y)
            elif op == "ROPE":
                c, s = p.rope[0][t], p.rope[1][t]
                buf[o["buf"]] = rope_apply(buf[o["buf"]], o["nh"], o["dh"], o["style"], c, s)
            elif op == "KVSTORE":
                l = o["layer"]
                self.K.setdefault(l, []).append(buf[o["k"]].copy())
                self.Vc.setdefault(l, []).append(buf[o["v"]].copy())
            elif op == "ATTN":
                l, nh, nkv, dh = o["layer"], o["nh"], o["nkv"], o["dh"]
                q = buf[o["q"]]
                K, Vv = np.array(self.K[l]), np.array(self.Vc[l])
                y = np.zeros(nh * dh)
                rep = nh // nkv
                for h in range(nh):
                    kv = h // rep
                    sc = o["alpha"] * (K[:, kv * dh:(kv + 1) * dh] @ q[h * dh:(h + 1) * dh])
                    w = np.exp(sc - sc.max())
                    w /= w.sum()
                    y[h * dh:(h + 1) * dh] = w @ Vv[:, kv * dh:(kv + 1) * dh]
                buf[o["dst"]] = y
            elif op == "MUL":
                y = buf[o["a"]] * buf[o["b"]]
                buf[o["dst"]] = y
                self.note((i, "out"), y)
            elif op == "GRUUPD":
                z = buf[o["z"]]
                self.HS = (1 - z) * self.HS + z * buf[o["hb"]]
                buf["HS"] = self.HS
                self.note(("HS",), self.HS)
            elif op == "COPY":
                buf[o["dst"]] = buf[o["src"]].copy()
                if o["dst"] == "HS":
                    self.HS = buf["HS"]
                    self.note(("HS",), self.HS)
            elif op == "REQ":
                buf[o["dst"]] = buf[o["src"]].copy()
                self.note((i, "out"), buf[o["dst"]])
            elif op == "BOW":
                if t == 0:
                    self.S = np.zeros_like(buf["R"])
                self.S = self.S + buf["R"]
                buf["R"] = buf["R"] + self.S / (t + 1)
                self.note(("R",), buf["R"])
            elif op == "TABLOGIT":
                buf["L"] = p.tab[o["table"]][tok].copy()
                self.note(("L",), buf["L"])
            else:
                raise ValueError(op)
        return buf["L"]


def pair_norms(y, o, p, i):
    """Max norm of RoPE pairs if the destination of this GEMV is rotated later (pair-norm calibration)."""
    for o2 in p.ops[i + 1:]:
        if o2["op"] == "ROPE" and o2["buf"] == o["dst"]:
            dh, style = o2["dh"], o2["style"]
            out = []
            for base in range(0, len(y), dh):
                for j in range(dh // 2):
                    a, b = (y[base + 2 * j], y[base + 2 * j + 1]) if style == "inter" else (y[base + j], y[base + j + dh // 2])
                    out.append(math.hypot(a, b))
            return np.array(out)
        if o2["op"] == "KVSTORE":
            break
    return np.zeros(0)


def attach_rope(prog):
    for o in prog.ops:
        if o["op"] == "ROPE":
            prog.rope = rope_tables(o["dh"], prog.T)
            return prog
    prog.rope = None
    return prog


def float_logits(prog, seq):
    """Logits for every position of a token sequence (teacher forcing)."""
    fr = FloatRunner(prog)
    return np.array([fr.step(int(tok), t) for t, tok in enumerate(seq)])
