"""Decision models (WP3.4-3.5): budget-matched GRU and a small RoPE transformer with banded attention.

DecisionLlama = sw/zoo Llama (RMSNorm, interleaved RoPE, GQA, SwiGLU, tied embeddings) with
  * periodic RoPE: theta_k = 2*pi*m_k/P, P = 256, m = (32, 8, 2, 1) -> a 256-row table indexed by pos mod 256
    (attention scores depend only on relative position, so the stream can grow without bound)
  * banded causal attention: key j visible from query i iff 0 <= i-j < W (W = 64 per layer); with 2 layers the
    decision at position i depends only on tokens i-126 .. i (effective context 127 tokens).
The IR compiler sees cos/sin with 256 rows, so the engine is built with T = 256 (contract.model.engine_T).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zoo"))
from models import RNN, Llama, merge_heads, n_params, rope_interleaved, source_init, split_heads  # noqa: E402

from contract import MODEL, V_SIZE  # noqa: E402

P_ROPE = MODEL["rope"]["period"]
M_ROPE = MODEL["rope"]["m"]
W_BAND = MODEL["band_window_per_layer"]


def periodic_rope(d_head, P=P_ROPE, m=M_ROPE):
    assert len(m) == d_head // 2
    ang = torch.outer(torch.arange(P).float(), 2 * math.pi * torch.tensor(m, dtype=torch.float) / P)
    return torch.cos(ang), torch.sin(ang)


def band_attention(q, k, v, n_rep, W):
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    T, d = q.size(2), q.size(3)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    i = torch.arange(T, device=q.device)
    rel = i[:, None] - i[None, :]
    mask = (rel >= 0) & (rel < W) if W else (rel >= 0)
    att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1) @ v


class DecisionLlama(Llama):
    def __init__(self, V=V_SIZE, T=P_ROPE, n_layer=2, n_embd=32, n_head=4, n_kv_head=2, hidden=168, W=W_BAND):
        super().__init__(V, T, n_layer, n_embd, n_head, n_kv_head, hidden)
        cos, sin = periodic_rope(self.dh)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.W = W

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device) % self.cos.shape[0]
        c, s = self.cos[pos], self.sin[pos]
        x = self.tok_embeddings(idx)
        for b in self.layers:
            a = b["attention_norm"](x)
            q = rope_interleaved(split_heads(b["wq"](a), self.n_head), c, s)
            k = rope_interleaved(split_heads(b["wk"](a), self.n_kv), c, s)
            v = split_heads(b["wv"](a), self.n_kv)
            x = x + b["wo"](merge_heads(band_attention(q, k, v, self.n_head // self.n_kv, self.W)))
            f = b["ffn_norm"](x)
            x = x + b["w2"](F.silu(b["w1"](f)) * b["w3"](f))
        return self.output(self.norm(x))


def make_gru(V=V_SIZE, n_embd=48, n_hidden=82):
    return RNN(V, P_ROPE, n_embd, n_hidden, cell="gru")


SPECS = {
    "lm": dict(cls="DecisionLlama", kwargs=dict(V=V_SIZE, T=P_ROPE, n_layer=2, n_embd=32, n_head=4, n_kv_head=2,
                                                 hidden=168, W=W_BAND)),
    "lm1": dict(cls="DecisionLlama", kwargs=dict(V=V_SIZE, T=P_ROPE, n_layer=1, n_embd=32, n_head=4, n_kv_head=2,
                                                  hidden=360, W=128)),
    "gru": dict(cls="RNN", kwargs=dict(V=V_SIZE, T=P_ROPE, n_embd=48, n_hidden=82, cell="gru")),
}


def build(spec):
    cls = {"DecisionLlama": DecisionLlama, "RNN": RNN}[spec["cls"]]
    m = cls(**spec["kwargs"])
    if isinstance(m, Llama):
        source_init(m)
    return m


if __name__ == "__main__":
    for k, s in SPECS.items():
        m = build(s)
        x = torch.randint(0, V_SIZE, (2, 300))
        assert m(x).shape == (2, 300, V_SIZE)
        print(f"{k:4s} {n_params(m):6d} params")
