"""
The Karpathy language-model family, re-implemented with one interface:
    logits = model(idx)          idx: (B, T) token ids  ->  logits: (B, T, V)

Each class follows its source repository (see report Part II):
    bigram, mlp, rnn, gru, bow   karpathy/makemore (makemore.py)
    gpt2                         minGPT / nanoGPT / makemore Transformer
    microgpt                     karpathy microgpt gist (2026)
    llama                        karpathy/llama2.c model.py
    nanochat                     karpathy/nanochat gpt.py, core architecture only
                                 (no value embeddings, smear, lambdas, backout, sliding window)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
def rms(x, eps=1e-5):
    """Parameter-free RMSNorm over the last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable gain (llama2.c)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms(x, self.eps) * self.weight


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def causal_attention(q, k, v, n_rep=1):
    """q: (B,H,T,d), k,v: (B,Hkv,T,d). GQA by repeating kv heads (head h uses kv head h//n_rep)."""
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    T, d = q.size(2), q.size(3)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1) @ v


def rope_tables(d_head, T, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2)[: d_head // 2].float() / d_head))
    ang = torch.outer(torch.arange(T).float(), freqs)          # (T, d/2)
    return torch.cos(ang), torch.sin(ang)


def rope_interleaved(x, cos, sin):
    """llama2.c: pairs (2i, 2i+1). x: (B,H,T,d)."""
    xr, xi = x.reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    c, s = cos[: x.size(2)], sin[: x.size(2)]
    out_r = xr * c - xi * s
    out_i = xr * s + xi * c
    return torch.stack([out_r, out_i], dim=-1).flatten(-2)


def rope_half(x, cos, sin):
    """nanochat: pairs (i, i+d/2), rotation by -theta (y1 = x1 c + x2 s, y2 = -x1 s + x2 c)."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos[: x.size(2)], sin[: x.size(2)]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], dim=-1)


def split_heads(x, n):
    B, T, C = x.shape
    return x.view(B, T, n, C // n).transpose(1, 2)


def merge_heads(x):
    B, H, T, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d)


# ----------------------------------------------------------------------------- makemore models
class Bigram(nn.Module):
    def __init__(self, V, T):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(V, V))

    def forward(self, idx):
        return self.logits[idx]


class MLP(nn.Module):
    """Bengio et al. 2003 as in makemore: embeddings of the last T tokens (current first)."""

    def __init__(self, V, T, n_embd, n_hidden):
        super().__init__()
        self.V, self.T = V, T
        self.wte = nn.Embedding(V + 1, n_embd)            # +1: <BLANK> padding token
        self.fc1 = nn.Linear(T * n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, V)

    def forward(self, idx):
        idx = idx.clone()
        embs = []
        for _ in range(self.T):
            embs.append(self.wte(idx))
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.V
        return self.fc2(torch.tanh(self.fc1(torch.cat(embs, -1))))


class RNN(nn.Module):
    """Vanilla RNN or GRU language model as in makemore."""

    def __init__(self, V, T, n_embd, n_hidden, cell="rnn"):
        super().__init__()
        self.cell = cell
        self.start = nn.Parameter(torch.zeros(1, n_hidden))
        self.wte = nn.Embedding(V, n_embd)
        n_in = n_embd + n_hidden
        if cell == "rnn":
            self.xh_to_h = nn.Linear(n_in, n_hidden)
        else:
            self.xh_to_z = nn.Linear(n_in, n_hidden)
            self.xh_to_r = nn.Linear(n_in, n_hidden)
            self.xh_to_hbar = nn.Linear(n_in, n_hidden)
        self.lm_head = nn.Linear(n_hidden, V)

    def step(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=-1)
        if self.cell == "rnn":
            return torch.tanh(self.xh_to_h(xh))
        r = torch.sigmoid(self.xh_to_r(xh))
        hbar = torch.tanh(self.xh_to_hbar(torch.cat([xt, r * hprev], dim=-1)))
        z = torch.sigmoid(self.xh_to_z(xh))
        return (1 - z) * hprev + z * hbar

    def forward(self, idx):
        B, T = idx.shape
        emb = self.wte(idx)
        h = self.start.expand(B, -1)
        hs = []
        for t in range(T):
            h = self.step(emb[:, t], h)
            hs.append(h)
        return self.lm_head(torch.stack(hs, 1))


class BoW(nn.Module):
    """Causal bag of words (makemore): x += mean_{j<=t} x_j ; x += c_proj(tanh(c_fc(x)))."""

    def __init__(self, V, T, n_embd, n_hidden):
        super().__init__()
        self.wte = nn.Embedding(V, n_embd)
        self.wpe = nn.Embedding(T, n_embd)
        self.c_fc = nn.Linear(n_embd, n_hidden)
        self.c_proj = nn.Linear(n_hidden, n_embd)
        self.lm_head = nn.Linear(n_embd, V)

    def forward(self, idx):
        T = idx.size(1)
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        cnt = torch.arange(1, T + 1, device=idx.device, dtype=x.dtype).view(1, T, 1)
        x = x + x.cumsum(1) / cnt
        x = x + self.c_proj(torch.tanh(self.c_fc(x)))
        return self.lm_head(x)


# ----------------------------------------------------------------------------- transformers
class GPT2(nn.Module):
    """GPT-2 block (minGPT/makemore): LayerNorm, GELU, learned positions, biases; lm_head without bias."""

    def __init__(self, V, T, n_layer, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        C = n_embd
        self.wte = nn.Embedding(V, C)
        self.wpe = nn.Embedding(T, C)
        self.h = nn.ModuleList()
        for _ in range(n_layer):
            self.h.append(nn.ModuleDict(dict(
                ln_1=nn.LayerNorm(C), c_attn=nn.Linear(C, 3 * C), c_proj=nn.Linear(C, C),
                ln_2=nn.LayerNorm(C), c_fc=nn.Linear(C, 4 * C), mlp_proj=nn.Linear(4 * C, C))))
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V, bias=False)

    def forward(self, idx):
        T = idx.size(1)
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.h:
            q, k, v = b["c_attn"](b["ln_1"](x)).chunk(3, dim=-1)
            y = causal_attention(split_heads(q, self.n_head), split_heads(k, self.n_head), split_heads(v, self.n_head))
            x = x + b["c_proj"](merge_heads(y))
            x = x + b["mlp_proj"](new_gelu(b["c_fc"](b["ln_2"](x))))
        return self.lm_head(self.ln_f(x))


class MicroGPT(nn.Module):
    """microgpt: GPT-2 with RMSNorm (no params), ReLU, no biases, no final norm; normalized embedding."""

    def __init__(self, V, T, n_layer, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        C = n_embd
        self.wte = nn.Embedding(V, C)
        self.wpe = nn.Embedding(T, C)
        self.h = nn.ModuleList()
        for _ in range(n_layer):
            self.h.append(nn.ModuleDict(dict(
                wq=nn.Linear(C, C, bias=False), wk=nn.Linear(C, C, bias=False),
                wv=nn.Linear(C, C, bias=False), wo=nn.Linear(C, C, bias=False),
                fc1=nn.Linear(C, 4 * C, bias=False), fc2=nn.Linear(4 * C, C, bias=False))))
        self.lm_head = nn.Linear(C, V, bias=False)

    def forward(self, idx):
        T = idx.size(1)
        x = rms(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        H = self.n_head
        for b in self.h:
            a = rms(x)
            y = causal_attention(split_heads(b["wq"](a), H), split_heads(b["wk"](a), H), split_heads(b["wv"](a), H))
            x = x + b["wo"](merge_heads(y))
            x = x + b["fc2"](F.relu(b["fc1"](rms(x))))
        return self.lm_head(x)


class Llama(nn.Module):
    """llama2.c: RMSNorm with gain, RoPE (interleaved), GQA, SwiGLU, tied embeddings, no biases."""

    def __init__(self, V, T, n_layer, n_embd, n_head, n_kv_head, hidden):
        super().__init__()
        C = n_embd
        self.n_head, self.n_kv, self.dh = n_head, n_kv_head, C // n_head
        self.tok_embeddings = nn.Embedding(V, C)
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.ModuleDict(dict(
                attention_norm=RMSNorm(C),
                wq=nn.Linear(C, C, bias=False), wk=nn.Linear(C, n_kv_head * self.dh, bias=False),
                wv=nn.Linear(C, n_kv_head * self.dh, bias=False), wo=nn.Linear(C, C, bias=False),
                ffn_norm=RMSNorm(C),
                w1=nn.Linear(C, hidden, bias=False), w3=nn.Linear(C, hidden, bias=False),
                w2=nn.Linear(hidden, C, bias=False))))
        self.norm = RMSNorm(C)
        self.output = nn.Linear(C, V, bias=False)
        self.output.weight = self.tok_embeddings.weight            # weight tying
        cos, sin = rope_tables(self.dh, T)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for b in self.layers:
            a = b["attention_norm"](x)
            q = rope_interleaved(split_heads(b["wq"](a), self.n_head), self.cos, self.sin)
            k = rope_interleaved(split_heads(b["wk"](a), self.n_kv), self.cos, self.sin)
            v = split_heads(b["wv"](a), self.n_kv)
            x = x + b["wo"](merge_heads(causal_attention(q, k, v, self.n_head // self.n_kv)))
            f = b["ffn_norm"](x)
            x = x + b["w2"](F.silu(b["w1"](f)) * b["w3"](f))
        return self.output(self.norm(x))


class NanoChat(nn.Module):
    """nanochat core: param-free RMSNorm (also after embedding and before head), RoPE (half split),
    QK-norm with 1.2 scale, GQA, ReLU^2 MLP, untied head, no biases."""

    def __init__(self, V, T, n_layer, n_embd, n_head, n_kv_head, eps=1e-6):
        super().__init__()
        C = n_embd
        self.n_head, self.n_kv, self.dh, self.eps = n_head, n_kv_head, C // n_head, eps
        self.wte = nn.Embedding(V, C)
        self.h = nn.ModuleList()
        for _ in range(n_layer):
            self.h.append(nn.ModuleDict(dict(
                c_q=nn.Linear(C, C, bias=False), c_k=nn.Linear(C, n_kv_head * self.dh, bias=False),
                c_v=nn.Linear(C, n_kv_head * self.dh, bias=False), c_proj=nn.Linear(C, C, bias=False),
                c_fc=nn.Linear(C, 4 * C, bias=False), mlp_proj=nn.Linear(4 * C, C, bias=False))))
        self.lm_head = nn.Linear(C, V, bias=False)
        cos, sin = rope_tables(self.dh, T)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        e = self.eps
        x = rms(self.wte(idx), e)
        for b in self.h:
            a = rms(x, e)
            q = rope_half(split_heads(b["c_q"](a), self.n_head), self.cos, self.sin)
            k = rope_half(split_heads(b["c_k"](a), self.n_kv), self.cos, self.sin)
            q, k = rms(q, e) * 1.2, rms(k, e) * 1.2
            v = split_heads(b["c_v"](a), self.n_kv)
            x = x + b["c_proj"](merge_heads(causal_attention(q, k, v, self.n_head // self.n_kv)))
            x = x + b["mlp_proj"](F.relu(b["c_fc"](rms(x, e))).square())
        return self.lm_head(rms(x, e))


# ----------------------------------------------------------------------------- the zoo (~50k params each)
V, T = 27, 16
ZOO = {
    "bigram":   lambda: Bigram(V, T),
    "mlp":      lambda: MLP(V, T, n_embd=16, n_hidden=176),
    "rnn":      lambda: RNN(V, T, n_embd=48, n_hidden=184, cell="rnn"),
    "gru":      lambda: RNN(V, T, n_embd=48, n_hidden=100, cell="gru"),
    "bow":      lambda: BoW(V, T, n_embd=64, n_hidden=352),
    "gpt2":     lambda: GPT2(V, T, n_layer=4, n_embd=32, n_head=4),
    "microgpt": lambda: MicroGPT(V, T, n_layer=4, n_embd=32, n_head=4),
    "llama":    lambda: Llama(V, T, n_layer=4, n_embd=32, n_head=4, n_kv_head=2, hidden=96),
    "nanochat": lambda: NanoChat(V, T, n_layer=4, n_embd=32, n_head=4, n_kv_head=2),
}


def n_params(model):
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


if __name__ == "__main__":
    idx = torch.randint(0, V, (2, T))
    for name, make in ZOO.items():
        m = make()
        out = m(idx)
        assert out.shape == (2, T, V), (name, out.shape)
        print(f"{name:9s} {n_params(m):7d} params")
