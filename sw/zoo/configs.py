"""
Model configurations for the experiments: every architecture at a given parameter budget on a given dataset.

    make(arch, dataset, budget) -> (constructor name, kwargs)      build(spec) -> nn.Module

Rules (the same for all architectures, see report Part IV):
  * transformers: n_layer fixed per dataset, head size 8 (the hardware's attention lane count),
    width C from the budget tier, MLP width searched so that the parameter count matches the budget;
    GQA models use n_kv = n_head / 2
  * MLP / RNN / GRU / BoW: embedding size from the budget tier, hidden width searched
  * bigram has no free size (V*V parameters)
Families follow their source repositories; LADDER changes one design choice per step
(GPT-2 -> Llama 2, and a nanochat branch).
"""
import torch

from models import BoW, Bigram, GPT2, Ladder, Llama, MLP, MicroGPT, NanoChat, RNN, n_params

DATASETS = dict(names=dict(V=27, T=16, L=4), stories=dict(V=512, T=256, L=5))
FAMILY = ["bigram", "mlp", "rnn", "gru", "bow", "gpt2", "microgpt", "llama", "nanochat"]

# one design change per step (equal budget via MLP width)
LADDER = {
    "lad0_gpt2":   dict(),                                                      # LayerNorm, learned pos, GELU, MHA
    "lad1_rms":    dict(norm="rms"),                                            # + RMSNorm
    "lad2_rope":   dict(norm="rms", pos="rope"),                                # + RoPE instead of learned pos
    "lad3_swiglu": dict(norm="rms", pos="rope", mlp="swiglu"),                  # + SwiGLU
    "lad4_gqa":    dict(norm="rms", pos="rope", mlp="swiglu", gqa=True),        # + GQA
    "lad5_llama":  dict(norm="rms", pos="rope", mlp="swiglu", gqa=True, bias=False, tied=True),  # = llama2.c
    "lad6_relu2":  dict(norm="rms", pos="rope", mlp="relu2", gqa=True),         # branch: ReLU^2 instead of SwiGLU
    "lad7_qknorm": dict(norm="rms", pos="rope", mlp="relu2", gqa=True, qk_norm=True),  # + QK-norm
}

# width tiers: budget -> (transformer width C, embedding size for MLP/RNN/GRU, BoW width)
TIERS = {
    "names": {12000: (16, 8, 24, 24), 50000: (32, 16, 48, 64), 200000: (64, 32, 64, 128)},
    "stories": {260000: (64, 32, 64, 64)},
}


def _tier(dataset, budget):
    t = TIERS[dataset]
    return t[min(t, key=lambda b: abs(b - budget))]


def _search(make, budget, lo, hi, step):
    """free width h in [lo, hi] (multiples of step) whose model is closest to the budget."""
    best = None
    for h in range(lo, hi + 1, step):
        n = n_params(make(h))
        if best is None or abs(n - budget) < abs(best[1] - budget):
            best = (h, n)
    return best[0]


def make(arch, dataset="names", budget=50000):
    d = DATASETS[dataset]
    V, T, L = d["V"], d["T"], d["L"]
    C, e_mlp, e_rnn, c_bow = _tier(dataset, budget)
    nh = C // 8
    k = min(T, 16)                                          # MLP context window
    if arch == "bigram":
        return "Bigram", dict(V=V, T=T)
    if arch == "mlp":
        f = lambda h: dict(V=V, T=T, n_embd=e_mlp, n_hidden=h, k=k)
        return "MLP", f(_search(lambda h: MLP(**f(h)), budget, 8, 4096, 4))
    if arch in ("rnn", "gru"):
        f = lambda h: dict(V=V, T=T, n_embd=e_rnn, n_hidden=h, cell=arch)
        return "RNN", f(_search(lambda h: RNN(**f(h)), budget, 8, 2048, 4))
    if arch == "bow":
        f = lambda h: dict(V=V, T=T, n_embd=c_bow, n_hidden=h)
        return "BoW", f(_search(lambda h: BoW(**f(h)), budget, 8, 8192, 8))
    if arch == "gpt2":
        f = lambda h: dict(V=V, T=T, n_layer=L, n_embd=C, n_head=nh, hidden=h)
        return "GPT2", f(_search(lambda h: GPT2(**f(h)), budget, 8, 16 * C, 8))
    if arch == "microgpt":
        f = lambda h: dict(V=V, T=T, n_layer=L, n_embd=C, n_head=nh, hidden=h)
        return "MicroGPT", f(_search(lambda h: MicroGPT(**f(h)), budget, 8, 16 * C, 8))
    if arch == "llama":
        f = lambda h: dict(V=V, T=T, n_layer=L, n_embd=C, n_head=nh, n_kv_head=max(1, nh // 2), hidden=h)
        return "Llama", f(_search(lambda h: Llama(**f(h)), budget, 8, 16 * C, 4))
    if arch == "nanochat":
        f = lambda h: dict(V=V, T=T, n_layer=L, n_embd=C, n_head=nh, n_kv_head=max(1, nh // 2), hidden=h)
        return "NanoChat", f(_search(lambda h: NanoChat(**f(h)), budget, 8, 16 * C, 8))
    if arch in LADDER:
        sw = dict(LADDER[arch])
        gqa = sw.pop("gqa", False)
        f = lambda h: dict(V=V, T=T, n_layer=L, n_embd=C, n_head=nh, n_kv_head=max(1, nh // 2) if gqa else nh,
                           hidden=h, **sw)
        return "Ladder", f(_search(lambda h: Ladder(**f(h)), budget, 8, 16 * C, 4))
    raise KeyError(arch)


CLASSES = dict(Bigram=Bigram, MLP=MLP, RNN=RNN, BoW=BoW, GPT2=GPT2, MicroGPT=MicroGPT, Llama=Llama,
               NanoChat=NanoChat, Ladder=Ladder)


def build(spec):
    """spec = {"cls": name, "kwargs": {...}} (stored next to every checkpoint)"""
    return CLASSES[spec["cls"]](**spec["kwargs"])


if __name__ == "__main__":
    torch.manual_seed(0)
    for ds, tiers in TIERS.items():
        for budget in tiers:
            print(f"--- {ds} {budget}")
            for a in FAMILY + list(LADDER):
                cls, kw = make(a, ds, budget)
                m = CLASSES[cls](**kw)
                extra = {k: v for k, v in kw.items() if k not in ("V", "T")}
                print(f"  {a:12s} {n_params(m):7d}  {extra}")
