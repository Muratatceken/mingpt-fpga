"""
stories260K (llama2.c, Karpathy's TinyStories series) through the same flow as the zoo:
pretrained float checkpoint -> IR -> INT8 -> microcoded RTL (rtl/zoo/zoo_core.sv).
  python3 stories.py [--ncal N] [--ntest N] [--gen N]
Data: data/stories260K/{stories260K.bin, tok512.bin, valid_400.txt}
  (huggingface.co/karpathy/tinyllamas, stories260K; first 400 stories of TinyStories-valid.txt)
Outputs: build/zoo/stories260k.qprog.pkl, model/zoo/int_metrics.json["stories260k"]
"""
import json
import os
import pickle
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import ROOT
from ir import attach_rope, compile_llama, float_logits
from models import Llama, n_params
from quant import IntRunner, calibrate, nll, quantize

DIR = os.path.join(ROOT, "data", "stories260K")
BUILD = os.path.join(ROOT, "build", "zoo")
NAME = "stories260k"
BOS, EOS = 1, 2


# ----------------------------------------------------------------------------- checkpoint
def load_model():
    """llama2.c legacy .bin: 7 int32 header, then float32 tensors in a fixed order (run.c memory_map_weights)."""
    raw = open(os.path.join(DIR, "stories260K.bin"), "rb").read()
    dim, hidden, nl, nh, nkv, V, T = struct.unpack("7i", raw[:28])
    shared, V = V > 0, abs(V)
    w = np.frombuffer(raw[28:], dtype=np.float32)
    dh, off = dim // nh, 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        a = torch.tensor(w[off:off + n].reshape(shape).copy())
        off += n
        return a

    emb = take(V, dim)
    att, wq, wk = take(nl, dim), take(nl, nh * dh, dim), take(nl, nkv * dh, dim)
    wv, wo, ffn = take(nl, nkv * dh, dim), take(nl, dim, nh * dh), take(nl, dim)
    w1, w2, w3 = take(nl, hidden, dim), take(nl, dim, hidden), take(nl, hidden, dim)
    fin = take(dim)
    take(T, dh // 2), take(T, dh // 2)                     # legacy RoPE tables (recomputed)
    assert shared and off == len(w), "unexpected checkpoint layout"
    sd = {"tok_embeddings.weight": emb, "norm.weight": fin, "output.weight": emb}
    for l in range(nl):
        for k, v in dict(attention_norm=att, wq=wq, wk=wk, wv=wv, wo=wo, ffn_norm=ffn, w1=w1, w2=w2, w3=w3).items():
            sd[f"layers.{l}.{k}.weight"] = v[l]
    m = Llama(V, T, nl, dim, nh, nkv, hidden)
    m.load_state_dict(sd)
    m.eval()
    return m


# ----------------------------------------------------------------------------- tokenizer (run.c)
class Tokenizer:
    def __init__(self, path=os.path.join(DIR, "tok512.bin"), V=512):
        raw = open(path, "rb").read()
        self.vocab, self.scores, p = [], [], 4
        for _ in range(V):
            sc, n = struct.unpack("fi", raw[p:p + 8])
            self.vocab.append(raw[p + 8:p + 8 + n])
            self.scores.append(sc)
            p += 8 + n
        self.lookup = {s: i for i, s in enumerate(self.vocab)}

    def encode(self, text, bos=True):
        toks = [BOS] if bos else []
        if text:
            toks.append(self.lookup[b" "])                 # dummy prefix, as sentencepiece
        for ch in text:
            b = ch.encode("utf-8")
            if b in self.lookup:
                toks.append(self.lookup[b])
            else:
                toks += [x + 3 for x in b]                 # byte fallback
        while True:                                        # greedy BPE merges by score
            best, bi, bid = -1e10, -1, -1
            for i in range(len(toks) - 1):
                k = self.lookup.get(self.vocab[toks[i]] + self.vocab[toks[i + 1]])
                if k is not None and self.scores[k] > best:
                    best, bi, bid = self.scores[k], i, k
            if bi < 0:
                return toks
            toks[bi:bi + 2] = [bid]

    def decode(self, toks):
        out, prev = b"", BOS
        for t in toks:
            if t == BOS:
                prev = t
                continue
            piece = self.vocab[t]
            if prev == BOS and piece.startswith(b" "):
                piece = piece[1:]
            if piece.startswith(b"<0x") and piece.endswith(b">") and len(piece) == 6:
                piece = bytes([int(piece[3:5], 16)])
            out += piece
            prev = t
        return out.decode("utf-8", errors="replace")


def stories(T=512):
    txt = open(os.path.join(DIR, "valid_400.txt"), encoding="utf-8").read()
    tok = Tokenizer()
    return [tok.encode(s.strip())[:T + 1] for s in txt.split("<|endoftext|>") if s.strip()]


def split():
    """first 100 stories: calibration, stories 200..: test."""
    s = stories()
    return s[:100], s[200:]


def greedy(step, n, bos=BOS):
    toks = [bos]
    for t in range(n):
        lg = step(toks[-1], t)
        toks.append(int(np.argmax(lg)))
    return toks


# ----------------------------------------------------------------------------- main
def main():
    args = sys.argv[1:]
    if "--decode" in args:                     # python3 stories.py --decode build/zoo/stories260k_gen_rtl.txt
        print(Tokenizer().decode([int(x) for x in open(args[args.index("--decode") + 1]).read().split()]))
        return
    opt = lambda k, d: int(args[args.index(k) + 1]) if k in args else d
    ncal, ntest, ngen = opt("--ncal", 100), opt("--ntest", 50), opt("--gen", 200)
    t0 = time.time()
    m = load_model()
    tok = Tokenizer()
    cal, test = split()
    cal, test = cal[:ncal], test[:ntest]
    print(f"stories260K: {n_params(m)} parameters, V={m.output.weight.shape[0]}, T={m.cos.shape[0]}, "
          f"{len(cal)} calibration / {len(test)} test stories ({sum(len(s) - 1 for s in test)} tokens)")

    # 1. float model (validates checkpoint loader + tokenizer: Karpathy reports val loss 1.297)
    with torch.no_grad():
        tot = n = 0
        for s in test:
            x = torch.tensor(s[:-1])[None]
            y = torch.tensor(s[1:])
            tot += F.cross_entropy(m(x)[0], y, reduction="sum").item()
            n += len(y)
    print(f"PyTorch test loss: {tot / n:.4f}")

    # 2. compiler check
    prog = attach_rope(compile_llama(m))
    prog.name = NAME
    with torch.no_grad():
        err = max(np.abs(float_logits(prog, s[:-1]) - m(torch.tensor(s[:-1])[None])[0].double().numpy()).max()
                  for s in test[:3])
    print(f"IR vs torch: max |diff| = {err:.2e}")

    # 3. INT8
    rec = calibrate(prog, [s[:-1] for s in cal])
    q = quantize(prog, rec)
    from ir import FloatRunner
    tf = ti = 0.0
    n = agree = 0
    for s in test:
        fr, ir = FloatRunner(prog), IntRunner(q)
        for t, (a, b) in enumerate(zip(s[:-1], s[1:])):
            lf = fr.step(a, t)
            li = ir.step(a, t) * q.s_logit
            tf += nll(lf, b)
            ti += nll(li, b)
            agree += int(np.argmax(lf) == np.argmax(li))
            n += 1
    res = dict(float_nll=tf / n, int_nll=ti / n, delta=(ti - tf) / n, argmax_agree=agree / n, tokens=n,
               FR=q.FR, ops=len(q.ops), params=n_params(m), seconds=round(time.time() - t0, 1))
    print(f"{NAME}: float {res['float_nll']:.4f}  int8 {res['int_nll']:.4f}  Δ {res['delta']:+.4f}  "
          f"argmax uyumu {100 * res['argmax_agree']:.1f}%")

    # 4. greedy story from BOS: float vs integer (the integer one is what the RTL must reproduce)
    fr, ir = FloatRunner(prog), IntRunner(q)
    gf, gi = greedy(fr.step, ngen), greedy(ir.step, ngen)
    same = next((k for k in range(len(gf)) if gf[k] != gi[k]), len(gf))
    res["gen_float"], res["gen_int"], res["gen_same_prefix"] = tok.decode(gf), tok.decode(gi), same
    print(f"\n--- float greedy ---\n{res['gen_float']}\n--- INT8 greedy (first {same} tokens identical) ---\n"
          f"{res['gen_int']}")

    os.makedirs(BUILD, exist_ok=True)
    pickle.dump(q, open(os.path.join(BUILD, f"{NAME}.qprog.pkl"), "wb"))
    path = os.path.join(ROOT, "model", "zoo", "int_metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {}
    metrics[NAME] = res
    json.dump(metrics, open(path, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
