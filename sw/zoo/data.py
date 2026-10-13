"""
names.txt dataset exactly as in makemore: vocabulary = 26 letters + token 0 (start/end),
x = [0, w...] padded with 0, y = [w..., 0] padded with -1, block size = max length + 1 = 16.
Split: torch.manual_seed(3407); randperm; last 1000 names are the test set.
"""
import os

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NAMES = os.path.join(ROOT, "data", "names.txt")


def load(seed=3407):
    words = [w.strip() for w in open(NAMES).read().splitlines()]
    words = [w for w in words if w]
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    T = max(len(w) for w in words) + 1
    torch.manual_seed(seed)
    rp = torch.randperm(len(words)).tolist()
    n_test = min(1000, int(len(words) * 0.1))
    train = [words[i] for i in rp[:-n_test]]
    test = [words[i] for i in rp[-n_test:]]

    def tensors(ws):
        X = torch.zeros(len(ws), T, dtype=torch.long)
        Y = torch.full((len(ws), T), -1, dtype=torch.long)
        for n, w in enumerate(ws):
            ix = torch.tensor([stoi[c] for c in w], dtype=torch.long)
            X[n, 1:1 + len(ix)] = ix
            Y[n, :len(ix)] = ix
            Y[n, len(ix)] = 0
        return X, Y

    itos = {i: c for c, i in stoi.items()}
    itos[0] = "."
    return dict(train=tensors(train), test=tensors(test), test_words=test,
                stoi=stoi, itos=itos, V=len(chars) + 1, T=T)
