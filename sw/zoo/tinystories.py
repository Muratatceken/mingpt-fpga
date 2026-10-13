"""
TinyStories with the 512-token vocabulary of stories260K (llama2.c's tok512), as in llama2.c/tinystories.py:
every story is encoded with BOS and the stories of a shard are concatenated into one uint16 stream.
  python3 tinystories.py pretokenize        data/tinystories/shards/*.json -> data/tinystories/tok512/*.bin
Shard data00 is the validation split (as in llama2.c), the other 49 are training data.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

from data import ROOT

DIR = os.path.join(ROOT, "data", "tinystories")
BIN = os.path.join(DIR, "tok512")
BOS = 1


def _encode_shard(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(DIR, "tok512.model"))
    out = os.path.join(BIN, os.path.basename(path).replace(".json", ".bin"))
    if os.path.exists(out):
        return out
    stories = json.load(open(path))
    toks = []
    for ex in stories:
        toks += [BOS] + sp.encode(ex["story"].strip())
    np.array(toks, dtype=np.uint16).tofile(out + ".tmp")
    os.replace(out + ".tmp", out)
    return out


def pretokenize(procs=6):
    os.makedirs(BIN, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(DIR, "shards", "*.json")))
    with Pool(procs) as pool:
        for i, out in enumerate(pool.imap_unordered(_encode_shard, shards)):
            print(f"{i + 1}/{len(shards)} {os.path.basename(out)} {os.path.getsize(out) // 2:,} tokens", flush=True)


class Batches:
    """random (x, y) windows of length T from the train or val stream (memory-mapped)."""

    def __init__(self, split, T, seed=0):
        files = sorted(glob.glob(os.path.join(BIN, "*.bin")))
        files = files[1:] if split == "train" else files[:1]
        self.data = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.T = T
        self.g = np.random.default_rng(seed)

    def get(self, B):
        X = np.empty((B, self.T), dtype=np.int64)
        Y = np.empty((B, self.T), dtype=np.int64)
        for b in range(B):
            d = self.data[self.g.integers(len(self.data))]
            i = self.g.integers(len(d) - self.T - 1)
            w = d[i:i + self.T + 1].astype(np.int64)
            X[b], Y[b] = w[:-1], w[1:]
        return torch.from_numpy(X), torch.from_numpy(Y)


def val_windows(T, which="val", n=400, seed=1234):
    """fixed held-out sets from shard data00: 2n windows of T+1 tokens that start at a story boundary (BOS),
    so every model sees whole-story prefixes (needed for the per-position analysis); the first n are the
    validation set (model selection), the other n the test set (reporting)."""
    d = np.memmap(sorted(glob.glob(os.path.join(BIN, "*.bin")))[0], dtype=np.uint16, mode="r")
    starts = np.flatnonzero(d[: len(d) - T - 1] == BOS)
    g = np.random.default_rng(seed)
    sel = g.choice(starts, size=2 * n, replace=False)
    sel = np.sort(sel[:n] if which == "val" else sel[n:])
    w = np.stack([d[s:s + T + 1].astype(np.int64) for s in sel])
    return torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:])


if __name__ == "__main__":
    if sys.argv[1:] == ["pretokenize"]:
        pretokenize()
