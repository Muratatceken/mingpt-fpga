"""
Shared helpers: minGPT import path, model config, and the Sort task dataset
(taken from minGPT's demo.ipynb).
"""
import os
import sys
import pickle

import torch
from torch.utils.data import Dataset

from config import (ROOT, BUILD_DIR, CKPT_PATH, QPARAM_PATH, MEM_DIR,  # noqa: F401
                    SORT_LENGTH, SORT_DIGITS, MODEL_TYPE, all_sort_inputs)

sys.path.insert(0, os.path.join(ROOT, "ref", "minGPT"))

from mingpt.model import GPT  # noqa: E402


class SortDataset(Dataset):
    """
    Sort problem from minGPT demo.ipynb. For length 6:
      input : 0 0 2 1 0 1 -> output: 0 0 0 1 1 2
    The transformer sees the concatenation (minus the last token):
      x: 0 0 2 1 0 1 0 0 0 1 1
      y: I I I I I 0 0 0 1 1 2     (I = ignore, -1)
    """

    def __init__(self, split, length=SORT_LENGTH, num_digits=SORT_DIGITS):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000

    def get_vocab_size(self):
        return self.num_digits

    def get_block_size(self):
        return self.length * 2 - 1

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    continue
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = "test" if h % 4 == 0 else "train"
            if inp_split == self.split:
                break
        sol = torch.sort(inp)[0]
        cat = torch.cat((inp, sol), dim=0)
        x = cat[:-1].clone()
        y = cat[1:].clone()
        y[: self.length - 1] = -1
        return x, y


def make_model():
    cfg = GPT.get_default_config()
    cfg.model_type = MODEL_TYPE
    cfg.vocab_size = SORT_DIGITS
    cfg.block_size = SORT_LENGTH * 2 - 1
    return GPT(cfg)


def load_model():
    model = make_model()
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()
    return model

