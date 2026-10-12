"""
Step 1: train minGPT (gpt-nano) on the Sort task, exactly as in minGPT's demo.ipynb.
Output: model/gpt_sort.pt
"""
import os

import torch

from common import CKPT_PATH, SortDataset, make_model, all_sort_inputs
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


def float_accuracy(model, inputs):
    x = torch.tensor(inputs, dtype=torch.long)
    with torch.no_grad():
        out = model.generate(x, len(inputs[0]), do_sample=False)[:, len(inputs[0]):]
    ref = torch.sort(x, dim=1)[0]
    return (out == ref).all(dim=1).float().mean().item()


def main():
    set_seed(3407)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    train_dataset = SortDataset("train")
    model = make_model()

    tc = Trainer.get_default_config()
    tc.learning_rate = 5e-4
    tc.max_iters = 5000
    tc.num_workers = 0
    tc.device = "cpu"
    trainer = Trainer(tc, model, train_dataset)

    def on_batch_end(tr):
        if tr.iter_num % 500 == 0:
            print(f"iter {tr.iter_num}: train loss {tr.loss.item():.5f}")

    trainer.set_callback("on_batch_end", on_batch_end)
    trainer.run()

    model.eval()
    acc = float_accuracy(model, all_sort_inputs())
    print(f"float model accuracy on all 729 inputs: {acc * 100:.2f}%")
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"saved {CKPT_PATH}")


if __name__ == "__main__":
    main()
