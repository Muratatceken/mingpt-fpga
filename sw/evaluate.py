"""
Step 3: compare float minGPT vs. the bit-exact integer model.

  - task accuracy (all 729 sort inputs, greedy decoding)
  - token agreement between float and integer model
  - logit error of the integer model under teacher forcing (same input sequences)
"""
import numpy as np
import torch

from common import SORT_LENGTH, load_model, all_sort_inputs
from intmodel import IntGPT, load_qparams


def main():
    model = load_model()
    q = load_qparams()
    inputs = all_sort_inputs()
    n = SORT_LENGTH

    with torch.no_grad():
        full = model.generate(torch.tensor(inputs), n, do_sample=False)
    float_out = full[:, n:].numpy()
    ref = np.sort(np.array(inputs), axis=1)

    int_out, err, sig = [], [], []
    with torch.no_grad():
        float_logits = model(full[:, :-1])[0].double().numpy()
    for i, prompt in enumerate(inputs):
        gen, _ = IntGPT(q).generate(prompt, n)
        int_out.append(gen)
        # teacher-forced logits on the float model's sequence
        m = IntGPT(q)
        for t, tok in enumerate(full[i, :-1].tolist()):
            _, lg, _ = m.step(tok, t)
            lf = float_logits[i, t]
            li = np.array(lg) * q["s_logit"]
            err.append(li - lf)
            sig.append(lf - lf.mean())
    int_out = np.array(int_out)
    err, sig = np.concatenate(err), np.concatenate(sig)

    print(f"float  accuracy : {(float_out == ref).all(1).mean() * 100:6.2f}%  (729 inputs)")
    print(f"int    accuracy : {(int_out == ref).all(1).mean() * 100:6.2f}%")
    print(f"token agreement : {(int_out == float_out).mean() * 100:6.2f}%  (float vs int, per token)")
    print(f"logit error     : max |e| = {np.abs(err).max():.4f}, rms = {np.sqrt((err ** 2).mean()):.4f}")
    snr = 10 * np.log10((sig ** 2).mean() / ((err - err.mean()) ** 2).mean())
    print(f"logit SQNR      : {snr:.1f} dB (offset-invariant)")


if __name__ == "__main__":
    main()
