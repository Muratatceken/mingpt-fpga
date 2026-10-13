"""
Differential fuzzing of the RTL against the integer reference (IntRunner).
  python3 fuzz.py [N] [--seed S]
Every case draws an architecture (family or ladder), vocabulary, context, depth and width at random,
fills the weights with random values of random scale (so that saturation, clipping and large dynamic
ranges are exercised), calibrates on random tokens, builds the INT8 program, exports it to
build/zoo/fuzz/<case>/ and simulates it with Verilator (assertions on). Every logit of random token
sequences of random length must match. Cases the compiler rejects (a numeric range the hardware
formats cannot hold) are counted separately.
Results: model/zoo/fuzz.json
"""
import json
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import torch

import configs
from configs import FAMILY, LADDER, build, make
from data import ROOT
from export_hw import export
from ir import attach_rope, compile_model
from models import n_params, source_init
from quant import IntRunner, calibrate, quantize

OUT = os.path.join(ROOT, "build", "zoo", "fuzz")


def randomize(m, rng):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("bias"):
                p.normal_(0.0, 0.1)
            elif p.dim() == 1:                                  # norm gains, RNN start state
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif "wte" in name or "wpe" in name or "tok_embeddings" in name or name == "logits":
                p.normal_(0.0, 10 ** rng.uniform(-1.0, 0.3))
            else:
                p.normal_(0.0, 10 ** rng.uniform(-1.6, -0.4))
    return m


def case(i, rng):
    arch = rng.choice(FAMILY + list(LADDER))
    V = rng.choice([27, 50, 97, 200, 512])
    T = rng.choice([8, 16, 24, 32, 64])
    L = rng.choice([1, 2, 3])
    C = rng.choice([16, 32, 48, 64]) if arch in ("llama", "nanochat") or "gqa" in str(LADDER.get(arch, "")) \
        else rng.choice([8, 16, 24, 32, 48, 64])
    tier = (C, rng.choice([4, 8, 16]), rng.choice([8, 16, 32]), rng.choice([8, 16, 32]))
    budget = int(10 ** rng.uniform(3.6, 4.8))
    configs.DATASETS["fuzz"] = dict(V=V, T=T, L=L)
    configs.TIERS["fuzz"] = {budget: tier}
    cls, kw = make(arch, "fuzz", budget)
    spec = dict(cls=cls, kwargs=kw)
    torch.manual_seed(1000 + i)
    m = randomize(source_init(build(spec)), rng).eval()
    prog = attach_rope(compile_model(arch, m))
    toks = lambda n: [rng.randrange(V) for _ in range(n)]
    info = dict(case=i, arch=arch, V=V, T=T, L=L, C=C, params=n_params(m), spec=spec)
    try:
        rec = calibrate(prog, [toks(T) for _ in range(24)])
        q = quantize(prog, rec)
        seqs = [toks(rng.randint(1, T)) for _ in range(6)]
        export(f"f{i:03d}", seqs, q=q, out=os.path.join(OUT, f"f{i:03d}"), quiet=True)
    except AssertionError as e:
        return dict(info, status="rejected", reason=str(e)[:200])
    r = subprocess.run([os.path.join(ROOT, "scripts", "zoo_sim.sh"), os.path.join("build", "zoo", "fuzz", f"f{i:03d}")],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    m_ = re.search(r"RESULT \S+ steps (\d+) bad_steps (\d+) errors (\d+) avg_cycles (\d+)", out)
    asserts = len(re.findall(r"Assertion failed|%Error", out))
    if not m_:
        return dict(info, status="sim_failed", log=out[-1500:])
    steps, bad, err, cyc = map(int, m_.groups())
    ok = err == 0 and asserts == 0
    return dict(info, status="pass" if ok else "FAIL", steps=steps, errors=err, assertions=asserts, cycles=cyc,
                logits=steps * V)


def main():
    args = sys.argv[1:]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 1
    n = int(args[0]) if args and args[0].isdigit() else 40
    rng = random.Random(seed)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(ROOT, "model", "zoo", "fuzz.json")
    res = []
    for i in range(n):
        r = case(i, rng)
        res.append(r)
        print(f"case {i:3d} {r['arch']:12s} V={r['V']:3d} T={r['T']:2d} L={r['L']} C={r['C']:2d} "
              f"{r['params']:6d} params: {r['status']}"
              + (f" ({r['steps']} steps, {r['logits']} logits, {r['errors']} errors)" if "steps" in r else "")
              + (f" [{r.get('reason', '')}]" if r["status"] == "rejected" else ""), flush=True)
        json.dump(dict(seed=seed, cases=res), open(path, "w"), indent=1)
    s = {k: sum(r["status"] == k for r in res) for k in ("pass", "FAIL", "rejected", "sim_failed")}
    print("summary:", s, "logits compared:", sum(r.get("logits", 0) for r in res))


if __name__ == "__main__":
    main()
