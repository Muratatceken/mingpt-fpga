"""
Per-run pipeline for every trained checkpoint (resumable, safe to run while training continues):
  float model -> IR -> INT8 (quant.CALIB) -> float / INT8 quality on the test set -> export ->
  bit-exact RTL check of every logit on a verification engine (one Verilator build per dataset, models
  loaded through ld_*; cycle counts equal those of a per-model build) -> model/zoo/pipeline.json
  python3 pipeline.py [tag ...]            (default: every tag under model/zoo/runs/)
names: the whole test set (1000 names) in INT8 and in RTL. stories: INT8 quality on 64 test windows,
RTL on 4 windows (256 positions each), float per-position loss on the 400 test windows.
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from configs import build
from data import ROOT, load
from evaluate import seqs_of
from export_hw import export, make_images, write_pkg
from ir import FloatRunner, attach_rope, compile_model
from quant import IntRunner, calibrate, nll, quantize

RUNS = os.path.join(ROOT, "model", "zoo", "runs")
OUT = os.path.join(ROOT, "build", "zoo", "pipe")
PATH = os.path.join(ROOT, "model", "zoo", "pipeline.json")
VERIF = dict(   # verification engines (grown automatically if a model does not fit)
    names=dict(V=27, T=16, NL=4, C=128, VLEN=1024, KVD=64, TAB_W=32, PROG_DEPTH=64, W_DEPTH=32768,
               PRM_DEPTH=4096, TAB_DEPTH=8192, LUT_DEPTH=2048, N_INSTR=64, OPS_USED=0xFFFE, N_PARAMS=0),
    stories=dict(V=512, T=256, NL=5, C=64, VLEN=2048, KVD=64, TAB_W=32, PROG_DEPTH=128, W_DEPTH=40960,
                 PRM_DEPTH=8192, TAB_DEPTH=65536, LUT_DEPTH=2048, N_INSTR=128, OPS_USED=0xFFFE, N_PARAMS=0))


def verif_engine(ds, dims):
    """path of the verification engine of a dataset, enlarged (and rebuilt) if `dims` does not fit"""
    vdir = os.path.join(ROOT, "build", "zoo", f"verif_{ds}")
    cur = VERIF[ds]
    if os.path.exists(os.path.join(vdir, "dims.json")):
        cur = json.load(open(os.path.join(vdir, "dims.json")))
    need = dict(cur)
    for k, v in dims.items():
        if k in need and k not in ("OPS_USED", "N_PARAMS", "TAB_W") and v > need[k]:
            need[k] = v
    if need != cur or not os.path.exists(os.path.join(vdir, "zoo_pkg.sv")):
        os.makedirs(vdir, exist_ok=True)
        write_pkg(vdir, f"verif_{ds}", need)
        json.dump(need, open(os.path.join(vdir, "dims.json"), "w"), indent=1)
        exp = os.path.join(ROOT, "mem", "zoo", "gpt2", "exp.mem")
        open(os.path.join(vdir, "exp.mem"), "w").write(open(exp).read())
    return vdir, need


def rtl(vdir, imgdir):
    env = dict(os.environ, LOAD="1", MODEL=os.path.relpath(imgdir, ROOT), VJOBS="2")
    out = subprocess.run([os.path.join(ROOT, "scripts", "zoo_sim.sh"), os.path.relpath(vdir, ROOT)],
                         capture_output=True, text=True, cwd=ROOT, env=env)
    txt = out.stdout + out.stderr
    m = re.search(r"RESULT \S+ steps (\d+) bad_steps (\d+) errors (\d+) avg_cycles (\d+) min_cycles (\d+) "
                  r"max_cycles (\d+)", txt)
    if not m:
        return dict(rtl_error=txt[-1500:])
    steps, bad, err, avg, lo, hi = map(int, m.groups())
    prof = {k: int(v) for k, v in re.findall(r"^PROF \S+ (\S+) (\d+)", txt, re.M)}
    return dict(rtl_steps=steps, rtl_errors=err + len(re.findall(r"Assertion failed", txt)), cycles=avg,
                cycles_min=lo, cycles_max=hi, profile={k: round(v / steps, 1) for k, v in prof.items()})


def process(run, d_names):
    spec, tag = run["spec"], run["tag"]
    ds = "stories" if run["data"] == "stories" else "names"
    m = build(spec)
    m.load_state_dict(torch.load(run["path"] + ".pt", map_location="cpu"))
    m.eval()
    prog = attach_rope(compile_model(run["arch"], m))
    res = dict(arch=run["arch"], tag=tag, seed=run["seed"], lr=run["lr"], params=run["params"],
               test_loss=run["test_loss"], val_loss=run["val_loss"])
    if ds == "names":
        cal = [x for x, _ in seqs_of(*d_names["train"])[:2000]]
        test = seqs_of(*d_names["test"])
        tv = [x for x, _ in test]
    else:
        from tinystories import Batches, val_windows
        g = Batches("train", spec["kwargs"]["T"], seed=777)
        X, _ = g.get(100)
        cal = [x.tolist() for x in X]
        Xt, Yt = val_windows(spec["kwargs"]["T"], "test")
        test = [(Xt[i].tolist(), Yt[i].tolist()) for i in range(64)]
        tv = [Xt[i].tolist() for i in range(4)]
        with torch.no_grad():                                  # float per-position loss, all test windows
            lp = []
            for i in range(0, len(Xt), 50):
                lg = m(Xt[i:i + 50])
                lp.append(F.cross_entropy(lg.transpose(1, 2), Yt[i:i + 50], reduction="none").mean(0))
            res["pos_loss"] = [round(float(v), 4) for v in torch.stack(lp).mean(0)]
    rec = calibrate(prog, cal)
    q = quantize(prog, rec)
    tf = ti = n = ag = 0
    for x, y in test:
        fr, ir = FloatRunner(prog), IntRunner(q)
        for t, (a, b) in enumerate(zip(x, y)):
            if b < 0:
                break
            lf, li = fr.step(int(a), t), ir.step(int(a), t) * q.s_logit
            tf, ti, ag, n = tf + nll(lf, b), ti + nll(li, b), ag + int(np.argmax(lf) == np.argmax(li)), n + 1
    res.update(float_nll=tf / n, int_nll=ti / n, int_delta=(ti - tf) / n, argmax_agree=ag / n, int_tokens=n)
    name = os.path.basename(run["path"])
    edir = os.path.join(OUT, tag, name)
    export(name, tv, q=q, out=edir, quiet=True)
    dims = json.load(open(os.path.join(edir, "dims.json")))
    res["dims"] = dims
    vdir, vd = verif_engine(ds, dims)
    make_images(edir, os.path.join(edir, "u"), dims, vd["TAB_W"], vd["T"])
    r = rtl(vdir, os.path.join(edir, "u"))
    res.update(r)
    res["rtl_logits"] = r.get("rtl_steps", 0) * dims["V"]
    return res


def main():
    tags = sys.argv[1:] or sorted(os.listdir(RUNS))
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    d = load()
    for tag in tags:
        if not os.path.isdir(os.path.join(RUNS, tag)):
            continue
        for f in sorted(glob.glob(os.path.join(RUNS, tag, "*.json"))):
            key = f"{tag}/{os.path.basename(f)[:-5]}"
            if key in res and "rtl_errors" in res[key]:
                continue
            run = json.load(open(f))
            if not os.path.exists(run["path"] + ".pt"):
                continue
            try:
                res[key] = process(run, d)
            except AssertionError as e:
                res[key] = dict(error=f"compiler rejected: {e}")
            r = res[key]
            print(f"{key:42s} test {r.get('test_loss', 0):.4f} int8 Δ {r.get('int_delta', float('nan')):+.4f} "
                  f"rtl {r.get('rtl_steps', '-')} steps {r.get('rtl_errors', '?')} errors "
                  f"{r.get('cycles', '-')} cycles {r.get('error', '')}{r.get('rtl_error', '')[:200]}", flush=True)
            json.dump(res, open(PATH, "w"), indent=1)


if __name__ == "__main__":
    main()
