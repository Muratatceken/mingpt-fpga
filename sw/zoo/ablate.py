"""
Where does the INT8 loss come from? Fake-quantisation ablation on the float IR:
quantise one tensor category at a time (on the same grids the integer program uses)
and measure the test NLL.
  python3 ablate.py <model|stories260k> [--n N]
Also compares calibration methods on the real integer program (quant.CALIB variants).
Results: model/zoo/quant_study.json
Categories: weights (per-row INT8), o8 (q/k/v/w3 outputs), act (LUT input+output),
mul (SwiGLU / GRU products), norm (normalised INT8 vectors), resid (INT16 residual),
logit (INT16 logits)
"""
import json
import math
import os
import sys

import numpy as np

import quant

import ir
from ir import ACTS, FloatRunner, attach_rope
from data import ROOT
from quant import IntRunner, calibrate, nll, quantize

CATS = ["weights", "o8", "act", "mul", "norm", "resid", "logit"]


def fq(x, s, bits=8):
    lim = (1 << (bits - 1)) - 1
    return np.clip(np.round(x / s), -lim - 1, lim) * s


def make_hooks(prog, rec, cats):
    """Monkey-patch helpers: return patched (ACTS, linear table) for the chosen categories."""
    p2 = attach_rope(prog)
    lin = dict(p2.lin)
    if "weights" in cats:
        for k, (W, b) in lin.items():
            s = np.abs(W).max(axis=1, keepdims=True) / 127.0
            s[s == 0] = 1.0
            lin[k] = (np.round(W / s) * s, b)
    return lin


def run(prog, rec, cats, seqs):
    """Float IR with fake-quant at the integer program's quantisation points."""
    q8 = lambda key, x: fq(x, rec[key] / 127.0) if rec.get(key, 0) > 0 else x
    sR = 2.0 ** -math.floor(math.log2(32767.0 / rec[("R",)]))
    sL = rec[("L",)] / 32767.0
    lin = make_hooks(prog, rec, cats)
    tot = n = 0
    for s in seqs:
        K, Vc = {}, {}
        for t, (tok, tgt) in enumerate(zip(s[:-1], s[1:])):
            buf = {}
            for i, o in enumerate(prog.ops):
                op = o["op"]
                if op == "EMB16":
                    x = prog.tab[o["tok"]][tok].copy()
                    if o.get("pos"):
                        x = x + prog.tab[o["pos"]][t]
                    buf["R"] = fq(x, sR, 16) if "resid" in cats else x
                elif op == "NORM":
                    x, nn = buf[o["src"]], o["n"]
                    y = np.empty_like(x)
                    for s0 in range(0, len(x), nn):
                        seg = x[s0:s0 + nn]
                        if o["kind"] == "ln":
                            seg = seg - seg.mean()
                        y[s0:s0 + nn] = seg / math.sqrt((seg ** 2).mean() + o["eps"])
                    if "norm" in cats and o["dst"] != "R":
                        bound = math.sqrt(nn - 1) if o["kind"] == "ln" else math.sqrt(nn)
                        y = fq(y, 2.0 ** -math.floor(math.log2(127.0 / bound)))
                    buf[o["dst"]] = y
                elif op == "GEMV":
                    W, b = lin[o["lin"]]
                    y = W @ np.concatenate([buf[x] for x in o["srcs"]]) + b
                    if o["mode"] == "o8":
                        key = (i, "pair") if rec.get((i, "pair"), 0) > 0 else (i, "out")
                        buf[o["dst"]] = q8(key, y) if "o8" in cats else y
                    elif o["mode"] == "radd":
                        buf["R"] = buf["R"] + y
                        if "resid" in cats:
                            buf["R"] = fq(buf["R"], sR, 16)
                    elif o["mode"] == "lut":
                        if "act" in cats:
                            y = q8((i, "pre"), y)
                        a = ACTS[o["act"]](y)
                        if "act" in cats:
                            a = q8((i, "post"), a)
                        buf[o["dst"]] = a
                    else:
                        buf["L"] = fq(y, sL, 16) if "logit" in cats else y
                elif op == "ROPE":
                    buf[o["buf"]] = ir.rope_apply(buf[o["buf"]], o["nh"], o["dh"], o["style"],
                                                  prog.rope[0][t], prog.rope[1][t])
                elif op == "KVSTORE":
                    K.setdefault(o["layer"], []).append(buf[o["k"]].copy())
                    Vc.setdefault(o["layer"], []).append(buf[o["v"]].copy())
                elif op == "ATTN":
                    l, nh, nkv, dh = o["layer"], o["nh"], o["nkv"], o["dh"]
                    Kl, Vl, qv = np.array(K[l]), np.array(Vc[l]), buf[o["q"]]
                    y = np.zeros(nh * dh)
                    for h in range(nh):
                        kv = h // (nh // nkv)
                        sc = o["alpha"] * (Kl[:, kv * dh:(kv + 1) * dh] @ qv[h * dh:(h + 1) * dh])
                        w = np.exp(sc - sc.max())
                        y[h * dh:(h + 1) * dh] = (w / w.sum()) @ Vl[:, kv * dh:(kv + 1) * dh]
                    buf[o["dst"]] = y
                elif op == "MUL":
                    y = buf[o["a"]] * buf[o["b"]]
                    buf[o["dst"]] = q8((i, "out"), y) if "mul" in cats else y
                elif op == "REQ":
                    buf[o["dst"]] = buf[o["src"]]
                else:
                    raise NotImplementedError(op)
            tot += nll(buf["L"], tgt)
            n += 1
    return tot / n


def int_nll(prog, q, seqs):
    tf = ti = n = ag = 0
    for s in seqs:
        f, ir_ = FloatRunner(prog), IntRunner(q)
        for t, (a, b) in enumerate(zip(s[:-1], s[1:])):
            lf, li = f.step(a, t), ir_.step(a, t) * q.s_logit
            tf, ti, ag, n = tf + nll(lf, b), ti + nll(li, b), ag + int(np.argmax(lf) == np.argmax(li)), n + 1
    return dict(float_nll=tf / n, int_nll=ti / n, delta=(ti - tf) / n, argmax_agree=ag / n)


def main():
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 20
    name = args[0]
    if name == "stories260k":
        from stories import load_model, split
        from ir import compile_llama
        m = load_model()
        prog = attach_rope(compile_llama(m))
        cal, test = split()
        cal = [s[:-1] for s in cal]
        seqs = test[:n]
    else:
        from check_ir import load_trained
        from data import load
        from evaluate import N_CALIB, seqs_of
        from ir import compile_model
        d = load()
        prog = attach_rope(compile_model(name, load_trained(name)))
        cal = [x for x, _ in seqs_of(*d["train"])[:N_CALIB]]
        seqs = [list(x) + [y[-1]] for x, y in seqs_of(*d["test"])[:n * 10]]
    res = dict(n_seqs=len(seqs))

    # 1. fake-quant ablation with max calibration and bound-based NORM scales (the original method)
    rec = calibrate(prog, cal, pct=0)
    base = run(prog, rec, [], seqs)
    res["ablation"] = {"float": base}
    print(f"{name}: float {base:.4f}")
    for c in CATS + ["all"]:
        v = run(prog, rec, CATS if c == "all" else [c], seqs)
        res["ablation"][c] = v
        print(f"  {c:8s} {v:.4f}  Δ {v - base:+.4f}", flush=True)

    # 2. calibration methods on the bit-exact integer program
    keys = {k: v for k, v in rec.items() if len(k) == 2 and k[1] in ("out", "pre", "post", "pair")}
    fr = FloatRunner(prog)
    fr.calhist = (keys, {})
    for s in cal:
        for t, tok in enumerate(s):
            fr.step(int(tok), t)
    res["calibration"] = []
    saved = dict(quant.CALIB)
    for pct, nm in [(0, "bound"), (0, "cal"), (99.99, "bound"), (99.9, "cal"), (99.99, "cal"), (99.999, "cal")]:
        r = dict(rec)
        if pct:
            for k, h in fr.calhist[1].items():
                c = np.cumsum(h)
                r[k] = min(r[k], keys[k] * (int(np.searchsorted(c, pct / 100.0 * c[-1])) + 1) / len(h))
        quant.CALIB["NORM"] = nm
        e = int_nll(prog, quantize(prog, r), seqs)
        e.update(pct=pct, norm=nm)
        res["calibration"].append(e)
        print(f"  pct {pct:7} norm {nm:5s}: int8 {e['int_nll']:.4f} Δ {e['delta']:+.4f} "
              f"argmax {100 * e['argmax_agree']:.1f}%", flush=True)
    quant.CALIB.update(saved)

    path = os.path.join(ROOT, "model", "zoo", "quant_study.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[name] = res
    json.dump(allres, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
