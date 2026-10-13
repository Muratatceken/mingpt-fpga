"""
Closed-form cycle model of the zoo engine, derived from the RTL state machines (rtl/zoo/zoo_core.sv):
the cycles of the token at position t follow from the microcode alone, without simulation.
  python3 perf_model.py [memdir ...]       compare with the measured cycles (hw_metrics.json, pipeline.json, dse.json)

Per step: 2 cycles of the start/done handshake; per instruction 2 cycles fetch/decode (+1 for the final step), and
  table streaming   N + 2      (N words; EMB16 with positions: 2N + 2; HRESET only at t = 0)
  GEMV              n_in + (B-1) max(n_in, P+2) + P + 5 (+1 activation table),   B = ceil(n_out / P)
  NORM              per segment of n elements: 2n + 73   (n + 1 + 1 + 29 isqrt + 42 divide + n)
  ATTN              per head: 2(t+1) + 30               (scores, exp/accumulate, 19 reciprocal, 8 outputs, 3)
  ROPE              per head: 2 + 3 d_h/2               KVSTORE: KV words;  MUL, REQ: n + 3;  COPY, GRUUPD: n;  BOW: n + 3
"""
import glob
import json
import os
import sys

from data import ROOT

NAMES = {0: "END", 1: "EMB16", 2: "EMB8", 3: "EMBWIN", 4: "TABLOGIT", 5: "HRESET", 6: "NORM", 7: "GEMV", 8: "ROPE",
         9: "KVSTORE", 10: "ATTN", 11: "MUL", 12: "GRUUPD", 13: "COPY", 14: "REQ", 15: "BOW"}


def field(w, off, n):
    return (w >> off) & ((1 << n) - 1)


def program(memdir):
    pkg = open(os.path.join(memdir, "zoo_pkg.sv")).read()
    P = int(pkg.split("localparam int    P          = ")[1].split(";")[0])
    words = [int(x, 16) for x in open(os.path.join(memdir, "prog.mem")).read().split()]
    prog = []
    for w in words:
        op = field(w, 0, 8)
        prog.append(dict(op=NAMES[op], mode=field(w, 20, 4), n0=field(w, 32, 16), n1=field(w, 48, 16),
                         n2=field(w, 64, 16), n3=field(w, 80, 16)))
        if op == 0:
            break
    return prog, P


def cycles(prog, P, t):
    """cycles of the token at position t, per operation (FETCH = fetch/decode overhead)"""
    c = {"FETCH": 2 * len(prog) + 1, "IO": 2}         # IO: start accepted, done observed (step interface)
    add = lambda k, v: c.__setitem__(k, c.get(k, 0) + v)
    for o in prog:
        op, n0, n1, n2 = o["op"], o["n0"], o["n1"], o["n2"]
        if op == "END":
            continue
        if op in ("EMB8", "TABLOGIT"):
            add(op, n0 + 2)
        elif op == "EMB16":
            add(op, (2 if o["mode"] & 1 else 1) * n0 + 2)
        elif op == "EMBWIN":
            add(op, n0 * n1 + 2)
        elif op == "HRESET":
            if t == 0:
                add(op, n0 + 2)
        elif op == "GEMV":
            nin, blocks = n0 + n1, n2 // P
            add(op, nin + (blocks - 1) * max(nin, P + 2) + P + 5 + (1 if o["mode"] == 2 else 0))
        elif op == "NORM":
            add(op, n1 * (2 * n0 + 73))
        elif op == "ATTN":
            add(op, n0 * (2 * (t + 1) + 30))
        elif op == "ROPE":
            add(op, n0 * (2 + 3 * (n1 // 2)))
        elif op == "KVSTORE":
            add(op, n0 // 8)
        elif op in ("MUL", "REQ", "BOW"):
            add(op, n0 + 3)
        elif op in ("COPY", "GRUUPD"):
            add(op, n0)
    return c


def positions(memdir):
    """positions of the teacher-forced steps of tv.txt (to average like the testbench)"""
    tv = open(os.path.join(memdir, "tv.txt")).read().split("\n")
    nseq = int(tv[0].split()[0])
    pos, i = [], 1
    for _ in range(nseq):
        n = int(tv[i])
        pos += list(range(n))
        i += 1 + n
    return pos


def predict(memdir):
    prog, P = program(memdir)
    pos = positions(memdir)
    tot, per_op = 0, {}
    for t in pos:
        c = cycles(prog, P, t)
        tot += sum(c.values())
        for k, v in c.items():
            per_op[k] = per_op.get(k, 0) + v
    return tot / len(pos), {k: v / len(pos) for k, v in per_op.items()}


def main():
    hw = json.load(open(os.path.join(ROOT, "model", "zoo", "hw_metrics.json")))
    rows = []
    dirs = sys.argv[1:] or [os.path.join(ROOT, "mem", "zoo", m) for m in hw if m != "universal"]
    for d in dirs:
        m = os.path.basename(d.rstrip("/"))
        if m not in hw or not os.path.exists(os.path.join(d, "prog.mem")):
            continue
        pred, ops = predict(d)
        meas = hw[m]["cycles_avg"]
        rows.append((m, pred, meas))
        bad = {k: (round(v, 1), hw[m]["profile"].get(k)) for k, v in ops.items()
               if k != "IO" and abs(v - hw[m]["profile"].get(k, 0)) > 0.5}
        print(f"{m:12s} model {pred:9.1f}  measured {meas:7d}  diff {pred - meas:+7.1f}  {bad if bad else ''}")
    # design-space points (different P): the engine's GEMV stall rule
    dse = json.load(open(os.path.join(ROOT, "model", "zoo", "dse.json"))) if os.path.exists(
        os.path.join(ROOT, "model", "zoo", "dse.json")) else {}
    for k, e in sorted(dse.items()):
        d = os.path.join(ROOT, "build", "zoo", "dse", f"P{e['P']}", e["model"])
        if os.path.exists(os.path.join(d, "prog.mem")):
            pred, _ = predict(d)
            print(f"DSE {k:16s} model {pred:9.1f}  measured {e['cycles']:7d}  diff {pred - e['cycles']:+7.1f}")
    return rows


if __name__ == "__main__":
    main()
