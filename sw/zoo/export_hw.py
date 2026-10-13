"""
Export an integer program (build/zoo/<name>.qprog.pkl) for the microcoded RTL (rtl/zoo/).
  python3 export_hw.py [model ...] [--nseq N]
Writes mem/zoo/<name>/: prog.mem weights.mem params.mem tables.mem luts.mem exp.mem tv.txt zoo_pkg.sv

Instruction word (256 bit)                         Buffers: A..G=0..6, Q=7, K=8, Vb=9, HS=10, R=14, L=15
  [7:0] opcode  [11:8] dst  [15:12] src0  [19:16] src1  [23:20] mode  [31:24] aux8
  [47:32] n0  [63:48] n1  [79:64] n2  [95:80] n3  [127:96] base0  [159:128] base1  [223:160] k64  [255:224] e32
"""
import json
import math
import os
import pickle
import sys

import numpy as np

from data import ROOT, load
from evaluate import seqs_of
from quant import EXP_LUT, IntRunner

P = int(os.environ.get("ZOO_P", 8))     # GEMV lanes (8, 16, 32); ZOO_P=16 python3 export_hw.py ...
DH = 8                                   # attention / KV lanes = head size
OPC = dict(END=0, EMB16=1, EMB8=2, EMBWIN=3, TABLOGIT=4, HRESET=5, NORM=6, GEMV=7, ROPE=8,
           KVSTORE=9, ATTN=10, MUL=11, GRUUPD=12, COPY=13, REQ=14, BOW=15)
BUF = dict(A=0, B=1, C=2, D=3, E=4, F=5, G=6, Q=7, K=8, Vb=9, HS=10, R=14, L=15)
MODE = dict(o8=0, radd=1, lut=2, logit=3)


def word(**f):
    lay = dict(op=(0, 8), dst=(8, 4), src0=(12, 4), src1=(16, 4), mode=(20, 4), aux8=(24, 8),
               n0=(32, 16), n1=(48, 16), n2=(64, 16), n3=(80, 16), base0=(96, 32), base1=(128, 32),
               k64=(160, 64), e32=(224, 32))
    w = 0
    for k, v in f.items():
        off, bits = lay[k]
        v = int(v)
        assert 0 <= v < (1 << bits), (k, v)
        w |= v << off
    return w


def hexw(v, bits):
    return format(v & ((1 << bits) - 1), "0{}x".format(bits // 4))


class Tables:
    """32-bit table ROM with named regions."""

    def __init__(self):
        self.data, self.base = [], {}

    def put(self, key, values):
        if key not in self.base:
            self.base[key] = len(self.data)
            self.data += [int(v) for v in np.asarray(values).reshape(-1)]
        return self.base[key]


def export(name, seqs, gen=0, gen_start=0, q=None, out=None, quiet=False):
    """seqs: input token sequences for the teacher-forced golden vectors; gen: greedy steps for gen.txt."""
    if q is None:
        q = pickle.load(open(os.path.join(ROOT, "build", "zoo", f"{name}.qprog.pkl"), "rb"))
    out = out or os.path.join(ROOT, "mem", "zoo", name)            # MEM_PATH is relative to the repo root
    os.makedirs(out, exist_ok=True)
    tabs = Tables()
    luts = []
    wlines, plines, prog = [], [], []
    vlen, kvd, C = 1, 1, 1
    lin_seen = {}

    def lin_rom(key):
        if key in lin_seen:
            return lin_seen[key]
        L = q.lin[key]
        W, b, m0, sh = L["W"], L["b"], L["m0"], L["sh"]
        n_out, n_in = W.shape
        n_pad = -(-n_out // P) * P
        wb, pb = len(wlines), len(plines)
        Wp = np.zeros((n_pad, n_in), dtype=np.int64)
        Wp[:n_out] = W
        for ob in range(n_pad // P):
            for i in range(n_in):
                wd = 0
                for p in range(P):
                    wd |= (int(Wp[ob * P + p, i]) & 0xFF) << (8 * p)
                wlines.append(hexw(wd, 8 * P))
        for o in range(n_pad):
            if o < n_out:
                bb, mm, ss = int(b[o]), int(m0[o]), int(sh[o])
            else:
                bb, mm, ss = 0, 1 << 15, 16
            plines.append(hexw(((bb & 0xFFFFFFFF) << 32) | (mm << 16) | ss, 64))
        lin_seen[key] = (wb, pb, n_in, n_out, n_pad)
        return lin_seen[key]

    for i, o in enumerate(q.ops):
        op = o["op"]
        if op == "EMB16":
            C = q.tab[o["tok"]].shape[1]
            b0 = tabs.put(o["tok"], q.tab[o["tok"]])
            b1 = tabs.put(o["pos"], q.tab[o["pos"]]) if o.get("pos") else 0
            prog.append(word(op=OPC[op], dst=BUF["R"], mode=int(bool(o.get("pos"))), n0=C, base0=b0, base1=b1))
        elif op == "EMB8":
            tb = q.tab[o["table"]]
            vlen = max(vlen, tb.shape[1])
            prog.append(word(op=OPC[op], dst=BUF[o["dst"]], n0=tb.shape[1], base0=tabs.put(o["table"], tb)))
        elif op == "EMBWIN":
            tb = q.tab[o["table"]]
            vlen = max(vlen, tb.shape[1] * o["k"])
            prog.append(word(op=OPC[op], dst=BUF[o["dst"]], n0=tb.shape[1], n1=o["k"], n2=tb.shape[0] - 1,
                             base0=tabs.put(o["table"], tb)))
        elif op == "TABLOGIT":
            tb = q.tab[o["table"]]
            prog.append(word(op=OPC[op], dst=BUF["L"], n0=tb.shape[1], base0=tabs.put(o["table"], tb)))
        elif op == "HRESET":
            tb = q.tab[o["table"]]
            vlen = max(vlen, len(tb))
            prog.append(word(op=OPC[op], dst=BUF["HS"], n0=len(tb), base0=tabs.put(o["table"], tb)))
        elif op == "NORM":
            src16, dst16 = o["src"] == "R", o["dst"] == "R"
            mode = int(o["kind"] == "ln") | (int(src16) << 1) | (int(dst16) << 2)
            src_len = C if src16 else None
            n = o["n"]
            nseg = (src_len // n) if src16 else _buflen_of(q, o, n)
            assert o["E"] < 2 ** 32 and o["K"] < 2 ** 40 and n < 256   # RTL widths: K << 19 < 2^64, 1/sqrt < 2^40
            prog.append(word(op=OPC[op], dst=BUF[o["dst"]], src0=BUF[o["src"]], mode=mode, aux8=o["shift"],
                             n0=n, n1=nseg, k64=o["K"], e32=o["E"]))
        elif op == "GEMV":
            wb, pb, n_in, n_out, n_pad = lin_rom(o["lin"])
            srcs = o["srcs"]
            if len(srcs) == 1:
                n0, n1, s1 = n_in, 0, 0
            else:
                n1 = len(q.tab["h0"])
                n0, s1 = n_in - n1, BUF[srcs[1]]
            assert o["dst"] not in srcs and n_in >= 2
            lut_id = 0
            if o["mode"] == "lut":
                lut_id = len(luts)
                lut = q.lut[i]
                luts.append([int(lut[((a ^ 0x80) & 0xFF)]) for a in range(256)])
            if o["mode"] in ("o8", "lut"):
                vlen = max(vlen, n_out)
            prog.append(word(op=OPC[op], dst=BUF[o["dst"]] if o["dst"] in BUF else 0, src0=BUF[srcs[0]], src1=s1,
                             mode=MODE[o["mode"]], aux8=lut_id, n0=n0, n1=n1, n2=n_pad, n3=n_out, base0=wb, base1=pb))
            vlen = max(vlen, n0)
        elif op == "ROPE":
            dh = o["dh"]
            assert dh == DH, "RoPE/attention unit assumes head size 8"
            rc, rs = q.rope
            prog.append(word(op=OPC[op], src0=BUF[o["buf"]], dst=BUF[o["buf"]], mode=int(o["style"] == "half"),
                             n0=o["nh"], n1=dh, base0=tabs.put("rope_cos", rc), base1=tabs.put("rope_sin", rs)))
        elif op == "KVSTORE":
            n = _kvdim(q, o)
            assert n % DH == 0
            kvd = max(kvd, n)
            prog.append(word(op=OPC[op], src0=BUF[o["k"]], src1=BUF[o["v"]], aux8=o["layer"], n0=n))
        elif op == "ATTN":
            assert o["dh"] == DH and o["FEXP"] < 64
            prog.append(word(op=OPC[op], src0=BUF[o["q"]], dst=BUF[o["dst"]], aux8=o["layer"], n0=o["nh"],
                             n1=o["nkv"], n2=o["dh"], n3=o["nh"] // o["nkv"], base1=o["CEXP"], k64=o["FEXP"]))
        elif op == "MUL":
            n = _len_of(q, o["a"])
            prog.append(word(op=OPC[op], src0=BUF[o["a"]], src1=BUF[o["b"]], dst=BUF[o["dst"]], n0=n,
                             k64=(o["m0"] << 8) | o["sh"]))
        elif op == "GRUUPD":
            n = len(q.tab["h0"])
            prog.append(word(op=OPC[op], src0=BUF[o["z"]], src1=BUF[o["hb"]], dst=BUF["HS"], n0=n))
        elif op == "COPY":
            n = len(q.tab["h0"]) if o["dst"] == "HS" else None
            prog.append(word(op=OPC[op], src0=BUF[o["src"]], dst=BUF[o["dst"]], n0=n))
        elif op == "REQ":
            prog.append(word(op=OPC[op], src0=BUF["R"], dst=BUF[o["dst"]], n0=C, k64=(o["m0"] << 8) | o["sh"]))
        elif op == "BOW":
            hl = [x for r in q.rcp for x in (int(r) >> 16, ((int(r) & 0xFFFF) ^ 0x8000) - 0x8000)]  # 16-bit halves
            prog.append(word(op=OPC[op], dst=BUF["R"], n0=C, base0=tabs.put("rcp", hl)))
        else:
            raise ValueError(op)
    prog.append(word(op=OPC["END"]))
    if not luts:
        luts.append([0] * 256)
    vlen = max(vlen, C)
    tab_w = 16 if all(-32768 <= v < 32768 for v in tabs.data) else 32

    def w(fname, lines, depth=None):
        lines = list(lines)
        while len(lines) < 2:                      # every ROM has at least two words
            lines.append(lines[0] if lines else "0")
        with open(os.path.join(out, fname), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return len(lines)

    d_prog = w("prog.mem", [hexw(x, 256) for x in prog])
    d_w = w("weights.mem", wlines or [hexw(0, 8 * P)])
    d_p = w("params.mem", plines or [hexw(0, 64)])
    d_t = w("tables.mem", [hexw(x, tab_w) for x in tabs.data] or [hexw(0, tab_w)])
    d_l = w("luts.mem", [hexw(v, 8) for lut in luts for v in lut])
    w("exp.mem", [hexw(int(v), 16) for v in EXP_LUT])

    # golden vectors: teacher-forced test sequences, every logit of every step
    lines = [f"{len(seqs)} {q.V}"]
    nsteps = 0
    for x in seqs:
        ir = IntRunner(q)
        lines.append(str(len(x)))
        for t, tok in enumerate(x):
            lg = ir.step(int(tok), t)
            lines.append(" ".join(map(str, [tok] + [int(v) for v in lg])))
            nsteps += 1
    w("tv.txt", lines)
    if gen:                                    # greedy generation from gen_start (closed loop in tb_zoo +GEN)
        ir, toks = IntRunner(q), [gen_start]
        for t in range(gen):
            toks.append(int(np.argmax(ir.step(toks[-1], t))))
        w("gen.txt", [f"{gen} {gen_start}", " ".join(map(str, toks[1:]))])

    dims = dict(V=q.V, T=q.T, NL=max(1, q.n_layer), C=C, VLEN=vlen, KVD=kvd, TAB_W=tab_w, PROG_DEPTH=d_prog,
                W_DEPTH=d_w, PRM_DEPTH=d_p, TAB_DEPTH=d_t, LUT_DEPTH=d_l, N_INSTR=len(prog),
                OPS_USED=sum(1 << op for op in sorted({x & 0xFF for x in prog}) if op),
                N_PARAMS=sum(L["W"].size + L["b"].size for L in q.lin.values()) + sum(q.tab[k].size for k in q.tab))
    json.dump(dims, open(os.path.join(out, "dims.json"), "w"), indent=1)
    write_pkg(out, name, dims)
    if not quiet:
        print(f"{name:9s} prog {len(prog):3d} instr | weights {len(wlines):5d}x{8 * P}b | params {len(plines):4d} | "
              f"tables {len(tabs.data):5d}x{tab_w}b | luts {len(luts)} | VLEN {vlen} KVD {kvd} | tv {len(seqs)} seq / {nsteps} steps")


def _len_of(q, buf):
    """length of a buffer produced earlier in the program."""
    for o in q.ops:
        if o["op"] == "GEMV" and o["dst"] == buf:
            return q.lin[o["lin"]]["W"].shape[0]
    raise ValueError(buf)


def _kvdim(q, o):
    return _len_of(q, o["k"])


def _buflen_of(q, o, n):
    return _len_of(q, o["src"]) // n


def write_pkg(out, name, d):
    aw = lambda n: max(1, math.ceil(math.log2(n)))
    vw, nkvw = -(-d["VLEN"] // P), -(-d["KVD"] // DH)
    kdep = d["NL"] * d["T"] * nkvw
    pkg = f"""// GENERATED by sw/zoo/export_hw.py for model '{name}' -- do not edit.
package zoo_pkg;
  localparam string NAME       = "{name}";
  localparam string MEM_PATH   = "{os.path.relpath(out, ROOT)}";
  localparam int    P          = {P};
  localparam int    V          = {d["V"]};
  localparam int    T          = {d["T"]};
  localparam int    NL         = {d["NL"]};
  localparam int    C          = {d["C"]};
  localparam int    NBUF       = 11;
  localparam int    VLEN       = {d["VLEN"]};
  localparam int    VW         = {vw};
  localparam int    VA_W       = {aw(11 * vw)};
  localparam int    KVD        = {d["KVD"]};
  localparam int    NKVW       = {nkvw};
  localparam int    KDEP       = {kdep};
  localparam int    KA_W       = {aw(kdep)};
  localparam int    TAB_W      = {d["TAB_W"]};
  localparam int    PROG_DEPTH = {d["PROG_DEPTH"]};
  localparam int    W_DEPTH    = {d["W_DEPTH"]};
  localparam int    PRM_DEPTH  = {d["PRM_DEPTH"]};
  localparam int    TAB_DEPTH  = {d["TAB_DEPTH"]};
  localparam int    LUT_DEPTH  = {d["LUT_DEPTH"]};
  localparam int    PC_W       = {aw(d["PROG_DEPTH"])};
  localparam int    WA_W       = {aw(d["W_DEPTH"])};
  localparam int    PA_W       = {aw(d["PRM_DEPTH"])};
  localparam int    TA_W       = {aw(d["TAB_DEPTH"])};
  localparam int    LUA_W      = {aw(d["LUT_DEPTH"])};
  localparam int    TOK_W      = {aw(d["V"] + 1)};
  localparam int    POS_W      = {aw(d["T"])};
  localparam int    LA_W       = {aw(d["V"])};
  localparam int    RA_W       = {aw(d["C"])};
  localparam int    N_INSTR    = {d["N_INSTR"]};
  localparam logic [15:0] OPS_USED = 16'h{d["OPS_USED"]:04x};
  localparam int    N_PARAMS   = {d["N_PARAMS"]};
endpackage
"""
    with open(os.path.join(out, "zoo_pkg.sv"), "w") as fh:
        fh.write(pkg)


def make_images(src, dst, d, tab_w, T):
    """images of one exported model for an engine with wider tables (tab_w) and context T: loaded at run time
    through ld_* (sizes.txt gives the number of words), golden vectors cut to T positions"""
    os.makedirs(dst, exist_ok=True)
    for f in ("prog.mem", "weights.mem", "params.mem", "luts.mem"):
        open(os.path.join(dst, f), "w").write(open(os.path.join(src, f)).read())
    tw = d["TAB_W"]
    vals = [int(x, 16) for x in open(os.path.join(src, "tables.mem")).read().split()]
    vals = [v - (1 << tw) if v >= 1 << (tw - 1) else v for v in vals]              # sign
    open(os.path.join(dst, "tables.mem"), "w").write("\n".join(hexw(v, tab_w) for v in vals) + "\n")
    open(os.path.join(dst, "sizes.txt"), "w").write(
        f"{d['PROG_DEPTH']} {d['W_DEPTH']} {d['PRM_DEPTH']} {d['TAB_DEPTH']} {d['LUT_DEPTH']}\n")
    tv = open(os.path.join(src, "tv.txt")).read().split("\n")
    nseq, v = map(int, tv[0].split())
    lines, i = [tv[0]], 1
    for _ in range(nseq):
        n = int(tv[i])
        k = min(n, T)
        lines += [str(k)] + tv[i + 1:i + 1 + k]
        i += 1 + n
    open(os.path.join(dst, "tv.txt"), "w").write("\n".join(lines) + "\n")
    if os.path.exists(os.path.join(src, "gen.txt")):
        g = open(os.path.join(src, "gen.txt")).read().split("\n")
        n, start = map(int, g[0].split())
        k = min(n, T)
        open(os.path.join(dst, "gen.txt"), "w").write(f"{k} {start}\n" + " ".join(g[1].split()[:k]) + "\n")


def universal(models, T=256, out=None):
    """One engine for many models (one FPGA bitstream): sizes = maximum over the models, every unit enabled,
    context capped at T. Every model gets an image directory <out>/<model>/ that is loaded at run time
    through the ld_* port (tables re-emitted at the common width, golden vectors cut to T positions)."""
    out = out or os.path.join(ROOT, "mem", "zoo", "universal")
    os.makedirs(out, exist_ok=True)
    dims = {m: json.load(open(os.path.join(ROOT, "mem", "zoo", m, "dims.json"))) for m in models}
    u = {k: max(d[k] for d in dims.values()) for k in dims[models[0]] if k not in ("N_PARAMS", "OPS_USED")}
    u.update(T=min(u["T"], T), OPS_USED=0xFFFE, N_PARAMS=max(d["N_PARAMS"] for d in dims.values()))
    write_pkg(out, "universal", u)
    json.dump(dict(u, models=models), open(os.path.join(out, "dims.json"), "w"), indent=1)
    src0 = os.path.join(ROOT, "mem", "zoo", models[0], "exp.mem")
    open(os.path.join(out, "exp.mem"), "w").write(open(src0).read())
    for m in models:
        make_images(os.path.join(ROOT, "mem", "zoo", m), os.path.join(out, m), dims[m], u["TAB_W"], u["T"])
    print(f"universal engine for {len(models)} models: " + ", ".join(f"{k}={u[k]}" for k in
          ("V", "T", "NL", "C", "VLEN", "KVD", "TAB_W", "W_DEPTH", "TAB_DEPTH", "PRM_DEPTH", "LUT_DEPTH")))
    return u


def main():
    args = sys.argv[1:]
    nseq = None
    if "--nseq" in args:
        k = args.index("--nseq")
        nseq = int(args[k + 1])
        args = args[:k] + args[k + 2:]
    if "--universal" in args:                  # python3 export_hw.py --universal m1 m2 ... [--T 256]
        args.remove("--universal")
        T = 256
        if "--T" in args:
            k = args.index("--T")
            T = int(args[k + 1])
            args = args[:k] + args[k + 2:]
        universal(args, T=T)
        return
    root = None
    if "--out-root" in args:                   # write <out-root>/<model>/ instead of mem/zoo/<model>/
        k = args.index("--out-root")
        root = os.path.join(ROOT, args[k + 1])
        args = args[:k] + args[k + 2:]
    d = load()
    names = args or [f[:-10] for f in sorted(os.listdir(os.path.join(ROOT, "build", "zoo"))) if f.endswith(".qprog.pkl")]
    for name in names:
        if name == "stories260k":              # TinyStories: 2 test stories + a 200-token greedy story
            from stories import BOS, split
            export(name, [s[:-1] for s in split()[1][:nseq or 2]], gen=200, gen_start=BOS,
                   out=root and os.path.join(root, name))
        else:                                  # names.txt: 50 test names
            export(name, [x for x, _ in seqs_of(*d["test"])[:nseq or 50]], out=root and os.path.join(root, name))


if __name__ == "__main__":
    main()
