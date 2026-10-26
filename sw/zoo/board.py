"""
Stand-alone Nexys A7-100T build of one zoo model: the whole language model runs on the FPGA.
Weights, tables and the microcode are BRAM contents of the bitstream; the generation loop, sampling,
detokenisation and the VGA text screen are logic on the chip (board/nexys_a7/). No host computer.

This module is the bit-exact reference of board/nexys_a7/lm_gen.sv and writes its memories:
  python3 board.py export <model> [--T 256] [--mhz 50]      -> mem/nexys/<model>/
  python3 board.py ref <model> [--greedy] [--tau i] [--minp j] [--seed s]
                                                            -> generated tokens and the expected screen
Models: the names models of the zoo (bigram ... nanochat), stories260k (Karpathy's pretrained
TinyStories model) and ts_<arch> (our TinyStories runs of the family, model/zoo/runs/stories260k/).

Generation (identical in lm_gen.sv)
  start token START at position 0; each step: logits = engine(tok, pos), next = pick(logits)
  names   (MULTI=1): STOP (= token 0, '.') ends a name -> newline, the next name starts again at
                     position 0; MAX_SEQ names per run; a name reaching the context T is cut
  stories (MULTI=0): the run ends when STOP (= BOS, as in llama2.c run.c) is sampled or pos = T-1
pick(logits)
  greedy:  first index of the largest logit
  sample:  e_k = 2^-((m - l_k) c)  with c = s_logit log2(e) / tau as (CEXP, FEXP) = qmul(c) and the
           exponential LUT of the attention unit (quant.EXP_LUT): u = (m - l_k) CEXP, n = u >> FEXP,
           e_k = 0 if n >= NCUT (min-p: drops tokens with p < p_max 2^-NCUT) else EXP_LUT[u >> (FEXP-8) & 255] >> n;
           r = xorshift32(state), thr = (r E) >> 32 with E = sum e_k, next = first k with e_0 + ... + e_k > thr
Screen: 80 x 30 characters (640 x 480, 8 x 16 font); row 0 title, rows 1..28 text, row 29 status.
"""
import json
import math
import os
import pickle
import re
import sys

import numpy as np

from data import ROOT
from quant import EXP_LUT, IntRunner, qmul

COLS, ROWS = 80, 28                       # text area: rows 1..28 of the 30-row screen
TAUS = [0.8, 1.0, 0.9, 0.7, 0.6, 0.5, 1.2, 1.5]     # SW[3:1] (all switches down: T = 0.8, min-p 1/32)
NCUTS = [5, 16, 7, 4]                     # SW[5:4]: min-p 1/32, off, 1/128, 1/16
MINP_TXT = ["1/32 ", "off  ", "1/128", "1/16 "]
BUILD = os.path.join(ROOT, "build", "zoo")
OUTROOT = os.path.join(ROOT, "mem", "nexys")
NAMES = ["bigram", "mlp", "rnn", "gru", "bow", "gpt2", "microgpt", "llama", "nanochat"]
ARCH = dict(bigram="Bigram", mlp="MLP", rnn="RNN", gru="GRU", bow="BoW", gpt2="GPT-2", microgpt="microgpt",
            llama="Llama 2", nanochat="nanochat")
# status row: fixed text with fields filled by the hardware (column, width)
STATUS_TXT = " ----- tok | ------- tok/s | ------- cyc/tok | ------------------ | check ---- "
STATUS = dict(zip(["ntok", "toks", "cpt", "mode", "check"],          # (column, width) of the '-' runs
                  [(m.start(), len(m.group())) for m in re.finditer("-+", STATUS_TXT)]))
assert [w for _, w in STATUS.values()] == [5, 7, 7, 18, 4]          # widths fixed in lm_gen.sv


# ----------------------------------------------------------------------------- integer programs
def float_model(name):
    """float PyTorch model of a board model name: <zoo model> | stories260k | ts_<arch>, optionally with
    -<method>: the weights of a low-bit QAT run (model/zoo/lowbit/<base>/<method>.pt, sw/zoo/lowbit.py)"""
    import torch
    base, _, method = name.partition("-")
    if base == "stories260k":
        from stories import load_model
        m = load_model()
    elif base.startswith("ts_"):
        from configs import build
        run = json.load(open(os.path.join(ROOT, "model", "zoo", "runs", "stories260k", f"{base[3:]}_lr0.003_s0.json")))
        m = build(run["spec"])
        m.load_state_dict(torch.load(run["path"] + ".pt", map_location="cpu"))
    else:
        from check_ir import load_trained
        m = load_trained(base)
    if method:
        m.load_state_dict(torch.load(os.path.join(ROOT, "model", "zoo", "lowbit", base, f"{method}.pt"), map_location="cpu"))
    m.eval()
    return m


def calib_seqs(name):
    """calibration inputs of the integer program, as in evaluate.py / stories.py / pipeline.py"""
    base = name.partition("-")[0]
    if base == "stories260k":
        from stories import split
        return [s[:-1] for s in split()[0]]
    if base.startswith("ts_"):
        from tinystories import Batches
        X, _ = Batches("train", 256, seed=777).get(100)
        return [x.tolist() for x in X]
    from evaluate import N_CALIB, seqs_of
    from data import load
    return [x for x, _ in seqs_of(*load()["train"])[:N_CALIB]]


def qprog(name):
    """quantised program of a model (cached in build/zoo/<name>.qprog.pkl)"""
    path = os.path.join(BUILD, f"{name}.qprog.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from ir import attach_rope, compile_model
    from quant import calibrate, quantize
    base = name.partition("-")[0]
    arch = "llama" if base == "stories260k" else base[3:] if base.startswith("ts_") else base
    prog = attach_rope(compile_model(arch, float_model(name)))
    method = name.partition("-")[2]
    if method == "bin":                            # 1-bit layers -> packed binary GEMV (GEMVB) + SCALE
        from ir import factor_binary
        prog = factor_binary(prog)
    elif method.startswith("lb"):                  # LittleBit layers -> SCALE + two GEMVBs
        import torch
        from ir import factor_littlebit
        fac = torch.load(os.path.join(ROOT, "model", "zoo", "lowbit", base, f"{method}.factors.pt"))
        prog = factor_littlebit(prog, fac)
        assert set(prog.littlebit.values()) == set(fac), (sorted(prog.littlebit), sorted(fac))   # every layer placed
    prog.name = name
    q = quantize(prog, calibrate(prog, calib_seqs(name)))
    os.makedirs(BUILD, exist_ok=True)
    pickle.dump(q, open(path, "wb"))
    return q


def cap_context(q, T):
    """the same program with a shorter context: KV cache, token history and position tables shrink"""
    if T >= q.T:
        return q
    q.T = T
    if hasattr(q, "rope"):
        q.rope = (q.rope[0][:T], q.rope[1][:T])
    if hasattr(q, "rcp"):
        q.rcp = q.rcp[:T]
    for o in q.ops:
        if o["op"] == "EMB16" and o.get("pos"):
            q.tab[o["pos"]] = q.tab[o["pos"]][:T]
    return q


# ----------------------------------------------------------------------------- tokens -> characters
ASCII = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
         "…": "...", " ": " ", "é": "e", "ñ": "n"}


def pieces(name, V):
    """display bytes of every token (printable ASCII and '\\n' only)"""
    if V == 27:                                            # names: '.' ends a name -> newline
        return [b"\n"] + [bytes([ord("a") + k]) for k in range(26)]
    from stories import Tokenizer
    out = []
    for k, p in enumerate(Tokenizer().vocab):
        if k < 3:                                          # <unk>, BOS, EOS
            out.append(b"")
        elif p.startswith(b"<0x") and len(p) == 6:         # byte fallback
            b = int(p[3:5], 16)
            out.append(b"\n" if b == 10 else bytes([b]) if 32 <= b < 127 else b"?" if b >= 0xC0 else b"")
        else:
            s = "".join(ASCII.get(ch, ch) for ch in p.decode("utf-8"))
            s = "".join(ch if 32 <= ord(ch) < 127 or ch == "\n" else "?" for ch in s)
            out.append(s.encode())
    return out


# ----------------------------------------------------------------------------- reference generator
def xorshift32(x):
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


def seed_of(sw_seed):
    """32-bit generator state from the 8 seed switches (never 0)"""
    return 0x9E3779B9 ^ (sw_seed << 16) ^ sw_seed


def pick(lg, sample, cexp, fexp, ncut, rng):
    lg = np.asarray(lg, dtype=np.int64)
    m = int(lg.max())
    am = int(np.argmax(lg))
    if not sample:
        return am, rng
    u = (m - lg) * cexp
    n = u >> fexp
    kk = (u >> (fexp - 8)) & 255
    e = np.where(n >= ncut, 0, EXP_LUT[kk] >> np.minimum(n, 15))
    rng = xorshift32(rng)
    thr = (rng * int(e.sum())) >> 32
    return int(np.argmax(np.cumsum(e) > thr)), rng


class Terminal:
    """the text area exactly as lm_gen.sv writes it: a word that does not fit at the end of a row moves to
    the next row (unless it fills the whole row), a space at the end of a row is dropped, scrolling"""

    def __init__(self):
        self.lines = [" " * COLS] * ROWS
        self.row = self.col = self.wstart = 0
        self.sp = True                                     # previous character was a space / row start

    def newline(self):
        """-> index of the row the cursor left (it moved up by one if the screen scrolled)"""
        self.col, self.sp = 0, True
        if self.row == ROWS - 1:
            self.lines = self.lines[1:] + [" " * COLS]
            return ROWS - 2
        self.row += 1
        return self.row - 1

    def write(self, ch):
        ln = self.lines[self.row]
        self.lines[self.row] = ln[:self.col] + chr(ch) + ln[self.col + 1:]
        self.col += 1

    def char(self, ch):
        if ch == 10:
            self.newline()
        elif ch == 32:
            if self.col == COLS:
                self.newline()
            else:
                self.write(ch)
                self.sp = True
        else:
            if self.col == COLS:
                ws = self.wstart if not self.sp else COLS
                old = self.newline()
                if 0 < ws < COLS:                           # move the unfinished word down
                    word = self.lines[old][ws:]
                    self.lines[old] = self.lines[old][:ws] + " " * (COLS - ws)
                    for c in word.encode():
                        self.write(c)
                    self.sp, self.wstart = False, 0
            if self.sp:
                self.wstart = self.col
            self.write(ch)
            self.sp = False

    def put(self, s, strip=False, force_nl=False):
        if s and strip and s[0] == 32:
            s = s[1:]
        for ch in s:
            self.char(ch)
        if force_nl:
            self.newline()

    def screen(self):
        return list(self.lines)


def config(q, name):
    names = q.V == 27
    return dict(START=0 if names else 1, STOP=0 if names else 1, MULTI=int(names), MAX_SEQ=ROWS - 1 if names else 1,
                STRIP=int(not names), STRIP_TOK=1)


def generate(q, name, sample=True, tau=0, minp=0, seed=0, cfg=None, pcs=None, verbose=False, rng=None):
    """the run lm_gen.sv performs for the given switch settings -> tokens, screen, hash, final random state.
    rng: state left by the previous run (BTNC continues it); None = freshly seeded (reset, BTNU)"""
    cfg = cfg or config(q, name)
    pcs = pcs or pieces(name, q.V)
    cexp, fexp = qmul(q.s_logit * math.log2(math.e) / TAUS[tau])
    ncut = NCUTS[minp]
    rng = seed_of(seed) if rng is None else rng
    ir, term = IntRunner(q), Terminal()
    tok, pos, nseq, prev = cfg["START"], 0, 0, cfg["START"]
    toks, h = [], 0
    while True:
        nxt, rng = pick(ir.step(tok, pos)[:q.V], sample, cexp, fexp, ncut, rng)
        toks.append(nxt)
        h = (h * 31 + nxt) & 0xFFFFFFFF
        end = nxt == cfg["STOP"] or pos == q.T - 1
        term.put(pcs[nxt], strip=bool(cfg["STRIP"]) and prev == cfg["STRIP_TOK"],
                 force_nl=bool(cfg["MULTI"]) and pos == q.T - 1 and nxt != cfg["STOP"])
        prev = nxt
        if verbose:
            print(nxt, end=" ", flush=True)
        if cfg["MULTI"]:
            if end:
                nseq += 1
                if nseq == cfg["MAX_SEQ"]:
                    break
                tok, pos = cfg["START"], 0
                continue
        elif end:
            break
        tok, pos = nxt, pos + 1
    return dict(tokens=toks, hash=h, screen=term.screen(), cexp=cexp, fexp=fexp, rng=rng)


# ----------------------------------------------------------------------------- export
def font_rom():
    """8 x 16 bitmap font for ASCII 0..127, rendered from DejaVu Sans Mono (Bitstream Vera license)"""
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSansMono.ttf")
    f = ImageFont.truetype(path, 13)
    rom = []
    for c in range(128):
        img = Image.new("L", (8, 16), 0)
        if 33 <= c < 127:
            ImageDraw.Draw(img).text((0, 1), chr(c), font=f, fill=255)
        px = np.array(img)
        if c == 127:                                       # solid block (cursor)
            px[2:14, 1:7] = 255
        for y in range(16):
            rom.append(sum(1 << (7 - x) for x in range(8) if px[y, x] >= 110))
    return rom


def param_count(name):
    """parameters of the float model (as in the report), not of the integer tables"""
    from models import n_params
    if name == "stories260k":
        from stories import load_model
        return n_params(load_model())
    if name.startswith("ts_"):
        return json.load(open(os.path.join(ROOT, "model", "zoo", "runs", "stories260k",
                                           f"{name[3:]}_lr0.003_s0.json")))["params"]
    from check_ir import load_trained
    return n_params(load_trained(name))


WFMT = {"": "INT8", "int4": "INT4", "tern": "ternary", "bin": "1-bit", "lb25p1": "LittleBit 0.6b"}


def title_of(name, q):
    base, _, method = name.partition("-")
    arch = "Llama 2" if base == "stories260k" else ARCH.get(base[3:] if base.startswith("ts_") else base, base)
    data = "names.txt" if q.V == 27 else "TinyStories"
    fmt = "LittleBit" if method.startswith("lb") else WFMT.get(method, method)
    t = f" {name}: {arch}, {param_count(base) / 1000:.0f}K params, {fmt}, {data} | Nexys A7-100T FPGA"
    assert len(t) <= COLS, t
    return t


def mode_rom():
    """status text of every switch setting {greedy, tau, minp} (18 characters each)"""
    out = []
    for g in range(2):
        for t in range(8):
            for p in range(4):
                txt = "greedy" if g else f"T={TAUS[t]:.1f} min-p {MINP_TXT[p]}"
                out.append(txt.ljust(18)[:18])
    return out


def hexw(v, bits):
    return format(int(v) & ((1 << bits) - 1), f"0{bits // 4}x")


def export(name, T=256, mhz=50):
    from export_hw import export as export_model
    q = cap_context(qprog(name), T)
    out = os.path.join(OUTROOT, name)
    export_model(name, [], q=q, out=out, quiet=True)          # program, weights, tables, ..., zoo_pkg.sv
    os.remove(os.path.join(out, "tv.txt"))                   # (no golden vectors: board_sim checks whole runs)
    dims = json.load(open(os.path.join(out, "dims.json")))
    pcs, cfg = pieces(name, q.V), config(q, name)
    # detokeniser: tokmeta[k] = {len[3:0], off[15:0]}, tokstr = bytes
    off, meta, raw = 0, [], []
    for p in pcs:
        assert len(p) < 16
        meta.append((len(p) << 16) | off)
        raw += list(p)
        off += len(p)
    raw = raw or [0]
    w = lambda f, lines: open(os.path.join(out, f), "w").write("\n".join(lines) + "\n")
    w("tokmeta.mem", [hexw(v, 20) for v in meta])
    w("tokstr.mem", [hexw(v, 8) for v in raw + [0] * (len(raw) < 2)])
    w("font.mem", [hexw(v, 8) for v in font_rom()])
    scr = [" " * COLS] * 30
    scr[0] = title_of(name, q).ljust(COLS)
    scr[29] = STATUS_TXT.ljust(COLS)
    w("screen.mem", [hexw(ord(c), 8) for row in scr for c in row])
    w("mode.mem", [hexw(ord(c), 8) for s in mode_rom() for c in s])
    taus = [qmul(q.s_logit * math.log2(math.e) / t) for t in TAUS]
    assert all(f >= 9 for _, f in taus)
    ref = generate(q, name, cfg=cfg, pcs=pcs)              # run after power-up (all switches down): on-board check
    gr = generate(q, name, sample=False, cfg=cfg, pcs=pcs)  # greedy (SW0 up): deterministic at any time
    json.dump(dict(cfg, name=name, T=q.T, V=q.V, mhz=mhz, hash=ref["hash"], tokens=ref["tokens"],
                   hash_greedy=gr["hash"], tokens_greedy=gr["tokens"],
                   taus=TAUS, cexp=[c for c, _ in taus], fexp=[f for _, f in taus]),
              open(os.path.join(out, "board.json"), "w"), indent=1)
    pkg = f"""// GENERATED by sw/zoo/board.py for model '{name}' -- do not edit.
package board_pkg;
  localparam int          NV          = {q.V};
  localparam int          START_TOK   = {cfg["START"]};
  localparam int          STOP_TOK    = {cfg["STOP"]};
  localparam bit          MULTI       = 1'b{cfg["MULTI"]};
  localparam int          MAX_SEQ     = {cfg["MAX_SEQ"]};
  localparam bit          STRIP       = 1'b{cfg["STRIP"]};
  localparam int          STRIP_TOK   = {cfg["STRIP_TOK"]};
  localparam logic [31:0] EXPECT_SAMPLE = 32'h{ref["hash"]:08x};  // first run after a reseed, switches down
  localparam logic [31:0] EXPECT_GREEDY = 32'h{gr["hash"]:08x};  // any greedy run
  localparam int          TOKSTR_DEPTH = {len(raw) + (len(raw) < 2)};
  localparam int          COLS        = {COLS};
  localparam int          ROWS        = {ROWS};
  localparam logic [15:0] CEXP [8] = '{{{", ".join(f"16'd{c}" for c, _ in taus)}}};
  localparam logic [5:0]  FEXP [8] = '{{{", ".join(f"6'd{f}" for _, f in taus)}}};
  localparam logic [4:0]  NCUT [4] = '{{{", ".join(f"5'd{n}" for n in NCUTS)}}};
  localparam int          ST_NTOK  = {STATUS["ntok"][0]};
  localparam int          ST_TOKS  = {STATUS["toks"][0]};
  localparam int          ST_CPT   = {STATUS["cpt"][0]};
  localparam int          ST_MODE  = {STATUS["mode"][0]};
  localparam int          ST_CHECK = {STATUS["check"][0]};
endpackage
"""
    open(os.path.join(out, "board_pkg.sv"), "w").write(pkg)
    print(f"{name}: V={q.V} T={q.T} {dims['N_PARAMS']} params, W_DEPTH {dims['W_DEPTH']}, TAB_DEPTH "
          f"{dims['TAB_DEPTH']}, first run {len(ref['tokens'])} tokens, hash {ref['hash']:08x} -> {out}")
    return out


def evaluate(name, n_names=1000, n_windows=16):
    """test loss of the float model and of the integer program the board runs (bit-exact with the FPGA):
    names: the test names (as evaluate.py); TinyStories: the first test windows of pipeline.py"""
    from ir import FloatRunner, attach_rope
    from quant import nll
    q = qprog(name)
    base = name.partition("-")[0]
    if q.V == 27:
        from evaluate import seqs_of
        from data import load
        test = seqs_of(*load()["test"])[:n_names]
    else:
        from tinystories import val_windows
        Xt, Yt = val_windows(256, "test")
        test = [(Xt[i].tolist(), Yt[i].tolist()) for i in range(n_windows)]
    ti = n = 0
    for x, y in test:
        ir = IntRunner(q)
        for t, (tok, tgt) in enumerate(zip(x, y)):
            ti += nll(ir.step(int(tok), t) * q.s_logit, int(tgt))
            n += 1
    res = dict(int_nll=ti / n, tokens=n)
    path = os.path.join(ROOT, "model", "zoo", "nexys.json")
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr.setdefault(name, {}).update(res)
    json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
    print(f"{name}: integer test loss {res['int_nll']:.4f} ({n} tokens)")
    return res


def main():
    a = sys.argv[1:]
    opt = lambda k, d: int(a[a.index(k) + 1]) if k in a else d
    if a and a[0] == "eval":
        for name in a[1:]:
            evaluate(name)
        return
    if not a or a[0] not in ("export", "ref"):
        print(__doc__)
        return
    names = [x for x in a[1:] if not x.startswith("--") and not x.lstrip("-").isdigit()]
    for name in names:
        if a[0] == "export":
            export(name, T=opt("--T", 256), mhz=opt("--mhz", 50))
        else:
            q = cap_context(qprog(name), opt("--T", 256))
            r = generate(q, name, sample="--greedy" not in a, tau=opt("--tau", 0), minp=opt("--minp", 0),
                         seed=opt("--seed", 0))
            print(f"{len(r['tokens'])} tokens, hash {r['hash']:08x}\n" + "\n".join(r["screen"]))


if __name__ == "__main__":
    main()
