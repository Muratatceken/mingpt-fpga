"""
Mac test of the stand-alone Nexys A7-100T design: simulation of the whole board top
(board/nexys_a7/nexys_a7_top.sv, tb/tb_nexys.sv) against the bit-exact reference sw/zoo/board.py.

  python3 board_sim.py <model> [--press CCU] [--sw HEX] [--sw<i> HEX] [--dly<i> N] [--frame]
                               [--sim verilator | icarus] [--pinonly]
      --press   keys after the automatic first run (tb/tb_nexys.sv):
                  C new run, U reseed + run, D clear, R CPU_RESET    (at idle: after the previous run)
                  c u d r  the same during the run in progress, once it has generated --dly<i> tokens
      --sw      switches for every run (SW0 greedy, SW[3:1] temperature, SW[5:4] min-p, SW6 typewriter,
                SW7 7-segment shows cycles/token, SW[15:8] seed)
      --sw<i>   switches set before key i
      --frame   capture one VGA frame from the pins -> build/nexys/<model>/frame.png (checked against the
                text RAM rendered with the font)
      --sim     verilator (default, 2-state) or icarus (sv2v + Icarus, 4-state, X checks on the pins and the
                text RAM write port: scripts/nexys_icarus.sh) or netlist (the post-synthesis netlist built by
                scripts/nexys_netlist_sim.sh, against the pin-only RTL: every pin in every cycle)
      --pinonly the testbench uses only the board pins (as for the netlist)
Checks, for every run: every generated token, the run's hash, the on-board check result (LED2/LED3), and at
every idle point the 80 x 28 text area, the status row, the title and the 7-segment display; runs aborted by
CPU_RESET must be a prefix of the reference run. Prints cycles/token and tokens/s at 50 MHz.
Behaviour of the board (lm_gen.sv, modelled by Board below):
  BTNC / BTNU during a run: queued, the next run starts when the current one ends (BTNU: reseeded from the
  switches at that time; several presses during one run give one run, reseeded if one was BTNU; a press in
  the very cycle a queued run starts counts for that new run); BTND during a run: ignored;
  CPU_RESET at any time: the run is aborted, the board starts again with a reseeded run (the text RAM is
  cleared by that run); switches are read when a run starts (seed: only when it is reseeded).
Needs mem/nexys/<model>/ (python3 board.py export <model>).
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

from board import (COLS, ROWS, STATUS, STATUS_TXT, cap_context, config, generate, mode_rom, pieces, qprog,
                   seed_of)
from data import ROOT

SRC = ["rtl/zoo/zrom.sv", "rtl/zoo/zdiv.sv", "rtl/requant.sv", "rtl/isqrt_seq.sv", "rtl/zoo/zoo_core.sv",
       "board/nexys_a7/bin2bcd.sv", "board/nexys_a7/vga_text.sv", "board/nexys_a7/seg7.sv",
       "board/nexys_a7/lm_gen.sv", "board/nexys_a7/nexys_a7_top.sv", "tb/tb_nexys.sv"]
MHZ = 50                                   # tb_nexys.sv: CLK_MHZ
CHECK_TXT = ["--  ", "OK  ", "FAIL", "... "]


def toolchain_env():
    """macOS: Homebrew Verilator, the system linker (not conda's), and the SDK of the selected Xcode when xcrun
    picks a newer Command Line Tools SDK that Xcode's linker cannot read"""
    env = dict(os.environ, PATH="/opt/homebrew/bin:/usr/bin:" + os.environ["PATH"])
    if sys.platform == "darwin" and "SDKROOT" not in env:
        run = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
        dev = run("xcode-select", "-p")
        sdk = os.path.join(dev, "Platforms", "MacOSX.platform", "Developer", "SDKs", "MacOSX.sdk")
        if os.path.isdir(sdk) and not run("xcrun", "--show-sdk-path").startswith(dev):
            env["SDKROOT"] = sdk
    return env


# ----------------------------------------------------------------------------- simulator builds
def build(model, sim="verilator", pinonly=False):
    """simulator of the board top for one model -> command (list) to run it from the repo root"""
    mem = os.path.join("mem", "nexys", model)
    tag = sim + ("_pinonly" if pinonly else "")
    obj = os.path.join(ROOT, "build", "nexys", model, "obj" if tag == "verilator" else f"obj_{tag}")
    os.makedirs(obj, exist_ok=True)
    exe = os.path.join(obj, "Vtb_nexys" if sim == "verilator" else "sim.vvp")
    deps = [os.path.join(ROOT, f) for f in SRC] + [os.path.join(ROOT, mem, f) for f in os.listdir(os.path.join(ROOT, mem))]
    defs = ["-DSIM"] + (["-DPINONLY"] if pinonly else [])
    cmd_run = [exe] if sim == "verilator" else ["vvp", "-n", exe]
    if os.path.exists(exe) and all(os.path.getmtime(f) < os.path.getmtime(exe) for f in deps):
        return cmd_run
    env = toolchain_env()
    pkgs = [f"{mem}/zoo_pkg.sv", f"{mem}/board_pkg.sv"]
    if sim == "verilator":
        cmds = [["verilator", "--binary", "--timing", "--assert", "--timescale", "1ns/1ps", "-j", "0", "-Wall",
                 "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM"] + defs +
                ["--x-assign", "fast"] + pkgs + SRC + ["--top-module", "tb_nexys", "-Mdir", obj, "-o", "Vtb_nexys"]]
    elif sim == "icarus":                                  # 4-state: X checks in the testbench
        cmds = [["sv2v"] + defs + ["-DXCHECK"] + pkgs + SRC + ["-w", f"{obj}/sim.v"],
                ["iverilog", "-g2012", "-s", "tb_nexys", "-o", exe, f"{obj}/sim.v"]]
    else:
        raise SystemExit(f"unknown simulator {sim}")
    log = ""
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        log += r.stdout + r.stderr
        if r.returncode:
            open(os.path.join(obj, "build.log"), "w").write(log)
            print(log[-4000:])
            raise SystemExit(f"{cmd[0]} failed ({obj}/build.log)")
    open(os.path.join(obj, "build.log"), "w").write(log)
    warn = [ln for ln in log.splitlines() if ln.startswith("%Warning") and "board/nexys_a7" in ln]
    for ln in warn[:20]:
        print("  " + ln)
    return cmd_run


def switch_list(keys, sw0, per_key):
    """switches before every key: sws[0] for the automatic run, sws[i + 1] set before key i"""
    sws, cur = [sw0], sw0
    for i in range(len(keys)):
        cur = per_key.get(i, cur)
        sws.append(cur)
    return sws


def run_tb(cmd, keys, sws, dlys, out, frame=False, maxcyc=None, extra=()):
    """run the testbench -> (log, screen dump text or None, seconds)"""
    os.makedirs(out, exist_ok=True)
    plus = [f"+PRESS={keys}", f"+SW={sws[0]:x}", f"+SCREEN={out}/screen.txt"] + list(extra)
    plus += [f"+SW{i}={sws[i + 1]:x}" for i in range(len(keys))]
    plus += [f"+DLY{i}={d}" for i, d in dlys.items()]
    if frame:
        plus.append(f"+FRAME={out}/frame.ppm")
    if maxcyc:
        plus.append(f"+MAXCYC={maxcyc}")
    scr = os.path.join(out, "screen.txt")
    if os.path.exists(scr):
        os.remove(scr)
    t0 = time.time()
    r = subprocess.run(cmd + plus, cwd=ROOT, capture_output=True, text=True, env=toolchain_env())
    t = time.time() - t0
    log = r.stdout + r.stderr
    open(os.path.join(out, "sim.log"), "w").write(log)
    if "TBDONE" not in log:
        print(log[-3000:])
        raise RuntimeError(f"simulation did not finish ({out}/sim.log)")
    txt = open(scr, encoding="latin-1").read() if os.path.exists(scr) else None
    return log, txt, t


def parse(log, txt):
    """testbench output -> runs (completed and aborted), idle points, keys, token time stamps"""
    runs, idles, keys, stamps, tbtoks, cur, ccur, reset_running = [], [], [], [], [], [], [], 0
    xcheck = None
    for ln in log.splitlines():
        m = re.match(r"GENTOK (\d+) (\d+) (\d+)", ln)
        if m:
            cur.append(int(m.group(1)))
            ccur.append(int(m.group(3)))                  # counted cycles (lm_gen.sv cyc) at the token
        m = re.match(r"GENRUN ntok (\d+) cycles (\d+) hash (\w+) rng (\w+)", ln)
        if m:
            runs.append(dict(tokens=cur, tok_cycles=ccur, ntok=int(m.group(1)), cycles=int(m.group(2)),
                             hash=int(m.group(3), 16), rng=int(m.group(4), 16), aborted=False))
            cur, ccur = [], []
        m = re.match(r"TBRESET cycle (\d+) running (\d) ntok (\d+)", ln)
        if m:
            reset_running = int(m.group(2))
        if ln.startswith("GENRESET"):
            if reset_running:
                runs.append(dict(tokens=cur, tok_cycles=ccur, aborted=True))
            elif cur:
                raise RuntimeError("tokens of a run that did not finish before an idle reset")
            cur, ccur = [], []
        m = re.match(r"TBIDLE (\d+) cycle (\d+) check (\d) ntok (\d+) sw (\w+) seg7 \[(.{8})\] sig (\w+) hash (\w+)", ln)
        if m:
            idles.append(dict(cycle=int(m.group(2)), check=int(m.group(3)), ntok=int(m.group(4)),
                              sw=int(m.group(5), 16), seg7=m.group(6), sig=m.group(7), hash=m.group(8)))
        m = re.match(r"TBKEY (\d+) (\S) running (\d) (\d) ntok (\d+) cycle (\d+)", ln)
        if m:
            keys.append(dict(key=m.group(2), r0=int(m.group(3)), r1=int(m.group(4)), ntok=int(m.group(5)),
                             cycle=int(m.group(6))))
        m = re.match(r"TBTOK (\d+) (\d+) (\d+)", ln)
        if m:
            stamps.append((int(m.group(1)), int(m.group(2))))
            tbtoks.append((int(m.group(1)), int(m.group(3))))
        m = re.match(r"XCHECK unknown (\d+)", ln)
        if m:
            xcheck = int(m.group(1))
    screens = []
    if txt is not None:
        lines = txt.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("RUN "):
                screens.append(dict(head=ln, rows=lines[i + 1:i + 31],
                                    check=int(re.search(r"check (\d)", ln).group(1))))
    sig = re.search(r"TBDONE cycles (\d+) sig (\w+)", log)
    frame = re.search(r"FRAME (\S+) sync_errors (\d+)", log)
    return dict(runs=runs, idles=idles, keys=keys, stamps=stamps, tbtoks=tbtoks, screens=screens, xcheck=xcheck,
                cycles=int(sig.group(1)), sig=sig.group(2), frame_sync_errors=int(frame.group(2)) if frame else None,
                xerr=[ln for ln in log.splitlines() if ln.startswith("XERR")][:10],
                errors=[ln for ln in log.splitlines() if "%Error" in ln or "ERROR" in ln or "Assertion" in ln][:10])


# ----------------------------------------------------------------------------- reference behaviour
class Board:
    """what the board does for a key sequence (see the module docstring), built key by key so that a fuzzer
    can choose the token count of a mid-run key from the run in progress"""

    def __init__(self, model, sw0):
        self.model = model
        self.q = cap_context(qprog(model), 256)
        self.cfg, self.pcs = config(self.q, model), pieces(model, self.q.V)
        self.cache = {}
        self.runs, self.idles = [], []          # expected runs (in order), idle points
        self.rng, self.pending, self.blank = None, None, False
        self.sw = sw0                           # switches currently set
        self.mid_floor = 0                      # mid-run keys: token counts never decrease within a run
        self.presses = []                       # keys, "end" for a press at the end of a run
        self.start(True)

    def reference(self, sw, rng):
        greedy, tau, minp = sw & 1, (sw >> 1) & 7, (sw >> 4) & 3
        key = ("greedy",) if greedy else (tau, minp, rng)
        if key not in self.cache:
            self.cache[key] = generate(self.q, self.model, sample=not greedy, tau=tau, minp=minp, cfg=self.cfg,
                                       pcs=self.pcs, rng=rng if rng is not None else 1)
        return self.cache[key]

    def start(self, reseed):
        sw = self.sw
        greedy = sw & 1
        if reseed:
            self.rng = seed_of((sw >> 8) & 0xFF)   # also for a greedy run: the state is set, not used
        r = self.reference(sw, self.rng)
        if not greedy:
            self.rng = r["rng"]
        default = (not greedy) and reseed and sw & 0x3E == 0 and (sw >> 8) == 0   # T 0.8, min-p 1/32, seed 0
        self.runs.append(dict(tokens=r["tokens"], hash=r["hash"], screen=r["screen"], sw=sw, reseed=reseed,
                              check=1 if greedy or default else 0, abort=None))
        self.blank, self.mid_floor = False, 0

    def current(self):
        """the run in progress (mid-run keys act on it)"""
        return self.runs[-1]

    def flush(self):
        if self.pending:
            self.start(self.pending == "u")
            self.pending = None

    def idle(self):
        self.flush()
        self.idles.append(dict(run=len(self.runs) - 1, blank=self.blank, sw=self.sw))

    def key(self, k, sw, dly=0):
        self.presses.append(k)
        if k.isupper():
            self.idle()
            self.sw = sw
            if k == "C":
                self.start(False)
            elif k in "UR":
                self.start(True)
            elif k == "D":
                self.blank = True
            else:
                raise ValueError(k)
            self.mid_floor = 0
            return
        assert not self.blank, "a mid-run key needs a run in progress (not after D)"
        cur = self.current()
        n = len(cur["tokens"])
        # c / u may also fall on the end of the run (dly = n: queued or at idle, the same run follows) unless a
        # run is queued already (it starts at the end: the press may then come before or after its start);
        # d / r must come before the last token (at the end the outcome depends on the exact cycle)
        assert self.mid_floor <= dly <= n - (k in "dr" or self.pending is not None), (k, dly, n)
        self.mid_floor = dly if dly < n else n + 1        # after a press at the end: no more mid-run keys
        if dly == n:
            self.presses[-1] = "end"
        self.sw = sw
        if k == "c":
            self.pending = self.pending or "c"
        elif k == "u":
            self.pending = "u"
        elif k == "r":
            cur["abort"] = dly
            self.pending = None
            self.start(True)
        elif k != "d":
            raise ValueError(k)

    def finish(self):
        self.idle()


def status_row(run, hw):
    """status row (row 29) after a run: fields as lm_gen.sv writes them, cycles from the simulation"""
    row = list(STATUS_TXT.ljust(COLS))
    n, cyc = len(run["tokens"]), hw["cycles"]
    toks = 0 if cyc == 0 else min(n * MHZ * 1_000_000 // cyc, 9_999_999)
    cpt = 0 if n == 0 else min(cyc // n, 9_999_999)
    sw = run["sw"]
    mode = mode_rom()[(sw & 1) * 32 + ((sw >> 1) & 7) * 4 + ((sw >> 4) & 3)]
    for f, s in dict(ntok=str(n).rjust(5), toks=str(toks).rjust(7), cpt=str(cpt).rjust(7), mode=mode,
                     check=CHECK_TXT[run["check"]]).items():
        c, w = STATUS[f]
        assert len(s) == w
        row[c:c + w] = s
    return "".join(row), toks, cpt


def title_row(model):
    v = open(os.path.join(ROOT, "mem", "nexys", model, "screen.mem")).read().split()
    return "".join(chr(int(x, 16)) for x in v[:COLS])


def render(rows, model):
    """30 x 80 characters -> the 640 x 480 RGB frame vga_text.sv shows"""
    font = np.array([int(x, 16) for x in open(os.path.join(ROOT, "mem", "nexys", model, "font.mem")).read().split()],
                    dtype=np.uint8).reshape(128, 16)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for r in range(30):
        bar = r == 0 or r == ROWS + 1
        on = np.array([0xF, 0xF, 0xF] if bar else [0xD, 0xD, 0xD]) * 17
        off = np.array([0x1, 0x3, 0x7] if bar else [0, 0, 0]) * 17
        for c in range(COLS):
            g = font[ord(rows[r][c]) & 0x7F]
            bits = (g[:, None] >> (7 - np.arange(8))[None, :]) & 1
            img[16 * r:16 * r + 16, 8 * c:8 * c + 8] = np.where(bits[:, :, None] == 1, on, off)
    return img


def read_ppm(path):
    data = open(path, "rb").read()
    hdr = b"P6\n640 480\n255\n"
    assert data.startswith(hdr)
    return np.frombuffer(data[len(hdr):], dtype=np.uint8).reshape(480, 640, 3)


def compare(board, sim, model, keys, quiet=False):
    """simulation vs reference -> (ok, report lines)"""
    rep, ok = [], True
    exp_runs = board.runs
    hw_runs = sim["runs"]
    if len(hw_runs) != len(exp_runs):
        rep.append(f"FAIL: {len(hw_runs)} runs in the simulation, {len(exp_runs)} expected")
        ok = False
    for i, (exp, hw) in enumerate(zip(exp_runs, hw_runs)):
        tag = f"run {i} ({'U' if exp['reseed'] else 'C'} sw {exp['sw']:04x})"
        if exp["abort"] is not None:
            n = len(hw["tokens"])
            good = hw["aborted"] and hw["tokens"] == exp["tokens"][:n] and n >= exp["abort"]
            rep.append(f"{tag}: aborted by CPU_RESET after {n} tokens (>= {exp['abort']}), "
                       f"{'prefix of the reference' if good else 'MISMATCH'}")
            ok &= good
            continue
        if hw["aborted"]:
            rep.append(f"{tag}: FAIL, aborted in the simulation")
            ok = False
            continue
        same = hw["tokens"] == exp["tokens"]
        bad = next((k for k, (x, y) in enumerate(zip(hw["tokens"], exp["tokens"])) if x != y),
                   min(len(hw["tokens"]), len(exp["tokens"])))
        n = len(hw["tokens"])
        cpt = hw["cycles"] / max(1, n)
        rep.append(f"{tag}: {n} tokens ({'identical' if same else f'FIRST MISMATCH at {bad}'}), hash "
                   f"{hw['hash']:08x} {'=' if hw['hash'] == exp['hash'] else '!='} {exp['hash']:08x}, "
                   f"{cpt:.0f} cycles/token = {MHZ * 1e6 / cpt:.0f} tokens/s at {MHZ} MHz")
        ok &= same and hw["hash"] == exp["hash"]
    # idle points
    title = title_row(model)
    if len(sim["idles"]) != len(board.idles):
        rep.append(f"FAIL: {len(sim['idles'])} idle points, {len(board.idles)} expected")
        ok = False
    for j, (ei, hi) in enumerate(zip(board.idles, sim["idles"])):
        run = exp_runs[ei["run"]]
        hw = hw_runs[ei["run"]] if ei["run"] < len(hw_runs) else None
        if hw is None or hw["aborted"]:
            rep.append(f"idle {j}: FAIL, no completed run")
            ok = False
            continue
        status, toks, cpt = status_row(run, hw)
        seg = str(cpt if ei["sw"] & 0x80 else toks).rjust(8)
        errs = []
        if hi["check"] != run["check"]:
            errs.append(f"check LED {CHECK_TXT[hi['check']]} expected {CHECK_TXT[run['check']]}")
        if hi["ntok"] != len(run["tokens"]) & 0xFFF:
            errs.append(f"LED ntok {hi['ntok']} expected {len(run['tokens'])}")
        if hi["seg7"] != seg:
            errs.append(f"7-segment [{hi['seg7']}] expected [{seg}]")
        if hi["sw"] != ei["sw"]:
            errs.append(f"switches {hi['sw']:04x} expected {ei['sw']:04x}")
        if sim.get("probes", True) and int(hi["hash"], 16) != run["hash"]:
            errs.append(f"hash register {hi['hash']} expected {run['hash']:08x}")
        if sim["screens"]:
            scr = sim["screens"][j]
            text = scr["rows"][1:1 + ROWS]
            want = [" " * COLS] * ROWS if ei["blank"] else run["screen"]
            if scr["check"] != run["check"]:
                errs.append(f"check {scr['check']} expected {run['check']}")
            if text != want:
                k = next(k for k in range(ROWS) if text[k] != want[k])
                errs.append(f"text row {k}: hw |{text[k]}| ref |{want[k]}|")
            if scr["rows"][ROWS + 1] != status:
                errs.append(f"status |{scr['rows'][ROWS + 1]}| ref |{status}|")
            if scr["rows"][0] != title:
                errs.append(f"title |{scr['rows'][0]}|")
        what = "clear" if ei["blank"] else f"after run {ei['run']}"
        rep.append(f"idle {j}: {what}, " + ("screen, status, check, LEDs, 7-segment OK" if not errs else
                                            "FAIL: " + "; ".join(errs)))
        ok &= not errs
    for i, (k, hk) in enumerate(zip(keys, sim["keys"])):
        if hk["key"] != k:
            rep.append(f"key {i}: FAIL, testbench pressed {hk['key']}")
            ok = False
        elif k.islower() and board.presses[i] != "end" and not (hk["r0"] and (hk["r1"] or k == "r")):
            rep.append(f"key {i} '{k}': FAIL, not pressed during a run (LED1 {hk['r0']} {hk['r1']})")
            ok = False
        elif k.isupper() and hk["r0"]:
            rep.append(f"key {i} '{k}': FAIL, pressed during a run")
            ok = False
    if len(sim["keys"]) != len(keys):
        rep.append(f"FAIL: {len(sim['keys'])} keys pressed, {len(keys)} expected")
        ok = False
    if sim["xcheck"] is not None:
        rep.append(f"X check: {sim['xcheck']} unknown values on the pins / text RAM write port")
        rep += ["  " + x for x in sim["xerr"]]
        ok &= sim["xcheck"] == 0
    if sim["errors"]:
        rep += ["  " + x for x in sim["errors"]]
        ok = False
    return bool(ok), rep


def expected_rows(board, runs_hw, model, j=-1):
    """the 30 screen rows at idle point j: title, text area, status row (cycles of the simulated run)"""
    ei = board.idles[j]
    run = board.runs[ei["run"]]
    text = [" " * COLS] * ROWS if ei["blank"] else run["screen"]
    return [title_row(model)] + text + [status_row(run, runs_hw[ei["run"]])[0]]


def check_frame(model, sim, out, board, runs_hw):
    """the VGA frame captured from the pins = the reference screen (and the last text RAM dump, if any)
    rendered with the font"""
    img = read_ppm(os.path.join(out, "frame.ppm"))
    ok = sim["frame_sync_errors"] == 0
    msg = f"VGA frame: {sim['frame_sync_errors']} sync/blanking errors"
    try:
        diff = int(np.count_nonzero(np.any(render(expected_rows(board, runs_hw, model), model) != img, axis=2)))
    except (IndexError, KeyError):
        diff = -1
    msg += f", {diff} pixels differ from the reference screen"
    ok &= diff == 0
    if sim["screens"]:
        diff = int(np.count_nonzero(np.any(render(sim["screens"][-1]["rows"], model) != img, axis=2)))
        msg += f", {diff} from the text RAM"
        ok &= diff == 0
    from PIL import Image
    Image.fromarray(img).save(os.path.join(out, "frame.png"))
    return ok, msg + f" -> {os.path.relpath(out, ROOT)}/frame.png"


def make_board(model, keys, sws, dlys):
    board = Board(model, sws[0])
    for i, k in enumerate(keys):
        board.key(k, sws[i + 1], dlys.get(i, 0))
    board.finish()
    return board


def simulate(model, keys="", sws=None, dlys=None, frame=False, sim="verilator", pinonly=False, out=None, board=None):
    """build + run + compare -> (ok, report lines, parsed simulation, board)"""
    sws = sws or [0] * (len(keys) + 1)
    dlys = dlys or {}
    out = out or os.path.join(ROOT, "build", "nexys", model)
    cmd = build(model, sim, pinonly)
    log, txt, t_sim = run_tb(cmd, keys, sws, dlys, out, frame=frame)
    s = parse(log, txt)
    s["seconds"] = t_sim
    board = board or make_board(model, keys, sws, dlys)
    ok, rep = compare(board, s, model, keys)
    if frame:
        fok, msg = check_frame(model, s, out, board, s["runs"])
        rep.append(msg)
        ok &= fok
    return ok, rep, s, board


def netlist(model, keys, sws, dlys, frame):
    """post-synthesis netlist (scripts/nexys_netlist_sim.sh built it) against the pin-only RTL: every pin in every
    cycle (signatures), the LED token counter, the 7-segment display, the VGA frame; the RTL against board.py"""
    base = os.path.join(ROOT, "build", "nexys", model)
    ok_rtl, rep, rtl, board = simulate(model, keys, sws, dlys, frame=frame, pinonly=True, out=f"{base}_pinonly")
    rep = ["RTL (Verilator, pin-only) against board.py:"] + ["  " + x for x in rep]
    rep.append(f"  RTL: {'PASS' if ok_rtl else 'FAIL'}, {rtl['cycles']} cycles, {rtl['seconds']:.0f} s")
    ndir = os.environ.get("NETLIST_DIR", os.path.join(base, "netlist"))
    cmd = os.environ.get("NETSIM_CMD", f"vvp -n {ndir}/sim.vvp").split()
    four = cmd[0].endswith("vvp")
    log, _, t_net = run_tb(cmd, keys, sws, dlys, ndir, frame=frame, maxcyc=os.environ.get("MAXCYC"))
    net = parse(log, None)
    net["seconds"] = t_net
    probes = os.environ.get("NETLIST_PROBES") == "1"           # \u_gen.nxt, \u_gen.hash kept in the netlist
    if not probes:
        for d in net["idles"] + rtl["idles"]:
            d.pop("hash")
    ok = ok_rtl
    rep.append(f"netlist (Yosys synth_xilinx; {'Icarus, 4-state' if four else 'Verilator, 2-state'}) against the RTL:")
    mm = [ln for ln in log.splitlines() if ln.startswith("MMCME2_BASE")]
    rep += ["  " + x for x in mm]
    ok &= bool(mm) and all("ERROR" not in x for x in mm)
    if probes:
        same = net["tbtoks"] == rtl["tbtoks"]
        n = sum(1 for k, _ in net["tbtoks"] if k > 0)
        rep.append(f"  {n} generated tokens (netlist register \\u_gen.nxt): {'identical' if same else 'DIFFERENT'}; "
                   f"run hashes (\\u_gen.hash) at the idle points: "
                   f"{[d['hash'] for d in net['idles']]}")
        ok &= same
    for what in ("idles", "keys"):
        same = net[what] == rtl[what]
        rep.append(f"  {len(net[what])} {'idle points (cycle, LEDs, 7-segment, pin signature)' if what == 'idles' else 'key presses'}"
                   f": {'identical' if same else 'DIFFERENT'}")
        if not same:
            for x, y in zip(net[what], rtl[what]):
                if x != y:
                    rep.append(f"    netlist {x}\n    RTL     {y}")
                    break
        ok &= same
    same = net["stamps"] == rtl["stamps"]
    bad = next((k for k, (x, y) in enumerate(zip(net["stamps"], rtl["stamps"])) if x != y),
               min(len(net["stamps"]), len(rtl["stamps"])))
    rep.append(f"  {len(net['stamps'])} LED token-counter changes: "
               f"{'identical (same cycles)' if same else f'DIFFERENT from change {bad}'}")
    ok &= same
    same = (net["cycles"], net["sig"]) == (rtl["cycles"], rtl["sig"])
    rep.append(f"  end: {net['cycles']} cycles, pin signature {net['sig']} "
               f"{'=' if same else '!='} RTL {rtl['cycles']} cycles, {rtl['sig']}")
    ok &= same
    if four:
        rep.append(f"  X check: {net['xcheck']} unknown values on the pins")
        rep += ["    " + x for x in net["xerr"]]
        ok &= net["xcheck"] == 0
    if net["errors"]:
        rep += ["    " + x for x in net["errors"]]
        ok = False
    if frame:
        a = open(os.path.join(base + "_pinonly", "frame.ppm"), "rb").read()
        b = open(os.path.join(ndir, "frame.ppm"), "rb").read()
        fok, msg = check_frame(model, net, ndir, board, rtl["runs"])
        rep.append(f"  {msg}; {'identical to' if a == b else 'DIFFERENT from'} the RTL frame")
        ok &= fok and a == b
    return ok, rep, rtl, net


def main():
    a = sys.argv[1:]
    model = a[0]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    keys = opt("--press", "")
    sws = switch_list(keys, int(opt("--sw", "0"), 16),
                      {i: int(opt(f"--sw{i}", "0"), 16) for i in range(len(keys)) if f"--sw{i}" in a})
    dlys = {i: int(opt(f"--dly{i}", "0")) for i in range(len(keys)) if f"--dly{i}" in a}
    sim = opt("--sim", "verilator")
    if sim == "netlist":
        ok, rep, rtl, net = netlist(model, keys, sws, dlys, "--frame" in a)
        print("\n".join(rep))
        print(f"{model} netlist: {'PASS' if ok else 'FAIL'}  (RTL {rtl['seconds']:.0f} s, netlist {net['seconds']:.0f} s "
              f"for {net['cycles']} cycles)")
        ndir = os.environ.get("NETLIST_DIR", os.path.join(ROOT, "build", "nexys", model, "netlist"))
        json.dump(dict(model=model, keys=keys, ok=bool(ok), cycles=net["cycles"], sig=net["sig"],
                       seconds=round(net["seconds"]), report=rep), open(os.path.join(ndir, "sim.json"), "w"), indent=1)
        sys.exit(0 if ok else 1)
    pinonly = "--pinonly" in a
    out = os.path.join(ROOT, "build", "nexys", model if sim == "verilator" and not pinonly else
                       f"{model}_{sim}{'_pinonly' if pinonly else ''}")
    t0 = time.time()
    build(model, sim, pinonly)
    t_build = time.time() - t0
    ok, rep, s, board = simulate(model, keys, sws, dlys, frame="--frame" in a, sim=sim, pinonly=pinonly, out=out)
    print("\n".join(rep))
    if s["screens"]:
        print("\n".join("    |" + row + "|" for row in s["screens"][-1]["rows"]))
    print(f"{model}: {'PASS' if ok else 'FAIL'}  ({sim}{' pin-only' if pinonly else ''}: build {t_build:.0f} s, "
          f"simulation {s['seconds']:.0f} s, {s['cycles']} cycles, pin signature {s['sig']})")
    res = dict(model=model, keys=keys, sim=sim, ok=bool(ok), sig=s["sig"], cycles=s["cycles"],
               runs=[dict(tokens=len(h["tokens"]), cycles=h.get("cycles"), aborted=h["aborted"]) for h in s["runs"]])
    json.dump(res, open(os.path.join(out, "sim.json"), "w"), indent=1)
    done = [h for h in s["runs"] if not h["aborted"]]
    if sim == "verilator" and not pinonly and done:
        path = os.path.join(ROOT, "model", "zoo", "nexys.json")         # results of every model
        allr = json.load(open(path)) if os.path.exists(path) else {}
        r0 = done[0]
        allr.setdefault(model, {}).update(sim_pass=bool(ok), sim_runs=len(done), sim_tokens=sum(len(h["tokens"]) for h in done),
                                          cycles_per_token=round(r0["cycles"] / len(r0["tokens"]), 1),
                                          tokens_per_s_50mhz=round(MHZ * 1e6 * len(r0["tokens"]) / r0["cycles"], 1))
        json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
