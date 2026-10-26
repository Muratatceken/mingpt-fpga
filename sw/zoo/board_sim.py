"""
Mac test of the stand-alone Nexys A7-100T design: Verilator simulation of the whole board top
(board/nexys_a7/nexys_a7_top.sv, tb/tb_nexys.sv) against the bit-exact reference sw/zoo/board.py.

  python3 board_sim.py <model> [--press CCU] [--sw HEX] [--sw<i> HEX] [--frame]
      --press   buttons after the automatic first run: C new run, U reseed + run, D clear
      --sw      switches for every run (SW0 greedy, SW[3:1] temperature, SW[5:4] min-p, SW[15:8] seed)
      --sw<i>   switches set before key i
      --frame   capture one VGA frame from the pins -> build/nexys/<model>/frame.png
Checks, for every run: every generated token, the run's hash and the on-board check result (LED16), and
the 80 x 28 text area character by character. Prints cycles/token and tokens/s at 50 MHz.
Needs mem/nexys/<model>/ (python3 board.py export <model>).
"""
import json
import os
import re
import subprocess
import sys
import time

from board import ROWS, cap_context, config, generate, pieces, qprog
from data import ROOT

SRC = ["rtl/zoo/zrom.sv", "rtl/zoo/zdiv.sv", "rtl/requant.sv", "rtl/isqrt_seq.sv", "rtl/zoo/zoo_core.sv",
       "board/nexys_a7/bin2bcd.sv", "board/nexys_a7/vga_text.sv", "board/nexys_a7/seg7.sv",
       "board/nexys_a7/lm_gen.sv", "board/nexys_a7/nexys_a7_top.sv", "tb/tb_nexys.sv"]
MHZ = 50


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


def build(model):
    mem = os.path.join("mem", "nexys", model)
    obj = os.path.join(ROOT, "build", "nexys", model, "obj")
    os.makedirs(obj, exist_ok=True)
    exe = os.path.join(obj, "Vtb_nexys")
    deps = [os.path.join(ROOT, f) for f in SRC] + [os.path.join(ROOT, mem, f) for f in os.listdir(os.path.join(ROOT, mem))]
    if os.path.exists(exe) and all(os.path.getmtime(f) < os.path.getmtime(exe) for f in deps):
        return exe
    env = toolchain_env()
    cmd = ["verilator", "--binary", "--timing", "--assert", "--timescale", "1ns/1ps", "-j", "0", "-Wall", "-Wno-fatal",
           "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-DSIM", "--x-assign", "fast",
           f"{mem}/zoo_pkg.sv", f"{mem}/board_pkg.sv"] + SRC + ["--top-module", "tb_nexys", "-Mdir", obj, "-o", "Vtb_nexys"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    open(os.path.join(obj, "build.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"Verilator build failed ({obj}/build.log)")
    warn = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("%Warning") and "board/nexys_a7" in ln]
    for ln in warn[:20]:
        print("  " + ln)
    return exe


def expected(model, keys, sws):
    """reference runs for the key sequence (random state continues on C, reseed on U, D only clears)"""
    q = cap_context(qprog(model), 256)
    cfg, pcs = config(q, model), pieces(model, q.V)
    runs, rng = [], None
    for i, key in enumerate(["U"] + list(keys)):         # the automatic run after reset is a reseeded run
        sw = sws[i]
        if key == "D":
            runs.append(None)
            continue
        greedy, tau, minp, seed = sw & 1, (sw >> 1) & 7, (sw >> 4) & 3, (sw >> 8) & 0xFF
        r = generate(q, model, sample=not greedy, tau=tau, minp=minp, seed=seed, cfg=cfg, pcs=pcs,
                     rng=None if key == "U" else rng)
        if not greedy:
            rng = r["rng"]
        default = (not greedy) and key == "U" and sw & 0xFF3E == 0     # tau, min-p, seed all zero
        r["check"] = 1 if greedy or default else 0
        runs.append(r)
    return runs


def main():
    a = sys.argv[1:]
    model = a[0]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    keys = opt("--press", "")
    sw0 = int(opt("--sw", "0"), 16)
    sws = [sw0]
    cur = sw0
    for i in range(len(keys)):
        cur = int(opt(f"--sw{i}", hex(cur)), 16)
        sws.append(cur)
    out = os.path.join(ROOT, "build", "nexys", model)
    t0 = time.time()
    exe = build(model)
    t_build = time.time() - t0
    plus = [f"+PRESS={keys}", f"+SW={sw0:x}", f"+SCREEN={out}/screen.txt"]
    plus += [f"+SW{i}={sws[i + 1]:x}" for i in range(len(keys))]
    if "--frame" in a:
        plus.append(f"+FRAME={out}/frame.ppm")
    t0 = time.time()
    r = subprocess.run([exe] + plus, cwd=ROOT, capture_output=True, text=True)
    t_sim = time.time() - t0
    log = r.stdout
    open(os.path.join(out, "sim.log"), "w").write(log + r.stderr)
    if "TBDONE" not in log:
        print(log[-3000:], r.stderr[-2000:])
        raise SystemExit("simulation did not finish")
    # tokens per run (a run ends with GENRUN)
    toks, runs_hw, cur = [], [], []
    for ln in log.splitlines():
        m = re.match(r"GENTOK (\d+) (\d+) (\d+)", ln)
        if m:
            cur.append(int(m.group(1)))
        m = re.match(r"GENRUN ntok (\d+) cycles (\d+) hash (\w+)", ln)
        if m:
            runs_hw.append(dict(tokens=cur, cycles=int(m.group(2)), hash=int(m.group(3), 16)))
            cur = []
    # screens
    txt = open(os.path.join(out, "screen.txt"), encoding="latin-1").read().split("\n")
    screens = []
    for i, ln in enumerate(txt):
        if ln.startswith("RUN "):
            screens.append(dict(head=ln, rows=txt[i + 1:i + 31]))
    ref = expected(model, keys, sws)
    ok = True
    hw_i = 0
    for i, (exp, scr) in enumerate(zip(ref, screens)):
        if exp is None:                                  # D: the text area must be empty
            blank = all(row == " " * 80 for row in scr["rows"][1:1 + ROWS])
            print(f"run {i}: clear  {'OK' if blank else 'FAIL'}")
            ok &= blank
            continue
        hw = runs_hw[hw_i]
        hw_i += 1
        same_tok = hw["tokens"] == exp["tokens"]
        first_bad = next((k for k, (x, y) in enumerate(zip(hw["tokens"], exp["tokens"])) if x != y),
                         min(len(hw["tokens"]), len(exp["tokens"])))
        same_scr = [row for row in scr["rows"][1:1 + ROWS]] == exp["screen"]
        chk = int(re.search(r"check (\d)", scr["head"]).group(1))
        chk_ok = (chk == 1) if exp["check"] else (chk == 0)
        n = len(hw["tokens"])
        cpt = hw["cycles"] / max(1, n)
        print(f"run {i}: {n} tokens ({'identical' if same_tok else f'FIRST MISMATCH at {first_bad}'}), hash "
              f"{hw['hash']:08x} {'=' if hw['hash'] == exp['hash'] else '!='} {exp['hash']:08x}, screen "
              f"{'identical' if same_scr else 'DIFFERENT'}, on-board check {['--', 'OK', 'FAIL', '?'][chk]}"
              f"{'' if chk_ok else ' (UNEXPECTED)'}, {cpt:.0f} cycles/token = {MHZ * 1e6 / cpt:.0f} tokens/s at {MHZ} MHz")
        if not same_scr:
            for k, (x, y) in enumerate(zip(scr["rows"][1:1 + ROWS], exp["screen"])):
                if x != y:
                    print(f"    row {k}: hw  |{x}|\n            ref |{y}|")
                    break
        ok &= same_tok and same_scr and hw["hash"] == exp["hash"] and chk_ok
    last = screens[-1]["rows"]
    print("\n".join("    |" + row + "|" for row in last))
    if "--frame" in a:
        from PIL import Image
        Image.open(os.path.join(out, "frame.ppm")).save(os.path.join(out, "frame.png"))
        print(f"VGA frame -> {os.path.relpath(out, ROOT)}/frame.png")
    print(f"{model}: {'PASS' if ok else 'FAIL'}  (build {t_build:.0f} s, simulation {t_sim:.0f} s)")
    res = dict(model=model, keys=keys, ok=bool(ok), runs=[dict(tokens=len(h["tokens"]), cycles=h["cycles"])
                                                         for h in runs_hw])
    json.dump(res, open(os.path.join(out, "sim.json"), "w"), indent=1)
    path = os.path.join(ROOT, "model", "zoo", "nexys.json")             # results of every model (tracked)
    allr = json.load(open(path)) if os.path.exists(path) else {}
    r0 = runs_hw[0]
    allr.setdefault(model, {}).update(sim_pass=bool(ok), sim_runs=len(runs_hw), sim_tokens=sum(len(h["tokens"]) for h in runs_hw),
                                      cycles_per_token=round(r0["cycles"] / len(r0["tokens"]), 1),
                                      tokens_per_s_50mhz=round(MHZ * 1e6 * len(r0["tokens"]) / r0["cycles"], 1))
    json.dump(dict(sorted(allr.items())), open(path, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
