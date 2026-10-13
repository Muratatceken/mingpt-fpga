"""
On-board measurements on the PYNQ-Z2 (run as root on the board, from the repository root):

    sudo python3 board/pynq_z2/run_board.py                # every model of mem/zoo/universal/
    sudo python3 board/pynq_z2/run_board.py --cpu          # + the ARM Cortex-A9 baseline (sw/c/zoo_run)
    sudo python3 board/pynq_z2/run_board.py --story 200    # + a TinyStories story generated on the FPGA

For every model: load through AXI, run all golden vectors (bit-exactness ON THE BOARD), engine cycles
per token (hardware counter), wall-clock time per step from Python, and optionally the same integer
work on the ARM core. Results: board_results.json (copy it to model/zoo/board_results.json).
"""
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoo import Zoo  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UNI = os.path.join(ROOT, "mem", "zoo", "universal")
OVERLAY = os.path.join(ROOT, "build", "zoo", "pynq_overlay", "zoo.bit")


def decode(toks):
    """TinyStories tok512 decoding (llama2.c run.c), vocabulary from data/stories260K/tok512.bin"""
    import struct
    raw = open(os.path.join(ROOT, "data", "stories260K", "tok512.bin"), "rb").read()
    vocab, p = [], 4
    for _ in range(512):
        _, n = struct.unpack("fi", raw[p:p + 8])
        vocab.append(raw[p + 8:p + 8 + n])
        p += 8 + n
    out, prev = b"", 1
    for t in toks:
        if t == 1:
            prev = t
            continue
        piece = vocab[t]
        if prev == 1 and piece.startswith(b" "):
            piece = piece[1:]
        if piece.startswith(b"<0x") and len(piece) == 6:
            piece = bytes([int(piece[3:5], 16)])
        out += piece
        prev = t
    return out.decode("utf-8", errors="replace")


def cpu_baseline(model):
    exe = os.path.join(ROOT, "build", "c", "zoo_run")
    if not os.path.exists(exe):
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        subprocess.run(["gcc", "-O3", "-std=c99", "-mcpu=cortex-a9", "-mfpu=neon", "-o", exe,
                        os.path.join(ROOT, "sw", "c", "zoo_run.c")], check=True)
    r = subprocess.run([exe, os.path.join(ROOT, "mem", "zoo", model), "--reps", "3"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            f = line.split()
            return dict(steps=int(f[3]), errors=int(f[5]), us_per_token=float(f[7]))
    return dict(error=r.stdout[-500:] + r.stderr[-500:])


def main():
    args = sys.argv[1:]
    models = sorted(m for m in os.listdir(UNI) if os.path.isdir(os.path.join(UNI, m)))
    z = Zoo(OVERLAY)
    res = dict(board=platform.node(), time=time.strftime("%Y-%m-%d %H:%M:%S"), engine=dict(V=z.V, T=z.T, C=z.C),
               models={})
    for m in models:
        print(f"== {m}", flush=True)
        r = dict(load=z.load(os.path.join(UNI, m)))
        r.update(z.check_tv())
        if "--cpu" in args:
            r["cpu"] = cpu_baseline(m)
        res["models"][m] = r
        print(json.dumps(r), flush=True)
        json.dump(res, open("board_results.json", "w"), indent=1)
    if "--story" in args:
        n = int(args[args.index("--story") + 1])
        z.load(os.path.join(UNI, "stories260k"))
        t0 = time.perf_counter()
        toks, cyc = z.generate(n, start=1)
        dt = time.perf_counter() - t0
        exp = open(os.path.join(UNI, "stories260k", "gen.txt")).read().split("\n")[1].split()
        same = sum(int(a) == int(b) for a, b in zip(toks[1:], exp))
        res["story"] = dict(tokens=toks, text=decode(toks), cycles_avg=sum(cyc) / len(cyc),
                            tokens_per_s_wall=n / dt, identical_to_reference=same, n=n)
        print(res["story"]["text"])
        print(f"{same}/{min(n, len(exp))} tokens identical to the integer reference, "
              f"{n / dt:.1f} tokens/s including Python")
        json.dump(res, open("board_results.json", "w"), indent=1)
    print("written board_results.json")


if __name__ == "__main__":
    main()
