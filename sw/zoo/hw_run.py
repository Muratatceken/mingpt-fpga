"""
Run the bit-exact RTL simulation (and optionally Yosys synthesis) of the microcoded engine
for every exported model and collect the results.
  python3 hw_run.py [model ...] [--synth] [--nseq N]
Output: model/zoo/hw_metrics.json   (needs mem/zoo/<model>/ from export_hw.py)
"""
import json
import os
import re
import subprocess
import sys

from data import ROOT
from models import ZOO


def sim(name, nseq):
    out = subprocess.run([os.path.join(ROOT, "scripts", "zoo_sim.sh"), name, str(nseq)], capture_output=True,
                         text=True, cwd=ROOT).stdout
    m = re.search(r"^RESULT \S+ steps (\d+) bad_steps (\d+) errors (\d+) avg_cycles (\d+) min_cycles (\d+) "
                  r"max_cycles (\d+)", out, re.M)
    if not m:
        print(out[-3000:])
        raise RuntimeError(f"simulation of {name} failed")
    steps, bad, err, avg, lo, hi = map(int, m.groups())
    prof = {k: int(v) for k, v in re.findall(r"^PROF \S+ (\S+) (\d+)", out, re.M)}
    return dict(steps=steps, bad_steps=bad, errors=err, cycles_avg=avg, cycles_min=lo, cycles_max=hi,
                profile={k: round(v / steps, 1) for k, v in prof.items()})


def synth(name):
    subprocess.run([os.path.join(ROOT, "scripts", "zoo_synth.sh"), name], capture_output=True, text=True, cwd=ROOT,
                   check=True)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "yosys_summary.py"),
                          os.path.join(ROOT, "build", "zoo", "yosys", f"stat_{name}.txt"), name, "--json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main():
    args = sys.argv[1:]
    do_synth = "--synth" in args
    nseq = 1000000
    if "--nseq" in args:
        k = args.index("--nseq")
        nseq = int(args[k + 1])
        del args[k:k + 2]
    args = [a for a in args if not a.startswith("--")]
    names = args or [n for n in ZOO if os.path.exists(os.path.join(ROOT, "mem", "zoo", n, "zoo_pkg.sv"))]
    path = os.path.join(ROOT, "model", "zoo", "hw_metrics.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        r = res.get(name, {})
        r.update(sim(name, nseq))
        if do_synth:
            r["yosys"] = synth(name)
        res[name] = r
        y = r.get("yosys")
        ys = f" | LUT {y['lut']} FF {y['ff']} BRAM {y['bram']} DSP {y['dsp']}" if y else ""
        print(f"{name:9s} {'PASS' if r['errors'] == 0 else 'FAIL'}  {r['steps']} steps, {r['errors']} errors, "
              f"{r['cycles_avg']} cycles/token{ys}", flush=True)
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
