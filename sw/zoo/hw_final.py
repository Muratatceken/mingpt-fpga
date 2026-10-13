"""
After the training campaigns: synthesis of the final models and promotion of the main models.
  python3 hw_final.py synth [tag ...]     Yosys (specialized engine; + all-units engine for names50k family)
                                          of the best-LR seed-0 run of every architecture -> model/zoo/hw_final.json
  python3 hw_final.py promote             names50k family best-LR seed-0 -> model/zoo/<arch>.pt + <arch>.spec.json
                                          (used by evaluate.py / export_hw.py / the board images)
"""
import json
import os
import shutil
import subprocess
import sys

from data import ROOT
from paper_data import FAMILY, runs

MZ = os.path.join(ROOT, "model", "zoo")
PATH = os.path.join(MZ, "hw_final.json")


def yosys(memdir, spec):
    args = [os.path.join(ROOT, "scripts", "zoo_synth.sh"), os.path.relpath(memdir, ROOT)] + (["spec"] if spec else [])
    subprocess.run(args, capture_output=True, cwd=ROOT)
    tag = os.path.relpath(memdir, ROOT).replace("/", "_") + ("_spec" if spec else "")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "yosys_summary.py"),
                          os.path.join(ROOT, "build", "zoo", "yosys", f"stat_{tag}.txt"), tag, "--json"],
                         capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else None


def synth(tags):
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for tag in tags:
        for arch, info in sorted(runs(tag).items()):
            key = f"{tag}/{arch}"
            run = os.path.basename(info["runs"][0]["path"])                 # best LR, seed 0
            memdir = os.path.join(ROOT, "build", "zoo", "pipe", tag, run)
            if not os.path.exists(os.path.join(memdir, "zoo_pkg.sv")):
                print(f"{key}: not exported yet (run pipeline.py)")
                continue
            r = res.get(key, {})
            if "spec" not in r:
                r["spec"] = yosys(memdir, True)
            if tag == "names50k" and arch in FAMILY and "gen" not in r:
                r["gen"] = yosys(memdir, False)
            r["run"] = run
            res[key] = r
            json.dump(res, open(PATH, "w"), indent=1)
            print(key, {k: (v or {}).get("lut") for k, v in r.items() if k in ("spec", "gen")},
                  {k: (v or {}).get("dsp") for k, v in r.items() if k in ("spec", "gen")}, flush=True)


def promote():
    n50 = runs("names50k")
    for arch in FAMILY:
        if arch not in n50:
            print(f"{arch}: no runs")
            continue
        r = n50[arch]["runs"][0]
        shutil.copy(r["path"] + ".pt", os.path.join(MZ, f"{arch}.pt"))
        json.dump(dict(spec=r["spec"], run=os.path.basename(r["path"]), lr=r["lr"], seed=r["seed"],
                       test_loss=r["test_loss"]), open(os.path.join(MZ, f"{arch}.spec.json"), "w"), indent=1)
        print(f"{arch}: {os.path.basename(r['path'])} (test {r['test_loss']:.4f})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["promote"]:
        promote()
    else:
        synth(sys.argv[2:] or ["names50k", "stories260k"])
