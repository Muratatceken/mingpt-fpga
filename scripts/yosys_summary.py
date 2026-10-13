"""Summarise a Yosys `stat` report for xc7z020 (PYNQ-Z2).  python3 yosys_summary.py stat.txt [name] [--json]"""
import json
import re
import sys

CAP = dict(LUT=53200, FF=106400, BRAM=140, DSP=220)


def summary(path):
    txt = open(path).read()
    cells = {m.group(2): int(m.group(1)) for m in re.finditer(r"^\s+(\d+)\s+(\w+)\s*$", txt, re.M)}
    g = lambda *k: sum(cells.get(x, 0) for x in k)
    lut_logic = g("LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6", "INV")
    lut_ram = (4 * g("RAM64M", "RAM32M") + 2 * g("RAM64X1D", "RAM32X1D") + 4 * g("RAM128X1D")
               + g("RAM64X1S", "RAM32X1S") + 2 * g("RAM128X1S") + 4 * g("RAM256X1S") + g("SRL16E", "SRLC32E"))
    r = dict(lut_logic=lut_logic, lut_ram=lut_ram, lut=lut_logic + lut_ram, ff=g("FDRE", "FDSE", "FDCE", "FDPE"),
             bram=g("RAMB36E1") + 0.5 * g("RAMB18E1"), dsp=g("DSP48E1"), carry=g("CARRY4"))
    ltp = path.replace("stat_", "ltp_")                 # longest topological path (cells), a depth proxy
    if ltp != path:
        try:
            m = re.search(r"length=(\d+)", open(ltp).read())
            if m:
                r["depth"] = int(m.group(1))
        except OSError:
            pass
    return r


if __name__ == "__main__":
    s = summary(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else ""
    if "--json" in sys.argv:
        print(json.dumps(s))
    else:
        print(f"Yosys sentezi {name} (xc7, zamanlama yok)")
        for k, lab, cap in [("lut_logic", "LUT (mantık)", None), ("lut_ram", "LUT (dağıtık RAM/SRL)", None),
                            ("lut", "LUT toplam", "LUT"), ("ff", "FF", "FF"), ("bram", "BRAM (RAMB36 eşd.)", "BRAM"),
                            ("dsp", "DSP48E1", "DSP")]:
            pct = f"  {100 * s[k] / CAP[cap]:5.1f}% / xc7z020" if cap else ""
            print(f"  {lab:24s} {s[k]:8g}{pct}")
