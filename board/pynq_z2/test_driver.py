"""
Off-board test of the PYNQ driver (zoo.py) with a register-level mock of the AXI engine:
  * load(): every LD_* write is captured and the rebuilt memory images must equal the image files;
  * check_tv(): the mock answers each step with the golden logits of tv.txt, so the driver's
    stepping / logit reading / comparison must report 0 errors (and N errors when one logit is corrupted).
The register protocol itself is verified against the RTL by tb/tb_zoo_axi.sv.
  python3 board/pynq_z2/test_driver.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zoo import MEMS, REG, ZOO_ID, Zoo  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class MockEngine:
    def __init__(self, model_dir, V=512, T=256, corrupt=False):
        self.regs, self.data, self.mem = {}, [0] * 8, {s: {} for s in range(5)}
        self.V, self.T, self.corrupt = V, T, corrupt
        tv = open(os.path.join(model_dir, "tv.txt")).read().split("\n")
        nseq, _ = map(int, tv[0].split())
        self.answers, i = [], 1                       # golden logits in step order
        for _ in range(nseq):
            n = int(tv[i])
            self.answers += [[int(x) for x in tv[i + 1 + t].split()[1:]] for t in range(n)]
            i += 1 + n
        self.k, self.cur = 0, None

    def write(self, off, v):
        if REG["LD_DATA"] <= off < REG["LD_DATA"] + 32:
            self.data[(off - REG["LD_DATA"]) // 4] = v
        elif off == REG["LD_GO"]:
            word = sum(d << (32 * i) for i, d in enumerate(self.data))
            sel = self.regs[REG["LD_SEL"]]
            self.mem[sel][self.regs[REG["LD_ADDR"]]] = word & ((1 << MEMS[sel][1]) - 1)
        elif off == REG["CTRL"] and v & 1:
            self.cur = list(self.answers[self.k])
            if self.corrupt and self.k == 3:
                self.cur[0] += 1
            self.k += 1
        else:
            self.regs[off] = v

    def read(self, off):
        if off == REG["ID"]:
            return ZOO_ID
        if off == REG["INFO0"]:
            return (self.V << 16) | self.T
        if off == REG["INFO1"]:
            return (5 << 24) | (8 << 16) | 64
        if off == REG["STATUS"]:
            return 2
        if off == REG["ARGMAX"]:
            return self.cur.index(max(self.cur))
        if off == REG["NVOCAB"]:
            return self.regs.get(off, self.V)
        if off == REG["CYCLES"]:
            return 1234
        if off >= REG["LOGIT"]:
            return self.cur[(off - REG["LOGIT"]) // 4] & 0xFFFFFFFF
        return self.regs.get(off, 0)


def main():
    fails = 0
    for m in ("gru", "llama", "stories260k"):
        d = os.path.join(ROOT, "mem", "zoo", "universal", m)
        eng = MockEngine(d)
        z = Zoo(mmio=eng)
        info = z.load(d)
        sizes = [int(x) for x in open(os.path.join(d, "sizes.txt")).read().split()]
        for sel, ((fname, bits), n) in enumerate(zip(MEMS, sizes)):
            words = [int(x, 16) for x in open(os.path.join(d, fname)).read().split()[:n]]
            got = [eng.mem[sel].get(a) for a in range(n)]
            if got != words:
                print(f"FAIL {m}: {fname} image differs")
                fails += 1
        r = z.check_tv(max_seq=3)
        bad = Zoo(mmio=MockEngine(d, corrupt=True))
        bad.model_dir = d
        rb = bad.check_tv(max_seq=3)
        ok = r["errors"] == 0 and rb["errors"] > 0
        fails += not ok
        print(f"{m:12s} load {info['words']} words OK | tv {r['steps']} steps, {r['errors']} errors | "
              f"corrupted run detected: {rb['errors']} errors -> {'PASS' if ok else 'FAIL'}")
    print("PASS: driver logic" if fails == 0 else f"FAIL: {fails}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
