"""
PYNQ driver of the universal zoo engine (rtl/zoo/zoo_axi.sv, overlay built by scripts/vivado_zoo_pynq.tcl).

    from zoo import Zoo
    z = Zoo("zoo.bit")                       # loads the overlay (zoo.hwh next to it)
    z.load("mem/zoo/universal/gpt2")         # model images -> engine memories (through LD_* registers)
    tok, cycles = z.step(token, position)    # one decode step: argmax and clock cycles
    logits = z.logits(27)

The register map is documented in rtl/zoo/zoo_axi.sv. Any object with read(offset) / write(offset, value)
can stand in for pynq.MMIO (used by tb-less tests on a PC).
"""
import os
import time

REG = dict(CTRL=0x00, TOK=0x04, POS=0x08, STATUS=0x0C, CYCLES=0x10, ARGMAX=0x14, MAXVAL=0x18, ID=0x1C,
           INFO0=0x20, INFO1=0x24, NVOCAB=0x28, LD_SEL=0x30, LD_ADDR=0x34, LD_GO=0x38, LD_DATA=0x40, LOGIT=0x1000)
MEMS = [("prog.mem", 256), ("weights.mem", 64), ("params.mem", 64), ("tables.mem", 32), ("luts.mem", 8)]
ZOO_ID = 0x5A4F4F01


def s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


class Zoo:
    def __init__(self, bitfile="zoo.bit", mmio=None):
        if mmio is None:
            from pynq import MMIO, Overlay
            ol = Overlay(bitfile)
            ip = next(v for k, v in ol.ip_dict.items() if "zoo" in k)
            mmio = MMIO(ip["phys_addr"], ip["addr_range"])
        self.m = mmio
        ident = self.m.read(REG["ID"])
        if ident != ZOO_ID:
            raise RuntimeError(f"zoo engine not found (ID {ident:#x})")
        i0, i1 = self.m.read(REG["INFO0"]), self.m.read(REG["INFO1"])
        self.V, self.T, self.C, self.P, self.NL = i0 >> 16, i0 & 0xFFFF, i1 & 0xFFFF, (i1 >> 16) & 0xFF, i1 >> 24

    # ---------------------------------------------------------------- model loading
    def load(self, model_dir):
        sizes = [int(x) for x in open(os.path.join(model_dir, "sizes.txt")).read().split()]
        n = 0
        t0 = time.perf_counter()
        for sel, ((fname, bits), count) in enumerate(zip(MEMS, sizes)):
            words = [int(x, 16) for x in open(os.path.join(model_dir, fname)).read().split()[:count]]
            nw = max(1, bits // 32)
            self.m.write(REG["LD_SEL"], sel)
            for a, v in enumerate(words):
                for i in range(nw):
                    self.m.write(REG["LD_DATA"] + 4 * i, (v >> (32 * i)) & 0xFFFFFFFF)
                self.m.write(REG["LD_ADDR"], a)
                self.m.write(REG["LD_GO"], 1)
            n += len(words)
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "tv.txt")) as fh:       # model vocabulary -> argmax range
            self.nvocab = int(fh.readline().split()[1])
        self.m.write(REG["NVOCAB"], self.nvocab)
        return dict(words=n, seconds=time.perf_counter() - t0)

    # ---------------------------------------------------------------- inference
    def step(self, tok, pos):
        """one decode step; returns (argmax token, engine clock cycles incl. the argmax scan)"""
        self.m.write(REG["TOK"], tok)
        self.m.write(REG["POS"], pos)
        self.m.write(REG["CTRL"], 1)
        while not self.m.read(REG["STATUS"]) & 2:
            pass
        return self.m.read(REG["ARGMAX"]), self.m.read(REG["CYCLES"])

    def logits(self, n):
        return [s32(self.m.read(REG["LOGIT"] + 4 * k)) for k in range(n)]

    def generate(self, n, start=1):
        toks, cyc = [start], []
        for t in range(min(n, self.T)):
            nxt, c = self.step(toks[-1], t)
            toks.append(nxt)
            cyc.append(c)
        return toks, cyc

    # ---------------------------------------------------------------- golden-vector check
    def check_tv(self, tv_file=None, max_seq=None):
        """run the golden vectors of the loaded model; every logit must match. Returns statistics."""
        tv = open(tv_file or os.path.join(self.model_dir, "tv.txt")).read().split("\n")
        nseq, V = map(int, tv[0].split())
        nseq = min(nseq, max_seq or nseq)
        i, steps, errors, cycles, wall = 1, 0, 0, [], []
        for _ in range(nseq):
            n = int(tv[i])
            for t in range(n):
                row = [int(x) for x in tv[i + 1 + t].split()]
                t0 = time.perf_counter()
                am, c = self.step(row[0], t)
                wall.append(time.perf_counter() - t0)
                lg = self.logits(V)
                errors += sum(a != b for a, b in zip(lg, row[1:]))
                gold = row[1:]
                errors += int(am != gold.index(max(gold)))
                cycles.append(c)
                steps += 1
            i += 1 + n
        return dict(steps=steps, errors=errors, cycles_avg=sum(cycles) / len(cycles), cycles_min=min(cycles),
                    cycles_max=max(cycles), step_wall_us=1e6 * sum(wall) / len(wall))
