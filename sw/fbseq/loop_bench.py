"""
Physical cost of the feedback loop: Fmax of S-bit state machines whose next state is computed by
different transition circuits, placed and routed on the XC7A100T (Yosys + nextpnr-himbaechel).

    python3 loop_bench.py [--seeds 1,2] [--S 120]          -> model/fbseq/loop.json

The tables d_t come from a rotating register (no logic), as they would from a pipelined
feed-forward network; the state is XOR-reduced onto one LED pin with a register after every LUT6, so
every path outside the state loop is a single LUT (until 29 Oct 2026 the reduction had a 2-3 LUT stage
that, not the loop, limited Fmax for loops of one LUT level: see the critical path in each result).
  mux1   s'_j = s_j ? a_j : b_j                    candidate B, kb = 1 (keep / set / toggle)
  mux4   s'_j = d_j[s_blk]   (2-bit blocks)        candidate B, kb = 2 (one LUT6)
  mux8   s'_j = d_j[s_blk]   (3-bit blocks)        candidate B, kb = 3 (LUT6 pair + MUXF7)
  dep1   s'_j = f_j(5 random state bits, d_j)      state-dependent logic, 1 LUT level
  dep2 / dep3 / dep4                               the same with 2 / 3 / 4 LUT levels (GRU-like)
  a5exact  hand-built A5 tracker: per bit LUT5(3 block bits, 2 input bits)  (controls.py a5)
  a5c7     smallest block-local A5 realization: V4 cosets (4 bits, LUT6) + A4 cosets (3 bits, LUT5)
  a5sat6   6-bit A5 realization from the SAT search (theory_sat.py): each bit LUT6(4 state, 2 control)
  gc:NAME  a group-machine code from core_fiber.py --export (model/fbseq/group_codes.json), e.g.
           gc:A5_gauge, gc:PSL(2,7)_dense: each bit one truth table over its exact deps and the w letter bits
"""
import argparse
import json
import os
import random
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
X = os.environ.get("XC7_DIR", os.path.expanduser("~/tools/xc7"))
NEXTPNR = os.path.join(X, "nextpnr", "build", "nextpnr-himbaechel")
PART = "xc7a100tcsg324-1"
XDC = """set_property LOC E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 2.500 -name clk [get_ports clk]
set_property LOC H17 [get_ports led]
set_property IOSTANDARD LVCMOS33 [get_ports led]
"""


def verilog(kind, S, rnd):
    """the benchmark circuit as Verilog text"""
    if kind == "a5sat6":                      # 6-bit A5 realization found by theory_sat.py (model/fbseq/sat_a5.json)
        from theory_sat import a5
        sol = [r for r in json.load(open(os.path.join(ROOT, "model", "fbseq", "sat_a5.json")))
               if r["S"] == 6 and r["result"] == "sat"][0]
        _, step6, _ = a5()
        enc6 = [int(e[::-1], 2) for e in sol["encoding"]]                    # bit b of the code = e[b]
        copies = S // 6
        S, nd = copies * 6, 2 * copies
    elif kind == "a5c7":                        # smallest block-local A5 realization (theory_sat.best_block_local(6))
        from theory_sat import best_block_local, coset_blocks
        blocks7, _ = coset_blocks(best_block_local(6)[2])
        copies = S // 7
        S, nd = copies * 7, 2 * copies
    elif kind == "a5exact":                   # hand-built A5 tracker (controls.a5_tracker): 5 blocks of 3 bits
        from controls import a5_tracker
        tt = a5_tracker()[0]
        copies = S // 15
        S, nd = copies * 15, 2 * copies
    elif kind.startswith("gc:"):              # group-machine code from core_fiber.export
        spec = next(r for r in json.load(open(os.path.join(ROOT, "model", "fbseq", "group_codes.json")))
                    if r["name"] == kind[3:])
        gS, gw = spec["S"], spec["w"]
        copies = S // gS
        S, nd = copies * gS, gw * copies
    elif kind == "mux1":
        nd = 2 * S
    elif kind in ("mux4", "mux8"):
        kb = 2 if kind == "mux4" else 3
        S -= S % kb
        nd = S * 2 ** kb
    else:
        nd = S
    init = "".join(rnd.choice("0123456789abcdef") for _ in range((nd + 3) // 4))
    v = [f"module bench(input clk, output reg led);",
         f"  reg [{nd - 1}:0] d = {nd}'h{init};",
         f"  reg [{S - 1}:0] s = {S}'b" + ("".join(format(j, "03b") for j in reversed(range(5))) * (S // 15)
                                            if kind == "a5exact" else
                                            "".join(format(bl["label"][tuple(range(5))], f"0{bl['bits']}b")
                                                    for bl in reversed(blocks7)) * (S // 7)
                                            if kind == "a5c7" else
                                            format(spec["start"], f"0{gS}b") * copies
                                            if kind.startswith("gc:") else "0") + ";",    # a5sat6: e(identity) = 0
         f"  always @(posedge clk) d <= {{d[{nd - 2}:0], d[{nd - 1}]}};      // tables: no logic"]
    nxt = []
    if kind == "a5sat6":
        code = {c: p for p, c in enumerate(enc6)}
        for c in range(S // 6):
            for j, dep in enumerate(sol["deps"]):
                tt = [0] * 64                                                 # index: {dep bits, x1, x0}
                for cur, p in code.items():
                    for x in range(4):
                        nxt_bit = ((enc6[step6[x][p]] >> j) & 1) if x < 3 else ((cur >> j) & 1)
                        idx = 0
                        for i in dep:
                            idx = (idx << 1) | ((cur >> i) & 1)
                        tt[(idx << 2) | x] = nxt_bit
                v.append(f"  localparam [63:0] TS{c}_{j} = 64'b" + "".join(str(t) for t in reversed(tt)) + ";")
                sel = "{" + ",".join(f"s[{6 * c + i}]" for i in dep) + f",d[{2 * c + 1}],d[{2 * c}]" + "}"
                nxt.append(f"TS{c}_{j}[{sel}]")
    elif kind.startswith("gc:"):
        code = {c: p for p, c in enumerate(spec["codes"])}
        for c in range(copies):
            for j, dep in enumerate(spec["deps"]):
                m = len(dep) + gw
                tt = [0] * 2 ** m                                             # index: {dep bits, letter bits}
                for idx in range(2 ** m):
                    x, st = idx & (2 ** gw - 1), idx >> gw
                    cur = None
                    for cc, p in code.items():                                # a reachable code with these dep bits
                        if all(((cc >> i) & 1) == ((st >> (len(dep) - 1 - q)) & 1) for q, i in enumerate(dep)):
                            cur = cc
                            break
                    if cur is None:
                        continue                                              # unreachable pattern: don't care (0)
                    p = code[cur]
                    x = min(x, spec["letters"] - 1)                         # unused letter codes = identity (last)
                    tt[idx] = (spec["codes"][spec["step"][p][x]] >> j) & 1
                name = f"G{c}_{j}"
                v.append(f"  localparam [{2 ** m - 1}:0] {name} = {2 ** m}'b" + "".join(str(t) for t in reversed(tt)) + ";")
                sel = "{" + ",".join([f"s[{gS * c + i}]" for i in dep] + [f"d[{gw * c + b}]" for b in reversed(range(gw))]) + "}"
                nxt.append(f"{name}[{sel}]" if dep or gw else name)
    elif kind == "a5c7":
        for c in range(S // 7):
            off = 7 * c
            for bl in blocks7:
                nb = bl["bits"]
                Q = 2 ** (nb + 2)
                for k in range(nb):                  # truth table over (block bits, control): unused codes hold
                    tt = []
                    for idx in range(Q):
                        st, x = idx >> 2, idx & 3
                        nx = bl["next"][x][st] if (st < bl["index"] and x < 3) else st
                        tt.append((nx >> k) & 1)
                    name = f"T{c}_{off}_{k}"
                    v.append(f"  localparam [{Q - 1}:0] {name} = {Q}'b" + "".join(str(t) for t in reversed(tt)) + ";")
                    sel = "{" + ",".join(f"s[{off + i}]" for i in reversed(range(nb))) + f",d[{2 * c + 1}],d[{2 * c}]" + "}"
                    nxt.append(f"{name}[{sel}]")
                off += nb
    elif kind == "a5exact":
        for k in range(3):
            v.append(f"  localparam [31:0] TA{k} = 32'b" + "".join(str(tt[k][i]) for i in reversed(range(32))) + ";")
        for blk in range(S // 3):
            c = blk // 5
            idx = "{" + ",".join(f"s[{blk * 3 + i}]" for i in (2, 1, 0)) + f",d[{2 * c + 1}],d[{2 * c}]" + "}"
            nxt += [f"TA{k}[{idx}]" for k in range(3)]
    elif kind == "mux1":
        nxt = [f"s[{j}] ? d[{2 * j}] : d[{2 * j + 1}]" for j in range(S)]
    elif kind in ("mux4", "mux8"):
        Q = 2 ** kb
        for j in range(S):
            blk = j // kb
            sel = "{" + ",".join(f"s[{blk * kb + i}]" for i in reversed(range(kb))) + "}"
            v.append(f"  wire [{Q - 1}:0] t{j} = d[{(j + 1) * Q - 1}:{j * Q}];")
            nxt.append(f"t{j}[{sel}]")
    else:
        levels = int(kind[3:])
        prev = [f"s[{i}]" for i in range(S)]
        for lev in range(levels):
            cur = []
            for j in range(S):
                ins = rnd.sample(prev, 5) + ([f"d[{j}]"] if lev == 0 else [rnd.choice(prev)])
                tt = "".join(rnd.choice("0123456789abcdef") for _ in range(16))
                name = f"w{lev}_{j}"
                v.append(f"  localparam [63:0] T{name} = 64'h{tt};")
                v.append(f"  wire {name} = T{name}[{{{','.join(ins)}}}];")
                cur.append(name)
            prev = cur
        nxt = prev
    v.append("  always @(posedge clk) begin")
    v += [f"    s[{j}] <= {e};" for j, e in enumerate(nxt)]
    v.append("  end")
    level, stage = [f"s[{i}]" for i in range(S)], 1                         # XOR reduction, one LUT6 per
    v.append("  always @(posedge clk) begin")                                 # registered stage, so that only
    body = []                                                                 # the state loop can be deeper
    while len(level) > 6:
        g = [level[i:i + 6] for i in range(0, len(level), 6)]
        v.insert(len(v) - 1, f"  reg [{len(g) - 1}:0] x{stage} = 0;")
        body += [f"    x{stage}[{k}] <= " + " ^ ".join(grp) + ";" for k, grp in enumerate(g)]
        level, stage = [f"x{stage}[{k}]" for k in range(len(g))], stage + 1
    v += body
    v.append("    led <= " + " ^ ".join(level) + ";")
    v.append("  end")
    v.append("endmodule")
    return "\n".join(v) + "\n", S


def critical_path(report):
    """logic on the worst path of a nextpnr report: LUT and MUXF7/F8 stages, logic and routing delay (ns),
    and whether it starts and ends at a state register s[] (the feedback loop) or elsewhere"""
    p = report["critical_paths"][0]
    seg = p["path"]
    logic = [x for x in seg if x["type"] == "logic"]
    mux = sum(1 for x in logic if ".mux" in x["to"]["cell"] or "MUXF" in x["to"]["cell"])
    nets = [x.get("net", "") for x in seg if x["type"] == "routing"]
    return dict(luts=len(logic) - mux, muxf=mux, logic_ns=round(sum(x["delay"] for x in logic), 3),
                route_ns=round(sum(x["delay"] for x in seg if x["type"] == "routing"), 3),
                first_net=nets[0] if nets else "", feedback=bool(nets) and nets[0].startswith("s["))


def run(kind, S, seed):
    out = os.path.join(ROOT, "build", "fbseq", "loop", f"{kind.replace(':', '_').replace('(', '').replace(')', '').replace(',', '')}_s{seed}")
    os.makedirs(out, exist_ok=True)
    text, S = verilog(kind, S, random.Random(1000 + seed))
    open(os.path.join(out, "bench.v"), "w").write(text)
    open(os.path.join(out, "bench.xdc"), "w").write(XDC)
    subprocess.run(["yosys", "-q", "-l", os.path.join(out, "synth.log"), "-p",
                    f"read_verilog {out}/bench.v; synth_xilinx -family xc7 -top bench -flatten -nodsp; "
                    f"tee -q -o {out}/stat.txt stat; write_json {out}/bench.json"], check=True)
    chipdb = subprocess.run(["find", os.path.join(X, "nextpnr", "build"), "-name", "chipdb-xc7a100t.bin"],
                            capture_output=True, text=True).stdout.split()[0]
    subprocess.run([NEXTPNR, "--device", PART, "--chipdb", chipdb, "--json", f"{out}/bench.json", "--top", "bench",
                    "--vopt", f"xdc={out}/bench.xdc", "--report", f"{out}/report.json", "--timing-allow-fail",
                    "--seed", str(seed), "--log", f"{out}/pnr.log", "-q"], check=True)
    rep = json.load(open(f"{out}/report.json"))
    fmax = min(v["achieved"] for v in rep["fmax"].values())
    stat = open(f"{out}/stat.txt").read()
    cells = {}
    for line in stat.splitlines():
        p = line.split()
        if len(p) == 2:
            name, n = (p[1], p[0]) if p[0].isdigit() else (p[0], p[1])
            if name.startswith(("LUT", "MUXF", "FD", "CARRY")) and n.isdigit():
                cells[name] = int(n)
    return dict(kind=kind, S=S, seed=seed, fmax_mhz=round(fmax, 1), cells=cells, path=critical_path(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--S", type=int, default=120)
    ap.add_argument("--kinds", default="mux1,mux4,mux8,dep1,dep2,dep3,dep4", help="separated by ';' or ','")
    ap.add_argument("--out", default="loop.json")
    a = ap.parse_args()
    res = []
    for kind in (a.kinds.split(";") if ";" in a.kinds else a.kinds.split(",")):
        for seed in (int(s) for s in a.seeds.split(",")):
            r = run(kind, a.S, seed)
            print(json.dumps(r), flush=True)
            res.append(r)
    os.makedirs(os.path.join(ROOT, "model", "fbseq"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "model", "fbseq", a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
