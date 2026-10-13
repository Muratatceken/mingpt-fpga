"""
Numbers, tables and plot data of the paper (paper/data/). Every number in the paper comes from here.
  python3 paper_data.py
Inputs (model/zoo/): runs/<tag>/*.json (training), pipeline.json (INT8 + RTL per run), hw_final.json
(Yosys per final model), dse.json, fuzz.json, quant_study.json, int_metrics.json, hw_metrics.json, and --
once measured -- vivado_sweep.jsonl, board_results.json, power.json, board/pynq_z2/overlay/overlay.json.
Missing measurements are typeset as a visible \\tbd{}, never as invented numbers.
"""
import glob
import json
import math
import os
import statistics as st

from data import ROOT

MZ = os.path.join(ROOT, "model", "zoo")
OUT = os.path.join(ROOT, "paper", "data")
FAMILY = ["bigram", "mlp", "rnn", "gru", "bow", "gpt2", "microgpt", "llama", "nanochat"]
LADDER = ["lad0_gpt2", "lad1_rms", "lad2_rope", "lad3_swiglu", "lad4_gqa", "lad5_llama", "lad6_relu2", "lad7_qknorm"]
LABEL = dict(bigram="Bigram", mlp="MLP", rnn="RNN", gru="GRU", bow="BoW", gpt2="GPT-2", microgpt="microgpt",
             llama="Llama~2", nanochat="nanochat", lad0_gpt2="L0 GPT-2", lad1_rms="L1 +RMSNorm",
             lad2_rope="L2 +RoPE", lad3_swiglu="L3 +SwiGLU", lad4_gqa="L4 +GQA", lad5_llama="L5 Llama~2",
             lad6_relu2="L6 ReLU$^2$", lad7_qknorm="L7 +QK-norm")
TBD = r"\tbd{}"
GROUPS = [("GEMV", ["GEMV"]), ("NORM", ["NORM"]), ("ATTN", ["ATTN", "KVSTORE"]), ("ROPE", ["ROPE"]),
          ("EW", ["MUL", "GRUUPD", "COPY", "REQ", "BOW"]), ("EMB", ["EMB16", "EMB8", "EMBWIN", "TABLOGIT", "HRESET"]),
          ("FETCH", ["FETCH"])]


def js(name, default=None):
    p = name if os.path.isabs(name) else os.path.join(MZ, name)
    return json.load(open(p)) if os.path.exists(p) else default


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def num(v, d=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return TBD
    return f"{v:,.{d}f}".replace(",", "{,}")


def sgn(v, d=3):
    return TBD if v is None else ("+" if v >= 0 else "$-$") + num(abs(v), d)


def pm(xs, d=3):
    if not xs:
        return TBD
    return num(xs[0], d) if len(xs) == 1 else f"{num(st.mean(xs), d)}\\,$\\pm$\\,{num(st.stdev(xs), d)}"


def mean(xs):
    return st.mean(xs) if xs else None


def runs(tag):
    """per architecture: best learning rate (validation loss, seed 0) and all runs at that rate"""
    rs = [json.load(open(f)) for f in glob.glob(os.path.join(MZ, "runs", tag, "*.json"))]
    out = {}
    for arch in sorted({r["arch"] for r in rs}):
        a = [r for r in rs if r["arch"] == arch]
        s0 = [r for r in a if r["seed"] == 0]
        best = min(s0, key=lambda r: r["val_loss"])["lr"] if s0 else a[0]["lr"]
        sel = sorted([r for r in a if r["lr"] == best], key=lambda r: r["seed"])
        out[arch] = dict(lr=best, test=[r["test_loss"] for r in sel], params=sel[0]["params"], runs=sel,
                         keys=[f"{tag}/{os.path.basename(r['path'])}" for r in sel])
    return out


def pipe_of(pipe, info, field):
    """values of a pipeline field over the selected runs of one architecture"""
    if not info:
        return []
    return [pipe[k][field] for k in info["keys"] if k in pipe and field in pipe[k]]


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    pipe, fuzz = js("pipeline.json", {}), js("fuzz.json", {})
    hwf = js("hw_final.json", {})                      # Yosys of the final (best-LR, seed-0) models
    n50, n12, n200, sto = runs("names50k"), runs("names12k"), runs("names200k"), runs("stories260k")
    mac = {}

    # ------------------------------------------------------------------ verification
    ok = [p for p in pipe.values() if p.get("rtl_errors") == 0]
    mac["zooVerifRuns"] = num(len(ok)) if ok else TBD
    mac["zooVerifiedLogits"] = num(sum(p.get("rtl_logits", 0) for p in ok)) if ok else TBD
    mac["zooVerifFailures"] = num(sum(1 for p in pipe.values() if p.get("rtl_errors", 0) != 0 or "rtl_error" in p))
    fc = fuzz.get("cases", [])
    mac["zooFuzzCases"] = num(sum(c["status"] == "pass" for c in fc)) if fc else TBD
    mac["zooFuzzLogits"] = num(sum(c.get("logits", 0) for c in fc)) if fc else TBD
    mac["zooArithCases"] = num(36421426)                # tb/tb_zarith.sv (fixed test set)
    board = js("board_results.json")
    mac["zooBoardVerif"] = (f"{len(board['models'])} models, "
                            f"{sum(m.get('errors', 0) for m in board['models'].values())} mismatches") if board else TBD
    mac["zooIrErr"] = "$10^{-4}$"
    mac["zooRepoLink"] = r"\url{https://github.com/Muratatceken/mingpt-fpga}"
    mac["zooBudgetTol"] = r"3\%"
    seeds = [len(v["runs"]) for v in n50.values()]
    mac["zooNamesSeeds"] = str(max(seeds)) if seeds else TBD
    if sto:
        r0 = next(iter(sto.values()))["runs"][0]
        mac["zooStoriesSteps"] = num(r0["steps"])
        mac["zooStoriesTokensM"] = num(r0["steps"] * r0["batch"] * 256 / 1e6)
    else:
        mac["zooStoriesSteps"] = mac["zooStoriesTokensM"] = TBD

    # ------------------------------------------------------------------ table: family
    rows = []
    for a in FAMILY:
        r, s = n50.get(a), sto.get(a)
        cyc = pipe_of(pipe, r, "cycles")
        scyc = pipe_of(pipe, s, "cycles")
        h = hwf.get(f"names50k/{a}", {}).get("spec", {})
        rows.append(" & ".join([LABEL[a], pm(r["test"]) if r else TBD, sgn(mean(pipe_of(pipe, r, "int_delta")), 4),
                                num(mean(cyc)), pm(s["test"]) if s else TBD, num(mean(scyc)),
                                num(h.get("lut")), num(h.get("dsp"))]) + r" \\")
    write("tab_family.tex", rows)

    # ------------------------------------------------------------------ table + figure: ladder
    rows, fig = [], ["step names dnames stories dstories cycles dcycles dsp lut"]
    base = {}
    for i, a in enumerate(LADDER):
        r, s = n50.get(a), sto.get(a)
        nl, sl = mean(r["test"]) if r else None, mean(s["test"]) if s else None
        cyc = mean(pipe_of(pipe, r, "cycles"))
        h = hwf.get(f"names50k/{a}", {}).get("spec", {})
        if i == 0:
            base = dict(n=nl, s=sl, c=cyc, dsp=h.get("dsp"), lut=h.get("lut"))
        d = lambda x, b: None if x is None or b is None else x - b
        rows.append(" & ".join([LABEL[a], pm(r["test"]) if r else TBD, sgn(d(nl, base.get("n"))),
                                num(sl, 3), sgn(d(sl, base.get("s"))), num(cyc),
                                sgn(d(cyc, base.get("c")), 0), num(h.get("dsp")), num(h.get("lut"))]) + r" \\")
        fig.append(" ".join(str(x if x is not None else "nan") for x in
                            [i, nl, d(nl, base.get("n")), sl, d(sl, base.get("s")), cyc, d(cyc, base.get("c")),
                             h.get("dsp"), h.get("lut")]))
    write("tab_ladder.tex", rows)
    write("fig_ladder.dat", fig)

    # ------------------------------------------------------------------ cycle profile (names 50k, seed 0)
    prof = {}
    for a in FAMILY:
        p = pipe.get(n50[a]["keys"][0]) if a in n50 else None
        if p and "profile" in p:
            prof[a] = {g: sum(p["profile"].get(k, 0.0) for k in ks) for g, ks in GROUPS}
    order = [a for a in reversed(FAMILY) if a in prof]
    write("fig_profile.tex", [f"\\addplot[zoo{g.lower()}] coordinates {{" +
                              " ".join(f"({prof[a][g]:.1f},{k + 1})" for k, a in enumerate(order)) + "};"
                              for g, _ in GROUPS] if order else ["% no data"])
    write("fig_profile_labels.tex", ["\\def\\zooproflabels{" + ",".join(LABEL[a] for a in order) + "}",
                                     f"\\def\\zooprofn{{{max(1, len(order))}}}"])

    # ------------------------------------------------------------------ context: per-position loss and cycles (stories)
    for a, info in sto.items():
        p = pipe.get(info["keys"][0], {})
        if "pos_loss" in p:
            write(f"pos_{a}.dat", ["t loss cycles"] + [
                f"{t} {l} {p['pos_cycles'][t] if t < len(p.get('pos_cycles', [])) else 'nan'}"
                for t, l in enumerate(p["pos_loss"])])

    # ------------------------------------------------------------------ scale
    COL = dict(mlp="orange!60", rnn="red!60", gru="red!85!black", bow="black!40", gpt2="blue!45",
               microgpt="cyan!60!black", llama="blue!80!black", nanochat="green!50!black")
    plots = []
    for a in FAMILY[1:]:
        pts = [(rr[a]["params"], mean(rr[a]["test"])) for rr in (n12, n50, n200) if a in rr]
        if pts:
            plots += [f"\\addplot[color={COL[a]}, mark=*] coordinates {{" +
                      " ".join(f"({p},{l:.4f})" for p, l in pts) + "};", f"\\addlegendentry{{{LABEL[a]}}}"]
    write("fig_scale.tex", plots or ["\\addplot coordinates {(1e4,2) (1e5,2)}; % no data yet"])

    # ------------------------------------------------------------------ context plots (TinyStories)
    ctx = [("mlp", "orange!60"), ("bow", "black!40"), ("rnn", "red!60"), ("gru", "red!85!black"),
           ("gpt2", "blue!45"), ("llama", "blue!80!black"), ("nanochat", "green!50!black")]
    lp = [x for a, c in ctx if os.path.exists(os.path.join(OUT, f"pos_{a}.dat"))
          for x in (f"\\addplot[color={c}] table[x=t, y=loss] {{data/pos_{a}.dat}};",
                    f"\\addlegendentry{{{LABEL[a]}}}")]
    cp = [f"\\addplot[color={c}] table[x=t, y=cycles] {{data/pos_{a}.dat}};" for a, c in ctx
          if a in ("gru", "gpt2", "llama", "nanochat") and os.path.exists(os.path.join(OUT, f"pos_{a}.dat"))]
    write("fig_context_loss.tex", lp or ["\\addplot coordinates {(0,2) (255,2)}; % no data yet"])
    write("fig_context_cycles.tex", cp or ["\\addplot coordinates {(0,0) (255,0)}; % no data yet"])

    # ------------------------------------------------------------------ design space P
    dse = js("dse.json", {})
    lines = ["model P cycles gemv lut dsp bram"]
    for k, e in sorted(dse.items()):
        y = e["yosys"]
        lines.append(f"{e['model']} {e['P']} {e['cycles']} {e['gemv']} {y['lut']} {y['dsp']} {y['bram']}")
    write("dse.dat", lines)

    # ------------------------------------------------------------------ quantization study
    qs = js("quant_study.json", {})
    cats = ["weights", "o8", "act", "mul", "norm", "resid", "logit", "all"]
    lab = dict(weights="weights (INT8, per row)", o8="$q,k,v$ / linear outputs", act="activation tables",
               mul="SwiGLU product", norm="normalized vectors", resid="residual (INT16)", logit="logits (INT16)",
               all="all together")
    ms = [m for m in ("gpt2", "llama", "stories260k") if m in qs]
    rows = []
    for c in cats:
        v = [sgn(qs[m]["ablation"][c] - qs[m]["ablation"]["float"], 4) if c in qs[m]["ablation"] else TBD for m in ms]
        rows.append(("\\midrule " if c == "all" else "") + lab[c] + " & " + " & ".join(v) + r" \\")
    write("tab_ablation.tex", rows)
    if "stories260k" in qs:
        write("tab_calib.tex", [f"{'max' if c['pct'] == 0 else ('%g' % c['pct'])} & "
                                f"{'bound' if c['norm'] == 'bound' else 'calibrated'} & {num(c['int_nll'], 4)} & "
                                f"{sgn(c['delta'], 4)} & {num(100 * c['argmax_agree'], 1)} \\\\"
                                for c in qs["stories260k"]["calibration"]])

    # ------------------------------------------------------------------ one-bitstream engine, Vivado, board, energy
    hw = js("hw_metrics.json", {})
    uni = hw.get("universal", {}).get("yosys")
    for k in ("lut", "ff", "bram", "dsp"):
        mac[f"zooUni{k.capitalize()}"] = num(uni.get(k), 1 if k == "bram" else 0) if uni else TBD
    viv = {(e["model"], e["config"]): e for e in jsonl(os.path.join(MZ, "vivado_sweep.jsonl"))}
    ov = js(os.path.join(ROOT, "board", "pynq_z2", "overlay", "overlay.json"))
    mac["zooFmax"] = num(viv.get(("universal", "gen"), {}).get("fmax_mhz"), 0) if viv else TBD
    mac["zooClockMHz"] = num(ov.get("clock_mhz"), 0) if ov else TBD
    power = js("power.json")
    rows = []
    if board:
        f_hz = (ov or {}).get("clock_mhz", 100) * 1e6
        for m, r in board["models"].items():
            us = r["cycles_avg"] / f_hz * 1e6
            cpu = r.get("cpu", {}).get("us_per_token")
            e_f = e_c = None
            if power:
                e_f = (power["fpga_run_w"] - power["pl_idle_w"]) * us          # uJ per token (engine only)
                e_c = (power["cpu_run_w"] - power["idle_w"]) * cpu if cpu else None
            rows.append(" & ".join([m.replace("_", r"\_"), num(us, 1), num(cpu, 1),
                                    num(cpu / us, 1) if cpu else TBD, num(e_f, 2), num(e_c, 2)]) + r" \\")
    write("tab_board.tex", rows or [r"\multicolumn{6}{c}{\tbd{} (on-board measurements pending)} \\"])

    mac["zooAbstractFinding"] = mac.get("zooAbstractFinding", "")
    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("% GENERATED by sw/zoo/paper_data.py -- do not edit\n")
        fh.write("\\providecommand{\\tbd}{\\textcolor{red}{\\textbf{TBD}}}\n")
        for k, v in sorted(mac.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("paper data written to", OUT, "|", len(pipe), "pipeline entries,", len(n50), "archs at 50k,",
          len(sto), "archs on stories")


if __name__ == "__main__":
    main()
