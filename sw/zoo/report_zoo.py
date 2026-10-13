"""
Tables, macros and plot data for the model-family results part of the PDF report.
  python3 report_zoo.py
Reads  model/zoo/{metrics,int_metrics,hw_metrics,quant_study}.json, build/zoo/*.qprog.pkl,
       build/zoo/stories260k_gen_rtl.txt
Writes report/data/zoo_*.tex|dat   (all numbers in the text come from here)
"""
import json
import os
import pickle

import numpy as np

from data import ROOT

OUT = os.path.join(ROOT, "report", "data")
MZ = os.path.join(ROOT, "model", "zoo")
NAMES = ["bigram", "mlp", "rnn", "gru", "bow", "gpt2", "microgpt", "llama", "nanochat"]
LABEL = dict(bigram="Bigram", mlp="MLP", rnn="RNN", gru="GRU", bow="BoW", gpt2="GPT-2", microgpt="microgpt",
             llama="Llama~2", nanochat="nanochat", stories260k="stories260K")
MACRO = dict(bigram="Bigram", mlp="Mlp", rnn="Rnn", gru="Gru", bow="Bow", gpt2="Gpt", microgpt="Micro",
             llama="Llama", nanochat="Nano", stories260k="Stories")
GROUPS = [("GEMV", ["GEMV"]), ("NORM", ["NORM"]), ("ATTN", ["ATTN", "KVSTORE"]), ("ROPE", ["ROPE"]),
          ("EW", ["MUL", "GRUUPD", "COPY", "REQ", "BOW"]), ("EMB", ["EMB16", "EMB8", "EMBWIN", "TABLOGIT", "HRESET"]),
          ("FETCH", ["FETCH"])]
P = 8


def js(name):
    p = os.path.join(MZ, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def num(v, d=0):
    """Turkish number: thin-space thousands, decimal comma (for text and tables)."""
    s = f"{v:,.{d}f}".replace(",", "X").replace(".", "{,}").replace("X", "\\,")
    return s.replace("-", "$-$") if v < 0 else s


def sgn(v, d=4):
    return ("+" if v >= 0 else "$-$") + num(abs(v), d)


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def qprog(name):
    p = os.path.join(ROOT, "build", "zoo", f"{name}.qprog.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def hw_facts(q):
    """GEMV MACs/token, ideal GEMV cycles, INT8 weight bytes, state bytes per context position / fixed."""
    macs = sum(L["W"].size for L in q.lin.values())
    ideal = sum(L["W"].shape[1] * -(-L["W"].shape[0] // P) for L in q.lin.values())
    wbytes = macs + sum(L["b"].size * 4 + L["W"].shape[0] * 4 for L in q.lin.values())  # + bias, m0/sh
    tbytes = sum(t.size * 2 for t in q.tab.values())
    per_pos = fixed = 0
    for o in q.ops:
        if o["op"] == "KVSTORE":
            per_pos += 2 * q.lin[next(x["lin"] for x in q.ops if x["op"] == "GEMV" and x["dst"] == o["k"])]["W"].shape[0]
        if o["op"] == "HRESET":
            fixed += len(q.tab[o["table"]])
        if o["op"] == "BOW":
            fixed += 4 * q.tab[next(x["tok"] for x in q.ops if x["op"] == "EMB16")].shape[1]
        if o["op"] == "EMBWIN":
            fixed += o["k"]                                   # token window
    attn_layers = [o for o in q.ops if o["op"] == "ATTN"]
    att_mac_per_t = sum(2 * o["nh"] * o["dh"] for o in attn_layers)   # q.k and e.v per cached position
    return dict(macs=macs, ideal=ideal, wbytes=wbytes, tbytes=tbytes, per_pos=per_pos, fixed=fixed,
                att_mac_per_t=att_mac_per_t, n_instr=len(q.ops) + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, im, hw, qs = js("metrics.json"), js("int_metrics.json"), js("hw_metrics.json"), js("quant_study.json")
    models = [n for n in NAMES + ["stories260k"] if n in im and n in hw]
    mac = []          # \newcommand macros
    rows_q, rows_h, rows_r = [], [], []
    prof_rows = {}
    for n in models:
        q = qprog(n)
        f = hw_facts(q)
        I, H = im[n], hw[n]
        params = tr.get(n, {}).get("params", I.get("params"))
        cyc = H["cycles_avg"]
        gemv = H["profile"].get("GEMV", 0.0)
        eff = f["ideal"] / gemv if gemv else float("nan")
        M = MACRO[n]
        mac += [f"\\newcommand{{\\zoo{M}Cyc}}{{{num(cyc)}}}",
                f"\\newcommand{{\\zoo{M}Us}}{{{num(cyc / 100, 1)}}}",
                f"\\newcommand{{\\zoo{M}Float}}{{{num(I['float_nll'], 4)}}}",
                f"\\newcommand{{\\zoo{M}Int}}{{{num(I['int_nll'], 4)}}}",
                f"\\newcommand{{\\zoo{M}Delta}}{{{sgn(I['delta'])}}}",
                f"\\newcommand{{\\zoo{M}Agree}}{{{num(100 * I['argmax_agree'], 1)}}}",
                f"\\newcommand{{\\zoo{M}Params}}{{{num(params)}}}",
                f"\\newcommand{{\\zoo{M}Eff}}{{{num(100 * eff, 1)}}}",
                f"\\newcommand{{\\zoo{M}Steps}}{{{num(H['steps'])}}}",
                f"\\newcommand{{\\zoo{M}CycMin}}{{{num(H['cycles_min'])}}}",
                f"\\newcommand{{\\zoo{M}CycMax}}{{{num(H['cycles_max'])}}}"]
        for key, tag in (("yosys", "Gen"), ("yosys_spec", "Spec")):
            y = H.get(key)
            if y:
                mac += [f"\\newcommand{{\\zoo{M}{tag}Lut}}{{{num(y['lut'])}}}",
                        f"\\newcommand{{\\zoo{M}{tag}Dsp}}{{{num(y['dsp'])}}}",
                        f"\\newcommand{{\\zoo{M}{tag}Bram}}{{{num(y['bram'], 1)}}}"]
        if "gen_steps" in H:
            mac += [f"\\newcommand{{\\zoo{M}GenSteps}}{{{num(H['gen_steps'])}}}",
                    f"\\newcommand{{\\zoo{M}GenMatch}}{{{num(H['gen_match'])}}}",
                    f"\\newcommand{{\\zoo{M}GenCyc}}{{{num(H['gen_cycles_avg'])}}}"]
        lab = LABEL[n]
        rows_q.append(f"{lab} & {num(params)} & {num(I['float_nll'], 4)} & {num(I['int_nll'], 4)} & "
                      f"{sgn(I['delta'])} & {num(100 * I['argmax_agree'], 1)} \\\\")
        state = (f"{num(f['per_pos'])}\\,B/konum" if f["per_pos"] else "") + \
                (" + " if f["per_pos"] and f["fixed"] else "") + (f"{num(f['fixed'])}\\,B" if f["fixed"] else "")
        rows_h.append(f"{lab} & {f['n_instr']} & {num(f['macs'])} & {num(cyc)} & {num(cyc / 100, 1)} & "
                      f"{num(100 * eff, 1) if gemv else '--'} & {state or '--'} \\\\")
        y, ys = H.get("yosys"), H.get("yosys_spec")
        if y and ys:
            rows_r.append(f"{lab} & {num(y['lut'])} & {num(y['ff'])} & {num(y['dsp'])} & {num(ys['lut'])} & "
                          f"{num(ys['ff'])} & {num(ys['dsp'])} & {num(y['bram'], 1)} \\\\")
        prof_rows[n] = {g: sum(H["profile"].get(k, 0.0) for k in ks) for g, ks in GROUPS}

    w = lambda fn, lines: open(os.path.join(OUT, fn), "w").write("\n".join(lines) + "\n")
    effs = [hw_facts(qprog(m))["ideal"] / hw[m]["profile"]["GEMV"] for m in models
            if m in NAMES and hw[m]["profile"].get("GEMV")]
    mac += [f"\\newcommand{{\\zooEffMin}}{{{num(100 * min(effs), 1)}}}",
            f"\\newcommand{{\\zooEffMax}}{{{num(100 * max(effs), 1)}}}"]
    if "stories260k" in hw and "gen_cycles_avg" in hw["stories260k"]:
        mac.append(f"\\newcommand{{\\zooStoriesTps}}{{{num(1e8 / hw['stories260k']['gen_cycles_avg'])}}}")
    if "stories260k" in im:
        mac += [f"\\newcommand{{\\zooStoriesSame}}{{{num(im['stories260k']['gen_same_prefix'])}}}",
                f"\\newcommand{{\\zooStoriesTokens}}{{{num(im['stories260k']['tokens'])}}}"]
    have = {x.split("}")[0].split("{")[1] for x in mac}
    need = [f"\\zoo{MACRO[m]}{f}" for m in NAMES + ["stories260k"]
            for f in ["Cyc", "Us", "Float", "Int", "Delta", "Agree", "Params", "Eff", "Steps", "CycMin", "CycMax",
                      "GenLut", "GenDsp", "GenBram", "SpecLut", "SpecDsp", "SpecBram"]]
    need += ["\\zooStoriesGenSteps", "\\zooStoriesGenMatch", "\\zooStoriesGenCyc", "\\zooStoriesTps",
             "\\zooStoriesSame", "\\zooStoriesTokens", "\\zooEffMin", "\\zooEffMax"]
    missing = [k for k in need if k not in have]
    if missing:
        print("WARNING: missing report numbers:", " ".join(missing))
    mac += [f"\\newcommand{{{k}}}{{\\textbf{{??}}}}" for k in missing]
    w("zoo_sayilar.tex", ["% GENERATED by sw/zoo/report_zoo.py"] + mac + [
        f"\\newcommand{{\\zooNModels}}{{{len([m for m in models if m in NAMES])}}}"])
    split = lambda rows: rows[:-1] + ["\\midrule", rows[-1]] if models[-1] == "stories260k" else rows
    w("zoo_kalite.tex", split(rows_q))
    w("zoo_donanim.tex", split(rows_h))
    w("zoo_kaynak.tex", split(rows_r) if len(rows_r) == len(models) else rows_r)

    # cycle profile: stacked bars (bottom model first)
    order = list(reversed([m for m in models if m in NAMES]))
    lines = []
    for g, _ in GROUPS:
        co = " ".join(f"({prof_rows[m][g]:.1f},{k + 1})" for k, m in enumerate(order))
        lines.append(f"\\addplot[zoo{g.lower()}] coordinates {{{co}}};")
    w("zoo_profil_veri.tex", lines)
    w("zoo_profil_etiket.tex", ["\\def\\zooproflabels{" + ",".join(LABEL[m] for m in order) + "}"])
    anchor = dict(mlp="west", rnn="south east", gru="north west", bow="west", gpt2="south east",
                  microgpt="north", llama="south west", nanochat="north")
    w("zoo_pareto.dat", ["x y label anchor"] + [f"{hw[m]['cycles_avg']} {im[m]['int_nll']:.4f} "
                                                 f"{LABEL[m]} {{{anchor[m]}}}"
                                                 for m in models if m in NAMES and m != "bigram"])
    for m in NAMES:
        if m in tr:
            w(f"zoo_egitim_{m}.dat", ["step test"] + [f"{s} {te}" for s, _, te in tr[m]["curve"]])

    # quantisation study
    for m, r in qs.items():
        a = r["ablation"]
        base = a["float"]
        M = MACRO[m]
        cname = dict(weights="Weights", o8="Lin", act="Act", mul="Mul", norm="Norm", resid="Resid", logit="Logit",
                     all="All")                    # TeX control sequences: letters only
        rows = [f"\\newcommand{{\\zoo{M}Abl{cname[c]}}}{{{sgn(a[c] - base)}}}" for c in a if c != "float"]
        rows += [f"\\newcommand{{\\zoo{M}AblFloat}}{{{num(base, 4)}}}"]
        open(os.path.join(OUT, "zoo_sayilar.tex"), "a").write("\n".join(rows) + "\n")
        cal = [f"{'maks.' if c['pct'] == 0 else ('%g' % c['pct']).replace('.', '{,}')} & "
               f"{'sınır' if c['norm'] == 'bound' else 'kalibre'} & {num(c['int_nll'], 4)} & {sgn(c['delta'])} & "
               f"{num(100 * c['argmax_agree'], 1)} \\\\" for c in r["calibration"]]
        w(f"zoo_kalib_{m}.tex", cal)
    if qs:
        cats = ["weights", "o8", "act", "mul", "norm", "resid", "logit", "all"]
        lab = dict(weights="ağırlıklar (satır başına INT8)", o8="$q,k,v$ ve doğrusal çıkışlar (INT8)",
                   act="aktivasyon tablosu giriş/çıkışı", mul="eleman bazlı çarpım (SwiGLU)",
                   norm="normalize edilmiş vektörler", resid="artık akış (INT16)", logit="logitler (INT16)",
                   all="hepsi birlikte")
        ms = [m for m in ["gpt2", "llama", "stories260k"] if m in qs]
        rows = []
        for c in cats:
            vals = ["--" if c == "mul" and not any(o["op"] == "MUL" for o in qprog(m).ops) else
                    sgn(qs[m]["ablation"][c] - qs[m]["ablation"]["float"]) for m in ms]
            rows.append(("\\midrule " if c == "all" else "") + f"{lab[c]} & " + " & ".join(vals) + " \\\\")
        w("zoo_ablasyon.tex", rows)
        w("zoo_ablasyon_bas.tex", [" & ".join(["\\thead{Nicemlenen tensör}"] + [f"\\thead{{{LABEL[m]}}}" for m in ms])])

    # the RTL's story
    g = os.path.join(ROOT, "build", "zoo", "stories260k_gen_rtl.txt")
    if os.path.exists(g):
        from stories import Tokenizer
        toks = [int(x) for x in open(g).read().split()]
        txt = Tokenizer().decode(toks)
        paras = [tex_escape(p) for p in txt.split("\n") if p.strip()]
        paras[-1] += "\\ldots"                    # 200 tokens end mid-sentence
        w("zoo_hikaye.tex", [p + "\\par" for p in paras])
    print("report data written:", ", ".join(sorted(f for f in os.listdir(OUT) if f.startswith("zoo_"))))


if __name__ == "__main__":
    main()
