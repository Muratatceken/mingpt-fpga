"""Validation comparison of the four decision methods (Project I; the test split stays closed until Project II).

    python3 evaluate_dec.py            -> docs/decision/sonuclar/val_results.{json,md}, fig_val_by_d.png
Per method: scenario accuracy (decision in the acceptable set at t0) and pair discrimination (both members
right), overall and by template x d, mean and spread over the training seeds, pair-level bootstrap 95 % CI.
Methods: rule, FSM (token-level, deterministic), GRU and LM (float, every seed in model/decision/runs/v1);
the INT8 decision agreement of the exported models is read from mem/decision/*/export.json.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from contract import NORMAL, ROOT  # noqa: E402
from models_dec import build  # noqa: E402
from policies import FSMPolicy, rule_action  # noqa: E402
from tokenizer import bins_from_tokens  # noqa: E402
from train_policy import eval_pairs, load_val  # noqa: E402

OUT = os.path.join(ROOT, "docs", "decision", "sonuclar")
RUNS = os.path.join(ROOT, "model", "decision", "runs", "v1")
plt.rcParams.update({"font.family": "Arial", "font.size": 8})


def token_policy(seq, meta, kind):
    E = len(seq)
    t0 = meta[0]["t0"]
    pred = np.zeros(E, dtype=np.int64)
    fsm = FSMPolicy(E)
    prev = np.full(E, NORMAL, dtype=np.int64)
    for k in range(t0 + 1):
        tV, tI, tT = seq[:, 1 + 5 * k], seq[:, 2 + 5 * k], seq[:, 3 + 5 * k]
        if kind == "fsm":
            a = fsm.act(tV, tI, tT, 0, prev)
        else:
            vb, ib, tb, inv = bins_from_tokens(tV, tI, tT)
            a = rule_action(vb, ib, tb, inv)
        if k == t0:
            pred = a
        prev = seq[:, 5 + 5 * k]
    return pred


def summarise(pred, acc, meta, rng, nboot=2000):
    ok = acc[np.arange(len(pred)), pred]
    pair = ok.reshape(-1, 2).all(1)
    out = dict(acc=float(ok.mean()), pair=float(pair.mean()), n_scen=int(len(ok)), n_pair=int(len(pair)))
    bs = rng.integers(0, len(pair), (nboot, len(pair)))
    out["pair_ci"] = [float(np.quantile(pair[bs].mean(1), 0.025)), float(np.quantile(pair[bs].mean(1), 0.975))]
    by = {}
    for j, m in enumerate(meta[::2]):
        by.setdefault(f"{m['template']}/d{m['d']}", []).append(pair[j])
        by.setdefault(f"all/d{m['d']}", []).append(pair[j])
        by.setdefault(f"noise_{m['noise']}", []).append(pair[j])
    out["by"] = {k: float(np.mean(v)) for k, v in sorted(by.items())}
    sc = {}
    for j, m in enumerate(meta):                      # scenario accuracy per pair member, inside the window
        if m["d"] <= 16:
            sc.setdefault(f"{m['template']}/{m['member']}", []).append(ok[j])
    out["member_in_window"] = {k: float(np.mean(v)) for k, v in sorted(sc.items())}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    seq, y, acc, meta = load_val()
    rng = np.random.default_rng(0)
    res = {}
    for kind in ("rule", "fsm"):
        res[kind] = dict(seeds=[summarise(token_policy(seq, meta, kind), acc, meta, rng)])
    for model in ("gru", "lm"):
        runs = sorted(glob.glob(os.path.join(RUNS, f"{model}_lr*_s*.json")))
        seeds = []
        for r in runs:
            rec = json.load(open(r))
            m = build(rec["spec"])
            m.load_state_dict(torch.load(r[:-5] + ".pt", map_location="cpu"))
            _, pred = eval_pairs(m, seq, acc, meta, "cpu")
            s = summarise(pred, acc, meta, rng)
            s.update(run=os.path.basename(r[:-5]), params=rec["params"], train_s=rec["seconds"])
            seeds.append(s)
        if seeds:
            res[model] = dict(seeds=seeds)
    for k, v in res.items():
        p = np.array([s["pair"] for s in v["seeds"]])
        a = np.array([s["acc"] for s in v["seeds"]])
        v["pair_mean"], v["pair_sd"] = float(p.mean()), float(p.std(ddof=1)) if len(p) > 1 else 0.0
        v["acc_mean"] = float(a.mean())
        keys = v["seeds"][0]["by"].keys()
        v["by_mean"] = {kk: float(np.mean([s["by"][kk] for s in v["seeds"]])) for kk in keys}
        keys = v["seeds"][0]["member_in_window"].keys()
        v["member_in_window_mean"] = {kk: float(np.mean([s["member_in_window"][kk] for s in v["seeds"]]))
                                      for kk in keys}
    int8 = {}
    for f in glob.glob(os.path.join(ROOT, "mem", "decision", "*", "export.json")):
        e = json.load(open(f))
        if not e.get("random"):
            int8[e["name"]] = e["agreement_val"]
    res["_int8_agreement"] = int8
    json.dump(res, open(os.path.join(OUT, "val_results.json"), "w"), indent=1)

    methods = [m for m in ("rule", "fsm", "gru", "lm") if m in res]
    lines = ["# Doğrulama kümesi sonuçları (Proje I)\n",
             f"{len(seq) // 2} eşlenik çift (48 hücre × 10). Çift doğruluğu: iki üyede de kabul edilebilir eylem. "
             "Öğrenen modeller için 3 tohumun ortalaması ± standart sapma; parantezde tohum 0'ın çift düzeyinde %95 "
             "bootstrap güven aralığı. Test bölümü Proje I'de açılmadı.\n",
             "| Yöntem | Senaryo doğruluğu | Çift doğruluğu | d=2 | d=8 | d=16 | d=32 |", "|---|---|---|---|---|---|---|"]
    for mth in methods:
        v = res[mth]
        ci = v["seeds"][0]["pair_ci"]
        sd = f" ± {v['pair_sd']:.3f}" if len(v["seeds"]) > 1 else ""
        bd = v["by_mean"]
        lines.append(f"| {mth} | {v['acc_mean']:.3f} | {v['pair_mean']:.3f}{sd} ({ci[0]:.3f}–{ci[1]:.3f}) | "
                     + " | ".join(f"{bd[f'all/d{d}']:.3f}" for d in (2, 8, 16, 32)) + " |")
    lines += ["\n## Şablon × d (çift doğruluğu)\n", "| Yöntem | " + " | ".join(
        f"{t}/d{d}" for t in ("BAT_SEN", "SEN", "OC") for d in (2, 8, 16, 32)) + " |",
              "|---|" + "---|" * 12]
    for mth in methods:
        bd = res[mth]["by_mean"]
        lines.append(f"| {mth} | " + " | ".join(f"{bd[f'{t}/d{d}']:.2f}" for t in ("BAT_SEN", "SEN", "OC")
                                               for d in (2, 8, 16, 32)) + " |")
    mem_cols = [f"{t}/{u}" for t in ("BAT_SEN", "SEN", "OC") for u in ("A", "B")]
    lines += ["\n## Pencere içinde (d ≤ 16) üye düzeyinde senaryo doğruluğu\n",
              "BAT_SEN: A = sensör arızası → KISITLI_MOD, B = batarya kapasite kaybı → GUVENLI_MOD. SEN: A = sağlam → "
              "NORMAL, B = takılı sensör → KISITLI_MOD. OC: A = tek geçici aşırı akım → NORMAL, B = aralıklı aşırı akım "
              "→ KISITLI_MOD.\n",
              "| Yöntem | " + " | ".join(mem_cols) + " |", "|---|" + "---|" * len(mem_cols)]
    for mth in methods:
        bd = res[mth]["member_in_window_mean"]
        lines.append(f"| {mth} | " + " | ".join(f"{bd[c]:.2f}" for c in mem_cols) + " |")
    if int8:
        lines += ["\n## INT8 karar uyuşması (doğrulama kümesi, float ↔ tamsayı)\n",
                  "Marj: float üst-2 logit farkı. Uyuşmazlıklar ve hangi tarafın kabul edilebilir eylemi seçtiği "
                  "`mem/decision/<model>/export.json` içinde (`disagreements`).\n",
                  "| Model | n | Uyuşma | Uyuşmazlık | Marj ≥ 0,2 iken uyuşma | Float doğruluk | INT8 doğruluk |",
                  "|---|---|---|---|---|---|---|"]
        for k, a in sorted(int8.items()):
            w = a.get("agree_margin_ge_0p2")
            ws = f"{w:.3f} (n = {a['n_margin_ge_0p2']})" if w is not None else "—"
            lines.append(f"| {k} | {a['n']} | {a['agree']:.4f} | {a.get('n_disagree', '—')} | {ws} | "
                         f"{a['float_acc']:.3f} | {a['int_acc']:.3f} |")
    with open(os.path.join(OUT, "val_results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    col = dict(rule="#9aa5b1", fsm="#52606d", gru="#b44d12", lm="#0b6e75")
    for mth in methods:
        bd = res[mth]["by_mean"]
        ys = [bd[f"all/d{d}"] for d in (2, 8, 16, 32)]
        ax.plot([2, 8, 16, 32], ys, marker="o", color=col[mth], label=mth.upper() if mth != "rule" else "kural")
    ax.axvspan(25.4, 34, color="#eef2f6")
    ax.text(29.5, 0.04, "pencere\ndışında", ha="center", fontsize=7, color="#52606d")
    ax.set_xticks([2, 8, 16, 32])
    ax.set_xlabel("belirleyici olay uzaklığı d (karar adımı)")
    ax.set_ylabel("çift doğruluğu (doğrulama)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_val_by_d.png"), dpi=180)
    print("\n".join(lines[:4 + len(methods)]))


if __name__ == "__main__":
    main()
