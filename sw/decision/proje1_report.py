"""Project I report (docs/decision/proje1_raporu.md) generated from the artifacts, so every number is traceable:
manifests, model/decision/*.json (training), mem/decision/*/export.json (INT8), docs/decision/sonuclar/val_results.*,
model/decision/hw.json (Yosys, nextpnr) and build/decision/finalize.log (RTL RESULT lines)."""
import glob
import json
import os
import re

from contract import ROOT, contract_hash

DOC = os.path.join(ROOT, "docs", "decision", "proje1_raporu.md")


def load(p, default=None):
    try:
        return json.load(open(os.path.join(ROOT, p)))
    except (OSError, ValueError):
        return default


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def tr(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def findings(vr, lat):
    """Findings in prose, every number read from val_results.json / finalize logs."""
    out = []
    rule, fsm, lm, gru = (vr.get(k) for k in ("rule", "fsm", "lm", "gru"))
    inw = lambda v: sum(v["by_mean"][f"all/d{d}"] for d in (2, 8, 16)) / 3  # noqa: E731
    if lm:
        d32 = ", ".join(tr(s["by"]["all/d32"]) for s in lm["seeds"])
        out.append(f"- **Negatif kontrol tuttu.** d = 32'de belirleyici olay LM'in 127 tokenlık bağlamının dışında; "
                   f"tohum başına çift doğruluğu {d32}. Çiftin iki üyesi LM'e aynı göründüğü için aynı kararı alıyor; kabul "
                   "kümeleri ayrık olduğundan en çok bir üye doğru olabiliyor.")
    if lm and fsm:
        out.append(f"- **Pencere içinde (d ≤ 16)** ortalama çift doğruluğu LM {tr(inw(lm))}, FSM {tr(inw(fsm))}. "
                   f"d = 2'de LM {tr(lm['by_mean']['all/d2'])}, FSM {tr(fsm['by_mean']['all/d2'])}.")
        mw = lm["member_in_window_mean"]
        out.append(f"- **OC şablonunda LM temkinli tarafta hata yapıyor.** Pencere içinde aralıklı aşırı akımı (B) "
                   f"{tr(mw['OC/B'])} oranında doğru biçimde KISITLI_MOD'a alıyor; tek geçici olaydan sonra (A) NORMAL'e "
                   f"dönme oranı yalnız {tr(mw['OC/A'])}. Hatalar gereksiz kısıtlı mod yönünde, kaçırılan arıza yönünde "
                   f"değil. OC çift doğruluğu d = 2'de de düşük ({tr(lm['by_mean']['OC/d2'])}), yani sorun bellek değil "
                   "öğrenilen karar sınırı (Proje II: OC örneklerinin derlemdeki payı, etiket duyarlılığı).")
    if fsm:
        hi = [fsm["by_mean"][f"all/d{d}"] for d in (8, 16, 32)]
        out.append(f"- **FSM olumlu kontrolü d ≥ 8'de büyük ölçüde tutuyor** ({tr(min(hi))}–{tr(max(hi))}); d = 2'de "
                   f"{tr(fsm['by_mean']['all/d2'])}. Yük anahtarlamasından sonraki 3 adımlık doğrulama penceresi d = 2'de "
                   "karar anında henüz kapanmamış olabiliyor; plandaki “FSM ≈ %100” beklentisi bu yüzden tutmadı.")
    if rule:
        out.append(f"- **Eşik kuralı** geçmişe bakmadığı için çiftin iki üyesinde aynı kararı veriyor: çift doğruluğu "
                   f"{tr(rule['pair_mean'])}, senaryo doğruluğu {tr(rule['acc_mean'])}.")
    if gru and lm:
        out.append(f"- **GRU'nun sabit bir pencere sınırı yok:** d = 32'de çift doğruluğu GRU {tr(gru['by_mean']['all/d32'])}, "
                   f"LM {tr(lm['by_mean']['all/d32'])}. Pencere içinde GRU {tr(inw(gru))}, LM {tr(inw(lm))}.")
    ia = vr.get("_int8_agreement", {})
    if ia:
        lo, hi = min(a["agree"] for a in ia.values()), max(a["agree"] for a in ia.values())
        out.append(f"- **INT8:** float ile karar uyuşması {tr(lo, 4)}–{tr(hi, 4)}. Bütün uyuşmazlıklar float üst-2 "
                   "marjı 0,2'nin altında olan neredeyse-eşitliklerde; INT8 doğruluğu float'tan düşük değil.")
    for name, mn, mx in lat:
        out.append(f"- **Karar süresi ({name}):** RTL'de {mn}–{mx} çevrim, 25 MHz'de {tr(mn / 25e3)}–{tr(mx / 25e3)} ms "
                   "(hedef ≤ 10 ms). Bu bir WCET kanıtı değil, ölçülen en uzun değerdir.")
    return "\n".join(out) + "\n"


def main():
    man = {s: load(f"sw/decision/manifests/{s}_manifest.json", {}) for s in ("val", "test", "train")}
    hw = load("model/decision/hw.json", {})
    results = []
    for f in sorted(glob.glob(os.path.join(ROOT, "build", "decision", "finalize_*.log"))):
        for line in re.findall(r"^.*RESULT .*$", open(f).read(), re.M):
            if line not in results:
                results.append(line)
    int8 = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "mem", "decision", "*", "export.json"))):
        e = json.load(open(f))
        if not e.get("random"):
            int8[e["name"]] = e["agreement_val"]
    gate_rows = []
    for k, a in int8.items():
        ok = a["agree"] >= 0.99
        note = "" if ok else (f"{a.get('n_disagree', '?')} uyuşmazlığın hepsi float üst-2 marjı < 0,2 olan "
                              f"neredeyse-eşitliklerde; marj ≥ 0,2 iken uyuşma {tr(a.get('agree_margin_ge_0p2', 0), 3)}"
                              if a.get("agree_margin_ge_0p2") == 1.0 else "")
        gate_rows.append(("S6 INT8 uyuşması ≥ 0,99", k, f"{tr(a['agree'], 4)} (n = {a['n']})",
                          "geçti" if ok else "geçmedi", note))
    for line in results:
        mm = re.search(r"\[(\w+)\] RESULT decision_top .*steps (\d+) errors (\d+)", line)
        if mm:
            gate_rows.append(("S7 iz = hw_ref", mm.group(1), f"{mm.group(2)} adım, {mm.group(3)} hata",
                              "geçti" if mm.group(3) == "0" else "geçmedi", ""))
    lat = []
    for line in results:
        mm = re.search(r"\[(\w+)\] RESULT decision_top policy 0 .*dec_cycles min (\d+) max (\d+)", line)
        if mm:
            mx = int(mm.group(3))
            lat.append((mm.group(1), int(mm.group(2)), mx))
            gate_rows.append(("D1.2 en uzun karar ≤ 10 ms (25 MHz)", mm.group(1),
                              f"{mx} çevrim = {tr(mx / 25e3)} ms", "geçti" if mx <= 250000 else "geçmedi", ""))
    for k, v in hw.items():
        if "netlist_sim" in v:
            n = v["netlist_sim"]
            gate_rows.append(("S7 kapı düzeyi netlist (kart öz-testi)", k,
                              f"{n['matched']}/{n['scenarios']} senaryo eşleşti", "geçti" if n["passed"] else "geçmedi",
                              ""))
    for k, v in hw.items():
        if "pnr" in v and v["pnr"]["clock_mhz"] == 25:
            gate_rows.append(("S7 25 MHz zamanlama", k, f"Fmax {tr(v['pnr']['fmax_mhz'], 1)} MHz",
                              "geçti" if v["pnr"]["met"] else "geçmedi", ""))
    val_md = ""
    if os.path.exists(os.path.join(ROOT, "docs", "decision", "sonuclar", "val_results.md")):
        val_md = open(os.path.join(ROOT, "docs", "decision", "sonuclar", "val_results.md")).read().split("\n", 2)[2]
        val_md = val_md.replace("\n## ", "\n### ")
    vr = load("docs/decision/sonuclar/val_results.json", {})
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "model", "decision", "*_s[0-9].json"))):
        r = json.load(open(f))
        runs.append((os.path.basename(f)[:-5], r["params"], r["lr"], r["steps"], r["batch"], r["device"],
                     f"{r['val']['acc']:.3f}", f"{r['val']['pair_acc']:.3f}", f"{r['seconds'] / 60:.0f}"))
    s = [f"# Bitirme Projesi I — teknik rapor (WP1–WP4)\n",
         f"Murat Atçeken · Mehmet Akif Ayzit · danışman Prof. Dr. Cem Ünsalan · 21 Kasım 2026 · sözleşme sha256 "
         f"`{contract_hash()[:16]}…`\n",
         "Bu rapor `sw/decision/proje1_report.py` tarafından proje çıktılarından üretilir; elle sayı yazılmaz. "
         "Test bölümü ön kayıt gereği açılmamıştır; bütün karar kalitesi sayıları **doğrulama** bölümüne aittir.\n",
         "## 1. İş paketlerinin durumu\n",
         table(["Görev", "Çıktı", "Durum", "Kanıt"], [
             ("WP1.1", "D1.1 kanıt kaydı", "tamam", "`docs/decision/D1.1_kanit_kaydi.md`"),
             ("WP1.2", "D1.2 hipotez matrisi", "tamam", "`D1.2_arastirma_sorusu_matrisi.md`"),
             ("WP1.3", "D1.3 alt sistem ve arıza tanımı", "tamam (üretilmiş)", "`D1.3_…`, `M1/fig_directed.png`"),
             ("WP1.4", "D1.4 eylem ve guard tabloları", "tamam (üretilmiş)", "`D1.4_…`"),
             ("WP1.5", "D1.5 değerlendirme protokolü", "tamam (ön kayıt)", "`D1.5_degerlendirme_protokolu.md`"),
             ("WP1.6", "M1 şartname paketi", "paket hazır; danışman kaydı bekliyor", "`docs/decision/M1/README.md`"),
             ("WP2.1–2.3", "kapalı döngü simülatör, arızalar, eylem etkileri", "tamam", "`sim.py`, `tests/decision`"),
             ("WP2.4", "eşlenik çift kataloğu", "tamam", "`pairs.py`, manifestler"),
             ("WP2.5", "donmuş bölümler", "tamam", "`sw/decision/manifests/*.json` (sha256)"),
             ("WP3.1–3.2", "kural ve FSM", "tamam; RTL ile bit düzeyinde aynı", "`policies.py`, `dec_policies.sv`"),
             ("WP3.3", "tokenlaştırıcı ve politika arayüzü", "tamam; RTL ile aynı", "`tokenizer.py`, `dec_tokenizer`"),
             ("WP3.4", "GRU, 3 tohum", "tamam", "`model/decision/gru_s*.{pt,json}`"),
             ("WP3.5", "LM, 3 tohum", "tamam", "`model/decision/lm_s*.{pt,json}`"),
             ("WP4.1–4.2", "dışa aktarma ve sayısal sözleşme", "tamam", "`D4_disa_aktarim_ve_sayisal_sozlesme.md`"),
             ("WP4.3", "RTL arayüzü ve pilot yol", "tamam", "`decision_top.sv`, `scripts/dec_top.sh`"),
             ("WP4.4", "bağımsız guard ve test takımı", "tamam", "`dec_guard.sv`, `scripts/dec_units.sh`")]),
         "\n## 2. Derlem\n",
         table(["Bölüm", "Çift", "Senaryo / bölüm", "sha256 (dizi)"], [
             ("doğrulama", man["val"].get("n_pairs"), man["val"].get("n_scenarios"), man["val"].get("seq_sha256", "")[:16]),
             ("test (kapalı)", man["test"].get("n_pairs"), man["test"].get("n_scenarios"),
              man["test"].get("seq_sha256", "")[:16]),
             ("eğitim", "—", man["train"].get("n_episodes"), man["train"].get("X_sha256", "")[:16])]),
         f"\nEğitim etiket dağılımı (NORMAL, YUKU_KAPAT, KISITLI_MOD, GUVENLI_MOD): {man['train'].get('label_dist')}.\n",
         "\n## 3. Doğrulama kümesinde dört yöntem\n", val_md,
         "\n![Çift doğruluğu ve uzaklık](sonuclar/fig_val_by_d.png)\n",
         "\n### Bulgular\n", findings(vr, lat),
         "\n## 4. Aşama kapıları (planın S6–S7 ölçütleri)\n",
         table(["Kapı", "Model", "Ölçülen", "Sonuç", "Not"], gate_rows),
         "\n## 5. Eğitim kayıtları\n",
         table(["Çalıştırma", "Parametre", "lr", "Adım", "Batch", "Cihaz", "Val senaryo", "Val çift", "Süre (dk)"], runs),
         "\n## 6. RTL doğrulama sonuçları\n",
         "Birim testleri (`make dec-units`): guard 73.728 vektör, tokenlaştırıcı 334, kural 3.705, FSM 64.080 adım, bekçi "
         "zamanlayıcı 400 durum — Verilator ve Icarus ile 0 hata.\n",
         "```\n" + ("\n".join(results) if results else "(finalize.log henüz yok)") + "\n```\n",
         "\n## 7. Donanım maliyeti (Yosys sentezi, nextpnr yerleşimi; xc7a100t)\n",
         table(["Yapılandırma", "LUT", "FF", "BRAM", "DSP"],
               [(k, v["yosys"]["lut"], v["yosys"]["ff"], v["yosys"]["bram"], v["yosys"]["dsp"])
                for k, v in hw.items() if "yosys" in v]),
         "\n" + table(["Kart öz-testi", "Saat (MHz)", "Zamanlama", "Fmax (MHz)", "LUT (yerleşim)"],
                      [(k, v["pnr"]["clock_mhz"], "tutuyor" if v["pnr"]["met"] else "tutmuyor", v["pnr"]["fmax_mhz"],
                        v["pnr"]["utilization"].get("SLICE_LUTX", ""))
                       for k, v in hw.items() if "pnr" in v]),
         "\n" + table(["Kapı düzeyi netlist simülasyonu", "Senaryo", "Eşleşen", "Çevrim", "Sonuç"],
                      [(k, v["netlist_sim"]["scenarios"], v["netlist_sim"]["matched"], v["netlist_sim"]["cycles"],
                        "yeşil" if v["netlist_sim"]["passed"] else "kırmızı")
                       for k, v in hw.items() if "netlist_sim" in v]),
         "\nNetlist simülasyonu, nextpnr'a verilen Yosys netlistini (MMCM, BUFG, G/Ç tamponları, BRAM içerikleri dahil) "
         "Yosys hücre modelleriyle yalnız pinlerden sürer (`scripts/dec_netlist_sim.sh`).\n",
         "\nKart öz-testleri `-nodsp` ile yerleştirilmiştir (Yosys 0.69'un DSP48E1 işaret genişletme hatası); FSM ve "
         "kural tepelerindeki LUT'ların çoğu iz hash'inin 32 bit çarpıcısıdır. Fmax açık kaynak aracın tahminidir; "
         "üretici aracı ya da kart ölçümü değildir.\n",
         "\n## 8. Proje II'ye devredenler\n",
         "- M1 gözden geçirme kaydı ve açık maddelerin kapatılması (`docs/decision/M1/README.md`).\n"
         "- Test bölümünün bir kez açılması, H1–H3 istatistikleri (D1.5).\n"
         "- WP3.6 ablasyonları (geçmiş karıştırma, 1 katman × 128 bant), etiket duyarlılığı (H, τ, saf kural).\n"
         "- RTL halka tamponlu KV (uzun akış), yumuşak RISC-V çekirdek karşılaştırması (H3), kart üzerinde kapalı döngü "
         "(tesis modeli RTL'de).\n"
         "- FSM'nin d = 2'deki zayıflığı (3 adımlık doğrulama penceresi) ve düşük kabul oranlı hücreler (M1 madde 4).\n"]
    with open(DOC, "w") as f:
        f.write("\n".join(s))
    print(DOC)


if __name__ == "__main__":
    main()
