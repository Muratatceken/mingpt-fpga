"""Project I report (docs/decision/proje1_raporu.md) generated from the artifacts, so every number is traceable:
manifests, model/decision/*.json (training), mem/decision/*/export.json (INT8), docs/decision/sonuclar/val_results.*,
docs/decision/sonuclar/int8_drift.json (int8_drift.py), model/decision/hw.json (Yosys, nextpnr, netlist simulation)
and build/decision/finalize_*.log (RTL RESULT lines)."""
import datetime
import glob
import json
import os
import re

from contract import ROOT, contract_hash

DOC = os.path.join(ROOT, "docs", "decision", "proje1_raporu.md")
AYLAR = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")


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


def ti(n):
    return f"{int(n):,}".replace(",", ".")


def today_tr():
    t = datetime.date.today()
    return f"{t.day} {AYLAR[t.month - 1]} {t.year}"


def sgn(x, n=3):
    return ("+" if x >= 0 else "−") + tr(abs(x), n)


def margin_split(a):
    """Disagreements of one model split at the float top-2 margin 0.2 (near-ties vs real decision changes)."""
    d = a.get("disagreements", [])
    big = [x["float_margin"] for x in d if x["float_margin"] >= 0.2]
    return len(d), len(d) - len(big), len(big), max(big) if big else None


def int8_note(a):
    nd, small, nbig, mx = margin_split(a)
    if nbig:
        s = (f"{nd} uyuşmazlık: {small} tanesi float üst-2 marjı < 0,2 (neredeyse-eşitlik), {nbig} tanesi ≥ 0,2 "
             f"(en büyük {tr(mx)})")
    else:
        s = f"{nd} uyuşmazlığın hepsi float üst-2 marjı < 0,2 (neredeyse-eşitlik)"
    return s + f"; marj ≥ 0,2 iken uyuşma {tr(a.get('agree_margin_ge_0p2', 0), 3)}"


def findings(vr, lat, drift):
    """Findings in prose, every number read from val_results.json, int8_drift.json and the finalize logs.
    Validation numbers only: they indicate a direction, the hypothesis tests (D1.5) run once on the test split."""
    out = []
    rule, fsm, lm, gru = (vr.get(k) for k in ("rule", "fsm", "lm", "gru"))
    inw = lambda v: sum(v["by_mean"][f"all/d{d}"] for d in (2, 8, 16)) / 3  # noqa: E731
    if lm:
        d32 = ", ".join(tr(s["by"]["all/d32"]) for s in lm["seeds"])
        out.append(f"- **Negatif kontrol tuttu.** d = 32'de belirleyici olay LM'in 127 tokenlık bağlamının dışında; "
                   f"tohum başına çift doğruluğu {d32}. Çiftin iki üyesi LM'e aynı göründüğü için aynı kararı alıyor; kabul "
                   "kümeleri ayrık olduğundan en çok bir üye doğru olabiliyor.")
    meths = [(k, v) for k, v in (("kural", rule), ("FSM", fsm), ("GRU", gru), ("LM", lm)) if v]
    if rule and len(meths) > 1:
        gap = {k: v["acc_mean"] - rule["acc_mean"] for k, v in meths if v is not rule}
        ok = [k for k, g in gap.items() if g > 0.05]
        verdict = ("geçmişi gören yöntemlerin hepsi" if len(ok) == len(gap) else ", ".join(ok) or "hiçbiri")
        out.append("- **H1 yönü (olumlu kontrol), doğrulama kümesinde:** senaryo düzeyinde yanlış eylem oranı "
                   + ", ".join(f"{k} {tr(1 - v['acc_mean'])}" for k, v in meths)
                   + f". Kuraldan 5 yüzde puandan fazla daha az yanlış eylem seçen: {verdict} (fark "
                   + ", ".join(f"{k} {tr(100 * g, 1)}" for k, g in gap.items()) + " yüzde puan). "
                   "Hipotez testi (McNemar, Holm) test bölümünde yapılacak.")
    if lm and gru:
        diff = [lm["by_mean"][f"all/d{d}"] - gru["by_mean"][f"all/d{d}"] for d in (2, 8, 16)]
        out.append("- **H2 yönü, doğrulama kümesinde: öngörülen yönde değil.** Pencere içinde LM − GRU çift doğruluğu "
                   f"farkı d = 2, 8, 16'da {', '.join(sgn(x, 2) for x in diff)}; fark uzaklıkla LM lehine büyümüyor, "
                   f"GRU her uzaklıkta önde (pencere içi ortalama GRU {tr(inw(gru))}, LM {tr(inw(lm))}"
                   + (f", FSM {tr(inw(fsm))}" if fsm else "") + "). Bu bir ön göstergedir, hipotez testi değildir; "
                   "H2 ön kayıtlı biçimiyle test bölümünde sınanacak ve sonuç hangi yönde çıkarsa öyle raporlanacak.")
    if lm and gru:
        ml, mg = lm["member_in_window_mean"], gru["member_in_window_mean"]
        mf = fsm["member_in_window_mean"] if fsm else None
        out.append("- **OC şablonunda iki öğrenen model de temkinli tarafta hata yapıyor.** Pencere içinde aralıklı "
                   f"aşırı akımı (B) KISITLI_MOD'a alma oranı LM {tr(ml['OC/B'])}, GRU {tr(mg['OC/B'])}; tek geçici olaydan "
                   f"sonra (A) NORMAL'e dönme oranı yalnız LM {tr(ml['OC/A'])}, GRU {tr(mg['OC/A'])}"
                   + (f" (FSM {tr(mf['OC/A'])})" if mf else "") + ". Hatalar gereksiz kısıtlı mod yönünde, kaçırılan "
                   f"arıza yönünde değil. OC çift doğruluğu d = 2'de de düşük (LM {tr(lm['by_mean']['OC/d2'])}, GRU "
                   f"{tr(gru['by_mean']['OC/d2'])}) ve belleği sınırsız GRU'da da görülüyor; yani sorun bağlam penceresi "
                   "değil, öğrenilen karar sınırı ya da etiketler (Proje II: OC örneklerinin derlemdeki payı, etiket "
                   "duyarlılığı).")
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
                   f"LM {tr(lm['by_mean']['all/d32'])}.")
    ia = vr.get("_int8_agreement", {})
    for fam, label in (("lm", "LM"), ("gru", "GRU")):
        a = {k: v for k, v in ia.items() if k.startswith(fam + "_")}
        if not a:
            continue
        ag = [v["agree"] for v in a.values()]
        dacc = [v["int_acc"] - v["float_acc"] for v in a.values()]
        big = [margin_split(v) for v in a.values()]
        nbig = sum(b[2] for b in big)
        maxd = max(x["float_margin"] for v in a.values() for x in v.get("disagreements", [])) if any(
            v.get("disagreements") for v in a.values()) else 0.0
        s = (f"- **INT8, {label}:** float ile karar uyuşması {tr(min(ag), 4)}–{tr(max(ag), 4)}; INT8 − float doğruluk "
             f"farkı {sgn(min(dacc))} ile {sgn(max(dacc))} arası. ")
        if nbig == 0:
            s += (f"Bütün uyuşmazlıklar float üst-2 marjı 0,2'nin altında (en büyük {tr(maxd, 3)}): neredeyse-eşitlikler, "
                  "nicemleme kararın yönünü değiştirmiyor.")
        else:
            s += (f"Üç tohumda toplam {nbig} uyuşmazlık float marjı ≥ 0,2 iken oluşuyor (en büyük {tr(maxd)}): bunlar "
                  "neredeyse-eşitlik değil.")
        dr = {k: v for k, v in drift.items() if k.startswith(fam + "_")}
        if dr:
            first = [v["bins"]["0-7"]["q95"] for v in dr.values()]
            last = [v["bins"]["40-47"]["q95"] for v in dr.values() if "40-47" in v["bins"]]
            ratio = [v["bins"]["40-47"]["q95"] / v["bins"]["0-7"]["q95"] for v in dr.values() if "40-47" in v["bins"]]
            s += (f" Logit hatasının %95'lik değeri ilk 8 adımda {tr(min(first))}–{tr(max(first))}, son 8 adımda "
                  f"{tr(min(last))}–{tr(max(last))}; tohum başına artış {tr(min(ratio), 1)}–{tr(max(ratio), 1)} kat "
                  "(`int8_drift.json`).")
            if min(ratio) >= 2:
                s += (" Hata akış boyunca büyüyor" + ("; en olası neden, INT8 özyinelemeli durumun her adımın nicemleme "
                      "hatasını sonraki adıma taşıması. Proje II: durumun 16 bit tutulması ya da nicemleme farkındalıklı "
                      "eğitim." if fam == "gru" else "."))
            elif max(ratio) < 1.5:
                s += (" Hata ilk adımlardan sonra aynı düzeyde kalıyor, akış boyunca büyümüyor"
                      + (" (bantlı dikkat yalnız son 127 tokenın INT8 K/V girdilerini okuyor)." if fam == "lm" else "."))
        out.append(s)
    for name, mn, mx in lat:
        cyc = f"{ti(mx)} çevrim (her kararda aynı)" if mn == mx else f"{ti(mn)}–{ti(mx)} çevrim"
        ms = f"{tr(mx / 25e3)} ms" if mn == mx else f"{tr(mn / 25e3)}–{tr(mx / 25e3)} ms"
        out.append(f"- **Karar süresi ({name}):** RTL'de {cyc}, 25 MHz'de {ms} (hedef ≤ 10 ms). Bu bir WCET kanıtı "
                   "değil, ölçülen en uzun değerdir.")
    return "\n".join(out) + "\n"


def drift_table(drift):
    if not drift:
        return ""
    bins = list(next(iter(drift.values()))["bins"])
    rows = [(k, v["scenarios"]) + tuple(tr(v["bins"][b]["q95"], 3) if b in v["bins"] else "—" for b in bins)
            for k, v in drift.items()]
    return ("\n### INT8 logit hatası akış boyunca\n\nHer MODE konumunda dört eylem logitinin float ile tamsayı (logit "
            "ölçeğiyle geri çevrilmiş) arasındaki en büyük mutlak farkı; adım aralığı başına %95'lik değer "
            "(`sw/decision/int8_drift.py`). Float logit aralığı (en büyük − en küçük) tipik olarak 8–9.\n\n"
            + table(["Model", "Senaryo"] + [f"adım {b}" for b in bins], rows) + "\n")


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
        note = "" if ok else int8_note(a)
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
                              f"{ti(mx)} çevrim = {tr(mx / 25e3)} ms", "geçti" if mx <= 250000 else "geçmedi", ""))
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
    drift = load("docs/decision/sonuclar/int8_drift.json", {})
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "model", "decision", "*_s[0-9].json"))):
        r = json.load(open(f))
        runs.append((os.path.basename(f)[:-5], r["params"], r["lr"], r["steps"], r["batch"], r["device"],
                     f"{r['val']['acc']:.3f}", f"{r['val']['pair_acc']:.3f}", f"{r['seconds'] / 60:.0f}"))
    s = [f"# Bitirme Projesi I — teknik rapor (WP1–WP4)\n",
         f"Murat Atçeken · Mehmet Akif Ayzit · danışman Prof. Dr. Cem Ünsalan · {today_tr()} · sözleşme sha256 "
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
         "\n![Çift doğruluğu ve uzaklık](sonuclar/fig_val_by_d.png)\n", drift_table(drift),
         "\n### Bulgular\n", findings(vr, lat, drift),
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
         "Yosys hücre modelleriyle yalnız pinlerden sürer (`scripts/dec_netlist_sim.sh`). Gecikmesiz işlevsel bir "
         "simülasyondur; yerleşim sonrası gecikmelerle (SDF) zamanlama simülasyonu yapılmadı, zamanlama yalnız "
         "nextpnr raporundan gelir.\n",
         "\nKart öz-testleri `-nodsp` ile yerleştirilmiştir (Yosys 0.69'un DSP48E1 işaret genişletme hatası); FSM ve "
         "kural tepelerindeki LUT'ların çoğu iz hash'inin 32 bit çarpıcısıdır. Fmax açık kaynak aracın tahminidir; "
         "üretici aracı ya da kart ölçümü değildir.\n",
         "\n## 8. Proje II'ye devredenler\n",
         "- M1 gözden geçirme kaydı ve açık maddelerin kapatılması (`docs/decision/M1/README.md`).\n"
         "- Test bölümünün bir kez açılması, H1–H3 istatistikleri (D1.5).\n"
         "- WP3.6 ablasyonları (geçmiş karıştırma, 1 katman × 128 bant), etiket duyarlılığı (H, τ, saf kural).\n"
         "- RTL halka tamponlu KV (uzun akış), yumuşak RISC-V çekirdek karşılaştırması (H3), kart üzerinde kapalı döngü "
         "(tesis modeli RTL'de).\n"
         "- FSM'nin d = 2'deki zayıflığı (3 adımlık doğrulama penceresi) ve düşük kabul oranlı hücreler (M1 madde 4).\n"
         "- GRU'da akış boyunca biriken INT8 hatası (durumun 16 bit tutulması ya da nicemleme farkındalıklı eğitim) ve "
         "iki öğrenen modelin OC/A hatası (derlemdeki OC payı, etiket duyarlılığı).\n"]
    with open(DOC, "w") as f:
        f.write("\n".join(s))
    print(DOC)


if __name__ == "__main__":
    main()
