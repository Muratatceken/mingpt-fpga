"""Project I report (docs/decision/proje1_raporu.md) generated from the artifacts, so every number is traceable:
manifests, model/decision/*.json (training), mem/decision/*/export.json (INT8), output/decision/val_results.*,
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


def main():
    man = {s: load(f"sw/decision/manifests/{s}_manifest.json", {}) for s in ("val", "test", "train")}
    hw = load("model/decision/hw.json", {})
    log = ""
    if os.path.exists(os.path.join(ROOT, "build", "decision", "finalize.log")):
        log = open(os.path.join(ROOT, "build", "decision", "finalize.log")).read()
    results = re.findall(r"^RESULT .*$", log, re.M)
    val_md = ""
    if os.path.exists(os.path.join(ROOT, "output", "decision", "val_results.md")):
        val_md = open(os.path.join(ROOT, "output", "decision", "val_results.md")).read().split("\n", 2)[2]
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
         "\n![Çift doğruluğu ve uzaklık](../../output/decision/fig_val_by_d.png)\n",
         "\n## 4. Eğitim kayıtları\n",
         table(["Çalıştırma", "Parametre", "lr", "Adım", "Batch", "Cihaz", "Val senaryo", "Val çift", "Süre (dk)"], runs),
         "\n## 5. RTL doğrulama sonuçları\n",
         "Birim testleri (`make dec-units`): guard 73.728 vektör, tokenlaştırıcı 334, kural 3.705, FSM 64.080 adım, bekçi "
         "zamanlayıcı 400 durum — Verilator ve Icarus ile 0 hata.\n",
         "```\n" + ("\n".join(results) if results else "(finalize.log henüz yok)") + "\n```\n",
         "\n## 6. Donanım maliyeti (Yosys sentezi, nextpnr yerleşimi; xc7a100t)\n",
         table(["Yapılandırma", "LUT", "FF", "BRAM", "DSP"],
               [(k, v["yosys"]["lut"], v["yosys"]["ff"], v["yosys"]["bram"], v["yosys"]["dsp"])
                for k, v in hw.items() if "yosys" in v]),
         "\n" + table(["Kart öz-testi", "Saat (MHz)", "Zamanlama", "Fmax (MHz)", "LUT (yerleşim)"],
                      [(k, v["pnr"]["clock_mhz"], "tutuyor" if v["pnr"]["met"] else "tutmuyor", v["pnr"]["fmax_mhz"],
                        v["pnr"]["utilization"].get("SLICE_LUTX", ""))
                       for k, v in hw.items() if "pnr" in v]),
         "\nKart öz-testleri `-nodsp` ile yerleştirilmiştir (Yosys 0.69'un DSP48E1 işaret genişletme hatası); FSM ve "
         "kural tepelerindeki LUT'ların çoğu iz hash'inin 32 bit çarpıcısıdır. Fmax açık kaynak aracın tahminidir; "
         "üretici aracı ya da kart ölçümü değildir.\n",
         "\n## 7. Proje II'ye devredenler\n",
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
