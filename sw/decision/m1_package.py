"""M1 specification-review package (WP1.3, 1.4, 1.6): documents generated from the contract and the code,
so that they cannot drift from what is implemented.

    python3 m1_package.py   -> docs/decision/D1.3_*.md, D1.4_*.md, docs/decision/M1/{README.md, fig_*.png, *.md}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contract import ACTIONS, C, EDGES, LAB, MODES, ROOT, SCN, TT, contract_hash  # noqa: E402
from episodes import run_episodes  # noqa: E402
from pairs import INTENDED, generate_cell  # noqa: E402
from policies import RulePolicy, Scripted  # noqa: E402
from sim import OCV_SOC, OCV_V, NoiseBank, Plant, sample_params, seed_of  # noqa: E402

DOC = os.path.join(ROOT, "docs", "decision")
M1 = os.path.join(DOC, "M1")
plt.rcParams.update({"font.family": "Arial", "font.size": 8})


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def d13():
    f = C["faults"]
    noise = C["noise"]
    s = [f"# D1.3 Alt sistem ve arıza tanımı\n\nSözleşme `sw/decision/spec/contract_v0.json` (sha256 `{contract_hash()[:16]}…`) "
         "ve `sw/decision/sim.py` dosyalarından üretilmiştir; elle düzenlenmez (`make dec-m1`).\n",
         "## Gözlenebilir kanallar (sensör kodları, adım 1 s)\n",
         md_table(["Kanal", "Birim", "Açıklama", "Aralık"],
                  [(k, v["unit"], v["desc"], f"{v['range'][0]} … {v['range'][1]}") for k, v in C["channels"].items()]),
         "\n## Sensör gürültüsü ve veri kaybı\n",
         md_table(["Düzey", "σ_V (mV)", "σ_I (mA)", "σ_T (0,1 °C)", "Veri kaybı olasılığı"],
                  [(k, v["V_mV"], v["I_mA"], v["T_dC"], v["dropout"]) for k, v in noise.items()]),
         "\n## Gizli durum ve dinamikler (`sim.py`)\n",
         "Durum: batarya doluluğu SoC, polarizasyon gerilimi v_p, batarya sıcaklığı T, fiziksel yük anahtarı, temel "
         "tüketim çarpanı, arıza durumu. Adım içindeki sıra: (1) zamanı gelen eylemler, (2) gözlem y_k, "
         "(3) politika ve guard, (4) fizik k→k+1.\n",
         "```\nI_bus = yük*(I_nom + ΔI_OC) + I_hk*hk          I_b = I_bus − I_sa\n"
         "V     = OCV(SoC) − ΔV_f − I_b*R0(T) − v_p          R0(T) = R0*r0m*(1 + 0,01*max(0, 25 − T))\n"
         "SoC'  = SoC − I_b/(3600*Q_eff)    v_p' = a*v_p + (1−a)*Rp*I_b,  a = exp(−1/τ_p)\n"
         "T'    = T + (I_b²*(R0(T)+Rp) − (T − T_amb)/R_th)/C_th\n```\n",
         "OCV(SoC) tablosu (2S Li-iyon):\n",
         md_table(["SoC"] + [f"{x:.2f}" for x in OCV_SOC], [["OCV (V)"] + [f"{x:.2f}" for x in OCV_V]]),
         "\n## Nominal parametre aralıkları (eğitim örneklemesi, `sample_params`)\n",
         md_table(["Parametre", "Aralık", "Not"], [
             ("SoC₀", "0,55 – 0,90", "şablonlarda hedef gerilime göre çözülür"),
             ("Q", "5,2 Ah", "2S2P"), ("R0 (25 °C)", "0,12 – 0,30 Ω", "+%1/°C, 25 °C altında"),
             ("Rp / τ_p", "0,03 – 0,08 Ω / 2 – 6 s", "tek RC dalı"),
             ("C_th / R_th / T_amb", "100 – 200 J/K / 2 – 4 K/W / 0 – 30 °C", ""),
             ("I_nom / I_hk / I_sa", "0,9 – 1,3 A / 0,08 – 0,15 A / 0 – 0,3 A", "faydalı yük / temel / güneş katkısı"),
             ("Aktüatör gecikmesi", "0 – 1 adım (eğitim), 0 (test)", ""),
             ("Etkisiz anahtarlama", "%5 (eğitim), %0 (test)", "anahtar komutu uygulanmaz")]),
         "\n## Arıza sınıfları\n",
         md_table(["Arıza", "Başlangıç", "Belirti", "Gelişim", "Temizlenme", "Şiddet (düşük / yüksek)", "Gereksinim"], [
             ("OC aşırı akım", "t_f; yük kapalıysa ilk açılışta", "I_bus = I_nom + ΔI",
              "geçici: tek olay; kesintili: açık kalınan 5–40 s sonra tekrar", "geçici: yük kesilince kalıcı olarak",
              f"ΔI {f['OC']['severity']['low']} / {f['OC']['severity']['high']} mA",
              "geçici: olay süresince yük kapalı, sonra NORMAL; kesintili: mod ≥ KISITLI"),
             ("BAT kapasite kaybı", "t_f (bölüm başından önce de olabilir)", "V düşer; yük azalınca kısmen toparlanır",
              "ΔV_f = A(1 − e^(−(t−t_f)/τ_f)), Q×0,2–0,5, R0×1,3", "yok",
              f"A {f['BAT']['severity']['low']} / {f['BAT']['severity']['high']} mV", "mod GUVENLI"),
             ("SEN takılı V sensörü", "t_f", "okuma sabit + ADC gürültüsü; yük değişimine tepki yok",
              "sabit", "yok", f"|gerçek − okunan| {f['SEN']['severity']['low']} / {f['SEN']['severity']['high']} mV",
              "mod ≥ KISITLI")]),
         f"\nGereksinimler arıza başlangıcından {f['requirement_grace_steps']} adım sonra etkin olur.\n",
         "\n## Bin sınırları (tokenlaştırma; bin = sınır ≤ kod sayısı)\n",
         md_table(["Kanal", "Sınırlar"], [(k, ", ".join(str(x) for x in v)) for k, v in EDGES.items()]),
         "\n## Yönlendirilmiş testler\n\n`docs/decision/M1/fig_directed.png`: her arıza sınıfı için yük kesme/geri açma "
         "senaryosunda gerçek ve okunan gerilim ile akım. Arızasız denge, yük kesilince akımın düşmesi ve fiziksel olarak "
         "imkânsız değer olmaması `tests/decision` içinde sınanır.\n"]
    return "\n".join(s)


def d14():
    g = C["guard"]
    rows = []
    for mi, m in enumerate(MODES):
        rows.append([m] + [("izin" if TT[(mi, ai)] is None else f"→ {ACTIONS[TT[(mi, ai)]]}")
                           for ai in range(4)])
    s = [f"# D1.4 Eylem ve guard geçiş tabloları\n\nSözleşme (sha256 `{contract_hash()[:16]}…`) ve "
         "`sw/decision/guard.py`, `rtl/decision/dec_guard.sv` ile birebir; elle düzenlenmez (`make dec-m1`).\n",
         "## Eylemler (şiddet sırasıyla)\n",
         md_table(["Kod", "Eylem", "Faydalı yük", "Temel tüketim", "Sonraki mod"],
                  [(a["id"], a["name"], "açık" if a["payload"] else "kapalı",
                    "değişmez" if a["hk_mult"] is None else f"×{a['hk_mult']}", a["mode_after"]) for a in C["actions"]]),
         "\n“En az X” = max(öneri, X); argmax eşitliğinde daha ağır eylem seçilir.\n",
         "## Modlar\n", md_table(["Kod", "Mod", "Temel tüketim"], [(m["id"], m["name"], f"×{m['hk_mult']}")
                                                                  for m in C["modes"]]),
         "\nGeçersiz mod kodu (3) GUVENLI kabul edilir. GUVENLI moddan çıkış yalnız yer komutu RESUME ile, aktif sınır "
         f"yokken ve {g['resume_to_mode']} moduna olur.\n",
         "## Öncelik sırası\n", "\n".join(f"{i + 1}. {p[2:]}" for i, p in enumerate(g["priority"])),
         "\n\n## Ham kod sınırları\n",
         md_table(["Sınır", "Değer"], [("V alt (mV)", g["limits_raw"]["V_low_mV"]), ("I üst (mA)", g["limits_raw"]["I_high_mA"]),
                                       ("T üst (0,1 °C)", g["limits_raw"]["T_high_dC"]),
                                       ("Geçersiz örnek kalıcılığı (adım)", g["invalid_persist_steps"]),
                                       ("Bekçi zamanlayıcı (çevrim, 25 MHz)", g["watchdog_cycles_25MHz"])]),
         "\n## Geçiş tablosu (mod × öneri)\n", md_table(["Mod \\ öneri"] + ACTIONS, rows),
         "\n## Gerekçe kodları\n", md_table(["Kod", "Anlam"], list(g["reason_codes"].items())),
         "\n## Eşik kuralı (son gözlem)\n", "\n".join(f"- {r}" for r in C["rule"]["order"]),
         "\n\n## Geçmiş tutan FSM (`policies.FSMPolicy`, `dec_fsm`; 41 bit durum)\n",
         "- OC olay sayacı (I ≥ 2000 mA yükselen kenarı, 2 bit, doygun): ilk olayda YUKU_KAPAT, sonra bir kez NORMAL ile "
         "yeniden deneme; ikinci olayda KISITLI_MOD.\n"
         "- Yük anahtarlama yanıtı: anahtarlamadan sonraki 3 adımda akım ≥ 2 bin değişip gerilim hiç değişmezse "
         "`sens` (sensör şüphesi) → en az KISITLI_MOD.\n"
         "- Yük kesildikten sonra gerilim ≥ 2 bin yükselip yük kapalıyken tepe değerin altına inerse ve 7,0 V altında "
         "kalırsa, ya da yüklüyken 6,6 V altına sarkmışsa `batt` (batarya şüphesi) → GUVENLI_MOD.\n"
         "- 7,0 V altında yük açıksa önce YUKU_KAPAT ile yanıtı sınar (prob), şüphe yoksa KISITLI_MOD.\n",
         "\n## Doğrulama\n\n`make dec-units`: guard için 73 728 vektörün tamamı (mod × geçerlilik × öneri × sınırların bir "
         "altı/üstü × geçerlilik bayrakları × yer komutu × sayaç durumu), FSM için 64 080 adım, bekçi zamanlayıcı kenarları için "
         "400 durum; Verilator ve Icarus (4 durumlu) ile 0 hata.\n"]
    return "\n".join(s)


def fig_directed():
    """one episode per fault class with a scripted load cycle: true vs read V, current."""
    B = 4
    rng = np.random.default_rng(seed_of("m1fig"))
    P = sample_params(rng, B, noise=0, train=False)
    P["ftype"] = np.array([0, 1, 2, 3])
    P["f_onset"] = np.array([1 << 20, 10, 10, 10])
    P["oc_kind"] = np.array([0, 1, 0, 0])
    P["oc_interval"] = np.array([8, 8, 8, 8])
    P["oc_dI"] = np.array([1.2, 1.2, 1.2, 1.2])
    P["bat_A"] = np.array([0.5, 0.5, 0.5, 0.5])
    P["sen_gap"] = np.array([-0.4, -0.4, -0.4, -0.4])
    N = 50
    sched = np.full((B, N), 0, dtype=np.int64)
    sched[:, 20:26] = 1                                  # load shed 21..26, back on from 27
    bank = NoiseBank([seed_of("m1fig", b) for b in range(B)], N + 70)
    r = run_episodes(Plant(P, bank), Scripted(sched, RulePolicy(B)), N, label=False, record_hidden=True)
    fig, axs = plt.subplots(2, 4, figsize=(11, 4.2), sharex=True)
    names = ["Arızasız", "OC (kesintili)", "BAT kapasite kaybı", "SEN takılı sensör"]
    for b in range(B):
        ax = axs[0, b]
        ax.plot(r["v_true"][b] * 1000, color="#0b6e75", label="gerçek V")
        ax.plot(r["v"][b], color="#b44d12", lw=0.8, label="okunan V")
        ax.axvspan(21, 27, color="#eef2f6")
        ax.set_title(names[b])
        ax.set_ylabel("mV" if b == 0 else "")
        axs[1, b].plot(r["i"][b], color="#1f2933")
        axs[1, b].axvspan(21, 27, color="#eef2f6")
        axs[1, b].set_xlabel("adım (s)")
        if b == 0:
            axs[1, b].set_ylabel("I (mA)")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Yönlendirilmiş testler: yük 21–26. adımlarda kesik (gri), arıza başlangıcı t=10", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(M1, "fig_directed.png"), dpi=160)
    plt.close(fig)


def fig_pairs_and_costs():
    lines = ["# Eşlenik çift örnekleri ve etiketleyici dal maliyetleri\n",
             "Her şablon için d = 8, yüksek şiddet, düşük gürültü örnek çiftinde karar adımındaki (t0 = 47) dört dalın "
             "(G, K, T+R+S) maliyetleri. Etiket a* ve kabul edilebilir küme buradan türetilir.\n"]
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.5))
    for row, tpl in enumerate(("BAT_SEN", "SEN", "OC")):
        prs, st = generate_cell("m1", tpl, "high", "low", 8, 1)
        p = prs[0]
        ax = axs[row]
        for side, col in (("A", "#b44d12"), ("B", "#0b6e75")):
            ax.plot(p[side]["v"], color=col, label=f"{side}: okunan V")
        ax.axvline(47 - 8, color="#52606d", ls="--", lw=0.8)
        ax.axvline(47, color="#1f2933", lw=0.8)
        iA, iB = INTENDED[tpl]
        ax.set_title(f"{tpl}: A → {ACTIONS[iA]}, B → {ACTIONS[iB]}  (son 8 adımda tokenlar özdeş)", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.set_ylabel("mV")
        lines.append(f"\n## {tpl}\n")
        rows = []
        for side in ("A", "B"):
            c = p[side]["costs"][47]
            acc = p[side]["acc"][47]
            for a in range(4):
                rows.append((side, ACTIONS[a], int(c[a, 0]), int(c[a, 1]), int(c[a, 2]),
                             "a*" if p[side]["a_star"][47] == a else ("kabul" if acc[a] else "")))
        lines.append(md_table(["Üye", "İlk eylem", "G", "K", "T+R+S", "Etiket"], rows))
    axs[-1].set_xlabel("adım (s)  — kesikli çizgi: t0 − d, düz çizgi: karar adımı t0")
    fig.tight_layout()
    fig.savefig(os.path.join(M1, "fig_pairs.png"), dpi=160)
    plt.close(fig)
    with open(os.path.join(M1, "labeler_costs.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def readme():
    man = {s: json.load(open(os.path.join(ROOT, "sw", "decision", "manifests", f"{s}_manifest.json")))
           for s in ("val", "test", "train")}
    low = {k: v["rate"] for k, v in man["test"]["acceptance"].items() if v["rate"] < SCN["acceptance_min_rate"]}
    s = [f"# M1 — Şartname gözden geçirme paketi (WP1.6)\n\nTarih: 21 Kasım 2026 · Sözleşme sha256 `{contract_hash()}`\n",
         "Bu paket danışman gözden geçirmesi içindir. Gözden geçirme sonucu (kabul / açık maddeler) bu dosyanın sonundaki "
         "kayda işlenir; yalnız toplantı yapılması tamamlanma sayılmaz (WP1.6 tamamlanma ölçütü).\n",
         "## İçerik\n",
         "- `../D1.1_kanit_kaydi.md`: yeniden kullanılan her sayının kaynağı ve ölçüm sınırı\n"
         "- `../D1.2_arastirma_sorusu_matrisi.md`: hipotezler, karşılaştırma tabanları, ölçütler, çürütme koşulları\n"
         "- `../D1.3_alt_sistem_ve_ariza_tanimi.md`: kanallar, dinamikler, parametreler, arıza sınıfları (üretilmiş)\n"
         "- `../D1.4_eylem_ve_guard_tablolari.md`: eylemler, modlar, guard öncelikleri, geçiş tablosu (üretilmiş)\n"
         "- `../D1.5_degerlendirme_protokolu.md`: ön kayıt — d tanımı, etiketleyici, bölümler, istatistik, karar süresi\n"
         "- `fig_directed.png`: arıza sınıflarının yönlendirilmiş testleri\n"
         "- `fig_pairs.png`, `labeler_costs.md`: eşlenik çift örnekleri ve dal maliyetleri\n"
         "- `../../../sw/decision/manifests/*.json`: bölüm manifestleri (sha256, kabul oranları)\n",
         "## Derlem özeti\n",
         md_table(["Bölüm", "Çift", "Senaryo / bölüm", "Not"], [
             ("val", man["val"]["n_pairs"], man["val"]["n_scenarios"], "hücre başına 10 çift (D2)"),
             ("test", man["test"]["n_pairs"], man["test"]["n_scenarios"],
              "48 hücre × (30 D2 + 20 D3) + 480 D4; Proje I'de açılmaz"),
             ("train", "—", man["train"]["n_episodes"], f"{man['train']['n_labels']} etiket")]),
         "\n## Açık maddeler (danışman kararı gereken)\n",
         "1. **Bağlam:** etkin pencere 127 token (2 katman × 64 bant). 2209 metnindeki “C = 128” ile tutarlı; "
         "1 katman × 128 B planı ablasyon olarak tutuluyor.\n"
         "2. **Etiketleyici devam politikası:** saf eşik kuralı ilk eylemi bir adım sonra geri aldığından kilitli GUVENLI "
         "neredeyse her etiketi kazanıyordu; tekdüze kural `max(kural, ilk eylem)` seçildi, saf kural duyarlılık olarak "
         "raporlanacak (D1.5).\n"
         "3. **Kısıtlı mod maliyeti:** gereksiz KISITLI/GUVENLI moda R maliyeti eklendi; yoksa YUKU_KAPAT ile KISITLI eşit "
         "kalıp eşitlik kuralı hep KISITLI seçiyordu.\n"
         f"4. **Düşük kabul oranlı hücreler** (test, < %{int(SCN['acceptance_min_rate'] * 100)}): {low}. Yüksek gürültü + "
         "d = 32 SEN hücrelerinde yüklü gerilim SoC ile kayarken takılı okuma sabit kalıyor; parametre seçim yanlılığı "
         "manifestte kayıtlı. Şablon revizyonu ya da kabul eşiği kararı gerekli.\n"
         "5. **Kart üzerinde kapalı döngü:** pilot ROM'dan açık döngü yeniden oynatmadır; kart üzerinde tesis modeli "
         "(RTL'de basit güç modeli) Proje II kararıdır.\n"
         "6. **RTL halka tamponu:** kayan pencere Python'da; RTL'de bant başlangıcı (j0) var, T = 256 ile 48 adımlık "
         "senaryolar sığıyor; uzun akış için halka tampon Proje II.\n",
         "## Değişiklik kaydı\n", md_table(["Tarih", "Sürüm", "Değişiklik", "Veri/model etkisi"],
                                          [("21.11.2026", "v0.1", "ilk sözleşme", "derlem v1")]),
         "\n## Gözden geçirme kaydı\n\n| Tarih | Gözden geçiren | Karar | Açık maddeler |\n|---|---|---|---|\n| | | | |\n"]
    with open(os.path.join(M1, "README.md"), "w") as f:
        f.write("\n".join(s))


def main():
    os.makedirs(M1, exist_ok=True)
    with open(os.path.join(DOC, "D1.3_alt_sistem_ve_ariza_tanimi.md"), "w") as f:
        f.write(d13())
    with open(os.path.join(DOC, "D1.4_eylem_ve_guard_tablolari.md"), "w") as f:
        f.write(d14())
    fig_directed()
    fig_pairs_and_costs()
    readme()
    print("M1 package ->", M1)


if __name__ == "__main__":
    main()
