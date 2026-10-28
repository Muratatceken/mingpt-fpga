# İlerleme raporu: case arayışından geri besleme kısıtlı sıralı hesaplamaya

**Kapsam.** 27–28 Ekim 2026 arasında yapılanlar. Başlangıç noktası, hocanın "bu proje hangi problemi çözüyor?" sorusu ve bunun sonucunda araştırmanın "modeli FPGA'ye sığdırmak" yerine "modeli FPGA'nin fiziksel kısıtlarından türetmek" yönüne dönmesi.

**İlgili belgeler:**

- `docs/08_case_arastirmasi.pdf`: case araştırması;
- `docs/09_faz0.pdf`: Faz 0 sonuçları;
- `docs/10_teori.pdf`: teori çerçevesi.

**Kod ve sonuçlar:** `sw/fbseq/`, `model/fbseq/*.json`, `scripts/`.

---

## 1. Tek bakışta

| | Durum |
|---|---|
| Hocanın case sorusu | **Cevaplandı:** deterministik dil modeliyle kayıpsız telemetri ve log sıkıştırma; yan case olarak aynı modelle anomali tespiti |
| Yeni araştırma yönü | **Belirlendi:** geri besleme kısıtlı (feedback-constrained) sıralı hesaplama. Döngüdeki mantık derinliği tasarımın merkez değişkeni |
| Faz 0 (fizibilite) | **Büyük ölçüde tamam:** 4 hipotezden ikisi tuttu, biri kısmen tuttu, biri Faz 1'e kaldı |
| İlk teori sonucu | **Var:** A5 için tek LUT seviyesinde en az state, sıkı biçimde 6 bit (SAT tanığı ve bilgi alt sınırı) |
| Gerçek veride kalite | Boolean model, doğru eğitimle pencere modelini geçiyor. 100 KB sözlüklü zstd'den %25 daha az bit harcıyor; GRU %32 |
| Literatürdeki konum | Üç hatlı tarama yapıldı; rakipler ve boşluklar belirlendi |
| Çalışmakta | Transition codebook (A5 ve Loghub), 2-bit ve 3-bit bloklarla Loghub |

---

## 2. Kronoloji

### 2.1 Hız sorusunun cevabı: CPU baseline
- Karpathy'nin `run.c`'si, stories260K'yı kartla aynı koşulda çalıştırdı: Apple M1, tek thread, 256 token, greedy. `make cpu-baseline`; `model/zoo/cpu_baseline.json`.
- Sonuç:

  | Platform | token/s | Token başına cycle |
  |---|---|---|
  | M1, -Ofast | 14 397 | ~222 bin |
  | M1, -O3 | 6 718 | – |
  | FPGA, 25 MHz | 496 | 50 397 |

  CPU mutlak hızda 29 kat önde; token başına cycle'da FPGA 4,4 kat verimli.

- Sunuma eklendi: slayt 2 ("dört soru, dört cevap") ve yeni slayt 46 (kaybın kaynağı ve CPU karşılaştırması). Commit `bee67be`.

### 2.2 Case araştırması (`docs/08`, commit `9f6939a`)
Dört paralel araştırma (sıkıştırma, CAN bus, sertifikasyon, rakip platformlar) ve bir puanlama matrisi:

- **Ana case: kayıpsız sıkıştırma.** Arithmetic coding'de encoder ve decoder olasılığı bit bit aynı hesaplamak zorunda. Float modeller bunu farklı donanımlarda garanti edemiyor (Ballé 2019, llama-zip, NNCP v2). Bizim bit-exact integer engine'imiz bunu yapısı gereği sağlıyor.
- **Yan case: anomali tespiti.** −log p hem kodlama maliyeti hem anomali skoru olduğu için aynı model iki işi birden yapıyor (telemetri ya da CAN bus).
- **Çapraz gereksinim: sertifikalanabilirlik.** EASA AI rehberi FPGA'yi mevcut sertifikasyon yolunun içinde sayıyor, GPU'yu kapsam dışında bırakıyor.
- **Dürüst konumlandırma.** Ham hızda ve enerjide üstünlük iddia edemiyoruz. MCU'lara göre 5–26 kat hızlıyız.

### 2.3 Yön tartışması
Hocanın yönü, iki ayrı yorum metni üzerinden tartışıldı. Kendi ölçümlerimiz yönü belirledi:

- **BRAM darboğaz:** stories260K'da %72; iki model RAMB18 parçalanması yüzünden place edilemedi.
- **Logic kaynakları boş:** LUT bel'lerinin en fazla %31'i kullanılıyor, DSP hiç kullanılmıyor.
- **Kritik yol:** gecikmenin %73'ü routing.

Bunun sonucunda:

- **Memory-centric model elendi**, çünkü BRAM zaten dolu.
- **Logic-native yön seçildi:** boş duran logic kaynağını kullanmak.
- **"Neden recurrent?" sorusunun cevabı:** belirleyici olan recurrence değil, loop-carried dependency'nin derinliği. Döngüsü bir mux kadar olan, geri kalanı pipeline'lanan bir yapı hedeflendi. Encoder tarafında girdinin tamamı baştan bilindiği için paralellik mümkün.
- **Dört aday:**
  - A: pencereli feedforward;
  - B: input-gated Boolean recurrence (set/keep, toggle, 2- ve 3-bitlik bloklar);
  - C: GRU;
  - D: microgpt.
- **Teorinin önerdiği tasarım:** transition codebook. Her blok, keyfi bir tablo yerine m öğrenilmiş dönüşümden birini seçiyor; döngüde her bit için tek bir LUT(kb + log₂m) kalıyor.

### 2.4 Faz 0 altyapısı (commit `985fbba`)
- **`sw/fbseq/fbseq.py` (CPU):** dört aday, cebirsel görevler (flag, parity, mod 3, A5), Loghub byte LM, klasik sıkıştırıcı baseline'ları (gzip, xz, zstd, zstd + sözlük).
- **`sw/fbseq/fbseq_gpu.py`:** aynı modeller, B için associative scan, cuDNN GRU, paralel işler. Mac'te test edildi.
- **`scripts/colab_faz0.sh`:** Colab'da tek komutluk çalıştırma. Kullanıcının hesabına A100 yerine T4 düştü. Küçük modellerde T4, M1'den belirgin hızlı değil; tek istisna GRU (~2 kat).
- **Colab CLI** (`google-colab-cli 0.7.5`) kuruldu ve giriş yapıldı.
  - Tarayıcıdaki runtime'a bağlanamıyor; CLI yalnız kendi açtığı runtime'ları yönetebiliyor.
  - Hesapta 0 compute unit var; A100 açılmadı.
- **`sw/fbseq/loop_bench.py`:** döngü devrelerini Yosys + nextpnr ile place edip Fmax ölçüyor.

### 2.5 Faz 0 deneyleri
Ayrıntı `docs/09_faz0.md`'de; özet:

1. **Döngü derinliği** (`loop.json`): tek mux 478, 4:1 mux 432, 8:1 mux 399 MHz. State'e bağlı 1/2/3/4 seviye: 366 / 259 / 192 / 154 MHz. **H1 tuttu.**
2. **Cebirsel görevler** (`synth.json`, 3 seed): set/keep → flag; + toggle → parity; 2 bit → mod 3; 3 bit → A5'te 0,32. Sıralama teoriyle bire bir. **H2 tuttu.**
3. **Byte LM** (`bytes.json`, `base.json`, Loghub, 256 B bağımsız paket):

   | Model | bpb |
   |---|---|
   | GRU | **1,294** |
   | Pencere 8 | 1,453 |
   | B · set/keep (soft→hard) | 1,583 |
   | zstd + 100 KB sözlük | 1,90 |
   | zstd, sözlüksüz | 5,51 |
   | microgpt | 1,917 (bu protokolde yetersiz eğitim) |

4. **Soft–hard farkı** (`gap.json`):
   - B'nin soft hali 1,398; pencere modelini geçiyor.
   - Doğrudan hard 2,77; uyarlanmış hard 1,583; Δ_hard = 0,185.
   - Soft state'ler 0 ve 1'den uzak.
   - D(t), farklı bit oranı: %6 ile %12 arasında; t ≈ 8–16'da doyuyor.
5. **GRU bellek ufku** (`gru_horizon.json`): 128 byte'ta sıfırlama 1,39, 64'te 1,58. 64–128 byte ötesindeki bilgi işe yarıyor; ölçümde karışık etkenler var (§5).
6. **Hard-aware eğitim** (`hardaware_*.json`):
   - **Loghub:** baştan Boolean forward ve eğim tavlamasıyla **1,418** (pencere 8: 1,453; soft: 1,398). Sade Hard-ST 1,545, Gumbel-ST 1,685. **H3 kısmen tuttu.**
   - **A5:** soft→hard seed'lerin yarısı %100; baştan hard 8 koşunun hiçbiri 0,13'ü geçmiyor.
   - **Ödünleşim:** keşif ile uyum. Dil modelinde baştan Boolean, durum izlemede soft aşama gerekiyor.

### 2.6 Teori (`docs/10_teori.md`; commit'ler `1c348ab`, `f45da5e`)
- **Biçimsel model:** K-LUT recurrent devre, block-local yapı, maliyet dörtlüsü (S, D, L, R).
- **Bilinen ve atıflı:** cascade'in iteration bound'u (Leiserson–Saxe, Thm 11), grup word problem'leri için encoding'ler, Myhill–Nerode alt sınırı.
- **Bizim sonuçlarımız (A5 küçük vakası):**

  | Gerçekleştirme | State | Döngü | Fmax (MHz) |
  |---|---|---|---|
  | Öğrenilen, tam tablolu 3-bit bloklar | 24 | 2 seviye | 399 |
  | Elle, noktasal | 15 | LUT5 | 433 |
  | Block-local en küçük (V4 + A4 coset'leri; K = 5'te 9 bit; K = 4'te imkânsız) | 7 | LUT6 | 458 |
  | SAT, alt sınırda | **6** | LUT6 | 455 |

  İspat yolu: Hartmanis–Stearns SP partition'ları, grup makinelerinde alt grupların coset'lerine karşılık gelir; injectivity ⇔ alt grupların kesişimi birim eleman; A5'in 59 alt grubu tarandı. Kod: `sw/fbseq/theory_sat.py`.

- **Kararlılık:**
  - hata rekürsiyonu sınırı ve koşullu beklenen değer sınırı;
  - tek bit için 𝔽₂ tam eşitliği;
  - bloklara genelleme: birleşen map'ler farkı siler, permütasyonlar korur;
  - kayba bağlantı (Lipschitz sınırı);
  - **bellek ile dayanıklılık aynı mekanizmaya bağlı.**
- **Sanı** (literatür ajanının taslağı, kontrol edilmeli): k-bitlik bloklar tam olarak, sözdizimsel monoidinin basit grup bölenleri S_{2^k}'yi bölen dilleri tanır. Test edilebilir öngörüleri var.

### 2.7 Literatür taraması (üç hat)

**Klasik:**

- Leiserson–Saxe 1991; Lin–Messerschmitt 1991 (FSM look-ahead);
- Stearns–Hartmanis 1961 (partition pair: Teorem C'nin çekirdeği);
- Derby 2001 (CRC state-space dönüşümü: doğrusal durum; doğrusal olmayan genellemesi bulunamadı);
- Barkalov 2021 (LUT tabanlı FSM);
- Shende 2003 (parity engeli).

**İfade gücü:** Sarrof 2024, Grazzi 2025, Merrill 2024, PD-SSM 2025 (en yakın nöral akraba), Knorozova & Ronca 2024, Zhang & Li 2026 (0,32'nin teşhisi: coset aşaması).

**Logic ağları:**

- RDDLGN (EdgeFM 2025 atölyesi): keep yolu yok, donanım yok.
- Gating LGN (NeurIPS 2026 AXIOM atölyesi): bit başına keep/write mux, donanım yok.
- Mind the Gap 2025, Kim 2026 (Hard-ST ve CAGE), HM-RNN 2017.

**Bulunamayanlar (bizim için boşluklar):**

- sentezlenmiş bir recurrent logic ya da LUT modeli ve Fmax'i;
- logic veya LUT ağlarıyla byte düzeyinde bpb;
- döngü derinliğini tasarım kısıtı yapan bir çalışma;
- Boolean transition codebook;
- adım bazında discretization gap ölçümü;
- LUT fan-in kısıtlı en küçük grup gerçekleştirmeleri.

---

## 3. Bugün ne iddia edebiliriz, neyi edemeyiz?

**Edebiliriz:**

1. Döngüdeki mantık derinliği ile Fmax arasındaki ilişkiyi XC7A100T'de ölçtük.
2. Boolean blok recurrence'ın ifade gücü sıralaması teoriyle örtüşüyor; ölçtük.
3. A5 için tek seviyeli fan-in 6 gerçekleştirmenin en küçük state'i tam olarak 6 bit; block-local kısıtla 7 bit. Hepsi doğrulanmış ve place edilmiş.
4. Gerçek log verisinde Boolean recurrence, doğru eğitimle aynı bütçedeki pencere modelini ve sözlüklü zstd'yi geçiyor. Bu ideal bit maliyeti; model float.
5. Soft–hard farkı ve adım bazında state ayrışması ölçüldü. Hard-forward ve eğim tavlaması bu farkı büyük ölçüde kapatıyor.

**Edemeyiz (henüz):**

1. Gate ağları ve olasılık head'i float. Model bütünüyle bir LUT devresi değil (H4, Faz 1).
2. Gerçek sıkıştırma yok: range coder ve bit-exact encode/decode yapılmadı.
3. Görülmemiş log sistemlerine genelleme test edilmedi.
4. Bütün donanım sonuçları simülasyon ve STA'ya dayanıyor; enerji ölçülmedi.
5. A5'in eğitimle güvenilir biçimde öğrenilmesi çözülmedi (seed'lerin yarısı).
6. Teoride yeni olan yalnız A5 küçük vakası; genel teorem (Problem C) açık.

---

## 4. Q1 açısından değerlendirme
- **Yeni ve savunulabilir bir çekirdek oluştu:** ifade gücü, encoding, döngü maliyeti ve fiziksel maliyet arasındaki ilişki; ölçülmüş FPGA karşılığı ve gerçek bir uygulama (sıkıştırma) ile birlikte.
- **Henüz eksik:**
  - genel bir teorem (A5'in ötesi);
  - Boolean devreleri güvenilir biçimde eğiten bir yöntem;
  - LUT'a çevrilmiş tam model.
- **Değerlendirme:** Bugün sağlam bir bitirme ve iyi bir konferans çalışması düzeyindeyiz. Q1 için bu üç eksikten en az ikisinin kapanması gerekiyor.

---

## 5. Hatalar, düzeltmeler ve sınırlar (şeffaflık için)

| Konu | Ne oldu | Durum |
|---|---|---|
| "A5 için en az iki LUT seviyesi gerekir" iddiası | Yanlıştı; A5 tek seviyede (LUT5) gerçekleştirilebiliyor | Düzeltildi |
| "Büyük basit grup → derin döngü" alt sınır sanısı | Genel olarak yanlış; one-hot encoding ile her basit grup fan-in 5'te | Karşı örnek belgelendi |
| "B'nin geride kalmasının sebebi mimari değil" | Erken bir iddiaydı; soft model aynı fonksiyon sınıfında değil | Düzeltildi |
| hardst yöntemi | Yanlışlıkla eğim tavlaması kullanıyordu | Düzeltildi, yeniden koşuldu |
| kb ≥ 2 bloklarda NaN | Bit bit multilinear relaxation'ın Jacobian'ı 1'den büyük; gradyan patladı | Kararlı dağılım formuna geçildi; CPU ve GPU soft ve hard eşit |
| microgpt 1,917 | Aynı protokol bir transformer için yetersiz | Adil eğitim yapılacak |
| GRU bellek ufku | Eğitim/değerlendirme uyumsuzluğu; blok sıfırlama ≠ kayan pencere | Kayan pencere GRU'su yapılacak |
| T4 hızlanması | Küçük modellerde beklenen hızlanma yok | Kayıt altında |
| Araştırma ajanı | Kullanıcının derlediği `run.c` binary'lerini kendiliğinden çalıştırdı; repoya yazmadı | Kullanıcıya bildirildi |
| PDF oluşturucu | Chrome 156 PDF'i yazdıktan sonra hata koduyla çıkıyor | Dosyanın varlığı kontrol ediliyor |
| Veri | Loghub küçük; test, aynı dosyaların son %20'si | Leave-one-system-out yapılacak |

---

## 6. Çalışan ve sıradaki işler

**Çalışıyor:**

- transition codebook: A5'te (softhard ve eğim tavlaması, 4 seed) ve Loghub'da (eğim tavlaması, 2 seed);
- 2-bit ve 3-bit bloklar, eğim tavlamasıyla Loghub'da.

**Öncelik sırasıyla sıradakiler:**

1. Codebook sonuçları: aynı ifade gücü daha kısa döngüyle, öğrenilebilir mi?
2. A5'i güvenilir öğrenmek: PD-SSM reçetesi, CAGE, curriculum; amaç seed başarı oranını %50'den yukarı çekmek.
3. Problem C'nin genelleşmesi:
   - K = 5 ve K = 4 için block-local olmayan SAT;
   - S₄, mod 5, mod 7 görevleriyle sanının testi;
   - β_K ölçüsünün diğer gruplarda hesaplanması.
4. Kararlılık: tek bit çevirme deneyi ve head'in Lipschitz sabiti.
5. Faz 1:
   - gate ağlarını ve head'i LUT ağlarına çevirmek;
   - bit bit olasılık head'i ve range coder ile gerçek sıkıştırma;
   - leave-one-system-out değerlendirme;
   - adil microgpt eğitimi ve kayan pencere GRU'su.

---

## 7. Dosya ve commit dizini

| Commit | İçerik |
|---|---|
| `bee67be` | CPU baseline, sunum slayt 2 ve 46 |
| `9f6939a` | Case araştırması (`docs/08`) |
| `985fbba` | `sw/fbseq` altyapısı, GPU sürümü, Colab script'i, döngü ölçümleri, baseline'lar |
| `1c348ab` | Teori belgesi, A5 gerçekleştirmeleri, SAT, soft–hard deneyi, kararlı relaxation |
| `f45da5e` | Faz 0 belgesi, düzeltilmiş hard-aware sonuçları, codebook modeli |

| Dosya | Görev |
|---|---|
| `sw/fbseq/fbseq.py` | Modeller, görevler, Loghub, baseline'lar (CPU) |
| `sw/fbseq/fbseq_gpu.py` | GPU sürümü, scan == döngü kontrolü |
| `sw/fbseq/loop_bench.py` | Döngü devrelerinin Fmax'i |
| `sw/fbseq/controls.py` | Soft–hard, A5 elle çözüm, maliyet, bellek ufku |
| `sw/fbseq/hardaware.py` | Hard-forward eğitim yöntemleri |
| `sw/fbseq/theory_sat.py` | A5 alt grupları, block-local minimum, SAT |
| `sw/fbseq/codebook.py` | Transition codebook |
| `scripts/get_loghub.sh`, `scripts/colab_faz0.sh`, `scripts/cpu_baseline.sh` | Veri, Colab, CPU baseline |
| `model/fbseq/*.json` | Bütün sonuçlar |
