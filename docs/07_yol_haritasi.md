# Q1 dergi yayınına giden yol haritası

**Amaç:** Projeyi bir bitirme çalışmasından Q1 bir dergide yayımlanabilecek bir araştırmaya taşımak. Bu belge, bugünkü konumu bir hakem gözüyle değerlendirir, literatürdeki boşluklara göre üç araştırma hattı önerir ve bunları algoritmik, matematiksel ve donanım görevlerine böler. Literatür bilgileri Ekim 2026 sonunda yapılan taramaya dayanır; “bulunamadı” ifadeleri kanıt değil, taramanın sonucudur.

---

## 1. Hedef dergiler

| Dergi | Scimago 2025 | Uygunluk |
|---|---|---|
| IEEE Transactions on Circuits and Systems I (TCAS-I) | Q1, SJR 1,947 | En iyi seçenek; EdgeLLM (FPGA LLM hızlandırıcısı) burada yayımlandı |
| IEEE Transactions on Computers (TC) | Q1, SJR 1,243 | Mimari karşılaştırma ve maliyet modeli ağırlıklı bir makale için |
| IEEE Transactions on VLSI Systems (TVLSI) | Q1 (2025), SJR 1,041 | Donanım ağırlıklı; 2024 çeyreği doğrulanmadı |
| IEEE JETCAS | Q1, SJR 0,953 | Özel sayılar (edge AI, düşük bit) için |
| IEEE TCAD | Q1, SJR 0,928 | Doğrulama, derleyici ve ispatlı sınırlar ağırlıklı bir makale için |
| ACM TRETS | Q2, SJR 0,741 | Konuya en uygun dergi, ancak Q1 değil |

**Öneri:** Ana makale için TCAS-I, ikinci (doğrulama ağırlıklı) makale için TCAD. Kategori bazında çeyreği başvurudan önce yeniden kontrol et.

---

## 2. Bugünkü konum: bir hakem ne der?

**Güçlü yanlar**
- Tek mikrokodlu engine’de 10 model; hepsi pin seviyesinde bit-exact.
- Sampling dahil tamamen bağımsız kart; host yok.
- Eşit bütçeyle mimari karşılaştırma ve SPECIALIZE ile özellik bazlı maliyet.
- İspatlı aralık sınırları; dört araç hatasının tespiti.
- Tamamen açık kaynak akış ve kod.

**Hakemin büyük ihtimalle yazacağı eleştiriler**
1. **Ölçek küçük.** 50 bin – 260 bin parametre “oyuncak” olarak görülür. Çip içi çalışmak bu ölçekte beklenen bir sonuçtur, katkı sayılmaz.
2. **Performans zayıf.** 25 MHz, DSP kullanılmıyor, enerji ölçümü yok. Donanım dergileri token/J ve gerçek ölçüm bekler.
3. **Karşılaştırma yok.** CPU, mikrodenetleyici, GPU ve önceki FPGA çalışmalarıyla (HLSTransform, TALOS-V2, TeLLMe) ölçülebilir bir kıyas yok.
4. **Yenilik sorgulanır.** Komutla programlanan engine (FlightLLM, DFX, VTA) ve tamsayı çekirdekler (I-BERT, I-LLM) biliniyor. TALOS-V2 de Karpathy’nin microGPT’sini SystemVerilog’da çalıştırıyor.
5. **Fiziksel ölçüm yok.** Simülasyon ve statik zamanlama analizi doğruluğu kanıtlar, ama güç ve gerçek hız için donanım dergileri çoğunlukla kart ölçümü ister.

**Sonuç:** Bugünkü hâliyle sağlam bir bitirme ve iyi bir konferans/workshop çalışması. Q1 için en az üç şey gerekiyor:
- açık bir araştırma sorusu ve onu cevaplayan kontrollü deneyler;
- ölçek (milyonlarca parametre) ve bellekten sınırlı (memory-bound) bir rejim;
- enerji dahil karşılaştırmalı ölçümler.

---

## 3. Literatürdeki boşluklar

| Boşluk | Durum | En yakın çalışmalar |
|---|---|---|
| Her mimari özelliğin (RMSNorm, RoPE, GQA, SwiGLU, QK-norm) eşit bütçede FPGA maliyeti | Bulunamadı; RoPE’un FPGA maliyetini ölçen çalışma yok | HW-GPT-Bench (yalnız genişlik/derinlik, FPGA yok), MobileLLM (yalnız kalite), HAAN ve SOLE (tek operatör) |
| Bütün operatörlerde (softmax, RMSNorm, RoPE dahil) ispatlı aralık ve bit-exact doğrulama | Bulunamadı | A2Q/A2Q+ (yalnız linear akümülatörleri), I-ViT, I-LLM (deneysel), QEBVerif (küçük MLP’ler) |
| LittleBit tarzı low-rank binary factor’ların donanım uygulaması | Bulunamadı | LittleBit, LittleBit-2, NanoQuant, DBF (hepsi GPU/CPU); COBRA (1 bit, üretim yok) |
| Host’suz, sampling dahil tamamen fabric içinde LM | Bulunamadı (hakemli) | HLSTransform (sampling host’ta), TALOS-V2 (JTAG ile sürülür, hakemsiz), Zynq tabanlı çalışmalar (ARM kullanır) |
| Açık kaynak akışla (Yosys + nextpnr) LM hızlandırıcı ve Vivado ile QoR karşılaştırması | Bulunamadı | – |

---

## 4. Üç araştırma hattı

Bir Q1 makalesi tek ve net bir soruya cevap vermeli. Aşağıdaki üç hattan **biri ana makale**, diğerleri destek ya da ikinci makale olur.

### Hat A · “Mimari fikirlerin donanım fiyatı” (önerilen ana hat)

**Araştırma sorusu:** Sabit bir parametre bütçesinde, modern dil modeli tasarım fikirlerinin her biri kalitede ne kazandırır, donanımda ne kaybettirir? Bir edge FPGA için Pareto-optimal mimari nedir?

**Neden güçlü:** Literatürde bu karşılaştırma yok. Karpathy’nin depoları tarihsel bir “merdiven” sunuyor (bigram → nanochat) ve bizim engine’imiz her basamağı aynı donanımda ölçebiliyor.

**Yapılacaklar**
1. **Ladder deneyleri.** `sw/zoo/models.py`’deki `Ladder` sınıfıyla GPT-2’den Llama’ya ve nanochat’e her özelliği tek tek aç ve kapat:
   - norm türü: LayerNorm, RMSNorm, parametresiz RMSNorm, normalization’sız DyT;
   - pozisyon: learned, RoPE, ALiBi;
   - attention: MHA, GQA, MQA;
   - activation: GELU, SwiGLU, ReLU²;
   - QK-norm var ya da yok.
2. **Üç bütçe, iki görev, çoklu tohum.** 12K / 50K / 260K / 1M parametre; names ve TinyStories; her nokta için 3–5 tohum ve güven aralıkları.
3. **Her nokta için ölçümler.** Kalite, cycle/token, LUT/FF/BRAM/DSP, Fmax, güç (Vivado tahmini ya da kart) ve token/J.
4. **Analitik maliyet modeli.** Cycle’lar için kapalı form zaten var (GEMV: ⌈n_out/P⌉·n_in, attention: 2·L·H·(t+1)). Bunu LUT ve Fmax için de regresyonla kur ve tasarım uzayını model üzerinden tara.
5. **Pareto analizi.** Kalite–enerji ve kalite–gecikme sınırları; “bu bütçede şu özellikler kendini öder” türünden somut öneriler.

**Matematiksel katkı fırsatı:** Her özelliğin maliyetini cycle, alan ve enerji olarak ayrıştıran toplamsal bir model, ve modelin ölçümlerle doğrulanması.

### Hat B · “İspatlı doğrulukla tamsayı dil modeli çıkarımı” (ikinci makale, TCAD)

**Araştırma sorusu:** Tamsayı bir dil modeli çıkarımının float modelden sapması önceden, matematiksel olarak sınırlanabilir mi? Bu sınır, üretilen token’ların doğruluğunu sertifikalamak için kullanılabilir mi?

**Yapılacaklar**
1. **Operatör bazlı hata sınırları.** Her çekirdek için en kötü durum ve istatistiksel hata sınırı:
   - requant: bağıl hata ≤ 2⁻¹⁶;
   - exp tablosu: < %0,27;
   - isqrt ve bölme: ispatlı;
   - RoPE: sin/cos tablosunun hatası;
   - SwiGLU: iki quantization hatasının çarpımı.
2. **Ağ boyunca yayılım.** Interval ya da affine aritmetiğiyle layer’lar boyunca hata yayılımı ve logit’ler için bir üst sınır.
3. **Sertifikalı decoding.** Greedy üretimde, en büyük iki logit arasındaki fark yayılmış hata sınırından büyükse, tamsayı ve float model aynı token’ı seçmek zorundadır. Adımların yüzde kaçının sertifikalandığını ölç; sertifikalanamayan adımları ayrıca incele. Bu, literatürde görmediğimiz, matematiksel olarak ilginç bir katkı.
4. **Sınırdan bit genişliği optimizasyonu.** A2Q’nun linear katmanlarla sınırlı fikrini bütün operatörlere genişlet. İspatlı sınırlarla her akümülatörün ve tamponun en küçük genişliğini seç; problemi “kısıtlar altında alan minimizasyonu” (ILP) olarak kur.
5. **Biçimsel eşdeğerlik.** Her operatör için RTL ile referans arasında SymbiYosys ile biçimsel eşdeğerlik kanıtı. Netlist simülasyonunun bulduğu üç araç hatası bu bölümün motivasyonu olur.

### Hat C · “Low-rank binary weight’ler ne zaman kendini öder?” (LittleBit hattı)

**Araştırma sorusu:** Sub-1-bit low-rank binary factorization (LittleBit, NanoQuant, DBF), binary ve INT4’e göre hangi katman genişliğinde ve hangi bellek bant genişliğinde avantajlı hâle gelir?

**Temel gözlem:** Bizim ölçeğimizde model çip içine sığdığı için BPW’nin hıza etkisi sınırlı. LittleBit’in asıl değeri, weight’lerin DRAM’den aktığı ve bant genişliğinin darboğaz olduğu rejimde ortaya çıkar. Bu yüzden bu hat DDR yolunu gerektirir.

**Yapılacaklar**
1. **Kapalı form geçiş noktası.** BPW formülümüzü (k[r(dₒ+dᵢ) + 16(dₒ+dᵢ+r)] / (dₒdᵢ)) bir hata modeliyle birleştir. Aynı bit bütçesinde LittleBit’in binary’yi geçtiği katman genişliğini (d*) analitik olarak türet. Ölçtüğümüz bulgu (geçiş 64 ile 256 arasında) bu türetimi doğrulamalı. Teorem ve deney birlikte güçlü bir katkı olur.
2. **Katman başına rank seçimi.** Bit bütçesi altında hatayı en aza indiren rank dağılımını bir optimizasyon problemi olarak çöz (Lagrange ya da dinamik programlama).
3. **İki yollu LittleBit ve LittleBit-2’nin donanımı.** Mevcut GEMVB birimi aynen kullanılır; yalnızca op sayısı artar.
4. **DDR2 akışlı engine.** Nexys A7’nin 128 MB DDR2’si üzerinden weight akıtan bir sürüm. Karpathy’nin stories15M, stories42M ve stories110M modelleri “yalnızca Karpathy modelleri” kuralına uyar; stories110M ile HLSTransform’la doğrudan karşılaştırma yapılabilir.
5. **Roofline analizi.** Hesap ve bellek sınırlarını çiz; INT8, INT4, binary ve LittleBit’in roofline üzerindeki yerlerini göster.

---

## 5. Algoritmik fikir havuzu

| Fikir | Kısa açıklama | Beklenen etki |
|---|---|---|
| Normalization’sız model (DyT) | Norm yerine tanh(αx); tamsayıda tek bir 256 satırlık tablo | isqrt ve bölme ortadan kalkar; nanochat’te cycle’ların %33’ü ve kritik yolun bir kısmı |
| Speculative decoding | Taslak model olarak bigram ya da mlp, doğrulayıcı olarak transformer; ikisi de çipe sığar | Token/s’de kabul oranına bağlı 1,5–3 kat; engine’e birden çok token’ı aynı anda doğrulama (GEMM) gerekir |
| Tamsayı sampling’in analizi | Min-p ve temperature’ın tamsayı yaklaşımının dağılım hatası (TV mesafesi) için sınır | Sampling’in de sertifikalanması |
| Karışık hassasiyet | Katman başına INT8, INT4 ya da binary, Hessian ya da duyarlılık analiziyle | Aynı kalitede daha az bellek |
| Activation quantization | INT4 activation’lar, LUT tabanlı çarpma (Geens vd. ISPASS 2026’nın uyarısına dikkat) | Daha küçük çarpıcılar |
| LUT-native binary GEMV | Popcount ağaçlarını doğrudan LUT6’ya eşleme ve toplama ağacıyla karşılaştırma | Alan ve Fmax iyileşmesi |
| KV cache sıkıştırma | KV cache’i INT4’e indirme, uzun context’te BRAM kazancı | stories modellerinde daha uzun context |

---

## 6. Donanım ve ölçüm yol haritası

1. **Performans.**
   - Engine’e pipeline ekle; hedef 50 MHz, ardından 100 MHz.
   - DSP’leri Vivado ile ya da Yosys hatası aşılarak kullan.
   - P = 8 → 16/32 tasarım uzayı (kod hazır: `zoo-dse`).
   - Attention’da paralel head.
2. **Enerji.** Vivado’nun güç tahmini en az gereklilik. Hakemler gerçek ölçüm ister: kartta ölçüm için USB güç ölçer ya da kartın besleme hattına bir akım sensörü.
3. **Karşılaştırmalar.**
   - Aynı modeller llama2.c (run.c) ile bir dizüstü CPU’da, bir mikrodenetleyicide (ör. RP2040 / Cortex-M) ve bir GPU’da.
   - HLSTransform, TALOS-V2 ve TeLLMe ile normalize edilmiş ölçütler: token/s, token/J, token/s/LUT.
4. **Vivado ile açık kaynak akışın QoR karşılaştırması.** Aynı RTL için Fmax, LUT ve güç farkı; literatürde ölçülmemiş ek bir katkı.
5. **Doğrulamanın tamamlanması.** Vivado post-route simülasyonu, SymbiYosys biçimsel kanıtları ve mümkünse kısa bir kart testi (hesap doğruluğu simülasyonla kanıtlı; kart ölçümü güç ve gerçek hız içindir).
6. **Bellek düzeni.** ts_gpt2 ve ts_microgpt’nin yerleşmesini engelleyen RAMB18 bölünmesinin giderilmesi.

---

## 7. Önerilen makale planı

**Çalışma başlığı (örnek):** “What Does Each Architectural Idea Cost? An Equal-Budget, Bit-Exact Study of Karpathy’s Language Models on a Standalone FPGA”

**Katkılar (taslak)**
1. Dokuz mimari ve ablation merdiveni için eşit bütçede kalite, cycle, alan, Fmax ve enerji ölçümleri; özellik bazlı toplamsal maliyet modeli.
2. Bütün operatörleri ispatlı sınırlarla kapsayan, tamamen bağımsız ve bit-exact doğrulanmış bir tamsayı engine (host’suz sampling dahil).
3. Low-rank binary factorization’ın ilk donanım uygulaması ve binary’ye karşı geçiş noktasının analitik türetimi.
4. Açık kaynak araç zinciriyle tekrarlanabilir akış ve Vivado ile karşılaştırma.

**Zaman planı (yaklaşık)**

| Ay | İş |
|---|---|
| 1 | Pipeline ile 50 MHz; DSP; Vivado ile Fmax ve güç; ts_gpt2/ts_microgpt bellek düzeni |
| 2–3 | Ladder deneyleri (3–4 bütçe, 2 görev, çoklu tohum); maliyet modeli |
| 2–4 | Hat B: hata sınırları, sertifikalı decoding, bit genişliği optimizasyonu |
| 3–5 | Hat C: DDR2 akışı, stories15M/42M/110M; geçiş noktası teoremi; roofline |
| 4–5 | Karşılaştırmalar (CPU, MCU, GPU, önceki FPGA çalışmaları); enerji ölçümü |
| 6 | Yazım; TCAS-I’ye başvuru; Hat B için TCAD taslağı |

---

## 8. Riskler ve önlemler

| Risk | Önlem |
|---|---|
| TALOS-V2 ya da TeLLMe gibi çalışmalar yenilik iddiasını zayıflatır | Katkıyı “birleşim ve kontrollü ölçüm” olarak konumla; bu çalışmaları açıkça karşılaştır |
| Kart ölçümü olmadan donanım sonuçları zayıf kalır | En azından Vivado güç tahmini; mümkünse kısa süreli bir kart erişimiyle güç ölçümü |
| Küçük ölçek eleştirisi | DDR2 ile stories110M’e kadar ölçekle; küçük modelleri kontrollü deney aracı olarak savun |
| Açık kaynak araçların olgunluğu (nextpnr 7 serisi deneysel) | Sonuçları Vivado ile çapraz doğrula; açık kaynak akışı ek katkı olarak sun |
| Yeniden eğitim maliyeti (çoklu tohum, çoklu bütçe) | Küçük modeller dakikalar içinde eğitiliyor; TinyStories için bütçeyi sınırla |

---

## 9. Kaynaklar (seçme, Ekim 2026 taraması)

- HLSTransform: He vd., ICML 2024 ES-FoMo, arXiv:2405.00738
- TALOS-V2: Abeykoon ve Chhajer, GitHub 2026 (hakemsiz)
- FlightLLM: Zeng vd., FPGA 2024, arXiv:2401.03868
- EdgeLLM: Huang vd., IEEE TCAS-I 2025, arXiv:2407.21325
- TeLLMe: Qiao vd., arXiv:2504.16266; TerEffic: Yin vd., arXiv:2502.16473
- COBRA: Qiao vd., arXiv:2504.16269
- MatMul-free LM: Zhu vd., arXiv:2406.02528
- I-LLM: Hu vd., arXiv:2405.17849; I-ViT: Li ve Gu, ICCV 2023
- A2Q / A2Q+: Colbert vd., ICCV 2023 / ICML 2024
- HW-GPT-Bench: Sukthanker vd., NeurIPS 2024 D&B; MobileLLM: Liu vd., ICML 2024
- DyT: Zhu vd., CVPR 2025, arXiv:2503.10622
- LittleBit: Lee vd., NeurIPS 2025, arXiv:2506.13771; LittleBit-2: arXiv:2603.00042; NanoQuant: ICML 2026, arXiv:2602.06694; DBF: TMLR 2026, arXiv:2505.11076
- Geens vd., ISPASS 2026, arXiv:2604.25183 (ternary LUT hızlandırıcılarının maliyet modeli)
