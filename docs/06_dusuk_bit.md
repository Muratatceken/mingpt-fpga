# Düşük bitli weight'ler: Larq tarzı binary/ternary ve LittleBit

Hocamızın önerdiği iki kaynak:

- **LittleBit** (Lee vd., NeurIPS 2025, arXiv:2506.13771): low-rank factorization ile 1 bitin altına
  inme.
- **Larq**: TensorFlow/Keras için binary neural network kütüphanesi.

Bu bölüm, ikisinin Karpathy modellerinde nasıl kullanıldığını ve donanıma neyin girdiğini anlatır.
Kural değişmedi: **FPGA'de eğitim yok.** Eğitim (QAT) Mac'te yapılır, FPGA yalnızca inference çalıştırır.

## 1. Ne denendi

`sw/zoo/lowbit.py`, modelin içindeki bütün linear katmanların weight'lerini değiştirir.

- **Değişmeyenler:** token embedding ve output head INT8 kalır (LittleBit makalesindeki ve binary ağ
  literatüründeki gibi). Activation'lar engine'de zaten INT8.
- **Eğitim:** float model öğretmen olur ve sonraki token dağılımlarına KL divergence ile knowledge
  distillation yapılır. Optimizer Adam, öğrenme oranı cosine.

| Yöntem | Weight | Kaynağı | Bizdeki karşılığı |
|---|---|---|---|
| `int8` | 8 bit, kanal başına scale | engine'in bugünkü formatı | eğitimsiz (PTQ) |
| `int4` | 4 bit, kanal başına scale | standart QAT | STE ile yuvarlama |
| `tern` | {−a, 0, +a} | Larq `SteTern(ternary_weight_networks=True)` (TWN) | eşik 0,7·ort\|w\|, STE \|w\| ≤ 1 |
| `bin` | {−a, +a}, a = ort\|w\| | Larq `MagnitudeAwareSign` (Bi-Real Net) + `SteSign` + `weight_clip` | STE \|w\| ≤ 1, latent weight [−1, 1] |
| `lb<r>p<k>` | diag(h)·sign(U)·diag(ℓ)·sign(V)ᵀ·diag(g) | LittleBit, Dual-SVID başlangıcı | rank = min(d)'nin %r'si, k yol (k = 2: residual yol) |

Larq'ı TensorFlow'a geçmeden kullandık. Kütüphanenin quantizer'ları birkaç satırlık fonksiyonlar
(ileri yönde sign, geri yönde kırpılmış özdeşlik); bunları Larq'ın belgelenen tanımlarına göre
PyTorch'ta yeniden yazdık (`lowbit.py`). Böylece modeller, eğitim düzeni ve integer akışı aynı kaldı.
Gerçek `larq` paketiyle sayısal bir karşılaştırma henüz yapılmadı. Bunun için TensorFlow kurulumu
gerekiyor.

**BPW** yalnızca değiştirilen katmanlar için hesaplanır: işaret bitleri ve 16 bitlik scale'ler.
LittleBit'te bunlar her yol için h, g ve ℓ; diğer yöntemlerde çıkış kanalı başına bir scale.

## 2. Sonuçlar

**names.txt** (~50 bin parametre, 1 000 test ismi). Her hücre, QAT sonrası test kaybı ve parantez içinde
BPW'dir. `int8` eğitimsizdir; diğerlerinin hepsi 3 000 adım QAT'tır.

| Model | float | int8 | int4 | tern | bin | lb50p2 | lb25p2 | lb25p1 | lb12p2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bow | 2,045 | 2,045 (8,15) | 2,049 (4,15) | 2,154 (1,73) | 2,122 (1,15) | 2,136 (1,82) | 2,186 (1,20) | 2,246 (0,60) | 2,239 (0,90) |
| gpt2 | 2,000 | 2,000 (8,38) | 1,983 (4,38) | 2,045 (1,96) | 2,029 (1,38) | 2,030 (2,83) | 2,090 (2,08) | 2,162 (1,04) | 2,125 (1,71) |
| gru | 1,956 | 1,957 (8,11) | 1,965 (4,11) | 2,052 (1,69) | 2,028 (1,11) | 1,990 (2,32) | 2,030 (1,43) | 2,086 (0,71) | 2,088 (0,96) |
| llama | 2,003 | 2,002 (8,42) | 1,980 (4,42) | 2,051 (2,00) | 2,052 (1,42) | 2,038 (3,29) | 2,090 (2,44) | 2,167 (1,22) | 2,123 (2,01) |
| microgpt | 1,971 | 1,971 (8,38) | 1,966 (4,38) | 2,086 (1,96) | 2,059 (1,38) | 2,031 (3,25) | 2,088 (2,38) | 2,156 (1,19) | 2,128 (1,94) |
| mlp | 1,986 | 1,986 (8,06) | 2,003 (4,06) | 2,178 (1,65) | 2,160 (1,06) | 2,030 (2,06) | 2,075 (1,18) | 2,142 (0,59) | 2,138 (0,72) |
| nanochat | 1,969 | 1,970 (8,36) | 1,958 (4,36) | 2,072 (1,95) | 2,062 (1,36) | 2,047 (3,18) | 2,094 (2,36) | 2,163 (1,18) | 2,128 (1,95) |
| rnn | 1,964 | 1,964 (8,07) | 1,972 (4,07) | 2,065 (1,65) | 2,076 (1,07) | 1,990 (2,17) | 2,040 (1,24) | 2,108 (0,62) | 2,102 (0,76) |

Okuma kılavuzu:
- **INT4:** her modelde float'la aynı; KD sayesinde bazen biraz daha iyi.
- **Ternary:** binary'den fazla bit harcıyor, sonucu binary'den iyi değil.
- **Binary:** transformer'larda (C = 32) +0,03…0,09 nat; en iyi 1 bitlik seçenek bu.
- **LittleBit:**
  - transformer'larda aynı bitte binary'nin gerisinde (gpt2: lb25p2 2,08 BPW'de 2,090, binary 1,38 BPW'de 2,029);
  - girişi geniş katmanlarda öne geçiyor (mlp: lb25p2 1,18 BPW'de 2,075, binary 1,06 BPW'de 2,160;
    rnn: lb25p2 1,24 BPW'de 2,040, binary 1,07 BPW'de 2,076).
- **Dual-SVID başlangıcı tek başına yetmiyor:** eğitimsiz kayıplar 3–12 nat; LittleBit'te QAT şart.

**TinyStories, stories260K** (Karpathy'nin hazır modeli, float 1,268; 100 test penceresi × 256 token).
Eğitilmiş bir modelin küçük weight'lerinde öğrenme oranı kritik: INT4'te 5e-4 modeli bozuyor, 1e-4 iyileştiriyor.

| Yöntem | BPW | Eğitimsiz | QAT (adım) |
|---|---:|---:|---:|
| int8 | 8,21 | 1,269 | – |
| int4 | 4,21 | 1,428 | 1,311 (2 000, lr 1e-4) |
| bin | 1,21 | 7,281 | 1,933 (3 000, lr 1,5e-3) |
| lb25p2 | 1,63 | 12,58 | 2,743 (300, lr 1e-3) |
| lb25p1 | 0,82 | 17,86 | 2,486 (3 000, lr 1,5e-3) |

LittleBit (tek yol, 0,82 BPW) aynı eğitim bütçesinde binary'nin 0,55 nat gerisinde kalıyor. Katmanların
genişliği 64 ve §3'teki formüle göre scale'ler bu genişlikte bitlerin önemli bir kısmını tutuyor.
Binary 3 000 adımda +0,66 nat kaybediyor. TinyStories, isimlerden çok daha zor bir görev ve 260 binlik
model kapasitesinin sınırında. Bu yüzden binary'nin bedeli burada ağır. Kayıp, son adımlarda hâlâ
hızla düşüyordu (2,10 → 1,93); daha uzun bir QAT farkı kapatacaktır.

## 3. LittleBit küçük modellerde ne zaman işe yarar?

Bir LittleBit katmanı ($d_\text{out}\times d_\text{in}$, rank $r$, $k$ yol) şu kadar bit harcar:

$$\text{BPW} = k\,\frac{r(d_\text{out}+d_\text{in}) + 16\,(d_\text{out}+d_\text{in}+r)}{d_\text{out}\,d_\text{in}}$$

İkinci terim scale vektörlerinin bedelidir. Binary katmanın bedeli ise $1 + 16/d_\text{in}$'dir.

| Katman | Binary | LittleBit, yalnız scale'ler ($r=0$, 1 yol) | r = %25, 1 yol | r = %25, 2 yol |
|---|---:|---:|---:|---:|
| 32 × 32 (names modelleri) | 1,50 | 1,00 | 1,63 | 3,25 |
| 64 × 64 (stories260K) | 1,25 | 0,50 | 1,06 | 2,13 |
| 172 × 64 (stories260K MLP) | 1,25 | 0,34 | 0,71 | 1,42 |
| 4096 × 4096 (Llama 2 7B) | 1,00 | 0,008 | 0,51 | 1,02 |

32 genişliğindeki bir katmanda yalnızca scale'ler weight başına 1 bit tutar; bu, binary bir
katmanın toplam maliyetinin üçte ikisidir. LittleBit'in 0,1 BPW sonucu, scale payının ihmal
edilebildiği 4096 genişliğindeki katmanlara aittir. Makaledeki 4096², r = 86, iki yollu örnek
0,0998 BPW eder.

Ölçümler formülle örtüşüyor:
- **Dar katmanlarda binary önde:** C = 32 genişliğindeki transformer'larda LittleBit aynı bit
  bütçesinde binary'nin gerisinde. Binary aynı bitle bütün rank'ı kullanıyor; LittleBit'in bitleri
  scale'lere gidiyor.
- **Geniş girişli katmanlarda LittleBit önde:** makemore MLP'nin ilk katmanının girişi 256, RNN'ninki
  ~200. Bu katmanlarda scale payı küçülüyor ve LittleBit binary'yi geçiyor.

## 4. Karar: donanıma 1-bit weight

Bu ölçekte üç yöntemin yeri şöyle:

- **INT4:** kalite olarak float'la aynı, ama donanım açısından INT8'e göre tek kazancı belleği yarıya
  indirmesi. Bizim modellerimiz BRAM'e zaten INT8'le sığıyor.
- **Ternary:** binary'den fazla bit harcıyor, binary'den iyi de değil.
- **Binary (Larq tarzı):**
  - weight'i 8 kat küçültür;
  - matris-vektör çarpımını çarpıcısız toplama-çıkarmaya çevirir;
  - names modellerinde +0,03…0,17 nat kaybettirir (transformer'larda +0,03…0,09).
- **LittleBit:** geniş girişli katmanlarda (MLP, RNN) aynı ya da daha az bitle binary'den iyi.
  Faktörleri binary matrisler olduğu için aynı birimi kullanır.

Engine'e bu yüzden **1-bit weight'li bir matris-vektör birimi** eklendi; binary ve LittleBit
modelleri bu birimde çalışır.

**GEMVB** (`rtl/zoo/zoo_core.sv`):

- `wbin.mem`'deki 64 bitlik bir kelime, bir girişin 64 çıkış için işaretlerini taşır (1 = +1, 0 = −1).
- 64 toplama/çıkarma hattı vardır, DSP kullanılmaz.
- Çıkış başına scale $a_o$, yeniden kuantizasyonun $m_0\,2^{-sh}$ çarpanına katlanır.
- Akümülatör ve shadow bankası ayrıdır ve 20 bittir: $|acc| \le 128\,n_\text{in} < 2^{19}$. Issue mantığı
  ve seri requantizer INT8 GEMV ile ortaktır. Banka yalnızca modelde GEMVB varsa sentezlenir.
- Çıkış sayısı 64'ün altındaysa boşaltma gerçek çıkış sayısında durur.

**SCALE:** Derleyici, norm katmanının gain'lerini $\gamma$ normalde sonraki weight'lere katlar:
$W\,\text{diag}(\gamma)$. Bu, binary weight'i bozar. `ir.py:factor_binary` katlanmış matrisi
$\text{diag}(a)\,S\,\text{diag}(g)$ olarak ayırır; burada $S\in\{\pm 1\}$.

- $\text{diag}(a)$ yeniden kuantizasyona gider.
- $g$, aynı girişi okuyan bütün katmanlar (q, k, v) için tek bir SCALE op'u olur:
  $y_i = \text{requant}(x_i\,g_i)$, $g$ INT16.

Bu, LittleBit'in $g$ vektörüyle aynı yapıdır. **LittleBit için yeni donanım gerekmez.**
`ir.py:factor_littlebit`, QAT'ın kaydettiği faktörlerden (`<yöntem>.factors.pt`) her katmanı üç op'a ayırır:

1. SCALE(g·γ)
2. GEMVB(sign(V)ᵀ), ℓ requant'ta
3. GEMVB(sign(U)), h requant'ta

Norm gain'leri γ ve $1/\sqrt{d_h}$ gibi katlanan çarpanlar, katlanmış matristen geri çıkarılır. GPT-2'nin
birleşik c_attn'ı q, k ve v'ye bölünür; ilk iki op paylaşılır. Faktörlere ayrılmış program, yoğun weight'li
modelle aynı logit'leri verir (fark ≤ 2·10⁻⁶).

İki yollu LittleBit, iki yolun çıkışlarını bir activation'dan önce toplamayı gerektirir. Bu yüzden şimdilik
yalnızca tek yollu LittleBit karta gider. Residual'a eklenen katmanlarda (wo, w2) iki yol da çalışabilir,
çünkü iki yol da residual'a ayrı ayrı eklenir.

**Kartta ölçülenler.** Değerler bütün kart tasarımı içindir: engine, sampler, VGA, 7-segment.

- Test kaybı, kartın bit-exact olarak çalıştırdığı integer modelden gelir (`board.py eval`).
- Weight belleği; weight ROM'larını ve çıkış başına parametreleri (bias, $m_0$, sh: 64 bit) içerir.
- Kaynak sayıları Yosys'tendir, cycle sayıları kart simülasyonundan.

| Model | Weight'ler | Test kaybı | Weight belleği | cycle/token | token/s (50 MHz) | LUT | FF | BRAM36 | DSP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | INT8 | 2,002 | 477 Kbit | 8 713 | 5 739 | 7 112 | 3 082 | 19,5 | 58 |
| gpt2-bin | binary (Larq) | 2,028 | 166 Kbit | 4 983 | 10 034 | 11 153 | 5 778 | 10 | 60 |
| mlp | INT8 | 1,987 | 419 Kbit | 6 722 | 7 438 | 4 949 | 2 167 | 14 | 22 |
| mlp-lb25p1 | LittleBit, 0,59 BPW | 2,142 | 86 Kbit | 1 874 | 26 681 | 7 940 | 4 767 | 5 | 25 |
| stories260k | INT8 | 1,291 | 2 322 Kbit | 50 397 | 992 | 9 002 | 3 312 | 97,5 | 66 |
| stories260k-bin | binary (Larq) | 1,927 | 747 Kbit | 27 992 | 1 786 | 12 118 | 5 920 | 56 | 68 |

Bütün düşük bitli modeller kart tasarımında Python referansıyla token token, hash ve ekran olarak aynı;
kart üstü kontrol LED'i OK. stories260k-bin'de belleğin üçte biri INT8 kalan output head'e
(512 × 64), üçte biri de çıkış başına parametrelere gidiyor. Binary katmanların kendisi
260 Kbit tutuyor.

**Karar.** Hız ve bellek için binary/LittleBit; kalite için INT8.

- **names modelleri:** binary (GPT-2) ve LittleBit (MLP) kartta 1,7–3,6 kat hız ve 3–5 kat bellek
  kazandırıyor, kalite bedeli 0,03–0,16 nat.
- **TinyStories:** INT8 kalıyor; binary'nin bedeli şimdilik çok yüksek.
- **Donanım:** iki yolu da destekliyor. Bir bitstream'deki model, hangi formatta dışa aktarıldıysa
  onu çalıştırıyor.

## 5. Sunumdaki üç araştırma sorusunun cevabı

**1. Küçük modellerde LittleBit'in anlamlı BPW alt sınırı nedir?**

- **Scale'lerin tabanı:** 16 bitlik scale'lerle bir LittleBit yolunun tabanı $16(d_\text{out}+d_\text{in}+r)/(d_\text{out}d_\text{in})$'dir.
  Bu taban 32 genişliğindeki katmanda 1 BPW, 64'te 0,5 BPW eder (§3).
- **Yöntemlerin sırası:** Ölçümlerde LittleBit, girişi dar transformer'larda (C = 32) aynı bitte binary'nin
  gerisinde kalıyor. Girişi geniş katmanlarda (makemore MLP'de 256, RNN'de ~200) ise binary'yi geçiyor.
- **Ölçülen alt sınır:** En düşük anlamlı nokta MLP'de 0,59 BPW, bedeli +0,16 nat.
- **0,1 BPW:** Bu bütçe, scale payının ihmal edilebildiği LLM katmanlarına (d ≈ 4096) özgü.

**2. Binary matris-vektör birimi kaç LUT tüketir, token başına kaç cycle sürer?**

- **Kaynak:** GEMVB'nin 64 toplama/çıkarma hattı ve 20 bitlik akümülatör/shadow bankası, SCALE dahil,
  Yosys'e göre yaklaşık **+4 000 LUT ve +2 700 FF** tutar; **DSP kullanmaz**.
- **Karşılaştırma:** 8 DSP'li INT8 dizisi cycle başına 8 MAC yapar; binary birim cycle başına 64 işaretli
  toplama yapar.
- **Token başına cycle:** aşağıdaki tabloda. GPT-2'de 8 713 → 4 983, MLP'de (LittleBit) 6 722 → 1 874,
  stories260K'da 50 397 → 27 992.

**3. FP16 scale'ler integer'a indirilip bit-exact bir donanım kurulabilir mi?**

Evet. Scale'ler şöyle taşınır:

- **Çıkış başına scale'ler** (binary'de $a$, LittleBit'te $h$ ve $\ell$) yeniden kuantizasyonun 16 bitlik
  $m_0$ çarpanına katlanır.
- **Giriş başına gain'ler** ($g$ ve katlanan norm gain'leri) INT16 bir tabloyla SCALE op'una gider.

Bütün akış, kart tasarımının tamamında Python'daki integer referansla token token ve ekran karakteri
karakterine aynı sonucu verir. Bu, Larq tarzı binary GPT-2, binary stories260K ve LittleBit MLP için
doğrulandı.

## 6. Çalıştırma

```bash
cd sw/zoo
python3 lowbit.py gpt2 bin                 # tek koşu -> model/zoo/lowbit/gpt2/bin.{json,pt}
python3 lowbit.py sweep gpt2 llama         # bütün yöntemler
python3 lowbit.py table                    # özet
python3 board.py export gpt2-bin           # QAT'lı modeli karta: GEMVB + SCALE -> mem/nexys/gpt2-bin/
python3 board_sim.py gpt2-bin --press CUD --sw1 1    # bütün kart tasarımı, referansla bit-exact
```
