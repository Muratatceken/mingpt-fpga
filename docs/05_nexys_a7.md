# Nexys A7-100T: model kartın üzerinde, bilgisayarsız çalışır

Bu bölüm, Karpathy model ailesinin Digilent **Nexys A7-100T** kartında (AMD/Xilinx Artix-7
**xc7a100tcsg324-1**) tek başına çalışan hâlini anlatır. Kart programlandıktan sonra hiçbir bilgisayar
gerekmez:

- **Model:** weight'ler, tablolar ve mikrokod bitstream'le birlikte BRAM'e yüklenir.
- **Üretim:** üretim döngüsü, sampling ve detokenization FPGA'deki mantıkta çalışır.
- **Çıktı:** sonuç VGA monitörde metin olarak görünür; hız 7-segment göstergede okunur.

## 1. Kart ve kaynaklar

| | Nexys A7-100T (xc7a100t) | Eski hedef PYNQ-Z2 (xc7z020) |
|---|---|---|
| LUT | 63 400 | 53 200 |
| Flip-flop | 126 800 | 106 400 |
| BRAM (36 Kb) | 135 (4 860 Kb ≈ 607 KB) | 140 (≈ 630 KB) |
| DSP48E1 | 240 | 220 |
| İşlemci | yok (yalnız FPGA) | çift çekirdek ARM |
| Kullandığımız G/Ç | 100 MHz osilatör, VGA (12 bit renk), 8 haneli 7-segment, 16 anahtar, 4 düğme + reset, 16 LED + 2 RGB LED, QSPI flash | |

Pin atamaları Digilent'in resmî `Nexys-A7-100T-Master.xdc` dosyasından alınmıştır
(`board/nexys_a7/nexys_a7.xdc`).

## 2. Mimari

```
   CLK100MHZ ─ MMCM ─ clk (50 MHz; 100 / 25 seçilebilir)
                                                      ┌──────────── lm_gen.sv ────────────┐
  BRAM: mikrokod, INT8 weight'ler,  ──►  zoo_core   ──► logits ──► sampler ──► next token ─┐
  requant parametreleri, tablolar       (rtl/zoo)    (INT16)   (greedy / sıcaklık, min-p,   │
        ▲                                   ▲                    xorshift32)               │
        └──────── bitstream ile yüklenir    └─────────── tok, pos ◄────────────────────────┘
                                                                 │
                                         detokenizer (token → karakterler) ─► terminal ─► text RAM ─► vga_text ─► VGA
                                                                                         (80 × 30)
```

Modüller (`board/nexys_a7/`):

| Dosya | Görev |
|---|---|
| `nexys_a7_top.sv` | MMCM, reset, düğme debounce, anahtar senkronizasyonu, LED ve RGB LED |
| `lm_gen.sv` | Üretim döngüsü, sampler, detokenizer, 80 sütunluk terminal (kelime kaydırma, scroll), durum satırı, kart üstü kontrol |
| `vga_text.sv` | 640 × 480 @ 60 Hz metin ekranı: 80 × 30 karakter, 8 × 16 font, 3 kademeli pipeline |
| `seg7.sv`, `bin2bcd.sv` | 7-segment sürücü, sıralı ikili→BCD dönüştürücü |

**Sampling (tamamen integer, bit-exact).** Engine'in INT16 logit'leri $l_k$ üç kez taranır:

1. En büyük logit $m$ bulunur.
2. Her token için $e_k = 2^{-(m-l_k)c}$ hesaplanır ve toplanır. Burada
   $c = s_\text{logit}\log_2 e/\tau$, sabit noktada $(\text{CEXP}, \text{FEXP})$ olarak tutulur.
   Üs alma, attention biriminin kullandığı $2^{-k/256}$ tablosuyla yapılır. Min-p kesmesi
   $p_k < p_\text{max}\,2^{-\text{NCUT}}$ olan token'ları atar.
3. $\text{thr} = (r \cdot \sum e)\gg 32$ hesaplanır; $r$ xorshift32 üretecinden gelir. Birikimli toplamı
   thr'yi ilk aşan token seçilir.

Toplam ek maliyet yaklaşık $3V$ cycle'dır: stories modellerinde ~1 500, names modellerinde ~80 cycle.
Aynı algoritma `sw/zoo/board.py:pick` içinde birebir yazılıdır.

**Kart üstü kontrol.** Üretilen token'ların hash'i ($h \leftarrow 31h + \text{tok}$), `board.py`'nin
dışa aktarırken hesapladığı değerle karşılaştırılır. Kontrol iki durumda yapılır:

- her greedy koşuda;
- reset ya da reseed sonrasındaki ilk sampling koşusunda, anahtarlar aşağıdayken.

LED16 yeşil yanarsa kart, Python'daki integer modelle token token aynı sonucu üretmiş demektir.

## 3. Kartın kullanımı

| Giriş | İşlev |
|---|---|
| güç / CPU_RESET | İlk koşu kendiliğinden başlar: hikâye ya da 27 isim |
| BTNC | Yeni koşu; rastgele durum devam eder, her basışta yeni hikâye |
| BTNU | Üreteci SW[15:8]'den yeniden tohumla ve koş |
| BTND | Metin alanını temizle |
| SW0 | yukarı: greedy · aşağı: sampling |
| SW[3:1] | sıcaklık τ: 000 = 0,8, sonra 1,0 · 0,9 · 0,7 · 0,6 · 0,5 · 1,2 · 1,5 |
| SW[5:4] | min-p: 00 = 1/32, 01 = kapalı, 10 = 1/128, 11 = 1/16 |
| SW6 | daktilo modu, ~25 token/s (üretimi gözle izlemek için) |
| SW7 | 7-segment: aşağı = token/s · yukarı = token başına cycle |
| SW[15:8] | seed |

| Çıkış | Anlamı |
|---|---|
| VGA | 1. satır: model başlığı · 28 satır: üretilen metin · son satır: token sayısı, token/s, cycle/token, mod, kontrol |
| LED0 | heartbeat |
| LED1 | üretim sürüyor |
| LED2 / LED3 | kontrol OK / FAIL |
| LED[15:4] | koşudaki token sayısı |
| LED16 (RGB) | mavi = üretiyor · yeşil = kontrol OK · kırmızı = FAIL |

## 4. Mac'te test

Vivado macOS'ta çalışmaz. Bu yüzden kartın **bütün üst modülü** Verilator'da simüle edilir:

- MMCM devre dışıdır; testbench saati doğrudan verir.
- Testbench reset'i bırakır, otomatik koşuyu bekler ve sonra düğmelere basar.
- Her koşudan sonra ekran RAM'i okunur. İstenirse VGA pinlerinden tam bir kare yakalanır.

```bash
make nexys-sim MODEL=stories260k   # = board.py export + board_sim.py --press CUD --sw1 1 --frame
make nexys-all                     # 15 INT8 modelin hepsi
scripts/nexys_ci.sh gru gpt2-bin   # Python'suz kısa kontrol (CI'da da çalışır)
make nexys-synth MODEL=stories260k # Yosys kaynak tahmini
```

`board_sim.py` her koşuda şunları Python referansıyla (`board.generate`) karşılaştırır: üretilen her
token, koşu hash'i, kart üstü kontrolün sonucu ve 80 × 28'lik metin alanı (karakter karakter). Örnek
VGA karesi: `build/nexys/stories260k/frame.png`.

**Sonuçlar** (`model/zoo/nexys.json`). Her model için 4 koşu yapıldı: otomatik ilk koşu (sampling),
BTNC (devam), SW0 açıkken BTNU (greedy) ve BTND (temizle). 18 modelin **hepsi bit-exact**: token'lar,
hash'ler, kontrol LED'i ve ekran metni referansla aynı. Bu 18 modelin 15'i INT8; 3'ü düşük bitli QAT
modeli ([06_dusuk_bit.md](06_dusuk_bit.md)).

| Model | Veri | Weight'ler | cycle/token | token/s (50 MHz) |
|---|---|---|---:|---:|
| bigram | names | INT8 | 157 | 318 098 |
| mlp | names | INT8 | 6 722 | 7 438 |
| rnn | names | INT8 | 6 464 | 7 735 |
| gru | names | INT8 | 6 605 | 7 570 |
| bow | names | INT8 | 6 368 | 7 852 |
| gpt2 | names | INT8 | 8 712 | 5 739 |
| microgpt | names | INT8 | 8 746 | 5 717 |
| llama | names | INT8 | 9 490 | 5 268 |
| nanochat | names | INT8 | 10 813 | 4 624 |
| stories260k (Karpathy) | TinyStories | INT8 | 50 397 | 992 |
| ts_gpt2 | TinyStories | INT8 | 41 781 | 1 197 |
| ts_gru | TinyStories | INT8 | 30 407 | 1 644 |
| ts_llama | TinyStories | INT8 | 50 397 | 992 |
| ts_microgpt | TinyStories | INT8 | 40 570 | 1 232 |
| ts_nanochat | TinyStories | INT8 | 50 680 | 987 |
| gpt2-bin | names | binary | 4 983 | 10 034 |
| mlp-lb25p1 | names | LittleBit | 1 874 | 26 681 |
| stories260k-bin | TinyStories | binary | 27 992 | 1 786 |

cycle/token, otomatik ilk koşunun ortalamasıdır. Daktilo beklemesi dışındaki her çevrim sayılır.
Ekran yazımı normalde bir sonraki adımın hesabıyla paralel yürür ve maliyetsizdir. Yalnızca bigram,
durum satırının güncellenmesinden hızlı olduğu için ekranı bekler. TinyStories modellerinde context
$T = 256$ ile sınırlanmıştır, böylece KV cache BRAM'e sığar.

**Ek doğrulama.** Testbench, kartın pinlerinden okunabilen her şeyi denetler.

- **Giriş/çıkış zamanlaması:** girişler düşen kenarda sürülür, çıkışlar düşen kenarda örneklenir.
- **Pin imzası:** her çevrimde bütün pinlerin imzası tutulur.
- **Görüntü:** 7-segment okunur; VGA karesi pinlerden alınır ve senkron sinyalleri kontrol edilir.
- **Ekran:** her boşta kalma anında durum satırı, başlık, LED token sayacı ve hash yazmacı referansla
  karşılaştırılır.

| Test | Komut | Kapsam | Sonuç |
|---|---|---|---|
| Rastgele (fuzz) | `make nexys-fuzz` | bigram, gru, nanochat, llama: 40'ar durum; stories260k: 6 | 655 koşu, ~121 000 token, sampling ayarlarının 30–32/32'si; hepsi aynı |
| Köşe durumları | `board_fuzz.py --directed` | 15 durum (aşağıda) | hepsi geçti |
| 4-state, X kontrolü | `make nexys-icarus` | gru, koşu ortasında reset dahil 2,7 M çevrim | pinlerde ve metin RAM yazma portunda 0 X |
| Sentez sonrası netlist | `make nexys-netlist` | gru: gerçek üst modülün Yosys netlist'i (MMCM, BUFG, BRAM) | her pinde RTL ile çevrimi çevrimine aynı; VGA karesi aynı |
| CI | `scripts/nexys_ci.sh` | gru, nanochat, gpt2-bin, mlp-lb25p1 | kart üstü kontrol OK |

Köşe durumları:
- koşu sırasında basılan BTNC/BTNU kuyruğa alınır; BTND yok sayılır;
- koşu sırasında CPU_RESET verilirse yarım kalan koşunun token'ları referansın önekidir ve ardından
  yeniden tohumlanmış bir koşu gelir;
- daktilo modu token'ları ve sayılan çevrimleri değiştirmez.

Bu testler iki gerçek RTL hatası buldu, ikisi de düzeltildi:
1. Kuyruktaki bir koşunun başladığı çevrime denk gelen BTNU basışı kayboluyordu.
2. Heartbeat sayacının reset'i yoktu; 4-state simülasyonda LED0 X kalıyordu.

**Kaynaklar** (Yosys `synth_xilinx`; zamanlama yok, Vivado'nun sayıları farklı olabilir). Her bitstream tek
bir model taşıdığı için engine yalnızca o modelin kullandığı birimlerle sentezlenir (`SPECIALIZE=1`):

| Model | LUT | FF | BRAM36 | DSP |
|---|---:|---:|---:|---:|
| bigram | 2 118 (%3) | 1 314 (%1) | 2 (%1) | 10 (%4) |
| mlp | 4 949 (%8) | 2 167 (%2) | 14 (%10) | 22 (%9) |
| gru | 5 021 (%8) | 2 199 (%2) | 16 (%12) | 25 (%10) |
| gpt2 | 7 112 (%11) | 3 082 (%2) | 19,5 (%14) | 58 (%24) |
| nanochat | 7 525 (%12) | 3 221 (%3) | 16 (%12) | 65 (%27) |
| stories260k | 9 002 (%14) | 3 312 (%3) | 97,5 (%72) | 66 (%28) |
| ts_gru | 5 911 (%9) | 2 227 (%2) | 69,5 (%51) | 25 (%10) |
| gpt2-bin | 11 153 (%18) | 5 778 (%5) | 10 (%7) | 60 (%25) |
| mlp-lb25p1 | 7 940 (%13) | 4 767 (%4) | 5 (%4) | 25 (%10) |
| stories260k-bin | 12 118 (%19) | 5 920 (%5) | 56 (%41) | 68 (%28) |

## 5. Bitstream ve kartı programlama

Bitstream, Vivado ile Linux ya da Windows bir bilgisayarda üretilir:

```bash
make nexys-bit MODEL=stories260k            # vivado -mode batch -source scripts/vivado_nexys_a7.tcl -tclargs stories260k 50
```

Betik 50 MHz'de zamanlama tutmazsa tasarımı kendiliğinden 25 MHz'de yeniden kurar; yani bıraktığı
bitstream her zaman zamanlamayı tutar. Çıktılar `build/nexys/<model>/vivado/` altındadır: `.bit`,
`.bin`, kaynak/zamanlama/güç raporları ve `result.json`.

Programlama micro-USB (PROG/UART) portundan yapılır:

```bash
vivado -mode batch -source scripts/vivado_program_nexys.tcl -tclargs stories260k          # FPGA'ye (güç kesilene kadar)
vivado -mode batch -source scripts/vivado_program_nexys.tcl -tclargs stories260k flash    # QSPI flash'a (her açılışta)
# Vivado'suz, Mac'ten:  brew install openfpgaloader
openFPGALoader -b nexys_a7_100 build/nexys/stories260k/vivado/nexys_stories260k.bit
openFPGALoader -b nexys_a7_100 -f build/nexys/stories260k/vivado/nexys_stories260k.bit   # flash
```

Flash'tan açılış için JP1 jumper'ı QSPI konumunda olmalıdır.

## 6. Bilinen sınırlar

- **Yosys'in DSP eşlemesinde bir hata var.** Yosys 0.69 `synth_xilinx`, requant çarpımını DSP48E1'e
  eşlediğinde (girişler registerden geliyorsa) çarpımın işaret uzatmasını düşürüyor. Bu yüzden negatif
  akümülatörler +max'a doyuyor. RTL doğru; `scripts/requant_netlist_check.sh` hatayı tek başına gösteriyor.
  Netlist simülasyonu bu yüzden çarpıcıları LUT'a eşler (`-nodsp`). Aynı sebeple ve BRAM modelleri boş
  olduğu için eski `scripts/zoo_netlist_sim.sh` bu Yosys sürümüyle geçmez. Vivado bu hatadan etkilenmez,
  ama kaynak tahminlerindeki DSP sayıları Yosys'in eşlemesidir.
- **Fmax ölçülmedi.** Vivado olmadan zamanlama analizi yapılamıyor. Engine'deki geniş çarpıcı
  zincirleri (NORM çıkışı, attention'daki üs alma) 50 MHz'i zorlayabilir; betik bu durumda 25 MHz'e
  iner. Gerçek Fmax ilk Vivado koşusunun `result.json` dosyasında görülecek.
- **Bir bitstream, bir model.** Modeller BRAM'de bitstream'le birlikte gelir. stories260k tek başına
  BRAM'in %72'sini kullanır (binary sürümü %41).
- **TinyStories modellerinde context 256 token'la sınırlı.** Model 512 token'la eğitildi, ama
  BRAM'e sığması için KV cache yarıya indirildi.
