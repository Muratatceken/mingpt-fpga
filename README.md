# minGPT → FPGA: Tamsayı Transformer Çıkarım Hızlandırıcısı

Bitirme projesi. Karpathy'nin [minGPT](https://github.com/karpathy/minGPT) modeli eğitilir, yalnızca
tamsayı aritmetiğine (INT8/INT16) dönüştürülür ve SystemVerilog ile yazılmış bir FPGA hızlandırıcısında
çalıştırılır. Donanım, Python'daki tamsayı referans modeliyle **bit düzeyinde aynı** sonucu üretir.

**Mevcut durum**

| | |
|---|---|
| Model | minGPT `gpt-nano` (3 katman, 3 head, $C=48$), minGPT demo'sundaki **sıralama** görevi |
| Doğruluk | float %100 · tamsayı %100 · RTL %100 (729 girdinin tamamı) |
| RTL doğrulaması | 729 dizi × 11 decode adımı = 8019 adımın tamamı **bit-exact** |
| Performans | 14 544 çevrim/token → 100 MHz'de 145 µs/token (GEMV verimi %98) |
| Bellek | 83 KB INT8 ağırlık (BRAM), 3 KB KV önbelleği |
| Hedef kart | **Nexys A7-100T** (Artix-7 xc7a100t): model kartta tek başına çalışır, sonuç VGA monitörde ([docs/05_nexys_a7.md](docs/05_nexys_a7.md)) |

**Nexys A7-100T: model kartın üzerinde, bilgisayarsız.**

- Weight'ler bitstream'le BRAM'e yüklenir.
- Üretim döngüsü, sampling (sıcaklık + min-p, xorshift32) ve detokenization FPGA'de çalışır.
- Hikâye ya da isimler VGA monitöre yazılır; token/s 7-segment göstergede okunur.
- Mac'te bütün kart tasarımı Verilator'da simüle edildi. 18 modelin hepsi (9 names modeli,
  stories260K, TinyStories'te eğittiğimiz 5 aile modeli, 3 düşük bitli model) Python referansıyla token
  token, ekran karakteri karakterine ve VGA pikseli pikseline aynı.
- **Açık kaynak yerleşim (Vivado'suz, Mac'te):** Yosys + nextpnr-himbaechel + Project X-Ray ile 16 model
  çipe yerleşti ve çalıştığı clock'ta zamanlamayı tutuyor: bigram, mlp, rnn 50 MHz (66–71 MHz Fmax),
  diğerleri 25 MHz (transformer'larda Fmax 26–31 MHz). ts_gpt2 ve ts_microgpt yerleşemedi (RAMB18
  bölünmesi). Kritik yol engine'in tek cycle'lık "tampondan oku → çarp → yaz" yolu.
- stories260K kartta 25 MHz'te 496 token/s üretiyor (256 token'lık hikâye yarım saniye); BRAM'in %72'si.

Ayrıntılar: [docs/05_nexys_a7.md](docs/05_nexys_a7.md). Kısa yol: `make nexys-sim MODEL=stories260k`,
bitstream için `make nexys-pnr MODEL=stories260k` (açık kaynak, Mac; önce bir kez `make xc7-toolchain`)
ya da `make nexys-bit MODEL=stories260k` (Vivado).

**Düşük bitli weight'ler (Larq, LittleBit).** Hocamızın önerdiği iki yöntem Mac'te QAT ile denendi
(`sw/zoo/lowbit.py`; FPGA yalnızca inference).

- **Engine:** 1-bit weight'li bir matris-vektör birimi (GEMVB, 64 toplama/çıkarma hattı, DSP yok) ve
  SCALE op'u eklendi. Larq tarzı binary modeller ve tek yollu LittleBit aynı birimde çalışır.
- **Kartta bit-exact çalışan düşük bitli modeller:**
  - binary GPT-2: 1,75 kat hızlı, +0,03 nat;
  - LittleBit makemore MLP: 0,59 BPW, 3,6 kat hızlı, 4,9 kat küçük, +0,16 nat;
  - binary stories260K: 1,8 kat hızlı, BRAM %72 → %41, +0,64 nat.
- **Bulgu:** LittleBit'in scale vektörleri dar katmanlarda (d ≤ 64) weight başına ~0,5–1 bit tutuyor;
  bu yüzden LittleBit ancak geniş katmanlarda binary'yi geçiyor.

Ayrıntılar: [docs/06_dusuk_bit.md](docs/06_dusuk_bit.md).

**Karpathy model ailesi (Kısım IV):** dokuz mimari (Bigram, MLP, RNN, GRU, BoW, GPT-2, microgpt,
Llama 2, nanochat) `names.txt` üzerinde ~50K parametreyle eğitildi, INT8'e çevrildi ve **tek bir
mikrokodlu RTL'de** (`rtl/zoo/zoo_core.sv`) çalıştırıldı. Modele özgü olan yalnızca bellek içerikleridir.
On modelin (dokuzu + Karpathy'nin hazır eğitilmiş **stories260K** TinyStories modeli) hepsi Verilator'da
tamsayı referansla **bit-exact**; RTL, referansla token token aynı 200 tokenlık bir hikâye üretir.

| Model | Float kayıp | INT8 kayıp | Çevrim/token |
|---|---|---|---|
| Bigram | 2,451 | 2,451 | 36 |
| MLP | 1,986 | 1,987 | 6 632 |
| RNN | 1,964 | 1,967 | 6 374 |
| GRU | **1,956** | **1,956** | 6 515 |
| BoW | 2,045 | 2,045 | 6 277 |
| GPT-2 | 2,000 | 2,002 | 8 620 |
| microgpt | 1,971 | 1,974 | 8 657 |
| Llama 2 | 2,003 | 2,004 | 9 396 |
| nanochat | 1,969 | 1,975 | 10 711 |
| stories260K | 1,400 | 1,422 | 38 652 (t=0) … 79 532 (t=511) |

## Dokümanlar

**Teknik rapor (PDF):** [report/rapor.pdf](report/rapor.pdf): minGPT'nin matematiksel ve görsel anlatımı (eğitilmiş modelin dikkat haritaları dahil), Karpathy'nin bigram'dan nanochat'e dil modeli ailesi (RMSNorm, RoPE, SwiGLU, GQA, QK-norm; ispatlı tamsayı sınırları), kuantizasyon teorisi, donanım mimarisi, performans ve doğrulama; Kısım IV'te mikrokodlu motor, nicemleme incelemesi ve ailenin donanım karşılaştırması. `make report` ile yeniden üretilir (XeLaTeX).

1. [docs/01_teori.md](docs/01_teori.md): Matematiksel model, KV önbelleği, kuantizasyon teorisi,
   tamsayı LayerNorm/Softmax/GELU türetmeleri ve ispatlar (Samuelson sınırı, dışbükeylik, taşma sınırları).
2. [docs/02_mimari.md](docs/02_mimari.md): Donanım mimarisi, GEMV dataflow'u, bellek haritası, çevrim analizi.
3. [docs/03_dogrulama.md](docs/03_dogrulama.md): Doğrulama metodolojisi ve sonuçları.
4. [docs/04_kart.md](docs/04_kart.md): Eski hedef PYNQ-Z2, yerleşik öz-test.
5. [docs/05_nexys_a7.md](docs/05_nexys_a7.md): **Nexys A7-100T**: model kartta tek başına çalışır (VGA, sampling, kart üstü kontrol), Mac'te tam kart simülasyonu, Vivado/openFPGALoader ile programlama.
6. [docs/06_dusuk_bit.md](docs/06_dusuk_bit.md): **Düşük bitli weight'ler**: Larq tarzı binary/ternary ve LittleBit QAT sonuçları, LittleBit'in küçük modellerde BPW sınırı, engine'deki 1-bit GEMV (GEMVB) ve SCALE, kartta ölçülen hız/bellek/kalite.
7. [docs/07_yol_haritasi.pdf](docs/07_yol_haritasi.pdf): **Q1 dergi yol haritası**: hedef dergiler, literatür boşlukları, üç araştırma hattı, ölçüm ve zaman planı. Literatür taraması: [paper/related_work_2026.md](paper/related_work_2026.md).

**Tarihsel gelişim raporu (PDF):** [report/gelisim/gelisim_raporu.pdf](report/gelisim/gelisim_raporu.pdf): 12–27 Ekim arasında projenin evre evre gelişimi (kararlar, sayılar, hatalar, dersler). `make gelisim` ile yeniden üretilir.

**Sunumlar** (yerel, `sunum/`):
- [sunum/sunum.pdf](sunum/sunum.pdf): Bölüm 1, minGPT adım adım, FPGA potansiyeli ve LittleBit (55 slayt).
- [sunum/uygulama/sunum.pdf](sunum/uygulama/sunum.pdf): Bölüm 2, implementasyon (50 slayt); yanında
  [terimler.pdf](sunum/uygulama/terimler.pdf) (teknik terimler sözlüğü),
  [konusma_notlari.pdf](sunum/uygulama/konusma_notlari.pdf) (zaman planı, olası sorular) ve
  [konusma_metni.pdf](sunum/uygulama/konusma_metni.pdf). `make sunum-yerel` ile yeniden üretilir.

## Dizin yapısı

```
ref/minGPT/      minGPT (git submodule, değiştirilmedi)
model/           eğitilmiş ağırlıklar (gpt_sort.pt)
sw/              Python akışı
  train.py         1) gpt-nano'yu sıralama görevinde eğit          -> model/gpt_sort.pt
  quantize.py      2) katlama + kalibrasyon + INT8/INT16 kuantizasyon -> build/qparams.pkl
  intmodel.py         bit-exact tamsayı altın model (donanımın spesifikasyonu)
  evaluate.py      3) float vs tamsayı karşılaştırması
  export.py        4) mem/*.mem, mem/tv.txt, rtl/gpt_pkg.sv üretimi
rtl/             SystemVerilog (sentezlenebilir)
  gpt_pkg.sv       (üretilir) boyutlar, sabitler, adres haritası
  gpt_top.sv       üst modül + kontrol FSM'i + tamponlar
  gpt_selftest.sv  çip üzeri öz-test: 729 sıralama problemi, sonucu kontrol eder
  gemv.sv          P-şeritli matris-vektör motoru
  requant.sv       kanal başına yeniden kuantizasyon
  layernorm.sv     tamsayı LayerNorm
  attention.sv     KV önbellekli nedensel çok-başlı dikkat
  div_seq.sv, isqrt_seq.sv, rom.sv
board/nexys_a7/   Nexys A7-100T: lm_gen (üretim, sampling, detokenizer, terminal), vga_text, seg7, üst modül, XDC
board/pynq_z2/    eski hedef PYNQ-Z2: üst modül ve pin kısıtları
tb/              testbench'ler (xsim ve Verilator uyumlu)
mem/             (üretilir) ROM içerikleri ve altın test vektörleri
scripts/         Vivado Tcl scriptleri (simülasyon, sentez), dosya listesi
                 zoo_sim.sh / zoo_synth.sh: model ailesi için Verilator / Yosys
                 nexys_*.sh: kart simülasyonu, netlist, Yosys; nexys_pnr.sh + xc7_toolchain.sh: açık kaynak yerleşim
sunum/           sunumlar: kaynak slaytlar (sunum/uygulama/kaynak), yerel HTML/PDF, sözlük ve notlar (yerel_sunum.py)
report/gelisim/  tarihsel gelişim raporu (XeLaTeX; tablolar veri.py ile JSON'dan)
data/            names.txt; stories260K/ (llama2.c checkpoint, tok512, 400 doğrulama hikâyesi)
sw/zoo/          model ailesi akışı
  models.py        dokuz mimari (kaynak depoları izleyerek)
  train.py         1) names.txt üzerinde eğitim                 -> model/zoo/<ad>.pt, metrics.json
  ir.py            PyTorch -> token başına komut programı (IR) + float yorumlayıcı
  check_ir.py         IR ile PyTorch'un eşleştiğinin kontrolü
  quant.py         INT8 nicemleme + bit-exact tamsayı yorumlayıcı (IntRunner = donanım spesifikasyonu)
  evaluate.py      2) float vs INT8                              -> model/zoo/int_metrics.json
  stories.py          stories260K: yükleyici, tokenizer, INT8, hikâye üretimi
  ablate.py           nicemleme hatasının kaynakları, kalibrasyon yöntemleri
  export_hw.py     3) mikrokod + bellekler + altın vektörler    -> mem/zoo/<ad>/
  hw_run.py        4) RTL simülasyonu (+ Yosys)                 -> model/zoo/hw_metrics.json
  report_zoo.py       rapor tabloları ve şekil verileri
rtl/zoo/         mikrokodlu motor: zoo_core.sv, zdiv.sv (sınırlı bölümlü bölücü), zrom.sv
```

## Hızlı başlangıç

**1) Python (macOS/Linux/Windows):** `torch` ve `numpy` gerekir.
```bash
git submodule update --init
make model        # eğit + kuantize et + dışa aktar  (~2 dk)
make eval         # float vs tamsayı karşılaştırması (~4 dk)
```
`mem/` ve `rtl/gpt_pkg.sv` depoda hazır bulunur. Yalnızca RTL ile çalışacaksanız bu adım atlanabilir.

**2) Yerel RTL simülasyonu (Verilator ≥ 5.0):**
```bash
make test-arith   # bölücü / karekök / requant birim testleri
make sim          # 729 dizi, bit-exact kontrol + çevrim profili
make selftest     # çip üzeri öz-test (normal + hata enjeksiyonu)
```

**3) Vivado (Windows veya Linux PC):**
```bash
vivado -source scripts/vivado_sim.tcl                              # GUI: proje + xsim + dalga formu
vivado -mode batch -source scripts/vivado_sim.tcl -tclargs 20      # batch: 20 dizi
vivado -mode batch -source scripts/vivado_synth.tcl -tclargs xc7z020clg400-1 10
vivado -mode batch -source scripts/vivado_build_pynq_z2.tcl         # PYNQ-Z2 bitstream
vivado -mode batch -source scripts/vivado_program.tcl              # JTAG ile kartı programla
```
Scriptler `rtl/gpt_pkg.sv`'nin `MEM_PATH`'i mutlak yol olan bir kopyasını `build/gen/` altında üretip
onu derler. Bu yüzden depo başka bir bilgisayara klonlandığında ek bir ayar gerekmez.

> **Not:** Vivado macOS'ta çalışmaz (yalnızca x86-64 Windows/Linux). Apple Silicon Mac'te bir x86
> sanal makinesi çok yavaş olur. Geliştirme Mac'te Verilator ile yapılıp Vivado adımı laboratuvar
> veya Linux/Windows bir bilgisayarda çalıştırılabilir. RTL ve testbench'ler iki simülatörde de
> çalışacak şekilde yazılmıştır.

**4) Model ailesi (Verilator + Yosys):**
```bash
make zoo-train    # dokuz modeli eğit (~40 dk, CPU)
make zoo-quant    # INT8 + stories260K + dışa aktarma
make zoo-sim      # on modelin bit-exact RTL simülasyonu (+ hikâye üretimi)
make zoo-synth    # simülasyon + Yosys sentezi (genel ve özelleşmiş motor)
```

## Yol haritası

- [x] Eğitim, katlama, kalibrasyon, INT8/INT16 kuantizasyon
- [x] Bit-exact tamsayı altın model, %100 doğruluk
- [x] RTL: GEMV, requant, LayerNorm, attention (KV önbelleği), top FSM
- [x] Birim + sistem testbench'leri, negatif test, çevrim profili
- [x] Hedef kart: PYNQ-Z2, çip üzeri öz-test (simülasyonda doğrulandı)
- [ ] Vivado xsim'de çalıştırma ve dalga formu ekran görüntüleri (rapor için)
- [ ] PYNQ-Z2 bitstream: kaynak kullanımı, Fmax, güç raporu, kartta yeşil LED
- [x] Karpathy ailesi: 9 mimari + stories260K, tek mikrokodlu RTL, hepsi bit-exact; Yosys kaynakları
- [x] Tek geçişli LN, 8 şeritli paralel dikkat, sınırlı bölümlü bölücü (zoo motorunda)
- [x] Tasarım uzayı deneyleri: P ∈ {8, 16, 32}, 6 model, 18 noktanın hepsi bit-exact
- [x] Nexys A7-100T: model kartta bağımsız; 18 model tam kart simülasyonunda bit-exact
- [x] Düşük bitli weight'ler: binary ve LittleBit, 1-bit GEMV birimi
- [x] Açık kaynak yerleşim: 16 model, Fmax ve bitstream
- [ ] Engine'e pipeline: 50 MHz (transformer'lar bugün 25 MHz)
- [ ] Yerleştirilen netlist'in 18 modelde simülasyonu; ts_gpt2 / ts_microgpt bellek düzeni
- [ ] minGPT gpt-nano sıralama demosu kart tasarımında (19. model)
- [ ] Q1 makale hatları: [docs/07_yol_haritasi.pdf](docs/07_yol_haritasi.pdf)
