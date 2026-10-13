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
| Hedef kart | **PYNQ-Z2** (Zynq-7020): yerleşik öz-test bitstream'i hazır, LED ile sonuç |

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
4. [docs/04_kart.md](docs/04_kart.md): Kart seçimi (PYNQ-Z2), yerleşik öz-test, kartta çalıştırma adımları.

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
board/pynq_z2/    PYNQ-Z2 üst modülü (MMCM, buton, LED) ve pin kısıtları (XDC)
tb/              testbench'ler (xsim ve Verilator uyumlu)
mem/             (üretilir) ROM içerikleri ve altın test vektörleri
scripts/         Vivado Tcl scriptleri (simülasyon, sentez), dosya listesi
                 zoo_sim.sh / zoo_synth.sh: model ailesi için Verilator / Yosys
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
- [ ] Tasarım uzayı deneyleri: P ∈ {4, 8, 16, 32}
- [ ] PS + AXI-Lite + PYNQ (Jupyter) arayüzü, ARM CPU ile hız karşılaştırması
- [ ] (opsiyonel) Daha büyük görev: karakter seviyesi Shakespeare
