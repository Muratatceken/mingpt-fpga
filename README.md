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

## Dokümanlar

**Teknik rapor (PDF):** [report/rapor.pdf](report/rapor.pdf), 38 sayfa: minGPT'nin matematiksel anlatımı, kuantizasyon teorisi ve ispatlar, donanım mimarisi, performans ve doğrulama. `make report` ile yeniden üretilir (XeLaTeX).

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

## Yol haritası

- [x] Eğitim, katlama, kalibrasyon, INT8/INT16 kuantizasyon
- [x] Bit-exact tamsayı altın model, %100 doğruluk
- [x] RTL: GEMV, requant, LayerNorm, attention (KV önbelleği), top FSM
- [x] Birim + sistem testbench'leri, negatif test, çevrim profili
- [x] Hedef kart: PYNQ-Z2, çip üzeri öz-test (simülasyonda doğrulandı)
- [ ] Vivado xsim'de çalıştırma ve dalga formu ekran görüntüleri (rapor için)
- [ ] PYNQ-Z2 bitstream: kaynak kullanımı, Fmax, güç raporu, kartta yeşil LED
- [ ] Tasarım uzayı deneyleri: P ∈ {4, 8, 16, 32}, tek geçişli LN, paralel dikkat
- [ ] PS + AXI-Lite + PYNQ (Jupyter) arayüzü, ARM CPU ile hız karşılaştırması
- [ ] (opsiyonel) Daha büyük görev: karakter seviyesi Shakespeare
