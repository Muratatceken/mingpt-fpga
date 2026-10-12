# 3. Doğrulama Metodolojisi

## 3.1 Referans model hiyerarşisi

Donanımın doğruluğu tek bir büyük adımda değil, her biri bir öncekine karşı doğrulanan dört
seviyede gösterilir:

| Seviye | Dosya | Aritmetik | Bir öncekine karşı kontrol |
|---|---|---|---|
| 0. minGPT | `ref/minGPT/mingpt/model.py` | float32 | eğitim: %100 sıralama doğruluğu |
| 1. Katlanmış model | `sw/quantize.py::folded_forward` | float64 | logitler $10^{-4}$ içinde aynı (729/729) |
| 2. Tamsayı altın model | `sw/intmodel.py` | yalnızca tamsayı | görev doğruluğu %100, token uyumu %100, SQNR 41.5 dB |
| 3. RTL | `rtl/*.sv` | donanım | **bit-exact** (her adımda logitler + 48 artık değer + token) |

Kilit fikir şudur: **Seviye 2, donanımın bit düzeyindeki spesifikasyonudur.** Python tamsayıları sınırsız
hassasiyetli olduğu için RTL'deki her taşma veya yanlış bit genişliği bir uyumsuzluk olarak
yakalanır. Seviye 2 ile 3 arasında tolerans yoktur, karşılaştırma `==` ile yapılır.

## 3.2 Birim testleri (`tb/tb_arith.sv`)

| Modül | Test | Kontrol (referanssız, matematiksel) |
|---|---|---|
| `div_seq` | 2000 rastgele (48 bit) | $q\,d \le n < (q+1)\,d$ |
| `isqrt_seq` | 2000 rastgele ($<2^{62}$) | $r^2 \le n < (r+1)^2$ |
| `requant` | 2000 rastgele | davranışsal referansa eşitlik |

## 3.3 Sistem testi (`tb/tb_gpt_top.sv`)

* `mem/tv.txt`: altın modelin **729 girdinin tamamı** için ürettiği bütün decode adımları
  (adım başına token, $V$ logit ve 48 artık değer).
* **Kapalı döngü:** prompt verildikten sonra DUT'nin kendi ürettiği token geri beslenir, yani gerçek
  kullanım senaryosu test edilir.
* Her adımda token, logitler ve 48 artık değer (`dbg` portu üzerinden) karşılaştırılır. Dizi
  sonunda çıktının gerçekten sıralı olup olmadığı da kontrol edilir.
* Çevrim profili: FSM durumlarına göre token başına çevrim sayısı.

**Sonuç (Verilator 5.052):**
```
decode steps checked : 8019
bit-exact sequences  : 729 / 729
sorted correctly     : 729 / 729
mismatching values   : 0
avg cycles / token   : 14544  (145.4 us @ 100 MHz)
RESULT: PASS
```

## 3.4 Testin kendisinin testi (negatif test)

Testbench'in hataları gerçekten yakaladığını göstermek için altın dosyaya bilerek iki hata eklendi
(bir artık değer +1, bir token değiştirildi). Testbench ikisini de raporladı ve `FAIL` verdi.
Bu adım, "hep PASS veren" bir testbench olmadığını kanıtlar.

> Önemli bir SystemVerilog tuzağı: `initial` bloğu içindeki bir döngüde ilk değerle tanımlanan
> değişkenler (`int x = ...;`) varsayılan olarak **static**'tir. İlk değer yalnızca bir kez atanır
> (IEEE 1800, §6.21). Rastgele testlerde bu değişkenler `automatic` olarak tanımlanmalıdır, aksi hâlde
> 2000 test aslında aynı girdiyi 2000 kez dener.

## 3.5 Çalıştırma

```bash
make test-arith          # birim testleri
make sim                 # 729 dizi, bit-exact (≈40 s, Verilator)
make sim NSEQ=20         # hızlı
```

Vivado (Windows/Linux):
```bash
vivado -mode batch -source scripts/vivado_sim.tcl -tclargs 20      # xsim, 20 dizi
vivado -source scripts/vivado_sim.tcl                              # GUI + dalga formu
vivado -mode batch -source scripts/vivado_synth.tcl                # kaynak / zamanlama raporu
```

## 3.6 Hata ayıklama ipuçları

* `mem/trace.txt`: 0 numaralı dizi için her alt katmandan sonraki artık vektör
  (`t layer {0:attn,1:mlp} res[0..47]`). İlk uyumsuz alt katman bu dosya ile bulunur.
* Dalga formunda izlenecek sinyaller: `dut/st`, `dut/layer`, `dut/u_gemv/{issuing,acc,out_valid,out_idx,out_data}`,
  `dut/u_ln/{st,s,Q,rt,inv}`, `dut/u_att/{st,h,j,m,e_sum,R}`.
* Yeniden eğitim veya kuantizasyon değişikliğinden sonra `make model` (veya yalnızca `make export`)
  çalıştırılmalıdır. `rtl/gpt_pkg.sv` ve `mem/` otomatik olarak yeniden üretilir.
