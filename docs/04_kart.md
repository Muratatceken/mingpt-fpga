# 4. Hedef Kart: PYNQ-Z2

## 4.1 Seçim gerekçesi

| Kart | FPGA | LUT | BRAM (36 Kb) | DSP | İşlemci | Değerlendirme |
|---|---|---|---|---|---|---|
| **PYNQ-Z2 (TUL)** | Zynq-7020 (xc7z020) | 53 200 | 140 (630 KB) | 220 | 2× Cortex-A9 + PYNQ (Python) | **seçildi** |
| Basys 3 (Digilent) | Artix-7 35T | 20 800 | 50 (225 KB) | 90 | yok | sığar ama büyümeye yer yok, PC arayüzü yalnızca UART |
| Zybo Z7-20 (Digilent) | Zynq-7020 | 53 200 | 140 | 220 | 2× Cortex-A9 | aynı çip; resmî PYNQ desteği PYNQ-Z2'de |
| Arty A7-100T (Digilent) | Artix-7 100T | 63 400 | 135 | 240 | yok (MicroBlaze) | iyi ama işlemci tarafı daha zahmetli |
| Kria KV260 (AMD) | Zynq UltraScale+ K26 | 117 120 | 144 + 64 URAM | 1 248 | 4× Cortex-A53 | güçlü; bu boyut için gereğinden büyük, akışı daha karmaşık |

PYNQ-Z2'yi seçmemin nedenleri:

1. **Kaynak payı:** Tasarımın ihtiyacı ≈ 24 RAMB36'dır (83 KB ağırlık + 10 KB parametre). Bu,
   kartın BRAM'inin ~%17'sine denk gelir. Kalan pay, tez deneyleri için yeterlidir: $P = 32$ şerit veya
   ~4 katman × $C = 96$ boyutlu (~440 k parametre) daha büyük bir model sığar.
2. **Ücretsiz araç:** xc7z020, ücretsiz **Vivado ML Standard** sürümünde desteklenir.
3. **ARM + PYNQ:** Bir sonraki aşamada hızlandırıcı AXI-Lite ile PS'e bağlanabilir ve Jupyter'den
   Python ile sürülebilir. Böylece tezde aynı kart üzerindeki ARM işlemciye karşı (NumPy/PyTorch) **hız ve
   enerji karşılaştırması** yapılabilir. Değerlendirme bölümü için bu güçlü bir argümandır.
4. **Yaygınlık:** Üniversitelerde yaygın, dokümantasyonu ve topluluğu geniş bir karttır.

> Laboratuvarda başka bir kart varsa geçiş kolaydır. Gereken üç değişiklik: part numarası, XDC dosyası
> ve saat girişi. Öz-test Basys 3'e de sığar.

## 4.2 İlk kart hedefi: yerleşik öz-test (BIST)

`board/pynq_z2/pynq_z2_top.sv` → `rtl/gpt_selftest.sv` → `rtl/gpt_top.sv`

Öz-test modülü, sıralama görevinin **729 girdisinin hepsini** çip üzerinde sayaçla üretir. Her girdi için
hızlandırıcıyla kapalı döngü üretim yapar ve sonucu kontrol eder: çıktı azalmayan sıralı olmalı ve
girdiyle aynı histograma sahip olmalıdır. Çipte altın veri saklanmaz, yani kontrol modelden bağımsızdır.
PC veya yazılım gerekmez.

| Kart öğesi | Anlamı |
|---|---|
| LD0 | yanıp söner: bitstream çalışıyor (heartbeat) |
| LD1 / LD2 | test sürüyor / bitti |
| LD3 | hata enjeksiyonu açık (SW0) |
| **LD4 (RGB)** | **mavi** = sürüyor, **yeşil** = 729/729 doğru, **kırmızı** = hata var |
| BTN0 | testi yeniden başlat (konfigürasyondan sonra test kendiliğinden bir kez çalışır) |
| BTN1 | reset |
| SW0 | hata enjeksiyonu: her dizinin ilk çıktısı bozulur → **kırmızı** beklenir |

Süre: 116.6 M çevrim. Bu, 50 MHz'de **≈ 2.3 s** sürer (100 MHz'de 1.17 s).
Simülasyonda doğrulanmıştır (`make selftest`): normal modda 729/729 geçer, hata enjeksiyonunda 729/729 kalır.

**Saat:** 125 MHz `sysclk` (H16) girişinden MMCM ile **50 MHz** üretilir. Bu değer, zamanlama raporu
görülene kadar bilinçli olarak temkinli seçildi. Rapor uygunsa `CLKOUT_DIV = 10.0` ile 100 MHz'e
çıkılabilir. (Not: PL'deki H15 "Ethernet" saati kablo takılı değilken kararsızdır, bu yüzden H16 kullanılır.)

## 4.3 Kartta çalıştırma adımları

1. Vivado ML Standard'ı (ücretsiz) bir Windows/Linux bilgisayara kur.
2. Kart ayarları: **Power** jumper'ı → USB, **Boot** jumper'ı → JTAG. Micro-USB (PROG/UART) kablosunu
   tak ve kartı aç.
3. Bitstream'i üret (~10–20 dk):
   ```bash
   vivado -mode batch -source scripts/vivado_build_pynq_z2.tcl
   ```
   Bitstream ve raporlar `build/pynq_z2/` altına yazılır (`utilization.rpt`, `timing.rpt`, `power.rpt`).
   Script en kötü setup slack'ini yazdırır; değer negatifse saat düşürülmelidir.
4. Kartı programla:
   ```bash
   vivado -mode batch -source scripts/vivado_program.tcl
   ```
5. Beklenen: LD4 ~2.3 s mavi yanar, sonra **yeşil** olur. SW0'ı açıp BTN0'a basınca **kırmızı** olmalıdır.

Rapor için bu raporlardan kaynak kullanımı, Fmax ve güç değerleri alınabilir. Kartın fotoğrafı
(yeşil LED) da sunumda iyi bir görsel kanıttır.

## 4.4 Sonraki aşama: PS + AXI + Python

1. `gpt_top` için AXI4-Lite sarmalayıcı yazmak: token/konum yazmaçları, `start`, `done`, `tok_out` ve
   logitler.
2. Vivado blok tasarımı kurmak: Zynq PS7 + AXI + sarmalayıcı, saat olarak PS'in `FCLK_CLK0`'ı.
3. PYNQ `Overlay` ile Jupyter'den prompt göndermek ve çıktıyı okumak.
4. Aynı işi ARM üzerinde NumPy/PyTorch ile yapıp gecikmeyi karşılaştırmak (tez değerlendirme bölümü).
