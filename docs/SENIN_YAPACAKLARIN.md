# Senin yapacakların

Bu not, bu bilgisayarda (Apple M1, macOS) yapılamayan işleri anlatıyor. Bu işler bir x86 bilgisayarda Vivado çalıştırmayı, PYNQ-Z2 kartıyla ölçüm yapmayı, güç ölçmeyi ve makale başvurusunu kapsıyor. Bu Mac'te yapılabilecek her şey yapıldı ve depoda duruyor; aşağıdaki bölüm 0 neyin hazır olduğunu özetliyor.

Sıra şöyle: **V (Vivado) → K (Kart) → G (Güç) → S (Sonuçları ekle) → Y (Yazım ve başvuru)**. Her adımın sonunda kontrol edebileceğin bir "beklenen çıktı" var. Bir adım beklenen çıktıyı vermezse "Sorun giderme" kısmına bak.

---

## 0. Durum: hazır olanlar ve sana kalanlar

| Konu | Durum | Nerede |
|---|---|---|
| 9 mimari + stories260K, INT8, tek mikrokodlu RTL | **Bitti**, 10 modelin hepsi bit-exact | `rtl/zoo/`, `sw/zoo/` |
| Tek bitstream (evrensel motor) + AXI4-Lite + model yükleme | **Bitti**, simülasyonda 10 model + 200 tokenlık hikâye bit-exact | `rtl/zoo/zoo_axi.sv`, `tb/tb_zoo_axi.sv` |
| Doğrulama: fuzzing, 4 durumlu simülasyon, assertion'lar, aritmetik birim testleri | **Bitti** | `sw/zoo/fuzz.py`, `tb/tb_zarith.sv` |
| Kaynak tahmini (Yosys), P taraması | **Bitti** (zamanlama yok) | `model/zoo/hw_metrics.json`, `model/zoo/dse.json` |
| Çok tohumlu eğitimler, özellik merdiveni, TinyStories eğitimleri | **Bitti** | `model/zoo/runs/`, `model/zoo/*.json` |
| CPU temel çizgisi: C ISA simülatörü (M1'de ölçüldü) | **Bitti**; ARM'da ölçmek sana kaldı | `sw/c/zoo_run.c` |
| Makale taslağı (İngilizce, IEEE) ve Türkçe rapor | **Bitti**; kart sayıları yer tutucu olarak duruyor | `paper/`, `report/` |
| **Vivado: OOC tarama (Fmax, güç tahmini) ve PYNQ bitstream'i** | **Sende** (bölüm V) | `scripts/vivado_zoo_*.tcl` |
| **Kartta çalıştırma, gecikme ve ARM karşılaştırması** | **Sende** (bölüm K) | `board/pynq_z2/` |
| **Güç ölçümü** | **Sende** (bölüm G) | — |
| **Danışman, yazar listesi, başvuru, arXiv, Zenodo** | **Sende** (bölüm Y) | — |

Vivado'yu **yalnızca bir kez** çalıştırman gerekiyor. Tek bitstream bütün modelleri çalıştırıyor; kartta model değiştirmek için yeniden sentez yapmıyorsun, Python sürücüsü yeni modeli yüklüyor.

---

## Gerekenler

- **x86-64 bilgisayar:** Windows 10/11 ya da Ubuntu 22.04/24.04. En az 16 GB RAM ve 100 GB boş disk gerekiyor (Vivado yalnızca Zynq-7000 desteğiyle kurulursa ~40 GB). Laboratuvar bilgisayarı yeterli. Apple Silicon Mac olmaz; Vivado orada çalışmıyor.
- **PYNQ-Z2 kartı:** microSD kart (en az 16 GB), Ethernet kablosu, micro-USB kablo.
- **USB güç ölçer** (G bölümü için): USB-A girişli, akım ve gerilim gösteren bir ara ölçer. Ucuz modeller yeterli. Kart micro-USB üzerinden beslenecek ve ölçer arada duracak.
- **Hesaplar:** AMD (Xilinx) hesabı (Vivado indirmek için, ücretsiz), arXiv hesabı, GitHub hesabın (zaten var), Zenodo (GitHub ile giriş).

---

## V. Vivado (x86 bilgisayarda)

### V1. Kurulum
1. AMD hesabıyla giriş yapıp Vivado'nun "Unified Installer" indiricisini al. Herhangi bir güncel sürüm uygun (2023.2 veya sonrası önerilir). Betikler sürüme bağlı değil.
2. Kurulumda **Vivado ML Standard** seç. Bu sürüm ücretsiz ve lisans gerektirmiyor; XC7Z020'yi destekliyor ([AMD UG973, Supported Devices](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Devices)).
3. Aygıt listesinde yalnızca **Zynq-7000** seçersen kurulum çok küçülür.
4. Linux'ta kurulumdan sonra her terminalde ortamı yükle: `source <Kurulum>/Vivado/<sürüm>/settings64.sh`.

**Beklenen:** `vivado -version` komutu sürümü yazar.

### V2. PYNQ-Z2 kart dosyaları
Vivado, PS (ARM tarafı) DDR ayarlarını kart dosyasından alır. Kart dosyaları Vivado'yla gelmez.
1. TUL'un ürün sayfasından "PYNQ-Z2 Board File" dosyasını indir: <https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html> (Downloads bölümü; [PYNQ belgeleri](https://pynq.readthedocs.io/en/v3.1/_sources/overlay_design_methodology/board_settings.rst.txt) de bu bağlantıyı veriyor).
2. Zip'i aç. İçinden çıkan `pynq-z2` klasörünü şuraya kopyala: `<Kurulum>/Vivado/<sürüm>/data/xhub/boards/XilinxBoardStore/boards/Xilinx/`. Eski sürümlerde (2020.x ve öncesi) hedef klasör `data/boards/board_files/`.
3. Vivado açıksa kapatıp yeniden aç.

**Beklenen:** Vivado Tcl konsolunda `get_board_parts *pynq-z2*` bir sonuç döndürür.

### V3. Depo
```bash
git clone https://github.com/Muratatceken/mingpt-fpga.git
cd mingpt-fpga
```
Bütün bellek görüntüleri (`mem/zoo/...`) depoda hazır. Vivado adımları için Python gerekmiyor.

### V4. Model başına uygulama taraması (kaynak, Fmax, güç tahmini)
Her model için motor tek başına sentezlenir, yerleştirilir ve yönlendirilir. Yaklaşık 21 koşu, model başına 5–15 dakika:
```bash
vivado -mode batch -source scripts/vivado_zoo_sweep.tcl
# yalnızca birkaç model:  vivado -mode batch -source scripts/vivado_zoo_sweep.tcl -tclargs gpt2 llama universal
```
**Beklenen:** `build/zoo/vivado/sweep.jsonl` dosyası oluşur ve her satırda `lut, ff, bram, dsp, wns, fmax_mhz, power_w` bulunur. Raporlar da aynı klasörde (`util_*.rpt`, `timing_*.rpt`, `power_*.rpt`).
**Yapacağın:** `sweep.jsonl` dosyasını depoda `model/zoo/vivado_sweep.jsonl` olarak kopyala.

Fmax değeri 100 MHz kısıtıyla yapılan uygulamadan türetiliyor: `1000 / (10 − WNS)`. WNS negatifse tasarım 100 MHz'e yetişmiyor demektir; bunu raporda olduğu gibi yaz.

### V5. PYNQ overlay'i (tek bitstream)
```bash
vivado -mode batch -source scripts/vivado_zoo_pynq.tcl            # 100 MHz
```
Betik sırasıyla şunları yapar:
1. Zynq PS7, AXI ara bağlantısı ve motorun kendisiyle bir blok tasarım kurar.
2. Motoru 0x4000_0000 adresine, 64 KB'lık bir alana yerleştirir.
3. Bitstream'i üretir.

**Beklenen:** `build/zoo/pynq_overlay/` klasöründe `zoo.bit`, `zoo.hwh` ve `overlay.json` oluşur; konsolda "timing NOT met" uyarısı görünmez.

- **Zamanlama tutmazsa:** Betiği daha düşük bir saatle yeniden çalıştır: `-tclargs 80` (ya da 75). Yeni frekansı not et; kartta ölçülen süreler bu frekansa göre olacak.
- **Kaydetmen gerekenler:** `zoo.bit`, `zoo.hwh` ve `overlay.json` dosyalarını depoda `board/pynq_z2/overlay/` klasörüne kopyalayıp commit et. Kart bu dosyaları oradan alacak, rapor da `overlay.json`'u okuyacak.

### V Sorun giderme
- **"PYNQ-Z2 board files not found":** V2 adımı eksik. Ayrıca klasör adının tam `pynq-z2` olduğunu kontrol et.
- **`apply_bd_automation` hata verirse:** Vivado sürümün çok eskidir. 2022.1 veya sonrası bir sürüm kur.
- **`$readmemh` dosya bulamıyor:** Betikler `build/gen/` altında mutlak yollu bir paket üretir. Depo klasörünü taşıdıysan betiği yeniden çalıştır.
- **Yetersiz BRAM:** Evrensel motor yaklaşık 120–135 RAMB36 kullanıyor (xc7z020'de 140 var). Sığmazsa motorun bağlamını 256'dan 128'e düşür ve tekrar dene: `cd sw/zoo && python3 export_hw.py --universal <modeller> --T 128`.

---

## K. Kart (PYNQ-Z2)

### K1. SD kart
1. PYNQ-Z2 imajını indir: <https://download.amd.com/opendownload/pynq/pynq_z2_v3.1.1.zip> (kaynak: [pynq.io board listesi](https://pynq.io/index.html)).
2. Zip'i aç ve `.img` dosyasını balenaEtcher (ya da Raspberry Pi Imager) ile microSD karta yaz.
3. Kartta **SD** önyükleme jumper'ını (JP1) SD konumuna, güç jumper'ını (J9) **USB** konumuna getir. USB konumu G bölümündeki güç ölçümü için gerekli.

### K2. Bağlantı
1. Ethernet'i bilgisayara ya da ağa bağla ve kartı aç.
2. Tarayıcıdan <http://pynq:9090> adresine git. Açılmazsa kartın IP'sini bul ya da doğrudan bağlantıda bilgisayara 192.168.2.1 statik IP verip <http://192.168.2.99:9090> adresini dene. Şifre: `xilinx`.
3. Jupyter'da New → Terminal ile kartta bir terminal aç. Kullanıcı `xilinx`, şifre `xilinx`; komutları `sudo` ile çalıştır.

### K3. Depo ve overlay
Kartın internet bağlantısı varsa:
```bash
cd /home/xilinx && git clone https://github.com/Muratatceken/mingpt-fpga.git && cd mingpt-fpga
mkdir -p build/zoo/pynq_overlay && cp board/pynq_z2/overlay/zoo.bit board/pynq_z2/overlay/zoo.hwh build/zoo/pynq_overlay/
```
İnternet yoksa depoyu bilgisayarından `scp -r` ile kopyala.

### K4. Ölçüm
```bash
sudo python3 board/pynq_z2/run_board.py --cpu --story 200
```
Bu komut sırasıyla şunları yapar:
1. **Modelleri yükler:** `mem/zoo/universal/` altındaki 10 modelin her birini AXI üzerinden motora yükler.
2. **Kartta bit-exact kontrolü yapar:** Her modelin altın vektörlerini kartta çalıştırıp her logiti karşılaştırır.
3. **Gecikmeyi ölçer:** Donanım çevrim sayacını ve Python'dan duvar saati süresini kaydeder.
4. **ARM temel çizgisini çalıştırır:** `--cpu` ile aynı tamsayı işi ARM Cortex-A9'da yapılır (C programı kartta `gcc -O3` ile derlenir).
5. **Hikâye üretir:** `--story 200` ile FPGA'de 200 tokenlık bir hikâye üretip metni yazar.

**Beklenen:**
- Her model için `"errors": 0`.
- Hikâyenin sonunda "200/200 tokens identical to the integer reference".
- `board_results.json` dosyası.

**Yapacağın:** `board_results.json` dosyasını depoda `model/zoo/board_results.json` olarak kaydet ve commit et.

### K Sorun giderme
- **"zoo engine not found (ID ...)":** Overlay yanlış ya da eski. `zoo.bit` ve `zoo.hwh` aynı Vivado koşusundan çıkmış olmalı.
- **`errors` sıfırdan büyük:** Bu ciddi bir durum. Hangi model, kaç hata ve `cycles_*` değerleri neydi, bunları not al. Aynı modeli Mac'te `LOAD=1 MODEL=mem/zoo/universal/<model> scripts/zoo_sim.sh universal` ile çalıştırınca simülasyon geçiyorsa sorun sentez/zamanlama tarafındadır: V5'i daha düşük bir saatle tekrarla.
- **`gcc` bulunamadı:** `sudo apt install build-essential` (PYNQ imajında genelde kurulu gelir).

---

## G. Güç ölçümü

Amaç token başına enerjiyi hesaplamak (FPGA ve ARM için ayrı ayrı). Ölçer kart girişindeki toplam gücü gösterir. Farkları kullanacağız.

1. Kartı ölçer üzerinden besle (J9 = USB). Önyükleme bitince 1 dakika bekle.
2. **P_bos:** Hiçbir şey çalışmazken ölçerdeki gücü (W) oku.
3. Overlay yüklü ama boşta: `sudo python3 -c "import sys; sys.path.insert(0,'board/pynq_z2'); from zoo import Zoo; Zoo('build/zoo/pynq_overlay/zoo.bit'); input('oku, Enter')"`. Ölçeri oku: **P_pl_bos**.
4. FPGA sürekli üretim yaparken: `sudo python3 board/pynq_z2/run_board.py --power fpga 60`. Değer oturduktan sonra ölçeri oku: **P_fpga**. Komutun yazdığı tokens/s değerini de not al.
5. ARM sürekli çalışırken: `sudo python3 board/pynq_z2/run_board.py --power cpu 60`. Ölçeri oku: **P_cpu**; tokens/s değerini de not al.
6. Değerleri depoda `model/zoo/power.json` dosyasına şu biçimde yaz:
   ```json
   {"meter": "ölçerin marka/model", "supply_v": 5.0,
    "idle_w": 0.00, "pl_idle_w": 0.00, "fpga_run_w": 0.00, "cpu_run_w": 0.00,
    "fpga_tokens_per_s": 0.0, "cpu_tokens_per_s": 0.0, "notes": ""}
   ```
   Token başına enerjiyi ben yazdığım betikle hesaplatacağım: (P_fpga − P_pl_bos) × (çevrim/token ÷ f), ve ARM için (P_cpu − P_bos) / (tokens/s). Değerleri yalnızca dosyaya yazman yeterli.

---

## S. Sonuçları rapora ve makaleye ekleme

Şu dosyalar depoya eklendiğinde rapor ve makale kendini günceller (`sw/zoo/report_zoo.py` ve `paper/` bunları okur; dosya yoksa ilgili yerde "ölçülmedi" yazar):
- `model/zoo/vivado_sweep.jsonl` (V4)
- `board/pynq_z2/overlay/overlay.json` (V5)
- `model/zoo/board_results.json` (K4)
- `model/zoo/power.json` (G)

Sonra LaTeX kurulu herhangi bir bilgisayarda (bu Mac'te kurulu):
```bash
make report      # report/rapor.pdf  (Türkçe tez raporu)
make paper       # paper/main.pdf    (İngilizce makale)
git add -A && git commit -m "Kart ve Vivado ölçümleri" && git push
```

---

## Y. Yazım ve başvuru

### Y1. Nereye gönderilir?
Aşağıdaki tarihler web'de 13 Ekim 2026'da kontrol edildi. "Tahmini" yazanlar önceki yılın takviminden çıkarıldı; resmî duyuruyu kontrol et.

| Konferans | Son tarih | Not |
|---|---|---|
| **DATE 2027 – Late Breaking Results** | 29 Kasım 2026 | Kısa (2 sayfa) bildiri; en yakın gerçekçi hedef |
| **FCCM 2027** | ~Ocak 2027 (tahmini) | FPGA odaklı; tam makale |
| **FPL 2027** | ~Mart–Nisan 2027 (tahmini) | FPGA odaklı; tam makale |
| **SIU 2027** | ~Mart 2027 (tahmini) | Türkiye; 4 sayfa, Türkçe olabilir. SIU 2026'da bir lisans projesi yarışması da vardı (son tarih Haziran) |
| ISCAS 2027 | 13 Ekim 2026 | Kaçtı |
| FPGA 2027 / FPT 2026 | Geçti | — |

**Öneri:** Kart ölçümleri Kasım sonuna yetişirse DATE LBR'a kısa bir bildiri gönder. Aynı anda arXiv'e tam sürümü koy. Asıl hedef FCCM ya da FPL olsun; SIU'ya Türkçe kısa sürüm gönderilebilir. Kaynaklar ve gerekçeler `paper/related_work.md` dosyasında (64 doğrulanmış kaynak).

### Y2. Başvurudan önce
- **Yazar listesi:** Danışmanın büyük ihtimalle ortak yazar olur; bunu kendisiyle konuş. Kurum adını ve e-postaları `paper/main.tex` içine yaz.
- **Kör hakemlik:** FCCM ve FPL çift kör olabilir. Bu durumda makaleden isimleri ve GitHub bağlantısını çıkar. Kod bağlantısı için anonim bir kopya kullan (örneğin <https://anonymous.4open.science>).
- **Konumlandırma:** Yakın çalışmalar HLSTransform, SECDA-LLM, DFX, I-BERT ve I-LLM. Makaledeki "Related work" bölümü bunları açıkça kabul ediyor; bu dürüstlüğü koru. Asıl katkı, her mimari yeniliğin eşit bütçede donanım maliyeti ve tek motorda 9 mimari.
- **Sınırlamalar bölümü:** Modeller küçük (≤260K), kart ölçümleri tek kartta yapıldı ve Hat B'de tek tohum kullanıldı. Bunlar makalede yazılı.

### Y3. arXiv
1. arXiv hesabı aç. İlk gönderide cs.AR (Hardware Architecture) kategorisi için "endorsement" (onay) gerekebilir; danışmanın ya da daha önce gönderi yapmış bir hoca onay verebilir.
2. `make paper` ile üretilen kaynak dosyaları (main.tex, refs.bib, şekiller) yükle. Lisans olarak CC BY 4.0 uygun.

### Y4. Zenodo (kalıcı DOI)
1. <https://zenodo.org> sitesine GitHub ile giriş yap. Ayarlarda GitHub entegrasyonunu açıp `mingpt-fpga` deposunu etkinleştir.
2. GitHub'da bir release oluştur (örneğin `v1.0-paper`). Zenodo otomatik olarak DOI verir.
3. DOI'yi makaledeki "Artifact" cümlesine ve README'ye ekle.

---

## D. Danışmanla konuşulacaklar
1. Hedef konferans (Y1) ve yazar sırası.
2. Vivado'lu bilgisayar ve PYNQ-Z2 kartı: laboratuvarda var mı, ne zaman kullanabilirsin?
3. USB güç ölçer: laboratuvarda var mı?
4. arXiv endorsement.
5. Tezin son teslim tarihi ve formatı (Türkçe rapor `report/rapor.pdf` hazır).

---

## Ek A. Neyin nerede doğrulandığı

| Doğrulama | Araç | Sonuç dosyası |
|---|---|---|
| PyTorch ↔ IR (float) | `sw/zoo/check_ir.py` | konsol |
| Tamsayı referans ↔ RTL, bütün modeller ve tohumlar | Verilator (`scripts/zoo_sim.sh`) | `model/zoo/hw_metrics.json`, `model/zoo/verify.json` |
| Özelleşmiş motor, yükleme portu, evrensel motor, AXI sistemi | Verilator | `model/zoo/runs/universal_sim.log` |
| Rastgele farklılık testi (60 vaka, 1,26 M logit) | `sw/zoo/fuzz.py` | `model/zoo/fuzz.json` |
| 4 durumlu simülasyon, X kontrolü | Icarus (`SIM=icarus`) | `model/zoo/verify.json` |
| Aritmetik birimler (36,4 M vaka; W=12'de tam) | `tb/tb_zarith.sv` | konsol / rapor |
| C ISA simülatörü ↔ altın vektörler | `sw/c/zoo_run.c` | konsol |
| **Kartta bit-exact** | `board/pynq_z2/run_board.py` | **sende** (K4) |

## Ek B. Motorun yazmaç haritası
Yazmaçlar `rtl/zoo/zoo_axi.sv` dosyasının başında belgelendi. Sürücü `board/pynq_z2/zoo.py` dosyasında.
