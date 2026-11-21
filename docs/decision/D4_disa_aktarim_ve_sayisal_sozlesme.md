# D4.1–D4.4 Model dışa aktarma, sayısal sözleşme, RTL arayüzü ve guard

21 Kasım 2026. Kod: `sw/decision/{export_dec,hw_ref,gen_rtl_pkg,gen_rtl_vectors}.py`, `sw/zoo/{ir,quant,export_hw}.py`, `rtl/zoo/zoo_core.sv`, `rtl/decision/*.sv`.

## D4.1 Dışa aktarma sözleşmesi

| Öğe | Anlam | Kaynak |
|---|---|---|
| Sözlük | 64 token: 0–3 eylem, 4–6 mod, 7 BOS, 8–25 V binleri, 26 V_INV, 27–40 I binleri, 41 I_INV, 42–53 T binleri, 54 T_INV, 63 PAD | `contract_v0.json` `vocab` |
| Akış | BOS, sonra adım başına V I T MODE ACT; MODE = karardan önceki mod, ACT = uygulanan eylem | `tokenizer.py`, `decision_top.sv` |
| Karar okuma | MODE tokenından sonra logit 0..3; maskeli argmax, eşitlikte büyük indeks (daha ağır eylem) | `train_policy.decide`, `dec_action_select` |
| Motor bağlamı | T = 256 (periyodik RoPE tablosunun satır sayısı); 48 adımlık senaryo 241 token | `models_dec.periodic_rope`, `compile_decision` |
| ATTN komutu | `e32` alanı = bant genişliği `win` (64); 0 ise tam nedensel aralık | `export_hw.py`, `zoo_core.sv` (`att_j0`) |
| RoPE | serpiştirilmiş çiftler, θ_k = 2π·m_k/256, m = (32, 8, 2, 1); tablo indeksi t mod 256 | `models_dec.py`, `ir.FloatRunner`, `quant.IntRunner` |
| Ağırlık / program dosyaları | `mem/decision/<model>/{prog,weights,params,tables,luts,exp}.mem`, `zoo_pkg.sv`, `dims.json` | `export_hw.export` |
| Altın vektörler | `tv.txt`: doğrulama senaryolarının karar adımına kadar öğretmen zorlamalı 64 logit | `export_dec.py` |
| Sürüm | sözleşme sha256, kalibrasyon indekslerinin sha256'sı, eğitim derleminin sha256'sı | `export.json`, eğitim `*.json` |

Desteklenmeyen işlem sessizce atlanmaz: `compile_model` bilinmeyen modelde hata verir, `export_hw` her işlemi bir komut sözcüğüne çevirir.

## D4.2 Sayısal sözleşme ve kalibrasyon

| Tensör | Biçim | Not |
|---|---|---|
| Ağırlıklar | INT8, çıktı başına ölçek (m0·2^−sh, m0 ∈ [2^15, 2^16)) | `quant.qmul` |
| Aktivasyonlar | INT8, tampon başına statik ölçek | %99,99 yüzdelik, ikinci geçişte histogramla (`quant.CALIB`) |
| Kalıntı (R) | INT16, 2'nin kuvveti ölçek | |
| NORM çıkışı | INT8, kanıtlanmış sınırdan ya da kalibrasyondan (hangisi daha ince değilse) | |
| tanh / sigmoid / GRU durumu | INT8 Q0.7 | GRU güncellemesi tamsayı |
| Logitler | INT16, kalibre ölçek | |
| Birikim | INT32 (aşım `assert` ile denetlenir) | |
| RoPE cos/sin | Q1.14; giriş çift normuna göre kalibre | |
| Softmax | maksimum çıkarma, 256 girişli 2^(−k/256) tablosu, tamsayı bölme | `attn_int` |

Kalibrasyon yalnız **eğitim** bölümünden yapılır: 300 bölüm, rastgele tohum 12345, motor bağlamına (256 token) kırpılmış. İndekslerin sha256'sı `export.json` içindedir; doğrulama ve test verisi ölçekleri belirlemez. Float→INT8 karar uyuşması doğrulama kümesinde ölçülür ve `export.json` içindeki `agreement_val` alanında, float ve INT8 doğruluğuyla birlikte raporlanır (eğitilmiş modellerin değerleri `docs/decision/sonuclar/val_results.md`).

Doygunluk ile iki'nin tümleyeni sarması aynı değildir: bütün dönüşümler `sat()` ile doyurulur, RTL'de de aynı.

## D4.3 RTL arayüzü ve pilot yol

`rtl/decision/decision_top.sv`:

- Gözlem ROM'u (64 bit sözcük: V, I, T, geçerlilik bitleri, yer komutu, öneri bastırma).
- Tokenlaştırıcı (`dec_tokenizer`).
- Politika: `POLICY` 0 ise `zoo_core` üzerinde tamsayı LM/GRU; 1 kural, 2 FSM.
- Guard (`dec_guard`) ve iz RAM'i (16 bit sözcük, FNV-1a tarzı hash).
- El sıkışmalar: `start` (bölüm), `zoo_core.start/done` (token adımı), `l_addr/l_data` (logit okuma), `dec_guard.start/prop_valid/done`.
- Sıfırlama: bölüm başında guard modu NORMAL, geçersiz sayacı 0, FSM yazmaçları temiz; motor durumu pos = 0'da sıfırlanır.

Doğrulama: `scripts/dec_top.sh <model> all` her politika için Python `hw_ref.py` ile iz sözcüğü ve hash karşılaştırır. Senaryolarda geçersiz örnek dizisi, FORCE_SAFE ve RESUME, bastırılmış öneri (zaman aşımı) ve sınırın tam üstü/altı kodlar bulunur.

| Politika | Adım | Hata | Karar süresi (çevrim) |
|---|---|---|---|
| LM (rastgele ağırlık, pilot) | 1.152 | 0 | 27.406 – 31.278 |
| kural | 1.152 | 0 | 2 |
| FSM | 1.152 | 0 | 2 |

Eğitilmiş LM ve GRU için aynı tablo `docs/decision/proje1_raporu.md` dosyasındadır.

Kart üzerinde bağımsız öz-test (`dec_board_top.sv`): BTNC bütün senaryoları çalıştırır, her bölümün hash'ini Python'ın beklediği değerle karşılaştırır; LED'ler çalışan/eşleşen senaryo sayısını, LED16 yeşil/kırmızı sonucu gösterir. UART ya da host protokolü yoktur.

## D4.4 Guard RTL'i ve test takımı

- `dec_guard_comb`: tek karar (kombinasyonel). Öncelik: FORCE_SAFE > V/T sınırı > RESUME > {zaman aşımı geri dönüşü, I sınırı, kalıcı geçersiz örnek, geçiş tablosu}.
- `dec_guard`: mod yazmacı, geçersiz sayacı ve bekçi zamanlayıcı (WDT çevrim). Aynı çevrimde gelen öneriye karşı zaman aşımı kazanır. Politika iç durumuna bağlı değildir (aynı yongada ortak saat, reset ve güç bağımlılığı vardır).
- Testler (`make dec-units`): 73.728 vektörün tamamı, 400 sıralı bekçi zamanlayıcı kenar durumu. Verilator ve Icarus (4 durumlu, X denetimi) ile 0 hata. `tests/decision` önceliklerin yazılı kurallarını ayrıca sınar.
- Sentez (Yosys, xc7): 61 LUT, 44 FF.
