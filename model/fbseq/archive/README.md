# İlk hard-aware koşusunun kayıtları (27 Ekim 2026)

Bu dosyalar `sw/fbseq/hardaware.py`'nin ilk sürümüyle üretildi. Bu sürümde iki hata vardı; ikisi de commit edilmeden önce düzeltildi.

- `hardaware_bytes_run1.json` (Loghub, B_setkeep, 2 seed). Hata: eğim tavlaması (a: 1 → 5) yalnız `slope`'a değil, `hardst`'ye de uygulanıyordu. Bu yüzden `hardst` satırları `slope` satırlarıyla birebir aynı. Düzeltilmiş `hardst` ve `slope` sonuçları `../hardaware_bytes.json` içinde. `gumbel` satırları bu hatadan etkilenmedi: surrogate ölçeği 1/τ, eğim değişkenini kullanmıyor. Bu satırlar (1,6905 ve 1,6799; ortalama 1,685) belgelerdeki Gumbel-ST değerinin kaynağı. Değer, düzeltilmiş kodla yeniden üretildi ve birebir aynı çıktı (`../hardaware_bytes.json`, 29 Ekim).
- `hardaware_a5_run1.json` (A5, B_block3). Fark: kb > 1 için eski multilinear relaxation kullanıldı; bu relaxation gradyan patlamasına yol açıyordu. Sonuçlar güncel dağılım relaxation'ıyla karşılaştırılamaz; yalnız kayıt amaçlı tutuluyor. Güncel sonuçlar `../hardaware_a5.json` içinde.
