# Doğrulama kümesi sonuçları (Proje I)

480 eşlenik çift (48 hücre × 10). Çift doğruluğu: iki üyede de kabul edilebilir eylem. Öğrenen modeller için 3 tohumun ortalaması ± standart sapma; parantezde tohum 0'ın çift düzeyinde %95 bootstrap güven aralığı. Test bölümü Proje I'de açılmadı.

| Yöntem | Senaryo doğruluğu | Çift doğruluğu | d=2 | d=8 | d=16 | d=32 |
|---|---|---|---|---|---|---|
| rule | 0.493 | 0.000 (0.000–0.000) | 0.000 | 0.000 | 0.000 | 0.000 |
| fsm | 0.940 | 0.890 (0.860–0.919) | 0.733 | 0.942 | 0.933 | 0.950 |
| lm | 0.792 | 0.590 ± 0.030 (0.554–0.646) | 0.781 | 0.811 | 0.767 | 0.000 |

## Şablon × d (çift doğruluğu)

| Yöntem | BAT_SEN/d2 | BAT_SEN/d8 | BAT_SEN/d16 | BAT_SEN/d32 | SEN/d2 | SEN/d8 | SEN/d16 | SEN/d32 | OC/d2 | OC/d8 | OC/d16 | OC/d32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fsm | 0.60 | 0.90 | 0.85 | 0.97 | 0.90 | 0.97 | 0.95 | 1.00 | 0.70 | 0.95 | 1.00 | 0.88 |
| lm | 0.87 | 0.96 | 0.93 | 0.00 | 0.95 | 0.95 | 0.88 | 0.00 | 0.53 | 0.53 | 0.50 | 0.00 |

## Pencere içinde (d ≤ 16) üye düzeyinde senaryo doğruluğu

BAT_SEN: A = sensör arızası → KISITLI_MOD, B = batarya kapasite kaybı → GUVENLI_MOD. SEN: A = sağlam → NORMAL, B = takılı sensör → KISITLI_MOD. OC: A = tek geçici aşırı akım → NORMAL, B = aralıklı aşırı akım → KISITLI_MOD.

| Yöntem | BAT_SEN/A | BAT_SEN/B | SEN/A | SEN/B | OC/A | OC/B |
|---|---|---|---|---|---|---|
| rule | 0.94 | 0.00 | 0.99 | 0.01 | 1.00 | 0.00 |
| fsm | 0.91 | 0.83 | 0.97 | 0.97 | 0.99 | 0.89 |
| lm | 0.94 | 0.97 | 0.99 | 0.94 | 0.55 | 0.96 |

## INT8 karar uyuşması (doğrulama kümesi, float ↔ tamsayı)

Marj: float üst-2 logit farkı. Uyuşmazlıklar ve hangi tarafın kabul edilebilir eylemi seçtiği `mem/decision/<model>/export.json` içinde (`disagreements`).

| Model | n | Uyuşma | Uyuşmazlık | Marj ≥ 0,2 iken uyuşma | Float doğruluk | INT8 doğruluk |
|---|---|---|---|---|---|---|
| lm_s0 | 960 | 0.9875 | 12 | 1.000 (n = 897) | 0.795 | 0.797 |
| lm_s1 | 960 | 0.9896 | 10 | 1.000 (n = 886) | 0.777 | 0.777 |
| lm_s2 | 960 | 0.9948 | 5 | 1.000 (n = 906) | 0.804 | 0.809 |
