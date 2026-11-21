# Eşlenik çift örnekleri ve etiketleyici dal maliyetleri

Her şablon için d = 8, yüksek şiddet, düşük gürültü örnek çiftinde karar adımındaki (t0 = 47) dört dalın (G, K, T+R+S) maliyetleri. Etiket a* ve kabul edilebilir küme buradan türetilir.


## BAT_SEN

| Üye | İlk eylem | G | K | T+R+S | Etiket |
|---|---|---|---|---|---|
| A | NORMAL | 0 | 1 | 0 |  |
| A | YUKU_KAPAT | 0 | 1 | 0 |  |
| A | KISITLI_MOD | 0 | 0 | 0 | a* |
| A | GUVENLI_MOD | 0 | 0 | 60 |  |
| B | NORMAL | 0 | 1 | 0 |  |
| B | YUKU_KAPAT | 0 | 60 | 0 |  |
| B | KISITLI_MOD | 0 | 60 | 0 |  |
| B | GUVENLI_MOD | 0 | 0 | 0 | a* |

## SEN

| Üye | İlk eylem | G | K | T+R+S | Etiket |
|---|---|---|---|---|---|
| A | NORMAL | 0 | 0 | 0 | a* |
| A | YUKU_KAPAT | 0 | 0 | 60 |  |
| A | KISITLI_MOD | 0 | 0 | 60 |  |
| A | GUVENLI_MOD | 0 | 0 | 60 |  |
| B | NORMAL | 0 | 60 | 0 |  |
| B | YUKU_KAPAT | 0 | 60 | 0 |  |
| B | KISITLI_MOD | 0 | 0 | 0 | a* |
| B | GUVENLI_MOD | 0 | 0 | 60 |  |

## OC

| Üye | İlk eylem | G | K | T+R+S | Etiket |
|---|---|---|---|---|---|
| A | NORMAL | 0 | 0 | 0 | a* |
| A | YUKU_KAPAT | 0 | 0 | 60 |  |
| A | KISITLI_MOD | 0 | 0 | 60 |  |
| A | GUVENLI_MOD | 0 | 0 | 60 |  |
| B | NORMAL | 0 | 60 | 0 |  |
| B | YUKU_KAPAT | 0 | 60 | 0 |  |
| B | KISITLI_MOD | 0 | 0 | 0 | a* |
| B | GUVENLI_MOD | 0 | 0 | 60 |  |
