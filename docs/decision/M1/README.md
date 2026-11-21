# M1 — Şartname gözden geçirme paketi (WP1.6)

Tarih: 21 Kasım 2026 · Sözleşme sha256 `0b1ea82508605a72f9dfbb8c3f0e1ad9c8e463e5fa36c26b79e788067af99b6c`

Bu paket danışman gözden geçirmesi içindir. Gözden geçirme sonucu (kabul / açık maddeler) bu dosyanın sonundaki kayda işlenir; yalnız toplantı yapılması tamamlanma sayılmaz (WP1.6 tamamlanma ölçütü).

## İçerik

- `../D1.1_kanit_kaydi.md`: yeniden kullanılan her sayının kaynağı ve ölçüm sınırı
- `../D1.2_arastirma_sorusu_matrisi.md`: hipotezler, karşılaştırma tabanları, ölçütler, çürütme koşulları
- `../D1.3_alt_sistem_ve_ariza_tanimi.md`: kanallar, dinamikler, parametreler, arıza sınıfları (üretilmiş)
- `../D1.4_eylem_ve_guard_tablolari.md`: eylemler, modlar, guard öncelikleri, geçiş tablosu (üretilmiş)
- `../D1.5_degerlendirme_protokolu.md`: ön kayıt — d tanımı, etiketleyici, bölümler, istatistik, karar süresi
- `fig_directed.png`: arıza sınıflarının yönlendirilmiş testleri
- `fig_pairs.png`, `labeler_costs.md`: eşlenik çift örnekleri ve dal maliyetleri
- `../../../sw/decision/manifests/*.json`: bölüm manifestleri (sha256, kabul oranları)

## Derlem özeti

| Bölüm | Çift | Senaryo / bölüm | Not |
|---|---|---|---|
| val | 480 | 960 | hücre başına 10 çift (D2) |
| test | 2880 | 5760 | 48 hücre × (30 D2 + 20 D3) + 480 D4; Proje I'de açılmaz |
| train | — | 20630 | 1513070 etiket |

## Açık maddeler (danışman kararı gereken)

1. **Bağlam:** etkin pencere 127 token (2 katman × 64 bant). 2209 metnindeki “C = 128” ile tutarlı; 1 katman × 128 B planı ablasyon olarak tutuluyor.
2. **Etiketleyici devam politikası:** saf eşik kuralı ilk eylemi bir adım sonra geri aldığından kilitli GUVENLI neredeyse her etiketi kazanıyordu; tekdüze kural `max(kural, ilk eylem)` seçildi, saf kural duyarlılık olarak raporlanacak (D1.5).
3. **Kısıtlı mod maliyeti:** gereksiz KISITLI/GUVENLI moda R maliyeti eklendi; yoksa YUKU_KAPAT ile KISITLI eşit kalıp eşitlik kuralı hep KISITLI seçiyordu.
4. **Düşük kabul oranlı hücreler** (test, < %70): {'SEN/low/high/d32/D2': 0.683, 'SEN/low/high/d32/D3': 0.321, 'SEN/high/high/d32/D2': 0.634, 'SEN/high/high/d32/D3': 0.375, 'OC/low/low/d2/D3': 0.679}. Yüksek gürültü + d = 32 SEN hücrelerinde yüklü gerilim SoC ile kayarken takılı okuma sabit kalıyor; parametre seçim yanlılığı manifestte kayıtlı. Şablon revizyonu ya da kabul eşiği kararı gerekli.
5. **Kart üzerinde kapalı döngü:** pilot ROM'dan açık döngü yeniden oynatmadır; kart üzerinde tesis modeli (RTL'de basit güç modeli) Proje II kararıdır.
6. **RTL halka tamponu:** kayan pencere Python'da; RTL'de bant başlangıcı (j0) var, T = 256 ile 48 adımlık senaryolar sığıyor; uzun akış için halka tampon Proje II.

## Değişiklik kaydı

| Tarih | Sürüm | Değişiklik | Veri/model etkisi |
|---|---|---|---|
| 21.11.2026 | v0.1 | ilk sözleşme | derlem v1 |

## Gözden geçirme kaydı

| Tarih | Gözden geçiren | Karar | Açık maddeler |
|---|---|---|---|
| | | | |
