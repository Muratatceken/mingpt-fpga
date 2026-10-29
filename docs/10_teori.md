# Geri besleme kısıtlı sıralı hesaplama: teori çerçevesi

**Merkezi soru.** *Can we learn hardware-realizable discrete state-transition systems without sacrificing the sequence-modeling capability of their continuous relaxations?*

**Bugünkü merkez iddia** (28 Ekim): Boolean recurrent modellerde durum kapasitesi, geçişlerin yerelliği ve geri besleme derinliği ayrı kısıtlardır. A5'te bunların bit maliyeti kesin olarak ayrıştırılabilir: 6, 7 ve 9 bit; 6 bit tek LUT6 seviyesinde bile sıkıdır. Input-gated yapıda discretization hatası döngü tarafından büyütülemez. Loghub'da geçiş kaybının büyük kısmı eğitimle kapanıyor (1,583 → 1,418; soft 1,398). Kalan fark, Boolean bitlerin sızdıran belleği temsil edememesiyle tutarlıdır; bu bir hipotez.

Bu belge, Faz 0 deneylerinden çıkan teorik çerçeveyi toplar. Beş ana bölümde ilerler:

1. Biçimsel model.
2. Bilinen sonuçların bize uyarlanması: zamanlama, grup yapıları, bilgi alt sınırı.
3. Kendi sonuçlarımız: A5 küçük vakasında kesin gerçekleştirme sınırları.
4. Öğrenilebilirlik ve kararlılık bulguları.
5. Literatürdeki yerimiz.

Literatür taraması Ekim 2026 sonunda üç hatta yapıldı. "Bulunamadı" ifadeleri bir arama sonucudur, yokluğun kanıtı değildir. Bütün sayılar `model/fbseq/*.json` dosyalarından gelir; bunları üreten kodlar `sw/fbseq/` altındadır.

---

## 0. Özet

| | Sonuç | Durum |
|---|---|---|
| Zamanlama | Cascade'in iteration bound'u, bileşen döngülerinin maksimumuna eşittir. Bizim mimari sınıfında bu sınıra ulaşılabilir. | **Bilinen** (Leiserson–Saxe 1991, Thm 11); uyarlama |
| Grup yapıları | m üreteçli G ≤ Sₙ'nin word problem'i tek LUT seviyesinde çözülebilir. Encoding'e göre fan-in ile state arasında bir ödünleşim var. | Parçaları bilinen; biçimsel ifadesi bulunamadı |
| Bilgi alt sınırı | Grup elemanını izlemek için S ≥ ⌈log₂\|G\|⌉ | **Bilinen** (Myhill–Nerode) |
| **A5, block-local** | Tek seviyede en az state: K = 6 için **7 bit**, K = 5 için **9 bit**; K = 4'te imkânsız | **Bizim** (alt grup taraması ile ispat) |
| **A5, genel** | K = 6 ile **6 bit**, yani alt sınır. Sınır sıkı. | **Bizim** (SAT tanığı + alt sınır) |
| Fiziksel karşılık | 6 bit 455 MHz, 7 bit 458 MHz, 15 bit 433 MHz, öğrenilen 3-bit bloklar 399 MHz | **Bizim** ölçüm |
| Öğrenilen sınıfta temsil | Doğru A5 çözümü, eğitilen mimarinin içine yazıldığında %100 (yalnız head eğitildi) | **Bizim** |
| A5'in basit grup genellemesi | Yörünge argümanı: her basit grup için blok boyutuna göre bir alt sınır | Eleştirel okumadan; genelleme ve ispat bizde |
| Bilgi ve kayıp | L_min = H(Y \| Z, W); fazlalık I(Y; H \| Z, W) | **Bilinen** (standart) |
| İfade gücü | Boolean blok sıralaması: set/keep → star-free; + toggle → Z₂; 2 bit → Z₃; 3 bit → A5 | Sürekli karşılığı bilinen (Sarrof 2024, Grazzi 2025); Boolean ölçümü bizim |
| Öğrenilebilirlik | Soft–hard farkı 0,185 bpb. Hard-forward eğitim bunu ~0,02'ye indirdi. A5'te seed'e bağlı kararsızlık var. | Ölçüm bizim; yöntemler literatürden |
| Kararlılık | Hata rekürsiyonu sınırı; 𝔽₂ tam eşitliği; birleşen map'ler farkı siler, permütasyonlar korur; input-gated yapıda hata döngüde büyümez; sızdıran bellek hipotezi | Matematiği standart; input-gated önermesi ve ölçümler bizim |

---

## 1. Biçimsel model

**Tanım 1 (K-LUT recurrent devre).** Bir devrenin parçaları:

- State: s ∈ {0,1}^S, register'larda.
- Her adımda bir feedforward devre, girdiden ve alt katmanların state'inden bir **kontrol kelimesi** c_t üretir. Feedforward devre istenildiği kadar derin olabilir ve pipeline'lanır.
- Yeni state: s_{t+1} = F(s_t, c_t). F'nin her çıkış biti en fazla K girişli tek bir LUT'tur. LUT'un girdileri, bir D_j ⊆ {1..S} kümesindeki state bitleri ile w_j kontrol bitidir; |D_j| + w_j ≤ K.

Bu durumda döngüdeki mantık derinliği **1 LUT seviyesidir**. Genel durumda F, d seviyeli bir LUT ağıdır; döngü derinliği d olur.

**Tanım 2 (block-local, dense).** State bitleri bloklara ayrılır ve her bitin bağımlılık kümesi kendi bloğunun içinde kalır (D_j ⊆ B(j)). **Dense** ise D_j = B(j), yani her bit bloğunun tamamını okur. Faz 0'daki B modelleri (kb = 1, 2, 3) dense block-local'dir.

**Tanım 3 (gerçekleştirme).** Bir M = (Q, Σ, δ, q₀) otomatının gerçekleştirmesi:

- injective bir encoding e : Q → {0,1}^S;
- her a ∈ Σ için F(e(q), c(a)) = e(δ(q, a)) koşulunu sağlayan bir devre.

Maliyet dörtlüsü (S, D, L, R):

- S: state bit sayısı;
- D: döngü derinliği;
- L: LUT sayısı;
- R: routing (ölçümle).

ℛ_K(M), fan-in K ile ulaşılabilir maliyet dörtlülerinin kümesidir; Pareto-optimal elemanları araştırıyoruz.

---

## 2. Zamanlama: Önerme A

**A1 (ayrışma).** Bileşenler arası bağlantı grafiği çevrimsizse bütün yönlü çevrimler bileşenlerin içindedir. Dolayısıyla global iteration bound

T_IB = max_C (Σ_{v∈C} d(v)) / (Σ_{e∈C} w(e))

bileşenlerin kendi bound'larının maksimumudur. *İspat:* Çevrimsiz bir grafikte bileşenler arası hiçbir kenar bir çevrimin üzerinde olamaz.

**A2 (ulaşılabilirlik).** Her düğümü tek LUT olan ve bileşenler arası yolları ileri yönlü olan sınıfımızda bu sınıra ulaşılabilir. Leiserson–Saxe'nin Theorem 11'i, unit-delay bir devrede Φ ≤ c yapan bir retiming'in var olmasını, G − 1/c grafiğinde negatif ağırlıklı çevrim olmamasına bağlar. Bizde her çevrim tek LUT ve tek register olduğu için c = 1 sağlanır.

Tezde açıkça yazılması gerekenler:

1. Bileşenler arasına register eklemek retiming değil pipelining'dir. Latency artar, üst katman girişi aynı miktarda geciktirilir (skew).
2. Retiming başlangıç state'ini değiştirebilir (Touati–Brayton 1993).
3. Bağımsız dizilerde c-slow (Leiserson–Saxe Sec. 7), derin bir döngünün throughput'unu geri kazanır.
4. Mantıksal sınır ile post-route clock period aynı şey değildir. Routing gecikmesi ölçümle verilir (§5.3).

**İki önemli ayrım** (eleştirel bir okumadan, 28 Ekim):

- **Iteration bound bir alt sınırdır; her devrede ulaşılamaz.** Bölünemeyen düğümlerde fark açılabilir. Örnek: bir çevrimde birim gecikmeli üç düğüm ve iki register. Alt sınır 3/2, ama bir aralıkta iki düğüm bulunmak zorunda olduğundan en iyi değer 2. Bizim sınıfta her çevrim 1 LUT ve 1 register olduğundan alt sınıra ulaşılır.
- **Akış işleme hızı, üretim hızı değildir.** "Döngüde tek mux" iddiası girdinin baştan bilindiği durumlar içindir: puanlama ve sıkıştırmada encoder. Metin üretiminde ya da FPGA'de decode'da bir sonraki girdi head'in ve örneklemenin çıktısına bağlıdır; head de döngüye girer. Sıkıştırma case'inde encoder FPGA'de, decoder yerde olduğu için iddia encoder için geçerlidir.

**Konum:** Bilinen sonuçların bir corollary'si; atıflı bir lemma olarak sunulur.

---

## 3. Grup yapıları: Önerme B

m üreteçli G ≤ Sₙ, artı identity harfi (m+1 harf, w = ⌈log₂(m+1)⌉ kontrol biti):

| Encoding | State | Bit başına fan-in | Yerellik |
|---|---|---|---|
| Positional, base ile (bir base β₁..β_k'nin görüntüleri) | k·⌈log₂n⌉ | ⌈log₂n⌉ + w (sola çarpmada) | Koordinat başına yerel |
| One-hot (bit (i,j) = [g(i)=j]) | n² | (m+1) + w | Her iki çarpma yönünde yerel |
| Sağa çarpma g∘σ, positional | k·⌈log₂n⌉ | m koordinat arasında mux | Yerel değil |

Bütün sonlu basit gruplar 2 üreteçlidir: Steinberg 1962, Aschbacher–Guralnick 1984, Miller; CFSG'ye dayanır. Bu yüzden one-hot encoding ile fan-in 3 + 2 = 5 her basit grup için yeter.

**Önemli uyarı:** Bu sonuç yalnız her adımda alfabe {e, a, b} ise geçerlidir. Bir Krohn–Rhodes cascade'inde grup bileşeninin etkin girdisi (harf, alt state) çiftidir ve G'nin herhangi bir elemanı olabilir; bu durumda fan-in n ile büyür.

**Konum:** Parçaları bilinen (one-hot encoding, base); K-LUT diliyle yazılmış biçimsel bir ifadesi bulunamadı.

---

## 4. Bilgi alt sınırı

**Önerme (Myhill–Nerode).** Grup elemanını izleyen deterministik bir makine için S ≥ ⌈log₂|G|⌉.

**Önerme (bilgi ve kayıp).** Y bir sonraki byte, H geçmiş, Z = f(H) saklanan state, W head'in gördüğü pencere olsun. Sınırsız kapasiteli ve en iyi ayarlanmış bir head için en küçük kayıp L_min = H(Y | Z, W), bpb cinsinden. Geçmişin tamamını kullanan bir modele göre fazlalık:

H(Y | Z, W) − H(Y | H) = I(Y; H | Z, W)

Yani bellek, geçmişin tamamını değil, geleceği tahmin etmeye yarayan kısmını taşımak zorundadır. *İspat:* E[−log₂ q] = H(Y|Z,W) + E[KL(P ‖ q)] ≥ H(Y|Z,W); fark ise koşullu karşılıklı bilginin tanımıdır.

**Sonuç.** d Boolean bit en fazla 2^d durum tutar. Sürekli bir relaxation aynı d koordinatta çok daha fazla ayırt edilebilir değer kullanabilir. Bu yüzden soft modelin başarısı, aynı d ile hard bir gerçekleştirmenin var olduğunu göstermez.

*İspat:* İki girdi dizisi farklı grup elemanlarına (g ≠ h) götürüyorsa, g⁻¹'e götüren bir sonek ilkini identity'ye götürür, ikincisini götürmez. Bu yüzden iki dizi aynı state'e düşemez; state sayısı en az |G| olmalıdır. A5 için: **6 bit.**

---

## 5. Gerçekleştirilebilirlik: A5 küçük vakası (bizim sonuçlarımız)

Görev: pₜ = pₜ₋₁ ∘ aₜ, aₜ ∈ {(01234), (012), e}, w = 2. Bu, `fbseq.py`'deki A5 görevi.

### 5.1 Block-local yapı ve alt gruplar

**Önerme C1.** Dense block-local bir gerçekleştirmede her blok, makinenin substitution-property (SP) partition'ına karşılık gelen bir bölüm makinesidir. Sağdan çarpan grup makinesinde SP partition'lar tam olarak alt grupların sağ coset'leridir (H p). Bütün state'in injective olması ∩ᵢ Hᵢ = {1} koşuluna denktir.

*İspat:*

1. **SP partition → alt grup.** Bir blok yalnız kendi bitlerine baktığı için, bloğun ayırt ettiği state'ler partition'ı SP özelliğini sağlar: p ≡ q ⇒ p∘a ≡ q∘a. Σ, G'yi ürettiğinden bu her g ∈ G için p∘g ≡ q∘g demektir. H = [1] sınıfı çarpmaya göre kapalıdır, yani bir alt gruptur. p ≡ q ⇔ p q⁻¹ ≡ 1 ⇔ Hp = Hq.
2. **Injectivity.** p ile q bütün bloklarda aynı ⇔ p q⁻¹ ∈ ∩ᵢHᵢ.

SP partition kavramı Hartmanis–Stearns'e (1961–1966) aittir; grup makinelerindeki karşılığı klasiktir. Bizim katkımız, bunun LUT fan-in kısıtıyla birleştirilmesi.

**Sonuç.** Dense block-local gerçekleştirmenin en küçük state'i:

β_K(G) = min Σᵢ ⌈log₂[G : Hᵢ]⌉, koşullar ∩Hᵢ = {1} ve ⌈log₂[G : Hᵢ]⌉ + w ≤ K.

### 5.2 A5 için değerler

A5'in 59 alt grubu tarandı (`theory_sat.subgroups`). Mertebe dağılımı 1, 2 (15), 3 (10), 4 (5), 5 (6), 6 (10), 10 (6), 12 (5), 60.

- **K = 4:** Blok en fazla 2 bit olabilir, yani indeks ≤ 4. A5'in indeksi ≤ 4 olan öz alt grubu yok. **İmkânsız.**
- **K = 5:** Blok en fazla 3 bit, yani indeks ≤ 8. Adaylar yalnızca A4 (indeks 5) ve D10 (indeks 6).
  - İki farklı A4'ün kesişimi Z₃'tür (iki nokta sabit).
  - A4 ∩ D10 = Z₂: D10 beş nokta üzerinde transitive, bu yüzden bir noktanın stabilizer'ı mertebe 2.
  - İki D10'un kesişimi her zaman tek bir involution: 15 involution'ın her biri tam iki D10'da bulunuyor.

  İki blok yetmiyor, üç blok yetiyor. **β₅ = 9** (örnek: üç D10).

- **K = 6:** Blok en fazla 4 bit, yani indeks ≤ 16. A4 ∩ V4 = {1} olan bir çift var; örneğin A4 ile mertebeleri aralarında asal olan Z5 de aynı işi görür. **β₆ = 7 = 3 + 4.** Daha azı mümkün değil: 3+3 bileşimleri yukarıdaki kesişimler yüzünden tutmuyor; indeksi ≤ 4 olan alt grup olmadığından 2+4 de yok.

Doğrulama: Üç çözüm de rastgele dizilerde %100 doğru (`theory_sat.check_family`: 500 dizi × 512 adım).

**Yörünge argümanı ve basit gruplar için genelleme** (eleştirel okumadan; tarama ispatını genelleştirir). G basit, blok b bitlik olsun ve yalnız kendi state'ini ve girdiyi görsün. G'nin bloğun başlangıç durumunun yörüngesi üzerindeki etkisi ya triviyaldir ya da sadıktır, çünkü çekirdek G'nin bir normal alt grubudur. Yörünge boyutu bir alt grubun indeksidir; dolayısıyla |G|'yi böler ve triviyal değilse G'nin en küçük öz alt grup indeksinden μ(G) küçük olamaz.

Bundan bir alt sınır çıkar. Blokların yörünge boyutları o₁, …, o_r, her biri 2^{bᵢ}'den küçük ya da eşit, |G|'yi bölen ve 1 ya da ≥ μ(G) olan sayılardır. State'in G'nin elemanlarını ayırabilmesi için ∏ oᵢ ≥ |G| gerekir. Bu, en küçük Σ bᵢ için genel bir alt sınırdır.

A5 için (μ = 5):

- 2 bitlik bloklar triviyaldir.
- 3 bitlik bloklarda yörünge 5 ya da 6 olabilir. İki blok en fazla 6 · 6 = 36 < 60 ayırt eder, yani en az 9 bit gerekir.
- 4 bitlik bloklarda yörünge en fazla 15 olabilir. 4 + 2 bit yetmez, 3 + 3 bit 36 verir; yani en az 7 bit gerekir.

Bu, §5.2'deki taramayı doğrular ve aynı akıl yürütme her basit grup için çalışır.

### 5.3 Block-local olmayan: sınır sıkı
SAT/SMT sorgusu (`theory_sat.py`, z3, 6 344 s): S = 6 ve K = 6 için, her bitin en fazla 4 state bitine ve 2 kontrol bitine bağlı olduğu injective bir encoding **var**. Bağımsız doğrulama:

- injectivity ve bütün bit/harf koşulları sağlanıyor;
- rastgele dizilerde %100 doğru (300 dizi × 512).

Alt sınır da 6 olduğundan **A5 için tek seviyeli, fan-in 6 gerçekleştirmenin en küçük state'i tam olarak 6 bittir.** Block-local kısıt bu değeri 7'ye çıkarıyor.

Bir notu açıkça yazmak gerekir: genel bir 6 bitlik kodda bir sonraki state biti 6 state bitine ve kontrol bitlerine bağlı olabilir. Bu da tek bir LUT6'ya sığmayabilir. SAT tanığı tam bu engeli aşan bir encoding buluyor: her bit yalnız 4 state bitine ve 2 kontrol bitine bağlı. Yani 6 bitlik minimum **fan-in 6 ve tek seviye kısıtı altında da** geçerli. Block-local kısıtın A5'teki bedeli tam 1 bit.

### 5.4 Fiziksel karşılık
XC7A100T üzerinde, 120 bitlik state'e kopyalanarak ölçüldü (Yosys + nextpnr-himbaechel, iki seed):

| Gerçekleştirme | State | Döngü | Fmax (MHz) |
|---|---|---|---|
| Öğrenilen mimari, tam tablolu 3-bit bloklar (8:1 mux) | 24 | 2 seviye | 378 / 421 |
| Elle, noktasal 5 × 3 bit | 15 | LUT5 | 424 / 443 |
| β₆: V4 + A4 coset'leri | 7 | LUT6 / LUT5 | 443 / 473 |
| SAT, 6 bit | **6** | LUT6 | 454 / 456 |

Bundan çıkan sonuç: Daha kompakt encoding, döngü derinliği ve Fmax açısından bir bedel ödetmedi. Bu küçük vakada maliyet Pareto'sunun ucunda 6 bit, tek seviye ve ~455 MHz var. Tam tablolu öğrenilen mimari ise hem daha fazla bit hem daha uzun döngü kullanıyor: gereğinden esnek. **Transition codebook** fikrinin dayanağı bu.

**Öğrenilen sınıfın içinde (`probe_a5embed.json`).** Doğru çözüm, eğitilen modelle birebir aynı mimarinin içine yazıldı: 3-bit bloklu `fbseq.Gated`, gate MLP'leri ile. Gate ağlarına tam tablolar yerleştirildi; üç blok r(j) = p⁻¹(j) değerlerini tutuyor (24 bitin 9'u). Sonuç:

- state'ten doğrudan çözülen grup elemanı: %100;
- yalnız head eğitildiğinde, 64 ve 256 uzunlukta: %100.

Yani bu sınıf A5'i temsil edebiliyor; eğitimde görülen başarısızlık bir öğrenme sorunu.

### 5.5 Açık sorular
1. K = 5 ve K = 4'te block-local olmayan en küçük state (SAT).
2. Genel bir G için β_K ile block-local olmayan minimum arasındaki ilişki. Grup teorisinde yakın kavramlar var (minimal faithful degree, base size); tam olarak bu ölçü bulunamadı.
3. Cascade'lerde etkin alfabe (§3 uyarısı) ve injective olmayan encoding'ler (state splitting).
4. Encoding bulmanın karmaşıklığı. Sabit bir netlist'te en küçük döngü derinliği polinom zamanda bulunuyor (Pan–Liu); zorluk tamamen encoding seçiminde.
5. Derby'nin (2001) CRC/LFSR için state-space dönüşümüyle döngüyü kısaltmasının doğrusal olmayan genellemesi bulunamadı. Teorem C için doğal bir şablon.
6. Shende vd. (2003) Lemma 11: tam 2^S kodu kullanan ve her adımda en az bir biti hiç değiştirmeyen devreler yalnız çift permütasyon üretir. Tek permütasyonlu otomatlar için boş bir kod ya da ek bir bit gerekir.

---

## 6. İfade gücü sınıfları

Ölçtüğümüz Boolean sıralama (64'te eğitim, 256'da test, 3 seed):

| Model | flag | parity | mod 3 | A5 |
|---|---|---|---|---|
| set/keep | 1,00 | 0,51 | 0,34 | 0,03 |
| + toggle | 1,00 | **1,00** | 0,34 | 0,03 |
| 2-bit blok | 1,00 | 1,00 | **1,00** | 0,05 |
| 3-bit blok | 1,00 | 1,00 | 1,00 | **0,32** |

**Bilinen karşılıklar:**

- Sarrof, Veitsman, Hahn (NeurIPS 2024): nonnegative SSM ⇔ star-free.
- Grazzi vd. (ICLR 2025): parity için negatif eigenvalue; 2'nin kuvveti olmayan m ile mod m için non-real eigenvalue.
- Krohn–Rhodes, Schützenberger, McNaughton–Papert: star-free ⇔ aperiodic.
- Barrington (1989): non-solvable grupların word problem'i NC¹-complete.
- Knorozova & Ronca (AAAI 2024): cascade çerçevesi.

Sıralamanın sürekli halinin teorisi yani zaten var; bizim katkımız Boolean blok karşılığı ve ölçümü.

**Sanı (literatür ajanının Krohn–Rhodes'tan türettiği bir taslak, bizim tarafımızdan kontrol edilmeli):** Girdi kontrollü k-bitlik bloklarla tanınabilen düzenli diller tam olarak şunlardır: sözdizimsel monoidin her basit grup böleni S_{2^k}'yi böler.

Test edilebilir öngörüleri:

- mod 5 ve mod 7, 2-bit bloklarla hiçbir derinlikte çözülemez, tek bir 3-bit blokla çözülür;
- mod 6 ve S₄, 2-bit bloklarla çözülür;
- Z₁₁ 4-bit blok ister;
- Z_{2^j}, 1-bit bloklarla j katman ister.

---

## 7. Öğrenilebilirlik

**Ölçülen fark (Loghub, 256 B paket, set/keep):**

- soft val 1,398;
- uyarlamasız hard 2,769;
- soft→hard uyarlanmış 1,583.

Δ_hard = 0,185 bpb. Soft model state'leri sürekli kullanıyor: ortalama min(s, 1−s) değeri 0,16–0,21. Yani soft model ile Boolean makine aynı fonksiyon sınıfında değil.

**Hard-forward eğitim** (Loghub, set/keep, 2 seed):

- baştan Boolean forward, backward'da sigmoid(a·l), a 1'den 5'e (eğim tavlaması): **1,418 bpb**. Pencere 8 modelini (1,453) geçiyor, soft modele 0,02 kalıyor.
- sade Hard-ST (a = 1): 1,545.
- Gumbel-ST (τ 1 → 0,25): 1,685. Kim'in (2026, arXiv 2603.14157) düşük sıcaklıkta Gumbel-ST çöküşü bulgusuyla uyumlu.
- toggle'lı model: eğim tavlamasıyla 1,462.

**Not (Gumbel).** 1[(l+n)/τ > 0] = 1[l+n > 0]. τ'yu düşürmek forward'daki gürültüyü azaltmaz, yalnız backward surrogate'ini değiştirir. Uygulamamız standart Gumbel-ST: logistic gürültü, yani iki Gumbel'in farkı; forward'da sabit gürültü. Gürültünün kendisini sönümleyen bir varyant denenmedi.

**A5 trainability** (3-bit blok, 4 seed, kararlı relaxation):

- Elle kurulan devre %100 (representability).
- soft→hard: seed'lerin yarısı %100'e ve tam uzunluk genellemesine ulaşıyor. Diğer ikisi 1/3 civarındaki coset aşamasında takılıyor (Zhang & Li 2026); birinde model 0,25'i bulduktan sonra 0,09'a geriliyor.
- Baştan hard (Hard-ST ve eğim tavlaması): 8 koşunun hiçbiri 0,13'ü geçemiyor.
- **Keşif ile uyum arasında bir ödünleşim var:** dil modelinde baştan Boolean eğitim en iyisi, durum izlemede ise soft aşama çözümü keşfetmek için şart.

**Transition codebook: olumsuz sonuç** (`codebook_*.json`). Her 3-bitlik blok, m = 8 öğrenilmiş dönüşümden birini seçiyor. Rastgele başlangıçla:

- A5: softhard ve eğim tavlaması, 8 koşunun hiçbiri 0,06'yı geçmedi.
- Loghub: 2,07–2,16 bpb.

Teori, A5'in tek seviyede yalnız 3 dönüşümle temsil edilebildiğini söylüyor. Ama dönüşümleri ve seçimi birlikte öğrenmek bu haliyle çalışmadı. Identity'ye yakın başlangıçla tekrar sürüyor.

**Literatürdeki reçeteler:**

- PD-SSM (Terzić vd., NeurIPS 2025): one-hot geçiş matrisleri, forward hardmax, backward softmax; A5 tek katman %100.
- Mind the Gap (NeurIPS 2025, feedforward): Gumbel + ST.
- Kim 2026: Hard-ST ve CAGE.
- HM-RNN (ICLR 2017): hard ST ve eğim tavlaması; hard model soft modeli geçmiş.
- R-DTLGN (2026): recurrent modellerde hatanın döngüde büyümesi, trajectory distillation.

---

## 8. Kararlılık

**Hata rekürsiyonu (set/keep, soft ve hard karşılaştırması).** eₜ = |sₜ − ŝₜ| için:

eₜ ≤ (1−gₜ)·eₜ₋₁ + gₜ·|uₜ − ûₜ| + |gₜ − ĝₜ|

Toggle'lı model (sₜ = bₜ + cₜ·sₜ₋₁, cₜ = aₜ − bₜ ∈ [−1, 1]) için:

eₜ ≤ |cₜ|·eₜ₋₁ + |bₜ − b̂ₜ| + |cₜ − ĉₜ|

**Yörünge boyunca açık sınır.** aₜ = 1 − gₜ ve εₜ = gₜ|uₜ − ûₜ| + |gₜ − ĝₜ| yazılırsa:

eₜ ≤ e₀ ∏ⱼ aⱼ + Σₖ εₖ ∏_{j>k} aⱼ

Bu, ortalama almadan her yörünge için geçerlidir. Her adımda aₜ ≤ ρ < 1 ve εₜ ≤ ε ise eₜ ≤ ρᵗe₀ + ε(1−ρᵗ)/(1−ρ); ayrıca e ≤ 1.

**Önerme (input-gated yapıda hata büyümez).** Gate'ler state'e bağlıysa, |g(s) − ĝ(h)| ≤ L_g‖s−h‖ + δ gibi bir bağıntı gerekir. Toplam katsayı 1'i aşabilir ve daralma kaybolur; GRU gibi hücrelerde durum budur. B'de ise gate'ler katmanın **kendi** state'ine bağlı değildir:

- Bir katmanın içinde εₜ yalnız girdiye ve alt katmanların state'ine bağlıdır; kendi hatasından bağımsızdır. Yukarıdaki sınır ek varsayım gerektirmeden geçerlidir.
- Katmanlar arasında hata ileri yönde akar: e⁽ˡ⁾ₜ ≤ ρ·e⁽ˡ⁾ₜ₋₁ + L·e⁽ˡ⁻¹⁾ₜ + δ. Alt katman sınırlıysa üst katman da sınırlıdır.

Sonuç: **discretization hatası state döngüsü tarafından büyütülemez.** Aynı yapı paralel scan'i ve tek mux'luk döngüyü de sağlıyordu.

**Sızdıran bellek: temsil farkının olası bir kaynağı.** Soft model sₜ = r·sₜ₋₁ ile (g = 1 − r, u = 0) yavaşça unutabilir; Boolean bit ya tam tutar ya sıfırlar. Bu durumda tam çözüm eₜ = 1 − rᵗ olur. r = 0,999 için e₂₅₆ ≈ 0,23, e₄₀₉₆ ≈ 0,98. Yani adım başına çok küçük bir gate farkı, uzun bir akışta büyük bir fark yaratabilir; bu örnekte sınır tam olarak eşitliktir. Soft state'lerin 0 ve 1'den uzak olması (0,16–0,21) böyle kısmi unutmanın izi olabilir. Boolean dünyada sızdıran belleğin karşılığı **sayaçlardır**: "k adım sonra unut" diyen çok bitli bloklar. Bu bir hipotez.

**Koşullu sınır.** E[|cₜ| | 𝓕ₜ₋₁] ≤ ρ < 1 ve adım başına ek hata ≤ ε ise:

E[eₜ] ≤ ρᵗ·E[e₀] + ε(1 − ρᵗ)/(1 − ρ)

**𝔽₂ tam eşitliği (hard, tek bit).** Dört map'in hepsi 𝔽₂ üzerinde affine: sₜ₊₁ = aₜsₜ ⊕ bₜ. Aynı kontrol dizisi altında başlangıç farkı için:

sₜ ⊕ s′ₜ = (∏ᵢ aᵢ)(s₀ ⊕ s′₀)

Fark ilk reset'e kadar korunur, reset'te silinir.

**Bloklara genelleme.** Aynı girdiyle ilerleyen iki yörünge, iki durumu aynı duruma eşleyen bir adımda kalıcı olarak birleşir; permütasyon adımları farkı korur. Krohn–Rhodes diliyle:

- reset bileşenleri farkı siler;
- grup bileşenleri farkı taşır.

**Bellek ile dayanıklılık aynı mekanizmaya bağlı.** Uzun hatırlayan bitler, hataları da uzun süre taşır.

**Kayba bağlantı.** Cross-entropy'nin logit gradyanının ℓ₁ normu en fazla 2. Head'in state'e göre ℓ∞ Lipschitz sabiti L ise:

|ΔBPB| ≤ (2L / (T ln 2)) · Σₜ ‖sₜ − ŝₜ‖∞

**Ölçüm.** D(t), soft ve hard state'lerin farklı olan bit oranı: birinci katmanda ~%6, ikinci katmanda ~%11–13; t ≈ 8–16'da doyuyor (`gap.json`).

Bu bölümün matematiği standart araçlardan türetiliyor. Bizim kısmımız, ölçümler ve bunların donanımdaki döngü yapısıyla ilişkisi.

**Düzeltme: D(t)'nin doyması bellek kalitesini kanıtlamaz.** Eşik uyuşmazlığı Dₜ ile sürekli uzaklık Eₜ farklı şeyler ölçer: Eₜ ≤ Qₜ + Dₜ ve Dₜ ≤ 2Eₜ, burada Qₜ yuvarlama uzaklığı. Sabit bir uyuşmazlık oranı, eski bilginin hızla silinmesiyle de oluşabilir. Örnek: her adımda r olasılıkla iki yörünge aynı değere sıfırlansın, aksi halde p olasılıkla ayrışsınlar. Denge (1−r)p/(r+p−rp) olur; r = 0,15 ve p = 0,02 bizim eğrimize benzer bir doygunluk verir, ama başlangıç bilgisinin 256 adım hayatta kalma olasılığı ~10⁻¹⁸'dir. Bu yüzden D(t)'nin doyması yalnız "uyuşmazlık incelenen ufukta artmıyor" demektir.

**Uyarı (Lipschitz sınırı).** Head'in ölçeği kontrol edilmezse sınır işe yaramaz. Tek bit için z = (M(2s−1), −M(2s−1)) alınırsa bir bitlik fark kayıpta 2M/ln 2 bit değişime yol açar. L, eğitilmiş head için ölçülmelidir.

**Deney** (`probes.py`; sonuçları çalışıyor): Hard modelde tek bir state biti çevrilir. Farkın yaşam süresi, üst katmana yayılması ve bpb'ye etkisi ölçülür. Ayrıca bir lineer probe ile state'in k adım önceki byte'ı ne kadar taşıdığına bakılır (gecikmeli okuma).

---

## 9. Literatürdeki konum

**Bilinen ve kullandığımız:**

- retiming ve iteration bound: Leiserson–Saxe 1991; Renfors–Neuvo 1981; Parhi–Messerschmitt 1991;
- FSM look-ahead: Lin–Messerschmitt 1991;
- partition pair ve reduced dependence: Stearns–Hartmanis 1961, kitap 1966;
- Krohn–Rhodes, holonomy;
- LUT tabanlı FSM synthesis: Barkalov vd. 2021;
- ifade gücü: Sarrof 2024, Grazzi 2025, Merrill 2024, PD-SSM 2025, Liu 2023, Barrington 1989;
- discretization: Mind the Gap 2025, Kim 2026, HM-RNN 2017.

**En yakın rakipler:**

- RDDLGN (EdgeFM 2025 atölyesi): concatenation recurrence, keep yolu yok, donanım yok; future work olarak associative recurrent blokları öneriyor.
- Ovin & Liwicki (NeurIPS 2026 AXIOM atölyesi): bit başına keep/write mux, sentetik recall; donanım yok. Tam metne erişilemedi.

**Bulunamayanlar (bizim için boşluklar):**

- sentezlenip Fmax'i raporlanmış bir recurrent logic ya da LUT modeli;
- logic ya da LUT ağlarıyla byte düzeyinde bpb ve kayıpsız sıkıştırma;
- döngüdeki LUT seviyesini tasarım kısıtı olarak kullanan bir çalışma;
- Boolean transition codebook;
- adım bazında discretization gap ölçümü;
- Boolean mux monoidinin scan ile eğitimi;
- A5 gibi gruplar için LUT fan-in kısıtlı en küçük gerçekleştirme sınırları (§5).

Ayrıntılı künyeler: `docs/08_case_arastirmasi.md`, `paper/related_work_2026.md` ve Faz 0 belgesinin kaynakçası.

---

## 10. Sonraki adımlar
1. K = 5 ve K = 4 için block-local olmayan SAT; S₄, mod 5 ve mod 7 ile §6'daki sanının testi.
2. Transition codebook: öğrenilen bir modelin her bloğu, keyfi tablo yerine m dönüşümlük bir codebook'tan seçsin. Hem A5'te hem Loghub'da.
3. Hard-aware eğitimin düzeltilmiş sonuçları ve finalistlerin daha fazla seed'le tekrarı; CAGE eklenmesi.
4. Kararlılık: tek bit çevirme deneyi ve L sabitinin ölçülmesi.
5. Gate ağları ve head hâlâ float. LUT ağlarına çevrildiklerinde kalite ve döngü dışı mantığın maliyeti (Faz 1).
