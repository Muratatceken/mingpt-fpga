# 1. Teorik Temel

Bu bölümde minGPT'nin matematiksel modeli, FPGA'ya uygun **yalnızca tamsayı (integer-only)** bir
çıkarım (inference) modeline nasıl dönüştürüldüğü ve her dönüşümün neden doğru/güvenli olduğu
türetilmektedir. Buradaki her formül `sw/intmodel.py` (altın model) ve `rtl/*.sv` (donanım) içinde
**birebir aynı** şekilde uygulanmıştır.

Notasyon: $C$ = `n_embd` (48), $H$ = `n_head` (3), $d_h = C/H$ (16), $L$ = `n_layer` (3),
$V$ = sözlük boyutu (3), $T$ = `block_size` (11), $N_{ff} = 4C$ (192).

---

## 1.1 minGPT'nin matematiksel tanımı

minGPT, GPT-2 ile aynı yapıda bir *decoder-only* transformer'dır (`ref/minGPT/mingpt/model.py`).
$t$ konumundaki token $\tau_t$ için:

$$x^{(0)}_t = W_{te}[\tau_t] + W_{pe}[t]$$

Her blok $\ell = 1..L$ için (pre-LN yapı):

$$
\begin{aligned}
a_t &= \mathrm{LN}_1(x_t), \qquad [q_t, k_t, v_t] = W_{attn}\, a_t + b_{attn} \\
s_{t,j}^{(h)} &= \frac{q_t^{(h)} \cdot k_j^{(h)}}{\sqrt{d_h}}, \quad j \le t \qquad \text{(nedensel maske)} \\
y_t^{(h)} &= \sum_{j \le t} \mathrm{softmax}_j\!\left(s_{t,\cdot}^{(h)}\right) v_j^{(h)} \\
x_t &\leftarrow x_t + W_{proj}\, y_t + b_{proj} \\
x_t &\leftarrow x_t + W_{mproj}\, \mathrm{GELU}\!\left(W_{fc}\, \mathrm{LN}_2(x_t) + b_{fc}\right) + b_{mproj}
\end{aligned}
$$

Çıkış: $\;z_t = W_{head}\, \mathrm{LN}_f(x_t)$, açgözlü (greedy) kod çözmede $\tau_{t+1} = \arg\max z_t$.

$\mathrm{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$, ve minGPT'nin GELU'su
tanh yaklaşımıdır: $\mathrm{GELU}(x) = \tfrac{1}{2}x\left(1+\tanh\!\left(\sqrt{2/\pi}\,(x+0.044715x^3)\right)\right)$.

## 1.2 Otoregresif çıkarım ve KV önbelleği

**Önerme 1.** Nedensel maske altında, $j$ konumundaki tüm ara değerler (her katmandaki $x_j, k_j, v_j$)
yalnızca $\tau_0..\tau_j$'ye bağlıdır.

*İspat (katman üzerinden tümevarım).* Gömme (embedding) sadece $\tau_j$ ve $j$'ye bağlıdır.
LN, Linear, GELU ve artık (residual) toplama konum-bazlı (token başına bağımsız) işlemlerdir.
Dikkat (attention) $j$ konumunda yalnızca $i \le j$ konumlarının $k_i, v_i$ değerlerini kullanır ve
bunlar tümevarım hipotezine göre yalnızca $\tau_0..\tau_i$'ye bağlıdır. ∎

**Sonuç.** Yeni bir token eklendiğinde önceki konumların $k_j, v_j$ değerleri **değişmez**. Bu değerler
önbellekte tutulursa (KV cache), her adımda sadece **yeni token** işlenir. Böylece:

* tüm Linear katmanlar matris-matris (GEMM) yerine **matris-vektör (GEMV)** çarpımına indirgenir,
* dikkat, $t+1$ uzunluğunda iki iç çarpım dizisine indirgenir: $q\cdot k_j$ ve $\sum_j p_j v_j$.

Donanım tam olarak bu "decode adımını" uygular: `start` darbesi başına bir token işlenir.

## 1.3 İş yükü analizi (gpt-nano, sıralama görevi)

| Bileşen | Boyut | MAC / token |
|---|---|---|
| `c_attn` (QKV) | $3C \times C$ | 6 912 |
| `c_proj` | $C \times C$ | 2 304 |
| `c_fc` | $4C \times C$ | 9 216 |
| `mlp.c_proj` | $C \times 4C$ | 9 216 |
| **Blok toplamı** | | **27 648** |
| $L=3$ blok + `lm_head` ($V \times C$) | | **83 088** |
| Dikkat (katman başına) | $2C(t+1)$ | 96 … 1 056 |

Parametre sayısı ≈ 85.7 k (INT8 ile ≈ 84 KB). Dikkatin payı $t \le 10$ için küçüktür; hesap yükünün
~%97'si GEMV'dir. Mimari bu nedenle **GEMV motoru** etrafında kurulmuştur.

---

## 1.4 Kuantizasyon teorisi

### 1.4.1 Düzgün simetrik kuantizasyon
Gerçel $r$ değeri, ölçek $S>0$ ile $b$-bit tamsayıya eşlenir:

$$q = \mathrm{clip}\!\left(\mathrm{round}(r/S),\, -2^{b-1},\, 2^{b-1}-1\right), \qquad \hat r = S\,q .$$

Aralık içindeki hata $|r-\hat r| \le S/2$'dir. Hata düzgün dağılımlı kabul edilirse (Bennett modeli)
varyans $S^2/12$ olur ve her ek bit SQNR'yi ≈ 6.02 dB artırır [Nagel21].

* **Ağırlıklar:** INT8, *çıkış-kanalı başına* ölçek $S_w[o] = \max_i |W_{oi}| / 127$.
* **Aktivasyonlar:** INT8, tensör başına sabit (statik) ölçek. Ölçek, kalibrasyondan $\max|r|/127$ olarak alınır.
* **Kalibrasyon kümesi:** Sıralama görevinin **tüm** girdi uzayı (3⁶ = 729 dizi). Kalibrasyon kümesi
  girdi uzayının tamamı olduğu için çıkarım sırasında aralık dışı (clipping) değer **oluşamaz**.

### 1.4.2 Yalnızca tamsayı GEMV ve yeniden kuantizasyon (requantization)
$y_o = \sum_i W_{oi} x_i + b_o$ için $W_{oi} \approx S_w[o]\, q^w_{oi}$, $x_i \approx S_x q^x_i$ yazılırsa:

$$y_o \approx S_w[o]\,S_x \underbrace{\Big(\sum_i q^w_{oi} q^x_i + q^b_o\Big)}_{acc_o\ \in\ \mathbb{Z}},\qquad q^b_o = \mathrm{round}\!\left(\frac{b_o}{S_w[o] S_x}\right)$$

Çıkışın ölçeği $S_y$ ise $q^y_o = \mathrm{round}(M_o \cdot acc_o)$ olur, burada $M_o = S_w[o]S_x/S_y$.
$M_o$ gerçel sayısı donanımda **tamsayı çarpan ve kaydırma** olarak temsil edilir [Jacob18]:

$$M_o \approx m_0 \cdot 2^{-sh}, \quad m_0 \in [2^{15}, 2^{16}) \;\Rightarrow\; \text{bağıl hata} \le 2^{-16}$$

$$q^y_o = \mathrm{sat}\!\left(\left(acc_o \cdot m_0 + 2^{sh-1}\right) \gg sh\right) = \mathrm{sat}\!\left(\left\lfloor acc_o M_o + \tfrac12 \right\rfloor\right)$$

(aritmetik sağa kaydırma = taban (floor) bölme, $+2^{sh-1}$ = yarıyı yukarı yuvarlama).
Bu işlem `rtl/requant.sv`'de 2 çevrimlik bir boru hattıdır. $(b, m_0, sh)$ kanal başına parametre
ROM'unda saklanır.

**Akümülatör genişliği.** $|acc| \le N_{in}\cdot 127 \cdot 128 + |q^b| \approx 3.1\cdot10^6 < 2^{22}$
($N_{in}=192$). Dolayısıyla INT32 akümülatör taşmaz.

### 1.4.3 Artık akış (residual stream)
Artık akış **INT16**, ölçeği **2'nin kuvveti** olacak şekilde seçilir: $S_r = 2^{-FR}$,
$FR = \lfloor \log_2(32767/\max|x|) \rfloor$. Kalibrasyonda $\max|x| = 5.94$ çıktığı için $FR=12$,
yani format **Q3.12** olur. Ölçek 2'nin kuvveti olduğundan gömme toplamı ve artık toplama ölçek
dönüşümü gerektirmez; bunlar doğrudan (doymalı) tamsayı toplamadır. `c_proj` ve `mlp.c_proj` çıkışları
doğrudan $S_r$ ölçeğine requantize edilir.

---

## 1.5 Matematiksel olarak eşdeğer dönüşümler (folding)

**LayerNorm afin katlama.** LN'den sonra her zaman bir Linear gelir:

$$W(\gamma \odot \hat z + \beta) + b = \underbrace{(W\,\mathrm{diag}(\gamma))}_{W'}\hat z + \underbrace{(W\beta + b)}_{b'}$$

Böylece donanımdaki LN biriminin yalnızca $\hat z = (x-\mu)/\sqrt{\sigma^2+\epsilon}$ hesaplaması yeterli
olur ve $\gamma, \beta$ için bellek ve çarpıcı gerekmez. `ln_f` de `lm_head`'e katlanır (bu durumda
`lm_head` bir bias kazanır).

**Dikkat ölçeği katlama.** $\frac{q\cdot k}{\sqrt{d_h}} = \left(\frac{q}{\sqrt{d_h}}\right)\cdot k$ olduğundan
$1/\sqrt{d_h} = 1/4$ çarpanı $W_q, b_q$ satırlarına katlanır.

Katlanmış float64 model, orijinal minGPT ile 729 dizinin tamamında $10^{-4}$ toleransla aynı
logitleri üretir (`sw/quantize.py` bunu otomatik olarak doğrular).

---

## 1.6 Tamsayı LayerNorm

Girdi INT16 ($x_i$, ölçek $S_r$). Bölme ve kayan nokta kullanmadan:

$$
\begin{aligned}
s &= \textstyle\sum_i x_i, \qquad d_i = N x_i - s = N (x_i - \mu) \quad\text{(tam)}\\
Q &= \textstyle\sum_i d_i^2 + E, \qquad E = \mathrm{round}(\epsilon N^3 / S_r^2) \;\Rightarrow\; Q = N^3(\sigma^2+\epsilon)/S_r^2 \\
rt &= \lfloor \sqrt{Q} \rfloor, \qquad inv = \lfloor K / rt \rfloor, \quad K = \mathrm{round}(\sqrt N\, 2^{P}) \\
z_i &= \mathrm{sat}_8\!\left((d_i \cdot inv + 2^{P-ZF-1}) \gg (P - ZF)\right)
\end{aligned}
$$

Burada $\hat z_i = \sqrt N\, d_i / \sqrt Q$ olduğu kolayca gösterilebilir. Çıkış formatı INT8
**Q3.4**'tür ($ZF = 4$, ölçek $2^{-4}$), $P = 36$. Donanımda bir ardışık karekök (`isqrt_seq`, 32 çevrim) ve
bir ardışık bölme (`div_seq`, 48 çevrim) kullanılır. Bunlar her LN için yalnızca **bir kez** çalışır.

**Önerme 2 (Samuelson eşitsizliği [Samuelson68]).** $N$ elemanlı her vektör için
$|x_i - \mu| \le \sigma\sqrt{N-1}$.

*İspat.* $\sum_j (x_j-\mu) = 0$ olduğundan $(x_i-\mu) = -\sum_{j\ne i}(x_j-\mu)$ yazılabilir.
Cauchy–Schwarz ile $(x_i-\mu)^2 \le (N-1)\sum_{j\ne i}(x_j-\mu)^2 = (N-1)\left(N\sigma^2 - (x_i-\mu)^2\right)$
elde edilir. Buradan $N(x_i-\mu)^2 \le N(N-1)\sigma^2$ çıkar. ∎

**Sonuç.** $|\hat z_i| \le \sqrt{N-1} = \sqrt{47} \approx 6.86 < 7.94$ olur ($\epsilon > 0$ bu sınırı daha da
küçültür). Yani Q3.4 formatı LN çıkışında **asla doymaz**. Bu sınır veriden bağımsız, matematiksel bir
garantidir ve kalibrasyon gerektirmez.

**Önerme 3 (taşma yok).** $|d_i| \le \sqrt{\sum_j d_j^2} \le \sqrt Q$'dur ve $d_i$ tamsayı olduğundan
$|d_i| \le \lfloor\sqrt Q\rfloor = rt$ sağlanır. Dolayısıyla $|d_i \cdot inv| \le rt \cdot K/rt = K < 2^{39}$
olur: 64-bit çarpım asla taşmaz.

**Hassasiyet.** $rt \ge \sqrt E \approx 4300$ olduğundan $inv$'nin bağıl hatası $< 3\cdot10^{-4}$'tür.
Baskın hata, çıkıştaki ±½ LSB (±1/32) yuvarlamadır.

---

## 1.7 Tamsayı Softmax

**(i) Kaydırma değişmezliği:** $\mathrm{softmax}(s) = \mathrm{softmax}(s - m)$. $m = \max_j s_j$
seçilirse bütün üsler $\le 0$ olur ve $e_j \in (0, 1]$ aralığında kalır (taşma yok).

**(ii) 2 tabanına geçiş ve aralık indirgeme:** $e^{x} = 2^{x\log_2 e}$. Skorların ölçeği
$S_s = S_q S_k$ olduğundan:

$$u_j = (m - s_j)\cdot C_{exp}, \qquad C_{exp} = \mathrm{round}\!\left(S_q S_k \log_2 e \cdot 2^{F}\right)$$

$u_j = n_j 2^F + f_j$ ayrışımıyla $e_j = 2^{-u_j/2^F} = 2^{-n_j}\cdot 2^{-f_j/2^F}$ yazılır.
$2^{-n_j}$ bir **sağa kaydırmadır**. $2^{-f}\in(\tfrac12, 1]$ ise $f$'nin en üst 8 bitiyle adreslenen
256 girişlik bir **LUT**'tan okunur (UQ1.15). $C_{exp}$ de requant çarpanları gibi
$[2^{15}, 2^{16})$ aralığına normalize edilir ve $F$ katman başına seçilir.

**Hata analizi.** $f$'nin 8 bite kesilmesi üste $< 2^{-8}$ hata ekler. Bu da $e_j$'de
$< 2^{2^{-8}}-1 \approx \%0.27$ bağıl hataya karşılık gelir. $n_j > 15$ için $e_j = 0$ alınır; atılan terim
en büyük terimin $2^{-16}$'sından küçüktür.

**(iii) Ertelenmiş normalizasyon:** Her olasılık ayrı ayrı bölünmez:

$$y_d = \frac{\sum_j e_j v_{j,d}}{\sum_j e_j} = \frac{A_d}{E}, \qquad R = \left\lfloor \frac{2^{32}}{E} \right\rfloor, \quad y_d = \mathrm{sat}_8\!\left((A_d R + 2^{31}) \gg 32\right)$$

Böylece **head başına tek bölme** yapılır. En büyük eleman için $u=0$ ve $e = 2^{15}$ olduğundan
$E \ge 2^{15}$'tir; sıfıra bölme olmaz.

**Önerme 4 (dışbükeylik).** $p_j \ge 0$ ve $\sum p_j = 1$ olduğundan
$\min_j v_{j,d} \le y_d \le \max_j v_{j,d}$ sağlanır. Bu nedenle dikkat çıkışı için $S_y = S_v$ seçilir ve
çıkış **asla doymaz**. Tamsayı tarafta da $R$ taban alındığı için $|A_d R / 2^{32}| \le |A_d|/E \le 127$ olur.

Bu yaklaşım I-BERT'in [Kim21] polinom tabanlı `i-exp` fonksiyonuna bir alternatiftir. FPGA'da küçük
LUT-ROM ucuz olduğu için LUT tercih edilmiştir.

---

## 1.8 GELU: tam tablo

`c_fc` çıkışı INT8'e ($S_h$) requantize edildiği için GELU'nun girişi yalnızca **256 farklı** değer
alabilir. Bu yüzden fonksiyon bir tabloyla **tam olarak** temsil edilir:

$$\mathrm{GELU\_LUT}_\ell[q] = \mathrm{round}\!\left(\mathrm{GELU}(q\,S_h)/S_g\right), \quad q \in [-128, 127]$$

Tabloda yaklaşım hatası yoktur. Hata yalnızca giriş ve çıkış kuantizasyonundan gelir. Her katman
için 256 bayt tutulur (toplam 768 B, LUT-ROM).

## 1.9 Argmax

Softmax monoton olduğu için $\arg\max\mathrm{softmax}(z) = \arg\max z$'dir. Açgözlü üretimde
softmax'a gerek yoktur. `lm_head` çıkışları ortak bir ölçeğe ($S_{logit}$, INT16) requantize edildiği
için tamsayı karşılaştırma doğrudur. Eşitlik durumunda en küçük indeks seçilir; altın model de aynı
kuralı uygular.

---

## 1.10 Sayısal sonuçlar

`sw/evaluate.py` çıktısı (729 girdinin tamamı, açgözlü üretim):

| Model | Görev doğruluğu | Float ile token uyumu |
|---|---|---|
| minGPT (float32) | %100.00 | – |
| Tamsayı model (INT8/INT16) | %100.00 | %100.00 |
| RTL (Verilator) | %100.00 | **bit-exact** (8019/8019 adım) |

Zorla öğretimli (teacher-forced) logit hatası: max $|e| = 0.35$, RMS $= 0.046$,
**logit SQNR = 41.5 dB**.

Kuantizasyon formatlarının özeti:

| Tensör | Format | Ölçek | Seçim gerekçesi |
|---|---|---|---|
| Ağırlıklar | INT8 | kanal başına | Jacob18 |
| Bias | INT32 | $S_w S_x$ | akümülatörle aynı ölçek |
| Artık akış | INT16 | $2^{-12}$ (Q3.12) | 2'nin kuvveti → bedava toplama |
| LN çıkışı | INT8 | $2^{-4}$ (Q3.4) | Samuelson sınırı → doyma yok |
| q, k, v | INT8 | tensör başına | kalibrasyon |
| Dikkat çıkışı | INT8 | $S_v$ | dışbükeylik → doyma yok |
| GELU giriş/çıkış | INT8 | katman başına | kalibrasyon, tam LUT |
| Logitler | INT16 | ortak | argmax için ortak ölçek |
| $e_j$ (softmax) | UQ1.15 | $2^{-15}$ | $e_{max} = 1$ |

---

## Kaynaklar

* [Vaswani17] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
* [Radford19] A. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI, 2019.
* [Karpathy] A. Karpathy, *minGPT*, https://github.com/karpathy/minGPT
* [Jacob18] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," *CVPR*, 2018.
* [Kim21] S. Kim et al., "I-BERT: Integer-only BERT Quantization," *ICML*, 2021.
* [Nagel21] M. Nagel et al., "A White Paper on Neural Network Quantization," arXiv:2106.08295, 2021.
* [Ba16] J. L. Ba, J. R. Kiros, G. E. Hinton, "Layer Normalization," arXiv:1607.06450, 2016.
* [Hendrycks16] D. Hendrycks, K. Gimpel, "Gaussian Error Linear Units (GELUs)," arXiv:1606.08415, 2016.
* [Samuelson68] P. A. Samuelson, "How Deviant Can You Be?," *J. American Statistical Association*, 63(324), 1968.
* [Milakov18] M. Milakov, N. Gimelshein, "Online normalizer calculation for softmax," arXiv:1805.02867, 2018.
* [Pope23] R. Pope et al., "Efficiently Scaling Transformer Inference," *MLSys*, 2023. (KV önbelleği, decode maliyeti)
* [Sze17] V. Sze et al., "Efficient Processing of Deep Neural Networks: A Tutorial and Survey," *Proc. IEEE*, 2017. (dataflow sınıflandırması)
* [Li20] B. Li et al., "FTRANS: Energy-Efficient Acceleration of Transformers using FPGA," *ISLPED*, 2020.
* [Ham20] T. J. Ham et al., "A³: Accelerating Attention Mechanisms in Neural Networks with Approximation," *HPCA*, 2020.
