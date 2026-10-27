# İlgili çalışmalar taraması (Ekim 2026)

Bütün kaynaklar arXiv, bildiri, DOI ya da GitHub sayfasından doğrulandı; doğrulanamayanlar **(belirsiz)** olarak işaretli. Scimago çeyrekleri, scimagojr.com sayfalarının arama motorundaki metninden alındı (site doğrudan erişimi engelledi).

## 1. FPGA dil modeli / transformer hızlandırıcıları

| Kaynak | Bağlantı | Ne yapıyor | Bizden farkı |
|---|---|---|---|
| Zeng vd., FlightLLM, FPGA’24 | arXiv:2401.03868 | Komut tabanlı LLaMA2-7B overlay (U280/Versal), seyrek DSP zinciri, derleyici | Komut tabanlı, ama HBM ölçeğinde ve host’lu; tek model ailesi |
| Chen vd., ACM TRETS 2024 (10.1145/3656177) | arXiv:2312.15159 | Uzamsal dataflow HLS çekirdekleri ve analitik model; BERT/GPT-2, U280 | Analitik model karşılaştırması için iyi; veri merkezi FPGA’si, dış bellek |
| Huang vd., EdgeLLM, IEEE TCAS-I 2025 | arXiv:2407.21325 | CPU+FPGA, FP16×INT4 PE dizisi, uçtan uca derleyici (VCU128) | CPU gerekiyor, attention float |
| He vd., HLSTransform, ICML’24 ES-FoMo | arXiv:2405.00738 | Karpathy llama2.c 110M (TinyStories) için HLS hızlandırıcı, INT8 weight, VU9P | En yakın model ailesi; weight’ler dış bellekte, RMSNorm FP32, sampling host’ta |
| Xu, Li, Ji, LlamaF, arXiv 2024 | arXiv:2409.11424 | PTQ ve pipeline’lı GEMV, TinyLlama-1.1B, ZCU102 | PS ile birlikte çalışıyor, weight’ler DDR’de |
| Li vd., DATE 2025 | arXiv:2502.10659 | KV260 üzerinde bare-metal LLaMA2-7B, ~5 token/s | Muhtemelen ARM PS kullanıyor **(belirsiz)** |
| Li vd., Hummingbird, ICCAD 2025 | arXiv:2507.03308 | Gömülü LLM hızlandırıcısı, GQA’ya özel dataflow (KV260/ZCU104) | GQA’ya özel bulunan tek FPGA tasarımı |
| He vd., LUT-LLM, arXiv 2025 | arXiv:2511.06174 | Vektör quantization’lı tablo okuma linear katmanları; Qwen3-1.7B, V80 | Büyük ölçekte bellek-olarak-hesap fikri |
| Jiang vd., hls4ml transformers, JINST 2025 | arXiv:2409.05207 | Sabit noktalı MHA/softmax/LayerNorm tamamen çip içinde, VU13P’de < 2 µs | Çip içi küçük transformer’lar, ama fizik sınıflandırıcıları |
| Abeykoon ve Chhajer, TALOS-V2, GitHub 2026 (**hakemsiz**) | github.com/Luthiraa/TALOS-V2 | Karpathy microGPT (4 192 parametre) SystemVerilog’da, Q4.12 ROM weight’ler, RTL sampler, Cyclone V | Ruh olarak en yakın; mutlaka atıf. Tek model, Python’dan JTAG ile sürülüyor, bit-exact değil |

Bağlam (2023 öncesi): DFX (Hong vd., MICRO 2022, arXiv:2209.10797), 4×U280 GPT-2, özel ISA. SECDA-LLM (Haris vd., arXiv:2408.00462, yeri **belirsiz**): llama.cpp ve SystemC, PYNQ-Z1’de TinyLlama.

## 2. Ternary/binary LLM donanımı

| Kaynak | Bağlantı | Ne yapıyor | Bizden farkı |
|---|---|---|---|
| Zhu vd., MatMul-free LM, arXiv 2024 | arXiv:2406.02528 | Ternary, MatMul’süz LM; v1’de 13 W’lık özel FPGA hızlandırıcı | SystemVerilog RTL, ama weight’ler DRAM’de |
| Yin vd., TerEffic, arXiv 2025 | arXiv:2502.16473 | Tamamen çip içi ternary 370M model (~16,3 bin token/s) ve HBM sürümü | En güçlü “tamamen çip içi” iddiası, büyük bir FPGA’de |
| Qiao vd., TeLLMe ve TeLLMe v2, arXiv 2025 | arXiv:2504.16266, 2510.15926 | W1.58/A8, tablo okumalı matmul, edge FPGA’de prefill ve decode, 5–7 W | v2’de RoPE ve SiLU FP16 |
| Huang vd., TENET, arXiv 2025 | arXiv:2509.13765 | Seyrek ternary LUT çekirdeği; FPGA ve ASIC prototipleri | LUT merkezli ternary tasarım |
| Kim, Lee, Yoo, Slim-Llama, ISSCC 2025 | Hot Chips 2025 posteri | 4,69 mW binary/ternary LLM ASIC’i | ASIC |
| Qiao vd., COBRA, arXiv 2025 | arXiv:2504.16269 | XNOR-popcount 1 bitlik birimle binary transformer, KV260/ZCU102 | 1 bitlik GEMV’mize en yakın; yalnız BERT/GLUE, metin üretimi yok |
| Ji vd., BMT/BAT, ICCAD 2024 | arXiv:2407.12070 | Binary transformer ve akış hızlandırıcısının birlikte tasarımı | Karışık binary/ternary |
| Geens vd., ISPASS 2026 | arXiv:2604.25183 | Ternary LUT hızlandırıcıları için üreteç ve maliyet modeli | Küçük tamsayı activation’larda LUT yeniden kullanımının getirisi azalıyor; LUT ile toplama ağacı seçimi için doğrudan ilgili |

## 3. LUT-native ağlar ve LUT-GEMM

| Kaynak | Bağlantı | Ne yapıyor | İlgisi |
|---|---|---|---|
| Andronic ve Constantinides, PolyLUT, FPT 2023 | arXiv:2309.02334 | Polinom nöronlar LUT’lara gömülü | Yalnız sınıflandırma |
| Andronic ve Constantinides, NeuraLUT, FPL 2024 | arXiv:2403.00849 | L-LUT’ların içinde küçük MLP’ler | Aynı |
| Lou vd., PolyLUT-Add, FPL 2024 | arXiv:2406.04910 | Alt nöronları toplayarak geniş fan-in | Aynı |
| Cassidy vd., ReducedLUT, FPGA’25 | arXiv:2412.18579 | Görülmemiş girişleri don’t-care sayarak tablo sıkıştırma | Aynı |
| Bacellar vd., DWN, ICML 2024 | arXiv:2410.11112 | Türevlenebilir weightless LUT ağları | Aynı |
| Khataei ve Bazargan, TreeLUT, FPGA’25 | arXiv:2501.01511 | Gradient boosted ağaçlar LUT’lara, BRAM/DSP’siz | Aynı |
| Nag vd., LL-ViT, FPT 2025 | arXiv:2511.00812 | ViT içinde LUT-nöron kanal karıştırıcı | Transformer ölçeğinde tek LUT-nöron çalışması; görüntü |
| Park vd., LUT-GEMM, ICLR 2024 | arXiv:2206.09557 | Binary kodlu weight’ler için GPU GEMM | Binary factor GEMV için ilgili |
| Wei vd., T-MAC, EuroSys 2025 | arXiv:2407.00088 | CPU’da tablo okumalı 1–4 bit GEMM | CPU |
| Mo vd., LUT Tensor Core, ISCA 2025 | arXiv:2408.06003 | LUT tabanlı karışık hassasiyet GEMM | ASIC yönelimli |

## 4. Tamsayı çekirdekler ve doğrulama

| Kaynak | Bağlantı | Ne yapıyor | İlgisi |
|---|---|---|---|
| Li ve Gu, I-ViT, ICCV 2023 | arXiv:2207.01405 | Shiftmax, ShiftGELU, tamsayı LayerNorm | Hata deneysel, sınır yok |
| Hu vd., I-LLM, arXiv 2024 | arXiv:2405.17849 | Tamsayı LLM PTQ: DI-Softmax, DI-Exp, DI-Norm | En yakın operatör kümesi; GPU simülasyonu, ispatlı sınır yok |
| Marchisio vd., SwiftTron, IJCNN 2023 | arXiv:2304.03986 | INT8/INT32 tamsayı transformer ASIC’i, açık RTL | Yalnız encoder |
| Colbert vd., A2Q (ICCV 2023), A2Q+ (ICML 2024) | arXiv:2308.13504, 2401.10432 | ℓ1 weight sınırıyla **ispatlı** akümülatör taşmazlığı | “İspatlı sınırlar”a en yakın; yalnız linear katmanlar |
| Zhang, Song, Sun, QEBVerif, CAV 2023 | arXiv:2212.02781 | Quantization hata sınırlarının sağlam ve tam doğrulaması | Biçimsel, ama küçük MLP’ler |
| Huang vd., 3LA, ACM TODAES 2024 | 10.1145/3639051 | Hızlandırıcı doğrulaması için biçimsel ILA arayüzü | Doğrulama metodolojisi |
| Sun vd., HGQ, arXiv 2024 | arXiv:2405.00645 | Ağırlık başına QAT, yazılımdan firmware’e bit-doğru akış (hls4ml) | Bit-exact iddiası, sınıflandırıcılar için |
| Wang vd., SOLE, ICCAD 2023 (yer **belirsiz**) | arXiv:2510.17189 | Düşük hassasiyetli softmax ve LayerNorm donanımı | Tek operatör |

## 5. Tasarım seçimlerinin donanım maliyeti / mimari arama

| Kaynak | Bağlantı | Ne yapıyor | İlgisi |
|---|---|---|---|
| Sukthanker vd., HW-GPT-Bench, NeurIPS 2024 D&B | arXiv:2405.10299 | GPT-2 arama uzayı için 13 cihazda vekil donanım ölçütleri | Eksenler genişlik/derinlik/head; FPGA yok; norm/RoPE/GQA yok |
| Liu vd., MobileLLM, ICML 2024 | arXiv:2402.14905 | Milyar altı tasarım çalışması: SwiGLU, derin-ince, GQA, embedding paylaşımı | Yalnız kalite |
| Peng vd., HAAN, arXiv 2025 | arXiv:2502.11832 | LayerNorm/RMSNorm hızlandırıcısı | Tek operatör |
| Zhu vd., DyT, CVPR 2025 | arXiv:2503.10622 | Normalization yerine Dynamic Tanh | Yeni bir karşılaştırma ekseni olabilir |
| Li vd., Quasar-ViT, ICS 2024 | arXiv:2407.18175 | FPGA gecikme/kaynak modeliyle NAS (ZCU102) | Görüntü |
| Agostinelli vd., ELiTeFormer, arXiv 2026 | arXiv:2607.03652 | Linear attention ve ternary transformer, Versal | Maliyet block başına, özellik başına değil |
| Jiang vd., LLMForge, arXiv 2026 | arXiv:2605.17653 | Edge LM’ler için donanım bilinçli NAS | FPGA yok |

RoPE’un FPGA donanım maliyetini ölçen bir çalışma bulunamadı.

## 6. LittleBit sonrası sub-1-bit / binary factorization

| Kaynak | Bağlantı | Ne yapıyor | İlgisi |
|---|---|---|---|
| Lee ve Kim, LittleBit-2, arXiv 2026 (depo ICML 2026 diyor, **belirsiz**) | arXiv:2603.00042 | Latent rotation ve Joint-ITQ; 1–0,1 bpp’de en iyi sonuç iddiası | Aynı çıkarım biçimi; GEMVB birimimiz uygulanabilir |
| Chong vd., NanoQuant, ICML 2026 | arXiv:2602.06694 | ADMM ile low-rank binary factorization olarak PTQ | Donanım yok |
| Boža ve Macko, DBF, TMLR 2026 | arXiv:2505.11076 | İki işaret matrisinin scale’li çarpımı; GPU’da 2–3,5× | Yalnız GPU |
| Xu vd., OneBit, NeurIPS 2024 | arXiv:2402.11295 | İşaret/değer ayrıştırması (SVID); LittleBit’in atası | – |
| Dong vd., STBLLM, ICLR 2025 | arXiv:2408.01803 | N:M seyrek binarization ile 1 bitin altı | – |
| Gu vd., BTC-LLM, ACL 2026 | arXiv:2506.12040 | Binary codebook ve öğrenilen rotation; XNOR/POPCNT çekirdekleri | Yalnız CPU/GPU |

Bunların hiçbirinin FPGA ya da ASIC uygulaması bulunamadı; LittleBit yalnızca “potansiyel 11,6× hızlanma” belirtiyor.

## 7. Dergiler (Scimago SJR 2025, en iyi çeyrek)

| Dergi | SJR 2025 | En iyi çeyrek |
|---|---|---|
| IEEE TCAS-I | 1,947 | Q1 |
| IEEE TC | 1,243 | Q1 |
| IEEE TVLSI | 1,041 | Q1 |
| IEEE JETCAS | 0,953 | Q1 |
| IEEE TCAD | 0,928 | Q1 |
| ACM TECS | 0,864 | Q2 |
| ACM TRETS | 0,741 | Q2 |

“En iyi çeyrek” en az bir kategoride Q1 demektir; ilgili kategorinin Q1 olduğu ayrıca kontrol edilmeli. 2024 çeyrekleri doğrulanmadı (TVLSI için bir kaynak 2024 SJR’sini 0,744 veriyor, o yıl Q2 olabilir).

## Savunulabilir boşluklar (temkinli: “bulunamadı”)

1. **Tek engine’de genişlik.** Birçok LM ailesini (bigram/MLP/RNN/GRU/BoW ve GPT-2/Llama tarzı transformer’lar) tek bir FPGA’de tek bir mikrokodlu veri yolunda çalıştıran hakemli bir çalışma yok. Komut tabanlı overlay’ler var (FlightLLM, DFX); yeni olan mikrokod değil, kapsam.
2. **Kontrollü özellik maliyeti çalışması.** RMSNorm, RoPE, GQA, SwiGLU ve QK-norm’un eşit bütçede cycle/LUT/DSP/Fmax maliyetini ayrı ayrı ölçen çalışma yok.
3. **Tamamen bağımsız fabric.** Sampling, detokenization ve ekranın host ya da sert CPU olmadan fabric içinde yapılması: HLSTransform host’ta örnekliyor, TALOS-V2 JTAG ile sürülüyor, gömülü tasarımlar Zynq PS kullanıyor.
4. **Bit-exact eşdeğerlik ve ispatlı aralıklar.** Softmax, RMSNorm ve RoPE dahil bütün operatörlerde Python tamsayı referansıyla bit-exact eşdeğerliği ispatlı aralık sınırlarıyla birleştiren çalışma yok.
5. **LittleBit tarzı donanım.** Low-rank binary factor’ların donanım uygulaması yok.
6. **Açık araç zinciri.** Yosys + nextpnr-himbaechel + Project X-Ray akışıyla kurulmuş hakemli bir LM hızlandırıcısı yok; Vivado ile QoR karşılaştırması da ek katkı olabilir.

**Uyarılar:** “Bulunamadı” yokluğun kanıtı değildir; “ilk” yazmadan önce IEEE Xplore’da doğrudan arama yapılmalı. TALOS-V2 ve TeLLMe/TerEffic çizgisi hakemlerin en olası karşı örnekleridir. 50K–260K parametrede tamamen çip içi bir tasarım beklenen bir sonuçtur; bunu tek başına katkı değil, kontrollü karşılaştırmayı mümkün kılan bir seçim olarak sunmak gerekir.
