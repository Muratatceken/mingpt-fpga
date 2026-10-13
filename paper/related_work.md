# Related work: verified references, positioning, and venue deadlines

Compiled 2026-10-13. Every entry below was checked against at least one primary record (publisher/DOI metadata via Crossref, arXiv API record, PMLR/NeurIPS/JMLR proceedings page, ACL Anthology DOI, GitHub API, or the author's own page). The `% verified:` line above each entry gives the URL used. DBLP was not reachable (bot wall), so Crossref + arXiv + proceedings pages were used instead.

Conventions:
- Author lists are complete except for two very long ones (Llama 2: 68 authors; ViT-22B: 42 authors), which use `and others`. IEEE style prints "et al." for more than six authors anyway.
- Where the published title differs from the arXiv title, the **published** title is used and the difference is noted in the comment (GPTQ/OPTQ, LLM.int8()/GPT3.int8(), Primer, McKinstry, TeLLMe).
- Page numbers are included only where the publisher record confirmed them.

Count: **64 verified entries** (A 13, B 20, C 7, D 21, E 3). That's more than a 6–8 page IEEE paper needs. Section 3 suggests a core subset of about 32.

---

## 1. BibTeX

```bibtex
% =====================================================================
% A. Integer-only / quantized inference foundations
% =====================================================================

% verified: https://doi.org/10.1109/CVPR.2018.00286 (Crossref record)
@inproceedings{jacob2018integer,
  author    = {Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  title     = {Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  booktitle = {Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2704--2713},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00286}
}

% verified: https://arxiv.org/abs/1806.08342
@misc{krishnamoorthi2018quantizing,
  author        = {Krishnamoorthi, Raghuraman},
  title         = {Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper},
  howpublished  = {arXiv:1806.08342},
  year          = {2018},
  eprint        = {1806.08342},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2004.09602 (percentile calibration 99.9%-99.9999% confirmed in the PDF, Table 5 / App. A)
@misc{wu2020integer,
  author        = {Wu, Hao and Judd, Patrick and Zhang, Xiaojie and Isaev, Mikhail and Micikevicius, Paulius},
  title         = {Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation},
  howpublished  = {arXiv:2004.09602},
  year          = {2020},
  eprint        = {2004.09602},
  archivePrefix = {arXiv}
}

% verified: https://arxiv.org/abs/2106.08295
@misc{nagel2021white,
  author        = {Nagel, Markus and Fournarakis, Marios and Amjad, Rana Ali and Bondarenko, Yelysei and van Baalen, Mart and Blankevoort, Tijmen},
  title         = {A White Paper on Neural Network Quantization},
  howpublished  = {arXiv:2106.08295},
  year          = {2021},
  eprint        = {2106.08295},
  archivePrefix = {arXiv}
}

% verified: https://proceedings.mlr.press/v139/kim21d.html
@inproceedings{kim2021ibert,
  author    = {Kim, Sehoon and Gholami, Amir and Yao, Zhewei and Mahoney, Michael W. and Keutzer, Kurt},
  title     = {{I-BERT}: Integer-only {BERT} Quantization},
  booktitle = {Proc. 38th Int. Conf. Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {5506--5518},
  year      = {2021},
  url       = {https://proceedings.mlr.press/v139/kim21d.html}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html
% NOTE: the NeurIPS proceedings title is "GPT3.int8()"; the arXiv version (2208.07339) is titled "LLM.int8()".
@inproceedings{dettmers2022llmint8,
  author    = {Dettmers, Tim and Lewis, Mike and Belkada, Younes and Zettlemoyer, Luke},
  title     = {{GPT3.int8()}: 8-bit Matrix Multiplication for Transformers at Scale},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  pages     = {30318--30332},
  year      = {2022},
  note      = {arXiv:2208.07339 (as ``LLM.int8()'')}
}

% verified: https://proceedings.mlr.press/v202/xiao23c.html
@inproceedings{xiao2023smoothquant,
  author    = {Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  title     = {{SmoothQuant}: Accurate and Efficient Post-Training Quantization for Large Language Models},
  booktitle = {Proc. 40th Int. Conf. Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {38087--38099},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v202/xiao23c.html}
}

% verified: https://iclr.cc/virtual/2023/poster/10855 and https://research-explorer.ista.ac.at/record/17378 ; arXiv https://arxiv.org/abs/2210.17323
% NOTE: published at ICLR 2023 under the name "OPTQ"; arXiv/code name is "GPTQ".
@inproceedings{frantar2023gptq,
  author    = {Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  title     = {{OPTQ}: Accurate Quantization for Generative Pre-trained Transformers},
  booktitle = {Proc. 11th Int. Conf. Learning Representations (ICLR)},
  year      = {2023},
  note      = {arXiv:2210.17323 (as ``GPTQ'')}
}

% verified: https://doi.org/10.1109/EMC2-NIPS53020.2019.00016 (Crossref) ; https://arxiv.org/abs/1910.06188
@inproceedings{zafrir2019q8bert,
  author    = {Zafrir, Ofir and Boudoukh, Guy and Izsak, Peter and Wasserblat, Moshe},
  title     = {{Q8BERT}: Quantized 8Bit {BERT}},
  booktitle = {Proc. 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing -- NeurIPS Edition (EMC2-NIPS)},
  pages     = {36--39},
  year      = {2019},
  doi       = {10.1109/EMC2-NIPS53020.2019.00016}
}

% verified: original URL https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf now redirects;
%           archived copy opened and read: http://web.archive.org/web/20240728112810/https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
%           (title slide: "8-bit Inference with TensorRT, Szymon Migacz, NVIDIA, May 8, 2017")
@misc{migacz2017tensorrt,
  author       = {Migacz, Szymon},
  title        = {8-bit Inference with {TensorRT}},
  howpublished = {NVIDIA GPU Technology Conference (GTC), San Jose, CA, presentation S7310},
  month        = may,
  year         = {2017},
  url          = {http://web.archive.org/web/20240728112810/https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf}
}

% verified: https://doi.org/10.1109/EMC2-NIPS53020.2019.00009 (Crossref) ; arXiv https://arxiv.org/abs/1809.04191
% NOTE: published title ends "...for Efficient Inference"; arXiv title ends "...for Efficient Embedded Inference".
%       99.99th-percentile activation range calibration confirmed in the PDF.
@inproceedings{mckinstry2019discovering,
  author    = {McKinstry, Jeffrey L. and Esser, Steven K. and Appuswamy, Rathinakumar and Bablani, Deepika and Arthur, John V. and Yildiz, Izzet B. and Modha, Dharmendra S.},
  title     = {Discovering Low-Precision Networks Close to Full-Precision Networks for Efficient Inference},
  booktitle = {Proc. 5th Workshop on Energy Efficient Machine Learning and Cognitive Computing -- NeurIPS Edition (EMC2-NIPS)},
  pages     = {6--9},
  year      = {2019},
  doi       = {10.1109/EMC2-NIPS53020.2019.00009}
}

% verified: https://doi.org/10.18653/v1/2021.emnlp-main.627 (Crossref) ; https://arxiv.org/abs/2109.12948
@inproceedings{bondarenko2021understanding,
  author    = {Bondarenko, Yelysei and Nagel, Markus and Blankevoort, Tijmen},
  title     = {Understanding and Overcoming the Challenges of Efficient Transformer Quantization},
  booktitle = {Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {7947--7969},
  year      = {2021},
  doi       = {10.18653/v1/2021.emnlp-main.627}
}

% verified: https://arxiv.org/abs/2405.17849
@misc{hu2024illm,
  author        = {Hu, Xing and Cheng, Yuan and Yang, Dawei and Yuan, Zhihang and Yu, Jiangyong and Xu, Chen and Zhou, Sifan},
  title         = {{I-LLM}: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models},
  howpublished  = {arXiv:2405.17849},
  year          = {2024},
  eprint        = {2405.17849},
  archivePrefix = {arXiv}
}

% =====================================================================
% B. FPGA/ASIC transformer & LLM accelerators, overlays, edge LLM on FPGA
% =====================================================================

% verified: https://doi.org/10.1145/3370748.3406567 (Crossref) ; https://arxiv.org/abs/2007.08563
@inproceedings{li2020ftrans,
  author    = {Li, Bingbing and Pandey, Santosh and Fang, Haowen and Lyv, Yanjun and Li, Ji and Chen, Jieyang and Xie, Mimi and Wan, Lipeng and Liu, Hang and Ding, Caiwen},
  title     = {{FTRANS}: Energy-Efficient Acceleration of Transformers using {FPGA}},
  booktitle = {Proc. ACM/IEEE Int. Symp. Low Power Electronics and Design (ISLPED)},
  pages     = {175--180},
  year      = {2020},
  doi       = {10.1145/3370748.3406567}
}

% verified: https://doi.org/10.1109/HPCA47549.2020.00035 (Crossref)
@inproceedings{ham2020a3,
  author    = {Ham, Tae Jun and Jung, Sung Jun and Kim, Seonghak and Oh, Young H. and Park, Yeonhong and Song, Yoonho and Park, Jung-Hun and Lee, Sanghee and Park, Kyoung and Lee, Jae W. and Jeong, Deog-Kyoon},
  title     = {{A}$^3$: Accelerating Attention Mechanisms in Neural Networks with Approximation},
  booktitle = {Proc. IEEE Int. Symp. High Performance Computer Architecture (HPCA)},
  pages     = {328--341},
  year      = {2020},
  doi       = {10.1109/HPCA47549.2020.00035}
}

% verified: https://doi.org/10.1109/HPCA51647.2021.00018 (Crossref) ; https://arxiv.org/abs/2012.09852
@inproceedings{wang2021spatten,
  author    = {Wang, Hanrui and Zhang, Zhekai and Han, Song},
  title     = {{SpAtten}: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning},
  booktitle = {Proc. IEEE Int. Symp. High-Performance Computer Architecture (HPCA)},
  pages     = {97--110},
  year      = {2021},
  doi       = {10.1109/HPCA51647.2021.00018}
}

% verified: https://doi.org/10.1109/ISCA52012.2021.00060 (Crossref)
@inproceedings{ham2021elsa,
  author    = {Ham, Tae Jun and Lee, Yejin and Seo, Seong Hoon and Kim, Soosung and Choi, Hyunji and Jung, Sung Jun and Lee, Jae W.},
  title     = {{ELSA}: Hardware-Software Co-design for Efficient, Lightweight Self-Attention Mechanism in Neural Networks},
  booktitle = {Proc. ACM/IEEE 48th Annu. Int. Symp. Computer Architecture (ISCA)},
  pages     = {692--705},
  year      = {2021},
  doi       = {10.1109/ISCA52012.2021.00060}
}

% verified: https://doi.org/10.1109/MICRO56248.2022.00051 (Crossref) ; https://arxiv.org/abs/2209.10797
@inproceedings{hong2022dfx,
  author    = {Hong, Seongmin and Moon, Seungjae and Kim, Junsoo and Lee, Sungjae and Kim, Minsub and Lee, Dongsoo and Kim, Joo-Young},
  title     = {{DFX}: A Low-latency Multi-{FPGA} Appliance for Accelerating Transformer-based Text Generation},
  booktitle = {Proc. 55th IEEE/ACM Int. Symp. Microarchitecture (MICRO)},
  pages     = {616--630},
  year      = {2022},
  doi       = {10.1109/MICRO56248.2022.00051}
}

% verified: https://doi.org/10.1145/3626202.3637562 (Crossref) ; https://arxiv.org/abs/2401.03868
@inproceedings{zeng2024flightllm,
  author    = {Zeng, Shulin and Liu, Jun and Dai, Guohao and Yang, Xinhao and Fu, Tianyu and Wang, Hongyi and Ma, Wenheng and Sun, Hanbo and Li, Shiyao and Huang, Zixiao and Dai, Yadong and Li, Jintao and Wang, Zehao and Zhang, Ruoyu and Wen, Kairui and Ning, Xuefei and Wang, Yu},
  title     = {{FlightLLM}: Efficient Large Language Model Inference with a Complete Mapping Flow on {FPGAs}},
  booktitle = {Proc. ACM/SIGDA Int. Symp. Field Programmable Gate Arrays (FPGA)},
  pages     = {223--234},
  year      = {2024},
  doi       = {10.1145/3626202.3637562}
}

% verified: https://doi.org/10.1145/3656177 (Crossref: vol 18, iss 1, pp 1-29; online 2024-12-17, print 2025-03) ; https://arxiv.org/abs/2312.15159
@article{chen2025understanding,
  author  = {Chen, Hongzheng and Zhang, Jiahao and Du, Yixiao and Xiang, Shaojie and Yue, Zichao and Zhang, Niansong and Cai, Yaohui and Zhang, Zhiru},
  title   = {Understanding the Potential of {FPGA}-based Spatial Acceleration for Large Language Model Inference},
  journal = {ACM Trans. Reconfigurable Technology and Systems},
  volume  = {18},
  number  = {1},
  pages   = {1--29},
  year    = {2025},
  doi     = {10.1145/3656177}
}

% verified: https://arxiv.org/abs/2405.00738 and https://arxiv.org/html/2405.00738v1 (builds on llama2.c; 110M TinyStories model; INT8; Virtex UltraScale+ VU9P)
@misc{he2024hlstransform,
  author        = {He, Andy and Key, Darren and Bulling, Mason and Chang, Andrew and Shapiro, Skyler and Lee, Everett},
  title         = {{HLSTransform}: Energy-Efficient {Llama} 2 Inference on {FPGAs} Via High Level Synthesis},
  howpublished  = {arXiv:2405.00738},
  year          = {2024},
  eprint        = {2405.00738},
  archivePrefix = {arXiv}
}

% verified: https://doi.org/10.1109/WF-IoT62078.2024.10811385 (Crossref) ; https://arxiv.org/abs/2409.11424
@inproceedings{xu2024llamaf,
  author    = {Xu, Han and Li, Yutong and Ji, Shihao},
  title     = {{LlamaF}: An Efficient {Llama2} Architecture Accelerator on Embedded {FPGAs}},
  booktitle = {Proc. IEEE 10th World Forum on Internet of Things (WF-IoT)},
  pages     = {1--7},
  year      = {2024},
  doi       = {10.1109/WF-IoT62078.2024.10811385}
}

% verified: https://arxiv.org/abs/2408.00462 (arXiv only; no peer-reviewed venue confirmed)
@misc{haris2024secdallm,
  author        = {Haris, Jude and Saha, Rappy and Hu, Wenhao and Cano, Jos{\'e}},
  title         = {Designing Efficient {LLM} Accelerators for Edge Devices},
  howpublished  = {arXiv:2408.00462},
  year          = {2024},
  eprint        = {2408.00462},
  archivePrefix = {arXiv}
}

% verified: https://doi.org/10.1145/3731545.3735115 (Crossref: HPDC'25, pp 1-3, i.e., a short paper) ; https://arxiv.org/abs/2507.14139
@inproceedings{wang2025speedllm,
  author    = {Wang, Peipei and Guan, Wu and Liang, Liping and Wang, Zhijun and Luo, Hanqing and Zhang, Zhibin},
  title     = {{SpeedLLM}: An {FPGA} Co-design of Large Language Model Inference Accelerator},
  booktitle = {Proc. 34th Int. Symp. High-Performance Parallel and Distributed Computing (HPDC)},
  pages     = {1--3},
  year      = {2025},
  doi       = {10.1145/3731545.3735115}
}

% verified: https://doi.org/10.1145/3748173.3779191 (Crossref, FPGA'26) ; arXiv https://arxiv.org/abs/2504.16266 (arXiv title differs: "TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs")
@inproceedings{qiao2026tellme,
  author    = {Qiao, Ye and Chen, Zhiheng and Zhang, Yifan and Wang, Yian and Huang, Sitao},
  title     = {{TeLLMe}: An Efficient End-to-End Ternary {LLM} Prefill and Decode Accelerator with Table-Lookup Matmul on Edge {FPGAs}},
  booktitle = {Proc. ACM/SIGDA Int. Symp. Field Programmable Gate Arrays (FPGA)},
  pages     = {247--257},
  year      = {2026},
  doi       = {10.1145/3748173.3779191}
}

% verified: https://arxiv.org/abs/1710.10296 (DRNN LM on PYNQ, XC7Z020; arXiv only)
@misc{hao2017drnn,
  author        = {Hao, Yufeng and Quigley, Steven},
  title         = {The Implementation of a Deep Recurrent Neural Network Language Model on a {Xilinx} {FPGA}},
  howpublished  = {arXiv:1710.10296},
  year          = {2017},
  eprint        = {1710.10296},
  archivePrefix = {arXiv}
}

% verified: https://doi.org/10.1109/TCAD.2017.2705069 (Crossref) ; preprint https://people.eecs.berkeley.edu/~alanmi/publications/other/tcad18_angeleye.pdf (search snippet: evaluated on XC7Z020 and XC7Z045)
@article{guo2018angeleye,
  author  = {Guo, Kaiyuan and Sui, Lingzhi and Qiu, Jiantao and Yu, Jincheng and Wang, Junbin and Yao, Song and Han, Song and Wang, Yu and Yang, Huazhong},
  title   = {{Angel-Eye}: A Complete Design Flow for Mapping {CNN} Onto Embedded {FPGA}},
  journal = {IEEE Trans. Computer-Aided Design of Integrated Circuits and Systems},
  volume  = {37},
  number  = {1},
  pages   = {35--47},
  year    = {2018},
  doi     = {10.1109/TCAD.2017.2705069}
}

% verified: https://doi.org/10.1109/ISCA.2018.00012 (Crossref) ; abstract at https://www.microsoft.com/en-us/research/publication/a-configurable-cloud-scale-dnn-processor-for-real-time-ai/
@inproceedings{fowers2018brainwave,
  author    = {Fowers, Jeremy and Ovtcharov, Kalin and Papamichael, Michael and Massengill, Todd and Liu, Ming and Lo, Daniel and Alkalay, Shlomi and Haselman, Michael and Adams, Logan and Ghandi, Mahdi and Heil, Stephen and Patel, Prerak and Sapek, Adam and Weisz, Gabriel and Woods, Lisa and Lanka, Sitaram and Reinhardt, Steven K. and Caulfield, Adrian M. and Chung, Eric S. and Burger, Doug},
  title     = {A Configurable Cloud-Scale {DNN} Processor for Real-Time {AI}},
  booktitle = {Proc. ACM/IEEE 45th Annu. Int. Symp. Computer Architecture (ISCA)},
  pages     = {1--14},
  year      = {2018},
  doi       = {10.1109/ISCA.2018.00012}
}

% verified: https://doi.org/10.1109/TVLSI.2019.2939726 (Crossref)
@article{yu2020opu,
  author  = {Yu, Yunxuan and Wu, Chen and Zhao, Tiandong and Wang, Kun and He, Lei},
  title   = {{OPU}: An {FPGA}-Based Overlay Processor for Convolutional Neural Networks},
  journal = {IEEE Trans. Very Large Scale Integration (VLSI) Systems},
  volume  = {28},
  number  = {1},
  pages   = {35--47},
  year    = {2020},
  doi     = {10.1109/TVLSI.2019.2939726}
}

% verified: https://doi.org/10.1145/3431920.3439477 (Crossref: FPGA'21, page 227 only => abstract-length proceedings entry) ; full text https://arxiv.org/abs/2104.06535
@inproceedings{khan2021npe,
  author    = {Khan, Hamza and Khan, Asma and Khan, Zainab and Huang, Lun Bin and Wang, Kun and He, Lei},
  title     = {{NPE}: An {FPGA}-based Overlay Processor for Natural Language Processing},
  booktitle = {Proc. ACM/SIGDA Int. Symp. Field-Programmable Gate Arrays (FPGA)},
  pages     = {227},
  year      = {2021},
  doi       = {10.1145/3431920.3439477},
  note      = {Extended version: arXiv:2104.06535}
}

% verified: https://doi.org/10.1109/MM.2019.2928962 (Crossref) ; https://arxiv.org/abs/1807.04188
@article{moreau2019vta,
  author  = {Moreau, Thierry and Chen, Tianqi and Vega, Luis and Roesch, Jared and Yan, Eddie and Zheng, Lianmin and Fromm, Josh and Jiang, Ziheng and Ceze, Luis and Guestrin, Carlos and Krishnamurthy, Arvind},
  title   = {A Hardware--Software Blueprint for Flexible Deep Learning Specialization},
  journal = {IEEE Micro},
  volume  = {39},
  number  = {5},
  pages   = {8--16},
  year    = {2019},
  doi     = {10.1109/MM.2019.2928962}
}

% verified: https://doi.org/10.1088/1748-0221/13/07/P07027 (Crossref)
@article{duarte2018hls4ml,
  author  = {Duarte, J. and Han, S. and Harris, P. and Jindariani, S. and Kreinar, E. and Kreis, B. and Ngadiuba, J. and Pierini, M. and Rivera, R. and Tran, N. and Wu, Z.},
  title   = {Fast Inference of Deep Neural Networks in {FPGAs} for Particle Physics},
  journal = {Journal of Instrumentation},
  volume  = {13},
  number  = {07},
  pages   = {P07027},
  year    = {2018},
  doi     = {10.1088/1748-0221/13/07/P07027}
}
% (Author given names are initials-only in the Crossref/IOP record; kept as initials.)

% verified: https://doi.org/10.1109/FPL57034.2022.00027 (Crossref)
@inproceedings{li2022autovitacc,
  author    = {Li, Zhengang and Sun, Mengshu and Lu, Alec and Ma, Haoyu and Yuan, Geng and Xie, Yanyue and Tang, Hao and Li, Yanyu and Leeser, Miriam and Wang, Zhangyang and Lin, Xue and Fang, Zhenman},
  title     = {{Auto-ViT-Acc}: An {FPGA}-Aware Automatic Acceleration Framework for Vision Transformer with Mixed-Scheme Quantization},
  booktitle = {Proc. 32nd Int. Conf. Field-Programmable Logic and Applications (FPL)},
  pages     = {109--116},
  year      = {2022},
  doi       = {10.1109/FPL57034.2022.00027}
}

% =====================================================================
% C. Integer / approximate nonlinearities; RNN/LSTM FPGA accelerators
% =====================================================================

% verified: https://doi.org/10.1109/DAC18074.2021.9586134 (Crossref) ; https://arxiv.org/abs/2103.09301
@inproceedings{stevens2021softermax,
  author    = {Stevens, Jacob R. and Venkatesan, Rangharajan and Dai, Steve and Khailany, Brucek and Raghunathan, Anand},
  title     = {Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers},
  booktitle = {Proc. 58th ACM/IEEE Design Automation Conf. (DAC)},
  pages     = {469--474},
  year      = {2021},
  doi       = {10.1109/DAC18074.2021.9586134}
}

% verified: https://doi.org/10.1109/ICCV51070.2023.01565 (Crossref) ; https://arxiv.org/abs/2207.01405
@inproceedings{li2023ivit,
  author    = {Li, Zhikai and Gu, Qingyi},
  title     = {{I-ViT}: Integer-only Quantization for Efficient Vision Transformer Inference},
  booktitle = {Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)},
  pages     = {17019--17029},
  year      = {2023},
  doi       = {10.1109/ICCV51070.2023.01565}
}

% verified: https://doi.org/10.24963/ijcai.2022/164 (Crossref) ; https://arxiv.org/abs/2111.13824
@inproceedings{lin2022fqvit,
  author    = {Lin, Yang and Zhang, Tianyu and Sun, Peiqin and Li, Zheng and Zhou, Shuchang},
  title     = {{FQ-ViT}: Post-Training Quantization for Fully Quantized Vision Transformer},
  booktitle = {Proc. 31st Int. Joint Conf. Artificial Intelligence (IJCAI)},
  pages     = {1173--1179},
  year      = {2022},
  doi       = {10.24963/ijcai.2022/164}
}

% verified: https://doi.org/10.1145/3489517.3530505 (Crossref) ; https://arxiv.org/abs/2112.02191
@inproceedings{yu2022nnlut,
  author    = {Yu, Joonsang and Park, Junki and Park, Seongmin and Kim, Minsoo and Lee, Sihwa and Lee, Dong Hyun and Choi, Jungwook},
  title     = {{NN-LUT}: Neural Approximation of Non-Linear Operations for Efficient Transformer Inference},
  booktitle = {Proc. 59th ACM/IEEE Design Automation Conf. (DAC)},
  pages     = {577--582},
  year      = {2022},
  doi       = {10.1145/3489517.3530505}
}

% verified: https://doi.org/10.1109/ICCAD57390.2023.10323725 (Crossref) ; abstract https://arxiv.org/abs/2510.17189
@inproceedings{wang2023sole,
  author    = {Wang, Wenxun and Zhou, Shuchang and Sun, Wenyu and Sun, Peiqin and Liu, Yongpan},
  title     = {{SOLE}: Hardware-Software Co-design of Softmax and {LayerNorm} for Efficient Transformer Inference},
  booktitle = {Proc. IEEE/ACM Int. Conf. Computer Aided Design (ICCAD)},
  pages     = {1--9},
  year      = {2023},
  doi       = {10.1109/ICCAD57390.2023.10323725}
}

% verified: https://doi.org/10.1145/3020078.3021745 (Crossref) ; https://arxiv.org/abs/1612.00694
@inproceedings{han2017ese,
  author    = {Han, Song and Kang, Junlong and Mao, Huizi and Hu, Yiming and Li, Xin and Li, Yubin and Xie, Dongliang and Luo, Hong and Yao, Song and Wang, Yu and Yang, Huazhong and Dally, William J.},
  title     = {{ESE}: Efficient Speech Recognition Engine with Sparse {LSTM} on {FPGA}},
  booktitle = {Proc. ACM/SIGDA Int. Symp. Field-Programmable Gate Arrays (FPGA)},
  pages     = {75--84},
  year      = {2017},
  doi       = {10.1145/3020078.3021745}
}

% verified: https://doi.org/10.1145/3174243.3174253 (Crossref) ; https://arxiv.org/abs/1803.06305
@inproceedings{wang2018clstm,
  author    = {Wang, Shuo and Li, Zhe and Ding, Caiwen and Yuan, Bo and Qiu, Qinru and Wang, Yanzhi and Liang, Yun},
  title     = {{C-LSTM}: Enabling Efficient {LSTM} using Structured Compression Techniques on {FPGAs}},
  booktitle = {Proc. ACM/SIGDA Int. Symp. Field-Programmable Gate Arrays (FPGA)},
  pages     = {11--20},
  year      = {2018},
  doi       = {10.1145/3174243.3174253}
}

% =====================================================================
% D. The models and architectural components
% =====================================================================

% verified: https://www.jmlr.org/papers/v3/bengio03a.html
@article{bengio2003neural,
  author  = {Bengio, Yoshua and Ducharme, R{\'e}jean and Vincent, Pascal and Jauvin, Christian},
  title   = {A Neural Probabilistic Language Model},
  journal = {Journal of Machine Learning Research},
  volume  = {3},
  pages   = {1137--1155},
  year    = {2003}
}

% verified: https://doi.org/10.21437/Interspeech.2010-343 (Crossref)
@inproceedings{mikolov2010recurrent,
  author    = {Mikolov, Tom{\'a}{\v{s}} and Karafi{\'a}t, Martin and Burget, Luk{\'a}{\v{s}} and {\v{C}}ernock{\'y}, Jan and Khudanpur, Sanjeev},
  title     = {Recurrent Neural Network Based Language Model},
  booktitle = {Proc. Interspeech},
  pages     = {1045--1048},
  year      = {2010},
  doi       = {10.21437/Interspeech.2010-343}
}

% verified: https://doi.org/10.3115/v1/D14-1179 (Crossref)
@inproceedings{cho2014learning,
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  title     = {Learning Phrase Representations using {RNN} Encoder--Decoder for Statistical Machine Translation},
  booktitle = {Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {1724--1734},
  year      = {2014},
  doi       = {10.3115/v1/D14-1179}
}

% verified: https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html (pages not shown on the record -> omitted)
@inproceedings{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention is All you Need},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {30},
  year      = {2017}
}

% verified: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf (PDF opened; author line read)
@techreport{radford2019language,
  author      = {Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  title       = {Language Models are Unsupervised Multitask Learners},
  institution = {OpenAI},
  year        = {2019},
  url         = {https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf}
}

% verified: https://arxiv.org/abs/2307.09288 (68 authors; truncated with "others")
@misc{touvron2023llama2,
  author        = {Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and others},
  title         = {{Llama} 2: Open Foundation and Fine-Tuned Chat Models},
  howpublished  = {arXiv:2307.09288},
  year          = {2023},
  eprint        = {2307.09288},
  archivePrefix = {arXiv}
}

% verified: https://doi.org/10.1016/j.neucom.2023.127063 (Crossref; author order of the journal version) ; arXiv https://arxiv.org/abs/2104.09864
@article{su2024roformer,
  author  = {Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Bo, Wen and Liu, Yunfeng},
  title   = {{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  journal = {Neurocomputing},
  volume  = {568},
  pages   = {127063},
  year    = {2024},
  doi     = {10.1016/j.neucom.2023.127063}
}

% verified: https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
@inproceedings{zhang2019rmsnorm,
  author    = {Zhang, Biao and Sennrich, Rico},
  title     = {Root Mean Square Layer Normalization},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {32},
  year      = {2019}
}

% verified: https://arxiv.org/abs/2002.05202
@misc{shazeer2020glu,
  author        = {Shazeer, Noam},
  title         = {{GLU} Variants Improve Transformer},
  howpublished  = {arXiv:2002.05202},
  year          = {2020},
  eprint        = {2002.05202},
  archivePrefix = {arXiv}
}

% verified: https://doi.org/10.18653/v1/2023.emnlp-main.298 (Crossref) ; https://arxiv.org/abs/2305.13245
@inproceedings{ainslie2023gqa,
  author    = {Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebr{\'o}n, Federico and Sanghai, Sumit},
  title     = {{GQA}: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  booktitle = {Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {4895--4901},
  year      = {2023},
  doi       = {10.18653/v1/2023.emnlp-main.298}
}

% verified: https://proceedings.neurips.cc/paper/2021/hash/2f3c6a4cd8af177f6456e7e51a916ff3-Abstract.html ; arXiv https://arxiv.org/abs/2109.08668
% NOTE: proceedings-page title omits the "Primer:" prefix used on arXiv; the model is called Primer in the paper.
@inproceedings{so2021primer,
  author    = {So, David R. and Ma{\'n}ke, Wojciech and Liu, Hanxiao and Dai, Zihang and Shazeer, Noam and Le, Quoc V.},
  title     = {Searching for Efficient Transformers for Language Modeling},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {34},
  pages     = {6010--6022},
  year      = {2021}
}

% verified: https://doi.org/10.18653/v1/2020.findings-emnlp.379 (Crossref) ; https://arxiv.org/abs/2010.04245
@inproceedings{henry2020qknorm,
  author    = {Henry, Alex and Dachapally, Prudhvi Raj and Pawar, Shubham Shantaram and Chen, Yuxuan},
  title     = {Query-Key Normalization for Transformers},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {4246--4253},
  year      = {2020},
  doi       = {10.18653/v1/2020.findings-emnlp.379}
}

% verified: https://proceedings.mlr.press/v202/dehghani23a.html (42 authors; truncated with "others")
@inproceedings{dehghani2023scaling,
  author    = {Dehghani, Mostafa and Djolonga, Josip and Mustafa, Basil and Padlewski, Piotr and Heek, Jonathan and Gilmer, Justin and others},
  title     = {Scaling Vision Transformers to 22 Billion Parameters},
  booktitle = {Proc. 40th Int. Conf. Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {7480--7512},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v202/dehghani23a.html}
}

% verified: https://arxiv.org/abs/2305.07759
@misc{eldan2023tinystories,
  author        = {Eldan, Ronen and Li, Yuanzhi},
  title         = {{TinyStories}: How Small Can Language Models Be and Still Speak Coherent {English}?},
  howpublished  = {arXiv:2305.07759},
  year          = {2023},
  eprint        = {2305.07759},
  archivePrefix = {arXiv}
}

% ---- Karpathy repositories (GitHub API checked 2026-10-13; created_at shown) ----

% verified: https://github.com/karpathy/makemore (created 2022-06-09)
@misc{karpathy2022makemore,
  author       = {Karpathy, Andrej},
  title        = {makemore: An Autoregressive Character-Level Language Model for Making More Things},
  howpublished = {\url{https://github.com/karpathy/makemore}},
  year         = {2022},
  note         = {Accessed: 2026-10-13}
}

% verified: https://github.com/karpathy/minGPT (created 2020-08-17)
@misc{karpathy2020mingpt,
  author       = {Karpathy, Andrej},
  title        = {{minGPT}: A Minimal {PyTorch} Re-implementation of the {OpenAI} {GPT}},
  howpublished = {\url{https://github.com/karpathy/minGPT}},
  year         = {2020},
  note         = {Accessed: 2026-10-13}
}

% verified: https://github.com/karpathy/nanoGPT (created 2022-12-28)
@misc{karpathy2022nanogpt,
  author       = {Karpathy, Andrej},
  title        = {{nanoGPT}: The Simplest, Fastest Repository for Training/Finetuning Medium-Sized {GPTs}},
  howpublished = {\url{https://github.com/karpathy/nanoGPT}},
  year         = {2022},
  note         = {Accessed: 2026-10-13}
}

% verified: https://github.com/karpathy/llama2.c (created 2023-07-23); README confirms stories260K row and the int8 (Q8_0) scheme
@misc{karpathy2023llama2c,
  author       = {Karpathy, Andrej},
  title        = {llama2.c: Inference {Llama} 2 in One File of Pure {C}},
  howpublished = {\url{https://github.com/karpathy/llama2.c}},
  year         = {2023},
  note         = {Accessed: 2026-10-13}
}

% verified: https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K (files stories260K.bin/.pt, tok512.bin/.model present)
@misc{karpathy2023tinyllamas,
  author       = {Karpathy, Andrej},
  title        = {tinyllamas: {Llama} 2 Architecture Models Trained on {TinyStories} (incl. stories260K)},
  howpublished = {\url{https://huggingface.co/karpathy/tinyllamas}},
  year         = {2023},
  note         = {Accessed: 2026-10-13}
}

% verified: https://github.com/karpathy/nanochat (created 2025-10-13); nanochat/gpt.py docstring lists: rotary embeddings, QK norm,
%           untied embeddings, relu^2 MLP, norm after token embedding, parameter-free rmsnorm, no bias, GQA
@misc{karpathy2025nanochat,
  author       = {Karpathy, Andrej},
  title        = {nanochat: The Best {ChatGPT} That \$100 Can Buy},
  howpublished = {\url{https://github.com/karpathy/nanochat}},
  year         = {2025},
  note         = {Accessed: 2026-10-13}
}

% verified: blog post https://karpathy.github.io/2026/02/12/microgpt/ (dated Feb 12, 2026) and
%           gist https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 (page title "microgpt · GitHub");
%           companion build-up gist https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a (created 2026-02-13)
%           Code comment: "Follow GPT-2 ... with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU"
@misc{karpathy2026microgpt,
  author       = {Karpathy, Andrej},
  title        = {microgpt},
  howpublished = {Blog post, \url{https://karpathy.github.io/2026/02/12/microgpt/}; code: \url{https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95}},
  month        = feb,
  year         = {2026},
  note         = {Accessed: 2026-10-13}
}

% =====================================================================
% E. Architecture-modification / hardware-cost studies
% =====================================================================

% verified: https://doi.org/10.18653/v1/2021.emnlp-main.465 (Crossref; published author name "Liam Fedus") ; https://arxiv.org/abs/2102.11972
@inproceedings{narang2021transformer,
  author    = {Narang, Sharan and Chung, Hyung Won and Tay, Yi and Fedus, Liam and Fevry, Thibault and Matena, Michael and Malkan, Karishma and Fiedel, Noah and Shazeer, Noam and Lan, Zhenzhong and Zhou, Yanqi and Li, Wei and Ding, Nan and Marcus, Jake and Roberts, Adam and Raffel, Colin},
  title     = {Do Transformer Modifications Transfer Across Implementations and Applications?},
  booktitle = {Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {5758--5773},
  year      = {2021},
  doi       = {10.18653/v1/2021.emnlp-main.465}
}

% verified: https://arxiv.org/abs/2302.14017 (journal_ref: ASSYST workshop at ISCA 2023)
@misc{kim2023fullstack,
  author        = {Kim, Sehoon and Hooper, Coleman and Wattanawong, Thanakul and Kang, Minwoo and Yan, Ruohan and Genc, Hasan and Dinh, Grace and Huang, Qijing and Keutzer, Kurt and Mahoney, Michael W. and Shao, Yakun Sophia and Gholami, Amir},
  title         = {Full Stack Optimization of Transformer Inference: a Survey},
  howpublished  = {arXiv:2302.14017; presented at the ASSYST Workshop at ISCA 2023},
  year          = {2023},
  eprint        = {2302.14017},
  archivePrefix = {arXiv}
}

% verified: https://doi.org/10.18653/v1/2020.acl-main.686 (Crossref) ; https://arxiv.org/abs/2005.14187
@inproceedings{wang2020hat,
  author    = {Wang, Hanrui and Wu, Zhanghao and Liu, Zhijian and Cai, Han and Zhu, Ligeng and Gan, Chuang and Han, Song},
  title     = {{HAT}: Hardware-Aware Transformers for Efficient Natural Language Processing},
  booktitle = {Proc. 58th Annu. Meeting of the Association for Computational Linguistics (ACL)},
  pages     = {7675--7688},
  year      = {2020},
  doi       = {10.18653/v1/2020.acl-main.686}
}
```

---

## 2. What each work does and how it relates to this paper

### A. Integer-only / quantized inference

- **jacob2018integer**: Introduces affine (scale + zero-point) INT8 quantization with INT32 accumulation and a fixed-point "multiplier + shift" requantization, so inference needs integer arithmetic only. It's the direct ancestor of our requantization datapath. The difference is that it targets CNNs on ARM/DSP, while we extend integer-only execution to every nonlinearity in nine LM architectures.
- **krishnamoorthi2018quantizing**: A practical guide to post-training and quantization-aware INT8 quantization: per-channel weights, per-layer activations, and a recommendation that accelerators support 4/8/16-bit. Our INT8 weight/activation + INT16 residual choice follows that "8/16-bit" advice. The whitepaper is CNN-only.
- **wu2020integer**: NVIDIA's systematic study of quantization parameters and calibrators (max, entropy, and 99.9–99.9999 % percentile) across vision, speech and BERT. It's the main citation for our percentile calibration. Our contribution is applying the comparison to tiny autoregressive LMs and attributing error per tensor class.
- **nagel2021white**: Qualcomm's white paper on PTQ/QAT pipelines, range setting and debugging. It's a good general reference for our calibration and error-attribution methodology. It doesn't address integer-only nonlinearities or LMs.
- **kim2021ibert**: The closest algorithmic prior work. It runs BERT/RoBERTa end-to-end in integer arithmetic, with polynomial i-GELU, i-exp softmax and Newton-iteration integer sqrt for LayerNorm. We should credit it explicitly for integer LayerNorm/softmax. Our differences are integer RMSNorm, RoPE, SwiGLU, ReLU², QK-norm, GRU/RNN gates, proven error bounds, and a bit-exact RTL implementation rather than GPU TensorRT kernels.
- **dettmers2022llmint8**: Shows that large-magnitude outlier activation features emerge in LLMs at around 6.7B parameters and handles them with mixed-precision decomposition. It's relevant to our finding that activations, not weights, dominate INT8 error. Our models are 4–6 orders of magnitude smaller, so the mechanism may differ, and we should say so.
- **xiao2023smoothquant**: Moves quantization difficulty from activations to weights with per-channel smoothing, enabling W8A8 LLMs. It's a natural alternative or complement to our percentile calibration, and the error-attribution section should mention or compare against it.
- **frantar2023gptq**: Second-order, one-shot, weight-only 3/4-bit quantization (published at ICLR as "OPTQ"). Since we find weights are not the dominant error source in tiny LMs, GPTQ-style weight-only methods address the smaller term. Cite it as a contrast.
- **zafrir2019q8bert**: Quantization-aware INT8 fine-tuning of BERT's GEMM/embedding layers. Nonlinearities stay in floating point, which is a useful foil for "integer-only".
- **migacz2017tensorrt**: The TensorRT GTC talk that introduced KL-divergence (entropy) calibration for INT8 activations. It's the canonical citation for entropy calibration next to percentile calibration. The original URL is dead, so the BibTeX URL points to the Internet Archive copy.
- **mckinstry2019discovering**: Fine-tuning low-precision networks after quantization, with activation ranges set from the 99.99th percentile (confirmed in the paper). It's an early percentile-calibration reference.
- **bondarenko2021understanding**: Finds that transformer quantization difficulty comes from high-dynamic-range activations, with structured outliers in the residual connections. This directly motivates our INT16 residual stream and activation-side error attribution, so cite it as supporting evidence rather than claiming the observation as new.
- **hu2024illm**: Integer-only PTQ for LLMs, including integer RMSNorm ("DI-Normalization"), DI-Exp and DI-ClippedSoftmax. It's the closest prior work for integer RMSNorm/softmax in Llama-style models. It is GPU/algorithm-level, with no hardware or error bounds.

### B. Transformer/LLM accelerators, overlays, edge LLM on FPGA

- **li2020ftrans**: FPGA transformer accelerator built on block-circulant weight compression. It's a fixed-function design for one model family, while ours is a programmable engine across nine families.
- **ham2020a3**: An ASIC accelerator that approximates attention by candidate selection. It optimizes only the attention kernel, while we measure attention's whole-system cost in the feature ladder.
- **wang2021spatten**: ASIC with cascade token/head pruning and progressive quantization for attention. It's algorithm–architecture co-design at large-model scale, where we keep model semantics exact (bit-exact to the reference).
- **ham2021elsa**: ASIC that filters irrelevant query–key pairs with a lightweight approximation. Like A³, it's attention-only and approximate.
- **hong2022dfx**: A four-U280 multi-FPGA appliance for end-to-end GPT-2 generation, whose compute cores run custom instructions. It's the closest instruction-driven GPT-2 FPGA work, and shows that instruction-based GPT-2 on FPGA is not new. Our differences are a single low-cost Zynq-7020, integer-only (not FP16) arithmetic, and multiple architectures.
- **zeng2024flightllm**: An LLM (LLaMA2-7B) mapping flow for U280/VHK158, with a configurable sparse DSP chain, always-on-chip decode and length-adaptive compilation. It's state of the art for large models and is datacenter-class.
- **chen2025understanding**: Analytical model and HLS kernel library for spatial (dataflow) vs. temporal (overlay) LLM accelerators on U280. It's useful for justifying our temporal, microcoded choice on a small device, where a spatial design of several architectures would not fit.
- **he2024hlstransform**: **The closest prior work.** It builds on llama2.c, runs a 110M TinyStories Llama 2 model with llama2.c's INT8 forward pass, and uses HLS on a Virtex UltraScale+ VU9P. In llama2.c's int8 scheme only the matmul weights and activations are int8, and the other parameters (e.g., RMSNorm) stay fp32 (llama2.c README). We differ in being integer-only, handwritten SystemVerilog, on a $100-class Zynq-7020, and covering eight more architectures.
- **xu2024llamaf**: A post-training-quantized TinyLlama-1.1B accelerator on a ZCU102 (Zynq UltraScale+), with a pipelined matrix–vector engine. It's an embedded-FPGA Llama 2 architecture on a larger device and model, focused on matmul.
- **haris2024secdallm**: SECDA-LLM design platform that offloads llama.cpp block-floating-point MatMul to a **PYNQ-Z1** (TinyLlama, about 1.7 s/token). It's the same board class as ours, but only the MatMul is offloaded and it isn't integer-only.
- **wang2025speedllm**: A short HPDC'25 paper with a U280 accelerator for a TinyLlama/llama2.c-style model, using operator fusion and memory reuse. Search results say it uses stories15M; I didn't confirm that in the paper. It's a datacenter card and a single model.
- **qiao2026tellme**: Ternary-weight (1.58-bit) / 8-bit-activation LLM accelerator for edge FPGAs (AMD KV260), supporting prefill and decode. It's recent edge-LLM-on-FPGA state of the art at FPGA'26. It uses a different numeric regime and a single model family.
- **hao2017drnn**: A deep RNN language model accelerated on a **PYNQ (XC7Z020)** through an AXI-Stream overlay. It's prior art for an LM on exactly our chip, but RNN-only and not integer-exact or multi-architecture.
- **guo2018angeleye**: Programmable, instruction-driven CNN accelerator with an 8-bit quantization strategy and compiler, evaluated on Zynq XC7Z020/XC7Z045. It's the classic "compiler + instruction stream on embedded Zynq" design, which we extend from CNNs to LMs.
- **fowers2018brainwave**: Brainwave NPU, a soft DNN processor with a single-threaded SIMD ISA and model-specialized precision, serving RNNs on Stratix 10. It's the canonical "microcoded/ISA overlay for sequence models" reference, at cloud scale.
- **yu2020opu**: Instruction-set overlay processor for CNNs, so new networks need recompilation rather than re-synthesis. It's the same "one bitstream, many models" philosophy as ours, for CNNs.
- **khan2021npe**: An overlay processor for NLP (BERT) that is software-programmable and upgradable to new NLP models without reconfiguration. It's the most direct precedent for our "single engine, many models" claim. The FPGA'21 entry is abstract-length, with the full paper on arXiv.
- **moreau2019vta**: VTA, a parametrizable accelerator with a two-level (task + microcode) ISA and a TVM JIT compiler, demonstrated on edge-class FPGAs. It's the closest open-source analogue of our microcode + compiler stack, but it's CNN-oriented and uses integer GEMM plus ALU ops.
- **duarte2018hls4ml**: hls4ml generates fully spatial, low-latency, per-model HLS designs. It's the opposite design point to our time-multiplexed overlay, and useful for contrast.
- **li2022autovitacc**: An FPGA (FPL'22) ViT accelerator with mixed-scheme quantization. It's an FPL-community example of quantized-transformer FPGA work, but vision-only and a single architecture.

### C. Integer/approximate nonlinearities; RNN FPGA accelerators

- **stevens2021softermax**: Base-2 softmax with low-precision arithmetic and online normalization, in an ASIC. Our integer softmax should be compared against its base-replacement and online-normalization ideas.
- **li2023ivit**: Integer-only ViT with Shiftmax, ShiftGELU and I-LayerNorm (bit-shift based). It's a close precedent for shift-based integer softmax and LayerNorm, but it runs on GPU via TVM and comes without error bounds.
- **lin2022fqvit**: Power-of-Two Factor quantization for LayerNorm inputs and Log-Int-Softmax. It's another fully-quantized-transformer recipe for the same two operators.
- **yu2022nnlut**: Replaces GELU, softmax and LayerNorm with small-NN-derived LUTs. It's an alternative, LUT-based hardware strategy that we can contrast with our arithmetic integer kernels.
- **wang2023sole**: E2Softmax (log2-quantized exponent, log-based division) and AILayerNorm (low-precision statistics), without retraining. It's the most recent hardware co-design for exactly softmax and LayerNorm.
- **han2017ese**: Engine for a pruned and quantized (20x compressed) sparse LSTM on a Xilinx XCKU060 FPGA, using load-balance-aware pruning. It's the classic RNN-on-FPGA reference for our RNN/GRU members.
- **wang2018clstm**: Block-circulant/FFT-compressed LSTM on FPGA. It's another recurrent-FPGA baseline. Both C-LSTM and ESE target speech at datacenter-FPGA scale.

### D. Models and components

- **bengio2003neural**: The neural probabilistic (embedding + MLP) LM. It's the basis of makemore's MLP model.
- **mikolov2010recurrent**: The RNN language model, our RNN member.
- **cho2014learning**: Introduces the gated recurrent unit (GRU), our GRU member. Note that the makemore README cites a different Cho et al. 2014 paper for its GRU (arXiv:1409.1259, "On the Properties of Neural Machine Translation: Encoder–Decoder Approaches"). D14-1179 is the paper that introduces the gated unit, so it's the standard GRU citation.
- **vaswani2017attention**: The Transformer, the base of the GPT/Llama members.
- **radford2019language**: GPT-2, the reference architecture implemented in minGPT/nanoGPT and the bottom of our feature ladder.
- **touvron2023llama2**: Llama 2, the top of the ladder (RMSNorm, RoPE, SwiGLU, GQA). llama2.c implements this architecture.
- **su2024roformer**: Rotary position embedding (RoPE), one ladder step.
- **zhang2019rmsnorm**: RMSNorm, one ladder step and an integer-kernel target.
- **shazeer2020glu**: GLU variants including SwiGLU, one ladder step.
- **ainslie2023gqa**: Grouped-query attention, one ladder step.
- **so2021primer**: Primer, where architecture search found squared ReLU (ReLU²), used in nanochat.
- **henry2020qknorm**: QK-norm, which normalizes queries and keys before the dot product. We can co-cite **dehghani2023scaling** (ViT-22B), which popularized QK-layernorm for training stability at scale. nanochat uses QK norm.
- **eldan2023tinystories**: The TinyStories dataset, on which stories260K (llama2.c / tinyllamas) is trained.
- **karpathy\***: The educational source implementations: makemore (README lists Bigram, MLP, CNN, RNN, LSTM, GRU and Transformer; `makemore.py` also defines a causal BoW model), minGPT/nanoGPT (GPT-2), llama2.c and tinyllamas (Llama 2, stories260K), nanochat (RoPE, QK-norm, ReLU², parameter-free RMSNorm, no bias, GQA support), and microgpt (Feb 2026; GPT-2-like with RMSNorm, no biases, ReLU).

### E. Architecture-modification / hardware-cost studies

- **narang2021transformer**: Re-implements many Transformer modifications in one shared setting and finds that "most modifications do not meaningfully improve performance". Most of the beneficial ones "were either developed in the same codebase ... or are relatively minor changes" (abstract). It's the methodological model for our controlled feature ladder. It measures quality only, at T5 scale; we add per-feature hardware cost (cycles, LUT/DSP/BRAM) under an equal parameter budget, at tiny scale.
- **kim2023fullstack**: Survey plus a Gemmini case study that explicitly analyzes how nonlinear ops (LayerNorm, softmax, GELU) affect hardware design. It supports our motivation that "architectural innovations have a hardware price".
- **wang2020hat**: Hardware-aware NAS for Transformers using measured latency on CPU, GPU and Raspberry Pi. It's a related "architecture vs. hardware cost" study, but it searches over size/shape hyperparameters rather than isolating specific innovations, and it doesn't involve FPGAs.

---

## 3. Positioning the novelty (honest version)

**What is not new, and should be credited up front:**

- Integer-only transformer inference, including integer softmax and LayerNorm, is established (I-BERT, I-ViT, FQ-ViT). Integer RMSNorm/softmax for Llama-style LLMs also exists (I-LLM).
- Hardware-friendly softmax/LayerNorm units exist (Softermax, SOLE, NN-LUT).
- Programmable, instruction-driven FPGA overlays that run many networks from one bitstream are a long line of work: Angel-Eye (on the same XC7Z020), OPU, VTA, Brainwave, and for NLP, NPE.
- Instruction-driven GPT-2 on FPGA exists (DFX).
- llama2.c / TinyStories-scale Llama 2 has already been put on FPGAs (HLSTransform, and SpeedLLM if its stories15M use is confirmed), as has TinyLlama on embedded and PYNQ boards (LlamaF, SECDA-LLM).
- An RNN LM has already run on a PYNQ XC7Z020 (Hao & Quigley).
- Activations being harder to quantize than weights, residual-stream outliers, and percentile/entropy calibration are all known (Bondarenko, LLM.int8(), SmoothQuant, Wu et al., McKinstry, Migacz).

**What appears to be new (no prior work found that combines these):**

1. **Breadth on one integer engine.** A single microcoded INT8/INT16 engine plus compiler executes nine architecturally distinct LM families: n-gram/bigram, MLP, BoW, RNN, GRU, GPT-2, a GPT-2-like microgpt, Llama 2 with a real pretrained checkpoint (stories260K), and a nanochat-style transformer. The engine is bit-exact against a Python integer reference, on a low-cost Zynq-7020. Prior overlays are CNN-only, or BERT/GPT-2-only. Prior llama2.c/TinyLlama FPGA designs implement one model, keep floating point outside the matmuls (llama2.c's int8 path keeps RMSNorm etc. in fp32), and use larger devices (VU9P, U280, ZCU102) — except SECDA-LLM, whose PYNQ-Z1 offloads only MatMul.
2. **A controlled hardware-cost-vs-quality ladder.** Each step adds one innovation (attention, RMSNorm, RoPE, SwiGLU, GQA, QK-norm, ReLU²) at an equal parameter budget. Narang et al. ran this kind of controlled ablation for quality only. HAT and Kim et al. relate architecture to latency but don't isolate these innovations or report FPGA resources. We didn't find a study reporting cycles/LUT/DSP/BRAM per innovation; this is the paper's strongest and most defensible claim.
3. **Integer nonlinear kernels with proven error bounds, verified bit-exact in RTL.** I-BERT, I-ViT and I-LLM give integer approximations evaluated empirically. Frame the novelty as the bounds plus covering the "modern" operator set (RMSNorm, RoPE, SwiGLU, QK-norm, ReLU²) in one datapath, not as "first integer LayerNorm/softmax".
4. **Quantization-error attribution at tiny scale.** Present it as confirming and quantifying, for 10^5–10^7-parameter LMs, an effect known for BERT/LLMs, plus a practical calibration recipe. Avoid claiming the activation-dominance finding itself is novel.

**Suggested framing sentence:** "Unlike prior FPGA LLM accelerators that specialize hardware for one large model and retain floating-point nonlinearities, and unlike prior integer-only transformer schemes that are evaluated in software, we provide a single bit-exact integer engine that spans nine generations of language-model architectures on a low-cost SoC FPGA. We use it to measure, under a fixed parameter budget, what each architectural innovation costs in cycles and FPGA resources."

**Caveat to state in the paper:** the results are at tiny scale (stories260K-class). Narang et al. show that architecture conclusions may not transfer across scales, and LLM.int8() shows outlier behaviour changes with scale.

**Suggested core subset for a 6–8 page IEEE paper (about 32 references):** jacob2018integer, wu2020integer, kim2021ibert, hu2024illm, li2023ivit, dettmers2022llmint8, xiao2023smoothquant, bondarenko2021understanding, migacz2017tensorrt, stevens2021softermax, wang2023sole, hong2022dfx, zeng2024flightllm, chen2025understanding, he2024hlstransform, xu2024llamaf, haris2024secdallm, qiao2026tellme, khan2021npe, guo2018angeleye, moreau2019vta, fowers2018brainwave, narang2021transformer, kim2023fullstack, vaswani2017attention, radford2019language, touvron2023llama2, su2024roformer, zhang2019rmsnorm, shazeer2020glu, ainslie2023gqa, so2021primer, henry2020qknorm, eldan2023tinystories, plus the Karpathy @misc entries (cite them together in one bracket). Drop or merge the rest as space requires (e.g., Bengio, Mikolov and Cho can go into one sentence about the makemore members).

---

## 4. Submission deadlines (checked 2026-10-13)

"Official" means read on the conference's own site or CFP. "Estimate" means extrapolated from the most recent cycle, not an announced date.

| Venue | Edition / where & when | Paper deadline(s) | Status as of 2026-10-13 | Source |
|---|---|---|---|---|
| **ISCAS 2027** | Bordeaux, France, 6–9 Jun 2027 | Regular papers **13 Oct 2026**; notification 11 Jan 2027; final 27 Jan 2027. Special-session papers 22 Oct 2026 (only for accepted special sessions). Live demos 13 Oct 2026 ("NO further extensions" for demos). | **Official. Due today**; no regular-paper extension posted yet. | https://2027.ieee-iscas.org/call-for-papers ; CFP PDF https://ieee-cas.org/files/ieeecass/2026-07/iscas2027-cfp.pdf |
| **DATE 2027** | Dresden, Germany, 22–24 Mar 2027 | Regular: abstract 13 Sep 2026, full paper 20 Sep 2026 (passed). **Late Breaking Results (extended abstract): 29 Nov 2026**, notification 13 Jan 2027. PhD Forum 30 Nov 2026. "No extensions". | Official. Main track closed; **LBR track open**. | https://www.date-conference.com/call-for-papers ; https://date27.date-conference.com/date-2027-call-papers |
| **ACM/SIGDA FPGA 2027** (35th ISFPGA) | California, USA, 14–16 Mar 2027 | Abstracts 1 Oct 2026, full papers 8 Oct 2026 ("No Extensions"); notification 24 Nov 2026. | Official. **Closed.** The next chance is FPGA 2028 (estimate: early Oct 2027). | https://www.isfpga.org/ |
| **FCCM 2027** (35th) | Provo, Utah, USA, 9–12 May 2027 | Site lists all deadlines as "TBD". **Estimate from FCCM 2026:** abstract ~10 Jan 2027, full paper ~17 Jan 2027 (2026 cycle: abstract 10 Jan 2026, full 17 Jan 2026, notification 16 Mar 2026, "No Extensions"). | Event announced; **deadline not yet announced (estimate)**. | https://www.fccm.org/ ; 2026 cycle https://www.fccm.org/past/2026/ |
| **FPL 2027** | Not yet announced (2027.fpl.org didn't resolve) | **Estimate from FPL 2026** (Ghent, 7–11 Sep 2026): CFP text says abstract 20 Mar 2026, full paper 27 Mar 2026, notification 1 Jun 2026. The EasyChair header shows abstract 27 Mar / submission 3 Apr 2026, which looks like an extension. Long papers 8 pp + refs, short 5 pp + refs (per third-party listing; not confirmed on the official page). Expect late Mar to early Apr 2027. | **Estimate only** (previous-year cycle). | https://easychair.org/cfp/fpl2026 ; https://2026.fpl.org/ |
| **FPT 2026** | Honolulu, Hawaii, USA, 7–10 Dec 2026 | Title/abstract and full paper 19 Jun 2026 (extended from 5/12 Jun); decisions 21 Aug 2026; camera-ready 15 Oct 2026. TRETS journal track 1 May 2026. | Official. **Closed.** | https://fpt2026.uark.edu/ |
| **FPT 2027** | Not announced | **Estimate from FPT 2026:** about early–mid Jun 2027 (journal track about 1 May 2027). | **Estimate only.** | (based on https://fpt2026.uark.edu/) |
| **SIU 2027** (IEEE Sinyal İşleme ve İletişim Uygulamaları Kurultayı, 35th) | Not announced (no 2027 site or call found) | **Estimate from SIU 2026** (Piri Reis Univ., Tuzla/İstanbul, 7–10 Jul 2026): paper deadline 29 Mar 2026, notification 20 May 2026, camera-ready and registration 5 Jun 2026. Max **4 pages**. Papers must be in **Turkish** (English accepted only if a co-author is not a native Turkish speaker). SIU 2026 also ran a **Lisans Proje Yarışması** (undergraduate project competition): application 16 Jun 2026, for final-year students' 2025–26 projects (technical report PDF + optional ≤3 min video; poster). Expect paper deadline about late Mar 2027. | **Estimate only** (previous-year cycle). | https://siu2026.pirireis.edu.tr/yazarlar-icin/bildiri-cagrisi/ ; https://siu2026.pirireis.edu.tr/program/lisans-proje-yarismasi/ |

Practical reading, as of 13 Oct 2026:
- ISCAS 2027 closes today.
- The DATE 2027 Late Breaking Results track (29 Nov 2026) is the nearest open option.
- After that, likely FCCM 2027 (~mid-Jan 2027, est.), then SIU 2027 (~late Mar 2027, est.; 4 pp, Turkish) and FPL 2027 (~late Mar–early Apr 2027, est.).
- Then FPT 2027 (~Jun 2027, est.) and FPGA 2028 (~early Oct 2027, est.).

---

## 5. Unverified / not included

Each item below was looked for but could not be verified to the required standard, or is flagged with a caveat.

1. **A dedicated "TinyStories FPGA" / "llama2.c FPGA" / "tiny LM on Zynq-7020" peer-reviewed paper beyond those listed:** none found. Searched arXiv full-text for TinyStories+FPGA, llama2.c+FPGA and Zynq+language model, plus web searches. HLSTransform (arXiv only, not peer-reviewed as far as found) and SpeedLLM are the closest.
2. **SpeedLLM using stories15M specifically:** stated in a web-search summary, but the arXiv abstract only says "Tinyllama framework" / "Llama2 operator fusion". The citation itself is verified (HPDC'25, 3 pp); the stories15M detail is not.
3. **SECDA-LLM peer-reviewed venue:** only the arXiv version (2408.00462) was confirmed. The Crossref lookup was rate-limited, so a workshop or conference version may exist and was not found.
4. **NPE full paper:** the FPGA'21 DOI resolves to a one-page entry (p. 227), most likely an abstract or poster. The full paper is arXiv:2104.06535, so cite both, as done.
5. **Migacz 2017 TensorRT slides:** the original NVIDIA URL now redirects to the On-Demand homepage. Verified only via the Internet Archive snapshot (2024-07-28). Kept in the BibTeX with the archive URL.
6. **Vaswani et al. 2017 and Zhang & Sennrich 2019 page numbers:** not shown on the proceedings pages I read, so omitted. Commonly cited pages (e.g., 5998–6008 for Vaswani) are **not** included because I didn't verify them.
7. **"Hardware Accelerator for Multi-Head Attention and Position-Wise Feed-Forward in the Transformer"** (Lu, Wang, Liang, Lin, Wang, IEEE SOCC 2020, DOI 10.1109/SOCC49529.2020.9524802): Crossref metadata verified but not added. It's optional if you want one more early FPGA transformer accelerator.
8. **arXiv 2607.01798 ("Approximate Attention Weighting for Sustainable FPGA-Based Vision Transformer Inference")**, reportedly a low-precision softmax on a Zynq-7020: seen only in a search snippet, not opened. Not included.
9. **GitHub taoFPGA/accelerator** (Bar-Ilan B.Sc. capstone: INT8 transformer block on PYNQ-Z2): seen only in a search snippet. It's not a publication and is not included. It might be worth a look as a same-board, student-project comparison point.
10. **TeLLMe published version board:** the arXiv abstract names the AMD KV260. I didn't open the FPGA'26 version to confirm it's unchanged.
11. **Angel-Eye on XC7Z020:** comes from a search-result summary of the paper and of reviews. The bibliographic record is verified via Crossref, but I didn't open the full text.
12. **FPL 2026 page limits (8 + refs / 5 + refs):** from a third-party listing, not the official site (2026.fpl.org is JavaScript-rendered and returned no text).
13. **FPGA 2027 city:** isfpga.org says only "California, USA". Third-party listings (and the FPGA'26 Crossref record for 2026) mention Seaside, CA, but this is not confirmed for 2027.
14. **SIU 2027, FPL 2027, FPT 2027:** no official 2027 information exists yet. All dates in the table for these are estimates from the previous cycle.
