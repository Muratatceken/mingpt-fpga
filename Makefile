# minGPT -> FPGA : local flow (Python + Verilator). Vivado flow: see scripts/*.tcl
PYTHON    ?= python3
VERILATOR ?= verilator
NSEQ      ?= 729

VFLAGS = --binary --timing --timescale 1ns/1ps -j 0 -Wall -Wno-fatal \
         -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM
RTL    = $(shell cat scripts/files.f)

.PHONY: all model train quantize eval export sim test-arith selftest lint synth report gelisim clean \
        zoo-train zoo-quant zoo-sim zoo-synth zoo-pipeline zoo-fuzz zoo-arith zoo-dse paper \
        nexys nexys-sim nexys-all nexys-fuzz nexys-icarus nexys-netlist nexys-synth nexys-bit nexys-prog nexys-pnr xc7-toolchain lowbit sunum-yerel

all: test-arith sim selftest

# ---------------- software (golden model) ----------------
model: train quantize export

train:
	cd sw && $(PYTHON) train.py

quantize:
	cd sw && $(PYTHON) quantize.py

eval:
	cd sw && $(PYTHON) evaluate.py

export:
	cd sw && $(PYTHON) export.py

# ---------------- RTL simulation (Verilator) ----------------
build/obj_top/Vtb_top: $(RTL) tb/tb_gpt_top.sv mem/tv.txt
	$(VERILATOR) $(VFLAGS) $(RTL) tb/tb_gpt_top.sv --top-module tb_gpt_top -Mdir build/obj_top -o Vtb_top

build/obj_arith/Vtb_arith: rtl/requant.sv rtl/div_seq.sv rtl/isqrt_seq.sv tb/tb_arith.sv
	$(VERILATOR) $(VFLAGS) $^ --top-module tb_arith -Mdir build/obj_arith -o Vtb_arith

build/obj_selftest/Vtb_selftest: $(RTL) tb/tb_selftest.sv
	$(VERILATOR) $(VFLAGS) $(RTL) tb/tb_selftest.sv --top-module tb_selftest -Mdir build/obj_selftest -o Vtb_selftest

sim: build/obj_top/Vtb_top
	./build/obj_top/Vtb_top +NSEQ=$(NSEQ)

test-arith: build/obj_arith/Vtb_arith
	./build/obj_arith/Vtb_arith

selftest: build/obj_selftest/Vtb_selftest
	./build/obj_selftest/Vtb_selftest

# ---------------- açık kaynak sentez (Yosys, kaynak tahmini) ----------------
synth:
	scripts/yosys_synth.sh gpt_top

# ---------------- Karpathy model ailesi (sw/zoo, rtl/zoo) ----------------
zoo-train:
	cd sw/zoo && $(PYTHON) train.py

zoo-quant:
	cd sw/zoo && $(PYTHON) check_ir.py && $(PYTHON) evaluate.py && $(PYTHON) stories.py && $(PYTHON) export_hw.py

zoo-sim:
	cd sw/zoo && $(PYTHON) hw_run.py

zoo-synth:
	cd sw/zoo && $(PYTHON) hw_run.py --synth

zoo-pipeline:      # every trained run: INT8 + bit-exact RTL on the test set -> model/zoo/pipeline.json
	cd sw/zoo && $(PYTHON) pipeline.py

zoo-fuzz:          # randomized differential tests RTL vs integer reference -> model/zoo/fuzz.json
	cd sw/zoo && $(PYTHON) fuzz.py 60

zoo-arith:         # exhaustive / random tests of the divider and square root
	mkdir -p build/zoo/obj_zarith && $(VERILATOR) --binary --timing --assert --timescale 1ns/1ps -j 0 -Wno-fatal \
	  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC rtl/zoo/zdiv.sv rtl/isqrt_seq.sv tb/tb_zarith.sv --top-module tb_zarith \
	  -Mdir build/zoo/obj_zarith -o Vtb && build/zoo/obj_zarith/Vtb

zoo-dse:           # GEMV lane count P = 8, 16, 32 -> model/zoo/dse.json
	cd sw/zoo && $(PYTHON) dse.py

# ---------------- Nexys A7-100T: the model runs on the FPGA (board/nexys_a7, docs/05_nexys_a7.md) ----------------
MODEL ?= stories260k
NEXYS_MODELS = bigram mlp rnn gru bow gpt2 microgpt llama nanochat stories260k ts_gpt2 ts_gru ts_llama ts_microgpt ts_nanochat
MHZ ?= 50

nexys:             # model images in BRAM form: mem/nexys/$(MODEL)/
	cd sw/zoo && $(PYTHON) board.py export $(MODEL)

nexys-sim: nexys  # whole board design in Verilator, bit-exact vs sw/zoo/board.py, VGA frame -> build/nexys/$(MODEL)/frame.png
	cd sw/zoo && $(PYTHON) board_sim.py $(MODEL) --press CUD --sw1 1 --frame

nexys-all:         # every model of the family
	for m in $(NEXYS_MODELS); do (cd sw/zoo && $(PYTHON) board.py export $$m && $(PYTHON) board_sim.py $$m --press CUD --sw1 1) || exit 1; done

nexys-fuzz:        # random switches + key sequences (incl. presses during a run, CPU_RESET), directed corner cases
	cd sw/zoo && $(PYTHON) board_fuzz.py && $(PYTHON) board_fuzz.py --directed && $(PYTHON) board_fuzz.py --race

nexys-icarus:      # 4-state (sv2v + Icarus) board simulation with X checks on every output pin
	scripts/nexys_icarus.sh $(MODEL)

nexys-netlist:     # Yosys netlist of the real top (MMCM, BRAM; multipliers in LUTs) simulated cycle-exactly vs the RTL
	scripts/nexys_netlist_sim.sh $(MODEL)

nexys-synth: nexys # Yosys resource estimate of the board design (no timing)
	scripts/nexys_synth.sh $(MODEL)

nexys-bit: nexys   # Vivado (Linux/Windows): bitstream -> build/nexys/$(MODEL)/vivado/nexys_$(MODEL).bit
	vivado -mode batch -source scripts/vivado_nexys_a7.tcl -tclargs $(MODEL) $(MHZ)

nexys-pnr: nexys   # open-source on the Mac (Yosys + nextpnr-himbaechel + Project X-Ray): Fmax + bitstream -> build/nexys/$(MODEL)/pnr/
	scripts/nexys_pnr.sh $(MODEL) $(or $(MHZ),50)

xc7-toolchain:     # build nextpnr-himbaechel (xc7a100t), xc7frames2bit and fasm2frames in ~/tools/xc7 (once)
	scripts/xc7_toolchain.sh

nexys-prog:        # program the board over USB-JTAG (Vivado); without Vivado: openFPGALoader -b nexys_a7_100 <bit>
	vivado -mode batch -source scripts/vivado_program_nexys.tcl -tclargs $(MODEL)

lowbit:            # low-bit weights study (Larq-style binary/ternary, LittleBit): QAT on the Mac
	cd sw/zoo && $(PYTHON) lowbit.py sweep gpt2 llama nanochat gru mlp && $(PYTHON) lowbit.py table

# ---------------- makale (İngilizce, IEEE) ----------------
paper:
	cd sw/zoo && $(PYTHON) paper_data.py
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

# ---------------- PDF rapor (XeLaTeX) ----------------
report:
	cd sw && $(PYTHON) report_data.py
	cd sw/zoo && $(PYTHON) report_zoo.py
	cd report && latexmk -xelatex -interaction=nonstopmode -outdir=build rapor.tex && cp build/rapor.pdf rapor.pdf

gelisim:           # tarihsel gelişim raporu -> report/gelisim/gelisim_raporu.pdf (tablolar model/zoo/*.json ve git log'dan)
	cd report/gelisim && $(PYTHON) veri.py && latexmk -xelatex -interaction=nonstopmode -outdir=build gelisim_raporu.tex \
	  && cp build/gelisim_raporu.pdf gelisim_raporu.tmp.pdf && mv gelisim_raporu.tmp.pdf gelisim_raporu.pdf

lint:
	$(VERILATOR) --lint-only -Wall -Wno-fatal -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL $(RTL) --top-module gpt_top

clean:
	rm -rf build/obj_* build/xsim build/vivado* report/build

sunum-yerel:       # implementasyon sunumu yerelde: sunum/uygulama/{sunum.html, sunum.pdf, konusma_metni.md, .pdf}
	python3 sunum/yerel_sunum.py sunum/uygulama --pdf
