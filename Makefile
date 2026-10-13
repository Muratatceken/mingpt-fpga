# minGPT -> FPGA : local flow (Python + Verilator). Vivado flow: see scripts/*.tcl
PYTHON    ?= python3
VERILATOR ?= verilator
NSEQ      ?= 729

VFLAGS = --binary --timing --timescale 1ns/1ps -j 0 -Wall -Wno-fatal \
         -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM
RTL    = $(shell cat scripts/files.f)

.PHONY: all model train quantize eval export sim test-arith selftest lint synth report clean \
        zoo-train zoo-quant zoo-sim zoo-synth zoo-pipeline zoo-fuzz zoo-arith zoo-dse paper

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

# ---------------- makale (İngilizce, IEEE) ----------------
paper:
	cd sw/zoo && $(PYTHON) paper_data.py
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

# ---------------- PDF rapor (XeLaTeX) ----------------
report:
	cd sw && $(PYTHON) report_data.py
	cd sw/zoo && $(PYTHON) report_zoo.py
	cd report && latexmk -xelatex -interaction=nonstopmode -outdir=build rapor.tex && cp build/rapor.pdf rapor.pdf

lint:
	$(VERILATOR) --lint-only -Wall -Wno-fatal -Wno-WIDTHEXPAND -Wno-UNUSEDSIGNAL $(RTL) --top-module gpt_top

clean:
	rm -rf build/obj_* build/xsim build/vivado* report/build
