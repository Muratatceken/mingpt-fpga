# Reproducible x86-64/arm64 Linux environment for the software and simulation flows (not Vivado).
#   docker build -t mingpt-fpga . && docker run --rm -it -v $PWD:/work -w /work mingpt-fpga bash
#   make zoo-sim            # bit-exact RTL simulation of every exported model
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      verilator iverilog yosys g++ make git python3 python3-pip python3-venv ca-certificates \
      texlive-xetex texlive-latex-extra texlive-fonts-extra texlive-science latexmk \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir numpy sentencepiece
ENV PATH=/opt/venv/bin:$PATH
