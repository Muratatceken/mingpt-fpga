#!/usr/bin/env bash
# Open-source Xilinx 7-series back end on macOS for the Nexys A7-100T (xc7a100t only): nextpnr-himbaechel (place and
# route, timing), Project X-Ray (xc7frames2bit) and fasm (fasm2frames). Follows https://zff.dev/gloriag/xc7-toolchain
# (build.sh) with the GUI and OpenMP off, so Apple clang is enough (no Qt, no Homebrew LLVM). Nothing is installed
# outside $XC7_DIR; scripts/nexys_pnr.sh finds the tools there.
#   brew install boost eigen@3 cmake ninja gflags cctz yaml-cpp googletest abseil openfpgaloader
#   scripts/xc7_toolchain.sh            ($XC7_DIR, default ~/tools/xc7; about 30-60 min, mostly the xc7a100t chipdb)
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
X="${XC7_DIR:-$HOME/tools/xc7}"
J="$(sysctl -n hw.ncpu)"
mkdir -p "$X" && cd "$X"
[ -d tc ] || git clone https://zff.dev/gloriag/xc7-toolchain tc     # recipe + patches (0BSD)

if [ ! -d prjxray ]; then
  git clone https://github.com/f4pga/prjxray
  git -C prjxray checkout c9f02d8576042325425824647ab5555b1bc77833
  git -C prjxray apply "$X/tc/prjxray.patch"
fi
[ -f prjxray/database/artix7/xc7a100tcsg324-1/part.yaml ] || (cd prjxray && ./download-latest-db.sh)
if [ ! -d nextpnr ]; then
  git clone https://github.com/YosysHQ/nextpnr.git
  git -C nextpnr checkout 3e53a0bf44d13c0de603dd089a323ea85d67d4ef
  git -C nextpnr submodule update --recursive --init
fi
if [ ! -d fasm ]; then                                              # patched: pure-Python (textX) parser, no Java
  git clone https://github.com/chipsalliance/fasm
  git -C fasm checkout ffafe821bae68637fe46e36bcfd2a01b97cdf6f2
  git -C fasm apply "$X/tc/fasm.patch"
fi

echo "== nextpnr-himbaechel (xilinx, xc7a100t)"
cmake -S nextpnr -B nextpnr/build -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-author \
  -DARCH=himbaechel -DHIMBAECHEL_UARCH=xilinx -DHIMBAECHEL_XILINX_DEVICES=xc7a100t \
  -DHIMBAECHEL_PRJXRAY_DB="$X/prjxray/database" -DUSE_OPENMP=OFF -DBUILD_GUI=OFF -DBUILD_PYTHON=OFF
cmake --build nextpnr/build -j "$J"

echo "== Project X-Ray tools (xc7frames2bit)"
cmake -S prjxray -B prjxray/build -G Ninja -DCMAKE_BUILD_TYPE=Release -Wno-author
cmake --build prjxray/build -j "$J"

echo "== fasm2frames (Python venv)"
PY="$(command -v python3.12 || command -v python3)"
[ -d prjxray/env ] || "$PY" -m venv prjxray/env
(cd prjxray && env/bin/pip install -q cython -e ../fasm -r requirements.txt)

ls -l nextpnr/build/nextpnr-himbaechel prjxray/build/tools/xc7frames2bit
find nextpnr/build -name 'chipdb-xc7a100t.bin'
prjxray/env/bin/python prjxray/utils/fasm2frames.py --help > /dev/null && echo "fasm2frames OK"
echo "Done: scripts/nexys_pnr.sh <model>"
