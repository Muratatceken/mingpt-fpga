# ---------------------------------------------------------------------------
# Post-synthesis functional simulation of the zoo engine (Vivado synthesis + xsim), OPTIONAL:
# the synthesized netlist of one model, with its memories as initialized ROMs, runs the same
# golden vectors as the RTL. Checks that synthesis preserved the integer semantics bit for bit.
#
#   vivado -mode batch -source scripts/vivado_zoo_postsynth_sim.tcl -tclargs gru 5
#        (model under mem/zoo/, number of test sequences)
# Output: PASS/FAIL line of tb_zoo.sv in the console and build/zoo/postsynth/<model>/sim.log
# (The same check with Yosys' open-source cell models is blocked by X-propagation in those models.)
# ---------------------------------------------------------------------------
source [file join [file dirname [info script]] vivado_zoo_common.tcl]

set part xc7z020clg400-1
set m    [expr {[llength $argv] > 0 ? [lindex $argv 0] : "gru"}]
set nseq [expr {[llength $argv] > 1 ? [lindex $argv 1] : 5}]
set out  [file join $ROOT build zoo postsynth $m]
file mkdir $out
set pkg [zoo_pkg_for $ROOT [file join mem zoo $m]]

# 1) synthesis of zoo_core alone (memories initialized from mem/zoo/<model>/)
create_project -in_memory -part $part
read_verilog -sv $pkg
read_verilog -sv $ZOO_RTL
synth_design -top zoo_core -part $part -mode out_of_context -generic SPECIALIZE=1 -flatten_hierarchy full
write_verilog -force -mode funcsim [file join $out zoo_core_funcsim.v]

# 2) xsim: testbench (NETLIST: no parameters) + funcsim netlist + unisim primitives
cd $out
set glbl [file join $::env(XILINX_VIVADO) data verilog src glbl.v]
exec xvlog -sv -d NETLIST $pkg [file join $ROOT tb tb_zoo.sv] >@ stdout
exec xvlog zoo_core_funcsim.v $glbl >@ stdout
exec xelab -L unisims_ver -L secureip tb_zoo glbl -s tb_snap -timescale 1ns/1ps >@ stdout
cd $ROOT
exec xsim tb_snap -R -testplusarg NSEQ=$nseq -log [file join $out sim.log] >@ stdout
