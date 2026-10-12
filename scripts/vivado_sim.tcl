# ---------------------------------------------------------------------------
# Vivado project-mode simulation (xsim) of tb_gpt_top.
#
#   GUI   : vivado -source scripts/vivado_sim.tcl
#   batch : vivado -mode batch -source scripts/vivado_sim.tcl -tclargs 20
#
# Optional tclarg: number of test sequences (default 20; all = 729).
# The project is created in build/vivado_sim/. In GUI mode the waveform window
# opens after the run -- add signals of dut/u_gemv, dut/u_att, ... and re-run.
# ---------------------------------------------------------------------------
source [file join [file dirname [info script]] common.tcl]

set nseq 20
if {$argc > 0} { set nseq [lindex $argv 0] }
set part xc7z020clg400-1
set proj_dir [file join $ROOT build vivado_sim]

create_project -force gpt_sim $proj_dir -part $part
set_property target_language Verilog [current_project]

add_files -norecurse $RTL_FILES
add_files -fileset sim_1 -norecurse [file join $ROOT tb tb_gpt_top.sv]
set_property file_type SystemVerilog [get_files *.sv]
set_property top tb_gpt_top [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]

set_property -name {xsim.simulate.runtime} -value {-all} -objects [get_filesets sim_1]
set_property -name {xsim.simulate.xsim.more_options} -value "-testplusarg NSEQ=$nseq" \
             -objects [get_filesets sim_1]

launch_simulation
