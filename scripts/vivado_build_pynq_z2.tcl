# ---------------------------------------------------------------------------
# PYNQ-Z2 bitstream of the PL-only self-test (board/pynq_z2/pynq_z2_top.sv).
#
#   vivado -mode batch -source scripts/vivado_build_pynq_z2.tcl
#
# Output: build/pynq_z2/gpt_selftest.bit + utilization/timing/power reports.
# Program the board afterwards with scripts/vivado_program.tcl.
# ---------------------------------------------------------------------------
source [file join [file dirname [info script]] common.tcl]

set part xc7z020clg400-1
set out  [file join $ROOT build pynq_z2]
file mkdir $out

read_verilog -sv $RTL_FILES
read_verilog -sv [file join $ROOT board pynq_z2 pynq_z2_top.sv]
read_xdc         [file join $ROOT board pynq_z2 pynq_z2.xdc]

synth_design -top pynq_z2_top -part $part
opt_design
place_design
phys_opt_design
route_design

report_utilization    -file [file join $out utilization.rpt]
report_utilization    -hierarchical -file [file join $out utilization_hier.rpt]
report_timing_summary -file [file join $out timing.rpt]
report_power          -file [file join $out power.rpt]

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "INFO: worst setup slack = $wns ns"
if {$wns < 0} {
  puts "WARNING: timing NOT met -- lower the clock (CLKOUT_DIV in pynq_z2_top.sv) or pipeline the critical path"
}

write_bitstream -force [file join $out gpt_selftest.bit]
puts "INFO: bitstream -> [file join $out gpt_selftest.bit]"
