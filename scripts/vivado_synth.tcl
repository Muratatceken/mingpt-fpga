# ---------------------------------------------------------------------------
# Out-of-context synthesis of gpt_top: resource and timing estimates.
#
#   vivado -mode batch -source scripts/vivado_synth.tcl -tclargs [part] [period_ns]
#
# defaults: xc7z020clg400-1 (PYNQ-Z2 / Zybo Z7-20 / ZedBoard), 10 ns (100 MHz)
# e.g. Basys 3: -tclargs xc7a35tcpg236-1 10
# Reports: build/vivado_synth/{utilization,timing}.rpt
# ---------------------------------------------------------------------------
source [file join [file dirname [info script]] common.tcl]

set part   xc7z020clg400-1
set period 10.0
if {$argc > 0} { set part   [lindex $argv 0] }
if {$argc > 1} { set period [lindex $argv 1] }
set out [file join $ROOT build vivado_synth]
file mkdir $out

read_verilog -sv $RTL_FILES
synth_design -top gpt_top -part $part -mode out_of_context

create_clock -name clk -period $period [get_ports clk]
report_utilization    -file [file join $out utilization.rpt]
report_utilization    -hierarchical -file [file join $out utilization_hier.rpt]
report_timing_summary -file [file join $out timing.rpt]
report_utilization
report_timing_summary -max_paths 5
