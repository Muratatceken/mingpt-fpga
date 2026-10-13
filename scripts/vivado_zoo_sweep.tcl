# ---------------------------------------------------------------------------
# Out-of-context implementation of the zoo engine for every model (Vivado, non-project mode):
# synthesis + place + route of zoo_core alone, 100 MHz clock constraint, then utilization,
# worst slack (-> Fmax estimate) and vectorless power. Two configurations per model:
# "gen" = all units (LOADABLE, the one-bitstream engine) and "spec" = only the model's units.
#
#   vivado -mode batch -source scripts/vivado_zoo_sweep.tcl
#   vivado -mode batch -source scripts/vivado_zoo_sweep.tcl -tclargs gpt2 llama     (subset)
#
# Output: build/zoo/vivado/sweep.jsonl (one JSON object per run) + reports per run.
# Copy sweep.jsonl to model/zoo/vivado_sweep.jsonl; sw/zoo/report_zoo.py and the paper read it.
# ---------------------------------------------------------------------------
source [file join [file dirname [info script]] vivado_zoo_common.tcl]

set part   xc7z020clg400-1
set period 10.0
set out    [file join $ROOT build zoo vivado]
file mkdir $out
set models $argv
if {[llength $models] == 0} {
  set models {bigram mlp rnn gru bow gpt2 microgpt llama nanochat stories260k universal}
}
set jf [open [file join $out sweep.jsonl] a]

foreach m $models {
  foreach cfg {gen spec} {
    if {$m eq "universal" && $cfg eq "spec"} { continue }
    set tag "${m}_${cfg}"
    puts "INFO: ===== $tag ====="
    close_project -quiet
    create_project -in_memory -part $part
    read_verilog -sv [zoo_pkg_for $ROOT [file join mem zoo $m]]
    read_verilog -sv $ZOO_RTL
    set spec [expr {$cfg eq "spec" ? 1 : 0}]
    synth_design -top zoo_core -part $part -mode out_of_context \
      -generic SPECIALIZE=$spec -generic LOADABLE=1 -flatten_hierarchy rebuilt
    create_clock -name clk -period $period [get_ports clk]
    opt_design
    place_design
    phys_opt_design
    route_design
    report_utilization    -file [file join $out util_$tag.rpt]
    report_timing_summary -file [file join $out timing_$tag.rpt]
    report_power          -file [file join $out power_$tag.rpt]
    set g [zoo_numbers $period]
    dict set g model $m
    dict set g config $cfg
    puts $jf [dict_to_json $g]
    flush $jf
    puts "INFO: $tag -> [dict_to_json $g]"
  }
}
close $jf
puts "INFO: results in [file join $out sweep.jsonl]"
