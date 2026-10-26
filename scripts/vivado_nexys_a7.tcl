# ---------------------------------------------------------------------------
# Nexys A7-100T bitstream of one model (stand-alone: the model runs on the FPGA, VGA output)
#
#   vivado -mode batch -source scripts/vivado_nexys_a7.tcl -tclargs <model> [MHz]
#     model  directory under mem/nexys/ (sw/zoo/board.py export <model>), e.g. stories260k, gru
#     MHz    core clock: 50 (default), 100 or 25. If timing is not met, the script builds again at the
#            next lower clock (100 -> 50 -> 25), so the bitstream it leaves always meets timing.
#
# Output: build/nexys/<model>/vivado/
#   nexys_<model>.bit   bitstream (JTAG: scripts/vivado_program_nexys.tcl or openFPGALoader -b nexys_a7_100)
#   nexys_<model>.bin   for the quad-SPI flash (openFPGALoader -b nexys_a7_100 -f ..., or Vivado hw manager)
#   utilization.rpt, timing.rpt, power.rpt, result.json (LUT, FF, BRAM, DSP, WNS, Fmax, power, clock)
# ---------------------------------------------------------------------------
set ROOT  [file normalize [file join [file dirname [info script]] ..]]
set part  xc7a100tcsg324-1
if {[llength $argv] < 1} { error "usage: -tclargs <model> \[MHz\]" }
set model [lindex $argv 0]
set mhz   [expr {[llength $argv] > 1 ? [lindex $argv 1] : 50}]
set mem   [file join $ROOT mem nexys $model]
set out   [file join $ROOT build nexys $model vivado]
if {![file exists [file join $mem board_pkg.sv]]} { error "$mem not found: run  cd sw/zoo && python3 board.py export $model" }
file mkdir $out

# zoo_pkg with an absolute MEM_PATH ($readmemh runs in Vivado's own directories)
set fh [open [file join $mem zoo_pkg.sv] r]; set txt [read $fh]; close $fh
if {[regsub {localparam string MEM_PATH   = "[^"]*";} $txt "localparam string MEM_PATH   = \"$mem\";" txt] != 1} {
  error "MEM_PATH not found in $mem/zoo_pkg.sv"
}
set pkg [file join $out zoo_pkg.sv]
set fh [open $pkg w]; puts -nonewline $fh $txt; close $fh

set srcs [list $pkg [file join $mem board_pkg.sv] \
  [file join $ROOT rtl zoo zrom.sv] [file join $ROOT rtl zoo zdiv.sv] [file join $ROOT rtl requant.sv] \
  [file join $ROOT rtl isqrt_seq.sv] [file join $ROOT rtl zoo zoo_core.sv] \
  [file join $ROOT board nexys_a7 bin2bcd.sv] [file join $ROOT board nexys_a7 vga_text.sv] \
  [file join $ROOT board nexys_a7 seg7.sv] [file join $ROOT board nexys_a7 lm_gen.sv] \
  [file join $ROOT board nexys_a7 nexys_a7_top.sv]]

proc build {mhz} {
  global srcs part ROOT out model
  close_project -quiet
  create_project -in_memory -part $part
  read_verilog -sv $srcs
  read_xdc [file join $ROOT board nexys_a7 nexys_a7.xdc]
  synth_design -top nexys_a7_top -part $part -generic CLK_MHZ=$mhz
  opt_design
  place_design
  phys_opt_design
  route_design
  return [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
}

set wns [build $mhz]
while {$wns < 0 && $mhz > 25} {
  puts "WARNING: timing not met at $mhz MHz (WNS $wns ns), building again at a lower clock"
  set mhz [expr {$mhz > 50 ? 50 : 25}]
  set wns [build $mhz]
}

report_utilization    -file [file join $out utilization.rpt]
report_utilization    -hierarchical -file [file join $out utilization_hier.rpt]
report_timing_summary -file [file join $out timing.rpt]
report_power          -file [file join $out power.rpt]
write_bitstream -force -bin_file [file join $out nexys_$model.bit]

set u [report_utilization -return_string]
set res [dict create model $model clock_mhz $mhz wns $wns]
foreach {key pat} {lut {Slice LUTs\*?\s*\|\s*(\d+)} ff {Slice Registers\s*\|\s*(\d+)} \
                   bram {Block RAM Tile\s*\|\s*([\d.]+)} dsp {DSPs\s*\|\s*(\d+)}} {
  if {[regexp $pat $u -> v]} { dict set res $key $v }
}
# Fmax of the core clock domain: 1 / (period - WNS)
dict set res fmax_mhz [format %.1f [expr {1000.0 / (1000.0 / $mhz - $wns)}]]
set p [report_power -return_string]
if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([\d.]+)} $p -> w]} { dict set res power_w $w }
set parts {}
dict for {k v} $res {
  if {[string is double -strict $v]} { lappend parts "\"$k\": $v" } else { lappend parts "\"$k\": \"$v\"" }
}
set fh [open [file join $out result.json] w]; puts $fh "{[join $parts {, }]}"; close $fh
if {$wns < 0} { puts "ERROR: timing not met even at 25 MHz (WNS $wns ns): do not use this bitstream" }
puts "INFO: $out/nexys_$model.bit  [join $parts {, }]"
