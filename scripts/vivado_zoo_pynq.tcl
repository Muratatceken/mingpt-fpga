# ---------------------------------------------------------------------------
# PYNQ-Z2 overlay with the universal zoo engine (one bitstream for every model):
#   Zynq PS7 (FCLK0 = 100 MHz)  --M_AXI_GP0-->  AXI interconnect  -->  zoo_axi_top (64 KB at 0x4000_0000)
#
#   vivado -mode batch -source scripts/vivado_zoo_pynq.tcl                 (default 100 MHz)
#   vivado -mode batch -source scripts/vivado_zoo_pynq.tcl -tclargs 125    (PL clock in MHz)
#
# Needs the PYNQ-Z2 board files (Vivado: board_files/pynq-z2/, see docs/SENIN_YAPACAKLARIN.md) for the
# DDR/MIO settings of the PS. Output in build/zoo/pynq_overlay/: zoo.bit, zoo.hwh (both are needed by
# PYNQ), reports and overlay.json (utilization, worst slack, power).
# ---------------------------------------------------------------------------
source [file join [file dirname [info script]] vivado_zoo_common.tcl]

set part xc7z020clg400-1
set mhz  [expr {[llength $argv] ? [lindex $argv 0] : 100}]
set out  [file join $ROOT build zoo pynq_overlay]
set proj [file join $ROOT build zoo pynq_proj]
file mkdir $out

create_project zoo_pynq $proj -part $part -force
set bp [get_board_parts -quiet -latest_file_version {*pynq-z2*}]
if {$bp eq ""} {
  error "PYNQ-Z2 board files not found. Install them (docs/SENIN_YAPACAKLARIN.md, step V2) and run again."
}
set_property board_part $bp [current_project]

set srcs [concat [list [zoo_pkg_for $ROOT [file join mem zoo universal]]] $ZOO_RTL \
          [list [file join $ROOT rtl zoo zoo_axi.sv] [file join $ROOT board pynq_z2 zoo_axi_top.v]]]
add_files -norecurse $srcs
foreach f [get_files *.sv] { set_property file_type SystemVerilog $f }
update_compile_order -fileset sources_1

create_bd_design zoo
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells ps7]
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 {1} CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $mhz] [get_bd_cells ps7]
create_bd_cell -type module -reference zoo_axi_top zoo
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Clk_master {/ps7/FCLK_CLK0} Clk_slave {/ps7/FCLK_CLK0} Clk_xbar {/ps7/FCLK_CLK0} \
           Master {/ps7/M_AXI_GP0} Slave {/zoo/s_axi} intc_ip {New AXI Interconnect} master_apm {0}] \
  [get_bd_intf_pins zoo/s_axi]
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data] -filter {NAME =~ *zoo*}]
set_property offset 0x40000000 $seg
set_property range 64K $seg
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files zoo.bd] -top]
add_files -norecurse $wrapper
set_property top zoo_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed, see $proj" }
open_run impl_1
report_utilization    -file [file join $out utilization.rpt]
report_utilization    -hierarchical -file [file join $out utilization_hier.rpt]
report_timing_summary -file [file join $out timing.rpt]
report_power          -file [file join $out power.rpt]
set g [zoo_numbers [expr {1000.0 / $mhz}]]
dict set g clock_mhz $mhz
set fh [open [file join $out overlay.json] w]; puts $fh [dict_to_json $g]; close $fh
if {[dict get $g wns] < 0} { puts "WARNING: timing NOT met at $mhz MHz -- rerun with a lower clock (-tclargs 80)" }

file copy -force [glob [file join $proj zoo_pynq.runs impl_1 *.bit]] [file join $out zoo.bit]
# hardware handoff: <proj>/zoo_pynq.gen/... (Vivado >= 2020.2) or <proj>/zoo_pynq.srcs/... (older)
set hwh [glob -nocomplain [file join $proj * sources_1 bd zoo hw_handoff zoo.hwh]]
if {$hwh eq ""} { error "zoo.hwh not found under $proj" }
file copy -force [lindex $hwh 0] [file join $out zoo.hwh]
puts "INFO: overlay -> $out/zoo.bit + zoo.hwh   [dict_to_json $g]"
