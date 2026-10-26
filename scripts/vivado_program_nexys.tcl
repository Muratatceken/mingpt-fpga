# ---------------------------------------------------------------------------
# Program the Nexys A7-100T over USB-JTAG (the PROG/UART micro-USB port, board power switch on).
#
#   vivado -mode batch -source scripts/vivado_program_nexys.tcl -tclargs <model>          (FPGA, until power-off)
#   vivado -mode batch -source scripts/vivado_program_nexys.tcl -tclargs <model> flash    (quad-SPI flash:
#          the board then loads the model by itself at every power-up; jumper JP1 on QSPI)
#
# Without Vivado (e.g. on a Mac):  openFPGALoader -b nexys_a7_100 build/nexys/<model>/vivado/nexys_<model>.bit
#                         flash:   openFPGALoader -b nexys_a7_100 -f build/nexys/<model>/vivado/nexys_<model>.bit
# ---------------------------------------------------------------------------
set ROOT  [file normalize [file join [file dirname [info script]] ..]]
set model [lindex $argv 0]
set dir   [file join $ROOT build nexys $model vivado]
set bit   [file join $dir nexys_$model.bit]
if {![file exists $bit]} { error "$bit not found: run scripts/vivado_nexys_a7.tcl first" }

open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xc7a100t*] 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev

if {[lindex $argv 1] eq "flash"} {
  set mcs [file join $dir nexys_$model.mcs]
  write_cfgmem -force -format mcs -size 16 -interface SPIx4 -loadbit "up 0x0 $bit" -file $mcs
  create_hw_cfgmem -hw_device $dev [lindex [get_cfgmem_parts {s25fl128sxxxxxx0-spi-x1_x2_x4}] 0]
  set cm [get_property PROGRAM.HW_CFGMEM $dev]
  set_property PROGRAM.ADDRESS_RANGE {use_file} $cm
  set_property PROGRAM.FILES [list $mcs] $cm
  set_property PROGRAM.BLANK_CHECK 0 $cm
  set_property PROGRAM.ERASE 1 $cm
  set_property PROGRAM.CFG_PROGRAM 1 $cm
  set_property PROGRAM.VERIFY 1 $cm
  create_hw_bitstream -hw_device $dev [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
  program_hw_devices $dev
  program_hw_cfgmem -hw_cfgmem $cm
  boot_hw_device $dev
  puts "INFO: $model written to the flash; the board boots it at power-up"
} else {
  set_property PROGRAM.FILE $bit $dev
  program_hw_devices $dev
  puts "INFO: $model loaded; the first story / names appear on the VGA monitor"
}
close_hw_target
disconnect_hw_server
