# ---------------------------------------------------------------------------
# Program the PYNQ-Z2 over JTAG (micro-USB "PROG/UART" port).
#
#   vivado -mode batch -source scripts/vivado_program.tcl [-tclargs <file.bit>]
# ---------------------------------------------------------------------------
set ROOT [file normalize [file join [file dirname [info script]] ..]]
set bit  [file join $ROOT build pynq_z2 gpt_selftest.bit]
if {$argc > 0} { set bit [lindex $argv 0] }

open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices xc7z020*] 0]
if {$dev eq ""} { error "xc7z020 not found on the JTAG chain -- is the board connected and powered?" }
current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
puts "INFO: programmed $dev with $bit"

close_hw_target
disconnect_hw_server
close_hw_manager
