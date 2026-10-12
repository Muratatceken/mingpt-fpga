# Shared helpers for the Vivado scripts.
#   ROOT     : repository root (derived from this file's location)
#   RTL_FILES: synthesizable sources in compile order (scripts/files.f), with
#              rtl/gpt_pkg.sv replaced by build/gen/gpt_pkg.sv (see below)

set ROOT [file normalize [file join [file dirname [info script]] ..]]

# gpt_pkg::MEM_PATH is relative ("mem") in the repository, which works when the
# simulator runs from the repo root (Verilator). Vivado runs xsim/synthesis in
# its own directories, so compile a copy whose MEM_PATH is absolute on THIS machine.
proc gen_pkg {root} {
  set src [file join $root rtl gpt_pkg.sv]
  set dst_dir [file join $root build gen]
  file mkdir $dst_dir
  set dst [file join $dst_dir gpt_pkg.sv]
  set fh [open $src r]; set txt [read $fh]; close $fh
  set mem [file join $root mem]
  set n [regsub {localparam string MEM_PATH = "[^"]*";} $txt \
         "localparam string MEM_PATH = \"$mem\";" txt]
  if {$n != 1} { error "MEM_PATH not found in $src" }
  set fh [open $dst w]; puts -nonewline $fh $txt; close $fh
  puts "INFO: $dst (MEM_PATH = $mem)"
  return $dst
}
set GEN_PKG [gen_pkg $ROOT]

set RTL_FILES {}
set fh [open [file join $ROOT scripts files.f] r]
foreach line [split [read $fh] "\n"] {
  set line [string trim $line]
  if {$line eq ""} { continue }
  if {$line eq "rtl/gpt_pkg.sv"} { lappend RTL_FILES $GEN_PKG; continue }
  lappend RTL_FILES [file join $ROOT $line]
}
close $fh
