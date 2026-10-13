# Shared helpers for the zoo-engine Vivado scripts.
#   ROOT                 repository root
#   zoo_pkg_for <memdir> copy of <memdir>/zoo_pkg.sv with an ABSOLUTE MEM_PATH (Vivado runs synthesis in
#                        its own directories, $readmemh paths must not be relative)
#   ZOO_RTL              engine sources (without the package)
set ROOT [file normalize [file join [file dirname [info script]] ..]]

proc zoo_pkg_for {root memdir} {
  set src [file join $root $memdir zoo_pkg.sv]
  set dst_dir [file join $root build gen [file tail $memdir]]
  file mkdir $dst_dir
  set dst [file join $dst_dir zoo_pkg.sv]
  set fh [open $src r]; set txt [read $fh]; close $fh
  set abs [file join $root $memdir]
  set n [regsub {localparam string MEM_PATH   = "[^"]*";} $txt "localparam string MEM_PATH   = \"$abs\";" txt]
  if {$n != 1} { error "MEM_PATH not found in $src" }
  set fh [open $dst w]; puts -nonewline $fh $txt; close $fh
  return $dst
}

set ZOO_RTL [list \
  [file join $ROOT rtl zoo zrom.sv] [file join $ROOT rtl zoo zdiv.sv] [file join $ROOT rtl requant.sv] \
  [file join $ROOT rtl isqrt_seq.sv] [file join $ROOT rtl zoo zoo_core.sv]]

# key numbers of an implemented design -> dict (LUT, FF, BRAM36 equivalents, DSP, WNS, Fmax, power)
proc zoo_numbers {period} {
  set u [report_utilization -return_string]
  set g {}
  foreach {key pat} {lut {Slice LUTs\*?\s*\|\s*(\d+)} ff {Slice Registers\s*\|\s*(\d+)} \
                     bram {Block RAM Tile\s*\|\s*([\d.]+)} dsp {DSPs\s*\|\s*(\d+)}} {
    if {[regexp $pat $u -> v]} { dict set g $key $v } else { dict set g $key -1 }
  }
  set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
  dict set g wns $wns
  dict set g fmax_mhz [format %.1f [expr {1000.0 / ($period - $wns)}]]
  set p [report_power -return_string]
  if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([\d.]+)} $p -> w]} { dict set g power_w $w }
  if {[regexp {Dynamic \(W\)\s*\|\s*([\d.]+)} $p -> w]} { dict set g dynamic_w $w }
  return $g
}

proc dict_to_json {d} {
  set parts {}
  dict for {k v} $d {
    if {[string is double -strict $v]} { lappend parts "\"$k\": $v" } else { lappend parts "\"$k\": \"$v\"" }
  }
  return "{[join $parts {, }]}"
}
