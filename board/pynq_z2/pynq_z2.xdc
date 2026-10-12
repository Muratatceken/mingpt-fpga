## ---------------------------------------------------------------------------
## PYNQ-Z2 (TUL), xc7z020clg400-1 -- pins as in the board's master constraints.
## Only the PL I/O used by pynq_z2_top is listed. All banks are 3.3 V.
## ---------------------------------------------------------------------------

## 125 MHz PL clock
set_property -dict { PACKAGE_PIN H16 IOSTANDARD LVCMOS33 } [get_ports sysclk]
create_clock -name sysclk -period 8.000 -waveform {0.000 4.000} [get_ports sysclk]

## LEDs LD0..LD3
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN N16 IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN M14 IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## RGB LEDs LD4, LD5
set_property -dict { PACKAGE_PIN N15 IOSTANDARD LVCMOS33 } [get_ports led4_r]
set_property -dict { PACKAGE_PIN G17 IOSTANDARD LVCMOS33 } [get_ports led4_g]
set_property -dict { PACKAGE_PIN L15 IOSTANDARD LVCMOS33 } [get_ports led4_b]
set_property -dict { PACKAGE_PIN M15 IOSTANDARD LVCMOS33 } [get_ports led5_r]
set_property -dict { PACKAGE_PIN L14 IOSTANDARD LVCMOS33 } [get_ports led5_g]
set_property -dict { PACKAGE_PIN G14 IOSTANDARD LVCMOS33 } [get_ports led5_b]

## Push buttons BTN0..BTN3
set_property -dict { PACKAGE_PIN D19 IOSTANDARD LVCMOS33 } [get_ports {btn[0]}]
set_property -dict { PACKAGE_PIN D20 IOSTANDARD LVCMOS33 } [get_ports {btn[1]}]
set_property -dict { PACKAGE_PIN L20 IOSTANDARD LVCMOS33 } [get_ports {btn[2]}]
set_property -dict { PACKAGE_PIN L19 IOSTANDARD LVCMOS33 } [get_ports {btn[3]}]

## Slide switches SW0, SW1
set_property -dict { PACKAGE_PIN M20 IOSTANDARD LVCMOS33 } [get_ports {sw[0]}]
set_property -dict { PACKAGE_PIN M19 IOSTANDARD LVCMOS33 } [get_ports {sw[1]}]

## Buttons/switches are synchronised in the fabric, LEDs are slow outputs
set_false_path -from [get_ports {btn[*] sw[*]}]
set_false_path -to   [get_ports {led[*] led4_* led5_*}]
