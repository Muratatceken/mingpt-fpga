// ---------------------------------------------------------------------------
// sim_mmcme2_base.v -- simulation stand-in for the Xilinx MMCME2_BASE in the post-synthesis netlist of
// board/nexys_a7/nexys_a7_top.sv (scripts/nexys_netlist_sim.sh). Yosys' cells_xtra.v has only a black box.
// The testbench drives the core clock on CLK100MHZ (as the RTL with -DSIM), so CLKOUT0 is CLKIN1 passed
// through; LOCKED rises after 4 input clocks (inside the testbench's CPU_RESETN pulse, so the reset sequence
// is the same as in the RTL). The frequency the parameters select on the board is printed and checked
// (VCO 600..1200 MHz, Artix-7 -1).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module MMCME2_BASE (CLKFBOUT, CLKFBOUTB, CLKOUT0, CLKOUT0B, CLKOUT1, CLKOUT1B, CLKOUT2, CLKOUT2B, CLKOUT3, CLKOUT3B,
                    CLKOUT4, CLKOUT5, CLKOUT6, LOCKED, CLKFBIN, CLKIN1, PWRDWN, RST);
  parameter BANDWIDTH = "OPTIMIZED";
  parameter real CLKFBOUT_MULT_F = 5.000;
  parameter real CLKFBOUT_PHASE = 0.000;
  parameter real CLKIN1_PERIOD = 0.000;
  parameter real CLKOUT0_DIVIDE_F = 1.000;
  parameter real CLKOUT0_DUTY_CYCLE = 0.500;
  parameter real CLKOUT0_PHASE = 0.000;
  parameter integer CLKOUT1_DIVIDE = 1;
  parameter real CLKOUT1_DUTY_CYCLE = 0.500;
  parameter real CLKOUT1_PHASE = 0.000;
  parameter integer CLKOUT2_DIVIDE = 1;
  parameter real CLKOUT2_DUTY_CYCLE = 0.500;
  parameter real CLKOUT2_PHASE = 0.000;
  parameter integer CLKOUT3_DIVIDE = 1;
  parameter real CLKOUT3_DUTY_CYCLE = 0.500;
  parameter real CLKOUT3_PHASE = 0.000;
  parameter CLKOUT4_CASCADE = "FALSE";
  parameter integer CLKOUT4_DIVIDE = 1;
  parameter real CLKOUT4_DUTY_CYCLE = 0.500;
  parameter real CLKOUT4_PHASE = 0.000;
  parameter integer CLKOUT5_DIVIDE = 1;
  parameter real CLKOUT5_DUTY_CYCLE = 0.500;
  parameter real CLKOUT5_PHASE = 0.000;
  parameter integer CLKOUT6_DIVIDE = 1;
  parameter real CLKOUT6_DUTY_CYCLE = 0.500;
  parameter real CLKOUT6_PHASE = 0.000;
  parameter integer DIVCLK_DIVIDE = 1;
  parameter real REF_JITTER1 = 0.010;
  parameter STARTUP_WAIT = "FALSE";
  output CLKFBOUT, CLKFBOUTB, CLKOUT0, CLKOUT0B, CLKOUT1, CLKOUT1B, CLKOUT2, CLKOUT2B, CLKOUT3, CLKOUT3B;
  output CLKOUT4, CLKOUT5, CLKOUT6;
  output reg LOCKED;
  input  CLKFBIN, CLKIN1, PWRDWN, RST;

  assign CLKOUT0  = CLKIN1;
  assign CLKFBOUT = CLKIN1;
  assign {CLKFBOUTB, CLKOUT0B, CLKOUT1, CLKOUT1B, CLKOUT2, CLKOUT2B, CLKOUT3, CLKOUT3B, CLKOUT4, CLKOUT5, CLKOUT6} = 11'd0;

  integer n = 0;
  initial LOCKED = 1'b0;
  always @(posedge CLKIN1) begin
    if (n < 4) n = n + 1;
    LOCKED <= (n >= 4) && !RST && !PWRDWN;
  end

  real vco, f0;
  initial begin
    vco = 1000.0 / CLKIN1_PERIOD * CLKFBOUT_MULT_F / DIVCLK_DIVIDE;
    f0  = vco / CLKOUT0_DIVIDE_F;
    $display("MMCME2_BASE: CLKIN1 %0.1f MHz, VCO %0.1f MHz, CLKOUT0 %0.3f MHz%s", 1000.0 / CLKIN1_PERIOD, vco, f0,
             (vco >= 600.0 && vco <= 1200.0) ? "" : "  ERROR: VCO outside 600..1200 MHz");
  end
endmodule
