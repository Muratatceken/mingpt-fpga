// ---------------------------------------------------------------------------
// pynq_z2_top.sv -- PYNQ-Z2 (xc7z020clg400-1), PL-only self-test bitstream.
//
//   sysclk 125 MHz (H16) -> MMCM -> clk = 125 * 8 / CLKOUT_DIV MHz (50 MHz default)
//
//   BTN0  : run the self-test again (it also runs once automatically after configuration)
//   BTN1  : reset (hold)
//   SW0   : fault injection (every sequence must then FAIL -> red)
//
//   LD0   : heartbeat (bitstream alive)      LD4 RGB : blue  = running
//   LD1   : running                                    green = PASS (729/729 sorted)
//   LD2   : done                                       red   = FAIL
//   LD3   : fault injection enabled
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module pynq_z2_top #(
  parameter real CLKOUT_DIV = 20.0      // 1000 MHz VCO / 20 = 50 MHz
) (
  input  logic       sysclk,
  input  logic [3:0] btn,
  input  logic [1:0] sw,
  output logic [3:0] led,
  output logic       led4_r, led4_g, led4_b,
  output logic       led5_r, led5_g, led5_b
);
  // ---------------- clocking: 125 MHz -> VCO 1000 MHz -> clk ----------------
  logic clk, clk_mmcm, clkfb, locked;

  MMCME2_BASE #(
    .CLKIN1_PERIOD   (8.0),
    .DIVCLK_DIVIDE   (1),
    .CLKFBOUT_MULT_F (8.0),
    .CLKOUT0_DIVIDE_F(CLKOUT_DIV)
  ) u_mmcm (
    .CLKIN1  (sysclk),
    .CLKFBIN (clkfb),
    .CLKFBOUT(clkfb),
    .CLKOUT0 (clk_mmcm),
    .LOCKED  (locked),
    .PWRDWN  (1'b0),
    .RST     (1'b0)
  );

  BUFG u_bufg (.I(clk_mmcm), .O(clk));

  // ---------------- synchronisers, reset, start ----------------
  logic [2:0] btn0_q, btn1_q;
  logic [1:0] sw0_q;
  always_ff @(posedge clk) begin
    btn0_q <= {btn0_q[1:0], btn[0]};
    btn1_q <= {btn1_q[1:0], btn[1]};
    sw0_q  <= {sw0_q[0], sw[0]};
  end

  logic [3:0] rst_pipe;
  always_ff @(posedge clk or negedge locked) begin
    if (!locked) rst_pipe <= '0;
    else         rst_pipe <= {rst_pipe[2:0], ~btn1_q[2]};
  end
  logic rst_n;
  assign rst_n = rst_pipe[3];

  // start once after reset, then on every BTN0 press
  logic auto_go, start;
  always_ff @(posedge clk) begin
    if (!rst_n) auto_go <= 1'b1;
    else        auto_go <= 1'b0;
  end
  assign start = (rst_n && auto_go) || (btn0_q[1] && !btn0_q[2]);

  // ---------------- self-test ----------------
  logic        busy, done, pass;
  logic [15:0] n_tested, n_fail;
  logic [47:0] cycles;

  gpt_selftest u_st (
    .clk(clk), .rst_n(rst_n), .start(start), .inject_fault(sw0_q[1]),
    .busy(busy), .done(done), .pass(pass),
    .n_tested(n_tested), .n_fail(n_fail), .cycles(cycles));

  // ---------------- LEDs ----------------
  logic [25:0] hb;
  always_ff @(posedge clk) hb <= hb + 1'b1;

  assign led    = {sw0_q[1], done, busy, hb[25]};
  assign led4_b = busy;
  assign led4_g = done &&  pass;
  assign led4_r = done && !pass;
  assign {led5_r, led5_g, led5_b} = 3'b000;
endmodule
