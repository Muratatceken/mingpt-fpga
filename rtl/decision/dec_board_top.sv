`timescale 1ns/1ps
// Nexys A7-100T stand-alone decision self-test (no host, no UART): the integer policy, the guard and the recorded
// telemetry live in BRAM.  BTNC runs every scenario of the observation ROM through decision_top and compares the
// trace hash of each episode with the expected hash computed by sw/decision/hw_ref.py.
//   LED[7:0]  scenarios run      LED[15:8] scenarios whose hash matched
//   LED16_G   all matched (done) LED16_R   at least one mismatch    LED16_B  running
module dec_board_top
  import dec_pkg::*;
#(
  parameter int    CLK_MHZ   = 25,
  parameter int    POLICY    = 0,
  parameter int    NSCEN     = 24,
  parameter int    NSTEP     = 48,
  parameter int    DEB_CYC   = CLK_MHZ * 10_000,          // 10 ms button debounce
  parameter string OBS_FILE  = "build/decision/obs.mem",
  parameter string HASH_FILE = "build/decision/exp_hash.mem"
) (
  input  logic        CLK100MHZ,
  input  logic        CPU_RESETN,
  input  logic        BTNC,
  output logic [15:0] LED,
  output logic        LED16_R, LED16_G, LED16_B
);
  logic clk, locked;
`ifdef SIM
  assign clk = CLK100MHZ;
  assign locked = 1'b1;
`else
  logic clk_mmcm, clkfb;
  MMCME2_BASE #(.CLKIN1_PERIOD(10.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(10.0),
                .CLKOUT0_DIVIDE_F(1000.0 / CLK_MHZ)) u_mmcm (
    .CLKIN1(CLK100MHZ), .CLKFBIN(clkfb), .CLKFBOUT(clkfb), .CLKOUT0(clk_mmcm), .LOCKED(locked),
    .PWRDWN(1'b0), .RST(1'b0), .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
    .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6());
  BUFG u_bufg (.I(clk_mmcm), .O(clk));
`endif
  // reset synchroniser
  logic [1:0] rs;
  logic [4:0] rc;
  logic       rst_n;
  always_ff @(posedge clk or negedge locked) begin
    if (!locked) begin rs <= '0; rc <= '0; rst_n <= 1'b0; end
    else begin
      rs <= {rs[0], CPU_RESETN};
      if (!rs[1]) begin rc <= '0; rst_n <= 1'b0; end
      else if (rc != 5'h1F) rc <= rc + 5'd1;
      else rst_n <= 1'b1;
    end
  end
  // BTNC debounce -> one-cycle press
  logic b1, b2, bst, press;
  logic [$clog2(DEB_CYC + 1)-1:0] deb;
  always_ff @(posedge clk) begin
    b1 <= BTNC; b2 <= b1; press <= 1'b0;
    if (!rst_n) begin bst <= 1'b0; deb <= '0; end
    else if (b2 == bst) deb <= '0;
    else if (deb == ($clog2(DEB_CYC + 1))'(DEB_CYC)) begin bst <= b2; deb <= '0; press <= b2; end
    else deb <= deb + 1'b1;
  end
  // expected hashes
  logic [31:0] exp_h [NSCEN];
  initial $readmemh(HASH_FILE, exp_h);
  // decision system
  logic        d_start, d_busy, d_done, tr_we;
  logic [15:0] d_base, tr_word;
  logic [31:0] d_hash, dec_cycles;
  decision_top #(.POLICY(POLICY), .NWORDS(NSCEN * NSTEP), .OBS_FILE(OBS_FILE)) u_dec (
    .clk(clk), .rst_n(rst_n), .start(d_start), .base(d_base), .nsteps(8'(NSTEP)), .busy(d_busy), .done(d_done),
    .hash(d_hash), .tr_we(tr_we), .tr_word(tr_word), .dec_cycles(dec_cycles));
  // scenario sequencer
  logic [7:0] s, npass;
  logic       running, fail, allok;
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s <= '0; npass <= '0; running <= 1'b0; fail <= 1'b0; allok <= 1'b0; d_start <= 1'b0; d_base <= '0;
    end else begin
      d_start <= 1'b0;
      if (press && !running) begin
        s <= '0; npass <= '0; fail <= 1'b0; allok <= 1'b0; running <= 1'b1; d_base <= '0; d_start <= 1'b1;
      end else if (running && d_done) begin
        if (d_hash == exp_h[s]) npass <= npass + 8'd1; else fail <= 1'b1;
        if (s == 8'(NSCEN - 1)) begin
          running <= 1'b0;
          allok <= (d_hash == exp_h[s]) && !fail;
          s <= s + 8'd1;
        end else begin
          s <= s + 8'd1;
          d_base <= d_base + 16'(NSTEP);
          d_start <= 1'b1;
        end
      end
    end
  end
  assign LED = {npass, s};
  assign LED16_G = allok;
  assign LED16_R = fail;
  assign LED16_B = running;
endmodule
