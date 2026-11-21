`timescale 1ns/1ps
// dec_board_top in simulation (+define+SIM): reset, press BTNC, wait for LED16_G (all hashes matched) or LED16_R.
module tb_dec_board;
  parameter int POLICY = 0;
  logic clk = 1'b0, rstn = 1'b0, btn = 1'b0;
  logic [15:0] led;
  logic r, g, b;
  always #5 clk = ~clk;
  dec_board_top #(.POLICY(POLICY), .DEB_CYC(4)) dut (.CLK100MHZ(clk), .CPU_RESETN(rstn), .BTNC(btn), .LED(led),
                                                      .LED16_R(r), .LED16_G(g), .LED16_B(b));
  longint cyc = 0;
  initial begin
    repeat (5) @(posedge clk); rstn = 1'b1;
    repeat (60) @(posedge clk); btn = 1'b1;
    repeat (20) @(posedge clk); btn = 1'b0;
    while (!g && !r && cyc < 64'd400_000_000) begin @(posedge clk); cyc++; end
    repeat (5) @(posedge clk);
    $display("RESULT board policy %0d: scenarios %0d matched %0d green %0d red %0d cycles %0d", POLICY, led[7:0],
             led[15:8], g, r, cyc);
    if (g && !r) $display("PASS: stand-alone self-test green"); else $display("FAIL");
    $finish;
  end
endmodule
