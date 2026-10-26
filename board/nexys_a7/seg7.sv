// ---------------------------------------------------------------------------
// seg7.sv -- eight-digit multiplexed 7-segment display (Nexys A7: common anode,
// segments and anodes active low). Leading zeros are blank.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module seg7 #(
  parameter int DIGIT_CYC = 50_000                  // clocks per digit (1 ms at 50 MHz)
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] bcd,
  output logic [6:0]  seg,                          // {g, f, e, d, c, b, a}, active low
  output logic        dp,
  output logic [7:0]  an
);
  logic [$clog2(DIGIT_CYC)-1:0] cnt;
  logic [2:0] d;
  logic [3:0] v;
  logic [7:0] lead;                                 // digit i is a leading zero

  always_comb begin
    lead = '0;
    for (int i = 7; i >= 1; i--)
      lead[i] = (bcd[4*i +: 4] == 4'd0) && (i == 7 || lead[(i + 1) % 8]);
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      cnt <= '0; d <= '0;
    end else if (cnt == ($clog2(DIGIT_CYC))'(DIGIT_CYC - 1)) begin
      cnt <= '0; d <= d + 1'b1;
    end else cnt <= cnt + 1'b1;
  end

  assign v  = bcd[4*d +: 4];
  assign dp = 1'b1;
  always_ff @(posedge clk) begin
    an <= lead[d] ? 8'hFF : ~(8'd1 << d);
    case (v)
      4'd0: seg <= 7'b1000000;
      4'd1: seg <= 7'b1111001;
      4'd2: seg <= 7'b0100100;
      4'd3: seg <= 7'b0110000;
      4'd4: seg <= 7'b0011001;
      4'd5: seg <= 7'b0010010;
      4'd6: seg <= 7'b0000010;
      4'd7: seg <= 7'b1111000;
      4'd8: seg <= 7'b0000000;
      4'd9: seg <= 7'b0010000;
      default: seg <= 7'b0111111;                   // '-'
    endcase
  end
endmodule
