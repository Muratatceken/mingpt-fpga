// ---------------------------------------------------------------------------
// bin2bcd.sv -- sequential binary -> BCD conversion (double dabble), 32 clocks.
// 8 BCD digits: the input must be below 10^8.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module bin2bcd (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [31:0] bin,
  output logic        done,
  output logic [31:0] bcd
);
  logic [31:0] b, adj;
  logic [5:0]  n;
  logic        busy;

  always_comb
    for (int i = 0; i < 8; i++)
      adj[4*i +: 4] = (bcd[4*i +: 4] >= 4'd5) ? bcd[4*i +: 4] + 4'd3 : bcd[4*i +: 4];

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      busy <= 1'b0;
      bcd  <= '0;
    end else if (start) begin
      b    <= bin;
      bcd  <= '0;
      n    <= 6'd32;
      busy <= 1'b1;
    end else if (busy) begin
      {bcd, b} <= {adj, b} << 1;
      n <= n - 1'b1;
      if (n == 6'd1) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end
endmodule
