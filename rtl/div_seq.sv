// ---------------------------------------------------------------------------
// div_seq.sv -- sequential restoring divider, unsigned.
//   quo = floor(num / den), one quotient bit per clock, W clocks per division.
// Used once per LayerNorm (reciprocal of the std) and once per attention head
// (reciprocal of the softmax denominator), so its latency is negligible.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module div_seq #(
  parameter int W = 48
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic [W-1:0] num,
  input  logic [W-1:0] den,
  output logic         done,
  output logic [W-1:0] quo
);
  logic [W-1:0]           q, d;
  logic [W:0]             r;
  logic [$clog2(W+1)-1:0] cnt;
  logic                   busy;

  logic [W:0] r_sh;
  assign r_sh = {r[W-1:0], q[W-1]};
  assign quo  = q;

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      busy <= 1'b0;
    end else if (start) begin
      q    <= num;
      d    <= den;
      r    <= '0;
      cnt  <= W[$clog2(W+1)-1:0];
      busy <= 1'b1;
    end else if (busy) begin
      if (r_sh >= {1'b0, d}) begin
        r <= r_sh - {1'b0, d};
        q <= {q[W-2:0], 1'b1};
      end else begin
        r <= r_sh;
        q <= {q[W-2:0], 1'b0};
      end
      cnt <= cnt - 1'b1;
      if (cnt == 1) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end
endmodule
