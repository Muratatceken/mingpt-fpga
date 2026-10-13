// ---------------------------------------------------------------------------
// zdiv.sv -- sequential restoring divider with a bounded quotient.
//   quo = floor(num / den), assuming quo < 2^QB (i.e. num < den * 2^QB).
// The first W-QB quotient bits of a W-bit division are then known to be zero
// and the partial remainder after them is num >> QB, so the divider starts
// there and needs only QB clocks instead of W (div_seq.sv is the QB = W case).
//   NORM: 1/sqrt  quotient < 2^41  -> 41 clocks instead of 64
//   ATTN: 2^32/E  E >= 2^15        -> 18 clocks instead of 34
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module zdiv #(
  parameter int W  = 64,
  parameter int QB = 41
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  input  logic [W-1:0] num,
  input  logic [W-1:0] den,
  output logic         done,
  output logic [W-1:0] quo
);
  logic [QB-1:0]           q;
  logic [W-1:0]            d;
  logic [W:0]              r;
  logic [$clog2(QB+1)-1:0] cnt;
  logic                    busy;

  logic [W:0] r_sh;
  assign r_sh = {r[W-1:0], q[QB-1]};
  assign quo  = W'(q);

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      busy <= 1'b0;
    end else if (start) begin
      r    <= (W + 1)'(num >> QB);
      q    <= num[QB-1:0];
      d    <= den;
      cnt  <= ($clog2(QB+1))'(QB);
      busy <= 1'b1;
    end else if (busy) begin
      if (r_sh >= {1'b0, d}) begin
        r <= r_sh - {1'b0, d};
        q <= {q[QB-2:0], 1'b1};
      end else begin
        r <= r_sh;
        q <= {q[QB-2:0], 1'b0};
      end
      cnt <= cnt - 1'b1;
      if (cnt == 1) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end
endmodule
