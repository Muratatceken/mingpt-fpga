// ---------------------------------------------------------------------------
// requant.sv -- per-channel requantisation (Jacob et al., 2018)
//
//   y = sat( ((acc + bias) * m0 + 2^(sh-1)) >>> sh )      (round half up)
//
// m0 in [2^15, 2^16) and sh come from the parameter ROM, so the real-valued
// multiplier s_in*s_w/s_out is represented as m0 * 2^-sh.
// Output saturates to INT8 (out16=0) or INT16 (out16=1). Latency: 2 clocks.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module requant #(
  parameter int IDX_W = 9
) (
  input  logic                    clk,
  input  logic                    in_valid,
  input  logic [IDX_W-1:0]        in_idx,
  input  logic signed [31:0]      acc,
  input  logic signed [31:0]      bias,
  input  logic        [15:0]      m0,
  input  logic        [5:0]       sh,
  input  logic                    out16,
  output logic                    out_valid,
  output logic [IDX_W-1:0]        out_idx,
  output logic signed [15:0]      out
);
  // ---- stage 1: (acc + bias) * m0 ----
  logic signed [32:0] sum;
  logic signed [16:0] m0_s;
  assign sum  = {acc[31], acc} + {bias[31], bias};
  assign m0_s = {1'b0, m0};

  logic signed [63:0] prod_r;
  logic        [5:0]  sh_r;
  logic               out16_r, v_r;
  logic [IDX_W-1:0]   idx_r;

  always_ff @(posedge clk) begin
    v_r     <= in_valid;
    idx_r   <= in_idx;
    prod_r  <= sum * m0_s;
    sh_r    <= sh;
    out16_r <= out16;
  end

  // ---- stage 2: round, shift, saturate ----
  logic signed [63:0] shifted;
  always_comb begin
    shifted = (prod_r + (64'sd1 <<< (sh_r - 6'd1))) >>> sh_r;
  end

  function automatic logic signed [15:0] sat(input logic signed [63:0] v, input logic is16);
    if (is16) begin
      if (v > 64'sd32767)       return 16'sh7FFF;
      else if (v < -64'sd32768) return 16'sh8000;
      else                      return v[15:0];
    end else begin
      if (v > 64'sd127)         return 16'sh007F;
      else if (v < -64'sd128)   return 16'shFF80;
      else                      return v[15:0];
    end
  endfunction

  always_ff @(posedge clk) begin
    out_valid <= v_r;
    out_idx   <= idx_r;
    out       <= sat(shifted, out16_r);
  end
endmodule
