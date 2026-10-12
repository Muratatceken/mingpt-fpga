// ---------------------------------------------------------------------------
// gemv.sv -- matrix-vector engine:  y = requant(W x + b)
//
// Output-stationary dataflow with P parallel MAC lanes:
//   * every clock one activation x[i] is broadcast to all lanes and one weight
//     ROM word {W[o+P-1][i], ..., W[o][i]} (P x INT8) is read,
//   * after n_in clocks the P INT32 accumulators hold outputs o..o+P-1,
//   * they are copied to a shadow bank and requantised one per clock while
//     the lanes already start the next block of P outputs (no bubbles).
//
// Cycles per call ~= n_out/P * n_in + P + 4.   Requires n_in > P.
//
// Pipeline:
//   issue : w_rom addr = w_ptr, x_addr = i            (x read asynchronously)
//   mac   : acc[p] += W[p] * x                        (ROM data valid)
//   rq    : param ROM read -> requant (2 stages)      -> out_valid
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module gemv
  import gpt_pkg::*;
#(
  parameter string MEM_DIR = MEM_PATH
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    start,
  input  logic [W_AW-1:0]         w_base,
  input  logic [PRM_AW-1:0]       p_base,
  input  logic [DIM_W-1:0]        n_in,
  input  logic [DIM_W-1:0]        n_out,   // multiple of P
  input  logic                    out16,
  // activation read port (asynchronous)
  output logic [DIM_W-1:0]        x_addr,
  input  logic signed [7:0]       x_data,
  // result stream
  output logic                    out_valid,
  output logic [DIM_W-1:0]        out_idx,
  output logic signed [15:0]      out_data,
  output logic                    done
);
  localparam int PW = (P > 1) ? $clog2(P) : 1;

  // ---------------- memories ----------------
  logic [W_AW-1:0]   w_ptr;
  logic [8*P-1:0]    w_word;
  logic [PRM_AW-1:0] prm_addr;
  logic [63:0]       prm_word;

  rom #(.WIDTH(8*P), .DEPTH(W_DEPTH),   .INIT({MEM_DIR, "/weights.mem"})) u_wrom (
    .clk(clk), .addr(w_ptr), .dout(w_word));
  rom #(.WIDTH(64),  .DEPTH(PRM_DEPTH), .INIT({MEM_DIR, "/params.mem"}))  u_prom (
    .clk(clk), .addr(prm_addr), .dout(prm_word));

  // ---------------- configuration ----------------
  logic [PRM_AW-1:0] p_base_r;
  logic [DIM_W-1:0]  n_in_r, n_out_r;
  logic              out16_r;

  // ---------------- issue stage ----------------
  logic             issuing;
  logic [DIM_W-1:0] i_cnt, obase;

  assign x_addr = i_cnt;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      issuing <= 1'b0;
    end else if (start) begin
      issuing  <= 1'b1;
      w_ptr    <= w_base;
      i_cnt    <= '0;
      obase    <= '0;
      p_base_r <= p_base;
      n_in_r   <= n_in;
      n_out_r  <= n_out;
      out16_r  <= out16;
    end else if (issuing) begin
      w_ptr <= w_ptr + 1'b1;
      if (i_cnt == n_in_r - 1'b1) begin
        i_cnt <= '0;
        obase <= obase + P[DIM_W-1:0];
        if ({1'b0, obase} + (DIM_W+1)'(P) >= {1'b0, n_out_r}) issuing <= 1'b0;
      end else begin
        i_cnt <= i_cnt + 1'b1;
      end
    end
  end

  // ---------------- MAC stage ----------------
  logic                    v1, first1, last1;
  logic [DIM_W-1:0]        obase1;
  logic signed [7:0]       x1;
  logic signed [15:0]      prod   [P];
  logic signed [31:0]      acc    [P];
  logic signed [31:0]      acc_nx [P];
  logic signed [31:0]      acc_sh [P];   // shadow bank feeding the requantiser

  always_ff @(posedge clk) begin
    if (!rst_n) v1 <= 1'b0;
    else        v1 <= issuing;
    first1 <= (i_cnt == '0);
    last1  <= (i_cnt == n_in_r - 1'b1);
    obase1 <= obase;
    x1     <= x_data;
  end

  always_comb begin
    for (int p = 0; p < P; p++) begin
      prod[p]   = $signed(w_word[8*p +: 8]) * x1;          // INT8 x INT8 -> INT16
      acc_nx[p] = (first1 ? 32'sd0 : acc[p]) + prod[p];    // INT32 accumulate
    end
  end

  logic handoff;
  assign handoff = v1 && last1;

  always_ff @(posedge clk) begin
    if (v1) begin
      for (int p = 0; p < P; p++) acc[p] <= acc_nx[p];
    end
    if (handoff) begin
      for (int p = 0; p < P; p++) acc_sh[p] <= acc_nx[p];
    end
  end

  // ---------------- requant feed ----------------
  logic             rq_busy, rqv1;
  logic [PW-1:0]    rq_p, rq_p1;
  logic [DIM_W-1:0] rq_obase, rq_idx1;

  assign prm_addr = p_base_r + PRM_AW'(rq_obase) + PRM_AW'(rq_p);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rq_busy <= 1'b0;
      rqv1    <= 1'b0;
    end else begin
      if (handoff) begin
        rq_busy  <= 1'b1;
        rq_p     <= '0;
        rq_obase <= obase1;
      end else if (rq_busy) begin
        rq_p <= rq_p + 1'b1;
        if (rq_p == PW'(P - 1)) rq_busy <= 1'b0;
      end
      rqv1 <= rq_busy;
    end
    rq_p1   <= rq_p;
    rq_idx1 <= rq_obase + DIM_W'(rq_p);
  end

  requant #(.IDX_W(DIM_W)) u_rq (
    .clk      (clk),
    .in_valid (rqv1),
    .in_idx   (rq_idx1),
    .acc      (acc_sh[rq_p1]),
    .bias     (prm_word[63:32]),
    .m0       (prm_word[31:16]),
    .sh       (prm_word[5:0]),
    .out16    (out16_r),
    .out_valid(out_valid),
    .out_idx  (out_idx),
    .out      (out_data)
  );

  // ---------------- completion ----------------
  logic [DIM_W-1:0] out_cnt;
  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n || start) begin
      out_cnt <= '0;
    end else if (out_valid) begin
      out_cnt <= out_cnt + 1'b1;
      if (out_cnt == n_out_r - 1'b1) done <= 1'b1;
    end
  end

`ifndef SYNTHESIS
  always @(posedge clk) if (start) assert (n_in > P) else $error("gemv: n_in must exceed P");
`endif
endmodule
