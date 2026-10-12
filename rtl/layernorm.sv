// ---------------------------------------------------------------------------
// layernorm.sv -- integer LayerNorm without affine part
// (gamma/beta are folded into the following Linear layer offline).
//
//   s   = sum x_i                     x_i : INT16 residual, scale 2^-FR
//   d_i = N x_i - s                   = N (x_i - mean)        (exact)
//   Q   = sum d_i^2 + LN_E            = N^3 (var + eps) / s_r^2
//   rt  = floor(sqrt(Q))              (isqrt_seq)
//   inv = floor(LN_K / rt)            LN_K = sqrt(N) 2^LN_P   (div_seq)
//   z_i = sat8((d_i inv + 2^(LN_P-ZF-1)) >>> (LN_P-ZF))       INT8, scale 2^-ZF
//
// |d_i| <= rt, so |d_i * inv| <= LN_K: the product never overflows 64 bits,
// and |z| <= sqrt(N-1) (Samuelson) so Q3.4 never clips for N <= 64.
// Latency ~ 3N + W_div + W_sqrt/2 clocks.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module layernorm
  import gpt_pkg::*;
(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    start,
  // residual read port (asynchronous)
  output logic [EMB_AW-1:0]       x_addr,
  input  logic signed [15:0]      x_data,
  // normalised output write port
  output logic                    z_we,
  output logic [EMB_AW-1:0]       z_addr,
  output logic signed [7:0]       z_data,
  output logic                    done
);
  localparam int N  = N_EMBD;
  localparam int SH = LN_P - ZF;

  typedef enum logic [2:0] {IDLE, SUM, VAR, VAR_WAIT, SQRT, DIV, OUT, FIN} state_t;
  state_t st;

  logic [EMB_AW-1:0]  i;
  logic signed [31:0] s;
  logic [63:0]        Q;
  logic [31:0]        rt;
  logic [47:0]        inv;

  // d_i = N x_i - s  (combinational from the asynchronous read)
  logic signed [31:0] d;
  assign d      = N * x_data - s;
  assign x_addr = i;

  // ---- sequential sqrt / divide ----
  logic        sq_start, sq_done, dv_start, dv_done;
  logic [63:0] sq_num;
  logic [31:0] sq_root;
  logic [47:0] dv_quo;

  isqrt_seq #(.W(64)) u_sqrt (.clk(clk), .rst_n(rst_n), .start(sq_start), .num(sq_num),
                              .done(sq_done), .root(sq_root));
  div_seq   #(.W(48)) u_div  (.clk(clk), .rst_n(rst_n), .start(dv_start), .num(LN_K[47:0]),
                              .den({16'd0, rt}), .done(dv_done), .quo(dv_quo));

  // ---- pipeline registers ----
  logic signed [31:0] d_r;          // VAR: d_i for squaring / OUT: d_i for scaling
  logic               dv_v, dv_last, o_v;
  logic [EMB_AW-1:0]  o_idx;

  logic signed [63:0] d_sq_s;       // d_i^2 >= 0
  logic        [63:0] d_sq;
  assign d_sq_s = d_r * d_r;
  assign d_sq   = unsigned'(d_sq_s);

  logic signed [63:0] zprod, zshift;
  assign zprod  = d_r * $signed({1'b0, inv});
  assign zshift = (zprod + (64'sd1 <<< (SH - 1))) >>> SH;

  function automatic logic signed [7:0] sat8(input logic signed [63:0] v);
    if (v > 64'sd127)       return 8'sh7F;
    else if (v < -64'sd128) return 8'sh80;
    else                    return v[7:0];
  endfunction

  always_ff @(posedge clk) begin
    sq_start <= 1'b0;
    dv_start <= 1'b0;
    done     <= 1'b0;
    z_we     <= 1'b0;
    dv_v     <= 1'b0;
    o_v      <= 1'b0;
    if (!rst_n) begin
      st <= IDLE;
    end else begin
      case (st)
        IDLE: if (start) begin
          st <= SUM; i <= '0; s <= '0; Q <= '0;
        end
        SUM: begin                                   // s = sum x_i
          s <= s + 32'(x_data);
          if (i == EMB_AW'(N - 1)) begin st <= VAR; i <= '0; end
          else i <= i + 1'b1;
        end
        VAR: begin                                   // issue d_i
          d_r <= d; dv_v <= 1'b1; dv_last <= (i == EMB_AW'(N - 1));
          if (i == EMB_AW'(N - 1)) st <= VAR_WAIT;
          else i <= i + 1'b1;
        end
        VAR_WAIT: ;                                  // last square accumulates below
        SQRT: if (sq_done) begin
          rt <= sq_root; dv_start <= 1'b1; st <= DIV;
        end
        DIV: if (dv_done) begin
          inv <= dv_quo; st <= OUT; i <= '0;
        end
        OUT: begin                                   // issue d_i for scaling
          d_r <= d; o_v <= 1'b1; o_idx <= i;
          if (i == EMB_AW'(N - 1)) st <= FIN;
          else i <= i + 1'b1;
        end
        FIN: if (!o_v && !z_we) begin done <= 1'b1; st <= IDLE; end
        default: st <= IDLE;
      endcase

      // Q accumulation (one clock behind VAR issue)
      if (dv_v) begin
        Q <= Q + d_sq;
        if (dv_last) begin
          sq_num   <= Q + d_sq + LN_E;
          sq_start <= 1'b1;
          st       <= SQRT;
        end
      end

      // output scaling (one clock behind OUT issue)
      if (o_v) begin
        z_we   <= 1'b1;
        z_addr <= o_idx;
        z_data <= sat8(zshift);
      end
    end
  end
endmodule
