// ---------------------------------------------------------------------------
// attention.sv -- causal multi-head attention for ONE new token (KV-cache decode)
//
// For every head h, over cached positions j = 0..pos:
//   SCORE : s_j = sum_d q[d] k_j[d]              (1/sqrt(hd) folded into W_q)
//           m   = max_j s_j
//   EXP   : u_j = (m - s_j) * CEXP               ( = -log2(e^x) in Q.FEXP )
//           e_j = EXP_LUT[frac8(u_j)] >> int(u_j)       UQ1.15, e_max = 1.0
//           E   = sum_j e_j                      (>= 2^15, never zero)
//   DIV   : R   = floor(2^ATT_RS / E)            (one division per head)
//   AV    : A_d = sum_j e_j v_j[d]
//           y_d = sat8((A_d R + 2^(ATT_RS-1)) >>> ATT_RS)
// Normalisation is deferred to the end (one reciprocal, no per-weight
// division). y is a convex combination of V rows, so it keeps V's scale.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module attention
  import gpt_pkg::*;
#(
  parameter string MEM_DIR = MEM_PATH
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    start,
  input  logic [LAYER_W-1:0]      layer,
  input  logic [POS_W-1:0]        pos,
  // query buffer and KV cache read ports (asynchronous)
  output logic [EMB_AW-1:0]       q_addr,
  input  logic signed [7:0]       q_data,
  output logic [KV_AW-1:0]        kv_addr,
  input  logic signed [7:0]       k_data,
  input  logic signed [7:0]       v_data,
  // output write port (concatenated heads, INT8, scale s_v)
  output logic                    y_we,
  output logic [EMB_AW-1:0]       y_addr,
  output logic signed [7:0]       y_data,
  output logic                    done
);
  localparam int DW = (HEAD_DIM > 1) ? $clog2(HEAD_DIM) : 1;

  typedef enum logic [2:0] {IDLE, SCORE, EXP, EXP_WAIT, DIV, AV, FIN} state_t;
  state_t st;

  // exp LUT: 2^(-k/256) in UQ1.15 (distributed ROM)
  logic [15:0] exp_lut [256];
  initial $readmemh({MEM_DIR, "/exp.mem"}, exp_lut);

  logic [LAYER_W-1:0] layer_r;
  logic [POS_W-1:0]   pos_r;
  logic [15:0]        cexp;
  logic [7:0]         fexp;
  logic [HEAD_W-1:0]  h;
  logic [POS_W-1:0]   j;
  logic [DW-1:0]      d;

  logic signed [31:0] s [BLOCK];     // raw scores
  logic        [15:0] e [BLOCK];     // exp weights, UQ1.15
  logic signed [31:0] m;             // running max
  logic        [31:0] e_sum;         // softmax denominator E
  logic        [47:0] R;             // 2^ATT_RS / E
  logic signed [31:0] acc;

  // ---- addresses ----
  logic [EMB_AW-1:0] col;
  assign col     = EMB_AW'(h * HEAD_DIM + d);
  assign q_addr  = col;
  assign kv_addr = KV_AW'((layer_r * BLOCK + j) * N_EMBD + col);

  // ---- SCORE / AV multiply-accumulate (one MAC per clock) ----
  logic signed [15:0] qk;
  logic signed [24:0] ev;
  logic signed [31:0] acc_qk, acc_av;
  assign qk     = q_data * k_data;                        // INT8 x INT8
  assign ev     = $signed({1'b0, e[j]}) * v_data;         // UQ1.15 x INT8
  assign acc_qk = ((d == '0) ? 32'sd0 : acc) + qk;
  assign acc_av = ((j == '0) ? 32'sd0 : acc) + ev;

  // ---- divider ----
  logic        dv_start, dv_done;
  logic [47:0] dv_quo;
  div_seq #(.W(48)) u_div (.clk(clk), .rst_n(rst_n), .start(dv_start),
                           .num(48'd1 << ATT_RS), .den({16'd0, e_sum}),
                           .done(dv_done), .quo(dv_quo));

  // ---- EXP pipeline: stage A computes u, stage B the LUT/shift ----
  logic        ea_v, ea_last;
  logic [POS_W-1:0] ea_j;
  logic [47:0] u_r;
  logic [31:0] delta;
  assign delta = unsigned'(m - s[j]);

  logic [47:0] u_int, u_frac;
  logic [15:0] e_new;
  always_comb begin
    u_int  = u_r >> fexp;
    u_frac = u_r >> (fexp - 8'd8);
    e_new  = (u_int > 48'd15) ? 16'd0 : (exp_lut[u_frac[7:0]] >> u_int[3:0]);
  end

  // ---- output stage ----
  logic               f_v;
  logic signed [31:0] a_fin;
  logic [EMB_AW-1:0]  f_idx;
  logic signed [63:0] yprod, yshift;
  assign yprod  = a_fin * $signed({1'b0, R});
  assign yshift = (yprod + (64'sd1 <<< (ATT_RS - 1))) >>> ATT_RS;

  function automatic logic signed [7:0] sat8(input logic signed [63:0] v);
    if (v > 64'sd127)       return 8'sh7F;
    else if (v < -64'sd128) return 8'sh80;
    else                    return v[7:0];
  endfunction

  always_ff @(posedge clk) begin
    done     <= 1'b0;
    dv_start <= 1'b0;
    ea_v     <= 1'b0;
    f_v      <= 1'b0;
    y_we     <= 1'b0;
    if (!rst_n) begin
      st <= IDLE;
    end else begin
      case (st)
        IDLE: if (start) begin
          layer_r <= layer; pos_r <= pos;
          cexp <= CEXP[layer]; fexp <= FEXP[layer];
          h <= '0; j <= '0; d <= '0;
          st <= SCORE;
        end

        SCORE: begin                                   // s_j = q . k_j
          acc <= acc_qk;
          if (d == DW'(HEAD_DIM - 1)) begin
            d    <= '0;
            s[j] <= acc_qk;
            if (j == '0 || acc_qk > m) m <= acc_qk;
            if (j == pos_r) begin j <= '0; e_sum <= '0; st <= EXP; end
            else j <= j + 1'b1;
          end else d <= d + 1'b1;
        end

        EXP: begin                                     // stage A: u_j
          u_r <= delta * cexp;                       // 32 x 16 bit
          ea_v <= 1'b1; ea_j <= j; ea_last <= (j == pos_r);
          if (j == pos_r) st <= EXP_WAIT;
          else j <= j + 1'b1;
        end

        EXP_WAIT: ;                                    // stage B finishes below

        DIV: if (dv_done) begin
          R <= dv_quo; j <= '0; d <= '0; st <= AV;
        end

        AV: begin                                      // A_d = sum_j e_j v_j[d]
          acc <= acc_av;
          if (j == pos_r) begin
            j <= '0;
            a_fin <= acc_av; f_v <= 1'b1; f_idx <= col;
            if (d == DW'(HEAD_DIM - 1)) begin
              d <= '0;
              if (h == HEAD_W'(N_HEAD - 1)) st <= FIN;
              else begin h <= h + 1'b1; st <= SCORE; end
            end else d <= d + 1'b1;
          end else j <= j + 1'b1;
        end

        FIN: if (!f_v && !y_we) begin done <= 1'b1; st <= IDLE; end

        default: st <= IDLE;
      endcase

      // EXP stage B
      if (ea_v) begin
        e[ea_j] <= e_new;
        e_sum <= e_sum + 32'(e_new);
        if (ea_last) begin dv_start <= 1'b1; st <= DIV; end
      end

      // output stage: y = A * R / 2^ATT_RS
      if (f_v) begin
        y_we   <= 1'b1;
        y_addr <= f_idx;
        y_data <= sat8(yshift);
      end
    end
  end
endmodule
