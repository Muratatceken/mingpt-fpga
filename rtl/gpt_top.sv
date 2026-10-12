// ---------------------------------------------------------------------------
// gpt_top.sv -- minGPT inference accelerator, one token per start pulse.
//
// Given the token at position `pos`, runs the full forward pass using the
// KV cache of positions 0..pos-1 and returns argmax(logits) as tok_out.
// The control sequence mirrors mingpt/model.py:
//
//   x = wte[tok] + wpe[pos]                                     S_EMB
//   for each block:
//     x = x + c_proj(attn(c_attn(ln_1(x))))     S_LN1 S_QKV S_ATT S_PROJ
//     x = x + mlp.c_proj(gelu(c_fc(ln_2(x))))   S_LN2 S_FC  S_MPROJ
//   logits = lm_head(ln_f(x))                   S_LNF S_HEAD
//
// On-chip buffers (all small, asynchronous read):
//   res  : residual stream          N_EMBD   x INT16 (scale 2^-FR)
//   xa   : LayerNorm output          N_EMBD   x INT8
//   xb   : attention / GELU output   N_FF     x INT8
//   qbuf : current query             N_EMBD   x INT8
//   kc,vc: KV cache                  N_LAYER*BLOCK*N_EMBD x INT8 each
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module gpt_top
  import gpt_pkg::*;
#(
  parameter string MEM_DIR = MEM_PATH
) (
  input  logic                          clk,
  input  logic                          rst_n,
  input  logic                          start,
  input  logic [TOK_W-1:0]              tok_in,
  input  logic [POS_W-1:0]              pos,
  output logic                          busy,
  output logic                          done,
  output logic [TOK_W-1:0]              tok_out,
  output logic [N_VOCAB-1:0][15:0]      logits,
  // debug: asynchronous read of the residual stream
  input  logic [EMB_AW-1:0]             dbg_addr,
  output logic signed [15:0]            dbg_res
);
  // ======================= state machine =======================
  typedef enum logic [3:0] {
    S_IDLE, S_EMB, S_LN1, S_QKV, S_ATT, S_PROJ, S_LN2, S_FC, S_MPROJ,
    S_LNF, S_HEAD, S_DONE
  } state_t;

  state_t             st;
  logic               kick;        // first clock of a state: start the sub-unit
  logic [LAYER_W-1:0] layer;
  logic [TOK_W-1:0]   tok_r;
  logic [POS_W-1:0]   pos_r;

  assign busy = (st != S_IDLE);

  // ======================= buffers =======================
  logic signed [15:0] res  [N_EMBD];
  logic signed [7:0]  xa   [N_EMBD];
  logic signed [7:0]  xb   [N_FF];
  logic signed [7:0]  qbuf [N_EMBD];
  logic signed [7:0]  kc   [KV_DEPTH];
  logic signed [7:0]  vc   [KV_DEPTH];

  // GELU LUTs: address {layer, h[7:0]} (distributed ROM)
  logic signed [7:0]  gelu_lut [N_LAYER * 256];
  initial $readmemh({MEM_DIR, "/gelu.mem"}, gelu_lut);

  assign dbg_res = res[dbg_addr];

  // ======================= embedding =======================
  localparam int WTE_D = N_VOCAB * N_EMBD;
  localparam int WPE_D = BLOCK * N_EMBD;
  localparam int WTE_AW = $clog2(WTE_D);
  localparam int WPE_AW = $clog2(WPE_D);
  logic [WTE_AW-1:0]        wte_addr;
  logic [WPE_AW-1:0]        wpe_addr;
  logic [15:0]              wte_q, wpe_q;
  logic [EMB_AW-1:0]        emb_i, emb_i1;
  logic                     emb_issue, emb_v1, emb_last1;

  assign wte_addr = WTE_AW'(tok_r * N_EMBD + emb_i);
  assign wpe_addr = WPE_AW'(pos_r * N_EMBD + emb_i);

  rom #(.WIDTH(16), .DEPTH(WTE_D), .INIT({MEM_DIR, "/wte.mem"})) u_wte (
    .clk(clk), .addr(wte_addr), .dout(wte_q));
  rom #(.WIDTH(16), .DEPTH(WPE_D), .INIT({MEM_DIR, "/wpe.mem"})) u_wpe (
    .clk(clk), .addr(wpe_addr), .dout(wpe_q));

  // ======================= sub-units =======================
  // LayerNorm: res -> xa
  logic                     ln_start, ln_done, ln_we;
  logic [EMB_AW-1:0]        ln_xaddr, ln_zaddr;
  logic signed [7:0]        ln_zdata;
  assign ln_start = kick && (st == S_LN1 || st == S_LN2 || st == S_LNF);

  layernorm u_ln (
    .clk(clk), .rst_n(rst_n), .start(ln_start),
    .x_addr(ln_xaddr), .x_data(res[ln_xaddr]),
    .z_we(ln_we), .z_addr(ln_zaddr), .z_data(ln_zdata), .done(ln_done));

  // Attention: qbuf, kc, vc -> xb[0 .. N_EMBD-1]
  logic                     att_start, att_done, att_we;
  logic [EMB_AW-1:0]        att_qaddr, att_yaddr;
  logic [KV_AW-1:0]         att_kvaddr;
  logic signed [7:0]        att_ydata;
  assign att_start = kick && (st == S_ATT);

  attention #(.MEM_DIR(MEM_DIR)) u_att (
    .clk(clk), .rst_n(rst_n), .start(att_start), .layer(layer), .pos(pos_r),
    .q_addr(att_qaddr), .q_data(qbuf[att_qaddr]),
    .kv_addr(att_kvaddr), .k_data(kc[att_kvaddr]), .v_data(vc[att_kvaddr]),
    .y_we(att_we), .y_addr(att_yaddr), .y_data(att_ydata), .done(att_done));

  // GEMV: xa or xb -> stream of requantised outputs
  logic                     g_start, g_done, g_out16, g_ov;
  logic [W_AW-1:0]          g_wbase;
  logic [PRM_AW-1:0]        g_pbase;
  logic [DIM_W-1:0]         g_nin, g_nout, g_xaddr, g_oidx;
  logic signed [7:0]        g_xdata;
  logic signed [15:0]       g_odata;
  logic                     g_src_xb;

  assign g_start = kick && (st == S_QKV || st == S_PROJ || st == S_FC ||
                            st == S_MPROJ || st == S_HEAD);

  always_comb begin
    g_wbase = '0; g_pbase = '0; g_nin = '0; g_nout = '0; g_out16 = 1'b0; g_src_xb = 1'b0;
    case (st)
      S_QKV:   begin g_wbase = W_AW'(layer * W_PER_LAYER + W_OFF_ATTN);
                     g_pbase = PRM_AW'(layer * PRM_PER_LAYER + PRM_OFF_ATTN);
                     g_nin = DIM_W'(N_EMBD); g_nout = DIM_W'(3 * N_EMBD); end
      S_PROJ:  begin g_wbase = W_AW'(layer * W_PER_LAYER + W_OFF_PROJ);
                     g_pbase = PRM_AW'(layer * PRM_PER_LAYER + PRM_OFF_PROJ);
                     g_nin = DIM_W'(N_EMBD); g_nout = DIM_W'(N_EMBD);
                     g_out16 = 1'b1; g_src_xb = 1'b1; end
      S_FC:    begin g_wbase = W_AW'(layer * W_PER_LAYER + W_OFF_FC);
                     g_pbase = PRM_AW'(layer * PRM_PER_LAYER + PRM_OFF_FC);
                     g_nin = DIM_W'(N_EMBD); g_nout = DIM_W'(N_FF); end
      S_MPROJ: begin g_wbase = W_AW'(layer * W_PER_LAYER + W_OFF_MPROJ);
                     g_pbase = PRM_AW'(layer * PRM_PER_LAYER + PRM_OFF_MPROJ);
                     g_nin = DIM_W'(N_FF); g_nout = DIM_W'(N_EMBD);
                     g_out16 = 1'b1; g_src_xb = 1'b1; end
      S_HEAD:  begin g_wbase = W_AW'(W_OFF_HEAD);
                     g_pbase = PRM_AW'(PRM_OFF_HEAD);
                     g_nin = DIM_W'(N_EMBD); g_nout = DIM_W'(VOCAB_PAD);
                     g_out16 = 1'b1; end
      default: ;
    endcase
  end

  assign g_xdata = g_src_xb ? xb[g_xaddr] : xa[EMB_AW'(g_xaddr)];

  gemv #(.MEM_DIR(MEM_DIR)) u_gemv (
    .clk(clk), .rst_n(rst_n), .start(g_start),
    .w_base(g_wbase), .p_base(g_pbase), .n_in(g_nin), .n_out(g_nout), .out16(g_out16),
    .x_addr(g_xaddr), .x_data(g_xdata),
    .out_valid(g_ov), .out_idx(g_oidx), .out_data(g_odata), .done(g_done));

  // ======================= writeback =======================
  function automatic logic signed [15:0] sat16(input logic signed [16:0] v);
    if (v > 17'sd32767)       return 16'sh7FFF;
    else if (v < -17'sd32768) return 16'sh8000;
    else                      return v[15:0];
  endfunction

  logic [KV_AW-1:0] kv_wbase;
  assign kv_wbase = KV_AW'((layer * BLOCK + pos_r) * N_EMBD);

  logic signed [15:0] best_val;

  // residual stream write port: embedding, or residual add from c_proj / mlp.c_proj
  logic               res_we;
  logic [EMB_AW-1:0]  res_waddr;
  logic signed [15:0] res_wdata;
  always_comb begin
    res_we    = emb_v1;
    res_waddr = emb_i1;
    res_wdata = sat16($signed(wte_q) + $signed(wpe_q));
    if (g_ov && (st == S_PROJ || st == S_MPROJ)) begin
      res_we    = 1'b1;
      res_waddr = EMB_AW'(g_oidx);
      res_wdata = sat16(res[EMB_AW'(g_oidx)] + g_odata);
    end
  end

  // xb write port: attention output, or GELU(c_fc)
  logic               xb_we;
  logic [DIM_W-1:0]   xb_waddr;
  logic signed [7:0]  xb_wdata;
  always_comb begin
    xb_we    = att_we;
    xb_waddr = DIM_W'(att_yaddr);
    xb_wdata = att_ydata;
    if (g_ov && st == S_FC) begin
      xb_we    = 1'b1;
      xb_waddr = g_oidx;
      xb_wdata = gelu_lut[{layer, g_odata[7:0]}];
    end
  end

  always_ff @(posedge clk) begin
    if (res_we) res[res_waddr] <= res_wdata;
    if (ln_we)  xa[ln_zaddr]   <= ln_zdata;
    if (xb_we)  xb[xb_waddr]   <= xb_wdata;

    // QKV projection: q -> qbuf, k/v -> cache row (layer, pos)
    if (g_ov && st == S_QKV) begin
      if (g_oidx < DIM_W'(N_EMBD))
        qbuf[EMB_AW'(g_oidx)] <= g_odata[7:0];
      else if (g_oidx < DIM_W'(2 * N_EMBD))
        kc[kv_wbase + KV_AW'(g_oidx - DIM_W'(N_EMBD))] <= g_odata[7:0];
      else
        vc[kv_wbase + KV_AW'(g_oidx - DIM_W'(2 * N_EMBD))] <= g_odata[7:0];
    end
  end

  // ======================= control =======================
  always_ff @(posedge clk) begin
    kick   <= 1'b0;
    done   <= 1'b0;
    emb_v1 <= 1'b0;
    if (!rst_n) begin
      st        <= S_IDLE;
      emb_issue <= 1'b0;
      tok_out   <= '0;
    end else begin
      // embedding pipeline (ROM latency 1)
      if (emb_issue) begin
        emb_v1    <= 1'b1;
        emb_i1    <= emb_i;
        emb_last1 <= (emb_i == EMB_AW'(N_EMBD - 1));
        if (emb_i == EMB_AW'(N_EMBD - 1)) emb_issue <= 1'b0;
        else emb_i <= emb_i + 1'b1;
      end

      // argmax over the real vocabulary (first maximum wins)
      if (g_ov && st == S_HEAD && g_oidx < DIM_W'(N_VOCAB)) begin
        logits[g_oidx[TOK_W-1:0]] <= g_odata;
        if (g_oidx == '0 || g_odata > best_val) begin
          best_val <= g_odata;
          tok_out  <= g_oidx[TOK_W-1:0];
        end
      end

      case (st)
        S_IDLE:  if (start) begin
                   tok_r <= tok_in; pos_r <= pos; layer <= '0;
                   emb_i <= '0; emb_issue <= 1'b1;
                   st <= S_EMB;
                 end
        S_EMB:   if (emb_v1 && emb_last1) begin st <= S_LN1; kick <= 1'b1; end
        S_LN1:   if (ln_done)  begin st <= S_QKV;  kick <= 1'b1; end
        S_QKV:   if (g_done)   begin st <= S_ATT;  kick <= 1'b1; end
        S_ATT:   if (att_done) begin st <= S_PROJ; kick <= 1'b1; end
        S_PROJ:  if (g_done)   begin st <= S_LN2;  kick <= 1'b1; end
        S_LN2:   if (ln_done)  begin st <= S_FC;   kick <= 1'b1; end
        S_FC:    if (g_done)   begin st <= S_MPROJ; kick <= 1'b1; end
        S_MPROJ: if (g_done) begin
                   kick <= 1'b1;
                   if (layer == LAYER_W'(N_LAYER - 1)) st <= S_LNF;
                   else begin layer <= layer + 1'b1; st <= S_LN1; end
                 end
        S_LNF:   if (ln_done)  begin st <= S_HEAD; kick <= 1'b1; end
        S_HEAD:  if (g_done)   st <= S_DONE;
        S_DONE:  begin done <= 1'b1; st <= S_IDLE; end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
