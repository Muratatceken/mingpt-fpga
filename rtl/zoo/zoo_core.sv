// ---------------------------------------------------------------------------
// zoo_core.sv -- microcoded INT8 inference engine for the Karpathy model zoo
//
// One RTL for all nine architectures (bigram ... nanochat core): a small
// controller fetches 256-bit instructions from prog.mem and runs each one on
// a shared unit. Everything model-specific (program, weights, requantisation
// parameters, tables, activation LUTs) lives in ROMs written by
// sw/zoo/export_hw.py, and the sizes come from the generated zoo_pkg.
// Bit-exact with sw/zoo/quant.py:IntRunner.
//
// Instruction word (see export_hw.py)
//   [7:0] op  [11:8] dst  [15:12] src0  [19:16] src1  [23:20] mode  [31:24] aux
//   [47:32] n0  [63:48] n1  [79:64] n2  [95:80] n3  [127:96] base0  [159:128] base1
//   [223:160] k64  [255:224] e32
//
// Storage
//   vrf        NBUF int8 vector buffers (A..G, Q, K, Vb, HS) as P-byte words:
//              one byte-enable write port, three asynchronous read ports
//   rmem/lmem  residual R and logits L (int16)
//   kmem/vmem  KV cache, one DH-byte word per (layer, position, kv head); DH = 8 = head size,
//              independent of the GEMV lane count P (P = 8, 16, 32: a head is a slice of a P-byte word)
//   smem       bag-of-words running sums (int32); hist: token history
// Units
//   GEMV  P-lane output-stationary MAC array, shadow bank, serial requantiser
//         (the requantiser of block b overlaps the accumulation of block b+1)
//   GEMVB the same unit with 1-bit weights (binary layers, sw/zoo/ir.py:factor_binary): a 64-bit word of
//         wbin.mem holds the signs of 64 outputs for one input, 64 add/subtract lanes, no multipliers;
//         the per-output scale is part of the requantisation. SCALE: input gains of a binary layer
//   NORM  LayerNorm / RMSNorm: one pass for sum and sum of squares,
//         normalised integer square root, one division per segment
//   ATTN  P-lane dot products, exp LUT, P-lane weighted sum, one division per head
//   misc  embeddings, RoPE, KV store, MUL, GRU update, COPY, REQ, BoW
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module zoo_core
  import zoo_pkg::*;
#(
  parameter string MEM_DIR    = MEM_PATH,
  parameter bit    SPECIALIZE = 1'b0,     // 1: drop the units this model does not use (OPS_USED)
  parameter bit    LOADABLE   = 1'b0,     // 1: model memories are RAMs filled through the ld_* port
  parameter bit    PRELOAD    = 1'b1      // 0: model memories start empty (must be loaded via ld_*)
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,      // run one decode step
  input  logic [TOK_W-1:0]   tok_in,
  input  logic [POS_W-1:0]   pos,
  output logic               busy,
  output logic               done,       // one-cycle pulse, logits valid
  input  logic [LA_W-1:0]    l_addr,     // logit read port
  output logic signed [15:0] l_data,
  output logic [7:0]         dbg_op,     // opcode in execution (16 = fetch/decode)
  input  logic               ld_we,      // model load (LOADABLE = 1): write one word of memory ld_sel
  input  logic [2:0]         ld_sel,     //   0 prog, 1 weights, 2 params, 3 tables, 4 luts, 5 binary weights
  input  logic [19:0]        ld_addr,
  input  logic [255:0]       ld_data
);
  localparam int LP  = $clog2(P);
  localparam int DH  = 8;                // attention / RoPE / KV lanes = head size
  localparam int LDH = 3;
  localparam logic [31:0] EN = SPECIALIZE ? 32'(OPS_USED) : 32'hFFFF_FFFF;   // enabled opcodes (GEMV always)
  localparam int BL  = 64;               // GEMVB lanes = bits of a binary weight word
  localparam int BLN = EN[18] ? BL : 1;  // binary accumulator bank (only if the model has GEMVB)
  localparam int BAW = 20;               // binary accumulators: |acc| <= 128 n_in < 2^19 for n_in <= 4096
  localparam int RPW = $clog2((BLN > P) ? BLN : P);
  localparam int GWA_W = (WA_W > WBA_W) ? WA_W : WBA_W;
  localparam string F_PROG = PRELOAD ? {MEM_DIR, "/prog.mem"}    : "";
  localparam string F_W    = PRELOAD ? {MEM_DIR, "/weights.mem"} : "";
  localparam string F_PRM  = PRELOAD ? {MEM_DIR, "/params.mem"}  : "";
  localparam string F_TAB  = PRELOAD ? {MEM_DIR, "/tables.mem"}  : "";
  localparam string F_LUT  = PRELOAD ? {MEM_DIR, "/luts.mem"}    : "";
  localparam string F_WB   = PRELOAD ? {MEM_DIR, "/wbin.mem"}    : "";

  localparam logic [7:0] OP_END = 8'd0, OP_EMB16 = 8'd1, OP_EMB8 = 8'd2, OP_EMBWIN = 8'd3,
                         OP_TABLOGIT = 8'd4, OP_HRESET = 8'd5, OP_NORM = 8'd6, OP_GEMV = 8'd7,
                         OP_ROPE = 8'd8, OP_KVSTORE = 8'd9, OP_ATTN = 8'd10, OP_MUL = 8'd11,
                         OP_GRUUPD = 8'd12, OP_COPY = 8'd13, OP_REQ = 8'd14, OP_BOW = 8'd15,
                         OP_SCALE = 8'd17, OP_GEMVB = 8'd18;          // 16: fetch / decode (dbg_op)
  localparam logic [3:0] B_HS = 4'd10;
  localparam logic [3:0] M_O8 = 4'd0, M_RADD = 4'd1, M_LUT = 4'd2, M_LOGIT = 4'd3;

  // ======================================================================
  // helpers
  // ======================================================================
  function automatic logic [VA_W-1:0] vaddr(input logic [3:0] b, input logic [15:0] i);
    return VA_W'(32'(b) * VW + 32'(i >> LP));
  endfunction

  function automatic logic signed [7:0] lane(input logic [8*P-1:0] w, input logic [15:0] i);
    return w[8*i[LP-1:0] +: 8];
  endfunction

  function automatic logic signed [7:0] sat8(input logic signed [63:0] v);
    if (v > 64'sd127)       return 8'sh7F;
    else if (v < -64'sd128) return 8'sh80;
    else                    return v[7:0];
  endfunction

  function automatic logic signed [15:0] sat16(input logic signed [63:0] v);
    if (v > 64'sd32767)       return 16'sh7FFF;
    else if (v < -64'sd32768) return 16'sh8000;
    else                      return v[15:0];
  endfunction

  function automatic logic [6:0] bitlen(input logic [63:0] v);
    logic [6:0] n;
    n = '0;
    for (int b = 0; b < 64; b++) if (v[b]) n = 7'(b + 1);
    return n;
  endfunction

  // ======================================================================
  // controller state and instruction fields
  // ======================================================================
  typedef enum logic [2:0] {S_IDLE, S_FETCH, S_DEC, S_EXEC, S_DONE} st_t;
  st_t st;

  logic [PC_W-1:0]  pc;
  logic [255:0]     ir;
  logic [255:0]     prog_dout;
  logic [TOK_W-1:0] tok;
  logic [POS_W-1:0] t;
  logic [3:0]       ph;
  logic [15:0]      i, j, c, h, sb, seg, kvh, rc, wb_cnt;
  logic             sub, iss_done;

  logic [7:0]  f_op;
  logic [3:0]  f_dst, f_src0, f_src1, f_mode;
  logic [7:0]  f_aux;
  logic [15:0] f_n0, f_n1, f_n2, f_n3;
  logic [31:0] f_b0, f_b1, f_e;
  logic [63:0] f_k;
  assign f_op   = ir[7:0];
  assign f_dst  = ir[11:8];
  assign f_src0 = ir[15:12];
  assign f_src1 = ir[19:16];
  assign f_mode = ir[23:20];
  assign f_aux  = ir[31:24];
  assign f_n0   = ir[47:32];
  assign f_n1   = ir[63:48];
  assign f_n2   = ir[79:64];
  assign f_n3   = ir[95:80];
  assign f_b0   = ir[127:96];
  assign f_b1   = ir[159:128];
  assign f_k    = ir[223:160];
  assign f_e    = ir[255:224];

  assign dbg_op = (st == S_EXEC) ? f_op : 8'd16;

  // ======================================================================
  // memories
  // ======================================================================
  // vector register file
  logic [8*P-1:0]  vrf [NBUF*VW];
  logic [VA_W-1:0] va_a, vb_a, vc_a, vw_a;
  logic [8*P-1:0]  va_d, vb_d, vc_d, vw_d;
  logic [P-1:0]    vw_be;
  assign va_d = vrf[va_a];
  assign vb_d = vrf[vb_a];
  assign vc_d = vrf[vc_a];
  always_ff @(posedge clk)
    for (int p = 0; p < P; p++) if (vw_be[p]) vrf[vw_a][8*p +: 8] <= vw_d[8*p +: 8];

  // residual R
  logic signed [15:0] rmem [C];
  logic [RA_W-1:0]    r_ra, r_wa;
  logic signed [15:0] r_rd, r_wd;
  logic               r_we;
  assign r_rd = rmem[r_ra];
  always_ff @(posedge clk) if (r_we) rmem[r_wa] <= r_wd;

  // logits L
  logic signed [15:0] lmem [V];
  logic [LA_W-1:0]    l_wa;
  logic signed [15:0] l_wd;
  logic               l_we;
  assign l_data = lmem[l_addr];
  always_ff @(posedge clk) if (l_we) lmem[l_wa] <= l_wd;

  // KV cache (synchronous read)
  logic [8*DH-1:0] kmem [KDEP];
  logic [8*DH-1:0] vmem [KDEP];
  logic [KA_W-1:0] kv_ra, kv_wa;
  logic            kv_we;
  logic [8*DH-1:0] k_rd, v_rd;
  logic [8*DH-1:0] kv_kd, kv_vd;          // the head slices of the K / Vb words being stored
  always_ff @(posedge clk) begin
    if (kv_we) begin
      kmem[kv_wa] <= kv_kd;
      vmem[kv_wa] <= kv_vd;
    end
    k_rd <= kmem[kv_ra];
    v_rd <= vmem[kv_ra];
  end

  // attention scores of the current head
  logic signed [23:0] sbuf [T];
  logic               sb_we;
  logic [15:0]        sb_wa;
  logic signed [23:0] sb_wd;
  always_ff @(posedge clk) if (sb_we) sbuf[POS_W'(sb_wa)] <= sb_wd;

  // bag-of-words running sums
  logic signed [31:0] smem [C];
  logic               sm_we;
  logic signed [31:0] sm_wd;
  always_ff @(posedge clk) if (sm_we) smem[RA_W'(i)] <= sm_wd;

  // token history
  logic [TOK_W-1:0] hist [T];
  always_ff @(posedge clk) if (st == S_IDLE && start) hist[pos] <= tok_in;

  // ROMs
  logic [WA_W-1:0]  w_a;   logic [8*P-1:0]   w_d;
  logic [PA_W-1:0]  p_a;   logic [63:0]      p_d;
  logic [TA_W-1:0]  t_a;   logic [TAB_W-1:0] t_d;
  logic [LUA_W-1:0] lu_a;  logic [7:0]       lu_d;
  zrom #(.WIDTH(256),   .DEPTH(PROG_DEPTH), .AW(PC_W),  .INIT(F_PROG), .LOADABLE(LOADABLE))
    u_prog (.clk, .addr(pc), .dout(prog_dout),
            .we(ld_we && ld_sel == 3'd0), .waddr(PC_W'(ld_addr)), .wdata(ld_data));
  zrom #(.WIDTH(8*P),   .DEPTH(W_DEPTH),    .AW(WA_W),  .INIT(F_W),    .LOADABLE(LOADABLE))
    u_w    (.clk, .addr(w_a), .dout(w_d),
            .we(ld_we && ld_sel == 3'd1), .waddr(WA_W'(ld_addr)), .wdata(ld_data[8*P-1:0]));
  zrom #(.WIDTH(64),    .DEPTH(PRM_DEPTH),  .AW(PA_W),  .INIT(F_PRM),  .LOADABLE(LOADABLE))
    u_p    (.clk, .addr(p_a), .dout(p_d),
            .we(ld_we && ld_sel == 3'd2), .waddr(PA_W'(ld_addr)), .wdata(ld_data[63:0]));
  zrom #(.WIDTH(TAB_W), .DEPTH(TAB_DEPTH),  .AW(TA_W),  .INIT(F_TAB),  .LOADABLE(LOADABLE))
    u_t    (.clk, .addr(t_a), .dout(t_d),
            .we(ld_we && ld_sel == 3'd3), .waddr(TA_W'(ld_addr)), .wdata(ld_data[TAB_W-1:0]));
  zrom #(.WIDTH(8),     .DEPTH(LUT_DEPTH),  .AW(LUA_W), .INIT(F_LUT),  .LOADABLE(LOADABLE))
    u_lut  (.clk, .addr(lu_a), .dout(lu_d),
            .we(ld_we && ld_sel == 3'd4), .waddr(LUA_W'(ld_addr)), .wdata(ld_data[7:0]));
  logic [BL-1:0]    wb_d;
  zrom #(.WIDTH(BL),    .DEPTH(WB_DEPTH),   .AW(WBA_W), .INIT(F_WB),   .LOADABLE(LOADABLE))
    u_wb   (.clk, .addr(WBA_W'(g_wa)), .dout(wb_d),
            .we(ld_we && ld_sel == 3'd5), .waddr(WBA_W'(ld_addr)), .wdata(ld_data[BL-1:0]));

  logic [15:0] exp_lut [256];             // 2^15 * 2^(-k/256), asynchronous LUT-ROM
  initial $readmemh({MEM_DIR, "/exp.mem"}, exp_lut);

  logic signed [31:0] t_dx;               // table word, sign-extended
  assign t_dx = 32'($signed(t_d));

  // ======================================================================
  // arithmetic units
  // ======================================================================
  logic               rq_in_v, rq_ov, rq_o16;
  logic [15:0]        rq_in_idx, rq_oidx;
  logic signed [31:0] rq_acc, rq_bias;
  logic [15:0]        rq_m0;
  logic [5:0]         rq_sh;
  logic signed [15:0] rq_o;
  requant #(.IDX_W(16)) u_rq (
    .clk, .in_valid(rq_in_v), .in_idx(rq_in_idx), .acc(rq_acc), .bias(rq_bias), .m0(rq_m0),
    .sh(rq_sh), .out16(rq_o16), .out_valid(rq_ov), .out_idx(rq_oidx), .out(rq_o));

  logic        isq_start, isq_done;
  logic [55:0] isq_num;
  logic [27:0] isq_root;
  isqrt_seq #(.W(56)) u_isq (.clk, .rst_n, .start(isq_start), .num(isq_num), .done(isq_done), .root(isq_root));

  logic        divn_start, divn_done;
  logic [63:0] divn_num, divn_den, divn_quo;
  zdiv #(.W(64), .QB(41)) u_divn (.clk, .rst_n, .start(divn_start), .num(divn_num), .den(divn_den),
                            .done(divn_done), .quo(divn_quo));

  logic        diva_start, diva_done;
  logic [33:0] diva_den, diva_quo;
  zdiv #(.W(34), .QB(18)) u_diva (.clk, .rst_n, .start(diva_start), .num(34'h1_0000_0000), .den(diva_den),
                            .done(diva_done), .quo(diva_quo));

  // ======================================================================
  // per-unit registers
  // ======================================================================
  // table streaming (EMB16 / EMB8 / EMBWIN / TABLOGIT / HRESET)
  logic               d_v, d_sub;
  logic [15:0]        d_idx;
  logic signed [31:0] st_tmp;
  // GEMV
  logic [15:0]        g_ob, g_i;
  logic [GWA_W-1:0]   g_wa;                 // weight address (INT8 or binary ROM)
  logic               a_v, a_first, a_last;
  logic signed [7:0]  a_x;
  logic [15:0]        a_ob;
  logic signed [31:0] acc [P];
  logic signed [31:0] shadow [P];
  logic signed [BAW-1:0] bacc [BLN];
  logic signed [BAW-1:0] bshadow [BLN];
  logic               rq_act;
  logic [RPW-1:0]     rq_p;
  logic [15:0]        rq_ob;
  logic               g1_v;
  logic signed [31:0] g1_acc;
  logic [15:0]        g1_idx;
  logic               lw_v;
  logic [15:0]        lw_idx;
  // NORM
  logic signed [23:0] n_s;            // sum x      (|x| < 2^15, n < 256)
  logic [39:0]        n_sq;           // sum x^2
  logic [55:0]        n_q;            // n (n sum x^2 - (sum x)^2) + E  or  sum x^2 + E
  logic [4:0]         n_k;
  logic [40:0]        n_inv;          // K 2^k / isqrt(Q 4^k) < 2^40
  // ATTN
  logic               s1_v, e1_v;
  logic [15:0]        s1_j;
  logic [15:0]        e1_e;
  logic signed [23:0] a_m;
  logic signed [33:0] att_a [DH];     // sum e * v
  logic [33:0]        att_e;          // sum of exp weights (>= 2^15)
  logic [17:0]        att_r;          // 2^32 / att_e <= 2^17
  // SCALE
  logic               sc1_v;
  logic [15:0]        sc1_i;
  logic signed [7:0]  sc1_x;
  // ROPE / BOW
  logic [8*P-1:0]     rp_w, rp_n;
  logic signed [15:0] rp_c;
  logic signed [21:0] rcp;

  // ======================================================================
  // combinational datapath
  // ======================================================================
  // ---- GEMV issue ----
  logic        g_sel1, g_islast, g_can, g_bin, g_on;
  logic [15:0] g_idx, g_nblk, g_lanes;
  logic signed [7:0] g_x;
  assign g_bin    = (f_op == OP_GEMVB) && EN[18];
  assign g_on     = (f_op == OP_GEMV) || g_bin;
  assign g_lanes  = g_bin ? 16'(BL) : 16'(P);
  assign g_sel1   = (g_i >= f_n0);
  assign g_idx    = g_sel1 ? g_i - f_n0 : g_i;
  assign g_islast = (g_i == f_n0 + f_n1 - 16'd1);
  assign g_nblk   = g_bin ? f_n2 >> 6 : f_n2 >> LP;
  logic  g_drlast;                       // GEMVB: the last real output is drained (padding lanes are skipped)
  assign g_drlast = (32'(rq_ob) * BL + 32'(rq_p) == 32'(f_n3) - 32'd1);
  assign g_can    = (st == S_EXEC) && g_on && (g_ob < g_nblk) &&
                    !(g_islast && (rq_act || (a_v && a_last)));

  logic signed [31:0] accn [P];
  always_comb
    for (int p = 0; p < P; p++)
      accn[p] = (a_first ? 32'sd0 : acc[p]) + $signed(w_d[8*p +: 8]) * a_x;

  logic signed [BAW-1:0] bx, bxn;        // GEMVB: x and -x, shared by the 64 add/subtract lanes
  logic signed [BAW-1:0] baccn [BLN];
  assign bx  = BAW'(a_x);
  assign bxn = -bx;
  always_comb
    for (int p = 0; p < BLN; p++)
      baccn[p] = (a_first ? BAW'(0) : bacc[p]) + (wb_d[p] ? bx : bxn);         // weight +1 / -1

  // ---- NORM ----
  logic               n_ln, n_src16, n_dst16;
  logic signed [15:0] nx;
  logic signed [31:0] nx2;
  logic signed [8:0]  n9;
  logic signed [47:0] ns48;
  logic [47:0]        q_t1, q_s2;
  logic [55:0]        q_ln, q_rms;
  logic signed [25:0] nd;
  logic signed [63:0] nprod, ny;
  logic [6:0]         n_bl;
  logic [4:0]         nk_c;
  assign n_ln    = f_mode[0];
  assign n_src16 = f_mode[1];
  assign n_dst16 = f_mode[2];
  always_comb begin
    nx    = n_src16 ? r_rd : 16'(lane(va_d, sb + j));
    nx2   = nx * nx;
    n9    = $signed({1'b0, f_n0[7:0]});
    ns48  = n_s;
    q_t1  = 48'(n_sq) * 48'(f_n0[7:0]);
    q_s2  = ns48 * ns48;                                   // >= 0
    q_ln  = 56'(f_n0[7:0]) * 56'(q_t1 - q_s2) + 56'(f_e);  // = sum (n x - s)^2 + E
    q_rms = 56'(n_sq) + 56'(f_e);
    n_bl  = bitlen(64'(n_q));
    nk_c  = (n_bl < 7'd40) ? 5'((7'd40 - n_bl) >> 1) : 5'd0;
    nd    = n_ln ? n9 * nx - n_s : nx;
    nprod = nd * $signed({1'b0, n_inv});
    ny    = (nprod + (64'sd1 <<< (f_aux - 8'd1))) >>> f_aux;
  end

  // ---- ATTN ----
  logic signed [23:0] sc;
  logic [24:0]        e_diff;
  logic [63:0]        e_u, e_n, e_kk;
  logic [15:0]        e_val;
  logic [5:0]         e_f;
  logic signed [63:0] att_y;
  always_comb begin
    sc = '0;                                              // q: head h is a DH-byte slice of a P-byte word
    for (int d = 0; d < DH; d++)
      sc += $signed(va_d[8 * (((h << LDH) + d) % P) +: 8]) * $signed(k_rd[8*d +: 8]);
    e_f    = f_k[5:0];
    e_diff = 25'(a_m - sbuf[POS_W'(j)]);
    e_u    = 64'(e_diff) * 64'(f_b1[15:0]);
    e_n    = e_u >> e_f;
    e_kk   = (e_u >> (e_f - 6'd8)) & 64'hFF;
    e_val  = (e_n > 64'd15) ? 16'd0 : (exp_lut[e_kk[7:0]] >> e_n[3:0]);
    att_y  = (64'(att_a[j[LDH-1:0]]) * $signed({1'b0, att_r}) + 64'sh8000_0000) >>> 32;
  end

  // ---- ROPE ----
  logic        rp_half;
  logic [15:0] rp_i1, rp_i2;
  logic signed [7:0]  rp_a, rp_b;
  logic signed [15:0] rp_s;
  logic signed [31:0] rp_y1, rp_y2;
  always_comb begin
    rp_half = f_mode[0];
    rp_i1   = ((h << LDH) % P) + (rp_half ? i : (i << 1));             // byte lanes inside the word
    rp_i2   = ((h << LDH) % P) + (rp_half ? i + (f_n1 >> 1) : (i << 1) + 16'd1);
    rp_a    = lane(rp_w, rp_i1);
    rp_b    = lane(rp_w, rp_i2);
    rp_s    = t_dx[15:0];
    if (rp_half) begin
      rp_y1 = (rp_a * rp_c + rp_b * rp_s + 32'sd8192) >>> 14;
      rp_y2 = (-rp_a * rp_s + rp_b * rp_c + 32'sd8192) >>> 14;
    end else begin
      rp_y1 = (rp_a * rp_c - rp_b * rp_s + 32'sd8192) >>> 14;
      rp_y2 = (rp_a * rp_s + rp_b * rp_c + 32'sd8192) >>> 14;
    end
  end

  // ---- misc elementwise ----
  logic signed [31:0] gz, ghb, ghs, gnew;
  logic signed [31:0] bw_s;
  logic signed [63:0] bw_avg;
  logic [15:0]        win_row;
  always_comb begin
    gz     = 32'(lane(va_d, i));
    ghb    = 32'(lane(vb_d, i));
    ghs    = 32'(lane(vc_d, i));
    gnew   = ghs + ((gz * (ghb - ghs) + 32'sd64) >>> 7);
    bw_s   = ((t == '0) ? 32'sd0 : smem[RA_W'(i)]) + 32'(r_rd);
    bw_avg = (64'(bw_s) * 64'(rcp) + 64'sd524288) >>> 20;
    win_row = (16'(t) >= j) ? 16'(hist[POS_W'(16'(t) - j)]) : f_n2;
  end

  // ---- stream ops ----
  logic        st_two;
  logic [15:0] st_n;
  assign st_two = (f_op == OP_EMB16) && f_mode[0];
  assign st_n   = (f_op == OP_EMBWIN) ? f_n0 * f_n1 : f_n0;

  // ---- write-back of the requantiser (GEMV / MUL / REQ) ----
  logic wb_fire, rq_keep, lw_keep;
  assign rq_keep = (f_op != OP_GEMV) || (rq_oidx < f_n3);
  assign lw_keep = (lw_idx < f_n3);
  assign wb_fire = (f_op == OP_GEMV && f_mode == M_LUT) ? lw_v : rq_ov;

  // ======================================================================
  // port / write multiplexing
  // ======================================================================
  always_comb begin
    va_a = '0; vb_a = '0; vc_a = vaddr(B_HS, i);
    vw_a = '0; vw_be = '0; vw_d = '0;
    r_ra = '0; r_we = 1'b0; r_wa = '0; r_wd = '0;
    l_we = 1'b0; l_wa = '0; l_wd = '0;
    w_a = WA_W'(g_wa); p_a = '0; t_a = '0; lu_a = '0;
    kv_ra = '0; kv_we = 1'b0; kv_wa = '0; kv_kd = '0; kv_vd = '0;
    sb_we = 1'b0; sb_wa = '0; sb_wd = '0;
    sm_we = 1'b0; sm_wd = '0;
    rq_in_v = 1'b0; rq_in_idx = '0; rq_acc = '0; rq_bias = '0; rq_m0 = '0; rq_sh = '0; rq_o16 = 1'b0;
    isq_start = 1'b0; isq_num = '0;
    divn_start = 1'b0; divn_num = '0; divn_den = '0;
    diva_start = 1'b0; diva_den = '0;

    if (st == S_EXEC) begin
      case (f_op)
        // ------------------------------------------------ table streaming
        OP_EMB16, OP_EMB8, OP_EMBWIN, OP_TABLOGIT, OP_HRESET: if (EN[1] || EN[2] || EN[3] || EN[4] || EN[5]) begin
          case (f_op)
            OP_EMB16:  t_a = TA_W'(sub ? f_b1 + 32'(t) * f_n0 + i : f_b0 + 32'(tok) * f_n0 + i);
            OP_EMBWIN: t_a = TA_W'(f_b0 + 32'(win_row) * f_n0 + c);
            OP_HRESET: t_a = TA_W'(f_b0 + i);
            default:   t_a = TA_W'(f_b0 + 32'(tok) * f_n0 + i);
          endcase
          if (d_v && !(st_two && !d_sub)) begin
            case (f_op)
              OP_EMB16: begin
                r_we = 1'b1; r_wa = RA_W'(d_idx);
                r_wd = sat16(64'(st_two ? st_tmp + t_dx : t_dx));
              end
              OP_TABLOGIT: begin
                l_we = 1'b1; l_wa = LA_W'(d_idx); l_wd = t_dx[15:0];
              end
              default: begin
                vw_a = vaddr(f_dst, d_idx); vw_be = P'(1) << d_idx[LP-1:0]; vw_d = {P{t_dx[7:0]}};
              end
            endcase
          end
        end

        // ------------------------------------------------ NORM
        OP_NORM: if (EN[6]) begin
          r_ra = RA_W'(sb + j);
          va_a = vaddr(f_src0, sb + j);
          if (ph == 4'd2) begin
            isq_start = 1'b1; isq_num = n_q << (2 * nk_c);
          end
          if (ph == 4'd3 && isq_done) begin
            divn_start = 1'b1; divn_num = f_k << n_k; divn_den = 64'(isq_root);
          end
          if (ph == 4'd5) begin
            if (n_dst16) begin
              r_we = 1'b1; r_wa = RA_W'(sb + j); r_wd = sat16(ny);
            end else begin
              vw_a = vaddr(f_dst, sb + j); vw_be = P'(1) << (sb + j) % P; vw_d = {P{sat8(ny)}};
            end
          end
        end

        // ------------------------------------------------ GEMV / GEMVB
        OP_GEMV, OP_GEMVB: begin
          va_a = vaddr(g_sel1 ? f_src1 : f_src0, g_idx);
          p_a  = PA_W'(f_b1 + 32'(rq_ob) * 32'(g_lanes) + 32'(rq_p));
          rq_in_v = g1_v; rq_in_idx = g1_idx; rq_acc = g1_acc;
          rq_bias = p_d[63:32]; rq_m0 = p_d[31:16]; rq_sh = p_d[5:0];
          rq_o16  = (f_mode == M_RADD) || (f_mode == M_LOGIT);
          if (rq_ov) begin
            case (f_mode)
              M_RADD: begin
                r_ra = RA_W'(rq_oidx);
                if (rq_keep) begin r_we = 1'b1; r_wa = RA_W'(rq_oidx); r_wd = sat16(64'(r_rd) + 64'(rq_o)); end
              end
              M_LOGIT: if (rq_keep) begin l_we = 1'b1; l_wa = LA_W'(rq_oidx); l_wd = rq_o; end
              M_LUT:   lu_a = LUA_W'({f_aux, rq_o[7:0]});
              default: if (rq_keep) begin
                vw_a = vaddr(f_dst, rq_oidx); vw_be = P'(1) << rq_oidx[LP-1:0]; vw_d = {P{rq_o[7:0]}};
              end
            endcase
          end
          if (lw_v && lw_keep) begin
            vw_a = vaddr(f_dst, lw_idx); vw_be = P'(1) << lw_idx[LP-1:0]; vw_d = {P{lu_d}};
          end
        end

        // ------------------------------------------------ MUL / REQ
        OP_MUL, OP_REQ: if (EN[11] || EN[14]) begin
          va_a = vaddr(f_src0, i);
          vb_a = vaddr(f_src1, i);
          r_ra = RA_W'(i);
          rq_in_v   = !iss_done;
          rq_in_idx = i;
          rq_acc    = (f_op == OP_MUL) ? 32'(lane(va_d, i)) * 32'(lane(vb_d, i)) : 32'(r_rd);
          rq_m0     = f_k[23:8];
          rq_sh     = f_k[5:0];
          if (rq_ov) begin
            vw_a = vaddr(f_dst, rq_oidx); vw_be = P'(1) << rq_oidx[LP-1:0]; vw_d = {P{rq_o[7:0]}};
          end
        end

        // ------------------------------------------------ SCALE: dst[i] = requant(src[i] * gain[i])
        OP_SCALE: if (EN[17]) begin
          va_a      = vaddr(f_src0, i);
          t_a       = TA_W'(f_b0 + 32'(i));
          rq_in_v   = sc1_v;
          rq_in_idx = sc1_i;
          rq_acc    = 32'(sc1_x) * t_dx;
          rq_m0     = f_k[23:8];
          rq_sh     = f_k[5:0];
          if (rq_ov) begin
            vw_a = vaddr(f_dst, rq_oidx); vw_be = P'(1) << rq_oidx[LP-1:0]; vw_d = {P{rq_o[7:0]}};
          end
        end

        // ------------------------------------------------ ROPE
        OP_ROPE: if (EN[8]) begin
          va_a = vaddr(f_src0, h << LDH);
          t_a  = TA_W'(((ph == 4'd1) ? f_b0 : f_b1) + 32'(t) * (f_n1 >> 1) + i);
          if (ph == 4'd4) begin
            vw_a = vaddr(f_dst, h << LDH); vw_be = '1; vw_d = rp_n;
          end
        end

        // ------------------------------------------------ KVSTORE
        OP_KVSTORE: if (EN[9]) begin
          va_a  = vaddr(f_src0, i << LDH);
          vb_a  = vaddr(f_src1, i << LDH);
          kv_kd = va_d[8 * ((i << LDH) % P) +: 8 * DH];
          kv_vd = vb_d[8 * ((i << LDH) % P) +: 8 * DH];
          kv_we = 1'b1;
          kv_wa = KA_W'((32'(f_aux) * T + 32'(t)) * NKVW + i);
        end

        // ------------------------------------------------ ATTN
        OP_ATTN: if (EN[10]) begin
          va_a  = vaddr(f_src0, h << LDH);
          kv_ra = KA_W'((32'(f_aux) * T + 32'(j)) * NKVW + kvh);
          if (s1_v) begin sb_we = 1'b1; sb_wa = s1_j; sb_wd = sc; end
          if (ph == 4'd4) begin diva_start = 1'b1; diva_den = att_e; end
          if (ph == 4'd6) begin
            vw_a = vaddr(f_dst, (h << LDH) + j); vw_be = P'(1) << ((h << LDH) + j) % P; vw_d = {P{sat8(att_y)}};
          end
        end

        // ------------------------------------------------ elementwise
        OP_COPY: if (EN[13]) begin
          va_a = vaddr(f_src0, i);
          vw_a = vaddr(f_dst, i); vw_be = P'(1) << i[LP-1:0]; vw_d = {P{lane(va_d, i)}};
        end
        OP_GRUUPD: if (EN[12]) begin
          va_a = vaddr(f_src0, i);
          vb_a = vaddr(f_src1, i);
          vw_a = vaddr(B_HS, i); vw_be = P'(1) << i[LP-1:0]; vw_d = {P{sat8(64'(gnew))}};
        end
        OP_BOW: if (EN[15]) begin                      // 1/(t+1) in 2^-20 units: two 16-bit table words
          t_a  = TA_W'(f_b0 + 2 * 32'(t) + ((ph == 4'd1) ? 32'd1 : 32'd0));
          r_ra = RA_W'(i);
          if (ph == 4'd3) begin
            sm_we = 1'b1; sm_wd = bw_s;
            r_we = 1'b1; r_wa = RA_W'(i); r_wd = sat16(64'(r_rd) + bw_avg);
          end
        end
        default: ;
      endcase
    end
  end

`ifndef SYNTHESIS
  // ======================================================================
  // simulation-only checks (Verilator --assert): addresses in range, no
  // INT32 accumulator overflow, step position inside the context
  // ======================================================================
  function automatic logic acc_overflow(input logic signed [31:0] a, input logic signed [7:0] w,
                                        input logic signed [7:0] x);
    logic signed [63:0] s;
    s = 64'(a) + 64'(w) * 64'(x);
    return (s > 64'sd2147483647) || (s < -64'sd2147483648);
  endfunction

  always @(posedge clk) if (rst_n) begin
    if (st == S_IDLE && start)
      assert (int'(pos) < T) else $error("position %0d outside the context T=%0d", pos, T);
    if (|vw_be) assert (int'(vw_a) < NBUF * VW) else $error("vector write address %0d out of range", vw_a);
    if (r_we)   assert (int'(r_wa) < C)         else $error("residual write address %0d out of range", r_wa);
    if (l_we)   assert (int'(l_wa) < V)         else $error("logit write address %0d out of range", l_wa);
    if (kv_we)  assert (int'(kv_wa) < KDEP)     else $error("KV write address %0d out of range", kv_wa);
    if (st == S_EXEC && f_op == OP_ATTN)
      assert (int'(kv_ra) < KDEP) else $error("KV read address %0d out of range", kv_ra);
    if (a_v && !g_bin)
      for (int p = 0; p < P; p++)
        assert (!acc_overflow(a_first ? 32'sd0 : acc[p], $signed(w_d[8*p +: 8]), a_x))
          else $error("GEMV accumulator overflow (lane %0d)", p);
    if (st == S_EXEC && g_bin)
      assert (f_n0 + f_n1 <= 16'd4096) else $error("GEMVB with %0d inputs: binary accumulators too narrow", f_n0 + f_n1);
  end
`endif

  // ======================================================================
  // sequencer
  // ======================================================================
  always_ff @(posedge clk) begin
    done <= 1'b0;
    d_v  <= 1'b0;
    a_v  <= 1'b0;
    g1_v <= 1'b0;
    lw_v <= 1'b0;
    s1_v <= 1'b0;
    e1_v <= 1'b0;
    if (!rst_n) begin
      st   <= S_IDLE;
      busy <= 1'b0;
      pc   <= '0;
    end else begin
      case (st)
        S_IDLE: if (start) begin
          tok  <= tok_in;
          t    <= pos;
          pc   <= '0;
          busy <= 1'b1;
          st   <= S_FETCH;
        end

        S_FETCH: st <= S_DEC;

        S_DEC: begin
          ir       <= prog_dout;
          ph       <= '0;
          {i, j, c, h, sb, seg, kvh, rc, wb_cnt} <= '0;
          sub      <= 1'b0;
          iss_done <= 1'b0;
          g_ob     <= '0;
          g_i      <= '0;
          g_wa     <= GWA_W'(prog_dout[127:96]);
          sc1_v    <= 1'b0;
          rq_act   <= 1'b0;
          if (prog_dout[7:0] == OP_END)
            st <= S_DONE;
          else if (prog_dout[7:0] == OP_HRESET && t != '0) begin   // state persists after t = 0
            pc <= pc + 1'b1;
            st <= S_FETCH;
          end else
            st <= S_EXEC;
        end

        S_EXEC: begin
          case (f_op)
            // -------------------------------------------- table streaming
            OP_EMB16, OP_EMB8, OP_EMBWIN, OP_TABLOGIT, OP_HRESET: if (EN[1] || EN[2] || EN[3] || EN[4] || EN[5]) begin
              if (!iss_done) begin
                d_v   <= 1'b1;
                d_sub <= sub;
                d_idx <= i;
                if (st_two && !sub)
                  sub <= 1'b1;
                else begin
                  sub <= 1'b0;
                  i   <= i + 1'b1;
                  if (c == f_n0 - 1'b1) begin c <= '0; j <= j + 1'b1; end
                  else c <= c + 1'b1;
                  if (i == st_n - 1'b1) iss_done <= 1'b1;
                end
              end
              if (d_v && st_two && !d_sub) st_tmp <= t_dx;
              if (iss_done && !d_v) begin pc <= pc + 1'b1; st <= S_FETCH; end
            end

            // -------------------------------------------- NORM
            OP_NORM: if (EN[6]) case (ph)
              4'd0: begin
                n_s  <= ((j == '0) ? 24'sd0 : n_s) + nx;
                n_sq <= ((j == '0) ? 40'd0 : n_sq) + 40'(nx2);
                if (j == f_n0 - 1'b1) begin j <= '0; ph <= 4'd1; end
                else j <= j + 1'b1;
              end
              4'd1: begin n_q <= n_ln ? q_ln : q_rms; ph <= 4'd2; end
              4'd2: begin n_k <= nk_c; ph <= 4'd3; end
              4'd3: if (isq_done) ph <= 4'd4;
              4'd4: if (divn_done) begin n_inv <= divn_quo[40:0]; ph <= 4'd5; end
              default: begin
                if (j == f_n0 - 1'b1) begin
                  j  <= '0;
                  sb <= sb + f_n0;
                  if (seg == f_n1 - 1'b1) begin pc <= pc + 1'b1; st <= S_FETCH; end
                  else begin seg <= seg + 1'b1; ph <= 4'd0; end
                end else j <= j + 1'b1;
              end
            endcase

            // -------------------------------------------- GEMV / GEMVB
            OP_GEMV, OP_GEMVB: begin
              if (g_can) begin
                a_v     <= 1'b1;
                a_x     <= lane(va_d, g_idx);
                a_first <= (g_i == '0);
                a_last  <= g_islast;
                a_ob    <= g_ob;
                g_wa    <= g_wa + 1'b1;
                if (g_islast) begin g_i <= '0; g_ob <= g_ob + 1'b1; end
                else g_i <= g_i + 1'b1;
              end
              if (rq_act) begin
                g1_v   <= 1'b1;
                g1_acc <= g_bin ? 32'(bshadow[rq_p % BLN]) : shadow[rq_p % P];
                g1_idx <= 16'(32'(rq_ob) * 32'(g_lanes) + 32'(rq_p));
                rq_p   <= rq_p + 1'b1;
                if (16'(rq_p) == g_lanes - 16'd1 || (g_bin && g_drlast)) rq_act <= 1'b0;
              end
              if (a_v) begin
                if (g_bin) for (int p = 0; p < BLN; p++) bacc[p] <= baccn[p];
                else       for (int p = 0; p < P; p++)   acc[p]  <= accn[p];
                if (a_last) begin
                  if (g_bin) for (int p = 0; p < BLN; p++) bshadow[p] <= baccn[p];
                  else       for (int p = 0; p < P; p++)   shadow[p]  <= accn[p];
                  rq_act <= 1'b1;
                  rq_p   <= '0;
                  rq_ob  <= a_ob;
                end
              end
              lw_v   <= rq_ov && (f_mode == M_LUT);
              lw_idx <= rq_oidx;
              if (wb_fire) wb_cnt <= wb_cnt + 1'b1;
              if (g_ob == g_nblk && wb_cnt == (g_bin ? f_n3 : f_n2)) begin pc <= pc + 1'b1; st <= S_FETCH; end
            end

            // -------------------------------------------- MUL / REQ
            OP_MUL, OP_REQ: if (EN[11] || EN[14]) begin
              if (!iss_done) begin
                i <= i + 1'b1;
                if (i == f_n0 - 1'b1) iss_done <= 1'b1;
              end
              if (wb_fire) wb_cnt <= wb_cnt + 1'b1;
              if (wb_cnt == f_n0) begin pc <= pc + 1'b1; st <= S_FETCH; end
            end

            // -------------------------------------------- SCALE (table word one clock after its address)
            OP_SCALE: if (EN[17]) begin
              sc1_v <= !iss_done;
              sc1_i <= i;
              sc1_x <= lane(va_d, i);
              if (!iss_done) begin
                i <= i + 1'b1;
                if (i == f_n0 - 1'b1) iss_done <= 1'b1;
              end
              if (rq_ov) wb_cnt <= wb_cnt + 1'b1;
              if (wb_cnt == f_n0) begin pc <= pc + 1'b1; st <= S_FETCH; end
            end

            // -------------------------------------------- ROPE
            OP_ROPE: if (EN[8]) case (ph)
              4'd0: begin rp_w <= va_d; rp_n <= va_d; i <= '0; ph <= 4'd1; end
              4'd1: ph <= 4'd2;                                    // cos address out
              4'd2: begin rp_c <= t_dx[15:0]; ph <= 4'd3; end      // sin address out, cos back
              4'd3: begin                                          // sin back: rotate one pair
                rp_n[8*rp_i1[LP-1:0] +: 8] <= sat8(64'(rp_y1));
                rp_n[8*rp_i2[LP-1:0] +: 8] <= sat8(64'(rp_y2));
                if (i == (f_n1 >> 1) - 1'b1) ph <= 4'd4;
                else begin i <= i + 1'b1; ph <= 4'd1; end
              end
              default: begin                                       // head written back
                if (h == f_n0 - 1'b1) begin pc <= pc + 1'b1; st <= S_FETCH; end
                else begin h <= h + 1'b1; ph <= 4'd0; end
              end
            endcase

            // -------------------------------------------- KVSTORE
            OP_KVSTORE: if (EN[9]) begin
              i <= i + 1'b1;
              if (i == (f_n0 >> LDH) - 1'b1) begin pc <= pc + 1'b1; st <= S_FETCH; end
            end

            // -------------------------------------------- ATTN
            OP_ATTN: if (EN[10]) begin
              case (ph)
                4'd0: begin                                        // scores: K rows 0..t
                  s1_v <= 1'b1;
                  s1_j <= j;
                  if (j == 16'(t)) begin j <= '0; ph <= 4'd1; end
                  else j <= j + 1'b1;
                end
                4'd1: begin                                        // last score lands
                  for (int d = 0; d < DH; d++) att_a[d] <= '0;
                  att_e <= '0;
                  ph    <= 4'd2;
                end
                4'd2: begin                                        // exp + V rows 0..t
                  e1_v <= 1'b1;
                  e1_e <= e_val;
                  if (j == 16'(t)) begin j <= '0; ph <= 4'd3; end
                  else j <= j + 1'b1;
                end
                4'd3: ph <= 4'd4;                                  // last accumulate lands
                4'd4: ph <= 4'd5;                                  // divider started
                4'd5: if (diva_done) begin att_r <= diva_quo[17:0]; j <= '0; ph <= 4'd6; end
                default: begin                                     // write P outputs
                  if (j == 16'(DH - 1)) begin
                    j <= '0;
                    if (rc == f_n3 - 1'b1) begin rc <= '0; kvh <= kvh + 1'b1; end
                    else rc <= rc + 1'b1;
                    if (h == f_n0 - 1'b1) begin pc <= pc + 1'b1; st <= S_FETCH; end
                    else begin h <= h + 1'b1; ph <= 4'd0; end
                  end else j <= j + 1'b1;
                end
              endcase
              if (s1_v) a_m <= (s1_j == '0 || sc > a_m) ? sc : a_m;
              if (e1_v) begin
                for (int d = 0; d < DH; d++)
                  att_a[d] <= att_a[d] + $signed({1'b0, e1_e}) * $signed(v_rd[8*d +: 8]);
                att_e <= att_e + 34'(e1_e);
              end
            end

            // -------------------------------------------- elementwise
            OP_COPY, OP_GRUUPD: if (EN[12] || EN[13]) begin
              i <= i + 1'b1;
              if (i == f_n0 - 1'b1) begin pc <= pc + 1'b1; st <= S_FETCH; end
            end

            OP_BOW: if (EN[15]) case (ph)
              4'd0: ph <= 4'd1;                                          // high word address out
              4'd1: begin rcp <= 22'(t_dx) <<< 16; ph <= 4'd2; end       // low word address out
              4'd2: begin rcp <= rcp | 22'(t_dx[15:0]); ph <= 4'd3; end
              default: begin
                i <= i + 1'b1;
                if (i == f_n0 - 1'b1) begin pc <= pc + 1'b1; st <= S_FETCH; end
              end
            endcase

            default: begin pc <= pc + 1'b1; st <= S_FETCH; end
          endcase
        end

        S_DONE: begin
          busy <= 1'b0;
          done <= 1'b1;
          st   <= S_IDLE;
        end

        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
