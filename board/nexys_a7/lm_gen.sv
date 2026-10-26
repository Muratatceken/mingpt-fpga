// ---------------------------------------------------------------------------
// lm_gen.sv -- stand-alone text generation on the FPGA (Nexys A7-100T)
//
// The zoo engine (rtl/zoo/zoo_core.sv) with its model in BRAM, plus everything a language model needs
// around it to run without a host computer:
//   generator  start token -> engine step -> next token (greedy or sampled) -> next step ...
//   sampler    temperature sampling with min-p cut-off on the engine's INT16 logits, xorshift32 random
//              numbers; exponentials from the same 2^(-k/256) table as the attention unit
//   screen     detokeniser (token -> display bytes) and an 80-column terminal on the text RAM of the VGA
//              unit: word wrap, scrolling; status row with tokens, tokens/s, cycles/token, mode, check
//   check      hash of the generated tokens against the value sw/zoo/board.py computed for the greedy
//              run and for the first sampled run after a reseed (all switches down)
// Bit-exact with sw/zoo/board.py:generate (tokens, random state, screen contents).
//
// Text RAM: row 0 title, rows 1..ROWS text (circular, scroll offset `top`), row ROWS+1 status.
// Cycle count of a run: every cycle from the first engine step to the last token except the typewriter
// pause: engine, sampler, and any wait for the screen (it normally overlaps with the next engine step and
// costs nothing; only bigram, faster than a status-row update, waits for it).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module lm_gen
  import zoo_pkg::*;
  import board_pkg::*;
#(
  parameter string MEM_DIR  = MEM_PATH,
  parameter int    CLK_HZ   = 50_000_000,
  parameter int    SLOW_CYC = 2_000_000          // typewriter mode: pause after every token
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        run,                       // pulse: one generation run (random state continues)
  input  logic        reseed,                    // pulse: seed the random generator from `seed`, then run
  input  logic        clear,                     // pulse: clear the text area
  input  logic        greedy,
  input  logic [2:0]  tau,                       // temperature index (board_pkg::CEXP/FEXP)
  input  logic [1:0]  minp,                      // min-p index (board_pkg::NCUT)
  input  logic [7:0]  seed,
  input  logic        slow,
  output logic        tw_we,                     // text RAM port A
  output logic [11:0] tw_addr,
  output logic [7:0]  tw_data,
  input  logic [7:0]  tw_q,                      //   read data (1 clock)
  output logic [4:0]  top,                       // scroll offset of the text area
  output logic        running,
  output logic [1:0]  check,                     // 0 none, 1 OK, 2 FAIL, 3 running
  output logic [15:0] ntok,
  output logic [31:0] toks_bcd,                  // tokens/s, 8 BCD digits
  output logic [31:0] cpt_bcd                    // cycles/token, 8 BCD digits
);
  localparam int KW = $clog2(NV + 1);

  function automatic logic [31:0] xorshift32(input logic [31:0] x0);
    logic [31:0] x;
    x = x0 ^ (x0 << 13);
    x = x ^ (x >> 17);
    return x ^ (x << 5);
  endfunction

  function automatic logic [4:0] prow(input logic [4:0] r, input logic [4:0] tp);   // physical row of text row r
    logic [5:0] s;
    s = 6'(r) + 6'(tp);
    return 5'(1 + ((s >= 6'(ROWS)) ? s - 6'(ROWS) : s));
  endfunction

  function automatic logic [11:0] taddr(input logic [4:0] pr, input logic [6:0] col);
    return 12'(pr) * 12'(COLS) + 12'(col);
  endfunction

  // ======================================================================
  // engine
  // ======================================================================
  logic               c_start, c_busy, c_done;
  logic [TOK_W-1:0]   c_tok;
  logic [POS_W-1:0]   c_pos;
  logic [LA_W-1:0]    l_addr;
  logic signed [15:0] l_data;
  logic [7:0]         dbg_op;

  zoo_core #(.MEM_DIR(MEM_DIR), .SPECIALIZE(1'b1)) u_core (     // one model per bitstream: only its units
    .clk, .rst_n, .start(c_start), .tok_in(c_tok), .pos(c_pos), .busy(c_busy), .done(c_done),
    .l_addr, .l_data, .dbg_op, .ld_we(1'b0), .ld_sel(3'd0), .ld_addr(20'd0), .ld_data(256'd0));

  logic [15:0] exp_lut [256];                      // 2^15 * 2^(-k/256)
  initial $readmemh({MEM_DIR, "/exp.mem"}, exp_lut);

  // ======================================================================
  // generator + sampler
  // ======================================================================
  typedef enum logic [3:0] {G_IDLE, G_CLR, G_STEP, G_WAIT, G_MAX, G_SUM, G_SEL, G_PICK, G_THR, G_FIN} gst_t;
  gst_t gs;

  logic              run_p, reseed_p, fresh_run;
  logic              q_greedy, q_slow;
  logic [2:0]        q_tau;
  logic [1:0]        q_minp;
  logic [7:0]        q_seed;
  logic [15:0]       tok, pos, nseq, prev, nxt;
  logic [31:0]       rng, hash, thr, esum, cum;
  logic [39:0]       cyc;
  logic [31:0]       wait_cnt;
  logic              found;

  // scan pipeline: issue k -> s1 logit -> s2 (m - l) CEXP -> s3 weight
  logic [15:0]        k;
  logic               k_on;
  logic               s1_v, s2_v, s3_v;
  logic [15:0]        s1_k, s2_k, s3_k;
  logic signed [15:0] s1_l, mx;
  logic [15:0]        amx;
  logic [31:0]        s2_u;
  logic [15:0]        s3_e;

  logic [31:0] u_n, u_kk;
  logic [15:0] e_w;
  logic [5:0]  fx;
  always_comb begin
    fx   = FEXP[q_tau];
    u_n  = s2_u >> fx;
    u_kk = (s2_u >> (fx - 6'd8)) & 32'hFF;
    e_w  = (u_n >= 32'(NCUT[q_minp])) ? 16'd0 : (exp_lut[u_kk[7:0]] >> u_n[3:0]);
  end

  logic [31:0] rng_next;
  logic [56:0] thr_p;
  assign rng_next = xorshift32(rng);
  assign thr_p    = 57'(rng_next) * 57'(esum);

  logic [31:0] cum_n;
  logic [16:0] dm;                                 // m - l_k >= 0
  assign dm = 17'($signed({mx[15], mx}) - $signed({s1_l[15], s1_l}));
  assign cum_n  = cum + 32'(s3_e);
  assign l_addr = LA_W'(k);
  assign c_tok  = TOK_W'(tok);
  assign c_pos  = POS_W'(pos);

  // mailbox to the screen sequencer
  logic        m_tok_v, m_strip, m_fnl, m_clr, m_stat;
  logic [15:0] m_tok;
  logic        scr_idle;

  logic last_pos, is_stop;
  assign last_pos = (pos == 16'(T - 1));
  assign is_stop  = (nxt == 16'(STOP_TOK));

  always_ff @(posedge clk) begin
    c_start <= 1'b0;
    if (!rst_n) begin
      gs       <= G_IDLE;
      run_p    <= 1'b1;                          // run once after configuration / reset
      reseed_p <= 1'b1;
      running  <= 1'b0;
      check    <= 2'd0;
      ntok     <= '0;
      cyc      <= '0;
      rng      <= 32'h1;
      m_tok_v  <= 1'b0;
      m_clr    <= 1'b0;
      m_stat   <= 1'b0;
      k_on     <= 1'b0;
      s1_v     <= 1'b0; s2_v <= 1'b0; s3_v <= 1'b0;
    end else begin
      if (run)    run_p    <= 1'b1;
      if (reseed) begin reseed_p <= 1'b1; run_p <= 1'b1; end
      if (clear && gs == G_IDLE) m_clr <= 1'b1;
      // the screen sequencer takes requests when it is idle
      if (scr_idle) begin
        if (m_clr)        m_clr   <= 1'b0;
        else if (m_tok_v) m_tok_v <= 1'b0;
        else if (m_stat)  m_stat  <= 1'b0;
      end

      // ---------------- scan pipeline (G_MAX, G_SUM, G_SEL) ----------------
      s1_v <= k_on;
      s1_k <= k;
      s1_l <= l_data;
      s2_v <= s1_v && gs != G_MAX;
      s2_k <= s1_k;
      s2_u <= 32'(dm) * 32'(CEXP[q_tau]);
      s3_v <= s2_v;
      s3_k <= s2_k;
      s3_e <= e_w;
      if (k_on) begin
        if (k == 16'(NV - 1)) k_on <= 1'b0;
        k <= k + 1'b1;
      end

      case (gs)
        G_IDLE: if (run_p && !m_clr) begin
          run_p     <= run || reseed;            // a press in this very cycle counts as one during the run:
          reseed_p  <= reseed;                   // one more run (reseeded for BTNU), not dropped
          fresh_run <= reseed_p;
          if (reseed_p) begin
            rng    <= 32'h9E37_79B9 ^ (32'(seed) << 16) ^ 32'(seed);
            q_seed <= seed;
          end
          q_greedy <= greedy;
          q_tau    <= tau;
          q_minp   <= minp;
          q_slow   <= slow;
          tok      <= 16'(START_TOK);
          prev     <= 16'(START_TOK);
          pos      <= '0;
          nseq     <= '0;
          ntok     <= '0;
          cyc      <= '0;
          hash     <= '0;
          running  <= 1'b1;
          check    <= 2'd3;
          m_clr    <= 1'b1;
          m_stat   <= 1'b1;
          gs       <= G_CLR;
        end

        G_CLR: if (!m_clr && scr_idle) gs <= G_STEP;

        G_STEP: begin
          c_start <= 1'b1;
          cyc     <= cyc + 1'b1;
          gs      <= G_WAIT;
        end

        G_WAIT: begin
          cyc <= cyc + 1'b1;
          if (c_done) begin
            k    <= '0;
            k_on <= 1'b1;
            gs   <= G_MAX;
          end
        end

        G_MAX: begin
          cyc <= cyc + 1'b1;
          if (s1_v && (s1_k == '0 || s1_l > mx)) begin mx <= s1_l; amx <= s1_k; end
          if (s1_v && s1_k == 16'(NV - 1)) begin
            if (q_greedy) begin
              nxt <= (s1_l > mx) ? s1_k : amx;
              gs  <= G_PICK;
            end else begin
              k    <= '0;
              k_on <= 1'b1;
              esum <= '0;
              gs   <= G_SUM;
            end
          end
        end

        G_SUM: begin
          cyc <= cyc + 1'b1;
          if (s3_v) esum <= esum + 32'(s3_e);
          if (s3_v && s3_k == 16'(NV - 1)) begin
            k     <= '0;
            k_on  <= 1'b1;
            cum   <= '0;
            found <= 1'b0;
            gs    <= G_SEL;
          end
        end

        G_SEL: begin
          cyc <= cyc + 1'b1;
          if (s1_v && s1_k == '0) begin           // threshold from the completed sum (one new random number)
            rng <= rng_next;
            thr <= 32'(thr_p >> 32);
          end
          if (s3_v) begin
            cum <= cum_n;
            if (!found && cum_n > thr) begin found <= 1'b1; nxt <= s3_k; end
          end
          if (s3_v && s3_k == 16'(NV - 1)) gs <= G_PICK;
        end

        G_PICK: if (m_tok_v) cyc <= cyc + 1'b1;  // waiting for the screen to take the previous token
        else begin
          cyc     <= cyc + 1'b1;
          ntok    <= ntok + 1'b1;
          hash    <= (hash << 5) - hash + 32'(nxt);
          m_tok_v <= 1'b1;
          m_tok   <= nxt;
          m_strip <= STRIP && (prev == 16'(STRIP_TOK));
          m_fnl   <= MULTI && last_pos && !is_stop;
          m_stat  <= 1'b1;
          prev    <= nxt;
          wait_cnt <= '0;
          if (MULTI && (is_stop || last_pos)) begin
            nseq <= nseq + 1'b1;
            tok  <= 16'(START_TOK);
            pos  <= '0;
            gs   <= (nseq == 16'(MAX_SEQ - 1)) ? G_FIN : G_THR;
          end else if (!MULTI && (is_stop || last_pos)) begin
            gs <= G_FIN;
          end else begin
            tok <= nxt;
            pos <= pos + 1'b1;
            gs  <= G_THR;
          end
        end

        G_THR: begin                             // typewriter pause (not counted)
          wait_cnt <= wait_cnt + 1'b1;
          if (!q_slow || wait_cnt >= 32'(SLOW_CYC)) gs <= G_STEP;
        end

        default: if (!m_tok_v && scr_idle) begin // G_FIN: last token on the screen
          running <= 1'b0;
          if (q_greedy)
            check <= (hash == EXPECT_GREEDY) ? 2'd1 : 2'd2;
          else if (fresh_run && q_tau == 3'd0 && q_minp == 2'd0 && q_seed == 8'd0)
            check <= (hash == EXPECT_SAMPLE) ? 2'd1 : 2'd2;
          else
            check <= 2'd0;
          m_stat <= 1'b1;
          gs     <= G_IDLE;
        end
      endcase
    end
  end

`ifndef SYNTHESIS
  // simulation trace for sw/zoo/board_sim.py
  always @(posedge clk) if (rst_n && gs == G_PICK && !m_tok_v) $display("GENTOK %0d %0d %0d", nxt, pos, cyc);
  always @(posedge clk) if (rst_n && gs == G_FIN && !m_tok_v && scr_idle)
    $display("GENRUN ntok %0d cycles %0d hash %08x rng %08x", ntok, cyc, hash, rng);
`endif

  // ======================================================================
  // screen sequencer: clear, detokenise + terminal, status row
  // ======================================================================
  typedef enum logic [4:0] {W_IDLE, W_CLR, W_META, W_NEXT, W_CH, W_NL, W_SCR, W_MOVE, W_MVR, W_MVW, W_MVC,
                            W_MVB, W_PUT, W_ST0, W_ST1, W_ST2, W_ST3, W_ST4, W_ST5, W_ST6} wst_t;
  wst_t ws, ret;
  assign scr_idle = (ws == W_IDLE);

  logic [19:0] tokmeta [NV];
  logic [7:0]  tokstr  [TOKSTR_DEPTH];
  logic [7:0]  mode_rom [64 * 18];
  initial begin
    $readmemh({MEM_DIR, "/tokmeta.mem"}, tokmeta);
    $readmemh({MEM_DIR, "/tokstr.mem"}, tokstr);
    $readmemh({MEM_DIR, "/mode.mem"}, mode_rom);
  end
  logic [19:0]    meta_q;                  // ROM outputs one clock after the address (m_tok / off + ci)
  logic [7:0]     str_q;
  logic [15:0]    off;
  logic [3:0]     len, ci;
  always_ff @(posedge clk) begin
    meta_q <= tokmeta[KW'(m_tok)];
    str_q  <= tokstr[off + 16'(ci)];
  end

  logic [4:0]  row, old_pr, top_r;
  logic [6:0]  col, wstart, mv_k, ws_l;
  logic        sp, strip_l, fnl_l, first;
  logic [7:0]  cur_b;
  logic [11:0] ca;
  assign top = top_r;
  logic [4:0] top_n;
  assign top_n = (top_r == 5'(ROWS - 1)) ? 5'd0 : top_r + 1'b1;

  // status numbers
  logic [15:0] st_ntok;
  logic [39:0] st_cyc;
  logic [31:0] ntok_bcd;
  logic [5:0]  st_c;
  logic        nz;
  logic        div_start, div_done, bcd_start, bcd_done;
  logic [63:0] div_num, div_den, div_quo;
  logic [31:0] bcd_in, bcd_out;
  zdiv #(.W(64), .QB(40)) u_div (.clk, .rst_n, .start(div_start), .num(div_num), .den(div_den), .done(div_done),
                                .quo(div_quo));
  bin2bcd u_bcd (.clk, .rst_n, .start(bcd_start), .bin(bcd_in), .done(bcd_done), .bcd(bcd_out));

  function automatic logic [31:0] clamp7(input logic [63:0] v);  // 7 display digits
    return (v > 64'd9_999_999) ? 32'd9_999_999 : v[31:0];
  endfunction

  // status field character st_c (0..40): right-aligned numbers, mode text, check word
  localparam logic [31:0] TXT_OK = "OK  ", TXT_FAIL = "FAIL", TXT_RUN = "... ", TXT_NONE = "--  ";
  logic [7:0]  st_ch;
  logic [6:0]  st_col;
  logic [3:0]  st_dig;
  logic        st_last, st_first, st_isdig;
  logic [10:0] mode_a;
  logic [31:0] chk_txt;
  always_comb begin
    st_dig = '0; st_isdig = 1'b0; st_last = 1'b0; st_first = 1'b0; st_ch = " "; st_col = '0;
    mode_a = 11'({q_greedy, q_tau, q_minp}) * 11'd18 + 11'(st_c - 6'd19);
    case (check)
      2'd1:    chk_txt = TXT_OK;
      2'd2:    chk_txt = TXT_FAIL;
      2'd3:    chk_txt = TXT_RUN;
      default: chk_txt = TXT_NONE;
    endcase
    if (st_c < 6'd5) begin
      st_col = 7'(ST_NTOK) + 7'(st_c); st_dig = ntok_bcd[4 * (4 - st_c) +: 4];
      st_isdig = 1'b1; st_first = (st_c == 0); st_last = (st_c == 4);
    end else if (st_c < 6'd12) begin
      st_col = 7'(ST_TOKS) + 7'(st_c - 6'd5); st_dig = toks_bcd[4 * (11 - st_c) +: 4];
      st_isdig = 1'b1; st_first = (st_c == 5); st_last = (st_c == 11);
    end else if (st_c < 6'd19) begin
      st_col = 7'(ST_CPT) + 7'(st_c - 6'd12); st_dig = cpt_bcd[4 * (18 - st_c) +: 4];
      st_isdig = 1'b1; st_first = (st_c == 12); st_last = (st_c == 18);
    end else if (st_c < 6'd37) begin
      st_col = 7'(ST_MODE) + 7'(st_c - 6'd19); st_ch = mode_rom[mode_a];
    end else begin
      st_col = 7'(ST_CHECK) + 7'(st_c - 6'd37); st_ch = chk_txt[8 * (40 - st_c) +: 8];
    end
    if (st_isdig) st_ch = (st_dig != 0 || (!st_first && nz) || st_last) ? 8'h30 + 8'(st_dig) : " ";
  end

  always_ff @(posedge clk) begin
    tw_we     <= 1'b0;
    div_start <= 1'b0;
    bcd_start <= 1'b0;
    if (!rst_n) begin
      ws       <= W_IDLE;
      top_r    <= '0;
      row      <= '0;
      col      <= '0;
      sp       <= 1'b1;
      wstart   <= '0;
      toks_bcd <= '0;
      cpt_bcd  <= '0;
      ntok_bcd <= '0;
    end else case (ws)
      W_IDLE: begin
        if (m_clr) begin
          ca <= 12'(COLS);
          ws <= W_CLR;
        end else if (m_tok_v) begin
          strip_l <= m_strip;
          fnl_l   <= m_fnl;
          ws      <= W_META;
        end else if (m_stat) begin
          st_ntok <= ntok;
          st_cyc  <= cyc;
          ws      <= W_ST0;
        end
      end

      W_CLR: begin                               // text rows 1..ROWS
        tw_we <= 1'b1; tw_addr <= ca; tw_data <= " ";
        ca    <= ca + 1'b1;
        if (ca == 12'(COLS * (ROWS + 1) - 1)) begin
          top_r <= '0; row <= '0; col <= '0; sp <= 1'b1; wstart <= '0;
          ws    <= W_IDLE;
        end
      end

      W_META: begin
        off   <= meta_q[15:0];
        len   <= meta_q[19:16];
        ci    <= '0;
        first <= 1'b1;
        ws    <= W_NEXT;
      end

      W_NEXT: begin
        if (ci == len) begin
          if (fnl_l) begin fnl_l <= 1'b0; ret <= W_NEXT; ws <= W_NL; end
          else ws <= W_IDLE;
        end else begin
          ci <= ci + 1'b1;                       // tokstr[off + ci] arrives in W_CH
          ws <= W_CH;
        end
      end

      W_CH: begin                                // byte str_q of the piece
        first <= 1'b0;
        if (first && strip_l && str_q == " ")
          ws <= W_NEXT;                          // leading space after BOS (as llama2.c decode)
        else if (str_q == 8'h0A) begin
          ret <= W_NEXT; ws <= W_NL;
        end else if (str_q == " ") begin
          if (col == 7'(COLS)) begin ret <= W_NEXT; ws <= W_NL; end
          else begin
            tw_we <= 1'b1; tw_addr <= taddr(prow(row, top_r), col); tw_data <= " ";
            col <= col + 1'b1; sp <= 1'b1;
            ws  <= W_NEXT;
          end
        end else begin
          cur_b <= str_q;
          if (col == 7'(COLS)) begin
            ws_l   <= sp ? 7'(COLS) : wstart;
            old_pr <= prow(row, top_r);
            ret    <= W_MOVE;
            ws     <= W_NL;
          end else ws <= W_PUT;
        end
      end

      W_NL: begin                                // newline (scrolls at the bottom row)
        col <= '0;
        sp  <= 1'b1;
        if (row == 5'(ROWS - 1)) begin
          top_r <= top_n;
          ca    <= taddr(prow(5'(ROWS - 1), top_n), 7'd0);
          mv_k  <= '0;
          ws    <= W_SCR;
        end else begin
          row <= row + 1'b1;
          ws  <= ret;
        end
      end

      W_SCR: begin                               // clear the new bottom row
        tw_we <= 1'b1; tw_addr <= ca; tw_data <= " ";
        ca    <= ca + 1'b1;
        mv_k  <= mv_k + 1'b1;
        if (mv_k == 7'(COLS - 1)) ws <= ret;
      end

      W_MOVE: begin                              // unfinished word [ws_l, COLS) of the old row moves down
        mv_k <= ws_l;
        ws   <= (ws_l != '0 && ws_l < 7'(COLS)) ? W_MVR : W_PUT;
      end
      W_MVR: begin tw_addr <= taddr(old_pr, mv_k); ws <= W_MVW; end    // read the old character
      W_MVW: ws <= W_MVC;                                              // read data arrives
      W_MVC: begin
        tw_we <= 1'b1; tw_addr <= taddr(prow(row, top_r), col); tw_data <= tw_q;
        col   <= col + 1'b1;
        ws    <= W_MVB;
      end
      W_MVB: begin                                                     // blank the old position
        tw_we <= 1'b1; tw_addr <= taddr(old_pr, mv_k); tw_data <= " ";
        mv_k  <= mv_k + 1'b1;
        if (mv_k == 7'(COLS - 1)) begin sp <= 1'b0; wstart <= '0; ws <= W_PUT; end
        else ws <= W_MVR;
      end

      W_PUT: begin
        tw_we <= 1'b1; tw_addr <= taddr(prow(row, top_r), col); tw_data <= cur_b;
        if (sp) wstart <= col;
        col <= col + 1'b1;
        sp  <= 1'b0;
        ws  <= W_NEXT;
      end

      // ---------------- status row ----------------
      W_ST0: begin                               // tokens/s = ntok * CLK_HZ / cycles
        div_num   <= 64'(st_ntok) * 64'(CLK_HZ);
        div_den   <= (st_cyc == '0) ? 64'd1 : 64'(st_cyc);
        div_start <= 1'b1;
        ws        <= W_ST1;
      end
      W_ST1: if (div_done) begin
        bcd_in    <= (st_cyc == '0) ? 32'd0 : clamp7(div_quo);
        bcd_start <= 1'b1;
        ws        <= W_ST2;
      end
      W_ST2: if (bcd_done) begin                 // cycles/token = cycles / ntok
        toks_bcd  <= bcd_out;
        div_num   <= 64'(st_cyc);
        div_den   <= (st_ntok == '0) ? 64'd1 : 64'(st_ntok);
        div_start <= 1'b1;
        ws        <= W_ST3;
      end
      W_ST3: if (div_done) begin
        bcd_in    <= (st_ntok == '0) ? 32'd0 : clamp7(div_quo);
        bcd_start <= 1'b1;
        ws        <= W_ST4;
      end
      W_ST4: if (bcd_done) begin
        cpt_bcd   <= bcd_out;
        bcd_in    <= 32'(st_ntok);
        bcd_start <= 1'b1;
        ws        <= W_ST5;
      end
      W_ST5: if (bcd_done) begin
        ntok_bcd <= bcd_out;
        st_c     <= '0;
        nz       <= 1'b0;
        ws       <= W_ST6;
      end
      default: begin                             // W_ST6: write the 41 field characters
        tw_we   <= 1'b1;
        tw_addr <= taddr(5'(ROWS + 1), st_col);
        tw_data <= st_ch;
        nz      <= (!st_first && nz) || st_dig != 0;
        st_c    <= st_c + 1'b1;
        if (st_c == 6'd40) ws <= W_IDLE;
      end
    endcase
  end
endmodule
