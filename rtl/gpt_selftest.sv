// ---------------------------------------------------------------------------
// gpt_selftest.sv -- on-chip self-test of the accelerator on the sort task.
//
// Enumerates every prompt (N_VOCAB^SORT_LEN = 729), runs closed-loop greedy
// generation on gpt_top (prompt tokens, then its own outputs fed back) and
// checks that the SORT_LEN generated tokens are the sorted prompt:
//     non-decreasing  AND  same histogram as the prompt.
// No golden data is stored on chip, so the check is independent of the model.
//
// inject_fault replaces the first generated token of every sequence by a
// different valid token before checking -> every sequence must then fail
// (demonstrates that the checker is live).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module gpt_selftest
  import gpt_pkg::*;
#(
  parameter string MEM_DIR = MEM_PATH
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic        inject_fault,
  output logic        busy,
  output logic        done,       // stays high until the next start
  output logic        pass,       // valid when done
  output logic [15:0] n_tested,
  output logic [15:0] n_fail,
  output logic [47:0] cycles      // clock cycles of the whole run
);
  localparam int SL    = SORT_LEN;
  localparam int NSTEP = 2 * SL - 1;
  localparam int CW    = $clog2(SL + 1);

  // ---------------- accelerator ----------------
  logic                     g_start, g_busy, g_done;
  logic [TOK_W-1:0]         g_tok_in, g_tok_out;
  logic [POS_W-1:0]         g_pos;
  logic [N_VOCAB-1:0][15:0] g_logits;
  logic signed [15:0]       g_dbg;

  gpt_top #(.MEM_DIR(MEM_DIR)) u_gpt (
    .clk(clk), .rst_n(rst_n), .start(g_start), .tok_in(g_tok_in), .pos(g_pos),
    .busy(g_busy), .done(g_done), .tok_out(g_tok_out), .logits(g_logits),
    .dbg_addr('0), .dbg_res(g_dbg));

  // ---------------- prompt enumerator (base-N_VOCAB odometer) ----------------
  logic [TOK_W-1:0] prompt      [SL];
  logic [TOK_W-1:0] prompt_next [SL];
  logic             prompt_last;
  logic [CW-1:0]    hist_p [N_VOCAB];   // histogram of the prompt
  logic [CW-1:0]    hist_g [N_VOCAB];   // histogram of the generated tokens

  always_comb begin
    logic carry;
    carry       = 1'b1;
    prompt_last = 1'b1;
    for (int i = 0; i < SL; i++) begin
      prompt_next[i] = prompt[i];
      if (prompt[i] != TOK_W'(N_VOCAB - 1)) prompt_last = 1'b0;
      if (carry) begin
        if (prompt[i] == TOK_W'(N_VOCAB - 1)) prompt_next[i] = '0;
        else begin prompt_next[i] = prompt[i] + 1'b1; carry = 1'b0; end
      end
    end
    for (int v = 0; v < N_VOCAB; v++) begin
      hist_p[v] = '0;
      for (int i = 0; i < SL; i++) hist_p[v] += CW'(prompt[i] == TOK_W'(v));
    end
  end

  // ---------------- sequencer ----------------
  typedef enum logic [2:0] {IDLE, NEWSEQ, RUN, WAIT, CHECK} state_t;
  state_t st;

  logic [POS_W-1:0] t;
  logic [TOK_W-1:0] last_out, prev_gen, gen_tok;
  logic             mono;

  // token seen by the checker (optionally corrupted, always a valid token)
  assign gen_tok = (inject_fault && t == POS_W'(SL - 1))
                   ? ((g_tok_out == '0) ? TOK_W'(1) : TOK_W'(0))
                   : g_tok_out;

  logic seq_ok;
  always_comb begin
    seq_ok = mono;
    for (int v = 0; v < N_VOCAB; v++) if (hist_g[v] != hist_p[v]) seq_ok = 1'b0;
  end

  assign busy = (st != IDLE);
  assign pass = done && (n_fail == '0);

  always_ff @(posedge clk) begin
    g_start <= 1'b0;
    if (!rst_n) begin
      st   <= IDLE;
      done <= 1'b0;
    end else begin
      if (busy) cycles <= cycles + 1'b1;
      case (st)
        IDLE: if (start) begin
          for (int i = 0; i < SL; i++) prompt[i] <= '0;
          n_tested <= '0; n_fail <= '0; cycles <= '0; done <= 1'b0;
          st <= NEWSEQ;
        end
        NEWSEQ: begin
          t <= '0; mono <= 1'b1;
          for (int v = 0; v < N_VOCAB; v++) hist_g[v] <= '0;
          st <= RUN;
        end
        RUN: begin                                // one decode step
          g_tok_in <= (t < POS_W'(SL)) ? prompt[t] : last_out;
          g_pos    <= t;
          g_start  <= 1'b1;
          st       <= WAIT;
        end
        WAIT: if (g_done) begin
          last_out <= g_tok_out;
          if (t >= POS_W'(SL - 1)) begin          // output is a generated token
            hist_g[gen_tok] <= hist_g[gen_tok] + 1'b1;
            if (t > POS_W'(SL - 1) && gen_tok < prev_gen) mono <= 1'b0;
            prev_gen <= gen_tok;
          end
          if (t == POS_W'(NSTEP - 1)) st <= CHECK;
          else begin t <= t + 1'b1; st <= RUN; end
        end
        CHECK: begin
          n_tested <= n_tested + 1'b1;
          if (!seq_ok) n_fail <= n_fail + 1'b1;
          if (prompt_last) begin
            done <= 1'b1; st <= IDLE;
          end else begin
            for (int i = 0; i < SL; i++) prompt[i] <= prompt_next[i];
            st <= NEWSEQ;
          end
        end
        default: st <= IDLE;
      endcase
    end
  end
endmodule
