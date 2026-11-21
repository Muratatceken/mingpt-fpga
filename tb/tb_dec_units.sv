`timescale 1ns/1ps
// Decision RTL unit tests against Python golden vectors (sw/decision/gen_rtl_vectors.py):
//   guard (combinational, exhaustive), tokenizer, rule, FSM (step by step), guard wrapper watchdog edges.
module tb_dec_units;
  import dec_pkg::*;
  localparam int WDT = 8;
  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ---------------- combinational guard
  logic [1:0] g_prop, g_mode, g_gc, g_a, g_m;
  logic g_pv, g_vv, g_iv, g_tv;
  logic signed [15:0] g_v, g_i, g_t;
  logic [2:0] g_inv, g_r, g_invo;
  dec_guard_comb u_gc (.prop(g_prop), .prop_valid(g_pv), .mode_in(g_mode), .v(g_v), .i(g_i), .t(g_t), .vv(g_vv),
                       .iv(g_iv), .tv(g_tv), .gc(g_gc), .inv_in(g_inv), .applied(g_a), .mode_new(g_m),
                       .reason(g_r), .inv_out(g_invo));
  // ---------------- tokenizer / rule
  logic signed [15:0] k_v, k_i, k_t;
  logic k_vv, k_iv, k_tv;
  logic [5:0] k_tokv, k_toki, k_tokt;
  dec_tokenizer u_tok (.v(k_v), .i(k_i), .t(k_t), .vv(k_vv), .iv(k_iv), .tv(k_tv), .tok_v(k_tokv),
                       .tok_i(k_toki), .tok_t(k_tokt));
  logic [5:0] r_tv, r_ti, r_tt;
  logic [1:0] r_a;
  dec_rule u_rule (.tok_v(r_tv), .tok_i(r_ti), .tok_t(r_tt), .act(r_a));
  // ---------------- FSM
  logic f_clear = 1'b0, f_step = 1'b0;
  logic [5:0] f_tv, f_ti, f_tt;
  logic [1:0] f_prev, f_a;
  dec_fsm u_fsm (.clk(clk), .rst_n(rst_n), .clear(f_clear), .step(f_step), .tok_v(f_tv), .tok_i(f_ti),
                 .tok_t(f_tt), .prev_act(f_prev), .act(f_a));
  // ---------------- registered guard (watchdog)
  logic s_start = 1'b0, s_pv = 1'b0, s_done, s_busy;
  logic [1:0] s_prop, s_gc, s_a, s_m;
  logic signed [15:0] s_v, s_i, s_t;
  logic s_vv, s_iv, s_tv;
  logic [2:0] s_r;
  dec_guard #(.WDT(WDT)) u_gs (.clk(clk), .rst_n(rst_n), .clear(1'b0), .start(s_start), .prop_valid(s_pv), .prop(s_prop),
                               .v(s_v), .i(s_i), .t(s_t), .vv(s_vv), .iv(s_iv), .tv(s_tv), .gc(s_gc),
                               .done(s_done), .applied(s_a), .mode(s_m), .reason(s_r), .busy(s_busy));

  // initialised so that sv2v keeps them as variables (Icarus flow)
  int fd = 0, n = 0, err = 0, tot = 0;
  int a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0, a9 = 0, a10 = 0, a11 = 0, a12 = 0,
      a13 = 0, a14 = 0;
  int ep = 0, stp = 0, last_ep = 0;
  initial begin
    tot = 0;
    // guard
    fd = $fopen("build/decision/vec_guard.txt", "r"); void'($fscanf(fd, "%d", n)); err = 0;
    for (int k = 0; k < n; k++) begin
      void'($fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                    a11, a12, a13, a14));
      g_mode = 2'(a0); g_pv = a1[0]; g_prop = 2'(a2); g_v = 16'(a3); g_i = 16'(a4); g_t = 16'(a5);
      g_vv = a6[0]; g_iv = a7[0]; g_tv = a8[0]; g_gc = 2'(a9); g_inv = 3'(a10);
      #1;
      if (g_a != 2'(a11) || g_m != 2'(a12) || g_r != 3'(a13) || g_invo != 3'(a14)) begin
        if (err < 5) $display("GUARD MISMATCH vec %0d: got a=%0d m=%0d r=%0d inv=%0d exp %0d %0d %0d %0d", k, g_a, g_m,
                              g_r, g_invo, a11, a12, a13, a14);
        err++;
      end
    end
    $fclose(fd); $display("RESULT guard vectors %0d errors %0d", n, err); tot += err;
    // tokenizer
    fd = $fopen("build/decision/vec_tok.txt", "r"); void'($fscanf(fd, "%d", n)); err = 0;
    for (int k = 0; k < n; k++) begin
      void'($fscanf(fd, "%d %d %d %d %d %d %d %d %d", a0, a1, a2, a3, a4, a5, a6, a7, a8));
      k_v = 16'(a0); k_i = 16'(a1); k_t = 16'(a2); k_vv = a3[0]; k_iv = a4[0]; k_tv = a5[0];
      #1;
      if (k_tokv != 6'(a6) || k_toki != 6'(a7) || k_tokt != 6'(a8)) begin
        if (err < 5) $display("TOK MISMATCH vec %0d: got %0d %0d %0d exp %0d %0d %0d", k, k_tokv, k_toki, k_tokt,
                              a6, a7, a8);
        err++;
      end
    end
    $fclose(fd); $display("RESULT tokenizer vectors %0d errors %0d", n, err); tot += err;
    // rule
    fd = $fopen("build/decision/vec_rule.txt", "r"); void'($fscanf(fd, "%d", n)); err = 0;
    for (int k = 0; k < n; k++) begin
      void'($fscanf(fd, "%d %d %d %d", a0, a1, a2, a3));
      r_tv = 6'(a0); r_ti = 6'(a1); r_tt = 6'(a2);
      #1;
      if (r_a != 2'(a3)) begin
        if (err < 5) $display("RULE MISMATCH %0d %0d %0d: got %0d exp %0d", a0, a1, a2, r_a, a3);
        err++;
      end
    end
    $fclose(fd); $display("RESULT rule vectors %0d errors %0d", n, err); tot += err;
    // FSM (episodes are consecutive rows; clear at every new episode)
    rst_n = 1'b0; repeat (2) @(posedge clk); rst_n = 1'b1;
    fd = $fopen("build/decision/vec_fsm.txt", "r"); void'($fscanf(fd, "%d", n)); err = 0; last_ep = -1;
    for (int k = 0; k < n; k++) begin
      void'($fscanf(fd, "%d %d %d %d %d %d %d", ep, stp, a0, a1, a2, a3, a4));
      if (ep != last_ep) begin
        @(negedge clk); f_clear = 1'b1; @(negedge clk); f_clear = 1'b0; last_ep = ep;
      end
      @(negedge clk);
      f_tv = 6'(a0); f_ti = 6'(a1); f_tt = 6'(a2); f_prev = 2'(a3);
      #1;
      if (f_a != 2'(a4)) begin
        if (err < 5) $display("FSM MISMATCH ep %0d step %0d: got %0d exp %0d", ep, stp, f_a, a4);
        err++;
      end
      f_step = 1'b1; @(negedge clk); f_step = 1'b0;
    end
    $fclose(fd); $display("RESULT fsm steps %0d errors %0d", n, err); tot += err;
    // registered guard: watchdog edges
    rst_n = 1'b0; repeat (2) @(posedge clk); rst_n = 1'b1;
    fd = $fopen("build/decision/vec_gseq.txt", "r"); void'($fscanf(fd, "%d", n)); err = 0;
    for (int k = 0; k < n; k++) begin
      void'($fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d", a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11));
      @(negedge clk);
      s_prop = 2'(a1); s_v = 16'(a2); s_i = 16'(a3); s_t = 16'(a4); s_vv = a5[0]; s_iv = a6[0]; s_tv = a7[0];
      s_gc = 2'(a8); s_start = 1'b1;
      @(negedge clk); s_start = 1'b0;
      // busy cycle index b = 0, 1, ...: drive the proposal at index a0 (one-cycle pulse)
      for (int b = 0; b <= WDT + 1 && s_busy; b++) begin
        s_pv = (b == a0);
        @(negedge clk);
        s_pv = 1'b0;
      end
      while (!s_done && s_busy) @(negedge clk);
      if (s_a != 2'(a9) || s_m != 2'(a10) || s_r != 3'(a11)) begin
        if (err < 5) $display("GSEQ MISMATCH %0d (arrive %0d): got a=%0d m=%0d r=%0d exp %0d %0d %0d", k, a0, s_a,
                              s_m, s_r, a9, a10, a11);
        err++;
      end
    end
    $fclose(fd); $display("RESULT guard_seq steps %0d errors %0d", n, err); tot += err;
    if (tot == 0) $display("PASS: decision units bit-exact with the Python golden models");
    else $display("FAIL: %0d mismatches", tot);
    $finish;
  end
endmodule
