// ---------------------------------------------------------------------------
// tb_zoo.sv -- bit-exact testbench for the microcoded zoo engine.
//
// Reads mem/zoo/<model>/tv.txt (written by sw/zoo/export_hw.py from the
// integer interpreter quant.py:IntRunner): teacher-forced test names, and for
// every decode step the input token and all V int16 logits. Every logit of
// every step must match exactly. Prints a cycle profile per opcode.
// +GEN: closed-loop greedy generation (mem/zoo/<model>/gen.txt): the engine's
// own argmax is fed back as the next token and every token must equal the one
// produced by IntRunner; the tokens are written to build/zoo/<model>_gen_rtl.txt.
//
// Plusargs: +NSEQ=<n> (default: all sequences)  +TV=<file> (default MEM_PATH/tv.txt)  +GEN
// Parameters: SPECIALIZE   only the units this model uses
//             LOADTEST     memories start empty and are written through the
//                          ld_* port from the .mem files before the test
// `define XCHECK: every logit is also checked for X/Z (4-state simulators)
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_zoo;
  import zoo_pkg::*;

  parameter bit SPECIALIZE = 1'b0;   // -GSPECIALIZE=1: only the units this model uses
  parameter bit LOADTEST   = 1'b0;   // -GLOADTEST=1: load the model through the ld_* port

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;   // 100 MHz

  logic                start = 1'b0;
  logic [TOK_W-1:0]    tok_in = '0;
  logic [POS_W-1:0]    pos = '0;
  logic                busy, done;
  logic [LA_W-1:0]     l_addr = '0;
  logic signed [15:0]  l_data;
  logic [7:0]          dbg_op;
  logic                ld_we = 1'b0;
  logic [2:0]          ld_sel = '0;
  logic [19:0]         ld_addr = '0;
  logic [255:0]        ld_data = '0;

  zoo_core #(.SPECIALIZE(SPECIALIZE), .LOADABLE(LOADTEST), .PRELOAD(!LOADTEST)) dut (.*);

  // ---- model loading through the ld_* port (what the PS does on the board) ----
  logic [255:0]     m_prog [PROG_DEPTH];
  logic [8*P-1:0]   m_w    [W_DEPTH];
  logic [63:0]      m_prm  [PRM_DEPTH];
  logic [TAB_W-1:0] m_tab  [TAB_DEPTH];
  logic [7:0]       m_lut  [LUT_DEPTH];

  task automatic ld_word(input int sel, input int addr, input logic [255:0] data);
    @(posedge clk);
    ld_we   <= 1'b1;
    ld_sel  <= 3'(sel);
    ld_addr <= 20'(addr);
    ld_data <= data;
    @(posedge clk);
    ld_we   <= 1'b0;
  endtask

  task automatic load_model();
    $readmemh({MEM_PATH, "/prog.mem"}, m_prog);
    $readmemh({MEM_PATH, "/weights.mem"}, m_w);
    $readmemh({MEM_PATH, "/params.mem"}, m_prm);
    $readmemh({MEM_PATH, "/tables.mem"}, m_tab);
    $readmemh({MEM_PATH, "/luts.mem"}, m_lut);
    for (int a = 0; a < PROG_DEPTH; a++) ld_word(0, a, m_prog[a]);
    for (int a = 0; a < W_DEPTH; a++)    ld_word(1, a, 256'(m_w[a]));
    for (int a = 0; a < PRM_DEPTH; a++)  ld_word(2, a, 256'(m_prm[a]));
    for (int a = 0; a < TAB_DEPTH; a++)  ld_word(3, a, 256'(m_tab[a]));
    for (int a = 0; a < LUT_DEPTH; a++)  ld_word(4, a, 256'(m_lut[a]));
    $display("model loaded through ld_*: %0d words", PROG_DEPTH + W_DEPTH + PRM_DEPTH + TAB_DEPTH + LUT_DEPTH);
  endtask

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  // cycle profile per opcode (index 16 = fetch/decode)
  longint prof [17];
  initial for (int k = 0; k < 17; k++) prof[k] = 0;
  always @(posedge clk) if (busy) prof[dbg_op] <= prof[dbg_op] + 1;

  function automatic string op_name(input int k);
    case (k)
      0: return "END";      1: return "EMB16";    2: return "EMB8";     3: return "EMBWIN";
      4: return "TABLOGIT"; 5: return "HRESET";   6: return "NORM";     7: return "GEMV";
      8: return "ROPE";     9: return "KVSTORE"; 10: return "ATTN";    11: return "MUL";
     12: return "GRUUPD";  13: return "COPY";    14: return "REQ";     15: return "BOW";
      default: return "FETCH";
    endcase
  endfunction

  int fd = 0, rc = 0, nseq = 0, nv = 0, len = 0, tok = 0, maxseq = 0, xerr = 0;   // initialised: sv2v
  string tvfile;                                                                  // keeps them variables
  int gold [V];
  initial for (int k = 0; k < V; k++) gold[k] = 0;
  int errors = 0, steps = 0, bad_steps = 0;
  longint t0, step_cyc, tot_cyc = 0, max_cyc = 0, min_cyc = 64'h7fffffff;

  // one decode step: token tk at position tt, wait for done
  task automatic run_step(input int tk, input int tt);
    @(posedge clk);
    start  <= 1'b1;
    tok_in <= TOK_W'(tk);
    pos    <= POS_W'(tt);
    t0 = cyc;
    @(posedge clk);
    start <= 1'b0;
    do @(posedge clk); while (!done);
    step_cyc = cyc - t0;
  endtask

  // closed-loop greedy generation, checked token by token
  task automatic generate_check();
    int gfd = 0, ofd = 0, n = 0, tk = 0, exp_tok = 0, best = 0, bi = 0, ok = 0;
    longint gcyc;
    gfd = $fopen({MEM_PATH, "/gen.txt"}, "r");
    if (gfd == 0) begin $display("GEN: no %s/gen.txt", MEM_PATH); return; end
    ofd = $fopen({"build/zoo/", NAME, "_gen_rtl.txt"}, "w");
    rc = $fscanf(gfd, "%d %d", n, tk);
    $fwrite(ofd, "%0d", tk);
    ok = 0;
    gcyc = 0;
    for (int tt = 0; tt < n; tt++) begin
      run_step(tk, tt);
      gcyc += step_cyc;
      best = -1 << 30;
      bi = 0;
      for (int k = 0; k < V; k++) begin
        l_addr = LA_W'(k);
        #1;
        if (int'(l_data) > best) begin best = int'(l_data); bi = k; end
      end
      rc = $fscanf(gfd, "%d", exp_tok);
      if (bi == exp_tok) ok++;
      else if (ok == tt) $display("GEN: first divergence at step %0d: rtl %0d, reference %0d", tt, bi, exp_tok);
      $fwrite(ofd, " %0d", bi);
      tk = bi;
    end
    $fwrite(ofd, "\n");
    $fclose(ofd);
    $fclose(gfd);
    $display("GEN %s steps %0d match %0d avg_cycles %0d", NAME, n, ok, gcyc / n);
    if (ok == n) $display("GEN PASS: %0d greedy tokens identical to the integer reference", n);
    else         $display("GEN FAIL: %0d / %0d tokens identical", ok, n);
  endtask

  initial begin
    if (!$value$plusargs("NSEQ=%d", maxseq)) maxseq = 1 << 30;
    if (!$value$plusargs("TV=%s", tvfile)) tvfile = {MEM_PATH, "/tv.txt"};
    fd = $fopen(tvfile, "r");
    if (fd == 0) $fatal(1, "cannot open %s", tvfile);
    rc = $fscanf(fd, "%d %d", nseq, nv);
    if (nv != V) $fatal(1, "tv.txt has V=%0d, package V=%0d", nv, V);
    if (nseq > maxseq) nseq = maxseq;

    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
    if (LOADTEST) load_model();

    for (int s = 0; s < nseq; s++) begin
      rc = $fscanf(fd, "%d", len);
      for (int tt = 0; tt < len; tt++) begin
        int step_err;
        rc = $fscanf(fd, "%d", tok);
        for (int k = 0; k < V; k++) rc = $fscanf(fd, "%d", gold[k]);
        run_step(tok, tt);
        tot_cyc += step_cyc;
        if (step_cyc > max_cyc) max_cyc = step_cyc;
        if (step_cyc < min_cyc) min_cyc = step_cyc;
        step_err = 0;
        for (int k = 0; k < V; k++) begin
          l_addr = LA_W'(k);
          #1;
`ifdef XCHECK
          if ($isunknown(l_data)) xerr++;
`endif
          if (int'(l_data) != gold[k]) begin
            if (errors < 10)
              $display("MISMATCH seq %0d t %0d logit %0d: rtl %0d gold %0d", s, tt, k, l_data, gold[k]);
            errors++;
            step_err = 1;
          end
        end
        bad_steps += step_err;
        steps++;
      end
    end
    $fclose(fd);
    if (steps == 0) steps = 1;

    $display("");
    $display("=== %s%s: %0d sequences, %0d decode steps, %0d x %0d logits checked ===", NAME, SPECIALIZE ? " (specialised)" : "", nseq, steps, steps, V);
    $display("cycles/token: avg %0d  min %0d  max %0d   (%.1f us/token @ 100 MHz)",
             tot_cyc / steps, min_cyc, max_cyc, real'(tot_cyc) / steps / 100.0);
    $display("cycle profile:");
    for (int k = 0; k < 17; k++)
      if (prof[k] != 0)
        $display("  %-9s %10d  %5.1f%%", op_name(k), prof[k], 100.0 * real'(prof[k]) / real'(tot_cyc));
    $display("RESULT %s steps %0d bad_steps %0d errors %0d avg_cycles %0d min_cycles %0d max_cycles %0d",
             NAME, steps, bad_steps, errors, tot_cyc / steps, min_cyc, max_cyc);
    for (int k = 0; k < 17; k++)
      if (prof[k] != 0) $display("PROF %s %s %0d", NAME, op_name(k), prof[k]);
`ifdef XCHECK
    $display("XCHECK %s unknown_logits %0d", NAME, xerr);
    if (xerr != 0) errors += xerr;
`endif
    if (errors == 0) $display("PASS: RTL is bit-exact with the integer reference");
    else             $display("FAIL: %0d logit mismatches in %0d steps", errors, bad_steps);
    if ($test$plusargs("GEN")) generate_check();
    $finish;
  end
endmodule
