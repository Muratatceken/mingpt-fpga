// ---------------------------------------------------------------------------
// tb_gpt_top.sv -- system-level, bit-exact testbench.
//
// Reads mem/tv.txt (written by sw/export.py from the integer golden
// model) and runs closed-loop greedy generation: the prompt tokens are fed
// first, then every generated token is fed back. At every decode step it
// checks tok_out, all logits and the full final residual vector.
//
// Plusargs:  +NSEQ=<n>   number of test sequences (default: all)
//            +VERBOSE    print every sequence
// Works with Vivado xsim and Verilator (--binary --timing).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_gpt_top;
  import gpt_pkg::*;

  parameter string MEM_DIR = MEM_PATH;

  logic clk, rst_n = 1'b0;
  initial clk = 1'b0;
  always #5 clk = ~clk;   // 100 MHz

  logic                     start = 1'b0;
  logic [TOK_W-1:0]         tok_in = '0;
  logic [POS_W-1:0]         pos = '0;
  logic                     busy, done;
  logic [TOK_W-1:0]         tok_out;
  logic [N_VOCAB-1:0][15:0] logits;
  logic [EMB_AW-1:0]        dbg_addr = '0;
  logic signed [15:0]       dbg_res;

  gpt_top #(.MEM_DIR(MEM_DIR)) dut (.*);

  // golden step record
  int g_t, g_tok_in, g_tok_out;
  int g_logit [N_VOCAB];
  int g_res   [N_EMBD];

  int fd, n_total, plen, nsteps, nseq, rc;
  int errors = 0, steps_checked = 0, seq_ok = 0, task_ok = 0;
  longint cycles = 0, t0;
  int prompt [64];
  int gen    [64];
  int counts [N_VOCAB];

  longint cyc;
  initial cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  // cycle profile per controller state (hierarchical reference into the DUT)
  localparam int NST = 12;
  function automatic string st_name(input int k);
    case (k)
      1: return "EMB";   2: return "LN1";  3: return "QKV";  4: return "ATT";
      5: return "PROJ";  6: return "LN2";  7: return "FC";   8: return "MPROJ";
      9: return "LNF";  10: return "HEAD"; 11: return "DONE"; default: return "IDLE";
    endcase
  endfunction
  longint st_cycles [NST];
  initial foreach (st_cycles[k]) st_cycles[k] = 0;
  always @(posedge clk) if (rst_n && dut.busy) st_cycles[int'(dut.st)] <= st_cycles[int'(dut.st)] + 1;

  task automatic run_step(input int tok, input int p);
    @(posedge clk);
    tok_in <= TOK_W'(tok);
    pos    <= POS_W'(p);
    start  <= 1'b1;
    t0 = cyc;
    @(posedge clk);
    start  <= 1'b0;
    do @(posedge clk); while (!done);
    cycles += cyc - t0;
  endtask

  task automatic read_golden();
    rc = $fscanf(fd, "%d %d %d", g_t, g_tok_in, g_tok_out);
    for (int v = 0; v < N_VOCAB; v++) rc = $fscanf(fd, "%d", g_logit[v]);
    for (int c = 0; c < N_EMBD; c++)  rc = $fscanf(fd, "%d", g_res[c]);
  endtask

  task automatic check_step(input int s);
    int bad = 0;
    if (int'(tok_out) != g_tok_out) begin
      bad++;
      if (errors < 20) $display("  [seq %0d t=%0d] tok_out %0d != golden %0d", s, g_t, tok_out, g_tok_out);
    end
    for (int v = 0; v < N_VOCAB; v++)
      if (int'($signed(logits[v])) != g_logit[v]) begin
        bad++;
        if (errors < 20) $display("  [seq %0d t=%0d] logit[%0d] %0d != golden %0d",
                                  s, g_t, v, $signed(logits[v]), g_logit[v]);
      end
    for (int c = 0; c < N_EMBD; c++) begin
      dbg_addr = EMB_AW'(c);
      #0.1;
      if (int'(dbg_res) != g_res[c]) begin
        bad++;
        if (errors < 20) $display("  [seq %0d t=%0d] res[%0d] %0d != golden %0d",
                                  s, g_t, c, dbg_res, g_res[c]);
      end
    end
    errors += bad;
    steps_checked++;
  endtask

  initial begin
    fd = $fopen({MEM_DIR, "/tv.txt"}, "r");
    if (fd == 0) begin
      $display("ERROR: cannot open %s/tv.txt -- run sw/export.py and check MEM_DIR", MEM_DIR);
      $finish;
    end
    rc = $fscanf(fd, "%d %d %d", n_total, plen, nsteps);
    if (!$value$plusargs("NSEQ=%d", nseq) || nseq > n_total) nseq = n_total;
    $display("tb_gpt_top: %0d sequences, prompt %0d, %0d decode steps each", nseq, plen, nsteps);

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    for (int s = 0; s < nseq; s++) begin
      automatic int seq_err = errors;
      automatic int prev_out = 0;
      for (int t = 0; t < nsteps; t++) begin
        automatic int tok;
        read_golden();
        if (t < plen) begin
          tok = g_tok_in;          // prompt token
          prompt[t] = tok;
        end else begin
          tok = prev_out;          // closed loop: feed back the DUT's own output
          if (tok != g_tok_in) begin
            errors++;
            $display("  [seq %0d t=%0d] fed-back token %0d != golden %0d", s, t, tok, g_tok_in);
          end
        end
        run_step(tok, t);
        check_step(s);
        prev_out = int'(tok_out);
        if (t >= plen - 1) gen[t - plen + 1] = prev_out;
      end

      // task-level check: is the generated sequence the sorted prompt?
      begin
        automatic int k = 0, ok = 1;
        for (int v = 0; v < N_VOCAB; v++) counts[v] = 0;
        for (int i = 0; i < plen; i++) counts[prompt[i]]++;
        for (int v = 0; v < N_VOCAB; v++)
          for (int c = 0; c < counts[v]; c++) begin
            if (gen[k] != v) ok = 0;
            k++;
          end
        task_ok += ok;
      end
      if (errors == seq_err) seq_ok++;

      if ($test$plusargs("VERBOSE") || s < 4) begin
        automatic string line = "";
        for (int i = 0; i < plen; i++) line = {line, $sformatf("%0d ", prompt[i])};
        line = {line, "-> "};
        for (int i = 0; i < plen; i++) line = {line, $sformatf("%0d ", gen[i])};
        $display("  seq %3d: %s %s", s, line, (errors == seq_err) ? "bit-exact" : "MISMATCH");
      end
    end

    $display("------------------------------------------------------------");
    $display("decode steps checked : %0d", steps_checked);
    $display("bit-exact sequences  : %0d / %0d", seq_ok, nseq);
    $display("sorted correctly     : %0d / %0d", task_ok, nseq);
    $display("mismatching values   : %0d", errors);
    $display("avg cycles / token   : %0d  (%.1f us @ 100 MHz)",
             cycles / steps_checked, real'(cycles) / steps_checked / 100.0);
    $display("cycle profile (per token, summed over all layers):");
    for (int k = 1; k < NST; k++)
      $display("  %-6s %7.1f  (%4.1f%%)", st_name(k), real'(st_cycles[k]) / steps_checked,
               100.0 * real'(st_cycles[k]) / real'(cycles));
    $display("RESULT: %s", (errors == 0) ? "PASS" : "FAIL");
    $display("------------------------------------------------------------");
    $fclose(fd);
    $finish;
  end
endmodule
