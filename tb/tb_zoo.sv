// ---------------------------------------------------------------------------
// tb_zoo.sv -- bit-exact testbench for the microcoded zoo engine.
//
// Reads mem/zoo/<model>/tv.txt (written by sw/zoo/export_hw.py from the
// integer interpreter quant.py:IntRunner): teacher-forced test names, and for
// every decode step the input token and all V int16 logits. Every logit of
// every step must match exactly. Prints a cycle profile per opcode.
//
// Plusargs: +NSEQ=<n> (default: all sequences in tv.txt)
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_zoo;
  import zoo_pkg::*;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;   // 100 MHz

  logic                start = 1'b0;
  logic [TOK_W-1:0]    tok_in = '0;
  logic [POS_W-1:0]    pos = '0;
  logic                busy, done;
  logic [LA_W-1:0]     l_addr = '0;
  logic signed [15:0]  l_data;
  logic [7:0]          dbg_op;

  zoo_core dut (.*);

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

  int fd, rc, nseq, nv, len, tok, maxseq;
  int gold [V];
  int errors = 0, steps = 0, bad_steps = 0;
  longint t0, step_cyc, tot_cyc = 0, max_cyc = 0, min_cyc = 64'h7fffffff;

  initial begin
    if (!$value$plusargs("NSEQ=%d", maxseq)) maxseq = 1 << 30;
    fd = $fopen({MEM_PATH, "/tv.txt"}, "r");
    if (fd == 0) $fatal(1, "cannot open %s/tv.txt", MEM_PATH);
    rc = $fscanf(fd, "%d %d", nseq, nv);
    if (nv != V) $fatal(1, "tv.txt has V=%0d, package V=%0d", nv, V);
    if (nseq > maxseq) nseq = maxseq;

    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);

    for (int s = 0; s < nseq; s++) begin
      rc = $fscanf(fd, "%d", len);
      for (int tt = 0; tt < len; tt++) begin
        int step_err;
        rc = $fscanf(fd, "%d", tok);
        for (int k = 0; k < V; k++) rc = $fscanf(fd, "%d", gold[k]);
        @(posedge clk);
        start  <= 1'b1;
        tok_in <= TOK_W'(tok);
        pos    <= POS_W'(tt);
        t0 = cyc;
        @(posedge clk);
        start <= 1'b0;
        do @(posedge clk); while (!done);
        step_cyc = cyc - t0;
        tot_cyc += step_cyc;
        if (step_cyc > max_cyc) max_cyc = step_cyc;
        if (step_cyc < min_cyc) min_cyc = step_cyc;
        step_err = 0;
        for (int k = 0; k < V; k++) begin
          l_addr = LA_W'(k);
          #1;
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

    $display("");
    $display("=== %s: %0d sequences, %0d decode steps, %0d x %0d logits checked ===", NAME, nseq, steps, steps, V);
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
    if (errors == 0) $display("PASS: RTL is bit-exact with the integer reference");
    else             $display("FAIL: %0d logit mismatches in %0d steps", errors, bad_steps);
    $finish;
  end
endmodule
