`timescale 1ns/1ps
// decision_top against sw/decision/hw_ref.py: every trace word (proposal, applied, reason, modes, valid) and the
// episode hash; also reports the observation-to-applied-action cycle count per step.
module tb_dec_top;
  parameter int    POLICY = 0;
  parameter string TRACE = "build/decision/trace_lm.txt";
  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  logic        start = 1'b0, busy, done, tr_we;
  logic [15:0] base = '0, tr_word;
  logic [7:0]  nsteps = '0;
  logic [31:0] hash, dec_cycles;
  decision_top #(.POLICY(POLICY)) dut (.clk(clk), .rst_n(rst_n), .start(start), .base(base), .nsteps(nsteps),
                                       .busy(busy), .done(done), .hash(hash), .tr_we(tr_we), .tr_word(tr_word),
                                       .dec_cycles(dec_cycles));
  int fd = 0, ns = 0, b = 0, n = 0, err = 0, steps = 0, kk = 0;
  longint h = 0;
  int exp_w [256];
  int maxcyc = 0, mincyc = 0;
  initial begin
    mincyc = 1 << 30;
    repeat (3) @(posedge clk); rst_n = 1'b1;
    fd = $fopen(TRACE, "r");
    void'($fscanf(fd, "%d", ns));
    for (int s = 0; s < ns; s++) begin
      void'($fscanf(fd, "%d %d %d", b, n, h));
      for (int k = 0; k < n; k++) void'($fscanf(fd, "%d", exp_w[k]));
      @(negedge clk); base = 16'(b); nsteps = 8'(n); start = 1'b1;
      @(negedge clk); start = 1'b0;
      kk = 0;
      while (!done) begin
        @(posedge clk); #1;
        if (tr_we) begin
          if (int'(tr_word) != exp_w[kk]) begin
            if (err < 8) $display("TRACE MISMATCH scen %0d step %0d: got %h exp %h", s, kk, tr_word, exp_w[kk]);
            err++;
          end
          if (tr_word[11]) begin                   // proposal arrived: deadline not exercised
            if (int'(dec_cycles) > maxcyc) maxcyc = int'(dec_cycles);
            if (int'(dec_cycles) < mincyc) mincyc = int'(dec_cycles);
          end
          kk++; steps++;
        end
      end
      if (longint'(hash) != h) begin
        if (err < 8) $display("HASH MISMATCH scen %0d: got %h exp %h", s, hash, h);
        err++;
      end
    end
    $display("RESULT decision_top policy %0d scenarios %0d steps %0d errors %0d dec_cycles min %0d max %0d", POLICY,
             ns, steps, err, mincyc, maxcyc);
    if (err == 0) $display("PASS: decision_top trace and hash identical to hw_ref");
    else $display("FAIL");
    $finish;
  end
endmodule
