// ---------------------------------------------------------------------------
// tb_selftest.sv -- runs the on-chip self-test twice:
//   1) normal          : all 729 sequences must pass
//   2) inject_fault = 1: all 729 sequences must fail (checker is live)
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_selftest;
  import gpt_pkg::*;

  logic clk, rst_n = 1'b0;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  logic        start = 1'b0, inject_fault = 1'b0;
  logic        busy, done, pass;
  logic [15:0] n_tested, n_fail;
  logic [47:0] cycles;

  gpt_selftest dut (.*);

  int errors = 0;

  task automatic run(input bit fault, input int exp_fail);
    @(posedge clk);
    inject_fault <= fault;
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    do @(posedge clk); while (!done);
    $display("inject_fault=%0d : tested %0d, failed %0d, pass=%0d, %0d cycles (%.3f s @ 100 MHz)",
             fault, n_tested, n_fail, pass, cycles, real'(cycles) / 1.0e8);
    if (n_fail != 16'(exp_fail) || n_tested != 16'(N_VOCAB ** SORT_LEN)) errors++;
  endtask

  initial begin
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    run(1'b0, 0);
    run(1'b1, N_VOCAB ** SORT_LEN);
    $display("RESULT: %s", (errors == 0) ? "PASS" : "FAIL");
    $finish;
  end
endmodule
