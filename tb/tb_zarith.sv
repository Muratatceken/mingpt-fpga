// ---------------------------------------------------------------------------
// tb_zarith.sv -- unit tests of the zoo engine's sequential arithmetic.
//
//  zdiv  W=12, QB=6   EXHAUSTIVE: every (num, den) with den > 0 and num >> QB < den
//  zdiv  W=64, QB=41  (NORM reciprocal square root)      random + corner cases
//  zdiv  W=34, QB=18  (softmax reciprocal 2^32 / E)     every E in [2^15, 2^24)
//  isqrt W=56         random + corner cases (powers of two, squares, squares - 1)
// Results are checked by the defining properties, not by a second implementation:
//  q*den <= num < (q+1)*den      and      r^2 <= n < (r+1)^2
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_zarith;
  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;

  // ---- DUTs ----
  logic        s12, d12;  logic [11:0] n12, q12, den12;
  zdiv #(.W(12), .QB(6))  u12 (.clk, .rst_n, .start(s12), .num(n12), .den(den12), .done(d12), .quo(q12));
  logic        s64, d64;  logic [63:0] n64, q64, den64;
  zdiv #(.W(64), .QB(41)) u64 (.clk, .rst_n, .start(s64), .num(n64), .den(den64), .done(d64), .quo(q64));
  logic        s34, d34;  logic [33:0] q34, den34;
  zdiv #(.W(34), .QB(18)) u34 (.clk, .rst_n, .start(s34), .num(34'h1_0000_0000), .den(den34), .done(d34), .quo(q34));
  logic        ss, ds;    logic [55:0] nsq;  logic [27:0] rsq;
  isqrt_seq #(.W(56))     usq (.clk, .rst_n, .start(ss), .num(nsq), .done(ds), .root(rsq));

  longint errors = 0, n_12 = 0, n_64 = 0, n_34 = 0, n_sq = 0;

  function automatic bit div_ok(input logic [127:0] num, input logic [127:0] den, input logic [127:0] q);
    return (q * den <= num) && (num < (q + 1) * den);
  endfunction

  function automatic bit sqrt_ok(input logic [127:0] n, input logic [127:0] r);
    return (r * r <= n) && (n < (r + 1) * (r + 1));
  endfunction

  task automatic run12(input logic [11:0] a, input logic [11:0] b);
    @(posedge clk); n12 <= a; den12 <= b; s12 <= 1'b1;
    @(posedge clk); s12 <= 1'b0;
    do @(posedge clk); while (!d12);
    n_12++;
    if (!div_ok(128'(a), 128'(b), 128'(q12))) begin
      if (errors < 10) $display("zdiv12 FAIL %0d / %0d -> %0d", a, b, q12);
      errors++;
    end
  endtask

  task automatic run64(input logic [63:0] a, input logic [63:0] b);
    @(posedge clk); n64 <= a; den64 <= b; s64 <= 1'b1;
    @(posedge clk); s64 <= 1'b0;
    do @(posedge clk); while (!d64);
    n_64++;
    if (!div_ok(128'(a), 128'(b), 128'(q64))) begin
      if (errors < 10) $display("zdiv64 FAIL %0h / %0h -> %0h", a, b, q64);
      errors++;
    end
  endtask

  task automatic run34(input logic [33:0] b);
    @(posedge clk); den34 <= b; s34 <= 1'b1;
    @(posedge clk); s34 <= 1'b0;
    do @(posedge clk); while (!d34);
    n_34++;
    if (!div_ok(128'(34'h1_0000_0000), 128'(b), 128'(q34))) begin
      if (errors < 10) $display("zdiv34 FAIL 2^32 / %0d -> %0d", b, q34);
      errors++;
    end
  endtask

  task automatic runsq(input logic [55:0] a);
    @(posedge clk); nsq <= a; ss <= 1'b1;
    @(posedge clk); ss <= 1'b0;
    do @(posedge clk); while (!ds);
    n_sq++;
    if (!sqrt_ok(128'(a), 128'(rsq))) begin
      if (errors < 10) $display("isqrt FAIL sqrt(%0h) -> %0h", a, rsq);
      errors++;
    end
  endtask

  // random 64-bit division with quotient < 2^41 (the NORM operating range: num = K << k, den = isqrt(..))
  task automatic rand64();
    logic [63:0] b, q, r, a;
    b = {$urandom, $urandom} >> ($urandom % 64);
    if (b == 0) b = 1;
    q = {$urandom, $urandom} & ((64'd1 << 41) - 1);
    q = q >> ($urandom % 41);
    r = (b > 1) ? ({$urandom, $urandom} % b) : 0;
    if ((128'(q) * 128'(b) + 128'(r)) >> 64 != 0) begin q = 0; end   // keep num within 64 bits
    a = q * b + r;
    run64(a, b);
  endtask

  initial begin
    {s12, s64, s34, ss} = '0;
    repeat (3) @(posedge clk);
    rst_n <= 1'b1;

    // 1) exhaustive small divider: every valid pair
    for (int b = 1; b < 4096; b++)
      for (int a = 0; a < 4096; a++)
        if ((a >> 6) < b) run12(12'(a), 12'(b));
    $display("zdiv  W=12 QB=6  exhaustive : %0d cases", n_12);

    // 2) NORM divider: corner cases + random
    run64(64'd0, 64'd1);
    run64((64'd1 << 41) - 1, 64'd1);
    run64(64'hFFFF_FFFF_FFFF_FFFF, 64'hFFFF_FFFF_FFFF_FFFF);
    run64(64'h7FFF_FFFF_FFFF_FFFF, 64'd1 << 23);
    for (int i = 0; i < 2_000_000; i++) rand64();
    $display("zdiv  W=64 QB=41 random     : %0d cases", n_64);

    // 3) softmax reciprocal: every denominator up to 2^24 (T <= 512 positions of weight <= 2^15)
    for (int e = 1 << 15; e < (1 << 24); e++) run34(34'(e));
    $display("zdiv  W=34 QB=18 2^32/E     : %0d cases (all E in [2^15, 2^24))", n_34);

    // 4) square root: corner cases + random over every bit length
    for (int k = 0; k < 56; k++) begin
      runsq(56'(1) << k);
      runsq((56'(1) << k) - 1);
    end
    for (int i = 1; i < 1 << 14; i++) begin                     // squares and their neighbours
      runsq(56'(i) * 56'(i));
      runsq(56'(i) * 56'(i) - 1);
    end
    for (int i = 0; i < 1_000_000; i++) runsq(56'({$urandom, $urandom}) >> ($urandom % 56));
    $display("isqrt W=56       random     : %0d cases", n_sq);

    if (errors == 0) $display("PASS: all %0d arithmetic cases correct", n_12 + n_64 + n_34 + n_sq);
    else             $display("FAIL: %0d errors", errors);
    $finish;
  end
endmodule
