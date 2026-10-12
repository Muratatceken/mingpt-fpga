// ---------------------------------------------------------------------------
// tb_arith.sv -- unit tests for the arithmetic building blocks:
//   div_seq   : q = floor(n/d)        checked with  q*d <= n < (q+1)*d
//   isqrt_seq : r = floor(sqrt(n))    checked with  r^2 <= n < (r+1)^2
//   requant   : against a behavioural reference
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_arith;
  logic clk, rst_n = 1'b0;
  initial clk = 1'b0;
  always #5 clk = ~clk;

  int errors = 0;

  // ---------------- div_seq ----------------
  logic        d_start = 1'b0, d_done;
  logic [47:0] d_num, d_den, d_quo;
  div_seq #(.W(48)) u_div (.clk, .rst_n, .start(d_start), .num(d_num), .den(d_den),
                           .done(d_done), .quo(d_quo));

  // ---------------- isqrt_seq ----------------
  logic        s_start = 1'b0, s_done;
  logic [63:0] s_num;
  logic [31:0] s_root;
  isqrt_seq #(.W(64)) u_sqrt (.clk, .rst_n, .start(s_start), .num(s_num),
                              .done(s_done), .root(s_root));

  // ---------------- requant ----------------
  logic               r_v = 1'b0, r_ov, r_out16;
  logic signed [31:0] r_acc, r_bias;
  logic [15:0]        r_m0;
  logic [5:0]         r_sh;
  logic [8:0]         r_oidx;
  logic signed [15:0] r_out;
  requant u_rq (.clk, .in_valid(r_v), .in_idx(9'd0), .acc(r_acc), .bias(r_bias), .m0(r_m0),
                .sh(r_sh), .out16(r_out16), .out_valid(r_ov), .out_idx(r_oidx), .out(r_out));

  function automatic longint ref_requant(longint acc, longint m0, int sh, bit is16);
    longint v = ((acc * m0) + (64'sd1 <<< (sh - 1))) >>> sh;
    longint lo = is16 ? -32768 : -128, hi = is16 ? 32767 : 127;
    return (v < lo) ? lo : (v > hi) ? hi : v;
  endfunction

  function automatic longint rand48();
    return {$urandom(), $urandom()} & 64'hFFFF_FFFF_FFFF;
  endfunction

  initial begin
    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    // ---- divider ----
    for (int k = 0; k < 2000; k++) begin
      automatic longint n = rand48() >> ($urandom() % 40);
      automatic longint d = (rand48() >> ($urandom() % 47)) | 1;
      @(posedge clk); d_num <= n[47:0]; d_den <= d[47:0]; d_start <= 1'b1;
      @(posedge clk); d_start <= 1'b0;
      do @(posedge clk); while (!d_done);
      if (!(d_quo * d <= n && n < (d_quo + 1) * d)) begin
        errors++; $display("div  FAIL %0d / %0d = %0d", n, d, d_quo);
      end
    end
    $display("div_seq   : 2000 random tests done");

    // ---- square root (n < 2^62 so (r+1)^2 fits in 64 bits) ----
    for (int k = 0; k < 2000; k++) begin
      automatic longint unsigned n = {$urandom(), $urandom()} >> (2 + $urandom() % 60);
      automatic longint unsigned r;
      @(posedge clk); s_num <= n; s_start <= 1'b1;
      @(posedge clk); s_start <= 1'b0;
      do @(posedge clk); while (!s_done);
      r = s_root;
      if (!(r * r <= n && n < (r + 1) * (r + 1))) begin
        errors++; $display("sqrt FAIL sqrt(%0d) = %0d", n, r);
      end
    end
    $display("isqrt_seq : 2000 random tests done");

    // ---- requant ----
    for (int k = 0; k < 2000; k++) begin
      automatic longint acc  = $signed($urandom()) >>> ($urandom() % 24);
      automatic longint bias = $signed($urandom()) >>> (8 + $urandom() % 20);
      automatic longint m0   = 32768 + ($urandom() % 32768);
      automatic int     sh   = 1 + $urandom() % 40;
      automatic bit     is16 = $urandom() % 2;
      automatic longint exp_v = ref_requant(acc + bias, m0, sh, is16);
      @(posedge clk);
      r_acc <= acc[31:0]; r_bias <= bias[31:0]; r_m0 <= m0[15:0]; r_sh <= sh[5:0];
      r_out16 <= is16; r_v <= 1'b1;
      @(posedge clk); r_v <= 1'b0;
      do @(posedge clk); while (!r_ov);
      if (longint'(r_out) != exp_v) begin
        errors++; $display("requant FAIL acc=%0d bias=%0d m0=%0d sh=%0d -> %0d (exp %0d)",
                           acc, bias, m0, sh, r_out, exp_v);
      end
    end
    $display("requant   : 2000 random tests done");

    $display("RESULT: %s (%0d errors)", errors == 0 ? "PASS" : "FAIL", errors);
    $finish;
  end
endmodule
