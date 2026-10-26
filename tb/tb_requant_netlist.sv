// ---------------------------------------------------------------------------
// tb_requant_netlist.sv -- rtl/requant.sv with registered inputs (as in zoo_core: g1_acc register,
// parameter ROM output) against its Yosys synth_xilinx netlist (module requant_reg_net), random vectors.
// scripts/requant_netlist_check.sh: with DSP48E1 mapping the netlist is wrong (negative products saturate to
// +max), with -nodsp it matches. Reproduces the reason scripts/nexys_netlist_sim.sh synthesises with -nodsp.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module requant_reg (
  input  logic clk,
  input  logic v_in,
  input  logic signed [31:0] acc_in, bias_in,
  input  logic [15:0] m0_in,
  input  logic [5:0] sh_in,
  input  logic o16_in,
  output logic ov,
  output logic signed [15:0] o
);
  logic v, o16;
  logic signed [31:0] acc, bias;
  logic [15:0] m0;
  logic [5:0] sh;
  logic [15:0] oi;
  always_ff @(posedge clk) begin
    v <= v_in; acc <= acc_in; bias <= bias_in; m0 <= m0_in; sh <= sh_in; o16 <= o16_in;
  end
  requant #(.IDX_W(16)) u (.clk, .in_valid(v), .in_idx(16'd0), .acc, .bias, .m0, .sh, .out16(o16),
                           .out_valid(ov), .out_idx(oi), .out(o));
endmodule

`ifndef SYNTHESIS
module tb_requant_netlist;
  logic clk = 1'b0;
  always #5 clk = ~clk;
  logic v = 1'b0, o16 = 1'b0;
  logic [15:0] m0 = 16'h8000;
  logic signed [31:0] acc = 0, bias = 0;
  logic [5:0] sh = 6'd1;
  logic ov_r, ov_n;
  logic signed [15:0] o_r, o_n;
  requant_reg     r (.clk, .v_in(v), .acc_in(acc), .bias_in(bias), .m0_in(m0), .sh_in(sh), .o16_in(o16), .ov(ov_r), .o(o_r));
  requant_reg_net n (.clk, .v_in(v), .acc_in(acc), .bias_in(bias), .m0_in(m0), .sh_in(sh), .o16_in(o16), .ov(ov_n), .o(o_n));
  int err = 0, neg = 0;
  initial begin
    repeat (4) @(negedge clk);
    for (int i = 0; i < 20000; i++) begin
      @(negedge clk);
      if (i > 5 && (o_r !== o_n || ov_r !== ov_n)) begin
        if (err < 5) $display("MISMATCH vector %0d: rtl %0d netlist %0d", i, o_r, o_n);
        err++;
      end
      v    = 1'b1;
      acc  = (i % 2) ? $random : $random % 100000;
      bias = (i % 3 == 0) ? $random % 10000 : 0;
      m0   = 16'h8000 | 16'($random);
      sh   = 6'(1 + ($urandom % 40));
      o16  = $random;
      if (acc + bias < 0) neg++;
    end
    $display("RESULT requant netlist: %0d mismatches in 20000 vectors (%0d with a negative sum)", err, neg);
    $finish;
  end
endmodule
`endif
