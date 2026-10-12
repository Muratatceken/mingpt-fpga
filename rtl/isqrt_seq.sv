// ---------------------------------------------------------------------------
// isqrt_seq.sv -- sequential integer square root, root = floor(sqrt(num)).
// Classic digit-by-digit (restoring) method: consumes 2 input bits and
// produces 1 root bit per clock, W/2 clocks in total. W must be even.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module isqrt_seq #(
  parameter int W = 64
) (
  input  logic           clk,
  input  logic           rst_n,
  input  logic           start,
  input  logic [W-1:0]   num,
  output logic           done,
  output logic [W/2-1:0] root
);
  localparam int RW = W / 2 + 3;   // remainder width (rem <= 2*root, shifted by 2)

  logic [W-1:0]             v;
  logic [RW-1:0]            rem;
  logic [W/2-1:0]           r;
  logic [$clog2(W/2+1)-1:0] cnt;
  logic                     busy;

  logic [RW-1:0] rem_sh, trial;
  assign rem_sh = {rem[RW-3:0], v[W-1:W-2]};
  assign trial  = {{(RW - W/2 - 2){1'b0}}, r, 2'b01};
  assign root   = r;

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (!rst_n) begin
      busy <= 1'b0;
    end else if (start) begin
      v    <= num;
      rem  <= '0;
      r    <= '0;
      cnt  <= ($clog2(W/2+1))'(W / 2);
      busy <= 1'b1;
    end else if (busy) begin
      v <= {v[W-3:0], 2'b00};
      if (rem_sh >= trial) begin
        rem <= rem_sh - trial;
        r   <= {r[W/2-2:0], 1'b1};
      end else begin
        rem <= rem_sh;
        r   <= {r[W/2-2:0], 1'b0};
      end
      cnt <= cnt - 1'b1;
      if (cnt == 1) begin
        busy <= 1'b0;
        done <= 1'b1;
      end
    end
  end
endmodule
