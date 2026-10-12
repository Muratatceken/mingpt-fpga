// ---------------------------------------------------------------------------
// rom.sv -- synchronous-read ROM initialised from a hex file ($readmemh).
// Vivado infers block RAM for large depths (weights) and LUT-ROM for small ones.
// Read latency: 1 clock.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module rom #(
  parameter int    WIDTH = 8,
  parameter int    DEPTH = 256,
  parameter string INIT  = ""
) (
  input  logic                     clk,
  input  logic [$clog2(DEPTH)-1:0] addr,
  output logic [WIDTH-1:0]         dout
);
  logic [WIDTH-1:0] mem [DEPTH];

  initial $readmemh(INIT, mem);

  always_ff @(posedge clk) dout <= mem[addr];
endmodule
