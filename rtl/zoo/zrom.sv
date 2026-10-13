// ---------------------------------------------------------------------------
// zrom.sv -- synchronous-read ROM initialised from a hex file ($readmemh).
// Like rom.sv, but the address width is passed explicitly (from zoo_pkg), so
// tiny ROMs (depth 2) are fine. Read latency: 1 clock.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module zrom #(
  parameter int    WIDTH = 8,
  parameter int    DEPTH = 2,
  parameter int    AW    = 1,
  parameter string INIT  = ""
) (
  input  logic             clk,
  input  logic [AW-1:0]    addr,
  output logic [WIDTH-1:0] dout
);
  logic [WIDTH-1:0] mem [DEPTH];

  initial $readmemh(INIT, mem);

  always_ff @(posedge clk) dout <= mem[addr];
endmodule
