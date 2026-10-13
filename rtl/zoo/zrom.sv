// ---------------------------------------------------------------------------
// zrom.sv -- synchronous-read ROM initialised from a hex file ($readmemh).
// Like rom.sv, but the address width is passed explicitly (from zoo_pkg), so
// tiny ROMs (depth 2) are fine. Read latency: 1 clock.
// LOADABLE = 1 adds a write port: the memory becomes a RAM that the host (the
// Zynq PS) fills at run time, so one bitstream serves every model, and
// synthesis cannot constant-fold a particular model's contents.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module zrom #(
  parameter int    WIDTH = 8,
  parameter int    DEPTH = 2,
  parameter int    AW    = 1,
  parameter string INIT  = "",
  parameter bit    LOADABLE = 1'b0
) (
  input  logic             clk,
  input  logic [AW-1:0]    addr,
  output logic [WIDTH-1:0] dout,
  input  logic             we,
  input  logic [AW-1:0]    waddr,
  input  logic [WIDTH-1:0] wdata
);
  logic [WIDTH-1:0] mem [DEPTH];

  initial $readmemh(INIT, mem);

  always_ff @(posedge clk) begin
    if (LOADABLE && we) mem[waddr] <= wdata;
    dout <= mem[addr];
  end
endmodule
