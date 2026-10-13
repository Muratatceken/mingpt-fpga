// ---------------------------------------------------------------------------
// zoo_axi.sv -- AXI4-Lite slave around the universal zoo engine (one bitstream,
// every model loaded at run time by the Zynq PS).
//
// Register map (32-bit, byte addresses; WSTRB is ignored: full-word writes)
//   0x0000 CTRL     W  bit0: start one decode step (TOK at POS)
//   0x0004 TOK      RW token id
//   0x0008 POS      RW position in the sequence (0 resets the model state)
//   0x000C STATUS   R  bit0 busy, bit1 done (sticky, cleared by a start)
//   0x0010 CYCLES   R  clock cycles of the last step (start -> logits + argmax ready)
//   0x0014 ARGMAX   R  index of the largest logit of the last step (first one on ties)
//   0x0018 MAXVAL   R  value of that logit (sign-extended)
//   0x001C ID       R  0x5A4F4F01  ("ZOO", version 1)
//   0x0020 INFO0    R  {V[15:0], T[15:0]}
//   0x0024 INFO1    R  {NL[7:0], P[7:0], C[15:0]}
//   0x0028 NVOCAB   RW vocabulary size of the loaded model (<= V): logits beyond it are not
//                      written by the model and are excluded from ARGMAX (reset value V)
//   0x0030 LD_SEL   W  memory to write: 0 program, 1 weights, 2 parameters, 3 tables, 4 LUTs
//   0x0034 LD_ADDR  W  word address
//   0x0038 LD_GO    W  any value: write LD_DATA (256 bits) to LD_SEL[LD_ADDR]
//   0x0040..0x005C  LD_DATA0..7  W  data bits [32i+31 : 32i]
//   0x1000 + 4k     LOGIT[k]  R  logit k of the last step (sign-extended), k < V
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module zoo_axi
  import zoo_pkg::*;
#(
  parameter string MEM_DIR = MEM_PATH
) (
  input  logic        aclk,
  input  logic        aresetn,
  // write address / data / response
  input  logic [15:0] s_axi_awaddr,
  input  logic        s_axi_awvalid,
  output logic        s_axi_awready,
  input  logic [31:0] s_axi_wdata,
  input  logic        s_axi_wvalid,
  output logic        s_axi_wready,
  output logic [1:0]  s_axi_bresp,
  output logic        s_axi_bvalid,
  input  logic        s_axi_bready,
  // read address / data
  input  logic [15:0] s_axi_araddr,
  input  logic        s_axi_arvalid,
  output logic        s_axi_arready,
  output logic [31:0] s_axi_rdata,
  output logic [1:0]  s_axi_rresp,
  output logic        s_axi_rvalid,
  input  logic        s_axi_rready
);
  // ---------------- engine ----------------
  logic               start, busy, done;
  logic [TOK_W-1:0]   tok;
  logic [POS_W-1:0]   pos;
  logic [LA_W-1:0]    l_addr;
  logic signed [15:0] l_data;
  logic [7:0]         dbg_op;
  logic               ld_we;
  logic [2:0]         ld_sel;
  logic [19:0]        ld_addr;
  logic [255:0]       ld_data;

  zoo_core #(.MEM_DIR(MEM_DIR), .LOADABLE(1'b1), .PRELOAD(1'b0)) u_core (
    .clk(aclk), .rst_n(aresetn), .start, .tok_in(tok), .pos, .busy, .done, .l_addr, .l_data, .dbg_op,
    .ld_we, .ld_sel, .ld_addr, .ld_data);

  // ---------------- step control, argmax scan, cycle counter ----------------
  typedef enum logic [1:0] {A_IDLE, A_RUN, A_SCAN} ast_t;
  ast_t               ast;
  logic               done_s;
  logic [31:0]        cyc, cycles;
  logic [15:0]        k, best_k;
  logic signed [15:0] best_v;
  logic [15:0]        rd_logit;          // logit index requested by the AXI read port
  logic               go;                // start requested by a CTRL write
  logic [15:0]        nvocab;            // vocabulary of the loaded model

  assign l_addr = (ast == A_SCAN) ? LA_W'(k) : LA_W'(rd_logit);
  assign start  = go && (ast == A_IDLE);

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      ast    <= A_IDLE;
      done_s <= 1'b0;
      cyc    <= '0;
      cycles <= '0;
      best_k <= '0;
      best_v <= '0;
      k      <= '0;
    end else begin
      if (ast != A_IDLE) cyc <= cyc + 1'b1;
      case (ast)
        A_IDLE: if (go) begin ast <= A_RUN; done_s <= 1'b0; cyc <= 32'd1; end
        A_RUN:  if (done) begin ast <= A_SCAN; k <= '0; end
        default: begin                                     // one logit per cycle
          if (k == 16'd0 || l_data > best_v) begin best_v <= l_data; best_k <= k; end
          if (k == nvocab - 1'b1) begin ast <= A_IDLE; done_s <= 1'b1; cycles <= cyc + 1'b1; end
          else k <= k + 1'b1;
        end
      endcase
    end
  end

  // ---------------- AXI4-Lite write channel ----------------
  logic        aw_ok, w_ok;
  logic [15:0] aw_a;
  logic [31:0] w_d;

  assign s_axi_awready = !aw_ok;
  assign s_axi_wready  = !w_ok;
  assign s_axi_bresp   = 2'b00;

  always_ff @(posedge aclk) begin
    go    <= 1'b0;
    ld_we <= 1'b0;
    if (!aresetn) begin
      aw_ok <= 1'b0; w_ok <= 1'b0; s_axi_bvalid <= 1'b0;
      tok <= '0; pos <= '0; ld_sel <= '0; ld_addr <= '0; ld_data <= '0; nvocab <= 16'(V);
    end else begin
      if (s_axi_awvalid && s_axi_awready) begin aw_ok <= 1'b1; aw_a <= s_axi_awaddr; end
      if (s_axi_wvalid && s_axi_wready)   begin w_ok  <= 1'b1; w_d  <= s_axi_wdata;  end
      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
      if (aw_ok && w_ok && !s_axi_bvalid) begin
        aw_ok <= 1'b0; w_ok <= 1'b0; s_axi_bvalid <= 1'b1;
        case (aw_a)
          16'h0000: go      <= w_d[0];
          16'h0004: tok     <= TOK_W'(w_d);
          16'h0008: pos     <= POS_W'(w_d);
          16'h0030: ld_sel  <= w_d[2:0];
          16'h0034: ld_addr <= w_d[19:0];
          16'h0038: ld_we   <= 1'b1;
          16'h0028: nvocab  <= (w_d[15:0] == 16'd0 || w_d[15:0] > 16'(V)) ? 16'(V) : w_d[15:0];
          default:
            if (aw_a >= 16'h0040 && aw_a <= 16'h005C) ld_data[32 * aw_a[4:2] +: 32] <= w_d;
        endcase
      end
    end
  end

  // ---------------- AXI4-Lite read channel ----------------
  logic        is_logit;
  assign is_logit       = s_axi_araddr[15:12] == 4'h1;
  assign rd_logit       = 16'(s_axi_araddr[11:2]);
  assign s_axi_arready  = !s_axi_rvalid && (ast != A_SCAN);  // logit port busy while scanning
  assign s_axi_rresp    = 2'b00;

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      s_axi_rvalid <= 1'b0;
      s_axi_rdata  <= '0;
    end else begin
      if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
      if (s_axi_arvalid && s_axi_arready) begin
        s_axi_rvalid <= 1'b1;
        if (is_logit) s_axi_rdata <= 32'(l_data);
        else case (s_axi_araddr)
          16'h0004: s_axi_rdata <= 32'(tok);
          16'h0008: s_axi_rdata <= 32'(pos);
          16'h000C: s_axi_rdata <= {30'd0, done_s, ast != A_IDLE};
          16'h0010: s_axi_rdata <= cycles;
          16'h0014: s_axi_rdata <= 32'(best_k);
          16'h0018: s_axi_rdata <= 32'(best_v);
          16'h001C: s_axi_rdata <= 32'h5A4F_4F01;
          16'h0020: s_axi_rdata <= {16'(V), 16'(T)};
          16'h0024: s_axi_rdata <= {8'(NL), 8'(P), 16'(C)};
          16'h0028: s_axi_rdata <= 32'(nvocab);
          default:  s_axi_rdata <= 32'h0;
        endcase
      end
    end
  end
endmodule
