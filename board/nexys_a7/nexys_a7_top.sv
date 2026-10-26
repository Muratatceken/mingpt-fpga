// ---------------------------------------------------------------------------
// nexys_a7_top.sv -- Nexys A7-100T (xc7a100tcsg324-1): a Karpathy language model running on the FPGA
//
// No host computer: the model (microcode, INT8 weights, tables) is in BRAM, initialised by the bitstream
// (mem/nexys/<model>/, sw/zoo/board.py). After configuration the board generates a story (TinyStories
// models) or 27 names (names.txt models) and shows it on a VGA monitor.
//
//   CLK100MHZ -> MMCM -> clk = CLK_MHZ (25 / 50 / 100); VGA pixel clock = clk / (CLK_MHZ / 25)
//
//   BTNC  new run (the random state continues: a new story every press)
//   BTNU  reseed the random generator from SW[15:8], then run
//   BTND  clear the text area
//   CPU_RESET  reset; the first run after reset uses a fresh seed
//   SW0   greedy (up) / sampling (down)          SW[3:1]  temperature (000 = 0.8, see board.py TAUS)
//   SW[5:4] min-p cut-off (00 = 1/32)            SW6      typewriter: 25 tokens/s
//   SW7   7-segment: tokens/s (down) / cycles per token (up)       SW[15:8] seed
//
//   LED0 heartbeat   LED1 generating   LED2 check OK   LED3 check FAIL   LED[15:4] tokens of the run
//   LED16 RGB: blue = generating, green = check OK, red = FAIL (check: greedy runs, and the first
//   run after a reseed with all switches down, against the hash computed by sw/zoo/board.py)
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module nexys_a7_top
  import zoo_pkg::*;
#(
  parameter int CLK_MHZ  = 50,
  parameter int DEB_CYC  = CLK_MHZ * 10_000,        // 10 ms button debounce
  parameter int SLOW_CYC = CLK_MHZ * 40_000,        // typewriter pause: 40 ms
  parameter int SEG_CYC  = CLK_MHZ * 1_000          // 1 ms per 7-segment digit
) (
  input  logic        CLK100MHZ,
  input  logic        CPU_RESETN,
  input  logic        BTNC,
  input  logic        BTNU,
  input  logic        BTND,
  input  logic [15:0] SW,
  output logic [15:0] LED,
  output logic        LED16_R, LED16_G, LED16_B,
  output logic        LED17_R, LED17_G, LED17_B,
  output logic        CA, CB, CC, CD, CE, CF, CG, DP,
  output logic [7:0]  AN,
  output logic [3:0]  VGA_R, VGA_G, VGA_B,
  output logic        VGA_HS, VGA_VS
);
  // ---------------- clocking ----------------
  logic clk, locked;
`ifdef SIM
  assign clk    = CLK100MHZ;                         // the testbench drives the core clock directly
  assign locked = 1'b1;
`else
  logic clk_mmcm, clkfb;
  MMCME2_BASE #(
    .CLKIN1_PERIOD   (10.0),
    .DIVCLK_DIVIDE   (1),
    .CLKFBOUT_MULT_F (10.0),                         // VCO 1000 MHz
    .CLKOUT0_DIVIDE_F(1000.0 / CLK_MHZ)
  ) u_mmcm (
    .CLKIN1(CLK100MHZ), .CLKFBIN(clkfb), .CLKFBOUT(clkfb), .CLKOUT0(clk_mmcm), .LOCKED(locked),
    .PWRDWN(1'b0), .RST(1'b0),
    .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(), .CLKOUT3(), .CLKOUT3B(),
    .CLKOUT4(), .CLKOUT5(), .CLKOUT6());
  BUFG u_bufg (.I(clk_mmcm), .O(clk));
`endif

  // ---------------- reset, synchronisers, debouncers ----------------
  logic [1:0] rstn_s;
  logic [4:0] rst_cnt;
  logic       rst_n;
  always_ff @(posedge clk or negedge locked) begin
    if (!locked) begin
      rstn_s <= '0; rst_cnt <= '0; rst_n <= 1'b0;
    end else begin
      rstn_s <= {rstn_s[0], CPU_RESETN};
      if (!rstn_s[1]) begin rst_cnt <= '0; rst_n <= 1'b0; end
      else if (rst_cnt != 5'h1F) rst_cnt <= rst_cnt + 1'b1;
      else rst_n <= 1'b1;
    end
  end

  logic [15:0] sw_s1, sw;
  always_ff @(posedge clk) begin sw_s1 <= SW; sw <= sw_s1; end

  logic [2:0] btn_raw, btn_s1, btn_s2, btn_st, btn_p;
  logic [$clog2(DEB_CYC + 1)-1:0] deb [3];
  assign btn_raw = {BTND, BTNU, BTNC};
  always_ff @(posedge clk) begin
    btn_s1 <= btn_raw;
    btn_s2 <= btn_s1;
    btn_p  <= '0;
    for (int i = 0; i < 3; i++) begin
      if (!rst_n) begin
        btn_st[i] <= 1'b0; deb[i] <= '0;
      end else if (btn_s2[i] == btn_st[i]) deb[i] <= '0;
      else if (deb[i] == ($clog2(DEB_CYC + 1))'(DEB_CYC)) begin
        btn_st[i] <= btn_s2[i];
        deb[i]    <= '0;
        btn_p[i]  <= btn_s2[i];                      // press
      end else deb[i] <= deb[i] + 1'b1;
    end
  end

  // ---------------- generator + engine ----------------
  logic        tw_we;
  logic [11:0] tw_addr;
  logic [7:0]  tw_data, tw_q;
  logic [4:0]  top;
  logic        running;
  logic [1:0]  check;
  logic [15:0] ntok;
  logic [31:0] toks_bcd, cpt_bcd;

  lm_gen #(.CLK_HZ(CLK_MHZ * 1_000_000), .SLOW_CYC(SLOW_CYC)) u_gen (
    .clk, .rst_n, .run(btn_p[0]), .reseed(btn_p[1]), .clear(btn_p[2]),
    .greedy(sw[0]), .tau(sw[3:1]), .minp(sw[5:4]), .slow(sw[6]), .seed(sw[15:8]),
    .tw_we, .tw_addr, .tw_data, .tw_q, .top, .running, .check, .ntok, .toks_bcd, .cpt_bcd);

  // ---------------- VGA ----------------
  logic       px_v;
  logic [9:0] px_x, px_y;
  vga_text #(.SCREEN_INIT({MEM_PATH, "/screen.mem"}), .FONT_INIT({MEM_PATH, "/font.mem"}),
             .PIX_DIV(CLK_MHZ / 25), .ROWS(board_pkg::ROWS)) u_vga (
    .clk, .rst_n, .a_we(tw_we), .a_addr(tw_addr), .a_wdata(tw_data), .a_q(tw_q), .top,
    .vga_r(VGA_R), .vga_g(VGA_G), .vga_b(VGA_B), .vga_hs(VGA_HS), .vga_vs(VGA_VS), .px_v, .px_x, .px_y);

  // ---------------- 7-segment, LEDs ----------------
  logic [6:0] seg;
  seg7 #(.DIGIT_CYC(SEG_CYC)) u_seg (.clk, .rst_n, .bcd(sw[7] ? cpt_bcd : toks_bcd), .seg, .dp(DP), .an(AN));
  assign {CG, CF, CE, CD, CC, CB, CA} = seg;

  logic [25:0] hb;
  always_ff @(posedge clk) hb <= hb + 1'b1;
  logic dim;                                          // the RGB LEDs are very bright: 1/8 duty
  assign dim = (hb[2:0] == 3'd0);

  assign LED = {ntok[11:0], check == 2'd2, check == 2'd1, running, hb[25]};
  assign LED16_B = dim && running;
  assign LED16_G = dim && check == 2'd1;
  assign LED16_R = dim && check == 2'd2;
  assign {LED17_R, LED17_G, LED17_B} = 3'b000;
endmodule
