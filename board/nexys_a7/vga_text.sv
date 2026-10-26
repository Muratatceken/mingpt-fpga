// ---------------------------------------------------------------------------
// vga_text.sv -- 640 x 480 @ 60 Hz VGA text display, 80 x 30 characters of 8 x 16 pixels
//
// Text RAM (2400 bytes, one character per byte, initial contents SCREEN_INIT): port A belongs to the
// generator (lm_gen.sv: write, and read for moving words), port B to the display.
// Row 0 (title) and row ROWS+1 (status) are fixed; rows 1..ROWS are a circular text area whose first
// row is physical row 1 + top.
// Pixel clock = clk / PIX_DIV (25 MHz: 25.0 instead of 25.175 MHz, 59.5 Hz, accepted by monitors).
// Display pipeline: counters -> text RAM -> font ROM -> pixel, 3 pixel clocks; syncs delayed to match.
// px_v / px_x / px_y mark the pixel on the outputs (used by the testbench to capture frames).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module vga_text #(
  parameter string SCREEN_INIT = "",
  parameter string FONT_INIT   = "",
  parameter int    PIX_DIV     = 2,
  parameter int    ROWS        = 28
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        a_we,
  input  logic [11:0] a_addr,
  input  logic [7:0]  a_wdata,
  output logic [7:0]  a_q,
  input  logic [4:0]  top,
  output logic [3:0]  vga_r,
  output logic [3:0]  vga_g,
  output logic [3:0]  vga_b,
  output logic        vga_hs,
  output logic        vga_vs,
  output logic        px_v,
  output logic [9:0]  px_x,
  output logic [9:0]  px_y
);
  localparam int H_VIS = 640, H_FP = 16, H_SY = 96, H_BP = 48, H_TOT = H_VIS + H_FP + H_SY + H_BP;
  localparam int V_VIS = 480, V_FP = 10, V_SY = 2,  V_BP = 33, V_TOT = V_VIS + V_FP + V_SY + V_BP;

  // ---------------- memories ----------------
  logic [7:0] tram [2400];
  logic [7:0] font [2048];
  initial begin
    if (SCREEN_INIT != "") $readmemh(SCREEN_INIT, tram);
    if (FONT_INIT != "")   $readmemh(FONT_INIT, font);
  end

  logic        pce;
  logic [11:0] b_addr;
  logic [7:0]  b_q, f_q;
  logic [10:0] f_addr;

  always_ff @(posedge clk) begin
    if (a_we) tram[a_addr] <= a_wdata;
    a_q <= tram[a_addr];
  end
  always_ff @(posedge clk) if (pce) b_q <= tram[b_addr];
  always_ff @(posedge clk) if (pce) f_q <= font[f_addr];

  // ---------------- timing ----------------
  logic [2:0] pdiv;
  logic [9:0] hc, vc;
  assign pce = (pdiv == 3'(PIX_DIV - 1));
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      pdiv <= '0; hc <= '0; vc <= '0;
    end else begin
      pdiv <= pce ? 3'd0 : pdiv + 1'b1;
      if (pce) begin
        if (hc == 10'(H_TOT - 1)) begin
          hc <= '0;
          vc <= (vc == 10'(V_TOT - 1)) ? 10'd0 : vc + 1'b1;
        end else hc <= hc + 1'b1;
      end
    end
  end

  // stage 0: character address
  logic [4:0] trow, prow;
  logic [5:0] s;
  logic [6:0] tcol;
  always_comb begin
    trow = vc[8:4];
    tcol = hc[9:3];
    s    = 6'(trow) - 6'd1 + 6'(top);
    if (trow == 5'd0 || trow >= 5'(ROWS + 1)) prow = trow;
    else prow = 5'(1 + ((s >= 6'(ROWS)) ? s - 6'(ROWS) : s));
    b_addr = 12'(prow) * 12'd80 + 12'(tcol);
  end

  // stages 1..3
  logic [3:0] grow1;
  logic [2:0] gcol1, gcol2;
  logic [2:0] vis_d, hs_d, vs_d, bar_d;
  logic [9:0] x1, x2, y1, y2;
  logic       vis0, hs0, vs0, bar0;
  assign vis0 = (hc < 10'(H_VIS)) && (vc < 10'(V_VIS));
  assign hs0  = !((hc >= 10'(H_VIS + H_FP)) && (hc < 10'(H_VIS + H_FP + H_SY)));    // negative sync
  assign vs0  = !((vc >= 10'(V_VIS + V_FP)) && (vc < 10'(V_VIS + V_FP + V_SY)));
  assign bar0 = (trow == 5'd0) || (trow == 5'(ROWS + 1));
  assign f_addr = {b_q[6:0], grow1};

  logic pix;
  assign pix = f_q[3'd7 - gcol2];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      vis_d <= '0; hs_d <= '1; vs_d <= '1; bar_d <= '0;
      vga_r <= '0; vga_g <= '0; vga_b <= '0; vga_hs <= 1'b1; vga_vs <= 1'b1;
      px_v  <= 1'b0;
    end else begin
      px_v <= pce && vis_d[1];
      if (pce) begin
        vis_d <= {vis_d[1:0], vis0};
        hs_d  <= {hs_d[1:0], hs0};
        vs_d  <= {vs_d[1:0], vs0};
        bar_d <= {bar_d[1:0], bar0};
        grow1 <= vc[3:0];
        gcol1 <= hc[2:0];
        gcol2 <= gcol1;
        x1 <= hc; x2 <= x1; px_x <= x2;
        y1 <= vc; y2 <= y1; px_y <= y2;
        vga_hs <= hs_d[1];
        vga_vs <= vs_d[1];
        if (!vis_d[1]) begin
          vga_r <= 4'h0; vga_g <= 4'h0; vga_b <= 4'h0;
        end else if (bar_d[1]) begin                      // title / status: white on blue
          vga_r <= pix ? 4'hF : 4'h1; vga_g <= pix ? 4'hF : 4'h3; vga_b <= pix ? 4'hF : 4'h7;
        end else begin                                    // text: light grey on black
          vga_r <= pix ? 4'hD : 4'h0; vga_g <= pix ? 4'hD : 4'h0; vga_b <= pix ? 4'hD : 4'h0;
        end
      end
    end
  end
endmodule
