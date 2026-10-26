// ---------------------------------------------------------------------------
// tb_nexys.sv -- simulation of the whole Nexys A7-100T design (board/nexys_a7/nexys_a7_top.sv) as it
// runs on the board: reset, the automatic first run, button presses, VGA output.
//   RTL (sw/zoo/board_sim.py: Verilator; scripts/nexys_icarus.sh: sv2v + Icarus) is compiled with -DSIM
//   (the MMCM is bypassed; the testbench clock is the core clock).
//   NETLIST (scripts/nexys_netlist_sim.sh): the Yosys netlist of the real top, MMCM + BUFG as
//   pass-through models; parameters (DEB_CYC, SLOW_CYC, SEG_CYC) were fixed at synthesis.
//
// Keys (+PRESS=<keys>), after the automatic run that follows the reset:
//   C U D R   at idle: wait until the previous run (and the screen) is finished, report the idle state
//             (TBIDLE line, text screen), then press BTNC (new run) / BTNU (reseed + run) / BTND (clear)
//             or pulse CPU_RESETN
//   c u d r   during the run in progress: wait until it has generated +DLY<i> tokens (LED[15:4]), then
//             press the button / pulse CPU_RESETN (the TBKEY line records LED1 before and after). With
//             +DLY<i> = the run's token count the press falls on the end of the run (G_FIN / idle).
// Plusargs
//   +SW=<hex>       switches (all runs);  +SW<i>=<hex> switches set before key i (0-based)
//   +DLY<i>=<n>     token count for a mid-run key i (default 0);  +OFS<i>=<n> then n more clocks
//   +SCREEN=<file>  text RAM at every idle point (RTL only: title, 28 text rows in display order, status)
//   +FRAME=<file>   one VGA frame (640x480, binary PPM) captured from the pins at the end
//   +MAXCYC=<n>     abort after n clock cycles
// Output lines: GENTOK / GENRUN (lm_gen.sv, RTL only), TBTOK <ntok> <cycle> <token> (LED[15:4] changes),
//   TBKEY, TBRESET, TBIDLE <n> ... seg7 [<8 digits>] sig <pin signature> hash <run hash>, FRAME, XERR, TBDONE.
// Every input changes on the falling clock edge and every output is sampled there, so the RTL and the
// netlist see the same stimulus in the same cycle. `define PINONLY (implied by NETLIST): the testbench
// uses only the board pins (idle = LED1 low for QUIET cycles), so RTL and netlist runs are cycle-identical
// and their pin signatures (a hash of all outputs in every cycle) must be equal.
// `define XCHECK (4-state simulators): after the first reset, no X/Z on any output pin, nor on the
// text RAM write port while it is enabled (RTL).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`ifdef NETLIST
 `define PINONLY
`endif

module tb_nexys;
  localparam int CLK_MHZ = 50, DEB_CYC = 4, SLOW_CYC = 64, SEG_CYC = 8;
  localparam int PIX_DIV = CLK_MHZ / 25;
  localparam int QUIET   = 8192;                     // PINONLY idle: LED1 low this long (status, clear)

  logic clk = 1'b0;
  always #10 clk = ~clk;                             // 50 MHz

  logic        rstn = 1'b0;
  logic        btnc = 1'b0, btnu = 1'b0, btnd = 1'b0;
  logic [15:0] sw = '0;
  logic [15:0] led;
  logic        l16r, l16g, l16b, l17r, l17g, l17b;
  logic        ca, cb, cc, cd, ce, cf, cg, dp;
  logic [7:0]  an;
  logic [3:0]  vr, vg, vb;
  logic        hs, vs;

`ifdef NETLIST
  nexys_a7_top dut (
`else
  nexys_a7_top #(.CLK_MHZ(CLK_MHZ), .DEB_CYC(DEB_CYC), .SLOW_CYC(SLOW_CYC), .SEG_CYC(SEG_CYC)) dut (
`endif
    .CLK100MHZ(clk), .CPU_RESETN(rstn), .BTNC(btnc), .BTNU(btnu), .BTND(btnd), .SW(sw), .LED(led),
    .LED16_R(l16r), .LED16_G(l16g), .LED16_B(l16b), .LED17_R(l17r), .LED17_G(l17g), .LED17_B(l17b),
    .CA(ca), .CB(cb), .CC(cc), .CD(cd), .CE(ce), .CF(cf), .CG(cg), .DP(dp), .AN(an),
    .VGA_R(vr), .VGA_G(vg), .VGA_B(vb), .VGA_HS(hs), .VGA_VS(vs));

  // ---------------- board state seen on the pins ----------------
  logic        running;
  logic [11:0] ntok;
  logic [1:0]  chk;
  assign running = led[1];
  assign ntok    = led[15:4];
  assign chk     = led[2] ? 2'd1 : led[3] ? 2'd2 : 2'd0;
  logic [63:0] pins;
  assign pins = {12'd0, led, l16r, l16g, l16b, l17r, l17g, l17b, ca, cb, cc, cd, ce, cf, cg, dp, an,
                 vr, vg, vb, hs, vs};

  // generator state for the logs: the token just picked and the run's hash (RTL; netlist: the registers kept
  // under their RTL names, `define NETLIST_PROBES, set by scripts/nexys_netlist_sim.sh when they exist)
  logic [15:0] gen_tok;
  logic [31:0] gen_hash;
`ifndef NETLIST
  assign gen_tok  = dut.u_gen.nxt;
  assign gen_hash = dut.u_gen.hash;
`elsif NETLIST_PROBES
  assign gen_tok  = dut.\u_gen.nxt ;
  assign gen_hash = dut.\u_gen.hash ;
`else
  assign gen_tok  = '0;
  assign gen_hash = '0;
`endif

  longint cyc = 0, maxcyc = 64'd2_000_000_000;
  logic [31:0] psig = 32'h811C9DC5;                  // pin signature (FNV-1a over 64-bit samples)
  bit          sig_on = 1'b0;
  int          xerr = 0;
  logic [11:0] ntok_d = '0;
  always @(negedge clk) begin
    cyc <= cyc + 1;
    if (cyc > maxcyc) begin
      $display("TIMEOUT at cycle %0d", cyc);
`ifdef XCHECK
      $display("XCHECK unknown %0d", xerr);
`endif
      $finish;
    end
    if (sig_on) begin
      psig = (psig ^ pins[31:0]) * 32'h0100_0193;
      psig = (psig ^ pins[63:32]) * 32'h0100_0193;
      if (ntok !== ntok_d) $display("TBTOK %0d %0d %0d", ntok, cyc, gen_tok);
      ntok_d = ntok;
`ifdef XCHECK
      if ($isunknown(pins)) begin
        if (xerr < 10) $display("XERR cycle %0d pins %b", cyc, pins);
        xerr++;
      end
 `ifndef NETLIST
      if (dut.tw_we !== 1'b0 && ($isunknown(dut.tw_we) || $isunknown(dut.tw_addr) || $isunknown(dut.tw_data))) begin
        if (xerr < 10) $display("XERR cycle %0d text RAM write we %b addr %h data %h", cyc, dut.tw_we,
                                dut.tw_addr, dut.tw_data);
        xerr++;
      end
 `endif
`endif
    end
  end

`ifndef NETLIST
  // the generator's GENTOK lines stop here (lm_gen.sv prints them only while its reset is released)
  logic rst_q = 1'b0;
  always @(negedge clk) begin
    if (rst_q && !dut.rst_n) $display("GENRESET cycle %0d", cyc);
    rst_q <= dut.rst_n;
  end
`endif

  string scr_file, frame_file, keys;
  int    sfd = 0;

`ifndef PINONLY
  task automatic dump_screen(input int n);
    int p;
    if (sfd == 0) return;
    $fdisplay(sfd, "RUN %0d check %0d led16 %0d%0d%0d", n, dut.check, l16r, l16g, l16b);
    for (int r = 0; r < 30; r++) begin
      if (r == 0 || r == 29) p = r;
      else p = 1 + ((r - 1 + dut.top) % 28);
      for (int c = 0; c < 80; c++) $fwrite(sfd, "%c", dut.u_vga.tram[p * 80 + c]);
      $fwrite(sfd, "\n");
    end
  endtask
`endif

  // generator idle and no screen request pending (PINONLY: LED1 low for QUIET cycles)
  task automatic wait_idle();
    int quiet = 0;
`ifdef PINONLY
    while (quiet < QUIET) begin
      @(negedge clk);
      quiet = running ? 0 : quiet + 1;
    end
`else
    while (quiet < 64) begin
      @(negedge clk);
      if (!dut.running && !dut.u_gen.run_p && dut.u_gen.ws == 0 && !dut.u_gen.m_stat && !dut.u_gen.m_tok_v &&
          !dut.u_gen.m_clr)
        quiet++;
      else quiet = 0;
    end
`endif
  endtask

  // the 8 digits of the 7-segment display (blank = ' ', '-' = segment g only, '?' = unknown pattern)
  function automatic byte seg_char(input logic [6:0] s);   // {g, f, e, d, c, b, a}, active low
    case (s)
      7'b1000000: return "0";
      7'b1111001: return "1";
      7'b0100100: return "2";
      7'b0110000: return "3";
      7'b0011001: return "4";
      7'b0010010: return "5";
      7'b0000010: return "6";
      7'b1111000: return "7";
      7'b0000000: return "8";
      7'b0010000: return "9";
      7'b0111111: return "-";
      default:    return "?";
    endcase
  endfunction

  task automatic read_seg7(output string s);
    byte ch [8];
    for (int i = 0; i < 8; i++) ch[i] = " ";
    repeat (16 * SEG_CYC) begin                      // every digit twice
      @(negedge clk);
      for (int i = 0; i < 8; i++)
        if (an == ~(8'd1 << i)) ch[i] = seg_char({cg, cf, ce, cd, cc, cb, ca});
    end
    s = "";
    for (int i = 7; i >= 0; i--) s = $sformatf("%s%c", s, ch[i]);
  endtask

  int nidle = 0;
  task automatic idle_report();
    string s7;
    read_seg7(s7);
    $display("TBIDLE %0d cycle %0d check %0d ntok %0d sw %04h seg7 [%s] sig %08x hash %08x", nidle, cyc, chk, ntok, sw,
             s7, psig, gen_hash);
`ifndef PINONLY
    dump_screen(nidle);
`endif
    nidle++;
  endtask

  task automatic press(input byte k);
    @(negedge clk);
    case (k)
      "C", "c": btnc = 1'b1;
      "U", "u": btnu = 1'b1;
      default:  btnd = 1'b1;
    endcase
    repeat (12) @(negedge clk);
    {btnc, btnu, btnd} = 3'b000;
    repeat (12) @(negedge clk);
  endtask

  task automatic pulse_reset();
    @(negedge clk);
    $display("TBRESET cycle %0d running %0d ntok %0d", cyc, running, ntok);
    rstn = 1'b0;
    repeat (8) @(negedge clk);
    rstn = 1'b1;
  endtask

  // one VGA frame from the pins: the frame starts 35 lines after the falling edge of VSYNC (the sync is
  // lines 490..491); every pixel lasts PIX_DIV clocks. The syncs of all 800 x 525 pixel positions are
  // checked against the 640 x 480 @ 60 Hz timing.
  task automatic capture_frame();
    int fd, sync_err = 0;
    logic pv;
    fd = $fopen(frame_file, "wb");
    $fwrite(fd, "P6\n640 480\n255\n");
    pv = 1'b0;
    @(negedge clk);
    while (!(pv && !vs)) begin pv = vs; @(negedge clk); end
    repeat (PIX_DIV * 800 * 35) @(negedge clk);
    for (int y = 0; y < 525; y++)
      for (int x = 0; x < 800; x++) begin
        if (hs !== !(x >= 656 && x < 752) || vs !== !(y >= 490 && y < 492)) sync_err++;
        if (x < 640 && y < 480) $fwrite(fd, "%c%c%c", 8'(vr * 17), 8'(vg * 17), 8'(vb * 17));
        else if ({vr, vg, vb} !== 12'd0) sync_err++;   // blanking
        repeat (PIX_DIV) @(negedge clk);
      end
    $fclose(fd);
    $display("FRAME %s sync_errors %0d", frame_file, sync_err);
  endtask

  initial begin
    logic [15:0] v;
    int dly;
    if (!$value$plusargs("PRESS=%s", keys)) keys = "";
    if ($value$plusargs("SW=%h", v)) sw = v;
    if ($value$plusargs("MAXCYC=%d", maxcyc)) ;
`ifndef PINONLY
    if ($value$plusargs("SCREEN=%s", scr_file)) sfd = $fopen(scr_file, "w");
`endif
    repeat (8) @(negedge clk);
    rstn = 1'b1;
    repeat (64) @(negedge clk);                       // reset synchroniser + 32 cycles
    sig_on = 1'b1;
    // automatic first run after reset
    while (!running) @(negedge clk);
    for (int i = 0; i < keys.len(); i++) begin
      string key;
      byte   k;
      logic  r0;
      k = keys[i];
      if (k >= "A" && k <= "Z") begin
        wait_idle();
        idle_report();
      end else begin
        key = $sformatf("DLY%0d=%%d", i);
        if (!$value$plusargs(key, dly)) dly = 0;
        while (ntok < 12'(dly)) @(negedge clk);        // (the counter restarted at 0 with the run)
        key = $sformatf("OFS%0d=%%d", i);
        if ($value$plusargs(key, dly)) repeat (dly) @(negedge clk);
      end
      key = $sformatf("SW%0d=%%h", i);
      @(negedge clk);
      if ($value$plusargs(key, v)) sw = v;
      repeat (4) @(negedge clk);
      r0 = running;
      if (k == "R" || k == "r") pulse_reset();
      else press(k);
      $display("TBKEY %0d %c running %0d %0d ntok %0d cycle %0d", i, k, r0, running, ntok, cyc);
      if (k == "R" || k == "r") begin                 // reset: the board starts its automatic run again
        repeat (64) @(negedge clk);
        while (!running) @(negedge clk);
      end else if (k == "C" || k == "U")
        while (!running) @(negedge clk);
    end
    wait_idle();
    idle_report();
    if (sfd != 0) $fclose(sfd);
    if ($value$plusargs("FRAME=%s", frame_file)) capture_frame();
`ifdef XCHECK
    $display("XCHECK unknown %0d", xerr);
`endif
    $display("TBDONE cycles %0d sig %08x", cyc, psig);
    $finish;
  end
endmodule
