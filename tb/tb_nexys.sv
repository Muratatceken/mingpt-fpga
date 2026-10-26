// ---------------------------------------------------------------------------
// tb_nexys.sv -- simulation of the whole Nexys A7-100T design (board/nexys_a7/nexys_a7_top.sv) as it
// runs on the board: reset, the automatic first run, button presses, VGA output.
// Compiled with -DSIM (the MMCM is bypassed; the testbench clock is the core clock).
//
// Plusargs
//   +PRESS=<keys>   buttons pressed after the automatic run, one run each: C = BTNC (new run),
//                   U = BTNU (reseed + run), D = BTND (clear only)
//   +SW=<hex>       switches (all runs);  +SW<i>=<hex> switches before key i (0-based)
//   +SCREEN=<file>  text screen after every run (title, 28 text rows in display order, status)
//   +FRAME=<file>   one VGA frame (640x480, binary PPM) captured from the pins after the last run
//   +MAXCYC=<n>     abort after n clock cycles
// The generator prints GENTOK / GENRUN lines (lm_gen.sv), compared by sw/zoo/board_sim.py.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_nexys;
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

  nexys_a7_top #(.CLK_MHZ(50), .DEB_CYC(4), .SLOW_CYC(64), .SEG_CYC(8)) dut (
    .CLK100MHZ(clk), .CPU_RESETN(rstn), .BTNC(btnc), .BTNU(btnu), .BTND(btnd), .SW(sw), .LED(led),
    .LED16_R(l16r), .LED16_G(l16g), .LED16_B(l16b), .LED17_R(l17r), .LED17_G(l17g), .LED17_B(l17b),
    .CA(ca), .CB(cb), .CC(cc), .CD(cd), .CE(ce), .CF(cf), .CG(cg), .DP(dp), .AN(an),
    .VGA_R(vr), .VGA_G(vg), .VGA_B(vb), .VGA_HS(hs), .VGA_VS(vs));

  longint cyc = 0, maxcyc = 64'd2_000_000_000;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (cyc > maxcyc) begin $display("TIMEOUT at cycle %0d", cyc); $finish; end
  end

  string scr_file, frame_file, keys;
  int    sfd = 0;

  task automatic dump_screen(input int run);
    int r, p;
    if (sfd == 0) return;
    $fdisplay(sfd, "RUN %0d check %0d led16 %0d%0d%0d", run, dut.check, l16r, l16g, l16b);
    for (r = 0; r < 30; r++) begin
      if (r == 0 || r == 29) p = r;
      else p = 1 + ((r - 1 + dut.top) % 28);
      for (int c = 0; c < 80; c++) $fwrite(sfd, "%c", dut.u_vga.tram[p * 80 + c]);
      $fwrite(sfd, "\n");
    end
  endtask

  task automatic wait_screen_idle();
    int quiet = 0;
    while (quiet < 64) begin                          // generator idle, no screen request pending
      @(posedge clk);
      if (!dut.running && dut.u_gen.ws == 0 && !dut.u_gen.m_stat && !dut.u_gen.m_tok_v && !dut.u_gen.m_clr)
        quiet++;
      else quiet = 0;
    end
  endtask

  task automatic press(input byte k);
    case (k)
      "C": btnc = 1'b1;
      "U": btnu = 1'b1;
      default: btnd = 1'b1;
    endcase
    repeat (12) @(posedge clk);
    {btnc, btnu, btnd} = 3'b000;
    repeat (12) @(posedge clk);
  endtask

  task automatic capture_frame();
    int fd, n;
    fd = $fopen(frame_file, "wb");
    $fwrite(fd, "P6\n640 480\n255\n");
    do @(posedge clk); while (!(dut.px_v && dut.px_x == 0 && dut.px_y == 0));
    n = 0;
    while (n < 640 * 480) begin
      if (dut.px_v) begin
        $fwrite(fd, "%c%c%c", 8'(vr * 17), 8'(vg * 17), 8'(vb * 17));
        n++;
      end
      @(posedge clk);
    end
    $fclose(fd);
  endtask

  initial begin
    logic [15:0] v;
    if (!$value$plusargs("PRESS=%s", keys)) keys = "";
    if ($value$plusargs("SW=%h", v)) sw = v;
    if ($value$plusargs("MAXCYC=%d", maxcyc)) ;
    if ($value$plusargs("SCREEN=%s", scr_file)) sfd = $fopen(scr_file, "w");
    repeat (8) @(posedge clk);
    rstn = 1'b1;
    // automatic first run after reset
    wait (dut.running);
    wait_screen_idle();
    dump_screen(0);
    for (int i = 0; i < keys.len(); i++) begin
      string key;
      key = $sformatf("SW%0d=%%h", i);
      if ($value$plusargs(key, v)) sw = v;
      repeat (4) @(posedge clk);
      press(keys[i]);
      if (keys[i] != "D") wait (dut.running);
      wait_screen_idle();
      dump_screen(i + 1);
    end
    if (sfd != 0) $fclose(sfd);
    if ($value$plusargs("FRAME=%s", frame_file)) capture_frame();
    $display("TBDONE cycles %0d", cyc);
    $finish;
  end
endmodule
