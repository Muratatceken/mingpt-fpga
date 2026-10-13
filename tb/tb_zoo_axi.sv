// ---------------------------------------------------------------------------
// tb_zoo_axi.sv -- system test of the AXI4-Lite engine (rtl/zoo/zoo_axi.sv),
// doing exactly what the PYNQ driver does on the board (board/pynq_z2/zoo.py):
//   1. load a model through the LD_* registers (one 256-bit word per LD_GO),
//   2. run every step of the golden vectors: TOK, POS, CTRL.start, poll STATUS,
//      read every logit through the LOGIT window and check it, check ARGMAX,
//   3. closed-loop greedy generation through the ARGMAX register (if gen.txt).
// Plusargs: +MODELDIR=<dir> (default MEM_PATH)  +NSEQ=<n>
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_zoo_axi;
  import zoo_pkg::*;

  logic aclk = 1'b0, aresetn = 1'b0;
  always #5 aclk = ~aclk;

  logic [15:0] awaddr = '0, araddr = '0;
  logic        awvalid = 1'b0, wvalid = 1'b0, bready = 1'b0, arvalid = 1'b0, rready = 1'b0;
  logic [31:0] wdata = '0;
  logic        awready, wready, bvalid, arready, rvalid;
  logic [1:0]  bresp, rresp;
  logic [31:0] rdata;

  zoo_axi dut (
    .aclk, .aresetn,
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));

  // ---------------- AXI4-Lite master ----------------
  task automatic wr(input logic [15:0] a, input logic [31:0] d);
    bit aw_done = 0, w_done = 0;
    @(posedge aclk);
    awaddr <= a; awvalid <= 1'b1; wdata <= d; wvalid <= 1'b1;
    while (!(aw_done && w_done)) begin
      @(posedge aclk);
      if (awvalid && awready) begin aw_done = 1; awvalid <= 1'b0; end
      if (wvalid && wready)   begin w_done = 1;  wvalid  <= 1'b0; end
    end
    bready <= 1'b1;
    do @(posedge aclk); while (!bvalid);
    bready <= 1'b0;
  endtask

  task automatic rd(input logic [15:0] a, output logic [31:0] d);
    @(posedge aclk);
    araddr <= a; arvalid <= 1'b1;
    do @(posedge aclk); while (!arready);
    arvalid <= 1'b0;
    rready  <= 1'b1;
    while (!rvalid) @(posedge aclk);
    d = rdata;
    @(posedge aclk);
    rready <= 1'b0;
  endtask

  // ---------------- model loading ----------------
  logic [255:0]     m_prog [PROG_DEPTH];
  logic [8*P-1:0]   m_w    [W_DEPTH];
  logic [63:0]      m_prm  [PRM_DEPTH];
  logic [TAB_W-1:0] m_tab  [TAB_DEPTH];
  logic [7:0]       m_lut  [LUT_DEPTH];
  string mdir;
  longint ld_words = 0;

  task automatic ld_mem(input int sel, input int addr, input logic [255:0] data, input int nwords32);
    for (int i = 0; i < nwords32; i++) wr(16'h0040 + 16'(4 * i), data[32 * i +: 32]);
    wr(16'h0034, 32'(addr));
    wr(16'h0038, 32'd1);
    ld_words++;
  endtask

  task automatic load_model();
    int sfd = 0, r = 0, n_prog = PROG_DEPTH, n_w = W_DEPTH, n_prm = PRM_DEPTH, n_tab = TAB_DEPTH, n_lut = LUT_DEPTH;
    sfd = $fopen({mdir, "/sizes.txt"}, "r");
    if (sfd != 0) begin r = $fscanf(sfd, "%d %d %d %d %d", n_prog, n_w, n_prm, n_tab, n_lut); $fclose(sfd); end
    $readmemh({mdir, "/prog.mem"}, m_prog, 0, n_prog - 1);
    $readmemh({mdir, "/weights.mem"}, m_w, 0, n_w - 1);
    $readmemh({mdir, "/params.mem"}, m_prm, 0, n_prm - 1);
    $readmemh({mdir, "/tables.mem"}, m_tab, 0, n_tab - 1);
    $readmemh({mdir, "/luts.mem"}, m_lut, 0, n_lut - 1);
    wr(16'h0030, 32'd0); for (int a = 0; a < n_prog; a++) ld_mem(0, a, m_prog[a], 8);
    wr(16'h0030, 32'd1); for (int a = 0; a < n_w; a++)    ld_mem(1, a, 256'(m_w[a]), (8 * P + 31) / 32);
    wr(16'h0030, 32'd2); for (int a = 0; a < n_prm; a++)  ld_mem(2, a, 256'(m_prm[a]), 2);
    wr(16'h0030, 32'd3); for (int a = 0; a < n_tab; a++)  ld_mem(3, a, 256'(m_tab[a]), 1);
    wr(16'h0030, 32'd4); for (int a = 0; a < n_lut; a++)  ld_mem(4, a, 256'(m_lut[a]), 1);
    $display("loaded %s over AXI: %0d words", mdir, ld_words);
  endtask

  // ---------------- one step through the registers ----------------
  logic [31:0] st, am, cyc;
  task automatic step(input int tk, input int tt);
    wr(16'h0004, 32'(tk));
    wr(16'h0008, 32'(tt));
    wr(16'h0000, 32'd1);
    do rd(16'h000C, st); while (!st[1]);
    rd(16'h0014, am);
    rd(16'h0010, cyc);
  endtask

  int fd = 0, rc = 0, nseq = 0, nv = 0, len = 0, tok = 0, maxseq = 0, best = 0, bi = 0;
  int n = 0, t0 = 0, ok = 0, e = 0;
  int gold [V];
  initial for (int k = 0; k < V; k++) gold[k] = 0;
  logic [31:0] lg;
  longint errors = 0, steps = 0, am_err = 0, tot_cyc = 0;

  initial begin
    if (!$value$plusargs("MODELDIR=%s", mdir)) mdir = MEM_PATH;
    if (!$value$plusargs("NSEQ=%d", maxseq)) maxseq = 1 << 30;
    repeat (5) @(posedge aclk);
    aresetn <= 1'b1;
    rd(16'h001C, lg);
    if (lg != 32'h5A4F_4F01) $fatal(1, "bad ID %h", lg);
    load_model();

    fd = $fopen({mdir, "/tv.txt"}, "r");
    rc = $fscanf(fd, "%d %d", nseq, nv);
    wr(16'h0028, 32'(nv));                       // model vocabulary (argmax range)
    if (nseq > maxseq) nseq = maxseq;
    for (int s = 0; s < nseq; s++) begin
      rc = $fscanf(fd, "%d", len);
      for (int tt = 0; tt < len; tt++) begin
        rc = $fscanf(fd, "%d", tok);
        best = -(1 << 30); bi = 0;
        for (int k = 0; k < nv; k++) begin
          rc = $fscanf(fd, "%d", gold[k]);
          if (gold[k] > best) begin best = gold[k]; bi = k; end
        end
        step(tok, tt);
        tot_cyc += cyc;
        if (int'(am) != bi) am_err++;
        for (int k = 0; k < nv; k++) begin
          rd(16'h1000 + 16'(4 * k), lg);
          if (int'(lg) != gold[k]) begin
            if (errors < 5) $display("MISMATCH seq %0d t %0d logit %0d: %0d gold %0d", s, tt, k, int'(lg), gold[k]);
            errors++;
          end
        end
        steps++;
      end
    end
    $fclose(fd);

    // closed-loop greedy generation through the ARGMAX register
    fd = $fopen({mdir, "/gen.txt"}, "r");
    if (fd != 0) begin
      rc = $fscanf(fd, "%d %d", n, t0);
      tok = t0;
      for (int tt = 0; tt < n; tt++) begin
        step(tok, tt);
        rc = $fscanf(fd, "%d", e);
        if (int'(am) == e) ok++;
        tok = int'(am);
      end
      $fclose(fd);
      $display("AXIGEN steps %0d match %0d", n, ok);
      if (ok != n) errors++;
    end

    $display("AXI %s: %0d steps, %0d logit errors, %0d argmax errors, %0d cycles/step (incl. argmax scan)",
             mdir, steps, errors, am_err, steps ? tot_cyc / steps : 0);
    if (errors == 0 && am_err == 0) $display("PASS: AXI system bit-exact");
    else $display("FAIL");
    $finish;
  end
endmodule
