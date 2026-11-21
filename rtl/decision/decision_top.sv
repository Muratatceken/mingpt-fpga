`timescale 1ns/1ps
// Decision pilot top (WP4.3): stand-alone on the FPGA, no host protocol.
//   observation ROM (recorded telemetry codes, replayed one step per decision)
//     -> dec_tokenizer -> policy (POLICY 0: integer LM/GRU on zoo_core; 1: rule; 2: FSM)
//     -> dec_guard (mode register, watchdog) -> applied action -> trace RAM + running hash
// The engine sees exactly the training token stream: BOS, then per step V I T MODE ACT (MODE = mode before the
// decision, ACT = applied action); the decision logits are read after the MODE token (masked argmax over 0..3).
// ROM word (64 bit): [15:0] v [31:16] i [47:32] t  [48] vv [49] iv [50] tv  [52:51] ground cmd  [53] suppress
//   (suppress = the policy's proposal is withheld for this step -> guard timeout path)
// Trace word (16 bit): [1:0] proposal [3:2] applied [6:4] reason [8:7] mode before [10:9] mode after [11] valid
module decision_top
  import dec_pkg::*;
#(
  parameter int    POLICY = 0,                    // 0 engine (LM/GRU), 1 rule, 2 FSM
  parameter int    NWORDS = 4096,                 // observation ROM depth (steps)
  parameter string OBS_FILE = "build/decision/obs.mem",
  parameter int    WDT = 250000                   // decision deadline in cycles (10 ms at 25 MHz)
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,                      // run one episode
  input  logic [15:0] base,                       // first ROM word of the episode
  input  logic [7:0]  nsteps,                     // steps in the episode (engine: 1 + 5*nsteps <= T)
  output logic        busy,
  output logic        done,                       // one-cycle pulse at the end of the episode
  output logic [31:0] hash,                       // running hash of the trace (shown on LEDs / 7-segment)
  output logic        tr_we,                      // trace stream (also written to the trace RAM)
  output logic [15:0] tr_word,
  output logic [31:0] dec_cycles                  // cycles from observation to applied action (last step)
);
  // ---------------- observation ROM (synchronous)
  logic [63:0] rom [NWORDS];
  initial $readmemh(OBS_FILE, rom);
  logic [15:0] ra;
  logic [63:0] rd;
  always_ff @(posedge clk) rd <= rom[ra];

  // ---------------- latched observation, tokens
  logic signed [15:0] ov, oi, ot;
  logic               ovv, oiv, otv, osup;
  logic [1:0]         ogc;
  logic [5:0]         tok_v, tok_i, tok_t;
  dec_tokenizer u_tok (.v(ov), .i(oi), .t(ot), .vv(ovv), .iv(oiv), .tv(otv), .tok_v(tok_v), .tok_i(tok_i),
                       .tok_t(tok_t));

  // ---------------- guard
  logic       g_start, g_pv, g_done, g_busy;
  logic [1:0] g_prop, g_applied, g_mode;
  logic [2:0] g_reason;
  dec_guard #(.WDT(WDT)) u_guard (.clk(clk), .rst_n(rst_n), .clear(fsm_clear), .start(g_start), .prop_valid(g_pv), .prop(g_prop),
                                  .v(ov), .i(oi), .t(ot), .vv(ovv), .iv(oiv), .tv(otv), .gc(ogc), .done(g_done),
                                  .applied(g_applied), .mode(g_mode), .reason(g_reason), .busy(g_busy));

  // ---------------- policies
  logic [1:0] rule_a, fsm_a, prev_act;
  logic       fsm_step, fsm_clear;
  dec_rule u_rule (.tok_v(tok_v), .tok_i(tok_i), .tok_t(tok_t), .act(rule_a));
  dec_fsm  u_fsm  (.clk(clk), .rst_n(rst_n), .clear(fsm_clear), .step(fsm_step), .tok_v(tok_v), .tok_i(tok_i),
                   .tok_t(tok_t), .prev_act(prev_act), .act(fsm_a));

  // ---------------- engine (POLICY 0)
  logic                 c_start, c_busy, c_done;
  logic [5:0]           c_tok;
  logic [7:0]           c_pos;
  logic [7:0]           l_addr;
  logic signed [15:0]   l_data;
  logic signed [15:0]   lg [4];
  logic [1:0]           lm_a;
  generate
    if (POLICY == 0) begin : g_eng
      logic [zoo_pkg::LA_W-1:0] la;
      logic signed [15:0]       ld;
      assign la = zoo_pkg::LA_W'(l_addr);
      zoo_core #(.SPECIALIZE(1)) u_core (
        .clk(clk), .rst_n(rst_n), .start(c_start), .tok_in(zoo_pkg::TOK_W'(c_tok)), .pos(zoo_pkg::POS_W'(c_pos)),
        .busy(c_busy), .done(c_done), .l_addr(la), .l_data(ld), .dbg_op(),
        .ld_we(1'b0), .ld_sel(3'd0), .ld_addr(20'd0), .ld_data(256'd0));
      assign l_data = ld;
    end else begin : g_noeng
      assign c_busy = 1'b0;
      assign c_done = 1'b1;
      assign l_data = '0;
    end
  endgenerate
  dec_action_select u_sel (.l0(lg[0]), .l1(lg[1]), .l2(lg[2]), .l3(lg[3]), .act(lm_a));

  // ---------------- sequencer
  typedef enum logic [3:0] {D_IDLE, D_BOS, D_BOSW, D_ROM, D_LATCH, D_FEED, D_FEEDW, D_LOGIT, D_PROP, D_GWAIT,
                            D_ACT, D_ACTW, D_NEXT, D_FIN} dst_t;
  dst_t        st;
  logic [7:0]  k, nst;
  logic [1:0]  fi;                                // which of V I T MODE is being fed
  logic [2:0]  li;                                // logit read index
  logic [1:0]  prop_r, mode_b;
  logic [31:0] cyc;
  logic        prop_sent;

  always_comb begin
    case (fi)
      2'd0: c_tok = tok_v;
      2'd1: c_tok = tok_i;
      2'd2: c_tok = tok_t;
      default: c_tok = MODE_BASE + 6'(g_mode);
    endcase
    if (st == D_BOS || st == D_BOSW) c_tok = BOS;
    if (st == D_ACT || st == D_ACTW) c_tok = ACT_BASE + 6'(g_applied);
    l_addr  = 8'(li);
    g_prop  = (POLICY == 0) ? lm_a : (POLICY == 1) ? rule_a : fsm_a;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st <= D_IDLE; busy <= 1'b0; done <= 1'b0; hash <= 32'h811C9DC5; tr_we <= 1'b0; tr_word <= '0;
      c_start <= 1'b0; c_pos <= '0; k <= '0; nst <= '0; fi <= '0; li <= '0; ra <= '0;
      g_start <= 1'b0; g_pv <= 1'b0; fsm_step <= 1'b0; fsm_clear <= 1'b0; prev_act <= A_NORMAL;
      prop_r <= '0; mode_b <= '0; cyc <= '0; dec_cycles <= '0; prop_sent <= 1'b0;
      {ov, oi, ot} <= '0; {ovv, oiv, otv, osup} <= '0; ogc <= '0;
      for (int a = 0; a < 4; a++) lg[a] <= '0;
    end else begin
      done <= 1'b0; tr_we <= 1'b0; c_start <= 1'b0; g_start <= 1'b0; g_pv <= 1'b0; fsm_step <= 1'b0;
      fsm_clear <= 1'b0;
      cyc <= cyc + 32'd1;
      case (st)
        D_IDLE: if (start) begin
          busy <= 1'b1; k <= '0; nst <= nsteps; c_pos <= '0; ra <= base; hash <= 32'h811C9DC5;
          prev_act <= A_NORMAL; fsm_clear <= 1'b1;
          if (POLICY == 0) begin c_start <= 1'b1; st <= D_BOSW; end else st <= D_ROM;
        end
        D_BOSW: if (c_done && !c_start) begin c_pos <= 8'd1; st <= D_ROM; end
        D_ROM:  st <= D_LATCH;                                         // ROM read latency
        D_LATCH: begin
          ov <= rd[15:0]; oi <= rd[31:16]; ot <= rd[47:32];
          ovv <= rd[48]; oiv <= rd[49]; otv <= rd[50]; ogc <= rd[52:51]; osup <= rd[53];
          mode_b <= g_mode; cyc <= '0; prop_sent <= 1'b0;
          g_start <= 1'b1;                                             // guard waits for the proposal
          fi <= '0;
          st <= (POLICY == 0) ? D_FEED : D_PROP;
        end
        D_FEED: begin c_start <= 1'b1; st <= D_FEEDW; end
        D_FEEDW: if (c_done && !c_start) begin
          c_pos <= c_pos + 8'd1;
          if (fi == 2'd3) begin li <= '0; st <= D_LOGIT; end
          else begin fi <= fi + 2'd1; st <= D_FEED; end
        end
        D_LOGIT: begin                                                 // async logit port: one per cycle
          lg[li[1:0]] <= l_data;
          if (li == 3'd3) st <= D_PROP; else li <= li + 3'd1;
        end
        D_PROP: begin                                                  // proposal ready (combinational)
          prop_r <= g_prop;
          if (!osup) g_pv <= 1'b1;
          prop_sent <= !osup;
          fsm_step <= 1'b1;                                            // FSM registers advance every step
          st <= D_GWAIT;
        end
        D_GWAIT: if (g_done) begin
          dec_cycles <= cyc;
          prev_act <= g_applied;
          if (POLICY == 0) begin c_start <= 1'b1; st <= D_ACTW; end else st <= D_NEXT;
        end
        D_ACTW: if (c_done && !c_start) begin c_pos <= c_pos + 8'd1; st <= D_NEXT; end
        D_NEXT: begin
          tr_we   <= 1'b1;
          tr_word <= {4'd0, prop_sent, g_mode, mode_b, g_reason, g_applied, prop_r};
          hash    <= (hash ^ 32'({4'd0, prop_sent, g_mode, mode_b, g_reason, g_applied, prop_r})) * 32'h01000193;
          k <= k + 8'd1;
          ra <= ra + 16'd1;
          if (k + 8'd1 == nst) st <= D_FIN; else st <= D_ROM;
        end
        D_FIN: begin busy <= 1'b0; done <= 1'b1; st <= D_IDLE; end
        default: st <= D_IDLE;
      endcase
    end
  end
endmodule
