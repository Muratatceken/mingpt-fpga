`timescale 1ns/1ps
// Tokenizer and the two non-learned baselines (WP3.1-3.3), bit-equivalent to sw/decision/{tokenizer,policies}.py
//   dec_tokenizer   raw codes + valid flags -> V I T tokens (bin = number of edges <= code)
//   dec_rule        latest observation only
//   dec_fsm         history-aware FSM: OC event count, load-switch response tracking, sticky suspect flags
module dec_tokenizer
  import dec_pkg::*;
(
  input  logic signed [15:0] v, i, t,
  input  logic               vv, iv, tv,
  output logic [5:0]         tok_v, tok_i, tok_t
);
  assign tok_v = vv ? V_BASE + 6'(bin_v(v)) : V_INV;
  assign tok_i = iv ? I_BASE + 6'(bin_i(i)) : I_INV;
  assign tok_t = tv ? T_BASE + 6'(bin_t(t)) : T_INV;
endmodule

// token -> signed bin (-1 for an invalid channel), as tokenizer.bins_from_tokens
module dec_bins
  import dec_pkg::*;
(
  input  logic [5:0]        tok_v, tok_i, tok_t,
  output logic signed [6:0] vb, ib, tb,
  output logic              inv
);
  assign vb  = (tok_v == V_INV) ? -7'sd1 : $signed({1'b0, tok_v - V_BASE});
  assign ib  = (tok_i == I_INV) ? -7'sd1 : $signed({1'b0, tok_i - I_BASE});
  assign tb  = (tok_t == T_INV) ? -7'sd1 : $signed({1'b0, tok_t - T_BASE});
  assign inv = (tok_v == V_INV) || (tok_i == I_INV) || (tok_t == T_INV);
endmodule

module dec_rule
  import dec_pkg::*;
(
  input  logic [5:0] tok_v, tok_i, tok_t,
  output logic [1:0] act
);
  logic signed [6:0] vb, ib, tb;
  logic              inv;
  dec_bins u_b (.tok_v(tok_v), .tok_i(tok_i), .tok_t(tok_t), .vb(vb), .ib(ib), .tb(tb), .inv(inv));
  always_comb begin
    act = A_NORMAL;
    if (int'(vb) <= RULE_V_RESTR) act = A_KISITLI_MOD;
    if (int'(ib) >= RULE_I_OC) act = A_YUKU_KAPAT;
    if (int'(vb) <= RULE_V_SAFE || int'(tb) >= RULE_T_SAFE) act = A_GUVENLI_MOD;
    if (inv) act = A_KISITLI_MOD;
  end
endmodule

module dec_fsm
  import dec_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,
  input  logic       clear,                 // episode reset
  input  logic       step,                  // commit this step's registers
  input  logic [5:0] tok_v, tok_i, tok_t,
  input  logic [1:0] prev_act,              // action applied at the previous step (NORMAL before step 0)
  output logic [1:0] act
);
  // registers (42 bits): see policies.FSMPolicy
  logic [1:0]        ocn, swt;
  logic              ocp, swdir, cresp, rose, sens, batt, prev_pl, started;
  logic [1:0]        vcnt, n_vcnt;
  logic signed [6:0] vref, iref, vpeak, prev_ib, prev_vb, vmin_on;
  // next state
  logic [1:0]        n_ocn, n_swt;
  logic              n_ocp, n_swdir, n_vresp, n_cresp, n_rose, n_sens, n_batt;
  logic signed [6:0] n_vref, n_iref, n_vpeak, n_vmin;
  logic signed [6:0] vb, ib, tb, p_ib, p_vb, d_i, d_v;
  logic              inv, pl, p_pl, oc_now, oc_rise, jump, sw, active, new_rose, done_w, low_v, probing;
  logic [1:0]        a;
  dec_bins u_b (.tok_v(tok_v), .tok_i(tok_i), .tok_t(tok_t), .vb(vb), .ib(ib), .tb(tb), .inv(inv));

  function automatic logic signed [6:0] smax(input logic signed [6:0] x, input logic signed [6:0] y);
    return (x > y) ? x : y;
  endfunction
  function automatic logic signed [6:0] smin(input logic signed [6:0] x, input logic signed [6:0] y);
    return (x < y) ? x : y;
  endfunction

  always_comb begin
    pl   = (prev_act == A_NORMAL);
    p_pl = started ? prev_pl : pl;
    p_ib = started ? prev_ib : ib;
    p_vb = started ? prev_vb : vb;
    // OC events: I >= 2000 mA, or a current jump of >= 2 bins while the payload stays on
    jump    = started && pl && p_pl && !inv && (ib >= p_ib + 7'sd2);
    oc_now  = !inv && (int'(ib) >= RULE_I_OC || jump);
    oc_rise = jump || (!inv && int'(ib) >= RULE_I_OC && int'(p_ib) < RULE_I_OC);
    n_ocn   = (oc_rise && ocn != 2'd3) ? ocn + 2'd1 : ocn;
    n_ocp   = oc_now ? 1'b1 : ocp;
    // payload switch response tracking
    sw      = started && (pl != p_pl);
    n_swt   = sw ? 2'(SW_WIN) : swt;
    n_swdir = sw ? !pl : swdir;
    n_vref  = sw ? p_vb : vref;
    n_iref  = sw ? p_ib : iref;
    n_vcnt  = sw ? 2'd0 : vcnt;
    n_cresp = sw ? 1'b0 : cresp;
    active  = (n_swt != 2'd0) && !inv;
    d_i     = ib - n_iref;
    d_v     = vb - n_vref;
    if (active && (d_i >= 7'sd2 || d_i <= -7'sd2)) n_cresp = 1'b1;
    if (active && (d_v >= 7'sd1 || d_v <= -7'sd1) && n_vcnt != 2'd3) n_vcnt = n_vcnt + 2'd1;
    n_vresp = (n_vcnt >= 2'd2);
    new_rose = active && n_swdir && (vb >= n_vref + 7'sd2);
    n_rose   = new_rose ? 1'b1 : rose;
    n_vpeak  = (sw && !pl) ? smax(vb, 7'sd0) : vpeak;
    if (!pl && !inv) n_vpeak = smax(n_vpeak, vb);
    done_w   = (n_swt == 2'd1);
    if (n_swt != 2'd0) n_swt = n_swt - 2'd1;
    n_sens   = (done_w && n_cresp && !n_vresp) ? 1'b1 : sens;
    n_batt   = (n_rose && !pl && !inv && int'(vb) <= RULE_V_RESTR && vb < n_vpeak) ? 1'b1 : batt;
    n_vmin   = (pl && !inv) ? smin(vmin_on, vb) : vmin_on;
    if (int'(n_vmin) <= RULE_V_SAFE && !pl && !inv && int'(vb) <= RULE_V_RESTR) n_batt = 1'b1;
    // decision
    low_v   = !inv && int'(vb) <= RULE_V_RESTR;
    probing = !pl && (n_swt != 2'd0);
    a = A_NORMAL;
    if (low_v && pl) a = A_YUKU_KAPAT;
    if (low_v && probing) a = A_YUKU_KAPAT;
    if (low_v && !pl && n_swt == 2'd0) a = A_KISITLI_MOD;
    if (n_ocp && !oc_now) a = (n_ocn >= 2'd2) ? A_KISITLI_MOD : A_NORMAL;
    if (oc_now) a = (n_ocn >= 2'd2) ? A_KISITLI_MOD : A_YUKU_KAPAT;
    if (n_sens && a < A_KISITLI_MOD) a = A_KISITLI_MOD;
    if (n_batt) a = A_GUVENLI_MOD;
    if (!inv && !n_sens && int'(vb) <= RULE_V_SAFE) a = A_GUVENLI_MOD;
    if (!inv && int'(tb) >= RULE_T_SAFE) a = A_GUVENLI_MOD;
    if (inv && a < A_KISITLI_MOD) a = A_KISITLI_MOD;
    if (n_ocp && !oc_now) n_ocp = 1'b0;
    act = a;
  end

  always_ff @(posedge clk) begin
    if (!rst_n || clear) begin
      {ocn, swt, vcnt, ocp, swdir, cresp, rose, sens, batt, prev_pl, started} <= '0;
      {vref, iref, vpeak, prev_ib, prev_vb} <= '0;
      vmin_on <= 7'sd31;
    end else if (step) begin
      ocn <= n_ocn; ocp <= n_ocp; swt <= n_swt; swdir <= n_swdir; vref <= n_vref; iref <= n_iref;
      vcnt <= n_vcnt; cresp <= n_cresp; rose <= n_rose; vpeak <= n_vpeak; sens <= n_sens; batt <= n_batt;
      vmin_on <= n_vmin; prev_pl <= pl; started <= 1'b1;
      prev_ib <= inv ? p_ib : ib;
      prev_vb <= inv ? p_vb : vb;
    end
  end
endmodule

// masked argmax over the 4 action logits (ties -> more severe action), as train_policy.decide
module dec_action_select (
  input  logic signed [15:0] l0, l1, l2, l3,
  output logic [1:0]         act
);
  logic signed [15:0] m;
  always_comb begin
    act = 2'd0; m = l0;
    if (l1 >= m) begin act = 2'd1; m = l1; end
    if (l2 >= m) begin act = 2'd2; m = l2; end
    if (l3 >= m) begin act = 2'd3; m = l3; end
  end
endmodule
