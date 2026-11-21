`timescale 1ns/1ps
// Independent safety guard (WP4.4).  Golden model: sw/decision/guard.py (guard_step).
//   dec_guard_comb : one decision (combinational), exhaustively testable against Python
//   dec_guard      : owns the mode register and the invalid-sample counter; waits for a proposal after `start`
//                    and applies the timeout fallback when none arrives within WDT cycles (timeout wins over a
//                    proposal arriving in the same cycle).  It does not depend on the policy's internal state.
// Priority: FORCE_SAFE > hard limit V/T > RESUME > {timeout fallback, hard limit I, persistent invalid, table}
module dec_guard_comb
  import dec_pkg::*;
(
  input  logic [1:0]         prop,
  input  logic               prop_valid,
  input  logic [1:0]         mode_in,
  input  logic signed [15:0] v, i, t,
  input  logic               vv, iv, tv,
  input  logic [1:0]         gc,
  input  logic [2:0]         inv_in,
  output logic [1:0]         applied,
  output logic [1:0]         mode_new,
  output logic [2:0]         reason,
  output logic [2:0]         inv_out
);
  logic [1:0] mode, a, rep;
  logic       any_inv, inv_persist, lim_vt, lim_i;
  always_comb begin
    mode        = (mode_in > M_GUVENLI) ? M_GUVENLI : mode_in;      // illegal code -> safe mode
    any_inv     = !(vv && iv && tv);
    inv_out     = any_inv ? ((inv_in == 3'(INV_SAT)) ? inv_in : inv_in + 3'd1) : 3'd0;
    inv_persist = int'(inv_out) >= INV_PERSIST;
    lim_vt      = (vv && int'(v) < LIM_V) || (tv && int'(t) >= LIM_T);
    lim_i       = iv && int'(i) >= LIM_I;
    a      = prop_valid ? prop : ((mode == M_GUVENLI) ? A_GUVENLI_MOD : A_KISITLI_MOD);
    reason = prop_valid ? R_OK : R_TIMEOUT;
    if (lim_i && a < A_YUKU_KAPAT)        begin a = A_YUKU_KAPAT;  reason = R_LIM_I;   end
    if (inv_persist && a < A_KISITLI_MOD) begin a = A_KISITLI_MOD; reason = R_INVALID; end
    rep = tt(mode, a);
    if (rep != a)                         begin a = rep;           reason = R_DENY;    end
    if (gc == GC_RESUME && mode == M_GUVENLI && !lim_vt && !lim_i)
                                          begin a = A_KISITLI_MOD; reason = R_RESUME;  end
    if (lim_vt)                           begin a = A_GUVENLI_MOD; reason = R_LIM_VT;  end
    if (gc == GC_FORCE_SAFE)              begin a = A_GUVENLI_MOD; reason = R_FORCE;   end
    applied  = a;
    mode_new = mode_after(a, mode);
  end
endmodule

module dec_guard
  import dec_pkg::*;
#(
  parameter int WDT = 250000                      // decision deadline in cycles (10 ms at 25 MHz)
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,              // a new step: observation inputs are stable from here
  input  logic               prop_valid,         // proposal from the policy (one-cycle pulse is enough)
  input  logic [1:0]         prop,
  input  logic signed [15:0] v, i, t,
  input  logic               vv, iv, tv,
  input  logic [1:0]         gc,
  output logic               done,               // one-cycle pulse: applied/mode/reason valid
  output logic [1:0]         applied,
  output logic [1:0]         mode,               // mode register (before `done`: the current mode)
  output logic [2:0]         reason,
  output logic               busy
);
  logic [2:0]  inv_r, inv_n;
  logic [1:0]  a_n, m_n;
  logic [2:0]  r_n;
  logic [31:0] cnt;
  logic        timeout, decide, pv;
  assign timeout = busy && (cnt == 32'(WDT - 1));
  assign decide  = busy && (timeout || prop_valid);
  assign pv      = prop_valid && !timeout;
  dec_guard_comb u_c (.prop(prop), .prop_valid(pv), .mode_in(mode), .v(v), .i(i), .t(t), .vv(vv), .iv(iv),
                      .tv(tv), .gc(gc), .inv_in(inv_r), .applied(a_n), .mode_new(m_n), .reason(r_n),
                      .inv_out(inv_n));
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      busy <= 1'b0; done <= 1'b0; cnt <= '0; inv_r <= '0;
      mode <= M_NORMAL; applied <= A_NORMAL; reason <= R_OK;
    end else begin
      done <= 1'b0;
      if (start && !busy) begin busy <= 1'b1; cnt <= '0; end
      else if (decide) begin
        busy <= 1'b0; done <= 1'b1;
        applied <= a_n; mode <= m_n; reason <= r_n; inv_r <= inv_n;
      end else if (busy) cnt <= cnt + 32'd1;
    end
  end
endmodule
