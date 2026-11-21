"""Decision contract: single source of truth for bins, vocabulary, actions, modes, guard and labeler.

Everything that must agree between Python (simulator, models, labeler) and RTL (tokenizer, guard,
rule/FSM baselines) is read from spec/contract_v0.json through this module.
"""
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SPEC = os.path.join(HERE, "spec", "contract_v0.json")

with open(SPEC) as f:
    C = json.load(f)


def contract_hash():
    with open(SPEC, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- actions / modes
NORMAL, YUKU_KAPAT, KISITLI_MOD, GUVENLI_MOD = 0, 1, 2, 3
ACTIONS = [a["name"] for a in C["actions"]]
M_NORMAL, M_KISITLI, M_GUVENLI = 0, 1, 2
MODES = [m["name"] for m in C["modes"]]
ACT_PAYLOAD = np.array([1, 0, 0, 0], dtype=np.int64)            # NORMAL turns the payload on
MODE_HK = np.array([m["hk_mult"] for m in C["modes"]])
# mode after an applied action (YUKU_KAPAT keeps the mode)
MODE_AFTER = {NORMAL: M_NORMAL, KISITLI_MOD: M_KISITLI, GUVENLI_MOD: M_GUVENLI}

# ---------------------------------------------------------------- vocabulary
VOC = C["vocab"]
V_SIZE = VOC["size"]
ACT_BASE, MODE_BASE, BOS = VOC["ACT_BASE"], VOC["MODE_BASE"], VOC["BOS"]
V_BASE, V_INV, I_BASE, I_INV, T_BASE, T_INV, PAD = (VOC[k] for k in
                                                    ("V_BASE", "V_INV", "I_BASE", "I_INV", "T_BASE", "T_INV", "PAD"))
EDGES = {k: np.array(C["bins"][k], dtype=np.int64) for k in ("V", "I", "T")}
TOK_PER_STEP = len(VOC["step_grammar"])                          # V I T MODE ACT
assert V_INV == V_BASE + len(EDGES["V"]) + 1
assert I_INV == I_BASE + len(EDGES["I"]) + 1
assert T_INV == T_BASE + len(EDGES["T"]) + 1


def bin_code(code, ch):
    """bin = number of edges <= code (vectorised, int)."""
    e = EDGES[ch]
    return (np.asarray(code, dtype=np.int64)[..., None] >= e).sum(-1)


def tok_names():
    n = [f"ACT_{a}" for a in ACTIONS] + [f"MODE_{m}" for m in MODES] + ["BOS"]
    n += [f"V{b}" for b in range(len(EDGES["V"]) + 1)] + ["V_INV"]
    n += [f"I{b}" for b in range(len(EDGES["I"]) + 1)] + ["I_INV"]
    n += [f"T{b}" for b in range(len(EDGES["T"]) + 1)] + ["T_INV"]
    n += [f"RES{i}" for i in range(PAD - T_INV - 1)] + ["PAD"]
    assert len(n) == V_SIZE
    return n


# ---------------------------------------------------------------- guard constants
G = C["guard"]
LIM_V, LIM_I, LIM_T = G["limits_raw"]["V_low_mV"], G["limits_raw"]["I_high_mA"], G["limits_raw"]["T_high_dC"]
INV_PERSIST = G["invalid_persist_steps"]
GC_NONE, GC_RESUME, GC_FORCE_SAFE = 0, 1, 2
R_OK, R_FORCE, R_LIM_VT, R_LIM_I, R_RESUME, R_TIMEOUT, R_INVALID, R_DENY = range(8)
TT = {}                                                          # (mode, proposal) -> replacement or None
for mi, mname in enumerate(MODES):
    for ai, aname in enumerate(ACTIONS):
        v = G["transition_table"][mname][aname]
        TT[(mi, ai)] = None if v == "allow" else ACTIONS.index(v)

# ---------------------------------------------------------------- rule thresholds (token level)
RULE_V_SAFE = int(bin_code(6599, "V"))       # V_bin < 4  <=> V < 6600
RULE_V_RESTR = int(bin_code(6999, "V"))      # V_bin < 8  <=> V < 7000
RULE_T_SAFE = int(bin_code(400, "T"))        # T_bin >= 10 <=> T >= 40.0 C
RULE_I_OC = int(bin_code(2000, "I"))         # I_bin >= 11 <=> I >= 2000 mA
assert (RULE_V_SAFE, RULE_V_RESTR, RULE_T_SAFE, RULE_I_OC) == (3, 7, 10, 11)

LAB = C["labeler"]
SCN = C["scenarios"]
MODEL = C["model"]
