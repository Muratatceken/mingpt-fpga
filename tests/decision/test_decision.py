"""Decision stack unit tests (make dec-test).  Fast: no training, no RTL."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sw", "decision"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sw", "zoo"))

import torch  # noqa: E402

from contract import (EDGES, GUVENLI_MOD, KISITLI_MOD, NORMAL, TT, V_BASE, V_INV, YUKU_KAPAT, bin_code,  # noqa: E402
                      tok_names)
from episodes import run_episodes  # noqa: E402
from guard import guard_step  # noqa: E402
from labeler import labels_from_costs  # noqa: E402
from pairs import INTENDED, generate_cell  # noqa: E402
from policies import FSMPolicy, RulePolicy  # noqa: E402
from sim import NoiseBank, Plant, sample_params, seed_of  # noqa: E402
from tokenizer import first_diff_step_from_end, obs_tokens  # noqa: E402


def test_vocab_and_bins():
    assert len(tok_names()) == 64
    for ch, e in EDGES.items():
        assert int(bin_code(e[0] - 1, ch)) == 0
        assert int(bin_code(e[-1], ch)) == len(e)
        for k, x in enumerate(e):                       # bin = number of edges <= code
            assert int(bin_code(x, ch)) == k + 1 and int(bin_code(x - 1, ch)) == k
    tV, _, _ = obs_tokens(np.array([6950]), np.array([100]), np.array([200]), np.array([False]), np.array([True]),
                          np.array([True]))
    assert tV[0] == V_INV
    tV, _, _ = obs_tokens(np.array([6950]), np.array([100]), np.array([200]), np.array([True]), np.array([True]),
                          np.array([True]))
    assert tV[0] == V_BASE + 7


def _run(seed_tag, B=64, N=50, label=False):
    rng = np.random.default_rng(seed_of(seed_tag))
    P = sample_params(rng, B, train=True)
    bank = NoiseBank([seed_of(seed_tag, b) for b in range(B)], N + 70)
    return run_episodes(Plant(P, bank), RulePolicy(B), N, label=label)


def test_simulator_deterministic_and_physical():
    a, b = _run("det"), _run("det")
    assert np.array_equal(a["seq"], b["seq"])
    healthy = a["ftype"] == 0
    assert a["v"][healthy].min() > 6800 and a["v"][healthy].max() < 8500
    assert (a["i"] >= -100).all() and (a["i"] < 4000).all()


def test_guard_priorities():
    # FORCE_SAFE beats everything, V limit -> safe, I limit -> at least load off, timeout fallback, safe latch
    assert guard_step(NORMAL, True, 0, 7200, 1000, 200, True, True, True, 2, 0)[0] == GUVENLI_MOD
    assert guard_step(NORMAL, True, 0, 6399, 1000, 200, True, True, True, 0, 0)[0] == GUVENLI_MOD
    assert guard_step(NORMAL, True, 0, 7200, 2800, 200, True, True, True, 0, 0)[0] == YUKU_KAPAT
    assert guard_step(NORMAL, False, 0, 7200, 1000, 200, True, True, True, 0, 0)[0] == KISITLI_MOD
    assert guard_step(NORMAL, True, 2, 7200, 1000, 200, True, True, True, 0, 0)[0] == GUVENLI_MOD
    assert guard_step(NORMAL, True, 2, 7200, 1000, 200, True, True, True, 1, 0)[0] == KISITLI_MOD   # RESUME
    a, m, r, n = guard_step(NORMAL, True, 0, 7200, 1000, 200, False, True, True, 0, 2)
    assert a == KISITLI_MOD and n == 3                                                          # 3rd invalid
    for (mi, ai), rep in TT.items():                                                            # no exit from SAFE
        if mi == 2:
            assert guard_step(ai, True, 2, 7200, 1000, 200, True, True, True, 0, 0)[0] == GUVENLI_MOD


def test_labels_tie_rule():
    costs = np.zeros((1, 4, 3), dtype=np.int64)
    a, acc = labels_from_costs(costs)
    assert a[0] == GUVENLI_MOD and acc.all()          # all tied -> most severe
    costs[0, :, 2] = [5, 0, 0, 9]
    a, acc = labels_from_costs(costs)
    assert a[0] == KISITLI_MOD and acc[0].tolist() == [False, True, True, False]


def test_labels_no_lookahead():
    r = _run("lab", B=48, N=40, label=True)
    pre = np.arange(40)[None, :] < r["f_onset"][:, None]
    # before any fault has started, a healthy plant is always labelled NORMAL
    assert (r["a_star"][pre] == NORMAL).mean() > 0.99


@pytest.mark.parametrize("tpl", ["BAT_SEN", "SEN", "OC"])
def test_pairs_valid(tpl):
    prs, st = generate_cell("pytest", tpl, "high", "low", 8, 4)
    assert st["accepted"] >= 3
    for p in prs:
        assert first_diff_step_from_end(p["A"]["seq"], p["B"]["seq"], 47) == 8
        assert (p["A"]["a_star"][47], p["B"]["a_star"][47]) == INTENDED[tpl]
        assert not (p["A"]["acc"][47] & p["B"]["acc"][47]).any()
        assert not p["A"]["reason"][:47].any() and not p["B"]["reason"][:47].any()


def test_fsm_solves_designed_pairs_rule_does_not():
    for tpl in ("BAT_SEN", "SEN", "OC"):
        prs, _ = generate_cell("pytest2", tpl, "high", "low", 16, 3)
        for p in prs:
            res = {}
            for side in ("A", "B"):
                m = p[side]
                for cls in (RulePolicy, FSMPolicy):
                    pol = cls(1)
                    prev = np.array([NORMAL])
                    for k in range(48):
                        a = pol.act(m["tV"][None, k], m["tI"][None, k], m["tT"][None, k], m["mode"][None, k], prev)
                        prev = m["act"][None, k]
                    res[(side, cls.name)] = bool(m["acc"][47][a[0]])
            assert res[("A", "fsm")] and res[("B", "fsm")]
            assert res[("A", "rule")] != res[("B", "rule")]           # the rule gets exactly one member right


def test_band_receptive_field():
    from models_dec import SPECS, build
    torch.manual_seed(0)
    m = build(SPECS["lm"]).eval()
    x = torch.randint(0, 64, (1, 300))
    with torch.no_grad():
        a = m(x)[0, 299]
        x2 = x.clone()
        x2[0, :299 - 126] = torch.randint(0, 64, (299 - 126,))
        assert torch.equal(a, m(x2)[0, 299])                      # tokens >= 127 back have no influence


def test_ir_and_int_reference_match_torch():
    from export_dec import compile_decision
    from ir import FloatRunner
    from models_dec import SPECS, build
    torch.manual_seed(1)
    for name in ("lm", "gru"):
        m = build(SPECS[name]).eval()
        prog = compile_decision(m)
        x = torch.randint(0, 64, (1, 300))                       # beyond 256: periodic RoPE + band in the IR
        with torch.no_grad():
            ref = m(x)[0].numpy()
        fr = FloatRunner(prog)
        for t in range(300):
            lf = fr.step(int(x[0, t]), t)
            if t in (0, 63, 64, 200, 255, 256, 299):
                assert np.abs(lf - ref[t]).max() < 1e-4, (name, t)
