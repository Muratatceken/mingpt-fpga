"""
Randomised regression of the stand-alone Nexys A7-100T design (board_sim.py, tb/tb_nexys.sv): random switch
settings and key sequences, every simulation against the bit-exact reference (board.py, board_sim.Board).

  python3 board_fuzz.py [models ...] [--cases N] [--stories-cases N] [--seed S] [--jobs 2] [--no-mid]
  python3 board_fuzz.py --directed [case names ...]        directed corner cases (DIRECTED below)
      models          default: bigram gru nanochat llama stories260k
      --cases         cases per names model (default 24); --stories-cases for TinyStories models (default 5)
      --no-mid        only keys at idle (C / U / D); default: every third random case also presses keys during
                      a run (c / u / d) and may contain CPU_RESET (R at idle, r during a run)
Switches: greedy or sampling, every temperature (SW[3:1]) and min-p (SW[5:4]) index (drawn from a shuffled
deck of all 32 combinations), random seeds (SW[15:8]) and SW7 (7-segment: tokens/s or cycles/token), SW6
(typewriter) off; the all-zero setting (on-board check of the sampled run) now and then. Before C the switches
change with probability 1/2, before U with 0.7, never before D. Fixed cases: D right after the automatic run
(twice), U with changed switches, D between runs.
Each simulation checks every token, hash, on-board check (LED2/LED3), LED token count, the text area, status
row, title and 7-segment display at every idle point (board_sim.compare).
Summary: model/zoo/board_fuzz.json; failing cases are listed with the board_sim.py command that repeats them.
"""
import json
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

from board_sim import Board, build, compare, parse, run_tb, switch_list
from data import ROOT

MODELS = ["bigram", "gru", "nanochat", "llama", "stories260k"]


class Deck:
    """all 32 (temperature, min-p) settings in random order, refilled when empty; case i of a model starts at
    position 5 i of the model's first permutation, so that every setting is drawn early in some case"""

    def __init__(self, rnd, perm=None, start=0):
        self.rnd = rnd
        self.cards = list(reversed(perm[start % 32:] + perm[:start % 32])) if perm else []

    def draw(self):
        if not self.cards:
            self.cards = [(t, p) for t in range(8) for p in range(4)]
            self.rnd.shuffle(self.cards)
        return self.cards.pop()


def random_sw(rnd, deck):
    if rnd.random() < 0.12:
        return 0                                           # power-up setting: the on-board check applies
    tau, minp = deck.draw()
    greedy = rnd.random() < 0.25
    return greedy | tau << 1 | minp << 4 | (rnd.random() < 0.5) << 7 | rnd.randrange(256) << 8


def make_case(model, rnd, deck, mid, fixed=None):
    """-> keys, switches (sws[0] automatic run, sws[i + 1] before key i), token counts of mid-run keys, Board"""
    sws = [random_sw(rnd, deck)]
    board = Board(model, sws[0])
    keys, dlys = "", {}
    stories = model == "stories260k" or model.startswith("ts_")
    n = len(fixed) if fixed else rnd.randint(2, 5 if stories else 7)
    for i in range(n):
        if fixed:
            k = fixed[i]
        else:
            choices = "CCUUDD"
            if mid and not board.blank:
                choices += "ccuudr" + ("R" if rnd.random() < 0.3 else "")
            k = rnd.choice(choices)
            if k.islower() and len(board.current()["tokens"]) - (k in "dr") < board.mid_floor:
                k = k.upper()
        change = rnd.random() < (1.0 if fixed and k == "U" else 0.7 if k in "Uu" else 0.0 if k in "Dd" else 0.5)
        sw = random_sw(rnd, deck) if change else sws[-1]
        if k.islower():
            n = len(board.current()["tokens"])
            if k in "cu" and board.pending is None and rnd.random() < 0.2:
                dlys[i] = n                                 # pressed as the run ends
            else:
                dlys[i] = rnd.randint(board.mid_floor, max(board.mid_floor, n - 1))
        board.key(k, sw, dlys.get(i, 0))
        keys += k
        sws.append(sw)
    board.finish()
    return keys, sws, dlys, board


def command(model, keys, sws, dlys):
    c = f"python3 board_sim.py {model} --press {keys} --sw {sws[0]:04x}"
    for i in range(len(keys)):
        if sws[i + 1] != sws[i]:
            c += f" --sw{i} {sws[i + 1]:04x}"
    for i, d in dlys.items():
        c += f" --dly{i} {d}"
    return c


def run_case(arg):
    model, idx, seed, mid, fixed, cmd, perm = arg
    rnd = random.Random(seed)
    keys, sws, dlys, board = make_case(model, rnd, Deck(rnd, perm, 5 * idx), mid, fixed)
    out = os.path.join(ROOT, "build", "nexys", model, "fuzz", f"case{idx:03d}")
    t0 = time.time()
    try:
        log, txt, _ = run_tb(cmd, keys, sws, dlys, out)
        s = parse(log, txt)
        ok, rep = compare(board, s, model, keys)
    except Exception as e:                                 # a crashed simulation is a failure, not an abort
        ok, rep, s = False, [f"ERROR {type(e).__name__}: {e}"], dict(runs=[], idles=[], cycles=0)
    n_abort = sum(1 for r in s["runs"] if r.get("aborted"))
    return dict(case=idx, seed=seed, keys=keys, sws=[f"{v:04x}" for v in sws], dlys=dlys, ok=ok,
                runs=len(s["runs"]), aborted=n_abort, tokens=sum(len(r["tokens"]) for r in s["runs"]),
                idles=len(s["idles"]), cycles=s["cycles"], seconds=round(time.time() - t0, 1),
                greedy=sum(r["sw"] & 1 for r in board.runs), checked=sum(r["check"] for r in board.runs),
                settings=sorted({((r["sw"] >> 1) & 7, (r["sw"] >> 4) & 3) for r in board.runs if not r["sw"] & 1}),
                command=command(model, keys, sws, dlys), report=rep if not ok else [])


# ----------------------------------------------------------------------------- directed corner cases
# (name, model, keys, sw0, {key: switches}, {key: token count; -1 = before the run's last token, "end" = as the
#  run ends}, intended behaviour as implemented in lm_gen.sv)
SLOW_CYC = 64                                              # tb_nexys.sv
DIRECTED = [
    ("btnc_twice_during_run", "gru", "cc", 0x0000, {}, {0: 10, 1: 120},
     "BTNC twice during a run: one queued run (random state continues), not two"),
    ("btnu_then_btnc_during_run", "gru", "uc", 0x0000, {0: 0x5A0A}, {0: 30, 1: 31},
     "BTNU then BTNC during a run: one queued reseeded run with the switches when it starts (C keeps the reseed)"),
    ("btnc_then_btnu_during_run", "gru", "cu", 0x0300, {1: 0x0326}, {0: 5, 1: 150},
     "BTNC then BTNU during a run: one queued reseeded run"),
    ("btnd_during_run", "gru", "dC", 0x0000, {}, {0: 40},
     "BTND during a run: ignored (not queued); the run's text is complete at idle"),
    ("btnd_idle", "gru", "DCDD", 0x0000, {1: 0x0041}, {},
     "BTND at idle: the text area is cleared, status row / check / 7-segment keep the last run"),
    ("reset_at_first_step", "gru", "r", 0x0000, {0: 0x1100}, {0: 0},
     "CPU_RESET during the clear / first step: the board restarts with a reseeded run (seed from SW[15:8])"),
    ("reset_before_last_token", "gru", "r", 0x0000, {}, {0: -1},
     "CPU_RESET just before the last token: the run is aborted, a fresh reseeded run follows"),
    ("reset_new_seed", "gru", "Cr", 0x0300, {1: 0x7714}, {1: 80},
     "CPU_RESET during a BTNC run with new switches: the restart is seeded from the switches, not continued"),
    ("reset_drops_queued_run", "gru", "ur", 0x0000, {0: 0x0101}, {0: 20, 1: 60},
     "BTNU queued, then CPU_RESET: the queued run is dropped, only the run after the reset follows"),
    ("reset_idle", "gru", "R", 0x0000, {0: 0x2a02}, {},
     "CPU_RESET at idle: as at power-up"),
    ("btnc_at_end_of_run", "gru", "cU", 0x0000, {}, {0: "end"},
     "BTNC as the run ends (G_FIN / idle): exactly one more run"),
    ("typewriter", "gru", "U", 0x0000, {0: 0x0040}, {},
     "SW6 typewriter: same tokens, screen and counted cycles (status row), SLOW_CYC+1 clocks per token pause"),
    ("typewriter_queued", "gru", "Uc", 0x0000, {0: 0x0040}, {1: 25},
     "typewriter run with a BTNC queued during it: the queued run is also a typewriter run"),
    ("stories_reset_after_scroll", "stories260k", "r", 0x0000, {}, {0: 200},
     "CPU_RESET after the text has scrolled (TinyStories): the restart clears the scrolled text area"),
    ("stories_btnd_during_run", "stories260k", "dD", 0x0000, {}, {0: 100},
     "BTND during a TinyStories run: ignored; BTND at idle clears"),
]


def run_stamps(stamps):
    """TBTOK (ntok, cycle) -> per run the cycles of tokens 1..n (a run starts when the counter returns to 0)"""
    runs, cur, prev = [], [], 0
    for n, c in stamps:
        if n < prev or n == 0:
            if cur:
                runs.append(cur)
            cur = []
        if n > 0:
            cur.append(c)
        prev = n
    if cur:
        runs.append(cur)
    return runs


def run_directed(arg):
    name, model, keys, sw0, swk, dk, what, cmd = arg
    sws = switch_list(keys, sw0, swk)
    probe = Board(model, sws[0])                           # token counts of the runs the keys act on
    dlys = {}
    for i, k in enumerate(keys):
        if i in dk:
            n = len(probe.current()["tokens"])
            dlys[i] = {-1: n - 1, "end": n}.get(dk[i], dk[i])
        probe.key(k, sws[i + 1], dlys.get(i, 0))
    probe.finish()
    out = os.path.join(ROOT, "build", "nexys", model, "directed", name)
    try:
        log, txt, t = run_tb(cmd, keys, sws, dlys, out)
        s = parse(log, txt)
        ok, rep = compare(probe, s, model, keys)
    except Exception as e:
        return dict(name=name, ok=False, behaviour=what, report=[f"ERROR {type(e).__name__}: {e}"])
    extra = []
    if name.startswith("typewriter"):                     # run 0 normal, run 1 (and 2) typewriter
        st = run_stamps(s["stamps"])
        r0, r1 = s["runs"][0], s["runs"][1]
        same_cyc = r0["cycles"] == r1["cycles"] and r0["tokens"] == r1["tokens"]
        extra.append(f"typewriter: counted cycles {r1['cycles']} {'=' if same_cyc else '!='} {r0['cycles']} of the normal "
                     f"run with the same tokens")
        ok &= same_cyc
        # uncounted clocks before token i (relative to the first token): i (G_THR) or i (SLOW_CYC + 1) (pause)
        for k, (r, tw) in enumerate(zip(s["runs"], [False, True, True])):
            pause = [(t - st[k][0]) - (c - r["tok_cycles"][0]) for t, c in zip(st[k], r["tok_cycles"])]
            want = [i * (SLOW_CYC + 1 if tw else 1) for i in range(len(pause))]
            good = len(st[k]) == len(r["tok_cycles"]) and pause == want
            extra.append(f"run {k} ({'typewriter' if tw else 'normal'}): uncounted clocks between tokens "
                         f"{'exactly ' + str(SLOW_CYC + 1 if tw else 1) if good else 'WRONG'} per token "
                         f"({len(pause)} tokens)")
            ok &= good
    return dict(name=name, model=model, keys=keys, sws=[f"{v:04x}" for v in sws], dlys=dlys, ok=ok, behaviour=what,
                runs=len(s["runs"]), tokens=sum(len(r["tokens"]) for r in s["runs"]), seconds=round(t),
                command=command(model, keys, sws, dlys), report=rep + extra)


def directed(jobs, names=None):
    cmds = {m: build(m) for m in sorted({d[1] for d in DIRECTED})}
    todo = [d + (cmds[d[1]],) for d in DIRECTED if not names or d[0] in names]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(run_directed, todo))
    for r in res:
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['name']}: {r['behaviour']}")
        for ln in r["report"]:
            if not r["ok"] or ln.startswith("typewriter") or "uncounted" in ln or "aborted" in ln:
                print("    " + ln)
    return res


# ----------------------------------------------------------------------------- start-cycle race
# BTNC during the automatic run queues a run; BTNU is then pressed so that its debounced pulse sweeps over the
# cycle in which the queued run starts. Intended: every BTNU press gives a reseeded run, either the queued run
# itself (pulse before its start) or one more run after it (pulse after the start).
RACE_SW = 0x5500                                           # seed 0x55, otherwise as the automatic run


def race_case(arg):
    ofs, n, cmd = arg
    out = os.path.join(ROOT, "build", "nexys", "gru", "race", f"ofs{ofs:04d}")
    log, _, _ = run_tb(cmd, "cu", [0, 0, RACE_SW], {0: 10, 1: n - 1}, out, extra=[f"+OFS1={ofs}"])
    s = parse(log, None)
    return ofs, [r["tokens"] for r in s["runs"]]


def second_run(key, sw):
    b = Board("gru", 0)
    b.key(key, sw)
    return b.runs[1]["tokens"]


def race(jobs):
    cmd = build("gru")
    auto = Board("gru", 0).runs[0]["tokens"]
    c_run, u_run = second_run("C", 0), second_run("U", RACE_SW)
    n = len(auto)
    log, _, _ = run_tb(cmd, "c", [0, 0], {0: 10}, os.path.join(ROOT, "build", "nexys", "gru", "race", "calib"))
    st = run_stamps(parse(log, None)["stamps"])[0]
    d = st[n - 1] - st[n - 2]                              # clocks from token n-1 to the last token
    outcomes, ok = {}, True
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for ofs, runs in ex.map(race_case, [(o, n, cmd) for o in range(d - 30, d + 11)]):
            rest = runs[1:]
            kind = ("reseeded queued run" if rest == [u_run] else "RESEED LOST" if rest == [c_run] else
                    "queued run + reseeded run" if rest == [c_run, u_run] else "UNEXPECTED")
            outcomes[ofs] = kind
            ok &= runs[0] == auto and kind in ("reseeded queued run", "queued run + reseeded run")
    first = {}
    for o, k in sorted(outcomes.items()):
        first.setdefault(k, []).append(o - d)
    for k, v in first.items():
        print(f"  {k}: BTNU {min(v)} .. {max(v)} clocks after the last token's time ({len(v)} offsets)")
    print(f"start-cycle race: {'PASS' if ok else 'FAIL'}")
    return dict(ok=ok, outcomes={k: [min(v), max(v), len(v)] for k, v in first.items()},
                behaviour="BTNU pressed around the cycle a queued run starts: always a reseeded run")


def main():
    a = sys.argv[1:]
    opt = lambda k, d: int(a[a.index(k) + 1]) if k in a else d
    path = os.path.join(ROOT, "model", "zoo", "board_fuzz.json")
    if "--race" in a:
        r = race(opt("--jobs", 2))
        summary = json.load(open(path)) if os.path.exists(path) else {}
        summary.setdefault("directed", {})["btnu_at_queued_run_start"] = r
        json.dump(summary, open(path, "w"), indent=1)
        sys.exit(0 if r["ok"] else 1)
    if "--directed" in a:
        res = directed(opt("--jobs", 2), [x for x in a if not x.startswith("--") and not x.isdigit()])
        summary = json.load(open(path)) if os.path.exists(path) else {}
        summary.setdefault("directed", {}).update({r["name"]: {k: v for k, v in r.items() if k != "name"} for r in res})
        json.dump(summary, open(path, "w"), indent=1)
        sys.exit(0 if all(r["ok"] for r in res) else 1)
    skip = {a[a.index(k) + 1] for k in ("--cases", "--stories-cases", "--seed", "--jobs") if k in a}
    models = [x for x in a if not x.startswith("--") and x not in skip] or MODELS
    n_names, n_ts, seed0, jobs = opt("--cases", 24), opt("--stories-cases", 5), opt("--seed", 2026), opt("--jobs", 2)
    mid_on = "--no-mid" not in a
    summary = json.load(open(path)) if os.path.exists(path) else {}
    all_ok = True
    for model in models:
        t0 = time.time()
        cmd = build(model)
        stories = model == "stories260k" or model.startswith("ts_")
        fixed = ["DU", "UDC"] if stories else ["DD", "DDC", "UDU", "CUD"]
        cases = []
        mseed = seed0 * 100003 + zlib.crc32(model.encode()) % 10007 * 101
        perm = [(t, p) for t in range(8) for p in range(4)]
        random.Random(mseed).shuffle(perm)
        for i in range(n_ts if stories else n_names):
            f = fixed[i] if i < len(fixed) else None
            mid = mid_on and f is None and i % 3 == 0
            cases.append((model, i, mseed + i, mid, f, cmd, perm))
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            res = list(ex.map(run_case, cases))
        bad = [r for r in res if not r["ok"]]
        cover = sorted({tuple(s) for r in res for s in r["settings"]})
        taus, minps = sorted({t for t, _ in cover}), sorted({p for _, p in cover})
        keys = "".join(r["keys"] for r in res)
        summary[model] = dict(
            ok=not bad, cases=len(res), failed=len(bad), runs=sum(r["runs"] for r in res),
            aborted_runs=sum(r["aborted"] for r in res), tokens=sum(r["tokens"] for r in res),
            idle_points=sum(r["idles"] for r in res), cycles=sum(r["cycles"] for r in res),
            keys={k: keys.count(k) for k in "CUDRcudr" if keys.count(k)},
            greedy_runs=sum(r["greedy"] for r in res), runs_with_onboard_check=sum(r["checked"] for r in res),
            sampling_settings=f"{len(cover)} of 32", taus=taus, minps=minps, seed=seed0,
            seconds=round(time.time() - t0),
            failures=[dict(case=r["case"], command=r["command"], report=r["report"][:12]) for r in bad],
            cases_list=[dict(case=r["case"], keys=r["keys"], sws=r["sws"], dlys=r["dlys"], ok=r["ok"], runs=r["runs"],
                             tokens=r["tokens"]) for r in res])
        s = summary[model]
        print(f"{model}: {'PASS' if not bad else 'FAIL'}  {s['cases']} cases, {s['runs']} runs ({s['aborted_runs']} "
              f"aborted by reset, {s['greedy_runs']} greedy, {s['runs_with_onboard_check']} with on-board check), "
              f"{s['tokens']} tokens, {s['idle_points']} idle points, keys {s['keys']}, sampling settings "
              f"{s['sampling_settings']} (temperatures {taus}, min-p {minps}), {s['seconds']} s")
        for r in bad:
            print(f"  case {r['case']}: {r['command']}")
            for ln in r["report"]:
                if "FAIL" in ln or "MISMATCH" in ln or "ERROR" in ln or "!=" in ln:
                    print("    " + ln)
        all_ok &= not bad
        json.dump(summary, open(path, "w"), indent=1)
    print(f"summary -> {os.path.relpath(path, ROOT)}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
