"""Scenario corpus, partitions and manifests (WP2.4-2.5).

    python3 dataset.py all            # val + test (frozen) + train  -> data/decision/<split>/*.npz + manifests
    python3 dataset.py val|test|train

Splits are generated from disjoint seed namespaces (sha256 of the split name), so no parameter set, noise bank
or pair can appear in two splits; both members of a pair always stay together.  The test split is written once
and only its manifest + sha256 checksums are committed (pre-registration: it is not evaluated in Project I).
Training data = template episodes (d ~ U{1..40}, decision step t0 in [20, 47], continued 0-20 steps with a
behaviour policy) + free-running episodes (random fault, behaviour mix rule/FSM/oracle-eps/random), labelled
at every step by the rollout labeler.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from contract import ROOT, SCN, contract_hash
from episodes import run_episodes
from pairs import TEMPLATES, generate_cell
from policies import FSMPolicy, RandomSafe, RulePolicy, Scripted
from sim import NoiseBank, Plant, sample_params, seed_of

OUT = os.path.join(ROOT, "data", "decision")
MAN = os.path.join(ROOT, "sw", "decision", "manifests")
SEVS, NOISES, DS = ("low", "high"), ("low", "high"), (2, 8, 16, 32)


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- pair splits (val / test)
def gen_pair_split(split, n_d2, n_d3, n_d4):
    recs, stats = [], {}
    for tpl in TEMPLATES:
        for sev in SEVS:
            for noise in NOISES:
                for d in DS:
                    key = f"{tpl}/{sev}/{noise}/d{d}"
                    for diff, n in (("D2", n_d2), ("D3", n_d3)):
                        if n == 0:
                            continue
                        got = []
                        base = 0
                        for _ in range(4):                         # top up until n accepted (logged)
                            prs, st = generate_cell(split, tpl, sev, noise, d, int(n * 1.3) + 2, difficulty=diff,
                                                    base_index=base)
                            got += prs
                            base += int(n * 1.3) + 2
                            stats.setdefault(key + "/" + diff, []).append(st)
                            if len(got) >= n:
                                break
                        recs += got[:n]
    # D4: indistinguishable histories (SEN template without the revealing load cycle), per (sev, noise)
    if n_d4:
        for sev in SEVS:
            for noise in NOISES:
                prs, st = generate_cell(split, "SEN", sev, noise, 0, n_d4 * 12, difficulty="D4")
                stats[f"D4/{sev}/{noise}"] = [st]
                recs += prs[:n_d4 * 12]
    return recs, stats


def pack_pairs(recs):
    """pairs -> arrays: seq (2P, L), y (2P,), acc (2P, 4), meta list."""
    seqs, ys, accs, metas = [], [], [], []
    for r in recs:
        t0 = r["t0"]
        L = 1 + 5 * (t0 + 1)
        for side in ("A", "B"):
            m = r[side]
            seqs.append(m["seq"][:L])
            ys.append(int(m["a_star"][t0]))
            accs.append(m["acc"][t0])
            metas.append(dict(template=r["template"], difficulty=r["difficulty"], sev=r["sev"], noise=r["noise"],
                              d=r["d"], t0=t0, pair_index=r["pair_index"], member=side, bank_seed=r["bank_seed"],
                              meta_e=int(r["meta"]["e"]), params=m["params"]))
    return np.stack(seqs), np.array(ys), np.stack(accs), metas


def write_split(split, recs, stats, extra=None):
    os.makedirs(os.path.join(OUT, split), exist_ok=True)
    os.makedirs(MAN, exist_ok=True)
    seq, y, acc, metas = pack_pairs(recs)
    path = os.path.join(OUT, split, "pairs.npz")
    np.savez_compressed(path, seq=seq, y=y, acc=acc)
    with open(os.path.join(OUT, split, "pairs_meta.json"), "w") as f:
        json.dump(metas, f)
    man = dict(split=split, contract_sha256=contract_hash(), git=git_rev(), created=time.strftime("%Y-%m-%d %H:%M"),
               n_pairs=len(recs), n_scenarios=len(y), seq_sha256=sha(seq), y_sha256=sha(y), acc_sha256=sha(acc),
               counts=_counts(metas), acceptance=_acc_summary(stats))
    if extra:
        man.update(extra)
    with open(os.path.join(MAN, f"{split}_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def _counts(metas):
    c = {}
    for m in metas:
        k = f"{m['template']}/{m['difficulty']}/{m['sev']}/{m['noise']}/d{m['d']}"
        c[k] = c.get(k, 0) + 1
    return c


def _acc_summary(stats):
    out = {}
    for k, sts in stats.items():
        att = sum(s["attempted"] for s in sts)
        acc = sum(s["accepted"] for s in sts)
        out[k] = dict(attempted=att, accepted=acc, rate=round(acc / max(att, 1), 3))
    return out


# ---------------------------------------------------------------- training episodes
def gen_train(n_free=16000, n_tpl_per=60, seed="train", chunk=2000, extra_max=20):
    os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
    X, Y, info = [], [], []
    rng = np.random.default_rng(seed_of(seed, "master"))
    t_start = time.time()
    # (1) template episodes: d ~ U{1..40}, t0 in [20,47] (when the template fits), continued with behaviour policy
    tpl_jobs = []
    for tpl in TEMPLATES:
        for sev in SEVS:
            for noise in NOISES:
                for d in range(1, 41):
                    t0 = int(rng.integers(max(20, d + 12), 48)) if d + 12 < 48 else 47
                    tpl_jobs.append((tpl, sev, noise, d, t0))
    n_each = max(1, n_tpl_per * 4 // 40)
    for j, (tpl, sev, noise, d, t0) in enumerate(tpl_jobs):
        prs, st = generate_cell(seed, tpl, sev, noise, d, n_each, t0=t0, seeds_per_pair=2, base_index=0)
        if not prs:
            continue
        extra = int(rng.integers(0, extra_max + 1))
        x, y, inf = continue_pairs(prs, t0, extra, rng)
        X += x
        Y += y
        info += inf
        if j % 60 == 0:
            print(f"  templates {j}/{len(tpl_jobs)}  episodes {len(X)}  {time.time() - t_start:.0f}s", flush=True)
    # (2) free-running episodes
    for c0 in range(0, n_free, chunk):
        B = min(chunk, n_free - c0)
        x, y, inf = free_running(B, rng, f"{seed}/free/{c0}")
        X += x
        Y += y
        info += inf
        print(f"  free {c0 + B}/{n_free}  episodes {len(X)}  {time.time() - t_start:.0f}s", flush=True)
    return X, Y, info


def _behaviour(B, rng):
    """per-episode behaviour policy id: 0 rule, 1 FSM, 2 oracle-eps, 3 random."""
    return rng.choice(4, size=B, p=[0.3, 0.3, 0.2, 0.2])


class Mix:
    """mixes rule / FSM / random per episode (oracle handled by run_episodes)."""

    def __init__(self, B, beh, rng):
        self.beh = beh
        self.r, self.f, self.x = RulePolicy(B), FSMPolicy(B), RandomSafe(B, rng)

    def reset(self, idx=None):
        self.r.reset(), self.f.reset(), self.x.reset()

    def act(self, *a):
        ar, af, ax = self.r.act(*a), self.f.act(*a), self.x.act(*a)
        return np.select([self.beh == 0, self.beh == 1, self.beh == 3], [ar, af, ax], ar)


def free_running(B, rng, tag, N=80):
    P = sample_params(rng, B, train=True)
    bank = NoiseBank([seed_of(tag, b) for b in range(B)], N + 62)
    beh = _behaviour(B, rng)
    pol = Mix(B, beh, rng)
    rec = run_episodes(Plant(P, bank), pol, N, label=True, oracle_eps=(beh == 2, 0.2), rng=rng)
    xs = [rec["seq"][b] for b in range(B)]
    ys = [rec["a_star"][b] for b in range(B)]
    inf = [dict(kind="free", ftype=int(rec["ftype"][b]), beh=int(beh[b]), N=N) for b in range(B)]
    return xs, ys, inf


def continue_pairs(prs, t0, extra, rng):
    """re-run both members of each pair for t0+1+extra steps: scripted prefix, behaviour policy afterwards,
    labels at every step."""
    N = t0 + 1 + extra
    out_x, out_y, out_i = [], [], []
    for side in ("A", "B"):
        P = {k: np.array([p[side]["params"][k] for p in prs]) for k in prs[0][side]["params"]}
        B = len(prs)
        sched = np.full((B, N), -1, dtype=np.int64)
        for b, p in enumerate(prs):
            sched[b, :t0] = p[side]["schedule"][:t0]
        bank = NoiseBank([p["bank_seed"] for p in prs], N + 62)
        P["delay"] = np.zeros(B, dtype=np.int64)
        beh = _behaviour(B, rng)
        pol = Scripted(sched, Mix(B, beh, rng))
        rec = run_episodes(Plant(P, bank), pol, N, label=True, oracle_eps=(beh == 2, 0.2), rng=rng)
        for b in range(B):
            out_x.append(rec["seq"][b])
            out_y.append(rec["a_star"][b])
            out_i.append(dict(kind="template", template=prs[b]["template"], d=prs[b]["d"], t0=t0, member=side,
                              N=N))
    return out_x, out_y, out_i


def pack_train(X, Y, L=512):
    """variable-length episodes -> X (n, L) int16 PAD-padded, Y (n, L) int16 with labels at MODE positions."""
    from contract import PAD
    n = len(X)
    Xa = np.full((n, L), PAD, dtype=np.int16)
    Ya = np.full((n, L), -1, dtype=np.int16)
    for j, (x, y) in enumerate(zip(X, Y)):
        x = x[:L]
        Xa[j, :len(x)] = x
        steps = min(len(y), (len(x) - 1) // 5)
        pos = 4 + 5 * np.arange(steps)
        Ya[j, pos] = y[:steps]
    return Xa, Ya


def main(which):
    t = time.time()
    if which in ("val", "all"):
        n = SCN["pairs_per_cell"]["val"]
        recs, st = gen_pair_split("val", n, 0, 0)
        man = write_split("val", recs, st)
        print("val", man["n_pairs"], "pairs", f"{time.time() - t:.0f}s")
    if which in ("test", "all"):
        pp = SCN["pairs_per_cell"]
        recs, st = gen_pair_split("test", pp["test_D2"], pp["test_D3"], pp["D4_test"])
        man = write_split("test", recs, st)
        print("test", man["n_pairs"], "pairs", f"{time.time() - t:.0f}s")
    if which in ("train", "all"):
        X, Y, info = gen_train()
        Xa, Ya = pack_train(X, Y)
        np.savez_compressed(os.path.join(OUT, "train", "train.npz"), X=Xa, Y=Ya)
        with open(os.path.join(OUT, "train", "train_info.json"), "w") as f:
            json.dump(info, f)
        man = dict(split="train", contract_sha256=contract_hash(), git=git_rev(),
                   created=time.strftime("%Y-%m-%d %H:%M"), n_episodes=len(X), X_sha256=sha(Xa), Y_sha256=sha(Ya),
                   n_labels=int((Ya >= 0).sum()),
                   label_dist=np.bincount(Ya[Ya >= 0].astype(int), minlength=4).tolist())
        with open(os.path.join(MAN, "train_manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        print("train", len(X), "episodes", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
