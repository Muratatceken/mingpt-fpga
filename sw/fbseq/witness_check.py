"""
Independent re-check of the A5 realizability results and of the counterexamples in docs/10 (stdlib only,
shares no code with theory_sat.py; items follow an external audit of docs/10, 28 Oct 2026).

    python3 witness_check.py [--tries 1000]                            -> model/fbseq/witness_check.json

1. sat6        the S = 6, K = 6 SAT witness: 60 distinct codes, every next bit a function of its <= 4 state
               bits D_j and the letter, checked exhaustively (60 states x 3 letters x 6 bits).
2. core_fiber  structure of that witness: a closed 4-bit core whose 15 reachable values are the right cosets
               V4 p, and 2 fiber bits that each update as s_j' = s_j XOR f_j(core, letter).
3. lemma       the same shape from any V4, transversal, core labeling and V4 basis (coset core + Z2
               toggles), one LUT level; over random choices, how often the fiber fan-in drops from 1 + 4 + 2 to 6.
4. block_local beta_{K,w} for independent dense blocks with fixed letter control (w = 2), by subgroup closure.
5. counterexamples  CNOT keeps a wire and is odd; right multiplication on a partial point base is not local,
               the inverse code r = p^-1 is.
6. robustness  majority code keeps memory and corrects one flip; 60 states correcting any single flip need
               d >= 10 bits (Hamming bound), met by a shortened [10, 6, 3] Hamming code.
"""
import argparse
import itertools as it
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

G = [p for p in it.permutations(range(5)) if sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0]
IDX = {p: i for i, p in enumerate(G)}
E0 = IDX[tuple(range(5))]
GENS = [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4), tuple(range(5))]           # fbseq.A5_TABLE letters, w = 2


def cp(p, q):
    return tuple(p[q[i]] for i in range(5))                           # p o q


def inv(p):
    return tuple(sorted(range(5), key=lambda i: p[i]))


def step(p, a):
    return IDX[cp(G[p], GENS[a])]


def sat6():
    sol = next(v for v in json.load(open(os.path.join(ROOT, "model/fbseq/sat_a5.json")))
               if v["S"] == 6 and v["K"] == 6 and v["result"] == "sat")
    E = [tuple(map(int, s)) for s in sol["encoding"]]
    D = sol["deps"]
    assert len(set(E)) == 60 and all(len(d) + 2 <= 6 for d in D)
    n = 0
    for j, d in enumerate(D):
        for a in range(3):
            seen = {}
            for p in range(60):
                k = tuple(E[p][i] for i in d)
                assert seen.setdefault(k, E[step(p, a)][j]) == E[step(p, a)][j]
                n += 1
    return E, D, {"distinct_codes": 60, "max_state_fanin": max(map(len, D)), "exhaustive_checks": n, "passed": True}


def core_fiber(E, D):
    core = sorted(D[0])                           # smallest set closed under "reads only bits in the set"
    while True:
        nxt = sorted(set(core) | {i for j in core for i in D[j]})
        if nxt == core:
            break
        core = nxt
    fiber = [j for j in range(6) if j not in core]
    cval = lambda p: tuple(E[p][i] for i in core)
    H = [p for p in range(60) if cval(p) == cval(E0)]
    right = all((cval(p) == cval(q)) == (IDX[cp(G[p], inv(G[q]))] in H) for p in range(60) for q in range(60))
    flips = {}
    for j in fiber:
        f = {}
        for p in range(60):
            for a in range(3):
                assert f.setdefault((cval(p), a), E[p][j] ^ E[step(p, a)][j]) == E[p][j] ^ E[step(p, a)][j]
        reads = [i for i in D[j] if i != j]
        flips[str(j)] = {"toggle_given_core_and_letter": True, "core_bits_read": reads,
                         "fanin": len(D[j]) + 2, "flip_fraction": f"{sum(f.values())}/{len(f)}"}
    return {"core_bits": core, "fiber_bits": fiber, "reachable_core_values": len({cval(p) for p in range(60)}),
            "stabilizer": [list(G[p]) for p in H], "core_classes_are_right_cosets": right, "fiber": flips}


def lemma(tries, seed=0):
    """p = h t with t in a transversal of the right cosets H t; p o a = h c(t, a) t', c in H = V4 ~ Z2^2"""
    rng = random.Random(seed)
    best, hits, total = None, 0, 0
    for pt in range(5):                                               # the five V4, one in each point stabilizer
        V = [p for p in G if p[pt] == pt and cp(p, p) == G[E0]]
        cosets = []
        for p in G:
            c = frozenset(IDX[cp(h, p)] for h in V)
            if c not in cosets:
                cosets.append(c)
        for t in range(tries):
            x, y = rng.sample([h for h in V if h != G[E0]], 2)                    # basis of V4 ~ F2^2
            coord = {G[E0]: (0, 0), x: (1, 0), y: (0, 1), cp(x, y): (1, 1)}
            label = rng.sample(range(16), 15)
            if t == 0:
                label = list(range(15))
            T = [G[rng.choice(sorted(c))] if t else G[min(c)] for c in cosets]
            E = [None] * 60
            for k, c in enumerate(cosets):
                for h in V:
                    p = IDX[cp(h, T[k])]
                    E[p] = tuple((label[k] >> b) & 1 for b in range(4)) + coord[h]
            assert len(set(E)) == 60
            reads = []
            for j in range(6):
                f = {}                                    # core bit: next value; fiber bit: flip, given (core, a)
                for p in range(60):
                    for a in range(3):
                        v = E[step(p, a)][j] ^ (E[p][j] if j >= 4 else 0)
                        assert f.setdefault((E[p][:4], a), v) == v                 # core-only / toggle
                need = None                               # fewest core bits that still determine it (exact)
                for r in range(5):
                    for sub in it.combinations(range(4), r):
                        g = {}
                        if need is None and all(g.setdefault((tuple(c[i] for i in sub), a), v) == v
                                                for (c, a), v in f.items()):
                            need = list(sub)
                reads.append(need)
            fan = [len(reads[j]) + (j >= 4) + 2 for j in range(6)]
            total += 1
            hits += max(fan) <= 6
            if best is None or max(fan) < max(best["fanin"]):
                best = {"fanin": fan, "core_bits_read": reads}
    return {"construction": "4-bit coset core (fan-in <= 4 + 2) + 2 fiber toggles (fan-in <= 1 + 4 + 2)",
            "one_level_and_toggle_checked": True, "tries": total, "fraction_with_K6": round(hits / total, 4),
            "best": best}


def subgroups():
    subs, front = {frozenset([E0])}, [frozenset([E0])]
    while front:
        H = front.pop()
        for a in range(60):
            if a in H:
                continue
            C, q = {E0}, [E0]
            while q:
                p = q.pop()
                for g in list(H) + [a]:
                    z = IDX[cp(G[p], G[g])]
                    if z not in C:
                        C.add(z)
                        q.append(z)
            C = frozenset(C)
            if C not in subs:
                subs.add(C)
                front.append(C)
    return subs


def block_local(w=2):
    subs = subgroups()
    hist = {str(n): sum(len(H) == n for H in subs) for n in sorted({len(H) for H in subs})}
    out = {}
    for K in (4, 5, 6):
        cand = [H for H in subs if len(H) < 60 and math.ceil(math.log2(60 // len(H))) + w <= K]
        best = None
        for r in range(1, 4):
            for fam in it.combinations(cand, r):
                bits = sum(math.ceil(math.log2(60 // len(H))) for H in fam)
                if (best is None or bits < best[0]) and frozenset.intersection(*fam) == {E0}:
                    best = (bits, sorted(len(H) for H in fam))
        out[str(K)] = None if best is None else {"bits": best[0], "subgroup_orders": best[1]}
    return {"subgroups": len(subs), "orders": hist, "beta_K_w2": out}


def counterexamples():
    cnot = [0, 1, 3, 2]                                               # (a, b) -> (a, b xor a), a unchanged
    odd = sum(cnot[i] > cnot[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 1
    base = (0, 1, 2)
    a = GENS[0]
    bad = next({"p": p, "q": q, "a": a, "coord": 2, "old": p[2], "new": [cp(p, a)[2], cp(q, a)[2]]}
               for p in G for q in G if p[2] == q[2] and cp(p, a)[2] != cp(q, a)[2])
    local = all(len({inv(cp(p, g))[b] for p in G if inv(p)[b] == v}) == 1
                for g in GENS for b in base for v in range(5))
    return {"cnot_keeps_wire_and_is_odd": odd, "right_mult_partial_base_not_local": bad,
            "inverse_code_local": local}


def robustness():
    maj = lambda v: (int(sum(v) >= 2),) * 3
    ok = all(maj([c ^ (k == i) for i, c in enumerate((b, b, b))]) == (b, b, b) for b in (0, 1) for k in (-1, 0, 1, 2))
    d = next(d for d in range(6, 20) if 60 * (d + 1) <= 2 ** d)
    code = []
    for v in range(64):
        c = [0] * 10
        for j, pos in enumerate((3, 5, 6, 7, 9, 10)):
            c[pos - 1] = (v >> j) & 1
        syn = 0
        for pos, bit in enumerate(c, 1):
            syn ^= pos if bit else 0
        for pos in (1, 2, 4, 8):
            c[pos - 1] = int(bool(syn & pos))
        code.append(c)
    dist = min(sum(x != y for x, y in zip(u, v)) for u, v in it.combinations(code, 2))
    return {"majority_keeps_and_corrects": ok, "single_error_correcting_60_states_min_bits": d,
            "shortened_hamming": {"length": 10, "words": len(code), "min_distance": dist}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=1000)
    a = ap.parse_args()
    E, D, r1 = sat6()
    R = {"sat6": r1, "core_fiber": core_fiber(E, D), "lemma": lemma(a.tries), "block_local": block_local(),
         "counterexamples": counterexamples(), "robustness": robustness()}
    path = os.path.join(ROOT, "model", "fbseq", "witness_check.json")
    json.dump(R, open(path, "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
