"""
Coset core + Z2^k fiber realizations of group machines (docs/10 §5.3), for several groups.

    python3 core_fiber.py [--tries 300] [--labelings 300] [--groups A5+S4]   -> model/fbseq/core_fiber.json
    python3 core_fiber.py --export     dense / lemma / gauge / positional codes  -> model/fbseq/group_codes.json

Task: p_t = p_{t-1} o a_t, letters = generators + identity, w = ceil(log2 #letters) control bits.
Lemma: for H <= G with H ~ Z2^k, write p = h t (t in a transversal of the right cosets H t).  Then
p o a = h gamma(t, a) t' with gamma in H, so a dense c-bit core (c = ceil(log2 [G:H])) tracks the coset and
each of k fiber bits toggles: s_j' = s_j XOR gamma_j(core, a).  One LUT level, S = c + k = ceil(log2 |G|)
(since [G:H] = |G| / 2^k), fan-in <= c + w (core) and 1 + c + w (fiber), against S + w for one dense block.

Gauge: replacing t_k by h_k t_k shifts the fiber code of coset k by delta(h_k), so for a fiber functional l
the flip becomes l(gamma(k, a)) + beta(k) + beta(k') with free beta in F2.  "The flip reads only the core
bits D" means flip(k, a) = g(code_D(k), a) for some table g: linear equations over F2 in (beta, g), solved
exactly by elimination.  So for a core labeling we find, per functional, the fewest core bits; the
transversal and the basis follow from the solution, and the resulting code is re-checked by brute force.
Random transversals (the first column) almost never hit these solutions.
"""
import argparse
import itertools as it
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def cp(p, q):
    return tuple(p[q[i]] for i in range(len(p)))                     # p o q


def closure(gens):
    e = tuple(range(len(gens[0])))
    G, q = {e}, [e]
    while q:
        p = q.pop()
        for g in gens:
            z = cp(p, g)
            if z not in G:
                G.add(z)
                q.append(z)
    return sorted(G)


def groups():
    inv7 = {x: pow(x, 5, 7) for x in range(1, 7)}
    psl_a = tuple(list(range(1, 7)) + [0, 7])                         # x -> x + 1 on P1(F7), 7 = infinity
    psl_b = tuple([7] + [(-inv7[x]) % 7 for x in range(1, 7)] + [0])  # x -> -1/x
    return {
        "Z2^3": [(1, 0, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 2, 3, 5, 4)],
        "Z8": [(1, 2, 3, 4, 5, 6, 7, 0)],
        "S4": [(1, 2, 3, 0), (1, 0, 2, 3)],
        "A5": [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4)],                    # fbseq.A5_TABLE
        "PSL(2,7)": [psl_a, psl_b],
        "A6": [(1, 2, 3, 4, 0, 5), (0, 2, 3, 4, 5, 1)],                # two 5-cycles
    }


def elementary_abelian(G):
    """all subgroups H ~ Z2^k of largest k, as sorted element lists"""
    e = G[0]
    inv = [p for p in G if p != e and cp(p, p) == e]
    level = {frozenset([e])}
    while True:
        nxt = set()
        for H in level:
            for x in inv:
                if x not in H and all(cp(x, h) == cp(h, x) for h in H):
                    nxt.add(frozenset(H | {cp(x, h) for h in H}))
        if not nxt:
            return [sorted(H) for H in level], int(math.log2(len(next(iter(level)))))
        level = nxt


def fewest(f, c):
    """smallest set of core bits (exact search) that, with the letter, determines f(core code, letter)"""
    for r in range(c + 1):
        for sub in it.combinations(range(c), r):
            g, ok = {}, True
            for (code, a), v in f.items():
                if g.setdefault((tuple(code[i] for i in sub), a), v) != v:
                    ok = False
                    break
            if ok:
                return list(sub)


class Family:
    """codes of the coset core + fiber shape for one group: (subgroup, core labeling, transversal, basis)"""

    def __init__(self, gens):
        self.G = G = closure(gens)
        self.N = len(G)
        self.idx = {p: i for i, p in enumerate(G)}
        self.e = G[0]
        self.letters = gens + [self.e]
        self.w = math.ceil(math.log2(len(self.letters)))
        self.step = [[self.idx[cp(p, a)] for a in self.letters] for p in G]
        self.S = math.ceil(math.log2(self.N))
        self.subs, self.k = elementary_abelian(G)
        self.c = math.ceil(math.log2(self.N >> self.k))
        self.cosets = []
        for H in self.subs:
            cs, seen = [], set()
            for p in G:
                if p not in seen:
                    cs.append([cp(h, p) for h in H])
                    seen.update(cs[-1])
            self.cosets.append(cs)

    def random(self, rng):
        s = rng.randrange(len(self.subs))
        H = self.subs[s]
        return dict(s=s, label=rng.sample(range(2 ** self.c), len(self.cosets[s])),
                    T=[rng.randrange(len(H)) for _ in self.cosets[s]],
                    order=rng.sample([h for h in H if h != self.e], len(H) - 1))

    def code(self, x):
        H = self.subs[x["s"]]
        span = {self.e: ()}                                          # basis of H ~ F2^k, greedy from order
        for h in x["order"]:
            if h not in span:
                span = {z: v + (0,) for z, v in span.items()} | {cp(z, h): v + (1,) for z, v in span.items()}
        E = [None] * self.N
        for ci, cs in enumerate(self.cosets[x["s"]]):
            T = cs[x["T"][ci]]                                       # transversal element; cs = H T
            for h in H:
                E[self.idx[cp(h, T)]] = tuple((x["label"][ci] >> b) & 1 for b in range(self.c)) + span[h]
        assert len(set(E)) == self.N
        return E

    def fanin(self, E):
        c, fan, reads = self.c, [], []
        for j in range(c + self.k):
            f = {}
            for p in range(self.N):
                for a in range(len(self.letters)):
                    v = E[self.step[p][a]][j] ^ (E[p][j] if j >= c else 0)
                    assert f.setdefault((E[p][:c], a), v) == v       # core reads core; fiber toggles
            reads.append(fewest(f, c))
            fan.append(len(reads[-1]) + (j >= c) + self.w)
        return fan, reads


def gf2_solve(rows, n):
    """rows: (mask, rhs) over F2, n unknowns -> one solution as a bit mask, or None"""
    piv = {}
    for m, r in rows:
        while m:
            hb = m.bit_length() - 1
            if hb not in piv:
                piv[hb] = (m, r)
                break
            m, r = m ^ piv[hb][0], r ^ piv[hb][1]
        else:
            if r:
                return None
    x = 0
    for hb in sorted(piv):                                           # lower bits are settled first
        m, r = piv[hb]
        r ^= bin(m & x & ~(1 << hb)).count("1") & 1
        x |= r << hb
    return x


def gauge(F, s, label, ell, D):
    """beta(k) making the flip of fiber functional ell read only the core bits D, or None"""
    H, cos = F.subs[s], F.cosets[s]
    where = {z: ci for ci, cs in enumerate(cos) for z in cs}
    span = {F.e: ()}
    for h in H:
        if h not in span:
            span = {z: v + (0,) for z, v in span.items()} | {cp(z, h): v + (1,) for z, v in span.items()}
    nK, A = len(cos), len(F.letters)
    pats = {}
    for ci in range(nK):
        pats.setdefault(tuple((label[ci] >> b) & 1 for b in D), len(pats))
    rows = []
    for ci, cs in enumerate(cos):
        t = cs[0]
        for a, g in enumerate(F.letters):
            ta = cp(t, g)
            k2 = where[ta]
            gam = cp(ta, tuple(sorted(range(len(ta)), key=lambda i: cos[k2][0][i])))    # (t a) t'^-1 in H
            rhs = sum(x & y for x, y in zip(ell, span[gam])) & 1
            pi = pats[tuple((label[ci] >> b) & 1 for b in D)]
            rows.append(((1 << ci) ^ (1 << k2) ^ (1 << (nK + pi * A + a)), rhs))
    x = gf2_solve(rows, nK + len(pats) * A)
    return None if x is None else [(x >> ci) & 1 for ci in range(nK)], span


def build(F, s, label, chosen):
    """code from k independent functionals with their gauges: transversal h_k t_k, basis dual to them"""
    H = F.subs[s]
    ells = [ell for ell, _ in chosen]
    span = chosen[0][1][1]
    lin = lambda ell, h: sum(x & y for x, y in zip(ell, span[h])) & 1
    dual = [next(h for h in H if all(lin(ells[i], h) == (i == j) for i in range(F.k))) for j in range(F.k)]
    T = []
    for ci in range(len(F.cosets[s])):
        want = tuple(sol[0][ci] for _, sol in chosen)
        hk = next(h for h in H if tuple(lin(ell, h) for ell in ells) == want)
        T.append(H.index(hk))                                       # cosets[s][ci][i] = H[i] o t_ci
    return F.code(dict(s=s, label=label, T=T, order=dual + [h for h in H if h not in dual and h != F.e]))


def solve_label(F, s, label):
    """for a core labeling: the code whose k fiber functionals read the fewest core bits d (gauge solved)"""
    funcs = [f for f in it.product((0, 1), repeat=F.k) if any(f)]
    for d in range(F.c + 1):
        sols = []
        for ell in funcs:
            for D in it.combinations(range(F.c), d):
                r = gauge(F, s, label, ell, D)
                if r[0] is not None:
                    sols.append((ell, r))
                    break
        pick = []                                                    # k independent functionals among sols
        for ell, r in sols:
            cand = [e for e, _ in pick] + [ell]
            if len({tuple(sum(c * v[i] for c, v in zip(cs, cand)) % 2 for i in range(F.k))
                    for cs in it.product((0, 1), repeat=len(cand))}) == 2 ** len(cand):
                pick.append((ell, r))
        if len(pick) >= F.k:
            return build(F, s, label, pick[:F.k]), d


def deps_all(E, F, j):
    """exact smallest set of state bits that, with the letter, determines next bit j (unused codes free)"""
    f = {(E[p], a): E[F.step[p][a]][j] for p in range(F.N) for a in range(len(F.letters))}
    S = len(E[0])
    for r in range(S + 1):
        for sub in it.combinations(range(S), r):
            g = {}
            if all(g.setdefault((tuple(code[i] for i in sub), a), v) == v for (code, a), v in f.items()):
                return list(sub)


def positional_inverse(F):
    """r = p^-1 on a smallest base: r' = a^-1 r is local, each coordinate ceil(log2 n) bits"""
    n = len(F.e)
    for k in range(1, n + 1):
        for base in it.combinations(range(n), k):
            if sum(all(p[b] == b for b in base) for p in F.G) == 1:
                nb = math.ceil(math.log2(n))
                inv = lambda p: tuple(sorted(range(n), key=lambda i: p[i]))
                return [tuple((inv(p)[b] >> t) & 1 for b in base for t in range(nb)) for p in F.G]


def export(rng, names=("A5", "PSL(2,7)", "A6")):
    """codes for the FPGA loop bench (loop_bench.py kind gc:<group>_<code>) -> model/fbseq/group_codes.json"""
    specs = []
    for name in names:
        F = Family(groups()[name])
        x = F.random(rng)
        cands = {"dense": [tuple((v >> b) & 1 for b in range(F.S)) for v in rng.sample(range(2 ** F.S), F.N)],
                 "lemma": F.code(x),
                 "gauge": solve_label(F, x["s"], x["label"])[0],
                 "positional": positional_inverse(F)}
        for kind, E in cands.items():
            assert len(set(E)) == F.N
            D = [deps_all(E, F, j) for j in range(len(E[0]))]
            spec = dict(name=f"{name}_{kind}", group=name, code=kind, S=len(E[0]), w=F.w, letters=len(F.letters),
                        fanin=max(len(d) for d in D) + F.w, deps=D,
                        codes=[sum(b << i for i, b in enumerate(c)) for c in E],
                        start=sum(b << i for i, b in enumerate(E[0])), step=F.step)
            print(json.dumps({k: v for k, v in spec.items() if k not in ("codes", "step", "deps")}), flush=True)
            specs.append(spec)
    json.dump(specs, open(os.path.join(ROOT, "model", "fbseq", "group_codes.json"), "w"))


def study(name, gens, tries, labelings, rng):
    F = Family(gens)
    out = dict(group=name, order=F.N, letters=len(F.letters), w=F.w, S_lower_bound=F.S, two_rank=F.k,
               n_subgroups_max_rank=len(F.subs), dense_fanin=F.S + F.w, core_bits=F.c,
               lemma_fanin_bound=1 + F.c + F.w)
    best, hist = None, {}
    for _ in range(tries):                                           # random labeling, transversal, basis
        fan, _r = F.fanin(F.code(F.random(rng)))
        hist[max(fan)] = hist.get(max(fan), 0) + 1
        if best is None or max(fan) < max(best):
            best = fan
    out.update(random_tries=tries, random_best_fanin=max(best),
               random_histogram={str(m): hist[m] for m in sorted(hist)})
    best, hist, example = None, {}, None
    for _ in range(labelings):                                       # random labeling, solved gauge
        x = F.random(rng)
        s, label = x["s"], x["label"]
        E, d = solve_label(F, s, label)
        fan, reads = F.fanin(E)                                      # brute-force re-check of the code
        assert max(fan[F.c:]) <= 1 + d + F.w
        hist[max(fan)] = hist.get(max(fan), 0) + 1
        if best is None or max(fan) < max(best):
            best, example = fan, dict(subgroup=[list(h) for h in F.subs[s]], label=label, core_bits_read=reads)
    out.update(gauge_labelings=labelings, gauge_best_fanin=max(best), gauge_best_per_bit=best,
               gauge_histogram={str(m): hist[m] for m in sorted(hist)}, gauge_example=example)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=300)
    ap.add_argument("--labelings", type=int, default=300)
    ap.add_argument("--groups", default="")
    ap.add_argument("--export", action="store_true", help="only write the loop-bench codes (group_codes.json)")
    a = ap.parse_args()
    rng = random.Random(0)
    if a.export:
        return export(rng)
    R = []
    for name, gens in groups().items():
        if a.groups and name not in a.groups.split("+"):
            continue
        r = study(name, gens, a.tries, a.labelings, rng)
        print(json.dumps(r), flush=True)
        R.append(r)
    json.dump(R, open(os.path.join(ROOT, "model", "fbseq", "core_fiber.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
