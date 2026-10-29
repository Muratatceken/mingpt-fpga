"""
Smallest state found for a one-level, fan-in-K realization of a group machine (docs/10 §5.3), by one
construction that contains the earlier ones:

    python3 realize.py [--groups A5+PSL(2,7)] [--K 4,5,6,7] [--labelings 40]   -> model/fbseq/realize.json

Core: blocks for subgroups O_1..O_r that all contain H, with O_1 ∩ ... ∩ O_r = H; block i holds the right
coset O_i p in b_i = ceil(log2 [G:O_i]) bits and reads only itself and the letter (fan-in b_i + w).
Fiber: H ~ Z2^k, k toggle bits; the flip of fiber functional l is l(gamma(k, a)) + beta(k) + beta(k')
(gauge = choice of transversal), and "reads only the core bits D" is linear over F2 in (beta, g), solved
exactly (core_fiber.gf2_solve).  S = sum b_i + k, fan-in max(b_i + w, 1 + |D| + w).
  H = 1            -> independent dense blocks (beta_{K,w}, docs/10 §5.1)
  r = 1, O_1 = H   -> one dense coset core + fiber (core_fiber.py)
Every code is rebuilt and its exact fan-in re-measured by brute force (core_fiber.deps_all style).
Lower bound: S >= ceil(log2 |G|) (Myhill-Nerode).
"""
import argparse
import itertools as it
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core_fiber as cf  # noqa: E402

ROOT = cf.ROOT


class Group:
    def __init__(self, gens):
        G = cf.closure(gens)
        self.N = N = len(G)
        idx = {p: i for i, p in enumerate(G)}
        self.mul = [[idx[cf.cp(p, q)] for q in G] for p in G]
        self.inv = [next(j for j in range(N) if self.mul[i][j] == 0) for i in range(N)]
        self.letters = [idx[g] for g in gens] + [0]                   # element 0 = identity
        self.w = math.ceil(math.log2(len(self.letters)))
        self.S = math.ceil(math.log2(N))

    def subgroups(self):
        def close(gens):
            seen, q = {0}, [0]
            while q:
                p = q.pop()
                for g in gens:
                    z = self.mul[p][g]
                    if z not in seen:
                        seen.add(z)
                        q.append(z)
            return frozenset(seen)
        subs, front = {frozenset([0]): [0]}, [frozenset([0])]
        while front:
            H = front.pop()
            for a in range(self.N):
                if a not in H:
                    C = close(subs[H] + [a])
                    if C not in subs:
                        subs[C] = subs[H] + [a]
                        front.append(C)
        return list(subs)

    def conj_classes(self, subs):
        """one representative per conjugacy class"""
        seen, reps = set(), []
        for H in sorted(subs, key=len):
            if H in seen:
                continue
            reps.append(H)
            for g in range(self.N):
                seen.add(frozenset(self.mul[self.mul[self.inv[g]][h]][g] for h in H))
        return reps

    def right_cosets(self, O):
        """coset index of every element for O p"""
        lab, n = [-1] * self.N, 0
        for p in range(self.N):
            if lab[p] < 0:
                for h in O:
                    lab[self.mul[h][p]] = n
                n += 1
        return lab, n


def elementary(Gr, H):
    return all(Gr.mul[h][h] == 0 for h in H) and all(Gr.mul[x][y] == Gr.mul[y][x] for x in H for y in H)


def families(Gr, subs, H, K, slack):
    """subgroup families containing H with intersection H, blocks within fan-in K, smallest sum of bits"""
    cand = sorted([O for O in subs if H <= O and len(O) < Gr.N and math.ceil(math.log2(Gr.N // len(O))) + Gr.w <= K],
                  key=lambda O: -len(O))
    bits = {O: math.ceil(math.log2(Gr.N // len(O))) for O in cand}
    out, best = [], [None]

    def rec(start, inter, total, fam):
        if inter == H:
            if best[0] is None or total < best[0]:
                best[0] = total
            out.append((total, fam))
            return
        if best[0] is not None and total >= best[0] + slack:
            return
        for i in range(start, len(cand)):
            O = cand[i]
            nxt = inter & O
            if nxt != inter:
                rec(i + 1, nxt, total + bits[O], fam + [O])
    if H == frozenset(range(Gr.N)):
        return []
    rec(0, frozenset(range(Gr.N)), 0, [])
    return sorted([f for f in out if f[0] <= best[0] + slack], key=lambda f: f[0]) if out else []


def realize(Gr, H, fam, K, rng, labelings):
    """try block labelings; return a verified code (list of bit tuples) with fan-in <= K, or None"""
    k = int(math.log2(len(H)))
    Hl = sorted(H)
    cos_lab, nK = Gr.right_cosets(H)
    rep = [None] * nK
    for p in range(Gr.N):
        if rep[cos_lab[p]] is None:
            rep[cos_lab[p]] = p                                       # transversal t_k
    blocks = [(Gr.right_cosets(O), math.ceil(math.log2(Gr.N // len(O)))) for O in fam]
    c = sum(b for _, b in blocks)
    span = {0: ()}                                                    # coordinates of H ~ F2^k
    for h in Hl:
        if h not in span:
            span = {z: v + (0,) for z, v in span.items()} | {Gr.mul[z][h]: v + (1,) for z, v in span.items()}
    funcs = [f for f in it.product((0, 1), repeat=k) if any(f)]
    A = len(Gr.letters)
    nxt = [[cos_lab[Gr.mul[rep[ci]][a]] for a in Gr.letters] for ci in range(nK)]
    gam = [[Gr.mul[Gr.mul[rep[ci]][a]][Gr.inv[rep[nxt[ci][ai]]]] for ai, a in enumerate(Gr.letters)] for ci in range(nK)]
    dmax = K - Gr.w - 1
    for _ in range(labelings):
        perm = [rng.sample(range(2 ** b), n) for (_, n), b in blocks]
        code = [0] * nK                                               # core code of every H-coset
        for ci in range(nK):
            p, off, v = rep[ci], 0, 0
            for bi, ((lab, _), b) in enumerate(blocks):
                v |= perm[bi][lab[p]] << off
                off += b
            code[ci] = v
        chosen = []
        if k:
            if dmax < 0:
                return None
            sols = []
            for ell in funcs:
                done = False
                for d in range(dmax + 1):
                    for D in it.combinations(range(c), d):
                        pats = {}
                        for ci in range(nK):
                            pats.setdefault(tuple((code[ci] >> b) & 1 for b in D), len(pats))
                        rows = []
                        for ci in range(nK):
                            pi = pats[tuple((code[ci] >> b) & 1 for b in D)]
                            for ai in range(A):
                                rhs = sum(x & y for x, y in zip(ell, span[gam[ci][ai]])) & 1
                                rows.append(((1 << ci) ^ (1 << nxt[ci][ai]) ^ (1 << (nK + pi * A + ai)), rhs))
                        x = cf.gf2_solve(rows, nK + len(pats) * A)
                        if x is not None:
                            sols.append((ell, [(x >> ci) & 1 for ci in range(nK)]))
                            done = True
                            break
                    if done:
                        break
            for ell, beta in sols:                                    # k independent functionals
                cand = [e for e, _ in chosen] + [ell]
                if len({tuple(sum(cc * v[i] for cc, v in zip(cs, cand)) % 2 for i in range(k))
                        for cs in it.product((0, 1), repeat=len(cand))}) == 2 ** len(cand):
                    chosen.append((ell, beta))
            if len(chosen) < k:
                continue
            chosen = chosen[:k]
        E = []
        for p in range(Gr.N):
            ci = cos_lab[p]
            h = Gr.mul[p][Gr.inv[rep[ci]]]                            # p = h t
            E.append(tuple((code[ci] >> b) & 1 for b in range(c)) +
                     tuple((sum(x & y for x, y in zip(ell, span[h])) & 1) ^ beta[ci] for ell, beta in chosen))
        assert len(set(E)) == Gr.N
        fan, deps = verify(Gr, E, K)
        if max(fan) <= K:
            return E, fan, deps
    return None


def verify(Gr, E, K=None):
    """exact fan-in of every next bit (fewest state bits + letter bits that determine it); > K if above K"""
    step = [[Gr.mul[p][a] for a in Gr.letters] for p in range(Gr.N)]
    S, out, deps = len(E[0]), [], []
    for j in range(S):
        f = {(E[p], a): E[step[p][a]][j] for p in range(Gr.N) for a in range(len(Gr.letters))}
        for r in range(S + 1 if K is None else min(S, K - Gr.w) + 1):
            ok = None
            for sub in it.combinations(range(S), r):
                g = {}
                if all(g.setdefault((tuple(cd[i] for i in sub), a), v) == v for (cd, a), v in f.items()):
                    ok = sub
                    break
            if ok is not None:
                out.append(len(ok) + Gr.w)
                deps.append(list(ok))
                break
        else:
            out.append(K + 1)
        if out[-1] > (K or out[-1]):
            return out, deps
    return out, deps


def study(name, Ks, labelings, slack, rng):
    t0 = time.time()
    Gr = Group(cf.groups()[name])
    subs = Gr.subgroups()
    Hs = [H for H in Gr.conj_classes(subs) if elementary(Gr, H)]
    res = dict(group=name, order=Gr.N, w=Gr.w, lower_bound=Gr.S, n_subgroups=len(subs), results={})
    for K in Ks:
        best = None
        for H in sorted(Hs, key=len, reverse=True):
            k = int(math.log2(len(H)))
            for total, fam in families(Gr, subs, H, K, slack)[:12]:
                S = total + k
                if best is not None and S >= best["S"]:
                    continue
                if S < Gr.S:
                    continue
                r = realize(Gr, H, fam, K, rng, labelings if k else 1)
                if r:
                    E, fan, deps = r
                    best = dict(S=S, fanin=max(fan), H_order=len(H), blocks=[len(O) for O in fam],
                                block_bits=[math.ceil(math.log2(Gr.N // len(O))) for O in fam], fiber_bits=k,
                                deps=deps, codes=[sum(b << i for i, b in enumerate(c)) for c in E])
                    if S == Gr.S:
                        break
            if best is not None and best["S"] == Gr.S:
                break
        res["results"][str(K)] = best
        print(name, "K", K, json.dumps({k: v for k, v in (best or {}).items() if k not in ("deps", "codes")}),
              round(time.time() - t0, 1), "s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="S4+A5+PSL(2,7)+A6")
    ap.add_argument("--K", default="4,5,6,7")
    ap.add_argument("--labelings", type=int, default=40)
    ap.add_argument("--slack", type=int, default=1, help="also try families up to this many bits above the best")
    a = ap.parse_args()
    rng = random.Random(0)
    R = [study(n, [int(k) for k in a.K.split(",")], a.labelings, a.slack, rng) for n in a.groups.split("+")]
    json.dump(R, open(os.path.join(ROOT, "model", "fbseq", "realize.json"), "w"), indent=1)
    path = os.path.join(ROOT, "model", "fbseq", "group_codes.json")        # for loop_bench.py gc:<group>_K<K>
    specs = [r for r in json.load(open(path)) if r["code"] != "realize"] if os.path.exists(path) else []
    for res in R:
        Gr = Group(cf.groups()[res["group"]])
        for K, b in res["results"].items():
            if b and int(K) >= 5:
                specs.append(dict(name=f"{res['group']}_K{K}", group=res["group"], code="realize", S=b["S"], w=Gr.w,
                                  letters=len(Gr.letters), fanin=b["fanin"], deps=b["deps"], codes=b["codes"],
                                  start=b["codes"][0], step=[[Gr.mul[p][x] for x in Gr.letters] for p in range(Gr.N)]))
    json.dump(specs, open(path, "w"))


if __name__ == "__main__":
    main()
