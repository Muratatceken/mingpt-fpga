"""
Smallest state for a one-level feedback realization of the A5 word problem (problem C, small case).

    python3 theory_sat.py [--K 4,5,6] [--S 6-15] [--timeout 900]     (needs z3: pip install z3-solver)
                                                                       -> model/fbseq/sat_a5.json
    python3 theory_sat.py --K 5 --S 6-6 --coding 0,1,2 --out sat_a5_ctrl_a.json
        control inputs chosen per next bit (|D_j| + |C_j| <= K) for the letter coding a=00, b=01, e=10
Question: is there an injective encoding e: A5 -> {0,1}^S such that, for every letter a of the task
alphabet (two generators and the identity, control word of w = 2 bits), every next-state bit j is a
function of the control word and of at most K - w state bits D_j?  Then each next-state bit is one
LUT-K and the feedback loop is one LUT level.  The task's update is p_t = p_{t-1} o a_t (fbseq.A5_TABLE).
Condition, for all p, q and every letter a:  e(p)|D_j = e(q)|D_j  =>  e(p o a)_j = e(q o a)_j.
(The identity letter forces e(p)_j to be a function of D_j.)  Lower bound: S >= ceil(log2 60) = 6
(Myhill-Nerode: prefixes of different group elements are separated by the suffix of the inverse).
"""
import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def a5():
    perms = [p for p in itertools.permutations(range(5))
             if sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0]
    index = {p: i for i, p in enumerate(perms)}
    gens = [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4), (0, 1, 2, 3, 4)]
    step = [[index[tuple(p[g[i]] for i in range(5))] for p in perms] for g in gens]   # p o g
    return len(perms), step, index[(0, 1, 2, 3, 4)]


def solve(S, K, w=2, timeout=900, coding=None, fix=None):
    """coding=None: every next bit reads both control bits (|D_j| <= K - w).  coding = (c(a), c(b), c(e)), the
    letters' 2-bit codes: each next bit also chooses which control bits it reads, |D_j| + |C_j| <= K, and
    pairs with different letters are compared too (codes not listed are don't-care)"""
    import z3
    N, step, ident = a5()
    E = [[z3.Bool(f"e{p}_{b}") for b in range(S)] for p in range(N)]
    D = [[z3.Bool(f"d{j}_{i}") for i in range(S)] for j in range(S)]
    C = [[z3.Bool(f"c{j}_{i}") for i in range(w)] for j in range(S)]
    s = z3.Solver()
    s.set("timeout", timeout * 1000)
    for b in range(S):                                           # symmetry breaking: e(identity) = 0
        s.add(z3.Not(E[ident][b]))
    for p, code in enumerate(fix or []):                         # test hook: a fixed encoding
        for b in range(S):
            s.add(E[p][b] == bool((code >> b) & 1))
    for j in range(S):
        if coding is None:
            s.add(z3.PbLe([(D[j][i], 1) for i in range(S)], K - w))
        else:
            s.add(z3.PbLe([(D[j][i], 1) for i in range(S)] + [(C[j][i], 1) for i in range(w)], K))
    pairs = list(itertools.combinations(range(N), 2))
    for p, q in pairs:                                           # injective
        s.add(z3.Or([z3.Xor(E[p][b], E[q][b]) for b in range(S)]))
    L = len(step)
    for j in range(S):
        for p, q in pairs + ([(p, p) for p in range(N)] if coding is not None else []):
            agree = z3.Bool(f"a{j}_{p}_{q}")                     # p, q look the same to bit j
            s.add(agree == z3.And([z3.Implies(D[j][i], E[p][i] == E[q][i]) for i in range(S)]))
            for x in range(L):
                for y in range(L):
                    if x == y and p == q:
                        continue
                    if coding is None and x != y:
                        continue
                    diff = [i for i in range(w) if ((coding[x] ^ coding[y]) >> i) & 1] if x != y else []
                    s.add(z3.Implies(z3.And([agree] + [z3.Not(C[j][i]) for i in diff]),
                                     E[step[x][p]][j] == E[step[y][q]][j]))
    t0 = time.time()
    r = s.check()
    res = dict(S=S, K=K, w=w, result=str(r), seconds=round(time.time() - t0, 1))
    if coding is not None:
        res["coding"] = list(coding)
    if r == z3.sat:
        m = s.model()
        res["deps"] = [[i for i in range(S) if z3.is_true(m.eval(D[j][i]))] for j in range(S)]
        if coding is not None:
            res["control_deps"] = [[i for i in range(w) if z3.is_true(m.eval(C[j][i]))] for j in range(S)]
        res["encoding"] = ["".join("1" if z3.is_true(m.eval(E[p][b])) else "0" for b in range(S)) for p in range(N)]
        res["verified"] = verify(res, step)
    return res


def verify(res, step):
    """independent check of a solution: injective and every next bit a function of (letter, D_j), or with a
    coding of (the chosen control bits of the letter's code, D_j)"""
    enc = [tuple(int(c) for c in e) for e in res["encoding"]]
    if len(set(enc)) != len(enc):
        return False
    if "coding" in res:
        for j, (dep, cd) in enumerate(zip(res["deps"], res["control_deps"])):
            seen = {}
            for x, st in enumerate(step):
                for p, e in enumerate(enc):
                    key = (tuple(e[i] for i in dep), tuple((res["coding"][x] >> i) & 1 for i in cd))
                    if seen.setdefault(key, enc[st[p]][j]) != enc[st[p]][j]:
                        return False
        return True
    for j, dep in enumerate(res["deps"]):
        for st in step:
            seen = {}
            for p, e in enumerate(enc):
                key = tuple(e[i] for i in dep)
                if seen.setdefault(key, enc[st[p]][j]) != enc[st[p]][j]:
                    return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", default="4,5,6")
    ap.add_argument("--S", default="6-15")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--coding", default="", help="e.g. 0,1,2 = codes of a, b, e: control inputs chosen per bit")
    ap.add_argument("--out", default="sat_a5.json")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.S.split("-"))
    coding = tuple(int(c) for c in a.coding.split(",")) if a.coding else None
    path = os.path.join(ROOT, "model", "fbseq", a.out)
    out = json.load(open(path)) if os.path.exists(path) else []
    for K in (int(k) for k in a.K.split(",")):
        for S in range(lo, hi + 1):
            r = solve(S, K, timeout=a.timeout, coding=coding)
            print(json.dumps({k: v for k, v in r.items() if k != "encoding"}), flush=True)
            out = [o for o in out if (o["S"], o["K"]) != (S, K)] + [r]
            json.dump(sorted(out, key=lambda o: (o["K"], o["S"])), open(path, "w"), indent=1)
            if r["result"] == "sat":
                break                                            # smallest S for this K found


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- block-local realizations
def perm_group():
    perms = [p for p in itertools.permutations(range(5))
             if sum(p[i] > p[j] for i in range(5) for j in range(i + 1, 5)) % 2 == 0]
    return perms, {p: i for i, p in enumerate(perms)}


def compose(p, q):
    """(p o q)(i) = p(q(i))"""
    return tuple(p[q[i]] for i in range(5))


def inverse(p):
    return tuple(sorted(range(5), key=lambda i: p[i]))


def subgroups():
    """all subgroups of A5 (brute force over generated subgroups of pairs)"""
    perms, _ = perm_group()
    found = set()
    for a in perms:
        for b in perms:
            H = {tuple(range(5))}
            frontier = [a, b]
            while frontier:
                x = frontier.pop()
                if x not in H:
                    H.add(x)
                    frontier += [compose(x, y) for y in list(H)] + [compose(y, x) for y in list(H)]
            found.add(frozenset(H))
    return sorted(found, key=len)


def coset_blocks(subs):
    """block i holds the right coset H_i p; it updates by p -> p o a (well defined); returns the tables and
    whether the blocks determine p (intersection of the H_i trivial)"""
    perms, index = perm_group()
    gens = [(1, 2, 3, 4, 0), (1, 2, 0, 3, 4), (0, 1, 2, 3, 4)]
    blocks = []
    for H in subs:
        label, cosets = {}, []
        for p in perms:
            if p in label:
                continue
            c = frozenset(compose(h, p) for h in H)
            for q in c:
                label[q] = len(cosets)
            cosets.append(c)
        nxt = []
        for g in gens:                                   # coset transition, checked to be well defined
            t = {}
            for p in perms:
                a, b = label[p], label[compose(p, g)]
                assert t.setdefault(a, b) == b
            nxt.append([t[c] for c in range(len(cosets))])
        bits = max(1, (len(cosets) - 1).bit_length())
        blocks.append(dict(order=len(H), index=len(cosets), bits=bits, label=label, next=nxt))
    inter = set(perms)
    for H in subs:
        inter &= set(H)
    return blocks, len(inter) == 1


def best_block_local(K, w=2):
    """smallest total bits of a family of subgroups with trivial intersection, each block dense
    (every bit reads its whole block): bits_i + w <= K"""
    subs = [H for H in subgroups() if 1 < len(H) < 60 or len(H) == 1]
    perms, _ = perm_group()
    cands = []
    for H in subs:
        idx = 60 // len(H)
        b = max(1, (idx - 1).bit_length())
        if b + w <= K:
            cands.append((b, H))
    best = None
    for r in range(1, 5):
        for fam in itertools.combinations(cands, r):
            tot = sum(b for b, _ in fam)
            if best is not None and tot >= best[0]:
                continue
            inter = set(perms)
            for _, H in fam:
                inter &= set(H)
            if len(inter) == 1:
                best = (tot, [len(H) for _, H in fam], [H for _, H in fam])
    return best


def check_family(subs, n_seq=2000, length=1024, seed=0):
    """exhaustive transition check is the well-definedness assert above; here: random A5 words"""
    import numpy as np
    sys.path.insert(0, HERE)
    import fbseq
    blocks, ok = coset_blocks(subs)
    perms, index = perm_group()
    rng = np.random.default_rng(seed)
    x, y = fbseq.task("a5", n_seq, length, rng)
    ident = tuple(range(5))
    decode = {}
    for p in perms:
        decode[tuple(b["label"][p] for b in blocks)] = index[p]
    state = [np.full(n_seq, b["label"][ident]) for b in blocks]
    correct = 0
    for t in range(length):
        state = [np.array(b["next"])[x[:, t], s] for b, s in zip(blocks, state)]
        pred = np.array([decode[k] for k in zip(*state)])
        correct += int((pred == y[:, t]).sum())
    return dict(determines_p=ok, bits=[b["bits"] for b in blocks], orders=[b["order"] for b in blocks],
                accuracy=correct / x.size, sequences=n_seq, length=length)
