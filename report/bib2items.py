"""Turn the verified BibTeX entries cited by the report (paper/refs.bib) into Turkish \\bibitem lines:
   python3 report/bib2items.py  ->  report/bolumler/kaynaklar_ek.tex"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
bib = open(os.path.join(HERE, "..", "paper", "refs.bib")).read()
used = set()
for f in os.listdir(os.path.join(HERE, "bolumler")):
    if f.endswith(".tex"):
        for m in re.finditer(r"\\cite\{([^}]*)\}", open(os.path.join(HERE, "bolumler", f)).read()):
            used |= {k.strip() for k in m.group(1).split(",")}
existing = set(re.findall(r"\\bibitem\{([^}]*)\}", open(os.path.join(HERE, "bolumler", "kaynaklar.tex")).read()))


def fields(body):
    out = {}
    for m in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
        out[m.group(1).lower()] = re.sub(r"\s+", " ", m.group(2)).strip()
    return out


def authors(a):
    names = [n.strip() for n in a.split(" and ")]
    def short(n):
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
        else:
            parts = n.split()
            last, first = parts[-1], " ".join(parts[:-1])
        ini = " ".join(p[0] + "." for p in first.replace("-", " ").split() if p and p[0].isalpha())
        return f"{ini} {last}".strip()
    s = [short(n) for n in names[:3]]
    return ", ".join(s) + (" ve ark." if len(names) > 3 else "")


items = []
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S):
    kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
    if key not in used or key in existing:
        continue
    f = fields(body)
    venue = f.get("booktitle") or f.get("journal") or f.get("howpublished") or f.get("publisher") or ""
    if f.get("eprint") and not venue:
        venue = f"arXiv:{f['eprint']}"
    title = f.get("title", "").replace("{", "").replace("}", "")
    url = f" \\url{{{f['url']}}}" if kind == "misc" and f.get("url") else ""
    items.append(f"\\bibitem{{{key}}} {authors(f.get('author', ''))}, ``{title},'' "
                 f"\\emph{{{venue}}}, {f.get('year', '')}.{url}")
open(os.path.join(HERE, "bolumler", "kaynaklar_ek.tex"), "w").write("\n".join(items) + "\n")
print(len(items), "bibitems for", len(used), "cited keys")
