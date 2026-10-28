#!/usr/bin/env python3
"""Yerel sunum: Slides kaynaklarından (kaynak/deck.json + kaynak/slides/*.html) tek dosyalık bir HTML, PDF ve
konuşma metni üretir.

  python3 sunum/yerel_sunum.py sunum/uygulama          -> sunum.html, konusma_metni.md
  python3 sunum/yerel_sunum.py --md docs/07_yol_haritasi.md -> docs/07_yol_haritasi.pdf (tek bir Markdown)
  python3 sunum/yerel_sunum.py sunum/uygulama --pdf    -> ayrıca sunum.pdf, konusma_metni.pdf (Chrome headless);
                                                          klasördeki terimler.md ve konusma_notlari.md de PDF olur

- Fontlar sunum/sunum.html'deki gömülü @font-face bloğundan alınır (Source Serif 4, IBM Plex Sans, IBM Plex Mono).
- /_blob/<id> görselleri kaynak/assets/assets.json eşlemesiyle data URI olarak gömülür.
- <aside> konuşmacı notları slayttan çıkarılır, konuşma metnine yazılır.
- HTML'de P tuşu (ya da "Sun" düğmesi) tam ekran sunum modunu açar; oklar ilerletir, Esc çıkar.
"""
import base64, html, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """*{box-sizing:border-box}
html,body{margin:0;background:#2B2F36}
h1,h2,h3,p,ul,ol,table{margin:0}
section ul,section ol{padding-left:1.1em}
section{position:relative;width:1920px;height:1080px;overflow:hidden;flex:none;
  -webkit-print-color-adjust:exact;print-color-adjust:exact}
section table{border-collapse:collapse;width:100%}
section th,section td{padding:0.35em 0.6em;border-bottom:1px solid rgba(120,120,120,.35);vertical-align:top}
section th{font-weight:600}
x-shape{display:block;flex:none}
x-shape[kind=arrow-right]{clip-path:polygon(0 30%,60% 30%,60% 0,100% 50%,60% 100%,60% 70%,0 70%)}
x-shape[kind=arrow-left]{clip-path:polygon(100% 30%,40% 30%,40% 0,0 50%,40% 100%,40% 70%,100% 70%)}
x-shape[kind=arrow-down]{clip-path:polygon(30% 0,70% 0,70% 60%,100% 60%,50% 100%,0 60%,30% 60%)}
x-shape[kind=arrow-up]{clip-path:polygon(30% 100%,70% 100%,70% 40%,100% 40%,50% 0,0 40%,30% 40%)}
.deck{display:flex;flex-direction:column;align-items:center;gap:24px;padding:24px 0}
.wrap{width:1920px;height:1080px;box-shadow:0 4px 24px rgba(0,0,0,.35)}
.bar{position:fixed;top:12px;right:16px;z-index:10;font:14px -apple-system,Arial,sans-serif;color:#E6EAF2}
.bar button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid #5A6474;background:#14213D;color:#E6EAF2;cursor:pointer}
body.present{background:#000;overflow:hidden}
body.present .bar{display:none}
body.present .deck{display:block;padding:0}
body.present .wrap{display:none;position:fixed;left:50%;top:50%;box-shadow:none}
body.present .wrap.on{display:block}
@page{size:1920px 1080px;margin:0}
@media print{
  html,body{background:none}
  .bar{display:none}
  .deck{display:block;padding:0}
  .wrap{zoom:1 !important;transform:none !important;box-shadow:none;break-after:page}
}"""

JS = """const W = [...document.querySelectorAll('.wrap')];
let cur = 0;
function fitScroll() {
  const z = Math.min(1, (window.innerWidth - 48) / 1920);
  W.forEach(w => { w.style.transform = ''; w.style.zoom = z; });
}
function fitPresent() {
  const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  W.forEach((w, i) => {
    w.classList.toggle('on', i === cur);
    w.style.zoom = 1;
    w.style.transform = 'translate(-50%,-50%) scale(' + s + ')';
  });
}
function layout() { document.body.classList.contains('present') ? fitPresent() : fitScroll(); }
function present(on) {
  document.body.classList.toggle('present', on);
  if (on) {
    cur = Math.max(0, W.findIndex(w => w.getBoundingClientRect().bottom > 0));
    if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  } else {
    if (document.fullscreenElement) document.exitFullscreen();
    W.forEach(w => w.classList.remove('on'));
  }
  layout();
  if (!on) W[cur].scrollIntoView();
}
document.getElementById('sun').onclick = e => { e.stopPropagation(); present(true); };
document.addEventListener('keydown', e => {
  const p = document.body.classList.contains('present');
  if (!p && (e.key === 'p' || e.key === 'P')) return present(true);
  if (!p) return;
  if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(e.key)) cur = Math.min(W.length - 1, cur + 1);
  else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(e.key)) cur = Math.max(0, cur - 1);
  else if (e.key === 'Home') cur = 0;
  else if (e.key === 'End') cur = W.length - 1;
  else if (e.key === 'Escape') return present(false);
  else return;
  e.preventDefault();
  fitPresent();
});
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement && document.body.classList.contains('present')) present(false);
});
document.addEventListener('click', e => {
  if (document.body.classList.contains('present')) { cur = Math.min(W.length - 1, cur + 1); fitPresent(); }
});
window.addEventListener('resize', layout);
layout();"""

NOTES_CSS = """@page{size:A4;margin:22mm 20mm}
body{font-family:'IBM Plex Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.5;color:#1F2633}
h1{font-family:'Source Serif 4',Georgia,serif;font-size:20pt;color:#14213D;margin:0 0 10pt}
h2{font-family:'Source Serif 4',Georgia,serif;font-size:12.5pt;color:#14213D;margin:16pt 0 4pt;break-after:avoid}
p{margin:0 0 6pt}
hr{border:none;border-top:1px solid #C9C4B8;margin:12pt 0}
h3{font-size:11pt;color:#B4460F;margin:12pt 0 3pt;break-after:avoid}
table{border-collapse:collapse;margin:4pt 0 10pt}
th,td{border-bottom:1px solid #D8D3C8;padding:3pt 8pt;text-align:left}
th{color:#14213D}"""

EXTRA_MD = ["terimler.md", "konusma_notlari.md"]   # hand-written notes rendered next to the deck


def fonts_css():
    """the @font-face block (data URIs) of sunum/sunum.html"""
    s = open(os.path.join(ROOT, "sunum", "sunum.html"), encoding="utf-8").read()
    blocks = re.findall(r"<style>(.*?)</style>", s, re.S)
    face = [b for b in blocks if "@font-face" in b]
    if not face:
        sys.exit("sunum/sunum.html: @font-face bloğu bulunamadı")
    return face[0]


def text_of(fragment):
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def write_atomic(path, data, mode="w"):
    """temp file + rename (a PDF open in a viewer is not overwritten in place)"""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, mode, **({"encoding": "utf-8"} if mode == "w" else {})) as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def chrome_pdf(src_html, out_pdf):
    prof = tempfile.mkdtemp(prefix="chrome-prof-")
    tmp = out_pdf + ".tmp.pdf"
    r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                        "--virtual-time-budget=15000", f"--print-to-pdf={tmp}", "file://" + os.path.abspath(src_html)],
                       capture_output=True, text=True)
    # Chrome >= 156 may exit non-zero after writing the PDF ("Teardown watchdog expired"): trust the file
    if not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
        raise RuntimeError(f"Chrome did not write {tmp} (exit {r.returncode}):\n{r.stderr[-2000:]}")
    os.replace(tmp, out_pdf)


def render_md(path):
    """a single Markdown file -> PDF next to it (same fonts and style as the notes)"""
    import markdown
    path = os.path.abspath(path)
    body = markdown.markdown(open(path, encoding="utf-8").read(), extensions=["tables"])
    nh = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8"><style>{fonts_css()}</style>'
          f"<style>{NOTES_CSS}</style></head><body>{body}</body></html>")
    tmp_html = path[:-3] + ".tmp.html"
    open(tmp_html, "w", encoding="utf-8").write(nh)
    try:
        chrome_pdf(tmp_html, path[:-3] + ".pdf")
    finally:
        os.remove(tmp_html)
    print(path[:-3] + ".pdf")


def main():
    if sys.argv[1] == "--md":
        return render_md(sys.argv[2])
    d = os.path.abspath(sys.argv[1])
    src = os.path.join(d, "kaynak")
    deck = json.load(open(os.path.join(src, "deck.json"), encoding="utf-8"))
    assets = {}
    amap = os.path.join(src, "assets", "assets.json")
    if os.path.exists(amap):
        for bid, fn in json.load(open(amap)).items():
            data = open(os.path.join(src, "assets", fn), "rb").read()
            assets[bid] = "data:image/png;base64," + base64.b64encode(data).decode()

    slides, notes = [], []
    for i, sid in enumerate(deck["order"], 1):
        s = open(os.path.join(src, "slides", sid + ".html"), encoding="utf-8").read().strip()
        m = re.search(r"<aside>(.*?)</aside>", s, re.S)
        note = text_of(m.group(1)) if m else ""
        s = re.sub(r"\s*<aside>.*?</aside>", "", s, flags=re.S)
        s = re.sub(r'src="/_blob/([0-9a-f]{32})"', lambda mm: f'src="{assets[mm.group(1)]}"', s)
        title = re.search(r"<h[12][^>]*>(.*?)</h[12]>", s, re.S)
        notes.append((i, text_of(title.group(1)) if title else sid, note))
        slides.append(f'<div class="wrap">{s}</div>')

    page = (f'<!doctype html>\n<html lang="tr">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{html.escape(deck["title"])}</title>\n<style>{fonts_css()}</style>\n<style>\n{CSS}\n</style>\n'
            f'</head>\n<body>\n<div class="bar"><button id="sun">Sun (P)</button></div>\n<div class="deck">\n'
            + "\n".join(slides) + f"\n</div>\n<script>\n{JS}\n</script>\n</body>\n</html>\n")
    out_html = os.path.join(d, "sunum.html")
    write_atomic(out_html, page)

    secs = sorted(deck.get("sections", {}).values(), key=lambda v: deck["order"].index(v["start"]))
    md = [f"# {deck['title']} · konuşma metni", "",
          "**Nasıl kullanılır:** Her başlık bir slayta karşılık gelir; numaralar slayt numarasıdır. Aynı metin "
          "sunumda her slaytın konuşmacı notunda da var.", "",
          "**Akış:** " + " · ".join(f"{deck['order'].index(v['start']) + 1}: {v['description']}" for v in secs),
          "", "---", ""]
    for i, title, note in notes:
        md += [f"## {i}. {title}", "", note, ""]
    out_md = os.path.join(d, "konusma_metni.md")
    write_atomic(out_md, "\n".join(md))
    print(f"{out_html}: {len(slides)} slayt; {out_md}")

    if "--pdf" in sys.argv:
        chrome_pdf(out_html, os.path.join(d, "sunum.pdf"))
        import markdown
        docs = [("konusma_metni", "\n".join(md))]
        docs += [(fn[:-3], open(os.path.join(d, fn), encoding="utf-8").read())
                 for fn in EXTRA_MD if os.path.exists(os.path.join(d, fn))]
        for name, text in docs:
            body = markdown.markdown(text, extensions=["tables"])
            nh = (f'<!doctype html><html lang="tr"><head><meta charset="utf-8"><style>{fonts_css()}</style>'
                  f"<style>{NOTES_CSS}</style></head><body>{body}</body></html>")
            tmp_html = os.path.join(d, f".{name}.tmp.html")
            open(tmp_html, "w", encoding="utf-8").write(nh)
            try:
                chrome_pdf(tmp_html, os.path.join(d, name + ".pdf"))
            finally:
                os.remove(tmp_html)
        print(f"{d}/sunum.pdf; " + ", ".join(f"{d}/{n}.pdf" for n, _ in docs))


if __name__ == "__main__":
    main()
