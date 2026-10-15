"""Build web/streamomr.html: ONE self-contained page that runs the JavaScript port of the StreamOMR recognizer
(web/src/*.js) in a Web Worker, no Pyodide, no network (works from file:// and https).

    .venv/bin/python scripts/build_web.py          # web/streamomr.html
    .venv/bin/python scripts/build_web.py --dev    # web/streamomr-dev.html: + development logging (web/dev.js), research test page only

Inputs: web/src/*.js (load order below), web/{template.html,app.js,worker.js,dev.js}, app/static/style.css.
The worker is created from a Blob URL of the inlined source (<script type="text/plain" id="worker-src">): recognizer + scanner.js + worker.js.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ORDER = ["util", "markers", "layout", "types", "identity", "decide", "fusion", "revision", "stream", "cv", "aruco", "cvx",
             "fastops", "ink", "illum", "tracks", "detect", "recognize", "scanner"]
MAIN_ORDER = ["util", "markers", "layout", "sheet"]     # the sheet maker runs on the main thread
CSS_KEEP = re.compile(r"^(:root|\*|body|a |h1|\.card|label|input|button|\.row|\.scan|@media \(max-width:760px\)|\.stage|\.banner|\.state|\.hint|\.stats|ul\.scans|\.badge|\.mut|\.small|\.err|\.tag)")


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


def css():
    lines = [l for l in read("app", "static", "style.css").splitlines() if CSS_KEEP.match(l)]
    extra = ".loading { position:fixed; inset:0; background:rgba(17,20,24,.96); display:flex; align-items:center; justify-content:center; text-align:center; padding:24px; z-index:10; }\n.loading[hidden] { display:none; }\n.tag.warn { background:var(--warn); color:#000; }\n.badge.needs_review { background:#f0d54a; }\n"
    return "\n".join(lines) + "\n" + extra


def esc(s):
    return s.replace("</", "<\\/")


def worker_source(dev=False):
    parts = []
    for m in SRC_ORDER:
        s = read("web", "src", m + ".js")
        if m == "scanner":
            marker = "var DEV = false;  // __DEV__"
            assert marker in s
            if dev:
                s = s.replace(marker, "var DEV = true;  // __DEV__")
        parts.append(f"// ==== web/src/{m}.js\n{s}")
    parts.append("// ==== web/worker.js\n" + read("web", "worker.js"))
    return "\n".join(parts)


def build(dev=False):
    html = read("web", "template.html")
    for key, val in (("/*__CSS__*/", css()), ("/*__WORKER__*/", esc(worker_source(dev))),
                     ("/*__SHEET__*/", esc("\n".join(read("web", "src", m + ".js") for m in MAIN_ORDER))),
                     ("/*__MAKER__*/", esc(read("web", "maker.js"))),
                     ("/*__APP__*/", esc(read("web", "app.js")).replace("/*__DEV__*/false", "true" if dev else "false")),
                     ("/*__DEVJS__*/", esc(read("web", "dev.js")) if dev else "")):
        assert key in html, key
        html = html.replace(key, val)
    out = os.path.join(ROOT, "web", "streamomr-dev.html" if dev else "streamomr.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out, len(html.encode("utf-8"))


if __name__ == "__main__":
    dev = "--dev" in sys.argv[1:]
    path, n = build(dev)
    print(f"wrote {path} ({n / 1024:.0f} KB)")
