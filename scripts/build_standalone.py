"""Build standalone/streamomr.html: ONE self-contained page that runs the current omr/ recognizer in the
browser (Pyodide: CPython + numpy + OpenCV in WebAssembly), no server.  Also copies it to
app/static/standalone.html so the normal HTTPS server can serve it to a phone.

    .venv/bin/python scripts/build_standalone.py          # the normal page (also copied to app/static/standalone.html)
    .venv/bin/python scripts/build_standalone.py --dev    # standalone/streamomr-dev.html: + development logging (standalone/dev.js), research test page only

Inputs: omr/*.py (embedded as text), standalone/{template.html,app.js,worker.js,glue.py}, app/static/style.css.
Not embedded (not needed to scan): omr/db.py, omr/service.py, omr/render_pdf.py, omr/review.py.
"""
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYODIDE_VERSION = "0.27.7"      # ships numpy 2.0.2 + opencv-python 4.10 (with cv2.aruco)
PYODIDE_BASE = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
OMR_MODULES = ["__init__", "markers", "layout", "detect", "fastops", "illum", "ink", "tracks", "recognize", "fusion", "identity", "revision", "stream"]
CSS_KEEP = re.compile(r"^(:root|\*|body|a |h1|\.card|label|input|button|\.row|\.scan|@media \(max-width:760px\)|\.stage|\.banner|\.state|\.hint|\.stats|ul\.scans|\.badge|\.mut|\.small|\.err|\.tag)")


def read(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as f:
        return f.read()


def css():
    lines = [l for l in read("app", "static", "style.css").splitlines() if CSS_KEEP.match(l)]
    extra = ".loading { position:fixed; inset:0; background:rgba(17,20,24,.96); display:flex; align-items:center; justify-content:center; text-align:center; padding:24px; z-index:10; }\n.loading[hidden] { display:none; }\n.tag.warn { background:var(--warn); color:#000; }\n.badge.needs_review { background:#f0d54a; }\n"
    return "\n".join(lines) + "\n" + extra


def build(dev=False):
    sources = {f"omr/{m}.py": read("omr", f"{m}.py") for m in OMR_MODULES}
    glue = read("standalone", "glue.py")
    if dev:
        assert "DEV = False  # __DEV__" in glue
        glue = glue.replace("DEV = False  # __DEV__", "DEV = True  # __DEV__")
    sources["omr_web.py"] = glue
    blob = json.dumps(sources, ensure_ascii=False).replace("</", "<\\/")
    html = read("standalone", "template.html")
    # plain str.replace with function-free values (no regex escapes in the payload)
    for key, val in (("/*__CSS__*/", css()), ("/*__WORKER__*/", read("standalone", "worker.js").replace("</", "<\\/")),
                     ("/*__SOURCES__*/", blob), ("/*__PYODIDE_BASE__*/", PYODIDE_BASE),
                     ("/*__APP__*/", read("standalone", "app.js").replace("</", "<\\/").replace("/*__DEV__*/false", "true" if dev else "false")),
                     ("/*__DEVJS__*/", read("standalone", "dev.js").replace("</", "<\\/") if dev else "")):
        assert key in html, key
        html = html.replace(key, val)
    out = os.path.join(ROOT, "standalone", "streamomr-dev.html" if dev else "streamomr.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    if not dev:     # the dev build is for the research test page only: never served by the app or the main Pages site
        shutil.copyfile(out, os.path.join(ROOT, "app", "static", "standalone.html"))
    return out, len(html.encode("utf-8"))


if __name__ == "__main__":
    dev = "--dev" in sys.argv[1:]
    path, n = build(dev)
    print(f"wrote {path} ({n / 1024:.0f} KB)" + ("" if dev else " and app/static/standalone.html"))
