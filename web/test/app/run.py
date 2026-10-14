"""One command for the J4 app tests:   .venv/bin/python web/test/app/run.py [--frames 60] [--stride 9] [--no-dev] [--keep]

1. builds web/streamomr.html and web/streamomr-dev.html (scripts/build_web.py),
2. syntax-checks (jsc, new Function) every inline <script> of both built files,
3. unit checks of scanner.js against cv2: RGBA->grey bit-exact, JPEG encoder decodes (cv2.imdecode) to within cv2's own error,
4. runs the BUILT worker source under jsc with a fake self/postMessage on RGBA frames of the 2026-10-14 video (read here, written
   only to web/test/fixtures/app/, deleted afterwards) and compares every posted result with standalone/glue.py Scanner on the same frames
   (same fake stream clock): status, state, hint, overlay, commits/revisions (summary, detail), missed, dev trace (dev build).
Exit code 0 when all acceptance criteria hold."""
import argparse, base64, concurrent.futures as cf, json, math, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "standalone")]
JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
VIDEO = os.path.join(os.environ.get("STREAMOMR_REALASSET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "realasset")), "devlog_2026-10-14", "video", "segment_01.mp4")
FIX = os.path.join(ROOT, "web", "test", "fixtures", "app")
SRC = ["util", "markers", "layout", "types", "identity", "decide", "fusion", "revision", "stream", "cv", "aruco", "cvx",
       "fastops", "ink", "illum", "tracks", "detect", "recognize", "scanner"]

import numpy as np, cv2

FAILS = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def jsc(args, files=()):
    return subprocess.run([JSC, *files, *args], capture_output=True, text=True, cwd=ROOT)


# ------------------------------------------------------------------ 1+2: build and syntax
def scripts_of(html):
    return [(m.group(1), m.group(2)) for m in re.finditer(r"<script([^>]*)>(.*?)</script>", html, re.S)]


def build_and_syntax():
    import build_web
    outs = {}
    for dev in (False, True):
        path, n = build_web.build(dev)
        html = open(path, encoding="utf-8").read()
        outs[dev] = (path, html)
        print(f"built {os.path.relpath(path, ROOT)}: {n / 1024:.0f} KB")
        files = []
        for i, (attrs, body) in enumerate(scripts_of(html)):
            if "application/json" in attrs:
                continue
            f = os.path.join(FIX, f"{'dev' if dev else 'prod'}_script{i}.js")
            open(f, "w", encoding="utf-8").write(body)
            files.append(f)
        p = jsc([os.path.join(HERE, "syntax.js"), "--"] + files, [])
        check(p.returncode == 0, f"syntax {os.path.basename(path)}: {len(files)} inline scripts parse" + ("" if p.returncode == 0 else "\n" + p.stdout + p.stderr))
        # fully offline: no external script/link/src/fetch targets
        ext = re.findall(r"""(?:src|href)\s*=\s*["'](https?:[^"']+)""", html) + re.findall(r"importScripts\(|XMLHttpRequest|loadPyodide|cdn\.jsdelivr", html)
        # (the shared detail block keeps its two server-mode fetch('/api/...') calls; standalone never reaches them, scanId is null)
        check(not ext, f"{os.path.basename(path)}: no external resources, importScripts, XHR or Pyodide loader ({ext[:3]})")
        wsrc = [b for a, b in scripts_of(html) if 'id="worker-src"' in a][0]
        outs[dev] += (wsrc,)
    return outs


# ------------------------------------------------------------------ 3: unit checks
def units():
    rng = np.random.default_rng(1)
    rgba = rng.integers(0, 256, (61, 97, 4), dtype=np.uint8)
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, fr = cap.read()
    grey = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[300:300 + 531, 200:200 + 437].copy()     # sizes not multiples of 8
    for n, a in (("rgba.bin", rgba), ("grey.bin", grey)):
        open(os.path.join(FIX, n), "wb").write(a.tobytes())
    p = jsc(["--", os.path.join(FIX, "rgba.bin"), "97", "61", os.path.join(FIX, "grey.bin"), str(grey.shape[1]), str(grey.shape[0])],
            [os.path.join(ROOT, "web", "src", s + ".js") for s in SRC] + [os.path.join(HERE, "units.js")])
    line = [l for l in p.stdout.splitlines() if l.startswith("@")]
    check(bool(line) and p.returncode == 0, "units.js ran" + ("" if line else "\n" + p.stderr[:500]))
    if not line:
        return
    j = json.loads(line[0][1:])
    g = np.frombuffer(bytes.fromhex(j["gray"]), np.uint8).reshape(61, 97)
    ref = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    check(np.array_equal(g, ref), f"RGBA->grey bit-identical to cv2 COLOR_RGBA2GRAY ({int((g != ref).sum())} of {g.size} differ)")
    for q, b in zip((75, 90), j["jpeg"]):
        raw = base64.b64decode(b)
        dec = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        okc, cj = cv2.imencode(".jpg", grey, [cv2.IMWRITE_JPEG_QUALITY, q])
        cdec = cv2.imdecode(cj, cv2.IMREAD_UNCHANGED)
        good = dec is not None and dec.shape == grey.shape
        e_js = float(np.abs(dec.astype(float) - grey).mean()) if good else 99
        e_cv = float(np.abs(cdec.astype(float) - grey).mean())
        check(good and e_js <= e_cv * 1.1 + 0.1, f"JPEG q{q}: decodes, mean abs error {e_js:.3f} (cv2 {e_cv:.3f}); size {len(raw)} B (cv2 {len(cj)} B), 1 channel={dec is not None and dec.ndim == 2}")


# ------------------------------------------------------------------ 4: frames, python reference, jsc worker
def make_frames(n, stride):
    os.makedirs(FIX, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    frames, i = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgba = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGBA))
            path = os.path.join(FIX, f"f{i:05d}.rgba")
            rgba.tofile(path)
            frames.append({"i": i, "t": i / 30.0, "w": rgba.shape[1], "h": rgba.shape[0], "path": path})
        i += 1
    return frames


def python_events(frames, dev):
    import glue
    glue.DEV = dev
    sc = glue.Scanner()
    clk = [0.0]
    real = glue.time.monotonic
    glue.time.monotonic = lambda: clk[0]
    out = []
    try:
        for f in frames:
            clk[0] = f["t"]
            rgba = np.fromfile(f["path"], np.uint8)
            out.append(json.loads(sc.step_rgba(memoryview(rgba), f["w"], f["h"])))
    finally:
        glue.time.monotonic = real
        glue.DEV = False
    return out


def js_events(wsrc, frames, tag):
    wf = os.path.join(FIX, f"worker_{tag}.js")
    open(wf, "w", encoding="utf-8").write(wsrc)
    mf = os.path.join(FIX, "manifest.json")
    json.dump({"frames": frames}, open(mf, "w"))
    p = jsc(["--", wf, mf], [os.path.join(HERE, "harness.js")])
    msgs = [json.loads(l[1:]) for l in p.stdout.splitlines() if l.startswith("@")]
    if p.returncode != 0 or p.stderr.strip():
        print(f"[jsc {tag}] rc={p.returncode} {p.stderr[:800]}")
    return msgs


def skeleton(o, ignore=()):
    """Key structure: dict -> sorted keys with children; list -> structure of the union over its elements."""
    if isinstance(o, dict):
        return {k: skeleton(v, ignore) for k, v in o.items() if k not in ignore}
    if isinstance(o, list):
        return [merge([skeleton(x, ignore) for x in o])] if o else []
    return type(o).__name__ if o is not None else "null"


def merge(sks):
    if not sks:
        return None
    if all(isinstance(s, dict) for s in sks):
        keys = sorted({k for s in sks for k in s})
        return {k: merge([s[k] for s in sks if k in s]) for k in keys}
    if all(isinstance(s, list) for s in sks):
        return [merge([x for s in sks for x in s])] if any(sks) else []
    ts = {s if isinstance(s, str) else "x" for s in sks}
    return "number" if ts <= {"int", "float"} else ("null" if ts == {"null"} else "/".join(sorted(ts - {"null"})))


def keydiff(a, b, path=""):
    """Key-set differences between two skeletons (value kinds are ignored apart from dict vs non-dict)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: missing in JS")
            elif k not in b:
                out.append(f"{path}/{k}: missing in Python")
            else:
                out += keydiff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if a and b:
            out += keydiff(a[0], b[0], path + "[]")
    elif isinstance(a, dict) != isinstance(b, dict) or isinstance(a, list) != isinstance(b, list):
        out.append(f"{path}: structure differs ({type(a).__name__} vs {type(b).__name__})")
    return out


def vdiff(a, b, path, out, tol=1e-3):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                vdiff(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path + "#len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            vdiff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if abs(a - b) > tol * max(1.0, abs(a)):
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))


def img_diff(pa, ja):
    da = cv2.imdecode(np.frombuffer(base64.b64decode(pa), np.uint8), 0)
    dj = cv2.imdecode(np.frombuffer(base64.b64decode(ja), np.uint8), 0)
    if da is None or dj is None or da.shape != dj.shape:
        return None
    return float(np.abs(da.astype(float) - dj).mean())


IGN_TOP = {"ms", "frame_index"}      # timings; frame_index equal anyway but not interesting


def compare(frames, P, J, dev, tag):
    n = len(frames)
    check(len(J) == n, f"[{tag}] worker answered every frame ({len(J)}/{n})")
    if len(J) != n:
        return
    err = [j for j in J if j.get("error")]
    check(not err, f"[{tag}] no frame returned an error ({[j.get('error') for j in err][:2]})")
    st_eq = sum(p["status"] == j.get("status") for p, j in zip(P, J))
    sta_eq = sum(p["state"] == j["state"] for p, j in zip(P, J))
    hint_eq = sum(p["hint"] == j["hint"] for p, j in zip(P, J))
    print(f"[{tag}] status equal {st_eq}/{n}, state equal {sta_eq}/{n}, hint equal {hint_eq}/{n}")
    check(st_eq >= 0.9 * n and sta_eq >= 0.9 * n, f"[{tag}] status and state agree with Python on >= 90 % of frames")
    # structure: every event, commit/revision with summary/detail/dev
    kd = set()
    for p, j in zip(P, J):
        a, b = skeleton(j, IGN_TOP | {"missed"}), skeleton(p, IGN_TOP | {"missed"})
        for k in ("commit", "revision"):
            if k in a and k in b:
                pass
        if "overlay" in a and "overlay" in b and p["overlay"] and j["overlay"]:
            kd |= set(keydiff(a["overlay"], b["overlay"], "/overlay"))
        for k in ("commit", "revision"):
            if k in p and k in j:
                kd |= set(keydiff(skeleton(j[k], {"img"}), skeleton(p[k], {"img"}), "/" + k))
        for k in ("dev", "ms"):
            if k in p and k in j and k == "dev":
                kd |= set(keydiff(a["dev"], b["dev"], "/dev"))
        kd |= {x for x in keydiff({kk: vv for kk, vv in a.items() if kk not in ("overlay", "commit", "revision", "dev")},
                                  {kk: vv for kk, vv in b.items() if kk not in ("overlay", "commit", "revision", "dev")}, "")
               if "/missed" not in x}
    check(not kd, f"[{tag}] JSON keys identical to Python (event, overlay, commit/revision summary+detail" + (", dev+trace" if dev else "") + ")" + ("" if not kd else ": " + "; ".join(sorted(kd)[:10])))
    msk = set()
    for p, j in zip(P, J):
        if "ms" in p and "ms" in j:
            msk |= set(keydiff(skeleton(j["ms"]), skeleton(p["ms"]), "/ms"))
    check(not msk, f"[{tag}] ms keys identical ({sorted(msk)})")
    # commits / revisions / missed
    for k in ("commit", "revision"):
        pi = [i for i, p in enumerate(P) if k in p]
        ji = [i for i, j in enumerate(J) if k in j]
        print(f"[{tag}] {k}s: python at frames {[frames[i]['i'] for i in pi]}, js at {[frames[i]['i'] for i in ji]}")
    pc = [i for i, p in enumerate(P) if "commit" in p]
    jc = [i for i, j in enumerate(J) if "commit" in j]
    check(len(set(pc) & set(jc)) >= max(1, len(pc) - 1) and abs(len(pc) - len(jc)) <= 1, f"[{tag}] commits: same frames as Python (at most one differs)")
    # values at frames where both committed
    both = sorted((set(pc) & set(jc)) | (set(i for i, p in enumerate(P) if "revision" in p) & set(i for i, j in enumerate(J) if "revision" in j)))
    nsum = nsum_eq = 0
    for i in both:
        k = "commit" if "commit" in P[i] else "revision"
        pk, jk = P[i][k], J[i][k]
        d = []
        vdiff(pk["summary"], jk["summary"], "summary", d)
        nsum += 1
        nsum_eq += not d
        if d:
            print(f"[{tag}] frame {frames[i]['i']} {k} summary differs: {d[:6]}{' ...' if len(d) > 6 else ''}")
        dd = []
        a, b = dict(pk["detail"]), dict(jk["detail"])
        pimg, jimg = a.pop("img", None), b.pop("img", None)
        vdiff(a, b, "detail", dd, 1e-2)
        mad = img_diff(pimg, jimg) if pimg and jimg else None
        print(f"[{tag}] frame {frames[i]['i']} {k}: detail value diffs {len(dd)}, warped jpeg mean abs diff {mad if mad is None else round(mad, 2)}")
        for x in dd[:5]:
            print("      ", x)
        check(mad is not None and mad < 6, f"[{tag}] frame {frames[i]['i']} detail image decodes, same size, mean abs diff < 6 ({mad})")
        if dev:
            td = []
            vdiff(pk["dev"], jk["dev"], "dev", td, 5e-3)
            print(f"[{tag}] frame {frames[i]['i']} dev/trace value diffs: {len(td)}" + ("" if not td else f" e.g. {td[:3]}"))
    check(nsum == 0 or nsum_eq >= 0.5 * nsum, f"[{tag}] commit summaries equal to Python's on {nsum_eq}/{nsum} common commits")
    # overlay / missed
    ov = sum(p["overlay"] == j["overlay"] or (p["overlay"] and j["overlay"] and bool(p["overlay"].get("quad")) == bool(j["overlay"].get("quad"))) for p, j in zip(P, J))
    print(f"[{tag}] overlay presence agrees on {ov}/{n}")
    pm = [i for i, p in enumerate(P) if "missed" in p]
    jm = [i for i, j in enumerate(J) if "missed" in j]
    print(f"[{tag}] frames with a 'missed' report: python {len(pm)}, js {len(jm)}")
    tms = [j["ms"]["total"] for j in J]
    print(f"[{tag}] js per-frame total ms (fake-free clock): median {np.median(tms):.0f}, max {max(tms):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--no-dev", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    os.makedirs(FIX, exist_ok=True)
    t0 = time.time()
    outs = build_and_syntax()
    units()
    frames = make_frames(a.frames, a.stride)
    print(f"{len(frames)} frames (every {a.stride}th of segment_01.mp4, {frames[0]['w']}x{frames[0]['h']})", flush=True)
    modes = [False] + ([] if a.no_dev else [True])
    with cf.ThreadPoolExecutor(4) as ex:
        # glue.DEV is a module global: the Python reference runs for the two modes one after the other (JS runs in parallel processes)
        pyfut = ex.submit(lambda: {dev: python_events(frames, dev) for dev in modes})
        jsfut = {dev: ex.submit(js_events, outs[dev][2], frames, "dev" if dev else "prod") for dev in modes}
        PY = pyfut.result()
        for dev in modes:
            P = PY[dev]
            msgs = jsfut[dev].result()
            tag = "dev" if dev else "prod"
            first = [m for m in msgs if m["type"] in ("progress", "ready", "error")]
            check([m["type"] for m in first] == ["progress", "ready"], f"[{tag}] init answered progress then ready ({[m['type'] for m in first]})")
            res = [m for m in msgs if m["type"] == "result"]
            for m in res:
                m["_j"] = json.loads(m["json"])
            check(all(isinstance(m.get("ms"), (int, float)) and m.get("t") and set(m["t"]) == {"xfer", "worker", "tx"} for m in res), f"[{tag}] result messages carry ms and t{{xfer,worker,tx}}")
            compare(frames, P, [m["_j"] for m in res], dev, tag)
    if not a.keep:
        shutil.rmtree(FIX, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(len(FAILS))} in {time.time() - t0:.0f}s")
    for f in FAILS:
        print("  -", f.splitlines()[0])
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
