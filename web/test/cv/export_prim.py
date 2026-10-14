"""Primitive fixtures for cv.js: inputs + cv2 outputs under web/test/fixtures/prim/ (never anywhere else).  Run via run.py."""
import json, os, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FX = os.path.join(ROOT, "web", "test", "fixtures")
OUT = os.path.join(FX, "prim")
os.makedirs(OUT, exist_ok=True)
man = json.load(open(os.path.join(FX, "aruco", "manifest.json")))
cases = []
_n = [0]


def load(c):
    return np.fromfile(os.path.join(FX, "aruco", c["file"]), np.uint8).reshape(c["h"], c["w"])


def save(a):
    a = np.ascontiguousarray(a)
    name = f"{_n[0]:05d}.bin"
    _n[0] += 1
    a.tofile(os.path.join(OUT, name))
    return {"file": name, "dtype": str(a.dtype), "shape": list(a.shape)}


def timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def add(op, params, inputs, fn, tol=0, note=""):
    out = fn()
    cases.append({"op": op, "params": params, "in": [save(a) for a in inputs],
                  "out": save(np.asarray(out)) if isinstance(out, np.ndarray) else {"value": out},
                  "cv2_ms": timeit(fn), "tol": tol, "note": note})


def images():
    real = [i for i, c in enumerate(man) if c["src"].startswith("real:6B86") and "@" not in c["src"]][:1]
    vid = [i for i, c in enumerate(man) if c["src"].startswith("video:")][:1]
    syn = [i for i, c in enumerate(man) if c["src"].startswith("synth:")][:1]
    out = {}
    for nm, ids in (("photo", real), ("video", vid), ("synth", syn)):
        for i in ids:
            out[nm] = load(man[i])
    out["noise"] = np.random.default_rng(1).integers(0, 256, (333, 517), dtype=np.uint8)
    out["small"] = out["synth"][100:260, 200:420].copy()
    out["tiny"] = out["noise"][:7, :9].copy()
    return out


def main():
    I = images()
    rng = np.random.default_rng(5)
    # ---- contours (adaptiveThreshold / findContours / approxPolyDP)
    for k, nm in enumerate(("photo", "video", "synth", "small")):
        g = I[nm]
        for win in (3, 13, 23):
            t = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, win, 7)
            cs, _ = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cs2, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ap = []
            for c in cs:
                if 20 < len(c) < 4 * max(g.shape) and len(c) < 5000:
                    a = cv2.approxPolyDP(c, len(c) * 0.03, True)
                    ap.append([c.reshape(-1).tolist(), a.reshape(-1).tolist(), cv2.arcLength(c, True), cv2.contourArea(c), bool(cv2.isContourConvex(a))])
            cases.append({"op": "contours", "name": f"{nm}_{win}", "win": win, "in": [save(g)], "at": save(t),
                          "list_none": [c.reshape(-1).tolist() for c in cs], "ext_simple": [c.reshape(-1).tolist() for c in cs2], "approx": ap, "cv2_ms": 0, "tol": 0})
    # ---- resize
    for nm in ("photo", "video", "small", "noise"):
        g = I[nm]; h, w = g.shape
        for f in (0.5, 0.25, 0.125, 1 / 3, 0.33, 0.4, 0.7, 0.9, 700 / max(h, w), 1920 / max(h, w)):
            if f >= 1: continue
            add("resize_by", {"fx": f, "mode": "area"}, [g], lambda g=g, f=f: cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), tol=0)
            gf = g.astype(np.float32)
            add("resize_by", {"fx": f, "mode": "area"}, [gf], lambda g=gf, f=f: cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), tol=1e-3)
        add("resize", {"w": max(w // 8, 1), "h": max(h // 8, 1), "mode": "area"}, [g], lambda g=g, w=w, h=h: cv2.resize(g, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA))
        s = 1920 / max(h, w)
        if s < 1:
            add("resize", {"w": max(int(round(w * s)), 1), "h": max(int(round(h * s)), 1), "mode": "area"}, [g], lambda g=g, w=w, h=h, s=s: cv2.resize(g, (max(int(round(w * s)), 1), max(int(round(h * s)), 1)), interpolation=cv2.INTER_AREA))
    for nm in ("small", "noise", "tiny"):
        g = I[nm]; h, w = g.shape
        add("resize", {"w": w * 3, "h": h * 3, "mode": "linear"}, [g], lambda g=g, w=w, h=h: cv2.resize(g, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR), tol=1, note="u8 up")
        add("resize", {"w": max(w // 2 + 1, 1), "h": max(h // 3 + 1, 1), "mode": "linear"}, [g], lambda g=g, w=w, h=h: cv2.resize(g, (max(w // 2 + 1, 1), max(h // 3 + 1, 1)), interpolation=cv2.INTER_LINEAR), tol=1, note="u8 down")
        gf = rng.random((max(h // 20, 2), max(w // 20, 2))).astype(np.float32) * 100
        ds = 9
        add("resize", {"w": gf.shape[1] * ds, "h": gf.shape[0] * ds, "mode": "linear"}, [gf], lambda g=gf: cv2.resize(g, (g.shape[1] * ds, g.shape[0] * ds), interpolation=cv2.INTER_LINEAR), tol=1e-3, note="float maps")
    # ---- Gaussian blur
    for nm in ("photo", "small", "noise"):
        g = I[nm]
        for sg in (0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0):
            add("gaussian", {"sigma": sg}, [g], lambda g=g, sg=sg: cv2.GaussianBlur(g, (0, 0), sg), tol=0)
            add("gaussian", {"sigma": sg}, [g.astype(np.float32)], lambda g=g, sg=sg: cv2.GaussianBlur(g.astype(np.float32), (0, 0), sg), tol=1e-3)
    row = (rng.random(700) * 200 + 50).astype(np.float32)
    for sg in (2.0, 7.5, 30.0):
        add("gaussian", {"sigma": sg}, [row.reshape(1, -1)], lambda sg=sg: cv2.GaussianBlur(row.reshape(1, -1), (0, 0), sg), tol=1e-3, note="1-row float")
    # ---- morphology
    for nm in ("photo", "small", "noise"):
        g = I[nm]
        for k in (3, 5, 9, 15, 23, 41):
            kk = np.ones((k, k), np.uint8)
            add("dilate", {"k": k}, [g], lambda g=g, kk=kk: cv2.dilate(g, kk))
            add("erode", {"k": k}, [g], lambda g=g, kk=kk: cv2.erode(g, kk))
            add("open", {"k": k}, [g], lambda g=g, kk=kk: cv2.morphologyEx(g, cv2.MORPH_OPEN, kk))
        for k in (4, 6):
            kk = np.ones((k, k), np.uint8)
            add("dilate", {"k": k}, [g], lambda g=g, kk=kk: cv2.dilate(g, kk), note="even")
    resp = (rng.random((120, 300)) * 2 - 0.5).astype(np.float32)
    for k in (3, 7, 11):
        add("dilate", {"k": k}, [resp], lambda kk=np.ones((k, k), np.uint8): cv2.dilate(resp, kk), note="float")
    # ---- box filter (float, replicate), uint8
    fb = (rng.random((150, 260)) * 200).astype(np.float32)
    for kx, ky in ((5, 7), (21, 15), (3, 3), (40, 30)):
        add("box", {"kx": kx, "ky": ky}, [fb], lambda kx=kx, ky=ky: cv2.boxFilter(fb, -1, (kx, ky), normalize=True, borderType=cv2.BORDER_REPLICATE), tol=2e-3)
    for kx in (3, 7, 12):
        add("box", {"kx": kx, "ky": kx}, [I["small"]], lambda kx=kx: cv2.boxFilter(I["small"], -1, (kx, kx), normalize=True, borderType=cv2.BORDER_REPLICATE))
    # ---- laplacian / integral / threshold / divide / addWeighted / gray / not
    for nm in ("small", "photo", "tiny"):
        g = I[nm]
        add("laplacian", {}, [g], lambda g=g: cv2.Laplacian(g, cv2.CV_32F), tol=1e-4)
        add("integral", {}, [g], lambda g=g: cv2.integral(g, sdepth=cv2.CV_64F))
        add("otsu", {"inv": True}, [g], lambda g=g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        add("not", {}, [g], lambda g=g: cv2.bitwise_not(g))
    a = I["small"]; b = np.maximum(cv2.GaussianBlur(cv2.dilate(a, np.ones((9, 9), np.uint8)), (0, 0), 3), 8)
    add("divide", {"scale": 235.0}, [a, b], lambda: cv2.divide(a, b, scale=235.0, dtype=cv2.CV_32F), tol=1e-3)
    add("addWeighted", {"a": 1.5, "b": -0.5, "g": 3.0}, [a, b], lambda: cv2.addWeighted(a, 1.5, b, -0.5, 3.0))
    bgr = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    add("gray", {"order": "bgr", "ch": 3}, [bgr], lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    add("gray", {"order": "rgb", "ch": 3}, [bgr], lambda: cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY))
    rgba = rng.integers(0, 256, (60, 80, 4), dtype=np.uint8)
    add("gray", {"order": "rgb", "ch": 4}, [rgba], lambda: cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY))
    # ---- geometry
    g = I["synth"]; h, w = g.shape
    for t in range(4):
        src = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]]) + rng.normal(0, 1, (4, 2)).astype(np.float32)
        dst = np.float32([[10, 20], [320, 15], [330, 230], [5, 215]]) + rng.normal(0, 8, (4, 2)).astype(np.float32)
        add("getPT", {}, [src, dst], lambda: cv2.getPerspectiveTransform(src, dst), tol=1e-6)
        H = cv2.getPerspectiveTransform(src, dst)
        for interp, name in ((cv2.INTER_LINEAR, "linear"), (cv2.INTER_NEAREST, "nearest")):
            for bv in (0, 255):
                add("warp", {"interp": name, "w": 380, "h": 260, "value": bv}, [g, H], lambda: cv2.warpPerspective(g, H, (380, 260), flags=interp, borderValue=bv), tol=1 if name == "linear" else 0)
        pts = (rng.random((50, 2)) * 300).astype(np.float64)
        add("perspT", {}, [pts, H], lambda: cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2), tol=1e-9)
        n = int(rng.integers(8, 17))
        sp = (rng.random((n, 2)) * 500).astype(np.float64); Ht = H.copy()
        dp = cv2.perspectiveTransform(sp.reshape(-1, 1, 2), Ht).reshape(-1, 2) + rng.normal(0, 0.4, (n, 2))
        add("findH", {}, [sp, dp], lambda: cv2.findHomography(sp, dp, 0)[0], tol=1e-5, note=f"n={n}")
    # ---- remap (float maps), N x 1 sample form
    for border, bm in (("replicate", cv2.BORDER_REPLICATE), ("constant", cv2.BORDER_CONSTANT)):
        mh, mw = 150, 260
        gy, gx = np.mgrid[0:mh, 0:mw].astype(np.float32)
        mx = (gx + rng.normal(0, 3, (mh, mw)).astype(np.float32) * 2 - 10).astype(np.float32)
        my = (gy + rng.normal(0, 3, (mh, mw)).astype(np.float32) * 2 + 7).astype(np.float32)
        add("remap", {"border": border, "value": 255, "w": mw, "h": mh}, [g[:200, :300].copy(), mx, my], lambda: cv2.remap(g[:200, :300].copy(), mx, my, cv2.INTER_LINEAR, borderMode=bm, borderValue=255), tol=1)
        gf = g[:200, :300].astype(np.float32)
        add("remap", {"border": border, "value": 255, "w": mw, "h": mh}, [gf, mx, my], lambda: cv2.remap(gf, mx, my, cv2.INTER_LINEAR, borderMode=bm, borderValue=255), tol=1e-3)
    px = (rng.random((200, 1)) * 330 - 15).astype(np.float32); py = (rng.random((200, 1)) * 230 - 15).astype(np.float32)
    add("remapLinear", {}, [g[:200, :300].copy(), px.reshape(-1), py.reshape(-1)], lambda: cv2.remap(g[:200, :300].copy(), px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float64), tol=1)
    # ---- cornerSubPix on aruco-like corners from real photos
    c = [c for c in man if c["src"].startswith("real:6B86") and "@" not in c["src"] and c["corners"]][0]
    g = load(c)
    for win in (2, 3, 4):
        pts = np.float32(c["corners"][0]).reshape(4, 2) + rng.normal(0, 0.8, (4, 2)).astype(np.float32)
        add("subpix", {"win": win}, [g, pts], lambda: cv2.cornerSubPix(g, pts.copy().reshape(-1, 1, 2), (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)).reshape(4, 2), tol=2e-3)
    # ---- matchTemplate
    sm = I["small"]
    for (ty, tx) in ((20, 20), (11, 15)):
        tm = sm[60:60 + ty, 80:80 + tx].copy()
        win = sm[30:30 + ty + 31, 50:50 + tx + 31].copy()
        add("matchT", {}, [win, tm], lambda: cv2.matchTemplate(win, tm, cv2.TM_CCOEFF_NORMED), tol=2e-3)
    flat = np.full((40, 40), 90, np.uint8); tmf = np.full((9, 9), 90, np.uint8)
    add("matchT", {}, [flat, tmf], lambda: cv2.matchTemplate(flat, tmf, cv2.TM_CCOEFF_NORMED), tol=2e-3, note="flat")
    json.dump(cases, open(os.path.join(OUT, "manifest.json"), "w"))
    print("prim cases", len(cases))


main()
