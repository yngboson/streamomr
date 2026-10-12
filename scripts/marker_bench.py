#!/usr/bin/env python
"""Corner-marker (ArUco) detection benchmark: per-marker detection rate on *uncovered* markers, wrong / misplaced
detections, corner accuracy and timing, for the real video, synthetic printer frames and harsher print variants.

    python scripts/marker_bench.py                       # all datasets, methods "v0" (pre-change) and "final"
    python scripts/marker_bench.py --methods v0,final --sets real,printer,light --scales 0,1280,960
    python scripts/marker_bench.py --list                # available methods / sets

Truth
  synth   : truth["marker_corners"] from tests/synth.make_frame (planar projection; every marker of the printed page
            is uncovered; markers cut by the frame edge are excluded).
  real    : realasset/IMG_4425.MOV has no marker truth, so it is labelled by an *oracle*: the union of every detector
            variant below plus a slow exhaustive pass, accepted only when >= 3 markers agree with the page geometry
            (homography reprojection < ORACLE_REPROJ_MM).  The page is then projected to get all four marker quads;
            a marker counts as visible when its quad + quiet zone lies inside the frame and holds (almost) no skin.
            Frames the oracle cannot label are reported (and can be dumped with --dump-unlabelled for review).
  Labels are cached (--cache, default in the system temp dir) because the oracle is slow.

Metrics (per set x scale x method)
  det%     detected / visible markers (correct id within SAME_PLACE of the true centre)
  frm3%    frames with >= 3 visible markers in which >= 3 were detected (what recognize() needs)
  wrong    detections with an id that is not on the page, or at the wrong place (misplaced)
  cerr     median corner error of correct detections, in marker modules (1 module = 16/6 mm)
  ms       median / p95 detection time, single-threaded (cv2.setNumThreads(1)); "none" = frames without a sheet
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from omr import detect  # noqa: E402
from omr import layout as lay  # noqa: E402
from omr import markers as mk  # noqa: E402

VIDEO = os.path.join(ROOT, "realasset", "IMG_4425.MOV")
SAME_PLACE = 0.35          # a detection is "at the right place" if its centre is within this * marker side
ORACLE_REPROJ_MM = 3.0     # oracle: markers must agree with the page layout to this (hand-held sheets curl by 1-2.6 mm;
                           # truth only needs centres to << SAME_PLACE * 16 mm = 5.6 mm)
QUIET_CHECK_MM = 2.0       # visibility: the quiet zone around the marker that must be skin-free / in frame
SKIN_FRAC = 0.03           # visibility: fraction of skin pixels in marker + quiet zone that counts as covered

Truth = Dict[str, object]   # {"quads": {id: 4x2}, "visible": [ids], "page": int}


# ============================================================================ geometry helpers
def marker_quads_mm(page: int) -> Dict[int, np.ndarray]:
    out = {}
    for c, (x, y, w, h) in lay.marker_rects(page).items():
        out[mk.marker_id(page, c)] = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)
    return out


def side(q: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]))


def fit_page(found: Dict[int, np.ndarray]) -> Optional[Tuple[int, np.ndarray, float, List[int]]]:
    """Best page homography (mm -> px) from >= 3 mutually consistent markers: (page, H, max reproj mm, ids used)."""
    groups: Dict[int, List[int]] = {}
    for i in found:
        groups.setdefault(i // 4, []).append(i)
    best = None
    for g, ids in groups.items():
        if len(ids) < 3:
            continue
        mm = marker_quads_mm(g)
        subsets = [ids] + ([[j for j in ids if j != k] for k in ids] if len(ids) == 4 else [])
        for sub in subsets:
            src = np.concatenate([mm[i] for i in sub])
            dst = np.concatenate([found[i] for i in sub])
            H, _ = cv2.findHomography(src, dst, 0)
            if H is None:
                continue
            back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
            err = float(np.linalg.norm(back - src, axis=1).max())
            if err < ORACLE_REPROJ_MM and (best is None or len(sub) > len(best[3]) or
                                           (len(sub) == len(best[3]) and err < best[2])):
                best = (g, H, err, sub)
    return best


# ============================================================================ methods
def v0_find_markers(gray):
    """The detector as it was before this benchmark (frozen copy of omr/detect.py, plain pass + opening ladder)."""
    d = _v0_det()
    m = detect._Merge()
    c, i, _ = d.detectMarkers(gray)
    m.add(c, i)
    passes = 1
    if not m.solid():
        steps = detect.ladder(gray.shape)
        hint = m.module_px()
        if hint is not None:
            steps = sorted(steps, key=lambda s: abs(np.log(s / hint)))
            steps = [hint] + steps[:2]
        else:
            steps = steps[1:] + steps[:1] if len(steps) > 2 else steps
        done = set()
        for s in steps:
            k = max(3, int(round(0.85 * s)) | 1)
            if k in done:
                continue
            done.add(k)
            c, i, _ = d.detectMarkers(cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)))
            m.add(c, i)
            passes += 1
            if m.solid():
                break
    return m.found, m.dup, {"detect_passes": float(passes)}


@functools.lru_cache(maxsize=None)
def _v0_det_cached(tid):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    p.perspectiveRemovePixelPerCell = 8
    p.maxErroneousBitsInBorderRate = 0.5
    return cv2.aruco.ArucoDetector(mk.aruco_dictionary(), p)


def _v0_det():
    import threading
    return _v0_det_cached(threading.get_ident())


def final_find_markers(gray):
    return detect.find_markers(gray)


METHODS: Dict[str, Callable] = {"v0": v0_find_markers, "final": final_find_markers}


def register(name: str, fn: Callable) -> None:
    METHODS[name] = fn


# ============================================================================ oracle (real video labelling)
def oracle_find(gray: np.ndarray) -> Dict[int, np.ndarray]:
    """Slow exhaustive union of detector variants (labelling only; ids seen at conflicting places are dropped):
    the pre-change detector, plain + every opening 3..17 px with the AprilTag-style re-decode of every rejected
    candidate, and white-biased local binarisations at every module-size hypothesis."""
    m = detect._Merge()
    f, _, _ = v0_find_markers(gray)
    _add(m, f)
    d = _v0_det()
    for k in [0] + list(range(3, 19, 2)):
        o = gray if k == 0 else cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        c, i, rej = d.detectMarkers(o)
        m.add(c, i)
        for r in rej:
            for img in ((gray, o) if k else (gray,)):
                for tau in (0.5, 0.65):
                    out = model_decode(img, r, tau)
                    if out:
                        _add(m, {out[0]: _subpix(gray, out[1])})
                        break
    mods = [s for s in detect.ladder(gray.shape)] + ([m.module_px()] if m.found else [])
    for mod in mods:
        for tau in (0.65, 0.8):
            for kf in (0, 0.6, 0.85):
                c, i, _ = d.detectMarkers(white_biased_binary(gray, mod, tau, kf))
                if i is not None:
                    m.add([_subpix(gray, cc).reshape(1, 4, 2) for cc in c], i)
    return m.found


def _add(m, f):
    if f:
        m.add([v.reshape(1, 4, 2).astype(np.float32) for v in f.values()], np.array(list(f)).reshape(-1, 1))


def _subpix(gray, q):
    return cv2.cornerSubPix(gray, np.asarray(q, np.float32).reshape(-1, 1, 2), (3, 3), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)).reshape(4, 2)


def white_biased_binary(gray, mod, tau, kf, wf=3.0, floor=0.15):
    """Black where the pixel is below B + tau (W - B), W / B = local max / min over ~wf modules (after an opening
    of kf modules); printing defects only lighten black, so tau > 0.5 keeps grey-printed cells black."""
    k = max(3, int(round(kf * mod)) | 1) if kf else 0
    o = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) if k else gray
    w = max(3, int(round(wf * mod)) | 1)
    K = np.ones((w, w), np.uint8)
    W = cv2.blur(cv2.dilate(o, K).astype(np.float32), (w, w))
    B = cv2.blur(cv2.erode(o, K).astype(np.float32), (w, w))
    blk = (o < B + tau * (W - B)) & ((W - B) > floor * W)
    return np.where(blk, 0, 255).astype(np.uint8)


# AprilTag-style decoding: bits are thresholded between a black model (fit to the border cells) and a white model
# (fit to a ring in the quiet zone), both planes over the marker (Olson 2011 / Wang & Olson 2016).
_G = (np.arange(3) - 1) * 0.25
_CU, _CV = np.meshgrid(np.arange(6) + 0.5, np.arange(6) + 0.5)
_CELL_UV = np.stack([(_CU[..., None, None] + _G[None, None, None, :]).repeat(3, 2),
                     (_CV[..., None, None] + _G[None, None, :, None]).repeat(3, 3)], -1).reshape(-1, 2)
_T = np.arange(6) + 0.5
_WHITE_UV = np.concatenate([np.stack([_T, np.full(6, -0.6)], 1), np.stack([_T, np.full(6, 6.6)], 1),
                            np.stack([np.full(6, -0.6), _T], 1), np.stack([np.full(6, 6.6), _T], 1)])
_BORDER = np.ones((6, 6), bool)
_BORDER[1:5, 1:5] = False


def _sample(img, H, uv):
    p = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2).astype(np.float32)
    return cv2.remap(img, p[:, :1], p[:, 1:], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float64)


def model_decode(img, quad, tau=0.5, min_contrast=0.25):
    q = np.asarray(quad, np.float64).reshape(4, 2)
    a, b = q[1] - q[0], q[2] - q[1]
    if a[0] * b[1] - a[1] * b[0] < 0:
        q = q[::-1].copy()
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [6, 0], [6, 6], [0, 6]]), q.astype(np.float32))
    cells = _sample(img, H, _CELL_UV).reshape(6, 6, 9).mean(2)
    wv = _sample(img, H, _WHITE_UV)
    A = lambda uv: np.c_[np.ones(len(uv)), uv]  # noqa: E731
    bc = np.linalg.lstsq(A(np.stack([_CU[_BORDER], _CV[_BORDER]], 1)), cells[_BORDER], rcond=None)[0]
    wc = np.linalg.lstsq(A(_WHITE_UV), wv, rcond=None)[0]
    grid = A(np.stack([_CU.ravel(), _CV.ravel()], 1))
    blk, wht = (grid @ bc).reshape(6, 6), (grid @ wc).reshape(6, 6)
    con = wht - blk
    if con.min() <= 0 or con.mean() < min_contrast * wht.mean():
        return None
    rel = (cells - blk) / con
    wrel = (wv - A(_WHITE_UV) @ bc) / (A(_WHITE_UV) @ (wc - bc))
    if (rel[_BORDER] > tau).any() or (wrel < tau).any():
        return None
    ok, mid, rot = mk.aruco_dictionary().identify((rel[1:5, 1:5] > tau).astype(np.uint8), 0.0)
    return (int(mid), np.roll(q, int(rot), axis=0)) if ok else None


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # classic YCrCb skin box (Chai & Ngan 1999), tightened on Cr so warm-white paper does not count
    return ((ycc[..., 1] >= 140) & (ycc[..., 1] <= 180) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)).astype(np.uint8)


def label_frame(bgr: np.ndarray) -> Optional[Truth]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = oracle_find(gray)
    fit = fit_page(found)
    if fit is None:
        return None
    page, H, err, used = fit
    h, w = gray.shape
    skin = skin_mask(bgr)
    quads, visible = {}, []
    for mid, q in marker_quads_mm(page).items():
        # oracle's own corners where it found the marker (exact), else the page projection (curl error ~1-3 mm)
        quads[mid] = found[mid] if mid in found else cv2.perspectiveTransform(q.reshape(-1, 1, 2), H).reshape(4, 2)
        ctr = q.mean(0)
        big = ctr + (q - ctr) * (lay.MARKER_MM + 2 * QUIET_CHECK_MM) / lay.MARKER_MM
        bq = cv2.perspectiveTransform(big.reshape(-1, 1, 2), H).reshape(4, 2)
        if (bq < 0).any() or (bq[:, 0] >= w).any() or (bq[:, 1] >= h).any():
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.round(bq).astype(np.int32), 1)
        if (skin[mask > 0].mean() if mask.any() else 1) > SKIN_FRAC:
            continue
        visible.append(mid)
    return {"quads": {int(k): v.tolist() for k, v in quads.items()}, "visible": visible, "page": page,
            "oracle_ids": sorted(int(i) for i in found), "oracle_err_mm": err}


def read_video(stride: int, max_frames: int = 0):
    cap = cv2.VideoCapture(VIDEO)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            yield idx, img
            if max_frames and idx // stride + 1 >= max_frames:
                break
        idx += 1
    cap.release()


def real_set(stride: int, cache: str):
    """-> [(gray, truth or None)], truth None = no sheet (timing only) or unlabelled ('unlabelled' key)."""
    path = os.path.join(cache, f"real_labels_s{stride}.json")
    labels = json.load(open(path)) if os.path.exists(path) else {}
    frames = list(read_video(stride))
    todo = [(idx, img) for idx, img in frames if str(idx) not in labels]
    if todo:
        from multiprocessing import Pool
        with Pool(min(8, os.cpu_count() or 1)) as pool:
            for (idx, _), lab in zip(todo, pool.imap(label_frame, [img for _, img in todo])):
                labels[str(idx)] = lab
    items = [(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), labels[str(idx)], idx) for idx, img in frames]
    if todo:
        os.makedirs(cache, exist_ok=True)
        json.dump(labels, open(path, "w"))
    return items


# ============================================================================ synthetic sets
HARSH = {
    # lighter toner (black prints 25-45 % grey) - the faintest the cheap printer produced
    "light": dict(black=(0.25, 0.45)),
    # thicker / denser white streaks and always a white band through one marker
    "streaky": dict(streak_mm=(0.3, 1.2), p_band=1.0),
    # both, plus extra bands through the other markers (see _extra_bands)
    "worst": dict(black=(0.3, 0.5), streak_mm=(0.3, 1.2), p_band=1.0, extra_bands=2),
}


def _defects(orig, kw):
    kw = dict(kw)
    extra = kw.pop("extra_bands", 0)

    def f(raster, page, rng, **_):
        out = orig(raster, page, rng, **kw)
        if extra:
            from tests import synth
            mm = synth.PPM_HI
            rects = list(lay.marker_rects(page).values())
            for k in rng.choice(len(rects), size=extra, replace=False):
                x, y, w, h = rects[int(k)]
                y0 = int((y + rng.uniform(0.15, 0.85) * h) * mm)
                out[y0:y0 + int(rng.uniform(0.5, 1.1) * mm), int((x - 2) * mm):int((x + w + 2) * mm)] = 250
        return out
    return f


def synth_set(cond: str, n: int, seed: int, workdir: str, harsh: Optional[str] = None):
    from tests import synth
    assets = synth.prepare(workdir)
    names = list(assets)
    orig = synth.print_defects
    if harsh:
        synth.print_defects = _defects(orig, HARSH[harsh])
    try:
        items = []
        for k in range(n):
            rng = np.random.default_rng(seed + k)
            L, pdf = assets[names[k % len(names)]]
            frame, tr = synth.make_frame(L, pdf, synth.CONDS[cond], rng, mode="sheet")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            quads = {int(i): np.array(q) for i, q in tr["marker_corners"].items()}
            vis = [i for i, q in quads.items() if (q >= 2).all() and (q[:, 0] < w - 2).all() and (q[:, 1] < h - 2).all()
                   and i not in [list(quads)[j] for j in tr.get("hidden", [])]]
            items.append((gray, {"quads": {i: q.tolist() for i, q in quads.items()}, "visible": vis,
                                 "page": int(tr["page"])}, k))
        return items
    finally:
        synth.print_defects = orig


def none_set(n: int, seed: int, workdir: str):
    from tests import synth
    assets = synth.prepare(workdir)
    L, pdf = next(iter(assets.values()))
    out = []
    for k in range(n):
        frame, _ = synth.make_frame(L, pdf, synth.CONDS["phone"], np.random.default_rng(seed + k), mode="none")
        out.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, k))
    return out


# ============================================================================ evaluation
def rescale(gray, truth, max_side):
    if not max_side or max(gray.shape) <= max_side:
        return gray, truth, 1.0
    s = max_side / max(gray.shape)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if truth is None:
        return g, None, s
    t = dict(truth)
    t["quads"] = {k: (np.array(v) * s).tolist() for k, v in truth["quads"].items()}
    return g, t, s


def score(found, truth):
    quads = {int(k): np.array(v) for k, v in truth["quads"].items()}
    vis = set(truth["visible"])
    hit, wrong, cerr = 0, 0, []
    for i, c in found.items():
        q = quads.get(i)
        if q is None or np.linalg.norm(c.mean(0) - q.mean(0)) > SAME_PLACE * side(q):
            wrong += 1
            continue
        if i in vis:
            hit += 1
            mod = side(q) / 6.0
            # corner order may start anywhere only if the detector were wrong; ArUco returns TL,TR,BR,BL of the marker
            cerr.append(float(np.median(np.linalg.norm(c - q, axis=1))) / mod)
    return hit, len(vis), wrong, cerr


def run(methods, sets, scales, items_by_set):
    rows = []
    for sname, items in items_by_set.items():
        for ms in scales:
            prepared = [rescale(g, t, ms) + (idx,) for g, t, idx in items]
            for mname in methods:
                fn = METHODS[mname]
                hit = tot = wrong = f3 = f3n = 0
                cerrs, t_sheet, t_none, misses = [], [], [], []
                for g, t, s, idx in prepared:
                    t0 = time.perf_counter()
                    found, dup, info = fn(g)
                    dt = (time.perf_counter() - t0) * 1000
                    if t is None or not t.get("quads"):
                        t_none.append(dt)
                        continue
                    t_sheet.append(dt)
                    h, n, w, ce = score(found, t)
                    hit += h; tot += n; wrong += w; cerrs += ce
                    if w:
                        misses.append(("WRONG", idx, sorted(found)))
                    if n >= 3:
                        f3n += 1
                        f3 += h >= 3
                    if h < n:
                        misses.append((idx, sorted(set(t["visible"]) - set(found))))
                rows.append(dict(set=sname, scale=ms or "native", method=mname, det=hit / max(tot, 1), n=tot, wrong=wrong,
                                 frm3=f3 / max(f3n, 1), nf3=f3n,
                                 cerr=float(np.median(cerrs)) if cerrs else float("nan"),
                                 ms_med=float(np.median(t_sheet)) if t_sheet else float("nan"),
                                 ms_p95=float(np.percentile(t_sheet, 95)) if t_sheet else float("nan"),
                                 none_med=float(np.median(t_none)) if t_none else float("nan"),
                                 none_p95=float(np.percentile(t_none, 95)) if t_none else float("nan"),
                                 misses=misses))
                r = rows[-1]
                print(f"{sname:9s} {str(r['scale']):>6s} {mname:16s} det {100 * r['det']:5.1f}% (n={tot:4d})  "
                      f"frm3 {100 * r['frm3']:5.1f}% (n={f3n:3d})  wrong {wrong:2d}  cerr {r['cerr']:.3f}  "
                      f"ms {r['ms_med']:6.1f}/{r['ms_p95']:6.1f}  none {r['none_med']:6.1f}/{r['none_p95']:6.1f}", flush=True)
    return rows


def load_sets(names, args):
    out = {}
    wd = args.workdir
    for s in names:
        if s == "real":
            items = real_set(args.stride, args.cache)
            lab = [(g, t, i) for g, t, i in items if t is not None]
            # frames after the last sheet (truth: no_sheet_after 10.3 s) are "none" frames for timing
            none = [(g, None, i) for g, t, i in items if t is None and i / 30.0 > 10.4]
            unl = [i for g, t, i in items if t is None and i / 30.0 <= 10.4]
            print(f"real: {len(lab)} labelled frames, {len(none)} no-sheet frames, {len(unl)} unlabelled: {unl}")
            out["real"] = lab + none
        elif s == "none":
            out["none"] = none_set(args.n, 9000, wd)
        elif s in HARSH:
            out[s] = synth_set("printer", args.n, 7000, wd, harsh=s)
        else:
            out[s] = synth_set(s, args.n, 5000, wd)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="v0,final")
    ap.add_argument("--sets", default="real,printer,light,streaky,worst,none")
    ap.add_argument("--scales", default="0,1280,960", help="max image side; 0 = native")
    ap.add_argument("--n", type=int, default=60, help="frames per synthetic set")
    ap.add_argument("--stride", type=int, default=2, help="real video: use every n-th frame")
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "streamomr_marker_bench"))
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "streamomr_marker_bench", "synth"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--json", help="write rows (incl. misses) here")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extra", help="python module path adding methods via marker_bench.register (experiments)")
    args = ap.parse_args(argv)
    if args.extra:
        import importlib.util
        spec = importlib.util.spec_from_file_location("bench_extra", args.extra)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["marker_bench"] = sys.modules[__name__]
        spec.loader.exec_module(mod)
    if args.list:
        print("methods:", ", ".join(METHODS))
        print("sets: real, none, <any synth cond>, " + ", ".join(HARSH))
        return
    cv2.setNumThreads(args.threads)
    sets = load_sets([s for s in args.sets.split(",") if s], args)
    rows = run(args.methods.split(","), sets, [int(x) for x in args.scales.split(",")], sets)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1, default=str)
    return rows


if __name__ == "__main__":
    main()
