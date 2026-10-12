#!/usr/bin/env python
"""Benchmark of the lighting-robust bubble darkness (omr/illum.py) and the auto ink floor (omr/ink.py) as integrated
in omr.recognize, on harder lighting (h_*) and ink (k_*) synthetic conditions.

    python scripts/illum_bench.py --hard --n 60            # the harder lighting cases defined here
    python scripts/illum_bench.py --ink --n 60             # mark lightness, erasures, printer black
    python scripts/illum_bench.py --methods E,F,B --conds shadow,glare --n 60 --jobs 14
    python scripts/illum_bench.py --video                  # real video replay (scripts/illum_video.py)

Every frame is generated once and recognised once per method (same pixels), so the columns are paired.
A method is ``<darkness>[.<ink>]``: darkness E (integrated default) or an alternative kept in omr.illum (F, B, D10,
D20: swapped in at runtime for the method name recognize passes); ink ``auto`` (integrated default), ``f45`` (fixed 0.45
x black, the pre-auto rule) or ``f50``.  The pre-integration pipeline (shipped max-filter paper level) no longer exists
in the code; its numbers were recorded before integration.

Harder conditions (h_*) wrap ``synth.paper_in_space`` with a version that has several shadows, an edge forced
through the ID grid, more glare and glossy graphite.
"""
import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import synth  # noqa: E402
from omr import illum  # noqa: E402
from omr import ink as inkmod  # noqa: E402
from omr import recognize as R  # noqa: E402

_ORIG = {"pil": synth.paper_in_space, "pd": synth.print_defects, "di": synth.draw_ink, "hk": synth.HARD_KINDS,
         "bd": illum.bubble_darkness, "cf": inkmod.choose_floor}
for _kv in filter(None, os.environ.get("ILLUM_PARAMS", "").split(",")):    # e.g. ILLUM_PARAMS=SECT_PCT=90,PAPER_GAIN=1.03
    setattr(illum, _kv.split("=")[0], float(_kv.split("=")[1]))
for _kv in filter(None, os.environ.get("RECOG_PARAMS", "").split(",")):   # e.g. RECOG_PARAMS=MIN_STRENGTH=0.55
    setattr(R, _kv.split("=")[0], float(_kv.split("=")[1]))
for _kv in filter(None, os.environ.get("INK_PARAMS", "").split(",")):     # e.g. INK_PARAMS=GHOST_PCT=95,LO_FRAC=0.38
    setattr(inkmod, _kv.split("=")[0], float(_kv.split("=")[1]))
_STATE = {"unrel": [], "ms": 0.0, "inkinfo": None}


# ------------------------------------------------------------------------------ method selection
def install(method: str) -> None:
    """Hook the method in (both hooks wrap the functions recognize calls through the module attributes)."""
    dm, _, im = method.partition(".")
    _STATE.update(unrel=[], ms=0.0, inkinfo=None)

    def bd(smooth, centers_mm, diameter_mm, ppm, _method="E"):
        t0 = time.perf_counter()
        D, rel = _ORIG["bd"](smooth, centers_mm, diameter_mm, ppm, dm)
        _STATE["ms"] += (time.perf_counter() - t0) * 1000
        _STATE["unrel"].append(~rel)
        return D, rel

    def cf(rows, black):
        ai = _ORIG["cf"](rows, black) if im in ("", "auto") else \
            inkmod.AutoInk((0.50 if im == "f50" else inkmod.DEFAULT_FRAC) * black, "default", 0.0, True)
        _STATE["inkinfo"] = (ai.source, ai.floor / black, ai.separation)
        return ai

    illum.bubble_darkness, inkmod.choose_floor = bd, cf


# ------------------------------------------------------------------------------ harder lighting
@dataclass
class HCond(synth.Cond):
    n_shadows: int = 0                          # sharp-edged shadows
    shadow_depth: Tuple[float, float] = (0.25, 0.6)
    edge_through_id: bool = False               # the first shadow's edge runs through the ID grid
    gloss: float = 0.6                          # glossiness of ink under glare (stock synth: 0.6; graphite is glossier)
    edge_pt: Optional[Tuple[float, float]] = None


def _pil(small, ppm, cam, cond, rng, truth):
    """synth.paper_in_space with several shadows, an edge through a chosen point, and a glossiness knob."""
    if not isinstance(cond, HCond):
        return _ORIG["pil"](small, ppm, cam, cond, rng, truth)
    step = 2.0
    gx, gy, X, N = synth.paper_surface(rng, cond, step)
    C = np.asarray(cam, np.float64)
    if C[2] < 0:
        C = C * np.array([1, 1, -1])
    s = C[2] / (C[2] - X[..., 2])
    qx = C[0] + s * (X[..., 0] - C[0])
    qy = C[1] + s * (X[..., 1] - C[1])
    Dx, Dy = (qx - gx).astype(np.float32), (qy - gy).astype(np.float32)
    truth["curl_max_mm"] = float(np.sqrt(Dx ** 2 + Dy ** 2).max())
    view = C - X
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(rng.uniform(12, 30))
    graze = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    Lc = view[view.shape[0] // 2, view.shape[1] // 2]
    L = (1 - cond.raking) * Lc + cond.raking * graze
    L /= np.linalg.norm(L)
    amb = 0.3
    lam = np.clip((N * L).sum(-1), 0, None)
    shade = amb + (1 - amb) * lam / max(float(lam.max()), 1e-6)
    for i in range(cond.n_shadows):
        a = rng.uniform(0, 2 * np.pi)
        d = (gx - 105) * np.cos(a) + (gy - 148.5) * np.sin(a)
        if i == 0 and cond.edge_pt is not None:
            ex, ey = cond.edge_pt
            c0 = (ex - 105) * np.cos(a) + (ey - 148.5) * np.sin(a)
        else:
            c0 = rng.uniform(-0.45, 0.45) * (abs(210 * np.cos(a)) + abs(297 * np.sin(a)))
        wob = rng.uniform(2, 10) * np.sin((-(gx - 105) * np.sin(a) + (gy - 148.5) * np.cos(a)) / rng.uniform(25, 80) + rng.uniform(0, 6.3))
        soft = rng.uniform(0.4, 2.0)
        inside = 1 / (1 + np.exp(-(d - c0 + wob) / soft))
        shade = shade * (1 - rng.uniform(*cond.shadow_depth) * inside)
    glare = np.zeros_like(gx, np.float32)
    if cond.glare[1] > 0 and rng.random() < 0.9:
        Lg = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 1.0])
        Lg /= np.linalg.norm(Lg)
        Rm = 2 * (N * Lg).sum(-1, keepdims=True) * N - Lg
        spec = np.clip((Rm * view).sum(-1), 0, 1) ** rng.uniform(150, 600)
        spec = cv2.GaussianBlur(spec.astype(np.float32), (0, 0), rng.uniform(2, 6))
        if spec.max() < 1e-3:
            cx, cy = rng.uniform(30, 180), rng.uniform(30, 267)
            spec = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * rng.uniform(12, 35) ** 2)).astype(np.float32)
        glare = (spec / spec.max() * rng.uniform(*cond.glare)).astype(np.float32)
    truth["glare_peak"] = float(glare.max())
    h, w = small.shape
    up = lambda a: cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    Dxp, Dyp = up(Dx) * ppm, up(Dy) * ppm
    qy_, qx_ = np.mgrid[0:h, 0:w].astype(np.float32)
    px_, py_ = qx_.copy(), qy_.copy()
    for _ in range(6):
        px_ = qx_ - cv2.remap(Dxp, px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        py_ = qy_ - cv2.remap(Dyp, px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    paper = cv2.remap(small.astype(np.float32), px_, py_, cv2.INTER_LINEAR, borderValue=0)
    sh = cv2.remap(up(shade), px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gl = cv2.remap(up(glare), px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inkness = 1.0 - paper / 255.0
    out = paper * sh + 255.0 * gl * (1.0 + cond.gloss * inkness)
    cover = ((px_ >= 0) & (py_ >= 0) & (px_ <= w - 1) & (py_ <= h - 1)).astype(np.uint8) * 255
    return out, cover


synth.paper_in_space = _pil

# ---- ink-related variations (omr.ink): lightness of the marks, erasures, printer black
_KX = {"pencil": None, "black": None, "kinds": None}


def _draw_ink(img, cx, cy, R_mm, rng, kind, **kw):
    if kind == "fill" and _KX["pencil"] is not None:
        kw["gray"] = rng.uniform(*_KX["pencil"])
    return _ORIG["di"](img, cx, cy, R_mm, rng, kind, **kw)


def _print_defects(raster, page, rng, **kw):
    if _KX["black"] is not None:
        kw["black"] = _KX["black"]
    return _ORIG["pd"](raster, page, rng, **kw)


synth.draw_ink = _draw_ink
synth.print_defects = _print_defects

_H = dict(frame=(1080, 1920), tilt=15, rot="upright", printer=True)
HCONDS = {
    "h_multi": HCond("h_multi", n_shadows=3, curl=(0, 6), **_H),
    "h_idedge": HCond("h_idedge", n_shadows=1, shadow_depth=(0.4, 0.65), edge_through_id=True, curl=(0, 4), **_H),
    "h_dimgrad": HCond("h_dimgrad", gradient=0.6, shadow=0.45, gain=(0.22, 0.4), noise=(3, 6), n_shadows=1,
                       shadow_depth=(0.2, 0.4), curl=(0, 4), **{**_H, "tilt": 15}),
    "h_glare": HCond("h_glare", glare=(0.5, 1.0), curl=(2, 10), **_H),
    "h_pencil": HCond("h_pencil", glare=(0.2, 0.5), gloss=3.0, light_pencil=True, curl=(2, 8), **_H),
    "h_all": HCond("h_all", n_shadows=2, glare=(0.2, 0.6), gloss=2.0, raking=0.7, curl=(4, 14), edge_curl=0.5,
                   gain=(0.4, 0.9), gradient=0.4, **_H),
}
synth.CONDS.update(HCONDS)
_P = dict(frame=(1080, 1920), tilt=15, rot="upright", printer=True)
KCONDS = {   # name: (cond, overrides)
    "k_light": (synth.Cond("k_light", **_P), {"pencil": (140, 185)}),            # very light pencil only
    "k_pencil": (synth.Cond("k_pencil", light_pencil=True, **_P), {}),            # light pencil 110-150
    "k_erase": (synth.Cond("k_erase", hard_rate=0.7, **_P), {"kinds": ("erased_blank", "erased_clear")}),
    "k_pale": (synth.Cond("k_pale", **_P), {"black": (0.25, 0.45)}),             # pale printer black
    "k_darkprt": (synth.Cond("k_darkprt", **_P), {"black": (0.9, 1.0)}),         # very dark printer black
    "k_pale_light": (synth.Cond("k_pale_light", **_P), {"black": (0.25, 0.45), "pencil": (120, 170)}),
    "k_mixed": (synth.Cond("k_mixed", hard_rate=0.3, **_P), {}),                 # pen + pencil mix, many hard cases
}
for _n, (_c, _o) in KCONDS.items():
    synth.CONDS[_n] = _c
KORDER = list(KCONDS)
HARD_ORDER = list(HCONDS)
STD_ORDER = ["clean", "tilted", "blurred", "dark", "phone", "lowres", "sp_blank", "sp_ghosts", "sp_fewghost",
             "sp_pencil", "sp_dark", "printer", "handheld", "curl", "glare", "shadow", "raking"]


# ------------------------------------------------------------------------------ one frame, every method
def run_job(args):
    assets, name, cond_name, seed, ppm, methods = args
    L, pdf = assets[name]
    cond = synth.CONDS[cond_name]
    rng = np.random.default_rng(seed)
    if isinstance(cond, HCond) and cond.edge_through_id:
        g = L.pages[0].id_grid
        b = g.columns[int(rng.integers(len(g.columns)))].bubbles[int(rng.integers(10))]
        cond = dataclasses.replace(cond, edge_pt=(b.cx + rng.uniform(-3, 3), b.cy + rng.uniform(-3, 3)))
    serial = 1 + seed % synth.mk_serial_max()
    ov = KCONDS[cond_name][1] if cond_name in KCONDS else {}
    _KX.update({"pencil": ov.get("pencil"), "black": ov.get("black"), "kinds": ov.get("kinds")})
    synth.HARD_KINDS = ov.get("kinds") or _ORIG["hk"]
    frame, truth = synth.make_frame(L, pdf, cond, rng, mode="sheet", serial=serial)
    kw = {} if ppm is None else {"px_per_mm": ppm}
    out = {"cond": cond_name, "seed": seed, "name": name, "res": {}}
    for m in methods:
        install(m)
        res = R.recognize(frame, **kw)
        ev = synth.evaluate(res, truth)
        if ev["id_wrong"] or ev["subject_wrong"]:
            print("WRONG ID/SUBJECT", m, cond_name, seed, name, "read", res.student_id, res.subject, "truth", truth["student_id"],
                  truth.get("subject"), "id_digits", [(d.state, d.value, d.scores) for d in res.id_digits if d.ambiguous or d.state != "single"], flush=True)
        out["res"][m] = {"status": res.status, "ev": ev, "ms": res.timing_ms["total"], "dark_ms": _STATE["ms"],
                         "score_ms": res.timing_ms.get("score", 0.0), "id": res.student_id,
                         "ink": _STATE.get("inkinfo"),
                         "unrel": float(np.mean(np.concatenate(_STATE["unrel"]))) if _STATE["unrel"] else 0.0}
    return out


def table(out, methods, conds, title=""):
    hdr = (f"{'condition':<11}{'method':<7}{'frames':>6}{'read%':>6}{'silent':>7}{'/qs':>6}{'clr acc%':>9}{'clr flg%':>9}"
           f"{'hard ok%':>9}{'ID read%':>9}{'ID wrong':>9}{'unrel%':>7}{'dark ms':>8}")
    print(title)
    print(hdr)
    print("-" * len(hdr))
    for cn in conds + ["ALL"]:
        rows = [r for r in out if cn == "ALL" or r["cond"] == cn]
        if not rows:
            continue
        for m in methods:
            ok = [r["res"][m] for r in rows if r["res"][m]["status"] == "ok"]
            tot = {}
            for r in ok:
                synth.merge(tot, r["ev"])
            g = lambda k: tot.get(k, 0)
            div = lambda a, b: 100.0 * a / b if b else float("nan")
            ms = np.median([r["dark_ms"] for r in ok]) if ok else float("nan")
            un = 100 * np.mean([r["unrel"] for r in ok]) if ok else float("nan")
            print(f"{cn:<11}{m:<7}{len(rows):>6}{div(len(ok), len(rows)):>6.0f}{g('q_silent'):>7}{g('q_total'):>6}"
                  f"{div(g('q_clear_ok'), g('q_clear')):>9.2f}{div(g('q_clear_flagged'), g('q_clear')):>9.2f}"
                  f"{div(g('q_hard_ok'), g('q_hard')):>9.1f}{div(g('id_ok'), g('id_readable')):>9.1f}{g('id_wrong'):>9}{un:>7.2f}{ms:>8.1f}")
        for m in methods:
            ii = [r["res"][m]["ink"] for r in rows if r["res"][m]["status"] == "ok" and r["res"][m].get("ink")]
            if ii:
                fr = [x[1] for x in ii]
                print(f"{'':<11}{m:<7} ink floor/black: auto {100 * np.mean([x[0] == 'auto' for x in ii]):.0f}%  "
                      f"floor p10/50/90 = {np.percentile(fr, 10):.2f}/{np.percentile(fr, 50):.2f}/{np.percentile(fr, 90):.2f}")
        print()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="E")
    ap.add_argument("--conds", default=None)
    ap.add_argument("--hard", action="store_true", help="the h_* conditions")
    ap.add_argument("--ink", action="store_true", help="the k_* conditions (mark lightness, erasures, printer black)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--ppm", type=float, default=None)
    ap.add_argument("--list-silent", action="store_true", help="also list every wrong student ID / subject")
    o = ap.parse_args(argv)
    methods = o.methods.split(",")
    conds = o.conds.split(",") if o.conds else (HARD_ORDER if o.hard else KORDER if o.ink else STD_ORDER)
    work = os.path.join(tempfile.gettempdir(), "streamomr_illum_bench")
    assets = synth.prepare(work)
    names = list(assets)
    jobs = []
    for ci, cn in enumerate(conds):
        for i in range(o.n):
            jobs.append((assets, names[i % len(names)], cn, o.seed * 1_000_003 + ci * 10_007 + i, o.ppm, methods))
    t0 = time.time()
    if o.jobs > 1:
        with mp.Pool(o.jobs) as pool:
            out = pool.map(run_job, jobs, chunksize=2)
    else:
        out = [run_job(j) for j in jobs]
    print(f"{len(jobs)} frames x {len(methods)} methods in {time.time() - t0:.0f}s\n")
    table(out, methods, conds)
    if o.list_silent:
        for r in out:
            for m in methods:
                ev = r["res"][m]["ev"]
                for f in ev.get("failures", []):
                    if not f[6]:
                        print("silent", m, r["cond"], r["seed"], r["name"], f[:6], np.round(f[7], 2))
    return out


if __name__ == "__main__":
    main()
