"""Record real inputs / outputs of the Python logic modules as JSON fixtures for the jsc tests (web/test/logic/test.js).

    .venv/bin/python web/test/logic/export_fixtures.py [--regen] [--no-video]

Recognition results (synthetic frames + the iPhone devlog video) are cached in web/test/fixtures/logic/_cache so a re-run only
re-exports.  Never copies realasset files: only recognition results (numbers) derived from them are written.
"""
import copy, dataclasses, json, math, os, pickle, random, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2  # noqa: E402
import synth  # noqa: E402
from omr import markers as mk, layout as lay, identity as idt, revision as rev, fusion as fus, stream as stm  # noqa: E402
from omr import recognize as R  # noqa: E402

FIX = os.path.join(ROOT, "web", "test", "fixtures", "logic")
CACHE = os.path.join(FIX, "_cache")
VIDEO = os.path.join(os.environ.get("STREAMOMR_REALASSET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "realasset")), "devlog_2026-10-14", "video", "segment_01.mp4")
os.makedirs(CACHE, exist_ok=True)


# ------------------------------------------------------------------ serialisation (the TYPES.md shape)
def nn(x):
    """float -> JSON-safe (NaN / inf -> None)."""
    return None if isinstance(x, float) and not math.isfinite(x) else x


def plain(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return nn(v)
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(plain(x) for x in v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    return v


def ser_field(f):
    if f is None:
        return None
    fn = {s: {"u": plain(u), "v": plain(vv)} for s, (u, vv) in f._fn.items()}
    return {"XL": R.trk._XL, "XR": R.trk._XR, "YT": R.trk._YT, "YB": R.trk._YB, "c": plain(f.c), "fn": fn}


def ser_raw(raw):
    if raw is None:
        return None
    d = {f.name: plain(getattr(raw, f.name)) for f in dataclasses.fields(raw)}
    d["decided_sig"] = plain(getattr(raw, "decided_sig", None))
    return d


def ser_result(r, raw=True):
    d = {}
    for f in dataclasses.fields(r):
        n = f.name
        if n in ("warped", "name_crop"):
            d[n] = None
        elif n == "raw":
            d[n] = ser_raw(r.raw) if raw else None
        elif n == "dewarp":
            d[n] = ser_field(r.dewarp) if raw else None
        elif n == "marker_corners":
            d[n] = [[int(i), plain(c)] for i, c in r.marker_corners.items()]
        elif n == "timing_ms":
            d[n] = {}
        else:
            d[n] = plain(getattr(r, n))
    return d


def ser_event_result(r):
    d = ser_result(r, raw=False)
    d.pop("raw"); d.pop("dewarp")
    ts = getattr(r, "trace_src", None)
    d["fused_raw"] = ser_raw(ts[0]) if ts else None
    d["trace_n"] = len(ts[1]) if ts else None
    return d


def ser_event(ev):
    out = {}
    for k, v in ev.items():
        if k in ("commit", "revision"):
            v = dict(v)
            v["result"] = ser_event_result(v["result"])
        out[k] = plain(v)
    return out


def ser_decision(d):
    return plain(d)


# ------------------------------------------------------------------ recognition pool
def jitter(base, rng, heavy=False):
    f = base.astype(np.float32)
    g = rng.uniform(0.6, 1.05) if heavy else rng.uniform(0.92, 1.05)
    f *= g
    M = np.float32([[1, 0, rng.uniform(-1.5, 1.5)], [0, 1, rng.uniform(-1.5, 1.5)]])
    f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    if heavy:
        s = rng.uniform(1.2, 2.6)
        f = cv2.GaussianBlur(f, (0, 0), s)
    f += rng.normal(0, rng.uniform(5, 12) if heavy else rng.uniform(1, 4), f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


SCENARIOS = [  # (template, cond, seed, serial)
    ("q30_abcd", "clean", 1, 11), ("q30_abcd", "handheld", 2, 12), ("q30_abcd", "printer", 3, 13), ("q30_abcd", "curl", 4, 14),
    ("q45", "clean", 5, 15), ("q45", "phone", 6, 16), ("q45", "glare", 7, 17), ("q45", "dark", 8, 18),
    ("q40_nosubj", "clean", 9, 19), ("q40_nosubj", "lowres", 10, 20), ("q100u80_2p", "tilted", 11, 21),
    ("q100_2p", "raking", 12, 22), ("q30_abcd", "sp_pencil", 13, 23), ("q45", "curl_hand", 14, 24),
    ("q30_abcd", "clean", 15, 0), ("q45", "printer", 16, 26), ("q30_abcd", "shadow", 17, 27), ("q45", "handheld", 18, 28),
]
NJ = 14


def build_pool():
    path = os.path.join(CACHE, "synth_pool.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    import tempfile
    tmp = tempfile.mkdtemp()
    names = sorted({s[0] for s in SCENARIOS} | {"q30_abcd"})
    assets = synth.prepare(tmp, {n: synth.STANDARD_TEMPLATES[n] for n in names})
    pool = {"scen": [], "empty": [], "bad": []}
    t0 = time.time()
    for si, (name, cond, seed, serial) in enumerate(SCENARIOS):
        L, pdf = assets[name]
        rng = np.random.default_rng(seed)
        frame, truth = synth.make_frame(L, pdf, synth.CONDS[cond], rng, serial=serial or None)
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames = []
        for j in range(NJ):
            heavy = j % 4 == 3
            img = g if j == 0 else jitter(g, rng, heavy)
            frames.append((R.recognize(img, keep_images=False), heavy))
        pool["scen"].append(frames)
        print(f"  scenario {si} {name}/{cond}: {[f[0].status for f in frames].count('ok')}/{NJ} ok  ({time.time() - t0:.0f}s)", flush=True)
    L, pdf = assets["q30_abcd"]
    for s in range(6):
        fr, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(100 + s), mode="none")
        pool["empty"].append(R.recognize(fr, keep_images=False))
    for s in range(30):
        mode = ["thumb2", "cut", "thumb2", "cut", "sheet"][s % 5]
        cond = ["clean", "phone", "tilted", "curl"][s % 4]
        fr, _ = synth.make_frame(L, pdf, synth.CONDS[cond], np.random.default_rng(200 + s), mode=mode)
        r = R.recognize(fr, keep_images=False)
        if r.status != "ok":
            pool["bad"].append(r)
    pickle.dump(pool, open(path, "wb"))
    return pool


def build_video(stride=3):
    path = os.path.join(CACHE, "video_pool.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    cap = cv2.VideoCapture(VIDEO)
    res, i, t0 = [], 0, time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            res.append((i / fps, R.recognize(fr, keep_images=False)))
            if len(res) % 50 == 0:
                print(f"  video frame {i} ({time.time() - t0:.0f}s) statuses so far {len(res)}", flush=True)
        i += 1
    pickle.dump(res, open(path, "wb"))
    return res


# ------------------------------------------------------------------ mutations (like tests/test_stream.py)
def flip_answer(r, q=0):
    r = copy.copy(r)
    r.answers = [dataclasses.replace(a) for a in r.answers]
    a = r.answers[q]
    a.marked = [(a.marked[0] + 1) % 4] if a.marked else [0]
    a.state = "single"
    return r


def other_student(r):
    r = copy.copy(r)
    r.student_id = "99999999"
    return r


def with_status(r, status):
    r = copy.copy(r)
    r.status = status
    return r


def sharper(r, s):
    r = copy.copy(r)
    r.quality = dict(r.quality, sharpness=s)
    return r


class Pool:
    """Collects every distinct result object that goes into a fixture; sequences refer to it by index."""
    def __init__(self):
        self.items, self.idx = [], {}

    def add(self, r):
        k = id(r)
        if k not in self.idx:
            self.idx[k] = len(self.items)
            self.items.append(r)
        return self.idx[k]

    def dump(self):
        return [ser_result(r) for r in self.items]



def redecide(r, raw):
    """Copy of result r whose raw measurements are `raw` and whose decisions are decided from them (a consistent, trustworthy frame)."""
    r = copy.copy(r)
    raw = copy.copy(raw)
    d = R.decide_page(raw)
    r.raw = raw
    r.answers, r.id_digits, r.student_id, r.calibration = d.answers, d.id_digits, d.student_id, d.calibration
    r.subject_digits, r.subject = d.subject_digits, d.subject
    r.flags = [f for f in r.flags if f not in fus.ID_FLAGS] + d.flags
    raw.decided_sig = R.decision_sig(d.answers, d.student_id, d.subject)
    return r


def swung(r, rng, nb, delta):
    raw = copy.copy(r.raw)
    raw.FA = raw.FA.copy()
    for _ in range(nb):
        i, j = rng.randrange(raw.FA.shape[0]), rng.randrange(raw.FA.shape[1])
        v = float(raw.FA[i, j])
        raw.FA[i, j] = np.float32(v - delta if v > delta else v + delta)
    return redecide(r, raw)


def scrambled(r, rng, frac):
    """Same frame with a fraction of the answer rows turned into another single mark (re-decided from the bubbles)."""
    raw = copy.copy(r.raw)
    raw.FA, raw.SA = raw.FA.copy(), raw.SA.copy()
    for i in range(raw.FA.shape[0]):
        if rng.random() < frac:
            raw.FA[i, :] = np.float32(0.02)
            raw.FA[i, rng.randrange(raw.FA.shape[1])] = np.float32(0.97)
            raw.SA[i, :] = 1.0
    return redecide(r, raw)


def make_sequences(pool, vid, rng):
    scen = [[f[0] for f in fr] for fr in pool["scen"]]
    heavy = [[f[1] for f in fr] for fr in pool["scen"]]
    oks = [i for i, fr in enumerate(scen) if sum(r.status == "ok" for r in fr) >= NJ // 2]
    empties, bads = pool["empty"], pool["bad"]
    seqs = []

    def sheet_frames(si, n, mode):
        fr = [r for r in scen[si]]
        if mode == "improve":     # blurry frames first, sharp last (revision)
            fr = [r for _, r in sorted(zip([r.quality.get("sharpness", 0.0) for r in fr], fr), key=lambda x: x[0])]
        elif mode == "shuffle":
            fr = fr[:]; rng.shuffle(fr)
        out = []
        while len(out) < n:
            out += fr
        return out[:n]

    for s in range(70):
        segs = []
        n_seg = rng.randint(2, 7)
        for _ in range(n_seg):
            kind = rng.choice(["E", "P", "S", "S", "S", "M"])
            if kind == "E":
                segs += [rng.choice(empties) for _ in range(rng.randint(1, 8))]
            elif kind == "P" and bads:
                segs += [rng.choice(bads) for _ in range(rng.randint(1, 5))]
            elif kind == "M":
                si = rng.choice(oks)
                base = rng.choice([r for r in scen[si] if r.status == "ok"] or scen[si])
                if base.status != "ok":
                    continue
                m = rng.choice(["flip", "other", "unknown", "bad_geometry", "inconsistent"])
                f = {"flip": flip_answer, "other": other_student, "unknown": lambda r: with_status(r, "unknown_form"),
                     "bad_geometry": lambda r: with_status(r, "bad_geometry"), "inconsistent": lambda r: with_status(r, "inconsistent")}[m]
                segs += [f(base) for _ in range(rng.randint(1, 6))]
            else:
                si = rng.choice(oks if rng.random() < 0.8 else range(len(scen)))
                segs += sheet_frames(si, rng.randint(2, 30), rng.choice(["order", "shuffle", "improve"]))
        params = dict(k=rng.choice([2, 3, 3, 4]), m=rng.choice([3, 5, 5]), hint_after=rng.choice([0.0, 0.5, 2.0]),
                      fuse=rng.random() < 0.85, hold_s=rng.choice([1.5, 1.5, 0.3, 5.0]))
        if rng.random() < 0.15:
            params["allowed_forms"] = sorted({r.form_key for r in segs if r.form_key})[:1]
        if rng.random() < 0.15:
            params["allowed_subjects"] = sorted({r.subject for r in segs if r.subject})[:2]
        dt = rng.choice([0.125, 0.33, 0.5])
        seqs.append({"name": f"synth{s}", "params": params, "frames": segs, "ts": [i * dt for i in range(len(segs))]})

    # borderline sequences: bubble swings around CONTRADICT_FILL, answer scrambles around SWAP_FRACTION, fine-grained timing
    for s2 in range(110):
        si = rng.choice(oks)
        base = [r for r in scen[si] if r.status == "ok" and r.raw is not None and r.raw.verified]
        if len(base) < 3:
            continue
        segs = []
        for _ in range(rng.randint(2, 4)):
            kind = rng.choice(["swing", "swing", "scramble", "plain"])
            n = rng.randint(3, 16)
            if kind == "swing":
                nb, delta = rng.randint(1, 3), rng.choice([0.5, 0.55, 0.59, 0.6, 0.61, 0.65, 0.7])
                src = [rng.choice(base) for _ in range(n)]
                segs += [swung(r, rng, nb, delta) if rng.random() < 0.4 else r for r in src]
            elif kind == "scramble":
                f = rng.choice([0.1, 0.2, 0.28, 0.3, 0.32, 0.4])
                b = rng.choice(base)
                segs += [scrambled(b, rng, f) for _ in range(n)] if rng.random() < 0.5 else [b] * 4 + [scrambled(b, rng, f) for _ in range(n)]
            else:
                segs += [rng.choice(base) for _ in range(n)]
            if rng.random() < 0.4:
                segs += [rng.choice(empties) for _ in range(rng.randint(1, 4))]
        ts, t = [], 0.0
        for _ in segs:
            t += rng.uniform(0.05, 0.6)
            ts.append(t)
        params = dict(k=rng.choice([2, 3, 4]), m=rng.choice([3, 5]), hint_after=rng.uniform(0.2, 2.5), fuse=rng.random() < 0.9,
                      hold_s=rng.uniform(0.1, 3.0))
        seqs.append({"name": f"border{s2}", "params": params, "frames": segs, "ts": ts})
    # handcrafted (tests/test_stream.py-like)
    A = next(r for si in oks for r in scen[si] if r.status == "ok" and r.student_id and not r.id_guessed)
    E, T_ = empties[0], (bads[0] if bads else empties[0])
    seqs.append({"name": "enter_leave", "params": {}, "frames": [E, E, T_, A, A, A, A, A, A, A, A, A, T_, E, E, E, E, E, E] + [A] * 6 + [E] * 6, "ts": [i * 0.125 for i in range(37)]})
    seqs.append({"name": "flicker", "params": {}, "frames": [A] * 4 + [flip_answer(A)] * 6 + [A] * 3 + [flip_answer(A)] * 5, "ts": [i * 0.125 for i in range(18)]})
    if vid:
        for stride, nm in ((1, "video_s3"), (3, "video_s9")):
            fr = [r for _, r in vid[::stride]]
            ts = [t for t, _ in vid[::stride]]
            seqs.append({"name": nm, "params": {}, "frames": fr, "ts": ts})
        fr = [r for _, r in vid[::3]]; ts = [t for t, _ in vid[::3]]
        seqs.append({"name": "video_s9_k2", "params": {"k": 2, "m": 3}, "frames": fr, "ts": ts})
        seqs.append({"name": "video_s9_nofuse", "params": {"fuse": False}, "frames": fr, "ts": ts})
        seqs.append({"name": "video_s9_hold", "params": {"hold_s": 5.0}, "frames": fr, "ts": ts})
    return seqs


def run_stream(seq, P):
    s = stm.StreamSession(**seq["params"])
    evs = []
    for i, (r, t) in enumerate(zip(seq["frames"], seq["ts"])):
        evs.append(ser_event(s.feed(r, t, payload=i)))
    fin = s.finish(seq["ts"][-1])
    return {"name": seq["name"], "params": seq["params"], "frames": [P.add(r) for r in seq["frames"]], "ts": seq["ts"],
            "events": evs, "finish": None if fin is None else dict(plain({k: v for k, v in fin.items() if k != "result"}), result=ser_event_result(fin["result"])),
            "missed": plain(s.missed()), "missed_final": plain(s.missed(True)), "unread": plain(s.unread_episodes()),
            "unread_final": plain(s.unread_episodes(True)), "state": s.state}


def fusion_trace(frames, P):
    track, steps = None, []
    for r in frames:
        st = {"idx": P.add(r), "trust": fus.trustworthy(r), "quad": plain(fus.sheet_quad(r)) if r.status == "ok" else None}
        if r.status == "ok" and st["trust"]:
            if track is not None:
                st["incompat"] = track.incompatible(r)
                if st["incompat"]:
                    track = None
            if track is None:
                track = fus.Track(len(steps) + 1, r, 0.0)
            else:
                track.add(r, 0.0, None)
            st.update(fused=ser_raw(track.fused_raw()), dec=ser_decision(track.decisions()), serial=track.serial(),
                      idu=track.id_unreadable(), sbu=track.subject_unreadable(), n=track.n,
                      applied=ser_event_result(track.apply(r)))
        steps.append(st)
    return steps


def perturbed_raws(pool_results, rng, n=400):
    out = []
    oks = [r for r in pool_results if r.raw is not None]
    for _ in range(n):
        r = rng.choice(oks)
        raw = copy.copy(r.raw)
        nrng = np.random.default_rng(rng.randint(0, 1 << 30))
        mode = rng.choice(["noise", "sentinel", "nan", "unverified", "low", "mix"])
        def noisy(a, s):
            return np.clip(a + nrng.normal(0, s, a.shape), 0, 1).astype(a.dtype) if a is not None else None
        if mode in ("noise", "mix"):
            raw.FA, raw.FI = noisy(raw.FA, 0.15), noisy(raw.FI, 0.15)
            raw.FS = noisy(raw.FS, 0.15)
            raw.SA = np.clip(raw.SA + nrng.normal(0, 0.2, raw.SA.shape), 0, 2)
            raw.MA = noisy(raw.MA, 0.1); raw.MI = noisy(raw.MI, 0.1); raw.MS = noisy(raw.MS, 0.1)
        if mode in ("sentinel", "mix"):
            for F, S in ((raw.FA, raw.SA), (raw.FI, raw.SI)):
                m = nrng.random(F.shape) < 0.08
                F = F.copy(); S = S.copy()
                F[m] = 0.5; S[m] = 0.3
            raw.FA = raw.FA.copy(); raw.SA = raw.SA.copy()
            m = nrng.random(raw.FA.shape) < 0.08
            raw.FA[m] = 0.5; raw.SA[m] = 0.3
            if raw.MA is not None:
                raw.MA = raw.MA.copy(); raw.MA[m] = np.nan
        if mode == "nan" and raw.MA is not None:
            raw.MA = raw.MA.copy(); raw.MA[nrng.random(raw.MA.shape) < 0.1] = np.nan
        if mode == "unverified":
            raw.verified = False
        if mode == "low":
            raw.FA = (raw.FA * 0.45).astype(raw.FA.dtype); raw.FI = (raw.FI * 0.45).astype(raw.FI.dtype)
            if raw.MA is not None:
                raw.MA = (raw.MA * 0.4).astype(raw.MA.dtype); raw.MI = (raw.MI * 0.4).astype(raw.MI.dtype)
        out.append(raw)
    return out


def export_decide(pool_results, rng):
    cases = []
    raws = [r.raw for r in pool_results if r.raw is not None] + perturbed_raws(pool_results, rng)
    for raw in raws:
        d = R.decide_page(raw)
        cases.append({"raw": ser_raw(raw), "dec": ser_decision(d), "sig": plain(R.decision_sig(d.answers, d.student_id, d.subject))})
    return cases


# ------------------------------------------------------------------ layouts / markers
def export_layouts():
    rng = random.Random(5)
    tpls = []
    for q, c, d in [(45, 5, 8), (30, 4, 6), (100, 5, 8), (10, 2, 1), (200, 5, 8), (60, 10, 12), (20, 3, 4), (150, 4, 10), (1, 5, 8), (300, 5, 8)]:
        tpls.append(dict(questions=q, choices=c, id_digits=d))
    for _ in range(90):
        q = rng.choice([5, 17, 30, 45, 60, 80, 100, 120, 170, 250, 400, 700])
        t = dict(questions=q, choices=rng.randint(2, 10), id_digits=rng.randint(1, 12), subject_digits=rng.randint(0, 3),
                 tracks=rng.random() < 0.7, min_bubble_mm=rng.choice([4.0, 4.25, 4.5, 5.0, 5.75]),
                 title=rng.choice(["", "Midterm", "2026 수학능력시험 모의고사 제1회 전국연합"]))
        if rng.random() < 0.4:
            t["pages"] = rng.randint(1, 6)
        if rng.random() < 0.4:
            t["used"] = rng.randint(1, q)
        if rng.random() < 0.5:
            st = rng.choice(["1", "A", "a", "가"])
            t["choice_labels"] = mk.choice_labels(mk.LABEL_STYLES.index(st), t["choices"]) if t["choices"] <= (10 if st == "가" else 26) else None
        tpls.append(t)
    out = []
    for t in tpls:
        try:
            L = lay.build_layout(lay.Template(**t))
        except Exception as e:      # noqa: BLE001
            out.append({"template": t, "error": True})
            continue
        item = {"template": t, "layout": L.to_dict(), "describe": lay.describe(L),
                "headers": [dataclasses.asdict(L.header(p)) for p in range(L.total_pages)]}
        h0 = L.header(0)
        item["from_header"] = lay.layout_from_header(h0).to_dict() if True else None
        item["form_key"] = h0.form_key
        out.append(item)
    ok = [o for o in out if "layout" in o]
    errs = [o for o in out if "error" in o]
    return {"layouts": out, "n_ok": len(ok), "n_err": len(errs),
            "rects": {"marker": {str(p): lay.marker_rects(p) for p in range(3)}, "meta": lay.meta_cell_rects(), "header": lay.header_cell_rects(),
                      "ticks": lay.track_tick_rects(), "centers": plain(lay.track_tick_centers()), "serial": lay.serial_cell_rects(),
                      "serial_label": dataclasses.asdict(lay.serial_label(37))}}


def export_markers():
    rng = random.Random(7)
    out = {"bits": [''.join('1' if x else '0' for x in mk.marker_bits(m).ravel()) for m in range(48)],
           "image": {str(m): {"sum": int(mk.marker_image(m, px).sum()), "shape": list(mk.marker_image(m, px).shape),
                              "hash": int(np.sum(mk.marker_image(m, px).astype(np.int64) * np.arange(mk.marker_image(m, px).size).reshape(mk.marker_image(m, px).shape)) % 1000003)}
                     for m, px in ((0, 200), (5, 100), (47, 37))},
           "ids": [[p, c, mk.marker_id(p, c)] for p in range(12) for c in mk.CORNERS],
           "crc8": [], "meta": [], "serial": [], "headers": [], "labels": [], "style": []}
    for _ in range(40):
        data = bytes(rng.randrange(256) for _ in range(rng.randint(0, 8)))
        out["crc8"].append([list(data), mk.crc8(data), mk.crc16(data)])
    for _ in range(80):
        total = rng.randint(1, 12); page = rng.randrange(total); fid = rng.randrange(65536)
        bits = mk.encode_meta(fid, page, total)
        flips = rng.choice([0, 0, 1, 2])
        b2 = list(bits)
        for _ in range(flips):
            i = rng.randrange(len(b2)); b2[i] = not b2[i]
        m = mk.decode_meta(b2)
        out["meta"].append({"args": [fid, page, total], "bits": bits, "bits2": b2, "dec": None if m is None else dataclasses.asdict(m)})
    for _ in range(80):
        s = rng.randint(1, mk.SERIAL_MAX)
        bits = mk.encode_serial(s)
        b2 = list(bits)
        for _ in range(rng.choice([0, 0, 1])):
            i = rng.randrange(len(b2)); b2[i] = not b2[i]
        out["serial"].append({"serial": s, "bits": bits, "bits2": b2, "dec": mk.decode_serial(b2)})
    for _ in range(150):
        pages = rng.randint(1, 12)
        q = rng.randint(1, 1023)
        h = mk.Header(q, rng.randint(1, q), rng.randint(2, 10), rng.randint(1, 12), rng.randint(0, 3), pages, rng.randrange(pages),
                      rng.randrange(4), rng.randrange(8), rng.choice([1, 2]))
        rows = mk.encode_header(h)
        r2 = [list(r) for r in rows]
        for _ in range(rng.choice([0, 0, 1, 2])):
            a = rng.randrange(2); i = rng.randrange(len(r2[a])); r2[a][i] = not r2[a][i]
        d = mk.decode_header(r2)
        out["headers"].append({"h": dataclasses.asdict(h), "form_key": h.form_key, "labels": h.choice_labels(), "rows": rows, "rows2": r2,
                               "dec": None if d is None else dataclasses.asdict(d),
                               "dec_data": None if (dd := mk.decode_header_data([c for r in rows for c in r[4:-4]])) is None else dataclasses.asdict(dd)})
    for st in range(4):
        for n in (1, 3, 5, 10):
            out["labels"].append([st, n, mk.choice_labels(st, n)])
    for labs in (["1", "2", "3"], ["A", "B"], ["a", "b", "c", "d"], ["가", "나"], ["x", "y"], ["1", "3"], [], ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "["]):
        out["style"].append([labs, mk.label_style(labs)])
    out["form_keys"] = [{"key": k, "header": dataclasses.asdict(lay.header_from_form_key(k, 0)) if k.startswith("Q") else None}
                        for k in ["Q100U80C5D8S2P2L0B0V1", "Q45U45C5D8S2P1L1B2V2"]]
    return out


def export_identity_revision(pool_results, rng):
    out = {"identity": [], "revision": []}
    muts = []
    for r in pool_results[:]:
        muts.append(r)
        if r.status == "ok":
            m = copy.copy(r); m.student_id = None; m.flags = list(r.flags) + ["id_guessed"]; muts.append(m)
            m = copy.copy(r); m.serial = None; m.subject = ""; m.subject_digits = []; muts.append(m)
            m = copy.copy(r); m.flags = list(r.flags) + ["subject_unreadable", "contradiction"]; muts.append(m)
    for r in muts:
        d = ser_result(r, raw=False)
        out["identity"].append({"r": d, "serial_of": idt.serial_of(r), "id_text": idt.student_id_text(r), "subject_text": idt.subject_text(r),
                                "sheet_key": idt.sheet_key(r), "severity": idt.severity(r), "doubtful": rev.doubtful(r) if r.status == "ok" else None,
                                "needs_review": r.needs_review, "id_guessed": r.id_guessed, "subject_guessed": r.subject_guessed,
                                "fmt": idt.format_serial(r.serial), "key": idt.make_sheet_key(5, "12?4", "0")})
    oks = [r for r in muts if r.status == "ok"]
    for _ in range(500):
        a, b = rng.choice(oks), rng.choice(oks)
        if len(a.answers) != len(b.answers) and rng.random() < 0.8:
            b = a
        m, ch, co = rev.merge(a, b)
        out["revision"].append({"base": ser_result(a, raw=False), "now": ser_result(b, raw=False), "merged": ser_result(m, raw=False),
                                "changed": ch, "contra": co})
    return out


def main():
    regen = "--regen" in sys.argv
    if regen:
        for f in os.listdir(CACHE):
            os.remove(os.path.join(CACHE, f))
    rng = random.Random(2026)
    t0 = time.time()
    json.dump(export_layouts(), open(os.path.join(FIX, "layouts.json"), "w"))
    json.dump(export_markers(), open(os.path.join(FIX, "markers.json"), "w"))
    print(f"layouts+markers exported ({time.time() - t0:.0f}s)", flush=True)
    pool = build_pool()
    vid = None if "--no-video" in sys.argv else build_video(3)
    print(f"pool ready ({time.time() - t0:.0f}s)", flush=True)
    allres = [f[0] for fr in pool["scen"] for f in fr] + pool["empty"] + pool["bad"]
    json.dump(export_decide(allres, rng), open(os.path.join(FIX, "decide.json"), "w"))
    json.dump(export_identity_revision(allres, rng), open(os.path.join(FIX, "identity_revision.json"), "w"))
    print(f"decide/identity/revision exported ({time.time() - t0:.0f}s)", flush=True)
    P = Pool()
    seqs = make_sequences(pool, vid, rng)
    streams = [run_stream(s, P) for s in seqs]
    fus_steps = []
    for si in range(len(pool["scen"])):
        fus_steps.append(fusion_trace([f[0] for f in pool["scen"][si]] * 2, P))
    for s in seqs[:25] + [q for q in seqs if q["name"].startswith("border")][:50]:
        fus_steps.append(fusion_trace(s["frames"][:60], P))
    if vid:
        fus_steps.append(fusion_trace([r for _, r in vid[::3]], P))
    json.dump({"pool": P.dump(), "streams": streams, "fusion": fus_steps}, open(os.path.join(FIX, "stream.json"), "w"))
    n_events = sum(len(s["events"]) for s in streams)
    n_commits = sum(1 for s in streams for e in s["events"] if "commit" in e)
    n_rev = sum(1 for s in streams for e in s["events"] if "revision" in e)
    print(f"stream fixtures: {len(streams)} sequences, {n_events} events, {n_commits} commits, {n_rev} revisions, pool {len(P.items)} results ({time.time() - t0:.0f}s)")
    json.dump({"streams": len(streams), "events": n_events, "commits": n_commits, "revisions": n_rev, "pool": len(P.items),
               "fusion_traces": len(fus_steps), "fusion_steps": sum(len(f) for f in fus_steps), "decide_cases": len(allres)},
              open(os.path.join(FIX, "counts.json"), "w"))


if __name__ == "__main__":
    main()
