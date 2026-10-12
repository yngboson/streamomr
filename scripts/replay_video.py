#!/usr/bin/env python
"""Replay a recorded video (or a folder of photos) through the exact live pipeline: recognize + StreamSession.

    python scripts/replay_video.py sheets.mp4 --data data              # sheets describe themselves: no exam needed
    python scripts/replay_video.py sheets.mp4 --data data --exam-id 1  # only that exam's form + subject
    python scripts/replay_video.py old.mp4 --layout out/q45.layout.json --fps 8 --annotate annotated.mp4
    python scripts/replay_video.py photos_dir/ --data data --dry-run

Frames are sampled at --fps (default 8, what the live client achieves) and fed with video timestamps.
Every commit is printed; the commits are ingested into the DB at --data unless --dry-run (always dry
without --data).  --layout (repeatable) is only for LEGACY sheets printed with the 16-bit form id strip; with
--data they are registered so their commits are ingested too.  A summary (frames, status histogram, commits,
ms/frame) follows.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr.layout import Layout  # noqa: E402
from omr.recognize import recognize  # noqa: E402
from omr.stream import DEFAULT_K, DEFAULT_M, StreamSession  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
MISSED_AFTER = 1.5   # seconds a sheet may be in view without a commit before it is reported
CODE_COLOR = [(150, 150, 150), (80, 200, 60), (80, 80, 255), (40, 160, 245)]   # BGR: empty, marked, multi, ambiguous


def iter_frames(source, fps, max_side):
    """Yield (t_seconds, bgr) sampled at `fps`."""
    def fit(img):
        if max_side and max(img.shape[:2]) > max_side:
            s = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return img
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source) if f.lower().endswith(IMG_EXT))
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(source, f), cv2.IMREAD_COLOR)
            if img is not None:
                yield i / fps, fit(img)
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, next_t = 0, 0.0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / src_fps
        idx += 1
        if t + 1e-6 >= next_t:
            ok, img = cap.retrieve()
            if not ok:
                break
            next_t += 1.0 / fps
            yield t, fit(img)
    cap.release()


def open_writer(path, fps, size):
    """VideoWriter with the first codec this OpenCV build supports for the container."""
    codecs = ("avc1", "mp4v", "H264") if path.lower().endswith(".mp4") else ("MJPG", "XVID")
    for cc in codecs:
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
    raise SystemExit(f"no usable video codec for {path} (try a .avi name)")


def draw(img, ev, t, status):
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, int(round(max(h, w) / 700)))
    ov = ev["overlay"]
    if ov.get("sheet"):
        cv2.polylines(out, [np.array(ov["sheet"], np.int32)], True, (220, 140, 0), th)
    for q in ov.get("markers", []):
        cv2.polylines(out, [np.array(q, np.int32)], True, (60, 200, 60), th + 1)
    r = int(max(2, ov.get("bubble_r", 4) * 0.8))
    for x, y, c in ov.get("bubbles", []):
        if c:
            cv2.circle(out, (x, y), r, CODE_COLOR[c], -1)
        else:
            cv2.circle(out, (x, y), max(1, r // 4), CODE_COLOR[0], -1)
    fs = max(h, w) / 1400
    cv2.rectangle(out, (0, 0), (w, int(fs * 70)), (0, 0, 0), -1)
    cv2.putText(out, f"{ev['state'].upper()}  {status}  t={t:.2f}s", (10, int(fs * 28)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    cv2.putText(out, ev["hint"], (10, int(fs * 58)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (120, 220, 255), max(1, th - 1))
    if ev["state"] == "committed":
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (60, 200, 60), th * 6)
    return out


def page_layout(r, resolve):
    """Layout of a committed result: derived from its form_key (self-describing) or resolved by form id (legacy)."""
    if r.form_id is not None:
        return resolve(r.form_id)
    from omr.layout import header_from_form_key
    from omr.recognize import layout_for_header
    return layout_for_header(header_from_form_key(r.form_key))


def render_commit(r, resolve):
    """The committed sheet as the recognizer saw it (canonical canvas) with every bubble's verdict drawn on top."""
    img = cv2.cvtColor(r.warped, cv2.COLOR_GRAY2BGR)
    ppm = r.px_per_mm
    dia = page_layout(r, resolve).pages[r.page].bubble_diameter
    rad = max(2, int(round(0.5 * dia * ppm * 0.8)))
    for a, row in zip(r.answers, r.bubble_centers_mm.get("answers", [])):
        for k, (x, y) in enumerate(row):
            c = (int(round(x * ppm)), int(round(y * ppm)))
            if k in a.marked:
                col = CODE_COLOR[2] if a.state == "multi" else CODE_COLOR[3] if a.ambiguous else CODE_COLOR[1]
                cv2.circle(img, c, rad, col, 3)
            elif a.ambiguous and a.scores and a.scores[k] > 0.1:
                cv2.circle(img, c, rad, CODE_COLOR[3], 2)
    for digits, rows in ((r.id_digits, r.bubble_centers_mm.get("id", [])), (r.subject_digits, r.bubble_centers_mm.get("subject", []))):
      for d, row in zip(digits, rows):
        for k, (x, y) in enumerate(row):
            c = (int(round(x * ppm)), int(round(y * ppm)))
            if d.value == k:
                cv2.circle(img, c, rad, CODE_COLOR[1], 3)
            elif d.ambiguous and d.scores and d.scores[k] > 0.1:
                cv2.circle(img, c, rad, CODE_COLOR[3], 2)
    for q in r.marker_corners.values():     # where the markers were found, in canvas coordinates
        pts = cv2.perspectiveTransform(np.array(q, np.float64).reshape(-1, 1, 2), np.asarray(r.homography)).reshape(-1, 2)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (60, 200, 60), 3)
    return img


def commit_text(i, t, r):
    marked = {a.number: ("".join(a.labels[m] for m in a.marked) or "-") for a in r.answers if a.state != "blank" or a.ambiguous}
    q15 = " ".join(f"Q{a.number}={('/'.join(a.labels[m] for m in a.marked) or '-')}{'?' if a.ambiguous else ''}"
                   f"{'(multi)' if a.state == 'multi' else ''}" for a in r.answers[:5])
    later = [a.number for a in r.answers[5:] if a.state != "blank" or a.ambiguous]
    return (f"commit_{i:02d}  t={t:.2f}s  subject={r.subject}  student={r.student_id}  id_digits={[d.value for d in r.id_digits]}  serial={r.serial}\n"
            f"  {q15}\n  Q6+ non-blank/doubtful: {later or 'none'}   flags={r.flags or '-'}  markers={r.marker_ids} missing={r.missing_corners}\n"
            f"  recognize {r.timing_ms.get('total', 0):.0f} ms  quality: "
            + ", ".join(f"{k}={v:.2f}" for k, v in (r.quality or {}).items() if k in ("reproj_max_mm", "source_px_per_mm", "meta_contrast", "align_resid_mm", "ink_abs", "detect_passes", "detect_kernel", "markers_used")) + "\n")


def layout_resolver(args):
    """(resolve_layout for legacy sheets, service or None, allowed_forms or None, allowed_subjects or None)."""
    legacy = {}
    for p in args.layout or []:
        with open(p, encoding="utf-8") as f:
            L = Layout.from_json(f.read())
        if L.form_id is None:
            print(f"note: {p} describes itself; --layout is only needed for legacy sheets", file=sys.stderr)
            continue
        legacy[L.form_id] = L
    svc, forms, subjects = None, None, None
    if args.exam_id is not None and not args.data:
        raise SystemExit("--exam-id needs --data")
    if args.data:
        from omr.service import OmrService
        svc = OmrService(args.data)
        for L in legacy.values():
            svc.register_legacy_layout(L)
        if args.exam_id is not None:
            try:
                e = svc.get_exam(args.exam_id)
            except KeyError:
                raise SystemExit(f"no exam {args.exam_id} in {args.data}")
            forms = {e["form_key"]}
            subjects = {e["subject"]} if e["subject"] else None
    resolve = svc.resolve_layout if svc is not None else legacy.get
    return resolve, (None if args.dry_run else svc), forms, subjects


def answer_summary(r):
    c = Counter(a.state for a in r.answers)
    dbt = sum(1 for a in r.answers if a.ambiguous)
    return f"{len(r.answers)}q: {c['single']} single, {c['blank']} blank, {c['multi']} multi, {dbt} doubtful"


def replay(source, args, out=print):
    resolve, svc, forms, subjects = layout_resolver(args)
    session = StreamSession(k=args.k, m=args.m, allowed_forms=forms, allowed_subjects=subjects, resolve_layout=resolve,
                            fuse=not args.no_fuse)
    hist, commits, ms_rec, ms_all = Counter(), [], [], []
    writer = None
    n = 0
    txt = []
    def handle_commit(c, t):
        r = c["result"]
        outcome = None
        if svc is not None:
            outcome = svc.ingest(r, images={"original": c["payload"]}, source="stream").to_dict()
        rec = {"t": round(t, 2), "student_id": r.student_id, "serial": r.serial, "page": r.page, "form_key": r.form_key,
               "subject": r.subject,
               "answers": answer_summary(r), "states": [(a.number, a.state, list(a.marked), a.ambiguous) for a in r.answers],
               "flags": list(r.flags), "needs_review": r.needs_review, "fused_frames": r.fused_frames,
               "outcome": outcome["kind"] if outcome else "dry-run"}
        commits.append(rec)
        if args.save_commits:
            os.makedirs(args.save_commits, exist_ok=True)
            cv2.imwrite(os.path.join(args.save_commits, f"commit_{len(commits):02d}.png"), render_commit(r, resolve))
            txt.append(commit_text(len(commits), t, r))
        out(f"[{t:7.2f}s] COMMIT subject={r.subject if r.subject is not None else '?'} student={r.student_id} serial={r.serial or '-'} page={r.page + 1} {rec['answers']} fused={r.fused_frames} flags={rec['flags'] or '-'} -> {rec['outcome']}")

    last_t = 0.0
    for t, img in iter_frames(source, args.fps, args.max_side):
        last_t = t
        t0 = time.perf_counter()
        res = recognize(img, resolve)
        ev = session.feed(res, t, payload=img)
        n += 1
        hist[res.status] += 1
        ms_rec.append(res.timing_ms.get("total", 0.0))
        if "commit" in ev:
            handle_commit(ev["commit"], t)
        ms_all.append((time.perf_counter() - t0) * 1000)
        if args.annotate:
            frame = draw(img, ev, t, res.status)
            if writer is None:
                h, w = frame.shape[:2]
                writer = open_writer(args.annotate, args.fps, (w, h))
            writer.write(frame)
    if writer is not None:
        writer.release()
    c = session.finish(last_t)
    if c is not None:
        handle_commit(c, last_t)
    missed = [{"from": round(e["t0"], 2), "to": round(e["t1"], 2), "frames": e["frames"], "ok": e["ok"],
               "statuses": e["statuses"]} for e in session.unread_episodes(final=True)
              if e["t1"] - e["t0"] > MISSED_AFTER]
    missed_sheets = session.missed(final=True)
    summary = {"missed": missed, "missed_sheets": missed_sheets,
               "missed_serials": sorted({m["serial"] for m in missed_sheets if m["serial"]}), "frames": n, "status": dict(hist), "commits": commits,
               "mean_recognize_ms": round(float(np.mean(ms_rec)), 1) if ms_rec else 0.0,
               "mean_frame_ms": round(float(np.mean(ms_all)), 1) if ms_all else 0.0}
    out(f"\nframes: {n}   commits: {len(commits)}   mean recognize: {summary['mean_recognize_ms']} ms   "
        f"mean per frame (incl. session/ingest/annotate): {summary['mean_frame_ms']} ms")
    out("status histogram: " + ", ".join(f"{k}={v}" for k, v in hist.most_common()))
    if missed:
        out(f"sheet in view (any marker visible, incl. partial) for >{MISSED_AFTER}s with NO commit ({len(missed)}):")
        for m in missed:
            out(f"  {m['from']:.2f}s - {m['to']:.2f}s   {m['frames']} frames, {m['ok']} ok   statuses: {m['statuses']}")
    else:
        out("no sheet-present stretch without a commit")
    if missed_sheets:
        out(f"MISSED SHEETS - seen in >= 2 ok frames but never committed ({len(missed_sheets)}):")
        for m in missed_sheets:
            who = f"No. {m['serial']:04d}" if m["serial"] else f"student {m['student_id'] or '?'} (no serial)"
            out(f"  {who}  page {m['page'] + 1}  ({m['frames']} frames)")
    else:
        out("no sheet was seen without being committed")
    if args.save_commits and txt:
        with open(os.path.join(args.save_commits, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt))
    if args.annotate:
        out(f"annotated video: {args.annotate}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file or folder of photos")
    ap.add_argument("--data", help="data directory (commits are stored there unless --dry-run)")
    ap.add_argument("--exam-id", type=int, help="only accept this exam's sheets (its form and subject)")
    ap.add_argument("--layout", action="append", help="legacy layout JSON (repeatable); only for sheets printed with the old form id strip")
    ap.add_argument("--fps", type=float, default=8.0, help="sampling rate simulating the live client")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--m", type=int, default=DEFAULT_M)
    ap.add_argument("--max-side", type=int, default=1920, help="downscale frames larger than this (0 = never)")
    ap.add_argument("--dry-run", action="store_true", help="do not touch the database")
    ap.add_argument("--no-fuse", action="store_true", help="disable multi-frame fusion (for before/after comparisons)")
    ap.add_argument("--save-commits", help="directory: per commit the warped sheet with the verdict overlay (commit_NN.png) + summary.txt")
    ap.add_argument("--annotate", help="write an annotated mp4 here")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON at the end")
    args = ap.parse_args(argv)
    s = replay(args.source, args)
    if args.json:
        print(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    main()
