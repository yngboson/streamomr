#!/usr/bin/env python
"""Recognise EVERY frame of a recorded video once and cache the results, so stream policies (omr.stream K / holds / sampling)
can be simulated without running the vision again (scripts/latency_bench.py).

    python scripts/video_cache.py realassets/<devlog>/video/segment_01.mp4 --out out/devlog_cache.pkl --jobs 8

The cache is a pickle: {"fps", "n", "results": [RecognitionResult without images, one per source frame], "ms": [native recognize ms]}.
Re-run it after changing the recogniser (the cache holds its outputs)."""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr.recognize import recognize  # noqa: E402


def work(args):
    i, img = args
    t0 = time.perf_counter()
    r = recognize(img, keep_images=False)
    return i, r, (time.perf_counter() - t0) * 1000


def frames(path):
    """Sequential decode: seeking in an H.264 file is not frame-exact (cap.set(POS_FRAMES) lands on the wrong frames)."""
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            return
        yield i, img
        i += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--prior", action="store_true", help="sequential run (1 process) feeding each frame the previous frame's markers as a prior, "
                                                          "like the live client (omr.detect.prior_from)")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    t0 = time.time()
    results, ms = [None] * n, [0.0] * n
    if a.prior:
        from omr.detect import prior_from
        prior = None
        for i, img in frames(a.video):
            t1 = time.perf_counter()
            r = recognize(img, keep_images=False, prior=prior)
            ms[i] = (time.perf_counter() - t1) * 1000
            results[i] = r
            p = prior_from(r.marker_corners)
            prior = p if p is not None else prior
    else:
        with mp.Pool(a.jobs) as pool:
            for i, r, m in pool.imap_unordered(work, frames(a.video), chunksize=2):
                if i < n:
                    results[i], ms[i] = r, m
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"fps": fps, "n": n, "results": results, "ms": ms}, f)
    print(f"{n} frames at {fps:.2f} fps recognised in {time.time() - t0:.0f}s -> {a.out}")
