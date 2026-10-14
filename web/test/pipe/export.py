"""Record the inputs (grey frames, raw .bin) and Python's outputs for the J3 pipeline tests.

    .venv/bin/python web/test/pipe/export.py [--sets photos,devlog,video,synth] [--regen] [--jobs 8]

Writes web/test/fixtures/pipe/<set>/NNNN.bin, <set>.manifest.json ([{name, file, w, h, t?}]) and <set>.py.json ({name: result}).
Never copies realasset files: frames are decoded to grey bytes in the (gitignored) fixtures dir.
"""
import argparse, glob, json, multiprocessing as mp, os, pickle, sys, time
import cv2
import numpy as np
from common import *  # noqa: F401,F403

from omr import recognize as R  # noqa: E402

SYNTH_CONDS = ["clean", "tilted", "blurred", "dark", "phone", "lowres", "sp_blank", "sp_ghosts", "sp_fewghost", "sp_pencil", "sp_dark",
               "printer", "handheld", "hand_only", "curl", "curl_hand", "glare", "shadow", "raking", "faint", "faint_glare", "faint_prt",
               "fold", "fold_curl", "fold_hand", "fold_big", "mix_pencil_id", "mix_pen_id", "mix_half_pen", "mix_rand_pen",
               "h_multi", "h_dimgrad", "h_glare", "h_pencil", "h_all"]
SYNTH_TEMPLATES = ["q45", "q100_2p", "q100u80_2p", "q30_abcd", "q40_nosubj", "q100_3p"]


def work(args):
    name, d = args
    gray = np.frombuffer(open(os.path.join(FIX, d["file"]), "rb").read(), np.uint8).reshape(d["h"], d["w"])
    t0 = time.perf_counter()
    r = R.recognize(gray, None)
    ms = (time.perf_counter() - t0) * 1000
    out = ser(r)
    out["py_ms"] = ms
    r.warped = r.name_crop = None            # the pickle feeds the python stream replay: no images
    return name, (out, r)


def save_set(setname, frames, workers):
    os.makedirs(os.path.join(FIX, setname), exist_ok=True)
    man = []
    for i, (name, gray, extra) in enumerate(frames):
        f = f"{setname}/{i:04d}.bin"
        with open(os.path.join(FIX, f), "wb") as fh:
            fh.write(np.ascontiguousarray(gray).tobytes())
        man.append(dict(name=name, file=f, w=int(gray.shape[1]), h=int(gray.shape[0]), **extra))
    json.dump(man, open(os.path.join(FIX, setname + ".manifest.json"), "w"))
    t0 = time.time()
    with mp.Pool(workers) as p:
        res = dict(p.map(work, [(m["name"], m) for m in man], chunksize=1))
    json.dump({k: v[0] for k, v in res.items()}, open(os.path.join(FIX, setname + ".py.json"), "w"))
    pickle.dump({k: v[1] for k, v in res.items()}, open(os.path.join(FIX, setname + ".py.pkl"), "wb"))
    print(f"{setname}: {len(man)} frames, python {time.time() - t0:.0f}s wall", flush=True)


def gray_of(bgr):
    return bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def set_photos():
    out = []
    for f in sorted(glob.glob(f"{REAL}/*.jpeg")) + [f"{REAL}/mixed_ink_pencil_id.jpg"]:
        out.append((os.path.basename(f), gray_of(cv2.imread(f)), {}))
    return out


def set_devlog():
    out = []
    for sd in sorted(glob.glob(f"{REAL}/devlog_2026-10-13/scans/*")):
        out.append((os.path.basename(sd).split("_", 1)[1], gray_of(cv2.imread(os.path.join(sd, "frame.jpg"))), {}))
    return out


def set_video(fps=8):
    import replay_video
    out = []
    for t, img in replay_video.iter_frames(f"{REAL}/devlog_2026-10-14/video/segment_01.mp4", fps, 0):
        out.append((f"t{t:07.3f}", gray_of(img), {"t": t}))
    return out


def set_synth(n):
    import synth
    try:
        import illum_bench  # noqa: F401  registers the h_* conditions
    except Exception as e:
        print("illum_bench unavailable:", e)
    workdir = os.path.join(FIX, "synth_assets")
    assets = synth.prepare(workdir, {k: synth.STANDARD_TEMPLATES[k] for k in SYNTH_TEMPLATES})
    names = list(assets)
    out, truths = [], {}
    jobs = []
    for ci, cn in enumerate(SYNTH_CONDS):
        if cn not in synth.CONDS:
            continue
        for k in range(n):
            jobs.append((names[(ci + k) % len(names)], cn, "sheet", 7000 + ci * 101 + k))
    for k, cn in enumerate(["clean", "phone", "tilted", "dark", "handheld"]):
        for mode in ("none", "thumb2", "cut"):
            jobs.append((names[k % len(names)], cn, mode, 9000 + k * 7 + len(mode)))
    for name, cn, mode, seed in jobs:
        L, pdf = assets[name]
        rng = np.random.default_rng(seed)
        serial = 1 + seed % synth.mk_serial_max() if mode != "none" else None
        mk = (synth.make_frame_faint if cn in synth.FAINT_CONDS else synth.make_frame_mixed if cn in synth.MIXED_CONDS else synth.make_frame)
        try:
            frame, truth = mk(L, pdf, synth.CONDS[cn], rng, mode=mode, serial=serial)
        except Exception as e:
            print("skip", cn, mode, e)
            continue
        key = f"{cn}.{mode}.{name}.{seed}"
        truths[key] = truth
        out.append((key, gray_of(frame), {"cond": cn, "mode": mode, "template": name, "seed": seed}))
    pickle.dump(truths, open(os.path.join(FIX, "synth.truth.pkl"), "wb"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="photos,devlog,video,synth")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--synth-n", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(FIX, exist_ok=True)
    for s in a.sets.split(","):
        if not a.regen and os.path.exists(os.path.join(FIX, s + ".py.json")):
            print(s, "exists (use --regen)")
            continue
        fr = {"photos": set_photos, "devlog": set_devlog, "video": set_video, "synth": lambda: set_synth(a.synth_n)}[s]()
        save_set(s, fr, a.jobs)


if __name__ == "__main__":
    main()
