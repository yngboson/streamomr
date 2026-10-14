"""Export ArUco fixtures: every grey image that omr.detect hands to cv2's ArucoDetector (plain, opened, flattened ...),
with cv2's corners / ids / rejected.  Sources: realasset photos, devlog frames, synthetic frames (tests/synth.py).
Usage: .venv/bin/python web/test/cv/export_aruco.py [--synth N]
Output: web/test/fixtures/aruco/{manifest.json, NNNN.bin}   (never written anywhere else)"""
import glob, json, os, sys, time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
from omr import detect  # noqa: E402

OUT = os.path.join(ROOT, "web", "test", "fixtures", "aruco")
REAL = os.environ.get("STREAMOMR_REALASSET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "realasset"))
cases = []


class Rec:
    def __init__(self, real, tag):
        self.real, self.tag = real, tag

    def detectMarkers(self, img):
        t = time.perf_counter()
        c, i, r = self.real.detectMarkers(img)
        ms = (time.perf_counter() - t) * 1000
        n = len(cases)
        np.ascontiguousarray(img).tofile(os.path.join(OUT, f"{n:04d}.bin"))
        cases.append({"file": f"{n:04d}.bin", "w": int(img.shape[1]), "h": int(img.shape[0]), "src": self.tag, "cv2_ms": ms,
                      "ids": [] if i is None else [int(x) for x in i.ravel()],
                      "corners": [np.asarray(x, np.float64).reshape(8).tolist() for x in c],
                      "rejected": [np.asarray(x, np.float64).reshape(8).tolist() for x in r]})
        return c, i, r


def run_gray(gray, tag):
    real = detect._detector()
    detect._tls.det = Rec(real, tag)
    try:
        detect.find_markers_scaled(gray)
    finally:
        detect._tls.det = real


def main():
    nsynth = int(sys.argv[sys.argv.index("--synth") + 1]) if "--synth" in sys.argv else 60
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*")):
        os.remove(f)
    files = sorted(glob.glob(f"{REAL}/*.jpeg")) + [f"{REAL}/mixed_ink_pencil_id.jpg"]
    files += sorted(glob.glob(f"{REAL}/devlog_2026-10-13/scans/*/frame.jpg"))
    files += sorted(glob.glob(f"{REAL}/devlog_2026-10-13/failures/*.jpg"))[:40]
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tag = "real:" + os.path.basename(os.path.dirname(f) if "scans" in f else f)[:24]
        run_gray(g, tag)
        if os.path.basename(f).startswith(("6B86", "mixed")):   # also the 0.5 / 0.33 scales the photo tests use
            for s in (0.5, 0.33):
                run_gray(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), tag + f"@{s}")
        print(f, len(cases), flush=True)
    vstep = int(sys.argv[sys.argv.index("--video") + 1]) if "--video" in sys.argv else 10
    vid = f"{REAL}/devlog_2026-10-14/video/segment_01.mp4"
    if vstep and os.path.exists(vid):
        cap = cv2.VideoCapture(vid)
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if n % vstep == 0:
                run_gray(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), f"video:{n}")
            n += 1
        cap.release()
        print("video frames", n, "cases", len(cases), flush=True)
    if nsynth:
        import tempfile, synth
        assets = synth.prepare(tempfile.mkdtemp(prefix="j1synth"), {k: v for k, v in synth.STANDARD_TEMPLATES.items() if k in ("q45", "q100_2p")})
        names = list(assets)
        conds = ["clean", "tilted", "blurred", "dark", "phone", "lowres", "printer", "handheld", "hand_only", "curl", "glare", "shadow", "raking"]
        for n in range(nsynth):
            cn = conds[n % len(conds)]
            L, pdf = assets[names[n % len(names)]]
            rng = np.random.default_rng(1000 + n)
            mode = "thumb" if n % 7 == 3 else ("none" if n % 11 == 5 else "sheet")
            frame, _ = synth.make_frame(L, pdf, synth.CONDS[cn], rng, mode=mode, serial=1 + n)
            g = cv2.cvtColor(frame.astype(np.uint8) if frame.ndim == 3 else frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            run_gray(g, f"synth:{cn}:{n}")
            if n % 10 == 0:
                print("synth", n, len(cases), flush=True)
    json.dump(cases, open(os.path.join(OUT, "manifest.json"), "w"))
    print("cases", len(cases), "with ids", sum(1 for c in cases if c["ids"]), "rejected total", sum(len(c["rejected"]) for c in cases))


main()
