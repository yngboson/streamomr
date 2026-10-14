"""cv2 timings (ms/call, median, single call at a time) on a 1080x1920 video frame, for the table web/test/cv/bench.js prints next to the JS numbers."""
import json, os, sys, time
import cv2
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FX = os.path.join(ROOT, "web", "test", "fixtures")
sys.path.insert(0, ROOT)
man = json.load(open(os.path.join(FX, "aruco", "manifest.json")))
c = [c for c in man if c["src"].startswith("video:") and c["w"] == 1080 and c["h"] == 1920 and len(c["ids"]) >= 3][0]
g = np.fromfile(os.path.join(FX, "aruco", c["file"]), np.uint8).reshape(c["h"], c["w"])
gf = g.astype(np.float32)


def T(fn, reps=15):
    fn(); fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 3)


from omr import detect
det = detect._detector()
src = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]]); dst = np.float32([[10, 20], [320, 15], [330, 230], [5, 215]])
H = cv2.getPerspectiveTransform(src, dst)
bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 7)
mx = np.random.default_rng(0).random((1000, 1400)).astype(np.float32) * 1000
my = np.random.default_rng(1).random((1000, 1400)).astype(np.float32) * 1800
out = {
    "frame": f"{c['w']}x{c['h']} ({c['src']})",
    "adaptiveThreshold win3": T(lambda: cv2.adaptiveThreshold(g, 255, 0, 1, 3, 7)),
    "adaptiveThreshold win13": T(lambda: cv2.adaptiveThreshold(g, 255, 0, 1, 13, 7)),
    "adaptiveThreshold win23": T(lambda: cv2.adaptiveThreshold(g, 255, 0, 1, 23, 7)),
    "findContours LIST/NONE (win13)": T(lambda: cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)),
    "findContours EXTERNAL/SIMPLE (win13)": T(lambda: cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)),
    "aruco detectMarkers (plain)": T(lambda: det.detectMarkers(g)),
    "aruco detectMarkers (opened k=9)": T(lambda: det.detectMarkers(cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)))),
    "resize area x0.25": T(lambda: cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)),
    "resize area x0.125 (u8)": T(lambda: cv2.resize(g, (g.shape[1] // 8, g.shape[0] // 8), interpolation=cv2.INTER_AREA)),
    "resize area x0.4 fractional": T(lambda: cv2.resize(g, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)),
    "resize linear small->full (u8)": T(lambda: cv2.resize(g[:135, :240], (1920, 1080), interpolation=cv2.INTER_LINEAR)),
    "GaussianBlur u8 sigma 1.2": T(lambda: cv2.GaussianBlur(g, (0, 0), 1.2)),
    "GaussianBlur u8 sigma 3": T(lambda: cv2.GaussianBlur(g, (0, 0), 3)),
    "GaussianBlur f32 sigma 1.2": T(lambda: cv2.GaussianBlur(gf, (0, 0), 1.2)),
    "dilate k=9": T(lambda: cv2.dilate(g, np.ones((9, 9), np.uint8))),
    "erode k=23": T(lambda: cv2.erode(g, np.ones((23, 23), np.uint8))),
    "morphology OPEN k=9": T(lambda: cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))),
    "boxFilter f32 21x15": T(lambda: cv2.boxFilter(gf, -1, (21, 15), normalize=True, borderType=cv2.BORDER_REPLICATE)),
    "Laplacian": T(lambda: cv2.Laplacian(g, cv2.CV_32F)),
    "integral f64": T(lambda: cv2.integral(g, sdepth=cv2.CV_64F)),
    "threshold otsu": T(lambda: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)),
    "warpPerspective linear 1400x1000": T(lambda: cv2.warpPerspective(g, H, (1400, 1000), flags=cv2.INTER_LINEAR, borderValue=255)),
    "remap linear 1400x1000": T(lambda: cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)),
    "cornerSubPix 4 corners": T(lambda: cv2.cornerSubPix(g, np.float32(c["corners"][0]).reshape(4, 1, 2), (4, 4), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)), 50),
    "bitwise_not": T(lambda: cv2.bitwise_not(g)),
}
json.dump(out, open(os.path.join(FX, "bench_cv2.json"), "w"), indent=1)
print("bench_cv2.json written")
