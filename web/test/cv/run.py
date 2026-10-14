#!/usr/bin/env python
"""One command for the cv.js / aruco.js tests:   .venv/bin/python web/test/cv/run.py [--regen] [--quick] [--bench]

  --regen  re-export the fixtures (needs the realasset folder + cv2; ~10 min for the aruco set)
  --quick  aruco parity on every 4th case only
  --bench  also print the timing table (JS vs cv2 on a 1080x1920 frame)
Fixtures live in web/test/fixtures/ (git-ignored).  JS runs under JavaScriptCore (no node needed)."""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FX = os.path.join(ROOT, "web", "test", "fixtures")
JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
PY = sys.executable
ok_all = True


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def jsc(script, *args):
    r = sh([JSC, os.path.join("web/test/cv", script), "--", FX, *args])
    return r.stdout + r.stderr


def report(name, good, detail=""):
    global ok_all
    ok_all &= good
    print(("PASS " if good else "FAIL ") + name + (("  " + detail) if detail else ""))


def main():
    regen = "--regen" in sys.argv
    if regen or not os.path.exists(os.path.join(FX, "aruco", "manifest.json")):
        print("exporting aruco fixtures ..."); r = sh([PY, "web/test/cv/export_aruco.py", "--synth", "60", "--video", "10"]); print(r.stdout[-300:], r.stderr[-300:])
    if regen or not os.path.exists(os.path.join(FX, "prim", "manifest.json")):
        print("exporting primitive fixtures ..."); r = sh([PY, "web/test/cv/export_prim.py"]); print(r.stdout[-200:], r.stderr[-300:])
    if regen or "--bench" in sys.argv:
        sh([PY, "web/test/cv/bench_cv2.py"])
    out = jsc("test_contours.js")
    m = re.search(r"^(PASS|FAIL) (\{.*\})", out, re.M)
    report("contours: adaptiveThreshold / findContours / approxPolyDP / arcLength / area / convex", bool(m) and m.group(1) == "PASS", m.group(2)[:200] if m else out[-300:])
    out = jsc("test_prim.js")
    for line in out.splitlines():
        if line.startswith(("PASS", "FAIL", "ERR")):
            report("prim " + line[5:].split(" cases")[0], line.startswith("PASS"), line[5:].split(" cases", 1)[-1][:150] if line.startswith("FAIL") else "")
    out = jsc("test_aruco.js", "100000" if "--quick" not in sys.argv else "100000", "", "") if "--quick" not in sys.argv else ""
    if "--quick" in sys.argv:
        import json
        n = len(json.load(open(os.path.join(FX, "aruco", "manifest.json"))))
        out = jsc("test_aruco.js", "100000", "", ",".join(str(i) for i in range(0, n, 4)))
    m = re.search(r"cases (\d+) idsOk (\d+) idsBad (\d+) rejOk (\d+) rejBad (\d+)", out)
    c = re.search(r"median ([\d.]+) p99 ([\d.]+) max ([\d.]+)", out)
    if m:
        n, ok = int(m.group(1)), int(m.group(2))
        report("aruco.detect vs cv2 ArucoDetector", ok / n >= 0.99 and float(c.group(2)) <= 0.01, f"id sets equal {ok}/{n} ({100*ok/n:.2f}%), rejected sets equal {m.group(4)}/{n}, corner err median {c.group(1)} p99 {c.group(2)} max {c.group(3)} px")
        print("\n".join(l for l in out.splitlines() if l.startswith(("ms/call", "  real", "  video", "  synth"))))
    else:
        report("aruco.detect", False, out[-400:])
    if "--bench" in sys.argv:
        print(jsc("bench.js"))
    print("ALL PASSED" if ok_all else "FAILURES")
    sys.exit(0 if ok_all else 1)


main()
