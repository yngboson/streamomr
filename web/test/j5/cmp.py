"""bit-identity check of web/src against a reference source dir:  .venv/bin/python web/test/j5/cmp.py <refsrc dir> [n]
runs a spread of video/photos/devlog/synth frames through both and compares json+canvas hashes; prints mean ms of each."""
import json, os, subprocess, sys, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIX = os.path.join(ROOT, "web", "test", "fixtures", "pipe"); PIPE = os.path.join(ROOT, "web", "test", "pipe")
JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
ORDER = ["util", "markers", "layout", "types", "identity", "decide", "fusion", "revision", "stream", "cv", "aruco", "cvx", "fastops", "ink", "illum", "tracks", "detect", "recognize"]
def run(srcdir, setname, idxs):
    files = [f"{srcdir}/{n}.js" for n in ORDER] + [PIPE + "/ser.js", HERE + "/dump.js"]
    p = subprocess.run([JSC] + files + ["--", FIX, setname, ",".join(map(str, idxs))], capture_output=True, text=True, cwd=ROOT)
    if p.stderr.strip(): print(p.stderr[:500])
    return [l.split() for l in p.stdout.splitlines() if l.strip()]
def main():
    ref = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    jobs = []
    for s, step in (("video", max(1, 424 // n)), ("photos", 1), ("devlog", 4), ("synth", max(1, 120 // 24))):
        man = json.load(open(f"{FIX}/{s}.manifest.json")); idx = list(range(0, len(man), step))
        for k in range(0, len(idx), 6): jobs.append((s, idx[k:k + 6]))
    new = os.path.join(ROOT, "web", "src")
    with cf.ThreadPoolExecutor(4) as ex:
        a = list(ex.map(lambda j: run(ref, *j), jobs)); b = list(ex.map(lambda j: run(new, *j), jobs))
    bad = 0; ta = tb = 0; cnt = 0
    for x, y, j in zip(a, b, jobs):
        for r, q in zip(x, y):
            cnt += 1; ta += float(r[2]); tb += float(q[2])
            if r[1] != q[1] or r[3] != q[3] or r[4] != q[4]:
                bad += 1; print("DIFF", j[0], r, q)
    print(f"{cnt} frames, {bad} differ; mean ms ref {ta/cnt:.0f} new {tb/cnt:.0f}")
main()
