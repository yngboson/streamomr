"""Quick look: .venv/bin/python web/test/pipe/one.py <set> <idx> [idx...]  -> run JS on those frames, print differences vs python."""
import json, sys
from common import *  # noqa
from compare import compare

setname = sys.argv[1]
py = json.load(open(f"{FIX}/{setname}.py.json"))
man = json.load(open(f"{FIX}/{setname}.manifest.json"))
for idx in map(int, sys.argv[2:]):
    p = run_jsc([FIX_ := FIX] if False else ["--", FIX, setname, f"{idx}-{idx + 1}"], files=src_files() + [HERE + "/ser.js", HERE + "/drv.js"])
    if p.stderr.strip():
        print(p.stderr[:2000])
    for line in p.stdout.splitlines():
        if line.startswith("@"):
            j = json.loads(line[1:])
            r = j["res"]; pr = py[j["name"]]
            n, diffs, st = compare(pr, r)
            print(f"{setname}[{idx}] {j['name']}: py {pr['status']} {pr['py_ms']:.0f}ms js {r['status']} {j['ms']:.0f}ms  fields {n} diffs {len(diffs)}")
            for d in diffs[:12]:
                print("   ", d)
            print("   ", {k: float(f"{v:.3g}") for k, v in st.items()})
            print("    js timing", {k: round(v) for k, v in r["timing_ms"].items()}, "py", {k: round(v) for k, v in pr["timing_ms"].items()})
        else:
            print(line[:500])
