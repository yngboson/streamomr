"""per-section minimum timings of one video frame:  .venv/bin/python web/test/pipe/stages.py <idx> [reps]  (idx 16 = ok, 0 = partial, 1 = no_sheet)"""
import sys
from common import *  # noqa
p = run_jsc(["--", FIX, "video"] + sys.argv[1:2] + sys.argv[2:3], files=src_files() + [HERE + "/ser.js", HERE + "/stages.js"])
print(p.stdout.strip(), p.stderr[:1500])
