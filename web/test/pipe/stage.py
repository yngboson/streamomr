import sys
from common import *  # noqa
key = sys.argv[1]
p = run_jsc(["--", FIX, key], files=src_files() + [HERE + "/ser.js", HERE + "/stage.js"])
print(p.stdout, p.stderr[:3000])
