import sys
from common import *  # noqa
p = run_jsc(["--", FIX] + sys.argv[1:], files=src_files() + [HERE + "/ser.js", HERE + "/prof.js"])
print(p.stdout, p.stderr[:2000])
