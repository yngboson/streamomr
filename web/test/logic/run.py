"""One command for the J2 logic port:  .venv/bin/python web/test/logic/run.py [--regen] [--no-video] [--no-export]

1. (re)exports the Python fixtures into web/test/fixtures/logic/ (skipped when they exist unless --regen / missing),
2. runs the JS modules under JavaScriptCore on them and prints per-area pass / fail.  Exit code 0 only if everything matched.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIX = os.path.join(ROOT, "web", "test", "fixtures", "logic")
JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
ORDER = ["util", "markers", "layout", "types", "identity", "decide", "fusion", "revision", "stream"]


def main():
    args = sys.argv[1:]
    need = "--regen" in args or not os.path.exists(os.path.join(FIX, "stream.json"))
    if need and "--no-export" not in args:
        cmd = [sys.executable, os.path.join(HERE, "export_fixtures.py")] + [a for a in args if a in ("--regen", "--no-video")]
        if subprocess.call(cmd, cwd=ROOT) != 0:
            return 2
    files = [os.path.join(ROOT, "web", "src", n + ".js") for n in ORDER] + [os.path.join(HERE, "test.js")]
    p = subprocess.run([JSC] + files + ["--", FIX], capture_output=True, text=True, cwd=ROOT)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    return 0 if "ALL PASSED" in p.stdout and p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
