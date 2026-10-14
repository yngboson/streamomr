"""Shared helpers of the J3 pipeline tests: paths, serialisation of Python RecognitionResults to the TYPES.md JSON shape, jsc runner."""
import dataclasses, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIX = os.path.join(ROOT, "web", "test", "fixtures", "pipe")
REAL = os.environ.get("STREAMOMR_REALASSET", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "realasset"))
JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
ORDER = ["util", "markers", "layout", "types", "identity", "decide", "fusion", "revision", "stream",
         "cv", "aruco", "cvx", "fastops", "ink", "illum", "tracks", "detect", "recognize"]
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def src_files():
    return [os.path.join(ROOT, "web", "src", n + ".js") for n in ORDER]


def clean(v):
    import numpy as np
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(clean(x) for x in v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: clean(getattr(v, f.name)) for f in dataclasses.fields(v)}
    return v


RAW_FIELDS = ["numbers", "labels", "digit_index", "FA", "SA", "FI", "SI", "ink", "min_strength", "verified", "FS", "SS", "ink_source",
              "ink_separation", "MA", "MI", "MS"]


def ser(res):
    """RecognitionResult -> plain dict (no images)."""
    d = {k: clean(getattr(res, k)) for k in ["status", "form_key", "form_id", "page", "total_pages", "serial", "student_id", "subject", "flags",
                                              "marker_ids", "missing_corners", "homography", "px_per_mm", "quality", "calibration",
                                              "bubble_centers_mm", "timing_ms", "fused_frames"]}
    d["answers"] = [clean(a) for a in res.answers]
    d["id_digits"] = [clean(a) for a in res.id_digits]
    d["subject_digits"] = [clean(a) for a in res.subject_digits]
    d["marker_corners"] = [[int(i), clean(c)] for i, c in res.marker_corners.items()]
    d["raw"] = None if res.raw is None else {k: clean(getattr(res.raw, k)) for k in RAW_FIELDS}
    d["dewarp"] = res.dewarp is not None
    return d


def run_jsc(args, files=None, cwd=ROOT):
    return subprocess.run([JSC] + (files if files is not None else src_files()) + args, capture_output=True, text=True, cwd=cwd)
