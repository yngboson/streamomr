"""Browser glue for the standalone scanner (runs inside Pyodide, next to the unmodified ``omr`` package).

``Scanner.step`` mirrors ``_frame_step`` in app/main.py for the "all subjects (auto)" path of /ws/scan:
recognize(frame) -> StreamSession.feed(...) -> event.  What differs: no database (a commit is only
summarised, the caller keeps the results in memory), no JPEG round trip (the browser hands over raw RGBA),
and no legacy-form lookup (resolve_layout=None: sheets describe themselves in their header).
Keep it in sync with app/main.py::_frame_step when that changes.
"""
import json
import time

import cv2
import numpy as np

from omr.recognize import recognize
from omr.stream import StreamSession

MIN_FRAME_SIDE = 64      # same as app/main.py


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not (isinstance(k, str) and k.endswith("_path"))}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def summarize(result) -> dict:
    """Same fields as app/main.py::summarize, plus the marked labels per question (for display / CSV)."""
    return {"form_key": result.form_key, "page": result.page, "total_pages": result.total_pages,
            "student_id": result.student_id, "subject": result.subject, "serial": result.serial,
            "n_questions": len(result.answers),
            "n_blank": sum(1 for a in result.answers if a.state == "blank"),
            "n_multi": sum(1 for a in result.answers if a.state == "multi"),
            "n_doubtful": sum(1 for a in result.answers if a.ambiguous),
            "flags": list(result.flags), "needs_review": result.needs_review,
            "sharpness": (result.quality or {}).get("sharpness"),
            "answers": [{"q": a.number, "state": a.state, "marked": "|".join(a.labels[i] for i in a.marked),
                         "doubt": bool(a.ambiguous)} for a in result.answers]}


class Scanner:
    def __init__(self):
        self.session = StreamSession(resolve_layout=None)

    def reset(self):
        self.session.reset()
        self.session.reset_tracking()

    def step_rgba(self, buf, w: int, h: int) -> str:
        """buf: RGBA bytes (memoryview / JsProxy Uint8Array) of a w x h frame.  Returns the event as JSON."""
        t0 = time.perf_counter()
        if hasattr(buf, "to_py"):
            buf = buf.to_py()
        rgba = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return json.dumps(self.step(img, t0))

    def step(self, img, t0=None) -> dict:
        session = self.session
        t0 = time.perf_counter() if t0 is None else t0
        if min(img.shape[:2]) < MIN_FRAME_SIDE:
            w, h = int(img.shape[1]), int(img.shape[0])
            return {"state": session.state, "hint": f"Camera frame too small ({w}x{h})", "overlay": {},
                    "error": f"frame too small: {w}x{h}", "w": w, "h": h}
        res = recognize(img, None)
        ev = session.feed(res, time.monotonic(), payload=None)
        ev["status"] = res.status
        ev["w"], ev["h"] = int(img.shape[1]), int(img.shape[0])
        commit = ev.pop("commit", None)
        if commit is not None:
            ev["commit"] = {"summary": summarize(commit["result"]), "frames": commit["frames"]}
        missed = session.missed()
        if missed:
            ev["missed"] = missed
        ev["ms"] = {"total": round((time.perf_counter() - t0) * 1000, 1),
                    "recognize": round(res.timing_ms.get("total", 0), 1)}
        return _clean(ev)


_scanner = None


def init():
    global _scanner
    _scanner = Scanner()


def reset():
    _scanner.reset()


def frame(buf, w, h):
    return _scanner.step_rgba(buf, w, h)
