"""Browser glue for the standalone scanner (runs inside Pyodide, next to the unmodified ``omr`` package).

``Scanner.step`` mirrors ``_frame_step`` in app/main.py for the "all subjects (auto)" path of /ws/scan:
recognize(frame) -> StreamSession.feed(...) -> event.  What differs: no database (a commit is only
summarised, the caller keeps the results in memory), no JPEG round trip (the browser hands over raw RGBA),
and no legacy-form lookup (resolve_layout=None: sheets describe themselves in their header).
Keep it in sync with app/main.py::_frame_step when that changes.
"""
import base64
import json
import time

import cv2
import numpy as np

from omr import recognize as R
from omr.recognize import recognize
from omr.stream import StreamSession
from omr.identity import severity

MIN_FRAME_SIDE = 64      # same as app/main.py
DEV = False  # __DEV__: scripts/build_standalone.py --dev sets this True (extra per-frame diagnostics for standalone/dev.js)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not (isinstance(k, str) and k.endswith("_path"))}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _plain(d) -> dict:
    """Numbers / strings / bools of a dict (quality, calibration) rounded for the dev log."""
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (d or {}).items() if isinstance(v, (int, float, str, bool))}


def _r(v, n=3):
    return None if v is None or (isinstance(v, float) and v != v) else round(float(v), n)


def _row_trace(f, st, md, cal, strict, labels, frame_f, trust):
    """Every intermediate value behind one question / ID digit / subject digit: per choice the fill (= the share of the
    bubble that is inked, the "marking %"), strength, disc darkness and its excess over the other choices, the in / out / grey
    class of the absolute rule, the spread of the fill over the fused frames; then the absolute rule's verdict, the relative
    rule's, the thresholds and a plain-text reason."""
    fmax = float(f.max())
    t_in = cal.t_hi + (R.DIGIT_MARGIN * np.clip((st - R.DIGIT_MARGIN_FROM) / (R.DIGIT_MARGIN_FULL - R.DIGIT_MARGIN_FROM), 0.0, 1.0) if strict else 0.0)
    rel_out = 0.25 if strict else 0.35
    ratio = f / fmax if fmax > 0 else np.zeros_like(f)
    is_in = (f >= t_in) & (ratio >= 0.7) & (st >= cal.min_strength)
    is_out = (f < cal.t_lo) | (ratio < rel_out)
    s_abs, m_abs, c_abs, a_abs, confident = R._decide_abs(f, st, cal, strict)
    rel, ex = R._relative_marks(md, strict)
    excess = None
    if md is not None and md.size >= 3 and not np.isnan(md).any():
        excess = [md[i] - np.median(np.delete(md, i)) for i in range(md.size)]
    ff = np.stack(frame_f) if frame_f else f[None]
    ch = [{"label": labels[i] if labels else str(i), "fill_pct": round(float(f[i]) * 100, 1), "strength": _r(st[i]), "disc_dark": _r(None if md is None else md[i]),
           "excess_vs_others": _r(None if excess is None else excess[i]), "ratio_to_max": _r(ratio[i]), "t_in": _r(t_in[i] if strict else t_in),
           "abs_class": "in" if is_in[i] else "out" if is_out[i] else "grey",
           "fill_pct_min": round(float(ff[:, i].min()) * 100, 1), "fill_pct_max": round(float(ff[:, i].max()) * 100, 1), "fill_pct_std": round(float(ff[:, i].std()) * 100, 1)}
          for i in range(len(f))]
    why = []
    if not (f >= t_in).any():
        why.append(f"no choice reaches t_in ({_r(float(np.min(t_in)))}); fullest fill {_r(fmax)} " +
                   ("< t_lo -> blank" if fmax < cal.t_lo else ">= (t_lo+t_hi)/2 -> weak single guess, flagged" if fmax >= 0.5 * (cal.t_lo + cal.t_hi) else "-> blank (but unsure)"))
    else:
        for i, c in enumerate(ch):
            if c["abs_class"] == "grey":
                why.append(f"{c['label']}: grey zone (fill {c['fill_pct']}%, ratio {c['ratio_to_max']}, strength {c['strength']}) -> doubt")
        if int(is_in.sum()) >= 2:
            why.append("two or more choices count as marked -> multi")
    if not confident and rel:
        why.append(f"relative rule: {[ch[i]['label'] for i in rel]} darker than the other choices by {[_r(x) for x in ex]} (needs >= {R.REL_MARGIN}; silent from {R.REL_SURE_DIGIT if strict else R.REL_SURE}); rule {'ON' if trust else 'OFF (region not trusted)'}")
    return {"choices": ch, "abs": {"state": s_abs, "marked": [int(i) for i in m_abs], "confidence_pct": round(float(c_abs) * 100), "ambiguous": bool(a_abs), "confident": bool(confident)},
            "rel": {"indices": rel, "excess": [_r(x) for x in ex], "trusted": bool(trust)}, "fmax_pct": round(fmax * 100, 1), "why": why}


def explain(result) -> dict:
    """Dev log: the decision trace of a committed sheet (see _row_trace), from the fused measurements of its track."""
    src = getattr(result, "trace_src", None) or ((result.raw, [result.raw]) if result.raw is not None else None)
    if src is None:
        return {}
    raw, frames = src
    cal = R._calibrate(raw.FA, raw.ink, min_strength=raw.min_strength)
    trust, page_trust = R._rel_trust_answers(raw, cal)
    FD = raw.FI if raw.FS is None else np.concatenate([raw.FI, raw.FS])
    ical = R._calibrate(FD, raw.ink, default_typ=cal.typ, min_strength=raw.min_strength)
    SD = raw.SI if raw.SS is None else np.concatenate([raw.SI, raw.SS])
    MD = None if raw.MI is None else (raw.MI if raw.MS is None else np.concatenate([raw.MI, raw.MS]))
    dtrust = page_trust or R._rel_trust_digits(FD, SD, MD, ical)
    out = {"page": {"frames_fused": len(frames), "ink": _r(raw.ink), "ink_source": raw.ink_source, "ink_separation": _r(raw.ink_separation), "min_strength": _r(raw.min_strength),
                    "typ_fill": _r(cal.typ), "t_lo": _r(cal.t_lo), "t_hi": _r(cal.t_hi), "id_typ_fill": _r(ical.typ), "id_t_lo": _r(ical.t_lo), "id_t_hi": _r(ical.t_hi),
                    "alignment_verified": bool(raw.verified), "relative_rule_page_trust": bool(page_trust), "relative_rule_digit_trust": bool(dtrust),
                    "consts": {"REL_MARGIN": R.REL_MARGIN, "REL_SURE": R.REL_SURE, "REL_SURE_DIGIT": R.REL_SURE_DIGIT, "GUESS_GAP": R.GUESS_GAP, "DIGIT_MARGIN": R.DIGIT_MARGIN,
                               "T_HI_OF_TYP": R.T_HI_OF_TYP, "T_LO_FRAC": R.T_LO_FRAC}},
           "answers": [], "id": [], "subject": []}
    for i, a in enumerate(result.answers):
        if i >= len(raw.FA):
            break
        t = _row_trace(raw.FA[i], raw.SA[i], None if raw.MA is None else raw.MA[i], cal, False, list(a.labels), [x.FA[i] for x in frames], trust[i])
        t.update(q=a.number, final={"state": a.state, "marked": list(a.marked), "confidence_pct": round(a.confidence * 100), "ambiguous": bool(a.ambiguous), "guessed": bool(a.guessed), "guess": a.guess})
        out["answers"].append(t)
    for key, digits, F, S_, M_, fr_key in (("id", result.id_digits, raw.FI, raw.SI, raw.MI, "FI"), ("subject", result.subject_digits, raw.FS, raw.SS, raw.MS, "FS")):
        if F is None:
            continue
        for i, d in enumerate(digits):
            if i >= len(F):
                break
            t = _row_trace(F[i], S_[i], None if M_ is None else M_[i], ical, True, None, [getattr(x, fr_key)[i] for x in frames], dtrust)
            t.update(digit=i + 1, final={"state": d.state, "value": d.value, "confidence_pct": round(d.confidence * 100), "ambiguous": bool(d.ambiguous), "guessed": bool(d.guessed)})
            out[key].append(t)
    return out


def _beacon(summary) -> None:
    """Tell the page about a commit the moment it is decided: the main thread beeps / flashes at once, before the rest of the event
    (detail sheet JPEG ~20 ms native, dev trace ~60 ms) is built, serialised and posted.  No-op outside Pyodide (native tests)."""
    try:
        import js
        from pyodide.ffi import to_js
        js.postMessage(to_js({"type": "beacon", "severity": summary.get("severity") or "ok"}, dict_converter=js.Object.fromEntries))
    except Exception:
        pass


def summarize(result) -> dict:
    """Same fields as app/main.py::summarize, plus the marked labels per question (for display / CSV)."""
    return {"form_key": result.form_key, "page": result.page, "total_pages": result.total_pages,
            "student_id": result.student_id, "subject": result.subject, "serial": result.serial,
            "n_questions": len(result.answers),
            "n_blank": sum(1 for a in result.answers if a.state == "blank"),
            "n_multi": sum(1 for a in result.answers if a.state == "multi"),
            "n_doubtful": sum(1 for a in result.answers if a.ambiguous),
            "flags": list(result.flags), "needs_review": result.needs_review,
            "sharpness": (result.quality or {}).get("sharpness"), "severity": severity(result),
            "answers": [{"q": a.number, "state": a.state, "marked": "|".join(a.labels[i] for i in a.marked),
                         "doubt": bool(a.ambiguous)} for a in result.answers]}


def _digit_rows(digits, centers):
    return [{"state": d.state, "value": d.value, "doubt": bool(d.ambiguous), "guessed": bool(getattr(d, "guessed", False)), "scores": d.scores,
             "centers": centers[i]} for i, d in enumerate(digits) if i < len(centers)]


def detail(result) -> dict:
    """What the review view needs to show *where* a sheet is doubtful: the canonical (rectified) sheet as a JPEG
    plus, per question / ID digit / subject digit, the bubble centres (mm), fill scores and the decision."""
    if result.warped is None:
        return {}
    ok, jpg = cv2.imencode(".jpg", result.warped, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return {}
    bc, cal = result.bubble_centers_mm or {}, result.calibration or {}
    ans = bc.get("answers", [])
    return {"img": base64.b64encode(jpg.tobytes()).decode("ascii"), "ppm": result.px_per_mm,
            "t_lo": cal.get("t_lo"), "t_hi": cal.get("t_hi"), "id_t_lo": cal.get("id_t_lo"), "id_t_hi": cal.get("id_t_hi"),
            "answers": [{"q": a.number, "state": a.state, "marked": list(a.marked), "labels": list(a.labels),
                         "scores": a.scores, "doubt": bool(a.ambiguous), "guessed": bool(getattr(a, "guessed", False)), "centers": ans[i]}
                        for i, a in enumerate(result.answers) if i < len(ans)],
            "id": _digit_rows(result.id_digits, bc.get("id", [])),
            "subject": _digit_rows(result.subject_digits, bc.get("subject", []))}


class Scanner:
    def __init__(self):
        self.session = StreamSession(resolve_layout=None)
        self._rgba = self._gray = None
        self._convert_ms = 0.0

    def reset(self):
        self.session.reset()
        self.session.reset_tracking()

    def step_rgba(self, buf, w: int, h: int) -> str:
        """buf: RGBA bytes (memoryview / JsProxy Uint8Array) of a w x h frame.  Returns the event as JSON.
        The recognizer only reads the grey image (omr.recognize converts BGR->GRAY first thing), so RGBA goes straight to
        GRAY (bit-identical to RGBA->BGR->GRAY) into reused buffers: one 8 MB copy in, no BGR image, no per-frame allocation."""
        t0 = time.perf_counter()
        if self._rgba is None or self._rgba.shape[:2] != (h, w):
            self._rgba = np.empty((h, w, 4), np.uint8)
            self._gray = np.empty((h, w), np.uint8)
        try:
            buf.assign_to(self._rgba)        # JsProxy (Uint8Array): memcpy straight into the numpy buffer
        except AttributeError:               # plain bytes / memoryview (native tests)
            self._rgba[...] = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
        except Exception:                    # any Pyodide-side surprise: fall back to the old copy path
            self._rgba[...] = np.frombuffer(buf.to_py(), np.uint8).reshape(h, w, 4)
        cv2.cvtColor(self._rgba, cv2.COLOR_RGBA2GRAY, dst=self._gray)
        self._convert_ms = (time.perf_counter() - t0) * 1000
        return json.dumps(self.step(self._gray, t0))

    def step(self, img, t0=None) -> dict:
        session = self.session
        t0 = time.perf_counter() if t0 is None else t0
        if min(img.shape[:2]) < MIN_FRAME_SIDE:
            w, h = int(img.shape[1]), int(img.shape[0])
            return {"state": session.state, "hint": f"Camera frame too small ({w}x{h})", "overlay": {},
                    "error": f"frame too small: {w}x{h}", "w": w, "h": h}
        res = recognize(img, None)
        t_feed = time.perf_counter()
        ev = session.feed(res, time.monotonic(), payload=None)
        t_feed = (time.perf_counter() - t_feed) * 1000
        ev["status"] = res.status
        ev["w"], ev["h"] = int(img.shape[1]), int(img.shape[0])
        for name in ("commit", "revision"):      # a revision = a better reading of the sheet committed before (omr/revision.py)
            c = ev.pop(name, None)
            if c is None:
                continue
            summ = summarize(c["result"])
            if name == "commit":
                _beacon(summ)        # the beep must not wait for the detail JPEG / the dev trace below
            ev[name] = {"summary": summ, "frames": c["frames"], "detail": detail(c["result"])}
            if name == "revision":
                ev[name].update(changed=c["changed"], contradiction=c["contradiction"], prev_severity=c["prev_severity"], n=c["revision"])
            if DEV:
                cr = c["result"]
                ev[name]["dev"] = {"quality": _plain(cr.quality), "calibration": _plain(cr.calibration), "fused_frames": cr.fused_frames,
                                   "frame_index": c.get("frame_index"), "marker_ids": list(cr.marker_ids), "missing_corners": list(cr.missing_corners),
                                   "trace": explain(cr)}
        if DEV:
            ev["dev"] = {"quality": _plain(res.quality), "flags": list(res.flags), "markers": list(res.marker_ids),
                         "missing": list(res.missing_corners), "form_key": res.form_key, "page": res.page,
                         "timing": {k: round(v, 1) for k, v in (res.timing_ms or {}).items()}}
        missed = session.missed()
        if missed:
            ev["missed"] = missed
        ev["ms"] = {"total": round((time.perf_counter() - t0) * 1000, 1),
                    "recognize": round(res.timing_ms.get("total", 0), 1)}
        if DEV:
            ev["ms"].update(convert=round(self._convert_ms, 2), feed=round(t_feed, 1))
        self._convert_ms = 0.0
        return _clean(ev)


_scanner = None


def init():
    global _scanner
    _scanner = Scanner()


def reset():
    _scanner.reset()


def frame(buf, w, h):
    return _scanner.step_rgba(buf, w, h)
