"""Field-level comparison of a JS result with the Python result of the same frame."""
import math

CORNER_TOL_PX = 0.5


def num_close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def decision_fields(r):
    """Flat {field: value} of everything a user sees: the 'fields' of the agreement metric."""
    f = {"status": r["status"], "form_key": r["form_key"], "page": r["page"], "serial": r["serial"], "student_id": r["student_id"],
         "subject": r["subject"], "flags": tuple(sorted(r["flags"])), "marker_ids": tuple(r["marker_ids"]),
         "missing_corners": tuple(r["missing_corners"])}
    for a in r["answers"]:
        f[f"q{a['number']}"] = (a["state"], tuple(a["marked"]), bool(a["ambiguous"]), a["guess"])
    for k, key in (("id", "id_digits"), ("sj", "subject_digits")):
        for d in r[key]:
            f[f"{k}{d['index']}"] = (d["state"], d["value"], bool(d["ambiguous"]), bool(d["guessed"]))
    return f


def compare(py, js):
    """-> (n_fields, [differences (field, py, js)], stats dict of numeric deviations)"""
    fp, fj = decision_fields(py), decision_fields(js)
    diffs, n = [], 0
    for k in sorted(set(fp) | set(fj)):
        n += 1
        if fp.get(k, "<absent>") != fj.get(k, "<absent>"):
            diffs.append((k, fp.get(k, "<absent>"), fj.get(k, "<absent>")))
    st = {}
    # marker corners
    if py["marker_corners"] and js["marker_corners"] and py["marker_ids"] == js["marker_ids"]:
        dj = {i: q for i, q in js["marker_corners"]}
        st["corner_px"] = max(math.hypot(p[0] - q[0], p[1] - q[1]) for i, qq in py["marker_corners"] for p, q in zip(qq, dj[i]))
    if py["homography"] and js["homography"]:
        st["H_rel"] = max(abs(a - b) / max(1.0, abs(a)) for ra, rb in zip(py["homography"], js["homography"]) for a, b in zip(ra, rb))
    if py["raw"] and js["raw"]:
        for key in ("FA", "SA", "FI", "SI", "MA", "MI"):
            a, b = py["raw"][key], js["raw"][key]
            if a is None or b is None:
                continue
            mx = 0.0
            for ra, rb in zip(a, b):
                for x, y in zip(ra, rb):
                    if x is None or y is None:
                        if (x is None) != (y is None):
                            mx = max(mx, 1.0)
                        continue
                    mx = max(mx, abs(x - y))
            st["raw_" + key] = mx
        st["ink"] = abs(py["raw"]["ink"] - js["raw"]["ink"])
    for k in ("meta_black", "align_resid_mm", "align_ring_corr", "reproj_max_mm", "ink_abs", "meta_contrast"):
        a, b = py["quality"].get(k), js["quality"].get(k)
        if a is not None and b is not None:
            st["q_" + k] = abs(a - b)
    return n, diffs, st
