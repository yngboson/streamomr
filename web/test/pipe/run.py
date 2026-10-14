"""One command for the J3 pipeline tests:   .venv/bin/python web/test/pipe/run.py [--sets photos,devlog,video,synth] [--regen] [--jobs 10] [--no-time] [--no-stream]

1. exports the Python fixtures (grey frames + Python results) when missing (export.py),
2. recognises every frame with the JS port under JavaScriptCore (parallel chunks) -> fixtures/pipe/<set>.js.json,
3. field-level agreement with Python, accuracy vs truth (photos, devlog, synthetic), StreamSession replay (video, devlog),
4. timing per frame class (JS vs Python-native) on the video frames.
Exit code 0 when every acceptance criterion holds."""
import argparse, collections, csv, concurrent.futures as cf, glob, json, math, os, pickle, subprocess, sys, time
from common import *  # noqa: F401,F403
from compare import compare

import numpy as np
from omr import recognize as R


# ------------------------------------------------------------------ running the JS side
def chunks(n, k):
    step = max(1, math.ceil(n / k))
    return [(i, min(i + step, n)) for i in range(0, n, step)]


PARITY = False


def js_chunk(setname, a, b, reps=1, idxs=None):
    spec = ",".join(map(str, idxs)) if idxs is not None else f"{a}-{b}"
    p = run_jsc(["--", FIX, setname, spec, str(reps)] + (["parity"] if PARITY else []), files=src_files() + [HERE + "/ser.js", HERE + "/drv.js"])
    out = [json.loads(l[1:]) for l in p.stdout.splitlines() if l.startswith("@")]
    if p.returncode != 0 or p.stderr.strip():
        print(f"[jsc {setname} {spec}] rc={p.returncode} {p.stderr[:600]}")
    return out


def run_js(setname, jobs):
    man = json.load(open(f"{FIX}/{setname}.manifest.json"))
    t0 = time.time()
    with cf.ThreadPoolExecutor(jobs) as ex:
        parts = list(ex.map(lambda ab: js_chunk(setname, *ab), chunks(len(man), jobs * 3)))
    res = {}
    for part in parts:
        for j in part:
            res[j["name"]] = j["res"]
            res[j["name"]]["js_ms"] = j["ms"]
    json.dump(res, open(f"{FIX}/{setname}.js.json", "w"))
    print(f"{setname}: js recognised {len(res)}/{len(man)} frames in {time.time() - t0:.0f}s wall", flush=True)
    return res


# ------------------------------------------------------------------ python objects from JSON (to reuse the python evaluators)
def to_res(j):
    r = R.RecognitionResult(status=j["status"], form_key=j["form_key"], form_id=j["form_id"], page=j["page"], total_pages=j["total_pages"],
                            serial=j["serial"], student_id=j["student_id"], subject=j["subject"], flags=list(j["flags"]),
                            answers=[R.QuestionResult(**a) for a in j["answers"]], id_digits=[R.DigitResult(**d) for d in j["id_digits"]],
                            subject_digits=[R.DigitResult(**d) for d in j["subject_digits"]])
    return r


# ------------------------------------------------------------------ truth: photos
def photo_truth(py, js, man):
    T = json.load(open(f"{REAL}/truth.json"))
    TM = json.load(open(f"{REAL}/truth_mixed.json"))
    rows = []
    for m in man:
        t = TM if m["name"].startswith("mixed_ink") else T
        line = {}
        for tag, res in (("py", py[m["name"]]), ("js", js[m["name"]])):
            ans = [(q["marked"][0] + 1 if q["state"] == "single" and len(q["marked"]) == 1 else None) for q in res["answers"][:len(t["answers"])]]
            n_ok = sum(x == y for x, y in zip(ans, t["answers"]))
            allok = res["status"] == "ok" and res["student_id"] == t["student_id"] and res["subject"] == t["subject"] and n_ok == len(t["answers"])
            line[tag] = dict(n_ok=n_ok, n=len(t["answers"]), allok=allok, sid=res["student_id"], subj=res["subject"])
        rows.append((m["name"], line))
    return rows


# ------------------------------------------------------------------ truth: devlog
def load_devlog_truth():
    out = {}
    with open(f"{REAL}/devlog_2026-10-13/truth.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["student_id"]] = (row["subject"], [int(row[f"q{i}"]) for i in range(1, 46)])
    return out


def answer_value(q):
    if q["state"] == "single" and len(q["marked"]) == 1 and not q["ambiguous"]:
        return q["marked"][0] + 1, False
    if q["state"] == "multi":
        return tuple(m + 1 for m in q["marked"]), True
    if q.get("guessed") and q.get("guess") is not None:
        return q["guess"] + 1, True
    return None, True


def devlog_score(results_by_sid, truth):
    c = fw = sw = id_ok = subj_ok = n = 0
    silent = []
    for sid, r in results_by_sid.items():
        if sid not in truth or r is None:
            continue
        subj, T = truth[sid]
        n += 1
        id_ok += r["student_id"] == sid
        subj_ok += r["subject"] == subj
        if r["status"] != "ok" and "states" not in r:
            continue
        answers = r["answers"]
        for i, q in enumerate(answers[:45]):
            v, flagged = answer_value(q)
            if v == T[i]:
                c += 1
            elif flagged:
                fw += 1
            else:
                sw += 1
                silent.append((sid, q["number"], v, T[i]))
    return dict(sheets=n, answers_correct=c, flagged_wrong=fw, silent_wrong=sw, id_ok=id_ok, subject_ok=subj_ok, silent=silent)


# ------------------------------------------------------------------ python stream replay
def py_rec(kind, t, r, **extra):
    d = dict(kind=kind, t=t, student_id=r.student_id, subject=r.subject, serial=r.serial, page=r.page, form_key=r.form_key,
             fused_frames=r.fused_frames, flags=list(r.flags), needs_review=r.needs_review,
             states=[[a.number, a.state, list(a.marked), bool(a.ambiguous), a.guess] for a in r.answers],
             id_values=[x.value for x in r.id_digits], subject_values=[x.value for x in r.subject_digits])
    d.update(extra)
    return d


def py_stream(setname, man):
    from omr.stream import StreamSession
    pk = pickle.load(open(f"{FIX}/{setname}.py.pkl", "rb"))
    out = []
    if setname == "video":
        s = StreamSession(resolve_layout=None)
        last = 0.0
        for m in man:
            ev = s.feed(pk[m["name"]], m["t"], payload=None)
            last = m["t"]
            if "commit" in ev:
                out.append(py_rec("commit", m["t"], ev["commit"]["result"], frames=ev["commit"]["frames"], frame_index=ev["commit"]["frame_index"]))
            elif "revision" in ev:
                out.append(py_rec("revision", m["t"], ev["revision"]["result"], frames=ev["revision"]["frames"], changed=ev["revision"]["changed"]))
        c = s.finish(last)
        if c is not None:
            out.append(py_rec("finish", last, c["result"], frames=c["frames"]))
        out.append(dict(kind="missed", episodes=len(s.unread_episodes(final=True)), missed=len(s.missed(final=True))))
    else:
        import copy
        for m in man:
            s = StreamSession(resolve_layout=None)
            got = None
            for q in range(6):
                ev = s.feed(copy.copy(pk[m["name"]]), q / 8, payload=None)
                if "commit" in ev and got is None:
                    got = py_rec("commit", q / 8, ev["commit"]["result"], name=m["name"])
            if got is None:
                c = s.finish(1)
                if c is not None:
                    got = py_rec("finish", 1, c["result"], name=m["name"])
            out.append(got or dict(kind="none", name=m["name"]))
    return out


def js_stream(setname):
    p = run_jsc(["--", FIX, setname, "video" if setname == "video" else "devlog"], files=src_files() + [HERE + "/ser.js", HERE + "/stream.js"])
    lines = [l for l in p.stdout.splitlines() if l.startswith("@")]
    if not lines:
        print("stream.js failed:", p.stderr[:1500], p.stdout[:500])
        return None
    return json.loads(lines[0][1:])


def stream_key(c):
    if c["kind"] == "missed":
        return ("missed", c["episodes"], c["missed"])
    return (c["kind"], c.get("name"), c["student_id"], c["subject"], c["serial"], c["page"], c["form_key"], c["fused_frames"], tuple(sorted(c["flags"])),
            tuple((n, st, tuple(mk), amb, g) for n, st, mk, amb, g in c["states"]), tuple(c["id_values"]), tuple(c["subject_values"]))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="photos,devlog,video,synth")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--parity", action="store_true", help="JS with detect.py's exact pass sequence (no J3 pass-policy optimisations)")
    ap.add_argument("--reuse", action="store_true", help="reuse the last <set>.js.json instead of re-running the JS side")
    a = ap.parse_args()
    global PARITY
    PARITY = a.parity
    sets = a.sets.split(",")
    missing = [s for s in sets if not os.path.exists(f"{FIX}/{s}.py.pkl")]
    if a.regen or missing:
        subprocess.check_call([sys.executable, HERE + "/export.py", "--sets", ",".join(sets if a.regen else missing), "--regen"], cwd=ROOT)
    ok_all = True
    tot_fields = tot_diffs = 0
    all_diffs = []
    stats_max = collections.defaultdict(float)
    for s in sets:
        py = json.load(open(f"{FIX}/{s}.py.json"))
        man = json.load(open(f"{FIX}/{s}.manifest.json"))
        js = json.load(open(f"{FIX}/{s}.js.json")) if (a.reuse and os.path.exists(f"{FIX}/{s}.js.json")) else run_js(s, a.jobs)
        miss = [m["name"] for m in man if m["name"] not in js]
        if miss:
            print(f"!! {s}: JS produced no result for {len(miss)} frames: {miss[:5]}")
            ok_all = False
        nf = nd = 0
        status_diff = []
        for m in man:
            if m["name"] not in js:
                continue
            n, diffs, st = compare(py[m["name"]], js[m["name"]])
            nf += n
            nd += len(diffs)
            for d in diffs:
                all_diffs.append((s, m["name"]) + d)
            for k, v in st.items():
                stats_max[(s, k)] = max(stats_max[(s, k)], v)
            if py[m["name"]]["status"] != js[m["name"]]["status"]:
                status_diff.append((m["name"], py[m["name"]]["status"], js[m["name"]]["status"]))
        tot_fields += nf
        tot_diffs += nd
        print(f"== {s}: {len(man)} frames, {nf} fields, {nd} differ -> agreement {100 * (1 - nd / max(nf, 1)):.3f}%   status differences: {len(status_diff)} {status_diff[:6]}")
        print("   max numeric deviation:", {k[1]: float(f"{v:.3g}") for k, v in stats_max.items() if k[0] == s})

        # ---------------- truth
        if s == "photos":
            rows = photo_truth(py, js, man)
            for name, line in rows:
                print(f"   {name[:12]:12s} py {line['py']['n_ok']}/{line['py']['n']} {'OK' if line['py']['allok'] else 'FAIL'} id={line['py']['sid']}  js {line['js']['n_ok']}/{line['js']['n']} {'OK' if line['js']['allok'] else 'FAIL'} id={line['js']['sid']}")
            jok = sum(l["js"]["allok"] for _, l in rows)
            pok = sum(l["py"]["allok"] for _, l in rows)
            print(f"   photo bench: python {pok}/{len(rows)}   js {jok}/{len(rows)}")
            ok_all &= jok >= pok and jok == len(rows)
        if s == "devlog":
            truth = load_devlog_truth()
            for tag, res in (("python", py), ("js", js)):
                sc = devlog_score({n: r for n, r in res.items()}, truth)
                print(f"   single-frame {tag}: sheets {sc['sheets']} ids {sc['id_ok']} subjects {sc['subject_ok']} answers correct {sc['answers_correct']} flagged-wrong {sc['flagged_wrong']} SILENT {sc['silent_wrong']}")
        if s == "synth":
            import synth
            truths = pickle.load(open(f"{FIX}/synth.truth.pkl", "rb"))
            tot = {"py": collections.Counter(), "js": collections.Counter()}
            bycond = collections.defaultdict(lambda: {"py": collections.Counter(), "js": collections.Counter()})
            for m in man:
                if m["name"] not in js or m["name"] not in truths:
                    continue
                for tag, res in (("py", py), ("js", js)):
                    ev = synth.evaluate(to_res(res[m["name"]]), truths[m["name"]])
                    for k, v in ev.items():
                        if isinstance(v, (int, float)):
                            tot[tag][k] += v
                            bycond[m["cond"]][tag][k] += v
                    st_ok = res[m["name"]]["status"] in truths[m["name"]].get("expect_status", ["ok"])
                    tot[tag]["status_expected"] += int(st_ok)
                    tot[tag]["frames"] += 1
            keys = ["frames", "status_expected", "q_total", "q_clear_ok", "q_hard_ok", "q_silent", "q_silent_clear", "q_guessed", "q_guess_ok", "id_ok", "id_wrong", "id_none", "id_guessed", "id_guess_ok",
                    "subject_ok", "subject_wrong", "subject_none", "nosubj_ok", "serial_ok", "serial_wrong"]
            print("   synthetic accuracy (sum over all frames):")
            for k in keys:
                print(f"     {k:16s} py {tot['py'][k]:6d}   js {tot['js'][k]:6d}")
            worse = [k for k in ("q_silent", "id_wrong", "subject_wrong", "serial_wrong") if tot["js"][k] > tot["py"][k]]
            better_ok = tot["js"]["q_clear_ok"] + tot["js"]["q_hard_ok"] >= tot["py"]["q_clear_ok"] + tot["py"]["q_hard_ok"]
            print("   js not worse than python:", "YES" if not worse and better_ok else f"NO {worse} correct-ge={better_ok}")
            for c, d in sorted(bycond.items()):
                dq = (d["js"]["q_clear_ok"] + d["js"]["q_hard_ok"]) - (d["py"]["q_clear_ok"] + d["py"]["q_hard_ok"])
                ds = d["js"]["q_silent"] - d["py"]["q_silent"]
                di = d["js"]["id_ok"] - d["py"]["id_ok"]
                if dq or ds or di:
                    print(f"     cond {c}: correct answers js-py {dq:+d}, silent js-py {ds:+d}, id_ok js-py {di:+d}")
            ok_all &= not worse and better_ok
        if s in ("video", "devlog") and not a.no_stream:
            jr = js_stream(s)
            pr = py_stream(s, man)
            if jr is None:
                ok_all = False
            else:
                ka, kb = [stream_key(c) for c in pr], [stream_key(c) for c in jr]
                same = ka == kb
                print(f"   StreamSession ({'one session' if s == 'video' else 'fresh session per frame, 6 copies'}): python {len(pr)} records, js {len(jr)}, identical: {same}")
                if s == "video":
                    for c in pr:
                        if c["kind"] != "missed":
                            print(f"     py  {c['kind']:8s} t={c['t']:7.2f} id={c['student_id']} subj={c['subject']} serial={c['serial']} fused={c['fused_frames']}")
                    for c in jr:
                        if c["kind"] != "missed":
                            print(f"     js  {c['kind']:8s} t={c['t']:7.2f} id={c['student_id']} subj={c['subject']} serial={c['serial']} fused={c['fused_frames']}")
                if s == "devlog":
                    truth = load_devlog_truth()
                    for tag, recs in (("python", pr), ("js", jr)):
                        by = {}
                        for c in recs:
                            if c.get("name"):
                                by[c["name"]] = None if c["kind"] == "none" else dict(student_id=c["student_id"], subject=c["subject"], status="ok", states=1,
                                                                                         answers=[dict(number=n, state=st, marked=mk, ambiguous=amb, guessed=g is not None and amb, guess=g) for n, st, mk, amb, g in c["states"]])
                        sc = devlog_score(by, truth)
                        print(f"   fused via StreamSession {tag}: sheets {sc['sheets']} ids {sc['id_ok']}/{sc['sheets']} subjects {sc['subject_ok']}/{sc['sheets']} "
                              f"answers correct {sc['answers_correct']}/{sc['sheets'] * 45} flagged-wrong {sc['flagged_wrong']} silent {sc['silent_wrong']}")
                        if tag == "js":
                            ok_all &= sc["answers_correct"] == sc["sheets"] * 45 and sc["id_ok"] == sc["sheets"] and sc["subject_ok"] == sc["sheets"]
                if not same:
                    for i, (x, y) in enumerate(zip(ka, kb)):
                        if x != y:
                            print("     first stream difference at record", i, "\n       py", x[:9], "\n       js", y[:9])
                            break
    print(f"\nTOTAL field agreement: {tot_fields - tot_diffs}/{tot_fields} = {100 * (1 - tot_diffs / max(tot_fields, 1)):.3f}%  ({tot_diffs} differing fields)")
    with open(f"{FIX}/diffs.txt", "w") as f:
        for d in all_diffs:
            f.write(repr(d) + "\n")
    for d in all_diffs[:60]:
        print("  diff", d)
    if len(all_diffs) > 60:
        print(f"  ... {len(all_diffs) - 60} more in {FIX}/diffs.txt")
    ok_all &= (tot_diffs / max(tot_fields, 1)) <= 0.01
    if not a.no_time:
        timing(a.jobs)
    print("\nRESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def timing(jobs):
    man = json.load(open(f"{FIX}/video.manifest.json"))
    py = json.load(open(f"{FIX}/video.py.json"))
    by = collections.defaultdict(list)
    for i, m in enumerate(man):
        by[py[m["name"]]["status"]].append(i)
    pick = []
    for st, idx in by.items():
        pick += idx[::max(1, len(idx) // 8)][:8]
    pick.sort()
    print(f"\ntiming: {len(pick)} video frames, status classes { {k: len(v) for k, v in by.items()} } (sequential, best of 3)")
    j = js_chunk("video", 0, 0, reps=3, idxs=pick)
    import cv2
    from omr.recognize import recognize
    rows = collections.defaultdict(lambda: ([], []))
    for item in j:
        m = next(x for x in man if x["name"] == item["name"])
        gray = np.frombuffer(open(f"{FIX}/{m['file']}", "rb").read(), np.uint8).reshape(m["h"], m["w"])
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            recognize(gray, None)
            best = min(best, (time.perf_counter() - t0) * 1000)
        st = py[item["name"]]["status"]
        rows[st][0].append(best)
        rows[st][1].append(item["ms"])
    print(f"{'class':12s}{'n':>4s}{'python ms (cv2 native)':>26s}{'js ms (jsc)':>16s}")
    for st, (p, q) in sorted(rows.items()):
        print(f"{st:12s}{len(p):4d}{np.mean(p):20.0f} (max {np.max(p):.0f}){np.mean(q):12.0f} (max {np.max(q):.0f})")


if __name__ == "__main__":
    sys.exit(main())
