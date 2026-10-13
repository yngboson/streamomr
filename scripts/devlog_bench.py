"""Dev-log bench: python scripts/devlog_bench.py [--dir realasset/devlog_2026-10-13]
(a) replays every scans/*/frame.jpg through recognize() (ONE frame, no fusion) and compares ID, subject and answers with
    truth.csv: correct / flagged-but-wrong / SILENT wrong.  An answer counts as 'flagged' when the row is ambiguous (its guess
    is then what is compared), otherwise it is a confident read.
(b) re-decides the rows of the logged (fused) scan.json whose bubbles were contaminated by the UNRELIABLE sentinel (fill exactly
    50 % = recognize.UNRELIABLE_FILL, shown as median or as min/max of the per-frame range): the log only keeps median + min..max
    per bubble, not every frame, so the 'after' is a PROXY: the sentinel end of the range is replaced by the measured end
    (fusion now ignores sentinel samples).  The before column is the logged decision (final)."""
import argparse, csv, glob, json, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omr import recognize as R


def load_truth(d):
    out = {}
    with open(os.path.join(d, 'truth.csv'), encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            out[row['student_id']] = (row['subject'], [int(row[f'q{i}']) for i in range(1, 46)])
    return out


def answer_value(q):
    """(value or tuple of marks or None, flagged)"""
    if q.state == 'single' and len(q.marked) == 1 and not q.ambiguous:
        return q.marked[0] + 1, False
    if q.state == 'multi':
        return tuple(m + 1 for m in q.marked), True
    if q.guessed and q.guess is not None:
        return q.guess + 1, True
    return None, True


def part_a(d, truth):
    c = fw = sw = 0; id_ok = id_bad = subj_ok = 0; n = 0; silent_list = []
    for sd in sorted(glob.glob(os.path.join(d, 'scans', '*'))):
        sid = os.path.basename(sd).split('_', 1)[1]
        if sid not in truth: continue
        subj, T = truth[sid]
        r = R.recognize(cv2.imread(os.path.join(sd, 'frame.jpg')), None)
        n += 1
        if r.student_id == sid: id_ok += 1
        elif r.student_id and not r.id_guessed: id_bad += 1; print(f'  SILENT WRONG ID {sid} -> {r.student_id}')
        subj_ok += r.subject == subj
        if r.status != 'ok': print(f'  {sid}: status {r.status}'); continue
        for i, q in enumerate(r.answers[:45]):
            v, flagged = answer_value(q)
            if v == T[i]: c += 1
            elif flagged: fw += 1
            else: sw += 1; silent_list.append((sid, q.number, v, T[i]))
    tot = c + fw + sw
    print(f'(a) single-frame replay: sheets {n}  ID right {id_ok} silent-wrong ID {id_bad}  subject right {subj_ok}')
    print(f'    answers {tot}: correct {c}  flagged-but-wrong {fw}  SILENT wrong {sw} {silent_list}')


def part_b(d, truth):
    print('(b) logged fused scans: rows touched by sentinel samples (before = logged final, after = proxy re-decision)')
    nrows = fixed = 0
    for sd in sorted(glob.glob(os.path.join(d, 'scans', '*'))):
        sid = os.path.basename(sd).split('_', 1)[1]
        j = json.load(open(os.path.join(sd, 'scan.json'), encoding='utf-8'))
        cal = j['dev']['calibration']; tr = j['dev']['trace']
        c = R._Calib(cal['ink'], cal['typ_fill'], cal['t_hi'], cal['t_lo'], tr['page'].get('min_strength', 0.6))
        for a in tr['answers']:
            ch = a['choices']
            S = lambda k: abs(k - 50) < 0.05
            if not any(S(x['fill_pct']) or S(x['fill_pct_min']) or S(x['fill_pct_max']) for x in ch): continue
            f = np.array([x['fill_pct'] / 100 for x in ch]); s = np.array([x['strength'] for x in ch])
            fa = f.copy(); sa = s.copy()
            for i, x in enumerate(ch):     # proxy for the non-sentinel median: the measured end of the min..max range
                lo, hi = x['fill_pct_min'] / 100, x['fill_pct_max'] / 100
                if S(x['fill_pct']) or S(x['fill_pct_min']) or S(x['fill_pct_max']):
                    fa[i] = lo if S(x['fill_pct_max']) else hi if S(x['fill_pct_min']) else f[i]
                    sa[i] = 1.0 if fa[i] >= c.t_hi else 0.5
            state, marked, conf, amb = R._decide(fa, sa, c)
            f0 = a.get('final', {}); b = f0.get('guess', None) if f0.get('guessed') else (f0.get('marked') or [None])[0]
            after = (marked[0] if state == 'single' and not amb else None)
            t = truth[sid][1][a['q'] - 1] - 1 if sid in truth else None
            nrows += 1; fixed += after == t and b != t
            print(f"    scan {sid} Q{a['q']}: before {state_s(f0)} -> after {state}{[m + 1 for m in marked]}{' amb' if amb else ''}   truth {t + 1 if t is not None else '?'}")
    print(f'    rows {nrows}, rows turned from wrong/flagged to confident-right {fixed}')


def state_s(f):
    g = f.get('guess')
    return f"{f.get('state')}{[m + 1 for m in f.get('marked', [])]}{' guess ' + str(g + 1) if g is not None else ''}{' amb' if f.get('ambiguous') else ''}"


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='realasset/devlog_2026-10-13'); a = ap.parse_args()
    tr = load_truth(a.dir)
    part_a(a.dir, tr); part_b(a.dir, tr)
