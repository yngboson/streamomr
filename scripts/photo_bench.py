"""Real-photo bench: python scripts/photo_bench.py [--scale 1.0] [--rot 0]
Checks every realasset/*.jpeg against realasset/truth.json, and realasset/mixed_ink_pencil_id.jpg against truth_mixed.json (pencil ID + pen answers): status, ID, subject, answers, review flags, time."""
import argparse, glob, json, os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omr.recognize import recognize

ap = argparse.ArgumentParser(); ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--rot', type=int, default=0)
a = ap.parse_args()
T = json.load(open('realasset/truth.json'))
good = q_ok = q_all = g_n = g_ok = 0; times = []
jobs = [(f, T) for f in sorted(glob.glob('realasset/*.jpeg'))]
if os.path.exists('realasset/truth_mixed.json') and os.path.exists('realasset/mixed_ink_pencil_id.jpg'):
    jobs.append(('realasset/mixed_ink_pencil_id.jpg', json.load(open('realasset/truth_mixed.json'))))
for f, T in jobs:
    img = cv2.imread(f)
    if a.scale != 1.0: img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    for _ in range(a.rot // 90): img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    t0 = time.perf_counter(); r = recognize(img, None); dt = (time.perf_counter() - t0) * 1000; times.append(dt)
    ans = [(q.marked[0] + 1 if q.state == 'single' and len(q.marked) == 1 else None) for q in (r.answers or [])[:len(T['answers'])]]
    n_ok = sum(x == y for x, y in zip(ans, T['answers'])); q_ok += n_ok; q_all += len(T['answers'])
    gq = [(q.answer, T['answers'][i]) for i, q in enumerate((r.answers or [])[:len(T['answers'])]) if q.guessed]
    g_n += len(gq); g_ok += sum((x + 1 if x is not None else None) == y for x, y in gq)
    amb = sum(bool(q.ambiguous) for q in (r.answers or [])[:len(T['answers'])])
    allok = r.status == 'ok' and r.student_id == T['student_id'] and r.subject == T['subject'] and n_ok == len(T['answers'])
    good += allok
    print(f"{os.path.basename(f)[:8]} {img.shape[1]}x{img.shape[0]} {dt:6.0f}ms {r.status:13s} id={r.student_id} subj={r.subject} ans={n_ok}/{len(T['answers'])} amb={amb} review={r.needs_review} {'OK' if allok else 'FAIL'} {r.flags}")
    if r.student_id != T['student_id'] and not r.id_guessed: print('  SILENT WRONG ID')
print(f"guessed answers {g_ok}/{g_n} right (flagged, not counted in 'answers' above)")
print(f"sheets fully correct {good}/{len(times)}  answers {q_ok}/{q_all}  time max {max(times):.0f}ms mean {sum(times)/len(times):.0f}ms")
