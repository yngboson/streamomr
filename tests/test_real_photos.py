"""Real iPhone photos of one filled sheet (realasset/*.jpeg) against realasset/truth.json."""
import glob, json, os
import cv2
import pytest
from omr.recognize import recognize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "realasset", "*.jpeg")))
TRUTH = os.path.join(ROOT, "realasset", "truth.json")
pytestmark = pytest.mark.skipif(not FILES or not os.path.exists(TRUTH), reason="realasset photos missing")


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.33])
@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(f)[:8] for f in FILES])
def test_photo_fully_correct(path, scale):
    T = json.load(open(TRUTH))
    img = cv2.imread(path)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    r = recognize(img, None)
    ans = [(q.marked[0] + 1 if q.state == "single" and len(q.marked) == 1 else None) for q in (r.answers or [])[:len(T["answers"])]]
    assert r.status == "ok"
    assert r.student_id == T["student_id"] and r.subject == T["subject"]
    assert ans == T["answers"]
