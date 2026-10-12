import os
import re

from fastapi.testclient import TestClient

from app.main import STATIC_DIR, create_app

PAGES = ["index.html", "exam.html", "conflicts.html", "unassigned.html", "student.html", "key.html", "scan.html"]


def test_every_page_is_served_and_references_existing_files(tmp_path):
    c = TestClient(create_app(str(tmp_path)))
    for name in PAGES:
        r = c.get(f"/static/{name}")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], name
        refs = re.findall(r'(?:src|href)="(/static/[^"?#]+)', r.text)
        assert refs, name
        for ref in refs:
            assert os.path.isfile(os.path.join(STATIC_DIR, ref[len("/static/"):])), (name, ref)
            assert c.get(ref).status_code == 200, (name, ref)
        assert "<title>" in r.text and 'lang="ko"' in r.text
    for name in os.listdir(STATIC_DIR):          # nothing orphaned: every page is covered above
        if name.endswith(".html") and name != "standalone.html":      # generated self-contained page (scripts/build_standalone.py)
            assert name in PAGES, name
    assert c.get("/", follow_redirects=False).status_code in (302, 307)
    assert c.get("/static/../app/main.py").status_code in (400, 404)
