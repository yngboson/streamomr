import itertools
import json
import math
import os

import pytest

from omr import markers as mk
from omr.layout import (Layout, Template, build_layout, describe, header_cell_rects, header_from_form_key,
                        layout_from_header, page_capacity)

CONFIGS = {
    "q45": Template(45, 5, 8),
    "q100_2p": Template(100, 5, 8, pages=2),
    "q100_3p": Template(100, 5, 8, pages=3),
    "odd": Template(30, 4, 6, choice_labels=list("ABCD")),
    "wide": Template(60, 10, 12, title="아주 긴 제목 " * 6),
    "min": Template(20, 2, 1),
    "used80": Template(100, 5, 8, pages=2, used=80),
    "nosubj": Template(40, 4, 8, subject_digits=0),
    "subj3": Template(60, 5, 6, subject_digits=3, choice_labels=list("abcde")),
}


def circles(lay, pg):
    for col in pg.columns:
        for q in col.questions:
            for b in q.bubbles:
                yield ("a", b.cx, b.cy, pg.bubble_diameter / 2)
    grids = [pg.id_grid] + ([pg.subject_grid] if pg.subject_grid else [])
    for g in grids:
        for c in g.columns:
            for b in c.bubbles:
                yield ("i", b.cx, b.cy, g.bubble_diameter / 2)


def rect_hit_circle(r, cx, cy, rad):
    nx = min(max(cx, r[0]), r[0] + r[2])
    ny = min(max(cy, r[1]), r[1] + r[3])
    return math.hypot(cx - nx, cy - ny) < rad


def rects_hit(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


@pytest.mark.parametrize("name", CONFIGS)
def test_layout_valid(name):
    t = CONFIGS[name]
    lay = build_layout(t)
    assert sum(lay.questions_per_page) == (t.used or t.questions)
    assert lay.form_key == lay.header(0).form_key and lay.form_id is None and lay.version == 1
    nxt = 1
    for pg in lay.pages:
        assert pg.first_question == nxt
        nums = [q.number for c in pg.columns for q in c.questions]
        assert nums == list(range(nxt, nxt + pg.n_questions))
        nxt += pg.n_questions
        for c in pg.columns:
            for q in c.questions:
                assert len(q.bubbles) == t.choices
        assert len(pg.id_grid.columns) == t.id_digits
        assert (len(pg.subject_grid.columns) if pg.subject_grid else 0) == t.subject_digits
        assert len(pg.meta_cells) == mk.HEADER_ROWS * mk.HEADER_ROW_CELLS
        assert pg.bubble_diameter >= t.min_bubble_mm - 1e-9
        assert pg.id_grid.bubble_diameter >= t.min_bubble_mm - 1e-9
        assert {m.id for m in pg.markers} == {4 * pg.index + k for k in range(4)}
        # no overlaps among bubbles
        cs = list(circles(lay, pg))
        for a, b in itertools.combinations(cs, 2):
            assert math.hypot(a[1] - b[1], a[2] - b[2]) >= a[3] + b[3] + 0.5
        # bubbles vs markers (quiet zones), strip, other zones
        for kind, cx, cy, r in cs:
            for m in pg.markers:
                assert not rect_hit_circle(m.quiet, cx, cy, r)
            assert not rect_hit_circle(pg.meta_rect, cx, cy, r)
            assert not rect_hit_circle(pg.name_box, cx, cy, r)
            if kind == "a":
                assert not rect_hit_circle(pg.info_box, cx, cy, r)
        # zones disjoint
        assert not rects_hit(pg.info_box, pg.answer_box)
        for m in pg.markers:
            for z in (pg.info_box, pg.answer_box, pg.meta_rect):
                assert not rects_hit(m.quiet, z)
        for col in pg.id_grid.columns:
            assert not rects_hit(col.writein, pg.name_box)
        # ink inside margins
        mg, W, H = lay.margin, lay.page_w, lay.page_h
        ink = [m.rect for m in pg.markers] + [pg.meta_rect, pg.info_box, pg.answer_box, pg.name_box]
        ink += [c.writein for c in pg.id_grid.columns]
        ink += [b.rect for c in pg.columns for b in c.bands]
        for r in ink:
            assert r[0] >= mg - 1e-6 and r[1] >= mg - 1e-6
            assert r[0] + r[2] <= W - mg + 1e-6 and r[1] + r[3] <= H - mg + 1e-6
        for kind, cx, cy, r in cs:
            assert cx - r >= mg and cx + r <= W - mg and cy - r >= mg and cy + r <= H - mg
        for tx in pg.texts:
            assert mg <= tx.x <= W - mg and mg <= tx.y <= H - mg
        # bubbles inside their column / answer box
        for c in pg.columns:
            assert c.rect[1] + c.rect[3] <= pg.answer_box[1] + pg.answer_box[3] + 1e-6


def test_expected_split():
    assert build_layout(CONFIGS["q100_2p"]).questions_per_page == [50, 50]
    assert build_layout(CONFIGS["q100_3p"]).total_pages == 3
    lay = build_layout(Template(100, 5, 8, pages=3))
    assert lay.pages[2].first_question == 68 and sum(lay.questions_per_page) == 100    # even split 34 / 33 / 33
    # 100 questions on 2 pages, 80 used: page 1 holds Q1-50, page 2 holds Q51-80 (the other slots are not printed)
    lay = build_layout(CONFIGS["used80"])
    assert lay.questions_per_page == [50, 30] and lay.pages[1].first_question == 51
    assert [q.number for c in lay.pages[1].columns for q in c.questions] == list(range(51, 81))


def test_used_questions_are_the_only_ones_printed():
    lay = build_layout(Template(100, 5, 8, pages=2, used=80))
    nums = [q.number for pg in lay.pages for c in pg.columns for q in c.questions]
    assert nums == list(range(1, 81))
    with pytest.raises(ValueError):          # page 2 would be empty: refuse instead of printing a blank page
        build_layout(Template(100, 5, 8, pages=2, used=40))
    with pytest.raises(ValueError):
        build_layout(Template(100, 5, 8, used=101))


def test_auto_split():
    t = Template(45, 5, 8)
    cap = page_capacity(t)
    assert build_layout(t).total_pages == 1
    lay = build_layout(Template(cap + 1, 5, 8))
    assert lay.total_pages == 2
    lay = build_layout(Template(cap * 2 + 1, 5, 8))
    assert lay.total_pages == 3
    # every page carries a full student-info area and same form id
    assert all(len(p.id_grid.columns) == 8 for p in lay.pages)
    for p in lay.pages:
        assert all(c.writein for c in p.id_grid.columns) and p.name_box


def test_limits():
    with pytest.raises(ValueError):
        build_layout(Template(45, 11, 8))
    with pytest.raises(ValueError):
        build_layout(Template(45, 5, 13))
    with pytest.raises(ValueError):
        build_layout(Template(45, 5, 8, choice_labels=["A"]))  # wrong label count
    with pytest.raises(ValueError):
        build_layout(Template(10000, 5, 8))  # > 12 pages
    with pytest.raises(ValueError):
        build_layout(Template(500, 5, 8, pages=1))  # too many for forced pages


def test_form_key():
    a, b = build_layout(Template(45, 5, 8)), build_layout(Template(45, 5, 8, title="other title"))
    assert a.form_key == b.form_key                       # texts are not part of the identity
    assert build_layout(Template(46, 5, 8)).form_key != a.form_key
    assert build_layout(Template(45, 5, 8, used=40)).form_key != a.form_key
    assert build_layout(Template(45, 5, 8, subject_digits=0)).form_key != a.form_key
    assert build_layout(Template(45, 5, 8, choice_labels=list("ABCDE"))).form_key != a.form_key
    assert {p.index for p in build_layout(Template(100, 5, 8, pages=2)).pages} == {0, 1}
    assert len({build_layout(Template(100, 5, 8, pages=2)).header(i).form_key for i in (0, 1)}) == 1   # same on every page
    with pytest.raises(ValueError):          # the header stores a label STYLE only
        build_layout(Template(45, 5, 8, choice_labels=list("VWXYZ")))
    with pytest.raises(ValueError):
        build_layout(Template(45, 5, 8, subject_digits=4))


@pytest.mark.parametrize("name", CONFIGS)
def test_layout_from_header_roundtrip(name):
    """The header alone yields the same geometry: what a scanner without any database rebuilds is what was printed."""
    t = CONFIGS[name]
    L = build_layout(t)
    for i in range(L.total_pages):
        h = L.header(i)
        cells = [[c.bit for c in L.pages[i].meta_cells[r * mk.HEADER_ROW_CELLS:(r + 1) * mk.HEADER_ROW_CELLS]] for r in range(mk.HEADER_ROWS)]
        assert mk.decode_header(cells) == h
    L2 = layout_from_header(L.header(0))
    ref = build_layout(Template(**{**t.__dict__, "title": ""}))
    assert L2.pages == ref.pages and L2.form_key == L.form_key and L2.questions_per_page == L.questions_per_page
    assert header_from_form_key(L.form_key, 0) == L.header(0)
    assert describe(L).startswith(f"{t.used or t.questions}문항") or "문항 양식 중" in describe(L)


def test_many_templates_roundtrip():
    n = 0
    for q, used, ch, idd, sd, pages, lab in itertools.product((10, 100, 250), (None, 7), (2, 5, 10), (1, 12), (0, 3), (None, 2), ("1", "가")):
        cap = lambda t: page_capacity(t)
        try:
            t = Template(q, ch, idd, pages=pages, used=used if used is None else min(used, q), subject_digits=sd,
                         choice_labels=mk.choice_labels(mk.LABEL_STYLES.index(lab), ch))
            L = build_layout(t)
        except ValueError:
            continue                       # impossible combination (no room / empty page): refused, not mis-printed
        assert layout_from_header(L.header(L.total_pages - 1)).form_key == L.form_key
        assert header_from_form_key(L.form_key).questions == q
        n += 1
    assert n > 40


def test_header_cells_are_template_independent():
    r = header_cell_rects()
    assert len(r) == mk.HEADER_ROWS and all(len(row) == mk.HEADER_ROW_CELLS for row in r)
    a, b = build_layout(CONFIGS["q45"]), build_layout(CONFIGS["wide"])
    assert [m.rect for m in a.pages[0].meta_cells] == [m.rect for m in b.pages[0].meta_cells]


def test_legacy_layout_still_loads():
    """out/*.layout.json were written before the redesign: they load as version 0 with a form id and no subject."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "q45.layout.json")
    if not os.path.exists(path):
        pytest.skip("no legacy layout file")
    L = Layout.from_json(open(path, encoding="utf-8").read())
    assert L.version == 0 and L.form_id is not None and L.form_key == f"F{L.form_id:04X}"
    assert L.template.subject_digits == 0 and L.pages[0].subject_grid is None


@pytest.mark.parametrize("name", CONFIGS)
def test_json_roundtrip(name):
    lay = build_layout(CONFIGS[name])
    again = Layout.from_dict(json.loads(json.dumps(lay.to_dict())))
    assert again == lay
    assert Layout.from_json(lay.to_json()) == lay
