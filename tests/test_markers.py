import random

import pytest

from omr import markers as mk


def test_meta_roundtrip():
    rnd = random.Random(1)
    for _ in range(500):
        tot = rnd.randint(1, 12)
        m = mk.Meta(rnd.randrange(65536), rnd.randrange(tot), tot)
        bits = mk.encode_meta(*[m.form_id, m.page, m.total_pages])
        assert len(bits) == mk.META_CELLS
        assert mk.decode_meta(bits) == m


def test_single_bit_flips_detected():
    rnd = random.Random(2)
    missed = total = 0
    for _ in range(100):
        tot = rnd.randint(1, 12)
        m = mk.Meta(rnd.randrange(65536), rnd.randrange(tot), tot)
        bits = mk.encode_meta(m.form_id, m.page, m.total_pages)
        for i in range(len(bits)):
            b = list(bits)
            b[i] = not b[i]
            total += 1
            if mk.decode_meta(b) is not None:
                missed += 1
    assert missed == 0  # CRC-8 + guards catch every single-bit flip


def test_bad_inputs():
    assert mk.decode_meta([True] * 10) is None
    for args in [(70000, 0, 1), (1, 0, 0), (1, 3, 3), (1, 0, 13)]:
        try:
            mk.encode_meta(*args)
            assert False
        except ValueError:
            pass


def test_marker_ids():
    ids = {mk.marker_id(p, c) for p in range(12) for c in mk.CORNERS}
    assert ids == set(range(48))
    assert mk.decode_marker_id(mk.marker_id(5, "BR")) == (5, "BR")
    assert mk.marker_bits(3).shape == (6, 6)
    assert mk.marker_image(3).ndim == 2


# ---------------------------------------------------------------- self-describing header
def _headers(rnd, n):
    for _ in range(n):
        q = rnd.randint(1, 1023)
        yield mk.Header(q, rnd.randint(1, q), rnd.randint(2, 10), rnd.randint(1, 12), rnd.randint(0, 3),
                        (pg := rnd.randint(1, mk.MAX_PAGES)), rnd.randrange(pg), rnd.randrange(len(mk.LABEL_STYLES)),
                        rnd.randrange(8))


def test_header_roundtrip():
    rnd = random.Random(3)
    for h in _headers(rnd, 400):
        rows = mk.encode_header(h)
        assert len(rows) == mk.HEADER_ROWS and all(len(r) == mk.HEADER_ROW_CELLS for r in rows)
        assert mk.decode_header(rows) == h


def test_header_flips_never_decode_to_something_else():
    """CRC-16 + guards: every single and double cell flip is rejected (or, for a flip in a guard, refused)."""
    rnd = random.Random(4)
    for h in _headers(rnd, 6):
        rows = mk.encode_header(h)
        flat = [b for r in rows for b in r]
        n = len(flat)
        def dec(bits):
            return mk.decode_header([bits[:mk.HEADER_ROW_CELLS], bits[mk.HEADER_ROW_CELLS:]])
        for i in range(n):
            b = list(flat)
            b[i] = not b[i]
            assert dec(b) is None, i
        for i in range(n):
            for j in range(i + 1, n):
                b = list(flat)
                b[i], b[j] = not b[i], not b[j]
                assert dec(b) is None, (i, j)


def test_header_field_ranges():
    ok = mk.Header(100, 80, 5, 8, 2, 2, 1)
    assert mk.decode_header(mk.encode_header(ok)) == ok
    bad = [dict(questions=1024, used=1), dict(used=0), dict(used=101), dict(choices=1), dict(choices=11), dict(id_digits=0),
           dict(id_digits=13), dict(subject_digits=4), dict(subject_digits=-1), dict(pages=0), dict(pages=mk.MAX_PAGES + 1),
           dict(page=2), dict(labels=len(mk.LABEL_STYLES)), dict(bubble=8), dict(version=0)]
    for kw in bad:
        args = dict(questions=100, used=80, choices=5, id_digits=8, subject_digits=2, pages=2, page=1, labels=0, bubble=0)
        args.update(kw)
        with pytest.raises(ValueError):
            mk.encode_header(mk.Header(**args))
    for hi in (mk.Header(1023, 1023, 10, 12, 3, mk.MAX_PAGES, mk.MAX_PAGES - 1, 3, 7), mk.Header(1, 1, 2, 1, 0, 1, 0)):
        assert mk.decode_header(mk.encode_header(hi)) == hi


def test_header_is_not_the_legacy_strip_and_labels():
    rows = mk.encode_header(mk.Header(45, 45, 5, 8, 2, 1, 0))
    assert mk.decode_meta(rows[0]) is None and mk.decode_header([rows[0]]) is None
    assert mk.choice_labels(0, 3) == ["1", "2", "3"] and mk.choice_labels(1, 4) == list("ABCD")
    assert mk.choice_labels(3, 5) == list("가나다라마")
    for i in range(len(mk.LABEL_STYLES)):
        assert mk.label_style(mk.choice_labels(i, 5)) == i
    assert mk.label_style(["x", "y"]) is None
