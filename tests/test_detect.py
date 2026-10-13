"""omr.detect: the AprilTag-style second-chance decoder and find_markers on a grey, unevenly printed marker."""
import cv2
import numpy as np

from omr import detect
from omr import markers as mk


def _page(mid=5, cell=12, ink=None):
    """A marker (cell px per module) on white paper; `ink` maps the 6x6 module grid to grey levels."""
    bits = mk.marker_bits(mid)
    lv = np.where(bits, 0, 235).astype(np.float32) if ink is None else ink(bits)
    img = np.kron(lv, np.ones((cell, cell), np.float32))
    img = np.pad(img, 6 * cell, constant_values=235)
    return cv2.GaussianBlur(img, (0, 0), 1.2).astype(np.uint8), 6 * cell


def test_model_decode_any_corner_order_returns_tl_first():
    img, o = _page()
    s = 6 * 12
    truth = np.float64([[o, o], [o + s, o], [o + s, o + s], [o, o + s]])
    for k in range(4):
        for q in (np.roll(truth, k, 0), np.roll(truth, k, 0)[::-1]):
            mid, c = detect.model_decode(img, q)
            assert mid == 5 and np.abs(c - truth).max() < 1e-6


def test_light_inner_row_recovered_without_wrong_ids():
    # grey "black" (border 125) whose first inner row printed much lighter (175) - toner dropout over a module row
    def ink(bits):
        lv = np.where(bits, 125, 230).astype(np.float32)
        lv[1, 1:5] = np.where(bits[1, 1:5], 175, 230)
        return lv
    for mid in (0, 1, 2, 3):
        img, _ = _page(mid, ink=ink)
        found, dup, _ = detect.find_markers(img)
        assert not dup and set(found) == {mid}      # plain ArUco (Otsu bits) misses ids 0, 1, 3, 4 here


def test_plain_texture_is_not_a_marker():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (400, 400)).astype(np.uint8), (0, 0), 3)
    found, dup, _ = detect.find_markers(img)
    assert found == {} and not dup


def test_empty_scene_costs_two_passes():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (600, 800)).astype(np.uint8), (0, 0), 3)
    _, _, info = detect.find_markers(img)
    assert info["detect_passes"] == 2                # no marker-like candidate: no speculative passes


def test_markerlike_counts_damaged_markers_not_texture():
    img, o = _page(5)
    q = np.float64([[o, o], [o + 72, o], [o + 72, o + 72], [o, o + 72]])
    # a marker whose inner bits are scrambled to a word not in the dictionary is still marker-like (border + quiet zone)
    bad = img.copy()
    bad[o + 14:o + 58, o + 14:o + 58] = 0
    assert detect.markerlike(bad, [q], 10.0) == 1
    rng = np.random.default_rng(2)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (400, 400)).astype(np.uint8), (0, 0), 3)
    assert detect.markerlike(tex, [q * 0.5 + 100], 10.0) == 0


def test_off_frame_prediction_needs_missing_markers_outside():
    from omr import layout as lay
    rects = lay.marker_rects(0)
    sc = 3.0
    def quad(c, shift=(0, 0)):
        x, y, w, h = rects[c]
        return np.float64([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]) * sc + shift
    m = detect._Merge()
    first = ['TL', 'TR']
    for c in first:
        m.found[mk.marker_id(0, c)] = quad(c)
    big = (4000, 4000)                                # whole sheet inside the frame: the other two corners must exist
    assert not detect._off_frame(m, big)
    assert detect._off_frame(m, (int(quad('TL')[:, 1].max()) + 60, 4000))   # frame ends under the top row: BL / BR are far below
