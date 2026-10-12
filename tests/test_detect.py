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
