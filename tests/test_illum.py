"""omr.illum: darkness of bubbles under hard shadows, gradients, glare; flags where it cannot know."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr import illum  # noqa: E402
from omr import recognize as R  # noqa: E402

PPM = 6.0
DIA = 5.25


def sheet(shadow_x=None, depth=0.5, gradient=0.0, filled=(), glare=0.0, seed=0, glare_at=None):
    """A 60 x 40 mm patch: 8 x 4 bubbles on a 7.5 mm pitch, optional vertical hard shadow edge at x = shadow_x mm."""
    rng = np.random.default_rng(seed)
    W, H = int(60 * PPM), int(40 * PPM)
    img = np.full((H, W), 230.0, np.float32)
    cents = [(8 + 7.5 * i, 6 + 7.5 * j) for j in range(4) for i in range(7)]
    for k, (cx, cy) in enumerate(cents):
        cv2.circle(img, (int(round(cx * PPM)), int(round(cy * PPM))), int(round(DIA / 2 * PPM)), 90.0, max(int(0.25 * PPM), 1))
        if k in filled:
            cv2.circle(img, (int(round(cx * PPM)), int(round(cy * PPM))), int(round(DIA / 2 * PPM)), 60.0, -1)
    xs = np.arange(W)[None, :] / PPM
    light = np.ones((H, W), np.float32) * (1 - gradient * xs / 60.0)
    if shadow_x is not None:
        light *= np.where(xs >= shadow_x, 1 - depth, 1.0)
    if glare_at is not None:                           # a local specular highlight (gaussian, 10 mm)
        ys = np.arange(H)[:, None] / PPM
        glare = glare * np.exp(-((xs - glare_at[0]) ** 2 + (ys - glare_at[1]) ** 2) / (2 * 10.0 ** 2))
    img = img * light + 255 * glare
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.float32), np.array(cents)


def darkness(img, cents, method):
    return illum.bubble_darkness(img, cents, DIA, PPM, method)


def shipped_like_centre_darkness(img, c):
    """What the pre-illum estimator saw at each bubble centre: the page value over a 10 mm max-filter window (it reaches
    across a shadow edge).  Kept to show the failure illum fixes."""
    bg = cv2.dilate(img, np.ones((int(10 * PPM) | 1,) * 2, np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), 3 * PPM)
    ix, iy = np.round(c[:, 0] * PPM).astype(int), np.round(c[:, 1] * PPM).astype(int)
    return 1.0 - img[iy, ix] / bg[iy, ix]


def test_contract_is_one_row_per_bubble_over_the_scored_disc():
    img, c = sheet()
    D, rel = darkness(img, c, "E")
    dy, _, _ = R._disc(illum.INNER_DISC_FRAC * DIA / 2 * PPM)
    assert D.shape == (len(c), len(dy)) and rel.shape == (len(c),) and rel.dtype == bool


def test_clean_sheet_all_reliable_and_marks_read():
    img, c = sheet(filled=(3, 10))
    D, rel = darkness(img, c, "E")
    assert rel.all()
    m = D.mean(1)
    assert m[3] > 0.6 and m[10] > 0.6
    assert np.delete(m, [3, 10]).max() < 0.1


def test_hard_shadow_edge_does_not_make_empty_bubbles_dark():
    img, c = sheet(shadow_x=29.0, depth=0.55, filled=(2, 12))
    D, rel = darkness(img, c, "E")
    m = D.mean(1)
    ok = rel.copy()
    assert ok.sum() >= len(c) - 5                       # only bubbles the edge touches may be flagged
    all_empty = np.array([k not in (2, 12) for k in range(len(c))])
    empty = all_empty & ok
    assert m[empty].max() < 0.12, m[empty].max()        # shadowed empty bubbles stay empty
    assert m[2] > 0.5 and m[12] > 0.5                   # a mark in the shadow and one in the light are both seen
    base = shipped_like_centre_darkness(img, c)
    assert base[all_empty & (c[:, 0] > 29.0)].max() > 0.25  # the max-filter estimator does fail here (this is the bug)


def test_edge_through_a_bubble_is_flagged_not_guessed():
    img, c = sheet(shadow_x=8.0, depth=0.6)             # the edge passes through the centre of the bubbles at x = 8
    D, rel = darkness(img, c, "E")
    through = np.isclose(c[:, 0], 8.0)
    assert (~rel[through]).all()
    assert np.all(D[~rel] == 0)                         # no fake darkness leaks into the ink statistics


def test_smooth_gradient_is_undone():
    img, c = sheet(gradient=0.5, filled=(5,))
    D, rel = darkness(img, c, "E")
    assert rel.all()
    m = D.mean(1)
    assert np.delete(m, [5]).max() < 0.08 and m[5] > 0.55


def test_neighbour_overflow_does_not_flag_the_bubble():
    img, c = sheet(filled=(3,))
    # a mark that spills over its ring towards the right neighbour
    cv2.circle(img, (int(round((c[3][0] + 0.6) * PPM)), int(round(c[3][1] * PPM))), int(round(3.2 * PPM)), 60.0, -1)
    D, rel = darkness(img, c, "E")
    assert rel[2] and rel[4]
    assert D[2].mean() < 0.1 and D[4].mean() < 0.1


def test_strong_glare_flags_instead_of_reading_blank():
    img, c = sheet(glare=1.2, filled=(3,), glare_at=(30.5, 6.0))              # clipped: the mark cannot be seen at all
    img = np.clip(img, 0, 255)
    D, rel = darkness(img, c, "E")
    assert not rel[3]                                   # a mark that vanished must not read "empty"
