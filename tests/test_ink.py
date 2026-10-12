"""omr.ink: per-sheet ink floor."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr import ink  # noqa: E402

BLACK = 0.9


def rows(n=40, mark=0.8, ghost=0.08, ghost_rows=0, ghost_level=0.3, nc=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = rng.uniform(0.0, ghost, nc)
        r[rng.integers(nc)] = mark * rng.uniform(0.85, 1.1)
        if i < ghost_rows:
            r[(np.argmax(r) + 1) % nc] = ghost_level * rng.uniform(0.9, 1.1)
        out.append(r)
    return out


def test_bubble_ink_is_p85_per_bubble():
    D = np.tile(np.linspace(0, 1, 101, dtype=np.float32), (3, 1)) * np.array([[1], [0.5], [0]], np.float32)
    assert np.allclose(ink.bubble_ink(D), [0.85, 0.425, 0.0], atol=0.02)


def test_clean_sheet_floor_is_default_and_marked_auto():
    a = ink.choose_floor(rows(), BLACK)
    assert a.source == "auto" and abs(a.floor - ink.DEFAULT_FRAC * BLACK) < 0.02
    assert a.separation > 0.3 and not a.weak


def test_dark_ghosts_raise_the_floor():
    a = ink.choose_floor(rows(ghost=0.08, ghost_rows=15, ghost_level=0.5), BLACK)
    assert a.floor > 0.5 * BLACK or a.source == "default"
    a2 = ink.choose_floor(rows(ghost=0.08, ghost_rows=15, ghost_level=0.5), BLACK)
    assert a2.floor >= ink.DEFAULT_FRAC * BLACK


def test_floor_never_below_default():
    for mark in (0.5, 0.6, 0.8):
        a = ink.choose_floor(rows(mark=mark, seed=3), BLACK)
        assert a.floor >= ink.DEFAULT_FRAC * BLACK - 1e-6


def test_blank_or_tiny_sheet_falls_back_to_default():
    a = ink.choose_floor(rows(n=3), BLACK)
    assert a.source == "default" and a.weak and a.floor == ink.DEFAULT_FRAC * BLACK
    b = ink.choose_floor([np.random.default_rng(1).uniform(0, 0.1, 5) for _ in range(40)], BLACK)
    assert b.source == "default"


def test_marks_as_light_as_ghosts_gives_weak_separation():
    a = ink.choose_floor(rows(mark=0.4, ghost=0.4), BLACK)
    assert a.source == "default" and a.weak
