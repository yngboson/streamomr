"""omr.fastops replaces numpy calls whose Python overhead dominates on small arrays: it must return exactly the same bits."""
import numpy as np
import pytest

from omr import fastops


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint8])
@pytest.mark.parametrize("q", [25, 50, 75, 85, 97])
def test_pct1_matches_numpy(dtype, q):
    rng = np.random.default_rng(q)
    for n in (1, 2, 3, 7, 40, 301):
        a = (rng.random((n,)) * (200 if dtype == np.uint8 else 1)).astype(dtype)
        assert fastops.pct1(a, q) == np.percentile(a, q)
        b = (rng.random((n, 3)) * (200 if dtype == np.uint8 else 1)).astype(dtype)
        assert fastops.pct1(b, q) == np.percentile(b, q)


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
@pytest.mark.parametrize("q", [35, 85])
def test_pct_axis0_matches_numpy(dtype, q):
    rng = np.random.default_rng(1)
    for n in (2, 9, 111):
        a = (rng.random((n, 57)) * (255 if dtype == np.uint8 else 1)).astype(dtype)
        assert np.array_equal(fastops.pct_axis0(a, q), np.percentile(a, q, axis=0))
