import numpy as np
from numpy.testing import assert_allclose

from ..utils import normalize_light_curves, weights


def _fake_lc(n=50, seed=42):
    rand = np.random.RandomState(seed)
    t = np.sort(365 * rand.rand(n))
    y = 12 + 0.1 * np.cos(2 * np.pi * t / 5.0) + 0.1 * rand.randn(n)
    dy = 0.1 * np.ones_like(y)
    return t, y, dy


def test_normalize_subtracts_means():
    t, y, dy = _fake_lc()
    (tn, yn, dyn), = normalize_light_curves([(t, y, dy)])

    assert_allclose(np.mean(tn), 0, atol=1e-9)
    assert_allclose(np.mean(yn), 0, atol=1e-9)
    assert_allclose(tn, t - np.mean(t))
    assert_allclose(yn, y - np.mean(y))
    # columns beyond (t, y) pass through unchanged
    assert_allclose(dyn, dy)


def test_normalize_does_not_mutate_input():
    t, y, dy = _fake_lc()
    t0, y0 = t.copy(), y.copy()
    normalize_light_curves([(t, y, dy)])
    assert_allclose(t, t0)
    assert_allclose(y, y0)


def test_normalize_passes_none_through():
    # Regression test: unweighted CE/LS callers can pass (t, y, None);
    # normalize_light_curves used to crash with AttributeError on
    # None.copy().
    t, y, _ = _fake_lc()
    (tn, yn, dyn), = normalize_light_curves([(t, y, None)])

    assert dyn is None
    assert_allclose(yn, y - np.mean(y))


def test_normalize_legacy_four_tuple():
    # Deprecated PDM format: (t, y, w, freqs) — w and freqs must pass
    # through untouched.
    t, y, dy = _fake_lc()
    w = weights(dy)
    freqs = np.linspace(0.1, 10.0, 100)
    (tn, yn, wn, fn), = normalize_light_curves([(t, y, w, freqs)])

    assert_allclose(wn, w)
    assert_allclose(fn, freqs)
    assert_allclose(yn, y - np.mean(y))


def test_conflict_scatter_perm_is_permutation():
    from cuvarbase.utils import conflict_scatter_perm

    for n in (64, 65, 1000, 20000, 65537):
        p = conflict_scatter_perm(n)
        assert p is not None
        assert len(p) == n
        # a true permutation of 0..n-1
        assert_allclose(np.sort(p), np.arange(n))
        # deterministic
        assert np.array_equal(p, conflict_scatter_perm(n))
        # actually scatters: adjacent outputs come from far-apart inputs
        assert np.min(np.abs(np.diff(p.astype(np.int64)))) > n // 4


def test_conflict_scatter_perm_small_n_passthrough():
    from cuvarbase.utils import conflict_scatter_perm

    for n in (0, 1, 2, 32, 63):
        assert conflict_scatter_perm(n) is None
