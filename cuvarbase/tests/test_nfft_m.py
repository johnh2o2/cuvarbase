"""CPU-side tests for the NFFT filter-radius (m) selection.

These exercise ``NFFTAsyncProcess.estimate_m``/``get_m`` only -- no
GPU work -- so they run on CPU-only machines (under the conftest
stubs) as well as on the pod.
"""
import numpy as np
import pytest

from ..cunfft import NFFTAsyncProcess
from ..memory.nfft_memory import next_fast_len


def _D(sigma):
    return np.pi * (1. - 1. / (2. * sigma - 1.))


class TestEstimateM(object):

    def _proc(self, tol=1e-8, sigma=4):
        return NFFTAsyncProcess(sigma=sigma, autoset_m=True, tol=tol)

    @pytest.mark.parametrize("tol", [1e-4, 1e-8, 1e-12])
    @pytest.mark.parametrize("sigma", [2, 4])
    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
    def test_l1_bound_is_rigorous_and_minimal(self, tol, sigma, scale):
        # The chosen m must satisfy 4 exp(-m D) ||y||_1 <= tol, and be
        # the smallest such integer (no over-padding).
        proc = self._proc(tol=tol, sigma=sigma)
        rand = np.random.RandomState(42)
        y = scale * rand.randn(500)

        m = proc.estimate_m(y=y)
        l1 = np.sum(np.abs(y))
        D = _D(sigma)

        assert 4 * np.exp(-m * D) * l1 <= tol
        if m > 1:
            assert 4 * np.exp(-(m - 1) * D) * l1 > tol

    def test_fallback_heuristic_unchanged(self):
        # Without data, estimate_m must reproduce the historical
        # jakevdp/nfft heuristic exactly.
        proc = self._proc()
        N = 1024
        expected = proc.m_from_C(proc.m_tol / N, proc.sigma)
        assert proc.estimate_m(N) == expected
        assert proc.get_m(N) == expected

    def test_fallback_clamps_m_to_at_least_one(self):
        # Pathological tolerance (m_tol > 4N) used to return m <= 0 on
        # the N-fallback path (the y-path was already clamped), giving
        # a negative Gaussian shape parameter b and garbage gridding.
        proc = self._proc(tol=1e6)
        assert proc.estimate_m(100) >= 1
        assert proc.get_m(100) >= 1

    def test_data_driven_m_scales_with_l1_norm(self):
        proc = self._proc(tol=1e-8, sigma=4)
        N = 1000
        m_small = proc.get_m(N, y=1e-3 * np.ones(N))
        m_heur = proc.get_m(N)
        m_big = proc.get_m(N, y=1e3 * np.ones(N))
        # ||y||_1 = 1 < N < ||y||_1 = 1e6
        assert m_small < m_heur < m_big

    def test_zero_data_returns_minimal_m(self):
        proc = self._proc()
        assert proc.estimate_m(y=np.zeros(16)) == 1

    def test_estimate_m_requires_N_or_y(self):
        proc = self._proc()
        with pytest.raises(ValueError, match="requires N"):
            proc.estimate_m()

    def test_autoset_false_ignores_data(self):
        proc = NFFTAsyncProcess(m=8, autoset_m=False)
        assert proc.get_m() == 8
        assert proc.get_m(100, y=1e6 * np.ones(100)) == 8


class TestNextFastLen(object):
    """``next_fast_len`` (7-smooth padding of the NFFT grids, Sep 2026)
    must return the smallest 2^a 3^b 5^c 7^d >= n."""

    @staticmethod
    def _smooth(x):
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        return x == 1

    def test_matches_brute_force(self):
        for n in list(range(1, 3000)) + [145996, 291996, 2920004]:
            got = next_fast_len(n)
            assert got >= max(n, 1)
            assert self._smooth(got)
            # minimal: nothing 7-smooth in [n, got)
            assert not any(self._smooth(x) for x in range(max(n, 1), got))

    def test_fixed_points_and_edges(self):
        assert next_fast_len(0) == 1
        assert next_fast_len(1) == 1
        assert next_fast_len(7) == 7
        assert next_fast_len(11) == 12
        assert next_fast_len(1024) == 1024
        assert next_fast_len(1025) == 1029     # 3 * 7^3
