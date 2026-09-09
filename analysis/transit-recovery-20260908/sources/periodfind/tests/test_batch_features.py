"""Tests for batch feature extraction modules: DmDt, BasicStats, RemoveHighCadence.

These test the Python-level wrappers for the Rust implementations.

Run with: pytest tests/test_batch_features.py -v
"""

import numpy as np

import periodfind
from periodfind.cpu import BasicStats, DmDt, RemoveHighCadence

# ---------------------------------------------------------------------------
# RemoveHighCadence
# ---------------------------------------------------------------------------


class TestRemoveHighCadence:
    """Tests for the RemoveHighCadence wrapper."""

    def test_basic_filtering(self):
        """Points within cadence are removed."""
        # cadence = 30 min = 30/1440 days ≈ 0.02083 days
        t = np.array([0.0, 0.001, 0.002, 0.025, 0.05], dtype=np.float32)
        m = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32)
        e = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        rhc = RemoveHighCadence(cadence_minutes=30.0)
        result = rhc.calc([t], [m], [e])

        assert len(result) == 1
        t_out, m_out, e_out = result[0]
        # keep indices 0, 3, 4 (gaps >= 0.02083 days)
        assert len(t_out) == 3
        assert t_out.dtype == np.float32

    def test_empty_input(self):
        """Empty input returns empty output."""
        t = np.array([], dtype=np.float32)
        m = np.array([], dtype=np.float32)
        e = np.array([], dtype=np.float32)

        rhc = RemoveHighCadence(cadence_minutes=30.0)
        result = rhc.calc([t], [m], [e])

        assert len(result) == 1
        assert len(result[0][0]) == 0

    def test_single_point(self):
        """Single point is preserved."""
        rhc = RemoveHighCadence(cadence_minutes=30.0)
        result = rhc.calc(
            [np.array([1.0], dtype=np.float32)],
            [np.array([10.0], dtype=np.float32)],
            [np.array([0.1], dtype=np.float32)],
        )
        assert len(result[0][0]) == 1

    def test_all_beyond_cadence(self):
        """Well-separated points are all kept."""
        t = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        m = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
        e = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        rhc = RemoveHighCadence(cadence_minutes=30.0)
        result = rhc.calc([t], [m], [e])
        assert len(result[0][0]) == 4

    def test_batch_processing(self):
        """Multiple curves processed correctly."""
        t1 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        m1 = np.array([10.0, 11.0, 12.0], dtype=np.float32)
        e1 = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        t2 = np.array([0.0, 0.001, 5.0], dtype=np.float32)
        m2 = np.array([20.0, 21.0, 22.0], dtype=np.float32)
        e2 = np.array([0.2, 0.2, 0.2], dtype=np.float32)

        rhc = RemoveHighCadence(cadence_minutes=30.0)
        result = rhc.calc([t1, t2], [m1, m2], [e1, e2])
        assert len(result) == 2
        assert len(result[0][0]) == 3  # all kept
        assert len(result[1][0]) == 2  # first and last kept

    def test_factory_function(self):
        """periodfind.remove_high_cadence() works end-to-end."""
        t = [np.array([0.0, 1.0, 2.0], dtype=np.float32)]
        m = [np.array([10.0, 11.0, 12.0], dtype=np.float32)]
        e = [np.array([0.1, 0.1, 0.1], dtype=np.float32)]

        result = periodfind.remove_high_cadence(t, m, e, cadence_minutes=30.0)
        assert len(result) == 1
        assert len(result[0][0]) == 3


# ---------------------------------------------------------------------------
# DmDt
# ---------------------------------------------------------------------------


class TestDmDt:
    """Tests for the DmDt wrapper."""

    @staticmethod
    def _make_edges():
        dt_edges = np.array(
            [
                0.0,
                1.0 / 145,
                2.0 / 145,
                3.0 / 145,
                4.0 / 145,
                5.0 / 145,
                6.0 / 145,
                1.5 / 23.2,
                2.0 / 23.2,
                3.0 / 23.2,
                1.0 / 3.5,
                2.0 / 3.5,
                3.0 / 3.5,
                4.0 / 3.5,
                5.0 / 3.5,
                7.0,
                10.0,
                20.0,
                30.0,
                60.0,
                90.0,
                120.0,
                240.0,
                600.0,
                960.0,
                2000.0,
            ],
            dtype=np.float32,
        )
        dm_edges = np.array(
            [
                -8.0,
                -3.2,
                -2.4,
                -2.0,
                -1.6,
                -1.2,
                -0.8,
                -0.6,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                -0.05,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.6,
                0.8,
                1.2,
                1.6,
                2.0,
                2.4,
                3.2,
                8.0,
            ],
            dtype=np.float32,
        )
        return dt_edges, dm_edges

    def test_output_shape(self):
        """Returns correct 3D shape."""
        rng = np.random.default_rng(42)
        n_curves = 3
        times, mags = [], []
        for _ in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 80)).astype(np.float32)
            m = (18.0 + rng.normal(0, 0.5, 80)).astype(np.float32)
            times.append(t)
            mags.append(m)

        dt_edges, dm_edges = self._make_edges()
        dd = DmDt()
        result = dd.calc(times, mags, dt_edges, dm_edges)

        n_dt_bins = len(dt_edges) - 1
        n_dm_bins = len(dm_edges) - 1
        assert result.shape == (n_curves, n_dm_bins, n_dt_bins)
        assert result.dtype == np.float32

    def test_single_point_curve(self):
        """Single-point curve returns all zeros."""
        dt_edges, dm_edges = self._make_edges()
        dd = DmDt()
        result = dd.calc(
            [np.array([1.0], dtype=np.float32)],
            [np.array([10.0], dtype=np.float32)],
            dt_edges,
            dm_edges,
        )
        assert np.all(result == 0.0)

    def test_l2_normalization(self):
        """Each curve's histogram is L2-normalised (or zero)."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 50)).astype(np.float32)
        m = (18.0 + rng.normal(0, 0.5, 50)).astype(np.float32)

        dt_edges, dm_edges = self._make_edges()
        dd = DmDt()
        result = dd.calc([t], [m], dt_edges, dm_edges)

        norm = np.linalg.norm(result[0])
        if norm > 0:
            assert abs(norm - 1.0) < 1e-5, f"L2 norm = {norm}"

    def test_factory_function(self):
        """periodfind.DmDt() factory works."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 50)).astype(np.float32)
        m = (18.0 + rng.normal(0, 0.5, 50)).astype(np.float32)

        dt_edges, dm_edges = self._make_edges()
        dd = periodfind.DmDt()
        result = dd.calc([t], [m], dt_edges, dm_edges)
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# BasicStats
# ---------------------------------------------------------------------------


class TestBasicStats:
    """Tests for the BasicStats wrapper."""

    def test_output_shape(self):
        """Returns (n_curves, 22) array."""
        rng = np.random.default_rng(42)
        n_curves = 3
        times, mags, errs = [], [], []
        for _ in range(n_curves):
            t = np.sort(rng.uniform(0, 100, 50)).astype(np.float32)
            m = (18.0 + rng.normal(0, 0.1, 50)).astype(np.float32)
            e = np.full(50, 0.1, dtype=np.float32)
            times.append(t)
            mags.append(m)
            errs.append(e)

        bs = BasicStats()
        result = bs.calc(times, mags, errs)
        assert result.shape == (n_curves, 22)
        assert result.dtype == np.float32

    def test_n_count(self):
        """First column is the number of points."""
        rng = np.random.default_rng(42)
        times, mags, errs = [], [], []
        for n in [20, 50, 100]:
            t = np.sort(rng.uniform(0, 100, n)).astype(np.float32)
            m = (18.0 + rng.normal(0, 0.1, n)).astype(np.float32)
            e = np.full(n, 0.1, dtype=np.float32)
            times.append(t)
            mags.append(m)
            errs.append(e)

        bs = BasicStats()
        result = bs.calc(times, mags, errs)
        assert result[0, 0] == 20.0
        assert result[1, 0] == 50.0
        assert result[2, 0] == 100.0

    def test_constant_signal(self):
        """Constant magnitude yields ~0 scatter stats."""
        n = 50
        t = (np.arange(n, dtype=np.float64) * 0.5).astype(np.float32)
        m = np.full(n, 17.0, dtype=np.float32)
        e = np.full(n, 0.1, dtype=np.float32)

        bs = BasicStats()
        result = bs.calc([t], [m], [e])
        r = result[0]

        assert r[0] == n  # N
        assert abs(r[1] - 17.0) < 0.01  # median
        assert abs(r[2] - 17.0) < 0.01  # wmean
        assert abs(r[3]) < 0.01  # chi2red
        assert abs(r[5]) < 0.01  # wstd

    def test_too_few_points(self):
        """Fewer than 4 points returns NaN."""
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m = np.array([10.0, 11.0, 12.0], dtype=np.float32)
        e = np.array([0.1, 0.1, 0.1], dtype=np.float32)

        bs = BasicStats()
        result = bs.calc([t], [m], [e])
        assert np.all(np.isnan(result[0]))

    def test_finite_for_normal_input(self):
        """All 22 stats are finite for well-behaved input."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 200)).astype(np.float32)
        m = (18.0 + rng.normal(0, 0.3, 200)).astype(np.float32)
        e = np.full(200, 0.1, dtype=np.float32)

        bs = BasicStats()
        result = bs.calc([t], [m], [e])
        for i in range(22):
            assert np.isfinite(result[0, i]), f"stat[{i}] is not finite: {result[0, i]}"

    def test_factory_function(self):
        """periodfind.BasicStats() factory works."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0, 100, 50)).astype(np.float32)
        m = (18.0 + rng.normal(0, 0.1, 50)).astype(np.float32)
        e = np.full(50, 0.1, dtype=np.float32)

        bs = periodfind.BasicStats()
        result = bs.calc([t], [m], [e])
        assert result.shape == (1, 22)
