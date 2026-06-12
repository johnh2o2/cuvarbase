"""
Tests for the Keplerian/uniform frequency grid utilities and the
per-frequency q-bound wiring into the batch BLS API.
"""
import numpy as np
import pytest

from ..bls_frequencies import (_q_transit, keplerian_freq_grid,
                               uniform_freq_grid, freq_grid_stats)


class TestKeplerianFreqGrid:

    def test_grid_covers_range(self):
        freqs = keplerian_freq_grid(1.0, 10.0, baseline=365.0)
        assert freqs[0] == pytest.approx(0.1, rel=1e-5)
        assert freqs[-1] >= 1.0 * 0.999
        assert np.all(np.diff(freqs) > 0)

    def test_fewer_freqs_than_uniform(self):
        kep = keplerian_freq_grid(1.0, 50.0, baseline=1000.0)
        uni = uniform_freq_grid(1.0, 50.0, baseline=1000.0)
        assert len(kep) < len(uni)

    def test_return_qvals(self):
        freqs, qvals = keplerian_freq_grid(1.0, 10.0, baseline=365.0,
                                           return_qvals=True)
        assert len(qvals) == len(freqs)
        assert qvals.dtype == np.float32
        assert np.all(qvals > 0)
        assert np.all(qvals <= 0.5)
        # Keplerian q grows with frequency (shorter periods -> larger
        # duration fraction)
        assert np.all(np.diff(qvals) >= 0)
        # consistent with the q model used to build the grid
        np.testing.assert_allclose(
            qvals, _q_transit(freqs.astype(np.float64)), rtol=1e-5)

    def test_default_return_unchanged(self):
        out = keplerian_freq_grid(1.0, 10.0, baseline=365.0)
        assert isinstance(out, np.ndarray)

    def test_grid_stats(self):
        freqs = keplerian_freq_grid(1.0, 50.0, baseline=1000.0)
        stats = freq_grid_stats(freqs, 1000.0)
        assert stats['nfreq'] == len(freqs)
        assert stats['reduction_factor'] > 1


class TestBatchPerFrequencyQBounds:
    """eebls_gpu_batch accepts per-frequency qmin/qmax arrays (the
    batch kernel reads per-frequency bin counts); combined with
    keplerian_freq_grid(return_qvals=True) this enables
    duration-constrained Keplerian searches in batch mode."""

    def test_batch_keplerian_q_bounds(self):
        # GPU only: skipped on CPU machines via the conftest stub.
        from ..bls import eebls_gpu_batch

        rand = np.random.RandomState(8)
        freq_inj, q_inj, delta = 0.5, 0.03, 0.05
        ndata, baseline = 300, 365.0

        lcs = []
        for seed in (1, 2):
            r = np.random.RandomState(seed)
            t = np.sort(baseline * r.rand(ndata))
            phase = (t * freq_inj) % 1.0
            y = 12.0 - delta * (phase < q_inj)
            y += 0.01 * r.randn(ndata)
            lcs.append((t, y, 0.01 * np.ones(ndata)))

        freqs, qvals = keplerian_freq_grid(1.5, 3.0, baseline,
                                           return_qvals=True)
        results = eebls_gpu_batch(lcs, freqs,
                                  qmin=0.5 * qvals, qmax=2.0 * qvals)
        assert len(results) == 2
        for power in results:
            best = freqs[int(np.argmax(power))]
            assert abs(best - freq_inj) / freq_inj < 0.02
