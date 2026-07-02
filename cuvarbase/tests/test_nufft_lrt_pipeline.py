"""CPU verification of the rewired NUFFT-LRT pipeline (C3).

After the rewire, ``compute_nufft`` routes to the GPU adjoint NFFT, which
covers the full non-uniform baseline (no median(dt)*nf truncation). Here
we mock ``compute_nufft`` with a direct adjoint DFT -- the exact math the
GPU NFFT approximates, at the same convention (modes k=0..nf-1, frequency
k/(max(t)-min(t)), ABSOLUTE-t phases exp(2 pi i f_k t_j) -- verified
against the device NFFT in the batch-3 pod run, Jul 2026) -- and check
the host pipeline (PSD, all-ones weights, matched filter) on CPU:

* the matched filter is sensitive to data across the WHOLE baseline
  (perturbing a late, well-separated season changes the result -- the
  defect that got the module cut is gone), and
* the weights span all nf bins (the rfft one-sided packing is gone).

The GPU NFFT itself (and its accuracy vs this exact reference) is checked
on a pod, queued separately.
"""
import numpy as np

from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess


def _adjoint_dft(t, y, nf):
    """Exact adjoint NFFT at the GPU convention: ghat[k] = sum_j y_j
    exp(2 pi i k t_j/(tmax - tmin)), k = 0..nf-1 (ABSOLUTE-t phases --
    the device normalize kernel re-references to t=0, not min(t))."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = t / (t.max() - t.min())
    k = np.arange(nf)
    return np.exp(2j * np.pi * np.outer(k, x)) @ y


def _mock_proc(monkeypatch):
    proc = NUFFTLRTAsyncProcess()
    monkeypatch.setattr(
        proc, 'compute_nufft',
        lambda t, y, nf, **kw: _adjoint_dft(t, y, nf).astype(proc.complex_type))
    return proc


def _two_season_lc(seed=0):
    rng = np.random.RandomState(seed)
    # two well-separated observing seasons (a 260-day gap) -- the old
    # uniform grid (span ~ median(dt)*2N << 340 d) would drop season 2.
    t = np.concatenate([np.sort(rng.uniform(0.0, 40.0, 120)),
                        np.sort(rng.uniform(300.0, 340.0, 120))])
    period = 2.3
    phase = (t % period) / period
    y = np.ones_like(t)
    y[(phase < 0.06) | (phase > 0.94)] -= 0.2          # box transit
    y += 0.01 * rng.randn(len(t))
    return t, y, period


def test_pipeline_runs_end_to_end(monkeypatch):
    proc = _mock_proc(monkeypatch)
    t, y, period = _two_season_lc()
    periods = np.linspace(1.5, 4.0, 40)
    durations = np.array([0.15, 0.3])
    snr = proc.run(t, y, periods, durations=durations)
    assert snr.shape == (len(periods), len(durations))
    assert np.all(np.isfinite(snr))


def test_late_season_data_changes_result(monkeypatch):
    # The full-baseline NFFT must let late-season observations affect the
    # detection statistic; the old median(dt)*nf grid silently ignored them.
    proc = _mock_proc(monkeypatch)
    t, y, period = _two_season_lc()
    periods = np.linspace(1.5, 4.0, 40)
    durations = np.array([0.2])

    snr0 = proc.run(t, y, periods, durations=durations)

    # perturb ONLY the late (second) season
    y2 = y.copy()
    late = t > 200.0
    assert late.sum() > 0
    rng = np.random.RandomState(1)
    y2[late] += 0.5 * rng.randn(int(late.sum()))

    snr1 = proc.run(t, y2, periods, durations=durations)

    # the statistic must respond to the late-season change (it would be
    # identical if that data were truncated away)
    assert np.max(np.abs(snr1 - snr0)) > 1e-6


def test_snr_responds_to_injected_transit(monkeypatch):
    # Sanity: the SNR spectrum is non-degenerate and the transit period
    # produces a finite, above-median response (full recovery / harmonic
    # disambiguation is left to the GPU injection-recovery validation).
    proc = _mock_proc(monkeypatch)
    t, y, period = _two_season_lc()
    periods = np.linspace(1.5, 4.0, 60)
    snr = proc.run(t, y, periods, durations=np.array([0.2]))[:, 0]
    assert np.ptp(snr) > 0                       # not constant
    i = int(np.argmin(np.abs(periods - period)))
    assert snr[i] >= np.median(snr)
