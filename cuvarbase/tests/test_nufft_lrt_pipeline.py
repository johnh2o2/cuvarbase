"""CPU verification of the NUFFT-LRT host pipeline (no GPU).

``compute_nufft`` is mocked with a direct adjoint DFT -- the exact math
the GPU NFFT approximates, at the same convention (modes k=0..nf-1,
frequency k/(max(t)-min(t)); the transform's time reference is a common
per-mode phase that cancels in every whitened inner product) -- so the
host pipeline (epoch subtraction, PSD, weights, matched filter, return
shapes, input validation) runs on CPU:

* the matched filter is sensitive to data across the WHOLE baseline
  (perturbing a late, well-separated season changes the result -- the
  defect that got the module cut is gone),
* the weights span all nf bins (the rfft one-sided packing is gone), and
* absolute-time input is handled exactly (float64 epoch subtraction).

The GPU NFFT itself (and its accuracy vs this exact reference) is
checked in ``test_nufft_lrt.py`` on a GPU.
"""
import numpy as np
import pytest

from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess

pytestmark = pytest.mark.filterwarnings(
    "ignore:cuvarbase.nufft_lrt is EXPERIMENTAL")

BJD_OFFSET = 2457000.5


def _adjoint_dft(t, y, nf):
    """Exact adjoint NFFT: ghat[k] = sum_j y_j exp(2 pi i k t_j/(tmax -
    tmin)), k = 0..nf-1 (chunked over k to bound memory)."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = t / (t.max() - t.min())
    out = np.empty(nf, dtype=np.complex128)
    for a in range(0, nf, 512):
        k = np.arange(a, min(nf, a + 512))
        out[a:a + len(k)] = np.exp(2j * np.pi * np.outer(k, x)) @ y
    return out


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


def test_absolute_time_input_is_exact(monkeypatch):
    # run() subtracts floor(min t) in float64 before anything else, so a
    # BJD-scale offset (with explicit epochs shifted identically) gives
    # the same statistic.
    proc = _mock_proc(monkeypatch)
    t, y, period = _two_season_lc()
    periods = np.array([2.0, period, 3.1])
    durations = np.array([0.2])
    epochs = np.linspace(0.0, 2.0, 4)
    base = proc.run(t, y, periods, durations=durations, epochs=epochs)
    shifted = proc.run(t + BJD_OFFSET, y, periods, durations=durations,
                       epochs=epochs + BJD_OFFSET)
    np.testing.assert_allclose(shifted, base, rtol=1e-6, atol=1e-9)


def test_dy_is_ignored_with_a_warning(monkeypatch):
    proc = _mock_proc(monkeypatch)
    t, y, period = _two_season_lc()
    kw = dict(durations=np.array([0.2]), epochs=np.array([0.0]))
    ref = proc.run(t, y, np.array([period]), **kw)
    with pytest.warns(UserWarning, match="dy"):
        got = proc.run(t, y, np.array([period]), dy=np.full(len(t), 0.01),
                       **kw)
    np.testing.assert_array_equal(got, ref)


def test_input_validation(monkeypatch):
    proc = _mock_proc(monkeypatch)
    t, y, period = _two_season_lc()
    with pytest.raises(ValueError, match="same length"):
        proc.run(t[:-1], y, np.array([period]))
    with pytest.raises(ValueError, match="finite"):
        proc.run(t, np.where(np.arange(len(y)) == 3, np.nan, y),
                 np.array([period]))
    with pytest.raises(ValueError, match="periods"):
        proc.run(t, y, np.array([-1.0]))
    with pytest.raises(ValueError, match="durations"):
        proc.run(t, y, np.array([period]), durations=np.array([0.0]))
    with pytest.raises(ValueError, match="epochs"):
        proc.run(t, y, np.array([period]), epochs=np.array([np.nan]))
