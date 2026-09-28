"""CPU verification of the multiharmonic GLS hybrid path (C2, #...).

The GPU multiharmonic Lomb-Scargle reads the NFFT spectra of ``w`` and
``w*(y-ybar)`` off the device and does the small per-frequency 2H x 2H
solve on the host via ``_mh_power_from_spectra``. Here we build those
spectra on the CPU (direct exponential sums, the same convention as
``mhdirect_sums``) and assert the hybrid power matches the tested
``lomb_scargle_direct_sums`` reference for H = 2 and 3 -- no GPU needed.
The remaining GPU-only question (that the real ``ghat_g`` layout matches
this convention) is covered by a pod smoke-test, queued separately.
"""
import numpy as np
import pytest

from cuvarbase.lombscargle import (lomb_scargle_direct_sums,
                                   _mh_power_from_spectra,
                                   mhdirect_sums,
                                   _mh_assemble_from_centered)


def _spectra(coef, t, k0, df, max_index):
    """Adjoint-NFFT-style spectrum: entry k = sum_j coef_j exp(2 pi i
    (k0+k) df t_j), so entry k holds the transform at frequency
    (k0+k)*df (the layout the GPU emits)."""
    kk = np.arange(max_index + 1)
    f = (k0 + kk) * df
    ang = 2 * np.pi * np.outer(f, t)          # (n_k, n_data)
    return (np.cos(ang) + 1j * np.sin(ang)) @ coef


def _data(H, seed=3):
    rng = np.random.RandomState(seed)
    n = 200
    t = np.sort(20.0 * rng.rand(n))
    f0 = 1.3
    y = (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
         + 0.2 * np.cos(2 * np.pi * 3 * f0 * t) + 0.05 * rng.randn(n))
    dy = 0.05 * np.ones(n)
    w = dy ** -2
    w = w / w.sum()
    ybar = float(np.dot(w, y))
    YY = float(np.dot(w, (y - ybar) ** 2))

    df = 1.0 / (5.0 * (t.max() - t.min()))
    k0, nf = 3, 150

    # spectra the GPU would emit: w-spectrum to 2H harmonics, w*(y-ybar) to H
    sw = _spectra(w, t, k0, df, (2 * H - 1) * k0 + 2 * H * (nf - 1))
    syw = _spectra(w * (y - ybar), t, k0, df, (H - 1) * k0 + H * (nf - 1))

    freqs = df * (k0 + np.arange(nf))
    return t, y, w, ybar, YY, k0, nf, freqs, sw, syw


@pytest.mark.parametrize("H", [2, 3])
def test_mh_power_from_spectra_matches_direct_sums(H):
    t, y, w, ybar, YY, k0, nf, freqs, sw, syw = _data(H)

    p_ref = lomb_scargle_direct_sums(t, w * y, w, freqs, YY, nharms=H)
    p_hyb = _mh_power_from_spectra(sw, syw, k0, H, nf, YY)

    corr = np.corrcoef(p_ref, p_hyb)[0, 1]
    assert corr > 0.999, "corr=%.6f for H=%d" % (corr, H)
    # both float64 here -> should agree to ~machine precision
    np.testing.assert_allclose(p_hyb, p_ref, rtol=1e-6, atol=1e-9)


def test_assemble_refactor_matches_mhdirect_sums():
    # _mh_assemble_from_centered fed the same moments mhdirect_sums computes
    # internally must reproduce mhdirect_sums exactly (guards the refactor).
    rng = np.random.RandomState(7)
    n, H, freq = 120, 3, 0.37
    t = np.sort(15.0 * rng.rand(n))
    y = np.sin(2 * np.pi * freq * t) + 0.1 * rng.randn(n)
    w = np.ones(n) / n
    yw = w * y
    YY = float(np.dot(w, (y - np.dot(w, y)) ** 2))

    expected = mhdirect_sums(t, yw, w, freq, YY, nharms=H)

    phase = 2 * np.pi * ((t * freq) % 1.0)
    c = [np.dot(w, np.cos(m * phase)) for m in range(2 * H + 1)]
    s = [np.dot(w, np.sin(m * phase)) for m in range(2 * H + 1)]
    ybar = float(np.sum(yw))
    C = np.asarray(c)[1:H + 1]
    S = np.asarray(s)[1:H + 1]
    yc = np.array([np.dot(yw, np.cos(m * phase)) for m in range(1, H + 1)])
    ys = np.array([np.dot(yw, np.sin(m * phase)) for m in range(1, H + 1)])
    got = _mh_assemble_from_centered(c, s, yc - ybar * C, ys - ybar * S, H)

    for a, b in zip(expected, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-12)


def test_lombscargle_accepts_nharmonics_gt_1():
    # Previously LombScargleAsyncProcess(nharmonics>1) raised
    # NotImplementedError; the GPU multiharmonic path is now implemented.
    from cuvarbase.lombscargle import LombScargleAsyncProcess
    proc = LombScargleAsyncProcess(nharmonics=3)
    assert proc.nharmonics == 3
    with pytest.raises(ValueError):
        LombScargleAsyncProcess(nharmonics=0)


def _mh_power_loop(sw, syw, k0, nharms, nf, YY, reg_kwargs=None):
    """Reference: the per-frequency Python loop ``_mh_power_from_spectra``
    used before it was vectorized, written on the public helpers it
    called (``_mh_assemble_from_centered`` + ``add_regularization`` +
    ``mhgls_from_sums``, all unchanged)."""
    from cuvarbase.lombscargle import add_regularization, mhgls_from_sums
    H = int(nharms)
    i = np.arange(nf)
    cm = np.empty((2 * H + 1, nf), dtype=np.float64)
    sm = np.empty((2 * H + 1, nf), dtype=np.float64)
    cm[0], sm[0] = 1.0, 0.0
    for m in range(1, 2 * H + 1):
        vals = sw[(m - 1) * k0 + m * i]
        cm[m], sm[m] = vals.real, vals.imag
    YC = np.empty((H, nf), dtype=np.float64)
    YS = np.empty((H, nf), dtype=np.float64)
    for h in range(1, H + 1):
        vals = syw[(h - 1) * k0 + h * i]
        YC[h - 1], YS[h - 1] = vals.real, vals.imag
    power = np.empty(nf, dtype=np.float64)
    for j in range(nf):
        sums = _mh_assemble_from_centered(cm[:, j], sm[:, j],
                                          YC[:, j], YS[:, j], H)
        if reg_kwargs:
            sums = add_regularization(sums, **reg_kwargs)
        power[j] = mhgls_from_sums(sums, YY, 0.0)
    return power


@pytest.mark.parametrize("H", [1, 2, 3])
@pytest.mark.parametrize("prior", [None, 0.5, 'per-harmonic'])
def test_stacked_solve_matches_the_per_frequency_loop(H, prior):
    """LS-5: ``_mh_power_from_spectra`` solves the 2H x 2H systems for
    every frequency in one stacked ``np.linalg.solve`` instead of a
    Python loop (60-90 us per frequency before; 84x faster at
    nf = 20,000, H = 2 on the A40 pod host). Same arithmetic, so the
    powers must agree to ~1e-15."""
    t, y, w, ybar, YY, k0, nf, freqs, sw, syw = _data(H)
    if prior == 'per-harmonic':
        prior = list(0.3 + 0.1 * np.arange(H))
    reg = None if prior is None else dict(amplitude_priors=prior)

    ref = _mh_power_loop(sw, syw, k0, H, nf, YY, reg_kwargs=reg)
    got = _mh_power_from_spectra(sw, syw, k0, H, nf, YY, reg_kwargs=reg)

    assert got.shape == ref.shape
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    if H == 1:
        # the H = 1 assembly involves no reordering at all
        assert np.array_equal(got, ref)


@pytest.mark.parametrize("H", [2, 3])
def test_solve_chunking_does_not_change_the_result(H):
    """The stack is solved in chunks of ``_MH_SOLVE_CHUNK`` frequencies
    to bound the (chunk, 2H, 2H) temporary; the chunk size must not
    touch the numbers."""
    import cuvarbase.lombscargle as lsmod
    t, y, w, ybar, YY, k0, nf, freqs, sw, syw = _data(H)
    full = _mh_power_from_spectra(sw, syw, k0, H, nf, YY)
    old = lsmod._MH_SOLVE_CHUNK
    try:
        for chunk in (1, 7, nf - 1, nf, 10 * nf):
            lsmod._MH_SOLVE_CHUNK = chunk
            assert np.array_equal(
                _mh_power_from_spectra(sw, syw, k0, H, nf, YY), full), chunk
    finally:
        lsmod._MH_SOLVE_CHUNK = old


def test_stacked_solve_is_not_a_python_loop(monkeypatch):
    """Behavioural guard for LS-5: one ``np.linalg.solve`` call per
    chunk, not one per frequency."""
    H = 2
    t, y, w, ybar, YY, k0, nf, freqs, sw, syw = _data(H)
    calls = []
    real_solve = np.linalg.solve

    def counting_solve(a, b):
        calls.append(np.shape(a))
        return real_solve(a, b)

    monkeypatch.setattr(np.linalg, 'solve', counting_solve)
    _mh_power_from_spectra(sw, syw, k0, H, nf, YY)
    assert len(calls) == 1, calls
    assert calls[0] == (nf, 2 * H, 2 * H)
