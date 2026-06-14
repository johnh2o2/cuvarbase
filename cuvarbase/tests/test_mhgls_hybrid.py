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
