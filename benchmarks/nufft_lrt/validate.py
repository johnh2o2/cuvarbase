"""NUFFT-LRT injection-recovery validation vs BLS (and TLS).

The question this answers (audit follow-up, July 2026; re-run after the
Sep-2026 correctness fixes as Phase 4 of the 1.0 release plan): does the
PSD-whitened matched filter actually buy detection performance in
correlated noise, what does it cost in white noise -- i.e. when is it
the right tool -- and does the PUBLIC DEFAULT PATH (``epochs=None`` on
absolute, BJD-scale times) perform like the explicit-epoch search?

Protocol (per noise configuration):

1. NULL RUNS: generate signal-free lightcurves, run each method's search
   over the identical period range, record the maximum statistic. The
   95th percentile of the null maxima is that method's detection
   threshold at a fixed 5% per-search false-alarm rate. This
   self-calibration is what makes methods with different statistics
   (LRT SNR, BLS power, TLS SDE) comparable -- and it is exactly where
   red noise hurts BLS/TLS: their null maxima inflate, raising the bar.
2. INJECTION RUNS: inject box transits (random epoch, fixed
   period/duration, swept depth) into fresh noise; a detection requires
   the statistic to exceed the null threshold AND the best period to
   land within 1% of the truth or its 2:1 aliases.
3. Completeness(depth) per method per noise config + the LRT statistic's
   null calibration (mean/std on white noise at ONE fixed template).
   This is a calibration CONSTANT of the configuration, not a pass/fail
   check: the statistic is NOT N(0,1) by design (the NFFT modes of
   irregular sampling are not orthogonal, so the frequency-diagonal
   whitened correlation is over-dispersed even with the true PSD; null
   std ~1.8-2.7 for this harness's ground sampling at nf = 2n -- see the
   cuvarbase.nufft_lrt module docstring; the pre-fix campaign measured
   1.81 with the sigma = 2 NFFT). Expect a mean near 0 and a std above
   1; the std is what a threshold must be scaled by if it is ever
   quoted in "sigma" units.

Configurations (``CONFIG_ORDER``): ``white``; ``red_1x`` / ``red_3x``
(OU red noise at 1x / 3x the white level); ``red_sys`` (1x red plus
three shared systematics modes, searched with a PCA basis + coefficient
prior estimated from a signal-free population, as in Taaki et al. 2020);
and two PAIRED configurations added for the Phase-4 re-validation:

* ``white_bjd``: the ``white`` lightcurves on absolute timestamps,
  ``t + 2457000.5`` d relative to the generated sampling (every other
  configuration is handed ``t + 0.5`` d, so the two members differ by
  exactly ``BJD_OFFSET`` = 2457000 d and every method's internal
  ``floor(min t)``-anchored grid -- the automatic epoch grid, the BLS
  phase bins, the TLS epoch grid -- has the same phase in both; a
  fractional offset would re-phase those grids by the fraction and the
  comparison would measure grid alignment, not the float64 epoch
  subtraction it is meant to test);
* ``red_sys_nzm``: the ``red_sys`` data searched with the SAME basis
  plus a constant offset per column (``BASIS_COLUMN_OFFSETS``; the PCA
  columns are unit-norm over 600 points, rms 0.041, so the offsets are
  0.12-0.49 of a column's rms -- cotrending vectors that are far from
  zero-mean) with the unchanged coefficient prior -- exercises the
  intercept of the sequential cotrend and Detector A's centring (both
  centre the basis, so the fixed module should be invariant).

Each configuration draws its noise and injections from its OWN
``RandomState(seed + CONFIG_SEED_OFFSET[name])``; a paired
configuration shares its partner's seed and therefore sees exactly the
same lightcurves, so the per-lightcurve statistics recorded in the
JSON can be compared one-to-one (``summarize_lrt_validation.py`` does)
rather than only through completeness.

Arms: ``lrt`` (explicit epoch grid of ``round(2P / min duration)``
clipped to 8..96 epochs per period, epochs anchored at the first
observation in the caller's time scale), ``lrt_auto`` (the PUBLIC
DEFAULT: one ``run(t, y, periods, durations=...)`` call with
``epochs=None``, i.e. the module's automatic per-cell epoch grid and its
returned best epoch), ``bls`` (``eebls_gpu_fast``, its own q ladder),
``tls`` (``tls_search_batch``; the per-search score is its
un-normalized delta-chi-squared statistic ``SNR = sqrt(chi2_0 -
chi2_min)``, NOT the SDE: an SDE over the 32-point trial spectrum is
bounded by ``sqrt(31)`` and saturates), ``lrt_flat`` (PSD = ones;
``red_1x`` and ``red_3x``), ``lrt_marg`` (Detector A, default
``estimate_psd``) and ``lrt_seq`` (OLS cotrend + matched filter; both on
the systematics configurations).

Template grids: the LRT arms search durations {0.12, 0.21, 0.30} d
against the injected 0.22 d box (the nearest template recovers 97.7% of
the matched-filter statistic when centred). The explicit arm uses
round(2P / 0.12) epochs for EVERY duration (88 at P = 5.3 d, up to
0.030 d of misalignment), the default path ceil(2P / duration) per
cell (89/51/36 epochs at P = 5.3 d for the three durations, up to
0.030/0.052/0.074 d) -- the source of the default path's 4-9% deficit
against the explicit arm in the 2026-09-06 campaign. BLS's q ladder
(0.005..0.08, dlogq 0.3) has 0.2385 d at P = 5.3 d with P/200 phase
bins, so the comparators are slightly better matched to the injection
than the LRT grid is; the depth sweeps of each configuration bracket its
own detectability transition ([0.002, 0.003, 0.004, 0.008] in white
noise, [0.004, 0.006, 0.008, 0.016] at 1x red, [0.008, 0.016, 0.024,
0.032] at 3x red, [0.004, 0.008, 0.016, 0.032] with systematics). The
true period AND its 2P alias are placed on the shared period grid (P/2
already falls within 1% of a grid point).

Resolution: with n_inj = 200 the binomial 1-sigma of a completeness is
0.035 at p = 0.5; the null-p95 threshold from n_null = 200 maxima
carries its own sampling error (a common shift for all injections of
that arm). ``summarize_lrt_validation.py`` propagates both (bootstrap of
the null set + Wilson interval) into a per-cell uncertainty and reports
paired (same-lightcurve) arm differences within a configuration.

Noise model: white Gaussian + an exact Ornstein-Uhlenbeck (AR(1) in
continuous time) red component generated directly at the irregular
sample times (x_{i+1} = x_i e^{-dt/tau} + N(0, s^2(1-e^{-2 dt/tau}))),
so no uniform-grid interpolation is involved. The OU PSD is a Lorentzian
~ 1/(1+(2 pi f tau)^2) -- "stellar activity"-like low-frequency power.

Run on a GPU machine (full campaign, ~7 GPU-hours sequential on an A40;
four to six concurrent processes give ~2x throughput, the GPU's
context switching caps it there).
The configurations are independent, and within a configuration the
arms are too (the noise and injections are drawn from the
configuration's own seed in an order that does not depend on which
arms run), so the campaign can be split into parallel processes by
configuration and by arm and merged afterwards; ``--merge`` checks that
parts of the same configuration saw identical injections:
    python benchmarks/nufft_lrt/validate.py --configs white --arms lrt,bls,tls --out a.json
    python benchmarks/nufft_lrt/validate.py --configs white --arms lrt_auto --skip-calibration --out b.json
    python benchmarks/nufft_lrt/validate.py --configs red_sys,red_sys_nzm --out c.json
    ...
    python benchmarks/nufft_lrt/validate.py --merge a.json b.json c.json ... --out merged.json
The LRT null calibration runs in every process whose selection includes
``white`` unless ``--skip-calibration``. ``--quick`` runs smoke-test sizes.
"""
import argparse
import json
import os
import subprocess
import sys
import time

# One BLAS thread per process: the per-template host algebra is tiny and
# OpenBLAS's default (one thread per host core, 96 on the RunPod host
# against a ~8-CPU container quota) only adds spin-wait contention,
# especially when several campaign processes share the GPU.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402


CONFIG_ORDER = ['white', 'white_bjd', 'red_1x', 'red_3x',
                'red_sys', 'red_sys_nzm']
# Paired configurations share a sub-seed (identical noise + injections).
CONFIG_SEED_OFFSET = {'white': 1, 'white_bjd': 1, 'red_1x': 2, 'red_3x': 3,
                      'red_sys': 4, 'red_sys_nzm': 4}
CALIBRATION_SEED_OFFSET = 100
# Every configuration's times are t_base + REL_OFFSET; the BJD-scale
# configuration adds BJD_OFFSET on top, i.e. t_base + 2457000.5 d.
REL_OFFSET = 0.5
BJD_OFFSET = 2457000.0
BASIS_COLUMN_OFFSETS = (0.01, -0.005, 0.02)
HARNESS_VERSION = 2      # 1 = Jul/Sep-2026 (single rng, 60/60); 2 = Phase 4


# ---------------------------------------------------------------- data

def make_times(rng, mode='ground', baseline=90.0, n=600):
    """Irregular sampling. 'ground': nightly visibility windows with
    per-night jitter and random weather losses (the sampling regime the
    NUFFT path exists for)."""
    if mode == 'ground':
        nights = np.arange(int(baseline))
        keep = rng.rand(len(nights)) > 0.35          # weather
        nights = nights[keep]
        per_night = max(1, int(round(n / max(len(nights), 1))))
        t = (nights[:, None]
             + 0.25 * rng.rand(len(nights), per_night)).ravel()
        t = np.sort(t[:n])
        return t
    # 'space': near-uniform short-cadence with a mid-campaign gap
    t = np.linspace(0, baseline, n) + 1e-3 * rng.randn(n)
    gap = (t > 0.45 * baseline) & (t < 0.55 * baseline)
    return np.sort(t[~gap])


def ou_noise(rng, t, sigma_red, tau):
    """Exact OU process sampled at irregular times t."""
    x = np.zeros(len(t))
    x[0] = sigma_red * rng.randn()
    for i in range(1, len(t)):
        a = np.exp(-(t[i] - t[i - 1]) / tau)
        x[i] = x[i - 1] * a + sigma_red * np.sqrt(1 - a * a) * rng.randn()
    return x


def box_transit(t, period, epoch, duration, depth):
    phase = np.fmod(t - epoch, period) / period
    phase[phase < 0] += 1
    phase[phase > 0.5] -= 1
    y = np.zeros_like(t)
    y[np.abs(phase) <= duration / (2 * period)] = -depth
    return y


def make_lc(rng, t, sigma_white, sigma_red, tau, inject=None,
            sys_modes=None, sys_amps=None):
    y = 1.0 + sigma_white * rng.randn(len(t))
    if sigma_red > 0:
        y += ou_noise(rng, t, sigma_red, tau)
    if sys_modes is not None:
        y += sys_modes @ (rng.randn(sys_modes.shape[1]) * sys_amps)
    if inject is not None:
        y += box_transit(t, **inject)
    dy = np.full(len(t), sigma_white)   # what a pipeline would believe:
    return y, dy                        # formal (white) errors only


# ------------------------------------------------------------- methods
#
# Every search is called as search(t, y, dy) and returns
# (max statistic, best period, best epoch or nan) in the caller's time
# scale.

class LRTSearch:
    """PSD-whitened NUFFT matched filter over a (period, duration,
    epoch) template grid.

    ``auto_epochs=False`` (the ``lrt`` arm): one ``run()`` call per
    period with an explicit epoch grid that scales with period so
    template misalignment stays below ~half the narrowest trial duration
    (``epoch_oversample`` boxes per duration; a fixed epoch count would
    leave long periods unsearchable for box overlap). The epochs are
    anchored at the first observation, in the caller's time scale, as a
    user with absolute timestamps would write them.

    ``auto_epochs=True`` (the ``lrt_auto`` arm): the public default path,
    ONE ``run(t, y, periods, durations=durations)`` call with
    ``epochs=None`` and every other argument at its default; the module
    builds its own per-(period, duration) epoch grid and returns the max
    over it plus the best epoch.
    """

    def __init__(self, periods, durations, epoch_oversample=2.0,
                 max_epochs=96, flat_psd=False, auto_epochs=False,
                 run_kwargs=None, **proc_kwargs):
        from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess, epoch_grid
        self.proc = NUFFTLRTAsyncProcess(**proc_kwargs)
        self.periods = np.asarray(periods, dtype=np.float64)
        self.durations = np.asarray(durations, dtype=np.float64)
        self.epoch_oversample = epoch_oversample
        self.max_epochs = max_epochs
        self.flat_psd = flat_psd
        self.auto_epochs = auto_epochs
        self.run_kwargs = dict(run_kwargs or {})
        if auto_epochs:
            # the module's own grid at its defaults (2.0, 8, 96)
            self.n_templates = sum(len(epoch_grid(P, d))
                                   for P in self.periods
                                   for d in self.durations)
        else:
            self.n_templates = sum(
                self._n_epochs(P) * len(self.durations)
                for P in self.periods)

    def _n_epochs(self, P):
        n = int(round(self.epoch_oversample * P / self.durations.min()))
        return int(min(max(n, 8), self.max_epochs))

    def __call__(self, t, y, dy):
        kwargs = dict(self.run_kwargs)
        if self.flat_psd:
            nf = 2 * len(t)
            kwargs.update(estimate_psd=False,
                          psd=np.ones(nf, dtype=np.float32), nf=nf)
        if self.auto_epochs:
            snr, best_epoch = self.proc.run(t, y, self.periods,
                                            durations=self.durations,
                                            **kwargs)
            i, j = np.unravel_index(int(np.argmax(snr)), snr.shape)
            return (float(snr[i, j]), float(self.periods[i]),
                    float(best_epoch[i, j]))
        best = (-np.inf, np.nan, np.nan)
        t_ref = float(np.min(t))
        for P in self.periods:
            epochs = t_ref + np.linspace(0, P, self._n_epochs(P),
                                         endpoint=False)
            snr = self.proc.run(t, y, np.array([P]),
                                durations=self.durations,
                                epochs=epochs, **kwargs)
            k = int(np.argmax(snr))
            m = float(snr.ravel()[k])
            if m > best[0]:
                best = (m, float(P),
                        float(epochs[np.unravel_index(k, snr.shape)[2]]))
        return best


class BLSSearch:
    def __init__(self, periods, qvals):
        from cuvarbase.bls import eebls_gpu_fast
        self._bls = eebls_gpu_fast
        self.freqs = np.sort(1.0 / periods).astype(np.float64)
        self.qmin, self.qmax = qvals

    def __call__(self, t, y, dy):
        power = self._bls(t, y, dy, self.freqs,
                          qmin=self.qmin, qmax=self.qmax)
        i = int(np.argmax(power))
        return float(power[i]), float(1.0 / self.freqs[i]), np.nan


class TLSSearch:
    def __init__(self, periods, qvals):
        from cuvarbase.tls import tls_search_batch
        self._tls = tls_search_batch
        self.periods = np.asarray(periods, dtype=np.float64)
        q = np.full(len(self.periods), qvals[0]), \
            np.full(len(self.periods), qvals[1])
        self.qmin, self.qmax = q

    def __call__(self, t, y, dy):
        r = self._tls([(t, y, dy)], periods=self.periods,
                      qmin=self.qmin, qmax=self.qmax)[0]
        if 'error' in r:
            return 0.0, np.nan, np.nan
        # un-normalized per-search score (sqrt of the chi2 improvement
        # of the best template over the constant model); the SDE of a
        # 32-point spectrum is bounded by sqrt(31) and would saturate
        return float(r['SNR']), float(r['period']), np.nan


# ------------------------------------------- shared systematics (paper)

def make_systematics_modes(t, baseline):
    """Three plausible shared instrument/site modes: a slow drift, a
    within-night 'airmass' parabola, and a long-period thermal-like
    oscillation."""
    m1 = (t - t.mean()) / (0.5 * baseline)
    night = np.floor(t)
    tn = t - night - 0.125                       # hours from mid-window
    m2 = (tn / 0.125) ** 2 - 0.5
    m3 = np.sin(2 * np.pi * t / (0.4 * baseline))
    M = np.stack([m1, m2, m3], axis=1)
    return M / np.std(M, axis=0)


def build_basis_from_population(rng, t, baseline, sigma_white, sigma_red,
                                tau, amps, n_pop=60, K=3):
    """Paper-style systematics model: PCA basis from a population of
    signal-free lightcurves sharing the true modes, plus a Gaussian
    prior on coefficients from per-lightcurve least-squares fits. The
    population is row-centred, so the PCA modes are exactly zero-mean
    (the ``red_sys_nzm`` configuration adds column offsets afterwards)."""
    M = make_systematics_modes(t, baseline)
    pop = np.empty((n_pop, len(t)))
    for i in range(n_pop):
        c = rng.randn(M.shape[1]) * amps
        y, _ = make_lc(rng, t, sigma_white, sigma_red, tau)
        pop[i] = y + M @ c
    pop -= pop.mean(axis=1, keepdims=True)
    # PCA over the population (as in Taaki et al. 2020)
    _, _, VT = np.linalg.svd(pop, full_matrices=False)
    V = VT[:K].T
    coeffs = pop @ V           # per-lightcurve LS fits (V orthonormal)
    prior_mean = coeffs.mean(axis=0)
    prior_cov = np.cov(coeffs.T)
    return M, V, prior_mean, prior_cov


def period_hit(p_found, p_true, tol=0.01):
    if not np.isfinite(p_found):
        return False
    for target in (p_true, 2 * p_true, 0.5 * p_true):
        if abs(p_found - target) / target < tol:
            return True
    return False


def epoch_error(epoch_found, epoch_true, p_found, p_true):
    """Smallest |epoch_found - epoch_true| modulo the transit spacing the
    found period implies (``min(p_found, p_true)``: a 2P alias still
    lands on true transits; a P/2 alias on every other template
    transit). nan when no epoch was reported."""
    if not (np.isfinite(epoch_found) and np.isfinite(p_found)):
        return np.nan
    wrap = min(float(p_found), float(p_true))
    d = np.fmod(epoch_found - epoch_true, wrap)
    d = abs(d)
    return float(min(d, wrap - d))


# ------------------------------------------------------------ protocol

def run_config(cfg, methods, n_null, n_inj, depths, t_base, log=print):
    """One configuration. The data are generated on ``t_base`` (relative
    times) from the configuration's own RandomState and handed to every
    search at ``t = t_base + cfg['t_offset']``; injected epochs are
    recorded in that same (caller) time scale. Per-lightcurve results
    are kept so paired configurations can be compared one-to-one."""
    rng = np.random.RandomState(cfg['seed'])
    t_off = float(cfg.get('t_offset', 0.0))
    t = t_base + t_off
    out = {'config': {k: v for k, v in cfg.items()
                      if k != 'name' and not k.startswith('_')},
           'methods': {}}
    p_true, dur_true = cfg['p_true'], cfg['dur_true']
    sys_kw = dict(sys_modes=cfg.get('_sys_modes'),
                  sys_amps=cfg.get('_sys_amps'))
    wall = {name: 0.0 for name in methods}
    n_calls = {name: 0 for name in methods}

    # one untimed search per arm on a throwaway lightcurve (its own
    # RandomState, so the configuration's draws are untouched): kernel
    # compilation and first-call allocation stay out of the per-search
    # cost, and out of the first configuration a process runs
    y_w, dy_w = make_lc(np.random.RandomState(0), t_base, cfg['sigma_white'],
                        cfg['sigma_red'], cfg['tau'], **sys_kw)
    for search in methods.values():
        search(t, y_w, dy_w)

    def timed(name, search, y, dy):
        t0 = time.time()
        r = search(t, y, dy)
        wall[name] += time.time() - t0
        n_calls[name] += 1
        return r

    # 1. null threshold per method
    nulls = {name: [] for name in methods}
    t_start = time.time()
    for i in range(n_null):
        y, dy = make_lc(rng, t_base, cfg['sigma_white'], cfg['sigma_red'],
                        cfg['tau'], **sys_kw)
        for name, search in methods.items():
            stat, _, _ = timed(name, search, y, dy)
            nulls[name].append(stat)
        if (i + 1) % 25 == 0 or i + 1 == n_null:
            log('  [%s] null %d/%d  (%.0f s)' % (cfg['name'], i + 1, n_null,
                                                 time.time() - t_start))

    for name in methods:
        arr = np.sort(np.asarray(nulls[name]))
        thresh = float(np.percentile(arr, 95))
        out['methods'][name] = {
            'null_max_median': float(np.median(arr)),
            'null_max_p95': thresh,
            'completeness': {},
            'epoch_recovery': {},
            'null_stats': [round(float(v), 7) for v in nulls[name]],
            'injections': {},
        }
    out['injected_epochs'] = {}

    # 2. injections, swept depth
    for depth in depths:
        hits = {name: 0 for name in methods}
        rec = {name: {'stat': [], 'p_found': [], 'epoch_found': [],
                      'detected': []} for name in methods}
        epochs_true = []
        for i in range(n_inj):
            epoch = rng.rand() * p_true            # relative frame
            epochs_true.append(round(epoch + t_off, 7))
            y, dy = make_lc(rng, t_base, cfg['sigma_white'], cfg['sigma_red'],
                            cfg['tau'],
                            inject=dict(period=p_true, epoch=epoch,
                                        duration=dur_true, depth=depth),
                            **sys_kw)
            for name, search in methods.items():
                stat, p_found, e_found = timed(name, search, y, dy)
                det = bool(stat > out['methods'][name]['null_max_p95']
                           and period_hit(p_found, p_true))
                hits[name] += det
                r = rec[name]
                r['stat'].append(round(float(stat), 7))
                r['p_found'].append(round(float(p_found), 7)
                                    if np.isfinite(p_found) else None)
                r['epoch_found'].append(round(float(e_found), 7)
                                        if np.isfinite(e_found) else None)
                r['detected'].append(det)
            if (i + 1) % 50 == 0 or i + 1 == n_inj:
                log('  [%s] depth %s inj %d/%d  (%.0f s)'
                    % (cfg['name'], depth, i + 1, n_inj,
                       time.time() - t_start))
        out['injected_epochs'][str(depth)] = epochs_true
        for name in methods:
            m = out['methods'][name]
            m['completeness'][str(depth)] = hits[name] / n_inj
            m['injections'][str(depth)] = rec[name]
            # epoch recovery among detections (arms that report an epoch)
            errs = [epoch_error(e, e_true, p, p_true)
                    for e, p, e_true, det in zip(rec[name]['epoch_found'],
                                                 rec[name]['p_found'],
                                                 epochs_true,
                                                 rec[name]['detected'])
                    if det and e is not None]
            if errs:
                errs = np.asarray(errs)
                m['epoch_recovery'][str(depth)] = {
                    'n_detected': int(len(errs)),
                    'frac_within_half_duration':
                        float(np.mean(errs <= 0.5 * dur_true)),
                    'median_abs_error_d': float(np.median(errs)),
                    'max_abs_error_d': float(np.max(errs)),
                }
    for name in methods:
        out['methods'][name]['seconds_per_search'] = (
            wall[name] / max(n_calls[name], 1))
        out['methods'][name]['n_templates'] = getattr(
            methods[name], 'n_templates', None)
    return out


def snr_calibration(rng, t, proc_kwargs, n=200):
    """Null mean/std of the LRT statistic on pure white noise at ONE
    fixed template: the calibration constant of this (sampling, nf, PSD
    estimator) configuration. The statistic is a whitened correlation,
    not N(0,1): with irregular sampling the NFFT modes are not
    orthogonal and the null std is ~1.8-2.7 for the harness's ground
    sampling at nf = 2n even with the true PSD (module docstring of
    cuvarbase.nufft_lrt). A mean far from 0 would indicate a
    normalization bug; a std above 1 is expected."""
    from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
    proc = NUFFTLRTAsyncProcess(**proc_kwargs)
    vals = []
    for i in range(n):
        y = 1 + 1e-3 * rng.randn(len(t))
        # ONE fixed template (epochs=None now scans an epoch grid and
        # returns (max, best_epoch); the calibration is single-template)
        snr = proc.run(t, y - np.mean(y), np.array([3.7]),
                       durations=np.array([0.15]),
                       epochs=np.array([0.0]))
        vals.append(float(snr[0, 0, 0]))
    v = np.asarray(vals)
    return {'mean': float(v.mean()), 'std': float(v.std()),
            'n': n}


# ----------------------------------------------------------- bookkeeping

def _gpu_name():
    try:
        import pycuda.driver as cuda
        cuda.init()
        return cuda.Device(0).name()
    except Exception:              # noqa: BLE001 -- label only
        return None


def _git_sha():
    """``HEAD`` plus ``-dirty`` when the tree has uncommitted changes
    (the archived campaign should point at a commit that contains the
    harness that produced it)."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            text=True).strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            stderr=subprocess.DEVNULL, text=True).strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:              # noqa: BLE001 -- label only
        return None


def merge_results(paths):
    """Merge per-process JSONs (``--configs`` subsets of one campaign)
    into one campaign JSON: identical protocol meta required, configs
    ordered by CONFIG_ORDER, the calibration taken from the file that
    has it."""
    parts = []
    for p in paths:
        with open(p) as f:
            parts.append(json.load(f))
    keys = ['seed', 'n_null', 'n_inj', 'n_periods', 'ndata', 'baseline',
            'p_true', 'dur_true', 'depths', 'sigma_white',
            'harness_version', 'rel_offset', 'bjd_offset', 'lrt_sigma',
            'lrt_nf']
    ref = parts[0]['meta']
    for part, p in zip(parts, paths):
        if not part['meta'].get('complete', False):
            raise ValueError('%s is an incomplete checkpoint (its process '
                             'did not finish); refusing to merge it' % p)
        have = [c['name'] for c in part['configs']]
        if have != list(part['meta'].get('configs_run', have)):
            raise ValueError('%s holds configs %s but claims %s'
                             % (p, have, part['meta'].get('configs_run')))
        for k in keys:
            if part['meta'].get(k) != ref.get(k):
                raise ValueError('meta %r differs in %s: %r vs %r'
                                 % (k, p, part['meta'].get(k), ref.get(k)))
    merged = {'meta': dict(ref), 'snr_calibration': None, 'configs': []}
    merged['meta']['gpu'] = sorted({str(part['meta'].get('gpu'))
                                    for part in parts})
    merged['meta']['git_sha'] = sorted({str(part['meta'].get('git_sha'))
                                        for part in parts})
    merged['meta']['date'] = sorted({str(part['meta'].get('date'))
                                     for part in parts})
    merged['meta']['merged_from'] = [str(p) for p in paths]
    cals = [part['snr_calibration'] for part in parts
            if part.get('snr_calibration')]
    if cals:
        merged['snr_calibration'] = cals[0]
    seen = {}
    for part in parts:
        for cfg in part['configs']:
            name = cfg['name']
            if name not in seen:
                seen[name] = cfg
                continue
            # the same configuration run in another process with other
            # arms: identical protocol, identical injections required
            base = seen[name]
            if cfg['config'] != base['config']:
                raise ValueError('config %r: protocol differs between '
                                 'parts' % name)
            if cfg['injected_epochs'] != base['injected_epochs']:
                raise ValueError('config %r: parts did not see the same '
                                 'injections' % name)
            dup = set(cfg['methods']) & set(base['methods'])
            if dup:
                raise ValueError('config %r: arm(s) %s appear twice'
                                 % (name, sorted(dup)))
            base['methods'].update(cfg['methods'])
            base['wall_s'] = base.get('wall_s', 0.0) + cfg.get('wall_s', 0.0)
    merged['configs'] = [seen[n] for n in CONFIG_ORDER if n in seen]
    merged['configs'] += [c for n, c in seen.items() if n not in CONFIG_ORDER]
    merged['meta']['configs_run'] = [c['name'] for c in merged['configs']]
    merged['meta']['wall_s_total'] = float(sum(
        c.get('wall_s', 0.0) for c in merged['configs']))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='nufft_lrt_validation.json')
    ap.add_argument('--quick', action='store_true',
                    help='smoke-test sizes')
    ap.add_argument('--seed', type=int, default=20260711)
    ap.add_argument('--skip-tls', action='store_true')
    ap.add_argument('--n-null', type=int, default=None)
    ap.add_argument('--n-inj', type=int, default=None)
    ap.add_argument('--configs', default=None,
                    help='comma-separated subset of %s (default: all); '
                         'the LRT null calibration runs when the '
                         'selection includes "white"'
                         % ','.join(CONFIG_ORDER))
    ap.add_argument('--arms', default=None,
                    help='comma-separated subset of the arms each selected '
                         'configuration would run (default: all of them); '
                         'parts of one configuration merge with --merge')
    ap.add_argument('--skip-calibration', action='store_true',
                    help='do not run the LRT null calibration')
    ap.add_argument('--merge', nargs='+', metavar='JSON', default=None,
                    help='merge these per-process JSONs into --out and '
                         'exit')
    args = ap.parse_args()

    if args.merge:
        merged = merge_results(args.merge)
        with open(args.out, 'w') as f:
            json.dump(merged, f, indent=1)
        print('merged %d configs -> %s' % (len(merged['configs']), args.out))
        return 0

    selected = (list(CONFIG_ORDER) if args.configs is None
                else [s.strip() for s in args.configs.split(',') if s.strip()])
    unknown = [s for s in selected if s not in CONFIG_ORDER]
    if unknown:
        ap.error('unknown config(s) %s; choose from %s'
                 % (unknown, CONFIG_ORDER))

    rng = np.random.RandomState(args.seed)

    n_null = args.n_null or (12 if args.quick else 200)
    n_inj = args.n_inj or (8 if args.quick else 200)
    n_periods = 16 if args.quick else 32
    # depth sweeps bracket each configuration's own detectability
    # transition (transition depths from the pre-fix campaign)
    if args.quick:
        depths = {'white': [0.004, 0.008], 'red_1x': [0.008, 0.016],
                  'red_3x': [0.016, 0.032], 'red_sys': [0.008, 0.016]}
    else:
        depths = {'white': [0.002, 0.003, 0.004, 0.008],
                  'red_1x': [0.004, 0.006, 0.008, 0.016],
                  'red_3x': [0.008, 0.016, 0.024, 0.032],
                  'red_sys': [0.004, 0.008, 0.016, 0.032]}

    # The sampling and the population basis come from the master rng in
    # a fixed order, so every process of a split campaign builds the
    # same t, V, prior.
    t = make_times(rng, 'ground', baseline=90.0, n=600)
    p_true, dur_true = 5.3, 0.22
    periods = np.exp(np.linspace(np.log(2.0), np.log(18.0), n_periods))
    # inject exactly on the shared grid: completeness then measures
    # detection, not grid-resolution luck (all methods share the grid);
    # the 2P alias is put on the grid too so that period_hit's 2:1
    # credit is real (P/2 = 2.65 falls within 1% of a grid point anyway)
    periods[np.argmin(np.abs(periods - p_true))] = p_true
    periods[np.argmin(np.abs(periods - 2 * p_true))] = 2 * p_true
    durations = np.array([0.12, 0.21, 0.30])
    qvals = (0.005, 0.08)

    sigma_w = 3e-3

    def cfg_seed(name):
        return int(args.seed + CONFIG_SEED_OFFSET[name])

    configs = [
        dict(name='white', sigma_white=sigma_w, sigma_red=0.0, tau=1.0,
             p_true=p_true, dur_true=dur_true, depths=depths['white'],
             seed=cfg_seed('white'), t_offset=REL_OFFSET),
        dict(name='white_bjd', sigma_white=sigma_w, sigma_red=0.0, tau=1.0,
             p_true=p_true, dur_true=dur_true, depths=depths['white'],
             seed=cfg_seed('white_bjd'), t_offset=REL_OFFSET + BJD_OFFSET,
             paired_with='white'),
        dict(name='red_1x', sigma_white=sigma_w, sigma_red=1.0 * sigma_w,
             tau=0.8, p_true=p_true, dur_true=dur_true,
             depths=depths['red_1x'],
             seed=cfg_seed('red_1x'), t_offset=REL_OFFSET),
        dict(name='red_3x', sigma_white=sigma_w, sigma_red=3.0 * sigma_w,
             tau=0.8, p_true=p_true, dur_true=dur_true,
             depths=depths['red_3x'],
             seed=cfg_seed('red_3x'), t_offset=REL_OFFSET),
    ]

    # shared-systematics config (the paper's core contrast): PCA basis +
    # coefficient prior estimated from a signal-free population, exactly
    # as Taaki et al. (2020) do with Kepler PCA modes
    sys_amps = np.array([6.0, 3.0, 6.0]) * sigma_w
    true_modes, V_est, mu_c, cov_c = build_basis_from_population(
        rng, t, 90.0, sigma_w, 1.0 * sigma_w, 0.8, sys_amps,
        n_pop=20 if args.quick else 60)
    sys_cfg = dict(sigma_white=sigma_w, sigma_red=1.0 * sigma_w,
                   tau=0.8, p_true=p_true, dur_true=dur_true,
                   depths=depths['red_sys'],
                   sys_amp_over_white=[float(a / sigma_w) for a in sys_amps],
                   _sys_modes=true_modes, _sys_amps=sys_amps)
    configs.append(dict(sys_cfg, name='red_sys', seed=cfg_seed('red_sys'),
                        t_offset=REL_OFFSET))
    # the SAME data searched with a basis whose columns are not
    # zero-mean (constant offsets of 0.12-0.49 column-rms added to the
    # unit-norm PCA modes; the prior is unchanged, as a user with
    # un-centred cotrending vectors would have)
    col_off = np.asarray(BASIS_COLUMN_OFFSETS[:V_est.shape[1]], np.float64)
    V_nzm = V_est + col_off[None, :]
    configs.append(dict(sys_cfg, name='red_sys_nzm',
                        seed=cfg_seed('red_sys_nzm'), t_offset=REL_OFFSET,
                        paired_with='red_sys',
                        basis_column_offsets=[float(c) for c in col_off]))
    configs = [c for c in configs if c['name'] in selected]

    eo = 1.0 if args.quick else 2.0
    lrt = LRTSearch(periods, durations, epoch_oversample=eo)
    lrt_auto = LRTSearch(periods, durations, auto_epochs=True)
    methods = {
        'lrt': lrt,
        'lrt_auto': lrt_auto,
        'bls': BLSSearch(periods, qvals),
    }
    if not args.skip_tls:
        methods['tls'] = TLSSearch(periods, qvals)
    print('LRT templates per search: %d explicit-epoch, %d automatic '
          '(epochs=None)' % (lrt.n_templates, lrt_auto.n_templates),
          flush=True)

    # the flat-PSD arm isolates what the whitening itself buys; it runs
    # on the red-noise configs (in white noise the estimated PSD is
    # ~flat and the arms coincide)
    lrt_flat = LRTSearch(periods, durations, epoch_oversample=eo,
                         flat_psd=True)

    # joint (Detector A) and sequential-detrend arms for the
    # systematics configs, sharing the population-estimated basis/prior
    def marg_seq(V):
        return (LRTSearch(periods, durations, epoch_oversample=eo,
                          run_kwargs=dict(detector='marginal',
                                          systematics_basis=V,
                                          coeff_prior_mean=mu_c,
                                          coeff_prior_cov=cov_c)),
                LRTSearch(periods, durations, epoch_oversample=eo,
                          run_kwargs=dict(detector='sequential',
                                          systematics_basis=V)))
    lrt_marg, lrt_seq = marg_seq(V_est)
    lrt_marg_nzm, lrt_seq_nzm = marg_seq(V_nzm)

    arms = (None if args.arms is None
            else [a.strip() for a in args.arms.split(',') if a.strip()])

    results = {'meta': dict(seed=args.seed, n_null=n_null, n_inj=n_inj,
                            n_periods=n_periods,
                            ndata=len(t), baseline=90.0,
                            p_true=p_true, dur_true=dur_true,
                            depths=depths, sigma_white=sigma_w,
                            durations=[float(d) for d in durations],
                            qvals=list(qvals),
                            harness_version=HARNESS_VERSION,
                            rel_offset=REL_OFFSET, bjd_offset=BJD_OFFSET,
                            configs_run=[c['name'] for c in configs],
                            arms=arms, complete=False,
                            gpu=_gpu_name(), git_sha=_git_sha(),
                            date=time.strftime('%Y-%m-%d'),
                            lrt_sigma=float(lrt.proc.sigma),
                            lrt_nf=int(2 * len(t))),
               'snr_calibration': None, 'configs': []}

    if 'white' in selected and not args.skip_calibration:
        print('LRT statistic null calibration on white noise...',
              flush=True)
        # 1000 draws: with 200 the sample std/mean of the 2026-09-06
        # campaign came out 1.58/0.35 where 5000 draws from the same
        # seed give 1.81/0.03 (see benchmarks/results/nufft_lrt_validation_2026-09-06/)
        results['snr_calibration'] = snr_calibration(
            np.random.RandomState(args.seed + CALIBRATION_SEED_OFFSET),
            t, {}, n=40 if args.quick else 1000)
        print('  mean=%.3f std=%.3f (calibration constant: mean ~0 '
              'expected; std is NOT ~1 by design -- the pre-fix campaign '
              'measured 1.81 at nf = 2n with the sigma = 2 NFFT)'
              % (results['snr_calibration']['mean'],
                 results['snr_calibration']['std']), flush=True)

    for cfg in configs:
        t0 = time.time()
        print('config %s (seed %d, t_offset %g) ...'
              % (cfg['name'], cfg['seed'], cfg['t_offset']), flush=True)
        if cfg['name'] == 'red_sys_nzm':
            cfg_methods = {'lrt_marg': lrt_marg_nzm, 'lrt_seq': lrt_seq_nzm}
        else:
            cfg_methods = dict(methods)
            if cfg['name'] in ('red_1x', 'red_3x'):
                cfg_methods['lrt_flat'] = lrt_flat
            if cfg['name'] == 'red_sys':
                cfg_methods['lrt_marg'] = lrt_marg
                cfg_methods['lrt_seq'] = lrt_seq
        if arms is not None:
            missing = [a for a in arms if a not in cfg_methods]
            if missing:
                ap.error('arm(s) %s are not run on config %r (available: '
                         '%s)' % (missing, cfg['name'], sorted(cfg_methods)))
            cfg_methods = {a: cfg_methods[a] for a in arms}
        r = run_config(cfg, cfg_methods, n_null, n_inj, cfg['depths'], t,
                       log=lambda s: print(s, flush=True))
        r['name'] = cfg['name']
        r['wall_s'] = time.time() - t0
        results['configs'].append(r)
        for name, m in r['methods'].items():
            print('  %-8s null_p95=%8.3f  %.3f s/search  completeness=%s'
                  % (name, m['null_max_p95'], m['seconds_per_search'],
                     {d: c for d, c in m['completeness'].items()}),
                  flush=True)
        # checkpoint after every config
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=1)

    results['meta']['complete'] = True
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
