"""NUFFT-LRT injection-recovery validation vs BLS (and TLS).

The question this answers (audit follow-up, July 2026): does the
PSD-whitened matched filter actually buy detection performance in
correlated noise, and what does it cost in white noise — i.e. when is it
the right tool?

Protocol (per noise configuration):

1. NULL RUNS: generate signal-free lightcurves, run each method's search
   over the identical period range, record the maximum statistic. The
   95th percentile of the null maxima is that method's detection
   threshold at a fixed 5% per-search false-alarm rate. This
   self-calibration is what makes methods with different statistics
   (LRT SNR, BLS power, TLS SDE) comparable — and it is exactly where
   red noise hurts BLS/TLS: their null maxima inflate, raising the bar.
2. INJECTION RUNS: inject box transits (random epoch, fixed
   period/duration, swept depth) into fresh noise; a detection requires
   the statistic to exceed the null threshold AND the best period to
   land within 1% of the truth or its 2:1 aliases.
3. Completeness(depth) per method per noise config + the LRT SNR
   calibration check (statistic ~ N(0,1) on white noise at a fixed
   template).

Noise model: white Gaussian + an exact Ornstein-Uhlenbeck (AR(1) in
continuous time) red component generated directly at the irregular
sample times (x_{i+1} = x_i e^{-dt/tau} + N(0, s^2(1-e^{-2 dt/tau}))),
so no uniform-grid interpolation is involved. The OU PSD is a Lorentzian
~ 1/(1+(2 pi f tau)^2) — "stellar activity"-like low-frequency power.

Run on a GPU machine:
    python scripts/nufft_lrt_validation.py --out results.json [--quick]
"""
import argparse
import json
import sys
import time

import numpy as np


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


def make_lc(rng, t, sigma_white, sigma_red, tau, inject=None):
    y = 1.0 + sigma_white * rng.randn(len(t))
    if sigma_red > 0:
        y += ou_noise(rng, t, sigma_red, tau)
    if inject is not None:
        y += box_transit(t, **inject)
    dy = np.full(len(t), sigma_white)   # what a pipeline would believe:
    return y, dy                        # formal (white) errors only


# ------------------------------------------------------------- methods

class LRTSearch:
    """PSD-whitened NUFFT matched filter over a (period, duration,
    epoch) template grid. The epoch grid scales with period so template
    misalignment stays below ~half the narrowest trial duration
    (epoch_oversample boxes per duration) -- a fixed epoch count would
    leave long periods unsearchable for box overlap."""

    def __init__(self, periods, durations, epoch_oversample=2.0,
                 max_epochs=96, flat_psd=False, **proc_kwargs):
        from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
        self.proc = NUFFTLRTAsyncProcess(**proc_kwargs)
        self.periods = periods
        self.durations = durations
        self.epoch_oversample = epoch_oversample
        self.max_epochs = max_epochs
        self.flat_psd = flat_psd
        self.n_templates = sum(
            self._n_epochs(P) * len(durations) for P in periods)

    def _n_epochs(self, P):
        n = int(round(self.epoch_oversample * P / self.durations.min()))
        return int(min(max(n, 8), self.max_epochs))

    def __call__(self, t, y, dy):
        best = (-np.inf, np.nan)
        kwargs = {}
        if self.flat_psd:
            nf = 2 * len(t)
            kwargs = dict(estimate_psd=False,
                          psd=np.ones(nf, dtype=np.float32), nf=nf)
        for P in self.periods:
            epochs = np.linspace(0, P, self._n_epochs(P), endpoint=False)
            snr = self.proc.run(t, y, np.array([P]),
                                durations=self.durations,
                                epochs=epochs, **kwargs)
            m = float(np.max(snr))
            if m > best[0]:
                best = (m, float(P))
        return best   # (max statistic, best period)


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
        return float(power[i]), float(1.0 / self.freqs[i])


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
            return 0.0, np.nan
        return float(r['SDE']), float(r['period'])


def period_hit(p_found, p_true, tol=0.01):
    if not np.isfinite(p_found):
        return False
    for target in (p_true, 2 * p_true, 0.5 * p_true):
        if abs(p_found - target) / target < tol:
            return True
    return False


# ------------------------------------------------------------ protocol

def run_config(cfg, methods, rng, n_null, n_inj, depths, t):
    out = {'config': {k: v for k, v in cfg.items() if k != 'name'},
           'methods': {}}
    p_true, dur_true = cfg['p_true'], cfg['dur_true']

    # 1. null threshold per method
    nulls = {name: [] for name in methods}
    for i in range(n_null):
        y, dy = make_lc(rng, t, cfg['sigma_white'], cfg['sigma_red'],
                        cfg['tau'])
        for name, search in methods.items():
            stat, _ = search(t, y, dy)
            nulls[name].append(stat)

    for name in methods:
        arr = np.sort(np.asarray(nulls[name]))
        thresh = float(np.percentile(arr, 95))
        out['methods'][name] = {
            'null_max_median': float(np.median(arr)),
            'null_max_p95': thresh,
            'completeness': {},
        }

    # 2. injections, swept depth
    for depth in depths:
        hits = {name: 0 for name in methods}
        for i in range(n_inj):
            epoch = rng.rand() * p_true
            y, dy = make_lc(rng, t, cfg['sigma_white'], cfg['sigma_red'],
                            cfg['tau'],
                            inject=dict(period=p_true, epoch=epoch,
                                        duration=dur_true, depth=depth))
            for name, search in methods.items():
                stat, p_found = search(t, y, dy)
                if (stat > out['methods'][name]['null_max_p95']
                        and period_hit(p_found, p_true)):
                    hits[name] += 1
        for name in methods:
            out['methods'][name]['completeness'][str(depth)] = \
                hits[name] / n_inj
    return out


def snr_calibration(rng, t, proc_kwargs, n=200):
    """LRT statistic on pure white noise at ONE fixed template must be
    ~ N(0,1) if the whitened matched filter is correctly normalized."""
    from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
    proc = NUFFTLRTAsyncProcess(**proc_kwargs)
    vals = []
    for i in range(n):
        y = 1 + 1e-3 * rng.randn(len(t))
        snr = proc.run(t, y - np.mean(y), np.array([3.7]),
                       durations=np.array([0.15]))
        vals.append(float(snr[0, 0]))
    v = np.asarray(vals)
    return {'mean': float(v.mean()), 'std': float(v.std()),
            'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='nufft_lrt_validation.json')
    ap.add_argument('--quick', action='store_true',
                    help='smoke-test sizes')
    ap.add_argument('--seed', type=int, default=20260711)
    ap.add_argument('--skip-tls', action='store_true')
    ap.add_argument('--n-null', type=int, default=None)
    ap.add_argument('--n-inj', type=int, default=None)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    n_null = args.n_null or (12 if args.quick else 60)
    n_inj = args.n_inj or (8 if args.quick else 60)
    n_periods = 16 if args.quick else 48
    depths = [0.004, 0.008] if args.quick else [0.002, 0.004, 0.008, 0.016]

    t = make_times(rng, 'ground', baseline=90.0, n=600)
    p_true, dur_true = 5.3, 0.22
    periods = np.exp(np.linspace(np.log(2.0), np.log(18.0), n_periods))
    # inject exactly on the shared grid: completeness then measures
    # detection, not grid-resolution luck (all methods share the grid)
    periods[np.argmin(np.abs(periods - p_true))] = p_true
    durations = np.array([0.12, 0.25])
    qvals = (0.005, 0.08)

    sigma_w = 3e-3
    configs = [
        dict(name='white', sigma_white=sigma_w, sigma_red=0.0, tau=1.0,
             p_true=p_true, dur_true=dur_true),
        dict(name='red_1x', sigma_white=sigma_w, sigma_red=1.0 * sigma_w,
             tau=0.8, p_true=p_true, dur_true=dur_true),
        dict(name='red_3x', sigma_white=sigma_w, sigma_red=3.0 * sigma_w,
             tau=0.8, p_true=p_true, dur_true=dur_true),
    ]

    lrt = LRTSearch(periods, durations,
                    epoch_oversample=1.0 if args.quick else 2.0)
    methods = {
        'lrt': lrt,
        'bls': BLSSearch(periods, qvals),
    }
    if not args.skip_tls:
        methods['tls'] = TLSSearch(periods, qvals)
    print('LRT templates per search: %d' % lrt.n_templates, flush=True)

    # the flat-PSD arm isolates what the whitening itself buys; it runs
    # on the strongest-red config only (in white noise the estimated
    # PSD is ~flat and the arms coincide)
    lrt_flat = LRTSearch(periods, durations,
                         epoch_oversample=1.0 if args.quick else 2.0,
                         flat_psd=True)

    results = {'meta': dict(seed=args.seed, n_null=n_null, n_inj=n_inj,
                            n_periods=n_periods,
                            ndata=len(t), baseline=90.0,
                            p_true=p_true, dur_true=dur_true,
                            depths=depths, sigma_white=sigma_w),
               'snr_calibration': None, 'configs': []}

    print('LRT SNR calibration on white noise...', flush=True)
    results['snr_calibration'] = snr_calibration(
        rng, t, {}, n=40 if args.quick else 200)
    print('  mean=%.3f std=%.3f (want ~0, ~1)'
          % (results['snr_calibration']['mean'],
             results['snr_calibration']['std']), flush=True)

    for cfg in configs:
        t0 = time.time()
        print('config %s ...' % cfg['name'], flush=True)
        cfg_methods = dict(methods)
        if cfg['name'] == 'red_3x':
            cfg_methods['lrt_flat'] = lrt_flat
        r = run_config(cfg, cfg_methods, rng, n_null, n_inj, depths, t)
        r['name'] = cfg['name']
        r['wall_s'] = time.time() - t0
        results['configs'].append(r)
        for name, m in r['methods'].items():
            print('  %-4s null_p95=%8.3f  completeness=%s'
                  % (name, m['null_max_p95'],
                     {d: c for d, c in m['completeness'].items()}),
                  flush=True)

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
