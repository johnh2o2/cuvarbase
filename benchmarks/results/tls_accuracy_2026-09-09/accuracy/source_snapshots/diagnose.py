#!/usr/bin/env python3
"""CPU expected-SNR diagnostics for the fast TLS approximation.

This is a known-period, noiseless-signal calculation, not a recovery test or
an implementation of GTLS. Expected filter SNR uses its actual white-noise
variance. Native coarse TLS scores are evaluated separately because their
bin-averaged T-squared normalization is not that variance.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import batman
import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, differential_evolution, minimize


G, MSUN, RSUN, REARTH = 6.67430e-11, 1.98840e30, 6.95700e8, 6.371e6
LD = [.4804, .1867]


@dataclass(frozen=True)
class Regime:
    name: str
    period: float
    radius: float = 1.
    mass: float = 1.
    rp: float = .00916
    impact: float = 0.
    eccentricity: float = 0.
    omega: float = 90.
    exposure_seconds: float = 200.


REGIMES = [
    Regime('sun_jupiter_5d', 5., rp=.1),
    Regime('sun_subneptune_10d', 10., rp=.025, impact=.5),
    Regime('sun_earth_10d', 10.),
    Regime('sun_earth_10d_b08', 10., impact=.8),
    Regime('sun_earth_10d_b095', 10., impact=.95),
    Regime('sun_earth_10d_grazing', 10., impact=1.),
    Regime('sun_jupiter_5d_grazing', 5., rp=.1, impact=1.05),
    Regime('sun_earth_365d', 365.25),
    Regime('sun_earth_365d_b08', 365.25, impact=.8),
    Regime('sun_earth_1000d', 1000.),
    Regime('sun_earth_100d_e08', 100., impact=.5, eccentricity=.8),
    Regime('mdwarf02_earth_10d', 10., radius=.2, mass=.2, rp=.00916/.2),
    Regime('mdwarf01_earth_30d', 30., radius=.1, mass=.1, rp=.00916/.1),
    Regime('mdwarf01_earth_100d', 100., radius=.1, mass=.1, rp=.00916/.1),
    Regime('mdwarf01_earth_365d', 365.25, radius=.1, mass=.1, rp=.00916/.1),
    Regime('mdwarf01_earth_365d_b09', 365.25, radius=.1, mass=.1,
           rp=.00916/.1, impact=.9),
    Regime('white_dwarf_earth_1d', 1., radius=.012, mass=.6,
           rp=.00916/.012, exposure_seconds=30.),
    Regime('white_dwarf_earth_10d', 10., radius=.012, mass=.6,
           rp=.00916/.012, exposure_seconds=30.),
    Regime('sun_earth_10d_30min', 10., exposure_seconds=1800.),
]

CONFIGS = [
    ('api_default', 3., 15, None),
    ('benchmark_original', 4., 16, None),
    ('benchmark_fine', 16., 32, 8192),
]


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Template:
    def __init__(self, source):
        module = source_module(source, 'diagnostic_tls_models')
        if not module.BATMAN_AVAILABLE:
            raise RuntimeError('A physical batman template is required.')
        self.t, self.s1, self.s2 = [np.asarray(x, dtype=np.float64) for x in
                                  module.generate_template_tables(u=LD)]
        self.knots = np.linspace(-1., 1., len(self.t))
        # Exact integral of the squared piecewise-linear point template.
        self.point_norm = float(np.sum(np.diff(self.knots) *
                                      (self.t[:-1]**2+self.t[:-1]*self.t[1:]+self.t[1:]**2)/3))

    def point(self, x, center, width):
        return np.interp(2 * (x-center) / width, self.knots, self.t,
                         left=0., right=0.)

    def averages(self, lo, hi, center, width):
        a, b = 2*(lo-center)/width, 2*(hi-center)/width
        return tuple((np.interp(b, self.knots, table) -
                      np.interp(a, self.knots, table)) / (b-a)
                     for table in (self.s1, self.s2))


def parameters(regime, epoch=0.):
    p = batman.TransitParams()
    p.t0, p.per, p.rp = epoch, regime.period, regime.rp
    p.a = (G*MSUN*regime.mass*(p.per*86400)**2/(4*np.pi**2))**(1/3)
    p.a /= regime.radius*RSUN
    p.ecc, p.w = regime.eccentricity, regime.omega
    cosi = regime.impact / p.a * (1+p.ecc*np.sin(np.radians(p.w))) / (1-p.ecc**2)
    p.inc = float(np.degrees(np.arccos(cosi)))
    p.u, p.limb_dark = LD, 'quadratic'
    return p


def separation(t, p):
    """Projected separation in stellar radii; t0 is inferior conjunction.

    This independently solves Kepler's equation and is used only to locate
    geometric contacts. All fluxes come from batman. Catalog eccentric cases
    have omega=90 degrees, so inferior conjunction is also minimum separation.
    """
    omega = np.radians(p.w)
    f0 = np.pi/2 - omega
    e0 = 2*np.arctan2(np.sqrt(1-p.ecc)*np.sin(f0/2),
                      np.sqrt(1+p.ecc)*np.cos(f0/2))
    mean = e0-p.ecc*np.sin(e0) + 2*np.pi*(np.asarray(t)-p.t0)/p.per
    eccentric = mean.copy()
    for _ in range(30):
        step = (eccentric-p.ecc*np.sin(eccentric)-mean)/(1-p.ecc*np.cos(eccentric))
        eccentric -= step
        if np.max(np.abs(step)) < 1e-14:
            break
    anomaly = 2*np.arctan2(np.sqrt(1+p.ecc)*np.sin(eccentric/2),
                           np.sqrt(1-p.ecc)*np.cos(eccentric/2))
    r = p.a*(1-p.ecc*np.cos(eccentric))
    angle = anomaly+omega
    return r*np.sqrt(np.cos(angle)**2 + np.cos(np.radians(p.inc))**2*np.sin(angle)**2)


def durations(regime):
    p = parameters(regime)
    guess = p.per/np.pi*np.arcsin(np.sqrt((1+p.rp)**2-regime.impact**2)/
                                (p.a*np.sin(np.radians(p.inc))))
    guess *= np.sqrt(1-p.ecc**2)/(1+p.ecc*np.sin(np.radians(p.w)))

    def contacts(level):
        if float(separation(0., p)) >= level:
            return 0., 0.
        left = brentq(lambda t: float(separation(t, p))-level, -2*guess, 0., xtol=1e-14)
        right = brentq(lambda t: float(separation(t, p))-level, 0., 2*guess, xtol=1e-14)
        return left, right

    t1, t4 = contacts(1+p.rp)
    t2, t3 = contacts(abs(1-p.rp))
    return t4-t1, max(0., t3-t2), p.a


def physical_signal(regime, times, exposures, epoch=0., exposure_nodes=64):
    """Exposure integrals using Gauss-Legendre quadrature, in float64."""
    p = parameters(regime, epoch)
    times = np.asarray(times, dtype=np.float64)
    exposures = np.broadcast_to(np.asarray(exposures), times.shape)
    result = np.empty_like(times)
    nodes, weights = np.polynomial.legendre.leggauss(exposure_nodes)
    for exposure in np.unique(exposures):
        take = exposures == exposure
        if exposure == 0:
            result[take] = 1-batman.TransitModel(p, times[take]).light_curve(p)
        else:
            t = (times[take, None]+exposure*.5*nodes[None, :]).ravel()
            flux = batman.TransitModel(p, t).light_curve(p).reshape((-1, exposure_nodes))
            result[take] = np.dot(1-flux, weights*.5)
    return result


def expected_snr(signal, filt, errors, regularizer=1e-10):
    weights = 1/(errors*errors+regularizer)
    numerator = np.dot(weights*signal, filt)
    variance = np.dot(weights*weights*errors*errors, filt*filt)
    return float(numerator/np.sqrt(variance)) if variance > 0 else 0.


class UniformSignal:
    def __init__(self, x, signal):
        self.x, self.signal = np.asarray(x), np.asarray(signal)
        self.cumulative = cumulative_trapezoid(self.signal, self.x, initial=0.)
        self.oracle = float(np.sqrt(np.sum(np.diff(self.x) *
                            (self.signal[:-1]**2+self.signal[:-1]*self.signal[1:]+
                             self.signal[1:]**2)/3)))

    def integral(self, lo, hi):
        def antiderivative(z):
            z = np.clip(z, self.x[0], self.x[-1])
            index = np.clip(np.searchsorted(self.x, z, side='right')-1,
                            0, len(self.x)-2)
            delta = z-self.x[index]
            slope = ((self.signal[index+1]-self.signal[index]) /
                     (self.x[index+1]-self.x[index]))
            return self.cumulative[index]+self.signal[index]*delta+.5*slope*delta*delta
        return antiderivative(hi)-antiderivative(lo)

    def point_snr(self, template, center, width):
        f = template.point(self.x, center, width)
        # Integrate the whole filter even if a very wide duration prior
        # extends outside the compact grid containing the signal.
        variance = .5*width*template.point_norm
        return float(np.trapz(self.signal*f, self.x)/np.sqrt(variance)) if variance > 0 else 0.

    def binned(self, template, center, width, binwidth, offset):
        first = int(np.floor((center-.5*width)/binwidth+offset))
        last = int(np.ceil((center+.5*width)/binwidth+offset))
        lo = (np.arange(first, last)-offset)*binwidth
        hi = lo+binwidth
        f, f2 = template.averages(lo, hi, center, width)
        numerator = float(np.dot(self.integral(lo, hi), f))
        variance = float(np.dot(f, f)*binwidth)
        native_den = float(f2.sum()*binwidth)
        return metrics(numerator, variance, native_den)

    def projection_retention(self, binwidth, offset):
        """Best SNR possible from bin sums, given the true signal shape."""
        first = int(np.floor(self.x[0]/binwidth+offset))
        last = int(np.ceil(self.x[-1]/binwidth+offset))
        lo = (np.arange(first, last)-offset)*binwidth
        integrals = self.integral(lo, lo+binwidth)
        return float(np.sqrt(np.dot(integrals, integrals)/binwidth)/self.oracle)


def metrics(numerator, variance, native_den):
    return dict(snr=numerator/np.sqrt(variance) if variance > 0 else 0.,
                native_score=numerator*numerator/native_den if native_den > 0 else 0.,
                native_norm_over_noise=np.sqrt(native_den/variance) if variance > 0 else np.nan)


def best_uniform_box(signal):
    """Optimize both box boundaries through its epoch and duration."""
    bounds = [(-.4, .4), (.08, 2.5)]
    def box_objective(z):
        center, width = z
        return -float(signal.integral(center-.5*width, center+.5*width)/np.sqrt(width))

    box = differential_evolution(box_objective, bounds, seed=114,
                                 tol=1e-10, polish=True)
    if not box.success:
        raise RuntimeError(box.message)
    return box.x, -float(box.fun)


def best_uniform_fits(signal, template):
    """Continuous fits; the box's width and epoch are both free."""
    tls = minimize(lambda z: -signal.point_snr(template, *z), [0., 1.],
                   method='Powell', bounds=[(-.4, .4), (.08, 2.5)],
                   options={'xtol': 1e-8, 'ftol': 1e-10})
    if not tls.success:
        raise RuntimeError(tls.message)
    box_fit, box_snr = best_uniform_box(signal)
    return tls.x, -float(tls.fun), box_fit, box_snr


def automatic_bins(qmin, oversampling, cap=8192):
    need = oversampling/max(qmin, 1e-6)
    requested = 2**int(np.ceil(np.log2(max(256., need))))
    return min(cap, requested), requested


def epoch_trials(q, oversampling):
    return min(20000, max(30, int(np.ceil(oversampling/q))))


def correct_period_grid(evaluator, qmin, qmax, qtrue, epoch_phase,
                        oversampling, n_durations, support_width):
    """Expected-score maximum near the transit, on the complete native grid.

    Trials whose window does not overlap the deterministic signal have zero
    numerator and cannot win. Pruning those zero-overlap trials is exact for
    this noiseless calculation, not an acceleration usable for real searches.
    support_width is in units of the geometric transit duration.
    """
    winner_native = None
    winner_snr = None
    for q in np.geomspace(qmin, qmax, n_durations):
        width = q/qtrue
        n = epoch_trials(q, oversampling)
        half = .5*(width+support_width)*qtrue
        first = int(np.floor((epoch_phase-half)*n))-1
        last = int(np.ceil((epoch_phase+half)*n))+1
        for index in range(first, last+1):
            center = (index/n-epoch_phase)/qtrue
            value = dict(evaluator(center, width), center=center, width=width)
            if winner_native is None or value['native_score'] > winner_native['native_score']:
                winner_native = value
            if winner_snr is None or value['snr'] > winner_snr['snr']:
                winner_snr = value
    return winner_native, winner_snr


def flags(regime, duration, ingress, qmin, bins, requested, oversampling):
    q = duration/regime.period
    return dict(q=q, duration_hours=duration*24, ingress_minutes=ingress*1440,
                qmin=qmin, bins=bins, requested_bins=requested,
                bins_across_transit=q*bins,
                bins_across_ingress=ingress/regime.period*bins,
                bin_cap=requested > bins,
                epoch_cap=oversampling/qmin > 20000,
                duration_below_prior=q < qmin,
                depth_gate_caveat=regime.rp > .5,
                stellar_density_solar=regime.mass/regime.radius**3)


def uniform_cases(template, grids, offsets, samples_per_transit, exposure_nodes):
    rng = np.random.default_rng(20260909)
    rows = []
    for regime in REGIMES:
        duration, full_duration, a = durations(regime)
        ingress = .5*(duration-full_duration)
        qtrue = duration/regime.period
        x = np.linspace(-2., 2., 4*samples_per_transit+1)
        flux = physical_signal(regime, x*duration, regime.exposure_seconds/86400,
                               exposure_nodes=exposure_nodes)
        physical = UniformSignal(x, flux)
        own = UniformSignal(x, template.point(x, 0., 1.))
        fit, fit_snr, box_fit, box_snr = best_uniform_fits(physical, template)
        qmin, qmax = grids.duration_window(np.array([regime.period]),
                                          R_star=regime.radius, M_star=regime.mass)
        qmin, qmax = float(qmin[0]), min(float(qmax[0]), .333)
        for label, oversampling, nd, fixed_bins in CONFIGS:
            auto, requested = automatic_bins(qmin, oversampling)
            bins = auto if fixed_bins is None else fixed_bins
            duration_snrs = [physical.point_snr(template, fit[0], q/qtrue)
                             for q in np.geomspace(qmin, qmax, nd)]
            for i, offset in enumerate(np.arange(offsets)/offsets):
                # Cover every sub-bin offset and independently vary the
                # integer bin index, to sample epoch-grid alignment too.
                index = int(rng.integers(bins//4, 3*bins//4))
                epoch = (index+offset)/bins
                h = 1/(qtrue*bins)
                binned = physical.binned(template, *fit, h, offset)
                own_binned = own.binned(template, 0., 1., h, offset)
                uncapped_h = 1/(qtrue*max(bins, requested))
                uncapped = own.binned(template, 0., 1., uncapped_h, offset)
                # Use the fixed geometric width for this isolated epoch
                # test. Applying ceil(m/q) to a numerically fitted width
                # makes tiny optimizer jitter change the entire epoch grid.
                n = epoch_trials(qtrue, oversampling)
                nearest = int(round(epoch*n))
                epoch_snr = max(physical.point_snr(template, (j/n-epoch)/qtrue, 1.)
                                for j in (nearest-1, nearest, nearest+1))
                centered_width_snr = physical.point_snr(template, 0., 1.)
                evaluate = lambda c, w: physical.binned(template, c, w, h, offset)
                native, best_snr = correct_period_grid(
                    evaluate, qmin, qmax, qtrue, epoch, oversampling, nd,
                    1+regime.exposure_seconds/86400/duration+2*h)
                rows.append(dict(
                    kind='uniform', regime=regime.name, config=label, offset_index=i,
                    bin_phase_offset=offset, epoch_phase=epoch,
                    **asdict(regime), a_over_rstar=a,
                    **flags(regime, duration, ingress, qmin, bins, requested, oversampling),
                    template_fit_width_over_duration=float(fit[1]),
                    template_fit_center_over_duration=float(fit[0]),
                    box_fit_width_over_duration=float(box_fit[1]),
                    box_fit_center_over_duration=float(box_fit[0]),
                    template_snr_over_oracle=fit_snr/physical.oracle,
                    box_snr_over_oracle=box_snr/physical.oracle,
                    physical_projection_retention=physical.projection_retention(h, offset),
                    physical_binned_over_unbinned=binned['snr']/fit_snr,
                    own_template_binned_over_unbinned=own_binned['snr']/own.oracle,
                    own_template_uncapped_retention=uncapped['snr']/own.oracle,
                    native_norm_over_noise=binned['native_norm_over_noise'],
                    epoch_grid_only_retention=epoch_snr/centered_width_snr,
                    duration_grid_only_retention=max(duration_snrs)/fit_snr,
                    coarse_grid_native_selected_over_oracle=native['snr']/physical.oracle,
                    coarse_grid_best_snr_over_oracle=best_snr['snr']/physical.oracle,
                    coarse_grid_native_selected_over_best_template=native['snr']/fit_snr,
                    coarse_grid_native_amplitude_over_oracle=np.sqrt(native['native_score'])/physical.oracle,
                    coarse_selected_width_over_duration=native['width'],
                    coarse_selected_center_over_duration=native['center']))
        print('uniform', regime.name, 'complete', flush=True)
    return rows


def best_observed_box(phase, signal, errors):
    """Exhaust all nonempty contiguous intervals with signal at both ends.

    Including an outer zero-signal observation only adds variance; trimming
    it improves SNR. Thus these intervals contain a global optimum at the
    known period, with arbitrary duration and epoch. No duration-prior or
    trial-grid handicap is applied to this shape control.
    """
    order = np.argsort(phase)
    phase, signal, errors = phase[order], signal[order], errors[order]
    weights = 1/(errors*errors+1e-10)
    numerator = np.r_[0., np.cumsum(weights*signal)]
    variance = np.r_[0., np.cumsum(weights*weights*errors*errors)]
    positive = np.flatnonzero(signal > 0)
    best = 0.
    for j, start in enumerate(positive):
        ends = positive[j:]+1
        values = (numerator[ends]-numerator[start])/np.sqrt(variance[ends]-variance[start])
        best = max(best, float(values.max()))
    return best


def observed_cases(template, grids, cadence_dir, epochs, exposure_nodes):
    rng = np.random.default_rng(421990)
    regimes = [REGIMES[2], REGIMES[4],
               Regime('mdwarf01_earth_10d', 10., radius=.1, mass=.1, rp=.00916/.1)]
    rows = []
    for profile in ('tess_200s', 'tess_gap', 'ztf'):
        with np.load(cadence_dir/f'{profile}.npz') as data:
            times = np.array(data['t'], dtype=np.float64)
            errors = 1e-3*np.array(data['relative_error'], dtype=np.float64)
            exposures = np.array(data['exposure_days'], dtype=np.float64)
        origin = np.floor(times.min())
        for regime in regimes:
            duration, full_duration, a = durations(regime)
            qtrue = duration/regime.period
            qmin, qmax = grids.duration_window(np.array([regime.period]),
                                              R_star=regime.radius, M_star=regime.mass)
            qmin, qmax = float(qmin[0]), min(float(qmax[0]), .333)
            for epoch_index, epoch_phase in enumerate(rng.random(epochs)):
                epoch = origin+epoch_phase*regime.period
                signal = physical_signal(regime, times, exposures, epoch=epoch,
                                         exposure_nodes=exposure_nodes)
                phase = ((times-epoch+.5*regime.period) % regime.period)-.5*regime.period
                x = phase/duration
                oracle = float(np.sqrt(np.dot(signal/errors, signal/errors)))
                if oracle <= 0:
                    rows.append(dict(kind='observed', profile=profile, regime=regime.name,
                                     epoch_index=epoch_index, epoch_phase=epoch_phase,
                                     no_sampled_signal=True))
                    continue
                def objective(z):
                    return -expected_snr(signal, template.point(x, *z), errors)
                # Discrete cadences can create local optima; use a broad,
                # reproducible global fit followed by Powell refinement.
                fit_global = differential_evolution(objective, [(-.5, .5), (.08, 2.5)],
                                                     seed=714, tol=1e-7, popsize=12)
                fit = minimize(objective, fit_global.x, method='Powell',
                               bounds=[(-.5, .5), (.08, 2.5)],
                               options={'ftol': 1e-10, 'xtol': 1e-8})
                if fit.fun > fit_global.fun:
                    fit = fit_global
                direct_snr = -float(fit.fun)
                own_signal = template.point(x, 0., 1.)
                own_snr = expected_snr(own_signal, own_signal, errors)
                box_snr = best_observed_box(phase, signal, errors)
                weights = 1/(errors*errors+1e-10)
                for label, oversampling, nd, fixed_bins in CONFIGS:
                    auto, requested = automatic_bins(qmin, oversampling)
                    bins = auto if fixed_bins is None else fixed_bins
                    bindex = np.floor(((times-origin)/regime.period % 1)*bins).astype(int)
                    bphase = (((bindex+.5)/bins-epoch_phase+.5) % 1)-.5
                    blo = (bphase-.5/bins)/qtrue
                    bhi = (bphase+.5/bins)/qtrue
                    averaged, squared = template.averages(blo, bhi, *fit.x)
                    own_averaged, _ = template.averages(blo, bhi, 0., 1.)
                    binned_snr = expected_snr(signal, averaged, errors)
                    own_binned_snr = expected_snr(own_signal, own_averaged, errors)
                    noise_variance = float(np.dot(weights*weights*errors*errors, averaged*averaged))
                    native_den = float(np.dot(weights, squared))
                    bin_signal = np.bincount(bindex, weights=weights*signal, minlength=bins)
                    bin_variance = np.bincount(bindex, weights=weights*weights*errors*errors,
                                               minlength=bins)
                    valid_bins = bin_variance > 0
                    compressed_oracle = np.sqrt(np.sum(bin_signal[valid_bins]**2/bin_variance[valid_bins]))
                    rows.append(dict(kind='observed', profile=profile, regime=regime.name,
                                     config=label, epoch_index=epoch_index, epoch_phase=epoch_phase,
                                     no_sampled_signal=False, n_observations=len(times),
                                     n_signal_observations=int(np.sum(signal > 0)),
                                     n_transit_events=int(len(np.unique(np.floor((times[signal>0]-epoch)/
                                                                               regime.period+.5)))),
                                     **asdict(regime), a_over_rstar=a,
                                     **flags(regime, duration, .5*(duration-full_duration), qmin,
                                             bins, requested, oversampling),
                                     template_fit_width_over_duration=float(fit.x[1]),
                                     template_snr_over_oracle=direct_snr/oracle,
                                     box_snr_over_oracle=box_snr/oracle,
                                     physical_projection_retention=compressed_oracle/oracle,
                                     physical_binned_over_unbinned=binned_snr/direct_snr,
                                     own_template_binned_over_unbinned=own_binned_snr/own_snr if own_snr>0 else np.nan,
                                     native_norm_over_noise=np.sqrt(native_den/noise_variance) if noise_variance>0 else np.nan))
            print('observed', profile, regime.name, 'complete', flush=True)
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    keys = ('template_snr_over_oracle', 'box_snr_over_oracle',
            'physical_projection_retention',
            'physical_binned_over_unbinned', 'own_template_binned_over_unbinned',
            'own_template_uncapped_retention', 'native_norm_over_noise',
            'epoch_grid_only_retention', 'duration_grid_only_retention',
            'coarse_grid_native_selected_over_oracle',
            'coarse_grid_native_selected_over_best_template',
            'coarse_grid_native_amplitude_over_oracle')
    result = []
    groups = sorted(set((r['kind'], r.get('profile', ''), r['regime'], r.get('config', ''))
                        for r in rows if r.get('config')))
    for kind, profile, regime, config in groups:
        subset = [r for r in rows if (r['kind'], r.get('profile', ''), r['regime'],
                                     r.get('config')) == (kind, profile, regime, config)]
        row = dict(kind=kind, profile=profile, regime=regime, config=config, cases=len(subset))
        for flag in ('q', 'duration_hours', 'ingress_minutes', 'bins',
                     'bins_across_transit', 'bins_across_ingress', 'bin_cap',
                     'epoch_cap', 'duration_below_prior', 'depth_gate_caveat'):
            row[flag] = subset[0][flag]
        for key in keys:
            values = np.array([r[key] for r in subset if key in r], dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                for suffix, value in zip(('minimum', 'median', 'maximum'),
                                         np.quantile(values, [0., .5, 1.])):
                    row[f'{key}_{suffix}'] = float(value)
        result.append(row)
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--source-revision', default='HEAD',
                    help='Git revision supplying the model, grids, and kernel provenance (default: HEAD).')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cadences', type=Path)
    ap.add_argument('--offsets', type=int, default=32)
    ap.add_argument('--samples-per-transit', type=int, default=4096)
    ap.add_argument('--exposure-nodes', type=int, default=64)
    ap.add_argument('--observed-epochs', type=int, default=8)
    args = ap.parse_args()
    if (args.offsets < 2 or args.samples_per_transit < 128 or
            args.observed_epochs < 1 or args.exposure_nodes < 8):
        ap.error('Use at least two offsets, 128 samples per transit, one observed epoch, and eight exposure nodes.')
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', '-C', str(args.source_root), 'rev-parse',
                                        args.source_revision], text=True).strip()
    snapshot_dir = args.out/'source_snapshots'
    snapshot_dir.mkdir(exist_ok=True)
    # Freeze sources before a potentially long run; concurrent local edits
    # cannot alter the loaded mathematics or the recorded source hashes.
    snapshot_paths = []
    for relative in ('cuvarbase/tls_models.py', 'cuvarbase/tls_grids.py',
                     'cuvarbase/kernels/tls_fast.cu'):
        snapshot = snapshot_dir/Path(relative).name
        snapshot.write_bytes(subprocess.check_output(
            ['git', '-C', str(args.source_root), 'show', f'{revision}:{relative}']))
        snapshot_paths.append(snapshot)
    template_source, grids_source = snapshot_paths[:2]
    diagnostic_sha256 = sha256(Path(__file__).resolve())
    template = Template(template_source)
    grids = source_module(grids_source, 'diagnostic_tls_grids')
    rows = uniform_cases(template, grids, args.offsets, args.samples_per_transit,
                         args.exposure_nodes)
    if args.cadences:
        rows.extend(observed_cases(template, grids, args.cadences, args.observed_epochs,
                                    args.exposure_nodes))
    write_csv(args.out/'cases.csv', rows)
    summary = summarize(rows)
    write_csv(args.out/'summary.csv', summary)
    sources = snapshot_paths.copy()
    if args.cadences:
        sources += [args.cadences/f'{p}.npz' for p in ('tess_200s', 'tess_gap', 'ztf')]
    manifest = dict(
        scope='Expected white-noise SNR at the true period; not detection completeness, native SDE, or GTLS.',
        source_commit=revision,
        diagnostic_sha256=diagnostic_sha256,
        versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, batman=batman.__version__),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        limb_darkening=LD, configs=CONFIGS, regimes=[asdict(r) for r in REGIMES],
        source_sha256={str(path): sha256(path) for path in sources},
        output_sha256={name: sha256(args.out/name) for name in ('cases.csv', 'summary.csv')},
        rows=len(rows), no_sampled_signal_rows=sum(bool(r.get('no_sampled_signal')) for r in rows))
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Wrote', len(rows), 'rows to', args.out, flush=True)


if __name__ == '__main__':
    main()
