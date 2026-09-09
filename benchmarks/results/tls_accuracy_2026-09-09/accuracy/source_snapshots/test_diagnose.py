"""Independent mathematical checks for the CPU SNR diagnostic."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('tls_accuracy_diagnose', Path(__file__).with_name('diagnose.py'))
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)


def test_expected_snr_uses_actual_regularized_filter_variance():
    signal = np.array([.3, .8, .2])
    filt = np.array([.1, 1., .4])
    errors = np.array([.2, .05, .1])
    regularizer = .01
    coefficients = filt/(errors**2+regularizer)
    expected = (signal@coefficients)/np.linalg.norm(errors*coefficients)
    assert d.expected_snr(signal, filt, errors, regularizer) == pytest.approx(expected)
    assert expected <= np.linalg.norm(signal/errors)
    assert d.expected_snr(signal, signal, errors, 0.) == pytest.approx(np.linalg.norm(signal/errors))


def test_projected_bin_signal_obeys_information_loss_identity():
    signal = np.array([.2, .5, 1., .1, .4, .9])
    errors = np.array([.3, .2, .1, .1, .4, .3])
    groups = np.array([0, 0, 0, 1, 1, 2])
    weights = 1/errors**2
    sums = np.bincount(groups, weights=weights*signal)
    count = np.bincount(groups, weights=weights)
    binned_filter = (sums/count)[groups]
    coarse_squared = d.expected_snr(signal, binned_filter, errors, 0.)**2
    oracle_squared = np.dot(weights, signal*signal)
    lost = np.dot(weights, (signal-binned_filter)**2)
    assert coarse_squared+lost == pytest.approx(oracle_squared)
    assert coarse_squared <= oracle_squared


def test_uniform_projection_has_analytic_triangle_loss():
    # A triangle over [-1, 1] has integral(s^2)=2/3. Each of two
    # unit-width bins contains signal integral 1/2, giving SNR^2=1/2.
    signal = d.UniformSignal(np.array([-1., 0., 1.]), np.array([0., 1., 0.]))
    assert signal.oracle**2 == pytest.approx(2/3)
    assert signal.projection_retention(1., 0.) == pytest.approx(np.sqrt(3/4))
    flat = d.UniformSignal(np.array([-1., 0., 1.]), np.ones(3))
    assert flat.projection_retention(1., 0.) == pytest.approx(1.)


@pytest.mark.parametrize('impact', [0., .8, .95])
def test_contact_solver_matches_exact_circular_geometry(impact):
    regime = d.Regime('geometry', 10., rp=.025, impact=impact)
    duration, full, a = d.durations(regime)
    expected = 10/np.pi*np.arcsin(np.sqrt((1+.025)**2-impact**2)/np.sqrt(a*a-impact*impact))
    expected_full = 10/np.pi*np.arcsin(np.sqrt((1-.025)**2-impact**2)/np.sqrt(a*a-impact*impact))
    assert duration == pytest.approx(expected, rel=1e-10)
    assert full == pytest.approx(expected_full, rel=1e-10)


def test_optimal_box_recovers_analytic_trapezoid_width():
    # Unit full width; each ingress occupies fraction a. The best box
    # excludes part of each ingress and has an analytic stationary point.
    a = .1
    x = np.linspace(-2., 2., 40001)
    signal = np.clip((.5-np.abs(x))/a, 0., 1.)
    model = d.UniformSignal(x, signal)
    fit, snr = d.best_uniform_box(model)
    flat = 1-2*a
    z = (-(2*flat-2*a)+np.sqrt((2*flat-2*a)**2+12*a*flat))/6
    width = flat+2*z
    area = flat+2*z-z*z/a
    assert fit[0] == pytest.approx(0., abs=2e-5)
    assert fit[1] == pytest.approx(width, abs=3e-5)
    assert snr == pytest.approx(area/np.sqrt(width), rel=1e-7)
    full_width_snr = (1-a)
    assert snr > full_width_snr


def test_observed_box_exhaustive_result_matches_all_intervals():
    rng = np.random.default_rng(15)
    phase = rng.uniform(-1., 1., 21)
    signal = np.maximum(0., 1-3*np.abs(phase))
    errors = rng.uniform(.05, .2, len(phase))
    optimum = d.best_observed_box(phase, signal, errors)
    sorted_phase = np.sort(phase)
    brute = max(d.expected_snr(signal, ((phase >= lo) & (phase <= hi)).astype(float), errors)
                for lo in sorted_phase for hi in sorted_phase if hi >= lo)
    assert optimum == pytest.approx(brute, rel=1e-13)


def test_automatic_bins_guarantee_minimum_until_cap():
    for q in np.geomspace(4/8192, .1, 100):
        bins, requested = d.automatic_bins(q, 4.)
        assert bins == requested
        assert q*bins >= 4.-1e-12
    bins, requested = d.automatic_bins(1e-4, 4.)
    assert bins == 8192
    assert requested == 65536
    assert 1e-4*bins < 1.


def test_coarse_normalization_distinguishes_mean_square_from_square_mean():
    # One bin with a half-on box template. Actual filter noise variance
    # is mean(T)^2=1/4, while the native denominator is mean(T^2)=1/2.
    value = d.metrics(.25, .25, .5)
    assert value['snr'] == pytest.approx(.5)
    assert np.sqrt(value['native_score']) == pytest.approx(.5/np.sqrt(2))
    assert value['native_norm_over_noise'] == pytest.approx(np.sqrt(2))


def test_bin_integrals_match_independent_dense_point_filter():
    source = Path(__file__).resolve().parents[2]/'cuvarbase/tls_models.py'
    template = d.Template(source)
    x = np.linspace(-2., 2., 800001)
    signal = template.point(x, 0., 1.)
    model = d.UniformSignal(x, signal)
    h, offset = .25, .31
    value = model.binned(template, 0., 1., h, offset)
    index = np.floor(x/h+offset)
    lo = (index-offset)*h
    averaged, squared = template.averages(lo, lo+h, 0., 1.)
    variance = np.trapz(averaged*averaged, x)
    numerator = np.trapz(signal*averaged, x)
    native_den = np.trapz(squared, x)
    assert value['snr'] == pytest.approx(numerator/np.sqrt(variance), rel=2e-5)
    assert value['native_score'] == pytest.approx(numerator*numerator/native_den, rel=2e-5)


def test_zero_signal_epoch_pruning_matches_complete_grid():
    source = Path(__file__).resolve().parents[2]/'cuvarbase/tls_models.py'
    template = d.Template(source)
    x = np.linspace(-2., 2., 10001)
    model = d.UniformSignal(x, template.point(x, 0., 1.))
    qtrue, epoch, bins = .02, .413, 256
    h = 1/(qtrue*bins)
    offset = epoch*bins % 1
    evaluate = lambda c, w: model.binned(template, c, w, h, offset)
    native, best_snr = d.correct_period_grid(evaluate, .01, .04, qtrue,
                                            epoch, 3., 3, 1+2*h)
    complete = []
    for q in np.geomspace(.01, .04, 3):
        n = d.epoch_trials(q, 3.)
        complete.extend(evaluate((j/n-epoch)/qtrue, q/qtrue) for j in range(n))
    assert native['native_score'] == pytest.approx(max(r['native_score'] for r in complete))
    assert best_snr['snr'] == pytest.approx(max(r['snr'] for r in complete))
