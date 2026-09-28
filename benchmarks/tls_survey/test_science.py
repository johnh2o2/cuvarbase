"""Independent CPU checks of the survey's statistical and filter diagnostics.

Small exhaustive searches and dense covariance calculations are deliberately
independent of the optimized production diagnostic algorithms.
"""
import importlib.util
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import binom, multinomial


HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


common = _load('survey_science_common', HERE / 'common.py')
with patch.dict(sys.modules, {'common': common}):
    analysis = _load('survey_science_analysis', HERE / 'analyze.py')
    development = _load('survey_science_development', HERE / 'development.py')


def _projected_filter(signal, template, errors):
    """Whiten, project out the constant using least squares, then correlate."""
    constant = 1 / np.asarray(errors)
    whitened = np.asarray(template) / errors
    coefficient = np.linalg.lstsq(constant[:, None], whitened, rcond=None)[0]
    direction = whitened - constant * coefficient[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-12 * max(1., np.linalg.norm(whitened)):
        return 0.
    return max(0., float(np.dot(signal / errors, direction) / norm))


@pytest.mark.parametrize('n', [1, 19, 64, 256])
def test_binomial_intervals_include_exact_zero_and_all_success_limits(n):
    tail = .05 / 2
    np.testing.assert_allclose(analysis.binomial_interval(0, n),
                               [0., 1 - tail ** (1 / n)], rtol=2e-14)
    np.testing.assert_allclose(analysis.binomial_interval(n, n),
                               [tail ** (1 / n), 1.], rtol=2e-14)
    assert analysis.binomial_interval(0, 0) == [0., 1.]


@pytest.mark.parametrize('k,n', [(1, 19), (3, 10), (32, 64), (120, 128)])
def test_clopper_pearson_endpoints_invert_binomial_tail_probabilities(k, n):
    lower, upper = analysis.binomial_interval(k, n)
    assert binom.sf(k - 1, n, lower) == pytest.approx(.025, abs=2e-14)
    assert binom.cdf(k, n, upper) == pytest.approx(.025, abs=2e-14)


def test_binomial_intervals_have_nominal_or_greater_finite_sample_coverage():
    n = 20
    intervals = np.array([analysis.binomial_interval(k, n) for k in range(n + 1)])
    for probability in np.linspace(0., 1., 101):
        included = (intervals[:, 0] <= probability) & (probability <= intervals[:, 1])
        coverage = np.sum(binom.pmf(np.arange(n + 1), n, probability)[included])
        assert coverage >= .95 - 1e-13


def test_paired_intervals_use_discordance_and_keep_finite_sample_uncertainty():
    first = np.r_[np.ones(12, bool), np.zeros(8, bool)]
    second = np.r_[np.ones(9, bool), np.zeros(11, bool)]
    forward = analysis.paired_interval(first, second)
    reverse = analysis.paired_interval(second, first)
    assert (forward['first_only'], forward['second_only']) == (3, 0)
    assert forward['difference'] == .15
    np.testing.assert_allclose(reverse['interval'], -np.array(forward['interval'])[::-1])
    same = analysis.paired_interval(first, first)
    assert same['difference'] == 0
    assert same['interval'][0] < 0 < same['interval'][1]
    # Equal marginal recoveries do not imply paired equivalence: disjoint
    # discoveries have a wider interval than exact paired agreement.
    disjoint = analysis.paired_interval(first, first[::-1])
    assert np.ptp(disjoint['interval']) > np.ptp(same['interval'])


@pytest.mark.parametrize('probabilities', [(0.1, 0.2, 0.3, 0.4),
                                         (0.7, 0.01, 0.04, 0.25)])
def test_paired_interval_coverage_by_exhaustive_multinomial_outcomes(probabilities):
    # Categories: both, first only, second only, neither. Enumerate every
    # possible table rather than treating the two recovery rates as unpaired.
    n = 6
    truth = probabilities[1] - probabilities[2]
    covered = 0.
    for both in range(n + 1):
        for first_only in range(n - both + 1):
            for second_only in range(n - both - first_only + 1):
                neither = n - both - first_only - second_only
                first = [True] * (both + first_only) + [False] * (second_only + neither)
                second = ([True] * both + [False] * first_only +
                          [True] * second_only + [False] * neither)
                bounds = analysis.paired_interval(first, second)['interval']
                if bounds[0] <= truth <= bounds[1]:
                    covered += multinomial.pmf(
                        [both, first_only, second_only, neither], n, probabilities)
    assert covered >= .95 - 1e-13


@pytest.mark.parametrize('first,second', [([], []), ([True], []), ([[True]], [[True]])])
def test_paired_intervals_reject_empty_or_misaligned_populations(first, second):
    with pytest.raises(ValueError, match='Paired nonempty aligned'):
        analysis.paired_interval(first, second)


@pytest.mark.parametrize('alpha,n', [(.05, 19), (.05, 256), (.01, 99), (.01, 256)])
def test_conformal_threshold_has_exchangeable_rank_control_and_strict_ties(alpha, n):
    values = np.arange(n + 1, dtype=float)
    # Hold every possible observation out once. Every ordering is represented
    # because the statistic depends only on ranks, so this is exact coverage.
    for population in (values, np.floor(values / 3)):
        exceedances = 0
        for index in range(n + 1):
            cutoff = analysis.threshold(np.delete(population, index), alpha)
            exceedances += population[index] > cutoff['value']
        assert exceedances / (n + 1) <= alpha
    result = analysis.threshold(np.ones(n), alpha)
    assert result['value'] == 1.
    assert result['decision'] == 'strict exceedance'
    assert result['attainable_marginal_fpr'] <= alpha


@pytest.mark.parametrize('values,alpha', [([], .05), ([np.nan] * 20, .05),
                                       ([np.inf] * 20, .05), (range(18), .05),
                                       (range(98), .01)])
def test_calibration_cannot_drop_invalid_nulls_or_invent_finer_resolution(values, alpha):
    with pytest.raises(ValueError):
        analysis.threshold(values, alpha)


def test_detection_threshold_ties_and_failures_remain_nondetections():
    def row(value, recovered=True, valid=True):
        return dict(valid=valid, candidates={'native': dict(score=value, recovered=recovered)})
    rows = [row(8.), row(np.nextafter(8., np.inf)), row(10., recovered=False),
            row(20., valid=False), row(None, valid=False)]
    np.testing.assert_array_equal(analysis.detections(rows, 'native', 8.),
                                  [False, True, False, False, False])
    np.testing.assert_array_equal(analysis.detections(rows, 'native', 8., null=True),
                                  [False, True, True, False, False])


@pytest.mark.parametrize('tau', [.0001, .3, 200.])
@pytest.mark.parametrize('amplitude', [0., .05, 2.])
def test_ou_recursion_matches_dense_covariance_with_gaps_ties_and_heterogeneous_errors(tau, amplitude):
    rng = np.random.default_rng(1691)
    times = np.array([4., 0., .02, 4., 500., .4, 1., .02, 3., 70., .03])
    errors = rng.uniform(.1, 1., len(times))
    signal = np.array([.2, 0., .8, .1, 0., .3, 1., .4, .8, 0., .2])
    template = signal + rng.uniform(0, .5, len(times))
    weights = errors ** -2
    coefficients = weights * (template - np.average(template, weights=weights))
    covariance = np.diag(errors ** 2) + amplitude ** 2 * np.exp(
        -np.abs(times[:, None] - times[None, :]) / tau)
    expected = max(0., np.dot(coefficients, signal) /
                   np.sqrt(coefficients @ covariance @ coefficients))
    actual = development.ou_filter_snr(times, signal, template, errors, amplitude, tau)
    assert actual == pytest.approx(expected, rel=2e-13)
    if amplitude == 0:
        assert actual == pytest.approx(_projected_filter(signal, template, errors), rel=2e-13)


@pytest.mark.parametrize('seed', [11, 42, 918])
def test_centered_optimal_box_matches_every_admissible_interval(seed):
    rng = np.random.default_rng(seed)
    count = 21
    phase = np.linspace(-.5, .5, count)
    signal = np.zeros(count)
    signal[8:13] = rng.uniform(.05, 1., 5)
    signal[10] = 0.  # Sampling/exposure differences need not give a smooth row.
    errors = rng.uniform(.8, 1.2, count)
    permutation = rng.permutation(count)
    phase, signal, errors = (array[permutation] for array in (phase, signal, errors))
    order = np.argsort(phase)
    best = 0.
    weights = errors ** -2
    for start in range(count):
        for end in range(start + 1, count + 1):
            template = np.zeros(count)
            template[order[start:end]] = 1.
            if np.dot(weights, template) < .5 * np.sum(weights):
                best = max(best, _projected_filter(signal, template, errors))
    score, template = development.optimal_box(phase, signal, errors)
    assert score == pytest.approx(best, rel=2e-13)
    assert score == pytest.approx(_projected_filter(signal, template, errors), rel=2e-13)


def test_box_certificate_rejects_weight_dominated_support_and_handles_no_signal():
    phase = np.arange(9.)
    score, template = development.optimal_box(phase, np.zeros(9), np.ones(9))
    assert score == 0.
    assert not template.any()
    with pytest.raises(ValueError, match='support exceeds half the weight'):
        development.optimal_box(phase, np.r_[1., np.zeros(8)], np.r_[.01, np.ones(8)])


@pytest.mark.parametrize('count', [15, 16, 23])
def test_native_fft_family_matches_direct_cyclic_template_enumeration(count):
    rng = np.random.default_rng(910 + count)
    times = rng.uniform(.1, 20., count)
    period = 2.37
    errors = rng.uniform(.4, 1.7, count)
    order = np.argsort((times % period) / period)
    widths = [1, 2, 4, 7, count, count + 2]
    deficits = np.zeros((len(widths), count + 2), dtype=np.float32)
    for row, width in enumerate(widths):
        deficits[row, :width] = rng.uniform(.1, 1., width)
    deficits[2, :4] = [.2, .7, .6, 1.]  # Include literal native padding deficit.
    deficits[-2, :count] = 1.  # Pure constant has zero identifiable signal.
    signal = np.zeros(count)
    signal[order[(count - 2 + np.arange(4)) % count]] = [.2, .7, .6, 1.]
    best = 0.
    for row, width in enumerate(widths):
        if width > count:
            continue
        for start in range(count):
            template = np.zeros(count)
            template[order[(start + np.arange(width)) % count]] = deficits[row, :width]
            best = max(best, _projected_filter(signal, template, errors))
    actual, template, winner = development.optimal_native_family(
        times, period, signal, errors, dict(widths=widths, template_deficits=deficits))
    assert winner is not None
    assert actual == pytest.approx(best, rel=5e-13)
    assert actual == pytest.approx(_projected_filter(signal, template, errors), rel=5e-13)


def test_recovery_boundary_is_closed_and_harmonics_are_separate():
    metadata = dict(truth_period=2., baseline_days=16., duration_days=.5)
    boundary = 2.03125  # Exactly representable half-duration drift.
    assert common.recovered(boundary, metadata)
    assert not common.recovered(np.nextafter(boundary, np.inf), metadata)
    assert not common.recovered(1., metadata)
    assert common.recovered(1., metadata, aliases=True)
    assert not common.recovered(None, metadata)
    assert not common.recovered(np.nan, metadata)


def test_fixed_grid_policy_protects_declared_thin_regimes_without_truth_adaptation():
    expected = {'tess_highimpact': 9, 'ztf_highimpact': 9,
                'tess_eccentric': 9, 'hatpi_short': 9, 'tess_grazing_smeared': 24}
    for name, settings in common.REGIMES.items():
        assert settings.get('grid_oversampling', 3) == expected.get(name, 3)
    assert [common.BLS_CONFIGS[name]['qmin_factor'] for name in
            ('bls_medium', 'bls_fine', 'bls_finest')] == [1., .5, .25]


def test_fixed_grids_resolve_maximum_impact_and_eccentricity_design_boundaries():
    pytest.importorskip('batman')
    from cuvarbase import tls_reference_math as reference
    physics = _load('survey_science_physics', common.ROOT / 'benchmarks/tls_accuracy/diagnose.py')
    baseline = 90.
    for name, settings in common.REGIMES.items():
        radius, mass = settings.get('radius', 1.), settings.get('mass', 1.)
        lower, upper = settings.get('period', (2., 6.))
        grid = np.sort(reference.period_grid(
            baseline, R_star=radius, M_star=mass, period_min=.5 * lower,
            period_max=1.2 * upper, oversampling_factor=settings.get('grid_oversampling', 3)))
        for target in (lower, math.sqrt(lower * upper), upper):
            index = np.searchsorted(grid, target)
            before, after = grid[index - 1:index + 1]
            midpoint = (before + after) / 2
            model = physics.Regime(name, midpoint, radius=radius, mass=mass,
                rp=.00916 / radius, impact=max(settings['impact']),
                eccentricity=max(settings.get('eccentricity', (0., 0.))))
            duration, _, _ = physics.durations(model)
            # Every period in this grid interval is at most half a grid step
            # from a trial; this samples the declared physical boundary, not
            # the favorable realized injection coordinates.
            worst_nearest_drift = (after - before) / (2 * midpoint) * baseline
            assert worst_nearest_drift < .25 * duration, name
