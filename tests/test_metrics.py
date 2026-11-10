"""
    Characterization tests for the per-region metrics that
    bpnet-predict reports (the functions live in bpnet.cli.predict),
    against formulas written out here
"""

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import multinomial, rankdata

pytest.importorskip("tensorflow")

from bpnet.cli import predict
from bpnet.utils.exceptionhandler import NoTracebackException

from tests.checks import gaussian_smooth

TRACKER_KEYS = ["profile_mnlls", "profile_cross_entropys", "profile_jsds",
                "profile_mses", "profile_pearsonrs", "profile_spearmanrs",
                "all_true_logcounts", "all_pred_logcounts"]


# --------------------------------------------------------------------
# the formulas
# --------------------------------------------------------------------

def multinomial_logpmf(counts, probs):
    counts = np.asarray(counts, dtype=np.float64)
    nonzero = counts > 0
    return gammaln(counts.sum() + 1) - gammaln(counts + 1).sum() + \
        (counts[nonzero] * np.log(probs[nonzero])).sum()


def mnll(counts, probs):
    """negative multinomial log likelihood per position"""
    return -multinomial_logpmf(counts, probs) / len(counts)


def cross_entropy(counts, probs):
    p = counts / counts.sum()
    return -(p * np.log(probs + 1e-7)).sum()


def js_distance(p, q):
    """square root of the Jensen-Shannon divergence (natural log)"""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return (a[keep] * np.log(a[keep] / b[keep])).sum()
    return np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m))


def normalize(value, best, worst):
    """0 at the worst bound, 1 at the best bound, clipped"""
    if best == worst:
        return 0.0
    return float(np.clip((value - worst) / (best - worst), 0.0, 1.0))


def fix_sum(probs):
    """what predict._fix_sum_to_one does"""
    probs = probs.copy()
    if probs.sum() > 1.0:
        probs[np.argmax(probs)] -= probs.sum() - 1.0
    if probs.sum() < 1.0:
        probs[np.argmin(probs)] += 1.0 - probs.sum()
    return probs


def expected_metrics(true, pred):
    """
        The metrics of one region and one track: correlations and the
        Jensen-Shannon distance compare the prediction with the
        smoothed observed profile (sigma 7, window 81); likelihood,
        cross entropy and squared error with the raw one. The three
        distribution metrics are rescaled so that 1 is the best
        achievable value (the observed profile itself) and 0 a uniform
        prediction or worse.
    """

    n = len(true)
    uniform = np.ones(n) / n
    true_prob = true / true.sum()
    smooth = gaussian_smooth(true, 7.0, 81)
    smooth_prob = smooth / smooth.sum()
    pred_prob = fix_sum(pred / pred.sum())

    if len(np.unique(smooth_prob)) == 1 or len(np.unique(pred_prob)) == 1:
        pearson = spearman = 0.0
    else:
        pearson = np.corrcoef(smooth_prob, pred_prob)[0, 1]
        spearman = np.corrcoef(rankdata(smooth_prob),
                               rankdata(pred_prob))[0, 1]

    best_mnll = mnll(true, fix_sum(true_prob))
    if best_bound_collapses(true):
        best_mnll = 0.0
    return {
        "profile_pearsonrs": pearson,
        "profile_spearmanrs": spearman,
        "profile_mnlls": normalize(mnll(true, pred_prob), best_mnll,
                                   mnll(true, uniform)),
        "profile_cross_entropys": normalize(
            cross_entropy(true, pred_prob), cross_entropy(true, true_prob),
            cross_entropy(true, uniform)),
        "profile_jsds": normalize(js_distance(smooth_prob, pred_prob), 0.0,
                                  js_distance(true_prob, uniform)),
        "profile_mses": np.mean((true - pred) ** 2),
    }


def best_bound_collapses(true):
    """
        Whether the likelihood score's best-case bound is replaced by 0
        for this profile in this environment.

        The bound is the likelihood of the observed profile under its
        own proportions, computed with scipy.stats.multinomial. Some
        scipy versions recompute the last probability as 1 minus the
        sum of the others; when the last position has no reads,
        round-off can make that a tiny negative number, scipy returns
        NaN, and the package then uses 0 as the bound. This is a quirk
        of the implementation, not part of the metric's definition
        (see test_likelihood_bound_does_not_depend_on_round_off); the
        check is made with scipy itself so that the tests state the
        mathematical value wherever the quirk does not strike.
    """

    probs = fix_sum(true / true.sum())
    value = multinomial(true.sum(), probs).logpmf(true)
    return not np.isfinite(value)


def assert_metrics_equal(got, expected, rtol=1e-8):
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], value, rtol=rtol, atol=1e-10,
                                   err_msg=key)


def profiles(seed, n=200, noise=0.0):
    """an observed profile with a peak and a prediction of its shape"""
    rng = np.random.RandomState(seed)
    x = np.arange(n)
    shape = 0.02 + np.exp(-0.5 * ((x - 0.4 * n) / 12.0) ** 2)
    true = rng.poisson(40.0 * shape).astype(np.float64)
    pred = shape * true.sum() / shape.sum() * \
        np.exp(rng.normal(0.0, noise, size=n))
    return true, pred


def run_update(true, pred, true_logcounts=3.0, pred_logcounts=2.5):
    tracker = {key: np.zeros((3, 2)) for key in TRACKER_KEYS}
    predict.metrics_update(tracker, 1, 1, true, true_logcounts, pred,
                           pred_logcounts)
    # only the addressed cell is written
    for key in TRACKER_KEYS:
        others = tracker[key].copy()
        others[1, 1] = 0
        assert not others.any()
    return {key: tracker[key][1, 1] for key in TRACKER_KEYS}


# --------------------------------------------------------------------
# metrics_update
# --------------------------------------------------------------------

@pytest.mark.parametrize("seed,noise", [(0, 0.0), (1, 0.2), (2, 1.0),
                                        (3, 3.0)])
def test_metrics_match_the_formulas(seed, noise):
    true, pred = profiles(seed, noise=noise)
    got = run_update(true, pred)
    assert_metrics_equal(got, expected_metrics(true, pred))
    assert got["all_true_logcounts"] == 3.0
    assert got["all_pred_logcounts"] == 2.5


def test_likelihood_score_with_exactly_representable_proportions():
    """
        128 reads: every proportion is a multiple of 1/128 and they add
        up to exactly 1, so no round-off is involved and the score is
        the mathematical one in every environment
    """

    rng = np.random.RandomState(13)
    for _ in range(20):
        true = np.bincount(rng.randint(0, 100, size=128),
                           minlength=100).astype(np.float64)
        true[-1] = 0.0
        true[0] += 128 - true.sum()
        assert not best_bound_collapses(true)
        pred = true + rng.uniform(0.5, 1.5, size=100)
        got = run_update(true, pred)
        n = len(true)
        value = mnll(true, fix_sum(pred / pred.sum()))
        best = mnll(true, true / true.sum())
        worst = mnll(true, np.ones(n) / n)
        np.testing.assert_allclose(
            got["profile_mnlls"], normalize(value, best, worst), rtol=1e-9)
        assert 0.0 < got["profile_mnlls"] < 1.0


def test_metrics_of_a_random_prediction_match_the_formulas():
    rng = np.random.RandomState(4)
    true = rng.poisson(2.0, size=150).astype(np.float64)
    pred = rng.gamma(2.0, size=150)
    got = run_update(true, pred)
    assert_metrics_equal(got, expected_metrics(true, pred))
    # worse than a uniform prediction is reported as 0
    assert got["profile_mnlls"] == 0.0
    assert got["profile_cross_entropys"] == 0.0


def test_better_predictions_score_higher():
    scores = []
    for noise in (0.0, 0.5, 2.0):
        true, pred = profiles(5, noise=noise)
        scores.append(run_update(true, pred))
    for key in ("profile_mnlls", "profile_cross_entropys", "profile_jsds",
                "profile_pearsonrs", "profile_spearmanrs"):
        values = [s[key] for s in scores]
        assert values[0] > values[1] > values[2], key
        assert 0.0 <= values[2] and values[0] <= 1.0, key


def test_only_the_shape_of_the_prediction_matters():
    true, pred = profiles(6, noise=0.3)
    base = run_update(true, pred)
    scaled = run_update(true, 7.5 * pred)
    for key in ("profile_mnlls", "profile_cross_entropys", "profile_jsds",
                "profile_pearsonrs", "profile_spearmanrs"):
        np.testing.assert_allclose(scaled[key], base[key], rtol=1e-9)
    assert scaled["profile_mses"] != base["profile_mses"]


def test_region_without_observed_or_predicted_reads():
    true, pred = profiles(7)
    for t, p in ((np.zeros_like(true), pred), (true, np.zeros_like(pred))):
        got = run_update(t, p, true_logcounts=0.7, pred_logcounts=1.1)
        # the profile metrics stay at their initial value, the counts
        # are still recorded
        for key in TRACKER_KEYS[:6]:
            assert got[key] == 0.0
        assert got["all_true_logcounts"] == 0.7
        assert got["all_pred_logcounts"] == 1.1


def test_constant_prediction_has_zero_correlation_when_it_sums_to_one():
    # 256 equal probabilities add up to exactly 1 in floating point
    true, _ = profiles(8, n=256)
    got = run_update(true, np.ones_like(true))
    assert got["profile_pearsonrs"] == 0.0
    assert got["profile_spearmanrs"] == 0.0
    expected = expected_metrics(true, np.ones_like(true))
    np.testing.assert_allclose(got["profile_jsds"], expected["profile_jsds"],
                               rtol=1e-8)


@pytest.mark.xfail(strict=True, reason="a constant prediction whose "
                   "probabilities do not add up to exactly 1 has one element "
                   "adjusted before the constant-array check, so a "
                   "correlation with that single element is reported")
def test_constant_prediction_has_zero_correlation():
    true, _ = profiles(8, n=200)
    got = run_update(true, np.ones_like(true))
    assert got["profile_pearsonrs"] == 0.0
    assert got["profile_spearmanrs"] == 0.0


def test_smoothing_parameters_can_be_changed():
    true, pred = profiles(9, noise=0.4)
    tracker = {key: np.zeros((1, 1)) for key in TRACKER_KEYS}
    predict.metrics_update(tracker, 0, 0, true, 0.0, pred, 0.0,
                           true_profile_smoothing=[3.0, 21])
    smooth = gaussian_smooth(true, 3.0, 21)
    expected = np.corrcoef(smooth / smooth.sum(), pred / pred.sum())[0, 1]
    np.testing.assert_allclose(tracker["profile_pearsonrs"][0, 0], expected,
                               rtol=1e-8)


# --------------------------------------------------------------------
# the building blocks
# --------------------------------------------------------------------

def test_mnll_and_cross_entropy_with_probabilities_and_logits():
    rng = np.random.RandomState(10)
    true = rng.poisson(3.0, size=80).astype(np.float64)
    logits = rng.normal(size=80)
    probs = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(predict.mnll(true, probs=probs),
                               mnll(true, probs), rtol=1e-10)
    np.testing.assert_allclose(predict.mnll(true, logits=logits),
                               mnll(true, probs), rtol=1e-10)
    np.testing.assert_allclose(
        predict.profile_cross_entropy(true, probs=probs),
        cross_entropy(true, probs), rtol=1e-10)
    np.testing.assert_allclose(
        predict.profile_cross_entropy(true, logits=logits),
        cross_entropy(true, probs), rtol=1e-10)


@pytest.mark.parametrize("function", [predict.mnll,
                                      predict.profile_cross_entropy])
def test_invalid_metric_arguments(function):
    true = np.arange(5, dtype=np.float64)
    with pytest.raises(NoTracebackException):
        function(true)
    with pytest.raises(NoTracebackException):
        function(true, logits=np.zeros(4))
    with pytest.raises(NoTracebackException):
        function(true, probs=np.ones(4) / 4)
    with pytest.raises(NoTracebackException):
        function(true, probs=np.ones(5) / 6)


def test_bounds():
    rng = np.random.RandomState(11)
    true = rng.poisson(3.0, size=90).astype(np.float64)
    n = len(true)
    uniform = np.ones(n) / n
    true_prob = true / true.sum()

    low, high = predict.mnll_min_max_bounds(true)
    np.testing.assert_allclose(low, mnll(true, fix_sum(true_prob)),
                               rtol=1e-10)
    np.testing.assert_allclose(high, mnll(true, uniform), rtol=1e-10)
    assert low < high

    low, high = predict.cross_entropy_min_max_bounds(true)
    np.testing.assert_allclose(low, cross_entropy(true, true_prob),
                               rtol=1e-10)
    np.testing.assert_allclose(high, cross_entropy(true, uniform),
                               rtol=1e-10)

    low, high = predict.jsd_min_max_bounds(true)
    assert low == 0.0
    np.testing.assert_allclose(high, js_distance(true_prob, uniform),
                               rtol=1e-10)


@pytest.mark.xfail(strict=False, reason="the best-case likelihood bound "
                   "becomes 0 when round-off makes scipy's recomputed last "
                   "probability negative (depends on the scipy version)")
def test_likelihood_bound_does_not_depend_on_round_off():
    rng = np.random.RandomState(12)
    for _ in range(300):
        true = rng.poisson(0.7, size=200).astype(np.float64)
        true[-1] = 0.0
        if true.sum() == 0:
            continue
        low, _ = predict.mnll_min_max_bounds(true)
        np.testing.assert_allclose(low, mnll(true, true / true.sum()),
                                   rtol=1e-9)


def test_min_max_normalize():
    # (best, worst) bounds: 1 at the best bound, 0 at the worst
    assert predict.min_max_normalize(0.0, (0.0, 10.0)) == 1.0
    assert predict.min_max_normalize(10.0, (0.0, 10.0)) == 0.0
    assert predict.min_max_normalize(2.5, (0.0, 10.0)) == 0.75
    # clipped outside the bounds
    assert predict.min_max_normalize(-1.0, (0.0, 10.0)) == 1
    assert predict.min_max_normalize(11.0, (0.0, 10.0)) == 0
    # equal bounds give the default
    assert predict.min_max_normalize(3.0, (2.0, 2.0)) == 0.0
    assert predict.min_max_normalize(3.0, (2.0, 2.0), default=0.7) == 0.7


def test_fix_sum_to_one():
    for probs in (np.array([0.5, 0.3, 0.2000001]),
                  np.array([0.5, 0.3, 0.1999999]),
                  np.array([0.25, 0.25, 0.25, 0.25])):
        fixed = predict._fix_sum_to_one(probs)
        assert fixed.sum() == 1.0
        # the input is not modified and at most one element changes
        assert (fixed != probs).sum() <= 1
        np.testing.assert_allclose(fixed, fix_sum(probs), rtol=0, atol=0)
