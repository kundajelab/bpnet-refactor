"""
    Characterization tests for the loss functions in bpnet.model.losses
"""

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import multinomial

tf = pytest.importorskip("tensorflow")

from bpnet.model import losses

from tests import reference_model


def random_counts_and_logits(seed, rows=7, positions=60, rate=2.0):
    rng = np.random.RandomState(seed)
    counts = rng.poisson(rate, size=(rows, positions)).astype(np.float32)
    logits = rng.normal(0.0, 2.0, size=(rows, positions)).astype(np.float32)
    return counts, logits


def nll(counts, logits):
    return float(losses.multinomial_nll(tf.constant(counts),
                                        tf.constant(logits)))


@pytest.mark.parametrize("seed", range(4))
def test_multinomial_nll_matches_scipy(seed):
    counts, logits = random_counts_and_logits(seed)
    expected = 0.0
    for row_counts, row_logits in zip(counts.astype(np.float64),
                                      logits.astype(np.float64)):
        p = np.exp(row_logits - row_logits.max())
        p = p / p.sum()
        expected -= multinomial.logpmf(row_counts, row_counts.sum(), p)
    expected /= len(counts)
    np.testing.assert_allclose(nll(counts, logits), expected, rtol=1e-5)
    np.testing.assert_allclose(
        reference_model.multinomial_nll(counts, logits), expected, rtol=1e-10)


def test_multinomial_nll_is_the_mean_over_rows():
    counts, logits = random_counts_and_logits(10, rows=5)
    per_row = [nll(counts[i:i + 1], logits[i:i + 1]) for i in range(5)]
    np.testing.assert_allclose(nll(counts, logits), np.mean(per_row),
                               rtol=1e-5)


def test_multinomial_nll_ignores_a_constant_added_to_the_logits():
    counts, logits = random_counts_and_logits(11)
    shift = np.linspace(-5, 5, len(logits)).astype(np.float32)[:, None]
    np.testing.assert_allclose(nll(counts, logits + shift),
                               nll(counts, logits), rtol=1e-5)


def test_multinomial_nll_rows_without_counts_contribute_zero():
    counts, logits = random_counts_and_logits(12, rows=4)
    with_empty = np.concatenate([counts, np.zeros_like(counts[:2])])
    logits_with_empty = np.concatenate([logits, logits[:2]])
    # the sum over rows is unchanged, the mean is over more rows
    np.testing.assert_allclose(nll(with_empty, logits_with_empty) * 6,
                               nll(counts, logits) * 4, rtol=1e-5)


def test_multinomial_nll_minimum_is_at_the_observed_proportions():
    counts, _ = random_counts_and_logits(13, rows=3, rate=5.0)
    best = np.log(counts + 1e-6)
    worse = best + np.random.RandomState(0).normal(0, 0.3, size=best.shape)
    assert nll(counts, best.astype(np.float32)) < \
        nll(counts, worse.astype(np.float32))


def test_multinomial_nll_gradient():
    counts, logits = random_counts_and_logits(14)
    variable = tf.Variable(logits)
    with tf.GradientTape() as tape:
        value = losses.multinomial_nll(tf.constant(counts), variable)
    gradient = tape.gradient(value, variable).numpy()
    # d/dlogits of -log likelihood = total * softmax - counts, averaged
    # over the rows
    p = np.exp(logits - logits.max(axis=1, keepdims=True)).astype(np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    expected = (counts.sum(axis=1, keepdims=True) * p - counts) / len(counts)
    np.testing.assert_allclose(gradient, expected, rtol=1e-4, atol=1e-5)


def tensorflow_probability_nll(true_counts, logits):
    """
        multinomial_nll as it was while the package used
        tensorflow-probability (bpnet/model/losses.py at main@0b11a7c)
    """

    import tensorflow_probability as tfp

    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) /
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))


def value_and_gradient(function, counts, logits, compiled):
    variable = tf.Variable(logits)

    def run():
        with tf.GradientTape() as tape:
            value = function(tf.constant(counts), variable)
        return value, tape.gradient(value, variable)

    if compiled:
        run = tf.function(run)
    value, gradient = run()
    return value.numpy(), gradient.numpy()


NLL_CASES = {
    "profile of a batch": dict(rows=64, positions=2000, rate=0.3),
    "few positions": dict(rows=5, positions=7, rate=2.0),
    "one row": dict(rows=1, positions=1000, rate=0.1),
    "deep coverage": dict(rows=3, positions=2000, rate=400.0),
    "one position": dict(rows=4, positions=1, rate=3.0),
}


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("case", sorted(NLL_CASES))
def test_multinomial_nll_equals_the_tensorflow_probability_version(
        case, compiled):
    """
        The loss gives the value and the gradient that the version
        built on tensorflow-probability gave, bit for bit, eagerly and
        in a compiled function (the trainer's train step is compiled)
    """

    pytest.importorskip("tensorflow_probability")
    for seed in range(3):
        counts, logits = random_counts_and_logits(seed, **NLL_CASES[case])
        # some rows and positions without any counts, and some very
        # unlikely positions
        counts[0] = 0
        counts[:, 1::3] = 0
        logits[:, -1] = -80.0
        new = value_and_gradient(losses.multinomial_nll, counts, logits,
                                 compiled)
        old = value_and_gradient(tensorflow_probability_nll, counts, logits,
                                 compiled)
        assert new[0].dtype == old[0].dtype == np.float32
        assert np.array_equal(new[0], old[0]), (case, seed)
        assert np.array_equal(new[1], old[1]), (case, seed)
        assert np.isfinite(new[0]) and np.isfinite(new[1]).all()


def test_multinomial_log_prob_per_row():
    counts, logits = random_counts_and_logits(21, rows=6, positions=40)
    log_prob = losses.multinomial_log_prob(
        tf.constant(counts), tf.constant(logits)).numpy()
    assert log_prob.shape == (6,)
    for i in range(6):
        p = np.exp(logits[i].astype(np.float64))
        expected = multinomial.logpmf(counts[i], n=counts[i].sum(),
                                      p=p / p.sum())
        np.testing.assert_allclose(log_prob[i], expected, rtol=1e-4)
    # leading dimensions are kept
    stacked = losses.multinomial_log_prob(
        tf.constant(counts.reshape(2, 3, 40)),
        tf.constant(logits.reshape(2, 3, 40))).numpy()
    assert np.array_equal(stacked, log_prob.reshape(2, 3))


def test_multinomial_nll_position_with_zero_probability_and_no_counts():
    # log softmax is -inf at such a position; it has no counts, so it
    # contributes nothing (and no nan)
    counts = np.array([[3.0, 0.0, 2.0]], dtype=np.float32)
    logits = np.array([[0.0, -np.inf, 0.0]], dtype=np.float32)
    expected = -multinomial.logpmf([3, 2], n=5, p=[0.5, 0.5])
    assert nll(counts, logits) == pytest.approx(expected, rel=1e-6)


def test_multinomial_nll_needs_counts_and_logits_of_one_type():
    counts, logits = random_counts_and_logits(22)
    with pytest.raises((TypeError, ValueError,
                        tf.errors.InvalidArgumentError)):
        losses.multinomial_nll(tf.constant(counts.astype(np.float64)),
                               tf.constant(logits))


def test_the_model_code_does_not_import_tensorflow_probability():
    import subprocess
    import sys

    code = ("import sys; import bpnet.model.losses, bpnet.model.custommodel, "
            "bpnet.model.arch, bpnet.train.training; "
            "sys.exit('tensorflow_probability' in sys.modules)")
    assert subprocess.call([sys.executable, "-c", code]) == 0


def test_multinomial_nll_large_counts():
    rng = np.random.RandomState(15)
    counts = rng.poisson(300.0, size=(4, 2000)).astype(np.float32)
    logits = rng.normal(0.0, 1.0, size=(4, 2000)).astype(np.float32)
    # about 6e5 reads per row; float32 keeps 4 significant digits here
    np.testing.assert_allclose(
        nll(counts, logits), reference_model.multinomial_nll(counts, logits),
        rtol=1e-4)


def test_multinomial_nll_normalization_constant():
    """
        The log of the multinomial coefficient is part of the value (it
        does not depend on the logits, but the validation loss that
        drives early stopping includes it)
    """

    counts = np.array([[3.0, 0.0, 2.0, 5.0]], dtype=np.float32)
    logits = np.zeros((1, 4), dtype=np.float32)
    coefficient = gammaln(11) - gammaln(4) - gammaln(1) - gammaln(3) - \
        gammaln(6)
    expected = -(coefficient + 10 * np.log(0.25))
    np.testing.assert_allclose(nll(counts, logits), expected, rtol=1e-5)


def test_mse_loss_function_averages_over_the_batch():
    true = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    pred = np.array([1.5, 2.0, 2.0], dtype=np.float32)
    value = losses.mse_loss_function(tf.constant(true), tf.constant(pred))
    assert value.shape == ()
    np.testing.assert_allclose(float(value), (0.25 + 0.0 + 4.0) / 3,
                               rtol=1e-6)


def test_poisson_loss_function_is_per_example():
    log_true = np.array([0.5, 2.0, 3.0], dtype=np.float32)
    log_pred = np.array([1.0, 2.0, 2.5], dtype=np.float32)
    value = losses.poisson_loss_function(
        tf.constant(log_true), tf.constant(log_pred)).numpy()
    true, pred = np.exp(log_true.astype(np.float64)), \
        np.exp(log_pred.astype(np.float64))
    expected = pred - true * np.log(pred + 1e-8) + gammaln(true + 1.0)
    assert value.shape == (3,)
    np.testing.assert_allclose(value, expected, rtol=1e-5)
