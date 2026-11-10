"""
    Characterization of the Adam update the trainer uses, on prescribed
    gradients. The update rule is part of the training recipe: another
    backend (a newer TensorFlow, PyTorch) has to reproduce it, and its
    epsilon sits in a different place than in some other frameworks.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensorflow.keras.optimizers import Adam

BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-7


def prescribed_gradients():
    rng = np.random.RandomState(0)
    shape = (3, 8, 8)
    gradients = []
    for step in range(40):
        # magnitudes from far below epsilon / sqrt(1 - beta2) = 3.2e-6
        # to large, with sign changes and exact zeros
        g = rng.normal(size=shape) * 10.0 ** rng.uniform(-9, 2, size=shape)
        if step % 5 == 0:
            g[rng.rand(*shape) < 0.3] = 0.0
        gradients.append(g.astype(np.float32))
    return gradients


def learning_rate_at(step):
    # the trainer halves the rate on a plateau
    return 0.001 if step < 25 else 0.0005


def keras_rule(theta, gradients):
    theta = theta.astype(np.float64)
    m, v, out = np.zeros_like(theta), np.zeros_like(theta), []
    for t, g in enumerate(gradients, start=1):
        g = g.astype(np.float64)
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g * g
        lr_t = learning_rate_at(t - 1) * np.sqrt(1 - BETA2 ** t) / \
            (1 - BETA1 ** t)
        theta = theta - lr_t * m / (np.sqrt(v) + EPSILON)
        out.append(theta.copy())
    return out


def epsilon_after_bias_correction_rule(theta, gradients):
    """
        The variant with epsilon added to the bias-corrected second
        moment (e.g. torch.optim.Adam), for contrast
    """

    theta = theta.astype(np.float64)
    m, v, out = np.zeros_like(theta), np.zeros_like(theta), []
    for t, g in enumerate(gradients, start=1):
        g = g.astype(np.float64)
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g * g
        m_hat, v_hat = m / (1 - BETA1 ** t), v / (1 - BETA2 ** t)
        theta = theta - learning_rate_at(t - 1) * m_hat / \
            (np.sqrt(v_hat) + EPSILON)
        out.append(theta.copy())
    return out


def test_adam_defaults():
    optimizer = Adam(learning_rate=0.001)
    config = optimizer.get_config()
    assert float(config["beta_1"]) == pytest.approx(BETA1)
    assert float(config["beta_2"]) == pytest.approx(BETA2)
    assert float(config["epsilon"]) == pytest.approx(EPSILON)
    assert config["amsgrad"] is False


def test_adam_trajectory_on_prescribed_gradients():
    gradients = prescribed_gradients()
    start = np.random.RandomState(1).normal(
        size=gradients[0].shape).astype(np.float32)
    variable = tf.Variable(start)
    optimizer = Adam(learning_rate=0.001)

    expected = keras_rule(start, gradients)
    contrast = epsilon_after_bias_correction_rule(start, gradients)
    for step, gradient in enumerate(gradients):
        if step == 25:
            optimizer.learning_rate.assign(0.0005)
        optimizer.apply_gradients([(tf.constant(gradient), variable)])
        np.testing.assert_allclose(variable.numpy(), expected[step],
                                   rtol=0, atol=2e-6,
                                   err_msg="step {}".format(step))

    # the other epsilon placement is distinguishable on these gradients:
    # it differs by far more than the tolerance used above
    assert np.abs(contrast[-1] - expected[-1]).max() > 1e-3
