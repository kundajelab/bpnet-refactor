"""
    Characterization tests for the training objective and the train /
    test steps of bpnet.model.custommodel.CustomModel, against a numpy
    implementation of the forward pass and the loss
"""

import copy

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensorflow.keras.optimizers import Adam

from bpnet.model import arch

from tests import reference_model
from tests.model_helpers import (
    make_tasks, random_batch, randomize_weights, small_params)


def build(params=None, n_signal=2, n_bias=2, **kwargs):
    params = small_params() if params is None else params
    return arch.BPNet(make_tasks(n_signal, n_bias), copy.deepcopy(params),
                      name_prefix="main", **kwargs)


def reference_loss(weights, params, batch, orig_multi_loss=False):
    inputs, outputs, sample_weights = batch
    profile, counts = reference_model.forward(
        weights, params, inputs["sequence"],
        inputs.get("profile_bias_input_0"), inputs.get("counts_bias_input_0"),
        orig_multi_loss=orig_multi_loss)
    return reference_model.loss(
        outputs["profile_predictions"], outputs["logcounts_predictions"],
        sample_weights, profile, counts, params["loss_weights"],
        orig_multi_loss=orig_multi_loss)


def model_loss(model, batch, training=False):
    inputs, outputs, sample_weights = batch
    values = model._get_loss(inputs, outputs, tf.constant(sample_weights),
                             training=training)
    return [float(v) for v in values]


# --------------------------------------------------------------------
# the loss
# --------------------------------------------------------------------

WEIGHT_CASES = {
    "foreground and background": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    "all foreground": [1.0] * 6,
    "one foreground example": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    # the profile loss uses examples whose weight is at least 1
    "weights between 0 and 1 are left out": [1.0, 0.5, 0.99, 1.0, 0.0, 1.0],
    "weights above 1 count once": [2.0, 3.0, 0.0, 1.0, 0.0, 1.0],
}


@pytest.mark.parametrize("case", sorted(WEIGHT_CASES))
def test_loss_matches_the_reference(case):
    params = small_params()
    model = build(params)
    weights = randomize_weights(model, seed=1)
    batch = random_batch(params, seed=2, weights=WEIGHT_CASES[case])

    total, profile_loss, counts_loss = model_loss(model, batch)
    expected = reference_loss(weights, params, batch)
    np.testing.assert_allclose([total, profile_loss, counts_loss], expected,
                               rtol=2e-4)
    # total = profile loss + counts loss weight * counts loss
    np.testing.assert_allclose(
        total, profile_loss + params["loss_weights"][1] * counts_loss,
        rtol=1e-5)


def test_loss_of_a_batch_without_foreground():
    params = small_params()
    model = build(params)
    weights = randomize_weights(model, seed=3)
    batch = random_batch(params, seed=4, weights=[0.0] * 6)
    total, profile_loss, counts_loss = model_loss(model, batch)
    expected = reference_loss(weights, params, batch)
    assert profile_loss == 0.0
    np.testing.assert_allclose([total, counts_loss],
                               [expected[0], expected[2]], rtol=2e-4)
    assert counts_loss > 0


def test_weights_above_one_do_not_scale_the_loss():
    params = small_params()
    model = build(params)
    randomize_weights(model, seed=5)
    ones = random_batch(params, seed=6, weights=[1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    heavy = (ones[0], ones[1], np.array([5.0, 2.0, 0.0, 1.0, 0.0, 9.0]))
    assert model_loss(model, ones) == model_loss(model, heavy)


@pytest.mark.parametrize("n_signal,n_bias", [(2, 0), (1, 1), (1, 0)])
def test_loss_matches_the_reference_for_other_layouts(n_signal, n_bias):
    params = small_params()
    model = build(params, n_signal=n_signal, n_bias=n_bias)
    weights = randomize_weights(model, seed=7)
    batch = random_batch(params, n_signal=n_signal, n_bias=n_bias, seed=8)
    np.testing.assert_allclose(model_loss(model, batch),
                               reference_loss(weights, params, batch),
                               rtol=2e-4)


def test_original_loss_matches_the_reference():
    params = small_params()
    model = build(params, orig_multi_loss=True)
    weights = randomize_weights(model, seed=9)
    batch = random_batch(params, seed=10)
    np.testing.assert_allclose(
        model_loss(model, batch),
        reference_loss(weights, params, batch, orig_multi_loss=True),
        rtol=2e-4)


def test_count_target_is_the_log_of_total_counts_plus_two():
    """
        The generator gives log(1 + counts) per strand; the count head is
        trained against the log of their sum, log(total counts + 2)
    """

    params = small_params()
    model = build(params)
    weights = randomize_weights(model, seed=11)
    inputs, outputs, sample_weights = random_batch(params, seed=12)
    _, counts = reference_model.forward(
        weights, params, inputs["sequence"], inputs["profile_bias_input_0"],
        inputs["counts_bias_input_0"])
    total = outputs["profile_predictions"].sum(axis=(1, 2))
    expected = np.mean((np.log(total + 2.0) - counts[:, 0]) ** 2)
    _, _, counts_loss = model_loss(model, (inputs, outputs, sample_weights))
    np.testing.assert_allclose(counts_loss, expected, rtol=2e-4)


def test_loss_does_not_depend_on_the_order_of_the_examples():
    params = small_params()
    model = build(params)
    randomize_weights(model, seed=13)
    inputs, outputs, sample_weights = random_batch(params, seed=14)
    order = np.random.RandomState(0).permutation(len(sample_weights))
    shuffled = ({k: v[order] for k, v in inputs.items()},
                {k: v[order] for k, v in outputs.items()},
                sample_weights[order])
    np.testing.assert_allclose(
        model_loss(model, shuffled),
        model_loss(model, (inputs, outputs, sample_weights)), rtol=1e-5)


def test_gradient_matches_finite_differences_of_the_reference():
    params = small_params()
    model = build(params)
    weights = randomize_weights(model, seed=15)
    batch = random_batch(params, seed=16)
    inputs, outputs, sample_weights = batch

    with tf.GradientTape() as tape:
        total, _, _ = model._get_loss(inputs, outputs,
                                      tf.constant(sample_weights))
    gradients = tape.gradient(total, model.trainable_variables)
    by_name = {v.name: g.numpy()
               for v, g in zip(model.trainable_variables, gradients)}

    rng = np.random.RandomState(17)
    checked = 0
    for layer_name in ("main_conv_0", "main_dil_conv_2", "main_profile_head",
                       "main_counts_head", "profile_predictions",
                       "logcounts_predictions"):
        for position, suffix in ((0, "kernel:0"), (1, "bias:0")):
            array = weights[layer_name][position]
            index = tuple(rng.randint(0, n) for n in array.shape)
            # small enough that no ReLU changes side between the two
            # evaluations (the reference is float64)
            step = 1e-5
            values = []
            for sign in (1.0, -1.0):
                changed = {k: [w.astype(np.float64) for w in v]
                           for k, v in weights.items()}
                changed[layer_name][position][index] += sign * step
                values.append(reference_loss(changed, params, batch)[0])
            numeric = (values[0] - values[1]) / (2 * step)
            analytic = by_name["{}/{}".format(layer_name, suffix)][index]
            np.testing.assert_allclose(
                analytic, numeric, rtol=1e-3, atol=1e-4,
                err_msg="{} {} {}".format(layer_name, suffix, index))
            checked += 1
    assert checked == 12


# --------------------------------------------------------------------
# train and test steps
# --------------------------------------------------------------------

LOG_KEYS = ["batch_loss", "logcounts_predictions_loss", "loss",
            "profile_predictions_loss"]


def test_train_step_reports_running_mean_and_batch_values():
    params = small_params()
    model = build(params)
    randomize_weights(model, seed=18)
    model.compile(Adam(learning_rate=0.001), loss=None)

    batches = [random_batch(params, seed=s) for s in (19, 20, 21)]
    batch_losses = []
    for i, (inputs, outputs, sample_weights) in enumerate(batches):
        before = model_loss(model, (inputs, outputs, sample_weights),
                            training=True)
        logs = model.train_step((inputs, outputs, tf.constant(sample_weights)))
        logs = {k: float(v) for k, v in logs.items()}
        assert sorted(logs) == LOG_KEYS
        # the values describe the batch before the weight update
        np.testing.assert_allclose(logs["batch_loss"], before[0], rtol=1e-5)
        np.testing.assert_allclose(logs["profile_predictions_loss"],
                                   before[1], rtol=1e-5)
        np.testing.assert_allclose(logs["logcounts_predictions_loss"],
                                   before[2], rtol=1e-5)
        batch_losses.append(logs["batch_loss"])
        # 'loss' is the mean of the batch losses since the last reset
        np.testing.assert_allclose(logs["loss"], np.mean(batch_losses),
                                   rtol=1e-5)

    assert [m.name for m in model.metrics] == ["loss"]
    model.reset_metrics()
    inputs, outputs, sample_weights = batches[0]
    logs = model.train_step((inputs, outputs, tf.constant(sample_weights)))
    np.testing.assert_allclose(float(logs["loss"]), float(logs["batch_loss"]),
                               rtol=1e-6)


def test_test_step_does_not_change_the_weights():
    params = small_params()
    model = build(params)
    randomize_weights(model, seed=22)
    model.compile(Adam(learning_rate=0.001), loss=None)
    inputs, outputs, sample_weights = random_batch(params, seed=23)
    before = [v.numpy().copy() for v in model.trainable_variables]
    logs = model.test_step((inputs, outputs, tf.constant(sample_weights)))
    assert sorted(logs) == LOG_KEYS
    for old, variable in zip(before, model.trainable_variables):
        assert np.array_equal(old, variable.numpy())
    expected = model_loss(model, (inputs, outputs, sample_weights))
    np.testing.assert_allclose(float(logs["batch_loss"]), expected[0],
                               rtol=1e-5)


def test_first_adam_step_is_the_keras_update_rule():
    """
        theta <- theta - lr_t * m / (sqrt(v) + epsilon) with
        lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t), epsilon = 1e-7:
        the first step moves every weight by lr * g / (|g| + epsilon /
        sqrt(1 - beta2))
    """

    params = small_params()
    model = build(params)
    randomize_weights(model, seed=24)
    lr = 0.001
    model.compile(Adam(learning_rate=lr), loss=None)
    inputs, outputs, sample_weights = random_batch(params, seed=25)

    with tf.GradientTape() as tape:
        total, _, _ = model._get_loss(inputs, outputs,
                                      tf.constant(sample_weights))
    gradients = [g.numpy().astype(np.float64) for g in
                 tape.gradient(total, model.trainable_variables)]
    before = [v.numpy().astype(np.float64) for v in model.trainable_variables]

    model.train_step((inputs, outputs, tf.constant(sample_weights)))

    epsilon_hat = 1e-7 / np.sqrt(1 - 0.999)
    for old, grad, variable in zip(before, gradients,
                                   model.trainable_variables):
        expected = old - lr * grad / (np.abs(grad) + epsilon_hat)
        np.testing.assert_allclose(variable.numpy(), expected, rtol=0,
                                   atol=2e-6, err_msg=variable.name)


def test_fit_and_evaluate_report_the_same_keys():
    params = small_params()
    model = build(params)
    randomize_weights(model, seed=26)
    model.compile(Adam(learning_rate=0.001), loss=None,
                  loss_weights=params["loss_weights"])
    batches = [random_batch(params, seed=s) for s in (27, 28)]

    def generator():
        for batch in batches:
            yield batch

    history = model.fit(generator(), epochs=1, steps_per_epoch=2, verbose=0)
    assert sorted(history.history) == LOG_KEYS
    logs = model.evaluate(generator(), steps=2, return_dict=True, verbose=0)
    assert sorted(logs) == LOG_KEYS
    # evaluate reports the mean of the batch losses as 'loss'
    expected = np.mean([model_loss(model, b)[0] for b in batches])
    np.testing.assert_allclose(logs["loss"], expected, rtol=1e-5)


def test_dropout_is_only_active_in_train_steps():
    """
        A counts head with hidden layers has dropout: test steps must
        run the model in inference mode, train steps in training mode
    """

    params = small_params(counts_head_params={
        "units": [8, 1], "dropouts": [0.5, 0.0],
        "activations": ["relu", "linear"]})
    model = build(params, n_bias=0)
    randomize_weights(model, seed=29)
    model.compile(Adam(learning_rate=0.0), loss=None)
    inputs, outputs, sample_weights = random_batch(params, n_bias=0, seed=30)
    data = (inputs, outputs, tf.constant(sample_weights))

    test_values = {float(model.test_step(data)["logcounts_predictions_loss"])
                   for _ in range(4)}
    assert len(test_values) == 1
    assert test_values == {model_loss(model, (inputs, outputs,
                                              sample_weights))[2]}

    # learning rate 0: the weights do not move, only dropout varies
    train_values = {float(
        model.train_step(data)["logcounts_predictions_loss"])
        for _ in range(4)}
    assert len(train_values) > 1
