"""
    Characterization tests for the model builder (bpnet.model.arch):
    the graph it builds and the numbers the graph computes, against a
    numpy implementation of the architecture
"""

import copy

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bpnet.model import arch, bpnetdefaults

from tests import reference_model
from tests.model_helpers import (
    RELEASED_PARAMS, make_tasks, random_batch, randomize_weights,
    small_params, weights_by_layer)


def build(params, tasks=None, **kwargs):
    return arch.BPNet(tasks if tasks is not None else make_tasks(),
                      copy.deepcopy(params), name_prefix="main", **kwargs)


def predict(model, inputs):
    names = [tensor.name.split(":")[0] for tensor in model.inputs]
    outputs = model({name: inputs[name] for name in names}, training=False)
    return [np.asarray(o) for o in outputs]


# --------------------------------------------------------------------
# the graph of the released configuration
# --------------------------------------------------------------------

RELEASED_WEIGHT_SHAPES = {
    "main_conv_0": [(21, 4, 64), (64,)],
    "main_profile_head": [(75, 64, 2), (2,)],
    "main_counts_head": [(64, 1), (1,)],
    "profile_predictions": [(1, 4, 2), (2,)],
    "logcounts_predictions": [(2, 1), (1,)],
}
RELEASED_WEIGHT_SHAPES.update({
    "main_dil_conv_{}".format(i): [(3, 64, 64), (64,)] for i in range(1, 9)})


def test_released_graph():
    """
        One task, two strands, two control tracks: the layout of the
        released ENCODE models. Layer names and weight shapes are what
        saved weights are matched by.
    """

    model = build(RELEASED_PARAMS)

    assert [(t.name.split(":")[0], tuple(t.shape)) for t in model.inputs] == [
        ("sequence", (None, 2114, 4)),
        ("profile_bias_input_0", (None, 1000, 2)),
        ("counts_bias_input_0", (None, 2))]
    assert [tuple(t.shape) for t in model.outputs] == \
        [(None, 1000, 2), (None, 1)]
    assert model.output_names == \
        ["profile_predictions", "logcounts_predictions"]

    shapes = {name: [w.shape for w in weights]
              for name, weights in weights_by_layer(model).items()}
    assert shapes == RELEASED_WEIGHT_SHAPES
    assert model.count_params() == 113938

    names = {layer.name for layer in model.layers}
    expected = {
        "sequence", "profile_bias_input_0", "counts_bias_input_0",
        "main_conv_0", "main_conv_0_relu", "main_conv_0_cr",
        "main_profile_head", "main_profile_head_cropped",
        "main_global_avg_pooling", "main_counts_head",
        "main_logsumexp_counts_bias_0", "main_concat_with_prof_bias_0",
        "main_concat_with_counts_bias_0", "profile_predictions",
        "logcounts_predictions"}
    for i in range(1, 9):
        expected.update({"main_dil_conv_{}".format(i),
                         "main_add_{}".format(i),
                         "main_add_{}_relu".format(i)})
    for i in range(1, 8):
        expected.add("main_add_{}_cr".format(i))
    assert names == expected

    # dilation doubles from 2; each residual branch is cropped by the
    # dilation on both sides to match the 'valid' convolution
    for i in range(1, 9):
        layer = model.get_layer("main_dil_conv_{}".format(i))
        assert tuple(layer.dilation_rate) == (2 ** i,)
        assert layer.padding == "valid"
        assert layer.activation is tf.keras.activations.linear
    assert tuple(model.get_layer("main_conv_0_cr").cropping) == (2, 2)
    assert tuple(model.get_layer("main_add_7_cr").cropping) == (256, 256)
    assert tuple(
        model.get_layer("main_profile_head_cropped").cropping) == (0, 0)


def test_model_attributes():
    model = build(RELEASED_PARAMS)
    assert model.num_tasks == 1
    assert model.num_output_tracks == 2
    assert list(model.tracks_for_each_task) == [2]
    assert model.output_profile_len == 1000
    assert list(model.loss_weights) == [1.0, 88.3]
    assert model.counts_loss == "MSE"
    assert model.orig_multi_loss is False


def test_initial_weights_are_glorot_uniform_kernels_and_zero_biases():
    model = build(RELEASED_PARAMS)
    for name, (kernel, bias) in weights_by_layer(model).items():
        assert not bias.any(), name
        fan_in = np.prod(kernel.shape[:-1])
        fan_out = np.prod(kernel.shape[:-2]) * kernel.shape[-1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert np.abs(kernel).max() <= limit, name
        if kernel.size > 1000:
            # a uniform distribution on [-limit, limit]
            np.testing.assert_allclose(
                kernel.std(), limit / np.sqrt(3.0), rtol=0.05, err_msg=name)


# --------------------------------------------------------------------
# what the graph computes
# --------------------------------------------------------------------

FORWARD_CASES = {
    "two strands with control": dict(),
    "two strands without control": dict(n_bias=0),
    "unstranded with control": dict(n_signal=1, n_bias=1),
    "unstranded without control": dict(n_signal=1, n_bias=0),
    "one control track for two strands": dict(n_bias=1),
    "residual from activated input": dict(params=dict(
        syntax_module_params={"pre_activation_residual_unit": False})),
    "two motif convolutions": dict(params=dict(
        input_len=196,
        motif_module_params={"filters": [5, 6], "kernel_sizes": [11, 21]})),
    "one dilated layer": dict(params=dict(
        input_len=178, syntax_module_params={"num_dilation_layers": 1})),
    "wide control kernel": dict(params=dict(
        output_profile_len=76,
        profile_bias_module_params={"kernel_sizes": [5]})),
    "original loss with control": dict(orig_multi_loss=True),
    "counts head with hidden layers": dict(params=dict(
        counts_head_params={
            "units": [8, 4, 1], "dropouts": [0.5, 0.0, 0.0],
            "activations": ["relu", "leakyrelu", "linear"]})),
    "counts head with hidden layers, no control": dict(n_bias=0, params=dict(
        counts_head_params={
            "units": [5, 1], "dropouts": [0.0, 0.0],
            "activations": ["tanh", "linear"]})),
    "smoothed control tracks": dict(n_bias=4, smoothing=[[7.0, 81]] * 2),
}


@pytest.mark.parametrize("case", sorted(FORWARD_CASES))
def test_forward_pass_matches_the_reference(case):
    options = dict(FORWARD_CASES[case])
    params = small_params(**options.pop("params", {}))
    n_signal = options.pop("n_signal", 2)
    n_bias = options.pop("n_bias", 2)
    orig_multi_loss = options.pop("orig_multi_loss", False)
    smoothing = options.pop("smoothing", None)

    # a smoothed copy of each control track doubles the control channels
    n_control_files = n_bias // 2 if smoothing else n_bias
    model = build(params, make_tasks(n_signal, n_control_files,
                                     smoothing=smoothing),
                  orig_multi_loss=orig_multi_loss)
    weights = randomize_weights(model, seed=1)

    # the control profile enters after the profile head is cropped; a
    # control kernel wider than 1 shortens the output further
    batch_params = dict(params)
    control_kernel = params["profile_bias_module_params"]["kernel_sizes"][0]
    if n_bias > 0:
        batch_params["output_profile_len"] = \
            params["output_profile_len"] + control_kernel - 1
    inputs, _, _ = random_batch(batch_params, n_signal=n_signal,
                                n_bias=n_bias, seed=2)
    if n_bias > 0:
        # control input length is the model's (uncropped) output length
        assert model.get_layer("profile_bias_input_0").output.shape[1] == \
            params["output_profile_len"]
        inputs["profile_bias_input_0"] = \
            inputs["profile_bias_input_0"][:, :params["output_profile_len"]]

    profile, counts = predict(model, inputs)
    expected_profile, expected_counts = reference_model.forward(
        weights, params, inputs["sequence"],
        inputs.get("profile_bias_input_0"), inputs.get("counts_bias_input_0"),
        orig_multi_loss=orig_multi_loss)

    assert profile.shape == expected_profile.shape
    assert counts.shape == expected_counts.shape
    np.testing.assert_allclose(profile, expected_profile, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(counts, expected_counts, rtol=1e-4, atol=1e-4)
    assert profile.shape[2] == n_signal


def test_forward_pass_of_the_released_architecture():
    model = build(RELEASED_PARAMS)
    weights = randomize_weights(model, seed=3, scale=0.05)
    inputs, _, _ = random_batch(RELEASED_PARAMS, batch_size=2, seed=4)
    profile, counts = predict(model, inputs)
    expected_profile, expected_counts = reference_model.forward(
        weights, RELEASED_PARAMS, inputs["sequence"],
        inputs["profile_bias_input_0"], inputs["counts_bias_input_0"])
    assert profile.shape == (2, 1000, 2)
    assert counts.shape == (2, 1)
    np.testing.assert_allclose(profile, expected_profile, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(counts, expected_counts, rtol=1e-4, atol=1e-4)


def test_only_the_valid_region_of_the_input_matters():
    """
        With 'valid' padding every output position sees a fixed window
        of the input: changing the first base only changes the first
        output positions' logits
    """

    params = small_params()
    model = build(params, make_tasks(n_bias=0))
    randomize_weights(model, seed=5)
    inputs, _, _ = random_batch(params, n_bias=0, seed=6)
    changed = {"sequence": inputs["sequence"].copy()}
    changed["sequence"][:, 0, :] = changed["sequence"][:, 0, ::-1]
    base, _ = predict(model, inputs)
    other, _ = predict(model, changed)
    # the receptive field of output position j starts at input position j
    assert np.abs(base[:, 1:] - other[:, 1:]).max() == 0
    assert np.abs(base[:, 0] - other[:, 0]).max() > 0


# --------------------------------------------------------------------
# other configurations
# --------------------------------------------------------------------

def test_same_padding_keeps_the_length():
    params = small_params(
        input_len=100, output_profile_len=100,
        motif_module_params={"padding": "same"},
        syntax_module_params={"padding": "same"},
        profile_head_params={"padding": "same"})
    model = build(params)
    assert [tuple(t.shape) for t in model.outputs] == \
        [(None, 100, 2), (None, 1)]
    # no cropping layers on the residual branches
    assert not any(layer.name.endswith("_cr") for layer in model.layers)


def test_counts_head_with_hidden_layers():
    params = small_params(counts_head_params={
        "units": [8, 4, 1], "dropouts": [0.5, 0.0, 0.0],
        "activations": ["relu", "leakyrelu", "linear"]})
    model = build(params, make_tasks(n_bias=0))
    names = [layer.name for layer in model.layers]
    # the syntax module output is flattened (not averaged) and goes
    # through dropout and the hidden layers
    for name in ("main_flatten", "counts_flatten_dropout", "counts_dense_0",
                 "counts_dropout_0", "counts_dense_1", "counts_dense_LReLU_1"):
        assert name in names
    assert "main_global_avg_pooling" not in names
    assert "counts_dropout_1" not in names
    assert tuple(model.outputs[1].shape) == (None, 1)
    syntax_len = params["output_profile_len"] + 74
    assert model.get_layer("counts_dense_0").get_weights()[0].shape == \
        (syntax_len * 6, 8)


def test_counts_head_size_minus_one_follows_the_loss():
    params = small_params(counts_head_params={"units": [-1]})
    joint = build(params, make_tasks(n_bias=0))
    assert tuple(joint.outputs[1].shape) == (None, 1)
    per_strand = build(params, make_tasks(n_bias=0), orig_multi_loss=True)
    assert tuple(per_strand.outputs[1].shape) == (None, 2)


def test_original_loss_counts_outputs():
    model = build(small_params(), orig_multi_loss=True)
    # one count per strand; the control counts enter per strand
    assert tuple(model.outputs[1].shape) == (None, 2)
    assert model.get_layer("logcounts_predictions").get_weights()[0].shape == \
        (3, 2)
    assert "main_logsumexp_counts_bias_0" not in \
        [layer.name for layer in model.layers]


def test_smoothed_control_tracks_add_input_channels():
    tasks = make_tasks(smoothing=[[7.0, 81], [7.0, 81]])
    model = build(small_params(), tasks)
    assert tuple(model.get_layer("profile_bias_input_0").output.shape) == \
        (None, 80, 4)
    assert tuple(model.get_layer("counts_bias_input_0").output.shape) == \
        (None, 4)


def test_kernel_sizes_must_match_the_number_of_tasks():
    from bpnet.utils.exceptionhandler import NoTracebackException
    params = small_params(profile_bias_module_params={"kernel_sizes": [1, 1]})
    with pytest.raises(NoTracebackException):
        build(params)


def test_smoothing_list_must_match_the_control_tracks():
    from bpnet.utils.exceptionhandler import NoTracebackException
    with pytest.raises(NoTracebackException):
        build(small_params(), make_tasks(smoothing=[None]))


def test_load_params_fills_in_defaults():
    (input_len, output_len, motif, syntax, profile_head, counts_head,
     profile_bias, counts_bias, loss_weights, counts_loss) = \
        arch.load_params({"input_len": 500, "loss_weights": [1, 3],
                          "syntax_module_params": {"filters": 12}})
    assert input_len == 500
    assert output_len == 1000
    assert syntax["filters"] == 12
    assert syntax["num_dilation_layers"] == 8
    assert motif == {"filters": [64], "kernel_sizes": [21],
                     "padding": "valid"}
    assert profile_head["kernel_size"] == 75
    assert counts_head["units"] == [1]
    assert profile_bias == {"kernel_sizes": [1]}
    assert counts_bias == {}
    assert loss_weights == [1, 3]
    assert counts_loss == "MSE"


# --------------------------------------------------------------------
# known defects, written as the behaviour we want
# --------------------------------------------------------------------

def default_parameters():
    return {name: copy.deepcopy(getattr(bpnetdefaults, name))
            for name in dir(bpnetdefaults) if name.isupper()}


def test_building_a_model_leaves_the_defaults_alone():
    before = default_parameters()
    build(small_params(counts_head_params={"units": [-1]}),
          orig_multi_loss=True)
    arch.load_params({"motif_module_params": {"filters": [3]},
                      "loss_weights": [2, 3]})
    assert default_parameters() == before


def test_building_a_model_leaves_the_callers_parameters_alone():
    params = small_params(counts_head_params={"units": [-1]})
    before = copy.deepcopy(params)
    arch.BPNet(make_tasks(), params, name_prefix="main",
               orig_multi_loss=True)
    assert params == before
    # the returned dictionaries are copies
    loaded = arch.load_params(params)
    loaded[2]["filters"].append(99)
    loaded[8].append(99)
    assert params == before


def test_models_do_not_depend_on_what_was_built_before():
    build(small_params(
        motif_module_params={"filters": [5, 6], "kernel_sizes": [11, 21]},
        syntax_module_params={"num_dilation_layers": 1, "filters": 6},
        input_len=188))
    # the default architecture, with no parameters given
    model = arch.BPNet(make_tasks(), {}, name_prefix="main")
    assert model.count_params() == 113938
    assert [tuple(t.shape) for t in model.outputs] == \
        [(None, 1000, 2), (None, 1)]


def two_task_model():
    params = small_params(profile_bias_module_params={"kernel_sizes": [1, 1]})
    model = build(params, make_tasks(n_tasks=2))
    # the model builds today; its bookkeeping and counts output are wrong
    assert tuple(model.outputs[0].shape) == (None, 80, 4)
    return model


@pytest.mark.xfail(strict=True, reason="multi-task models: the number of "
                   "tracks per task is stored as a running total")
def test_two_task_model_tracks_per_task():
    assert list(two_task_model().tracks_for_each_task) == [2, 2]


@pytest.mark.xfail(strict=True, reason="multi-task models: every task's "
                   "counts layer has one unit per task, so two tasks give "
                   "four count outputs")
def test_two_task_model_has_one_count_per_task():
    assert tuple(two_task_model().outputs[1].shape) == (None, 2)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="an output length whose difference to the profile "
                   "head's length is odd cannot be cropped symmetrically; "
                   "the model is built with another length without notice")
def test_uncroppable_output_length_is_not_silently_changed():
    from bpnet.utils.exceptionhandler import NoTracebackException
    # the profile head produces 80 positions here; 79 was asked for
    params = small_params(output_profile_len=79)
    try:
        model = build(params, make_tasks(n_bias=0))
    except NoTracebackException:
        return
    assert tuple(model.outputs[0].shape) == (None, 79, 2)
