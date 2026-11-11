"""
    The models of this package in PyTorch (bpnet.pytorch.model): the
    layer graph of a saved model built from PyTorch modules gives the
    outputs of the Keras model and of the numpy reference, and the
    weights move between the two layouts without loss
"""

import copy
import json
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bpnet.model import loading
from bpnet.model.loading import UnsupportedModelError
from bpnet.pytorch import model as pytorch_model

from tests import reference_model
from tests.model_helpers import (FIXTURE_VERSIONS, fixture_files,
                                 unpack_fixture_models)


# --------------------------------------------------------------------
# without TensorFlow: a layer graph written out by hand
# --------------------------------------------------------------------

def entry(class_name, name, inputs=(), **config):
    return {"class_name": class_name, "name": name,
            "config": dict(config, name=name),
            "inbound_nodes": [[[other, 0, 0, {}] for other in inputs]]
            if inputs else []}


def hand_made():
    """
        A small graph with the layer classes of the package (two
        inputs, a residual unit with a crop, both ways from the
        convolutions to the dense layers, layers with and without a
        bias), and random weights for it
    """

    layers = [
        entry("InputLayer", "sequence", batch_input_shape=[None, 30, 4]),
        entry("InputLayer", "control", batch_input_shape=[None, 2]),
        entry("Conv1D", "first", ["sequence"], filters=5, kernel_size=[3],
              padding="valid", activation="relu"),
        entry("Conv1D", "dilated", ["first"], filters=5, kernel_size=[3],
              dilation_rate=[2], padding="valid", activation="linear",
              use_bias=False),
        entry("Cropping1D", "cropped", ["first"], cropping=[2, 2]),
        entry("Add", "sum", ["dilated", "cropped"]),
        entry("ReLU", "sum_relu", ["sum"]),
        entry("GlobalAveragePooling1D", "pooled", ["sum_relu"]),
        entry("Dense", "from_pooled", ["pooled"], units=5,
              activation="tanh", use_bias=False),
        entry("Flatten", "flat", ["sum_relu"]),
        entry("Dropout", "dropped", ["flat"], rate=0.5),
        entry("Dense", "from_flat", ["dropped"], units=3,
              activation="linear"),
        entry("LeakyReLU", "leaky", ["from_flat"], alpha=0.2),
        entry("Lambda", "x_logsumexp_counts_bias_0", ["control"]),
        entry("Concatenate", "together",
              ["from_pooled", "leaky", "x_logsumexp_counts_bias_0"],
              axis=-1),
        entry("Dense", "out", ["together"], units=1, activation="linear"),
    ]
    architecture = {"name": "hand", "layers": layers,
                    "input_layers": [["sequence", 0, 0], ["control", 0, 0]],
                    "output_layers": [["sum_relu", 0, 0], ["out", 0, 0]]}
    rng = np.random.RandomState(0)

    def pair(*shape):
        return {"kernel": rng.normal(size=shape),
                "bias": rng.normal(size=shape[-1])}

    # (the kernels of 'dilated' and 'from_pooled' have as many inputs
    # as outputs: their layout is not given away by their shape)
    weights = {"first": pair(3, 4, 5),
               "dilated": {"kernel": rng.normal(size=(3, 5, 5))},
               "from_pooled": {"kernel": rng.normal(size=(5, 5))},
               "from_flat": pair(24 * 5, 3), "out": pair(9, 1)}
    return architecture, weights


def hand_computed(weights, sequence, control):
    """
        The outputs of the hand-made graph in numpy, in Keras' layout
    """

    first = reference_model.relu(reference_model.conv1d_valid(
        sequence, weights["first"]["kernel"], weights["first"]["bias"]))
    dilated = reference_model.conv1d_valid(
        first, weights["dilated"]["kernel"], np.zeros(5), dilation=2)
    activated = reference_model.relu(dilated + first[:, 2:-2, :])
    from_pooled = np.tanh(activated.mean(axis=1) @
                          weights["from_pooled"]["kernel"])
    from_flat = activated.reshape(len(activated), -1) @ \
        weights["from_flat"]["kernel"] + weights["from_flat"]["bias"]
    leaky = np.where(from_flat > 0, from_flat, 0.2 * from_flat)
    largest = control.max(axis=1, keepdims=True)
    summed = largest + np.log(np.exp(control - largest).sum(
        axis=1, keepdims=True))
    out = np.concatenate([from_pooled, leaky, summed], axis=1) @ \
        weights["out"]["kernel"] + weights["out"]["bias"]
    return activated, out


def test_graph_written_by_hand():
    architecture, weights = hand_made()
    model = pytorch_model.from_architecture(architecture, weights)
    assert not model.training
    # the weights of the graph are float64, and so is the model
    assert all(p.dtype == torch.float64 for p in model.parameters())
    assert model.input_names == ["sequence", "control"]
    assert model.output_names == ["sum_relu", "out"]
    assert model.shapes["sequence"] == (4, 30)
    assert model.shapes["first"] == (5, 28)
    assert model.shapes["sum"] == (5, 24)
    assert model.shapes["flat"] == (120,) and model.shapes["out"] == (1,)

    rng = np.random.RandomState(1)
    sequence = rng.normal(size=(7, 30, 4))
    control = rng.normal(size=(7, 2))
    expected = hand_computed(weights, sequence, control)
    with torch.no_grad():
        found = model(torch.tensor(pytorch_model.channels_first(sequence)),
                      torch.tensor(control))
    assert len(found) == 2
    np.testing.assert_allclose(
        pytorch_model.channels_last(found[0].numpy()), expected[0],
        rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(found[1].numpy(), expected[1],
                               rtol=1e-12, atol=1e-12)

    # by name, and in any order
    with torch.no_grad():
        again = model(control=torch.tensor(control), sequence=torch.tensor(
            pytorch_model.channels_first(sequence)))
    assert all(torch.equal(a, b) for a, b in zip(found, again))

    # control counts whose exponentials cannot be represented
    control = np.array([[1000.0, 999.0], [-1000.0, -1001.0], [800.0, -800.0],
                        [0.0, 0.0], [3.0, -2.0], [1e-3, 2e-3], [5.0, 5.0]])
    expected = hand_computed(weights, sequence, control)
    assert np.isfinite(expected[1]).all()
    with torch.no_grad():
        found = model(torch.tensor(pytorch_model.channels_first(sequence)),
                      torch.tensor(control))
    np.testing.assert_allclose(found[1].numpy(), expected[1],
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("width,dilation", [
    (3, 1), (4, 1), (2, 3), (4, 3), (3, 2), (6, 5), (1, 1)])
def test_same_padding_as_tensorflow_defines_it(width, dilation):
    """
        A convolution that keeps the length is the convolution of the
        input with zeros added on both sides: as many as the kernel
        spans beyond one position, half of them on the left, and the
        odd one on the right (the 'SAME' padding of TensorFlow)
    """

    architecture = {
        "name": "same", "layers": [
            entry("InputLayer", "x", batch_input_shape=[None, 20, 3]),
            entry("Conv1D", "conv", ["x"], filters=2, kernel_size=[width],
                  dilation_rate=[dilation], padding="same",
                  activation="linear")],
        "input_layers": [["x", 0, 0]], "output_layers": [["conv", 0, 0]]}
    rng = np.random.RandomState(11)
    weights = {"conv": {"kernel": rng.normal(size=(width, 3, 2)),
                        "bias": rng.normal(size=2)}}
    model = pytorch_model.from_architecture(architecture, weights)
    assert model.shapes["conv"] == (2, 20)
    x = rng.normal(size=(4, 20, 3))
    span = dilation * (width - 1)
    left = span // 2
    padded = np.zeros((4, 20 + span, 3))
    padded[:, left:left + 20] = x
    expected = reference_model.conv1d_valid(
        padded, weights["conv"]["kernel"], weights["conv"]["bias"],
        dilation=dilation)
    with torch.no_grad():
        found = model(torch.tensor(pytorch_model.channels_first(x)))
    np.testing.assert_allclose(pytorch_model.channels_last(found.numpy()),
                               expected, rtol=1e-12, atol=1e-12)


def test_model_with_one_output_returns_a_tensor():
    architecture, weights = hand_made()
    one = copy.deepcopy(architecture)
    one["output_layers"] = [["out", 0, 0]]
    rng = np.random.RandomState(5)
    inputs = (torch.tensor(rng.normal(size=(3, 4, 30))),
              torch.tensor(rng.normal(size=(3, 2))))
    with torch.no_grad():
        both = pytorch_model.from_architecture(architecture, weights)(*inputs)
        single = pytorch_model.from_architecture(one, weights)(*inputs)
    assert isinstance(both, tuple) and len(both) == 2
    assert isinstance(single, torch.Tensor)
    assert torch.equal(single, both[1])


def test_every_module_runs_once_in_a_forward_pass():
    """
        What methods that hook the modules rely on: a forward pass
        calls every module of the model exactly once
    """

    architecture, weights = hand_made()
    model = pytorch_model.from_architecture(architecture, weights)
    calls = {name: 0 for name in model.layers}

    def count(name):
        def hook(module, inputs, output):
            calls[name] += 1
        return hook

    for name, module in model.layers.items():
        module.register_forward_hook(count(name))
    with torch.no_grad():
        model(torch.zeros(2, 4, 30, dtype=torch.float64),
              torch.zeros(2, 2, dtype=torch.float64))
    assert calls == {name: 1 for name in model.layers}
    assert len(calls) == 10


def test_every_activation_is_a_module():
    architecture, weights = hand_made()
    model = pytorch_model.from_architecture(architecture, weights)
    kinds = {name: type(module).__name__
             for name, module in model.layers.items()}
    assert kinds == {
        "first": "Conv1d", "first_activation": "ReLU", "dilated": "Conv1d",
        "sum_relu": "ReLU", "from_pooled": "Linear",
        "from_pooled_activation": "Tanh", "dropped": "Dropout",
        "from_flat": "Linear", "leaky": "LeakyReLU", "out": "Linear"}
    assert model.layers["leaky"].negative_slope == 0.2
    assert model.layers["dropped"].p == 0.5
    assert model.layers["dilated"].dilation == (2,)
    assert model.layers["dilated"].bias is None
    assert model.layers["from_pooled"].bias is None
    # every module once: a module that is used in two places could not
    # be told apart by a hook
    modules = list(model.layers.values())
    assert len({id(module) for module in modules}) == len(modules)

    # the slope of a LeakyReLU under the name later versions of keras
    # store it with, and the default of keras
    other = pytorch_model.from_architecture(
        changed(architecture, "leaky", alpha=None, negative_slope=0.1))
    assert other.layers["leaky"].negative_slope == 0.1
    other = pytorch_model.from_architecture(
        changed(architecture, "leaky", alpha=None))
    assert other.layers["leaky"].negative_slope == 0.3


def test_weights_go_there_and_back():
    architecture, weights = hand_made()
    model = pytorch_model.from_architecture(architecture, weights)
    back = pytorch_model.get_weights(model)
    assert sorted(back) == sorted(weights)
    for name in weights:
        assert sorted(back[name]) == sorted(weights[name])
        for key in weights[name]:
            assert back[name][key].shape == weights[name][key].shape
            assert np.array_equal(back[name][key], weights[name][key])
    # PyTorch's own layout: (out, in, width) and (out, in)
    assert tuple(model.layers["first"].weight.shape) == (5, 4, 3)
    assert tuple(model.layers["from_flat"].weight.shape) == (3, 120)
    for name in ("first", "dilated"):
        assert np.array_equal(
            model.layers[name].weight.detach().numpy(),
            weights[name]["kernel"].transpose(2, 1, 0))
    for name in ("from_pooled", "from_flat"):
        assert np.array_equal(model.layers[name].weight.detach().numpy(),
                              weights[name]["kernel"].T)

    # what is read back does not share memory with the model, in
    # either direction (also a kernel with one output, whose exchange
    # of axes needs no copy)
    before = pytorch_model.get_weights(model)
    for name in back:
        for key in back[name]:
            back[name][key][...] = 7.0
    after = pytorch_model.get_weights(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(-3.0)
    for name in weights:
        for key in weights[name]:
            assert np.array_equal(after[name][key], weights[name][key])
            assert np.array_equal(before[name][key], weights[name][key])
    # and the arrays that were given are not the model's memory either
    pytorch_model.set_weights(model, weights)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(-3.0)
    reference, _ = hand_made()[1], None
    for name in weights:
        for key in weights[name]:
            assert np.array_equal(weights[name][key], reference[name][key])
    pytorch_model.set_weights(model, weights)

    # the parameters under the names of the layers: what a state
    # dictionary of the model holds, and another model takes
    state = model.state_dict()
    assert sorted(state) == sorted(
        "layers.{}.{}".format(name, {"kernel": "weight"}.get(key, key))
        for name in weights for key in weights[name])
    other = pytorch_model.from_architecture(architecture).double()
    other.load_state_dict(state)
    for name in weights:
        for key in weights[name]:
            assert np.array_equal(pytorch_model.get_weights(other)[name][key],
                                  weights[name][key])



def as_type(weights, dtype):
    return {name: {key: value.astype(dtype) for key, value in saved.items()}
            for name, saved in weights.items()}


def test_model_has_the_type_of_the_saved_weights():
    architecture, weights = hand_made()
    for dtype, expected in ((np.float32, torch.float32),
                            (np.float64, torch.float64)):
        saved = as_type(weights, dtype)
        model = pytorch_model.from_architecture(architecture, saved)
        assert all(p.dtype == expected for p in model.parameters())
        back = pytorch_model.get_weights(model)
        for name in saved:
            for key in saved[name]:
                assert back[name][key].dtype == dtype
                assert np.array_equal(back[name][key], saved[name][key])

    # without weights: PyTorch's default, and weights that are set
    # later are converted to it
    model = pytorch_model.from_architecture(architecture)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    pytorch_model.set_weights(model, weights)
    assert model.layers["first"].weight.dtype == torch.float32
    assert np.array_equal(
        pytorch_model.get_weights(model)["first"]["kernel"],
        weights["first"]["kernel"].astype(np.float32))

    # one type for all the weights, and one the model can compute in
    mixed = as_type(weights, np.float32)
    mixed["out"]["bias"] = mixed["out"]["bias"].astype(np.float64)
    for saved in (mixed, as_type(weights, np.float16),
                  as_type(weights, np.int32)):
        with pytest.raises(UnsupportedModelError) as failure:
            pytorch_model.from_architecture(architecture, saved)
        assert "are not supported (all float32 or all float64)" in \
            str(failure.value)


def erf(x):
    return np.vectorize(math.erf)(x)


#: the activations by the names Keras stores, from their definitions
ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))),
    "selu": lambda x: 1.0507009873554805 * np.where(
        x > 0, x, 1.6732632423543772 * np.expm1(np.minimum(x, 0.0))),
    "softplus": lambda x: np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x))),
    "softsign": lambda x: x / (1.0 + np.abs(x)),
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1.0 + erf(x / np.sqrt(2.0))),
}


def one_dense_layer(activation, units=4):
    architecture = {
        "name": "one", "layers": [
            entry("InputLayer", "x", batch_input_shape=[None, 3]),
            entry("Dense", "dense", ["x"], units=units,
                  activation=activation)],
        "input_layers": [["x", 0, 0]], "output_layers": [["dense", 0, 0]]}
    rng = np.random.RandomState(7)
    weights = {"dense": {"kernel": rng.normal(size=(3, units)),
                         "bias": rng.normal(size=units)}}
    return architecture, weights


def test_supported_activations_are_the_tested_ones():
    assert sorted(pytorch_model._ACTIVATIONS) == sorted(ACTIVATIONS)


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_activation_built_into_a_layer(activation):
    architecture, weights = one_dense_layer(activation)
    model = pytorch_model.from_architecture(architecture, weights)
    assert sorted(model.layers) == ["dense", "dense_activation"]
    rng = np.random.RandomState(8)
    # also far out, where the exponentials are large or vanish
    x = np.concatenate([rng.normal(size=(40, 3)),
                        rng.normal(size=(40, 3)) * 30.0])
    before = x @ weights["dense"]["kernel"] + weights["dense"]["bias"]
    assert np.abs(before).max() > 40 and (np.abs(before) < 0.5).any()
    with torch.no_grad():
        found = model(torch.tensor(x)).numpy()
    np.testing.assert_allclose(found, ACTIVATIONS[activation](before),
                               rtol=1e-12, atol=1e-13)


def test_linear_layer_has_no_activation_module():
    for activation in ("linear", None):
        architecture, weights = one_dense_layer(activation)
        model = pytorch_model.from_architecture(architecture, weights)
        assert sorted(model.layers) == ["dense"]
    architecture, weights = one_dense_layer("linear")
    del architecture["layers"][1]["config"]["activation"]
    assert sorted(pytorch_model.from_architecture(
        architecture, weights).layers) == ["dense"]


def test_weights_must_have_the_type_the_layers_were_saved_with():
    """
        The layer graph says which type the variables of a layer had;
        weights of another type are not loaded as if nothing had
        happened (set_weights converts on purpose)
    """

    architecture, weights = hand_made()
    single = as_type(weights, np.float32)
    model = pytorch_model.from_architecture(architecture, weights)
    assert model.saved_types == {name: None for name in weights}

    double = changed(architecture, "first", dtype="float64")
    model = pytorch_model.from_architecture(double, weights)
    assert model.saved_types["first"] == "float64"
    assert model.layers["first"].weight.dtype == torch.float64
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(double, single)
    assert "Layer first: its saved weights are ['float32'], and the " \
        "layer was saved with the type float64" in str(failure.value)

    # under a mixed precision policy the variables are float32
    for policy in ({"class_name": "Policy",
                    "config": {"name": "mixed_float16"}},
                   {"module": "keras", "class_name": "DTypePolicy",
                    "config": {"name": "mixed_bfloat16"},
                    "registered_name": None}):
        mixed = changed(architecture, "out", dtype=policy)
        model = pytorch_model.from_architecture(mixed, single)
        assert model.saved_types["out"] == "float32"
        assert model.layers["out"].weight.dtype == torch.float32
        with pytest.raises(UnsupportedModelError) as failure:
            pytorch_model.from_architecture(mixed, weights)
        assert "Layer out: its saved weights are ['float64']" in \
            str(failure.value)

    # the other names TensorFlow has for the types (for numpy 'float'
    # is float64: these are TensorFlow's)
    for name, kind, right, wrong in (
            ("double", "float64", weights, single),
            ("float", "float32", single, weights),
            ("float32_ref", "float32", single, weights),
            ("float_ref", "float32", single, weights),
            ("float64_ref", "float64", weights, single),
            ("double_ref", "float64", weights, single)):
        other = changed(architecture, "first", dtype=name)
        model = pytorch_model.from_architecture(other, right)
        assert model.saved_types["first"] == kind
        assert str(model.layers["first"].weight.dtype) == "torch." + kind
        with pytest.raises(UnsupportedModelError) as failure:
            pytorch_model.from_architecture(other, wrong)
        assert "was saved with the type {}".format(kind) in \
            str(failure.value)
    for name, kind in (("half", "float16"), ("float16", "float16"),
                       ("half_ref", "float16")):
        assert pytorch_model.Model(changed(
            architecture, "first", dtype=name)).saved_types["first"] == kind
        with pytest.raises(UnsupportedModelError) as failure:
            pytorch_model.from_architecture(
                changed(architecture, "first", dtype=name), single)
        assert "was saved with the type float16" in str(failure.value)
    # a type that is none of these is not one the weights can have:
    # a model that declares it is refused, whatever its weights
    for name in ("bfloat16", "bfloat16_ref", "int32", "complex64",
                 "some_policy"):
        other = changed(architecture, "first", dtype=name)
        declared = name[:-4] if name.endswith("_ref") else name
        assert pytorch_model.Model(other).saved_types["first"] == declared
        for given in (single, weights):
            with pytest.raises(UnsupportedModelError) as failure:
                pytorch_model.from_architecture(other, given)
            assert "was saved with the type {}".format(declared) in \
                str(failure.value)
        # (without weights there is nothing to compare)
        pytorch_model.from_architecture(other)

    # on purpose
    model = pytorch_model.from_architecture(double)
    pytorch_model.set_weights(model, single)
    assert model.layers["first"].weight.dtype == torch.float32


def test_dropout_acts_in_training_mode_only():
    architecture, weights = hand_made()
    model = pytorch_model.from_architecture(architecture, weights)
    rng = np.random.RandomState(2)
    inputs = (torch.tensor(rng.normal(size=(4, 4, 30))),
              torch.tensor(rng.normal(size=(4, 2))))
    with torch.no_grad():
        a, b = model(*inputs)[1], model(*inputs)[1]
        assert torch.equal(a, b)
        model.train()
        torch.manual_seed(0)
        c = model(*inputs)[1]
        torch.manual_seed(1)
        d = model(*inputs)[1]
    assert not torch.equal(c, d) and not torch.equal(a, c)


def test_layout_of_batches():
    batch = np.arange(24.0).reshape(2, 3, 4)
    first = pytorch_model.channels_first(batch)
    assert first.shape == (2, 4, 3) and first.flags["C_CONTIGUOUS"]
    assert np.array_equal(first[1, 2], batch[1, :, 2])
    assert np.array_equal(pytorch_model.channels_last(first), batch)
    as_tensor = pytorch_model.channels_first(torch.tensor(batch))
    assert torch.equal(as_tensor, torch.tensor(first))
    counts = np.zeros((2, 5))
    assert pytorch_model.channels_first(counts) is counts


@pytest.mark.parametrize("change,message", [
    (lambda w: w.pop("dilated"), "No saved weights for layer dilated"),
    (lambda w: w.update(other={"kernel": np.zeros((1, 1))}),
     "Saved weights of layers ['other'] have no layer"),
    (lambda w: w["first"].update(kernel=np.zeros((3, 4, 6))),
     "Layer first: the saved kernel does not have the shape"),
    (lambda w: w["first"].update(kernel=np.zeros((3, 4))),
     "Layer first: the saved kernel has shape (3, 4)"),
    (lambda w: w["out"].update(bias=np.zeros(2)),
     "Layer out: the saved bias does not have the shape"),
    (lambda w: w["out"].pop("bias"), "Layer out: saved variables"),
    (lambda w: w["out"].update(moving_mean=np.zeros(1)),
     "Layer out: saved variables"),
    (lambda w: w["dilated"].update(bias=np.zeros(5)),
     "Layer dilated: saved variables ['bias', 'kernel'], the layer has "
     "['kernel']"),
    # a bias of one value is not spread over the five of the layer
    (lambda w: w["first"].update(bias=np.zeros(1)),
     "Layer first: the saved bias does not have the shape"),
])
def test_weights_must_match_the_layers(change, message):
    architecture, weights = hand_made()
    weights = copy.deepcopy(weights)
    change(weights)
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(architecture, weights)
    assert message in str(failure.value)


def changed(architecture, name, **settings):
    """
        The graph with other settings for one layer (None: without
        the setting)
    """

    architecture = copy.deepcopy(architecture)
    for layer in architecture["layers"]:
        if layer["name"] == name:
            layer["config"].update(settings)
            for key, value in settings.items():
                if value is None:
                    del layer["config"][key]
    return architecture


@pytest.mark.parametrize("name,settings,message", [
    ("together", {"axis": 0}, "concatenates along axis 0"),
    ("first", {"strides": [2]}, "strides"),
    ("first", {"padding": "causal"}, "padding causal is not supported"),
    ("first", {"activation": "exponential"},
     "activation exponential is not supported"),
    ("first", {"activation": {"class_name": "ReLU"}}, "activation"),
    ("first", {"kernel_size": [40]}, "shorter than its kernel"),
    ("cropped", {"cropping": [14, 14]}, "crops 28 of 28 positions"),
    ("cropped", {"cropping": [1, 2]}, "adds outputs of shapes"),
    # what Keras computes for a negative width is not a crop
    ("cropped", {"cropping": [0, -1]}, "cropping (0, -1) is not supported"),
    ("cropped", {"cropping": [-1, 0]}, "cropping (-1, 0) is not supported"),
    ("first", {"dtype": 5}, "dtype 5 is not supported"),
    ("sum_relu", {"max_value": 6.0}, "max_value"),
    ("first", {"groups": 2}, "groups"),
    ("first", {"data_format": "channels_first"}, "data_format"),
    ("dropped", {"noise_shape": [1, 120]}, "noise_shape"),
    ("pooled", {"keepdims": True}, "keepdims"),
])
def test_graphs_that_are_not_supported(name, settings, message):
    architecture, _ = hand_made()
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(
            changed(architecture, name, **settings))
    assert message in str(failure.value)


def test_layer_classes_and_lambdas_that_are_not_supported():
    architecture, _ = hand_made()
    other = copy.deepcopy(architecture)
    other["layers"][6]["class_name"] = "BatchNormalization"
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(other)
    assert "has class BatchNormalization" in str(failure.value)

    other = copy.deepcopy(architecture)
    for layer in other["layers"]:
        if layer["class_name"] == "Lambda":
            layer["name"] = layer["config"]["name"] = "slice_of_task_1"
        for node in layer["inbound_nodes"]:
            for source in node:
                if source[0] == "x_logsumexp_counts_bias_0":
                    source[0] = "slice_of_task_1"
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(other)
    assert "Lambda layer slice_of_task_1 is not supported" in \
        str(failure.value)

    other = copy.deepcopy(architecture)
    other["layers"][3]["inbound_nodes"][0][0][0] = "nowhere"
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(other)
    assert "take the output of layers that do not exist" in \
        str(failure.value)

    # a dense layer over the positions of a convolution's output
    other = copy.deepcopy(architecture)
    for layer in other["layers"]:
        if layer["name"] == "from_pooled":
            layer["inbound_nodes"][0][0][0] = "sum_relu"
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(other)
    assert "Layer from_pooled cannot take an input of shape (5, 24)" in \
        str(failure.value)

    other = copy.deepcopy(architecture)
    other["output_layers"][1][0] = "nowhere"
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(other)
    assert "The model has no layer nowhere" in str(failure.value)


def renamed(architecture, old, new):
    architecture = copy.deepcopy(architecture)
    for layer in architecture["layers"]:
        if layer["name"] == old:
            layer["name"] = layer["config"]["name"] = new
        for node in layer["inbound_nodes"]:
            for source in node:
                if source[0] == old:
                    source[0] = new
    for key in ("input_layers", "output_layers"):
        for endpoint in architecture[key]:
            if endpoint[0] == old:
                endpoint[0] = new
    return architecture


@pytest.mark.parametrize("old,new,message", [
    # two layers of one name
    ("leaky", "sum_relu", "Layer names are not unique"),
    # the name this module gives the activation that 'first' has
    # built in
    ("sum_relu", "first_activation",
     "Layer names are not unique: first_activation"),
    # also by a layer that has no module
    ("pooled", "first_activation",
     "Layer names are not unique: first_activation"),
    ("control", "from_pooled_activation",
     "Layer names are not unique: from_pooled_activation"),
    # names that a PyTorch module cannot have
    ("sum_relu", "sum.relu", "PyTorch cannot register a module"),
    ("sum_relu", "keys", "PyTorch cannot register a module"),
])
def test_layer_names_that_cannot_be_used(old, new, message):
    architecture, _ = hand_made()
    with pytest.raises(UnsupportedModelError) as failure:
        pytorch_model.from_architecture(renamed(architecture, old, new))
    assert message in str(failure.value)


def test_layers_listed_in_any_order():
    """
        The order of the layers in the file does not matter: every
        layer is built after the layers it takes its input from
    """

    architecture, weights = hand_made()
    rng = np.random.RandomState(3)
    inputs = (torch.tensor(rng.normal(size=(3, 4, 30))),
              torch.tensor(rng.normal(size=(3, 2))))
    with torch.no_grad():
        expected = pytorch_model.from_architecture(
            architecture, weights)(*inputs)
    for order in (list(reversed(range(16))),
                  list(np.random.RandomState(4).permutation(16))):
        other = copy.deepcopy(architecture)
        other["layers"] = [other["layers"][i] for i in order]
        model = pytorch_model.from_architecture(other, weights)
        with torch.no_grad():
            found = model(*inputs)
        assert all(torch.equal(a, b) for a, b in zip(found, expected))


def test_inputs_must_be_the_models():
    architecture, weights = hand_made()
    model = pytorch_model.from_architecture(architecture, weights)
    sequence, control = torch.zeros(2, 4, 30), torch.zeros(2, 2)
    with pytest.raises(ValueError):
        model(sequence)
    with pytest.raises(ValueError):
        model(sequence, control, control)
    with pytest.raises(ValueError):
        model(sequence, other=control)
    # one input twice, by position and by name
    with pytest.raises(ValueError):
        model(sequence, control, control=control)
    with pytest.raises(ValueError):
        model(sequence, sequence=sequence, control=control)
    # Keras' layout is refused, not misread
    with pytest.raises(ValueError) as failure:
        model(torch.zeros(2, 30, 4), control)
    assert "Input sequence has shape (30, 4)" in str(failure.value)


# --------------------------------------------------------------------
# models that TensorFlow saved
# --------------------------------------------------------------------

def to_torch(inputs, model, dtype):
    return [torch.tensor(pytorch_model.channels_first(
        np.asarray(inputs[name])).astype(dtype))
        for name in model.input_names]


@pytest.fixture(scope="module", params=FIXTURE_VERSIONS)
def saved_by(request, tmp_path_factory):
    """
        The models that the given TensorFlow version saved (see
        tests/data/saved_models/make_fixtures.py), unpacked
    """

    target = str(tmp_path_factory.mktemp("fixtures"))
    return {"version": request.param,
            "dir": unpack_fixture_models(request.param, target)}


@pytest.mark.parametrize("directory,file_name", fixture_files())
def test_models_saved_by_every_tensorflow_version(saved_by, directory,
                                                  file_name):
    """
        A model that TensorFlow 2.4.1, 2.11.0 or 2.15.1 saved gives in
        PyTorch the outputs it gave where it was saved. An HDF5 file
        is read without TensorFlow; the weights of a SavedModel
        directory are read with it.
    """

    case_dir = os.path.join(saved_by["dir"], directory)
    path = os.path.join(case_dir, file_name)
    if file_name == "model.h5":
        pytest.importorskip("h5py")
        if tuple(int(n) for n in saved_by["version"].split(".")[:2]) >= \
                (2, 13):
            # keras 2.13 and later write no layer graph to HDF5 for
            # this model class
            with pytest.raises(UnsupportedModelError,
                               match="without a layer graph"):
                pytorch_model.load_model(path)
            return
    else:
        pytest.importorskip("tensorflow")
    expected = np.load(os.path.join(case_dir, "expected.npz"))
    inputs = {key[len("input/"):]: expected[key] for key in expected.files
              if key.startswith("input/")}

    model = pytorch_model.load_model(path)
    assert sorted(model.input_names) == sorted(inputs)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with torch.no_grad():
        profile, counts = model(*to_torch(inputs, model, np.float32))
    profile = pytorch_model.channels_last(profile.numpy())
    counts = counts.numpy()
    assert profile.shape == expected["profile"].shape
    assert counts.shape == expected["counts"].shape
    for found, saved in ((profile, expected["profile"]),
                         (counts, expected["counts"])):
        np.testing.assert_allclose(found, saved, rtol=2e-5,
                                   atol=2e-5 * np.abs(saved).max())


# --------------------------------------------------------------------
# with TensorFlow: the models the package builds
# --------------------------------------------------------------------

@pytest.fixture(scope="module")
def tf():
    return pytest.importorskip("tensorflow")


def case_names():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return ["two strands with control"]
    from tests.test_arch import FORWARD_CASES
    return sorted(FORWARD_CASES)


@pytest.fixture(scope="module", params=case_names())
def saved(request, tf, tmp_path_factory):
    """
        A model of every architecture variant with random weights,
        saved the way the trainer saves it, and a random batch for it
    """

    from tests.test_arch import forward_case, predict

    made = forward_case(request.param)
    path = str(tmp_path_factory.mktemp("saved") / "model_split000")
    made["model"].save(path)
    made["path"] = path
    made["expected"] = predict(made["model"], made["inputs"])
    return made


def test_outputs_of_the_keras_model(saved):
    """
        float32, as the model was saved: the outputs agree with those
        of the Keras model to the rounding of float32
    """

    model = pytorch_model.load_model(saved["path"])
    assert not model.training
    assert model.input_names == [
        tensor.name.split(":")[0] for tensor in saved["model"].inputs]
    with torch.no_grad():
        found = model(*to_torch(saved["inputs"], model, np.float32))
    assert len(found) == len(saved["expected"]) == 2
    for a, b in zip(found, saved["expected"]):
        a = pytorch_model.channels_last(a.numpy())
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=2e-5,
                                   atol=2e-5 * np.abs(b).max())


def test_outputs_of_the_numpy_reference(saved):
    """
        In float64 the outputs are those of the numpy reference of the
        architecture, which shares no code with either framework
    """

    model = pytorch_model.load_model(saved["path"]).double()
    inputs = saved["inputs"]
    with torch.no_grad():
        profile, counts = model(*to_torch(inputs, model, np.float64))
    expected_profile, expected_counts = reference_model.forward(
        saved["weights"], saved["params"], inputs["sequence"],
        inputs.get("profile_bias_input_0"), inputs.get("counts_bias_input_0"),
        orig_multi_loss=saved["orig_multi_loss"])
    # (the saved weights are float32; the reference computes from them
    # in float64 as well)
    np.testing.assert_allclose(
        pytorch_model.channels_last(profile.numpy()), expected_profile,
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(counts.numpy(), expected_counts,
                               rtol=1e-10, atol=1e-10)


def test_weights_and_layers_of_the_keras_model(saved, tf):
    model = pytorch_model.load_model(saved["path"])
    stored = loading.read_weights(saved["path"])
    back = pytorch_model.get_weights(model)
    assert sorted(back) == sorted(stored)
    for name in stored:
        for key, value in stored[name].items():
            assert back[name][key].dtype == value.dtype == np.float32
            assert back[name][key].tobytes() == np.asarray(value).tobytes()

    # a module for every Keras layer that computes something, under
    # its name; activations that are built into a Keras layer besides
    by_class = {"Conv1D": "Conv1d", "Dense": "Linear", "ReLU": "ReLU",
                "LeakyReLU": "LeakyReLU", "Dropout": "Dropout"}
    expected = {}
    for layer in saved["model"].layers:
        kind = type(layer).__name__
        if kind in by_class:
            expected[layer.name] = by_class[kind]
            activation = getattr(layer, "activation", None)
            if kind in ("Conv1D", "Dense") and \
                    activation is not tf.keras.activations.linear:
                expected[layer.name + "_activation"] = {
                    "relu": "ReLU", "tanh": "Tanh"}[activation.__name__]
    assert {name: type(module).__name__
            for name, module in model.layers.items()} == expected


def test_gradient_of_the_outputs_with_respect_to_the_sequence(saved, tf):
    """
        What attribution methods start from: the gradient of the
        summed outputs with respect to the one hot input is that of
        the Keras model
    """

    inputs = saved["inputs"]
    keras = saved["model"]
    names = [tensor.name.split(":")[0] for tensor in keras.inputs]
    feed = {name: tf.constant(np.asarray(inputs[name]), dtype=tf.float32)
            for name in names}
    with tf.GradientTape() as tape:
        tape.watch(feed["sequence"])
        profile, counts = keras(feed, training=False)
        total = tf.reduce_sum(profile) + tf.reduce_sum(counts)
    expected = tape.gradient(total, feed["sequence"]).numpy()

    model = pytorch_model.load_model(saved["path"])
    tensors = to_torch(inputs, model, np.float32)
    tensors[0].requires_grad_(True)
    profile, counts = model(*tensors)
    (profile.sum() + counts.sum()).backward()
    found = pytorch_model.channels_last(tensors[0].grad.numpy())
    np.testing.assert_allclose(found, expected, rtol=1e-4,
                               atol=1e-4 * np.abs(expected).max())


def test_model_with_same_padding(tf, tmp_path):
    from tests.model_helpers import (random_batch, randomize_weights,
                                     small_params)
    from tests.test_arch import build, predict

    # convolutions that keep the length; kernels of even width are
    # padded as Keras pads them (the odd position on the right)
    params = small_params(
        input_len=100, output_profile_len=100,
        motif_module_params={"padding": "same", "kernel_sizes": [8]},
        syntax_module_params={"padding": "same"},
        profile_head_params={"padding": "same", "kernel_size": 6})
    keras = build(params)
    randomize_weights(keras, seed=4)
    inputs, _, _ = random_batch(params, seed=6)
    path = str(tmp_path / "model")
    keras.save(path)
    model = pytorch_model.load_model(path)
    assert model.shapes["main_profile_head"][1] == 100
    with torch.no_grad():
        found = model(*to_torch(inputs, model, np.float32))
    for a, b in zip(found, predict(keras, inputs)):
        np.testing.assert_allclose(
            pytorch_model.channels_last(a.numpy()), b, rtol=2e-5,
            atol=2e-5 * np.abs(b).max())


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_activations_are_those_of_keras(tf, activation):
    try:
        theirs = tf.keras.activations.get(activation)
    except ValueError:
        pytest.skip("this keras has no activation {}".format(activation))
    rng = np.random.RandomState(9)
    x = np.concatenate([rng.normal(size=200), rng.normal(size=200) * 30.0])
    expected = theirs(tf.constant(x, dtype=tf.float64)).numpy()
    module = pytorch_model._ACTIVATIONS[activation]()
    found = module(torch.tensor(x)).numpy()
    # (keras' gelu is 5e-9 from its definition in float64, on
    # TensorFlow 2.4.1 and 2.15.1 alike; its error function is exact)
    atol = 1e-8 if activation == "gelu" else 1e-12
    np.testing.assert_allclose(found, expected, rtol=1e-10, atol=atol)
    np.testing.assert_allclose(ACTIVATIONS[activation](x), expected,
                               rtol=1e-10, atol=atol)


def test_graph_that_keras_builds(tf, tmp_path):
    """
        Convolutions that keep the length with kernels and dilations
        for which the padding is not the same on both sides, one of
        them without a bias, activations built into layers, one
        output: against the Keras model, which is saved and read here
    """

    layers = tf.keras.layers
    x = layers.Input((50, 4), name="sequence")
    y = layers.Conv1D(5, 4, dilation_rate=3, padding="same",
                      activation="elu", name="a")(x)
    y = layers.Conv1D(3, 2, dilation_rate=5, padding="same",
                      use_bias=False, name="b")(y)
    y = layers.Conv1D(4, 6, padding="same", activation="softplus",
                      name="c")(y)
    y = layers.Flatten(name="flat")(y)
    y = layers.Dense(6, activation="selu", name="d")(y)
    y = layers.Dense(2, activation="sigmoid", use_bias=False, name="e")(y)
    keras = tf.keras.Model(x, y, name="small")
    rng = np.random.RandomState(10)
    keras.set_weights([rng.normal(0, 0.3, size=w.shape).astype(np.float32)
                       for w in keras.get_weights()])
    batch = rng.normal(size=(5, 50, 4)).astype(np.float32)
    expected = keras(tf.constant(batch), training=False).numpy()
    path = str(tmp_path / "model")
    keras.save(path)

    model = pytorch_model.load_model(path)
    assert {name: type(module).__name__
            for name, module in model.layers.items()} == {
        "a": "Conv1d", "a_activation": "ELU", "b": "Conv1d", "c": "Conv1d",
        "c_activation": "Softplus", "d": "Linear", "d_activation": "SELU",
        "e": "Linear", "e_activation": "Sigmoid"}
    assert model.layers["b"].bias is None and model.layers["e"].bias is None
    with torch.no_grad():
        found = model(torch.tensor(pytorch_model.channels_first(batch)))
    assert isinstance(found, torch.Tensor)
    np.testing.assert_allclose(found.numpy(), expected, rtol=2e-5,
                               atol=2e-5 * np.abs(expected).max())


@pytest.mark.parametrize("name", ["double", "float64"])
def test_layers_that_keras_saved_as_double(tf, tmp_path, name):
    """
        A model whose layers were given their type by one of the names
        TensorFlow has for float64: its weights are float64, and so is
        the model here, with the outputs of the Keras model
    """

    layers = tf.keras.layers
    # (a policy keeps the name it was given; a type that is given as
    # a string is stored as 'float64' whatever its spelling)
    x = layers.Input((6,), name="x", dtype="float64")
    y = layers.Dense(4, activation="tanh", name="d",
                     dtype=tf.keras.mixed_precision.Policy(name))(x)
    y = layers.Dense(2, name="e",
                     dtype=tf.keras.mixed_precision.Policy(name))(y)
    keras = tf.keras.Model(x, y, name="double")
    rng = np.random.RandomState(12)
    keras.set_weights([rng.normal(size=w.shape) for w in keras.get_weights()])
    batch = rng.normal(size=(5, 6))
    expected = keras(tf.constant(batch), training=False).numpy()
    assert expected.dtype == np.float64
    path = str(tmp_path / "model")
    keras.save(path)

    stored = {layer["name"]: layer["config"].get("dtype")
              for layer in loading.read_architecture(path)["layers"]}
    assert stored["d"] == stored["e"] == name
    model = pytorch_model.load_model(path)
    assert model.saved_types == {"d": "float64", "e": "float64"}
    assert all(p.dtype == torch.float64 for p in model.parameters())
    with torch.no_grad():
        found = model(torch.tensor(batch)).numpy()
    np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,kind", [
    ("float32_ref", "float32"), ("float_ref", "float32"),
    ("float64_ref", "float64"), ("double_ref", "float64")])
def test_layers_with_a_reference_type(tf, name, kind):
    """
        The reference types of TensorFlow 1 are valid types of a keras
        layer (with an initializer that accepts them), stored under
        their own names, with variables of the plain type
    """

    x = tf.keras.Input((3,), name="x", dtype=kind)
    dense = tf.keras.layers.Dense(
        2, name="dense", kernel_initializer="zeros",
        dtype=tf.keras.mixed_precision.Policy(name))
    keras = tf.keras.Model(x, dense(x))
    rng = np.random.RandomState(13)
    values = [rng.normal(size=w.shape).astype(kind)
              for w in keras.get_weights()]
    keras.set_weights(values)
    architecture = loading._plain(json.loads(keras.to_json()))["config"]
    stored = [layer["config"]["dtype"] for layer in architecture["layers"]
              if layer["name"] == "dense"]
    assert stored == [name]
    weights = {"dense": {"kernel": values[0], "bias": values[1]}}
    model = pytorch_model.from_architecture(architecture, weights)
    assert model.saved_types == {"dense": kind}
    batch = rng.normal(size=(4, 3)).astype(kind)
    expected = keras(tf.constant(batch)).numpy()
    with torch.no_grad():
        found = model(torch.tensor(batch)).numpy()
    assert found.dtype == expected.dtype == np.dtype(kind)
    np.testing.assert_allclose(found, expected, rtol=1e-5, atol=1e-6)
    # weights of the other precision are refused
    other = "float64" if kind == "float32" else "float32"
    with pytest.raises(UnsupportedModelError, match="was saved with"):
        pytorch_model.from_architecture(
            architecture, {"dense": {key: value.astype(other)
                                     for key, value in
                                     weights["dense"].items()}})


def test_model_saved_as_hdf5(tf, tmp_path):
    from tests.test_arch import forward_case, predict

    made = forward_case("two strands with control")
    path = str(tmp_path / "model.h5")
    made["model"].save(path)
    try:
        loading.read_architecture(path)
    except UnsupportedModelError as error:
        # keras 2.13 and later write no layer graph of this model
        # class to HDF5
        assert "without a layer graph" in str(error)
        pytest.skip(str(error))
    model = pytorch_model.load_model(path)
    with torch.no_grad():
        found = model(*to_torch(made["inputs"], model, np.float32))
    for a, b in zip(found, predict(made["model"], made["inputs"])):
        np.testing.assert_allclose(
            pytorch_model.channels_last(a.numpy()), b, rtol=2e-5,
            atol=2e-5 * np.abs(b).max())


@pytest.mark.parametrize("policy", ["float64", "mixed_float16"])
def test_model_saved_with_another_type(policy, tf, tmp_path):
    """
        A model that was saved in float64 computes in float64 here. A
        model saved under a mixed precision policy has float32 weights
        and computes in float32 here; Keras computes it in float16,
        and its outputs are those of this model to the rounding of
        float16 arithmetic.
    """

    from tests.test_loading import saved_under

    path, inputs, expected, types = saved_under(policy, tmp_path)
    model = pytorch_model.load_model(path)
    double = policy == "float64"
    assert all(p.dtype == (torch.float64 if double else torch.float32)
               for p in model.parameters())
    assert ("float16" in set(types.values())) == (not double)
    with torch.no_grad():
        found = model(*to_torch(inputs, model,
                                np.float64 if double else np.float32))
    # (float16: about 7e-4 of the largest output for these models)
    tolerance = 1e-10 if double else 5e-3
    for a, b in zip(found, expected):
        a = pytorch_model.channels_last(a.numpy())
        assert a.dtype == (np.float64 if double else np.float32)
        b = b.astype(np.float64)
        np.testing.assert_allclose(a, b, rtol=tolerance,
                                   atol=tolerance * np.abs(b).max())
