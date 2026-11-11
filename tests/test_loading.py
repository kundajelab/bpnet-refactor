"""
    Tests for bpnet.model.loading: a saved model is read from its
    files (layer graph and weights by name), rebuilt and gives the
    predictions of the model that was saved, in the TensorFlow version
    that saved it and in the others
"""

import contextlib
import json
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tensorflow.keras.optimizers import Adam

from bpnet.model import loading
from bpnet.model.custommodel import CustomModel

from tests.model_helpers import (FIXTURE_MODELS, FIXTURE_VERSIONS,
                                 fixture_files, make_tasks, random_batch,
                                 randomize_weights, small_params,
                                 unpack_fixture_models, weights_by_layer)
from tests.saved_weights import weights_by_layer_name
from tests.test_arch import FORWARD_CASES, build, forward_case, predict

TF_VERSION = tuple(int(x) for x in tf.__version__.split(".")[:2])


@pytest.fixture(scope="module", params=sorted(FORWARD_CASES))
def saved(request, tmp_path_factory):
    """
        A model of every architecture variant, saved the way the
        trainer saves it
    """

    made = forward_case(request.param)
    path = str(tmp_path_factory.mktemp("saved") / "model_split000")
    made["model"].save(path)
    made["path"] = path
    made["case"] = request.param
    return made


def test_loaded_model_predicts_what_the_saved_model_predicts(saved):
    loaded = loading.load_model(saved["path"])
    expected = predict(saved["model"], saved["inputs"])
    found = predict(loaded, saved["inputs"])
    assert len(found) == len(expected) == 2
    for a, b in zip(found, expected):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b)
    # and through model.predict, which bpnet-predict calls
    for a, b in zip(loaded.predict(saved["inputs"]), expected):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_loaded_model_has_the_layers_of_the_saved_model(saved):
    loaded = loading.load_model(saved["path"])
    model = saved["model"]
    # a plain keras model, without the training settings
    assert isinstance(loaded, tf.keras.Model)
    assert not isinstance(loaded, CustomModel)
    assert loaded.name == model.name
    assert [t.name.split(":")[0] for t in loaded.inputs] == \
        [t.name.split(":")[0] for t in model.inputs]
    assert [tuple(t.shape) for t in loaded.inputs] == \
        [tuple(t.shape) for t in model.inputs]
    assert [t.dtype for t in loaded.inputs] == \
        [t.dtype for t in model.inputs]
    assert [t.dtype for t in loaded.outputs] == \
        [t.dtype for t in model.outputs]
    assert [tuple(t.shape) for t in loaded.outputs] == \
        [tuple(t.shape) for t in model.outputs]
    assert {layer.name: type(layer).__name__ for layer in loaded.layers} == \
        {layer.name: type(layer).__name__ for layer in model.layers}
    expected = weights_by_layer(model)
    found = weights_by_layer(loaded)
    assert sorted(found) == sorted(expected)
    for name in expected:
        assert len(found[name]) == len(expected[name]) == 2
        for a, b in zip(found[name], expected[name]):
            assert np.array_equal(a, b), name


def test_architecture_and_weights_as_read_from_the_files(saved):
    model = saved["model"]
    architecture = loading.read_architecture(saved["path"])
    assert architecture["name"] == model.name
    assert {layer["name"]: layer["class_name"]
            for layer in architecture["layers"]} == \
        {layer.name: type(layer).__name__ for layer in model.layers}
    # plain JSON values only
    assert json.loads(json.dumps(architecture)) == architecture
    assert "__tuple__" not in json.dumps(architecture)

    weights = loading.read_weights(saved["path"])
    expected = weights_by_layer(model)
    assert sorted(weights) == sorted(expected)
    for name, (kernel, bias) in expected.items():
        assert sorted(weights[name]) == ["bias", "kernel"]
        assert np.array_equal(weights[name]["kernel"], kernel)
        assert np.array_equal(weights[name]["bias"], bias)
    # the reader of the tests agrees
    other = weights_by_layer_name(saved["path"])
    assert sorted(other) == sorted(weights)


def test_dropout_of_a_loaded_model_is_inactive_in_prediction(saved):
    if "hidden layers" not in saved["case"]:
        pytest.skip("only the counts heads with hidden layers have dropout")
    loaded = loading.load_model(saved["path"])
    first = predict(loaded, saved["inputs"])
    second = predict(loaded, saved["inputs"])
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def weight_digests(model):
    """
        layer name -> {variable name: sha256 of the array's bytes}
    """

    import hashlib

    digests = {}
    for layer in model.layers:
        for variable, value in zip(layer.weights, layer.get_weights()):
            key = variable.name.split("/")[-1].split(":")[0]
            digests.setdefault(layer.name, {})[key] = hashlib.sha256(
                np.ascontiguousarray(value).tobytes()).hexdigest()
    return digests


def trained_model(case="two strands with control"):
    """
        A model after two training steps: compiled, with optimizer
        state, as the trainer saves it
    """

    made = forward_case(case)
    model = made["model"]
    model.compile(Adam(learning_rate=0.01), loss=None)
    inputs, outputs, sample_weights = random_batch(made["params"], seed=5)
    for _ in range(2):
        model.train_step((inputs, outputs, tf.constant(sample_weights)))
    return made


def test_model_saved_after_training(tmp_path):
    made = trained_model()
    model = made["model"]
    path = str(tmp_path / "model")
    model.save(path)
    # the optimizer's variables are in the checkpoint, not in the result
    weights = loading.read_weights(path)
    assert sorted(weights) == sorted(weights_by_layer(model))
    loaded = loading.load_model(path)
    for a, b in zip(predict(loaded, made["inputs"]),
                    predict(model, made["inputs"])):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("case", [
    "two strands with control", "counts head with hidden layers",
    "unstranded without control"])
def test_model_saved_as_hdf5(case, tmp_path):
    import h5py

    made = forward_case(case)
    model = made["model"]
    path = str(tmp_path / "model.h5")
    model.save(path)
    with h5py.File(path, "r") as f:
        stored = f.attrs["model_config"]
    if isinstance(stored, bytes):
        stored = stored.decode("utf-8")
    if "layers" not in json.loads(stored)["config"]:
        # keras 2.13 and later write no layer graph for this model
        # class to HDF5; such a file cannot be rebuilt by anything
        assert TF_VERSION >= (2, 13)
        with pytest.raises(loading.UnsupportedModelError,
                           match="CustomModel model without a layer graph"):
            loading.load_model(path)
        return

    architecture = loading.read_architecture(path)
    assert {layer["name"] for layer in architecture["layers"]} == \
        {layer.name for layer in model.layers}
    weights = loading.read_weights(path)
    expected = weights_by_layer(model)
    assert sorted(weights) == sorted(expected)
    for name, (kernel, bias) in expected.items():
        assert sorted(weights[name]) == ["bias", "kernel"]
        assert np.array_equal(weights[name]["kernel"], kernel)
        assert np.array_equal(weights[name]["bias"], bias)
    loaded = loading.load_model(path)
    for a, b in zip(predict(loaded, made["inputs"]),
                    predict(model, made["inputs"])):
        assert np.array_equal(a, b)


def test_model_with_same_padding(tmp_path):
    # no cropping, and convolutions that keep the length
    params = small_params(
        input_len=100, output_profile_len=100,
        motif_module_params={"padding": "same"},
        syntax_module_params={"padding": "same"},
        profile_head_params={"padding": "same"})
    model = build(params)
    randomize_weights(model, seed=4)
    inputs, _, _ = random_batch(params, seed=6)
    path = str(tmp_path / "model")
    model.save(path)
    loaded = loading.load_model(path)
    for a, b in zip(predict(loaded, inputs), predict(model, inputs)):
        assert a.shape == b.shape and np.array_equal(a, b)


@contextlib.contextmanager
def default_type(policy):
    """
        'float64' or a mixed precision policy as the type of the
        layers that are built inside; back to keras' default after
    """

    if policy.startswith("mixed"):
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        tf.keras.backend.set_floatx(policy)
    try:
        yield
    finally:
        # no global policy: layers follow floatx again
        tf.keras.mixed_precision.set_global_policy(None)
        tf.keras.backend.set_floatx("float32")


def saved_under(policy, tmp_path, case="two strands with control"):
    """
        A model built and saved while `policy` was the default type of
        keras layers ('float64', 'mixed_float16'); the default is back
        to float32 afterwards

        Returns:
            tuple: (path, inputs, predictions of the model that was
                saved, the types its layers compute in)
    """

    with default_type(policy):
        made = forward_case(case)
        model = made["model"]
        path = str(tmp_path / "model")
        model.save(path)
        expected = predict(model, made["inputs"])
        types = {layer.name: layer.compute_dtype for layer in model.layers}
    return path, made["inputs"], expected, types


@pytest.mark.parametrize("policy", ["float64", "mixed_float16"])
def test_model_is_loaded_with_the_types_it_was_saved_with(policy, tmp_path):
    """
        The layers compute in the types the model was saved with, not
        in whatever is the default of the process that loads it
    """

    path, inputs, expected, types = saved_under(policy, tmp_path)
    loaded = loading.load_model(path)
    assert {layer.name: layer.compute_dtype for layer in loaded.layers} == \
        types
    found = predict(loaded, inputs)
    for a, b in zip(found, expected):
        assert a.dtype == b.dtype
        assert np.array_equal(a, b)
    if policy == "float64":
        assert found[0].dtype == np.float64
        assert all(w.dtype == np.float64 for w in loaded.get_weights())
    else:
        assert "float16" in set(types.values())


@pytest.mark.parametrize("ambient", ["float64", "mixed_float16"])
def test_float32_model_loaded_where_another_type_is_the_default(
        saved, ambient):
    with default_type(ambient):
        loaded = loading.load_model(saved["path"])
        found = predict(loaded, saved["inputs"])
    assert all(layer.compute_dtype == "float32" for layer in loaded.layers)
    for a, b in zip(found, predict(saved["model"], saved["inputs"])):
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a, b)


def test_unknown_layer_type_setting_is_rejected(released_like):
    architecture = changed(released_like["architecture"], "main_conv_0",
                           dtype={"class_name": "Quantized", "config": {}})
    with pytest.raises(loading.UnsupportedModelError,
                       match="main_conv_0: dtype"):
        loading.build_model(architecture)


@pytest.mark.parametrize("class_name,config,message", [
    ("Flatten", {"data_format": "channels_first"}, "data_format"),
    ("GlobalAveragePooling1D", {"data_format": "channels_first"},
     "data_format"),
    ("Dropout", {"rate": 0.5, "noise_shape": [1, 2]}, "noise_shape"),
    ("Conv1D", {"filters": 2, "kernel_size": [3], "padding": "valid",
                "strides": [2]}, "strides"),
])
def test_layer_settings_that_are_not_supported(class_name, config, message):
    config = dict(config, name="layer")
    with pytest.raises(loading.UnsupportedModelError, match=message):
        loading._make_layer(class_name, config)


def test_layer_settings_are_taken_from_the_file():
    """
        Settings that every model of this package leaves at one value
        are still read from the file, not assumed
    """

    layer = loading._make_layer("LeakyReLU", {"name": "a", "alpha": 0.1})
    assert float(layer.alpha) == pytest.approx(0.1)
    assert float(loading._make_layer(
        "LeakyReLU", {"name": "b"}).alpha) == pytest.approx(0.3)

    layer = loading._make_layer("Dropout", {"name": "c", "rate": 0.4})
    assert layer.rate == pytest.approx(0.4)

    x = tf.constant(np.arange(24, dtype=np.float32).reshape(1, 6, 4))
    layer = loading._make_layer("Cropping1D",
                                {"name": "d", "cropping": [1, 3]})
    assert np.array_equal(layer(x).numpy(), x.numpy()[:, 1:3])

    layer = loading._make_layer("Concatenate", {"name": "e", "axis": 1})
    assert layer([x, x]).shape == (1, 12, 4)

    layer = loading._make_layer("Conv1D", {
        "name": "f", "filters": 3, "kernel_size": [2], "padding": "same",
        "dilation_rate": [2], "activation": "relu", "use_bias": False})
    y = layer(x)
    assert y.shape == (1, 6, 3)
    assert layer.dilation_rate == (2,) and not layer.use_bias
    assert float(tf.reduce_min(y)) >= 0.0 and len(layer.weights) == 1

    layer = loading._make_layer("Dense", {
        "name": "g", "units": 2, "activation": "tanh", "use_bias": False})
    y = layer(tf.constant([[5.0, -7.0, 9.0]]))
    assert float(tf.reduce_max(tf.abs(y))) <= 1.0 and len(layer.weights) == 1


def test_hdf5_file_with_weights_only(tmp_path):
    made = forward_case("two strands with control")
    path = str(tmp_path / "weights.h5")
    made["model"].save_weights(path)
    with pytest.raises(loading.UnsupportedModelError,
                       match="no model configuration"):
        loading.load_model(path)


NATIVE_LOADER = """
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from bpnet.model.custommodel import CustomModel

for path in sys.argv[1:]:
    inputs = dict(np.load(path + '.inputs.npz'))
    # as bpnet-predict has always loaded its model
    with CustomObjectScope({'tf': tf, 'CustomModel': CustomModel}):
        model = load_model(path)
    profile, counts = model.predict(inputs)
    np.savez(path + '.native.npz', profile=profile, counts=counts)
"""


@pytest.mark.skipif(TF_VERSION >= (2, 5), reason="keras' own loader only "
                    "loads these models under TensorFlow 2.4")
def test_same_predictions_as_keras_own_loader(tmp_path):
    """
        Under TensorFlow 2.4, where keras' loader works, the model it
        returns and the model of this loader predict the same numbers,
        bit for bit. keras' loader runs in a process of its own: it
        only succeeds in a process that has not built a model yet (the
        first CustomModel that is built changes the base classes of
        the class, after which keras cannot rebuild a saved one).
    """

    import subprocess
    import sys

    cases = ["two strands with control", "unstranded without control",
             "counts head with hidden layers", "smoothed control tracks",
             "original loss with control", "two motif convolutions"]
    paths, inputs = [], []
    for i, case in enumerate(cases):
        made = forward_case(case)
        path = str(tmp_path / "model_{}".format(i))
        made["model"].save(path)
        np.savez(path + ".inputs.npz", **made["inputs"])
        paths.append(path)
        inputs.append(made["inputs"])

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    if not env.get("BPNET_TEST_GPU"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run(
        [sys.executable, "-c", NATIVE_LOADER] + paths, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:]

    for case, path, batch in zip(cases, paths, inputs):
        native = np.load(path + ".native.npz")
        profile, counts = loading.load_model(path).predict(batch)
        assert np.array_equal(profile, native["profile"]), case
        assert np.array_equal(counts, native["counts"]), case


def test_keras_own_loader_fails_once_a_model_was_built(tmp_path):
    """
        Why this loader exists, as a test: in a process that has built
        a model, keras cannot load a saved one in any supported
        TensorFlow version (and from 2.5 on in no process at all).
        When this starts to pass, the loader may no longer be needed.
    """

    made = forward_case("two strands with control")
    path = str(tmp_path / "model")
    made["model"].save(path)
    with pytest.raises(TypeError, match="missing .* required positional"):
        tf.keras.models.load_model(
            path, custom_objects={"tf": tf, "CustomModel": CustomModel})


# -------------------- models saved by other versions --------------------

@pytest.fixture(scope="module", params=FIXTURE_VERSIONS)
def fixture_models(request, tmp_path_factory):
    """
        The models that the given TensorFlow version saved (see
        tests/data/saved_models/make_fixtures.py), unpacked
    """

    target = str(tmp_path_factory.mktemp("fixtures"))
    return {"version": request.param,
            "dir": unpack_fixture_models(request.param, target)}


@pytest.mark.parametrize("directory,file_name", fixture_files())
def test_models_saved_by_every_version_load_in_this_one(
        fixture_models, directory, file_name):
    """
        A model saved under TensorFlow 2.4.1, 2.11.0 or 2.15.1, as a
        SavedModel directory or as an HDF5 file, loads here with the
        weights it was saved with and predicts what it predicted where
        it was saved
    """

    import hashlib

    case_dir = os.path.join(fixture_models["dir"], directory)
    path = os.path.join(case_dir, file_name)
    expected = np.load(os.path.join(case_dir, "expected.npz"))
    inputs = {key[len("input/"):]: expected[key] for key in expected.files
              if key.startswith("input/")}
    digests = json.loads(str(expected["weight_digests"]))

    saved_version = tuple(
        int(x) for x in fixture_models["version"].split(".")[:2])
    if file_name == "model.h5" and saved_version >= (2, 13):
        # keras 2.13 and later write no layer graph to HDF5 for this
        # model class
        with pytest.raises(loading.UnsupportedModelError,
                           match="without a layer graph"):
            loading.load_model(path)
        return

    weights = loading.read_weights(path)
    assert sorted(weights) == sorted(digests)
    for layer, variables in digests.items():
        assert sorted(weights[layer]) == sorted(variables)
        for variable, digest in variables.items():
            value = weights[layer][variable]
            assert value.dtype == np.float32
            assert hashlib.sha256(np.ascontiguousarray(
                value).tobytes()).hexdigest() == digest, (layer, variable)

    model = loading.load_model(path)
    profile, counts = predict(model, inputs)
    assert profile.shape == expected["profile"].shape
    assert counts.shape == expected["counts"].shape
    # float32 convolutions differ in their last bits between versions
    np.testing.assert_allclose(profile, expected["profile"],
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(counts, expected["counts"],
                               rtol=1e-5, atol=1e-5)


def test_fixture_models_cover_every_layer_class_the_package_builds(
        fixture_models):
    classes = set()
    for directory in FIXTURE_MODELS:
        architecture = loading.read_architecture(
            os.path.join(fixture_models["dir"], directory, "model"))
        classes.update(layer["class_name"]
                       for layer in architecture["layers"])
    assert classes == {
        "InputLayer", "Conv1D", "ReLU", "Cropping1D", "Add",
        "GlobalAveragePooling1D", "Flatten", "Dropout", "Dense", "LeakyReLU",
        "Lambda", "Concatenate"}


# -------------------- what cannot be loaded --------------------

@pytest.fixture(scope="module")
def released_like(tmp_path_factory):
    made = forward_case("two strands with control")
    path = str(tmp_path_factory.mktemp("saved") / "model")
    made["model"].save(path)
    return {"architecture": loading.read_architecture(path),
            "weights": loading.read_weights(path)}


def changed(architecture, layer_name, **settings):
    """
        A copy of the layer graph with settings of one layer replaced;
        'class_name' and 'inbound_nodes' replace those of the layer
    """

    architecture = json.loads(json.dumps(architecture))
    layer, = [l for l in architecture["layers"] if l["name"] == layer_name]
    for key, value in settings.items():
        if key in ("class_name", "inbound_nodes"):
            layer[key] = value
        else:
            layer["config"][key] = value
    return architecture


@pytest.mark.parametrize("layer,settings,message", [
    ("main_conv_0", {"strides": [2]}, "main_conv_0: strides"),
    ("main_conv_0", {"groups": 2}, "main_conv_0: groups"),
    ("main_conv_0", {"data_format": "channels_first"}, "data_format"),
    ("main_conv_0", {"kernel_size": [3, 3]}, "one value expected"),
    ("main_conv_0", {"activation": {"class_name": "X"}}, "activation"),
    ("main_conv_0_relu", {"max_value": 6.0}, "max_value"),
    ("main_conv_0_relu", {"negative_slope": 0.1}, "negative_slope"),
    ("main_conv_0_relu", {"threshold": 1.0}, "threshold"),
    ("main_global_avg_pooling", {"keepdims": True}, "keepdims"),
    ("main_conv_0_relu", {"class_name": "LSTM"}, "class LSTM"),
    ("main_logsumexp_counts_bias_0", {"name": "main_slice_0"},
     "Lambda layer main_slice_0"),
    ("sequence", {"sparse": True}, "sparse"),
    ("main_conv_0", {"inbound_nodes": [[["sequence", 0, 0, {}]],
                                       [["sequence", 0, 0, {}]]]},
     "applied 2 times"),
    ("main_conv_0", {"inbound_nodes": [[["sequence", 0, 1, {}]]]},
     "only single outputs"),
    ("main_conv_0", {"inbound_nodes": [[["nowhere", 0, 0, {}]]]},
     "do not exist"),
    ("main_conv_0", {"inbound_nodes": []}, "has no input"),
    ("main_conv_0", {"inbound_nodes": [[["sequence", 0, 0, {}],
                                        ["sequence", 0, 0, {}]]]},
     "takes 2 inputs"),
])
def test_unsupported_layer_graphs_are_rejected(released_like, layer,
                                               settings, message):
    architecture = changed(released_like["architecture"], layer, **settings)
    with pytest.raises(loading.UnsupportedModelError, match=message):
        loading.build_model(architecture)


def test_unchanged_layer_graph_builds(released_like):
    # the helper of the test above leaves a loadable graph when it
    # changes nothing
    model = loading.build_model(
        changed(released_like["architecture"], "main_conv_0"))
    loading.set_weights(model, released_like["weights"])


def test_settings_absent_from_older_files_are_accepted(released_like):
    architecture = json.loads(json.dumps(released_like["architecture"]))
    for layer in architecture["layers"]:
        for key in ("groups", "keepdims", "sparse", "ragged", "dtype",
                    "trainable"):
            layer["config"].pop(key, None)
    model = loading.build_model(architecture)
    loading.set_weights(model, released_like["weights"])


def test_weights_must_match_the_layers(released_like):
    architecture = released_like["architecture"]
    weights = released_like["weights"]

    def copy_of(weights):
        return {name: dict(variables) for name, variables in weights.items()}

    model = loading.build_model(architecture)

    missing = copy_of(weights)
    del missing["main_profile_head"]
    with pytest.raises(loading.UnsupportedModelError,
                       match="No saved weights for layer main_profile_head"):
        loading.set_weights(model, missing)

    no_bias = copy_of(weights)
    del no_bias["main_counts_head"]["bias"]
    with pytest.raises(loading.UnsupportedModelError,
                       match="No saved bias for layer main_counts_head"):
        loading.set_weights(model, no_bias)

    wrong_shape = copy_of(weights)
    wrong_shape["main_conv_0"]["kernel"] = \
        wrong_shape["main_conv_0"]["kernel"][:-1]
    with pytest.raises(loading.UnsupportedModelError,
                       match="main_conv_0: the saved kernel has shape"):
        loading.set_weights(model, wrong_shape)

    extra_variable = copy_of(weights)
    extra_variable["main_conv_0"]["moving_mean"] = np.zeros(3)
    with pytest.raises(loading.UnsupportedModelError,
                       match="main_conv_0: saved variables"):
        loading.set_weights(model, extra_variable)

    wrong_type = copy_of(weights)
    wrong_type["main_conv_0"]["kernel"] = \
        wrong_type["main_conv_0"]["kernel"].astype(np.float64)
    with pytest.raises(loading.UnsupportedModelError,
                       match="main_conv_0: the saved kernel is float64"):
        loading.set_weights(model, wrong_type)

    extra_layer = copy_of(weights)
    extra_layer["another_conv"] = dict(weights["main_conv_0"])
    with pytest.raises(loading.UnsupportedModelError,
                       match="another_conv"):
        loading.set_weights(model, extra_layer)

    # nothing was wrong with the model itself
    loading.set_weights(model, weights)
