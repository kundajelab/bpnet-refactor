"""
    Helpers shared by the model tests: task descriptions, small
    architectures, random weights and random batches
"""

import copy
import os

import numpy as np

#: receptive field arithmetic of SMALL_PARAMS: 20 + 2 * (2 + 4) + 74
SMALL_SHRINKAGE = 106

#: a small architecture with the structure of the released one
SMALL_PARAMS = {
    "input_len": 186,
    "output_profile_len": 80,
    "motif_module_params": {
        "filters": [6], "kernel_sizes": [21], "padding": "valid"},
    "syntax_module_params": {
        "num_dilation_layers": 2, "filters": 6, "kernel_size": 3,
        "padding": "valid", "pre_activation_residual_unit": True},
    "profile_head_params": {
        "filters": 1, "kernel_size": 75, "padding": "valid"},
    "counts_head_params": {
        "units": [1], "dropouts": [0.0], "activations": ["linear"]},
    "profile_bias_module_params": {"kernel_sizes": [1]},
    "counts_bias_module_params": {},
    "loss_weights": [1.0, 7.5],
    "counts_loss": "MSE",
}

#: the architecture of the released ENCODE models (bpnet_params.json of
#: the release; the counts loss weight differs per experiment)
RELEASED_PARAMS = {
    "input_len": 2114,
    "output_profile_len": 1000,
    "motif_module_params": {
        "filters": [64], "kernel_sizes": [21], "padding": "valid"},
    "syntax_module_params": {
        "num_dilation_layers": 8, "filters": 64, "kernel_size": 3,
        "padding": "valid", "pre_activation_residual_unit": True},
    "profile_head_params": {
        "filters": 1, "kernel_size": 75, "padding": "valid"},
    "counts_head_params": {
        "units": [1], "dropouts": [0.0], "activations": ["linear"]},
    "profile_bias_module_params": {"kernel_sizes": [1]},
    "counts_bias_module_params": {},
    "loss_weights": [1.0, 88.3],
    "counts_loss": "MSE",
}


def small_params(**overrides):
    """
        A deep copy of SMALL_PARAMS with top level keys replaced
    """

    params = copy.deepcopy(SMALL_PARAMS)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(params.get(key), dict):
            params[key].update(value)
        else:
            params[key] = value
    return params


def make_tasks(n_signal=2, n_bias=2, n_tasks=1, smoothing=None):
    """
        The tasks dictionary the model builder takes; only the number of
        signal and control tracks per task matters to it
    """

    tasks = {}
    for i in range(n_tasks):
        tasks[i] = {
            "signal": {"source": ["signal_{}_{}".format(i, j)
                                  for j in range(n_signal)]},
            "bias": {"source": ["control_{}_{}".format(i, j)
                                for j in range(n_bias)],
                     "smoothing": smoothing if smoothing is not None
                     else [None] * n_bias},
        }
    return tasks


def weights_by_layer(model):
    """
        layer name -> list of weight arrays, for layers that have weights
    """

    return {layer.name: layer.get_weights() for layer in model.layers
            if layer.get_weights()}


def randomize_weights(model, seed=0, scale=0.2):
    """
        Give every weight (kernels and biases) a random value, so that
        tests do not pass because Keras initializes biases to zero

        Returns:
            dict: layer name -> list of float32 arrays that were set
    """

    rng = np.random.RandomState(seed)
    out = {}
    for layer in model.layers:
        current = layer.get_weights()
        if not current:
            continue
        new = [rng.normal(0.0, scale, size=w.shape).astype(np.float32)
               for w in current]
        layer.set_weights(new)
        out[layer.name] = new
    return out


def random_batch(params, batch_size=6, n_signal=2, n_bias=2, seed=0,
                 weights=None):
    """
        A random batch in the format the generator produces

        Returns:
            tuple: (inputs dict, outputs dict, sample weights)
    """

    rng = np.random.RandomState(seed)
    input_len = params["input_len"]
    output_len = params["output_profile_len"]

    tokens = rng.randint(0, 5, size=(batch_size, input_len))
    sequence = np.zeros((batch_size, input_len, 4), dtype=np.float64)
    for base in range(4):
        sequence[:, :, base] = tokens == base

    profiles = rng.poisson(
        rng.uniform(0.05, 2.0, size=(batch_size, 1, n_signal)),
        size=(batch_size, output_len, n_signal)).astype(np.float32)
    inputs = {"sequence": sequence}
    if n_bias > 0:
        bias = rng.poisson(
            0.3, size=(batch_size, output_len, n_bias)).astype(np.float32)
        inputs["profile_bias_input_0"] = bias
        inputs["counts_bias_input_0"] = np.log(bias.sum(axis=1) + 1)
    outputs = {
        "profile_predictions": profiles,
        "logcounts_predictions": np.log(profiles.sum(axis=1) + 1),
    }
    if weights is None:
        weights = np.array(
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0] * (batch_size // 6 + 1),
            dtype=np.float64)[:batch_size]
    return inputs, outputs, np.asarray(weights, dtype=np.float64)


# -------------------- models saved by other versions --------------------

#: the archives of tests/data/saved_models: one per TensorFlow version
#: that saved the models (see make_fixtures.py there)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "saved_models")
FIXTURE_VERSIONS = ["2.4.1", "2.11.0", "2.15.1"]

#: the models of an archive: directory name in the archive ->
#: (architecture variant, saved after training steps with the
#: optimizer's state, also saved as HDF5)
FIXTURE_MODELS = {
    "released_structure": ("two strands with control", True, True),
    "hidden_counts_layers": ("counts head with hidden layers", False, False),
}


def fixture_files():
    """
        (directory, file name) of every saved model of an archive
    """

    files = []
    for directory, (_, _, hdf5) in sorted(FIXTURE_MODELS.items()):
        files.append((directory, "model"))
        if hdf5:
            files.append((directory, "model.h5"))
    return files


def unpack_fixture_models(version, target):
    """
        Unpack the models that a TensorFlow version saved

        Args:
            version (str): one of FIXTURE_VERSIONS
            target (str): directory to unpack into

        Returns:
            str: the directory that holds one folder per model
    """

    import tarfile

    name = "tf{}".format(version)
    with tarfile.open(os.path.join(FIXTURES, name + ".tar.gz")) as tar:
        members = tar.getmembers()
        # only plain files and directories below the archive's folder
        for member in members:
            parts = member.name.split("/")
            assert member.isfile() or member.isdir(), member.name
            assert parts[0] == name and ".." not in parts, member.name
            assert not os.path.isabs(member.name), member.name
        tar.extractall(target, members=members)
    return os.path.join(target, name)
