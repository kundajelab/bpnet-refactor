"""
    Tests for the seeded initial weights (bpnet.model.initialization):
    the numpy generator reproduces TensorFlow's random kernel and the
    weights Keras gave a model under TensorFlow 2.4.1
"""

import argparse
import copy
import hashlib
import random

import numpy as np
import pytest

from bpnet.cli import argparsers
from bpnet.model import initialization

# sha256 over the initial kernels and biases of the released
# architecture, recorded from the weights an unmodified bpnet-train run
# (upstream main at 0b11a7c, before this module existed) started from
# under TensorFlow 2.4.1 / Python 3.7, the stack the ENCODE models were
# trained with
RELEASED_SEED_1234_DIGEST = \
    "80359736a91cd5776a068eb5a1a0ee3554f31e32380141d7a9e9b7a268a17943"

# kernel shapes of the released architecture in creation order
RELEASED_SHAPES = [(21, 4, 64)] + [(3, 64, 64)] * 8 + \
    [(75, 64, 2), (64, 1), (1, 4, 2), (2, 1)]
RELEASED_BIAS_SIZES = [64] * 9 + [2, 1, 2, 1]


# --------------------------------------------------------------------
# without TensorFlow
# --------------------------------------------------------------------

def test_released_architecture_seed_1234_matches_the_tf241_record():
    kernels = initialization.initial_kernels(RELEASED_SHAPES, 1234)
    arrays = []
    for kernel, n in zip(kernels, RELEASED_BIAS_SIZES):
        arrays.extend([kernel, np.zeros(n, dtype=np.float32)])
    assert initialization.weights_digest(arrays) == RELEASED_SEED_1234_DIGEST
    # the two weights that combine the count prediction with the
    # control counts
    assert kernels[-1].ravel().tolist() == \
        [-0.08632349967956543, 0.41392314434051514]
    assert kernels[0][0, 0, :3].tolist() == \
        [0.004931449890136719, -0.017550937831401825, 0.010582588613033295]


def test_operation_seeds_are_the_python_random_stream():
    assert initialization.operation_seeds(1234, 4) == \
        [1892932127, 501869158, 32175636, 389311301]
    assert initialization.operation_seeds(7, 3) == \
        [1390851128, 647892279, 1695753998]
    stream = random.Random(99)
    assert initialization.operation_seeds(99, 5) == \
        [stream.randint(0, 2 ** 31 - 1) for _ in range(5)]


def test_random_uniform_reference_values():
    # values of tf.raw_ops.RandomUniform recorded under TF 2.4.1
    cases = {
        (1234, 5678): [0.666307806968689, 0.9674705266952515,
                       0.8894475698471069],
        (0, 1): [0.5397301912307739, 0.10493969917297363,
                 0.20118176937103271],
        (2147483647, 2147483647): [0.9800184965133667, 0.5922853946685791,
                                   0.10167217254638672],
        (42, 0): [0.8354591131210327, 0.9162973165512085,
                  0.4946495294570923],
    }
    for (seed, seed2), expected in cases.items():
        values = initialization.random_uniform(11, seed, seed2)
        assert values.dtype == np.float32
        assert values[:3].tolist() == expected
        # a longer request extends the same stream
        assert np.array_equal(
            initialization.random_uniform(1000, seed, seed2)[:11], values)


def test_random_uniform_is_uniform_on_the_unit_interval():
    values = initialization.random_uniform(200000, 5, 6)
    assert values.min() >= 0.0 and values.max() < 1.0
    np.testing.assert_allclose(values.mean(), 0.5, atol=0.005)
    np.testing.assert_allclose(values.std(), np.sqrt(1 / 12.0), atol=0.005)
    histogram, _ = np.histogram(values, bins=20, range=(0.0, 1.0))
    assert histogram.min() > 0.9 * 10000 and histogram.max() < 1.1 * 10000


@pytest.mark.parametrize("shape", [(21, 4, 64), (3, 64, 64), (64, 1), (2, 1),
                                   (1, 4, 2), (75, 8, 2)])
def test_glorot_uniform_limits(shape):
    kernel = initialization.glorot_uniform(shape, 1234, 99)
    assert kernel.shape == shape and kernel.dtype == np.float32
    receptive = int(np.prod(shape[:-2]))
    limit = np.sqrt(6.0 / (receptive * (shape[-2] + shape[-1])))
    assert np.abs(kernel).max() <= limit
    if kernel.size > 1000:
        assert np.abs(kernel).max() > 0.98 * limit
        np.testing.assert_allclose(kernel.std(), limit / np.sqrt(3.0),
                                   rtol=0.05)


def test_kernel_seeds():
    largest = 2 ** 31 - 1
    assert initialization.kernel_seeds(1234, 5) == (1234, 5)
    # both are reduced modulo 2**31 - 1
    assert initialization.kernel_seeds(largest, 5) == (0, 5)
    assert initialization.kernel_seeds(largest + 6, largest + 9) == (6, 9)
    assert initialization.kernel_seeds(-5, 3) == (largest - 5, 3)
    # the kernel reads (0, 0) as "not seeded"
    assert initialization.kernel_seeds(0, 0) == (0, largest)
    assert initialization.kernel_seeds(largest, largest) == (0, largest)
    assert initialization.kernel_seeds(0, 1) == (0, 1)


def test_initial_kernels_of_a_seed_beyond_the_kernel_range():
    seed = 2 ** 31 + 5
    kernel, = initialization.initial_kernels([(3, 4, 5)], seed)
    # the operation seeds come from the seed as given, the kernel gets
    # the seed modulo 2**31 - 1
    op_seed, = initialization.operation_seeds(seed, 1)
    assert np.array_equal(
        kernel, initialization.glorot_uniform((3, 4, 5), 6, op_seed))
    assert not np.array_equal(
        kernel, initialization.glorot_uniform((3, 4, 5), seed, op_seed))


def test_initial_kernels_depend_on_seed_and_position():
    shapes = [(3, 8, 8), (3, 8, 8), (8, 1)]
    a = initialization.initial_kernels(shapes, 1)
    b = initialization.initial_kernels(shapes, 1)
    c = initialization.initial_kernels(shapes, 2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])
    # two layers of the same shape get different kernels
    assert not np.array_equal(a[0], a[1])
    # a layer's kernel depends on how many layers were created before it
    assert np.array_equal(
        initialization.initial_kernels(shapes[:1], 1)[0], a[0])
    assert not np.array_equal(
        initialization.initial_kernels([(8, 1)], 1)[0], a[2])


def test_initialization_does_not_touch_the_global_random_state():
    random.seed(5)
    np.random.seed(5)
    expected = (random.random(), np.random.rand())
    random.seed(5)
    np.random.seed(5)
    initialization.initial_kernels(RELEASED_SHAPES, 1234)
    assert (random.random(), np.random.rand()) == expected


def test_weights_digest():
    a = [np.arange(6, dtype=np.float32).reshape(2, 3), np.zeros(2)]
    # sha256 over the arrays' bytes, one array after the other
    assert initialization.weights_digest(a) == hashlib.sha256(
        a[0].tobytes() + a[1].tobytes()).hexdigest()
    assert initialization.weights_digest(a) == \
        initialization.weights_digest([x.copy() for x in a])
    b = [a[0].copy(), a[1].copy()]
    b[0][0, 0] = 1e-6
    assert initialization.weights_digest(a) != \
        initialization.weights_digest(b)
    # a non contiguous view hashes like its copy
    view = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert initialization.weights_digest([view]) == \
        initialization.weights_digest([view.copy()])


class LayerStub(object):
    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        raise AssertionError("nothing may be set")


class ModelStub(object):
    def __init__(self, layers):
        self.layers = layers
        self.weight_layer_names = tuple(layers)

    def get_layer(self, name):
        return self.layers[name]


def test_unknown_weight_layout_is_rejected():
    kernel = np.zeros((2, 1), dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    for weights in ([kernel], [kernel, bias, bias]):
        model = ModelStub({
            "first": LayerStub([kernel, bias]), "dense": LayerStub(weights)})
        with pytest.raises(ValueError, match="dense has {} weight".format(
                len(weights))):
            initialization.set_initial_weights(model, 1)


def test_model_without_a_record_of_its_layers_is_rejected():
    # a model that arch.BPNet did not build, a loaded one for example
    with pytest.raises(ValueError, match="no record of its layers"):
        initialization.set_initial_weights(ModelStub({}), 1)


def test_seed_option_values():
    assert argparsers.seed_value("0") == 0
    assert argparsers.seed_value("7") == 7
    assert argparsers.seed_value("4294967295") == 2 ** 32 - 1
    for text in ("-1", "4294967296", "1.5", "seven", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            argparsers.seed_value(text)


def test_seed_option_of_the_training_parser(capsys):
    required = ["--model-arch-params-json", "p.json", "--reference-genome",
                "g.fa", "--chrom-sizes", "g.sizes", "--chroms", "chr1",
                "--input-data", "i.json"]
    parser = argparsers.training_argsparser()
    assert parser.parse_args(required).seed is None
    assert parser.parse_args(required + ["--seed", "7"]).seed == 7
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--seed", "-1"])
    assert "-1 is not between 0 and 4294967295" in capsys.readouterr().err


# --------------------------------------------------------------------
# with TensorFlow
# --------------------------------------------------------------------

tf = pytest.importorskip("tensorflow")

from bpnet.model import arch

from tests.model_helpers import (RELEASED_PARAMS, make_tasks,
                                 randomize_weights, small_params)

TF_VERSION = tuple(int(x) for x in tf.__version__.split(".")[:2])

CONFIGURATIONS = {
    "released": (RELEASED_PARAMS, {}, {}),
    "no control": (small_params(), dict(n_bias=0), {}),
    "unstranded": (small_params(), dict(n_signal=1, n_bias=1), {}),
    "two motif convolutions": (small_params(
        input_len=196, motif_module_params={
            "filters": [5, 6], "kernel_sizes": [11, 21]}), {}, {}),
    "hidden counts head": (small_params(counts_head_params={
        "units": [8, 4, 1], "dropouts": [0.5, 0.0, 0.0],
        "activations": ["relu", "leakyrelu", "linear"]}), {}, {}),
    "original loss": (small_params(), {}, dict(orig_multi_loss=True)),
    "two tasks": (small_params(
        profile_bias_module_params={"kernel_sizes": [1, 1]}),
        dict(n_tasks=2), {}),
}


def build(name):
    params, task_options, model_options = CONFIGURATIONS[name]
    return arch.BPNet(make_tasks(**task_options), copy.deepcopy(params),
                      name_prefix="main", **model_options)


@pytest.mark.parametrize("seed,seed2,n", [
    (1234, 5678, 4096), (0, 1, 7), (2147483646, 3, 33), (17, 2147483646, 1),
    (99, 0, 2050)])
def test_random_uniform_equals_the_tensorflow_kernel(seed, seed2, n):
    expected = tf.raw_ops.RandomUniform(
        shape=[n], dtype=tf.float32, seed=seed, seed2=seed2).numpy()
    assert np.array_equal(initialization.random_uniform(n, seed, seed2),
                          expected)


@pytest.mark.parametrize("seed,op_seed", [
    (1234, 5), (0, 0), (2 ** 31 - 1, 5), (2 ** 31 - 1, 2 ** 31 - 1),
    (2 ** 31 + 5, 2 ** 31 + 8), (7, 0), (0, 3)])
def test_kernel_seeds_are_the_ones_tensorflow_derives(seed, op_seed):
    from tensorflow.python.framework import random_seed

    try:
        tf.random.set_seed(seed)
        assert tuple(random_seed.get_seed(op_seed)) == \
            initialization.kernel_seeds(seed, op_seed)
        # and through the public interface
        drawn = tf.random.uniform([9], seed=op_seed).numpy()
    finally:
        tf.random.set_seed(initialization.DEFAULT_SEED)
    assert np.array_equal(drawn, initialization.random_uniform(
        9, *initialization.kernel_seeds(seed, op_seed)))


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_model_records_its_layers_with_weights(name):
    model = build(name)
    with_weights = {layer.name for layer in model.layers
                    if layer.get_weights()}
    assert set(model.weight_layer_names) == with_weights
    assert len(model.weight_layer_names) == len(with_weights)
    # a plain tuple: keras wraps the lists and tuples it tracks (and
    # would save with the model) in other types
    assert type(model.weight_layer_names) is tuple


def test_released_model_layer_order():
    model = build("released")
    assert list(model.weight_layer_names) == \
        ["main_conv_0"] + ["main_dil_conv_{}".format(i) for i in range(1, 9)] \
        + ["main_profile_head", "main_counts_head", "profile_predictions",
           "logcounts_predictions"]


def test_set_initial_weights_on_the_released_model():
    model = build("released")
    # whatever the weights were before, biases included
    randomize_weights(model, seed=3)
    assert all(model.get_layer(name).get_weights()[1].any()
               for name in model.weight_layer_names)
    digest = initialization.set_initial_weights(model, 1234)
    assert digest == RELEASED_SEED_1234_DIGEST
    arrays = []
    for name in model.weight_layer_names:
        kernel, bias = model.get_layer(name).get_weights()
        assert not bias.any()
        arrays.extend([kernel, bias])
    # the digest describes the weights the model now has
    assert hashlib.sha256(b"".join(
        a.tobytes() for a in arrays)).hexdigest() == digest
    kernel = model.get_layer("logcounts_predictions").get_weights()[0]
    assert kernel.ravel().tolist() == \
        [-0.08632349967956543, 0.41392314434051514]
    # the same seed again gives the same weights, another seed others
    assert initialization.set_initial_weights(model, 1234) == digest
    assert initialization.set_initial_weights(model, 1) != digest


def test_models_built_one_after_the_other_get_the_same_weights():
    first = build("no control")
    build("hidden counts head")
    second = build("no control")
    assert initialization.set_initial_weights(first, 3) == \
        initialization.set_initial_weights(second, 3)


@pytest.mark.skipif(TF_VERSION >= (2, 5), reason="only TensorFlow 2.4 draws "
                    "the initial weights from the global seed")
@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
@pytest.mark.parametrize("seed", [1234, 7, 2 ** 31 + 5])
def test_equals_what_tensorflow_24_draws_itself(name, seed):
    """
        Under TensorFlow 2.4 the weights Keras draws after
        tf.random.set_seed(seed) are the ones this module computes, for
        every architecture: assigning them there changes nothing
    """

    try:
        tf.random.set_seed(seed)
        model = build(name)
    finally:
        # the seed the package sets when it is imported
        tf.random.set_seed(initialization.DEFAULT_SEED)
    native = {n: model.get_layer(n).get_weights()
              for n in model.weight_layer_names}
    initialization.set_initial_weights(model, seed)
    for n in model.weight_layer_names:
        kernel, bias = model.get_layer(n).get_weights()
        assert np.array_equal(kernel, native[n][0]), n
        assert np.array_equal(bias, native[n][1]), n


def test_global_seed_is_what_tf_random_set_seed_was_given():
    from bpnet.train import training

    try:
        tf.random.set_seed(4321)
        assert training.global_seed() == 4321
        # no global seed: the seed the package has always set
        tf.random.set_seed(None)
        assert training.global_seed() == 1234
    finally:
        tf.random.set_seed(initialization.DEFAULT_SEED)
    assert training.global_seed() == 1234
