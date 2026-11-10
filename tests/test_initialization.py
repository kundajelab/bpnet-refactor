"""
    Tests for the seeded initial weights (bpnet.model.initialization)
    that need no TensorFlow: the numpy generator against values
    recorded from TensorFlow 2.4.1, the weights a training run started
    from under that version, the operation seeds, the Glorot limits.
    The comparisons with TensorFlow itself are in
    test_initialization_tf.py.
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

# tf.raw_ops.RandomUniform(shape=[3], seed=2**32 + 3, seed2=5) under
# TensorFlow 2.4.1
SEED_BEYOND_32_BITS = [0.4081333875656128, 0.30167901515960693,
                       0.42189598083496094]

# kernel shapes of the released architecture in creation order
RELEASED_SHAPES = [(21, 4, 64)] + [(3, 64, 64)] * 8 + \
    [(75, 64, 2), (64, 1), (1, 4, 2), (2, 1)]
RELEASED_BIAS_SIZES = [64] * 9 + [2, 1, 2, 1]


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
    # the seed is used as given, also when it is negative or large
    for seed in (-7, 2 ** 32 + 9, 2 ** 40):
        stream = random.Random(seed)
        assert initialization.operation_seeds(seed, 3) == \
            [stream.randint(0, 2 ** 31 - 1) for _ in range(3)]
    assert initialization.operation_seeds(2 ** 32 + 9, 3) != \
        initialization.operation_seeds(9, 3)


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


def test_random_uniform_from_a_position_in_the_stream():
    # four values per counter value
    whole = initialization.random_uniform(40, 1234, 5678)
    for skip in (1, 2, 7):
        assert np.array_equal(
            initialization.random_uniform(9, 1234, 5678, skip=skip),
            whole[4 * skip:4 * skip + 9])
    assert initialization.random_uniform(3, 1234, 5678, skip=2).tolist() == \
        [0.4489556550979614, 0.04433882236480713, 0.32650279998779297]
    # far positions: values recorded from tf.raw_ops.RandomUniform
    # under TensorFlow 2.4.1 (the last of 300,001 values, and values
    # from counter value 70,000 on)
    assert initialization.random_uniform(300001, 5, 6)[-3:].tolist() == \
        [0.10677707195281982, 0.6713669300079346, 0.5270742177963257]
    assert initialization.random_uniform(3, 5, 6, skip=70000).tolist() == \
        [0.5379114151000977, 0.638777494430542, 0.6014165878295898]
    # a seed beyond 32 bits fills the second word of the key
    assert initialization.random_uniform(3, 2 ** 32 + 3, 5).tolist() == \
        SEED_BEYOND_32_BITS
    assert not np.array_equal(initialization.random_uniform(3, 2 ** 32 + 3, 5),
                              initialization.random_uniform(3, 3, 5))
    # a position beyond 32 bits moves into the next counter word
    far = initialization.random_uniform(8, 3, 4, skip=2 ** 32 - 1)
    assert np.array_equal(
        far[4:], initialization.random_uniform(4, 3, 4, skip=2 ** 32))
    assert not np.array_equal(
        far[4:], initialization.random_uniform(4, 3, 4, skip=0))


def test_random_uniform_in_blocks(monkeypatch):
    # long requests are generated in blocks; the result does not
    # depend on the block size
    expected = initialization.random_uniform(1003, 11, 12, skip=5)
    for block in (1, 7, 250, 251):
        monkeypatch.setattr(initialization, "_BLOCK", block)
        assert np.array_equal(
            initialization.random_uniform(1003, 11, 12, skip=5), expected)
    assert len(initialization.random_uniform(0, 1, 2)) == 0


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


def test_repeated_operation_seed_continues_the_kernels_stream(monkeypatch):
    """
        TensorFlow keeps one random kernel per pair of seeds. When the
        stream hands out an operation seed a second time, the layer
        gets what follows the 256 counter values per weight that were
        reserved for the first layer.
    """

    monkeypatch.setattr(initialization, "operation_seeds",
                        lambda seed, n: [5, 9, 5, 5][:n])
    shapes = [(3, 2, 4), (2, 2), (3, 2, 4), (5, 1)]
    kernels = initialization.initial_kernels(shapes, 77)
    assert np.array_equal(
        kernels[0], initialization.glorot_uniform(shapes[0], 77, 5))
    assert np.array_equal(
        kernels[1], initialization.glorot_uniform(shapes[1], 77, 9))
    assert np.array_equal(kernels[2], initialization.glorot_uniform(
        shapes[2], 77, 5, skip=24 * 256))
    assert np.array_equal(kernels[3], initialization.glorot_uniform(
        shapes[3], 77, 5, skip=48 * 256))
    assert not np.array_equal(kernels[2], kernels[0])


def test_repeated_operation_seed_as_recorded_from_tensorflow_241():
    # seed 33780 hands out the same operation seed to the layers 164
    # and 191 of a model with 192 one-filter motif convolutions; the
    # first weights of the two layers as TensorFlow 2.4.1 drew them
    seeds = initialization.operation_seeds(33780, 192)
    assert seeds[164] == seeds[191] == 269732680
    assert len(set(seeds)) == 191
    shapes = [(1, 4, 1)] + [(1, 1, 1)] * 191
    kernels = initialization.initial_kernels(shapes, 33780)
    assert float(kernels[164].ravel()[0]) == -1.5604612827301025
    assert float(kernels[191].ravel()[0]) == -0.9455151557922363


def test_repeated_seed_pair_after_reduction(monkeypatch):
    # the random kernel is identified by the seeds as TensorFlow
    # reduces them: operation seeds 0 and 2**31 - 1 are the same
    # kernel, and so are 3 and 2**31 + 2
    monkeypatch.setattr(initialization, "operation_seeds",
                        lambda seed, n: [0, 2 ** 31 - 1, 3, 2 ** 31 + 2][:n])
    shapes = [(4, 3), (4, 3), (2, 5), (2, 5)]
    kernels = initialization.initial_kernels(shapes, 2 ** 31 + 6)
    assert initialization.kernel_seeds(2 ** 31 + 6, 0) == (7, 0)
    assert np.array_equal(
        kernels[0], initialization.glorot_uniform((4, 3), 7, 0))
    assert np.array_equal(kernels[1], initialization.glorot_uniform(
        (4, 3), 7, 0, skip=12 * 256))
    assert np.array_equal(
        kernels[2], initialization.glorot_uniform((2, 5), 7, 3))
    assert np.array_equal(kernels[3], initialization.glorot_uniform(
        (2, 5), 7, 3, skip=10 * 256))


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
    def __init__(self, name, weights):
        self.name = name
        self.weights = weights
        self.assigned = 0

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = [np.array(w) for w in weights]
        self.assigned += 1


class ModelStub(object):
    """
        What the initialization needs of a model: its layers, and the
        names of the layers with weights in the order the builder
        created them
    """

    def __init__(self, layers, recorded=None):
        self.layers = list(layers)
        self.weight_layer_names = tuple(
            layer.name for layer in layers if layer.get_weights()) \
            if recorded is None else tuple(recorded)

    def get_layer(self, name):
        layer, = [layer for layer in self.layers if layer.name == name]
        return layer


def stub_model(dtype=np.float32, seed=0):
    rng = np.random.RandomState(seed)
    shapes = [("conv", (3, 4, 5)), ("dense", (5, 1)), ("mix", (2, 1))]
    layers = [LayerStub(name, [
        rng.normal(size=shape).astype(dtype),
        rng.normal(size=shape[-1]).astype(dtype)]) for name, shape in shapes]
    # a layer without weights between them, as in a real model
    layers.insert(1, LayerStub("relu", []))
    return ModelStub(layers)


def stored_digest(model):
    return hashlib.sha256(b"".join(
        array.tobytes() for name in model.weight_layer_names
        for array in model.get_layer(name).get_weights())).hexdigest()


def stored_digest_of_all(model):
    return hashlib.sha256(b"".join(
        array.tobytes() for layer in model.layers
        for array in layer.get_weights())).hexdigest()


def test_set_initial_weights():
    model = stub_model()
    digest = initialization.set_initial_weights(model, 21)
    kernels = initialization.initial_kernels([(3, 4, 5), (5, 1), (2, 1)], 21)
    for name, kernel in zip(model.weight_layer_names, kernels):
        stored, bias = model.get_layer(name).get_weights()
        assert np.array_equal(stored, kernel) and stored.dtype == np.float32
        assert not bias.any() and bias.dtype == np.float32
    # the digest is of the weights the model now has
    assert digest == stored_digest(model)
    assert digest == initialization.current_weights_digest(model)
    assert initialization.set_initial_weights(model, 22) != digest


def test_unknown_weight_layout_is_rejected():
    kernel = np.zeros((2, 1), dtype=np.float32)
    bias = np.zeros(1, dtype=np.float32)
    for weights in ([kernel], [kernel, bias, bias]):
        model = ModelStub([LayerStub("first", [kernel, bias]),
                           LayerStub("dense", weights)])
        with pytest.raises(ValueError, match="dense has {} weight".format(
                len(weights))):
            initialization.set_initial_weights(model, 1)
        assert model.get_layer("first").assigned == 0


def test_model_without_a_record_of_its_layers_is_rejected():
    # a model that arch.BPNet did not build, a loaded one for example
    with pytest.raises(ValueError, match="no record of its layers"):
        initialization.set_initial_weights(ModelStub([]), 1)
    for drawn_from_seed in (False, True):
        with pytest.raises(ValueError, match="no record of its layers"):
            initialization.initialize(ModelStub([]), 1, drawn_from_seed)


def test_model_with_weights_the_builder_did_not_create():
    """
        A layer with weights that is not in the builder's record (one
        that a callable given to the builder added to the model): its
        weights cannot be derived from the seed. Such a model is left
        alone where TensorFlow follows the seed and refused where
        weights would have to be assigned.
    """

    model = stub_model()
    model.weight_layer_names = ("conv", "mix")
    before = stored_digest_of_all(model)
    with pytest.raises(ValueError, match=r"\['dense'\] have weights but"):
        initialization.set_initial_weights(model, 1)
    with pytest.raises(ValueError, match=r"\['dense'\] have weights but"):
        initialization.initialize(model, 1, False)
    assert stored_digest_of_all(model) == before
    assert all(layer.assigned == 0 for layer in model.layers)

    report = initialization.initialize(model, 1, True)
    assert report["drawn_by"] == "tensorflow"
    assert stored_digest_of_all(model) == before


def test_only_float32_models_are_given_seeded_weights():
    model = stub_model(dtype=np.float64)
    before = stored_digest(model)
    with pytest.raises(ValueError, match="only float32 models"):
        initialization.set_initial_weights(model, 1)
    assert stored_digest(model) == before
    # initialize leaves such a model as it was built
    report = initialization.initialize(model, 1, False)
    assert report == {"seed": 1, "sha256": before, "drawn_by": "tensorflow"}
    assert stored_digest(model) == before

    # also when only a later layer, or only a bias, is not float32
    for name, index in (("mix", 0), ("dense", 1)):
        model = stub_model()
        weights = model.get_layer(name).weights
        weights[index] = weights[index].astype(np.float64)
        before = stored_digest(model)
        report = initialization.initialize(model, 1, False)
        assert report["drawn_by"] == "tensorflow"
        assert stored_digest(model) == before == report["sha256"]
        with pytest.raises(ValueError, match="only float32 models"):
            initialization.set_initial_weights(model, 1)
        assert stored_digest(model) == before


def seeded_digest(seed):
    """
        Digest of the stub model with the weights of the seed
    """

    model = stub_model()
    names = model.weight_layer_names
    kernels = initialization.initial_kernels(
        [model.get_layer(n).get_weights()[0].shape for n in names], seed)
    for name, kernel in zip(names, kernels):
        model.get_layer(name).weights = [
            kernel, np.zeros(kernel.shape[-1], dtype=np.float32)]
    return stored_digest(model)


def test_initialize_leaves_weights_where_tensorflow_follows_the_seed():
    # as under TensorFlow 2.4: whatever TensorFlow drew stays, also
    # when it is not what a fresh seed gives (random operations or
    # other models before the build move TensorFlow's seed stream)
    model = stub_model(seed=3)
    before = stored_digest(model)
    assert before != seeded_digest(21)
    report = initialization.initialize(model, 21, True)
    assert report == {"seed": 21, "sha256": before, "drawn_by": "tensorflow"}
    assert stored_digest(model) == before
    assert all(layer.assigned == 0 for layer in model.layers)


def test_initialize_assigns_where_tensorflow_ignores_the_seed():
    # as under newer TensorFlow versions
    model = stub_model(seed=3)
    report = initialization.initialize(model, 21, False)
    assert report == {"seed": 21, "drawn_by": "bpnet",
                      "sha256": seeded_digest(21)}
    assert stored_digest(model) == report["sha256"]
    assert seeded_digest(21) != seeded_digest(22)


@pytest.mark.parametrize("drawn_from_seed", [False, True])
def test_initialize_without_a_global_seed_leaves_the_weights(
        drawn_from_seed):
    model = stub_model(seed=6)
    before = stored_digest(model)
    report = initialization.initialize(model, None, drawn_from_seed)
    assert report == {"seed": None, "sha256": before,
                      "drawn_by": "tensorflow"}
    assert stored_digest(model) == before


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


