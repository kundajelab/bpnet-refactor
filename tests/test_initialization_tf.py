"""
    Tests for the seeded initial weights (bpnet.model.initialization)
    against TensorFlow: the numpy generator equals TensorFlow's random
    kernel, the seed handling equals TensorFlow's, and under
    TensorFlow 2.4 the kernels equal the ones Keras draws itself
"""

import copy
import hashlib
import random
import threading

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bpnet.model import arch, initialization
from bpnet.train import training

from tests.model_helpers import (RELEASED_PARAMS, make_tasks,
                                 randomize_weights, small_params)
from tests.test_initialization import (RELEASED_SEED_1234_DIGEST,
                                       stored_digest)

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


#: Whether this TensorFlow draws the kernels of a layer from its global
#: seed. Tested for 2.4.1 (yes), 2.11.0 and 2.15.1 (no). In between,
#: from reading the keras sources: 2.5 to 2.9 still do, 2.10 is the
#: first version that draws them from seeds keras makes up
SEEDED_BY_TENSORFLOW = TF_VERSION < (2, 10)


def build(name):
    params, task_options, model_options = CONFIGURATIONS[name]
    return arch.BPNet(make_tasks(**task_options), copy.deepcopy(params),
                      name_prefix="main", **model_options)


def kernels_of(model):
    return [model.get_layer(name).get_weights()[0]
            for name in model.weight_layer_names]


@pytest.fixture(autouse=True)
def fresh_seed():
    """
        Every test starts and ends with the seed the package sets when
        it is imported. Setting the seed also empties TensorFlow's
        cache of random kernels, whose streams would otherwise continue
        from one test to the next.
    """

    tf.random.set_seed(initialization.DEFAULT_SEED)
    yield
    tf.random.set_seed(initialization.DEFAULT_SEED)


@pytest.mark.parametrize("seed,seed2,n", [
    (1234, 5678, 4096), (0, 1, 7), (2147483646, 3, 33), (17, 2147483646, 1),
    (99, 0, 2050), (5, 6, 300001), (2 ** 32 + 3, 5, 9), (7, 2 ** 33 + 1, 9)])
def test_random_uniform_equals_the_tensorflow_kernel(seed, seed2, n):
    expected = tf.raw_ops.RandomUniform(
        shape=[n], dtype=tf.float32, seed=seed, seed2=seed2).numpy()
    assert np.array_equal(initialization.random_uniform(n, seed, seed2),
                          expected)


def test_a_kernel_that_is_used_again_continues_its_stream():
    # TensorFlow reserves 256 counter values per value it returns
    sizes = [7, 1, 4096, 30]
    skip = 0
    for n in sizes:
        drawn = tf.raw_ops.RandomUniform(
            shape=[n], dtype=tf.float32, seed=31, seed2=32).numpy()
        assert np.array_equal(
            drawn, initialization.random_uniform(n, 31, 32, skip=skip)), n
        skip += n * 256
    # setting the global seed starts the kernels afresh
    tf.random.set_seed(1)
    drawn = tf.raw_ops.RandomUniform(
        shape=[5], dtype=tf.float32, seed=31, seed2=32).numpy()
    assert np.array_equal(drawn, initialization.random_uniform(5, 31, 32))


@pytest.mark.parametrize("seed,op_seed", [
    (1234, 5), (0, 0), (2 ** 31 - 1, 5), (2 ** 31 - 1, 2 ** 31 - 1),
    (2 ** 31 + 5, 2 ** 31 + 8), (7, 0), (0, 3)])
def test_kernel_seeds_are_the_ones_tensorflow_derives(seed, op_seed):
    from tensorflow.python.framework import random_seed

    tf.random.set_seed(seed)
    assert tuple(random_seed.get_seed(op_seed)) == \
        initialization.kernel_seeds(seed, op_seed)
    # and through the public interface
    drawn = tf.random.uniform([9], seed=op_seed).numpy()
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


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
@pytest.mark.parametrize("seed", [1234, 5])
def test_set_initial_weights(name, seed):
    model = build(name)
    # whatever the weights were before, biases included
    randomize_weights(model, seed=3)
    assert all(model.get_layer(n).get_weights()[1].any()
               for n in model.weight_layer_names)
    digest = initialization.set_initial_weights(model, seed)

    expected = initialization.initial_kernels(
        [kernel.shape for kernel in kernels_of(model)], seed)
    for n, kernel in zip(model.weight_layer_names, expected):
        stored, bias = model.get_layer(n).get_weights()
        assert np.array_equal(stored, kernel), n
        assert not bias.any(), n
    # the digest describes the weights the model now has
    assert digest == stored_digest(model)
    # the same seed again gives the same weights, another seed others
    assert initialization.set_initial_weights(model, seed) == digest
    assert initialization.set_initial_weights(model, seed + 1) != digest


def test_released_model_with_seed_1234_has_the_tf241_weights():
    model = build("released")
    randomize_weights(model, seed=3)
    assert initialization.set_initial_weights(model, 1234) == \
        RELEASED_SEED_1234_DIGEST
    kernel = model.get_layer("logcounts_predictions").get_weights()[0]
    assert kernel.ravel().tolist() == \
        [-0.08632349967956543, 0.41392314434051514]


def test_models_built_one_after_the_other_get_the_same_weights():
    first = build("no control")
    build("hidden counts head")
    second = build("no control")
    assert initialization.set_initial_weights(first, 3) == \
        initialization.set_initial_weights(second, 3)


needs_seeded_tensorflow = pytest.mark.skipif(
    not SEEDED_BY_TENSORFLOW,
    reason="this TensorFlow version does not draw the initial weights from "
    "the global seed")


@needs_seeded_tensorflow
@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
@pytest.mark.parametrize("seed", [1234, 7, 0, -7, 2 ** 31 + 5])
def test_equals_what_tensorflow_24_draws_itself(name, seed):
    """
        Under TensorFlow 2.4 the weights Keras draws after
        tf.random.set_seed(seed) are the ones this module computes, for
        every architecture
    """

    tf.random.set_seed(seed)
    model = build(name)
    native = {n: model.get_layer(n).get_weights()
              for n in model.weight_layer_names}
    initialization.set_initial_weights(model, seed)
    for n in model.weight_layer_names:
        kernel, bias = model.get_layer(n).get_weights()
        assert np.array_equal(kernel, native[n][0]), n
        assert np.array_equal(bias, native[n][1]), n


@needs_seeded_tensorflow
def test_equals_tensorflow_24_when_an_operation_seed_repeats():
    """
        Seed 33780 hands the same operation seed to two layers of this
        model; TensorFlow continues the stream of the random kernel the
        two layers share, and so does this module
    """

    params = small_params(
        input_len=8, output_profile_len=8,
        motif_module_params={"filters": [1] * 192, "kernel_sizes": [1] * 192,
                             "padding": "valid"},
        syntax_module_params={"num_dilation_layers": 0},
        profile_head_params={"filters": 1, "kernel_size": 1,
                             "padding": "valid"})
    tf.random.set_seed(33780)
    model = arch.BPNet(make_tasks(n_bias=0), params, name_prefix="main")
    native = kernels_of(model)
    expected = initialization.initial_kernels(
        [kernel.shape for kernel in native], 33780)
    assert len(native) == 194
    for n, a, b in zip(model.weight_layer_names, native, expected):
        assert np.array_equal(a, b), n
    assert float(native[191].ravel()[0]) == -0.9455151557922363


# --------------------------------------------------------------------
# what the trainer does after building its model
# --------------------------------------------------------------------

def seed_stream_state():
    from tensorflow.python.eager import context

    stream = getattr(context.context(), "_rng", None)
    return None if stream is None else stream.getstate()


def test_global_seed():
    tf.random.set_seed(4321)
    assert training.global_seed() == 4321
    assert type(training.global_seed()) is int
    tf.random.set_seed(0)
    assert training.global_seed() == 0
    tf.random.set_seed(None)
    assert training.global_seed() is None


def test_package_import_sets_seed_1234():
    # the fixture of this module sets the same seed again before every
    # test; the constant is the one the package modules set
    assert initialization.DEFAULT_SEED == 1234
    assert training.global_seed() == 1234


@pytest.mark.parametrize("seed", [1234, 0, 77])
def test_whether_tensorflow_draws_initial_weights_from_its_seed(seed):
    tf.random.set_seed(seed)
    assert training.initializers_follow_global_seed() is SEEDED_BY_TENSORFLOW
    # asking again gives the same answer
    assert training.initializers_follow_global_seed() is SEEDED_BY_TENSORFLOW


def test_finding_out_leaves_no_trace():
    """
        Everything random after initializers_follow_global_seed() is
        as if it had not been called: TensorFlow's stream of operation
        seeds, the random kernels of earlier operations, Python's
        random module and numpy's
    """

    def after(probe):
        tf.random.set_seed(99)
        random.seed(5)
        np.random.seed(5)
        first = tf.random.uniform([3], seed=9).numpy()
        if probe:
            training.initializers_follow_global_seed()
        return (first, seed_stream_state(), random.random(),
                np.random.rand(),
                # a kernel that was used before continues its stream
                tf.random.uniform([3], seed=9).numpy(),
                # the next operation seed of the stream
                tf.random.uniform([3]).numpy(),
                stored_digest(build("no control")))

    without, with_probe = after(False), after(True)
    assert np.array_equal(without[0], with_probe[0])
    assert without[1] == with_probe[1]
    assert without[2:4] == with_probe[2:4]
    assert np.array_equal(without[4], with_probe[4])
    assert np.array_equal(without[5], with_probe[5])
    if SEEDED_BY_TENSORFLOW:
        assert without[6] == with_probe[6]


def test_initial_weights_in_a_graph_are_left_to_tensorflow():
    # TensorFlow 1 style graphs have a seed and operation seeds of
    # their own; nothing is assigned there
    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(7)
        assert training.initializers_follow_global_seed() is True
        model = build("no control")
        before = stored_digest(model)
        report = initialization.initialize(
            model, training.global_seed(),
            training.initializers_follow_global_seed())
        assert report["drawn_by"] == "tensorflow"
        assert report["sha256"] == before == stored_digest(model)


def initialized(name, seed, before_build=lambda: None):
    """
        What the trainer does around building its model

        Returns:
            tuple: (model, digest of the weights as TensorFlow drew
                them, report of initialize)
    """

    tf.random.set_seed(seed)
    before_build()
    model = build(name)
    native = stored_digest(model)
    report = initialization.initialize(
        model, training.global_seed(),
        training.initializers_follow_global_seed())
    return model, native, report


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_initialize_after_a_fresh_seed(name):
    """
        Every TensorFlow version ends up with the weights TensorFlow
        2.4 draws for the seed: where TensorFlow follows the seed
        because it drew them, elsewhere because they were assigned
    """

    model, native, report = initialized(name, 1234)
    expected = initialization.initial_kernels(
        [kernel.shape for kernel in kernels_of(model)], 1234)
    for n, kernel in zip(model.weight_layer_names, expected):
        stored, bias = model.get_layer(n).get_weights()
        assert np.array_equal(stored, kernel), n
        assert not bias.any(), n
    assert report["seed"] == 1234
    assert report["sha256"] == stored_digest(model)
    if name == "released":
        assert report["sha256"] == RELEASED_SEED_1234_DIGEST
    if SEEDED_BY_TENSORFLOW:
        assert report["drawn_by"] == "tensorflow"
        assert report["sha256"] == native
    else:
        assert report["drawn_by"] == "bpnet"
        assert report["sha256"] != native


def a_random_operation():
    tf.random.uniform([1])


def another_model():
    build("no control")


def the_first_kernel_of_the_model_used_before():
    # 1892932127 is the first operation seed after set_seed(1234): the
    # model's first layer finds its random kernel already used
    tf.random.uniform([7], seed=1892932127)


@pytest.mark.parametrize("before_build", [
    a_random_operation, another_model,
    the_first_kernel_of_the_model_used_before])
def test_initialize_when_the_process_used_random_numbers_before(before_build):
    """
        Random operations or other models before the build move
        TensorFlow's stream of operation seeds, or leave a random
        kernel half used. Where TensorFlow follows the seed it then
        draws other weights than after a fresh seed, and they are left
        exactly as drawn. Elsewhere the model gets the weights of the
        fresh seed.
    """

    model, native, report = initialized("released", 1234, before_build)
    assert report["seed"] == 1234
    assert report["sha256"] == stored_digest(model)
    if SEEDED_BY_TENSORFLOW:
        assert report["drawn_by"] == "tensorflow"
        assert report["sha256"] == native != RELEASED_SEED_1234_DIGEST
    else:
        assert report == {"seed": 1234, "drawn_by": "bpnet",
                          "sha256": RELEASED_SEED_1234_DIGEST}


@needs_seeded_tensorflow
def test_weights_drawn_after_191_random_operations_are_left_alone():
    # after 191 operations the stream of seed 33780 repeats an
    # operation seed, so the model's first kernel continues the stream
    # of an earlier operation
    def operations():
        for _ in range(191):
            tf.random.uniform([1])

    model, native, report = initialized("no control", 33780, operations)
    assert report == {"seed": 33780, "sha256": native,
                      "drawn_by": "tensorflow"}
    assert stored_digest(model) == native


def test_initialize_without_a_global_seed_keeps_the_random_weights():
    model, native, report = initialized("released", None)
    assert report == {"seed": None, "sha256": native,
                      "drawn_by": "tensorflow"}
    assert stored_digest(model) == native
    # random: another model has other weights
    other, other_native, _ = initialized("released", None)
    assert other_native != native


def test_initialize_leaves_a_float64_model_alone():
    tf.keras.backend.set_floatx("float64")
    try:
        model, native, report = initialized("no control", 1234)
    finally:
        tf.keras.backend.set_floatx("float32")
    assert kernels_of(model)[0].dtype == np.float64
    assert report == {"seed": 1234, "sha256": native,
                      "drawn_by": "tensorflow"}
    assert stored_digest(model) == native
    with pytest.raises(ValueError, match="only float32 models"):
        initialization.set_initial_weights(model, 1234)
    assert stored_digest(model) == native


# --------------------------------------------------------------------
# the record of the layers
# --------------------------------------------------------------------

def test_a_build_that_fails_leaves_nothing_in_the_next_record():
    from bpnet.utils.exceptionhandler import NoTracebackException

    # fails after the convolutions and both heads were created
    with pytest.raises(NoTracebackException, match="kernel_sizes"):
        arch.BPNet(make_tasks(), small_params(
            profile_bias_module_params={"kernel_sizes": [1, 1]}),
            name_prefix="failed")
    model = build("no control")
    assert not any(name.startswith("failed")
                   for name in model.weight_layer_names)
    assert set(model.weight_layer_names) == {
        layer.name for layer in model.layers if layer.get_weights()}


def test_a_model_built_while_another_is_being_built():
    """
        A callable activation that builds a model of its own: both
        models get their own record
    """

    inner = []

    @tf.autograph.experimental.do_not_convert
    def activation(x):
        if not inner:
            inner.append(arch.BPNet(make_tasks(), small_params(),
                                    name_prefix="inner"))
        return tf.nn.relu(x)

    params = small_params(counts_head_params={
        "units": [8, 1], "dropouts": [0.0, 0.0],
        "activations": [activation, "linear"]})
    outer = arch.BPNet(make_tasks(), params, name_prefix="outer")
    assert len(inner) == 1
    for model, prefix in ((outer, "outer"), (inner[0], "inner")):
        assert set(model.weight_layer_names) == {
            layer.name for layer in model.layers if layer.get_weights()}
        assert model.weight_layer_names[0] == prefix + "_conv_0"
        assert initialization.set_initial_weights(model, 3) == \
            stored_digest(model)
    assert getattr(arch._build, "weight_layer_names", None) is None


def test_a_build_that_fails_inside_another_build():
    from bpnet.utils.exceptionhandler import NoTracebackException

    failures = []

    @tf.autograph.experimental.do_not_convert
    def activation(x):
        if not failures:
            try:
                arch.BPNet(make_tasks(), small_params(
                    profile_bias_module_params={"kernel_sizes": [1, 1]}),
                    name_prefix="failed")
            except NoTracebackException as error:
                failures.append(error)
        return tf.nn.relu(x)

    params = small_params(counts_head_params={
        "units": [8, 1], "dropouts": [0.0, 0.0],
        "activations": [activation, "linear"]})
    outer = arch.BPNet(make_tasks(), params, name_prefix="outer")
    assert len(failures) == 1
    assert not any(name.startswith("failed")
                   for name in outer.weight_layer_names)
    assert set(outer.weight_layer_names) == {
        layer.name for layer in outer.layers if layer.get_weights()}


def test_models_built_in_two_threads_keep_separate_records():
    """
        A model that is built in another thread while this thread is
        in the middle of building one does not record its layers here
    """

    arch._build.weight_layer_names = ["a_layer_of_an_unfinished_model"]
    built = {}

    def in_thread():
        built["model"] = build("no control")
        built["left_over"] = getattr(arch._build, "weight_layer_names", None)

    thread = threading.Thread(target=in_thread)
    thread.start()
    thread.join()
    try:
        assert arch._build.weight_layer_names == \
            ["a_layer_of_an_unfinished_model"]
    finally:
        arch._build.weight_layer_names = None
    assert built["left_over"] is None
    names = built["model"].weight_layer_names
    assert names == build("no control").weight_layer_names
    assert "a_layer_of_an_unfinished_model" not in names
