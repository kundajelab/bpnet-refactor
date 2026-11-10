"""
    Seeded initial weights that do not depend on the TensorFlow version

    Why this module exists. The package calls tf.random.set_seed(1234)
    when it is imported. Under TensorFlow 2.4.1 that fixed the initial
    weights: Keras drew every kernel with the stateful RandomUniform
    kernel, seeded with (the global seed, an operation seed taken from
    a Python random.Random(global seed) stream), so every model of a
    given architecture started from the same arrays. Newer Keras
    versions seed unseeded initializers from Python's `random` module,
    which is re-seeded from the operating system in the forked training
    process: the global seed no longer controls the initial weights and
    every run starts somewhere else.

    The starting point matters. The layer that combines the predicted
    log counts with the control log counts has two weights, drawn from
    [-1.41, 1.41]. Adam moves a weight by about the learning rate
    (0.001) per step, so an experiment with a few hundred peaks, which
    trains for a few hundred steps, ends close to where those weights
    started; a start with a negative weight on the control counts
    leaves the count predictions poor or anti-correlated with the
    observed counts. With the fixed start of TensorFlow 2.4.1 (weight
    +0.41 for seed 1234) this did not happen.

    This module reproduces the TensorFlow 2.4.1 arrays with numpy: the
    Philox4x32-10 counter based generator of the RandomUniform kernel,
    the conversion of its output to float32, the Glorot uniform limits
    and the operation seed stream. A model initialized here with seed
    1234 starts from exactly the weights the same model had in a
    training run under TensorFlow 2.4.1 (the released ENCODE models
    were trained that way), whatever the TensorFlow version, and so
    can a PyTorch model.

    What it does not do. Where TensorFlow itself still draws the
    initial weights from its global seed (2.4.1), `initialize` leaves
    them as drawn: nothing changes there, whatever random operations
    or model builds came before in the process. Weights are only
    assigned where TensorFlow no longer follows the seed, and then
    they are the weights of a fresh seed. Only float32 models are
    handled.

    Only numpy and the standard library are used.
"""

import hashlib
import random

import numpy as np

#: the seed the package has always set at import time
DEFAULT_SEED = 1234

_MASK32 = 0xFFFFFFFF
_PHILOX_W32_A = 0x9E3779B9
_PHILOX_W32_B = 0xBB67AE85
_PHILOX_M4X32_A = np.uint64(0xD2511F53)
_PHILOX_M4X32_B = np.uint64(0xCD9E8D57)
_LOW32 = np.uint64(_MASK32)
_SHIFT32 = np.uint64(32)

#: largest operation seed TensorFlow draws (2**31 - 1)
_MAX_OP_SEED = 2 ** 31 - 1


def _philox_4x32_10(counters, key):
    """
        Philox4x32 with 10 rounds

        Args:
            counters (numpy.ndarray): uint32 array of shape (n, 4)
            key (tuple): two integers below 2**32

        Returns:
            numpy.ndarray: uint32 array of shape (n, 4)
    """

    c0 = counters[:, 0].astype(np.uint64)
    c1 = counters[:, 1].astype(np.uint64)
    c2 = counters[:, 2].astype(np.uint64)
    c3 = counters[:, 3].astype(np.uint64)
    k0, k1 = int(key[0]) & _MASK32, int(key[1]) & _MASK32

    for round_number in range(10):
        if round_number > 0:
            k0 = (k0 + _PHILOX_W32_A) & _MASK32
            k1 = (k1 + _PHILOX_W32_B) & _MASK32
        product0 = _PHILOX_M4X32_A * c0
        product1 = _PHILOX_M4X32_B * c2
        low0, high0 = product0 & _LOW32, product0 >> _SHIFT32
        low1, high1 = product1 & _LOW32, product1 >> _SHIFT32
        c0 = high1 ^ c1 ^ np.uint64(k0)
        c1 = low1
        c2 = high0 ^ c3 ^ np.uint64(k1)
        c3 = low0

    return np.stack([c0, c1, c2, c3], axis=1).astype(np.uint32)


#: TensorFlow reserves this many counter values per output value when
#: a random kernel is used; a kernel that is used again continues after
#: what was reserved (PhiloxRandomOp::Compute, ReserveRandomOutputs)
_RESERVED_PER_VALUE = 256

#: number of counter values processed at a time (4 output values each)
_BLOCK = 1 << 18


def random_uniform(n, seed, seed2, skip=0):
    """
        Values of TensorFlow's stateful RandomUniform kernel for
        float32, in [0, 1)

        TensorFlow treats the pair (seed, seed2) = (0, 0) as "not
        seeded" and replaces it by seeds from the clock; see
        kernel_seeds for the pair it uses for given seeds. This
        function returns the generator's output for the pair as given.

        Args:
            n (int): number of values
            seed (int): the kernel's `seed` attribute
            seed2 (int): the kernel's `seed2` attribute
            skip (int): position in the kernel's stream to start from,
                in counter values (0 for a kernel that has not been
                used)

        Returns:
            numpy.ndarray: float32 array of length n
    """

    n, seed, seed2, skip = int(n), int(seed), int(seed2), int(skip)
    groups = (n + 3) // 4
    key = (seed & _MASK32, (seed >> 32) & _MASK32)
    out = np.empty(groups * 4, dtype=np.float32)
    for first in range(0, groups, _BLOCK):
        count = min(_BLOCK, groups - first)
        # the key is `seed`; `seed2` fills the upper half of the 128
        # bit counter, whose lower half counts groups of four values
        index = np.arange(skip + first, skip + first + count,
                          dtype=np.uint64)
        counters = np.zeros((count, 4), dtype=np.uint32)
        counters[:, 0] = (index & _LOW32).astype(np.uint32)
        counters[:, 1] = (index >> _SHIFT32).astype(np.uint32)
        counters[:, 2] = seed2 & _MASK32
        counters[:, 3] = (seed2 >> 32) & _MASK32
        bits = _philox_4x32_10(counters, key).reshape(-1)
        # 23 random mantissa bits with exponent 0 give a float in
        # [1, 2)
        as_float = ((np.uint32(127) << np.uint32(23))
                    | (bits & np.uint32(0x7FFFFF))).view(np.float32)
        out[4 * first:4 * (first + count)] = as_float - np.float32(1.0)
    return out[:n]


def glorot_uniform(shape, seed, op_seed, skip=0):
    """
        A kernel as Keras' default 'glorot_uniform' initializer drew it
        under TensorFlow 2.4.1

        Args:
            shape (tuple): kernel shape in Keras layout: (width, in,
                out) for Conv1D, (in, out) for Dense
            seed (int): the random kernel's `seed` attribute (see
                kernel_seeds)
            op_seed (int): the random kernel's `seed2` attribute
            skip (int): position in the random kernel's stream (see
                random_uniform)

        Returns:
            numpy.ndarray: float32 array of the given shape, uniform in
                [-limit, limit] with limit = sqrt(6 / (fan_in + fan_out))
    """

    shape = tuple(int(n) for n in shape)
    receptive_field = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    fan_in = shape[-2] * receptive_field
    fan_out = shape[-1] * receptive_field
    limit = np.sqrt(3.0 / max(1.0, (fan_in + fan_out) / 2.0))
    minval, maxval = np.float32(-limit), np.float32(limit)
    values = random_uniform(int(np.prod(shape)), seed, op_seed, skip)
    # the float32 arithmetic of tf.random.uniform
    return (values * (maxval - minval) + minval).reshape(shape)


def operation_seeds(seed, n):
    """
        The first n operation seeds TensorFlow hands out after
        tf.random.set_seed(seed)

        Args:
            seed (int): what tf.random.set_seed was given
            n (int): number of operation seeds

        Returns:
            list: n integers in [0, 2**31 - 1]
    """

    stream = random.Random(seed)
    return [stream.randint(0, _MAX_OP_SEED) for _ in range(n)]


def kernel_seeds(seed, op_seed):
    """
        The `seed` and `seed2` attributes TensorFlow gives a random
        kernel for a global seed and an operation seed: both reduced
        modulo 2**31 - 1, and the pair (0, 0), which the kernel would
        take as "not seeded", replaced

        Args:
            seed (int): global seed (tf.random.set_seed)
            op_seed (int): operation seed

        Returns:
            tuple: (seed, seed2)
    """

    seeds = (int(seed) % _MAX_OP_SEED, int(op_seed) % _MAX_OP_SEED)
    if seeds == (0, 0):
        return (0, _MAX_OP_SEED)
    return seeds


def initial_kernels(shapes, seed=DEFAULT_SEED):
    """
        Kernels for the layers of a model, in the order the layers are
        created, as TensorFlow 2.4.1 draws them for a model that is
        built right after tf.random.set_seed(seed)

        Args:
            shapes (list): kernel shapes in creation order
            seed (int): the seed (what tf.random.set_seed was given)

        Returns:
            list: one float32 array per shape
    """

    kernels = []
    # TensorFlow keeps one random kernel per pair of seeds; when the
    # stream hands out the same operation seed twice, the second layer
    # continues where the kernel stopped reserving for the first
    reserved = {}
    for shape, op_seed in zip(shapes, operation_seeds(seed, len(shapes))):
        pair = kernel_seeds(seed, op_seed)
        skip = reserved.get(pair, 0)
        kernels.append(glorot_uniform(shape, pair[0], pair[1], skip))
        reserved[pair] = skip + int(np.prod(shape)) * _RESERVED_PER_VALUE
    return kernels


def weights_digest(arrays):
    """
        sha256 over the bytes of a list of arrays, to compare sets of
        weights across runs and logs
    """

    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _weight_layers(model):
    """
        The layers with weights of a model in creation order, with
        their current weights

        Returns:
            list: (name, layer, [kernel, bias]) per layer
    """

    names = list(model.weight_layer_names)
    if not names:
        raise ValueError(
            "The model has no record of its layers with weights; only "
            "models built by bpnet.model.arch.BPNet can be initialized")
    found = []
    for name in names:
        layer = model.get_layer(name)
        weights = layer.get_weights()
        if len(weights) != 2:
            raise ValueError(
                "Layer {} has {} weight arrays; a kernel and a bias are "
                "expected".format(name, len(weights)))
        found.append((name, layer, weights))
    return found


def current_weights_digest(model):
    """
        Digest (see weights_digest) of the kernels and biases the model
        has now, in the order its layers were created
    """

    return weights_digest(
        [array for _, _, weights in _weight_layers(model)
         for array in weights])


def set_initial_weights(model, seed=DEFAULT_SEED):
    """
        Give a freshly built model the initial weights of the seed:
        Glorot uniform kernels from the seeded stream, zero biases

        Args:
            model: a float32 model built by bpnet.model.arch.BPNet (it
                records the names of its layers with weights, in
                creation order, in `weight_layer_names`)
            seed (int): the seed; 1234 gives the weights a model
                started from in a training run under TensorFlow 2.4.1

        Returns:
            str: digest of the weights that were set (see
                weights_digest)
    """

    layers = _weight_layers(model)
    for name, _, weights in layers:
        if any(w.dtype != np.float32 for w in weights):
            raise ValueError(
                "Layer {} has {} weights; only float32 models can be "
                "given seeded initial weights".format(
                    name, weights[0].dtype))

    kernels = initial_kernels([w[0].shape for _, _, w in layers], seed)
    for (_, layer, weights), kernel in zip(layers, kernels):
        layer.set_weights([kernel, np.zeros_like(weights[1])])
    # of the weights the model now has
    return current_weights_digest(model)


def initialize(model, seed, drawn_from_seed):
    """
        Make the initial weights of a freshly built model follow the
        global seed, whatever the TensorFlow version

        - The weights are left as TensorFlow drew them where TensorFlow
          draws initial weights from its global seed itself
          (TensorFlow 2.4: nothing changes there), when no global seed
          is set (random weights were asked for), and when the model is
          not float32.
        - Otherwise (newer TensorFlow versions, whose initializers
          ignore the global seed) the weights of the seed are
          assigned: the weights TensorFlow 2.4.1 draws for the model
          right after tf.random.set_seed(seed).

        Args:
            model: a model just built by bpnet.model.arch.BPNet
            seed (int): TensorFlow's global seed, None if none is set
            drawn_from_seed (bool): whether TensorFlow derives the
                initial weights of new layers from its global seed
                (bpnet.train.training.initializers_follow_global_seed)

        Returns:
            dict: 'seed', 'sha256' (digest of the initial weights, see
                weights_digest) and 'drawn_by': 'tensorflow' if the
                weights were left as drawn, 'bpnet' if they were
                assigned here
    """

    float32 = all(w.dtype == np.float32
                  for _, _, weights in _weight_layers(model)
                  for w in weights)
    if seed is None or drawn_from_seed or not float32:
        return {'seed': seed, 'sha256': current_weights_digest(model),
                'drawn_by': 'tensorflow'}
    return {'seed': seed, 'sha256': set_initial_weights(model, seed),
            'drawn_by': 'bpnet'}
