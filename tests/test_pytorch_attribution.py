"""
    The contribution scores of bpnet-shap from the PyTorch model
    (bpnet.pytorch.attribution, tangermeme's deep_lift_shap) and
    `bpnet-shap --backend pytorch`: the scores are those of DeepLIFT
    with the rescale rule as tests/reference_deeplift.py computes it
    independently, for the outputs and the references of bpnet-shap.
    The same reference is what the scores of the TensorFlow backend
    are compared with (tests/test_shap_scores.py), which ties the two
    backends together without an environment that has both.

    None of this needs TensorFlow: the model with control inputs is
    one that TensorFlow 2.4.1 saved (tests/data/saved_models), the
    model without is written out here layer by layer.
"""

import copy
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tangermeme")

from bpnet.model import loading
from bpnet.pytorch import attribution
from bpnet.pytorch import model as pytorch_model
from bpnet.utils.shaputils import dinuc_shuffle

from tests import reference_deeplift
from tests.model_helpers import small_params, unpack_fixture_models

SHUFFLES, SEED = 6, 11


# --------------------------------------------------------------------
# the models
# --------------------------------------------------------------------

def entry(class_name, name, inputs=(), **config):
    return {"class_name": class_name, "name": name,
            "config": dict(config, name=name),
            "inbound_nodes": [[[other, 0, 0, {}] for other in inputs]]
            if inputs else []}


def without_control(params, tracks=2, seed=3, counts=1):
    """
        The layer graph of a BPNet without control inputs, as
        tests/reference_model.py describes the architecture, and
        random weights for it (`counts`: how many log counts it
        predicts)

        Returns:
            tuple: (layer graph, weights by layer name as the loader
                returns them)
    """

    motif = params["motif_module_params"]
    syntax = params["syntax_module_params"]
    head = params["profile_head_params"]
    filters = syntax["filters"]
    assert len(motif["kernel_sizes"]) == 1 and \
        syntax["pre_activation_residual_unit"]
    rng = np.random.RandomState(seed)
    weights = {}

    def conv(name, source, n_in, n_out, width, dilation=1):
        weights[name] = {
            "kernel": rng.normal(0, 0.3, size=(width, n_in, n_out)),
            "bias": rng.normal(0, 0.3, size=n_out)}
        return entry("Conv1D", name, [source], filters=n_out,
                     kernel_size=[width], dilation_rate=[dilation],
                     padding="valid", activation="linear")

    length = params["input_len"]
    layers = [entry("InputLayer", "sequence",
                    batch_input_shape=[None, length, 4]),
              conv("main_conv_0", "sequence", 4, filters,
                   motif["kernel_sizes"][0])]
    length -= motif["kernel_sizes"][0] - 1
    x = "main_conv_0"
    for i in range(1, syntax["num_dilation_layers"] + 1):
        shrink = 2 ** i * (syntax["kernel_size"] - 1)
        layers += [
            entry("ReLU", "relu_{}".format(i), [x]),
            conv("main_dil_conv_{}".format(i), "relu_{}".format(i), filters,
                 filters, syntax["kernel_size"], 2 ** i),
            entry("Cropping1D", "crop_{}".format(i), [x],
                  cropping=[shrink // 2, shrink // 2]),
            entry("Add", "add_{}".format(i),
                  ["main_dil_conv_{}".format(i), "crop_{}".format(i)])]
        x = "add_{}".format(i)
        length -= shrink
    length -= head["kernel_size"] - 1
    trim = (length - params["output_profile_len"]) // 2
    layers += [
        entry("ReLU", "relu_last", [x]),
        conv("main_profile_head", "relu_last", filters, tracks,
             head["kernel_size"]),
        entry("Cropping1D", "profile_predictions", ["main_profile_head"],
              cropping=[trim, trim]),
        entry("GlobalAveragePooling1D", "pooled", ["relu_last"]),
        # (does nothing when the model is evaluated, as the scores
        # need it)
        entry("Dropout", "dropped", ["pooled"], rate=0.5),
        entry("Dense", "logcounts_predictions", ["dropped"], units=counts,
              activation="linear")]
    weights["logcounts_predictions"] = {
        "kernel": rng.normal(0, 0.3, size=(filters, counts)),
        "bias": rng.normal(0, 0.3, size=counts)}
    architecture = {
        "name": "bpnet", "layers": layers,
        "input_layers": [["sequence", 0, 0]],
        "output_layers": [["profile_predictions", 0, 0],
                          ["logcounts_predictions", 0, 0]]}
    return architecture, weights


def random_sequences(n, length, seed):
    rng = np.random.RandomState(seed)
    out = np.zeros((n, length, 4))
    out[np.arange(n)[:, None], np.arange(length)[None],
        rng.randint(0, 4, size=(n, length))] = 1
    return out


@pytest.fixture(scope="module", params=["with control", "without control"])
def case(request, tmp_path_factory):
    """
        A model in float64, sequences for it (one with positions
        without a base, one without any base), its control inputs,
        and what the reference needs: the weights by layer name and
        the architecture parameters
    """

    params = small_params()
    if request.param == "with control":
        models = unpack_fixture_models(
            "2.4.1", str(tmp_path_factory.mktemp("fixtures")))
        directory = os.path.join(models, "released_structure")
        path = os.path.join(directory, "model")
        saved = loading.read_weights(path)
        model = pytorch_model.load_model(path).double()
        stored = np.load(os.path.join(directory, "expected.npz"))
        rng = np.random.RandomState(1)
        controls = {
            "profile_bias_input_0": np.concatenate([
                stored["input/profile_bias_input_0"].astype(np.float64),
                rng.poisson(1.0, size=(3, 80, 2)).astype(np.float64)]),
            "counts_bias_input_0": np.concatenate([
                stored["input/counts_bias_input_0"].astype(np.float64),
                rng.uniform(1.0, 5.0, size=(3, 2))])}
    else:
        architecture, saved = without_control(params)
        model = pytorch_model.from_architecture(architecture, saved)
        path, controls = None, {}
    assert all(p.dtype == torch.float64 for p in model.parameters())
    sequences = random_sequences(5, params["input_len"], seed=2)
    sequences[1, 40:43] = 0
    sequences[4] = 0
    weights = {name: [np.asarray(w["kernel"], dtype=np.float64),
                      np.asarray(w["bias"], dtype=np.float64)]
               for name, w in saved.items()}
    return {"name": request.param, "model": model, "path": path,
            "sequences": sequences, "controls": controls,
            "weights": weights, "params": params}


def expected_scores(case, which, i, sequences=None):
    sequences = case["sequences"] if sequences is None else sequences
    shuffled = np.asarray(dinuc_shuffle(
        np.ascontiguousarray(sequences[i]), SHUFFLES,
        np.random.RandomState(SEED)))
    extra = {}
    if case["controls"]:
        extra = {
            "profile_bias": case["controls"]["profile_bias_input_0"][i],
            "counts_bias": case["controls"]["counts_bias_input_0"][i]}
    return reference_deeplift.hypothetical_scores(
        case["weights"], case["params"], which, sequences[i], shuffled,
        **extra)


def scores(case, which, **settings):
    settings = dict({"num_shuffles": SHUFFLES, "seed": SEED,
                     "device": "cpu"}, **settings)
    return attribution.hypothetical_scores(
        case["model"], which, case["sequences"], case["controls"],
        **settings)


# --------------------------------------------------------------------
# the scores
# --------------------------------------------------------------------

@pytest.mark.parametrize("which", ["counts", "profile"])
def test_scores_are_those_of_the_independent_deeplift(case, which):
    """
        In float64 the scores of all four bases at every position
        equal the independent computation to the rounding of float64,
        for sequences with and without positions that have no base;
        no warning is raised on the way
    """

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        found = scores(case, which)
    assert found.shape == case["sequences"].shape
    assert found.dtype == np.float64
    scales = []
    for i in range(len(found)):
        expected = expected_scores(case, which, i)
        scale = np.abs(expected).max()
        scales.append(scale)
        assert np.abs(found[i] - expected).max() <= 1e-10 * scale, i
        # the bases that are not there have scores of their own
        absent = case["sequences"][i] == 0
        assert np.abs(expected[absent]).max() > 0.1 * scale
    assert min(scales) > 0
    # the sequences have scores of their own
    assert np.abs(found[0] - found[2]).max() > 0.1 * scales[0]


@pytest.mark.parametrize("which", ["counts", "profile"])
def test_batches_and_chunks_do_not_change_the_scores(case, which):
    expected = scores(case, which)
    scale = np.abs(expected).max()
    for settings in ({"batch_size": 1}, {"batch_size": 5, "chunk": 2},
                     {"batch_size": 64, "chunk": 1},
                     {"batch_size": 7, "chunk": 100}):
        found = scores(case, which, **settings)
        assert np.abs(found - expected).max() <= 1e-12 * scale, settings


@pytest.mark.parametrize("which", ["counts", "profile"])
def test_scores_of_a_float32_model(case, which):
    """
        A model computes in the type of its parameters: float32 as
        the models are saved, with scores in float32 that are those of
        the float64 computation to the rounding of float32
    """

    model = case["model"]
    single = copy.deepcopy(model).float()
    found = attribution.hypothetical_scores(
        single, which, case["sequences"], case["controls"],
        num_shuffles=SHUFFLES, seed=SEED, device="cpu")
    assert found.dtype == np.float32
    expected = scores(case, which)
    assert np.abs(found - expected).max() <= 2e-4 * np.abs(expected).max()
    # the model that was given keeps its type
    assert all(p.dtype == torch.float64 for p in model.parameters())


def test_log_counts_of_several_values_are_summed():
    """
        The counts output that is explained is the sum of the
        predicted log counts, for a model that predicts more than one
    """

    params = small_params()
    architecture, saved = without_control(params, counts=2)
    model = pytorch_model.from_architecture(architecture, saved)
    weights = {name: [np.asarray(w["kernel"], dtype=np.float64),
                      np.asarray(w["bias"], dtype=np.float64)]
               for name, w in saved.items()}
    sequences = random_sequences(3, params["input_len"], seed=12)
    found = attribution.hypothetical_scores(
        model, "counts", sequences, num_shuffles=SHUFFLES, seed=SEED,
        device="cpu")
    one = dict(weights, logcounts_predictions=[
        weights["logcounts_predictions"][0][:, :1],
        weights["logcounts_predictions"][1][:1]])
    for i in range(len(sequences)):
        shuffled = np.asarray(dinuc_shuffle(
            np.ascontiguousarray(sequences[i]), SHUFFLES,
            np.random.RandomState(SEED)))
        expected = reference_deeplift.hypothetical_scores(
            weights, params, "counts", sequences[i], shuffled)
        scale = np.abs(expected).max()
        assert np.abs(found[i] - expected).max() <= 1e-10 * scale
        # (not the scores of the first value alone)
        first = reference_deeplift.hypothetical_scores(
            one, params, "counts", sequences[i], shuffled)
        assert np.abs(first - expected).max() > 0.05 * scale


def test_autocast_of_the_caller_does_not_change_the_scores(case):
    """
        The scores are computed in the type of the model's parameters
        also when the caller is inside an autocast (which would
        compute a float32 model in bfloat16 where it can)
    """

    single = copy.deepcopy(case["model"]).float()

    def compute():
        return attribution.hypothetical_scores(
            single, ("counts", "profile"), case["sequences"],
            case["controls"], num_shuffles=3, seed=SEED, device="cpu")

    plain = compute()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = compute()
        # (the autocast of the caller is on again afterwards)
        assert torch.ones(2, 2).mm(torch.ones(2, 2)).dtype == torch.bfloat16
    for a, b in zip(plain, inside):
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a, b)


def test_type_of_the_sequences_does_not_matter(case):
    expected = scores(case, "profile")
    for dtype in (np.int8, np.float32, bool):
        found = attribution.hypothetical_scores(
            case["model"], "profile", case["sequences"].astype(dtype),
            case["controls"], num_shuffles=SHUFFLES, seed=SEED,
            device="cpu")
        assert found.dtype == np.float64
        assert np.array_equal(found, expected)


def test_both_outputs_together(case, monkeypatch):
    """
        Both outputs in one call: the scores of the two separate
        calls, with the references of a chunk of sequences made once
    """

    made = []
    real = attribution.references

    def counting(*args, **kwargs):
        made.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(attribution, "references", counting)
    both = scores(case, ("counts", "profile"), chunk=2)
    assert made == [2, 2, 1]
    assert isinstance(both, list) and len(both) == 2
    assert np.array_equal(both[0], scores(case, "counts"))
    assert np.array_equal(both[1], scores(case, "profile"))
    # in the order asked for; one output asked for as a list
    other = scores(case, ["profile", "counts"])
    assert np.array_equal(other[0], both[1])
    assert np.array_equal(other[1], both[0])
    one = scores(case, ("profile",))
    assert isinstance(one, list) and np.array_equal(one[0], both[1])


def test_seed_and_number_of_references(case):
    first = scores(case, "counts")
    assert np.array_equal(first, scores(case, "counts"))
    other_seed = scores(case, "counts", seed=SEED + 1)
    more = scores(case, "counts", num_shuffles=SHUFFLES + 1)
    scale = np.abs(first).max()
    assert np.abs(other_seed - first).max() > 1e-3 * scale
    assert np.abs(more - first).max() > 1e-3 * scale


def test_references_are_those_of_bpnet_shap():
    """
        For every sequence the shuffles of dinuc_shuffle from a random
        state that starts from the seed: what the data function of
        bpnet-shap returns for the sequence
    """

    sequences = random_sequences(4, 120, seed=5)
    sequences[2] = sequences[0]
    sequences[3, 10:20] = 0
    for dtype in (np.float64, np.int8):
        x = sequences.astype(dtype)
        found = attribution.references(x, 7, 99)
        assert found.shape == (4, 7, 120, 4) and found.dtype == dtype
        for i in range(4):
            assert np.array_equal(found[i], dinuc_shuffle(
                np.ascontiguousarray(x[i]), 7, np.random.RandomState(99)))
        # the same sequence, the same shuffles
        assert np.array_equal(found[0], found[2])
        assert not np.array_equal(found[0], found[1])
    assert not np.array_equal(
        attribution.references(sequences, 7, 99),
        attribution.references(sequences, 7, 100))
    # a view of another array (not contiguous)
    reverse = sequences[:, ::-1]
    assert np.array_equal(
        attribution.references(reverse, 3, 1),
        attribution.references(np.ascontiguousarray(reverse), 3, 1))


# --------------------------------------------------------------------
# the targets
# --------------------------------------------------------------------

def as_tensors(case, rows):
    model = case["model"]
    x = torch.tensor(pytorch_model.channels_first(case["sequences"][rows]))
    controls = [torch.tensor(pytorch_model.channels_first(
        case["controls"][name][rows])) for name in model.input_names[1:]]
    return x, controls


def test_targets_of_a_batch_of_sequences_and_references(case):
    """
        What deep_lift_shap gets to explain, for a batch of two
        sequences followed by two references that it gives the
        control inputs of the sequences: the references are run
        without control, the counts target is the sum of the log
        counts, the profile target the weighted sum of the logits with
        the weights of each pair averaged
    """

    model = case["model"]
    x, controls = as_tensors(case, [0, 1])
    r, _ = as_tensors(case, [2, 3])
    batch = torch.cat([x, r])
    doubled = [torch.cat([c, c]) for c in controls]
    with torch.no_grad():
        profile_x, counts_x = model(x, *controls)
        profile_r, counts_r = model(r, *[torch.zeros_like(c)
                                         for c in controls])
        counts = attribution.CountsTarget(model)(batch, *doubled)
        profile = attribution.ProfileTarget(model)(batch, *doubled)
    assert counts.shape == profile.shape == (4, 1)
    np.testing.assert_allclose(
        counts[:, 0].numpy(),
        torch.cat([counts_x, counts_r]).sum(dim=-1).numpy(), rtol=1e-12)

    def normed(p):
        flat = p.reshape(p.shape[0], -1).numpy()
        return flat - flat.mean(axis=1, keepdims=True)

    def softmax(l):
        e = np.exp(l - l.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    l_x, l_r = normed(profile_x), normed(profile_r)
    w = 0.5 * (softmax(l_x) + softmax(l_r))
    np.testing.assert_allclose(
        profile[:, 0].numpy(),
        np.concatenate([(w * l_x).sum(axis=1), (w * l_r).sum(axis=1)]),
        rtol=1e-12)
    if controls:
        # with the control of the sequences for the references the
        # targets would be others
        with torch.no_grad():
            _, other = model(r, *controls)
        assert not np.allclose(other.sum(dim=-1).numpy(),
                               counts[2:, 0].numpy(), rtol=1e-6)

    # the weights are not explained: the gradient of the profile
    # target is that of the logits with the weights held
    batch = batch.clone().requires_grad_(True)
    target = attribution.ProfileTarget(model)(batch, *doubled)
    gradient, = torch.autograd.grad(target.sum(), batch)
    held = torch.tensor(np.concatenate([w, w]))
    again = batch.detach().clone().requires_grad_(True)
    profile_again = model(again, *[
        torch.cat([c, torch.zeros_like(c)]) for c in controls])[0]
    flat = profile_again.reshape(4, -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    expected, = torch.autograd.grad((held * flat).sum(), again)
    np.testing.assert_allclose(gradient.numpy(), expected.numpy(),
                               rtol=1e-10, atol=1e-14)


def test_targets_refuse_what_they_cannot_pair(case):
    model = case["model"]
    x, controls = as_tensors(case, [0, 1, 2])
    for target in (attribution.CountsTarget, attribution.ProfileTarget):
        with pytest.raises(ValueError, match="followed by as many "
                           "references is expected; it has 3 rows"):
            target(model)(x, *controls)


def test_models_the_targets_do_not_fit():
    params = small_params()
    architecture, weights = without_control(params)
    # one output
    one = dict(architecture, output_layers=[["logcounts_predictions", 0, 0]])
    with pytest.raises(ValueError, match="a profile and a counts output"):
        attribution.CountsTarget(pytorch_model.from_architecture(
            one, weights))
    # two profile outputs
    two = dict(architecture, output_layers=[["profile_predictions", 0, 0],
                                            ["main_profile_head", 0, 0]])
    with pytest.raises(ValueError, match="a profile and a counts output"):
        attribution.ProfileTarget(pytorch_model.from_architecture(
            two, weights))
    # the sequence is not the first input
    model = pytorch_model.from_architecture(architecture, weights)
    model.input_names = ["other"]
    with pytest.raises(ValueError, match="'sequence' is expected"):
        attribution.CountsTarget(model)
    # the outputs in the other order are found by their shapes
    swapped = dict(architecture, output_layers=[
        ["logcounts_predictions", 0, 0], ["profile_predictions", 0, 0]])
    model = pytorch_model.from_architecture(swapped, weights)
    straight = pytorch_model.from_architecture(architecture, weights)
    sequences = random_sequences(2, params["input_len"], seed=8)
    for which in ("counts", "profile"):
        assert np.array_equal(
            attribution.hypothetical_scores(
                model, which, sequences, num_shuffles=3, device="cpu"),
            attribution.hypothetical_scores(
                straight, which, sequences, num_shuffles=3, device="cpu"))


def test_models_with_other_nonlinear_layers_are_refused(case):
    """
        The scores are compared with an independent DeepLIFT for
        models whose nonlinear layers are ReLU (every model that
        bpnet-train makes). deep_lift_shap has rules of its own for
        other activations, and none for some; a model with one is
        refused rather than scored unchecked
    """

    attribution.check_model(case["model"])
    model = copy.deepcopy(case["model"])
    name = next(name for name, layer in model.layers.items()
                if isinstance(layer, torch.nn.ReLU))
    for other in (torch.nn.Tanh(), torch.nn.LeakyReLU(0.1), torch.nn.GELU(),
                  torch.nn.Softsign()):
        model.layers[name] = other
        message = "only nonlinear layers are ReLU; this model has layers " \
            "of the kind {}".format(type(other).__name__)
        with pytest.raises(ValueError, match=message):
            attribution.check_model(model)
        with pytest.raises(ValueError, match=message):
            attribution.hypothetical_scores(
                model, "counts", case["sequences"], case["controls"],
                num_shuffles=2, device="cpu")
    # dropout does nothing when a model is evaluated
    model.layers[name] = torch.nn.Dropout(0.5)
    attribution.check_model(model)
    # a model in half precision, or one of two types (the loader makes
    # models of float32 or float64)
    for change, types in (
            (lambda m: m.half(), r"\['float16'\]"),
            (lambda m: m.type(torch.bfloat16), r"\['bfloat16'\]"),
            (lambda m: next(iter(m.layers.values())).float(),
             r"\['float32', 'float64'\]")):
        other = copy.deepcopy(case["model"])
        change(other)
        message = "float32 or float64; the parameters of this model have " \
            "the types " + types
        with pytest.raises(ValueError, match=message):
            attribution.check_model(other)
        with pytest.raises(ValueError, match=message):
            attribution.hypothetical_scores(
                other, "counts", case["sequences"], case["controls"],
                num_shuffles=2, device="cpu")
    attribution.check_model(copy.deepcopy(case["model"]).float())
    # what is not a model of this package has layers and steps that
    # nothing here knows
    for other in (torch.nn.Linear(2, 2), attribution.CountsTarget(model)):
        with pytest.raises(ValueError, match="A model of bpnet.pytorch.model "
                           "is expected, not a " + type(other).__name__):
            attribution.check_model(other)


def with_logsumexp(params, of):
    """
        A model without control inputs whose two log counts go through
        the logsumexp that the models of this package take of their
        control counts: directly ('counts'), or together with an input
        of two values, the input first ('both')
    """

    architecture, weights = without_control(params, counts=2)
    layers, taken = architecture["layers"], ["logcounts_predictions"]
    if of == "both":
        layers.append(entry("InputLayer", "other",
                            batch_input_shape=[None, 2]))
        layers.append(entry("Concatenate", "together",
                            ["other", "logcounts_predictions"], axis=-1))
        architecture["input_layers"].append(["other", 0, 0])
        taken = ["together"]
    layers.append(entry("Lambda", "main_logsumexp_counts_bias_0", taken))
    architecture["output_layers"][1][0] = "main_logsumexp_counts_bias_0"
    return pytorch_model.from_architecture(architecture, weights)


@pytest.mark.parametrize("of", ["counts", "both"])
def test_a_logsumexp_of_what_comes_from_the_sequence_is_refused(case, of):
    """
        The logsumexp of the control counts has no module, so
        deep_lift_shap does not see it and takes its gradient. That is
        nothing to the scores while it is taken of control inputs, as
        in the models of bpnet-train (the model with control inputs
        here has one, and is scored like the reference); a model that
        takes it of something computed from the sequence is refused
    """

    kinds = [kind for _, kind, _ in case["model"].steps()]
    assert ("LogSumExp" in kinds) == bool(case["controls"])
    attribution.check_model(case["model"])

    params = small_params()
    model = with_logsumexp(params, of)
    message = "only nonlinear layers are ReLU; this model computes the " \
        "layer main_logsumexp_counts_bias_0 \\(LogSumExp\\) from the " \
        "sequence"
    with pytest.raises(ValueError, match=message):
        attribution.check_model(model)
    sequences = random_sequences(2, params["input_len"], seed=4)
    controls = {"other": np.ones((2, 2))} if of == "both" else None
    with pytest.raises(ValueError, match=message):
        attribution.hypothetical_scores(
            model, "counts", sequences, controls, num_shuffles=2,
            device="cpu")
    # the model itself computes: it is the scores that are not verified
    extra = [torch.ones(2, 2, dtype=torch.float64)] if of == "both" else []
    with torch.no_grad():
        _, counts = model(torch.tensor(pytorch_model.channels_first(
            sequences)), *extra)
    assert counts.shape == (2, 1) and bool(torch.isfinite(counts).all())


def test_devices_the_scores_are_computed_on(case, monkeypatch):
    """
        A CPU or a CUDA GPU that PyTorch sees; without a device, a
        GPU when there is one. Anything else is refused before
        anything is computed, also a GPU number that PyTorch would
        take for another (it keeps the number in eight bits: cuda:256
        is cuda:0 to it, cuda:255 the default GPU, cuda:128 a GPU
        with a negative number)
    """

    cpu = torch.device("cpu")
    assert attribution.resolve_device("cpu") == cpu
    assert attribution.resolve_device(cpu) == cpu
    there = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert attribution.resolve_device(None) == torch.device(
        "cuda" if there else "cpu")
    if there:
        assert attribution.resolve_device("cuda:0") == torch.device("cuda:0")
    for device, message in (
            ("nowhere", "device nowhere: not a device of PyTorch"),
            (3, "device 3: "),
            ("mps", "device mps: the scores are computed on a CPU or a CUDA "
             "GPU .*, not on a device of the kind mps"),
            ("xpu", "not on a device of the kind xpu"),
            ("meta", "not on a device of the kind meta"),
            (torch.device("meta"), "device meta: .* of the kind meta"),
            ("cuda:{}".format(there), "device cuda:{}: PyTorch sees {} "
             r"GPU\(s\) here".format(there, there)),
            ("cuda:99", r"device cuda:99: PyTorch sees {} GPU\(s\) "
             "here".format(there)),
            ("cuda:128", "device cuda:128: PyTorch sees"),
            ("cuda:255", "device cuda:255: PyTorch sees"),
            ("cuda:256", "device cuda:256: PyTorch sees"),
            (torch.device("cuda:128"), "device cuda:-128: PyTorch sees")):
        if device == 3 and there > 3:
            continue
        with pytest.raises(ValueError, match=message):
            attribution.resolve_device(device)
        with pytest.raises(ValueError, match=message):
            attribution.hypothetical_scores(
                case["model"], "counts", case["sequences"], case["controls"],
                num_shuffles=2, device=device)
    if not there:
        with pytest.raises(ValueError, match=r"device cuda: PyTorch sees 0 "
                           r"GPU\(s\) here"):
            attribution.resolve_device("cuda")
    # as if PyTorch saw one GPU: that one, and no number that wraps
    # around to it
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for device in (None, "cuda", "cuda:0", torch.device("cuda"),
                   torch.device("cuda:0")):
        assert attribution.resolve_device(device) == torch.device(
            "cuda" if device is None else device)
    for device in ("cuda:1", "cuda:128", "cuda:255", "cuda:256", "cuda:257",
                   "cuda:512", torch.device("cuda:1"),
                   torch.device("cuda:128")):
        with pytest.raises(ValueError, match=r"PyTorch sees 1 GPU\(s\) "
                           "here"):
            attribution.resolve_device(device)


def fake_deep_lift_shap(order):
    """
        A stand-in for deep_lift_shap that goes through the pairs of
        sequences and references as tangermeme 1.6 does, and hands
        the model each batch as `order` lays it out
    """

    def deep_lift_shap(model, X, args=None, references=None, batch_size=32,
                       dtype=torch.float32, **settings):
        pairs = [(i, j) for i in range(X.shape[0])
                 for j in range(references.shape[1])]
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            x = torch.stack([X[i] for i, _ in batch]).type(dtype)
            r = torch.stack([references[i, j] for i, j in batch]).type(dtype)
            a = None if args is None else torch.stack(
                [args[0][i] for i, _ in batch]).type(dtype)
            given = order(x, r, a)
            if given is not None:
                model(*given)
        return torch.zeros(X.shape)

    return deep_lift_shap


def twice(a):
    return None if a is None else torch.cat([a, a])


#: ways to lay out a batch of sequences x, their references r and the
#: extra inputs a of the sequences (None when there are none)
LAYOUTS = {
    "as tangermeme 1.6": lambda x, r, a: (torch.cat([x, r]), twice(a)),
    "references first": lambda x, r, a: (torch.cat([r, x]), twice(a)),
    "sequences and references in turn": lambda x, r, a: (
        torch.stack([x, r], dim=1).reshape((-1,) + x.shape[1:]),
        None if a is None else a.repeat_interleave(2, dim=0)),
    "references in another order": lambda x, r, a: (
        torch.cat([x, r.flip(0)]), twice(a)),
    "sequences in another order": lambda x, r, a: (
        torch.cat([x.flip(0), r]), twice(a)),
    "sequences only": lambda x, r, a: (x, a),
    "the model is not called": lambda x, r, a: None,
}

#: ways that differ from tangermeme 1.6 in the extra inputs alone
LAYOUTS_OF_EXTRA_INPUTS = {
    "no extra inputs for the references": lambda x, r, a: (
        torch.cat([x, r]), torch.cat([a, torch.zeros_like(a)])),
    "extra inputs of the references in another order": lambda x, r, a: (
        torch.cat([x, r]), torch.cat([a, a.flip(0)])),
    "extra inputs of the sequences in another order": lambda x, r, a: (
        torch.cat([x, r]), torch.cat([a.flip(0), a])),
    "extra inputs left out": lambda x, r, a: (torch.cat([x, r]), None),
}


@pytest.mark.parametrize(
    "layout", sorted(LAYOUTS) + sorted(LAYOUTS_OF_EXTRA_INPUTS))
def test_layout_of_a_batch_is_looked_at(case, monkeypatch, layout):
    """
        The targets rely on how deep_lift_shap lays out a batch: the
        sequences of its pairs, then their references row for row,
        the extra inputs for both halves. A deep_lift_shap that lays
        it out in another way is refused before a score is computed
    """

    from tangermeme import deep_lift_shap as module

    monkeypatch.setattr(attribution, "_LAYOUT_CHECKED", set())
    cpu, double = torch.device("cpu"), torch.float64
    extra_inputs_only = layout in LAYOUTS_OF_EXTRA_INPUTS
    fake = fake_deep_lift_shap(
        dict(LAYOUTS, **LAYOUTS_OF_EXTRA_INPUTS)[layout])
    if layout == "as tangermeme 1.6":
        attribution._check_layout(fake, cpu, double, True)
        attribution._check_layout(fake, cpu, double, False)
        assert attribution._LAYOUT_CHECKED == {
            (fake, "cpu", double, True), (fake, "cpu", double, False)}
        return
    for extra in (True,) if extra_inputs_only else (True, False):
        with pytest.raises(RuntimeError, match="hands a model other batches"):
            attribution._check_layout(fake, cpu, double, extra)
    assert attribution._LAYOUT_CHECKED == set()
    if extra_inputs_only and not case["controls"]:
        # (a model without control inputs has no extra inputs)
        return
    monkeypatch.setattr(module, "deep_lift_shap", fake)
    with pytest.raises(RuntimeError, match="hands a model other batches"):
        scores(case, "counts")


def test_layout_of_the_installed_tangermeme(case, monkeypatch):
    """
        The installed deep_lift_shap lays out its batches as the
        targets take them to be, with and without extra inputs, for
        float64 and float32. Scoring looks at it in a call like its
        own (sequences as small integers, hypothetical scores, the
        type of the model, extra inputs when the model has control
        inputs), once for each such way of calling and again for
        another deep_lift_shap.
    """

    from tangermeme import deep_lift_shap as module

    real = module.deep_lift_shap
    cpu = torch.device("cpu")
    monkeypatch.setattr(attribution, "_LAYOUT_CHECKED", set())
    for dtype in (torch.float64, torch.float32):
        for extra in (True, False):
            attribution._check_layout(real, cpu, dtype, extra)
    assert len(attribution._LAYOUT_CHECKED) == 4

    looked = []

    def watching(model, X, **settings):
        if isinstance(model, attribution._Layout):
            looked.append((X.dtype, settings["references"].dtype,
                           settings["args"] is not None,
                           settings["hypothetical"], settings["dtype"],
                           str(settings["device"])))
        return real(model, X, **settings)

    def other(model, X, **settings):
        return watching(model, X, **settings)

    with_controls = bool(case["controls"])
    as_scores_call = [torch.int8, torch.int8, with_controls, True]
    monkeypatch.setattr(attribution, "_LAYOUT_CHECKED", set())
    monkeypatch.setattr(module, "deep_lift_shap", watching)
    first = scores(case, "counts", num_shuffles=2)
    assert looked == [tuple(as_scores_call + [torch.float64, "cpu"])]
    assert attribution._LAYOUT_CHECKED == {
        (watching, "cpu", torch.float64, with_controls)}
    # not again for the same way of calling ...
    assert np.array_equal(scores(case, "counts", num_shuffles=2), first)
    scores(case, "profile", num_shuffles=2)
    assert len(looked) == 1
    # ... again for a model of another type, and for another
    # deep_lift_shap
    single = copy.deepcopy(case["model"]).float()
    attribution.hypothetical_scores(
        single, "counts", case["sequences"], case["controls"],
        num_shuffles=2, device="cpu")
    assert looked[1:] == [tuple(as_scores_call + [torch.float32, "cpu"])]
    monkeypatch.setattr(module, "deep_lift_shap", other)
    scores(case, "counts", num_shuffles=2)
    assert len(looked) == 3 and looked[2] == looked[0]
    assert attribution._LAYOUT_CHECKED == {
        (watching, "cpu", torch.float64, with_controls),
        (watching, "cpu", torch.float32, with_controls),
        (other, "cpu", torch.float64, with_controls)}


def test_arguments_that_are_refused(case):
    model, sequences = case["model"], case["sequences"]
    for which in ("both", (), ("counts", "other"), ["counts", None]):
        with pytest.raises(ValueError, match="'counts' or 'profile'"):
            attribution.hypothetical_scores(model, which, sequences,
                                            case["controls"])
    with pytest.raises(ValueError, match=r"shape \(n, length, 4\)"):
        attribution.hypothetical_scores(
            model, "counts", sequences.transpose(0, 2, 1), case["controls"])
    with pytest.raises(ValueError, match="takes the control inputs"):
        attribution.hypothetical_scores(
            model, "counts", sequences,
            dict(case["controls"], other=np.zeros(5)))
    # what is not one hot
    for change in (lambda x: x * 0.5, lambda x: x + 1, lambda x: -x,
                   lambda x: np.ones_like(x)):
        with pytest.raises(ValueError, match="one hot expected"):
            attribution.hypothetical_scores(
                model, "counts", change(sequences), case["controls"],
                device="cpu")
    if case["controls"]:
        with pytest.raises(ValueError, match="takes the control inputs"):
            attribution.hypothetical_scores(model, "counts", sequences)
    # numbers of shuffles, pairs per batch and sequences per chunk that
    # are none (deep_lift_shap takes a batch size below 1 for "all
    # pairs at once", and no chunk would be scored with a chunk below 1)
    for name in ("num_shuffles", "batch_size", "chunk"):
        for value in (0, -1, 2.5, True, "3", None):
            with pytest.raises(ValueError, match="{}: a whole number of at "
                               "least 1 is expected".format(name)):
                attribution.hypothetical_scores(
                    model, "counts", sequences, case["controls"],
                    device="cpu", **{name: value})
    one = attribution.hypothetical_scores(
        model, "counts", sequences[:2], {
            key: value[:2] for key, value in case["controls"].items()},
        num_shuffles=np.int64(2), batch_size=np.int32(1), chunk=1,
        seed=SEED, device="cpu")
    assert np.array_equal(one, scores(case, "counts", num_shuffles=2)[:2])


def test_model_is_left_as_it_was(case):
    """
        The model is run in evaluation mode (dropout does nothing) and
        is in the mode it was in afterwards; its parameters are not
        touched
    """

    model = case["model"]
    before = [p.detach().clone() for p in model.parameters()]
    expected = scores(case, "counts")
    model.train()
    try:
        assert np.array_equal(scores(case, "counts"), expected)
        assert model.training
    finally:
        model.eval()
    assert not model.training
    assert all(torch.equal(a, b)
               for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def precision_settings():
    """
        What decides the precision of a float32 computation in this
        PyTorch, as PyTorch reports it
    """

    read = getattr(torch._C, "_get_fp32_precision_getter", None)
    if read is None:
        return (torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
    return tuple(read(backend, kind) for backend, kind in (
        ("generic", "all"), ("cuda", "all"), ("cuda", "conv"),
        ("cuda", "rnn"), ("cuda", "matmul"), ("mkldnn", "all"),
        ("mkldnn", "conv"), ("mkldnn", "rnn"), ("mkldnn", "matmul")))


def test_full_precision_while_the_scores_are_computed(case, monkeypatch):
    """
        On a GPU, PyTorch computes convolutions in TF32 unless told
        otherwise, which is not precise enough for these scores; and
        a caller may have asked for less precision still. While the
        scores are computed (and while the layout of a batch is looked
        at) every setting is at full precision, and afterwards as it
        was, also after a failure. (How the settings are put back in
        every case: tests/test_pytorch_precision.py.)
    """

    import tangermeme.deep_lift_shap

    seen = []
    real = tangermeme.deep_lift_shap.deep_lift_shap

    def watching(*args, **kwargs):
        seen.append(precision_settings())
        return real(*args, **kwargs)

    def failing(*args, **kwargs):
        seen.append(precision_settings())
        raise RuntimeError("no scores today")

    # a reduced precision that can be put back exactly afterwards:
    # the top setting of PyTorch 2.9 and later, the flag of cuDNN
    # before
    newer = hasattr(torch._C, "_get_fp32_precision_getter")
    if newer:
        was = torch.backends.fp32_precision
        torch.backends.fp32_precision = "tf32"
        full = ("ieee",) * 9
    else:
        was = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        full = (False, "highest")
    try:
        before = precision_settings()
        assert before != full and ("tf32" in before or before[0] is True)
        monkeypatch.setattr(attribution, "_LAYOUT_CHECKED", set())
        monkeypatch.setattr(tangermeme.deep_lift_shap, "deep_lift_shap",
                            watching)
        scores(case, "counts")
        assert precision_settings() == before
        monkeypatch.setattr(tangermeme.deep_lift_shap, "deep_lift_shap",
                            failing)
        with pytest.raises(RuntimeError, match="no scores today"):
            scores(case, "counts")
        assert precision_settings() == before
    finally:
        if newer:
            torch.backends.fp32_precision = was
        else:
            torch.backends.cudnn.allow_tf32 = was
    # the look at the layout, the scores of the five sequences in one
    # chunk, and the call that failed
    assert seen == [full] * 3


def test_progress_is_reported(case):
    seen = []
    scores(case, "counts", chunk=2, progress=seen.append)
    assert seen == [2, 4, 5]


# --------------------------------------------------------------------
# the command
# --------------------------------------------------------------------

def save_as_hdf5(path, architecture, weights):
    """
        A layer graph and its weights in the HDF5 layout of keras'
        model.save() (the configuration as JSON in an attribute, the
        variables of every layer in a group of its name), so that the
        command has a file to load where no TensorFlow wrote one
    """

    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["model_config"] = json.dumps(
            {"class_name": "Functional", "config": architecture})
        group = f.create_group("model_weights")
        for layer, variables in weights.items():
            names = []
            for variable, value in variables.items():
                name = "{}/{}:0".format(layer, variable)
                group.create_dataset(layer + "/" + name, data=value)
                names.append(name.encode())
            group[layer].attrs["weight_names"] = names
    return path


def run_shap(arguments, cwd):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo, CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-m", "bpnet.cli.shap_scores"] +
        [str(a) for a in arguments], cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=600)
    return done.returncode, done.stdout


@pytest.fixture(scope="module")
def command(tmp_path_factory):
    """
        `bpnet-shap --backend pytorch` on a synthetic dataset, with
        the model that TensorFlow 2.4.1 saved
    """

    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    from tests import synth

    params = small_params()
    work = tmp_path_factory.mktemp("command")
    models = unpack_fixture_models("2.4.1", str(work))
    model = os.path.join(models, "released_structure", "model")
    d = synth.make_dataset(str(work / "data"),
                           input_len=params["input_len"],
                           output_len=params["output_profile_len"])
    out = work / "shap"
    out.mkdir()
    arguments = [
        "--backend", "pytorch", "--device", "cpu",
        "--reference-genome", d["fasta"],
        "--input-seq-len", params["input_len"],
        "--control-len", params["output_profile_len"], "--model", model,
        "--bed-file", d["peaks"], "--chroms", d["chroms"][-1],
        "--input-data", d["input_json"], "--num-shuffles", SHUFFLES,
        "--seed", SEED, "--batch-size", 50,
        "--generate-shap-bigWigs", "--chrom-sizes", d["chrom_sizes"]]
    code, output = run_shap(arguments + ["--output-directory", out], out)
    assert code == 0, output[-3000:]
    files = {}
    for which in ("counts", "profile"):
        with h5py.File(str(out / (which + "_scores.h5")), "r") as f:
            files[which] = {key: f[key][:] for key in f}
    return {"dataset": d, "model": model, "out": out, "output": output,
            "scores": files, "params": params, "arguments": arguments}


def test_command_writes_the_files_of_bpnet_shap(command):
    out = command["out"]
    assert sorted(os.listdir(str(out))) == [
        "config.json", "counts_scores.bw", "counts_scores.h5",
        "counts_scores.stats.txt", "peaks_valid_scores.bed",
        "profile_scores.bw", "profile_scores.h5", "profile_scores.stats.txt",
        "shap_scores.log"]
    with open(str(out / "config.json")) as fh:
        config = json.load(fh)
    assert config["backend"] == "pytorch" and config["batch_size"] == 50
    assert config["num_shuffles"] == SHUFFLES and config["seed"] == SEED
    n = sum(1 for _ in open(str(out / "peaks_valid_scores.bed")))
    assert n > 20
    length = command["params"]["input_len"]
    for which in ("counts", "profile"):
        f = command["scores"][which]
        assert sorted(f) == ["coords_chrom", "coords_end", "coords_start",
                             "hyp_scores", "input_seqs"]
        assert f["hyp_scores"].shape == (n, length, 4)
        assert f["hyp_scores"].dtype == np.float16
        assert f["input_seqs"].shape == (n, length, 4)
        assert f["input_seqs"].dtype == np.int8
        assert (f["coords_end"] - f["coords_start"] == length).all()
        assert np.isfinite(f["hyp_scores"]).all()
    assert np.array_equal(command["scores"]["counts"]["input_seqs"],
                          command["scores"]["profile"]["input_seqs"])
    # TensorFlow had no part in it
    assert "tensorflow" not in command["output"].lower()


def control_inputs(d, f, row, output_len):
    """
        The control inputs the command gives the model for a row of a
        score file: the control tracks in the output window around the
        summit, and the log of their totals plus one
    """

    import pyBigWig

    chrom = f["coords_chrom"][row]
    chrom = chrom.decode() if isinstance(chrom, bytes) else str(chrom)
    centre = (int(f["coords_start"][row]) + int(f["coords_end"][row])) // 2
    with open(d["input_json"]) as fh:
        tasks = json.load(fh)
    tracks = []
    for path in tasks["0"]["bias"]["source"]:
        bigwig = pyBigWig.open(path)
        tracks.append(np.nan_to_num(bigwig.values(
            chrom, centre - output_len // 2, centre + output_len // 2)))
        bigwig.close()
    profile = np.stack(tracks, axis=1)
    return profile, np.log(profile.sum(axis=0) + 1)


@pytest.mark.parametrize("which", ["counts", "profile"])
def test_command_scores_are_those_of_the_independent_deeplift(command,
                                                              which):
    """
        The scores in the files of the command (a float32 model,
        float16 files) are those of the independent computation to
        the rounding of the files, for every base of the first twelve
        peaks
    """

    f = command["scores"][which]
    params = command["params"]
    weights = {name: [np.asarray(w["kernel"], dtype=np.float64),
                      np.asarray(w["bias"], dtype=np.float64)]
               for name, w in loading.read_weights(command["model"]).items()}
    for i in range(12):
        sequence = f["input_seqs"][i].astype(np.float64)
        shuffled = np.asarray(dinuc_shuffle(
            np.ascontiguousarray(sequence), SHUFFLES,
            np.random.RandomState(SEED)))
        profile_bias, counts_bias = control_inputs(
            command["dataset"], f, i, params["output_profile_len"])
        expected = reference_deeplift.hypothetical_scores(
            weights, params, which, sequence, shuffled,
            profile_bias=profile_bias, counts_bias=counts_bias)
        scale = np.abs(expected).max()
        rounding = np.maximum(2.0 ** -11 * np.abs(expected), 2.0 ** -25)
        beyond = np.abs(f["hyp_scores"][i].astype(np.float64) - expected) - \
            rounding
        # (float32 arithmetic of the model: 1e-5 of the largest score)
        assert beyond.max() <= 1e-5 * scale, (i, beyond.max() / scale)
        assert np.abs(expected[sequence == 0]).max() > 0.1 * scale


def test_command_for_a_model_without_control_inputs(tmp_path):
    """
        The command with a model that has no control inputs (written
        here as an HDF5 file, in float64): the scores of both files
        against the independent computation
    """

    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    from tests import synth

    params = small_params()
    architecture, saved = without_control(params)
    model = save_as_hdf5(str(tmp_path / "model.h5"), architecture, saved)
    loaded = pytorch_model.load_model(model)
    assert loaded.input_names == ["sequence"]
    d = synth.make_dataset(str(tmp_path / "data"),
                           input_len=params["input_len"],
                           output_len=params["output_profile_len"])
    out = tmp_path / "shap"
    out.mkdir()
    code, output = run_shap([
        "--backend", "pytorch", "--device", "cpu",
        "--reference-genome", d["fasta"],
        "--input-seq-len", params["input_len"],
        "--control-len", params["output_profile_len"], "--model", model,
        "--bed-file", d["peaks"], "--chroms", d["chroms"][-1],
        "--input-data", d["input_json"], "--num-shuffles", SHUFFLES,
        "--seed", SEED, "--output-directory", out], out)
    assert code == 0, output[-3000:]
    weights = {name: [w["kernel"], w["bias"]] for name, w in saved.items()}
    for which in ("counts", "profile"):
        with h5py.File(str(out / (which + "_scores.h5")), "r") as f:
            stored = f["hyp_scores"][:8].astype(np.float64)
            sequences = f["input_seqs"][:8].astype(np.float64)
        for i in range(8):
            shuffled = np.asarray(dinuc_shuffle(
                np.ascontiguousarray(sequences[i]), SHUFFLES,
                np.random.RandomState(SEED)))
            expected = reference_deeplift.hypothetical_scores(
                weights, params, which, sequences[i], shuffled)
            rounding = np.maximum(2.0 ** -11 * np.abs(expected), 2.0 ** -25)
            beyond = np.abs(stored[i] - expected) - rounding
            assert beyond.max() <= 1e-9 * np.abs(expected).max(), (which, i)


def test_command_refuses_a_model_with_one_profile_track(tmp_path):
    """
        For a model with one profile track the TensorFlow backend
        writes half the profile scores of the definition; this
        backend refuses such a model rather than write the other half
    """

    pytest.importorskip("h5py")
    from tests import synth

    params = small_params()
    architecture, saved = without_control(params, tracks=1)
    model = save_as_hdf5(str(tmp_path / "model.h5"), architecture, saved)
    d = synth.make_dataset(str(tmp_path / "data"),
                           input_len=params["input_len"],
                           output_len=params["output_profile_len"],
                           stranded=False)
    code, output = run_shap([
        "--backend", "pytorch", "--device", "cpu",
        "--reference-genome", d["fasta"],
        "--input-seq-len", params["input_len"],
        "--control-len", params["output_profile_len"], "--model", model,
        "--bed-file", d["peaks"], "--chroms", d["chroms"][-1],
        "--input-data", d["input_json"], "--output-directory", tmp_path],
        tmp_path)
    assert code != 0, output[-2000:]
    assert "explains models with two profile tracks (both strands); this " \
        "model has 1" in output
    assert "Traceback" not in output
    assert not os.path.exists(str(tmp_path / "counts_scores.h5"))
    # the function itself scores such a model, by the definition
    loaded = pytorch_model.load_model(model)
    found = attribution.hypothetical_scores(
        loaded, "profile", random_sequences(2, params["input_len"], seed=6),
        num_shuffles=3, device="cpu")
    assert np.isfinite(found).all() and np.abs(found).max() > 0


def test_command_refuses_what_the_backend_does_not_do(command, tmp_path):
    for extra, message in (
            (["--orig-multi-loss"], "are not supported with it"),
            (["--task-id", 1], "are not supported with it")):
        code, output = run_shap(
            command["arguments"] + extra + ["--output-directory", tmp_path],
            tmp_path)
        assert code != 0 and message in output, output[-2000:]
        assert not os.path.exists(str(tmp_path / "counts_scores.h5"))
    # numbers that are none, devices that are not there
    for extra, message in (
            (["--batch-size", 0], "--batch-size must be at least 1, not 0"),
            (["--batch-size", -1], "--batch-size must be at least 1, not -1"),
            (["--num-shuffles", 0], "--num-shuffles must be at least 1"),
            (["--device", "nowhere"], "--device nowhere: not a device of "
             "PyTorch"),
            (["--device", "cuda:0"], "--device cuda:0: PyTorch sees 0 "
             "GPU(s) here"),
            (["--device", "cuda:128"], "--device cuda:128: PyTorch sees 0 "
             "GPU(s) here"),
            (["--device", "cuda:256"], "--device cuda:256: PyTorch sees 0 "
             "GPU(s) here"),
            (["--device", "mps"], "--device mps: the scores are computed on "
             "a CPU or a CUDA GPU ('cpu', 'cuda', 'cuda:1', ...), not on a "
             "device of the kind mps"),
            (["--device", "xpu"], "not on a device of the kind xpu"),
            (["--device", "meta"], "not on a device of the kind meta")):
        code, output = run_shap(
            command["arguments"] + extra + ["--output-directory", tmp_path],
            tmp_path)
        assert code != 0 and message in output, output[-2000:]
        assert "Traceback" not in output
        assert not os.path.exists(str(tmp_path / "counts_scores.h5"))
    # a model with an activation for which the scores are not verified
    # (the counts head of this one has a LeakyReLU)
    arguments = list(command["arguments"])
    arguments[arguments.index("--model") + 1] = os.path.join(
        os.path.dirname(os.path.dirname(command["model"])),
        "hidden_counts_layers", "model")
    code, output = run_shap(arguments + ["--output-directory", tmp_path],
                            tmp_path)
    assert code != 0, output[-2000:]
    assert "--backend pytorch: The scores of the PyTorch backend are " \
        "verified for models whose only nonlinear layers are ReLU; this " \
        "model has layers of the kind LeakyReLU" in output
    assert "Traceback" not in output
    assert not os.path.exists(str(tmp_path / "counts_scores.h5"))
    # a model that cannot be loaded is reported, as by the other backend
    arguments = list(command["arguments"])
    arguments[arguments.index("--model") + 1] = str(tmp_path)
    code, output = run_shap(arguments + ["--output-directory", tmp_path],
                            tmp_path)
    assert code != 0 and "Cannot load the model" in output, output[-2000:]
