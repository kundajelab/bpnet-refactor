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


def without_control(params, tracks=2, seed=3):
    """
        The layer graph of a BPNet without control inputs, as
        tests/reference_model.py describes the architecture, and
        random weights for it

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
        entry("Dense", "logcounts_predictions", ["dropped"], units=1,
              activation="linear")]
    weights["logcounts_predictions"] = {
        "kernel": rng.normal(0, 0.3, size=(filters, 1)),
        "bias": rng.normal(0, 0.3, size=1)}
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


def test_no_tf32_while_the_scores_are_computed(case, monkeypatch):
    """
        On a GPU, PyTorch computes convolutions in TF32 unless told
        otherwise, which is not precise enough for these scores: it
        is switched off while they are computed, and as it was
        afterwards (also after a failure)
    """

    import tangermeme.deep_lift_shap

    seen = []
    real = tangermeme.deep_lift_shap.deep_lift_shap

    def watching(*args, **kwargs):
        seen.append((torch.backends.cudnn.allow_tf32,
                     torch.backends.cuda.matmul.allow_tf32))
        return real(*args, **kwargs)

    def failing(*args, **kwargs):
        raise RuntimeError("no scores today")

    def flags():
        return (torch.backends.cudnn.allow_tf32,
                torch.backends.cuda.matmul.allow_tf32)

    at_first = flags()
    try:
        for before in ((True, True), (True, False), (False, False)):
            torch.backends.cudnn.allow_tf32 = before[0]
            torch.backends.cuda.matmul.allow_tf32 = before[1]
            monkeypatch.setattr(tangermeme.deep_lift_shap, "deep_lift_shap",
                                watching)
            scores(case, "counts")
            assert flags() == before
            monkeypatch.setattr(tangermeme.deep_lift_shap, "deep_lift_shap",
                                failing)
            with pytest.raises(RuntimeError, match="no scores today"):
                scores(case, "counts")
            assert flags() == before
    finally:
        torch.backends.cudnn.allow_tf32, \
            torch.backends.cuda.matmul.allow_tf32 = at_first
    assert seen == [(False, False)] * 3


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
    # a model that cannot be loaded is reported, as by the other backend
    arguments = list(command["arguments"])
    arguments[arguments.index("--model") + 1] = str(tmp_path)
    code, output = run_shap(arguments + ["--output-directory", tmp_path],
                            tmp_path)
    assert code != 0 and "Cannot load the model" in output, output[-2000:]
