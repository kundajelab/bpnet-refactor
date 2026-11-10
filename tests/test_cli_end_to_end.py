"""
    End to end tests of the command line tools on the synthetic
    dataset: bpnet-train, then bpnet-predict on the saved model. The
    predictions and the files derived from them are recomputed here
    from the saved weights with the numpy reference model and compared.

    The tools run as subprocesses, as a user runs them.
"""

import hashlib
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import rankdata

tf = pytest.importorskip("tensorflow")

from tests import reference_model
from tests.checks import one_hot
from tests.saved_weights import weights_by_layer_name
from tests.test_bigwig_writing import check_file
from tests.test_metrics import expected_metrics

pytestmark = pytest.mark.slow

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCHS = 4

HISTORY_KEYS = [
    "batch_loss", "elapsed", "end_time", "learning_rate",
    "logcounts_predictions_loss", "loss", "profile_predictions_loss",
    "start_time", "val_batch_loss", "val_logcounts_predictions_loss",
    "val_loss", "val_profile_predictions_loss"]


# sha256 over the initial kernels and biases of the synthetic dataset's
# architecture for three seeds. The one of seed 1234 is what the
# unmodified trainer (upstream main at 0b11a7c) started from on this
# dataset under TensorFlow 2.4.1; under that version all are the
# weights Keras draws itself after tf.random.set_seed(seed) (see
# test_initialization_tf.py). Every TensorFlow version must start there
INITIAL_WEIGHTS_SHA256 = {
    1234: "56a4b9e7de8f3a4363f0b3b4b08e24289fe9cd7f43204053f2a71c9c52199be6",
    7: "df30f8c0cc3c46ca982edec4ee1c79c0dd391c09ff6f5dad55cfacbe4b08af72",
    0: "e44c42849f007e53b798b9d2e864564b4a59587f75edcf04607b7f643bd73785"}

#: TensorFlow 2.4 draws the initial weights from the global seed
#: itself; from 2.10 on the trainer assigns them (see
#: test_initialization_tf.py for the versions in between)
DRAWN_BY = "tensorflow" if tuple(
    int(x) for x in tf.__version__.split(".")[:2]) < (2, 10) else "bpnet"

#: the layers with weights in the order the model creates them
LAYERS = ["main_conv_0", "main_dil_conv_1", "main_dil_conv_2",
          "main_profile_head", "main_counts_head", "profile_predictions",
          "logcounts_predictions"]

#: seconds after which a command line tool is considered hung (a worker
#: process that dies leaves the batch generator waiting forever)
TIMEOUT = 900


def run_cli(module, args, cwd, code=None):
    """
        Run `python -m <module> <args>` in its own process group;
        returns (return code, output). The whole group is killed when
        the tool does not finish in time.

        With `code`, `python -c <code> <args>` is run instead (a
        script that changes something and then calls the tool's
        main()).
    """

    import signal

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    if not env.get("BPNET_TEST_GPU"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = ["-m", module] if code is None else ["-c", code]
    proc = subprocess.Popen(
        [sys.executable] + command + [str(a) for a in args], cwd=cwd,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail("{} did not finish within {} s\n{}".format(
            module, TIMEOUT, output[-3000:]))
    return proc.returncode, output


def run_trainer(d, workdir, epochs=EPOCHS, max_jitter=8,
                learning_rate=0.001, extra=(), threads=2, code=None):
    """
        Run bpnet-train on the synthetic dataset `d` in `workdir`
    """

    model_dir = workdir / "model"
    model_dir.mkdir()
    code, output = run_cli("bpnet.cli.bpnettrainer", [
        "--input-data", d["input_json"], "--output-dir", model_dir,
        "--reference-genome", d["fasta"], "--chroms"] + d["chroms"] + [
        "--chrom-sizes", d["chrom_sizes"], "--splits", d["splits_json"],
        "--model-arch-name", "BPNet",
        "--model-arch-params-json", d["params_json"],
        "--sequence-generator-name", "BPNet",
        "--model-output-filename", "model",
        "--input-seq-len", d["input_len"], "--output-len", d["output_len"],
        "--max-jitter", max_jitter, "--shuffle", "--threads", threads,
        "--epochs", epochs, "--batch-size", 16,
        "--reverse-complement-augmentation",
        "--learning-rate", learning_rate] + list(extra), cwd=str(workdir),
        code=code)
    return {"code": code, "output": output, "dir": str(model_dir),
            "model": str(model_dir / "model_split000")}


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def initial_weights_log_lines(run):
    with open(os.path.join(run["dir"], "trainer_split000.log")) as fh:
        return [line.split("Initial weights: ")[1].strip()
                for line in fh if "Initial weights: " in line]


@pytest.fixture(scope="module")
def trained(synth_dataset, tmp_path_factory):
    """
        One bpnet-train run of a few epochs on the synthetic dataset
    """

    return run_trainer(synth_dataset, tmp_path_factory.mktemp("train"))


@pytest.fixture(scope="module")
def frozen(synth_dataset, tmp_path_factory):
    """
        bpnet-train runs of one epoch with a learning rate of zero,
        without jitter, and with background regions that count in the
        profile loss like peaks. The weights that are saved are the
        weights the model started from, and the training loss is the
        mean loss of that model over the examples of the epoch, however
        they were grouped into batches.
    """

    runs = {}
    for name, extra in [("default", []), ("seed 7", ["--seed", 7]),
                        ("seed 0", ["--seed", 0])]:
        runs[name] = run_trainer(
            synth_dataset, tmp_path_factory.mktemp("frozen"), epochs=1,
            max_jitter=0, learning_rate=0,
            extra=["--mnll-loss-background-sample-weight", 1] + extra)
    return runs


@pytest.fixture(scope="module")
def predicted(synth_dataset, trained, tmp_path_factory):
    """
        One bpnet-predict run on the test chromosome with reverse
        complement averaging and bigWig output
    """

    assert trained["code"] == 0, trained["output"][-3000:]
    out_dir = tmp_path_factory.mktemp("predict")
    d = synth_dataset
    test_chroms = d["splits"]["0"]["test"]
    code, output = run_cli("bpnet.cli.predict", [
        "--model", trained["model"], "--chrom-sizes", d["chrom_sizes"],
        "--chroms"] + test_chroms + [
        "--test-indices-file", "None", "--reference-genome", d["fasta"],
        "--output-dir", out_dir, "--input-data", d["input_json"],
        "--sequence-generator-name", "BPNet",
        "--input-seq-len", d["input_len"], "--output-len", d["output_len"],
        "--output-window-size", d["output_len"], "--batch-size", 16,
        "--reverse-complement-average", "--threads", 2,
        "--generate-predicted-profile-bigWigs"], cwd=str(out_dir))
    assert code == 0, output[-3000:]

    with h5py.File(str(out_dir / "model_split000_predictions.h5"), "r") as f:
        arrays = {
            "chrom": np.array([c.decode() if isinstance(c, bytes) else c
                               for c in f["coords/coords_chrom"][:]]),
            "start": f["coords/coords_start"][:],
            "end": f["coords/coords_end"][:],
            "pred_profs": f["predictions/pred_profs"][:],
            "pred_logcounts": f["predictions/pred_logcounts"][:],
            "true_profs": f["predictions/true_profs"][:],
            "true_logcounts": f["predictions/true_logcounts"][:],
        }
    return {"dir": str(out_dir), "arrays": arrays, "output": output}


# --------------------------------------------------------------------
# bpnet-train
# --------------------------------------------------------------------

def test_train_writes_model_history_and_config(synth_dataset, trained):
    assert trained["code"] == 0, trained["output"][-3000:]
    files = sorted(os.listdir(trained["dir"]))
    assert files == ["config_split000.json", "model_split000",
                     "model_split000.history.json", "trainer.log",
                     "trainer_split000.log"]
    # the model is a TensorFlow SavedModel directory
    assert os.path.isfile(os.path.join(trained["model"], "saved_model.pb"))
    assert os.path.isdir(os.path.join(trained["model"], "variables"))

    with open(trained["model"] + ".history.json") as fh:
        history = json.load(fh)
    assert sorted(history) == HISTORY_KEYS
    epochs = [str(i) for i in range(1, EPOCHS + 1)]
    for key in HISTORY_KEYS:
        assert sorted(history[key], key=int) == epochs, key
    assert all(history["learning_rate"][e] == pytest.approx(0.001)
               for e in epochs)
    # training makes progress on data that have structure
    assert history["loss"][epochs[-1]] < history["loss"]["1"]
    assert history["val_loss"][epochs[-1]] < history["val_loss"]["1"]

    with open(os.path.join(trained["dir"], "config_split000.json")) as fh:
        config = json.load(fh)
    splits = synth_dataset["splits"]["0"]
    assert config["training_epochs"] == EPOCHS
    # the best epoch is the one with the lowest validation loss
    val_losses = [history["val_loss"][e] for e in epochs]
    assert config["best_epoch"] == int(np.argmin(val_losses)) + 1
    assert config["train_chroms"] == splits["train"]
    assert config["val_chroms"] == splits["val"]
    assert config["model_filename"] == trained["model"]
    assert config["hyper_params"]["batch_size"] == 16
    assert config["batch_gen_params"]["max_jitter"] == 8
    # the sequences go from the worker processes to the model as int8
    assert config["batch_gen_params"]["sequence_dtype"] == "int8"


#: bpnet-train with the one hot sequences of the batches as float64,
#: the type they had before the tool asked the generator for int8
TRAIN_WITH_FLOAT64_SEQUENCES = """
from bpnet.cli import bpnettrainer

train = bpnettrainer.training.train_and_validate_ksplits


def with_float64_sequences(*args, **kwargs):
    settings = [value for value in list(args) + list(kwargs.values())
                if isinstance(value, dict) and 'sequence_dtype' in value]
    assert len(settings) == 1 and settings[0]['sequence_dtype'] == 'int8'
    settings[0]['sequence_dtype'] = 'float64'
    return train(*args, **kwargs)


bpnettrainer.training.train_and_validate_ksplits = with_float64_sequences
bpnettrainer.main()
"""


@pytest.fixture(scope="module")
def trained_with_either_type(synth_dataset, tmp_path_factory):
    """
        Two bpnet-train runs that see the same batches in the same
        order: one worker process and no jitter. (With more workers
        the order of the batches depends on which worker is done
        first, and the jitter comes from a random generator that
        Python seeds anew in every worker process.) One run as the
        tool is, one with the sequences as float64.
    """

    runs = {}
    for name, code in [("int8", None),
                       ("float64", TRAIN_WITH_FLOAT64_SEQUENCES)]:
        runs[name] = run_trainer(
            synth_dataset, tmp_path_factory.mktemp(name), epochs=2,
            max_jitter=0, threads=1, code=code)
    return runs


@pytest.mark.skipif(
    bool(os.environ.get("BPNET_TEST_GPU")),
    reason="two training runs on a GPU do not repeat to the last bit")
def test_trained_model_does_not_depend_on_the_type_of_the_sequences(
        trained_with_either_type):
    """
        The whole of bpnet-train with int8 sequences against the same
        run with float64 sequences: the same losses in every epoch and
        the same saved weights, bit for bit
    """

    runs = trained_with_either_type
    histories, weights = {}, {}
    for name, run in runs.items():
        assert run["code"] == 0, run["output"][-3000:]
        config = read_json(os.path.join(run["dir"], "config_split000.json"))
        assert config["batch_gen_params"]["sequence_dtype"] == name
        assert config["training_epochs"] == 2
        histories[name] = read_json(run["model"] + ".history.json")
        weights[name] = weights_by_layer_name(run["model"])

    timing = {"start_time", "end_time", "elapsed"}
    assert set(histories["int8"]) == set(HISTORY_KEYS)
    compared = 0
    for key in sorted(set(HISTORY_KEYS) - timing):
        assert histories["int8"][key] == histories["float64"][key], key
        compared += 1
    assert compared == len(HISTORY_KEYS) - 3 >= 9
    # the model was trained: the loss moved
    assert histories["int8"]["loss"]["2"] != histories["int8"]["loss"]["1"]

    assert sorted(weights["int8"]) == sorted(weights["float64"]) == \
        sorted(LAYERS)
    for layer in LAYERS:
        for a, b in zip(weights["int8"][layer], weights["float64"][layer]):
            assert a.dtype == b.dtype == np.float32
            assert a.tobytes() == b.tobytes(), layer


def test_last_validation_loss_is_the_saved_models_loss(synth_dataset, trained):
    """
        Training ran to the epoch limit here, so the saved weights are
        those of the last epoch, and the last validation loss in the
        history is the loss of the saved model on the validation loci:
        the mean over batches of (profile loss + counts loss weight *
        counts loss). Recomputed with the reference model on batches of
        the validation generator; the batches are grouped differently
        than during training, which changes the mean slightly.
    """

    from bpnet.generators import generators

    assert trained["code"] == 0, trained["output"][-3000:]
    d = synth_dataset
    with open(trained["model"] + ".history.json") as fh:
        history = json.load(fh)
    weights = weights_by_layer_name(trained["model"])

    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": d["input_len"], "output_len": d["output_len"],
              "rev_comp_aug": False, "max_jitter": 0, "shuffle": False,
              "mode": "val"}
    gen = generators.MBPNetSequenceGenerator(
        d["input_json"], params, d["fasta"], d["chrom_sizes"],
        d["splits"]["0"]["val"], num_threads=2, batch_size=16, epochs=1)
    losses = []
    for inputs, outputs, sample_weights in gen.gen():
        profile, counts = reference_model.forward(
            weights, d["params"], inputs["sequence"],
            inputs["profile_bias_input_0"], inputs["counts_bias_input_0"])
        losses.append(reference_model.loss(
            outputs["profile_predictions"], outputs["logcounts_predictions"],
            sample_weights, profile, counts, d["params"]["loss_weights"])[0])
    assert len(losses) == gen.len() > 1

    # regrouping the batches of a fixed model moves this mean by about
    # 1% on these data
    np.testing.assert_allclose(np.mean(losses),
                               history["val_loss"][str(EPOCHS)], rtol=0.03)


def test_trainer_reports_the_initial_weights(trained):
    assert trained["code"] == 0, trained["output"][-3000:]
    config = read_json(os.path.join(trained["dir"], "config_split000.json"))
    assert config["initial_weights"] == {
        "seed": 1234, "sha256": INITIAL_WEIGHTS_SHA256[1234],
        "drawn_by": DRAWN_BY}
    assert config["hyper_params"]["seed"] is None
    assert initial_weights_log_lines(trained) == [
        "seed 1234, sha256 {}, drawn by {}".format(
            INITIAL_WEIGHTS_SHA256[1234], DRAWN_BY)]


@pytest.mark.parametrize("run,seed", [("default", 1234), ("seed 7", 7),
                                      ("seed 0", 0)])
def test_training_starts_from_the_seeded_weights(frozen, run, seed):
    """
        With a learning rate of zero the saved weights are the initial
        weights: Glorot uniform kernels of the seeded stream and zero
        biases, on every TensorFlow version. Without --seed they are
        the weights of seed 1234.
    """

    from bpnet.model import initialization

    assert frozen[run]["code"] == 0, frozen[run]["output"][-3000:]
    weights = weights_by_layer_name(frozen[run]["model"])
    assert sorted(weights) == sorted(LAYERS)
    digest = hashlib.sha256()
    for name in LAYERS:
        kernel, bias = weights[name]
        assert kernel.dtype == bias.dtype == np.float32
        assert not bias.any(), name
        digest.update(kernel.tobytes() + bias.tobytes())
    assert digest.hexdigest() == INITIAL_WEIGHTS_SHA256[seed]

    kernels = initialization.initial_kernels(
        [weights[name][0].shape for name in LAYERS], seed)
    for name, kernel in zip(LAYERS, kernels):
        assert np.array_equal(weights[name][0], kernel), name

    config = read_json(os.path.join(frozen[run]["dir"],
                                    "config_split000.json"))
    assert config["initial_weights"] == {
        "seed": seed, "sha256": INITIAL_WEIGHTS_SHA256[seed],
        "drawn_by": DRAWN_BY}
    assert config["hyper_params"]["seed"] == (None if run == "default"
                                              else seed)
    assert initial_weights_log_lines(frozen[run]) == [
        "seed {}, sha256 {}, drawn by {}".format(
            seed, INITIAL_WEIGHTS_SHA256[seed], DRAWN_BY)]


def epoch_loss(d, weights, numpy_seed):
    """
        The mean loss of a model with the given weights over the
        training epoch of a generator that is set up right after
        np.random.seed(numpy_seed), the way the trainer sets its
        generator up. Which background regions are sampled and which
        rows are trimmed follows from that state. All examples have
        weight 1 and all batches the same size, so the mean does not
        depend on the grouping into batches.
    """

    from bpnet.generators import generators

    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": d["input_len"], "output_len": d["output_len"],
              "rev_comp_aug": True, "max_jitter": 0, "shuffle": True,
              "mode": "train"}
    state = np.random.get_state()
    try:
        np.random.seed(numpy_seed)
        gen = generators.MBPNetSequenceGenerator(
            d["input_json"], params, d["fasta"], d["chrom_sizes"],
            d["splits"]["0"]["train"], num_threads=2, batch_size=16,
            epochs=1, foreground_weight=1.0, background_weight=1.0)
        losses = []
        for inputs, outputs, sample_weights in gen.gen():
            assert len(sample_weights) == 32 and (sample_weights == 1).all()
            profile, counts = reference_model.forward(
                weights, d["params"], inputs["sequence"],
                inputs["profile_bias_input_0"],
                inputs["counts_bias_input_0"])
            losses.append(reference_model.loss(
                outputs["profile_predictions"],
                outputs["logcounts_predictions"], sample_weights, profile,
                counts, d["params"]["loss_weights"])[0])
    finally:
        np.random.set_state(state)
    assert len(losses) == gen.len() > 1
    return float(np.mean(losses))


def test_seed_governs_the_choice_of_training_loci(synth_dataset, frozen):
    """
        The background regions an epoch samples and the rows it trims
        follow from --seed: the training loss of each run (of a model
        that does not change) is the loss over the epoch that a
        generator yields after numpy is seeded with the run's seed.
        Without --seed that is the 1234 the package sets when it is
        imported.
    """

    losses, weights = {}, {}
    for name, run in frozen.items():
        assert run["code"] == 0, run["output"][-3000:]
        history = read_json(run["model"] + ".history.json")
        assert sorted(history["loss"]) == ["1"]
        losses[name] = history["loss"]["1"]
        weights[name] = weights_by_layer_name(run["model"])

    expected = epoch_loss(synth_dataset, weights["seed 7"], 7)
    assert losses["seed 7"] == pytest.approx(expected, rel=2e-4)
    assert losses["seed 0"] == pytest.approx(
        epoch_loss(synth_dataset, weights["seed 0"], 0), rel=2e-4)
    assert losses["default"] == pytest.approx(
        epoch_loss(synth_dataset, weights["default"], 1234), rel=2e-4)

    # the comparison tells the epochs of two seeds apart: the same
    # model on the epoch of another seed is far outside the tolerance
    for other_seed in (1234, 0):
        other = epoch_loss(synth_dataset, weights["seed 7"], other_seed)
        assert abs(other - expected) > 2e-3 * expected


def test_saved_weights_are_readable_by_layer_name(trained):
    assert trained["code"] == 0, trained["output"][-3000:]
    weights = weights_by_layer_name(trained["model"])
    assert sorted(weights) == sorted([
        "main_conv_0", "main_dil_conv_1", "main_dil_conv_2",
        "main_profile_head", "main_counts_head", "profile_predictions",
        "logcounts_predictions"])
    assert weights["main_conv_0"][0].shape == (21, 4, 8)
    assert weights["logcounts_predictions"][0].shape == (2, 1)


# --------------------------------------------------------------------
# bpnet-predict
# --------------------------------------------------------------------

#: bpnet-predict, with a note of the type of the sequences in every
#: batch the model is given
PREDICT_AND_NOTE_SEQUENCE_TYPES = """
import tensorflow as tf
from bpnet.cli import predict

model_predict = tf.keras.Model.predict


def noted(self, batch, *args, **kwargs):
    with open('sequence_types.txt', 'a') as note:
        note.write(str(batch['sequence'].dtype) + chr(10))
    return model_predict(self, batch, *args, **kwargs)


tf.keras.Model.predict = noted
predict.predict_main()
"""


def test_predict_gives_the_model_int8_sequences(synth_dataset, trained,
                                                tmp_path):
    assert trained["code"] == 0, trained["output"][-3000:]
    d = synth_dataset
    code, output = run_cli("bpnet.cli.predict", [
        "--model", trained["model"], "--chrom-sizes", d["chrom_sizes"],
        "--chroms"] + d["splits"]["0"]["test"] + [
        "--test-indices-file", "None", "--reference-genome", d["fasta"],
        "--output-dir", tmp_path, "--input-data", d["input_json"],
        "--sequence-generator-name", "BPNet",
        "--input-seq-len", d["input_len"], "--output-len", d["output_len"],
        "--output-window-size", d["output_len"], "--batch-size", 16,
        "--threads", 1], cwd=str(tmp_path),
        code=PREDICT_AND_NOTE_SEQUENCE_TYPES)
    assert code == 0, output[-3000:]
    with open(str(tmp_path / "sequence_types.txt")) as fh:
        types = fh.read().split()
    assert len(types) >= 2 and set(types) == {"int8"}
    assert os.path.isfile(str(tmp_path / "model_split000_predictions.h5"))


def test_predict_output_files(predicted):
    files = set(os.listdir(predicted["dir"]))
    expected = {
        "config.json", "predict.log", "model_split000_predictions.h5",
        "pearson.txt", "spearman.txt", "jsd.txt"}
    expected.update(name + ".npz" for name in (
        "mnll", "cross_entropy", "jsd", "mse", "pearson", "spearman",
        "counts_pearson", "counts_spearman"))
    for track in (0, 1):
        expected.update({
            "true_logcounts_{}.npy".format(track),
            "pred_logcounts_{}.npy".format(track),
            "model_split000_predictions_track_{}.bw".format(track),
            "model_split000_predictions_track_{}_stats.txt".format(track)})
    assert files == expected


def test_predict_covers_every_test_peak_once(synth_dataset, predicted):
    a = predicted["arrays"]
    test_chroms = synth_dataset["splits"]["0"]["test"]
    half = synth_dataset["output_len"] // 2
    peaks = sorted((c, s + m - half, s + m + half)
                   for c, s, e, _, m in synth_dataset["peak_rows"]
                   if c in test_chroms)
    got = sorted(zip(a["chrom"], a["start"].tolist(), a["end"].tolist()))
    assert got == peaks
    n = len(peaks)
    assert a["pred_profs"].shape == (n, synth_dataset["output_len"], 2)
    assert a["true_profs"].shape == (n, synth_dataset["output_len"], 2)
    assert a["pred_logcounts"].shape == (n, 2)
    assert a["true_logcounts"].shape == (n, 2)


def test_predict_observed_values_are_the_truth(synth_dataset, predicted):
    a = predicted["arrays"]
    for i in range(len(a["chrom"])):
        truth = np.stack([
            synth_dataset["tracks"][path][a["chrom"][i]]
            [a["start"][i]:a["end"][i]]
            for path in synth_dataset["signal"]], axis=1)
        assert np.array_equal(a["true_profs"][i], truth)
        np.testing.assert_allclose(a["true_logcounts"][i],
                                   np.log(truth.sum(axis=0) + 1), rtol=1e-6)


def test_predictions_match_the_reference_model(synth_dataset, trained,
                                               predicted):
    """
        Predictions are the average over the sequence and its reverse
        complement (logits and log counts averaged after turning the
        reverse complement prediction back); the predicted profile is
        the softmax over all positions of both strands times the
        predicted total counts; the per-strand log counts are the log
        of the sums of the predicted profile.
    """

    d = synth_dataset
    a = predicted["arrays"]
    weights = weights_by_layer_name(trained["model"])
    flank = d["input_len"] // 2

    for i in range(len(a["chrom"])):
        chrom, start, end = a["chrom"][i], a["start"][i], a["end"][i]
        mid = (start + end) // 2
        sequence = one_hot(d["sequences"][chrom][mid - flank:mid + flank])
        control = np.stack([d["tracks"][path][chrom][start:end]
                            for path in d["control"]], axis=1)
        control = control.astype(np.float32)
        control_counts = np.log(control.sum(axis=0) + 1)

        forward = reference_model.forward(
            weights, d["params"], sequence[None], control[None],
            control_counts[None])
        reverse = reference_model.forward(
            weights, d["params"], sequence[None, ::-1, ::-1],
            control[None, ::-1, ::-1], control_counts[None, ::-1])

        logits = 0.5 * (forward[0][0] + reverse[0][0][::-1, ::-1])
        logcount = 0.5 * (forward[1][0, 0] + reverse[1][0, 0])
        profile = np.exp(logits - logsumexp(logits)) * np.exp(logcount)

        np.testing.assert_allclose(a["pred_profs"][i], profile, rtol=2e-4,
                                   atol=1e-6)
        np.testing.assert_allclose(a["pred_logcounts"][i],
                                   np.log(profile.sum(axis=0)), rtol=2e-4)
        # the two strands' predicted counts add up to the predicted total
        np.testing.assert_allclose(a["pred_profs"][i].sum(),
                                   np.exp(logcount), rtol=2e-4)


def test_metric_files_follow_from_the_predictions(predicted):
    a = predicted["arrays"]
    out = predicted["dir"]
    n = len(a["chrom"])

    expected = {key: np.zeros((n, 2)) for key in (
        "profile_jsds", "profile_mnlls", "profile_cross_entropys",
        "profile_mses", "profile_pearsonrs", "profile_spearmanrs")}
    for i in range(n):
        for track in (0, 1):
            metrics = expected_metrics(a["true_profs"][i, :, track],
                                       a["pred_profs"][i, :, track])
            for key in expected:
                expected[key][i, track] = metrics[key]

    files = {"jsd": "profile_jsds", "mnll": "profile_mnlls",
             "cross_entropy": "profile_cross_entropys",
             "mse": "profile_mses", "pearson": "profile_pearsonrs",
             "spearman": "profile_spearmanrs"}
    for name, key in files.items():
        got = np.load(os.path.join(out, name + ".npz"))[name]
        np.testing.assert_allclose(got, expected[key], rtol=1e-7,
                                   atol=1e-9, err_msg=name)

    # counts correlations per strand, across regions
    for track in (0, 1):
        np.testing.assert_array_equal(
            np.load(os.path.join(out, "true_logcounts_{}.npy".format(track))),
            a["true_logcounts"][:, track])
        np.testing.assert_array_equal(
            np.load(os.path.join(out, "pred_logcounts_{}.npy".format(track))),
            a["pred_logcounts"][:, track])
    pearson = [np.corrcoef(a["true_logcounts"][:, t],
                           a["pred_logcounts"][:, t])[0, 1] for t in (0, 1)]
    np.testing.assert_allclose(
        np.load(os.path.join(out, "counts_pearson.npz"))["counts_pearson"],
        pearson, rtol=1e-8)

    # the three summary files: mean over strands of the counts
    # correlations, median of the per-region profile score
    def summary(name):
        with open(os.path.join(out, name)) as fh:
            return float(fh.read())
    assert summary("pearson.txt") == round(float(np.mean(pearson)), 3)
    assert summary("jsd.txt") == round(
        float(np.median(expected["profile_jsds"])), 3)
    spearman = [np.corrcoef(rankdata(a["true_logcounts"][:, t]),
                            rankdata(a["pred_logcounts"][:, t]))[0, 1]
                for t in (0, 1)]
    np.testing.assert_allclose(
        np.load(os.path.join(out, "counts_spearman.npz"))["counts_spearman"],
        spearman, rtol=1e-8)
    assert summary("spearman.txt") == round(float(np.mean(spearman)), 3)


def test_predict_log_summary(predicted):
    """
        The log ends with min / max / median of every profile metric
        over regions and strands
    """

    import re

    rows = {}
    for line in predicted["output"].splitlines():
        match = re.search(
            r"INFO - (\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)$",
            line)
        if match:
            rows[match.group(1)] = [float(v) for v in match.groups()[1:]]
    assert sorted(rows) == ["cross_entropy", "jsd", "mnll", "mse", "pearson",
                            "spearman"]
    for name, values in rows.items():
        array = np.load(os.path.join(predicted["dir"], name + ".npz"))[name]
        expected = [array.min(), array.max(), np.median(array)]
        np.testing.assert_allclose(values, expected, rtol=0, atol=5.1e-4,
                                   err_msg=name)


def test_missing_test_indices_file_is_reported(synth_dataset, trained,
                                               tmp_path):
    d = synth_dataset
    missing = tmp_path / "no_such_indices.txt"
    code, output = run_cli("bpnet.cli.predict", [
        "--model", trained["model"], "--chrom-sizes", d["chrom_sizes"],
        "--test-indices-file", missing, "--reference-genome", d["fasta"],
        "--output-dir", tmp_path, "--input-data", d["input_json"],
        "--input-seq-len", d["input_len"], "--output-len", d["output_len"],
        "--output-window-size", d["output_len"]], cwd=str(tmp_path))
    assert code != 0
    assert "File not found: {}".format(missing) in output
    assert "NameError" not in output


def test_model_that_cannot_be_loaded_is_reported(synth_dataset, tmp_path):
    d = synth_dataset
    arguments = [
        "--chrom-sizes", d["chrom_sizes"], "--test-indices-file", "None",
        "--reference-genome", d["fasta"], "--output-dir", tmp_path,
        "--input-data", d["input_json"], "--input-seq-len", d["input_len"],
        "--output-len", d["output_len"],
        "--output-window-size", d["output_len"]]

    missing = tmp_path / "no_such_model"
    code, output = run_cli("bpnet.cli.predict",
                           ["--model", missing] + arguments, cwd=str(tmp_path))
    assert code != 0
    assert "Cannot load the model {}: Model not found".format(missing) \
        in output
    assert "Traceback" not in output

    # a directory that is not a saved model
    code, output = run_cli("bpnet.cli.predict",
                           ["--model", tmp_path] + arguments,
                           cwd=str(tmp_path))
    assert code != 0
    assert "it has no saved_model.pb" in output
    assert "Traceback" not in output

    # a damaged file
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    with open(str(damaged / "saved_model.pb"), "wb") as fh:
        fh.write(b"\x12\xff\xff\xff")
    code, output = run_cli("bpnet.cli.predict",
                           ["--model", damaged] + arguments,
                           cwd=str(tmp_path))
    assert code != 0
    assert "Cannot load the model {}: Truncated".format(damaged) in output
    assert "Traceback" not in output


def test_predicted_bigwigs(synth_dataset, predicted):
    a = predicted["arrays"]
    test_chroms = sorted(synth_dataset["splits"]["0"]["test"])
    header = [(c, synth_dataset["chrom_len"]) for c in test_chroms]
    regions = [(a["chrom"][i], int(a["start"][i]), int(a["end"][i]),
                int(a["start"][i] + a["end"][i]) // 2)
               for i in range(len(a["chrom"]))]
    for track in (0, 1):
        prefix = os.path.join(
            predicted["dir"],
            "model_split000_predictions_track_{}".format(track))
        check_file(prefix + ".bw", header, regions,
                   list(a["pred_profs"][:, :, track]),
                   stats=prefix + "_stats.txt")
