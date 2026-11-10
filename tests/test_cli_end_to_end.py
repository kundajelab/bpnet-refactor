"""
    End to end tests of the command line tools on the synthetic
    dataset: bpnet-train, then bpnet-predict on the saved model. The
    predictions and the files derived from them are recomputed here
    from the saved weights with the numpy reference model and compared.

    The tools run as subprocesses, as a user runs them.
"""

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


#: seconds after which a command line tool is considered hung (a worker
#: process that dies leaves the batch generator waiting forever)
TIMEOUT = 900


def run_cli(module, args, cwd):
    """
        Run `python -m <module> <args>` in its own process group;
        returns (return code, output). The whole group is killed when
        the tool does not finish in time.
    """

    import signal

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    if not env.get("BPNET_TEST_GPU"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.Popen(
        [sys.executable, "-m", module] + [str(a) for a in args], cwd=cwd,
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


@pytest.fixture(scope="module")
def trained(synth_dataset, tmp_path_factory):
    """
        One bpnet-train run of a few epochs on the synthetic dataset
    """

    workdir = tmp_path_factory.mktemp("train")
    model_dir = workdir / "model"
    model_dir.mkdir()
    d = synth_dataset
    code, output = run_cli("bpnet.cli.bpnettrainer", [
        "--input-data", d["input_json"], "--output-dir", model_dir,
        "--reference-genome", d["fasta"], "--chroms"] + d["chroms"] + [
        "--chrom-sizes", d["chrom_sizes"], "--splits", d["splits_json"],
        "--model-arch-name", "BPNet",
        "--model-arch-params-json", d["params_json"],
        "--sequence-generator-name", "BPNet",
        "--model-output-filename", "model",
        "--input-seq-len", d["input_len"], "--output-len", d["output_len"],
        "--max-jitter", 8, "--shuffle", "--threads", 2,
        "--epochs", EPOCHS, "--batch-size", 16,
        "--reverse-complement-augmentation", "--learning-rate", 0.001],
        cwd=str(workdir))
    return {"code": code, "output": output, "dir": str(model_dir),
            "model": str(model_dir / "model_split000")}


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
    legacy_stack = tuple(int(x) for x in tf.__version__.split(".")[:2]) \
        < (2, 5)
    if code != 0 and not legacy_stack and "__init__() missing" in output \
            and "required positional argument" in output:
        # TF 2.11: "CustomModel.__init__() missing 6 required positional
        # arguments"; TF 2.15: "Functional.__init__() missing 2 ...".
        # On the TF 2.4 stack the prediction tests must run.
        pytest.xfail("bpnet-predict cannot load a saved model with this "
                     "TensorFlow version (keras cannot rebuild the "
                     "CustomModel class from the saved configuration)")
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
