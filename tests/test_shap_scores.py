"""
    bpnet-shap on the synthetic dataset: what it writes, and that the
    contribution scores are what DeepSHAP promises (for every peak they
    add up to the difference between the model's output for the
    sequence and its mean output for the shuffled references)

    The tool needs the shap fork of the lab (TFDeepExplainer with a
    function for the references); the tests are skipped where it is
    not installed.
"""

import json
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
shap = pytest.importorskip("shap")
pytest.importorskip("pysam")
if not hasattr(getattr(getattr(shap, "explainers", None), "deep", None),
               "TFDeepExplainer"):
    pytest.skip("the shap that is installed has no TFDeepExplainer",
                allow_module_level=True)

import h5py
import hdf5plugin  # noqa: F401 (the score files are Blosc compressed)
import pyBigWig

from bpnet.utils.shaputils import dinuc_shuffle

from tests import reference_model, synth
from tests.checks import one_hot
from tests.saved_weights import weights_by_layer_name
from tests.test_cli_end_to_end import run_cli, run_trainer

pytestmark = pytest.mark.slow

SHUFFLES = 6
SEED = 7


#: bpnet-shap with the model loaded by keras' own loader, as the tool
#: did before (that loader reads these models back under TensorFlow 2.4
#: only)
SHAP_WITH_THE_KERAS_LOADER = """
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from bpnet.cli import shap_scores
from bpnet.model.custommodel import CustomModel


def as_before(path):
    with CustomObjectScope({'tf': tf, 'CustomModel': CustomModel}):
        return load_model(path, compile=False)


shap_scores.loading.load_model = as_before
shap_scores.shap_scores_main()
"""


def run_shap(d, model, out_dir, seed=SEED, extra=(), code=None):
    code, output = run_cli("bpnet.cli.shap_scores", [
        "--reference-genome", d["fasta"], "--input-seq-len", d["input_len"],
        "--control-len", d["output_len"], "--model", model,
        "--bed-file", d["peaks"], "--chroms", d["chroms"][-1],
        "--input-data", d["input_json"], "--output-directory", out_dir,
        "--num-shuffles", SHUFFLES, "--seed", seed] + list(extra),
        cwd=str(out_dir), code=code)
    assert code == 0, output[-3000:]
    scores = {}
    for name in ("counts", "profile"):
        with h5py.File(str(out_dir / (name + "_scores.h5")), "r") as f:
            scores[name] = {key: f[key][:] for key in f}
    return {"dir": str(out_dir), "output": output, "scores": scores}


@pytest.fixture(scope="module")
def with_control(synth_dataset, tmp_path_factory):
    """
        A model of the default dataset (two strands, control tracks)
        after one epoch, and its scores with bigWigs
    """

    trained = run_trainer(synth_dataset, tmp_path_factory.mktemp("train"),
                          epochs=1)
    assert trained["code"] == 0, trained["output"][-3000:]
    extra = ["--generate-shap-bigWigs",
             "--chrom-sizes", synth_dataset["chrom_sizes"]]
    run = run_shap(synth_dataset, trained["model"],
                   tmp_path_factory.mktemp("shap"), extra=extra)
    return dict(run, model=trained["model"], dataset=synth_dataset,
                extra=extra)


@pytest.fixture(scope="module")
def without_control(tmp_path_factory):
    """
        A model without control inputs: the sequence is then the only
        thing its output can be attributed to
    """

    d = synth.make_dataset(str(tmp_path_factory.mktemp("data")),
                           with_control=False)
    trained = run_trainer(d, tmp_path_factory.mktemp("train"), epochs=1)
    assert trained["code"] == 0, trained["output"][-3000:]
    run = run_shap(d, trained["model"], tmp_path_factory.mktemp("shap"))
    return dict(run, model=trained["model"], dataset=d, extra=[])


def float16_bound(values):
    """
        A bound on the difference between a float64 value and its
        float16 rounding: 2**-11 of the value (half a unit in the last
        place is between half of that and that, so this is on the safe
        side by up to a factor of two), and 2**-25 for the values below
        the normal range of float16
    """

    return np.maximum(2.0 ** -11 * np.abs(values), 2.0 ** -25)


def control_inputs(d, scores, row):
    """
        The control inputs bpnet-shap gives the model for the peak of a
        row of a score file: the control tracks in the output window
        around the summit, and the log of their totals plus one
    """

    chrom = scores["coords_chrom"][row]
    chrom = chrom.decode() if isinstance(chrom, bytes) else str(chrom)
    centre = (int(scores["coords_start"][row]) +
              int(scores["coords_end"][row])) // 2
    start = centre - d["output_len"] // 2
    profile = []
    for path in d["control"]:
        bw = pyBigWig.open(path)
        try:
            profile.append(np.nan_to_num(np.array(
                bw.values(chrom, start, start + d["output_len"]))))
        finally:
            bw.close()
    profile = np.stack(profile, axis=-1)
    return profile, np.log(profile.sum(axis=0) + 1)


def test_peaks(d):
    """
        The peaks of the chromosome the scores are computed for, in the
        order of the peak file: (chromosome, summit)
    """

    return [(chrom, start + summit)
            for chrom, start, _, _, summit in d["peak_rows"]
            if chrom == d["chroms"][-1]]


test_peaks.__test__ = False


def test_output_files(with_control):
    assert sorted(os.listdir(with_control["dir"])) == [
        "config.json", "counts_scores.bw", "counts_scores.h5",
        "counts_scores.stats.txt", "peaks_valid_scores.bed",
        "profile_scores.bw", "profile_scores.h5", "profile_scores.stats.txt",
        "shap_scores.log"]
    with open(os.path.join(with_control["dir"], "config.json")) as fh:
        config = json.load(fh)
    assert config["num_shuffles"] == SHUFFLES and config["seed"] == SEED
    assert config["chroms"] == [with_control["dataset"]["chroms"][-1]]


@pytest.mark.parametrize("which", ["counts", "profile"])
def test_score_files_describe_the_peaks(with_control, which):
    d = with_control["dataset"]
    scores = with_control["scores"][which]
    peaks = test_peaks(d)
    flank = d["input_len"] // 2
    assert sorted(scores) == ["coords_chrom", "coords_end", "coords_start",
                              "hyp_scores", "input_seqs"]
    assert len(peaks) > 20
    chroms = [c.decode() if isinstance(c, bytes) else c
              for c in scores["coords_chrom"]]
    assert chroms == [chrom for chrom, _ in peaks]
    assert scores["coords_start"].tolist() == [s - flank for _, s in peaks]
    assert scores["coords_end"].tolist() == [s + flank for _, s in peaks]

    # the sequences of the peaks, one hot, and a score for each of the
    # four bases at every position
    assert scores["input_seqs"].dtype == np.int8
    assert scores["hyp_scores"].dtype == np.float16
    assert scores["hyp_scores"].shape == scores["input_seqs"].shape == \
        (len(peaks), d["input_len"], 4)
    for i, (chrom, summit) in enumerate(peaks):
        expected = one_hot(d["sequences"][chrom][summit - flank:
                                                 summit + flank])
        assert np.array_equal(scores["input_seqs"][i], expected)
    assert np.isfinite(scores["hyp_scores"].astype(np.float64)).all()
    assert np.abs(scores["hyp_scores"].astype(np.float64)).max() > 0


def test_counts_scores_add_up_to_the_difference_from_the_references(
        without_control):
    """
        DeepSHAP: the scores of the bases that are there add up to the
        output for the sequence minus the mean output for its
        references. The output that is explained for the counts is the
        sum of the predicted log counts of the tracks; the references
        are dinucleotide shuffles of the sequence. The outputs are
        computed here with the numpy model from the saved weights.
    """

    d = without_control["dataset"]
    scores = without_control["scores"]["counts"]
    weights = weights_by_layer_name(without_control["model"])

    def total_logcounts(sequences):
        _, logcounts = reference_model.forward(
            weights, d["params"], sequences.astype(np.float64))
        return logcounts.reshape(len(sequences), -1).sum(axis=1)

    found, expected, bound = [], [], []
    for i in range(len(scores["input_seqs"])):
        sequence = scores["input_seqs"][i].astype(np.float64)
        references = dinuc_shuffle(scores["input_seqs"][i], SHUFFLES,
                                   np.random.RandomState(SEED))
        expected.append(total_logcounts(sequence[None])[0] -
                        total_logcounts(np.asarray(references)).mean())
        held = scores["hyp_scores"][i].astype(np.float64) * sequence
        found.append(held.sum())
        bound.append(within_rounding(held, sequence))
    found, expected, bound = map(np.array, (found, expected, bound))
    # the differences are not all alike, and the scores follow them
    assert expected.std() > 100 * bound.max()
    assert (np.abs(found - expected) <= bound).all(), \
        np.abs(found - expected) / bound


def within_rounding(held, sequence):
    """
        How far the sum of the stored scores of the bases that are there
        may be from the exact sum: the float16 rounding of every score,
        and the float32 arithmetic of TensorFlow
    """

    return float16_bound(held[sequence > 0]).sum() + \
        1e-5 * np.abs(held).sum() + 1e-7


def test_profile_scores_add_up_to_the_share_of_the_logits(without_control):
    """
        What is explained for the profile is the sum over all positions
        of both strands of w * l, with l the logits minus their mean
        and w their softmax, where the weights w are not explained
        (stop_gradient). DeepSHAP divides the change of a product
        between its two factors; the scores are the share of the
        logits, 0.5 * (w(sequence) + w(reference)) * (l(sequence) -
        l(reference)) summed over the positions and averaged over the
        references. (The other share, that of the weights, is dropped:
        the scores do not add up to the change of the output.)
    """

    d = without_control["dataset"]
    scores = without_control["scores"]["profile"]
    weights = weights_by_layer_name(without_control["model"])

    def meannormed_logits(sequences):
        logits, _ = reference_model.forward(
            weights, d["params"], sequences.astype(np.float64))
        # the positions of both strands of a sequence in one row
        logits = logits.reshape(len(sequences), -1)
        return logits - logits.mean(axis=1, keepdims=True)

    def softmax(values):
        values = np.exp(values - values.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)

    found, expected, whole, bound = [], [], [], []
    for i in range(len(scores["input_seqs"])):
        sequence = scores["input_seqs"][i].astype(np.float64)
        references = np.asarray(dinuc_shuffle(
            scores["input_seqs"][i], SHUFFLES, np.random.RandomState(SEED)))
        of_sequence = meannormed_logits(sequence[None])
        of_references = meannormed_logits(references)
        w_sequence, w_references = softmax(of_sequence), softmax(of_references)
        expected.append((0.5 * (w_sequence + w_references) *
                         (of_sequence - of_references)).sum(axis=1).mean())
        whole.append((w_sequence * of_sequence).sum() -
                     (w_references * of_references).sum(axis=1).mean())
        held = scores["hyp_scores"][i].astype(np.float64) * sequence
        found.append(held.sum())
        bound.append(within_rounding(held, sequence))
    found, expected, whole, bound = map(np.array,
                                        (found, expected, whole, bound))
    assert expected.std() > 100 * bound.max()
    assert (np.abs(found - expected) <= bound).all(), \
        np.abs(found - expected) / bound
    # and not to the change of the whole output, which is about twice
    # as large
    assert np.abs(found - whole).max() > 0.5 * expected.std()


@pytest.mark.parametrize("which", ["counts", "profile"])
@pytest.mark.parametrize("model", ["with control", "without control"])
def test_hypothetical_scores_of_every_base(with_control, without_control,
                                           model, which):
    """
        The scores of all four bases at every position, also of the
        three that are not in the sequence (TF-MoDISco reads them), are
        those of DeepLIFT with the rescale rule as computed here
        independently (tests/reference_deeplift.py), to the float16
        rounding of the files. Six peaks of each model are checked.
        The independent computation is written in PyTorch: the test is
        skipped where PyTorch is not installed. (The references are
        made with the shuffling of the package: what is checked is the
        attribution given the references, not the shuffling.)
    """

    torch = pytest.importorskip("torch")
    from tests import reference_deeplift

    run = with_control if model == "with control" else without_control
    d = run["dataset"]
    scores = run["scores"][which]
    weights = weights_by_layer_name(run["model"])
    for i in range(6):
        sequence = scores["input_seqs"][i].astype(np.float64)
        references = np.asarray(dinuc_shuffle(
            scores["input_seqs"][i], SHUFFLES,
            np.random.RandomState(SEED))).astype(np.float64)
        extra = {}
        if model == "with control":
            profile_bias, counts_bias = control_inputs(d, scores, i)
            extra = {"profile_bias": profile_bias,
                     "counts_bias": counts_bias}
        ours = reference_deeplift.hypothetical_scores(
            weights, d["params"], which, sequence, references, **extra)
        stored = scores["hyp_scores"][i].astype(np.float64)
        # (and the float32 arithmetic of TensorFlow: up to 1.4e-6 of the
        # largest score was seen for the profile with control inputs)
        allowed = float16_bound(ours) + 1e-5 * np.abs(ours).max()
        assert (np.abs(stored - ours) <= allowed).all(), \
            (i, (np.abs(stored - ours) / allowed).max())
        # the bases that are not there have scores of their own
        assert np.abs(ours[sequence == 0]).max() > 0.1 * np.abs(ours).max()


def run_shap_cli(d, model, out_dir, input_data=None):
    return run_cli("bpnet.cli.shap_scores", [
        "--reference-genome", d["fasta"], "--input-seq-len", d["input_len"],
        "--control-len", d["output_len"], "--model", model,
        "--bed-file", d["peaks"], "--chroms", d["chroms"][-1],
        "--input-data", input_data or d["input_json"],
        "--output-directory", out_dir], cwd=str(out_dir))


def test_model_that_cannot_be_loaded_is_reported(synth_dataset, tmp_path):
    missing = tmp_path / "no_such_model"
    code, output = run_shap_cli(synth_dataset, missing, tmp_path)
    assert code != 0
    assert "File {} does not exist".format(missing) in output
    assert "Traceback" not in output

    # a directory that is not a saved model
    code, output = run_shap_cli(synth_dataset, tmp_path, tmp_path)
    assert code != 0
    assert "Cannot load the model" in output
    assert "it has no saved_model.pb" in output
    assert "Traceback" not in output

    # a damaged file
    damaged = tmp_path / "damaged"
    damaged.mkdir()
    with open(str(damaged / "saved_model.pb"), "wb") as fh:
        fh.write(b"\x12\xff\xff\xff")
    code, output = run_shap_cli(synth_dataset, damaged, tmp_path)
    assert code != 0
    assert "Cannot load the model {}: Truncated".format(damaged) in output
    assert "Traceback" not in output


def test_missing_input_data_file_is_reported(synth_dataset, with_control,
                                             tmp_path):
    missing = tmp_path / "no_such_file.json"
    code, output = run_shap_cli(synth_dataset, with_control["model"], tmp_path,
                                input_data=missing)
    assert code != 0
    assert "Input data file {} does not exist".format(missing) in output
    assert "Traceback" not in output


def assert_same_files(found_dir, expected_dir):
    """
        The two output directories of bpnet-shap hold the same files with
        the same content: the score files array by array, the bigWigs,
        statistics and peaks byte for byte, the settings but for the
        output directory (the log has the times of the run)
    """

    names = sorted(os.listdir(expected_dir))
    assert sorted(os.listdir(found_dir)) == names
    for name in names:
        found, expected = (os.path.join(found_dir, name),
                           os.path.join(expected_dir, name))
        if name.endswith(".log"):
            continue
        if name == "config.json":
            with open(found) as a, open(expected) as b:
                a, b = json.load(a), json.load(b)
            a.pop("output_directory")
            b.pop("output_directory")
            assert a == b
            continue
        if name.endswith(".h5"):
            continue
        with open(found, "rb") as a, open(expected, "rb") as b:
            assert a.read() == b.read(), name


def assert_same_scores(found, expected):
    for which in ("counts", "profile"):
        assert sorted(found[which]) == sorted(expected[which])
        for key, values in expected[which].items():
            other = found[which][key]
            if values.dtype == object:
                # (the chromosome names)
                assert values.tolist() == other.tolist()
            else:
                assert values.dtype == other.dtype
                assert values.tobytes() == other.tobytes(), (which, key)


def test_scores_are_the_same_for_the_same_seed(with_control, tmp_path):
    d = with_control["dataset"]
    again = run_shap(d, with_control["model"], tmp_path)
    assert_same_scores(again["scores"], with_control["scores"])


@pytest.mark.skipif(
    tuple(int(part) for part in tf.__version__.split(".")[:2]) >= (2, 5),
    reason="keras' loader reads these models back under TensorFlow 2.4 only")
@pytest.mark.parametrize("model", ["with control", "without control"])
def test_scores_are_those_of_the_model_loaded_by_keras(
        with_control, without_control, tmp_path, model):
    """
        The tool loads the model with the package's loader. The scores
        are, byte for byte, the scores of the model as keras' loader
        returns it, which is how the tool loaded it before.
    """

    run = with_control if model == "with control" else without_control
    before = run_shap(run["dataset"], run["model"], tmp_path,
                      code=SHAP_WITH_THE_KERAS_LOADER, extra=run["extra"])
    assert_same_scores(run["scores"], before["scores"])
    assert_same_files(run["dir"], before["dir"])


def test_scores_depend_on_the_references(with_control, tmp_path):
    d = with_control["dataset"]
    other = run_shap(d, with_control["model"], tmp_path, seed=SEED + 1)
    for which in ("counts", "profile"):
        a = with_control["scores"][which]
        b = other["scores"][which]
        assert np.array_equal(a["input_seqs"], b["input_seqs"])
        assert not np.array_equal(a["hyp_scores"], b["hyp_scores"])
        # other shuffles of the same sequences: related scores
        x = (a["hyp_scores"].astype(np.float64) * a["input_seqs"]).sum(2)
        y = (b["hyp_scores"].astype(np.float64) * b["input_seqs"]).sum(2)
        assert np.corrcoef(x.ravel(), y.ravel())[0, 1] > 0.3


@pytest.mark.parametrize("which", ["counts", "profile"])
def test_bigwig_holds_the_scores_of_the_bases_that_are_there(with_control,
                                                             which):
    d = with_control["dataset"]
    scores = with_control["scores"][which]
    projected = (scores["hyp_scores"] * scores["input_seqs"]).sum(axis=2)
    bw = pyBigWig.open(os.path.join(with_control["dir"],
                                    which + "_scores.bw"))
    try:
        # a peak whose window overlaps no other is in the file as it is
        starts = scores["coords_start"]
        alone = [i for i in range(len(starts))
                 if all(abs(int(starts[i]) - int(s)) >= d["input_len"]
                        for j, s in enumerate(starts) if j != i)]
        assert len(alone) > 3
        for i in alone[:5]:
            chrom = scores["coords_chrom"][i]
            chrom = chrom.decode() if isinstance(chrom, bytes) else chrom
            values = np.array(bw.values(chrom, int(starts[i]),
                                        int(scores["coords_end"][i])),
                              dtype=np.float32)
            assert np.array_equal(values, projected[i].astype(np.float32))
    finally:
        bw.close()
