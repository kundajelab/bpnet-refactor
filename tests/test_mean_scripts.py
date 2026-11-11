"""
    The two scripts that average the outputs of the folds of an
    experiment: bpnet/utils/mean_shap.py (the score files of
    bpnet-shap) and bpnet/utils/mean_predictions.py (the prediction
    files of bpnet-predict, with bigWigs of the mean). Both are run as
    the pipelines run them, as scripts with a command line, on files
    written here; no model and no TensorFlow are needed.

    The scripts stop with a message rather than write a wrong mean:
    when the folds are not of the same peaks or sequences, and when
    no fold can be read. One thing they do today is open to question
    and marked as such in its tests: a fold that is missing or cannot
    be read is left out of the mean with a message (left out whole:
    nothing of it is counted), and the mean of the others is written.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pyBigWig = pytest.importorskip("pyBigWig")
pytest.importorskip("pysam")

from bpnet.cli import shap_scores

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = "chrUn_KI270302v1"


def run_script(name, arguments, cwd):
    done = subprocess.run(
        [sys.executable, os.path.join(REPO, "bpnet", "utils", name)] +
        [str(a) for a in arguments], cwd=str(cwd),
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=600)
    return done.returncode, done.stdout


#: a script run in a Python in which opening one file (the first
#: argument) fails for want of memory
NO_MEMORY_FOR_ONE_FILE = """
import runpy, sys
import h5py
real = h5py.File
too_large, script = sys.argv[1:3]
def failing(path, *args, **kwargs):
    if str(path) == too_large:
        raise MemoryError("no memory for " + str(path))
    return real(path, *args, **kwargs)
h5py.File = failing
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
"""


def run_without_memory_for(path, name, arguments, cwd):
    done = subprocess.run(
        [sys.executable, "-c", NO_MEMORY_FOR_ONE_FILE, str(path),
         os.path.join(REPO, "bpnet", "utils", name)] +
        [str(a) for a in arguments], cwd=str(cwd),
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=600)
    return done.returncode, done.stdout


def text(names):
    return [name.decode() if isinstance(name, bytes) else str(name)
            for name in names]


# --------------------------------------------------------------------
# mean_shap.py
# --------------------------------------------------------------------

#: the peaks of the score files: (chromosome, start); one chromosome
#: has a name of more than 8 characters
PEAKS = [("chr1", 100), ("chr1", 5000), ("chr2", 40), (LONG, 700),
         (LONG, 90), ("chrX", 12)]
LENGTH = 30


def score_files(directory, folds=3, seed=0, peaks=PEAKS, eighths=False):
    """
        The score files of `folds` folds for the same peaks, as
        bpnet-shap writes them, and what went into them (`eighths`:
        the scores are multiples of 1/8 up to 4, whose sums and means
        over a few folds float16 holds exactly)

        Returns:
            tuple: (paths, one hot sequences, the scores of every
                fold as the files hold them (float16))
    """

    rng = np.random.RandomState(seed)
    n = len(peaks)
    table = pd.DataFrame({
        "chrom": pd.Series([chrom for chrom, _ in peaks], dtype=object),
        "start": [start for _, start in peaks],
        "end": [start + LENGTH for _, start in peaks]})
    sequences = np.zeros((n, LENGTH, 4))
    sequences[np.arange(n)[:, None], np.arange(LENGTH)[None],
              rng.randint(0, 4, size=(n, LENGTH))] = 1
    paths, stored = [], []
    for fold in range(folds):
        scores = rng.normal(0, 0.05, size=(n, LENGTH, 4))
        if eighths:
            scores = rng.randint(-32, 33, size=(n, LENGTH, 4)) / 8.0
        path = os.path.join(str(directory), "fold{}.h5".format(fold))
        shap_scores.save_scores(table, sequences, scores, path)
        paths.append(path)
        stored.append(scores.astype(np.float16))
    return paths, sequences, stored


def read_scores(path):
    with h5py.File(path, "r") as f:
        return {key: f[key][:] for key in f}, {
            key: f[key].dtype for key in f}


#: the number HDF5 has for the Blosc filter
BLOSC = 32001


def filters(path):
    """
        The filters of every dataset of a file, by their numbers
    """

    found = {}
    with h5py.File(path, "r") as f:
        for key in f:
            made_with = f[key].id.get_create_plist()
            found[key] = [made_with.get_filter(i)[0]
                          for i in range(made_with.get_nfilters())]
    return found


def float16_mean_bound(stored):
    """
        How far the mean a script computes in float16 can be from the
        mean of the stored scores: one rounding per addition and one
        for the division, each of half a unit in the last place of a
        value no larger than the sum of the magnitudes
    """

    magnitudes = np.sum(np.abs(np.asarray(stored, dtype=np.float64)), axis=0)
    return len(stored) * 2.0 ** -11 * magnitudes + 1e-7


@pytest.fixture(scope="module")
def mean_shap(tmp_path_factory):
    work = tmp_path_factory.mktemp("mean_shap")
    for kind in ("counts", "profile"):
        os.mkdir(str(work / kind))
    counts = score_files(work / "counts", seed=1)
    profile = score_files(work / "profile", seed=2)
    out = work / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(counts[0]),
        "--profile_shaps", ",".join(profile[0]), "--output_dir", out], work)
    return {"code": code, "output": output, "out": out, "counts": counts,
            "profile": profile, "work": work}


def test_mean_scores_are_written(mean_shap):
    """
        The script ends well and writes one file per kind of score
        (it used to end in a floating point exception while writing
        the chromosome names: the Blosc filter cannot compress text of
        variable length)
    """

    assert mean_shap["code"] == 0, mean_shap["output"][-3000:]
    assert sorted(os.listdir(str(mean_shap["out"]))) == [
        "counts_mean_shap_scores.h5", "profile_mean_shap_scores.h5"]


@pytest.mark.parametrize("kind", ["counts", "profile"])
def test_mean_score_file(mean_shap, kind):
    """
        The peaks and sequences of the folds, and for every base the
        mean of its scores over the folds
    """

    paths, sequences, stored = mean_shap[kind]
    found, types = read_scores(str(
        mean_shap["out"] / (kind + "_mean_shap_scores.h5")))
    assert sorted(found) == ["coords_chrom", "coords_end", "coords_start",
                             "hyp_scores", "input_seqs"]
    # the names whole (one has 16 characters), as text of any length
    assert text(found["coords_chrom"]) == [chrom for chrom, _ in PEAKS]
    kind_of_text = h5py.check_string_dtype(types["coords_chrom"])
    assert kind_of_text is not None and kind_of_text.length is None
    assert found["coords_start"].tolist() == [start for _, start in PEAKS]
    assert found["coords_end"].tolist() == [start + LENGTH
                                            for _, start in PEAKS]
    assert np.array_equal(found["input_seqs"], sequences)
    assert found["hyp_scores"].shape == (len(PEAKS), LENGTH, 4)
    expected = np.mean(np.asarray(stored, dtype=np.float64), axis=0)
    difference = np.abs(found["hyp_scores"].astype(np.float64) - expected)
    assert (difference <= float16_mean_bound(stored)).all()
    # (the mean, not the scores of one fold)
    for fold in stored:
        assert np.abs(fold - expected).max() > 0.01
    # the types of the released mean score files
    assert types["hyp_scores"] == np.float32
    assert types["input_seqs"] == np.float32
    assert types["coords_start"] == types["coords_end"] == np.int64
    # compressed as the released files are, except the names (text of
    # variable length cannot go through that filter)
    assert filters(str(mean_shap["out"] / (
        kind + "_mean_shap_scores.h5"))) == {
            "coords_chrom": [], "coords_start": [BLOSC],
            "coords_end": [BLOSC], "hyp_scores": [BLOSC],
            "input_seqs": [BLOSC]}


def test_mean_is_exact_where_the_arithmetic_is(tmp_path):
    """
        Scores that are multiples of 1/8: their sum over four folds
        and its quarter are exact in the float16 the folds are stored
        in, so the mean in the file is the mean, to the last bit
    """

    paths, _, stored = score_files(tmp_path, folds=4, seed=3, eighths=True)
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 0, output[-3000:]
    found, _ = read_scores(str(out / "counts_mean_shap_scores.h5"))
    stored = np.asarray(stored, dtype=np.float64)
    assert np.array_equal(stored * 8, np.round(stored * 8))
    expected = stored.mean(axis=0)
    assert np.array_equal(found["hyp_scores"], expected)
    assert len(np.unique(expected)) > 100


def test_mean_of_one_kind_of_score(tmp_path):
    paths, _, stored = score_files(tmp_path, folds=2)
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--profile_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 0, output[-3000:]
    assert os.listdir(str(out)) == ["profile_mean_shap_scores.h5"]
    found, _ = read_scores(str(out / "profile_mean_shap_scores.h5"))
    expected = np.mean(np.asarray(stored, dtype=np.float64), axis=0)
    assert (np.abs(found["hyp_scores"] - expected) <=
            float16_mean_bound(stored)).all()


def test_score_that_is_missing_in_a_fold_is_left_out_of_the_mean(tmp_path):
    """
        A score that is not a number in one fold: the mean of the
        other folds
    """

    paths, _, stored = score_files(tmp_path, folds=3)
    with h5py.File(paths[1], "r+") as f:
        f["hyp_scores"][2, 5, 1] = np.nan
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 0, output[-3000:]
    found, _ = read_scores(str(out / "counts_mean_shap_scores.h5"))
    two = [stored[0], stored[2]]
    expected = np.mean(np.asarray(two, dtype=np.float64), axis=0)[2, 5, 1]
    assert abs(found["hyp_scores"][2, 5, 1] - expected) <= \
        float16_mean_bound(two)[2, 5, 1]
    assert np.isfinite(found["hyp_scores"]).all()


def test_fold_that_is_not_there_is_left_out_today(tmp_path):
    """
        What the script does today (open to question): a file that is
        not there is named in a message and left out, and the mean of
        the other folds is written without an error
    """

    paths, _, stored = score_files(tmp_path, folds=3)
    missing = str(tmp_path / "fold_that_is_not_there.h5")
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths[:2] + [missing]),
        "--output_dir", out], tmp_path)
    assert code == 0, output[-3000:]
    assert missing + " does not exist" in output
    found, _ = read_scores(str(out / "counts_mean_shap_scores.h5"))
    expected = np.mean(np.asarray(stored[:2], dtype=np.float64), axis=0)
    assert (np.abs(found["hyp_scores"] - expected) <=
            float16_mean_bound(stored[:2])).all()


@pytest.mark.parametrize("other", ["chromosomes", "starts", "ends",
                                   "sequences", "numbers of scores",
                                   "fewer peaks first"])
def test_folds_that_do_not_agree_are_an_error(tmp_path, other):
    """
        Folds that are not of the same peaks and the same sequences
        (or not of as many peaks): the script names the two files and
        what differs, writes nothing and fails. (It used to write
        nothing and end as if all were well.)
    """

    paths, sequences, _ = score_files(tmp_path, folds=3)
    differs = paths[2]
    if other in ("numbers of scores", "fewer peaks first"):
        # a fold of fewer peaks (the first peaks of the others), as
        # the last fold or as the first
        os.mkdir(str(tmp_path / "fewer"))
        fewer, _, _ = score_files(tmp_path / "fewer", folds=1,
                                  peaks=PEAKS[:-1])
        if other == "fewer peaks first":
            paths[0], differs = fewer[0], paths[1]
        else:
            paths[2] = differs = fewer[0]
        other = "chromosomes"
    else:
        with h5py.File(paths[2], "r+") as f:
            if other == "chromosomes":
                f["coords_chrom"][3] = "chr2"
            elif other == "sequences":
                f["input_seqs"][1, 4] = np.roll(sequences[1, 4], 1)
            else:
                f["coords_" + other[:-1]][3] += 1
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 1, output[-3000:]
    assert "{} and {} do not have the same {}: {} is not written".format(
        differs, paths[0], other,
        str(out) + "/counts_mean_shap_scores.h5") in output
    assert "Traceback" not in output
    assert os.listdir(str(out)) == []


def test_difference_in_the_last_of_many_peaks_is_found(tmp_path):
    """
        The folds are compared a block of rows at a time: a
        difference in the last of 4,100 peaks is found like one in
        the first
    """

    peaks = [("chr1", 40 * i) for i in range(4100)]
    paths, sequences, stored = score_files(tmp_path, folds=2, peaks=peaks)
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 0, output[-3000:]
    found, _ = read_scores(str(out / "counts_mean_shap_scores.h5"))
    assert found["hyp_scores"].shape == (4100, LENGTH, 4)
    os.remove(str(out / "counts_mean_shap_scores.h5"))
    with h5py.File(paths[1], "r+") as f:
        f["input_seqs"][4099, 7] = np.roll(sequences[4099, 7], 1)
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 1, output[-3000:]
    assert "do not have the same sequences" in output
    assert os.listdir(str(out)) == []


def test_no_memory_for_a_fold_stops_the_script(tmp_path):
    """
        A fold that cannot be read for want of memory is not a fold
        that does not exist: the script fails instead of writing the
        mean of the others
    """

    paths, _, _ = score_files(tmp_path, folds=3)
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_without_memory_for(paths[1], "mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code != 0, output[-3000:]
    assert "MemoryError: no memory for " + paths[1] in output
    assert os.listdir(str(out)) == []
    # (the same call with memory for all three)
    code, output = run_without_memory_for("none", "mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 0 and os.listdir(str(out)) == [
        "counts_mean_shap_scores.h5"], output[-3000:]


def test_no_score_files_asked_for_is_an_error(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", ["--output_dir", out],
                              tmp_path)
    assert code == 2, output[-3000:]
    assert "no score files: give --counts_shaps, --profile_shaps or both" \
        in output
    assert os.listdir(str(out)) == []


def test_folds_of_other_sizes_are_an_error(tmp_path):
    """
        A fold with the peaks of the others and scores of another
        shape
    """

    paths, _, _ = score_files(tmp_path, folds=2)
    with h5py.File(paths[1], "r+") as f:
        scores = f["hyp_scores"][:, :-1]
        del f["hyp_scores"]
        f.create_dataset("hyp_scores", data=scores)
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 1, output[-3000:]
    assert "do not have the same numbers of scores" in output
    assert "Traceback" not in output and os.listdir(str(out)) == []


@pytest.mark.parametrize("missing", ["hyp_scores", "coords_chrom",
                                     "coords_start", "coords_end",
                                     "input_seqs"])
def test_fold_that_is_read_in_part_is_left_out_whole(tmp_path, missing):
    """
        A fold file without one of its datasets cannot be read: it is
        named in a message and nothing of it is in the mean. (Its
        scores used to be counted when the dataset that is missing
        came after them: folds of 1, 100 and 3 gave 34.7 with the
        message that the fold of 100 "does not exist".)
    """

    paths, sequences, stored = score_files(tmp_path, folds=3)
    with h5py.File(paths[1], "r+") as f:
        f["hyp_scores"][...] = (
            f["hyp_scores"][:].astype(np.float64) + 100).astype(np.float16)
        del f[missing]
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", ",".join(paths), "--output_dir", out], tmp_path)
    assert code == 0, output[-3000:]
    assert paths[1] + " does not exist or cannot be read (KeyError" in output
    found, _ = read_scores(str(out / "counts_mean_shap_scores.h5"))
    two = [stored[0], stored[2]]
    expected = np.mean(np.asarray(two, dtype=np.float64), axis=0)
    assert (np.abs(found["hyp_scores"] - expected) <=
            float16_mean_bound(two)).all()
    assert np.array_equal(found["input_seqs"], sequences)
    assert text(found["coords_chrom"]) == [chrom for chrom, _ in PEAKS]


def test_no_fold_that_can_be_read_is_an_error(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    absent = [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")]
    with open(absent[1], "w") as fh:
        fh.write("not a score file")
    code, output = run_script("mean_shap.py", [
        "--profile_shaps", ",".join(absent), "--output_dir", out], tmp_path)
    assert code == 1, output[-3000:]
    for path in absent:
        assert path + " does not exist or cannot be read" in output
    assert "None of the score files could be read" in output
    assert "Traceback" not in output and os.listdir(str(out)) == []


def test_mean_score_file_is_read_like_a_score_file(mean_shap):
    """
        A mean score file has the layout of a score file: what reads
        one reads the other (here: the script itself, for the mean of
        two means)
    """

    out = mean_shap["work"] / "again"
    out.mkdir()
    first = str(mean_shap["out"] / "counts_mean_shap_scores.h5")
    code, output = run_script("mean_shap.py", [
        "--counts_shaps", first + "," + first, "--output_dir", out],
        mean_shap["work"])
    assert code == 0, output[-3000:]
    once, _ = read_scores(first)
    twice, _ = read_scores(str(out / "counts_mean_shap_scores.h5"))
    assert text(twice["coords_chrom"]) == text(once["coords_chrom"])
    assert np.array_equal(twice["hyp_scores"], once["hyp_scores"])
    assert np.array_equal(twice["input_seqs"], once["input_seqs"])


# --------------------------------------------------------------------
# mean_predictions.py
# --------------------------------------------------------------------

#: the regions of the prediction files: (chromosome, start); two of
#: them overlap, two chromosomes have names of more than 8 characters,
#: and the first 8 characters of one are the name of another
PREFIX = LONG[:8]
REGIONS = [("chr1", 100), ("chr1", 130), ("chr1", 400), (LONG, 20),
           (PREFIX, 200), ("chr2", 60), ("chr10_random_0001", 10)]
WIDTH = 50
SIZES = [("chr1", 1000), ("chr2", 500), ("chr10_random_0001", 400),
         (PREFIX, 600), (LONG, 300)]


def prediction_files(directory, folds=2, seed=0, regions=REGIONS):
    """
        Prediction files of `folds` folds in the layout of
        bpnet-predict (groups 'coords' and 'predictions'), each with
        the regions in an order of its own, and what went into them

        Returns:
            tuple: (paths, the profiles of every fold (fold, region,
                position, strand) in the order of `regions`)
    """

    rng = np.random.RandomState(seed)
    n = len(regions)
    paths, profiles = [], []
    for fold in range(folds):
        profile = rng.gamma(2.0, 3.0, size=(n, WIDTH, 2)) + 1.0
        order = rng.permutation(n)
        path = os.path.join(str(directory), "predictions{}.h5".format(fold))
        with h5py.File(path, "w") as f:
            coords = f.create_group("coords")
            names = coords.create_dataset(
                "coords_chrom", (n,),
                dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip")
            names[:] = [regions[i][0] for i in order]
            coords.create_dataset(
                "coords_start", data=[regions[i][1] for i in order],
                dtype=int, compression="gzip")
            coords.create_dataset(
                "coords_end", data=[regions[i][1] + WIDTH for i in order],
                dtype=int, compression="gzip")
            predictions = f.create_group("predictions")
            predictions.create_dataset("pred_profs", data=profile[order],
                                       compression="gzip")
            predictions.create_dataset(
                "pred_logcounts", data=np.log(profile[order].sum(axis=1)),
                compression="gzip")
        paths.append(path)
        profiles.append(profile)
    return paths, np.asarray(profiles)


def expected_mean_profile(profiles):
    """
        The mean of the folds as the script defines it: the geometric
        mean of the shapes (each region's profile over both strands
        as probabilities), made a probability again, times the
        geometric mean of the totals
    """

    totals = profiles.sum(axis=(2, 3), keepdims=True)
    shape = np.exp(np.mean(np.log(profiles / totals), axis=0))
    shape = shape / shape.sum(axis=(1, 2), keepdims=True)
    return shape * np.exp(np.mean(np.log(totals), axis=0))


@pytest.fixture(scope="module")
def mean_predictions(tmp_path_factory):
    work = tmp_path_factory.mktemp("mean_predictions")
    paths, profiles = prediction_files(work)
    sizes = str(work / "chrom.sizes")
    with open(sizes, "w") as fh:
        for name, size in SIZES:
            fh.write("{}\t{}\n".format(name, size))
    out = work / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "--generate_bigwigs",
        "-c", sizes, "-o", out], work)
    return {"code": code, "output": output, "out": out, "work": work,
            "paths": paths, "profiles": profiles, "sizes": sizes}


def test_mean_predictions_are_written(mean_predictions):
    assert mean_predictions["code"] == 0, mean_predictions["output"][-3000:]
    assert sorted(os.listdir(str(mean_predictions["out"]))) == [
        "mean_predictions.h5", "mean_predictions_minus.bw",
        "mean_predictions_plus.bw"]


def regions_of_the_mean_file(out):
    with h5py.File(str(out / "mean_predictions.h5"), "r") as f:
        found = {key: f[key][:] for key in f}
    return found, list(zip(text(found["coords_chrom"]),
                           found["coords_start"].tolist()))


def test_mean_prediction_file(mean_predictions):
    """
        The regions of the folds (found whatever their order in each
        file), with the whole chromosome names, and for each the mean
        profile and the log of its total per strand
    """

    found, regions = regions_of_the_mean_file(mean_predictions["out"])
    assert sorted(found) == ["coords_chrom", "coords_end", "coords_start",
                             "pred_logcounts", "pred_prof"]
    assert sorted(regions) == sorted(REGIONS)
    assert found["coords_end"].tolist() == [start + WIDTH
                                            for _, start in regions]
    expected = expected_mean_profile(mean_predictions["profiles"])
    rows = [REGIONS.index(region) for region in regions]
    np.testing.assert_allclose(found["pred_prof"], expected[rows],
                               rtol=1e-5)
    np.testing.assert_allclose(found["pred_logcounts"],
                               np.log(expected[rows].sum(axis=1)), rtol=1e-5)
    # (the mean, not one fold)
    for fold in mean_predictions["profiles"]:
        assert np.abs(fold[rows] - expected[rows]).max() > 0.5
    # the types the script has always written: the names as text of
    # any length in ASCII
    with h5py.File(str(mean_predictions["out"] / "mean_predictions.h5"),
                   "r") as f:
        kind_of_text = h5py.check_string_dtype(f["coords_chrom"].dtype)
    assert kind_of_text is not None and kind_of_text.length is None
    assert kind_of_text.encoding == "ascii"
    assert found["pred_prof"].dtype == np.float32
    assert found["pred_logcounts"].dtype == np.float32
    assert found["coords_start"].dtype == np.int64
    assert found["coords_end"].dtype == np.int64


def expected_tracks(regions, profile, sizes, width=WIDTH):
    """
        What the bigWigs of the mean hold, by chromosome. Every base
        of a region has the value of that region's profile; a base
        that several regions cover has the value of one of them: of
        the regions of a chromosome in the order of their starts, the
        last one whose border with the region before it is at or
        before the base. That border is the middle between the two
        centres, rounded down, a centre being (start + end) // 2.
    """

    tracks = {name: np.full(size, np.nan) for name, size in sizes}
    by_chromosome = {}
    for row, (name, start) in enumerate(regions):
        by_chromosome.setdefault(name, []).append((start, row))

    def centre(start):
        return (start + start + width) // 2

    for name, starts in by_chromosome.items():
        starts.sort()
        for position in range(len(tracks[name])):
            owner = None
            for k, (start, row) in enumerate(starts):
                border = start if k == 0 else max(start, (
                    centre(starts[k - 1][0]) + centre(start)) // 2)
                if border <= position < start + width:
                    owner = (start, row)
            if owner is not None:
                tracks[name][position] = profile[owner[1],
                                                 position - owner[0]]
    return tracks


@pytest.mark.parametrize("strand", ["plus", "minus"])
def test_bigwigs_of_the_mean_predictions(mean_predictions, strand):
    """
        One bigWig per strand with the mean profile of every region
        on its own chromosome, also for the chromosomes whose names
        have more than 8 characters (the script cut the names to 8
        characters here: the regions of 'chrUn_KI270302v1' were
        written to 'chrUn_KI', and a name whose first 8 characters
        are no chromosome ended the script)
    """

    assert mean_predictions["code"] == 0, mean_predictions["output"][-3000:]
    found, regions = regions_of_the_mean_file(mean_predictions["out"])
    column = 0 if strand == "plus" else 1
    expected = expected_tracks(regions, found["pred_prof"][:, :, column],
                               SIZES)
    bw = pyBigWig.open(str(mean_predictions["out"] / (
        "mean_predictions_{}.bw".format(strand))))
    try:
        assert dict(bw.chroms()) == dict(SIZES)
        for name, size in SIZES:
            values = np.array(bw.values(name, 0, size), dtype=np.float64)
            there = ~np.isnan(expected[name])
            assert np.array_equal(~np.isnan(values), there), name
            np.testing.assert_allclose(values[there], expected[name][there],
                                       rtol=1e-6, err_msg=name)
            assert there.sum() > 0
    finally:
        bw.close()
    # the two overlapping regions of chr1 meet in the middle between
    # their centres
    assert not np.isnan(expected["chr1"][100:180]).any()
    assert np.isnan(expected["chr1"][180:400]).all()


def test_region_is_written_to_its_own_chromosome(tmp_path):
    """
        Two chromosomes, the name of one being the first 8 characters
        of the other's: each region is in the bigWig on its own
        chromosome (with the names cut to 8 characters the script
        ended well and had written both regions to the shorter name)
    """

    regions = [(LONG, 20), (PREFIX, 200)]
    sizes = [(PREFIX, 600), (LONG, 300)]
    paths, _ = prediction_files(tmp_path, seed=3, regions=regions)
    with open(str(tmp_path / "chrom.sizes"), "w") as fh:
        for name, size in sizes:
            fh.write("{}\t{}\n".format(name, size))
    out = tmp_path / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "--generate_bigwigs",
        "-c", tmp_path / "chrom.sizes", "-o", out], tmp_path)
    assert code == 0, output[-3000:]
    found, stored = regions_of_the_mean_file(out)
    assert sorted(stored) == sorted(regions)
    bw = pyBigWig.open(str(out / "mean_predictions_plus.bw"))
    try:
        for name, size in sizes:
            start = dict(regions)[name]
            values = np.array(bw.values(name, 0, size), dtype=np.float64)
            covered = np.zeros(size, dtype=bool)
            covered[start:start + WIDTH] = True
            assert np.array_equal(~np.isnan(values), covered), name
            np.testing.assert_allclose(
                values[covered],
                found["pred_prof"][stored.index((name, start)), :, 0],
                rtol=1e-6, err_msg=name)
    finally:
        bw.close()


@pytest.mark.parametrize("width", [10, 9])
def test_regions_that_overlap_each_other(tmp_path, monkeypatch, width):
    """
        Three regions of one chromosome that all overlap, of an even
        and of an odd width: each base of the bigWig has the value of
        one region, the regions taking turns at the middle between
        their centres (rounded down)
    """

    regions = [("chr1", 106), ("chr1", 100), ("chr1", 103), ("chr2", 7)]
    sizes = [("chr1", 200), ("chr2", 100)]
    monkeypatch.setattr(sys.modules[__name__], "WIDTH", width)
    paths, _ = prediction_files(tmp_path, seed=4, regions=regions)
    with open(str(tmp_path / "chrom.sizes"), "w") as fh:
        for name, size in sizes:
            fh.write("{}\t{}\n".format(name, size))
    out = tmp_path / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "--generate_bigwigs",
        "-c", tmp_path / "chrom.sizes", "-o", out], tmp_path)
    assert code == 0, output[-3000:]
    found, stored = regions_of_the_mean_file(out)
    assert found["pred_prof"].shape == (4, width, 2)
    # the borders, said outright: the middle between the centres of
    # the regions at 100 and 103, and of those at 103 and 106
    first = ((200 + width) // 2 + (206 + width) // 2) // 2
    second = ((206 + width) // 2 + (212 + width) // 2) // 2
    assert (first, second) == ((106, 109) if width == 10 else (105, 108))
    rows = {start: stored.index(("chr1", start)) for start in (100, 103, 106)}
    for column, strand in enumerate(("plus", "minus")):
        profile = found["pred_prof"][:, :, column]
        expected = expected_tracks(stored, profile, sizes, width=width)
        by_hand = np.concatenate([
            profile[rows[100], :first - 100],
            profile[rows[103], first - 103:second - 103],
            profile[rows[106], second - 106:]])
        assert np.array_equal(expected["chr1"][100:106 + width], by_hand)
        assert (~np.isnan(expected["chr1"])).sum() == 6 + width
        bw = pyBigWig.open(str(out / "mean_predictions_{}.bw".format(strand)))
        try:
            for name, size in sizes:
                values = np.array(bw.values(name, 0, size), dtype=np.float64)
                there = ~np.isnan(expected[name])
                assert np.array_equal(~np.isnan(values), there), name
                np.testing.assert_allclose(
                    values[there], expected[name][there], rtol=1e-6,
                    err_msg=name)
        finally:
            bw.close()


@pytest.mark.parametrize("other", ["chromosomes", "starts", "ends",
                                   "numbers of predictions"])
def test_prediction_folds_that_do_not_agree_are_an_error(tmp_path, other):
    """
        Folds that are not of the same regions: the script names the
        two files and what differs, writes no mean and no bigWigs, and
        fails. (It used to write nothing and end as if all were well;
        and with the mean file of an earlier run in the folder it
        wrote bigWigs of that one.)
    """

    paths, _ = prediction_files(tmp_path, folds=3, seed=7)
    with h5py.File(paths[2], "r+") as f:
        if other == "chromosomes":
            row = text(f["coords/coords_chrom"][:]).index("chr2")
            f["coords/coords_chrom"][row] = "chr1"
        elif other == "numbers of predictions":
            profiles = f["predictions/pred_profs"][:, :-1]
            del f["predictions/pred_profs"]
            f["predictions"].create_dataset("pred_profs", data=profiles)
        else:
            f["coords/coords_" + other[:-1]][2] += 1
    with open(str(tmp_path / "chrom.sizes"), "w") as fh:
        for name, size in SIZES:
            fh.write("{}\t{}\n".format(name, size))
    out = tmp_path / "out"
    # the mean file of an earlier run
    out.mkdir()
    os.mkdir(str(tmp_path / "earlier"))
    earlier, _ = prediction_files(tmp_path / "earlier", folds=2, seed=8)
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(earlier), "-o", out], tmp_path)
    assert code == 0 and os.listdir(str(out)) == ["mean_predictions.h5"]
    with open(str(out / "mean_predictions.h5"), "rb") as fh:
        before = fh.read()

    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "--generate_bigwigs",
        "-c", tmp_path / "chrom.sizes", "-o", out], tmp_path)
    assert code == 1, output[-3000:]
    assert "{} and {} do not have the same {}: {} is not written".format(
        paths[2], paths[0], other,
        str(out) + "/mean_predictions.h5") in output
    assert "Traceback" not in output
    # no bigWigs of the earlier mean, which is as it was
    assert os.listdir(str(out)) == ["mean_predictions.h5"]
    with open(str(out / "mean_predictions.h5"), "rb") as fh:
        assert fh.read() == before


@pytest.mark.parametrize("missing", [
    "predictions/pred_profs", "predictions/pred_logcounts",
    "coords/coords_chrom", "coords/coords_end", "predictions"])
def test_prediction_fold_read_in_part_is_left_out_whole(tmp_path, missing):
    """
        A prediction file without one of its datasets is named in a
        message and nothing of it is in the mean (parts of it used to
        be kept, whichever came before the one that is missing)
    """

    paths, profiles = prediction_files(tmp_path, folds=3, seed=9)
    with h5py.File(paths[1], "r+") as f:
        del f[missing]
    out = tmp_path / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "-o", out], tmp_path)
    assert code == 0, output[-3000:]
    assert paths[1] + " has problems or does not exist (KeyError" in output
    found, regions = regions_of_the_mean_file(out)
    rows = [REGIONS.index(region) for region in regions]
    np.testing.assert_allclose(
        found["pred_prof"],
        expected_mean_profile(profiles[[0, 2]])[rows], rtol=1e-5)


def test_no_prediction_file_that_can_be_read_is_an_error(tmp_path):
    out = tmp_path / "out"
    absent = [str(tmp_path / "a.h5"), str(tmp_path / "b.h5")]
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(absent), "--generate_bigwigs",
        "-c", tmp_path / "no.sizes", "-o", out], tmp_path)
    assert code == 1, output[-3000:]
    assert "None of the prediction files could be read" in output
    assert "Traceback" not in output and os.listdir(str(out)) == []


def test_no_memory_for_a_prediction_file_stops_the_script(tmp_path):
    paths, _ = prediction_files(tmp_path, folds=3, seed=11)
    out = tmp_path / "out"
    code, output = run_without_memory_for(
        paths[2], "mean_predictions.py",
        ["--prediction_h5s", ",".join(paths), "-o", out], tmp_path)
    assert code != 0, output[-3000:]
    assert "MemoryError: no memory for " + paths[2] in output
    assert os.listdir(str(out)) == []


def test_bigwigs_without_chromosome_sizes_are_an_error(tmp_path):
    """
        bigWigs asked for without the sizes of the chromosomes: said
        before anything is written (the mean used to be written, the
        bigWigs left out with a message, and the script ended well)
    """

    paths, _ = prediction_files(tmp_path, folds=2, seed=12)
    out = tmp_path / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "--generate_bigwigs",
        "-o", out], tmp_path)
    assert code == 2, output[-3000:]
    assert "--generate_bigwigs needs the chromosome sizes" in output
    assert not os.path.exists(str(out))


def test_counts_below_one_are_counts(tmp_path):
    """
        A fold in which a region has a total below one (a log count
        below zero) is said to have such values and is used like any
        other, as it always was (the message used to say that the
        file "has problems or does not exist")
    """

    paths, profiles = prediction_files(tmp_path, folds=2, seed=10)
    with h5py.File(paths[1], "r+") as f:
        row = text(f["coords/coords_chrom"][:]).index("chr2")
        small = f["predictions/pred_profs"][row] * 1e-4
        f["predictions/pred_profs"][row] = small
        f["predictions/pred_logcounts"][row] = np.log(small.sum(axis=0))
        assert (f["predictions/pred_logcounts"][row] < 0).all()
    profiles[1, REGIONS.index(("chr2", 60))] *= 1e-4
    out = tmp_path / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "-o", out], tmp_path)
    assert code == 0, output[-3000:]
    assert paths[1] + " has profile values or log counts that are not " \
        "above zero" in output
    assert "does not exist" not in output
    found, regions = regions_of_the_mean_file(out)
    rows = [REGIONS.index(region) for region in regions]
    np.testing.assert_allclose(
        found["pred_prof"], expected_mean_profile(profiles)[rows], rtol=1e-5)


def test_mean_predictions_without_bigwigs(tmp_path):
    paths, profiles = prediction_files(tmp_path, folds=3, seed=5)
    out = tmp_path / "out"
    out.mkdir()
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths), "-o", out], tmp_path)
    assert code == 0, output[-3000:]
    assert os.listdir(str(out)) == ["mean_predictions.h5"]
    found, regions = regions_of_the_mean_file(out)
    rows = [REGIONS.index(region) for region in regions]
    np.testing.assert_allclose(
        found["pred_prof"], expected_mean_profile(profiles)[rows], rtol=1e-5)


def test_prediction_file_that_is_not_there_is_left_out_today(tmp_path):
    """
        What the script does today (open to question): a file that is
        not there is named in a message and left out of the mean
    """

    paths, profiles = prediction_files(tmp_path, folds=2, seed=6)
    missing = str(tmp_path / "not_there.h5")
    out = tmp_path / "out"
    code, output = run_script("mean_predictions.py", [
        "--prediction_h5s", ",".join(paths + [missing]), "-o", out],
        tmp_path)
    assert code == 0, output[-3000:]
    assert missing + " has problems or does not exist" in output
    found, regions = regions_of_the_mean_file(out)
    rows = [REGIONS.index(region) for region in regions]
    np.testing.assert_allclose(
        found["pred_prof"], expected_mean_profile(profiles)[rows], rtol=1e-5)
