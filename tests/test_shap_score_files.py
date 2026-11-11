"""
    The score files of bpnet-shap (bpnet.cli.shap_scores.save_scores):
    what a file holds for given peaks, sequences and scores, without
    a model and without TensorFlow. The chromosome names in
    particular: they are stored whole, whatever their length.
"""

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")
pytest.importorskip("pysam")
pytest.importorskip("pyBigWig")

from bpnet.cli import shap_scores

#: names of 1 to 22 characters; two that share their first 8, and
#: one that is the first 8 characters of another
NAMES = ["chr1", "2", "chrUn_KI270302v1", "chrUn_KI270303v1",
         "chr1_KI270706v1_random", "chr1_KI2", "NC_000001.11",
         "CHROMOSOME_I", "chr1"]


def peaks(names):
    return pd.DataFrame({
        "chrom": names,
        "start": np.arange(len(names)) * 100 + 7,
        "end": np.arange(len(names)) * 100 + 27})


def arrays(n, length=20, seed=0):
    rng = np.random.RandomState(seed)
    sequences = np.zeros((n, length, 4))
    sequences[np.arange(n)[:, None], np.arange(length)[None],
              rng.randint(0, 4, size=(n, length))] = 1
    return sequences, rng.normal(size=(n, length, 4))


def stored_names(path):
    with h5py.File(path, "r") as f:
        return [name.decode("ascii") if isinstance(name, bytes) else name
                for name in f["coords_chrom"][:]]


def test_chromosome_names_are_stored_whole(tmp_path):
    """
        Every peak has the name of its chromosome in the file, also a
        name of more than 8 characters (the file used to hold the
        first 8, which made 'chrUn_KI270302v1' and 'chrUn_KI270303v1'
        one chromosome)
    """

    sequences, scores = arrays(len(NAMES))
    path = str(tmp_path / "scores.h5")
    shap_scores.save_scores(peaks(NAMES), sequences, scores, path)
    assert stored_names(path) == NAMES
    with h5py.File(path, "r") as f:
        # text of any length, in ASCII, as before
        kind = h5py.check_string_dtype(f["coords_chrom"].dtype)
        assert kind is not None and kind.length is None
        assert kind.encoding == "ascii"
        assert f["coords_chrom"].shape == (len(NAMES),)


def test_what_a_score_file_holds(tmp_path):
    sequences, scores = arrays(len(NAMES), seed=1)
    path = str(tmp_path / "scores.h5")
    shap_scores.save_scores(peaks(NAMES), sequences, scores, path)
    with h5py.File(path, "r") as f:
        assert sorted(f) == ["coords_chrom", "coords_end", "coords_start",
                             "hyp_scores", "input_seqs"]
        assert f["coords_start"].dtype == np.int32
        assert f["coords_end"].dtype == np.int32
        assert np.array_equal(f["coords_start"][:],
                              np.arange(len(NAMES)) * 100 + 7)
        assert np.array_equal(f["coords_end"][:],
                              np.arange(len(NAMES)) * 100 + 27)
        assert f["hyp_scores"].dtype == np.float16
        assert np.array_equal(f["hyp_scores"][:], scores.astype(np.float16))
        assert f["input_seqs"].dtype == np.int8
        assert np.array_equal(f["input_seqs"][:], sequences)


@pytest.mark.parametrize("kind", ["object", "numpy text", "pandas text",
                                  "categories"])
def test_names_in_the_types_a_table_of_peaks_can_have(tmp_path, kind):
    """
        The chromosome column as pandas reads it from a peak file
        (Python text, or pandas' own text type), as a numpy array of
        text, and as categories
    """

    table = peaks(NAMES)
    if kind == "numpy text":
        table["chrom"] = np.array(NAMES)
    elif kind == "pandas text":
        table["chrom"] = table["chrom"].astype("string")
    elif kind == "categories":
        table["chrom"] = table["chrom"].astype("category")
    sequences, scores = arrays(len(NAMES))
    path = str(tmp_path / "scores.h5")
    shap_scores.save_scores(table, sequences, scores, path)
    assert stored_names(path) == NAMES


def test_one_peak_and_one_long_name(tmp_path):
    name = "scaffold_" + "0123456789" * 6
    sequences, scores = arrays(1)
    path = str(tmp_path / "scores.h5")
    shap_scores.save_scores(peaks([name]), sequences, scores, path)
    assert stored_names(path) == [name]


def test_name_that_is_not_ascii_is_refused(tmp_path):
    """
        The names are stored as ASCII; a name that is not ASCII stops
        the tool instead of being stored as something else
    """

    sequences, scores = arrays(2)
    with pytest.raises(UnicodeEncodeError):
        shap_scores.save_scores(peaks(["chr1", "chr" + chr(228)]), sequences,
                                scores, str(tmp_path / "scores.h5"))
