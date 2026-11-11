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


def read_like_the_command(tmp_path):
    """
        The names as pandas reads them from a peak file, with the
        call of bpnet-shap
    """

    path = str(tmp_path / "peaks.bed")
    with open(path, "w") as fh:
        for i, name in enumerate(NAMES):
            fh.write("\t".join([name, str(i * 100), str(i * 100 + 40), ".",
                                "0", ".", "1.0", "-1", "-1", "20"]) + "\n")
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["chrom", "st", "stop", "name", "score", "strand", "signal",
               "p", "q", "summit"])["chrom"]


#: the types a chromosome column has: Python text in an object column
#: (pandas before 3), pandas' own text type (the default of pandas 3),
#: categories, and whatever the installed pandas makes of a peak file
COLUMNS = {
    "object": lambda tmp_path: pd.Series(NAMES, dtype=object),
    "pandas text": lambda tmp_path: pd.Series(NAMES, dtype="string"),
    "categories": lambda tmp_path: pd.Series(NAMES, dtype="category"),
    "read from a peak file": read_like_the_command,
}


@pytest.mark.parametrize("kind", sorted(COLUMNS))
def test_names_in_the_types_a_chromosome_column_has(tmp_path, kind):
    table = peaks(NAMES)
    table["chrom"] = COLUMNS[kind](tmp_path)
    if kind == "object":
        # (the type with which the names were cut, in any pandas)
        assert table["chrom"].values.dtype == object
    assert list(table["chrom"]) == NAMES
    sequences, scores = arrays(len(NAMES))
    path = str(tmp_path / "scores.h5")
    shap_scores.save_scores(table, sequences, scores, path)
    assert stored_names(path) == NAMES


def test_files_for_short_names_are_what_they_were(tmp_path):
    """
        For names of up to 8 characters the file holds what the line
        that cut the names wrote: the same names in the same type
    """

    names = ["c" * length for length in range(1, 9)] + ["chr1", "chrX"]
    table = peaks(names)
    table["chrom"] = pd.Series(names, dtype=object)
    sequences, scores = arrays(len(names))
    path = str(tmp_path / "scores.h5")
    shap_scores.save_scores(table, sequences, scores, path)
    before = str(tmp_path / "before.h5")
    with h5py.File(before, "w") as f:
        dset = f.create_dataset(
            "coords_chrom", (len(names),),
            dtype=h5py.string_dtype(encoding="ascii"))
        dset[:] = table["chrom"].values.astype("U8")
    with h5py.File(path, "r") as now, h5py.File(before, "r") as was:
        assert now["coords_chrom"].dtype == was["coords_chrom"].dtype
        assert h5py.check_string_dtype(now["coords_chrom"].dtype) == \
            h5py.check_string_dtype(was["coords_chrom"].dtype)
        found, expected = now["coords_chrom"][:], was["coords_chrom"][:]
        assert found.dtype == expected.dtype
        assert [type(name) for name in found] == \
            [type(name) for name in expected]
        assert list(found) == list(expected)


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
