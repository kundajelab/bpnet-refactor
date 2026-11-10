"""
    Tests for the synthetic dataset builder itself, so that the other
    tests can rely on the data it writes
"""

import hashlib
import json
import os

import numpy as np
import pyBigWig
import pyfaidx
import pytest

from tests import synth

# sha256 over the arrays, sequences and loci of make_dataset(seed=0)
# with default arguments. The builder draws everything from a legacy
# numpy RandomState, whose streams are stable across numpy versions,
# so this value must not depend on the environment. Golden test data
# derived from the builder rely on it.
DEFAULT_DATASET_DIGEST = \
    "01c213b48529a4c36e965414e3ca1aa803aaf0aa13fa0ed0b46a54223c15b158"

SMALL = dict(chrom_len=20000, n_peaks=30, n_background=30)


def read_raw(path, chrom, length):
    """
        All values of one chromosome of a bigWig, as stored (NaN where
        nothing was written)
    """

    bw = pyBigWig.open(path)
    values = np.array(bw.values(chrom, 0, length))
    bw.close()
    return values


def digest(dataset):
    h = hashlib.sha256()
    for path in sorted(dataset["tracks"], key=os.path.basename):
        for chrom in dataset["chroms"]:
            h.update(dataset["tracks"][path][chrom].tobytes())
    for chrom in dataset["chroms"]:
        h.update(dataset["sequences"][chrom].encode())
    h.update(repr(dataset["peak_rows"]).encode())
    h.update(repr(dataset["background_rows"]).encode())
    return h.hexdigest()


def assert_files_match_truth(dataset):
    """
        Every file of a dataset agrees with the arrays the builder
        returned
    """

    # FASTA, read through the index the builder wrote (pyfaidx must not
    # build its own)
    index_path = dataset["fasta"] + ".fai"
    assert os.path.exists(index_path)
    with open(index_path, "rb") as fh:
        index_bytes = fh.read()
    fasta = pyfaidx.Fasta(dataset["fasta"], build_index=False, rebuild=False)
    assert list(fasta.keys()) == dataset["chroms"]
    for chrom in dataset["chroms"]:
        truth = dataset["sequences"][chrom]
        assert len(truth) == dataset["chrom_len"]
        assert fasta[chrom][:].seq == truth
        # intervals that cross line boundaries and reach the last line
        assert fasta[chrom][55:187].seq == truth[55:187]
        assert fasta[chrom][len(truth) - 71:].seq == truth[-71:]
    fasta.close()
    with open(index_path, "rb") as fh:
        assert fh.read() == index_bytes

    with open(dataset["chrom_sizes"]) as fh:
        rows = [line.split() for line in fh]
    assert [r[0] for r in rows] == dataset["chroms"]
    assert all(int(r[1]) == dataset["chrom_len"] for r in rows)

    # bigWigs: stored values equal the truth, zeros are not stored
    assert sorted(dataset["tracks"]) == \
        sorted(dataset["signal"] + dataset["control"])
    for path in dataset["signal"] + dataset["control"]:
        for chrom in dataset["chroms"]:
            raw = read_raw(path, chrom, dataset["chrom_len"])
            truth = dataset["tracks"][path][chrom]
            assert raw.shape == truth.shape
            assert np.array_equal(np.isnan(raw), truth == 0)
            assert np.array_equal(np.nan_to_num(raw), truth)
        assert sum(t.sum() for t in dataset["tracks"][path].values()) > 0

    # loci
    for key, rows in (("peaks", dataset["peak_rows"]),
                      ("background", dataset["background_rows"])):
        with open(dataset[key]) as fh:
            lines = [line.rstrip("\n").split("\t") for line in fh]
        assert len(lines) == len(rows)
        assert all(len(line) == 10 for line in lines)
        for line, (chrom, start, end, _, summit) in zip(lines, rows):
            assert line[0] == chrom
            assert (int(line[1]), int(line[2]), int(line[9])) == \
                (start, end, summit)

    # json files agree with the returned dictionaries and the file lists
    with open(dataset["input_json"]) as fh:
        tasks = json.load(fh)
    assert tasks == dataset["input_data"]
    per_task = len(dataset["signal"]) // len(tasks)
    for i, key in enumerate(sorted(tasks, key=int)):
        task = tasks[key]
        assert task["signal"]["source"] == \
            dataset["signal"][i * per_task:(i + 1) * per_task]
        if dataset["control"]:
            assert task["bias"]["source"] == \
                dataset["control"][i * per_task:(i + 1) * per_task]
        else:
            assert task["bias"]["source"] == []
        assert task["bias"]["smoothing"] == \
            [None] * len(task["bias"]["source"])
        assert task["loci"]["source"] == [dataset["peaks"]]
        assert task["background_loci"]["source"] == [dataset["background"]]
    with open(dataset["params_json"]) as fh:
        assert json.load(fh) == dataset["params"]
    with open(dataset["splits_json"]) as fh:
        assert json.load(fh) == dataset["splits"]


def test_default_dataset_files(synth_dataset):
    assert_files_match_truth(synth_dataset)
    assert len(synth_dataset["signal"]) == 2
    assert len(synth_dataset["control"]) == 2
    assert len(synth_dataset["peak_rows"]) == 240
    assert len(synth_dataset["background_rows"]) == 240


def test_default_dataset_is_pinned(synth_dataset):
    assert digest(synth_dataset) == DEFAULT_DATASET_DIGEST


def test_builder_is_deterministic(tmp_path):
    a = synth.make_dataset(str(tmp_path / "a"), seed=3, **SMALL)
    b = synth.make_dataset(str(tmp_path / "b"), seed=3, **SMALL)
    c = synth.make_dataset(str(tmp_path / "c"), seed=4, **SMALL)
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)
    with open(a["fasta"], "rb") as fa, open(b["fasta"], "rb") as fb:
        assert fa.read() == fb.read()


def test_peaks_carry_the_motif_and_signal(synth_dataset):
    plus = synth_dataset["tracks"][synth_dataset["signal"][0]]
    minus = synth_dataset["tracks"][synth_dataset["signal"][1]]
    for chrom, start, end, signal, summit in synth_dataset["peak_rows"]:
        pos = start + summit
        seq = synth_dataset["sequences"][chrom]
        assert seq[pos:pos + len(synth.MOTIF)] == synth.MOTIF
        assert start < pos < end
        # with this seed: reads pile up upstream on the plus strand and
        # downstream on the minus strand
        assert plus[chrom][pos - 60:pos].sum() > plus[chrom][pos:pos + 60].sum()
        assert minus[chrom][pos:pos + 60].sum() > minus[chrom][pos - 60:pos].sum()


def test_background_has_little_signal(synth_dataset):
    plus = synth_dataset["tracks"][synth_dataset["signal"][0]]
    peak_counts = [plus[c][s + m - 60:s + m + 61].sum()
                   for c, s, e, _, m in synth_dataset["peak_rows"]]
    background_counts = [plus[c][s + m - 60:s + m + 61].sum()
                         for c, s, e, _, m in synth_dataset["background_rows"]]
    assert np.max(background_counts) < np.min(peak_counts)


def test_loci_are_distinct_and_leave_room_for_jitter(synth_dataset):
    rows = synth_dataset["peak_rows"] + synth_dataset["background_rows"]
    positions = sorted((c, s + m) for c, s, e, _, m in rows)
    assert len(set(positions)) == len(positions)
    flank = synth_dataset["input_len"] // 2
    max_jitter = flank
    for chrom, pos in positions:
        assert pos - flank - max_jitter >= 0
        assert pos + flank + max_jitter <= synth_dataset["chrom_len"]
    # signal windows of neighbouring loci do not overlap
    for (c1, p1), (c2, p2) in zip(positions[:-1], positions[1:]):
        assert c1 != c2 or p2 - p1 >= synth.SIGNAL_WIDTH


def test_densest_allowed_layout_keeps_signal_windows_apart(tmp_path):
    # 3 chromosomes x 141 slots of exactly SIGNAL_WIDTH + 10 bases
    n_loci = 3 * 141
    chrom_len = 141 * (synth.SIGNAL_WIDTH + 10) + 2 * 306
    dataset = synth.make_dataset(
        str(tmp_path), chrom_len=chrom_len, n_peaks=n_loci // 2,
        n_background=n_loci - n_loci // 2)
    rows = dataset["peak_rows"] + dataset["background_rows"]
    positions = sorted((c, s + m) for c, s, e, _, m in rows)
    gaps = [p2 - p1 for (c1, p1), (c2, p2)
            in zip(positions[:-1], positions[1:]) if c1 == c2]
    assert min(gaps) >= synth.SIGNAL_WIDTH

    with pytest.raises(ValueError):
        synth.make_dataset(str(tmp_path / "too_dense"),
                           chrom_len=chrom_len - 141, n_peaks=n_loci // 2,
                           n_background=n_loci - n_loci // 2)


def test_json_inputs(synth_dataset):
    tasks = synth_dataset["input_data"]
    assert list(tasks.keys()) == ["0"]
    assert tasks["0"]["bias"]["smoothing"] == [None, None]
    assert tasks["0"]["background_loci"]["ratio"] == [0.5]
    params = synth_dataset["params"]
    assert params["input_len"] - params["output_profile_len"] == \
        synth.SHRINKAGE
    splits = synth_dataset["splits"]["0"]
    chroms = splits["train"] + splits["val"] + splits["test"]
    assert sorted(chroms) == sorted(synth_dataset["chroms"])
    assert all(len(splits[k]) > 0 for k in ("train", "val", "test"))


def test_unstranded_variant(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), stranded=False, **SMALL)
    assert_files_match_truth(dataset)
    assert len(dataset["signal"]) == 1
    assert len(dataset["control"]) == 1


def test_no_control_variant(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), with_control=False, **SMALL)
    assert_files_match_truth(dataset)
    assert dataset["control"] == []
    assert dataset["input_data"]["0"]["bias"] == \
        {"source": [], "smoothing": []}


def test_two_task_variant(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), num_tasks=2, **SMALL)
    assert_files_match_truth(dataset)
    assert len(dataset["signal"]) == 4
    assert list(dataset["input_data"].keys()) == ["0", "1"]
    assert dataset["params"]["profile_bias_module_params"] == \
        {"kernel_sizes": [1, 1]}
    # with this seed the second task has more reads in the first peak
    chrom, start, end, _, summit = dataset["peak_rows"][0]
    window = slice(start + summit - 60, start + summit + 61)
    t0 = sum(dataset["tracks"][p][chrom][window].sum()
             for p in dataset["signal"][:2])
    t1 = sum(dataset["tracks"][p][chrom][window].sum()
             for p in dataset["signal"][2:])
    assert t1 > t0


def test_chromosome_length_not_a_multiple_of_the_fasta_line(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), chrom_len=20003,
                                 n_peaks=30, n_background=30)
    assert_files_match_truth(dataset)


def test_invalid_arguments(tmp_path):
    with pytest.raises(ValueError):
        synth.make_dataset(str(tmp_path / "a"), chroms=("chr1", "chr2"))
    with pytest.raises(ValueError):
        synth.make_dataset(str(tmp_path / "b"), n_peaks=0)
    with pytest.raises(ValueError):
        synth.make_dataset(str(tmp_path / "c"), input_len=306,
                           output_len=400)
    # nothing is written for a rejected request
    assert not any((tmp_path / d).exists() for d in "abc")


def test_all_zero_bigwig_can_be_read(tmp_path):
    path = str(tmp_path / "zeros.bw")
    sizes = {"chr1": 50, "chr2": 30}
    synth.write_bigwig(path, sizes, {c: np.zeros(n) for c, n in sizes.items()})
    for chrom, n in sizes.items():
        values = read_raw(path, chrom, n)
        assert values.shape == (n,)
        assert not np.nan_to_num(values).any()
