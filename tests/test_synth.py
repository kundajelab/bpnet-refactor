"""
    Tests for the synthetic dataset builder itself, so that the other
    tests can rely on the data it writes
"""

import hashlib
import json

import numpy as np
import pyBigWig
import pyfaidx

from tests import synth


def read_dense(path, chrom, length):
    """
        All values of one chromosome of a bigWig, NaN replaced by 0
    """

    bw = pyBigWig.open(path)
    values = np.nan_to_num(np.array(bw.values(chrom, 0, length)))
    bw.close()
    return values


def test_fasta_matches_truth(synth_dataset):
    fasta = pyfaidx.Fasta(synth_dataset["fasta"])
    assert list(fasta.keys()) == synth_dataset["chroms"]
    for chrom in synth_dataset["chroms"]:
        truth = synth_dataset["sequences"][chrom]
        assert len(fasta[chrom]) == synth_dataset["chrom_len"]
        assert fasta[chrom][:].seq == truth
        # an interval that crosses line boundaries exercises the index
        assert fasta[chrom][55:187].seq == truth[55:187]
    fasta.close()


def test_chrom_sizes(synth_dataset):
    with open(synth_dataset["chrom_sizes"]) as fh:
        rows = [line.split() for line in fh]
    assert [r[0] for r in rows] == synth_dataset["chroms"]
    assert all(int(r[1]) == synth_dataset["chrom_len"] for r in rows)


def test_bigwigs_match_truth(synth_dataset):
    paths = synth_dataset["signal"] + synth_dataset["control"]
    assert len(paths) == 4
    for path in paths:
        for chrom in synth_dataset["chroms"]:
            values = read_dense(path, chrom, synth_dataset["chrom_len"])
            truth = synth_dataset["tracks"][path][chrom]
            assert np.array_equal(values, truth)
            assert truth.sum() > 0


def test_peaks_carry_the_motif_and_signal(synth_dataset):
    plus = synth_dataset["tracks"][synth_dataset["signal"][0]]
    minus = synth_dataset["tracks"][synth_dataset["signal"][1]]
    for chrom, start, end, signal, summit in synth_dataset["peak_rows"]:
        pos = start + summit
        seq = synth_dataset["sequences"][chrom]
        assert seq[pos:pos + len(synth.MOTIF)] == synth.MOTIF
        assert start < pos < end
        # reads pile up upstream on the plus strand, downstream on minus
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
    assert len(synth_dataset["peak_rows"]) == 240
    assert len(synth_dataset["background_rows"]) == 240
    positions = sorted((c, s + m) for c, s, e, _, m in rows)
    assert len(set(positions)) == len(positions)
    flank = synth_dataset["input_len"] // 2
    max_jitter = flank
    for chrom, pos in positions:
        assert pos - flank - max_jitter >= 0
        assert pos + flank + max_jitter <= synth_dataset["chrom_len"]
    # signal windows of neighbouring loci do not overlap
    for (c1, p1), (c2, p2) in zip(positions[:-1], positions[1:]):
        assert c1 != c2 or p2 - p1 > 121


def test_bed_files_are_narrowpeak(synth_dataset):
    for key, rows in (("peaks", synth_dataset["peak_rows"]),
                      ("background", synth_dataset["background_rows"])):
        with open(synth_dataset[key]) as fh:
            lines = [line.rstrip("\n").split("\t") for line in fh]
        assert len(lines) == len(rows)
        assert all(len(line) == 10 for line in lines)
        for line, (chrom, start, end, _, summit) in zip(lines, rows):
            assert line[0] == chrom
            assert (int(line[1]), int(line[2]), int(line[9])) == \
                (start, end, summit)


def test_json_inputs(synth_dataset):
    with open(synth_dataset["input_json"]) as fh:
        tasks = json.load(fh)
    assert list(tasks.keys()) == ["0"]
    assert tasks["0"]["signal"]["source"] == synth_dataset["signal"]
    assert tasks["0"]["bias"]["source"] == synth_dataset["control"]
    assert tasks["0"]["bias"]["smoothing"] == [None, None]
    assert tasks["0"]["loci"]["source"] == [synth_dataset["peaks"]]
    assert tasks["0"]["background_loci"]["ratio"] == [0.5]

    with open(synth_dataset["params_json"]) as fh:
        params = json.load(fh)
    # receptive field: first conv, dilated convs, profile head
    shrink = 20 + 2 * (2 + 4) + 74
    assert params["input_len"] - params["output_profile_len"] == shrink

    with open(synth_dataset["splits_json"]) as fh:
        splits = json.load(fh)["0"]
    chroms = splits["train"] + splits["val"] + splits["test"]
    assert sorted(chroms) == sorted(synth_dataset["chroms"])


def digest(dataset):
    h = hashlib.sha256()
    for path in sorted(dataset["tracks"]):
        for chrom in dataset["chroms"]:
            h.update(dataset["tracks"][path][chrom].tobytes())
    for chrom in dataset["chroms"]:
        h.update(dataset["sequences"][chrom].encode())
    h.update(repr(dataset["peak_rows"]).encode())
    h.update(repr(dataset["background_rows"]).encode())
    return h.hexdigest()


def test_builder_is_deterministic(tmp_path, synth_dataset):
    small = dict(chrom_len=20000, n_peaks=30, n_background=30)
    a = synth.make_dataset(str(tmp_path / "a"), seed=3, **small)
    b = synth.make_dataset(str(tmp_path / "b"), seed=3, **small)
    c = synth.make_dataset(str(tmp_path / "c"), seed=4, **small)
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)
    with open(a["fasta"], "rb") as fa, open(b["fasta"], "rb") as fb:
        assert fa.read() == fb.read()


def test_variants(tmp_path):
    small = dict(chrom_len=20000, n_peaks=30, n_background=30)

    unstranded = synth.make_dataset(
        str(tmp_path / "unstranded"), stranded=False, **small)
    assert len(unstranded["signal"]) == 1
    assert len(unstranded["control"]) == 1

    no_control = synth.make_dataset(
        str(tmp_path / "no_control"), with_control=False, **small)
    assert no_control["control"] == []
    assert no_control["input_data"]["0"]["bias"] == \
        {"source": [], "smoothing": []}

    two_tasks = synth.make_dataset(
        str(tmp_path / "two_tasks"), num_tasks=2, **small)
    assert len(two_tasks["signal"]) == 4
    assert list(two_tasks["input_data"].keys()) == ["0", "1"]
    assert two_tasks["params"]["profile_bias_module_params"] == \
        {"kernel_sizes": [1, 1]}
    # the second task has more reads in the same peaks
    chrom, start, end, _, summit = two_tasks["peak_rows"][0]
    window = slice(start + summit - 60, start + summit + 61)
    t0 = sum(two_tasks["tracks"][p][chrom][window].sum()
             for p in two_tasks["signal"][:2])
    t1 = sum(two_tasks["tracks"][p][chrom][window].sum()
             for p in two_tasks["signal"][2:])
    assert t1 > t0
