"""
    Characterization tests for sequtils.getPeakPositions: which loci
    are kept, where their windows are, how the background is sampled
    and which weights the samples get
"""

import numpy as np
import pandas as pd
import pytest

from bpnet.generators import sequtils
from bpnet.utils.exceptionhandler import NoTracebackException

FLANK = 100

CHROM_SIZES = pd.DataFrame(
    {"chrom": ["chr1", "chr2", "chr3"], "size": [1000, 500, 800]})

# (chrom, start, end, summit); the position of a locus is start + summit
PEAKS = [
    ("chr1", 150, 250, 50),   # 0: pos 200, window [100, 300]
    ("chr1", 40, 140, 30),    # 1: pos 70, window starts before 0
    ("chr1", 880, 980, 20),   # 2: pos 900, window ends at the chrom end
    ("chr1", 881, 981, 20),   # 3: pos 901, window ends past the chrom end
    ("chr2", 90, 190, 10),    # 4: pos 100, window starts at 0
    ("chr2", 200, 300, 50),   # 5: pos 250
    ("chr3", 300, 400, 50),   # 6: pos 350
    ("chr1", 400, 500, 50),   # 7: pos 450
]

BACKGROUND = [
    ("chr1", 500, 600, 50),   # 0: pos 550
    ("chr1", 600, 700, 50),   # 1: pos 650
    ("chr2", 250, 350, 50),   # 2: pos 300
    ("chr3", 100, 200, 50),   # 3: pos 150
    ("chr3", 500, 600, 50),   # 4: pos 550
    ("chr1", 700, 800, 50),   # 5: pos 750
]

# the peaks whose window fits in the chromosome, in the order
# getPeakPositions returns them: sorted by chromosome, then window end
VALID_PEAKS = [("chr1", 200), ("chr1", 450), ("chr1", 900),
               ("chr2", 100), ("chr2", 250), ("chr3", 350)]


def write_bed(path, rows):
    with open(str(path), "w") as fh:
        for chrom, start, end, summit in rows:
            fh.write("\t".join(map(str, [
                chrom, start, end, ".", 1, ".", 10.0, -1, -1, summit])) + "\n")
    return str(path)


@pytest.fixture
def tasks(tmp_path):
    return {
        0: {
            "signal": {"source": ["UNUSED/plus.bw", "UNUSED/minus.bw"]},
            "loci": {"source": [write_bed(tmp_path / "peaks.bed", PEAKS)]},
            "background_loci": {
                "source": [write_bed(tmp_path / "background.bed",
                                     BACKGROUND)],
                "ratio": [0.5]},
            "bias": {"source": [], "smoothing": []},
        }
    }


def loci(df):
    return list(zip(df["chrom"], df["pos"]))


def peaks_only(tasks, **kwargs):
    return sequtils.getPeakPositions(
        tasks, CHROM_SIZES, FLANK, loci_keys=["loci"], **kwargs)


def test_columns_positions_and_windows(tasks):
    df = peaks_only(tasks)
    assert list(df.columns) == \
        ["chrom", "start_coord", "end_coord", "pos", "weight"]
    assert list(df.index) == list(range(len(df)))
    # windows that leave the chromosome are dropped, windows that touch
    # its ends are kept; rows are sorted by chromosome and window end
    assert loci(df) == VALID_PEAKS
    assert (df["start_coord"] == df["pos"] - FLANK).all()
    assert (df["end_coord"] == df["pos"] + FLANK).all()
    assert df["start_coord"].min() == 0
    assert (df["weight"] == 1).all()


def test_chromosome_filter(tasks):
    df = peaks_only(tasks, chroms=["chr1", "chr3"])
    assert loci(df) == [p for p in VALID_PEAKS if p[0] != "chr2"]
    assert len(peaks_only(tasks, chroms=["chr4"])) == 0


def test_loci_indices_select_rows_of_the_file(tasks):
    # indices refer to rows of the bed file, before any filtering
    df = peaks_only(tasks, loci_indices=[0, 2, 5])
    assert loci(df) == [("chr1", 200), ("chr1", 900), ("chr2", 250)]
    # row 1 is selected but its window does not fit
    df = peaks_only(tasks, loci_indices=[1, 4])
    assert loci(df) == [("chr2", 100)]


def test_chromosomes_take_precedence_over_indices(tasks):
    df = peaks_only(tasks, chroms=["chr2"], loci_indices=[0])
    assert loci(df) == [("chr2", 100), ("chr2", 250)]


def test_background_is_sampled_in_proportion_to_the_peaks(tasks):
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
    peaks = df[df["weight"] == 1]
    background = df[df["weight"] == 0]
    assert loci(peaks) == VALID_PEAKS
    # ratio 0.5 of the 6 peaks that were kept
    assert len(background) == 3
    pool = {(c, s + m) for c, s, e, m in BACKGROUND}
    assert set(loci(background)) <= pool
    assert len(set(loci(background))) == 3
    # peaks first, then the background
    assert df["weight"].tolist() == [1] * 6 + [0] * 3


def test_val_background_follows_random_state_1(tasks):
    """
        The validation background must be the same in every run: pandas
        sample(n, random_state=1) on the background loci that fit, in
        (chromosome, window end) order
    """

    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
    pool = sorted(((c, s + m) for c, s, e, m in BACKGROUND),
                  key=lambda locus: (locus[0], locus[1] + FLANK))
    rows = np.random.RandomState(1).choice(len(pool), size=3, replace=False)
    assert loci(df[df["weight"] == 0]) == [pool[i] for i in rows]


def test_background_windows_must_fit_too(tasks, tmp_path):
    rows = BACKGROUND + [("chr1", 0, 100, 50), ("chr2", 400, 500, 50)]
    tasks[0]["background_loci"]["source"] = \
        [write_bed(tmp_path / "background_edges.bed", rows)]
    tasks[0]["background_loci"]["ratio"] = [1.0]
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
    background = set(loci(df[df["weight"] == 0]))
    # exactly the 6 loci that fit are available for the 6 requested
    assert background == {(c, s + m) for c, s, e, m in BACKGROUND}


def test_several_background_files_each_have_a_ratio(tasks, tmp_path):
    first = write_bed(tmp_path / "bg_a.bed", BACKGROUND[:3])
    second = write_bed(tmp_path / "bg_b.bed", BACKGROUND[3:])
    tasks[0]["background_loci"] = {"source": [first, second],
                                   "ratio": [0.5, 1.0 / 3]}
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
    background = loci(df[df["weight"] == 0])
    from_first = {(c, s + m) for c, s, e, m in BACKGROUND[:3]}
    from_second = {(c, s + m) for c, s, e, m in BACKGROUND[3:]}
    # 6 peaks: int(6 * 0.5) = 3 from the first file, int(6 / 3) = 2
    # from the second
    assert len([b for b in background if b in from_first]) == 3
    assert len([b for b in background if b in from_second]) == 2


def test_a_locus_listed_as_peak_and_background_is_kept_twice(tasks, tmp_path):
    # same position, different weights: not a duplicate
    overlap = [("chr1", 150, 250, 50)] + BACKGROUND[:2]
    tasks[0]["background_loci"] = {
        "source": [write_bed(tmp_path / "bg_overlap.bed", overlap)],
        "ratio": [0.5]}
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val",
                                   drop_duplicates=True)
    assert len(df[df["weight"] == 0]) == 3
    both = df[(df["chrom"] == "chr1") & (df["pos"] == 200)]
    assert sorted(both["weight"].tolist()) == [0, 1]


def test_background_count_with_several_peak_files(tasks, tmp_path):
    """
        With several peak files in a task, the background count is taken
        from the peak file that was read last, not from the total
    """

    small = write_bed(tmp_path / "two_peaks.bed", PEAKS[5:7])
    tasks[0]["loci"]["source"] = [tasks[0]["loci"]["source"][0], small]
    tasks[0]["background_loci"]["ratio"] = [1.0]
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
    assert (df["weight"] == 1).sum() == len(VALID_PEAKS) + 2
    assert (df["weight"] == 0).sum() == 2


def test_background_count_follows_the_chromosome_filter(tasks):
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK,
                                   chroms=["chr1"], mode="val")
    background = df[df["weight"] == 0]
    # 3 peaks on chr1 -> int(3 * 0.5) = 1 background locus, from chr1
    assert (df["weight"] == 1).sum() == 3
    assert len(background) == 1
    assert background["chrom"].tolist() == ["chr1"]


def test_background_indices_select_rows_of_the_file(tasks):
    tasks[0]["background_loci"]["ratio"] = [1.0 / 3]
    for _ in range(5):
        df = sequtils.getPeakPositions(
            tasks, CHROM_SIZES, FLANK, mode="train",
            background_loci_indices=[2, 3])
        background = df[df["weight"] == 0]
        assert len(background) == 2
        assert set(loci(background)) == {("chr2", 300), ("chr3", 150)}


def test_val_background_is_fixed_and_train_background_is_redrawn(tasks):
    val = [frozenset(loci(d[d["weight"] == 0])) for d in (
        sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
        for _ in range(8))]
    assert len(set(val)) == 1

    train = [frozenset(loci(d[d["weight"] == 0])) for d in (
        sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="train")
        for _ in range(30))]
    assert all(len(s) == 3 for s in train)
    assert len(set(train)) > 1


def test_too_little_background_is_an_error(tasks):
    tasks[0]["background_loci"]["ratio"] = [2.0]
    with pytest.raises(NoTracebackException):
        sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK)


def test_weights(tasks):
    df = sequtils.getPeakPositions(
        tasks, CHROM_SIZES, FLANK, mode="val", foreground_weight=2.5,
        background_weight=0.25)
    assert df["weight"].tolist() == [2.5] * 6 + [0.25] * 3


def test_task_without_background(tasks):
    del tasks[0]["background_loci"]
    df = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK)
    assert loci(df) == VALID_PEAKS


def test_duplicates_across_files_and_tasks(tasks, tmp_path):
    # the same peaks listed twice in a task, and again in a second task
    source = tasks[0]["loci"]["source"][0]
    del tasks[0]["background_loci"]
    tasks[0]["loci"]["source"] = [source, source]
    tasks[1] = {"signal": {"source": ["UNUSED"]},
                "loci": {"source": [source]},
                "bias": {"source": [], "smoothing": []}}

    kept = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK)
    assert loci(kept) == VALID_PEAKS * 3

    dropped = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK,
                                        drop_duplicates=True)
    assert loci(dropped) == VALID_PEAKS
    assert list(dropped.index) == list(range(len(VALID_PEAKS)))


def test_task_keys_may_be_strings(tasks):
    string_keys = {str(k): v for k, v in tasks.items()}
    a = sequtils.getPeakPositions(tasks, CHROM_SIZES, FLANK, mode="val")
    b = sequtils.getPeakPositions(string_keys, CHROM_SIZES, FLANK, mode="val")
    assert a.equals(b)


def test_integer_columns(tasks):
    df = peaks_only(tasks)
    for column in ("start_coord", "end_coord", "pos"):
        assert np.issubdtype(df[column].dtype, np.integer)
