"""
    Tests for bpnet.generators.windows: the values of bigWig intervals
    and windows as the batch generator reads them, against the way it
    read them before (tests/legacy_reference.py)
"""

import json
import multiprocessing
import os
import pickle
import random

import numpy as np
import pyBigWig
import pytest

from bpnet.generators import generators, windows

from tests import legacy_reference, synth
from tests.legacy_reference import interval_values_v0

CHROM_SIZES = [("chr1", 2000), ("chr2", 500)]


@pytest.fixture(scope="module")
def bigwig_path(tmp_path_factory):
    """
        A bigWig with data in some places only, and with every kind of
        value a file can hold
    """

    path = str(tmp_path_factory.mktemp("bigwig") / "signal.bw")
    rng = np.random.RandomState(0)
    bw = pyBigWig.open(path, "w")
    bw.addHeader(CHROM_SIZES, maxZooms=0)
    # single positions, as in a track of read ends
    positions = np.sort(rng.choice(np.arange(100, 900), size=300,
                                   replace=False))
    bw.addEntries(["chr1"] * 300, [int(p) for p in positions],
                  ends=[int(p) + 1 for p in positions],
                  values=[float(v) for v in rng.poisson(3.0, size=300)])
    # a run of fractional and negative values, zero, very large and
    # very small values, infinities, a stored NaN and a negative zero
    special = [0.1, -2.5, 0.0, 3.4e38, 1e-38, float("inf"), float("-inf"),
               float("nan"), 7.0, -0.0]
    bw.addEntries(["chr1"] * len(special),
                  [1000 + i for i in range(len(special))],
                  ends=[1001 + i for i in range(len(special))],
                  values=special)
    # an interval with one value for many positions, up to the end
    bw.addEntries(["chr1"], [1500], ends=[2000], values=[2.0])
    bw.addEntries(["chr2"], [0], ends=[10], values=[5.0])
    bw.close()
    return path


@pytest.fixture(scope="module")
def bigwig(bigwig_path):
    bw = pyBigWig.open(bigwig_path)
    yield bw
    bw.close()


def as_stored(values, n):
    """
        The values as they end up in the float32 arrays of a batch
    """

    out = np.zeros(n, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        out[:] = values
    return out


def same_bits(a, b):
    """
        Whether two arrays have the same type, shape and bytes. (Equal
        values are not enough for "bit for bit": 0.0 equals -0.0.)
    """

    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and \
        a.tobytes() == b.tobytes()


INTERVALS = [
    ("chr1", 100, 900), ("chr1", 0, 2000), ("chr1", 0, 100),
    ("chr1", 900, 1000), ("chr1", 995, 1012), ("chr1", 1005, 1006),
    ("chr1", 1499, 1501), ("chr1", 1990, 2000), ("chr1", 0, 1),
    ("chr2", 0, 500), ("chr2", 5, 15), ("chr2", 200, 300)]


@pytest.mark.parametrize("chrom,start,end", INTERVALS)
def test_same_values_as_the_previous_way_of_reading(bigwig, chrom, start,
                                                    end):
    n = end - start
    new = windows.interval_values(bigwig, chrom, start, end)
    old = interval_values_v0(bigwig, chrom, start, end)
    assert len(new) == len(old) == n
    assert same_bits(as_stored(new, n), as_stored(old, n))
    assert not np.isnan(as_stored(new, n)).any()


def test_positions_without_data_are_zero_and_infinities_stay(bigwig):
    values = windows.interval_values(bigwig, "chr1", 995, 1012)
    expected = np.array(
        [0, 0, 0, 0, 0, 0.1, -2.5, 0.0, 3.4e38, 1e-38, np.inf, -np.inf, 0,
         7.0, -0.0, 0, 0], dtype=np.float32)
    # the stored negative zero keeps its sign; a position without data
    # and the stored NaN become positive zeros
    assert same_bits(values, expected)
    assert np.flatnonzero(np.signbit(values)).tolist() == [6, 11, 14]
    # nothing but gaps
    gaps = windows.interval_values(bigwig, "chr2", 200, 300)
    assert same_bits(gaps, np.zeros(100, dtype=np.float32))


def test_values_are_a_float32_array_of_the_callers(bigwig):
    values = windows.interval_values(bigwig, "chr1", 100, 900)
    assert isinstance(values, np.ndarray) and values.dtype == np.float32
    assert values.shape == (800,)
    kept = values.copy()
    values[:] = -1
    again = windows.interval_values(bigwig, "chr1", 100, 900)
    assert np.array_equal(again, kept)


@pytest.mark.parametrize("chrom,start,end", [
    ("chr1", 1990, 2001), ("chr3", 0, 10), ("chr1", 50, 50),
    ("chr1", 60, 50), ("chr2", 500, 510)])
def test_intervals_that_cannot_be_read_fail_as_before(bigwig, chrom, start,
                                                      end):
    with pytest.raises(RuntimeError):
        interval_values_v0(bigwig, chrom, start, end)
    with pytest.raises(RuntimeError):
        windows.interval_values(bigwig, chrom, start, end)


def test_file_without_data(tmp_path):
    """
        A bigWig with a header and no entries has no values to return.
        The previous reading gave an empty array, which cannot be
        stored into a batch; the present one must do the same (pyBigWig
        0.3.22 returns an array without dimensions for such a file,
        which numpy would broadcast into the batch without complaint).
    """

    path = str(tmp_path / "empty.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader(CHROM_SIZES)
    bw.close()
    bw = pyBigWig.open(path)
    try:
        old = interval_values_v0(bw, "chr1", 0, 10)
        new = windows.interval_values(bw, "chr1", 0, 10)
        assert old.shape == new.shape == (0,)
        for values in (old, new):
            with pytest.raises(ValueError):
                np.zeros(10, dtype=np.float32)[:] = values
    finally:
        bw.close()


def test_pybigwig_without_numpy(bigwig, monkeypatch):
    # a pyBigWig that was built without numpy returns lists; the
    # values are the same once stored
    monkeypatch.setattr(windows.pyBigWig, "numpy", 0)
    for chrom, start, end in INTERVALS:
        n = end - start
        slow = windows.interval_values(bigwig, chrom, start, end)
        assert slow.dtype == np.float64
        assert same_bits(
            as_stored(slow, n),
            as_stored(interval_values_v0(bigwig, chrom, start, end), n))


def batches(dataset, mode):
    from bpnet.generators.generators import MBPNetSequenceGenerator

    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": dataset["input_len"],
              "output_len": dataset["output_len"], "rev_comp_aug": True,
              "max_jitter": 0, "shuffle": False, "mode": mode}
    # in test mode the last batch is filled up with randomly chosen
    # loci; the same ones for both runs
    state = np.random.get_state()
    np.random.seed(0)
    try:
        gen = MBPNetSequenceGenerator(
            dataset["input_json"], params, dataset["fasta"],
            dataset["chrom_sizes"], dataset["splits"]["0"]["val"],
            num_threads=1, batch_size=16, epochs=1)
        return list(gen.gen())
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("mode", ["val", "test"])
def test_batches_are_the_same_as_with_the_previous_reading(
        synth_dataset, monkeypatch, mode):
    """
        Whole batches of the generator, array by array, with the
        present reading and with the previous one swapped in
    """

    new = batches(synth_dataset, mode)
    monkeypatch.setattr(windows, "interval_values", interval_values_v0)
    old = batches(synth_dataset, mode)
    assert len(new) == len(old) > 1

    def arrays(batch):
        if isinstance(batch, dict):
            return sorted(batch.items())
        inputs, outputs, weights = batch
        return sorted(inputs.items()) + sorted(outputs.items()) + \
            [("weights", weights)]

    compared = 0
    for a, b in zip(new, old):
        for (name, x), (other, y) in zip(arrays(a), arrays(b)):
            assert name == other
            assert x.dtype == y.dtype and x.shape == y.shape, name
            assert np.array_equal(x, y), name
            compared += 1
    assert compared >= 5 * len(new)


# --------------------------------------------------------------------
# WindowReader: all windows of a batch in one call
# --------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_files(bigwig, bigwig_path, tmp_path_factory):
    """
        The file of the `bigwig` fixture and a second one with other
        values, as two channels: (open pyBigWig files, their paths)
    """

    path = str(tmp_path_factory.mktemp("bigwig") / "second.bw")
    rng = np.random.RandomState(1)
    bw = pyBigWig.open(path, "w")
    bw.addHeader(CHROM_SIZES, maxZooms=0)
    positions = np.sort(rng.choice(np.arange(0, 2000), size=700,
                                   replace=False))
    bw.addEntries(["chr1"] * 700, [int(p) for p in positions],
                  ends=[int(p) + 1 for p in positions],
                  values=[float(v) for v in rng.gamma(2.0, 1.7, size=700)])
    bw.addEntries(["chr2"], [100], ends=[400], values=[0.123456789])
    bw.close()
    second = pyBigWig.open(path)
    yield [bigwig, second], [bigwig_path, path]
    second.close()


WINDOWS = {
    # forward and reverse complement copies of two loci: overlapping
    "copies with other jitters": (["chr1"] * 4, [300, 1200, 316, 1188], 400),
    "the same window twice": (["chr1", "chr1"], [500, 500], 100),
    "adjacent, not overlapping": (["chr1", "chr1"], [100, 200], 100),
    "two chromosomes, unsorted": (
        ["chr2", "chr1", "chr2", "chr1"], [100, 1500, 0, 0], 200),
    "special values": (["chr1", "chr1"], [995, 1003], 17),
    "chromosome ends": (["chr1", "chr1", "chr2"], [0, 1990, 490], 10),
    "one base": (["chr1", "chr2"], [1005, 5], 1),
    "whole chromosome": (["chr2"], [0], 500),
}


@pytest.mark.parametrize("case", sorted(WINDOWS))
def test_windows_of_a_batch(two_files, case):
    """
        Every window in every file: the values that reading each
        window on its own gave before
    """

    files, paths = two_files
    chroms, starts, width = WINDOWS[case]
    reader = windows.WindowReader(paths)
    try:
        values = reader.read(chroms, starts, width)
    finally:
        reader.close()
    assert values.dtype == np.float32
    assert values.shape == (len(chroms), 2, width)
    for row, (chrom, start) in enumerate(zip(chroms, starts)):
        for channel, bw in enumerate(files):
            expected = as_stored(
                interval_values_v0(bw, chrom, start, start + width), width)
            assert same_bits(values[row, channel], expected), (row, channel)


def test_many_random_windows(two_files):
    files, paths = two_files
    rng = np.random.RandomState(3)
    chroms = ["chr1"] * 300
    starts = [int(s) for s in rng.randint(0, 1900, size=300)]
    reader = windows.WindowReader(paths)
    try:
        values = reader.read(chroms, starts, 100)
        # the reader can be used again
        again = reader.read(chroms[:5], starts[:5], 100)
    finally:
        reader.close()
    for row in range(300):
        for channel, bw in enumerate(files):
            expected = as_stored(interval_values_v0(
                bw, "chr1", starts[row], starts[row] + 100), 100)
            assert same_bits(values[row, channel], expected)
    assert same_bits(again, values[:5])


@pytest.mark.parametrize("chroms,starts,width", [
    (["chr1"], [1990], 20), (["chr1", "chr1"], [0, 1999], 2),
    (["chr3"], [0], 10), (["chr1"], [-5], 20), (["chr2"], [500], 10)])
def test_windows_that_cannot_be_read(two_files, chroms, starts, width):
    # as pyBigWig always refused them
    _, paths = two_files
    reader = windows.WindowReader(paths)
    try:
        with pytest.raises(RuntimeError):
            reader.read(chroms, starts, width)
    finally:
        reader.close()


def test_reader_of_a_file_without_data(two_files, tmp_path):
    _, paths = two_files
    empty = str(tmp_path / "empty.bw")
    bw = pyBigWig.open(empty, "w")
    bw.addHeader(CHROM_SIZES)
    bw.close()
    # opening is not reading: a control file that is never read (the
    # control inputs can be set to zero) may be empty
    reader = windows.WindowReader([paths[0], empty])
    try:
        with pytest.raises(ValueError, match="empty.bw holds no data"):
            reader.read(["chr1"], [0], 10)
        # every time, not only at the first read
        with pytest.raises(ValueError, match="empty.bw holds no data"):
            reader.read(["chr1", "chr2"], [100, 0], 10)
    finally:
        reader.close()


def test_reader_of_a_file_whose_first_chromosome_is_empty(tmp_path):
    """
        Whether a file holds data is seen from what a read returns,
        not by trying its first chromosome: that one may have length
        zero, or no entries, while the windows are elsewhere
    """

    path = str(tmp_path / "first_chromosome_of_length_zero.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader([("zero", 0), ("gap", 50), ("chr1", 100)], maxZooms=0)
    bw.addEntries(["chr1"], [10], ends=[20], values=[2.0])
    bw.close()

    bw = pyBigWig.open(path)
    reader = windows.WindowReader([path])
    try:
        expected = as_stored(interval_values_v0(bw, "chr1", 5, 25), 20)
        assert expected.tolist() == [0.0] * 5 + [2.0] * 10 + [0.0] * 5
        values = reader.read(["chr1", "gap"], [5, 0], 20)
        assert same_bits(values[0, 0], expected)
        assert same_bits(values[1, 0], np.zeros(20, dtype=np.float32))
        # the chromosome of length zero has no window that can be read
        with pytest.raises(RuntimeError):
            interval_values_v0(bw, "zero", 0, 1)
        with pytest.raises(RuntimeError):
            reader.read(["zero"], [0], 1)
    finally:
        reader.close()
        bw.close()


def test_reader_without_windows_or_files(two_files):
    _, paths = two_files
    reader = windows.WindowReader(paths)
    assert reader.read([], [], 10).shape == (0, 2, 10)
    reader.close()
    reader = windows.WindowReader([])
    assert reader.read(["chr1"], [0], 10).shape == (1, 0, 10)
    reader.close()


def test_overlapping_windows_are_grouped():
    groups = windows._overlapping_windows(
        ["chr1", "chr2", "chr1", "chr1", "chr1", "chr2"],
        [300, 0, 316, 716, 1000, 99], 100)
    assert groups == [
        ("chr1", 300, 416, [0, 2]), ("chr1", 716, 816, [3]),
        ("chr1", 1000, 1100, [4]), ("chr2", 0, 199, [1, 5])]
    # windows that touch are read separately
    assert windows._overlapping_windows(["c", "c"], [0, 100], 100) == [
        ("c", 0, 100, [0]), ("c", 100, 200, [1])]
    assert windows._overlapping_windows([], [], 100) == []


def test_overlapping_windows_are_read_once(two_files, monkeypatch):
    _, paths = two_files
    calls = []
    original = windows.interval_values

    def counted(bigwig, chrom, start, end):
        calls.append((chrom, start, end))
        return original(bigwig, chrom, start, end)

    monkeypatch.setattr(windows, "interval_values", counted)
    reader = windows.WindowReader(paths[:1])
    reader.read(["chr1"] * 4, [300, 1200, 316, 1188], 400)
    reader.close()
    assert calls == [("chr1", 300, 716), ("chr1", 1188, 1600)]


# --------------------------------------------------------------------
# whole batches against the previous batch construction
# --------------------------------------------------------------------

SMALL = dict(chrom_len=20000, n_peaks=60, n_background=60)

DATASETS = {
    "two strands with control": dict(),
    "unstranded": dict(stranded=False),
    "without control": dict(with_control=False),
    "two tasks": dict(num_tasks=2),
    # the first task with one track, the second with two, the third
    # with one: the tracks of a task do not start at a multiple of
    # anything
    "tasks with one and two tracks": dict(num_tasks=3, one_track=["0", "2"]),
    "smoothed control": dict(smoothing=[[7.0, 81], [7.0, 81]]),
    "one smoothed control": dict(smoothing=[None, [3.0, 21]]),
}


@pytest.fixture(scope="module", params=sorted(DATASETS))
def variant(request, tmp_path_factory):
    options = dict(DATASETS[request.param])
    smoothing = options.pop("smoothing", None)
    one_track = options.pop("one_track", [])
    directory = str(tmp_path_factory.mktemp("variant"))
    options.update(SMALL)
    dataset = synth.make_dataset(directory, **options)
    if smoothing is not None or one_track:
        with open(dataset["input_json"]) as fh:
            tasks = json.load(fh)
        if smoothing is not None:
            tasks["0"]["bias"]["smoothing"] = smoothing
        for task in one_track:
            for kind in ("signal", "bias"):
                tasks[task][kind]["source"] = tasks[task][kind]["source"][:1]
            tasks[task]["bias"]["smoothing"] = [None]
        with open(dataset["input_json"], "w") as fh:
            json.dump(tasks, fh)
    return dataset


def batch_rows(generator, start, size):
    """
        The rows of a batch as a worker process hands them to
        _generate_batch
    """

    rows = generator.get_resized_samples()[0].iloc[start:start + size][
        ['chrom', 'pos', 'weight']].copy()
    rows['status'] = 1
    rows.loc[rows['weight'] == 0, 'status'] = -1
    rows['idx'] = rows.index.values
    return rows


def flat(batch):
    if isinstance(batch, dict):
        return sorted(batch.items())
    inputs, outputs, weights = batch
    return sorted(inputs.items()) + sorted(outputs.items()) + \
        [("weights", weights)]


MODES = {
    "train": dict(mode="train", rev_comp_aug=True, max_jitter=16),
    "train without reverse complements": dict(
        mode="train", rev_comp_aug=False, max_jitter=16),
    "val": dict(mode="val", rev_comp_aug=False, max_jitter=0),
    "test with reverse complements": dict(
        mode="test", rev_comp_aug=True, max_jitter=0),
    "control set to zero": dict(
        mode="test", rev_comp_aug=False, max_jitter=0,
        set_bias_as_zero=True),
}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_batches_equal_the_previous_batch_construction(variant, mode):
    """
        Every array of every batch, bit for bit, against the batch
        construction this one replaced (kept in
        tests/legacy_reference.py), with the same jitters
    """

    options = dict(MODES[mode])
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": variant["input_len"],
              "output_len": variant["output_len"],
              "rev_comp_aug": options.pop("rev_comp_aug"),
              "max_jitter": options.pop("max_jitter"), "shuffle": False,
              "mode": options.pop("mode")}
    state = np.random.get_state()
    np.random.seed(0)
    try:
        generator = generators.MBPNetSequenceGenerator(
            variant["input_json"], params, variant["fasta"],
            variant["chrom_sizes"], variant["chroms"], num_threads=1,
            batch_size=16, epochs=1, **options)
    finally:
        np.random.set_state(state)

    n_rows = len(generator.get_resized_samples()[0])
    compared = 0
    for number, start in enumerate(range(0, n_rows - 15, 16)):
        random.seed(number)
        new = generator._generate_batch(batch_rows(generator, start, 16))
        random.seed(number)
        old = legacy_reference.generate_batch_v0(
            generator, batch_rows(generator, start, 16))
        new, old = flat(new), flat(old)
        assert [name for name, _ in new] == [name for name, _ in old]
        for (name, a), (_, b) in zip(new, old):
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert same_bits(a, b), (name, start)
            compared += 1
    assert compared >= 5 * (n_rows // 16) > 0


def test_files_are_opened_once_per_process(synth_dataset):
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": synth_dataset["input_len"],
              "output_len": synth_dataset["output_len"],
              "rev_comp_aug": False, "max_jitter": 0, "shuffle": False,
              "mode": "val"}
    generator = generators.MBPNetSequenceGenerator(
        synth_dataset["input_json"], params, synth_dataset["fasta"],
        synth_dataset["chrom_sizes"], synth_dataset["chroms"], num_threads=1,
        batch_size=16, epochs=1)
    assert getattr(generator, "_opened_files", None) is None
    generator._generate_batch(batch_rows(generator, 0, 16))
    first = generator._opened_files
    assert first[0] == os.getpid()
    generator._generate_batch(batch_rows(generator, 16, 16))
    assert generator._opened_files is first

    # a forked process opens its own
    def in_child(queue):
        generator._generate_batch(batch_rows(generator, 0, 16))
        queue.put((generator._opened_files[0] == os.getpid(),
                   generator._opened_files[0] != first[0]))

    queue = multiprocessing.Queue()
    child = multiprocessing.Process(target=in_child, args=(queue,))
    child.start()
    assert queue.get(timeout=120) == (True, True)
    child.join()
    assert generator._opened_files is first


def test_open_files_are_not_pickled(synth_dataset):
    """
        A generator that has read a batch can still be pickled (worker
        processes that are spawned instead of forked get it that way);
        the copy opens the files for itself
    """

    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": synth_dataset["input_len"],
              "output_len": synth_dataset["output_len"],
              "rev_comp_aug": False, "max_jitter": 0, "shuffle": False,
              "mode": "val"}
    generator = generators.MBPNetSequenceGenerator(
        synth_dataset["input_json"], params, synth_dataset["fasta"],
        synth_dataset["chrom_sizes"], synth_dataset["chroms"], num_threads=1,
        batch_size=16, epochs=1)
    fresh = pickle.loads(pickle.dumps(generator))
    batch = generator._generate_batch(batch_rows(generator, 0, 16))
    opened = generator._opened_files
    assert opened is not None

    copy = pickle.loads(pickle.dumps(generator))
    assert generator._opened_files is opened
    for other in (fresh, copy):
        assert getattr(other, "_opened_files", None) is None
        again = other._generate_batch(batch_rows(other, 0, 16))
        assert other._opened_files[0] == os.getpid()
        assert other._opened_files is not opened
        for (name, a), (_, b) in zip(flat(again), flat(batch)):
            assert same_bits(a, b), name
