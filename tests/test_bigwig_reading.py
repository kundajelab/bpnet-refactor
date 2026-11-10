"""
    Tests for bpnet.generators.generators.interval_values: the values
    of a bigWig interval as the batch generator reads them, against
    the way it read them before (tests/legacy_reference.py)
"""

import numpy as np
import pyBigWig
import pytest

from bpnet.generators import generators

from tests.legacy_reference import interval_values_v0

CHROM_SIZES = [("chr1", 2000), ("chr2", 500)]


@pytest.fixture(scope="module")
def bigwig(tmp_path_factory):
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
    # very small values, infinities and a stored NaN
    special = [0.1, -2.5, 0.0, 3.4e38, 1e-38, float("inf"), float("-inf"),
               float("nan"), 7.0]
    bw.addEntries(["chr1"] * len(special),
                  [1000 + i for i in range(len(special))],
                  ends=[1001 + i for i in range(len(special))],
                  values=special)
    # an interval with one value for many positions, up to the end
    bw.addEntries(["chr1"], [1500], ends=[2000], values=[2.0])
    bw.addEntries(["chr2"], [0], ends=[10], values=[5.0])
    bw.close()
    bw = pyBigWig.open(path)
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


INTERVALS = [
    ("chr1", 100, 900), ("chr1", 0, 2000), ("chr1", 0, 100),
    ("chr1", 900, 1000), ("chr1", 995, 1012), ("chr1", 1005, 1006),
    ("chr1", 1499, 1501), ("chr1", 1990, 2000), ("chr1", 0, 1),
    ("chr2", 0, 500), ("chr2", 5, 15), ("chr2", 200, 300)]


@pytest.mark.parametrize("chrom,start,end", INTERVALS)
def test_same_values_as_the_previous_way_of_reading(bigwig, chrom, start,
                                                    end):
    n = end - start
    new = generators.interval_values(bigwig, chrom, start, end)
    old = interval_values_v0(bigwig, chrom, start, end)
    assert len(new) == len(old) == n
    assert np.array_equal(as_stored(new, n), as_stored(old, n))
    assert not np.isnan(as_stored(new, n)).any()


def test_positions_without_data_are_zero_and_infinities_stay(bigwig):
    values = generators.interval_values(bigwig, "chr1", 995, 1012)
    expected = np.array(
        [0, 0, 0, 0, 0, 0.1, -2.5, 0.0, 3.4e38, 1e-38, np.inf, -np.inf, 0,
         7.0, 0, 0, 0], dtype=np.float32)
    assert np.array_equal(values, expected)
    # nothing but gaps
    assert not generators.interval_values(bigwig, "chr2", 200, 300).any()


def test_values_are_a_float32_array_of_the_callers(bigwig):
    values = generators.interval_values(bigwig, "chr1", 100, 900)
    assert isinstance(values, np.ndarray) and values.dtype == np.float32
    assert values.shape == (800,)
    kept = values.copy()
    values[:] = -1
    again = generators.interval_values(bigwig, "chr1", 100, 900)
    assert np.array_equal(again, kept)


@pytest.mark.parametrize("chrom,start,end", [
    ("chr1", 1990, 2001), ("chr3", 0, 10), ("chr1", 50, 50),
    ("chr1", 60, 50), ("chr2", 500, 510)])
def test_intervals_that_cannot_be_read_fail_as_before(bigwig, chrom, start,
                                                      end):
    with pytest.raises(RuntimeError):
        interval_values_v0(bigwig, chrom, start, end)
    with pytest.raises(RuntimeError):
        generators.interval_values(bigwig, chrom, start, end)


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
        new = generators.interval_values(bw, "chr1", 0, 10)
        assert old.shape == new.shape == (0,)
        for values in (old, new):
            with pytest.raises(ValueError):
                np.zeros(10, dtype=np.float32)[:] = values
    finally:
        bw.close()


def test_pybigwig_without_numpy(bigwig, monkeypatch):
    # a pyBigWig that was built without numpy returns lists; the
    # values are the same once stored
    monkeypatch.setattr(generators.pyBigWig, "numpy", 0)
    for chrom, start, end in INTERVALS:
        n = end - start
        slow = generators.interval_values(bigwig, chrom, start, end)
        assert slow.dtype == np.float64
        assert np.array_equal(
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
    monkeypatch.setattr(generators, "interval_values", interval_values_v0)
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
