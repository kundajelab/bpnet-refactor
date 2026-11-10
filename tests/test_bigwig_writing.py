"""
    Characterization tests for the two bigWig writers:
    bpnet.utils.bigwig_helper.write_bigwig (predicted profiles) and
    bpnet.utils.importance_hdf5_to_bigwig (contribution scores)
"""

import gzip
import os
import re
import struct
import zlib

import h5py
import numpy as np
import pyBigWig
import pytest

from bpnet.utils import bigwig_helper
from bpnet.utils.bigwig_helper import write_bigwig
from bpnet.utils.importance_hdf5_to_bigwig import importance_hdf5_to_bigwig

from tests.legacy_reference import (
    importance_hdf5_to_bigwig_v0, write_bigwig_v0)

HEADER = [("chr1", 1000), ("chr2", 800), ("chr10", 600)]

STAT_LABELS = ["Min", ".1%", "1%", "50%", "99%", "99.9%", "99.95%", "99.99%",
               "Max"]
STAT_QUANTILES = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9995, 0.9999, 1.0]


def read_all(path):
    """
        chromosome -> all stored values (NaN where nothing was written),
        plus the chromosome order of the file
    """

    bw = pyBigWig.open(path)
    chroms = bw.chroms()
    out = {c: np.array(bw.values(c, 0, n)) for c, n in chroms.items()}
    bw.close()
    return out, chroms


def expected_tracks(header, regions, data):
    """
        What the file must contain. Regions are taken in order of
        (chromosome order of the header, start). Where two consecutive
        regions overlap, the first one supplies the positions before
        the midpoint between the two regions' centres (the 4th element
        of a region) and the second one the rest.

        This states the rule for what both writers are given: regions
        of one width, so that centres are ordered like starts. It is
        not a specification for arbitrary intervals; the writers do not
        handle a region nested inside a longer one.
    """

    order = {c: i for i, (c, _) in enumerate(header)}
    tracks = {c: np.full(n, np.nan) for c, n in header}
    ranked = sorted(range(len(regions)),
                    key=lambda i: (order[regions[i][0]], regions[i][1]))
    written = []
    for rank, i in enumerate(ranked):
        chrom, start, end, mid = regions[i]
        first, last = start, end
        if rank > 0:
            p = ranked[rank - 1]
            if regions[p][0] == chrom and start < regions[p][2]:
                first = (regions[p][3] + mid) // 2
        if rank + 1 < len(ranked):
            n = ranked[rank + 1]
            if regions[n][0] == chrom and regions[n][1] < end:
                last = (mid + regions[n][3]) // 2
        values = np.asarray(data[i], dtype=np.float32)[first - start:
                                                       last - start]
        tracks[chrom][first:last] = values
        written.append(values)
    return tracks, np.hstack(written)


def check_stats(path, values):
    with open(path) as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh]
    assert [r[0] for r in rows] == STAT_LABELS
    expected = [np.quantile(values, q) for q in STAT_QUANTILES]
    np.testing.assert_allclose([float(r[1]) for r in rows], expected,
                               rtol=0, atol=1e-6)


def check_file(path, header, regions, data, stats=None):
    tracks, written = expected_tracks(header, regions, data)
    got, chroms = read_all(path)
    assert list(chroms.items()) == [(c, n) for c, n in header]
    for chrom, values in tracks.items():
        assert np.array_equal(np.isnan(got[chrom]), np.isnan(values)), chrom
        assert np.array_equal(np.nan_to_num(got[chrom]),
                              np.nan_to_num(values)), chrom
    if stats is not None:
        check_stats(stats, written)


def region_data(regions, seed=0):
    rng = np.random.RandomState(seed)
    return [rng.gamma(2.0, size=end - start) for _, start, end, _ in regions]


def run_write_bigwig(tmp_path, regions, header=HEADER, seed=0):
    data = region_data(regions, seed)
    path, stats = str(tmp_path / "out.bw"), str(tmp_path / "stats.txt")
    write_bigwig(data, regions, header, path, stats)
    check_file(path, header, regions, data, stats)
    return path


# --------------------------------------------------------------------
# write_bigwig
# --------------------------------------------------------------------

def test_disjoint_regions_in_any_order(tmp_path):
    regions = [("chr2", 100, 150, 125), ("chr1", 300, 360, 330),
               ("chr10", 0, 40, 20), ("chr1", 10, 60, 35),
               ("chr1", 940, 1000, 970)]
    run_write_bigwig(tmp_path, regions)


def test_single_region(tmp_path):
    run_write_bigwig(tmp_path, [("chr2", 700, 800, 750)])


def test_chromosomes_follow_the_header_order(tmp_path):
    header = [("chr10", 600), ("chr2", 800), ("chr1", 1000)]
    regions = [("chr1", 10, 60, 35), ("chr10", 0, 40, 20),
               ("chr2", 100, 150, 125)]
    run_write_bigwig(tmp_path, regions, header=header)


def test_two_overlapping_regions_split_between_their_centres(tmp_path):
    regions = [("chr1", 100, 200, 150), ("chr1", 160, 260, 210)]
    path = run_write_bigwig(tmp_path, regions)
    data = region_data(regions)
    got, _ = read_all(path)
    # positions 100..179 from the first region, 180..259 from the second
    assert np.array_equal(got["chr1"][100:180],
                          np.float32(data[0][:80]).astype(np.float64))
    assert np.array_equal(got["chr1"][180:260],
                          np.float32(data[1][20:]).astype(np.float64))


def test_chain_of_overlapping_regions(tmp_path):
    regions = [("chr1", 100, 200, 150), ("chr1", 160, 260, 210),
               ("chr1", 240, 340, 290), ("chr1", 300, 400, 350),
               ("chr2", 0, 100, 50), ("chr2", 50, 150, 100)]
    run_write_bigwig(tmp_path, regions, seed=1)


def test_touching_and_identical_regions(tmp_path):
    regions = [("chr1", 100, 200, 150), ("chr1", 200, 300, 250),
               ("chr2", 400, 500, 450), ("chr2", 400, 500, 450)]
    run_write_bigwig(tmp_path, regions, seed=2)


def test_regions_that_touch_are_written_in_full(tmp_path):
    # also when the midpoint between their centres is not where they
    # meet (regions of different widths)
    regions = [("chr10", 0, 100, 50), ("chr10", 100, 140, 120),
               ("chr10", 140, 150, 145)]
    path = run_write_bigwig(tmp_path, regions, seed=8)
    data = region_data(regions, 8)
    got, _ = read_all(path)
    assert np.array_equal(got["chr10"][:150], np.float32(
        np.concatenate(data)).astype(np.float64))


def test_many_regions_like_predictions(tmp_path):
    """
        Regions of one length around sorted peak positions, as
        bpnet-predict writes them, many of them overlapping
    """

    rng = np.random.RandomState(3)
    width = 60
    regions = []
    for chrom, size in HEADER:
        centres = np.unique(rng.randint(width, size - width, size=25))
        regions.extend((chrom, int(c) - width // 2, int(c) + width // 2,
                        int(c)) for c in centres)
    order = rng.permutation(len(regions))
    run_write_bigwig(tmp_path, [regions[i] for i in order], seed=4)


def test_three_regions_at_one_place(tmp_path):
    """
        The middle one of three regions with one centre has no
        position left to write. (Writing it used to fail: an empty
        list of entries is refused by pyBigWig.)
    """

    regions = [("chr1", 100, 200, 150)] * 3 + [("chr2", 10, 60, 35)]
    run_write_bigwig(tmp_path, regions, seed=5)
    with pytest.raises(RuntimeError):
        write_bigwig_v0(region_data(regions, 5), regions, HEADER,
                        str(tmp_path / "v0.bw"), str(tmp_path / "v0.txt"))


# --------------------------------------------------------------------
# against the writer that stored an entry per position
# --------------------------------------------------------------------

def internals(path):
    """
        What a bigWig file holds, read from its bytes (pyBigWig rounds
        the statistics of the header to integers, and cannot report
        them at all when they are not finite)

        Returns:
            dict: 'summary': the 40 bytes of the total summary (number
                of positions with a value, minimum, maximum, sum, sum
                of squares); 'sections': (index entry, chromosome,
                start, end, step, span, type, number of items) of
                every data section; 'zooms': (reduction, sorted
                records of 32 bytes) of every zoom level
    """

    with open(path, "rb") as fh:
        data = fh.read()
    (magic, _, zoom_levels, _, _, full_index, _, _, _, total_summary,
     uncompress, _) = struct.unpack_from("<IHHQQQHHQQIQ", data, 0)
    assert magic == 0x888FFC26

    def blocks(index):
        assert struct.unpack_from("<I", data, index)[0] == 0x2468ACE0
        found = []

        def node(offset):
            is_leaf, _, count = struct.unpack_from("<BBH", data, offset)
            for i in range(count):
                if is_leaf:
                    item = struct.unpack_from("<IIIIQQ", data,
                                              offset + 4 + 32 * i)
                    raw = data[item[4]:item[4] + item[5]]
                    found.append((item[:4], zlib.decompress(raw)
                                  if uncompress else raw))
                else:
                    node(struct.unpack_from("<IIIIQ", data,
                                            offset + 4 + 24 * i)[4])

        node(index + 48)
        return found

    sections = []
    for entry, block in blocks(full_index):
        chrom, start, end, step, span, kind, _, count = struct.unpack_from(
            "<IIIIIBBH", block, 0)
        sections.append((entry, chrom, start, end, step, span, kind, count))
    zooms = []
    for level in range(zoom_levels):
        reduction, _, _, index = struct.unpack_from("<IIQQ", data,
                                                    64 + 24 * level)
        records = [block[i:i + 32] for _, block in blocks(index)
                   for i in range(0, len(block), 32)]
        zooms.append((reduction, sorted(records)))
    return {"summary": data[total_summary:total_summary + 40],
            "sections": sections, "zooms": zooms}


def content(path):
    """
        Everything that can be read from a bigWig file, whatever the
        layout of its data sections: the chromosomes, the float32
        value at every position, the stored intervals (which tell a
        stored NaN from a position without a value), the total
        summary and the records of every zoom level
    """

    bw = pyBigWig.open(path)
    chroms = list(bw.chroms().items())
    values = [bw.values(c, 0, n, numpy=True).tobytes() for c, n in chroms]
    intervals = [[(start, end, struct.pack("<f", value))
                  for start, end, value in bw.intervals(c) or ()]
                 for c, _ in chroms]
    bw.close()
    inside = internals(path)
    return {"chroms": chroms, "values": values, "intervals": intervals,
            "summary": inside["summary"], "zooms": inside["zooms"]}


def assert_same_content(new, old):
    new, old = content(new), content(old)
    for key in ("chroms", "values", "intervals", "summary"):
        assert new[key] == old[key], key
    assert [r for r, _ in new["zooms"]] == [r for r, _ in old["zooms"]]
    for (reduction, a), (_, b) in zip(new["zooms"], old["zooms"]):
        assert a == b, "zoom level {}".format(reduction)
    return new


def same_bytes(a, b):
    with open(a, "rb") as first, open(b, "rb") as second:
        return first.read() == second.read()


def summaries(path, seed=0):
    """
        What the file answers to requests for summaries (which pyBigWig
        takes from the zoom levels unless exact values are asked for)
    """

    bw = pyBigWig.open(path)
    rng = np.random.RandomState(seed)
    out = []
    for chrom, size in bw.chroms().items():
        for _ in range(6):
            start = int(rng.randint(0, size - 1))
            end = int(rng.randint(start + 1, size + 1))
            for kind in ("mean", "max", "min", "coverage", "std", "sum"):
                for bins in (1, 5):
                    for exact in (False, True):
                        out.append(np.array(bw.stats(
                            chrom, start, end, type=kind, nBins=bins,
                            exact=exact), dtype=np.float64).tobytes())
    bw.close()
    return out


def like_predictions(seed, n_per_chrom=120, width=200):
    """
        Many regions of one width, most of them overlapping others, in
        random order, with float32 values in one array (what
        bpnet-predict hands to the writer)
    """

    rng = np.random.RandomState(seed)
    header = [("chr1", 30000), ("chr2", 20000), ("chr10", 9000)]
    regions = []
    for chrom, size in header:
        centres = rng.randint(width, size - width, size=n_per_chrom)
        regions.extend((chrom, int(c) - width // 2, int(c) + width // 2,
                        int(c)) for c in centres)
    order = rng.permutation(len(regions))
    regions = [regions[i] for i in order]
    data = rng.gamma(0.3, 2.0, size=(len(regions), width)).astype(np.float32)
    return header, regions, data


SPECIAL = [-0.0, 0.0, np.nan, np.inf, -np.inf, 1e39, -1e39, 1e-45, 1e-50,
           0.1, 16777217.0, 3.4e38]


def special_values():
    """
        Values a float32 holds differently from a float64, or not at
        all, handed over as float64
    """

    header = [("chr1", 500), ("chr2", 100)]
    regions = [("chr1", 10, 10 + len(SPECIAL), 16),
               ("chr1", 100, 100 + len(SPECIAL), 106),
               ("chr2", 0, len(SPECIAL), 6)]
    data = [np.array(SPECIAL), np.array(SPECIAL[::-1]),
            np.array(SPECIAL) * -1.0]
    return header, regions, data


def other_types():
    """
        Rows that are not float32 or float64 arrays. The values go
        through a double on their way into the file, as when pyBigWig
        was given Python floats: an integer beyond 2**53 and a long
        double are rounded twice, a signalling NaN becomes a quiet
        one, and a masked value is stored as NaN.
    """

    header = [("chr1", 500)]
    rows = [
        np.array([2 ** 62 + 2 ** 38 + 1, -(2 ** 62 + 2 ** 38 + 1), 7, 0],
                 dtype=np.int64),
        np.array([2 ** 64 - 1, 2 ** 63 + 2 ** 39 + 1, 3, 16777217],
                 dtype=np.uint64),
        np.array([np.longdouble(1) + np.longdouble(2) ** -24 +
                  np.longdouble(2) ** -60, np.longdouble(0.1), 2, 3]),
        np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False]),
        np.array([0x7f800001, 0xffc00123, 0x3f800000, 0x00000001],
                 dtype=np.uint32).view(np.float32),
        np.array([0.1, 1e-7, 65504.0, -0.0], dtype=np.float16),
    ]
    regions = [("chr1", 20 * i, 20 * i + 4, 20 * i + 2)
               for i in range(len(rows))]
    return header, regions, rows


CASES = {
    "like predictions": lambda: like_predictions(0),
    "like predictions, float64": lambda: (
        lambda h, r, d: (h, r, [row.astype(np.float64) * 1.1 for row in d]))(
            *like_predictions(1)),
    "regions as numpy values": lambda: (
        lambda h, r, d: (h, [(np.str_(c), np.int64(s), np.int64(e),
                              np.int64(m)) for c, s, e, m in r], d))(
            *like_predictions(2, n_per_chrom=20)),
    "few short regions": lambda: (
        HEADER, [("chr2", 100, 150, 125), ("chr1", 300, 360, 330),
                 ("chr1", 320, 380, 350), ("chr10", 0, 40, 20)],
        region_data([("chr2", 100, 150, 125), ("chr1", 300, 360, 330),
                     ("chr1", 320, 380, 350), ("chr10", 0, 40, 20)], 6)),
    "special values": special_values,
    "rows of other types": other_types,
    # the part of a region before the start of its chromosome is left
    # out
    "region before the chromosome start": lambda: (
        HEADER, [("chr1", -5, 15, 5), ("chr1", 10, 30, 20),
                 ("chr2", -8, 12, 2)],
        region_data([("chr1", -5, 15, 5), ("chr1", 10, 30, 20),
                     ("chr2", -8, 12, 2)], 7)),
    # a region that reaches beyond the end of its chromosome is
    # written as it is
    "region beyond the chromosome end": lambda: (
        HEADER, [("chr10", 590, 610, 600), ("chr2", 100, 120, 110)],
        region_data([("chr10", 590, 610, 600), ("chr2", 100, 120, 110)], 8)),
}


def write_both(tmp_path, header, regions, data):
    paths = {}
    for name, write in (("new", write_bigwig), ("old", write_bigwig_v0)):
        paths[name] = str(tmp_path / (name + ".bw"))
        with np.errstate(invalid="ignore", over="ignore"):
            write(data, regions, header, paths[name],
                  str(tmp_path / (name + ".txt")))
    with open(str(tmp_path / "new.txt")) as a, \
            open(str(tmp_path / "old.txt")) as b:
        assert a.read() == b.read()
    return paths


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("runs", [True, False])
@pytest.mark.parametrize("case", sorted(CASES))
def test_same_file_content_as_an_entry_per_position(tmp_path, monkeypatch,
                                                    case, runs):
    """
        Against the writer this one replaced (kept in
        tests/legacy_reference.py): the same float32 value at every
        position byte for byte, the same stored intervals, total
        summary and zoom records, and a statistics file with the same
        text. Written as an entry per position (what happens with a
        pyBigWig before 0.3.29) the file is the same file.
    """

    monkeypatch.setattr(bigwig_helper, "RUNS", runs)
    header, regions, data = CASES[case]()
    paths = write_both(tmp_path, header, regions, data)
    found = assert_same_content(paths["new"], paths["old"])
    assert found["chroms"] == [(c, n) for c, n in header]
    assert sum(len(i) for i in found["intervals"]) > 0
    if not runs:
        assert same_bytes(paths["new"], paths["old"])
    if case.startswith("like predictions"):
        # enough values for zoom levels, and summaries from them
        assert len(found["zooms"]) >= 1 and len(found["zooms"][0][1]) > 10
        assert summaries(paths["new"]) == summaries(paths["old"])


def test_layout_of_the_file_follows_the_pybigwig_version(tmp_path):
    """
        Runs of positions are used where pyBigWig records their end
        correctly: then every data section is a run that ends where
        its last value is, in the section and in the index, and the
        file is smaller. Elsewhere the file is, byte for byte, what
        it always was.
    """

    version = tuple(int(part) for part in re.findall(
        r"\d+", pyBigWig.__version__)[:3])
    assert bigwig_helper.RUNS == (version >= (0, 3, 29))

    header, regions, data = like_predictions(3)
    paths = write_both(tmp_path, header, regions, data)
    if not bigwig_helper.RUNS:
        assert same_bytes(paths["new"], paths["old"])
        return

    sections = internals(paths["new"])["sections"]
    assert len(sections) > 3
    for (_, index_start, _, index_end), chrom, start, end, step, span, \
            kind, count in sections:
        # fixedStep, a value per position
        assert (kind, step, span) == (3, 1, 1)
        assert end == start + count
        assert (index_start, index_end) == (start, end)
    assert os.path.getsize(paths["new"]) < 0.85 * os.path.getsize(
        paths["old"])
    # the previous layout: bedGraph sections
    assert {section[6] for section in internals(
        paths["old"])["sections"]} == {1}


def test_special_values_are_stored_as_float32(tmp_path, monkeypatch):
    monkeypatch.setattr(bigwig_helper, "RUNS", True)
    header, regions, data = special_values()
    path = str(tmp_path / "special.bw")
    with np.errstate(invalid="ignore", over="ignore"):
        write_bigwig(data, regions, header, path, str(tmp_path / "s.txt"))
        expected = np.array(SPECIAL).astype(np.float32)
    bw = pyBigWig.open(path)
    got = bw.values("chr1", 10, 10 + len(SPECIAL), numpy=True)
    bw.close()
    assert got.tobytes() == expected.tobytes()
    # among them: the sign of a negative zero, infinity for what is
    # beyond the range, the smallest subnormal, a NaN
    assert np.signbit(got[0]) and not np.signbit(got[1])
    assert np.isnan(got[2]) and np.isposinf(got[5]) and np.isneginf(got[6])
    assert got[7] > 0 and got[8] == 0


@pytest.mark.filterwarnings("ignore:Warning. converting a masked element")
def test_values_go_through_a_double():
    as_float32 = bigwig_helper.as_float32
    # rounded twice: to the nearest double, then to the nearest float32
    big = np.array([2 ** 62 + 2 ** 38 + 1], dtype=np.int64)
    assert as_float32(big).tobytes() != big.astype(np.float32).tobytes()
    assert as_float32(big).tobytes() == np.float32(float(big[0])).tobytes()
    # a signalling NaN becomes a quiet one
    signalling = np.array([0x7f800001], dtype=np.uint32).view(np.float32)
    assert as_float32(signalling).view(np.uint32)[0] == 0x7fc00001
    # a masked value is stored as NaN
    masked = np.ma.array([1.0, 2.0], mask=[False, True])
    with pytest.warns(UserWarning):
        assert np.isnan(as_float32(masked)).tolist() == [False, True]
    # lists and arrays of objects, value by value
    assert as_float32([0.1, 2]).tolist() == [np.float32(0.1), 2.0]
    assert as_float32(np.array([0.5, 3], dtype=object)).dtype == np.float32
    for row in other_types()[2]:
        with np.errstate(invalid="ignore"):
            found = as_float32(row)
        assert found.dtype == np.float32 and found.shape == (4,)


class Recorder(object):
    """
        Stands in for a bigWig file that is being written
    """

    def __init__(self):
        self.calls = []

    def addEntries(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.mark.parametrize("runs", [True, False])
@pytest.mark.parametrize("with_numpy", [1, 0])
def test_what_pybigwig_is_given(monkeypatch, runs, with_numpy):
    monkeypatch.setattr(bigwig_helper, "RUNS", runs)
    monkeypatch.setattr(bigwig_helper.pyBigWig, "numpy", with_numpy)
    bw = Recorder()
    values = np.array([0.1, 2.0, -0.0], dtype=np.float64)
    bigwig_helper.add_run(bw, np.str_("chr1"), np.int64(7), np.int64(10),
                          values)
    bigwig_helper.add_run(bw, "chr1", 10, 10, np.array([]))
    assert len(bw.calls) == 1
    args, kwargs = bw.calls[0]
    if not runs:
        # an entry per position, as it always was
        assert args == (["chr1"] * 3, [7, 8, 9])
        assert kwargs == {"ends": [8, 9, 10], "values": [0.1, 2.0, -0.0]}
        assert all(type(v) is float for v in kwargs["values"])
        return
    assert args == ("chr1", 7) and type(args[0]) is str
    assert type(args[1]) is int
    assert sorted(kwargs) == ["span", "step", "values"]
    assert (kwargs["span"], kwargs["step"]) == (1, 1)
    expected = values.astype(np.float32)
    if with_numpy:
        assert isinstance(kwargs["values"], np.ndarray)
        assert kwargs["values"].dtype == np.float32
        assert kwargs["values"].tobytes() == expected.tobytes()
    else:
        assert type(kwargs["values"]) is list
        assert all(type(v) is float for v in kwargs["values"])
        assert np.array(kwargs["values"], dtype=np.float32).tobytes() == \
            expected.tobytes()


def test_pybigwig_without_numpy_writes_the_same_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bigwig_helper, "RUNS", True)
    header, regions, data = like_predictions(4, n_per_chrom=30)
    with_arrays = str(tmp_path / "arrays.bw")
    write_bigwig(data, regions, header, with_arrays, str(tmp_path / "a.txt"))
    monkeypatch.setattr(bigwig_helper.pyBigWig, "numpy", 0)
    with_lists = str(tmp_path / "lists.bw")
    write_bigwig(data, regions, header, with_lists, str(tmp_path / "l.txt"))
    monkeypatch.undo()
    assert same_bytes(with_arrays, with_lists)


@pytest.mark.parametrize("regions,data,error", [
    # fewer values than positions
    ([("chr1", 10, 30, 20)], [np.ones(12)], RuntimeError),
    ([("chr1", 10, 30, 20), ("chr1", 100, 120, 110)],
     [np.ones(20), np.ones(7)], RuntimeError),
    # a chromosome the header does not have
    ([("chr5", 10, 30, 20)], [np.ones(20)], KeyError),
])
def test_regions_that_cannot_be_written_fail_as_before(tmp_path, regions,
                                                       data, error):
    for write in (write_bigwig_v0, write_bigwig):
        with pytest.raises(error):
            write(data, regions, HEADER, str(tmp_path / "x.bw"),
                  str(tmp_path / "x.txt"))


@pytest.mark.parametrize("runs", [True, False])
def test_run_of_positions(tmp_path, monkeypatch, runs):
    monkeypatch.setattr(bigwig_helper, "RUNS", runs)
    path = str(tmp_path / "runs.bw")
    bw = pyBigWig.open(path, "w")
    bw.addHeader(HEADER, maxZooms=0)
    bigwig_helper.add_run(bw, "chr1", 10, 13, np.array([1.5, 2.5, 3.5]))
    # nothing to write
    bigwig_helper.add_run(bw, "chr1", 13, 13, np.array([]))
    bigwig_helper.add_run(bw, np.str_("chr1"), np.int64(13), np.int64(15),
                          [4.5, 5.5])
    # what pyBigWig would take without complaint for a run, and store
    # in the wrong place
    for start, end, values in [(-2, 1, [1.0, 2.0, 3.0]), (30, 34, [1.0]),
                               (40, 38, []), (50, 52, [1.0, 2.0, 3.0]),
                               (-1, -1, []), (60, 60, [1.0])]:
        with pytest.raises(RuntimeError, match="Cannot write"):
            bigwig_helper.add_run(bw, "chr1", start, end, np.array(values))
    bigwig_helper.add_run(bw, "chr2", 0, 2, np.array([6.5, 7.5]))
    bw.close()

    got, _ = read_all(path)
    assert got["chr1"][10:15].tolist() == [1.5, 2.5, 3.5, 4.5, 5.5]
    assert np.isnan(got["chr1"][:10]).all() and np.isnan(got["chr1"][15:]).all()
    assert got["chr2"][:2].tolist() == [6.5, 7.5]
    assert np.isnan(got["chr10"]).all()


def test_segments_of_overlapping_regions():
    regions = [("chr1", 100, 200, 150), ("chr1", 160, 260, 210),
               ("chr2", 0, 100, 50), ("chr1", 240, 340, 290),
               ("chr2", 0, 100, 50), ("chr2", 0, 100, 50)]
    order = [0, 1, 3, 2, 4, 5]
    assert bigwig_helper.region_segments(regions, order) == [
        (0, "chr1", 100, 180), (1, "chr1", 180, 250), (3, "chr1", 250, 340),
        (2, "chr2", 0, 50), (4, "chr2", 50, 50), (5, "chr2", 50, 100)]
    assert bigwig_helper.region_segments(regions, []) == []
    # a region that ends before the position reached so far
    with pytest.raises(AssertionError):
        bigwig_helper.region_segments(
            [("chr1", 0, 100, 90), ("chr1", 10, 40, 95)], [0, 1])


# --------------------------------------------------------------------
# importance_hdf5_to_bigwig
# --------------------------------------------------------------------

def make_scores(tmp_path, loci, seq_len=40, seed=0, gzipped=False,
                header=HEADER, change=None):
    """
        An importance scores file, its peak file and a chrom sizes
        file. `loci` are (chrom, peak start, summit offset); `change`
        turns the random float32 scores into the scores that are
        stored.
    """

    rng = np.random.RandomState(seed)
    n = len(loci)
    tokens = rng.randint(0, 4, size=(n, seq_len))
    one_hot = np.zeros((n, seq_len, 4), dtype=np.int8)
    for base in range(4):
        one_hot[:, :, base] = tokens == base
    hyp = rng.normal(size=(n, seq_len, 4)).astype(np.float32)
    if change is not None:
        hyp = change(hyp)

    h5_path = str(tmp_path / "scores.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("hyp_scores", data=hyp)
        f.create_dataset("input_seqs", data=one_hot)

    lines = ["\t".join(map(str, [chrom, start, start + 100, ".", 1, ".", 1,
                                 -1, -1, summit])) + "\n"
             for chrom, start, summit in loci]
    bed_path = str(tmp_path / ("peaks.bed.gz" if gzipped else "peaks.bed"))
    opener = gzip.open if gzipped else open
    with opener(bed_path, "wt") as fh:
        fh.writelines(lines)

    sizes_path = str(tmp_path / "chrom.sizes")
    with open(sizes_path, "w") as fh:
        for chrom, size in header:
            fh.write("{}\t{}\n".format(chrom, size))

    # the score of a position is the hypothetical score of the base
    # that is there
    projected = (hyp * one_hot).sum(axis=2)
    regions = [(chrom, start + summit - seq_len // 2,
                start + summit + seq_len // 2, start + summit)
               for chrom, start, summit in loci]
    return h5_path, bed_path, sizes_path, regions, projected


LOCI = [("chr2", 300, 40), ("chr1", 100, 50), ("chr1", 120, 50),
        ("chr10", 400, 10), ("chr1", 700, 0)]


@pytest.mark.parametrize("gzipped", [False, True])
def test_importance_bigwig(tmp_path, gzipped):
    h5_path, bed_path, sizes_path, regions, projected = make_scores(
        tmp_path, LOCI, gzipped=gzipped)
    out, stats = str(tmp_path / "scores.bw"), str(tmp_path / "stats.txt")
    importance_hdf5_to_bigwig(h5_path, bed_path, out, stats, sizes_path,
                              gzipped=gzipped)
    check_file(out, HEADER, regions, list(projected), stats)


def test_importance_bigwig_without_regions_is_an_error(tmp_path):
    # no region on the requested chromosome: a clear error before any
    # output file is created
    h5_path, bed_path, sizes_path, _, _ = make_scores(
        tmp_path, [l for l in LOCI if l[0] != "chr10"])
    out, stats = tmp_path / "scores.bw", tmp_path / "stats.txt"
    with pytest.raises(ValueError, match="No regions to write on "
                       "chromosome chr10"):
        importance_hdf5_to_bigwig(h5_path, bed_path, str(out), str(stats),
                                  sizes_path, debug_chr="chr10")
    assert not out.exists() and not stats.exists()


@pytest.mark.parametrize("debug_chr", ["chr1", "chr2", "chr10"])
def test_importance_bigwig_debug_chromosome(tmp_path, debug_chr):
    """
        debug_chr restricts the output to the regions of one chromosome
    """

    h5_path, bed_path, sizes_path, regions, projected = make_scores(
        tmp_path, LOCI)
    out, stats = str(tmp_path / "scores.bw"), str(tmp_path / "stats.txt")
    importance_hdf5_to_bigwig(h5_path, bed_path, out, stats, sizes_path,
                              debug_chr=debug_chr)
    keep = [i for i, r in enumerate(regions) if r[0] == debug_chr]
    assert 0 < len(keep) < len(regions)
    check_file(out, HEADER, [regions[i] for i in keep],
               [projected[i] for i in keep], stats)


# --------------------------------------------------------------------
# the importance writer against the one that stored an entry per
# position
# --------------------------------------------------------------------

def tiny_and_large(hyp):
    """
        Scores at the edges of float32: subnormal, close to the
        largest, and zeros of either sign
    """

    hyp = hyp.copy()
    hyp[0::4] *= np.float32(1e-38)
    hyp[1::4] *= np.float32(1e37)
    hyp[2::4] = np.where(hyp[2::4] > 0, np.float32(0.0), np.float32(-0.0))
    return hyp


def as_large_integers(hyp):
    # beyond 2**53: not every one of them is a double
    return (np.clip(hyp.astype(np.float64), -3, 3) * 2 ** 61).astype(
        np.int64) | 1


SCORES = {"ordinary": None, "tiny and large": tiny_and_large,
          "64 bit integers": as_large_integers}


@pytest.mark.parametrize("runs", [True, False])
@pytest.mark.parametrize("scores", sorted(SCORES))
@pytest.mark.parametrize("debug_chr", [None, "chr2"])
def test_importance_file_content_as_an_entry_per_position(
        tmp_path, monkeypatch, scores, runs, debug_chr):
    """
        The importance writer against its previous version (kept in
        tests/legacy_reference.py), on enough loci for zoom levels:
        the same values, stored intervals, total summary, zoom
        records and statistics file
    """

    monkeypatch.setattr(bigwig_helper, "RUNS", runs)
    header = [("chr1", 40000), ("chr2", 30000)]
    rng = np.random.RandomState(11)
    loci = [(header[int(rng.randint(2))][0], int(rng.randint(200, 29000)),
             int(rng.randint(0, 100))) for _ in range(150)]
    h5_path, bed_path, sizes_path, _, _ = make_scores(
        tmp_path, loci, seq_len=200, seed=12, header=header,
        change=SCORES[scores])

    paths = {}
    for name, write in (("new", importance_hdf5_to_bigwig),
                        ("old", importance_hdf5_to_bigwig_v0)):
        paths[name] = str(tmp_path / (name + ".bw"))
        with np.errstate(invalid="ignore", over="ignore"):
            write(h5_path, bed_path, paths[name],
                  str(tmp_path / (name + ".txt")), sizes_path,
                  debug_chr=debug_chr)
    found = assert_same_content(paths["new"], paths["old"])
    with open(str(tmp_path / "new.txt")) as a, \
            open(str(tmp_path / "old.txt")) as b:
        assert a.read() == b.read()
    assert len(found["zooms"]) >= 1 and len(found["zooms"][0][1]) > 10
    written = [len(intervals) for intervals in found["intervals"]]
    assert written[1] > 1000
    assert (written[0] == 0) == (debug_chr == "chr2")
    if not runs:
        assert same_bytes(paths["new"], paths["old"])

