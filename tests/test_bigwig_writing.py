"""
    Characterization tests for the two bigWig writers:
    bpnet.utils.bigwig_helper.write_bigwig (predicted profiles) and
    bpnet.utils.importance_hdf5_to_bigwig (contribution scores)
"""

import gzip

import h5py
import numpy as np
import pyBigWig
import pytest

from bpnet.utils.bigwig_helper import write_bigwig
from bpnet.utils.importance_hdf5_to_bigwig import importance_hdf5_to_bigwig

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


# --------------------------------------------------------------------
# importance_hdf5_to_bigwig
# --------------------------------------------------------------------

def make_scores(tmp_path, loci, seq_len=40, seed=0, gzipped=False):
    """
        An importance scores file, its peak file and a chrom sizes
        file. `loci` are (chrom, peak start, summit offset).
    """

    rng = np.random.RandomState(seed)
    n = len(loci)
    tokens = rng.randint(0, 4, size=(n, seq_len))
    one_hot = np.zeros((n, seq_len, 4), dtype=np.int8)
    for base in range(4):
        one_hot[:, :, base] = tokens == base
    hyp = rng.normal(size=(n, seq_len, 4)).astype(np.float32)

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
        for chrom, size in HEADER:
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


@pytest.mark.xfail(strict=True, reason="the debug_chr filter looks at a "
                   "region picked by a leftover loop variable instead of "
                   "the current region")
def test_importance_bigwig_debug_chromosome(tmp_path):
    h5_path, bed_path, sizes_path, regions, projected = make_scores(
        tmp_path, LOCI)
    out, stats = str(tmp_path / "scores.bw"), str(tmp_path / "stats.txt")
    importance_hdf5_to_bigwig(h5_path, bed_path, out, stats, sizes_path,
                              debug_chr="chr1")
    keep = [i for i, r in enumerate(regions) if r[0] == "chr1"]
    check_file(out, HEADER, [regions[i] for i in keep],
               [projected[i] for i in keep], stats)
