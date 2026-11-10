"""
    Time to write the predicted profiles of a run to a bigWig file,
    with the present writer (runs of positions) and with the one it
    replaced (an entry per position; tests/legacy_reference.py).

        python benchmarks/bench_bigwig_writing.py [--regions 5000]

    Run from the repository root. The regions are windows of 1,000
    bases at random places on five chromosomes, many of them
    overlapping their neighbours, with random float32 values, the
    shape of what bpnet-predict writes. Both writers get the same
    regions, in alternating order. The sizes of the two files and
    whether they hold the same values are reported as well.
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pyBigWig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.utils.bigwig_helper import write_bigwig
from tests.legacy_reference import write_bigwig_v0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.RandomState(0)
    size = 150000000
    header = [("chr{}".format(i), size) for i in range(1, 6)]
    # starts on a grid of 50 bases within the first part of each
    # chromosome, so that many windows overlap; no two at one place
    # (the previous writer fails on three)
    span = max(args.regions * args.width // 10, 4 * args.width)
    slots = span // 50
    places = rng.choice(len(header) * slots, size=args.regions, replace=False)
    regions = [(header[int(p) // slots][0], int(p) % slots * 50,
                int(p) % slots * 50 + args.width,
                int(p) % slots * 50 + args.width // 2) for p in places]
    data = rng.gamma(0.3, 2.0, size=(args.regions, args.width)).astype(
        np.float32)

    directory = tempfile.mkdtemp()
    try:
        paths = {name: os.path.join(directory, name + ".bw")
                 for name in ("old", "new")}

        def write(function, name):
            def run():
                function(data, regions, header, paths[name],
                         os.path.join(directory, name + ".txt"))
            return run

        # the progress bars of the writers go to stderr
        result = compare(write(write_bigwig_v0, "old"),
                         write(write_bigwig, "new"), rounds=args.rounds,
                         warmup=0)
        report("bigWig of {} regions of {} bases".format(
            args.regions, args.width), result)

        same = True
        old, new = pyBigWig.open(paths["old"]), pyBigWig.open(paths["new"])
        for chrom, _ in header:
            a = old.values(chrom, 0, span + args.width, numpy=True)
            b = new.values(chrom, 0, span + args.width, numpy=True)
            same = same and a.tobytes() == b.tobytes()
        same = same and old.header() == new.header()
        print("file size: old {:.1f} MB, new {:.1f} MB; same values and "
              "header statistics: {}".format(
                  os.path.getsize(paths["old"]) / 1e6,
                  os.path.getsize(paths["new"]) / 1e6, same))
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
