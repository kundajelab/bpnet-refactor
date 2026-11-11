"""
    Speed of the dinucleotide shuffle (the references of bpnet-shap)
    against the implementation it replaced.

        python benchmarks/bench_dinuc_shuffle.py

    Run from the repository root. One call makes the 20 shuffles of
    one sequence of 2,114 bases, as bpnet-shap does for every peak.
    The shuffles are the same in all three; the walk that builds a
    shuffle is done in Python, or by a function that numba compiles
    once per process (its first call is timed separately).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.utils import shaputils
from bpnet.utils.shaputils import dinuc_shuffle
from tests.legacy_reference import dinuc_shuffle_v0


def main():
    rng = np.random.RandomState(0)
    sequence = np.zeros((2114, 4), dtype=np.int8)
    sequence[np.arange(2114), rng.randint(0, 4, size=2114)] = 1

    def old():
        return dinuc_shuffle_v0(sequence, 20, np.random.RandomState(1))

    def new(compiled):
        return dinuc_shuffle(sequence, 20, np.random.RandomState(1),
                             compiled=compiled)

    start = time.perf_counter()
    available = shaputils._compiled_walk() is not None
    print("numba: {} (first use {:.2f} s)".format(
        "compiles the walk" if available else "not available",
        time.perf_counter() - start))
    assert np.array_equal(old(), new(False))
    assert np.array_equal(old(), new(True))

    report("20 shuffles of 2,114 bases, walk in Python",
           compare(old, lambda: new(False)))
    if available:
        report("20 shuffles of 2,114 bases, compiled walk",
               compare(old, lambda: new(True)))


if __name__ == "__main__":
    main()
