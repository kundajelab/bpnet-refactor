"""
    Speed of the dinucleotide shuffle (the references of bpnet-shap)
    against the implementation it replaced.

        python benchmarks/bench_dinuc_shuffle.py

    Run from the repository root. One call makes the 20 shuffles of
    one sequence of 2,114 bases, as bpnet-shap does for every peak.
    The shuffles are the same; the walk that builds a shuffle is done
    over lists instead of numpy arrays one element at a time.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.utils.shaputils import dinuc_shuffle
from tests.legacy_reference import dinuc_shuffle_v0


def main():
    rng = np.random.RandomState(0)
    for length, shuffles in ((2114, 20), (2114, 1), (546, 20)):
        sequence = np.zeros((length, 4), dtype=np.int8)
        sequence[np.arange(length), rng.randint(0, 4, size=length)] = 1

        def old():
            return dinuc_shuffle_v0(sequence, shuffles,
                                    np.random.RandomState(1))

        def new():
            return dinuc_shuffle(sequence, shuffles,
                                 np.random.RandomState(1))

        assert np.array_equal(old(), new())
        report("{} shuffles of {} bases".format(shuffles, length),
               compare(old, new))


if __name__ == "__main__":
    main()
