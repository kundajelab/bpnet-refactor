"""
    Speed of one_hot_encode against the implementation it replaced.

        python benchmarks/bench_one_hot.py

    Run from the repository root. Batches have the shape used in
    training: 128 sequences (64 loci and their reverse complements) of
    2,114 bases. This times the encoder alone; what it means for the
    time a worker spends on a batch is measured on real data by
    bench_batch_generation.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.generators.sequtils import one_hot_encode
from tests.legacy_reference import one_hot_encode_v0


def main():
    rng = np.random.RandomState(0)
    for n, length in ((128, 2114), (2048, 2114), (64, 546)):
        sequences = ["".join(rng.choice(list("ACGTN"), size=length,
                                        p=[0.245] * 4 + [0.02]))
                     for _ in range(n)]
        old = one_hot_encode_v0(sequences, length)
        assert np.array_equal(old, one_hot_encode(sequences, length))
        assert old.dtype == one_hot_encode(sequences, length).dtype

        name = "{} x {}".format(n, length)
        report(name + ", float64 (default)", compare(
            lambda: one_hot_encode_v0(sequences, length),
            lambda: one_hot_encode(sequences, length)))
        # not like for like: the previous implementation could only
        # return float64
        report(name + ", int8 (old: float64)", compare(
            lambda: one_hot_encode_v0(sequences, length),
            lambda: one_hot_encode(sequences, length, dtype=np.int8)))


if __name__ == "__main__":
    main()
