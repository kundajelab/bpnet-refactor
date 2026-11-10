"""
    Time to construct one training batch, on real data, with the
    present code and with one piece of it replaced by the
    implementation it had before.

        python benchmarks/bench_batch_generation.py \\
            --input-data input.json --reference-genome genome.fa \\
            --chrom-sizes genome.chrom.sizes --chroms chr1 chr2 ...

    Run from the repository root. `input.json` is the input-data file
    of bpnet-train (signal and control bigWigs, peaks, background). A
    batch is constructed the way a worker process of the trainer does
    it (64 loci and their reverse complements, 2,114 bases, jitter
    128): files are opened, sequences fetched and encoded, signal and
    control values read, reverse complements made. Both variants
    construct the same rows with the same jitters, in alternating
    order.

    Not included: starting the worker processes, preparing and
    shuffling the epoch, several workers competing for the disk, the
    queues between the workers and the trainer, and everything
    TensorFlow does with a batch. The result is the speed of batch
    construction, not of training.
"""

import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.generators import generators, sequtils
from tests import legacy_reference

#: the pieces that were replaced by faster ones: (what, the module and
#: the name the batch generator looks it up under, the implementation
#: it had before)
REPLACED = [
    ("one hot encoder", sequtils, "one_hot_encode",
     legacy_reference.one_hot_encode_v0),
    ("bigWig reading", generators, "interval_values",
     legacy_reference.interval_values_v0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--reference-genome", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--chroms", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=30)
    args = parser.parse_args()

    params = {"sequence_generator_name": "BPNet", "input_seq_len": 2114,
              "output_len": 1000, "rev_comp_aug": True, "max_jitter": 128,
              "shuffle": True, "mode": "train"}
    gen = generators.MBPNetSequenceGenerator(
        args.input_data, params, args.reference_genome, args.chrom_sizes,
        args.chroms, num_threads=2, batch_size=args.batch_size, epochs=1)
    rows = gen.get_resized_samples()[0]
    print("{} loci in the epoch; the construction of {} batches of {} (x2 "
          "with reverse complements) is timed".format(
              len(rows), args.batches, args.batch_size))

    warmup = 2

    def batch_rows(start):
        # the rows of a batch as a worker process prepares them
        rows_of_batch = rows.iloc[start:start + args.batch_size][
            ['chrom', 'pos', 'weight']].copy()
        rows_of_batch['status'] = 1
        rows_of_batch.loc[rows_of_batch['weight'] == 0, 'status'] = -1
        rows_of_batch['idx'] = rows_of_batch.index.values
        return rows_of_batch

    for what, module, name, previous in REPLACED:
        present = getattr(module, name)
        seconds = {"old": [], "new": []}
        calls = [0]
        position = [0]

        def timed(variant, function):
            def run(*arguments):
                start = time.perf_counter()
                result = function(*arguments)
                # the calls of the warm up are not part of the timing
                if calls[0] >= 2 * warmup:
                    seconds[variant].append(time.perf_counter() - start)
                return result
            return run

        def construct(variant, function):
            def run():
                start = position[0] % (len(rows) - args.batch_size)
                setattr(module, name, timed(variant, function))
                # the same jitters for both variants
                random.seed(start)
                try:
                    gen._generate_batch(batch_rows(start))
                finally:
                    setattr(module, name, present)
                # compare() calls the two alternately; after every
                # second call both have constructed the current slice
                # of rows, and the next slice is taken
                calls[0] += 1
                if calls[0] % 2 == 0:
                    position[0] += args.batch_size
            return run

        result = compare(construct("old", previous),
                         construct("new", present), rounds=args.batches,
                         warmup=warmup)
        report("batch, previous {}".format(what), result)
        per_batch = len(seconds["new"]) / float(args.batches)
        for variant in ("old", "new"):
            # time inside the piece, summed over its calls in a batch
            inside = np.sum(seconds[variant]) / args.batches
            print("  {} {}: {:.2f} ms per batch ({:.0f} calls), {:.0f}% of "
                  "the batch time".format(
                      variant, what, 1e3 * inside, per_batch,
                      100 * inside / result[variant + "_seconds"]))


if __name__ == "__main__":
    main()
