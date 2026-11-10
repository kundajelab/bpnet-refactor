"""
    Time to construct one training batch, on real data, with the
    present code and with pieces of it replaced by the implementations
    they had before.

        python benchmarks/bench_batch_generation.py \\
            --input-data input.json --reference-genome genome.fa \\
            --chrom-sizes genome.chrom.sizes --chroms chr1 chr2 ...

    Run from the repository root. `input.json` is the input-data file
    of bpnet-train (signal and control bigWigs, peaks, background). A
    batch is constructed the way a worker process of the trainer does
    it (64 loci and their reverse complements, 2,114 bases, jitter
    128): sequences fetched and encoded, signal and control values
    read, reverse complements made. Both variants construct the same
    rows with the same jitters, in alternating order.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.generators import generators, sequtils, windows
from tests import legacy_reference


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

    def batch_rows(start):
        # the rows of a batch as a worker process prepares them
        rows_of_batch = rows.iloc[start:start + args.batch_size][
            ['chrom', 'pos', 'weight']].copy()
        rows_of_batch['status'] = 1
        rows_of_batch.loc[rows_of_batch['weight'] == 0, 'status'] = -1
        rows_of_batch['idx'] = rows_of_batch.index.values
        return rows_of_batch

    def previous_construction(rows_of_batch):
        # rows one by one, one bigWig read per row and file through a
        # Python list, all files opened for every batch
        legacy_reference.generate_batch_v0(gen, rows_of_batch)

    def row_by_row_with_array_reads(rows_of_batch):
        # the same, with every read handing over a float32 array: the
        # construction right before all windows of a batch were read
        # in one call
        through_lists = legacy_reference.interval_values_v0
        legacy_reference.interval_values_v0 = windows.interval_values
        try:
            legacy_reference.generate_batch_v0(gen, rows_of_batch)
        finally:
            legacy_reference.interval_values_v0 = through_lists

    def with_previous_encoder(construct):
        def run(rows_of_batch):
            present = sequtils.one_hot_encode
            sequtils.one_hot_encode = legacy_reference.one_hot_encode_v0
            try:
                construct(rows_of_batch)
            finally:
                sequtils.one_hot_encode = present
        return run

    comparisons = [
        ("everything as it was", with_previous_encoder(previous_construction)),
        ("bigWigs read row by row through lists", previous_construction),
        ("bigWigs read row by row as arrays", row_by_row_with_array_reads),
        ("previous one hot encoder",
         with_previous_encoder(gen._generate_batch)),
    ]
    for what, previous in comparisons:
        calls = [0]
        position = [0]

        def construct(function):
            def run():
                start = position[0] % (len(rows) - args.batch_size)
                # the same jitters for both variants
                random.seed(start)
                function(batch_rows(start))
                # compare() calls the two alternately; after every
                # second call both have constructed the current slice
                # of rows, and the next slice is taken
                calls[0] += 1
                if calls[0] % 2 == 0:
                    position[0] += args.batch_size
            return run

        result = compare(construct(previous), construct(gen._generate_batch),
                         rounds=args.batches, warmup=2)
        report("batch, {}".format(what), result)


if __name__ == "__main__":
    main()
