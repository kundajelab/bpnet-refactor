"""
    Time a worker spends on one training batch, on real data, with the
    one hot encoder it had before and with the present one.

        python benchmarks/bench_batch_generation.py \\
            --input-data input.json --reference-genome genome.fa \\
            --chrom-sizes genome.chrom.sizes --chroms chr1 chr2 ...

    Run from the repository root. `input.json` is the input-data file
    of bpnet-train (signal and control bigWigs, peaks, background). A
    batch is generated the way a worker process of the trainer does it
    (64 loci and their reverse complements, 2,114 bases, jitter 128):
    sequences are fetched and encoded, signal and control values are
    read. Both variants generate the same rows, in alternating order.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.timing import compare, report
from bpnet.generators import generators, sequtils
from tests.legacy_reference import one_hot_encode_v0


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
    print("{} loci in the epoch; {} batches of {} (x2 with reverse "
          "complements) are timed".format(
              len(rows), args.batches, args.batch_size))

    present = sequtils.one_hot_encode
    encode_seconds = {"old": [], "new": []}

    def timed_encoder(name, function):
        def encode(sequences, seq_length):
            start = time.perf_counter()
            result = function(sequences, seq_length)
            encode_seconds[name].append(time.perf_counter() - start)
            return result
        return encode

    position = [0]

    def batch(name, function):
        def run():
            start = position[0] % (len(rows) - args.batch_size)
            # the rows of a batch as a worker process prepares them
            batch_rows = rows.iloc[start:start + args.batch_size][
                ['chrom', 'pos', 'weight']].copy()
            batch_rows['status'] = 1
            batch_rows.loc[batch_rows['weight'] == 0, 'status'] = -1
            batch_rows['idx'] = batch_rows.index.values
            sequtils.one_hot_encode = timed_encoder(name, function)
            try:
                gen._generate_batch(batch_rows)
            finally:
                sequtils.one_hot_encode = present
        return run

    old, new = batch("old", one_hot_encode_v0), batch("new", present)

    # compare() calls the two alternately; after every second call
    # both have generated the current slice of rows, and the next
    # slice is taken
    calls = [0]

    def paired(function):
        def run():
            function()
            calls[0] += 1
            if calls[0] % 2 == 0:
                position[0] += args.batch_size
        return run

    result = compare(paired(old), paired(new), rounds=args.batches, warmup=2)
    report("one training batch in a worker", result)
    for name in ("old", "new"):
        share = np.median(encode_seconds[name]) / result[name + "_seconds"]
        print("  {} encoder: {:.2f} ms per batch, {:.0f}% of the batch "
              "time".format(name, 1e3 * np.median(encode_seconds[name]),
                            100 * share))


if __name__ == "__main__":
    main()
