"""
    Batches per second that reach the consumer of the batch generator,
    on real data, with the one hot sequences as float64 (the
    generator's default) and as int8 (what bpnet-train and
    bpnet-predict ask for).

        python benchmarks/bench_batch_delivery.py \\
            --input-data input.json --reference-genome genome.fa \\
            --chrom-sizes genome.chrom.sizes --chroms chr1 chr2 ... \\
            --threads 2 10

    Run from the repository root. `input.json` is the input-data file
    of bpnet-train. Worker processes construct the batches of a
    training epoch (64 loci and their reverse complements, 2,114
    bases, jitter 128) and hand them over through their queues, as in
    training; the consumer here takes each batch and does nothing
    with it. So this is what the data path can deliver, not the speed
    of training: a model that takes longer per step than a batch
    takes to arrive is not held up by the data path at all.

    A timing starts after the first batches of an epoch have arrived
    and covers the next `--batches`; the epoch is then ended early.
    The two types are timed in alternating order.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpnet.generators import generators


def megabytes(batch):
    inputs, outputs, weights = batch
    arrays = list(inputs.values()) + list(outputs.values()) + [weights]
    return sum(a.nbytes for a in arrays) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--reference-genome", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--chroms", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, nargs="+", default=[2])
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    def generator(threads, dtype):
        params = {"sequence_generator_name": "BPNet", "input_seq_len": 2114,
                  "output_len": 1000, "rev_comp_aug": True,
                  "max_jitter": 128, "shuffle": True, "mode": "train",
                  "sequence_dtype": dtype}
        return generators.MBPNetSequenceGenerator(
            args.input_data, params, args.reference_genome, args.chrom_sizes,
            args.chroms, num_threads=threads, batch_size=args.batch_size,
            epochs=1)

    def rate(gen, skip):
        """
            Batches per second over `--batches` batches of an epoch,
            after its first `skip`; megabytes of a batch
        """

        batches = gen.gen()
        try:
            for _ in range(skip):
                size = megabytes(next(batches))
            start = time.perf_counter()
            for _ in range(args.batches):
                next(batches)
            return args.batches / (time.perf_counter() - start), size
        finally:
            # ends the workers of the epoch
            batches.close()

    for threads in args.threads:
        gens = {dtype: generator(threads, dtype)
                for dtype in ("float64", "int8")}
        needed = args.batches + 2 * threads
        if gens["int8"].len() < needed:
            raise SystemExit("an epoch has {} batches; {} are needed".format(
                gens["int8"].len(), needed))
        rates = {"float64": [], "int8": []}
        sizes = {}
        for i in range(args.rounds):
            order = ("float64", "int8") if i % 2 == 0 else ("int8", "float64")
            for dtype in order:
                found, sizes[dtype] = rate(gens[dtype], 2 * threads)
                rates[dtype].append(found)
        old, new = np.array(rates["float64"]), np.array(rates["int8"])
        ratios = new / old
        rng = np.random.RandomState(0)
        medians = [np.median(ratios[rng.randint(0, len(ratios),
                                                size=len(ratios))])
                   for _ in range(2000)]
        print("{:2d} workers: float64 {:6.1f} batches per second ({:.1f} MB "
              "per batch)   int8 {:6.1f} ({:.1f} MB)   {:.2f}x "
              "[{:.2f}, {:.2f}]".format(
                  threads, np.median(old), sizes["float64"], np.median(new),
                  sizes["int8"], np.median(ratios),
                  np.percentile(medians, 2.5), np.percentile(medians, 97.5)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
