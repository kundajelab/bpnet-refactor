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

    What the number is and is not: the rate at which batches come out
    of the generator once it runs, which is set by the slower of the
    workers (constructing and pickling) and the consuming process
    (reading and unpickling, one thread per worker under one
    interpreter lock). The two generators do not draw the same
    background loci or jitters. Starting the workers of an epoch and
    ending them are not timed.

    With `--compare queues` both generators make int8 sequences, and
    one of them gets queues without a bound, as they were before the
    workers and the collecting threads waited for room: the bound
    should not cost batches per second.
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
    parser.add_argument("--compare", choices=["types", "queues"],
                        default="types")
    args = parser.parse_args()
    first, second = ("float64", "int8") if args.compare == "types" else \
        ("unbounded", "bounded")

    def generator(threads, dtype):
        params = {"sequence_generator_name": "BPNet", "input_seq_len": 2114,
                  "output_len": 1000, "rev_comp_aug": True,
                  "max_jitter": 128, "shuffle": True, "mode": "train",
                  "sequence_dtype": dtype}
        return generators.MBPNetSequenceGenerator(
            args.input_data, params, args.reference_genome, args.chrom_sizes,
            args.chroms, num_threads=threads, batch_size=args.batch_size,
            epochs=1)

    bounds = (generators.WORKER_QUEUE_BATCHES,
              generators.consumer_queue_batches)

    def rate(gen, skip, unbounded=False):
        """
            Batches per second over `--batches` batches of an epoch,
            after its first `skip`; megabytes of a batch
        """

        if unbounded:
            generators.WORKER_QUEUE_BATCHES = 0
            generators.consumer_queue_batches = lambda workers: 0
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
            generators.WORKER_QUEUE_BATCHES, \
                generators.consumer_queue_batches = bounds

    for threads in args.threads:
        gens = {name: generator(
            threads, name if args.compare == "types" else "int8")
            for name in (first, second)}
        needed = args.batches + 2 * threads
        if gens[second].len() < needed:
            raise SystemExit("an epoch has {} batches; {} are needed".format(
                gens[second].len(), needed))
        rates = {first: [], second: []}
        sizes = {}
        for i in range(args.rounds):
            order = (first, second) if i % 2 == 0 else (second, first)
            for name in order:
                found, sizes[name] = rate(gens[name], 2 * threads,
                                          unbounded=name == "unbounded")
                rates[name].append(found)
        old, new = np.array(rates[first]), np.array(rates[second])
        ratios = new / old
        rng = np.random.RandomState(0)
        medians = [np.median(ratios[rng.randint(0, len(ratios),
                                                size=len(ratios))])
                   for _ in range(2000)]
        print("{:2d} workers: {} {:6.1f} batches per second ({:.1f} MB "
              "per batch)   {} {:6.1f} ({:.1f} MB)   {:.2f}x "
              "[{:.2f}, {:.2f}]".format(
                  threads, first, np.median(old), sizes[first], second,
                  np.median(new), sizes[second], np.median(ratios),
                  np.percentile(medians, 2.5), np.percentile(medians, 97.5)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
