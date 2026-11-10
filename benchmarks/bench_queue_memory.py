"""
    Memory of the batch generator when its consumer is slower than its
    workers, on real data, with the queues as they are (room for a few
    batches) and without a bound (as they were).

        python benchmarks/bench_queue_memory.py \\
            --input-data input.json --reference-genome genome.fa \\
            --chrom-sizes genome.chrom.sizes --chroms chr1 chr2 ... \\
            --threads 2 10

    Run from the repository root. The consumer takes a training batch
    (64 loci and their reverse complements, 2,114 bases, int8
    sequences) every `--consumer-ms` milliseconds, about what a GPU
    step takes, for `--batches` batches of one epoch. Reported: the
    largest number of collected batches that waited for the consumer,
    and the resident memory of the consuming process and of the
    workers together (sampled every tenth batch). Each case runs in a
    process of its own.
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bpnet.generators import generators


def resident_mb(pid="self"):
    try:
        with open("/proc/{}/status".format(pid)) as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def one_case(args, threads, bounded):
    waiting = []

    class Watched(generators.Queue):
        def put(self, item, *a, **k):
            super(Watched, self).put(item, *a, **k)
            waiting.append(self.qsize())

    generators.Queue = Watched
    if not bounded:
        generators.WORKER_QUEUE_BATCHES = 0
        generators.consumer_queue_batches = lambda workers: 0

    params = {"sequence_generator_name": "BPNet", "input_seq_len": 2114,
              "output_len": 1000, "rev_comp_aug": True, "max_jitter": 128,
              "shuffle": True, "mode": "train", "sequence_dtype": "int8"}
    gen = generators.MBPNetSequenceGenerator(
        args.input_data, params, args.reference_genome, args.chrom_sizes,
        args.chroms, num_threads=threads, batch_size=64, epochs=1)
    start_mb = resident_mb()
    most_consumer, most_workers, taken = start_mb, 0.0, 0
    batches = gen.gen()
    try:
        for batch in batches:
            time.sleep(args.consumer_ms / 1000.0)
            taken += 1
            if taken % 10 == 0:
                most_consumer = max(most_consumer, resident_mb())
                most_workers = max(most_workers, sum(
                    resident_mb(child.pid)
                    for child in multiprocessing.active_children()))
            if taken >= args.batches:
                break
    finally:
        batches.close()
    print("{:2d} workers, {:9s} queues: {:4d} of {} batches taken; most "
          "batches waiting for the consumer {:4d}; consumer {:5.0f} -> "
          "{:5.0f} MB; workers together up to {:5.0f} MB".format(
              threads, "bounded" if bounded else "unbounded", taken,
              gen.len(), max(waiting), start_mb, most_consumer, most_workers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--reference-genome", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--chroms", nargs="+", required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[2])
    parser.add_argument("--consumer-ms", type=float, default=40.0)
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.case:
        threads, queues = args.case.split(",")
        one_case(args, int(threads), queues == "bounded")
        return
    for threads in args.threads:
        for queues in ("unbounded", "bounded"):
            subprocess.check_call([sys.executable] + sys.argv + [
                "--case", "{},{}".format(threads, queues)])


if __name__ == "__main__":
    main()
