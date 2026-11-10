"""
    Paired timing of two implementations of the same function.

    The two are run in alternating order (A B B A ...) so that drift of
    the machine (other jobs, frequency scaling, caches) affects both
    alike, and the comparison is made within pairs.
"""

import time

import numpy as np


def compare(old, new, rounds=30, inner=1, warmup=2, seed=0):
    """
        Time `old()` and `new()` in alternating order

        Args:
            old (callable): the reference implementation
            new (callable): the candidate
            rounds (int): number of (old, new) pairs
            inner (int): calls per timing
            warmup (int): untimed calls of each before timing starts
            seed (int): seed of the bootstrap

        Returns:
            dict: median seconds per call of each, the median speedup
                (old / new within pairs) and its 95% bootstrap interval
    """

    for _ in range(warmup):
        old()
        new()

    def timed(function):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        return (time.perf_counter() - start) / inner

    t_old, t_new = [], []
    for i in range(rounds):
        if i % 2 == 0:
            a, b = timed(old), timed(new)
        else:
            b, a = timed(new), timed(old)
        t_old.append(a)
        t_new.append(b)
    t_old, t_new = np.array(t_old), np.array(t_new)

    ratios = t_old / t_new
    rng = np.random.RandomState(seed)
    medians = [np.median(ratios[rng.randint(0, rounds, size=rounds)])
               for _ in range(2000)]
    return {
        "old_seconds": float(np.median(t_old)),
        "new_seconds": float(np.median(t_new)),
        "speedup": float(np.median(ratios)),
        "speedup_low": float(np.percentile(medians, 2.5)),
        "speedup_high": float(np.percentile(medians, 97.5)),
        "rounds": rounds,
    }


def report(name, result):
    print("{:40s} old {:9.3f} ms   new {:9.3f} ms   speedup {:6.1f}x "
          "[{:.1f}, {:.1f}]".format(
              name, 1e3 * result["old_seconds"], 1e3 * result["new_seconds"],
              result["speedup"], result["speedup_low"],
              result["speedup_high"]))
