"""
    A worker process of the batch generator that fails, or dies, ends
    the epoch with an error that names the cause. (Before, the
    consumer of the batches waited forever.)
"""

import multiprocessing
import os
import re
import signal
import time

import numpy as np
import pyBigWig
import pytest

from bpnet.generators import generators
from bpnet.utils.exceptionhandler import NoTracebackException

from tests import synth

#: seconds a generator gets before the test declares it hung
TIME_LIMIT = 120

SMALL = dict(chrom_len=20000, n_peaks=60, n_background=60)


def make_generator(dataset, **kwargs):
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": dataset["input_len"],
              "output_len": dataset["output_len"], "rev_comp_aug": False,
              "max_jitter": 0, "shuffle": False, "mode": "val"}
    kwargs.setdefault("num_threads", 2)
    kwargs.setdefault("batch_size", 8)
    return generators.MBPNetSequenceGenerator(
        dataset["input_json"], params, dataset["fasta"],
        dataset["chrom_sizes"], dataset["chroms"], epochs=2, **kwargs)


def consume(generator, queue):
    """
        Runs in a process of its own: one epoch of the generator; what
        happened goes into the queue
    """

    try:
        batches = sum(1 for _ in generator.gen())
        queue.put(("finished", batches))
    except NoTracebackException as error:
        queue.put(("failed", str(error)))
    except BaseException as error:  # pragma: no cover
        queue.put(("other", repr(error)))


def outcome(generator):
    """
        ('finished', number of batches) or ('failed', message) of one
        epoch, found out in a child process so that a generator that
        hangs fails the test instead of stopping the test run
    """

    queue = multiprocessing.Queue()
    child = multiprocessing.Process(target=consume, args=(generator, queue))
    child.start()
    try:
        result = queue.get(timeout=TIME_LIMIT)
    except Exception:
        result = ("hung", "no result within {} s".format(TIME_LIMIT))
    child.join(10)
    if child.is_alive():
        os.kill(child.pid, signal.SIGKILL)
        child.join()
    return result


class FailsAtBatch(generators.MBPNetSequenceGenerator):
    """
        A generator whose batch construction stops working at one batch
        of one worker process
    """

    fail_at = 2
    how = "exception"

    def _generate_batch(self, coords):
        # each worker process counts its own batches
        self._made = getattr(self, "_made", 0) + 1
        first_of_worker_0 = coords.index[0] == self._first_index
        if self._made == self.fail_at and first_of_worker_0:
            if self.how == "exception":
                raise ValueError("no values for this batch")
            if self.how == "exit":
                os._exit(7)
            os.kill(os.getpid(), signal.SIGKILL)
        return super(FailsAtBatch, self)._generate_batch(coords)


def failing_generator(dataset, how):
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": dataset["input_len"],
              "output_len": dataset["output_len"], "rev_comp_aug": False,
              "max_jitter": 0, "shuffle": False, "mode": "val"}
    generator = FailsAtBatch(
        dataset["input_json"], params, dataset["fasta"],
        dataset["chrom_sizes"], dataset["chroms"], num_threads=2,
        batch_size=8, epochs=2)
    generator.how = how
    # the second batch of the first worker: its rows start 8 rows into
    # the epoch
    generator._first_index = generator.get_resized_samples()[0].index[8]
    return generator


def test_an_epoch_without_failures(synth_dataset):
    generator = make_generator(synth_dataset)
    kind, batches = outcome(generator)
    assert kind == "finished"
    assert batches == generator.len() > 4


def test_exception_in_a_worker_is_reported(synth_dataset):
    start = time.time()
    kind, message = outcome(failing_generator(synth_dataset, "exception"))
    assert kind == "failed", message
    assert "Batch generation failed in worker process 0 (val mode)" in message
    # with the worker's own traceback
    assert "ValueError: no values for this batch" in message
    assert "_generate_batch" in message
    assert time.time() - start < TIME_LIMIT


@pytest.mark.parametrize("how,exit_code", [("exit", 7), ("kill", -9)])
def test_worker_that_dies_is_reported(synth_dataset, how, exit_code):
    kind, message = outcome(failing_generator(synth_dataset, how))
    assert kind == "failed", message
    assert "Batch generation failed in worker process 0" in message
    # the batch the worker made before it died may or may not have
    # left the process
    assert re.search(r"the process ended \(exit code {}\) after [01] of its "
                     r"\d+ batches".format(exit_code), message), message


@pytest.mark.parametrize("how", ["exit", "kill"])
def test_death_of_a_worker_is_noticed_at_once(synth_dataset, monkeypatch,
                                              how):
    """
        The end of a worker process shows as the end of its queue; it
        is not found by looking at the process now and then (which
        would also not help when the worker dies while it hands over a
        batch: the rest of that batch would be waited for forever)
    """

    monkeypatch.setattr(generators, "WORKER_POLL_SECONDS", 60.0)
    start = time.time()
    kind, message = outcome(failing_generator(synth_dataset, how))
    assert kind == "failed", message
    assert time.time() - start < 30


def test_no_worker_is_left_behind(synth_dataset):
    generator = failing_generator(synth_dataset, "exception")
    with pytest.raises(NoTracebackException):
        for _ in generator.gen():
            pass
    # the other worker was stopped and collected
    deadline = time.time() + 30
    while multiprocessing.active_children() and time.time() < deadline:
        time.sleep(0.2)
    assert multiprocessing.active_children() == []


def test_window_beyond_the_chromosome_of_a_bigwig(tmp_path):
    """
        A signal file whose chromosome is shorter than the genome's:
        loci beyond its end cannot be read. The epoch fails with the
        reader's error instead of hanging.
    """

    dataset = synth.make_dataset(str(tmp_path), **SMALL)
    path = dataset["signal"][0]
    short = {chrom: 6000 for chrom in dataset["chroms"]}
    tracks = {chrom: values[:6000]
              for chrom, values in dataset["tracks"][path].items()}
    synth.write_bigwig(path, short, tracks)
    assert pyBigWig.open(path).chroms(dataset["chroms"][0]) == 6000

    kind, message = outcome(make_generator(dataset))
    assert kind == "failed", message
    assert "Batch generation failed in worker process" in message
    assert "RuntimeError" in message
