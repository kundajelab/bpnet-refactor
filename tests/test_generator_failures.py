"""
    A worker process of the batch generator that fails, or dies, ends
    the epoch with an error that names the cause. (Before, the
    consumer of the batches waited forever.) And an epoch that ends
    early, because of that or because its consumer stops taking
    batches, leaves no worker process or thread behind.
"""

import gc
import itertools
import multiprocessing
import os
import re
import signal
import threading
import time

import numpy as np
import pyBigWig
import pytest

from bpnet.generators import generators
from bpnet.utils.exceptionhandler import NoTracebackException

from tests import synth

#: seconds a generator gets before the test declares it hung
TIME_LIMIT = 120

#: seconds the process that consumed the batches gets to end by itself
#: once it has reported what happened
EXIT_LIMIT = 30

#: seconds an epoch may take to end early. The slow workers of these
#: tests would need SLOW_EPOCH seconds or more for the rest of theirs.
PROMPT = 15
SLOW_EPOCH = 60

SMALL = dict(chrom_len=20000, n_peaks=60, n_background=60)


class Worker(generators.MBPNetSequenceGenerator):
    """
        A generator whose worker processes can be made slow from their
        second batch on, or stop working at one batch
    """

    #: seconds every batch but the first of a slow worker takes
    delay = 0.0
    #: the workers that are slow (None: all of them)
    slow = None
    #: the worker that stops at its batch `fail_at` (None: no worker)
    failing = None
    fail_at = 2
    how = "exception"
    #: whether the workers ignore the request to end (SIGTERM)
    stubborn = False
    #: seconds a worker goes on after it has handed over its last batch
    lingers = 0.0

    def _proc_target(self, coords_df, mpq, proc_idx):
        self._worker = proc_idx
        if self.stubborn:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        super(Worker, self)._proc_target(coords_df, mpq, proc_idx)
        time.sleep(self.lingers)

    def _generate_batch(self, coords):
        # each worker process counts its own batches
        self._made = getattr(self, "_made", 0) + 1
        worker = getattr(self, "_worker", None)
        if worker is not None and worker == self.failing and \
                self._made == self.fail_at:
            if self.how == "exception":
                raise ValueError("no values for this batch")
            if self.how == "exit":
                os._exit(7)
            os.kill(os.getpid(), signal.SIGKILL)
        if self.delay and self._made > 1 and \
                (self.slow is None or worker in self.slow):
            time.sleep(self.delay)
        return super(Worker, self)._generate_batch(coords)


def make_generator(dataset, mode="val", **kwargs):
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": dataset["input_len"],
              "output_len": dataset["output_len"], "rev_comp_aug": False,
              "max_jitter": 0, "shuffle": False, "mode": mode}
    kwargs.setdefault("num_threads", 2)
    kwargs.setdefault("batch_size", 8)
    return Worker(
        dataset["input_json"], params, dataset["fasta"],
        dataset["chrom_sizes"], dataset["chroms"], epochs=2, **kwargs)


def failing_generator(dataset, how, mode="val"):
    generator = make_generator(dataset, mode=mode)
    generator.failing = 0
    generator.how = how
    return generator


def slow_generator(dataset, slow=None):
    """
        Workers whose batches after the first take so long that the
        rest of the epoch needs SLOW_EPOCH seconds or more
    """

    generator = make_generator(dataset)
    per_worker = generator.len() // 2
    assert per_worker >= 3
    generator.delay = SLOW_EPOCH / float(per_worker - 1)
    generator.slow = slow
    return generator


def open_files():
    """
        Number of open file descriptors of this process (0 where that
        cannot be told)
    """

    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:  # pragma: no cover
        return 0


class Before(object):
    """
        What a process had before an epoch: its threads and its open
        file descriptors
    """

    def __init__(self):
        self.threads = threading.active_count()
        self.files = open_files()


def leftovers(before):
    """
        (worker processes, threads, open file descriptors) of this
        process beyond what it had before the epoch
    """

    return (len(multiprocessing.active_children()),
            threading.active_count() - before.threads,
            open_files() - before.files)


NOTHING = (0, 0, 0)


def consume(generator, queue):
    """
        Runs in a process of its own: one epoch of the generator; what
        happened goes into the queue, with what the epoch left behind
    """

    before = Before()
    start = time.time()
    try:
        batches = sum(1 for _ in generator.gen())
        result = ("finished", batches)
    except NoTracebackException as error:
        result = ("failed", str(error))
    except BaseException as error:  # pragma: no cover
        result = ("other", repr(error))
    queue.put(result + (time.time() - start, leftovers(before)))


def outcome(generator, scenario=consume):
    """
        What `scenario` reports about the generator, found out in a
        child process so that a generator that hangs fails the test
        instead of stopping the test run. `consume` reports
        ('finished', number of batches) or ('failed', message) of one
        epoch, the seconds it took and what it left behind.

        The child must also end by itself after it has reported:
        nothing the generator started may keep it running.
    """

    def in_a_session_of_its_own(generator, queue):
        # so that everything it starts can be removed with it
        os.setsid()
        scenario(generator, queue)

    queue = multiprocessing.Queue()
    child = multiprocessing.Process(
        target=in_a_session_of_its_own, args=(generator, queue))
    child.start()
    try:
        result = queue.get(timeout=TIME_LIMIT)
    except Exception:
        result = ("hung", "no result within {} s".format(TIME_LIMIT))
    child.join(EXIT_LIMIT)
    if child.is_alive():
        if result[0] != "hung":
            result = ("did not exit", "reported {!r}, and was still running "
                      "{} s later".format(result, EXIT_LIMIT))
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except OSError:
        pass
    child.join()
    return result


def test_an_epoch_without_failures(synth_dataset):
    generator = make_generator(synth_dataset)
    kind, batches, _, left = outcome(generator)
    assert kind == "finished"
    assert batches == generator.len() > 4
    assert left == NOTHING


def exit_codes(generator, queue):
    ended = []
    end_epoch = generator._end_epoch

    def recording(procs, *args):
        end_epoch(procs, *args)
        ended.append([proc.exitcode for proc in procs if proc is not None])

    generator._end_epoch = recording
    try:
        batches = sum(1 for _ in generator.gen())
    except NoTracebackException:
        batches = None
    queue.put((batches, ended))


def test_workers_of_a_complete_epoch_end_by_themselves(synth_dataset):
    """
        Only an epoch that ends early stops its workers. After the
        last batch of a complete one they are waited for, as always.
    """

    generator = make_generator(synth_dataset)
    generator.lingers = 1.0
    batches, ended = outcome(generator, exit_codes)
    assert batches == generator.len()
    assert ended == [[0, 0]]

    # for comparison: the worker that was stopped when the other failed
    # (which is on its way out, and may be stopped as well)
    generator = slow_generator(synth_dataset, slow=[1])
    generator.failing = 0
    batches, ended = outcome(generator, exit_codes)
    assert batches is None
    assert len(ended) == 1 and ended[0][1] == -signal.SIGTERM
    assert ended[0][0] in (1, -signal.SIGTERM)


@pytest.mark.parametrize("mode", ["val", "train"])
def test_exception_in_a_worker_is_reported(synth_dataset, mode):
    kind, message, seconds, left = outcome(
        failing_generator(synth_dataset, "exception", mode=mode))
    assert kind == "failed", message
    assert "Batch generation failed in worker process 0 ({} mode)".format(
        mode) in message
    # with the worker's own traceback
    assert "ValueError: no values for this batch" in message
    assert "_generate_batch" in message
    assert seconds < TIME_LIMIT
    assert left == NOTHING


@pytest.mark.parametrize("how,exit_code", [("exit", 7), ("kill", -9)])
def test_worker_that_dies_is_reported(synth_dataset, how, exit_code):
    kind, message, _, left = outcome(failing_generator(synth_dataset, how))
    assert kind == "failed", message
    assert "Batch generation failed in worker process 0" in message
    # the batch the worker made before it died may or may not have
    # left the process
    assert re.search(r"the process ended \(exit code {}\) after [01] of its "
                     r"\d+ batches".format(exit_code), message), message
    assert left == NOTHING


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
    kind, message, seconds, _ = outcome(failing_generator(synth_dataset, how))
    assert kind == "failed", message
    assert seconds < 30


def test_handle_on_the_writing_end_of_a_workers_queue(synth_dataset):
    """
        What makes the death of a worker show at once: the parent has
        closed its own handle on the writing end of the worker's
        queue. The handle is a private attribute of the queue; if a
        Python does not have it, this test says so.
    """

    queue = multiprocessing.Queue()
    child = multiprocessing.Process(target=queue.put, args=("item",))
    child.start()
    assert generators.close_writing_end(queue) is True
    assert queue._writer.closed
    assert queue.get(timeout=TIME_LIMIT) == "item"
    child.join()
    # the child is gone and nobody else can write: the end of the data
    with pytest.raises(EOFError):
        queue.get(timeout=TIME_LIMIT)

    class WithoutHandle(object):
        pass

    assert generators.close_writing_end(WithoutHandle()) is False


def test_other_workers_are_stopped_when_one_fails(synth_dataset,
                                                  monkeypatch):
    """
        The worker next to the one that fails has most of its epoch
        ahead of it. It is stopped; the failure is not held back until
        it is done.
    """

    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 2.0 * SLOW_EPOCH)
    generator = slow_generator(synth_dataset, slow=[1])
    generator.failing = 0
    kind, message, seconds, left = outcome(generator)
    assert kind == "failed", message
    assert "Batch generation failed in worker process 0" in message
    assert seconds < PROMPT
    assert left == NOTHING


def test_worker_that_does_not_end_when_asked_is_killed(synth_dataset,
                                                       monkeypatch):
    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 1.0)
    generator = slow_generator(synth_dataset, slow=[1])
    generator.failing = 0
    generator.stubborn = True
    kind, message, seconds, left = outcome(generator)
    assert kind == "failed", message
    assert 1.0 <= seconds < PROMPT
    assert left == NOTHING


# --------------------------------------------------------------------
# a consumer that stops before the end of the epoch
# --------------------------------------------------------------------

def closes(generator, queue):
    before = Before()
    batches = generator.gen()
    next(batches)
    during = leftovers(before)
    start = time.time()
    batches.close()
    queue.put((during, time.time() - start, leftovers(before),
               generator.curr_epoch))


def closes_while_threads_are_slow_to_end(generator, queue):
    collect = generator._stealer

    def slow_to_end(*args):
        collect(*args)
        time.sleep(2)

    generator._stealer = slow_to_end
    closes(generator, queue)


def throws(generator, queue):
    before = Before()
    batches = generator.gen()
    next(batches)
    during = leftovers(before)
    start = time.time()
    try:
        batches.throw(KeyError("thrown into the generator"))
    except KeyError:
        pass
    queue.put((during, time.time() - start, leftovers(before),
               generator.curr_epoch))


def fails_while_consuming(generator, queue):
    before = Before()
    during = []

    def consumer():
        for batch in generator.gen():
            during.append(leftovers(before))
            raise RuntimeError("the consumer of the batches failed")

    start = time.time()
    try:
        consumer()
    except RuntimeError:
        pass
    gc.collect()
    queue.put((during[0], time.time() - start, leftovers(before),
               generator.curr_epoch))


def takes_two_batches(generator, queue):
    before = Before()
    start = time.time()
    batches = generator.gen()
    taken = list(itertools.islice(batches, 2))
    during = leftovers(before)
    del batches
    gc.collect()
    queue.put((during, time.time() - start, leftovers(before),
               generator.curr_epoch, len(taken)))


@pytest.mark.parametrize("scenario", [
    closes, closes_while_threads_are_slow_to_end, throws,
    fails_while_consuming, takes_two_batches])
def test_consumer_that_stops_early_leaves_nothing_behind(
        synth_dataset, monkeypatch, scenario):
    """
        The generator is closed, has an exception thrown into it, or is
        dropped by a consumer that fails or has enough, while its
        workers have most of the epoch ahead of them: the workers and
        the threads that collect their batches end with it, at once,
        and the queues between them are closed
    """

    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 2.0 * SLOW_EPOCH)
    generator = slow_generator(synth_dataset)
    result = outcome(generator, scenario)
    assert len(result) >= 4, result
    during, seconds, after, epoch = result[:4]
    # while the epoch runs: two worker processes, two threads, and at
    # least the reading end of the queue of each worker (how many
    # descriptors a child process takes depends on the Python version)
    assert during[:2] == (2, 2) and during[2] >= 2
    assert after == NOTHING
    assert seconds < PROMPT
    # the epoch does not count as done
    assert epoch == 0
    if scenario is takes_two_batches:
        assert result[4] == 2


def stops_then_runs_an_epoch(generator, queue):
    before = Before()
    batches = generator.gen()
    first = next(batches)
    batches.close()
    # the workers of the next epochs are not slow
    generator.delay = 0.0
    again = list(generator.gen())
    next_epoch = sum(1 for _ in generator.gen())
    same_start = all(
        np.array_equal(first[0][name], again[0][0][name])
        for name in first[0])
    queue.put((len(again), next_epoch, same_start, generator.curr_epoch,
               leftovers(before)))


def test_generator_can_be_used_again_after_an_early_stop(synth_dataset):
    generator = slow_generator(synth_dataset)
    result = outcome(generator, stops_then_runs_an_epoch)
    assert len(result) == 5, result
    batches, next_epoch, same_start, epoch, left = result
    # the epoch that was stopped is generated again, in full, and the
    # generator then goes on to the next one
    assert batches == next_epoch == generator.len()
    assert same_start
    assert epoch == 2
    assert left == NOTHING


def closed_in_a_forked_process(generator, queue):
    before = Before()
    batches = generator.gen()
    first = next(batches)
    # a copy of the running generator in a forked process (a worker of
    # another generator, for example) is closed there
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            batches.close()
            code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    # this process still gets the rest of its epoch
    rest = sum(1 for _ in batches)
    queue.put((os.WEXITSTATUS(status), 1 + rest, generator.curr_epoch,
               leftovers(before)))


def test_copy_of_a_running_generator_in_a_forked_process(synth_dataset):
    """
        The workers of an epoch belong to the process that started it.
        Closing a copy of the generator elsewhere leaves them alone.
    """

    generator = make_generator(synth_dataset)
    result = outcome(generator, closed_in_a_forked_process)
    assert len(result) == 4, result
    exit_code, batches, epoch, left = result
    assert exit_code == 0
    assert batches == generator.len()
    assert epoch == 1
    assert left == NOTHING


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

    kind, message, _, left = outcome(make_generator(dataset))
    assert kind == "failed", message
    assert left == NOTHING
    assert "Batch generation failed in worker process" in message
    assert "RuntimeError" in message
