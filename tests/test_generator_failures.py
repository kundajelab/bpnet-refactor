"""
    A worker process of the batch generator that fails, or dies, ends
    the epoch with an error that names the cause. (Before, the
    consumer of the batches waited forever.) And an epoch that ends
    early, because of that or because its consumer stops taking
    batches, leaves no worker process or thread behind.
"""

import copy
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
    #: a multiprocessing.Value in which the stubborn workers count
    #: themselves once they ignore the request
    ready = None
    #: seconds a worker goes on after it has handed over its last batch
    lingers = 0.0
    #: number of batches after which a worker waits until it is ended
    #: (None: it does not)
    parks_after = None
    #: seconds a thread that collects the batches of a worker goes on
    #: after its last batch, and seconds it waits before its first
    threads_linger = 0.0
    threads_wait = 0.0

    def _proc_target(self, coords_df, mpq, proc_idx):
        self._worker = proc_idx
        if self.stubborn:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if self.ready is not None:
                with self.ready.get_lock():
                    self.ready.value += 1
        super(Worker, self)._proc_target(coords_df, mpq, proc_idx)
        time.sleep(self.lingers)

    def _stealer(self, *args):
        time.sleep(self.threads_wait)
        super(Worker, self)._stealer(*args)
        time.sleep(self.threads_linger)

    def _generate_batch(self, coords):
        # each worker process counts its own batches
        self._made = getattr(self, "_made", 0) + 1
        worker = getattr(self, "_worker", None)
        if worker is not None and worker == self.failing and \
                self._made == self.fail_at:
            if self.ready is not None:
                # not before every worker ignores the request to end
                wait_until(lambda: self.ready.value >= 2)
            if self.how == "exception":
                raise ValueError("no values for this batch")
            if self.how == "exit":
                os._exit(7)
            os.kill(os.getpid(), signal.SIGKILL)
        if self.delay and self._made > 1 and \
                (self.slow is None or worker in self.slow):
            time.sleep(self.delay)
        if self.parks_after is not None and self._made > self.parks_after:
            threading.Event().wait()
        return super(Worker, self)._generate_batch(coords)


def wait_until(condition, seconds=TIME_LIMIT / 2.0):
    deadline = time.time() + seconds
    while not condition() and time.time() < deadline:
        time.sleep(0.02)
    return condition()


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

    def recording(epoch, *args):
        end_epoch(epoch, *args)
        ended.append([proc.exitcode for proc in epoch.procs
                      if proc is not None])

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
    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 2.0)
    generator = slow_generator(synth_dataset, slow=[1])
    generator.failing = 0
    generator.stubborn = True
    generator.ready = multiprocessing.Value("i", 0)
    kind, message, seconds, left = outcome(generator)
    assert kind == "failed", message
    # both workers ignore the request (the one that failed is still
    # on its way out) and are killed after one wait, not after one
    # wait each
    assert 2.0 <= seconds < 3.5
    assert left[:2] == (0, 0)


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
    # the batch that was taken before the stop is one of the batches
    # of the epoch (which of the workers delivers first can differ)
    same_start = any(
        all(np.array_equal(first[0][name], batch[0][name])
            for name in first[0])
        for batch in again)
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


# --------------------------------------------------------------------
# an epoch that is stopped from outside
# --------------------------------------------------------------------

def parked_generator(dataset, batches=1):
    """
        Workers that hand over `batches` batches each and then wait
        until they are ended: the epoch never ends by itself
    """

    generator = make_generator(dataset)
    assert generator.len() // 2 > batches
    generator.parks_after = batches
    return generator


def stopped_while_the_consumer_waits(generator, queue):
    before = Before()
    taken, errors = [], []

    def consumer():
        try:
            for batch in generator.gen():
                taken.append(batch)
        except BaseException as error:
            errors.append(repr(error))

    thread = threading.Thread(target=consumer)
    thread.start()
    # both batches the workers make were taken: the consumer waits for
    # a third
    wait_until(lambda: len(taken) == 2)
    time.sleep(0.5)
    during = leftovers(before)
    start = time.time()
    generator.stop()
    seconds = time.time() - start
    thread.join(PROMPT)
    queue.put((len(taken), errors, thread.is_alive(), during, seconds,
               leftovers(before), generator.curr_epoch,
               len(generators._EPOCHS)))


def test_stop_ends_an_epoch_whose_consumer_waits_for_a_batch(synth_dataset,
                                                             monkeypatch):
    """
        What Keras leaves behind when a training step fails: a thread
        that waits for the next batch, and workers that go on. stop()
        ends the workers; the waiting consumer sees the batches end,
        without an error.
    """

    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 2.0 * SLOW_EPOCH)
    result = outcome(parked_generator(synth_dataset),
                     stopped_while_the_consumer_waits)
    assert len(result) == 8, result
    taken, errors, alive, during, seconds, after, epoch, under_way = result
    assert taken == 2 and errors == [] and not alive
    # two workers, their two threads and the consumer's
    assert during[:2] == (2, 3)
    assert seconds < PROMPT
    assert after == NOTHING
    assert epoch == 0 and under_way == 0


def stopped_while_the_consumer_is_elsewhere(generator, queue):
    before = Before()
    batches = generator.gen()
    next(batches)
    if generator.ready is not None:
        wait_until(lambda: generator.ready.value >= 2)
    during = leftovers(before)
    start = time.time()
    generator.stop()
    seconds = time.time() - start
    # again, and for every generator: nothing left to do
    generator.stop()
    generators.stop_all()
    # without the consumer having taken another batch
    after = leftovers(before)
    rest = list(batches)
    done = leftovers(before)
    # the generator starts the same epoch again, with workers that
    # do not wait
    generator.parks_after = None
    again = sum(1 for _ in generator.gen())
    queue.put((during, seconds, after, len(rest), done, again,
               generator.curr_epoch, len(generators._EPOCHS)))


@pytest.mark.parametrize("workers", ["ordinary", "stubborn",
                                     "slow threads"])
def test_stop_does_not_need_the_consumer(synth_dataset, monkeypatch,
                                         workers):
    """
        The consumer holds the generator and does not take another
        batch (Keras after a failed step). stop() alone ends the
        workers, also workers that ignore the request to end, and
        waits for the threads; the queues are closed when the
        generator is.
    """

    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS",
                        2.0 if workers == "stubborn" else 2.0 * SLOW_EPOCH)
    generator = parked_generator(synth_dataset)
    if workers == "stubborn":
        generator.stubborn = True
        generator.ready = multiprocessing.Value("i", 0)
    if workers == "slow threads":
        generator.threads_linger = 1.5
    result = outcome(generator, stopped_while_the_consumer_is_elsewhere)
    assert len(result) == 8, result
    during, seconds, after, rest, done, again, epoch, under_way = result
    assert during[:2] == (2, 2)
    assert after[:2] == (0, 0)
    assert seconds < PROMPT
    if workers == "stubborn":
        # one wait for both workers, then they are killed
        assert 2.0 <= seconds < 3.5
    if workers == "slow threads":
        assert seconds >= 1.5
    # the batches end without an error, and nothing is left
    assert rest == 0 and done[:2] == (0, 0)
    if workers != "stubborn":
        # (the counter of the stubborn workers holds a descriptor)
        assert done == NOTHING
    assert again == generator.len() and epoch == 1 and under_way == 0


def stopped_one_of_two(generator, queue):
    before = Before()
    other = copy.copy(generator)
    first, second = generator.gen(), other.gen()
    next(first)
    next(second)
    both = leftovers(before)[0]
    generator.stop()
    one = leftovers(before)[0]
    rest_of_first = len(list(first))
    # the other generator is not affected: it delivers its second batch
    next(second)
    generators.stop_all()
    none = leftovers(before)[0]
    rest_of_second = len(list(second))
    queue.put((both, one, none, rest_of_first, rest_of_second,
               leftovers(before), len(generators._EPOCHS)))


def test_stop_of_one_generator_and_of_all(synth_dataset):
    result = outcome(parked_generator(synth_dataset),
                     stopped_one_of_two)
    assert result == (4, 2, 0, 0, 0, NOTHING, 0), result


def stopped_while_starting(generator, queue):
    before = Before()
    start_workers = generator._epoch_run
    stopper = threading.Thread(target=generator.stop)

    def slow_start(data, epoch):
        # stop() is called while the workers are being started: it
        # has to wait until all of them are known
        stopper.start()
        time.sleep(1.0)
        found = start_workers(data, epoch)
        started.append((len(generators._EPOCHS), epoch.stopped,
                        len(multiprocessing.active_children())))
        return found

    started = []
    generator._epoch_run = slow_start
    taken = sum(1 for _ in generator.gen())
    stopper.join(PROMPT)
    queue.put((started, taken, stopper.is_alive(), leftovers(before),
               generator.curr_epoch, len(generators._EPOCHS)))


def test_stop_while_the_workers_are_started(synth_dataset):
    result = outcome(parked_generator(synth_dataset), stopped_while_starting)
    assert len(result) == 6, result
    started, taken, alive, after, epoch, under_way = result
    # the epoch was registered, not stopped, and both workers ran
    # when the start was over
    assert started == [(1, False, 2)]
    # then it was stopped: at most the two batches the workers made
    assert taken <= 2 and not alive
    assert after == NOTHING and epoch == 0 and under_way == 0


def second_worker_cannot_be_started(generator, queue):
    before = Before()
    started = []
    start = multiprocessing.Process.start

    def failing_start(self):
        if started:
            raise OSError("cannot start another process")
        started.append(self)
        start(self)

    multiprocessing.Process.start = failing_start
    try:
        next(generator.gen())
        result = "an epoch was started"
    except OSError as error:
        result = str(error)
    finally:
        multiprocessing.Process.start = start
    first_worker, n_started = started[0].exitcode, len(started)
    # (the process object holds descriptors of its own)
    del started[:]
    gc.collect()
    queue.put((result, n_started, first_worker, leftovers(before),
               generator.curr_epoch, len(generators._EPOCHS)))


def test_failure_to_start_a_worker_ends_those_that_were_started(
        synth_dataset):
    result = outcome(parked_generator(synth_dataset),
                     second_worker_cannot_be_started)
    assert result == ("cannot start another process", 1, -signal.SIGTERM,
                      NOTHING, 0, 0), result


def stopped_while_the_last_workers_are_waited_for(generator, queue):
    before = Before()
    taken, errors = [], []

    def consumer():
        try:
            for batch in generator.gen():
                taken.append(batch)
        except BaseException as error:
            errors.append(repr(error))

    thread = threading.Thread(target=consumer)
    thread.start()
    # every batch was taken; the generator waits for its workers,
    # which are slow to end
    wait_until(lambda: len(taken) == generator.len())
    time.sleep(0.5)
    during = leftovers(before)
    start = time.time()
    generator.stop()
    seconds = time.time() - start
    thread.join(PROMPT)
    queue.put((len(taken), errors, thread.is_alive(), during[:2], seconds,
               leftovers(before), generator.curr_epoch,
               len(generators._EPOCHS)))


def test_stop_reaches_an_epoch_that_waits_for_its_workers(synth_dataset,
                                                          monkeypatch):
    """
        All batches of an epoch were taken, and its workers do not
        end (here they linger for a minute). The epoch is still under
        way and can be stopped, from another thread; the wait for the
        workers does not shut the stop out.
    """

    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 2.0 * SLOW_EPOCH)
    generator = make_generator(synth_dataset)
    generator.lingers = float(SLOW_EPOCH)
    result = outcome(generator, stopped_while_the_last_workers_are_waited_for)
    assert len(result) == 8, result
    taken, errors, alive, during, seconds, after, epoch, under_way = result
    assert taken == generator.len() and errors == [] and not alive
    # two workers, their threads have ended, the consumer's has not
    assert during == (2, 1)
    assert seconds < PROMPT
    assert after == NOTHING
    # the epoch was complete
    assert epoch == 1 and under_way == 0


def epoch_with_another_start_method(generator, queue):
    multiprocessing.set_start_method(generator.start_method, force=True)
    taken = sum(1 for _ in generator.gen())
    # an epoch that is stopped
    generator.delay = SLOW_EPOCH
    batches = generator.gen()
    next(batches)
    start = time.time()
    generator.stop()
    seconds = time.time() - start
    rest = sum(1 for _ in batches)
    queue.put((taken, generator.curr_epoch, seconds, rest,
               len(multiprocessing.active_children()),
               len(generators._EPOCHS)))


@pytest.mark.parametrize("method", ["forkserver", "spawn"])
def test_workers_that_are_not_forked(synth_dataset, monkeypatch, method):
    """
        With the other ways to start a process the worker gets a copy
        of the generator instead of sharing it, and with "forkserver"
        its parent is not the process that takes its batches. An epoch
        delivers all its batches, and can be stopped.
    """

    monkeypatch.setattr(generators, "WORKER_STOP_SECONDS", 2.0 * SLOW_EPOCH)
    generator = make_generator(synth_dataset)
    generator.start_method = method
    result = outcome(generator, epoch_with_another_start_method)
    assert len(result) == 6, result
    taken, epoch, seconds, rest, workers, under_way = result
    assert taken == generator.len() and epoch == 1
    assert seconds < PROMPT and rest == 0
    assert workers == 0 and under_way == 0


def stop_in_a_forked_process(generator, queue):
    before = Before()
    batches = generator.gen()
    next(batches)
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            generator.stop()
            generators.stop_all()
            code = 0 if len(generators._EPOCHS) == 0 else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    rest = sum(1 for _ in batches)
    queue.put((os.WEXITSTATUS(status), 1 + rest, generator.curr_epoch,
               leftovers(before)))


def test_stop_in_a_forked_process_leaves_the_parents_epoch_alone(
        synth_dataset):
    generator = make_generator(synth_dataset)
    result = outcome(generator, stop_in_a_forked_process)
    assert result == (0, generator.len(), 1, NOTHING), result


def holds_an_epoch(generator, queue):
    os.setsid()
    batches = generator.gen()
    next(batches)
    queue.put([child.pid for child in multiprocessing.active_children()])
    time.sleep(TIME_LIMIT)


def ended(pid):
    """
        Whether a process has ended (it is gone, or waits to be
        collected)
    """

    try:
        with open("/proc/{}/stat".format(pid)) as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (IOError, OSError):
        return True


@pytest.mark.skipif(not os.path.isdir("/proc/self"),
                    reason="needs /proc to look at other processes")
def test_workers_end_when_their_consumer_is_gone(synth_dataset):
    """
        The process that takes the batches is killed (out of memory,
        a scheduler): its workers find out at their next batch and
        end, instead of making the rest of the epoch for nobody
    """

    generator = make_generator(synth_dataset, batch_size=1)
    per_worker = generator.len() // 2
    assert per_worker >= 20
    # every batch after the first takes a quarter of a second: the
    # epoch would take a worker five seconds or more
    generator.delay = 0.25
    queue = multiprocessing.Queue()
    consumer = multiprocessing.Process(target=holds_an_epoch,
                                       args=(generator, queue))
    consumer.start()
    try:
        workers = queue.get(timeout=TIME_LIMIT)
        assert len(workers) == 2
        assert not any(ended(pid) for pid in workers)
        os.kill(consumer.pid, signal.SIGKILL)
        consumer.join()
        start = time.time()
        assert wait_until(lambda: all(ended(pid) for pid in workers), 4.0), \
            "workers still running {:.1f} s after their consumer was " \
            "killed".format(time.time() - start)
    finally:
        try:
            os.killpg(consumer.pid, signal.SIGKILL)
        except OSError:
            pass
        consumer.join()


def holds_an_epoch_whose_workers_are_done(generator, queue):
    os.setsid()
    batches = generator.gen()
    # the epoch starts; nothing collects the batches for a long time
    threading.Thread(target=next, args=(batches,), daemon=True).start()
    # the workers make their batches at once; then they wait for them
    # to be collected
    time.sleep(2.0)
    queue.put([child.pid for child in multiprocessing.active_children()])
    time.sleep(TIME_LIMIT)


@pytest.mark.skipif(not os.path.isdir("/proc/self"),
                    reason="needs /proc to look at other processes")
def test_workers_with_batches_to_hand_over_end_when_their_consumer_is_gone(
        synth_dataset):
    """
        The workers have made all their batches and wait for them to
        be taken. Their consumer is killed. They end instead of
        waiting forever: a worker holds no reading end of its own
        queue, so that writing to it fails once the consumer is gone.
    """

    generator = make_generator(synth_dataset, batch_size=1)
    assert generator.len() // 2 >= 20
    generator.threads_wait = float(TIME_LIMIT)
    queue = multiprocessing.Queue()
    consumer = multiprocessing.Process(
        target=holds_an_epoch_whose_workers_are_done, args=(generator, queue))
    consumer.start()
    try:
        workers = queue.get(timeout=TIME_LIMIT)
        assert len(workers) == 2
        assert not any(ended(pid) for pid in workers)
        os.kill(consumer.pid, signal.SIGKILL)
        consumer.join()
        assert wait_until(lambda: all(ended(pid) for pid in workers), 5.0)
    finally:
        try:
            os.killpg(consumer.pid, signal.SIGKILL)
        except OSError:
            pass
        consumer.join()


def test_handle_on_the_reading_end_of_a_workers_queue():
    queue = multiprocessing.Queue()
    assert generators.close_reading_end(queue) is True
    assert queue._reader.closed and not queue._writer.closed

    class WithoutHandle(object):
        pass

    assert generators.close_reading_end(WithoutHandle()) is False


def interrupted_while_stopping(generator, queue):
    stopped = []

    def slow_stop():
        # a second interrupt arrives while the batch generation is
        # being stopped
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.5)
        stopped.append("stopped")

    generators.stop_all = slow_stop
    try:
        generators.stop_after_failure()
        outcome = "returned"
    except KeyboardInterrupt:
        outcome = "interrupted"

    def failing_stop():
        raise OSError("cannot stop")

    generators.stop_all = failing_stop
    try:
        generators.stop_after_failure()
        second = "returned"
    except OSError:
        second = "raised"
    queue.put((outcome, stopped, second,
               signal.getsignal(signal.SIGINT) is signal.SIG_IGN))


def test_cleanup_after_a_failure_is_not_interrupted(synth_dataset):
    """
        What the command line tools call when they fail: an interrupt
        that arrives during the cleanup does not end it, and an error
        of the cleanup is not raised (the failure of the tool is the
        one to report)
    """

    result = outcome(make_generator(synth_dataset), interrupted_while_stopping)
    assert result == ("returned", ["stopped"], "returned", True), result


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
