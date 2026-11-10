"""
    The queues between the worker processes of the batch generator and
    its consumer have room for a few batches. Workers that are faster
    than the consumer wait, instead of filling memory with the batches
    of the whole epoch; failures, an early end and a stop still get
    through when the queues are full.
"""

import multiprocessing
import os
import signal
import time

import numpy as np
import pytest

from bpnet.generators import generators
from bpnet.utils.exceptionhandler import NoTracebackException

from tests.test_generator_failures import (
    NOTHING, PROMPT, TIME_LIMIT, Before, Worker, ended, leftovers,
    make_generator, outcome, wait_until)


class Counted(Worker):
    """
        A generator whose workers count the batches they have made, in
        a number the consumer can read
    """

    made = None

    def _generate_batch(self, coords):
        batch = super(Counted, self)._generate_batch(coords)
        if self.made is not None:
            with self.made.get_lock():
                self.made.value += 1
        return batch


def counted_generator(dataset, mode="val", **kwargs):
    """
        One locus per batch: an epoch of many batches, made much
        faster than the consumers of these tests take them
    """

    generator = make_generator(dataset, mode=mode, batch_size=1, **kwargs)
    generator.__class__ = Counted
    assert generator.len() >= 40
    return generator


def most_in_flight(workers):
    """
        The largest number of batches that can have been made and not
        yet taken: per worker the ones in its queue and the one it
        waits to put, per collecting thread the one it holds, the ones
        in the queue of the consumer, and one the consumer is taking
    """

    return workers * (generators.WORKER_QUEUE_BATCHES + 1) + workers + \
        generators.consumer_queue_batches(workers) + 1


def test_sizes_of_the_queues():
    assert generators.WORKER_QUEUE_BATCHES == 2
    assert [generators.consumer_queue_batches(n) for n in (1, 2, 3, 4, 10)] \
        == [2, 4, 6, 8, 8]


def slow_consumer(generator, queue):
    generator.made = multiprocessing.Value("i", 0)
    before = Before()
    taken, most, indices = 0, 0, []
    for inputs, _, _ in generator.gen():
        taken += 1
        indices.extend(inputs["index"].tolist())
        time.sleep(0.03)
        most = max(most, generator.made.value - taken)
    queue.put((taken, most, generator.made.value, sorted(indices),
               leftovers(before), generator.curr_epoch))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_batches_wait_for_a_slow_consumer_in_a_few_places(synth_dataset,
                                                          workers):
    """
        The workers make a batch in a few milliseconds, the consumer
        takes one every 30. The batches that were made and not yet
        taken stay within what the queues hold (they used to be the
        rest of the epoch), and the epoch still delivers every locus
        once.
    """

    generator = counted_generator(synth_dataset, mode="train",
                                  num_threads=workers)
    result = outcome(generator, slow_consumer)
    assert len(result) == 6, result
    taken, most, made, indices, left, epoch = result
    assert taken == made == generator.len()
    # the queues did fill up, and held the workers back
    assert workers + 2 <= most <= most_in_flight(workers)
    assert most < generator.len() // 2
    expected = generator.get_resized_samples()[0].index.values[:taken]
    assert indices == sorted(expected.tolist())
    assert left == NOTHING and epoch == 1


def consumer_that_pauses(generator, queue):
    before = Before()
    indices = []
    for number, (inputs, _, _) in enumerate(generator.gen()):
        indices.extend(inputs["index"].tolist())
        if number == 4:
            # long enough for every worker and collecting thread to
            # find its queue full several times over
            time.sleep(5 * generators.WORKER_POLL_SECONDS)
    queue.put((sorted(indices), leftovers(before), generator.curr_epoch))


def test_waiting_for_room_loses_nothing(synth_dataset, monkeypatch):
    """
        The consumer stops taking batches for a while. Workers and
        collecting threads wait, look up, and wait again; when the
        consumer goes on, every batch of the epoch arrives.
    """

    monkeypatch.setattr(generators, "WORKER_POLL_SECONDS", 0.2)
    generator = counted_generator(synth_dataset, mode="train")
    result = outcome(generator, consumer_that_pauses)
    assert len(result) == 3, result
    indices, left, epoch = result
    expected = generator.get_resized_samples()[0].index.values[
        :generator.len()]
    assert indices == sorted(expected.tolist())
    assert left == NOTHING and epoch == 1


def slow_consumer_of_a_failing_worker(generator, queue):
    before = Before()
    taken = 0
    try:
        for _ in generator.gen():
            taken += 1
            time.sleep(0.03)
        result = "finished"
    except NoTracebackException as error:
        result = str(error)
    queue.put((result, taken, leftovers(before)))


def test_failure_gets_through_full_queues(synth_dataset):
    """
        A worker fails at its tenth batch while the queues are full
        (the consumer is slow). Its failure waits behind the batches
        that are in the queues, and is then reported.
    """

    generator = counted_generator(synth_dataset)
    generator.failing = 0
    generator.fail_at = 10
    result = outcome(generator, slow_consumer_of_a_failing_worker)
    assert len(result) == 3, result
    message, taken, left = result
    assert "Batch generation failed in worker process 0" in message
    assert "ValueError: no values for this batch" in message
    # the nine batches of the worker that failed were delivered
    assert 9 <= taken < generator.len()
    assert left == NOTHING


def closes_when_the_queues_are_full(generator, queue):
    generator.made = multiprocessing.Value("i", 0)
    before = Before()
    batches = generator.gen()
    next(batches)
    # the workers go on until every queue is full, and then wait
    time.sleep(1.5)
    made = generator.made.value
    time.sleep(0.5)
    waiting = generator.made.value == made
    during = leftovers(before)
    start = time.time()
    batches.close()
    queue.put((made, waiting, during, time.time() - start,
               leftovers(before), generator.curr_epoch))


def stops_when_the_queues_are_full(generator, queue):
    generator.made = multiprocessing.Value("i", 0)
    before = Before()
    batches = generator.gen()
    next(batches)
    time.sleep(1.5)
    made = generator.made.value
    time.sleep(0.5)
    waiting = generator.made.value == made
    during = leftovers(before)
    start = time.time()
    generator.stop()
    seconds = time.time() - start
    after = leftovers(before)
    rest = len(list(batches))
    queue.put((made, waiting, during, seconds, after + (rest,),
               generator.curr_epoch))


@pytest.mark.parametrize("scenario", [closes_when_the_queues_are_full,
                                      stops_when_the_queues_are_full])
def test_epoch_ends_early_although_everything_waits(synth_dataset, scenario):
    """
        The consumer takes one batch and no more. The workers fill the
        queues and wait; their collecting threads wait for room in the
        queue of the consumer. Closing the generator, or stopping the
        epoch, ends all of them.
    """

    generator = counted_generator(synth_dataset)
    result = outcome(generator, scenario)
    assert len(result) == 6, result
    made, waiting, during, seconds, after, epoch = result
    # the workers stopped making batches long before the end of the
    # epoch
    assert waiting and made <= most_in_flight(2) < generator.len() // 2
    assert during[:2] == (2, 2)
    assert seconds < PROMPT
    if scenario is closes_when_the_queues_are_full:
        assert after == NOTHING
    else:
        # no worker, no thread, and no batch for the consumer
        assert after[:2] == (0, 0) and after[3] == 0
    assert epoch == 0


def holds_an_epoch_with_full_queues(generator, queue):
    os.setsid()
    batches = generator.gen()
    next(batches)
    time.sleep(1.5)
    queue.put([child.pid for child in multiprocessing.active_children()])
    time.sleep(TIME_LIMIT)


@pytest.mark.skipif(not os.path.isdir("/proc/self"),
                    reason="needs /proc to look at other processes")
def test_waiting_workers_end_when_their_consumer_is_gone(synth_dataset):
    """
        Workers that wait for room in their queue find out that the
        process that takes their batches was killed, and end
    """

    generator = counted_generator(synth_dataset)
    queue = multiprocessing.Queue()
    consumer = multiprocessing.Process(
        target=holds_an_epoch_with_full_queues, args=(generator, queue))
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


class Notes(Counted):
    """
        A generator whose batches are a few bytes. What a worker has
        handed over is then in the pipe at once, and the worker has
        nothing left to write while it waits for room.
    """

    def _generate_batch(self, coords):
        self._made = getattr(self, "_made", 0) + 1
        return ("batch", self._made)


@pytest.mark.skipif(not os.path.isdir("/proc/self"),
                    reason="needs /proc to look at other processes")
def test_waiting_workers_with_nothing_to_write_end_when_their_consumer_is_gone(
        synth_dataset):
    """
        A worker that waits for room and has written all it handed
        over is not told by a failing write that its consumer is gone.
        It looks for its consumer every time the wait runs out.
    """

    generator = counted_generator(synth_dataset)
    generator.__class__ = Notes
    queue = multiprocessing.Queue()
    consumer = multiprocessing.Process(
        target=holds_an_epoch_with_full_queues, args=(generator, queue))
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
        for pid in workers:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        consumer.join()


def test_one_worker_delivers_in_order(synth_dataset):
    """
        With one worker the batches arrive in the order of the epoch,
        as before (with more workers the order was never fixed)
    """

    generator = counted_generator(synth_dataset, mode="train",
                                  num_threads=1)
    order = [inputs["index"][0] for inputs, _, _ in generator.gen()]
    expected = generator.get_resized_samples()[0].index.values
    assert np.array_equal(order, expected[:len(order)])
    assert len(order) == generator.len()
