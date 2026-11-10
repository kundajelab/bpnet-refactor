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
    NOTHING, PROMPT, TIME_LIMIT, Before, Worker, ended,
    kill_with_its_session, leftovers, make_generator, outcome, wait_until)


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


def most_waiting(workers):
    """
        The largest number of batches that can have been made and not
        yet taken: per worker the ones in its queue and the one it
        waits to put, per collecting thread the one it holds, and the
        ones in the queue of the consumer. When nobody takes batches,
        exactly that many are made.
    """

    return workers * (generators.WORKER_QUEUE_BATCHES + 1) + workers + \
        generators.consumer_queue_batches(workers)


def until_the_queues_are_full(generator, taken=1):
    """
        Wait until the workers of an epoch from which `taken` batches
        were taken have filled every queue, then for another half
        second.

        Returns:
            tuple: (batches made when the queues were full, whether no
                batch was made after that)
    """

    full = taken + most_waiting(generator._num_threads)
    wait_until(lambda: generator.made.value >= full)
    made = generator.made.value
    time.sleep(0.5)
    return made, generator.made.value == made


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
    queue.put((taken, most, generator.made.value, indices,
               leftovers(before), generator.curr_epoch))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_batches_wait_for_a_slow_consumer_in_a_few_places(synth_dataset,
                                                          workers):
    """
        The workers make a batch in a few milliseconds, the consumer
        takes one every 30. The batches that were made and not yet
        taken stay within what the queues hold (they used to be the
        rest of the epoch), and the epoch still delivers every locus
        once, the loci of each worker in the order of the epoch.
    """

    generator = counted_generator(synth_dataset, mode="train",
                                  num_threads=workers)
    result = outcome(generator, slow_consumer)
    assert len(result) == 6, result
    taken, most, made, indices, left, epoch = result
    assert taken == made == generator.len()
    # the queues did fill up, and held the workers back
    assert workers + 2 <= most <= most_waiting(workers)
    assert most < generator.len() // 2
    expected = generator.get_resized_samples()[0].index.values[:taken]
    assert sorted(indices) == sorted(expected.tolist())
    assert len(set(indices)) == len(indices)
    # each worker has its share of the loci of the epoch, and what it
    # makes arrives in the order it made it
    share = len(expected) // workers
    for worker in range(workers):
        own = expected[worker * share:(worker + 1) * share].tolist()
        assert [index for index in indices if index in set(own)] == own
    assert left == NOTHING and epoch == 1


def fills_the_queues(generator, queue):
    generator.made = multiprocessing.Value("i", 0)
    before = Before()
    batches = generator.gen()
    next(batches)
    made, waiting = until_the_queues_are_full(generator)
    batches.close()
    queue.put((made, waiting, leftovers(before)))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_workers_make_as_many_batches_as_the_queues_hold(synth_dataset,
                                                         workers):
    """
        The consumer takes one batch and no more. Every worker makes
        batches until two are in its queue, its collecting thread
        holds one and it waits to put another, and the queue of the
        consumer is full: that many batches exactly, and then none.
    """

    generator = counted_generator(synth_dataset, num_threads=workers)
    in_the_queues = {1: 2 + 4, 2: 4 + 8, 4: 8 + 16}[workers]
    assert most_waiting(workers) == in_the_queues
    # no worker runs out of batches before the queues are full
    assert generator.len() // workers > in_the_queues
    result = outcome(generator, fills_the_queues)
    assert result == (1 + in_the_queues, True, NOTHING), result


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
    made, waiting = until_the_queues_are_full(generator)
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
    made, waiting = until_the_queues_are_full(generator)
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
    assert waiting and made == 1 + most_waiting(2) < generator.len() // 2
    assert during[:2] == (2, 2)
    assert seconds < PROMPT
    if scenario is closes_when_the_queues_are_full:
        assert after == NOTHING
    else:
        # no worker, no thread, and no batch for the consumer
        assert after[:2] == (0, 0) and after[3] == 0
    assert epoch == 0


class Notes(Counted):
    """
        A generator whose batches are a few bytes. What a worker has
        handed over is then in the pipe at once, and the worker has
        nothing left to write while it waits for room.
    """

    def _generate_batch(self, coords):
        with self.made.get_lock():
            self.made.value += 1
            return ("batch", self.made.value)


#: seconds the workers of a consumer that was killed get to end. They
#: look for their consumer every WORKER_POLL_SECONDS.
ORPHAN_SECONDS = 5.0


def a_bystander():
    """
        Fork a process that does nothing and goes on after its parent

        Returns:
            int: its id
    """

    pid = os.fork()
    if pid == 0:
        try:
            time.sleep(TIME_LIMIT)
        finally:
            os._exit(0)
    return pid


def unreadable(pid):
    raise OSError("no start time for process {}".format(pid))


def holds_an_epoch_with_full_queues(generator, queue):
    os.setsid()
    multiprocessing.set_start_method(generator.start_method, force=True)
    generator.made = multiprocessing.Value("i", 0)
    if generator.case == "start time cannot be read":
        # (as on a system without /proc: the workers are then told
        # the process id of their consumer only)
        generators.process_start = unreadable
    elif generator.case == "nothing is known of the consumer":
        generators.process_identity = lambda: None
    batches = generator.gen()
    next(batches)
    made, waiting = until_the_queues_are_full(generator)
    others = []
    if generator.case == "forked":
        # a process that is forked from the consumer while the epoch
        # is under way, and goes on after it
        others.append(a_bystander())
    queue.put((made, waiting,
               [child.pid for child in multiprocessing.active_children()],
               others))
    time.sleep(TIME_LIMIT)


def orphans_end(generator, method, case=None):
    """
        Start an epoch of two workers in a process of its own, with
        the given start method of multiprocessing; when the workers
        wait for room in their queues, kill that process.

        Returns:
            bool: whether the workers ended within ORPHAN_SECONDS
    """

    generator.start_method = method
    generator.case = case
    queue = multiprocessing.Queue()
    consumer = multiprocessing.Process(
        target=holds_an_epoch_with_full_queues, args=(generator, queue))
    consumer.start()
    workers, others = [], []
    try:
        made, waiting, workers, others = queue.get(timeout=TIME_LIMIT)
        assert waiting and made == 1 + most_waiting(2)
        assert len(workers) == 2
        assert not any(ended(pid) for pid in workers)
        os.kill(consumer.pid, signal.SIGKILL)
        consumer.join()
        gone = wait_until(lambda: all(ended(pid) for pid in workers),
                          ORPHAN_SECONDS)
        # (the other process is still there)
        assert not any(ended(pid) for pid in others)
        return gone
    finally:
        kill_with_its_session(consumer, workers + others)


needs_proc = pytest.mark.skipif(
    not os.path.isdir("/proc/self"),
    reason="needs /proc to look at other processes")


@needs_proc
@pytest.mark.parametrize("method", ["fork", "forkserver", "spawn"])
@pytest.mark.parametrize("batches", [Counted, Notes])
def test_waiting_workers_end_when_their_consumer_is_gone(synth_dataset,
                                                         batches, method):
    """
        Workers that wait for room in their queue find out that the
        process that takes their batches was killed, and end.

        With batches of a few bytes (Notes) a worker has written all
        it handed over, so that no failing write tells it. And with
        the "forkserver" start method its parent is the fork server,
        which is still there. It looks for the process of its
        consumer every time the wait runs out.
    """

    generator = counted_generator(synth_dataset)
    generator.__class__ = batches
    assert orphans_end(generator, method)


@needs_proc
@pytest.mark.parametrize("method", ["fork", "forkserver"])
def test_process_forked_from_the_consumer_is_not_the_consumer(synth_dataset,
                                                              method):
    """
        The consumer forks another process while the epoch is under
        way, which has copies of everything the consumer had open and
        goes on after it. The workers end all the same when the
        consumer is killed: they look for the process of the
        consumer, not for something a forked process inherits.
    """

    generator = counted_generator(synth_dataset)
    generator.__class__ = Notes
    assert orphans_end(generator, method, case="forked")


@needs_proc
@pytest.mark.parametrize("method", ["fork", "forkserver", "spawn"])
def test_workers_end_where_the_start_time_cannot_be_read(synth_dataset,
                                                         method):
    """
        Without the start time of the consumer (no /proc) a worker
        looks for a process with the id of the consumer
    """

    generator = counted_generator(synth_dataset)
    generator.__class__ = Notes
    assert orphans_end(generator, method, case="start time cannot be read")


@needs_proc
@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_workers_of_a_parent_end_by_their_parent_alone(synth_dataset, method):
    """
        Workers whose parent is the consumer (start methods "fork"
        and "spawn") are told by the change of their parent, whatever
        else they know of the consumer
    """

    generator = counted_generator(synth_dataset)
    generator.__class__ = Notes
    assert orphans_end(generator, method,
                       case="nothing is known of the consumer")


def sleeper(name=None):
    if name is not None:
        # (the name of the process, as /proc shows it)
        with open("/proc/self/comm", "w") as comm:
            comm.write(name)
    time.sleep(TIME_LIMIT)


def a_sleeper(name=None):
    child = multiprocessing.Process(target=sleeper, args=(name,))
    child.start()
    return child


def end(child):
    if child.is_alive():
        os.kill(child.pid, signal.SIGKILL)
    child.join()


def test_start_time_of_a_process(monkeypatch):
    if not os.path.isdir("/proc/self"):
        with pytest.raises(OSError):
            generators.process_start(os.getpid())
        assert generators.process_identity() == (os.getpid(), None)
        return
    mine = generators.process_start(os.getpid())
    assert isinstance(mine, int) and mine > 0
    assert generators.process_start(os.getpid()) == mine
    assert generators.process_identity() == (os.getpid(), mine)

    child = a_sleeper()
    try:
        started = generators.process_start(child.pid)
        assert isinstance(started, int) and started >= mine
        os.kill(child.pid, signal.SIGKILL)
        # it has ended and waits to be collected
        assert wait_until(lambda: ended(child.pid), 10.0)
        assert generators.process_start(child.pid) is None
    finally:
        end(child)
    # no entry: it cannot be told from here whether there is no such
    # process or a hidden one
    with pytest.raises(FileNotFoundError):
        generators.process_start(child.pid)

    # a system without /proc
    monkeypatch.setattr(generators.os.path, "isdir", lambda path: False)
    with pytest.raises(NotADirectoryError):
        generators.process_start(os.getpid())
    assert generators.process_identity() == (os.getpid(), None)


@needs_proc
@pytest.mark.parametrize("name", ["a b", "a) Z 1 (b", "(x)\nR 2 3", ")"])
def test_start_time_of_a_process_with_an_odd_name(name):
    """
        The name of a process is part of the line the start time is
        read from, and may hold spaces, parentheses and line ends
    """

    child = a_sleeper()
    try:
        started = generators.process_start(child.pid)
    finally:
        end(child)
    child = a_sleeper(name)
    try:
        def renamed():
            with open("/proc/{}/comm".format(child.pid)) as comm:
                return comm.read() == name + "\n"

        assert wait_until(renamed, 10.0)
        with open("/proc/{}/stat".format(child.pid), "rb") as stat:
            assert name.encode() in stat.read()
        found = generators.process_start(child.pid)
        # started after the first child, within the time of this test
        assert isinstance(found, int)
        assert started <= found <= started + 100 * os.sysconf("SC_CLK_TCK")
    finally:
        end(child)


def hidden(pid):
    raise FileNotFoundError("/proc/{}/stat".format(pid))


@needs_proc
def test_consumer_as_a_worker_sees_it(monkeypatch):
    # nothing known of the consumer: only the parent counts
    consumer = generators.Consumer()
    assert consumer.parent == os.getppid()
    assert consumer.gone() is False
    consumer.parent += 1
    assert consumer.gone() is True

    child = a_sleeper()
    try:
        identity = (child.pid, generators.process_start(child.pid))
        by_start = generators.Consumer(identity)
        by_id = generators.Consumer((child.pid, None))
        other = generators.Consumer((child.pid, identity[1] + 1))
        assert by_start.gone() is False and by_id.gone() is False
        # a process with that id that was started at another time is
        # not the consumer
        assert other.gone() is True
        # the parent of the worker has changed: the consumer has ended
        by_start.parent += 1
        assert by_start.gone() is True
        by_start.parent -= 1
        # when the start time cannot be read, or the process is hidden
        # from this one, the consumer counts as there
        for cannot_tell in (unreadable, hidden):
            monkeypatch.setattr(generators, "process_start", cannot_tell)
            assert by_start.gone() is False and other.gone() is False
            monkeypatch.undo()

        os.kill(child.pid, signal.SIGKILL)
        assert wait_until(lambda: ended(child.pid), 10.0)
        # ended and not collected yet: gone by its start time; by its
        # id alone, or hidden, it still counts as there
        assert by_start.gone() is True
        assert by_id.gone() is False
        monkeypatch.setattr(generators, "process_start", hidden)
        assert by_start.gone() is False
        monkeypatch.undo()
    finally:
        end(child)
    # collected: there is no process with that id, whatever /proc
    # shows
    assert by_start.gone() is True and by_id.gone() is True
    for cannot_tell in (unreadable, hidden):
        monkeypatch.setattr(generators, "process_start", cannot_tell)
        assert by_start.gone() is True
        monkeypatch.undo()

    # a process that may not be signalled is there
    if os.getuid() != 0:
        assert generators.Consumer((1, None)).gone() is False
        assert generators.Consumer(
            (1, generators.process_start(1))).gone() is False


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
