"""
    Interrupts (SIGINT) that Python would lose: the bookkeeping that
    lets a loop act on them at its next round
    (bpnet.generators.generators.Interrupts, interruptible)
"""

import multiprocessing
import os
import signal
import threading
import time

import pytest

from bpnet.generators import generators

#: seconds a scenario gets
LIMIT = 60


def in_a_process(scenario):
    """
        What `scenario(queue)` puts into its queue, in a process of
        its own (the scenarios send signals to themselves)
    """

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=scenario, args=(queue,))
    process.start()
    try:
        return queue.get(timeout=LIMIT)
    finally:
        process.join(LIMIT)
        if process.is_alive():  # pragma: no cover
            process.kill()
            process.join()


def interrupt_myself():
    os.kill(os.getpid(), signal.SIGINT)
    # (the handler runs at once; if it raises nothing, go on)
    time.sleep(0.2)


def raised(queue):
    before = signal.getsignal(signal.SIGINT)
    result = "went on"
    try:
        with generators.Interrupts() as interrupts:
            interrupt_myself()
    except KeyboardInterrupt:
        result = "interrupted"
    queue.put((result, interrupts.noted,
               signal.getsignal(signal.SIGINT) is before))


def test_interrupt_is_raised_as_always():
    assert in_a_process(raised) == ("interrupted", True, True)


def lost(queue):
    with generators.Interrupts() as interrupts:
        try:
            interrupt_myself()
        except KeyboardInterrupt:
            # as inside a destructor: the exception is discarded
            pass
        try:
            interrupts.check()
            result = "went on"
        except KeyboardInterrupt:
            result = "interrupted at the check"
    queue.put((result, interrupts.noted))


def test_interrupt_that_was_lost_is_raised_at_the_check():
    assert in_a_process(lost) == ("interrupted at the check", True)


def nothing_happens(queue):
    with generators.Interrupts() as interrupts:
        interrupts.check()
        time.sleep(0.1)
        interrupts.check()
    queue.put(interrupts.noted)


def test_no_interrupt_no_exception():
    assert in_a_process(nothing_happens) is False


def loop(queue):
    before = signal.getsignal(signal.SIGINT)
    fetched = []

    def source():
        for item in (1, 2, 3):
            fetched.append(item)
            yield item

    taken = []
    try:
        for item in generators.interruptible(source()):
            taken.append(item)
            if item == 2:
                try:
                    interrupt_myself()
                except KeyboardInterrupt:
                    pass
        result = "finished"
    except KeyboardInterrupt:
        result = "interrupted"
    whole = list(generators.interruptible([1, 2, 3]))
    queue.put((result, taken, fetched, whole,
               signal.getsignal(signal.SIGINT) is before))


def test_loop_ends_before_the_next_item():
    """
        An interrupt that is lost while an item is worked on ends the
        loop without fetching the next item (which can take long); a
        loop without interrupts hands out every item, and the handling
        of interrupts is as before afterwards
    """

    assert in_a_process(loop) == ("interrupted", [1, 2], [1, 2], [1, 2, 3],
                                  True)


def lost_while_fetching(queue):
    def source():
        yield 1
        try:
            interrupt_myself()
        except KeyboardInterrupt:
            pass
        yield 2
        yield 3

    taken = []
    try:
        for item in generators.interruptible(source()):
            taken.append(item)
        result = "finished"
    except KeyboardInterrupt:
        result = "interrupted"
    queue.put((result, taken))


def test_loop_ends_before_an_item_is_handed_out():
    """
        An interrupt that is lost while the next item is fetched ends
        the loop before that item is handed out
    """

    assert in_a_process(lost_while_fetching) == ("interrupted", [1])


def reported_as_another_error(queue):
    with generators.Interrupts() as interrupts:
        plain = RuntimeError("no interrupt yet")
        before = interrupts.interrupted(plain) is plain
        try:
            interrupt_myself()
        except KeyboardInterrupt as interrupt:
            same = interrupts.interrupted(interrupt) is interrupt
        other = TypeError()
        restored = interrupts.interrupted(other)
    queue.put((before, same, type(restored).__name__,
               restored.__cause__ is other, str(restored)))


def test_interrupt_that_came_out_as_another_error():
    assert in_a_process(reported_as_another_error) == (
        True, True, "KeyboardInterrupt", True,
        "interrupted (the interrupt was reported as: TypeError())")


def lost_before_the_loop(queue):
    fetched = []

    def source():
        for item in (1, 2):
            fetched.append(item)
            yield item

    with generators.Interrupts() as interrupts:
        try:
            interrupt_myself()
        except KeyboardInterrupt:
            pass
        taken = []
        try:
            for item in generators.interruptible(source(), interrupts):
                taken.append(item)
            result = "finished"
        except KeyboardInterrupt:
            result = "interrupted"
    queue.put((result, taken, fetched))


def test_interrupt_lost_before_the_loop_ends_it_before_the_first_item():
    """
        With the notes of a block that began before the loop, an
        interrupt that was lost before the loop ends it before the
        first item is fetched
    """

    assert in_a_process(lost_before_the_loop) == ("interrupted", [], [])


def ignored_before(queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with generators.Interrupts() as interrupts:
        interrupt_myself()
        interrupts.check()
        during = signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    queue.put((interrupts.noted, during,
               signal.getsignal(signal.SIGINT) == signal.SIG_IGN))


def test_interrupts_that_were_ignored_stay_ignored():
    assert in_a_process(ignored_before) == (False, True, True)


def own_handler(queue):
    calls = []
    signal.signal(signal.SIGINT, lambda number, frame: calls.append(number))
    with generators.Interrupts() as interrupts:
        interrupt_myself()
        try:
            interrupts.check()
            result = "went on"
        except KeyboardInterrupt:
            result = "interrupted at the check"
    queue.put((calls == [signal.SIGINT], result))


def test_handler_that_was_there_is_called():
    assert in_a_process(own_handler) == (True, "interrupted at the check")


def default_action(queue):
    # (the action of the system, not Python's handler)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        with generators.Interrupts():
            interrupt_myself()
        result = "went on"
    except KeyboardInterrupt:
        result = "interrupted"
    queue.put((result, signal.getsignal(signal.SIGINT) == signal.SIG_DFL))


def test_without_a_python_handler_the_interrupt_is_raised():
    assert in_a_process(default_action) == ("interrupted", True)


def in_another_thread(queue):
    found = []

    def run():
        with generators.Interrupts() as interrupts:
            interrupts.check()
            found.append(interrupts._installed)

    before = signal.getsignal(signal.SIGINT)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    queue.put((found, signal.getsignal(signal.SIGINT) is before))


def test_nothing_is_changed_outside_the_main_thread():
    assert in_a_process(in_another_thread) == ([False], True)
