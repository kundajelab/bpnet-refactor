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
        its own (the scenarios send signals to themselves). The
        process must also end by itself.
    """

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=scenario, args=(queue,))
    process.start()
    try:
        result = queue.get(timeout=LIMIT)
        process.join(LIMIT)
        assert not process.is_alive(), "the scenario did not end"
        return result
    finally:
        if process.is_alive():
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


def lost_at_the_end(queue):
    def ending():
        yield 1
        try:
            interrupt_myself()
        except KeyboardInterrupt:
            pass

    with generators.Interrupts() as interrupts:
        taken = []
        try:
            for item in generators.interruptible(ending(), interrupts):
                taken.append(item)
            result = "finished"
        except KeyboardInterrupt:
            result = "interrupted"
    queue.put((result, taken))


def test_interrupt_lost_while_the_end_is_fetched_ends_the_loop():
    assert in_a_process(lost_at_the_end) == ("interrupted", [1])


def failures_after_an_interrupt(queue):
    from bpnet.utils.exceptionhandler import NoTracebackException

    with generators.Interrupts() as interrupts:
        try:
            interrupt_myself()
        except KeyboardInterrupt:
            pass
        found = []
        for failure in (SystemExit(0), SystemExit(1),
                        NoTracebackException("Batch generation failed"),
                        ValueError("x")):
            reported = interrupts.interrupted(failure)
            found.append((type(reported).__name__,
                          reported.__cause__ is failure))
    queue.put(found)


def test_every_failure_after_an_interrupt_is_the_interrupt():
    """
        Also an exit (which may claim success) and the failure of a
        batch worker that followed the interrupt are reported as the
        interrupt
    """

    assert in_a_process(failures_after_an_interrupt) == \
        [("KeyboardInterrupt", True)] * 4


def nested(queue):
    before = signal.getsignal(signal.SIGINT)
    with generators.Interrupts() as outer:
        with generators.Interrupts() as inner:
            try:
                interrupt_myself()
            except KeyboardInterrupt:
                pass
            inside = signal.getsignal(signal.SIGINT) == inner._note
        between = signal.getsignal(signal.SIGINT) == outer._note
    queue.put((inner.noted, outer.noted, inside, between,
               signal.getsignal(signal.SIGINT) is before))


def test_nested_notes_both_take_the_interrupt():
    assert in_a_process(nested) == (True, True, True, True, True)


def handler_installed_meanwhile(queue):
    calls = []

    def later(number, frame):
        calls.append(number)

    with generators.Interrupts() as interrupts:
        signal.signal(signal.SIGINT, later)
    kept = signal.getsignal(signal.SIGINT) is later
    interrupt_myself()
    queue.put((kept, calls == [signal.SIGINT], interrupts.noted))


def handler_that_equals_everything(queue):
    class Later(object):
        calls = 0

        def __call__(self, number, frame):
            Later.calls += 1

        def __eq__(self, other):
            return True

        __hash__ = object.__hash__

    later = Later()
    with generators.Interrupts():
        signal.signal(signal.SIGINT, later)
    queue.put(signal.getsignal(signal.SIGINT) is later)


def test_handler_installed_meanwhile_is_told_apart_by_identity():
    """
        Whether the handler of the block is still installed is asked
        by identity: a handler installed meanwhile that claims to be
        equal to anything stays as well
    """

    assert in_a_process(handler_that_equals_everything) is True


def test_handler_installed_meanwhile_stays():
    """
        A handler that someone else installs while the block runs is
        not replaced at its end by the one that was there before
    """

    assert in_a_process(handler_installed_meanwhile) == (True, True, False)


def handler_not_from_python(queue):
    # (signal.getsignal returns None for a handler that was not
    # installed from Python)
    real = signal.getsignal
    generators.signal.getsignal = lambda number: None
    try:
        with generators.Interrupts() as interrupts:
            during = real(signal.SIGINT) is signal.default_int_handler
        result = ("ended", interrupts.noted, interrupts._installed, during)
    except BaseException as failure:  # pragma: no cover
        result = ("failed", repr(failure))
    finally:
        generators.signal.getsignal = real
    queue.put(result)


def test_handler_that_python_did_not_install_is_left_alone():
    assert in_a_process(handler_not_from_python) == \
        ("ended", False, False, True)


def interrupted_while_entering(queue):
    before = signal.getsignal(signal.SIGINT)
    real = signal.signal

    def installs_then_interrupt(number, handler):
        previous = real(number, handler)
        if getattr(handler, "__name__", None) == "_note":
            raise KeyboardInterrupt
        return previous

    generators.signal.signal = installs_then_interrupt
    try:
        with generators.Interrupts():
            result = "entered"
    except KeyboardInterrupt:
        result = "interrupted"
    finally:
        generators.signal.signal = real
    queue.put((result, signal.getsignal(signal.SIGINT) is before))


def test_interrupt_while_the_handler_is_installed():
    """
        An interrupt right after the handler was installed ends the
        entry of the block, whose end is then not called: the entry
        puts the handler that was there back itself
    """

    assert in_a_process(interrupted_while_entering) == ("interrupted", True)


def interrupted_at_the_end_of_the_entry(queue):
    import inspect
    import sys

    before = signal.getsignal(signal.SIGINT)
    lines, first = inspect.getsourcelines(generators.Interrupts.__enter__)
    # the line of the entry that returns once the handler is installed
    installs = [i for i, line in enumerate(lines)
                if line.strip() == "signal.signal(signal.SIGINT, self._handler)"]
    assert len(installs) == 1 and lines[installs[0] + 1].strip() == \
        "return self"
    last = first + installs[0] + 1
    sent = []

    def trace(frame, event, arg):
        if frame.f_code is generators.Interrupts.__enter__.__code__:
            if event == "line" and frame.f_lineno == last and not sent:
                sent.append(1)
                os.kill(os.getpid(), signal.SIGINT)
            return trace
        return None

    notes = generators.Interrupts()
    sys.settrace(trace)
    try:
        with notes:
            result = "entered"
    except KeyboardInterrupt:
        result = "interrupted"
    finally:
        sys.settrace(None)
    queue.put((result, sent == [1], notes.noted,
               signal.getsignal(signal.SIGINT) is before, notes._installed))


def test_interrupt_when_the_entry_returns():
    """
        An interrupt that arrives when the handler is installed and
        the entry is about to return ends the entry: the handler that
        was there is put back
    """

    assert in_a_process(interrupted_at_the_end_of_the_entry) == \
        ("interrupted", True, True, True, False)


def handler_that_raises_value_error(queue):
    def odd(number, frame):
        raise ValueError("the handler that was there")

    signal.signal(signal.SIGINT, odd)
    real = signal.signal

    def installs_then_interrupt(number, handler):
        previous = real(number, handler)
        if getattr(handler, "__name__", None) == "_note":
            # a real interrupt, right after the handler was installed
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.2)
        return previous

    generators.signal.signal = installs_then_interrupt
    notes = generators.Interrupts()
    try:
        with notes:
            result = "entered"
    except ValueError as failure:
        result = str(failure)
    finally:
        generators.signal.signal = real
    queue.put((result, notes.noted, signal.getsignal(signal.SIGINT) is odd,
               notes._installed))


def test_failure_of_the_handler_that_was_there_while_entering():
    """
        A ValueError on the way into the block (here from the handler
        that was there, called for an interrupt right after ours was
        installed) is not taken for the refusal of another thread:
        the entry fails and is undone
    """

    assert in_a_process(handler_that_raises_value_error) == \
        ("the handler that was there", True, True, False)


#: a fresh interpreter in which the module threading is imported first
#: by a thread that is not the main one: threading then takes that
#: thread for the main thread
THREADING_IMPORTED_ELSEWHERE = """
import _thread, os, signal, time
done = []


def imports():
    import threading
    done.append(threading.main_thread().ident)


_thread.start_new_thread(imports, ())
while not done:
    time.sleep(0.01)
import threading
from bpnet.generators import generators

with generators.Interrupts() as notes:
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)
    except KeyboardInterrupt:
        pass
print("main thread taken for another:",
      threading.main_thread().ident != _thread.get_ident())
print("noted:", notes.noted)
"""


def test_notes_where_threading_was_imported_by_another_thread():
    """
        Whether the handling of interrupts can be changed is not
        asked of the module threading, which can take another thread
        for the main one
    """

    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output = subprocess.run(
        [sys.executable, "-c", THREADING_IMPORTED_ELSEWHERE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=LIMIT,
        env=dict(os.environ, PYTHONPATH=repo, CUDA_VISIBLE_DEVICES="")).stdout
    assert "main thread taken for another: True" in output, output[-2000:]
    assert "noted: True" in output, output[-2000:]


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
