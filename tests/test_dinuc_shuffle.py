"""
    The dinucleotide shuffle (bpnet.utils.shaputils.dinuc_shuffle),
    which makes the references of bpnet-shap: the shuffles are those
    of the implementation it had before it was made faster
    (tests/legacy_reference.py), for the same random state, whichever
    way the walk is done; and they are dinucleotide shuffles.
"""

import collections

import numpy as np
import pytest

from bpnet.utils import shaputils
from bpnet.utils.shaputils import dinuc_shuffle

from tests.legacy_reference import dinuc_shuffle_v0


def has_numba():
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


WAYS = [False, pytest.param(True, marks=pytest.mark.skipif(
    not has_numba(), reason="numba is not installed"))]


def random_state(seed):
    """
        A random state that has drawn before, with a normal variate
        in its cache (part of its state, like the generator's array)
    """

    state = np.random.RandomState(seed)
    state.normal()
    assert state.get_state()[3] == 1
    return state


def assert_same_state(a, b):
    """
        Two random states are in the same state: the generator's
        array, the position in it, and the cached normal variate
    """

    a, b = a.get_state(), b.get_state()
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
    assert a[2:] == b[2:]


def one_hot(text, dtype=np.int8):
    out = np.zeros((len(text), 4), dtype=dtype)
    for i, base in enumerate(text):
        if base in "ACGT":
            out[i, "ACGT".index(base)] = 1
    return out


def sequences():
    rng = np.random.RandomState(0)
    texts = {
        "one base": "A",
        "two bases": "AC",
        "three bases": "ACA",
        "one letter": "A" * 50,
        "two letters in turn": "AC" * 40,
        "ends with a letter of its own": "ACGCAGCAGACT",
        "starts with a letter of its own": "TACGCAGCAGAC",
        "with N": "ACGTNNACGTTGCANACGTN",
        "only N": "N" * 12,
        "306 bases": "".join(rng.choice(list("ACGT"), size=306)),
        "2114 bases": "".join(rng.choice(list("ACGT"), size=2114)),
        "2114 bases, some N": "".join(rng.choice(
            list("ACGTN"), size=2114, p=[0.24] * 4 + [0.04])),
        "low complexity": "".join(rng.choice(
            list("ACGT"), size=500, p=[0.9, 0.05, 0.03, 0.02])),
    }
    return texts


@pytest.mark.parametrize("compiled", WAYS)
@pytest.mark.parametrize("name", sorted(sequences()))
def test_same_shuffles_as_before(name, compiled):
    text = sequences()[name]
    for seed, count in ((20210304, 20), (1, 1), (7, 3), (3, 0)):
        for dtype in (np.int8, np.float64, np.float32):
            x = one_hot(text, dtype)
            old_state, new_state = random_state(seed), random_state(seed)
            old = dinuc_shuffle_v0(x, count, old_state)
            new = dinuc_shuffle(x, count, new_state, compiled=compiled)
            assert new.dtype == old.dtype == dtype
            assert new.shape == old.shape == (count, len(text), 4)
            assert np.array_equal(new, old)
            # the random state is where the previous implementation
            # leaves it
            assert_same_state(old_state, new_state)

        old_state, new_state = random_state(seed), random_state(seed)
        old = dinuc_shuffle_v0(text, count, old_state)
        new = dinuc_shuffle(text, count, new_state, compiled=compiled)
        assert isinstance(new, list) and new == old
        assert all(isinstance(s, str) and len(s) == len(text) for s in new)
        assert_same_state(old_state, new_state)
        # a state that has not drawn before
        old_state = np.random.RandomState(seed)
        new_state = np.random.RandomState(seed)
        assert dinuc_shuffle(text, count, new_state, compiled=compiled) == \
            dinuc_shuffle_v0(text, count, old_state)
        assert_same_state(old_state, new_state)


def pairs(text):
    return collections.Counter(zip(text[:-1], text[1:]))


@pytest.mark.parametrize("compiled", WAYS)
def test_shuffles_keep_the_dinucleotides(compiled):
    """
        What a dinucleotide shuffle is, whatever the implementation:
        every shuffle has the dinucleotides of the sequence, each as
        often, starts and ends as the sequence does, and the shuffles
        differ from the sequence and from each other
    """

    rng = np.random.RandomState(5)
    text = "".join(rng.choice(list("ACGTN"), size=400,
                              p=[0.3, 0.2, 0.2, 0.25, 0.05]))
    shuffles = dinuc_shuffle(text, 30, np.random.RandomState(11),
                             compiled=compiled)
    assert len(set(shuffles)) == 30 and text not in shuffles
    for shuffled in shuffles:
        assert pairs(shuffled) == pairs(text)
        assert shuffled[0] == text[0] and shuffled[-1] == text[-1]

    # as one hot arrays (positions without a base stay without one)
    arrays = dinuc_shuffle(one_hot(text), 30, np.random.RandomState(11),
                           compiled=compiled)
    as_text = ["".join("ACGTN"[4 if not row.any() else int(row.argmax())]
                       for row in array) for array in arrays]
    assert len(set(as_text)) == 30 and text not in as_text
    for array, shuffled in zip(arrays, as_text):
        assert set(np.unique(array)) <= {0, 1} and array.sum(axis=1).max() == 1
        assert pairs(shuffled) == pairs(text)
        assert shuffled[0] == text[0] and shuffled[-1] == text[-1]
    # (the shuffles of the text are others: the random numbers go to
    # the letters in the order of the alphabet, in which N comes
    # before T. Without N the two are the same shuffles.)
    assert as_text != shuffles
    plain = text.replace("N", "")
    arrays = dinuc_shuffle(one_hot(plain), 5, np.random.RandomState(11),
                           compiled=compiled)
    shuffles = dinuc_shuffle(plain, 5, np.random.RandomState(11),
                             compiled=compiled)
    for array, shuffled in zip(arrays, shuffles):
        assert np.array_equal(array, one_hot(shuffled))


def test_same_shuffles_whichever_way_the_walk_is_done(monkeypatch):
    """
        Left to itself the function uses the compiled walk for calls
        of the size bpnet-shap makes (20 shuffles of 2,114 bases) and
        the walk in Python for small ones; without numba always the
        latter. The shuffles are the same.
    """

    used = []
    real = shaputils._compiled_walk

    def recording():
        used.append(True)
        return real()

    monkeypatch.setattr(shaputils, "_compiled_walk", recording)
    rng = np.random.RandomState(3)
    long = one_hot("".join(rng.choice(list("ACGT"), size=2114)))
    short = long[:300]
    expected_long = dinuc_shuffle_v0(long, 20, np.random.RandomState(4))
    expected_short = dinuc_shuffle_v0(short, 20, np.random.RandomState(4))

    assert np.array_equal(
        dinuc_shuffle(short, 20, np.random.RandomState(4)), expected_short)
    assert used == []
    assert 20 * 300 < shaputils.COMPILED_FROM <= 20 * 2114
    assert np.array_equal(
        dinuc_shuffle(long, 20, np.random.RandomState(4)), expected_long)
    assert used == [True]
    assert np.array_equal(
        dinuc_shuffle(long, 20, np.random.RandomState(4), compiled=False),
        expected_long)
    assert used == [True]

    # numba not installed, or not able to compile: the walk in Python
    monkeypatch.setattr(shaputils, "_compiled_walk", lambda: None)
    for compiled in (None, True):
        assert np.array_equal(
            dinuc_shuffle(long, 20, np.random.RandomState(4),
                          compiled=compiled), expected_long)


def test_compiled_walk_is_made_once(monkeypatch):
    monkeypatch.setattr(shaputils, "_COMPILED_WALK", [])
    first = shaputils._compiled_walk()
    assert shaputils._compiled_walk() is first
    assert len(shaputils._COMPILED_WALK) == 1
    assert (first is not None) == has_numba()
    if first is not None:
        tokens = np.array([2, 0, 1, 0, 2, 1])
        successors = [np.array([2, 4]), np.array([3]), np.array([1, 5])]
        assert np.array_equal(first(tokens, successors),
                              shaputils._walk(tokens, successors))


def test_numba_that_is_missing_or_cannot_be_used(monkeypatch):
    """
        Without numba, or with one that cannot be loaded, cannot
        compile the walk, or does not keep its compiler lock where
        this module takes it from, the walk in Python is used: no
        error, the same shuffles
    """

    import builtins
    import sys

    rng = np.random.RandomState(6)
    x = one_hot("".join(rng.choice(list("ACGT"), size=2114)))
    expected = dinuc_shuffle_v0(x, 20, np.random.RandomState(8))

    def falls_back():
        monkeypatch.setattr(shaputils, "_COMPILED_WALK", [])
        assert shaputils._compiled_walk() is None
        for compiled in (None, True):
            assert np.array_equal(
                dinuc_shuffle(x, 20, np.random.RandomState(8),
                              compiled=compiled), expected)
        assert shaputils._COMPILED_WALK == [None]

    # not installed
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "numba", None)
        falls_back()
    if not has_numba():
        return

    import numba
    from numba.core import compiler_lock

    # installed, and its libraries cannot be loaded
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise OSError("cannot open shared object file")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", broken_import)
        falls_back()

    # compiling fails
    def fails(*args, **kwargs):
        raise RuntimeError("no compiler today")

    with monkeypatch.context() as patch:
        patch.setattr(numba, "njit", fails)
        falls_back()
    # (and the compiler's lock was let go)
    assert not compiler_lock.global_compiler_lock._lock._is_owned()

    # a numba that keeps its lock elsewhere, or whose lock is taken
    # in another way
    class NoLock(object):
        pass

    class OtherLock(object):
        class _lock(object):
            @staticmethod
            def acquire():
                return True

            @staticmethod
            def release():
                pass

    for other in (NoLock(), OtherLock()):
        with monkeypatch.context() as patch:
            patch.setattr(compiler_lock, "global_compiler_lock", other)
            falls_back()

    # none of this is remembered by the next process, and numba works
    monkeypatch.setattr(shaputils, "_COMPILED_WALK", [])
    assert shaputils._compiled_walk() is not None


THREADS = r"""
import threading
import numpy as np
import numba
from numba.core.compiler_lock import global_compiler_lock
from bpnet.utils import shaputils
from bpnet.utils.shaputils import dinuc_shuffle
from tests.legacy_reference import dinuc_shuffle_v0

rng = np.random.RandomState(9)
x = np.zeros((2114, 4), dtype=np.int8)
x[np.arange(2114), rng.randint(0, 4, size=2114)] = 1
expected = dinuc_shuffle_v0(x, 20, np.random.RandomState(8))


def shuffle():
    return dinuc_shuffle(x, 20, np.random.RandomState(8), compiled=True)


def elsewhere(function):
    # the result of a function in another thread
    result = []
    thread = threading.Thread(target=lambda: result.append(function()))
    thread.start()
    thread.join()
    return result[0]


def lock_is_free():
    def probe():
        if global_compiler_lock._lock.acquire(False):
            global_compiler_lock._lock.release()
            return True
        return False
    return elsewhere(probe)


# another thread compiles the walk: this one does not wait
with shaputils._COMPILING:
    assert elsewhere(shaputils._compiled_walk) is None
    assert np.array_equal(elsewhere(shuffle), expected)
assert shaputils._COMPILED_WALK == []

# numba compiles something in another thread: the same
with global_compiler_lock:
    assert elsewhere(shaputils._compiled_walk) is None
    assert np.array_equal(elsewhere(shuffle), expected)
assert shaputils._COMPILED_WALK == [] and lock_is_free()

# a compilation that fails lets go of numba's lock
real = numba.njit
numba.njit = lambda *args, **kwargs: 1 / 0
assert shaputils._compiled_walk() is None
assert shaputils._COMPILED_WALK == [None] and lock_is_free()
numba.njit = real
shaputils._COMPILED_WALK[:] = []

# numba's lock is held while the walk is compiled, by this thread, and
# free afterwards
held = []


def watching(*args, **kwargs):
    held.append(global_compiler_lock._lock._is_owned())
    return real(*args, **kwargs)


numba.njit = watching
start = threading.Event()
results = []


def one_of_several():
    start.wait()
    results.append(shuffle())


threads = [threading.Thread(target=one_of_several) for _ in range(6)]
for thread in threads:
    thread.start()
start.set()
for thread in threads:
    thread.join()
numba.njit = real
# several threads at once: one walk was compiled, all shuffles are right
# (numba calls njit for pieces of its own while it compiles: every
# call found the lock held by its thread)
assert held and all(held), held
assert len(results) == 6
assert all(np.array_equal(result, expected) for result in results)
assert len(shaputils._COMPILED_WALK) == 1
assert shaputils._COMPILED_WALK[0] is not None
assert shaputils._compiled_walk() is shaputils._COMPILED_WALK[0]
assert lock_is_free() and not shaputils._COMPILING.locked()
assert np.array_equal(shuffle(), expected)
print("all as expected")
"""


FORKS = r"""
import os, signal, sys, threading
import numpy as np
from numba.core.compiler_lock import global_compiler_lock
from bpnet.utils import shaputils
from bpnet.utils.shaputils import dinuc_shuffle
from tests.legacy_reference import dinuc_shuffle_v0

text = "ACGT" * 600
expected = dinuc_shuffle_v0(text, 20, np.random.RandomState(1))


def right():
    return dinuc_shuffle(text, 20, np.random.RandomState(1)) == expected


def in_a_child(check):
    # the exit status of a forked process that runs the check, which
    # must be through within a minute
    pid = os.fork()
    if pid == 0:
        signal.alarm(60)
        try:
            ok = check()
        except BaseException:
            ok = False
        os._exit(0 if ok else 3)
    return os.waitpid(pid, 0)[1]


def in_a_new_thread(check):
    result = []
    thread = threading.Thread(target=lambda: result.append(check()))
    thread.start()
    thread.join()
    return result[0]


def walks_in_python():
    # right shuffles, from the main thread and from a new one (which
    # may get the identity of a thread of the parent), nothing compiled
    return right() and in_a_new_thread(right) and \
        shaputils._COMPILED_WALK == [] and \
        shaputils._compiled_walk() is None


# a thread of this process holds the compiler's lock, as a thread
# that compiles does; a process forked now inherits it held, by nobody
held, release = threading.Event(), threading.Event()


def hold_compiler():
    with global_compiler_lock:
        held.set()
        release.wait()


thread = threading.Thread(target=hold_compiler)
thread.start()
held.wait()
assert in_a_child(walks_in_python) == 0
# and a child of such a child
assert in_a_child(lambda: in_a_child(walks_in_python) == 0) == 0
release.set()
thread.join()

# forked while nothing is compiling, before the walk is compiled: the
# child does not compile either (it cannot know)
assert shaputils._COMPILED_WALK == []
assert in_a_child(walks_in_python) == 0

# this process compiles; a process forked afterwards has the walk
assert right() and shaputils._COMPILED_WALK[0] is not None
compiled = shaputils._COMPILED_WALK[0]


def uses_the_compiled_walk():
    return right() and in_a_new_thread(right) and \
        shaputils._compiled_walk() is compiled


assert in_a_child(uses_the_compiled_walk) == 0
print("all as expected")
"""


def in_a_process(script):
    import os
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=repo,
        env=dict(os.environ, PYTHONPATH=repo, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=180)
    assert done.returncode == 0, done.stdout[-3000:]
    assert "all as expected" in done.stdout, done.stdout[-3000:]


def test_nothing_waits_for_a_compilation():
    """
        One thread compiles the walk, holding numba's compiler lock
        for the time; a thread that asks meanwhile, or while numba
        compiles something else, walks in Python instead of waiting.
        Several threads at once get one compiled walk and the same
        shuffles, and the locks are free afterwards, also when the
        compilation fails. (In a process of its own: a thread that
        waited after all would be left behind.)
    """

    if not has_numba():
        pytest.skip("numba is not installed")
    in_a_process(THREADS)


def test_forked_processes_do_not_compile():
    """
        A process that is forked while a thread of its parent compiles
        with numba inherits the compiler's lock, held by nobody: a
        shuffle in it must not compile (it would wait forever). No
        forked process compiles; one that is forked after its parent
        compiled the walk uses that. The shuffles are the same in all.
    """

    if not has_numba():
        pytest.skip("numba is not installed")
    in_a_process(FORKS)


def test_walk_that_another_thread_has_just_compiled_is_used(monkeypatch):
    """
        A thread that finds no compiled walk, and gets the lock only
        after the thread that compiled it has let go, takes that walk
        and does not compile another
    """

    class JustReleased(object):
        def acquire(self, blocking=True):
            shaputils._COMPILED_WALK.append("the walk of the other thread")
            return True

        def release(self):
            pass

    monkeypatch.setattr(shaputils, "_COMPILED_WALK", [])
    monkeypatch.setattr(shaputils, "_COMPILING", JustReleased())
    assert shaputils._compiled_walk() == "the walk of the other thread"
    assert shaputils._COMPILED_WALK == ["the walk of the other thread"]


def test_without_a_random_state():
    # an unseeded one is made: shuffles differ from call to call
    text = "ACGTTGCAAGCTTAGCCGATCGATTAGCATCGACTAGCTAGCATCGAT" * 3
    a, b = dinuc_shuffle(text, 5), dinuc_shuffle(text, 5)
    assert a != b
    assert all(pairs(s) == pairs(text) for s in a + b)


def test_what_is_not_a_sequence():
    for value in (["A", "C"], np.zeros(8), np.zeros((2, 3, 4)), 5, None,
                  b"ACGT"):
        with pytest.raises(ValueError,
                           match="Expected string or one-hot encoded array"):
            dinuc_shuffle(value, 2, np.random.RandomState(0))
        with pytest.raises(ValueError,
                           match="Expected string or one-hot encoded array"):
            dinuc_shuffle_v0(value, 2, np.random.RandomState(0))
    # an empty sequence has no shuffle (and none is asked for when
    # the number of shuffles is zero), as before
    for empty in ("", np.zeros((0, 4), dtype=np.int8)):
        for function in (dinuc_shuffle, dinuc_shuffle_v0):
            with pytest.raises(IndexError):
                function(empty, 1, np.random.RandomState(0))
            assert len(function(empty, 0, np.random.RandomState(0))) == 0
