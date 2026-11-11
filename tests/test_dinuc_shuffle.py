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
            old_state = np.random.RandomState(seed)
            new_state = np.random.RandomState(seed)
            old = dinuc_shuffle_v0(x, count, old_state)
            new = dinuc_shuffle(x, count, new_state, compiled=compiled)
            assert new.dtype == old.dtype == dtype
            assert new.shape == old.shape == (count, len(text), 4)
            assert np.array_equal(new, old)
            # the same random numbers were drawn
            assert old_state.randint(2 ** 31) == new_state.randint(2 ** 31)

        old_state = np.random.RandomState(seed)
        new_state = np.random.RandomState(seed)
        old = dinuc_shuffle_v0(text, count, old_state)
        new = dinuc_shuffle(text, count, new_state, compiled=compiled)
        assert isinstance(new, list) and new == old
        assert all(isinstance(s, str) and len(s) == len(text) for s in new)
        assert old_state.randint(2 ** 31) == new_state.randint(2 ** 31)


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


def test_numba_that_is_missing_or_fails(monkeypatch):
    """
        Without numba, or with one that cannot compile the walk, the
        walk in Python is used: no error, the same shuffles
    """

    import sys

    rng = np.random.RandomState(6)
    x = one_hot("".join(rng.choice(list("ACGT"), size=2114)))
    expected = dinuc_shuffle_v0(x, 20, np.random.RandomState(8))

    # not installed
    monkeypatch.setattr(shaputils, "_COMPILED_WALK", [])
    monkeypatch.setitem(sys.modules, "numba", None)
    assert shaputils._compiled_walk() is None
    assert np.array_equal(dinuc_shuffle(x, 20, np.random.RandomState(8)),
                          expected)
    assert shaputils._COMPILED_WALK == [None]
    monkeypatch.undo()
    if not has_numba():
        return

    # installed, and compiling fails
    import numba

    def fails(*args, **kwargs):
        raise RuntimeError("no compiler today")

    monkeypatch.setattr(shaputils, "_COMPILED_WALK", [])
    monkeypatch.setattr(numba, "njit", fails)
    assert shaputils._compiled_walk() is None
    for compiled in (None, True):
        assert np.array_equal(
            dinuc_shuffle(x, 20, np.random.RandomState(8), compiled=compiled),
            expected)


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
