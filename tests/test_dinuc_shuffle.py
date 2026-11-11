"""
    The dinucleotide shuffle (bpnet.utils.shaputils.dinuc_shuffle),
    which makes the references of bpnet-shap: the shuffles are those
    of the implementation it had before it was made faster
    (tests/legacy_reference.py), for the same random state, which is
    left in the same state; and they are dinucleotide shuffles.
"""

import collections

import numpy as np
import pytest

from bpnet.utils import shaputils
from bpnet.utils.shaputils import dinuc_shuffle

from tests.legacy_reference import dinuc_shuffle_v0


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


@pytest.mark.parametrize("name", sorted(sequences()))
def test_same_shuffles_as_before(name):
    text = sequences()[name]
    for seed, count in ((20210304, 20), (1, 1), (7, 3), (3, 0)):
        for dtype in (np.int8, np.float64, np.float32):
            x = one_hot(text, dtype)
            old_state, new_state = random_state(seed), random_state(seed)
            old = dinuc_shuffle_v0(x, count, old_state)
            new = dinuc_shuffle(x, count, new_state)
            assert new.dtype == old.dtype == dtype
            assert new.shape == old.shape == (count, len(text), 4)
            assert np.array_equal(new, old)
            # the random state is where the previous implementation
            # leaves it
            assert_same_state(old_state, new_state)

        old_state, new_state = random_state(seed), random_state(seed)
        old = dinuc_shuffle_v0(text, count, old_state)
        new = dinuc_shuffle(text, count, new_state)
        assert isinstance(new, list) and new == old
        assert all(isinstance(s, str) and len(s) == len(text) for s in new)
        assert_same_state(old_state, new_state)
        # a state that has not drawn before
        old_state = np.random.RandomState(seed)
        new_state = np.random.RandomState(seed)
        assert dinuc_shuffle(text, count, new_state) == \
            dinuc_shuffle_v0(text, count, old_state)
        assert_same_state(old_state, new_state)


def test_arrays_as_they_come():
    """
        Views that are not contiguous or cannot be written, other
        integer types, rows that are not one hot: what the previous
        implementation returns for them
    """

    rng = np.random.RandomState(4)
    x = one_hot("".join(rng.choice(list("ACGTN"), size=300)))
    reverse = x[::-1]
    locked = x.copy()
    locked.setflags(write=False)
    two_ones = x.copy()
    two_ones[5] = [1, 0, 1, 0]
    other_values = x.astype(np.float64) * 0.5
    wide = np.zeros((300, 6), dtype=np.int16)
    wide[np.arange(300), rng.randint(0, 6, size=300)] = 1
    # more kinds of position than a small integer counts
    many = np.zeros((600, 300), dtype=np.int8)
    many[np.arange(600), rng.randint(0, 300, size=600)] = 1
    for array in (reverse, locked, two_ones, other_values, wide, many,
                  np.eye(130, dtype=np.int8), x.astype(np.uint8),
                  x.astype(bool), np.asfortranarray(x)):
        old_state, new_state = random_state(9), random_state(9)
        old = dinuc_shuffle_v0(array, 4, old_state)
        new = dinuc_shuffle(array, 4, new_state)
        assert new.dtype == old.dtype and np.array_equal(new, old)
        assert_same_state(old_state, new_state)


def pairs(text):
    return collections.Counter(zip(text[:-1], text[1:]))


def test_shuffles_keep_the_dinucleotides():
    """
        What a dinucleotide shuffle is, whatever the implementation:
        every shuffle has the dinucleotides of the sequence, each as
        often, starts and ends as the sequence does, and the shuffles
        differ from the sequence and from each other
    """

    rng = np.random.RandomState(5)
    text = "".join(rng.choice(list("ACGTN"), size=400,
                              p=[0.3, 0.2, 0.2, 0.25, 0.05]))
    shuffles = dinuc_shuffle(text, 30, np.random.RandomState(11))
    assert len(set(shuffles)) == 30 and text not in shuffles
    for shuffled in shuffles:
        assert pairs(shuffled) == pairs(text)
        assert shuffled[0] == text[0] and shuffled[-1] == text[-1]

    # as one hot arrays (positions without a base stay without one)
    arrays = dinuc_shuffle(one_hot(text), 30, np.random.RandomState(11))
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
    arrays = dinuc_shuffle(one_hot(plain), 5, np.random.RandomState(11))
    shuffles = dinuc_shuffle(plain, 5, np.random.RandomState(11))
    for array, shuffled in zip(arrays, shuffles):
        assert np.array_equal(array, one_hot(shuffled))


def test_walk_over_the_positions():
    """
        The walk that builds a shuffle: from the first position, and
        from a position with token t to the next entry of the list of
        successors of t
    """

    # positions       0  1  2  3  4  5
    tokens = np.array([2, 0, 1, 0, 2, 1])
    # after a 0: to position 2, then 4; after a 1: to 3, then 1;
    # after a 2: to 5
    successors = [np.array([2, 4]), np.array([3, 1]), np.array([5, 1])]
    # 0 (a 2) -> 5 (a 1) -> 3 (a 0) -> 2 (a 1) -> 1 (a 0) -> 4 (a 2)
    walked = shaputils._walk(tokens, successors)
    assert walked.tolist() == [2, 1, 0, 1, 0, 2]
    assert walked.dtype == tokens.dtype
    # one position: nothing to walk
    assert shaputils._walk(np.array([3]), [np.array([], dtype=int)] * 4
                           ).tolist() == [3]


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
    # letters outside ASCII cannot be shuffled as text, as before: the
    # failure comes after the random numbers of the first shuffle
    states = []
    for function in (dinuc_shuffle, dinuc_shuffle_v0):
        state = random_state(7)
        with pytest.raises(UnicodeDecodeError):
            function("AC\u00e9GT" * 5, 3, state)
        states.append(state)
    assert_same_state(*states)
    assert states[0].get_state()[2] != random_state(7).get_state()[2]
