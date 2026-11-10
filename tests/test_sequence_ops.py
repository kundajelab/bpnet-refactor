"""
    Characterization tests for the sequence helpers in
    bpnet.generators.sequtils: one hot encoding, reverse complement,
    rounding
"""

import numpy as np
import pytest

from bpnet.generators import sequtils

from tests.checks import one_hot
from tests.legacy_reference import one_hot_encode_v0


# --------------------------------------------------------------------
# one_hot_encode
# --------------------------------------------------------------------

def test_one_hot_encode_matches_reference():
    rng = np.random.RandomState(0)
    sequences = ["".join(rng.choice(list("ACGTN"), size=50))
                 for _ in range(20)]
    result = sequtils.one_hot_encode(sequences, 50)
    assert result.shape == (20, 50, 4)
    assert result.dtype == np.float64
    for i, sequence in enumerate(sequences):
        assert np.array_equal(result[i], one_hot(sequence))


def test_one_hot_encode_column_order_is_acgt():
    result = sequtils.one_hot_encode(["ACGT"], 4)
    assert np.array_equal(result[0], np.eye(4))


@pytest.mark.parametrize("letter", list("NYRMSWK"))
def test_one_hot_encode_ambiguity_codes_are_zeros(letter):
    result = sequtils.one_hot_encode(["A" + letter + "T"], 3)
    assert np.array_equal(result[0], one_hot("ANT"))


def test_one_hot_encode_pads_with_n_and_truncates():
    result = sequtils.one_hot_encode(["AC", "ACGTACGT", "ACGT"], 4)
    assert result.shape == (3, 4, 4)
    assert np.array_equal(result[0], one_hot("ACNN"))
    assert np.array_equal(result[1], one_hot("ACGT"))
    assert np.array_equal(result[2], one_hot("ACGT"))


def test_one_hot_encode_empty_list_returns_none():
    assert sequtils.one_hot_encode([], 4) is None


def test_fix_sequence_length():
    assert sequtils.fix_sequence_length("ACGT", 4) == "ACGT"
    assert sequtils.fix_sequence_length("AC", 4) == "ACNN"
    assert sequtils.fix_sequence_length("ACGTAC", 4) == "ACGT"
    assert sequtils.fix_sequence_length("", 2) == "NN"


@pytest.mark.parametrize("letter", list("BDHVUX-*.? \n56789") + [
    "\u00e9", "\u00b2", "\u0665"])
def test_one_hot_encode_any_other_character_is_zeros(letter):
    # the remaining IUPAC codes and anything else: zeros, like N. The
    # previous implementation raised ValueError or IndexError for these
    result = sequtils.one_hot_encode(["A" + letter + "T"], 3)
    assert np.array_equal(result[0], one_hot("ANT"))
    with pytest.raises((ValueError, IndexError)):
        one_hot_encode_v0(["A" + letter + "T"], 3)


def test_one_hot_encode_lower_case_bases():
    result = sequtils.one_hot_encode(["acgtn", "AcGtN"], 5)
    assert np.array_equal(result[0], one_hot("ACGTN"))
    assert np.array_equal(result[1], one_hot("ACGTN"))


#: every character the previous implementation accepted: the letters
#: it knew, and the digits 0 to 4 (it encoded a digit like the letter
#: it would have translated to that digit), in ASCII or as a decimal
#: digit of another script
PREVIOUSLY_ACCEPTED = list("ACGTNYRMSWK01234") + [
    "\u0660", "\u0661", "\u0662", "\u0663", "\u0664",   # Arabic-Indic
    "\uff10", "\uff13", "\uff14",                       # fullwidth
    "\U0001d7ce", "\U0001d7d1"]                          # mathematical bold


@pytest.mark.parametrize("character", PREVIOUSLY_ACCEPTED)
def test_one_hot_encode_of_every_previously_accepted_character(character):
    new = sequtils.one_hot_encode(["A" + character + "T", character], 3)
    old = one_hot_encode_v0(["A" + character + "T", character], 3)
    assert np.array_equal(new, old)
    assert new.dtype == old.dtype


def test_one_hot_encode_digits_are_bases():
    result = sequtils.one_hot_encode(["01234", "\u0660\u0661\u0662\u0663\u0664"], 5)
    assert np.array_equal(result[0], one_hot("ACGTN"))
    assert np.array_equal(result[1], one_hot("ACGTN"))


@pytest.mark.parametrize("sequences,length", [
    ([], 4), (["", ""], 0), ([""], 0), ([""], 3), (["AC"], 2.0),
    ([np.str_("ACGT"), np.str_("NN")], 4), (("ACG", "T"), 3),
    (["ACGT"], 0)])
def test_one_hot_encode_unusual_arguments_as_before(sequences, length):
    new = sequtils.one_hot_encode(sequences, length)
    old = one_hot_encode_v0(sequences, length)
    if old is None:
        assert new is None
    else:
        assert new.shape == old.shape and new.dtype == old.dtype
        assert np.array_equal(new, old)


@pytest.mark.parametrize("sequences,length", [
    ([b"ACGT"], 4), (["AC"], 3.5), ([None], 2), ("ACGT", None)])
def test_one_hot_encode_arguments_rejected_as_before(sequences, length):
    with pytest.raises(TypeError):
        one_hot_encode_v0(sequences, length)
    with pytest.raises(TypeError):
        sequtils.one_hot_encode(sequences, length)


@pytest.mark.parametrize("seed", range(3))
def test_one_hot_encode_equals_the_previous_implementation(seed):
    """
        Same values and same type as the implementation it replaced, on
        every character that one accepted, with sequences shorter and
        longer than the requested length
    """

    rng = np.random.RandomState(seed)
    letters = PREVIOUSLY_ACCEPTED
    sequences = ["".join(rng.choice(letters, size=rng.randint(1, 400)))
                 for _ in range(50)]
    for length in (1, 50, 200, 400, 450):
        new = sequtils.one_hot_encode(sequences, length)
        old = one_hot_encode_v0(sequences, length)
        assert new.dtype == old.dtype == np.float64
        assert new.shape == old.shape == (50, length, 4)
        assert np.array_equal(new, old)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.float32,
                                   np.float64, bool])
def test_one_hot_encode_dtype(dtype):
    rng = np.random.RandomState(0)
    sequences = ["".join(rng.choice(list("ACGTN"), size=30))
                 for _ in range(4)]
    result = sequtils.one_hot_encode(sequences, 30, dtype=dtype)
    assert result.dtype == np.dtype(dtype)
    assert np.array_equal(result, sequtils.one_hot_encode(sequences, 30))
    # the result is the caller's own array
    assert result.flags.writeable
    result[:] = 0
    assert sequtils.one_hot_encode(sequences, 30).sum() > 0


def test_one_hot_encode_has_at_most_one_base_per_position():
    rng = np.random.RandomState(1)
    sequences = ["".join(chr(c) for c in rng.randint(32, 127, size=200))
                 for _ in range(20)]
    result = sequtils.one_hot_encode(sequences, 200)
    assert set(np.unique(result)) <= {0.0, 1.0}
    assert result.sum(axis=2).max() == 1
    for i, sequence in enumerate(sequences):
        # lower case letters and the digits 0 to 3 are bases
        assert np.array_equal(result[i], one_hot(
            sequence.upper().translate(str.maketrans("0123", "ACGT"))))


# --------------------------------------------------------------------
# reverse complement of sequences
# --------------------------------------------------------------------

def test_reverse_complement_of_sequences():
    assert sequtils.reverse_complement_of_sequences(
        ["A", "ACGT", "AAGGTTCC", "NACGN", ""]) == \
        ["T", "ACGT", "GGAACCTT", "NCGTN", ""]
    assert sequtils.reverse_complement_of_sequences([]) == []


def test_reverse_complement_of_sequences_is_an_involution():
    rng = np.random.RandomState(1)
    sequences = ["".join(rng.choice(list("ACGTN"), size=n))
                 for n in (1, 7, 50, 200)]
    once = sequtils.reverse_complement_of_sequences(sequences)
    assert sequtils.reverse_complement_of_sequences(once) == sequences


def test_reverse_complement_agrees_with_one_hot_flip():
    """
        Reverse complementing the string and flipping both axes of the
        one hot encoding are the same operation
    """

    rng = np.random.RandomState(2)
    sequences = ["".join(rng.choice(list("ACGTN"), size=40))
                 for _ in range(10)]
    forward = sequtils.one_hot_encode(sequences, 40)
    reverse = sequtils.one_hot_encode(
        sequtils.reverse_complement_of_sequences(sequences), 40)
    assert np.array_equal(reverse, forward[:, ::-1, ::-1])


# --------------------------------------------------------------------
# reverse complement of profiles
# --------------------------------------------------------------------

def test_reverse_complement_of_profiles_stranded():
    rng = np.random.RandomState(3)
    profiles = rng.poisson(3.0, size=(5, 30, 4)).astype(np.float32)
    result = sequtils.reverse_complement_of_profiles(profiles, stranded=True)
    assert result.shape == profiles.shape
    assert result.dtype == profiles.dtype
    # positions flip; within every (plus, minus) pair the strands swap
    assert np.array_equal(result[:, :, 0], profiles[:, ::-1, 1])
    assert np.array_equal(result[:, :, 1], profiles[:, ::-1, 0])
    assert np.array_equal(result[:, :, 2], profiles[:, ::-1, 3])
    assert np.array_equal(result[:, :, 3], profiles[:, ::-1, 2])


def test_reverse_complement_of_profiles_unstranded():
    rng = np.random.RandomState(4)
    profiles = rng.poisson(3.0, size=(5, 30, 3)).astype(np.float32)
    result = sequtils.reverse_complement_of_profiles(profiles, stranded=False)
    assert np.array_equal(result, profiles[:, ::-1, :])


@pytest.mark.parametrize("stranded,channels", [(True, 2), (True, 4),
                                               (False, 1), (False, 3)])
def test_reverse_complement_of_profiles_is_an_involution(stranded, channels):
    rng = np.random.RandomState(5)
    profiles = rng.poisson(3.0, size=(4, 25, channels)).astype(np.float64)
    once = sequtils.reverse_complement_of_profiles(profiles, stranded)
    twice = sequtils.reverse_complement_of_profiles(once, stranded)
    assert np.array_equal(twice, profiles)
    # total counts per example are unchanged
    assert np.array_equal(once.sum(axis=(1, 2)), profiles.sum(axis=(1, 2)))


def test_reverse_complement_of_profiles_does_not_modify_its_input():
    profiles = np.arange(2 * 6 * 2, dtype=np.float32).reshape(2, 6, 2)
    copy = profiles.copy()
    sequtils.reverse_complement_of_profiles(profiles, stranded=True)
    assert np.array_equal(profiles, copy)


# --------------------------------------------------------------------
# round_to_multiple
# --------------------------------------------------------------------

@pytest.mark.parametrize("x,y,smallest,expected", [
    (500, 64, False, 448), (500, 64, True, 512),
    (512, 64, False, 512), (512, 64, True, 512),
    (10, 64, False, 0), (10, 64, True, 64),
    (0, 64, False, 0), (0, 64, True, 0),
    (7, 1, False, 7), (7, 1, True, 7)])
def test_round_to_multiple(x, y, smallest, expected):
    assert sequtils.round_to_multiple(x, y, smallest=smallest) == expected
