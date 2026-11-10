"""
    Characterization tests for the sequence helpers in
    bpnet.generators.sequtils: one hot encoding, reverse complement,
    rounding
"""

import numpy as np
import pytest

from bpnet.generators import sequtils

from tests.checks import one_hot


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


@pytest.mark.xfail(strict=True, reason="the IUPAC codes B, D, H, V (and any "
                   "other letter) raise ValueError instead of encoding as "
                   "zeros like N")
@pytest.mark.parametrize("letter", list("BDHV"))
def test_one_hot_encode_other_iupac_codes_are_zeros(letter):
    result = sequtils.one_hot_encode(["A" + letter + "T"], 3)
    assert np.array_equal(result[0], one_hot("ANT"))


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
