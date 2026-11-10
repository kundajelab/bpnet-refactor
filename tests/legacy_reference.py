"""
    Earlier implementations of functions that have since been replaced
    by faster ones, kept verbatim so the tests can show that the new
    code returns the same values.
"""

import logging

import numpy as np


def one_hot_encode_v0(sequences, seq_length):
    """
        one_hot_encode as it was before the lookup table version
        (bpnet/generators/sequtils.py at main@0b11a7c), with its helper
        fix_sequence_length. It turns the letters A C G T N Y R M S W K
        into the digits 0 to 4 with str.translate, converts every
        character with int() and indexes a five row matrix with the
        result. Any character int() does not accept raises ValueError;
        a digit above 4 raises IndexError.
    """

    def fix_sequence_length(sequence, length):
        if len(sequence) < length:
            sequence += 'N' * (length - len(sequence))
        elif len(sequence) > length:
            sequence = sequence[:length]
        return sequence

    if len(sequences) == 0:
        logging.error("'sequences' is empty")
        return None

    sequences = list(map(
        fix_sequence_length, sequences, [seq_length] * len(sequences)))
    _sequences = ''.join(sequences)
    transtab = str.maketrans('ACGTNYRMSWK', '01234444444')
    sequences_trans = _sequences.translate(transtab)
    int_sequences = list(map(int, sequences_trans))
    encoder = np.vstack([np.eye(4), np.zeros(4)])
    X = encoder[int_sequences]
    return X.reshape(len(sequences), len(sequences[0]), 4)


def interval_values_v0(bigwig, chrom, start, end):
    """
        How the batch generator read an interval of a bigWig file
        before interval_values (bpnet/generators/generators.py at
        main@0b11a7c): a Python list of floats from pyBigWig, converted
        by numpy.nan_to_num into a float64 array
    """

    return np.nan_to_num(bigwig.values(chrom, start, end))
