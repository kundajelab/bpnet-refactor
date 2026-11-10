"""
    Earlier implementations of functions that have since been replaced
    by faster ones, kept verbatim so the tests can show that the new
    code returns the same values.
"""

import numpy as np


def one_hot_encode_v0(sequences, seq_length):
    """
        one_hot_encode as it was before the lookup table version
        (bpnet/generators/sequtils.py at main@0b11a7c). It builds a
        Python list of integers through str.translate and supports the
        letters A C G T N Y R M S W K only; any other character raises
        ValueError.
    """

    def fix_sequence_length(sequence, length):
        if len(sequence) < length:
            sequence += 'N' * (length - len(sequence))
        elif len(sequence) > length:
            sequence = sequence[:length]
        return sequence

    sequences = list(map(
        fix_sequence_length, sequences, [seq_length] * len(sequences)))
    _sequences = ''.join(sequences)
    transtab = str.maketrans('ACGTNYRMSWK', '01234444444')
    sequences_trans = _sequences.translate(transtab)
    int_sequences = list(map(int, sequences_trans))
    encoder = np.vstack([np.eye(4), np.zeros(4)])
    X = encoder[int_sequences]
    return X.reshape(len(sequences), len(sequences[0]), 4)
