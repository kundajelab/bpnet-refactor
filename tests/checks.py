"""
    Reference computations for the tests: what a batch must contain,
    derived from the arrays the synthetic dataset builder wrote, with
    code that shares nothing with the package (no pyBigWig, no pyfaidx,
    no bpnet imports).
"""

import numpy as np

BASES = "ACGT"


def one_hot(sequence):
    """
        One hot encoding of a DNA string, N and other letters as zeros

        Args:
            sequence (str): upper case DNA sequence

        Returns:
            numpy.ndarray: array of shape (len(sequence), 4)
    """

    out = np.zeros((len(sequence), 4), dtype=np.float64)
    for i, base in enumerate(sequence):
        if base in BASES:
            out[i, BASES.index(base)] = 1.0
    return out


def gaussian_smooth(values, sigma, window):
    """
        Gaussian smoothing of a 1D array with a kernel of `window`
        positions (odd), the array mirrored at its ends

        This is written out here (weights exp(-x^2 / (2 sigma^2)) for x
        in -(window - 1) / 2 .. (window - 1) / 2, normalized to 1) so
        that the tests do not use the package's own smoothing function
        as their reference.
    """

    radius = (int(window) - 1) // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(np.asarray(values, dtype=np.float64), radius,
                    mode="symmetric")
    return np.convolve(padded, kernel[::-1], mode="valid")


def unwrap(value):
    """
        The generator currently stores some of its extra inputs as
        1-tuples; accept both forms
    """

    if isinstance(value, tuple):
        assert len(value) == 1
        value = value[0]
    return np.asarray(value)


def reverse_complement_tracks(tracks):
    """
        Reverse complement of one task's tracks, shape (length, n):
        positions are flipped and, for two tracks, the strands swapped
    """

    if tracks.shape[1] == 2:
        return tracks[::-1, ::-1]
    return tracks[::-1]


def expected_example(dataset, chrom, start, end, rev_comp, input_len,
                     smoothing_fn=gaussian_smooth):
    """
        What the generator must produce for one example

        Args:
            dataset (dict): return value of tests.synth.make_dataset
            chrom (str): chromosome of the example
            start (int): start of the output window
            end (int): end of the output window
            rev_comp (bool): whether the example is reverse complemented
            input_len (int): length of the input sequence
            smoothing_fn (callable): function(array, sigma, window)
                used for the smoothed control tracks

        Returns:
            dict: 'sequence' (input_len, 4), 'profile' (output_len,
                total signal tracks), 'logcounts' (total signal tracks),
                'profile_bias' and 'counts_bias' (dicts keyed by task
                index, only for tasks that have control tracks)
    """

    mid = (start + end) // 2
    flank = input_len // 2
    # soft-masked (lower case) bases are read as their upper case base
    sequence = one_hot(
        dataset["sequences"][chrom][mid - flank:mid + flank].upper())
    if rev_comp:
        sequence = sequence[::-1, ::-1]

    profiles, profile_bias, counts_bias = [], {}, {}
    tasks = dataset["input_data"]
    for key in sorted(tasks, key=int):
        task = tasks[key]
        signal = np.stack(
            [dataset["tracks"][path][chrom][start:end]
             for path in task["signal"]["source"]], axis=1)
        if rev_comp:
            signal = reverse_complement_tracks(signal)
        profiles.append(signal)

        sources = task["bias"]["source"]
        if len(sources) == 0:
            continue
        raw = np.stack([dataset["tracks"][path][chrom][start:end]
                        for path in sources], axis=1)
        smoothed = []
        for j, params in enumerate(task["bias"]["smoothing"]):
            if params is not None:
                smoothed.append(smoothing_fn(
                    raw[:, j].astype(np.float32), params[0], params[1]))
        if rev_comp:
            raw = reverse_complement_tracks(raw)
        channels = [raw]
        if smoothed:
            smoothed = np.stack(smoothed, axis=1)
            if rev_comp:
                smoothed = reverse_complement_tracks(smoothed)
            channels.append(smoothed)
        bias = np.concatenate(channels, axis=1).astype(np.float32)
        profile_bias[int(key)] = bias
        counts_bias[int(key)] = np.log(bias.sum(axis=0) + 1)

    profile = np.concatenate(profiles, axis=1).astype(np.float32)
    return {
        "sequence": sequence,
        "profile": profile,
        "logcounts": np.log(profile.sum(axis=0) + 1),
        "profile_bias": profile_bias,
        "counts_bias": counts_bias,
    }


def check_batch(dataset, inputs, input_len, output_len, profiles, logcounts,
                smoothing_fn=gaussian_smooth, bias_is_zero=False):
    """
        Assert that every example of a batch matches the truth at the
        coordinates the batch reports

        Args:
            dataset (dict): return value of tests.synth.make_dataset
            inputs (dict): the batch's inputs dictionary
            input_len (int): input sequence length
            output_len (int): output profile length
            profiles (numpy.ndarray): the batch's profile targets
            logcounts (numpy.ndarray): the batch's log count targets
            smoothing_fn (callable): see expected_example
            bias_is_zero (bool): True if the generator was asked to
                zero the control inputs
    """

    coordinates = inputs["coordinates"]
    n = len(coordinates)
    rev_comp = unwrap(inputs["rev_comp"]) if "rev_comp" in inputs \
        else np.zeros(n, dtype=int)
    assert inputs["sequence"].shape == (n, input_len, 4)
    assert profiles.shape[:2] == (n, output_len)

    for i in range(n):
        chrom, start, end = coordinates[i]
        start, end = int(start), int(end)
        assert end - start == output_len
        expected = expected_example(dataset, chrom, start, end,
                                    bool(rev_comp[i]), input_len,
                                    smoothing_fn)
        assert np.array_equal(inputs["sequence"][i], expected["sequence"])
        assert np.array_equal(profiles[i], expected["profile"])
        np.testing.assert_allclose(logcounts[i], expected["logcounts"],
                                   rtol=1e-6)
        for task, bias in expected["profile_bias"].items():
            got = inputs["profile_bias_input_{}".format(task)][i]
            got_counts = inputs["counts_bias_input_{}".format(task)][i]
            if bias_is_zero:
                assert not got.any() and not got_counts.any()
            else:
                np.testing.assert_allclose(got, bias, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(
                    got_counts, expected["counts_bias"][task], rtol=1e-5)
