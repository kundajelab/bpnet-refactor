"""
    Characterization tests for MBPNetSequenceGenerator on the synthetic
    dataset: what goes into an epoch and what every batch contains,
    checked against the arrays the dataset builder wrote.

    These tests describe the behaviour of the generator as it is, so
    that a later change of the data path can be shown to keep it.
"""

import json
import multiprocessing as mp
import os

import numpy as np
import pytest

from bpnet.generators import generators
from bpnet.utils.exceptionhandler import NoTracebackException
from bpnet.utils.misc import gaussian1D_smoothing

from tests import synth
from tests.checks import check_batch, unwrap


def make_generator(dataset, mode, chroms=("chr1",), rev_comp_aug=None,
                   max_jitter=None, shuffle=True, input_json=None, **kwargs):
    """
        A generator on the synthetic dataset with the settings the
        trainer uses for each mode (jitter and reverse complement
        augmentation only when training)
    """

    params = {
        "sequence_generator_name": "BPNet",
        "input_seq_len": dataset["input_len"],
        "output_len": dataset["output_len"],
        "rev_comp_aug": (mode == "train") if rev_comp_aug is None
        else rev_comp_aug,
        "max_jitter": (16 if mode == "train" else 0) if max_jitter is None
        else max_jitter,
        "shuffle": shuffle,
        "mode": mode,
    }
    kwargs.setdefault("num_threads", 2)
    kwargs.setdefault("batch_size", 16)
    kwargs.setdefault("epochs", 3)
    return generators.MBPNetSequenceGenerator(
        input_json or dataset["input_json"], params, dataset["fasta"],
        dataset["chrom_sizes"], list(chroms) if chroms else None, **kwargs)


def loci_on(dataset, key, chroms):
    return [(c, s + m) for c, s, e, _, m in dataset[key] if c in chroms]


def targets(batch, mode):
    """
        (inputs, profiles, logcounts, weights) of a batch in any mode
    """

    if mode == "test":
        return batch, batch["true_profiles"], batch["true_logcounts"], None
    inputs, outputs, weights = batch
    return (inputs, outputs["profile_predictions"],
            outputs["logcounts_predictions"], weights)


def positions(inputs, output_len, jitter=None):
    """
        (chrom, peak position) of every example of a batch, from the
        coordinates it reports and the jitter that was applied
    """

    out = []
    for i, (chrom, start, end) in enumerate(inputs["coordinates"]):
        shift = 0 if jitter is None else int(jitter[i])
        out.append((str(chrom), int(start) + output_len // 2 - shift))
    return out


# --------------------------------------------------------------------
# epoch composition
# --------------------------------------------------------------------

def test_epoch_sizes_train_val_test(synth_dataset):
    n_peaks = len(loci_on(synth_dataset, "peak_rows", ["chr1"]))
    n_background = int(n_peaks * 0.5)

    train = make_generator(synth_dataset, "train")
    val = make_generator(synth_dataset, "val")
    test = make_generator(synth_dataset, "test")

    # train and val: peaks plus ratio * peaks background loci, trimmed
    # down to a multiple of num_threads * batch_size
    for gen in (train, val):
        assert gen.get_samples_len() == n_peaks + n_background
        assert gen.get_resized_samples_len() == \
            (n_peaks + n_background) // 32 * 32
        assert gen.len() == gen.get_resized_samples_len() // 16

    # test: peaks only, padded up to a multiple of num_threads * batch_size
    assert test.get_samples_len() == n_peaks
    assert test.get_resized_samples_len() == -(-n_peaks // 32) * 32
    assert test.len() == test.get_resized_samples_len() // 16


def test_one_dataframe_of_samples_per_epoch(synth_dataset):
    gen = make_generator(synth_dataset, "train", epochs=4)
    assert len(gen.get_samples()) == 4
    assert len(gen.get_resized_samples()) == 4
    assert list(gen.get_samples()[0].columns) == \
        ["chrom", "start_coord", "end_coord", "pos", "weight"]


def test_train_background_is_resampled_every_epoch(synth_dataset):
    gen = make_generator(synth_dataset, "train", epochs=4)
    peaks = set(loci_on(synth_dataset, "peak_rows", ["chr1"]))
    pool = set(loci_on(synth_dataset, "background_rows", ["chr1"]))
    background_sets = []
    for samples in gen.get_samples():
        fg = samples[samples["weight"] == 1]
        bg = samples[samples["weight"] == 0]
        # every epoch has all the peaks ...
        assert set(zip(fg["chrom"], fg["pos"])) == peaks
        # ... and its own draw (without replacement) from the pool
        drawn = list(zip(bg["chrom"], bg["pos"]))
        assert len(drawn) == len(set(drawn)) == int(len(peaks) * 0.5)
        assert set(drawn) <= pool
        background_sets.append(frozenset(drawn))
    assert len(set(background_sets)) > 1


def test_val_samples_are_the_same_every_epoch(synth_dataset):
    gen = make_generator(synth_dataset, "val", epochs=4)
    first = gen.get_resized_samples()[0]
    for samples in gen.get_resized_samples()[1:]:
        assert samples.equals(first)
    # a second generator makes the same choice (fixed random_state)
    other = make_generator(synth_dataset, "val", epochs=1)
    assert other.get_resized_samples()[0].equals(first)


def test_test_mode_uses_no_background(synth_dataset):
    gen = make_generator(synth_dataset, "test")
    samples = gen.get_samples()[0]
    peaks = set(loci_on(synth_dataset, "peak_rows", ["chr1"]))
    assert set(zip(samples["chrom"], samples["pos"])) == peaks


@pytest.mark.parametrize("num_threads,batch_size", [
    (1, 16), (2, 16), (3, 8), (1, 64), (2, 64), (4, 64)])
def test_trim_rule(synth_dataset, num_threads, batch_size):
    """
        Train / val data are trimmed to the largest multiple of
        num_threads * batch_size. When not even one multiple fits, the
        batch size is halved (and at batch size 1 the thread count
        reduced) until one does.
    """

    gen = make_generator(synth_dataset, "val", num_threads=num_threads,
                         batch_size=batch_size)
    n = gen.get_samples_len()

    expected_batch, expected_threads = batch_size, num_threads
    while n // (expected_threads * expected_batch) == 0:
        if expected_batch > 1:
            expected_batch //= 2
        else:
            expected_threads -= 1
    expected = n // (expected_threads * expected_batch) * \
        (expected_threads * expected_batch)

    assert gen._batch_size == expected_batch
    assert gen._num_threads == expected_threads
    assert gen.get_resized_samples_len() == expected
    assert gen.len() == expected // expected_batch
    # this dataset has 114 samples on chr1: 2 x 64 and 4 x 64 do not fit
    if num_threads * batch_size > n:
        assert expected_batch < batch_size


def test_no_samples_raises(synth_dataset):
    with pytest.raises(NoTracebackException):
        make_generator(synth_dataset, "train", chroms=["chrMissing"])


# --------------------------------------------------------------------
# batch contents
# --------------------------------------------------------------------

def test_train_batches_match_truth(synth_dataset):
    max_jitter = 16
    gen = make_generator(synth_dataset, "train", max_jitter=max_jitter)
    samples = gen.get_samples()[0]
    resized = gen.get_resized_samples()[0]
    batches = list(gen.gen())
    assert len(batches) == gen.len()

    seen, pair_jitters_differ = [], False
    for batch in batches:
        inputs, profiles, logcounts, weights = targets(batch, "train")
        n = 2 * gen._batch_size
        assert len(weights) == n

        # dtypes of what is fed to the model
        assert inputs["sequence"].dtype == np.float64
        assert profiles.dtype == np.float32
        assert logcounts.dtype == np.float32
        assert weights.dtype == np.float64

        check_batch(synth_dataset, inputs, synth_dataset["input_len"],
                    synth_dataset["output_len"], profiles, logcounts)

        index = unwrap(inputs["index"])
        jitters = unwrap(inputs["jitters"])
        status = unwrap(inputs["status"])
        rev_comp = unwrap(inputs["rev_comp"])

        # the second half of a batch is the reverse complement of the
        # loci of the first half
        half = n // 2
        assert rev_comp.tolist() == [0] * half + [1] * half
        assert np.array_equal(index[:half], index[half:])
        assert np.array_equal(weights[:half], weights[half:])

        # the jitter recorded for an example is the one that was applied
        assert positions(inputs, synth_dataset["output_len"], jitters) == \
            list(zip(samples.loc[index, "chrom"], samples.loc[index, "pos"]))

        # jitter: within range for loci with a non zero weight, none
        # for the background (weight 0)
        assert np.abs(jitters).max() <= max_jitter
        assert not jitters[weights == 0].any()
        assert np.array_equal(weights, samples.loc[index, "weight"].values)
        assert np.array_equal(status, np.where(weights == 0, -1, 1))

        # the forward and the reverse complement copy of a locus are
        # jittered independently
        if (jitters[:half] != jitters[half:]).any():
            pair_jitters_differ = True
        seen.extend(index[:half].tolist())

    assert pair_jitters_differ
    # every locus of the epoch exactly once
    assert sorted(seen) == sorted(resized.index.tolist())


def test_jitter_covers_the_range(synth_dataset):
    max_jitter = 3
    gen = make_generator(synth_dataset, "train", max_jitter=max_jitter,
                         epochs=2)
    values = set()
    for _ in range(2):
        for inputs, _, weights in gen.gen():
            jitters = unwrap(inputs["jitters"])
            values.update(jitters[weights != 0].tolist())
    # both ends of the range are included
    assert values == set(range(-max_jitter, max_jitter + 1))


def test_reverse_complement_copy_without_jitter(synth_dataset):
    gen = make_generator(synth_dataset, "train", max_jitter=0, epochs=1)
    for inputs, outputs, weights in gen.gen():
        half = len(weights) // 2
        assert np.array_equal(inputs["sequence"][:half],
                              inputs["sequence"][half:, ::-1, ::-1])
        for key in ("profile_predictions", "logcounts_predictions"):
            fwd, rev = outputs[key][:half], outputs[key][half:]
            if key == "profile_predictions":
                assert np.array_equal(fwd, rev[:, ::-1, ::-1])
            else:
                # the strands swap, so do their counts
                assert np.array_equal(fwd, rev[:, ::-1])
        assert np.array_equal(inputs["profile_bias_input_0"][:half],
                              inputs["profile_bias_input_0"][half:, ::-1, ::-1])
        assert np.array_equal(inputs["counts_bias_input_0"][:half],
                              inputs["counts_bias_input_0"][half:, ::-1])


def test_val_batches_match_truth(synth_dataset):
    gen = make_generator(synth_dataset, "val", epochs=2)
    resized = gen.get_resized_samples()[0]
    expected = sorted(zip(resized["chrom"], resized["pos"]))
    weight_of = dict(zip(zip(resized["chrom"], resized["pos"]),
                         resized["weight"]))

    for _ in range(2):
        batches = list(gen.gen())
        assert len(batches) == gen.len()
        seen = []
        for batch in batches:
            inputs, profiles, logcounts, weights = targets(batch, "val")
            assert sorted(inputs.keys()) == [
                "coordinates", "counts_bias_input_0",
                "profile_bias_input_0", "sequence"]
            assert len(weights) == gen._batch_size
            check_batch(synth_dataset, inputs, synth_dataset["input_len"],
                        synth_dataset["output_len"], profiles, logcounts)
            # no jitter: the window is centred on the peak position
            pos = positions(inputs, synth_dataset["output_len"])
            assert weights.tolist() == [weight_of[p] for p in pos]
            seen.extend(pos)
        # the same loci in every epoch, each exactly once
        assert sorted(seen) == expected


def test_test_batches_match_truth(synth_dataset):
    gen = make_generator(synth_dataset, "test", epochs=1)
    peaks = set(loci_on(synth_dataset, "peak_rows", ["chr1"]))
    seen = []
    batches = list(gen.gen())
    assert len(batches) == gen.len()
    for batch in batches:
        assert isinstance(batch, dict)
        assert sorted(batch.keys()) == [
            "coordinates", "counts_bias_input_0", "profile_bias_input_0",
            "rev_comp", "sequence", "true_logcounts", "true_profiles"]
        inputs, profiles, logcounts, _ = targets(batch, "test")
        check_batch(synth_dataset, inputs, synth_dataset["input_len"],
                    synth_dataset["output_len"], profiles, logcounts)
        assert not batch["rev_comp"].any()
        seen.extend(positions(inputs, synth_dataset["output_len"]))
    # every peak at least once; padding repeats some of them
    assert set(seen) == peaks
    assert len(seen) == gen.get_resized_samples_len() >= len(peaks)


def test_test_mode_reverse_complement_copies(synth_dataset):
    """
        With rev_comp_aug in test mode (used to average predictions over
        both strands) the second half of every batch is the reverse
        complement of the first half
    """

    gen = make_generator(synth_dataset, "test", rev_comp_aug=True, epochs=1)
    for batch in gen.gen():
        half = len(batch["rev_comp"]) // 2
        assert batch["rev_comp"].tolist() == [0] * half + [1] * half
        assert np.array_equal(batch["coordinates"][:half],
                              batch["coordinates"][half:])
        assert np.array_equal(batch["sequence"][:half],
                              batch["sequence"][half:, ::-1, ::-1])
        assert np.array_equal(batch["true_profiles"][:half],
                              batch["true_profiles"][half:, ::-1, ::-1])
        check_batch(synth_dataset, batch, synth_dataset["input_len"],
                    synth_dataset["output_len"], batch["true_profiles"],
                    batch["true_logcounts"])


def test_sample_weights_are_configurable(synth_dataset):
    gen = make_generator(synth_dataset, "train", epochs=1,
                         foreground_weight=2.0, background_weight=0.5)
    values = set()
    jittered_background = False
    for inputs, _, weights in gen.gen():
        values.update(weights.tolist())
        jitters = unwrap(inputs["jitters"])
        if jitters[weights == 0.5].any():
            jittered_background = True
    assert values == {2.0, 0.5}
    # jitter applies to every locus whose weight is not zero
    assert jittered_background


def test_order_without_shuffle_single_thread(synth_dataset):
    gen = make_generator(synth_dataset, "val", shuffle=False, num_threads=1,
                         epochs=1)
    resized = gen.get_resized_samples()[0]
    seen = []
    for inputs, _, _ in gen.gen():
        seen.extend(positions(inputs, synth_dataset["output_len"]))
    assert seen == list(zip(resized["chrom"], resized["pos"]))


def test_shuffle_changes_the_order_between_epochs(synth_dataset):
    gen = make_generator(synth_dataset, "val", shuffle=True, num_threads=1,
                         epochs=2)
    orders = []
    for _ in range(2):
        seen = []
        for inputs, _, _ in gen.gen():
            seen.extend(positions(inputs, synth_dataset["output_len"]))
        orders.append(seen)
    assert sorted(orders[0]) == sorted(orders[1])
    assert orders[0] != orders[1]


def test_epoch_counter_and_cleanup(synth_dataset):
    gen = make_generator(synth_dataset, "train", epochs=2)
    assert gen.curr_epoch == 0
    list(gen.gen())
    assert gen.curr_epoch == 1
    list(gen.gen())
    assert gen.curr_epoch == 2
    # the worker processes of an exhausted epoch have exited
    assert mp.active_children() == []


def test_set_bias_as_zero(synth_dataset):
    gen = make_generator(synth_dataset, "test", epochs=1,
                         set_bias_as_zero=True)
    for batch in gen.gen():
        check_batch(synth_dataset, batch, synth_dataset["input_len"],
                    synth_dataset["output_len"], batch["true_profiles"],
                    batch["true_logcounts"], bias_is_zero=True)


# --------------------------------------------------------------------
# other dataset layouts
# --------------------------------------------------------------------

SMALL = dict(chrom_len=20000, n_peaks=60, n_background=60)


def all_batches_match(dataset, mode, **kwargs):
    smoothing_fn = kwargs.pop("smoothing_fn", None)
    gen = make_generator(dataset, mode, chroms=None, batch_size=8,
                         epochs=1, **kwargs)
    n = 0
    for batch in gen.gen():
        inputs, profiles, logcounts, _ = targets(batch, mode)
        check_batch(dataset, inputs, dataset["input_len"],
                    dataset["output_len"], profiles, logcounts,
                    smoothing_fn=smoothing_fn)
        n += 1
    assert n == gen.len() > 0
    return inputs, profiles


def test_unstranded(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), stranded=False, **SMALL)
    for mode in ("train", "val", "test"):
        inputs, profiles = all_batches_match(dataset, mode)
        assert profiles.shape[2] == 1
        assert inputs["profile_bias_input_0"].shape[2] == 1
        assert inputs["counts_bias_input_0"].shape[1] == 1


def test_without_controls(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), with_control=False, **SMALL)
    for mode in ("train", "val", "test"):
        inputs, profiles = all_batches_match(dataset, mode)
        assert profiles.shape[2] == 2
        assert not any(key.startswith(("profile_bias", "counts_bias"))
                       for key in inputs)


def test_two_tasks(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), num_tasks=2, **SMALL)
    for mode in ("train", "val", "test"):
        inputs, profiles = all_batches_match(dataset, mode)
        # tracks of all tasks side by side; one pair of bias inputs
        # per task
        assert profiles.shape[2] == 4
        for task in (0, 1):
            assert inputs["profile_bias_input_{}".format(task)].shape[2] == 2
            assert inputs["counts_bias_input_{}".format(task)].shape[1] == 2
    # both tasks list the same loci; duplicates are dropped
    gen = make_generator(dataset, "test", chroms=None, epochs=1)
    assert gen.get_samples_len() == len(dataset["peak_rows"])


@pytest.mark.parametrize("smoothing", [
    [[3.0, 21], [3.0, 21]], [[5.0, 41], None]])
def test_smoothed_control_tracks(tmp_path, smoothing):
    """
        A smoothed copy of a control track is appended after the raw
        control tracks; it is smoothed after the window was cut out, so
        values near the window edges do not see the signal outside
    """

    dataset = synth.make_dataset(str(tmp_path), **SMALL)
    tasks = json.loads(json.dumps(dataset["input_data"]))
    tasks["0"]["bias"]["smoothing"] = smoothing
    dataset["input_data"] = tasks
    input_json = os.path.join(str(tmp_path), "input_smoothing.json")
    with open(input_json, "w") as fh:
        json.dump(tasks, fh)

    n_channels = 2 + sum(1 for s in smoothing if s is not None)
    for mode in ("train", "test"):
        inputs, _ = all_batches_match(
            dataset, mode, input_json=input_json,
            smoothing_fn=gaussian1D_smoothing)
        assert inputs["profile_bias_input_0"].shape[2] == n_channels
        assert inputs["counts_bias_input_0"].shape[1] == n_channels


# --------------------------------------------------------------------
# construction errors and accessors
# --------------------------------------------------------------------

def generator_args(dataset, **overrides):
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": dataset["input_len"],
              "output_len": dataset["output_len"], "rev_comp_aug": False,
              "max_jitter": 0, "shuffle": False, "mode": "test"}
    args = {"tasks_json": dataset["input_json"], "batch_gen_params": params,
            "reference_genome": dataset["fasta"],
            "chrom_sizes": dataset["chrom_sizes"], "chroms": ["chr1"],
            "epochs": 1}
    args.update(overrides)
    return args


@pytest.mark.parametrize("argument", ["tasks_json", "reference_genome",
                                      "chrom_sizes"])
def test_missing_input_file_raises(synth_dataset, tmp_path, argument):
    args = generator_args(
        synth_dataset, **{argument: str(tmp_path / "missing")})
    with pytest.raises(NoTracebackException):
        generators.MBPNetSequenceGenerator(**args)


def test_invalid_tasks_json_raises(synth_dataset, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    with pytest.raises(NoTracebackException):
        generators.MBPNetSequenceGenerator(
            **generator_args(synth_dataset, tasks_json=str(broken)))


def test_accessors(synth_dataset):
    gen = generators.MBPNetSequenceGenerator(**generator_args(synth_dataset))
    assert gen.get_name() == "BPNet"
    # the tasks of the input json, with integer keys
    assert gen.get_input_tasks() == \
        {int(k): v for k, v in synth_dataset["input_data"].items()}


# --------------------------------------------------------------------
# lookup of generators by name
# --------------------------------------------------------------------

def test_generator_names():
    assert generators.list_generator_names() == ["BPNet"]
    assert generators.find_generator_by_name("BPNet") == \
        "MBPNetSequenceGenerator"
    assert generators.find_generator_by_name("NoSuchGenerator") is None
    gen_class = getattr(generators,
                        generators.find_generator_by_name("BPNet"))
    assert gen_class is generators.MBPNetSequenceGenerator


# --------------------------------------------------------------------
# known defects, written as the behaviour we want; each is expected to
# fail until its fix lands, and then must pass
# --------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="'jitters', 'index' and 'status' are "
                   "stored as 1-tuples because of trailing commas")
def test_extra_train_inputs_are_arrays(synth_dataset):
    gen = make_generator(synth_dataset, "train", epochs=1)
    inputs, _, _ = next(iter(gen.gen()))
    try:
        for key in ("jitters", "index", "status", "rev_comp"):
            assert isinstance(inputs[key], np.ndarray), key
    finally:
        # let the worker processes of this epoch finish
        for child in mp.active_children():
            child.join()


@pytest.mark.xfail(strict=True, reason="more than two signal files per task "
                   "are only reported through the log")
def test_more_than_two_signal_files_is_an_error(tmp_path):
    dataset = synth.make_dataset(str(tmp_path), **SMALL)
    tasks = json.loads(json.dumps(dataset["input_data"]))
    tasks["0"]["signal"]["source"] = tasks["0"]["signal"]["source"] * 2
    input_json = os.path.join(str(tmp_path), "input_four_tracks.json")
    with open(input_json, "w") as fh:
        json.dump(tasks, fh)
    with pytest.raises(NoTracebackException):
        gen = make_generator(dataset, "test", chroms=None, epochs=1,
                             input_json=input_json)
        list(gen.gen())


@pytest.mark.xfail(strict=True, reason="loci are filtered by the input "
                   "flank only, so jitter can move a window of a locus "
                   "near a chromosome end outside the chromosome")
def test_jittered_windows_stay_inside_the_chromosome(tmp_path):
    import random
    import pandas as pd

    dataset = synth.make_dataset(str(tmp_path), **SMALL)
    flank = dataset["input_len"] // 2
    max_jitter = 16

    # one locus whose unjittered input window starts 5 bases into chr1
    edge_bed = os.path.join(str(tmp_path), "edge.bed")
    synth.write_narrowpeak(edge_bed, [("chr1", flank - 45, flank + 55, 1.0, 50)])
    tasks = json.loads(json.dumps(dataset["input_data"]))
    tasks["0"]["loci"]["source"] = [edge_bed]
    del tasks["0"]["background_loci"]
    input_json = os.path.join(str(tmp_path), "input_edge.json")
    with open(input_json, "w") as fh:
        json.dump(tasks, fh)

    gen = make_generator(dataset, "train", chroms=None, max_jitter=max_jitter,
                         rev_comp_aug=False, num_threads=1, batch_size=1,
                         epochs=1, input_json=input_json)
    samples = gen.get_samples()[0]
    random.seed(0)
    for _ in range(100):
        coords = pd.DataFrame({
            "chrom": samples["chrom"].values, "pos": samples["pos"].values,
            "weight": samples["weight"].values, "status": 1,
            "idx": samples.index.values})
        inputs, _, _ = gen._generate_batch(coords)
        for chrom, start, end in inputs["coordinates"]:
            mid = (int(start) + int(end)) // 2
            assert mid - flank >= 0
            assert mid + flank <= dataset["chrom_len"]
