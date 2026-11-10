"""
    The type of the one hot encoded sequences of a batch: float64 as
    always unless the batch generator is asked for another one. The
    command line tools ask for int8 (the model turns either into the
    same float32 values; see test_arch.py and test_custommodel.py).
"""

import random

import numpy as np
import pytest

from bpnet.generators import generators
from bpnet.utils.exceptionhandler import NoTracebackException

from tests.test_bigwig_reading import batch_rows, flat, same_bits


def make_generator(dataset, mode="train", num_threads=1, **settings):
    params = {"sequence_generator_name": "BPNet",
              "input_seq_len": dataset["input_len"],
              "output_len": dataset["output_len"],
              "rev_comp_aug": mode == "train",
              "max_jitter": 16 if mode == "train" else 0, "shuffle": False,
              "mode": mode}
    params.update(settings)
    state = np.random.get_state()
    np.random.seed(0)
    try:
        return generators.MBPNetSequenceGenerator(
            dataset["input_json"], params, dataset["fasta"],
            dataset["chrom_sizes"], dataset["chroms"],
            num_threads=num_threads, batch_size=16, epochs=1)
    finally:
        np.random.set_state(state)


def test_sequences_are_float64_unless_asked_otherwise(synth_dataset):
    generator = make_generator(synth_dataset)
    inputs, _, _ = generator._generate_batch(batch_rows(generator, 0, 16))
    assert inputs["sequence"].dtype == np.float64
    assert set(np.unique(inputs["sequence"])) == {0.0, 1.0}


@pytest.mark.parametrize("dtype", ["int8", "uint8", "float32", "bool",
                                   "float64", "float16", ">f4", "uint64"])
@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_only_the_type_of_the_sequences_differs(synth_dataset, mode, dtype):
    """
        The sequences hold the same zeros and ones in the type that
        was asked for. Every other array of the batch is the same,
        byte for byte, and so are the jitters that were drawn.
    """

    usual = make_generator(synth_dataset, mode)
    asked = make_generator(synth_dataset, mode, sequence_dtype=dtype)
    compared = 0
    for number, start in enumerate((0, 16, 32)):
        random.seed(number)
        expected = flat(usual._generate_batch(batch_rows(usual, start, 16)))
        after_usual = random.getstate()
        random.seed(number)
        found = flat(asked._generate_batch(batch_rows(asked, start, 16)))
        assert random.getstate() == after_usual
        assert [name for name, _ in found] == [name for name, _ in expected]
        for (name, a), (_, b) in zip(found, expected):
            if name == "sequence":
                assert b.dtype == np.float64 and a.dtype == np.dtype(dtype)
                assert a.shape == b.shape
                assert np.array_equal(a, b)
                assert a.nbytes * b.itemsize == b.nbytes * a.itemsize
                compared += 1
            else:
                assert same_bits(a, b), name
    assert compared == 3


def test_a_batch_with_int8_sequences_is_a_third_of_the_size(synth_dataset):
    def megabytes(generator):
        inputs, outputs, weights = generator._generate_batch(
            batch_rows(generator, 0, 16))
        arrays = list(inputs.values()) + list(outputs.values()) + [weights]
        return sum(a.nbytes for a in arrays), inputs["sequence"].nbytes

    total, sequences = megabytes(make_generator(synth_dataset))
    small, small_sequences = megabytes(
        make_generator(synth_dataset, sequence_dtype="int8"))
    assert sequences == 8 * small_sequences
    assert total - sequences == small - small_sequences
    # the sequences were most of the batch
    assert sequences > 0.7 * total
    assert small < 0.4 * total


@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_batches_from_the_worker_processes(synth_dataset, mode):
    """
        Through gen(): the batches that reach the consumer from a
        worker process have sequences of the type that was asked for,
        and are otherwise the batches of the default type
    """

    def batches(dtype):
        settings = dict(max_jitter=0)
        if dtype is not None:
            settings["sequence_dtype"] = dtype
        generator = make_generator(synth_dataset, mode, **settings)
        return [flat(batch) for batch in generator.gen()], generator.len()

    usual, n_batches = batches(None)
    small, _ = batches("int8")
    assert len(usual) == len(small) == n_batches > 2
    for found, expected in zip(small, usual):
        assert [name for name, _ in found] == [name for name, _ in expected]
        for (name, a), (_, b) in zip(found, expected):
            if name == "sequence":
                assert a.dtype == np.int8 and b.dtype == np.float64
                assert np.array_equal(a, b)
            else:
                assert same_bits(a, b), name


@pytest.mark.parametrize("dtype", [
    "S1", "complex64", "object", "U4", "timedelta64[ns]", "datetime64[s]",
    # not the name of a type
    "one hot", 7.5,
    # specifications of structured types that numpy refuses
    {"names": ["x"], "formats": []}, [("a", "i1"), ("a", "i1")],
    # a structured type
    [("a", "i1")]])
def test_types_other_than_boolean_integer_and_floating_point(synth_dataset,
                                                             dtype):
    with pytest.raises(NoTracebackException, match="sequence_dtype"):
        make_generator(synth_dataset, sequence_dtype=dtype)
