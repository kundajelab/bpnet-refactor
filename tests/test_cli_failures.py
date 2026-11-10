"""
    The command line tools must report failure through their exit
    status: pipelines (WDL, shell scripts with `set -e`) rely on it
"""

import json
import os

import pytest

pytest.importorskip("tensorflow")

from tests.test_cli_end_to_end import run_cli

pytestmark = pytest.mark.slow


def train_args(dataset, model_dir, **overrides):
    values = {
        "--input-data": dataset["input_json"],
        "--output-dir": model_dir,
        "--reference-genome": dataset["fasta"],
        "--chrom-sizes": dataset["chrom_sizes"],
        "--splits": dataset["splits_json"],
        "--model-arch-name": "BPNet",
        "--model-arch-params-json": dataset["params_json"],
        "--sequence-generator-name": "BPNet",
        "--model-output-filename": "model",
        "--input-seq-len": dataset["input_len"],
        "--output-len": dataset["output_len"],
        "--threads": 2, "--epochs": 1, "--batch-size": 16,
    }
    values.update(overrides)
    args = ["--chroms"] + dataset["chroms"]
    for key, value in values.items():
        args.extend([key, value])
    return args


def test_training_process_failure_gives_an_error_status(synth_dataset,
                                                        tmp_path):
    """
        The model parameters are read inside the training process; when
        that process fails, bpnet-train must fail too
    """

    broken = tmp_path / "broken_params.json"
    broken.write_text("{ this is not json")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir,
        **{"--model-arch-params-json": broken}), cwd=str(tmp_path))
    assert code != 0, output[-2000:]
    assert "Unable to load json file" in output
    assert "Training of split 0 failed" in output
    # no model was written
    assert not any(name.startswith("model") for name in os.listdir(
        str(model_dir)))


def test_training_stops_at_the_first_failed_split(synth_dataset, tmp_path):
    # two splits; the broken model parameter file makes the first one
    # fail, the second must not be started
    with open(synth_dataset["splits_json"]) as fh:
        split = json.load(fh)["0"]
    splits = tmp_path / "two_splits.json"
    splits.write_text(json.dumps({"0": split, "1": split}))
    broken = tmp_path / "broken_params.json"
    broken.write_text("{ this is not json")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir,
        **{"--model-arch-params-json": broken, "--splits": splits}),
        cwd=str(tmp_path))
    assert code != 0
    assert "Training of split 0 failed" in output
    # the second split was not started
    assert "Split #0" in output
    assert "Split #1" not in output
    # the message points at the directory that holds the logs
    assert "trainer logs in {}".format(model_dir) in output


def test_missing_splits_file_message(synth_dataset, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing = tmp_path / "no_such_splits.json"
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir, **{"--splits": missing}),
        cwd=str(tmp_path))
    assert code != 0
    assert "File not found: {}".format(missing) in output


def test_failure_in_a_batch_worker_ends_the_training(tmp_path):
    """
        A signal file that does not cover the genome: the worker
        processes cannot read some of the loci. bpnet-train must stop
        with an error status and say why (it used to wait forever for
        the batches of the worker that failed).
    """

    import time

    from tests import synth

    dataset = synth.make_dataset(
        str(tmp_path / "data"), chrom_len=20000, n_peaks=60, n_background=60)
    path = dataset["signal"][0]
    short = {chrom: 6000 for chrom in dataset["chroms"]}
    synth.write_bigwig(path, short, {
        chrom: values[:6000]
        for chrom, values in dataset["tracks"][path].items()})

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    start = time.time()
    code, output = run_cli("bpnet.cli.bpnettrainer",
                           train_args(dataset, model_dir), cwd=str(tmp_path))
    assert code != 0, output[-2000:]
    assert "Batch generation failed in worker process" in output
    assert "Training of split 0 failed" in output
    # far below the time after which run_cli gives up on a hung tool
    assert time.time() - start < 300
    assert not any(name.startswith("model") for name in os.listdir(
        str(model_dir)))
