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
    assert "Training of split 1 failed" not in output
    assert not os.path.exists(str(model_dir / "trainer_split001.log"))


def test_missing_splits_file_message(synth_dataset, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing = tmp_path / "no_such_splits.json"
    code, output = run_cli("bpnet.cli.bpnettrainer", train_args(
        synth_dataset, model_dir, **{"--splits": missing}),
        cwd=str(tmp_path))
    assert code != 0
    assert "File not found: {}".format(missing) in output
