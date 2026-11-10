"""
    Characterization tests for bpnet-counts-loss-weight: the suggested
    weight of the counts loss, from the average number of reads per
    peak window
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from bpnet.utils.counts_loss_weight import get_recommended_counts_loss_weight
from bpnet.utils.exceptionhandler import NoTracebackException

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def average_reads_per_track(dataset, width):
    """
        Mean over signal tracks and peaks of the reads within `width`
        bases around the peak position
    """

    totals = []
    for path in dataset["signal"]:
        for chrom, start, end, _, summit in dataset["peak_rows"]:
            pos = start + summit
            totals.append(dataset["tracks"][path][chrom]
                          [pos - width // 2:pos + width // 2].sum())
    return float(np.mean(totals))


def peaks_frame(dataset, width):
    rows = [(c, s + m - width // 2, s + m + width // 2)
            for c, s, e, _, m in dataset["peak_rows"]]
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_weight_is_alpha_times_the_average_reads(synth_dataset, alpha):
    width = 200
    n_obs = average_reads_per_track(synth_dataset, width)
    peaks = peaks_frame(synth_dataset, width)

    # joint loss over both strands: alpha * average reads per track
    weight = get_recommended_counts_loss_weight(
        synth_dataset["signal"], peaks, alpha=alpha)
    np.testing.assert_allclose(weight, alpha * n_obs, rtol=1e-9)

    # original loss, one multinomial per strand: half of that
    weight = get_recommended_counts_loss_weight(
        synth_dataset["signal"], peaks, alpha=alpha, orig_multi_loss=True)
    np.testing.assert_allclose(weight, 0.5 * alpha * n_obs, rtol=1e-9)


def test_windows_outside_the_chromosome_are_skipped(synth_dataset):
    width = 200
    peaks = peaks_frame(synth_dataset, width)
    outside = pd.DataFrame(
        [("chr1", synth_dataset["chrom_len"] - 50,
          synth_dataset["chrom_len"] + 150)],
        columns=["chrom", "start", "end"])
    with_outside = get_recommended_counts_loss_weight(
        synth_dataset["signal"], pd.concat([peaks, outside]))
    assert with_outside == get_recommended_counts_loss_weight(
        synth_dataset["signal"], peaks)


def test_missing_bigwig_is_an_error(synth_dataset):
    with pytest.raises(NoTracebackException):
        get_recommended_counts_loss_weight(
            ["/no/such/file.bw"], peaks_frame(synth_dataset, 200))


def run_cli(args):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "bpnet.cli.counts_loss_weight"] + args,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_command_line_prints_the_weight(synth_dataset):
    code, out, err = run_cli(["--input-data", synth_dataset["input_json"],
                              "--peak-width", "200", "--alpha", "0.5"])
    assert code == 0, err
    np.testing.assert_allclose(
        float(out.strip().splitlines()[-1]),
        0.5 * average_reads_per_track(synth_dataset, 200), rtol=1e-9)


def test_command_line_prints_the_default_when_the_input_is_missing(tmp_path):
    code, out, err = run_cli(["--input-data", str(tmp_path / "missing.json"),
                              "--default", "42.5"])
    # the default goes to standard output, the message to standard error
    assert out.strip() == "42.5"
    assert "does not exist" in err
    assert code != 0
