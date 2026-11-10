"""
    Shared pytest fixtures and settings for the bpnet test suite
"""

import os

# Tests run on the CPU unless BPNET_TEST_GPU is set: results must not
# depend on which GPU happens to be free, and a shared machine's GPUs
# should not be claimed by a test run. This has to happen before
# tensorflow is imported anywhere.
if not os.environ.get("BPNET_TEST_GPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pytest

from tests import synth


@pytest.fixture(scope="session")
def synth_dataset(tmp_path_factory):
    """
        The default synthetic dataset (single task, two strands, with
        controls), built once per test session. Tests must treat the
        files as read-only.
    """

    return synth.make_dataset(str(tmp_path_factory.mktemp("synth")), seed=0)

