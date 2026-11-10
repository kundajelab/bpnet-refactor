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


@pytest.fixture(autouse=True)
def restore_bpnet_defaults():
    """
        Put the default architecture parameters back after every test.
        Building a model with non default parameters currently changes
        the module level defaults (see
        test_arch.py::test_building_a_model_leaves_the_defaults_alone),
        which would make test results depend on the test order.
    """

    import copy
    from bpnet.model import bpnetdefaults

    names = [n for n in dir(bpnetdefaults) if n.isupper()]
    saved = {n: copy.deepcopy(getattr(bpnetdefaults, n)) for n in names}
    yield
    for name, value in saved.items():
        current = getattr(bpnetdefaults, name)
        if isinstance(current, dict):
            current.clear()
            current.update(value)
        elif isinstance(current, list):
            current[:] = value
        else:
            setattr(bpnetdefaults, name, value)
