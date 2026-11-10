"""
    Writes the saved-model fixture of the TensorFlow version it is run
    with: tf<version>.tar.gz next to this file

        PYTHONPATH=. python tests/data/saved_models/make_fixtures.py

    (from the repository root). The archive holds two small models:
    one with the structure of the released models after two training
    steps, saved the way the trainer saves it (a SavedModel directory
    with the optimizer's state) and as an HDF5 file; and one with every
    other kind of layer the package can build (hidden layers in the
    counts head), saved as a SavedModel directory. Next to each is
    expected.npz with a batch of inputs, the predictions of the model
    that was saved and checksums of its weights.
    tests/test_loading.py loads every archive in every TensorFlow
    version.

    The archives in the repository were written with the versions in
    their names (Python 3.7 for 2.4.1, 3.10 for 2.11.0, 3.11 for
    2.15.1). An archive only needs to be rewritten when a new
    TensorFlow version is added.
"""

import json
import os
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import tensorflow as tf

from tests.test_arch import forward_case, predict
from tests.test_loading import FIXTURE_MODELS, trained_model, weight_digests


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    name = "tf{}".format(tf.__version__)
    work = tempfile.mkdtemp()
    root = os.path.join(work, name)
    for directory, (case, trained, hdf5) in sorted(FIXTURE_MODELS.items()):
        out = os.path.join(root, directory)
        os.makedirs(out)
        made = trained_model(case) if trained else forward_case(case)
        model = made["model"]
        model.save(os.path.join(out, "model"))
        if hdf5:
            model.save(os.path.join(out, "model.h5"))
        inputs = {key: value[:2] for key, value in made["inputs"].items()}
        profile, counts = predict(model, inputs)
        arrays = {"input/" + key: value for key, value in inputs.items()}
        np.savez_compressed(
            os.path.join(out, "expected.npz"), profile=profile,
            counts=counts, weight_digests=json.dumps(weight_digests(model)),
            **arrays)

    archive = os.path.join(here, name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname=name)
    shutil.rmtree(work)
    print("wrote", archive, os.path.getsize(archive), "bytes; python",
          sys.version.split()[0])


if __name__ == "__main__":
    main()
