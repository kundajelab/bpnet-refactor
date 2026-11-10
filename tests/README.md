# Tests

Run from the repository root (the package does not need to be installed):

```
python -m pytest                 # everything
python -m pytest -m "not slow"   # skip the tests that train a small model
```

Tests run on the CPU. Set `BPNET_TEST_GPU=1` to let TensorFlow use a GPU.

## Layout

| file | what it covers |
|---|---|
| `synth.py` | builder for a small synthetic dataset (genome, bigWigs, peaks, background, input JSON files); returns the arrays it wrote so tests can compare against the truth |
| `conftest.py` | shared fixtures (`synth_dataset`: the default synthetic dataset, built once per session) |
| `test_synth.py` | the builder itself |
| `test_sequtils.py` | peak loading, one-hot encoding, reverse complement |
| `test_generators.py` | batch generation in train / val / test mode on the mini genome in `data/` |
| `data/` | small hand-made input files |
