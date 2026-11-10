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
| `checks.py` | reference computations that share no code with the package: what a batch must contain, from the builder's arrays |
| `test_sequence_ops.py` | one-hot encoding, reverse complement of sequences and profiles, rounding |
| `test_peak_positions.py` | `getPeakPositions`: windows, chromosome-end filtering, index and chromosome selection, background sampling, weights |
| `test_generators_synth.py` | `MBPNetSequenceGenerator` on the synthetic dataset: epoch composition (trimming, padding, background resampling), batch contents against the truth, jitter, reverse-complement copies, smoothed controls, unstranded / no-control / two-task layouts |
| `test_sequtils.py` | peak loading, one-hot encoding, reverse complement |
| `test_generators.py` | batch generation in train / val / test mode on the mini genome in `data/` |
| `data/` | small hand-made input files |

## Known defects

Tests marked `xfail(strict=True)` describe behaviour we want but do not have yet. They fail
today, and the PR that fixes a defect removes the marker (a strict xfail that starts passing
fails the suite, so it cannot be forgotten).
