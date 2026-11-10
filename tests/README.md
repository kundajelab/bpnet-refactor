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
| `reference_model.py` | the forward pass and the training loss in plain numpy (float64), written from the description of the architecture; the yardstick for every backend |
| `model_helpers.py` | small and released architecture parameters, random weights, random batches |
| `test_arch.py` | the graph `arch.BPNet` builds (layer names, weight shapes, cropping, dilation) and its outputs against the numpy reference for ten configurations |
| `test_losses.py` | multinomial NLL against scipy, its gradient and normalization constant, MSE and Poisson count losses |
| `test_custommodel.py` | the training objective against the numpy reference (masking by weight, joint two-strand multinomial, log(total + 2) count target), gradients against finite differences, the Adam update rule, train / test step bookkeeping |
| `test_training_helpers.py` | early stopping and learning-rate-on-plateau rules |
| `test_initialization.py` | the seeded initial weights without TensorFlow: the numpy generator against values recorded from TensorFlow 2.4.1, the weights a training run started from under that version, operation seeds, repeated seeds, when weights are left as drawn and when they are assigned; the `--seed` option's values |
| `test_initialization_tf.py` | the same against TensorFlow: the generator against TensorFlow's random kernel, the seed handling against TensorFlow's, on TensorFlow 2.4 the kernels against what Keras draws itself for seven architectures and five seeds; what the trainer does before and after building its model |
| `test_optimizer.py` | the Adam update on prescribed gradients (40 steps, magnitudes around epsilon, a learning-rate change) |
| `test_metrics.py` | the per-region metrics of `bpnet-predict` (correlations, normalized likelihood, cross entropy and Jensen-Shannon scores, MSE) against formulas written out in the test |
| `test_bigwig_writing.py` | both bigWig writers: region order, the split of overlapping regions between their centres, the statistics file |
| `test_counts_loss_weight.py` | `bpnet-counts-loss-weight` |
| `saved_weights.py` | reads a SavedModel's weights by layer name from its checkpoint |
| `test_loading_formats.py` | the file reading of `bpnet.model.loading` without TensorFlow: protocol buffer fields, where keras keeps the layer graph in a SavedModel, what is not a saved model |
| `test_loading.py` | `bpnet.model.loading`: models of thirteen architecture variants saved and loaded again (same predictions, bit for bit), on TensorFlow 2.4 against keras' own loader in a fresh process, models saved by TensorFlow 2.4.1, 2.11.0 and 2.15.1 (`data/saved_models/`) loaded in the running version, layer graphs and weights that cannot be loaded |
| `test_cli_end_to_end.py` | (slow) `bpnet-train` then `bpnet-predict` as subprocesses on the synthetic dataset, on every TensorFlow version; predictions, metric files and bigWigs recomputed from the saved weights with the numpy reference; the weights a run starts from and the loci `--seed` selects, from runs with a learning rate of zero |
| `test_cli_failures.py` | (slow) failures inside `bpnet-train` reach the exit status |
| `legacy_reference.py` | earlier implementations of functions that were replaced by faster ones, kept verbatim to prove equality |
| `test_bigwig_reading.py` | how the batch generator reads a bigWig interval: same values as the previous way for every kind of value a file can hold (gaps, infinities, a stored NaN), same failures, and whole batches array by array |
| `data/` | small hand-made input files; `data/saved_models/` holds small models saved by each supported TensorFlow version and the script that wrote them |

## Known defects

Tests marked `xfail(strict=True)` describe behaviour we want but do not have yet. They fail
today, and the PR that fixes a defect removes the marker (a strict xfail that starts passing
fails the suite, so it cannot be forgotten).

## Numerical tolerances

Model outputs and losses are compared with the float64 numpy reference at a relative tolerance
of 1e-4 (2e-4 for losses); weight updates at an absolute tolerance of a few 1e-6 for updates of
about 1e-3. These bounds hold for float32 arithmetic on the CPU. On Ampere or newer NVIDIA GPUs
TensorFlow computes convolutions in TF32 by default, which is less precise: run GPU parity checks
with `NVIDIA_TF32_OVERRIDE=0`.

## Benchmarks

`benchmarks/` holds timing scripts for code that was made faster, each comparing the new
implementation with the one in `tests/legacy_reference.py` in alternating order
(`benchmarks/timing.py`). Run from the repository root, for example
`python benchmarks/bench_one_hot.py`.
