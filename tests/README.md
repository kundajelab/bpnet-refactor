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
| `test_bigwig_writing.py` | both bigWig writers: region order, the split of overlapping regions between their centres, the statistics file; the files against those of the writers that stored an entry per position (`legacy_reference.py`), read from the bytes of the file: the same float32 value at every position (special values and rows of other types included), the same stored intervals, total summary, zoom records and statistics file, in both layouts (runs of positions with pyBigWig 0.3.29 or later, where the section ends are exact and the file is smaller; the identical file with older versions); regions that cannot be written |
| `test_counts_loss_weight.py` | `bpnet-counts-loss-weight` |
| `saved_weights.py` | reads a SavedModel's weights by layer name from its checkpoint |
| `test_loading_formats.py` | the file reading of `bpnet.model.loading` without TensorFlow: protocol buffer fields, where keras keeps the layer graph in a SavedModel, what is not a saved model |
| `test_loading.py` | `bpnet.model.loading`: models of thirteen architecture variants saved and loaded again (same predictions, bit for bit), on TensorFlow 2.4 against keras' own loader in a fresh process, models saved by TensorFlow 2.4.1, 2.11.0 and 2.15.1 (`data/saved_models/`) loaded in the running version, layer graphs and weights that cannot be loaded |
| `test_cli_end_to_end.py` | (slow) `bpnet-train` then `bpnet-predict` as subprocesses on the synthetic dataset, on every TensorFlow version; predictions, metric files and bigWigs recomputed from the saved weights with the numpy reference; the weights a run starts from and the loci `--seed` selects, from runs with a learning rate of zero |
| `test_cli_failures.py` | (slow) failures inside `bpnet-train` reach the exit status: a failing training process, a failing batch worker; a training step, a validation step or a prediction that fails while the batch workers have work left ends the tool at once with that error and leaves no process behind; an interrupt of `bpnet-train` alone, or of its process group, ends the training process and its workers; what `bpnet-train` does with a training process that does not end (interrupt, terminate, kill), on stand-in processes |
| `test_generator_failures.py` | a worker process of the batch generator that raises, exits or is killed ends the epoch with an error naming the cause (it used to leave the consumer waiting forever); an epoch that ends early (a worker failed, or the consumer closed the generator, raised, or took only some batches) stops its other workers at once and leaves no process, thread or open queue behind, and the generator can be used again; an epoch that is stopped from outside (`stop()`, `stop_all()`) ends without an error whether its consumer waits for a batch, is elsewhere, the workers are just being started or the last batch was taken and the workers are waited for, also with workers that ignore the request (one wait for all of them), in one generator or all, with forked, spawned and fork-server workers, and not from a forked process; workers end when the process that takes their batches is gone, also with batches left to hand over; the cleanup of a failing tool is not ended by a second interrupt; each case runs in a child process that must also end by itself, under a time limit |
| `test_generator_queues.py` | the queues between the worker processes and the consumer hold a few batches: with a slow consumer the batches that were made and not yet taken stay within what the queues hold (for 1, 2 and 4 workers) and the epoch still delivers every locus once, the loci of each worker in its order; when nobody takes batches the workers make exactly as many as the queues hold; a worker's failure gets through full queues; closing the generator and stopping the epoch end workers and threads that wait for room, and a stop does not wait for the polling interval of the workers; workers that wait for room end when their consumer is gone, with large and with tiny batches and with forked, fork-server and spawned workers (they look for the process of the consumer by its id and start time; a process forked from the consumer is not taken for it; without the start time the id alone is used; workers of a parent end by their parent alone); the start time of a process, also one with spaces, parentheses or a line end in its name; a consumer whose entry in /proc cannot be read or is hidden counts as there while a process with its id exists; one worker delivers in the order of the epoch |
| `test_interrupts.py` | interrupts (SIGINT) that Python would lose (`Interrupts`, `interruptible`): an interrupt is raised as always; one whose KeyboardInterrupt was discarded is raised at the next check; a loop ends without fetching the next item, before the first one when the interrupt was lost before the loop, and before an item is handed out when the interrupt was lost while it was fetched; an interrupt that came out as another error (also an exit, or the failure of a batch worker) is reported as an interrupt with that error as its cause; a loop ends when the interrupt was lost while its end was fetched; nested notes both take the interrupt; a handler installed meanwhile, or one that Python did not install, is left alone; an interrupt while the handler is installed, or when the entry returns, undoes the entry, and so does a ValueError on the way in (from the handler that was there, before or after ours was installed, also one that installed another handler, which stays); a handler installed meanwhile is told apart by identity; notes are taken also where the module threading takes another thread for the main one; ignored interrupts stay ignored; a handler that was there is called; nothing is changed outside the main thread, and the handling is as before afterwards (the command line cases are in `test_cli_failures.py`: `bpnet-predict` ends after an interrupt that is discarded inside a call of the model or while the model is loaded, fails after one discarded when its files are written, and reports one that came out as another error as an interrupt; `bpnet-train` ends promptly when its training process discards the interrupt in a training or validation step; the training process ignores interrupts once it is on its way out, reports such an error as an interrupt, and does not end as if it had succeeded after a lost one) |
| `test_sequence_dtype.py` | the type of the one hot sequences of a batch: float64 unless the generator is asked for another; with int8 (what `bpnet-train` and `bpnet-predict` ask for) only the type of the sequences differs, every other array and the jitters drawn are the same, also for batches that come from worker processes; types that cannot hold the encoding are refused. That the model computes the same from int8 as from float64 is in `test_arch.py` (outputs), `test_custommodel.py` (training steps: losses, weights, optimizer state) and `test_cli_end_to_end.py` (`bpnet-train` with either type gives the same saved model) |
| `legacy_reference.py` | earlier implementations of functions that were replaced by faster ones (one-hot encoding, bigWig interval reading, the construction of a batch, the two bigWig writers), kept verbatim to prove equality |
| `test_bigwig_reading.py` | how the batch generator reads bigWigs (`bpnet.generators.windows`): an interval and all windows of a batch against the previous way of reading, byte for byte, for every kind of value a file can hold (gaps, infinities, a stored NaN, a negative zero), overlapping and repeated windows, the same failures, a file without data and one whose first chromosome is empty; whole batches, array by array, against the previous batch construction for seven kinds of dataset (one with tasks of one and of two tracks) in five settings; files opened once per process and left out when a generator is pickled |
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
`python benchmarks/bench_one_hot.py`. `bench_batch_generation.py` times the construction of a
batch in one process; `bench_batch_delivery.py` measures how many batches per second reach the
consumer of the generator from its worker processes; `bench_queue_memory.py` measures the memory
of the generator when its consumer is slower than its workers.
