"""
    The contribution scores of bpnet-shap from the PyTorch model

    bpnet-shap explains two outputs of a model with DeepLIFT against
    dinucleotide shuffles of each sequence (DeepSHAP): the sum of the
    predicted log counts, and the sum of the profile logits (minus
    their mean) weighted by their softmax, the weights not explained.
    This module computes the same hypothetical contribution scores
    with tangermeme's deep_lift_shap (Jacob Schreiber) on the PyTorch
    model of bpnet.pytorch.model: the same outputs, the same
    references (the shuffles of bpnet.utils.shaputils.dinuc_shuffle
    for the same seed, zeros for the control inputs of the
    references), several times faster on a GPU.

    Two things are done here rather than left to deep_lift_shap,
    because bpnet-shap does them:

    - the references get no control: deep_lift_shap runs the model on
      a batch of sequences followed by their references and gives
      both halves the control inputs of the sequences; the targets
      below replace those of the second half by zeros;
    - the weights of the profile logits are those of the pair: the
      explainer of bpnet-shap attributes the product of weights and
      logits to both factors and the weights are not explained, which
      leaves the mean of the weights of the sequence and of the
      reference as the multiplier of a logit. (The scores then add up
      to the mean over the references of sum_j 0.5 (w_j(x) + w_j(r))
      (l_j(x) - l_j(r)), about half of the change of the output.)

    Both rely on the layout of a batch inside deep_lift_shap
    (sequences, then references, row for row); the tests compare the
    scores with an independent DeepLIFT, which a change of that
    layout in a later tangermeme would fail.

    On a GPU the scores are computed with TF32 switched off. PyTorch
    lets cuDNN compute convolutions in TF32 (10 bits of mantissa) by
    default on the GPUs that have it; with it, 4% of the profile
    scores of a trained ENCODE model were more than 1e-5 of the
    largest score of their peak off, by up to 9e-4; without it, all
    were within 5e-8 of an independent float64 computation, at the
    same speed.
"""

import warnings

import numpy as np
import torch

from bpnet.pytorch.model import channels_first, channels_last
from bpnet.utils.shaputils import dinuc_shuffle


class _FullPrecision(object):
    """
        While inside: no TF32 on the GPU (convolutions and matrix
        products in float32 proper); afterwards as it was
    """

    def __enter__(self):
        self.previous = (torch.backends.cudnn.allow_tf32,
                         torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *failure):
        torch.backends.cudnn.allow_tf32, \
            torch.backends.cuda.matmul.allow_tf32 = self.previous


def _no_control_for_the_references(control):
    """
        The control input of a batch of sequences followed by their
        references, with zeros for the references
    """

    half = control.shape[0] // 2
    return torch.cat([control[:half], torch.zeros_like(control[half:])])


class _Target(torch.nn.Module):
    """
        A model as deep_lift_shap needs it: sequences first, the
        control inputs after them in the order of the model's inputs,
        one number per row out
    """

    def __init__(self, model):
        super(_Target, self).__init__()
        self.model = model
        if model.input_names[:1] != ["sequence"]:
            raise ValueError(
                "The first input of the model is {}; 'sequence' is "
                "expected".format(model.input_names[:1]))
        outputs = [model.shapes[name] for name in model.output_names]
        self.profile = [i for i, shape in enumerate(outputs)
                        if len(shape) == 2]
        self.counts = [i for i, shape in enumerate(outputs)
                       if len(shape) == 1]
        if len(outputs) != 2 or len(self.profile) != 1 or \
                len(self.counts) != 1:
            raise ValueError(
                "A model with a profile and a counts output is expected; "
                "the outputs {} have the shapes {}".format(
                    model.output_names, outputs))

    def outputs(self, sequence, controls):
        """
            (profile logits, log counts) of a batch of sequences
            followed by their references
        """

        if sequence.shape[0] % 2:
            raise ValueError(
                "A batch of sequences followed by as many references is "
                "expected; it has {} rows".format(sequence.shape[0]))
        out = self.model(sequence, *[_no_control_for_the_references(control)
                                     for control in controls])
        return out[self.profile[0]], out[self.counts[0]]


class CountsTarget(_Target):
    """
        The sum of the predicted log counts
    """

    def forward(self, sequence, *controls):
        _, counts = self.outputs(sequence, controls)
        return counts.sum(dim=-1, keepdim=True)


class ProfileTarget(_Target):
    """
        The profile logits of all positions of all tracks minus their
        mean, weighted by their softmax; the weights are not
        explained, and are the mean of those of a sequence and of its
        reference
    """

    def forward(self, sequence, *controls):
        profile, _ = self.outputs(sequence, controls)
        logits = profile.reshape(profile.shape[0], -1)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        weights = torch.softmax(logits, dim=-1).detach()
        half = logits.shape[0] // 2
        weights = 0.5 * (weights[:half] + weights[half:])
        return (torch.cat([weights, weights]) * logits).sum(
            dim=-1, keepdim=True)


def references(sequences, num_shuffles, seed):
    """
        The references of bpnet-shap: for every sequence its
        dinucleotide shuffles, from a random state that starts from
        the seed for every sequence

        Args:
            sequences (numpy.ndarray): one hot, (n, length, 4)
            num_shuffles (int): shuffles per sequence
            seed (int): seed of the shuffles

        Returns:
            numpy.ndarray: (n, num_shuffles, length, 4), of the type
                of the sequences
    """

    out = np.empty((len(sequences), num_shuffles) + sequences.shape[1:],
                   dtype=sequences.dtype)
    for i, sequence in enumerate(sequences):
        out[i] = dinuc_shuffle(np.ascontiguousarray(sequence), num_shuffles,
                               np.random.RandomState(seed))
    return out


def hypothetical_scores(model, which, sequences, controls=None,
                        num_shuffles=20, seed=20210304, batch_size=128,
                        device=None, chunk=256, progress=None):
    """
        The hypothetical contribution scores of bpnet-shap for one
        output of a model, or for both

        Args:
            model (bpnet.pytorch.model.Model): the model; it computes
                in the type of its parameters
            which (str or tuple): 'counts' or 'profile', or both (the
                references of a sequence are then made once)
            sequences (numpy.ndarray): one hot, (n, length, 4), as
                bpnet-shap has them (a position without a base is all
                zeros)
            controls (dict): for a model with control inputs, input
                name -> array (n, ...) in the layout of the Keras
                model ((n, length, tracks) for the control profiles)
            num_shuffles (int): references per sequence
            seed (int): seed of the shuffles
            batch_size (int): pairs of a sequence and a reference per
                forward pass (the model sees twice as many rows)
            device (str or torch.device): where to compute; None: a
                GPU when there is one
            chunk (int): sequences whose references are held in
                memory at a time
            progress (callable): called with the number of sequences
                done after every chunk

        Returns:
            numpy.ndarray: (n, length, 4), of the type of the model's
                parameters: the score each of the four bases would
                have at each position. A list of such arrays, in the
                order of `which`, when `which` names several outputs.
    """

    from tangermeme.deep_lift_shap import deep_lift_shap

    several = not isinstance(which, str)
    names_of_targets = list(which) if several else [which]
    kinds = {"counts": CountsTarget, "profile": ProfileTarget}
    if not names_of_targets or \
            any(name not in kinds for name in names_of_targets):
        raise ValueError("which: 'counts' or 'profile', not {!r}".format(
            which))
    targets = [kinds[name](model) for name in names_of_targets]
    names = model.input_names[1:]
    controls = dict(controls or {})
    if sorted(controls) != sorted(names):
        raise ValueError(
            "The model takes the control inputs {}; given: {}".format(
                names, sorted(controls)))
    sequences = np.asarray(sequences)
    if sequences.ndim != 3 or sequences.shape[2] != 4:
        raise ValueError(
            "sequences: one hot of shape (n, length, 4) expected, not "
            "{}".format(sequences.shape))
    dtype = next(model.parameters()).dtype
    outs = [np.zeros(sequences.shape,
                     dtype=torch.zeros(0, dtype=dtype).numpy().dtype)
            for _ in targets]

    was_training = model.training
    model.eval()
    try:
        with _FullPrecision():
            _score(deep_lift_shap, targets, sequences, controls, names,
                   num_shuffles, seed, batch_size, device, chunk, progress,
                   dtype, outs)
    finally:
        model.train(was_training)
    return outs if several else outs[0]


def _score(deep_lift_shap, targets, sequences, controls, names, num_shuffles,
           seed, batch_size, device, chunk, progress, dtype, outs):
    """
        The scores of all sequences for every target, a chunk of
        sequences at a time, into `outs`
    """

    for start in range(0, len(sequences), chunk):
        rows = slice(start, start + chunk)
        # as small integers: the shuffles of a chunk are large,
        # and deep_lift_shap converts a batch at a time
        x = np.ascontiguousarray(sequences[rows], dtype=np.int8)
        if not np.array_equal(x, sequences[rows]) or \
                x.min() < 0 or x.max() > 1 or x.sum(axis=2).max() > 1:
            raise ValueError(
                "sequences: one hot expected (zeros and at most one "
                "1 per position)")
        shuffled = torch.as_tensor(np.ascontiguousarray(
            references(x, num_shuffles, seed).transpose(0, 1, 3, 2)))
        x = torch.as_tensor(channels_first(x))
        args = tuple(
            torch.as_tensor(channels_first(np.asarray(controls[name])[
                rows])).type(dtype) for name in names)
        for target, out in zip(targets, outs):
            with warnings.catch_warnings():
                # positions without a base (N) are all zeros here, as
                # in bpnet-shap; deep_lift_shap is told to go on
                warnings.filterwarnings(
                    "ignore", message=".*one-hot.*", category=Warning)
                if names:
                    # deep_lift_shap compares the change of the output
                    # with what the scores add up to. With control
                    # inputs the output also changes because the
                    # references have none, which no score of the
                    # sequence accounts for: the comparison does not
                    # apply
                    warnings.filterwarnings(
                        "ignore", message="Convergence deltas.*",
                        category=RuntimeWarning)
                scores = deep_lift_shap(
                    target, x, args=args or None, references=shuffled,
                    hypothetical=True, batch_size=batch_size,
                    device=device, dtype=dtype, only_warn=True)
            out[rows] = channels_last(scores.numpy())
        if progress is not None:
            progress(min(start + chunk, len(sequences)))
