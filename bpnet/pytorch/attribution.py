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
    (sequences, then references, row for row). The layout is looked
    at once per process with a small model that notes what it is
    given (_check_layout), so that a later tangermeme with another
    layout is refused instead of giving other scores; and the tests
    compare the scores with an independent DeepLIFT.

    The scores are verified for models whose only nonlinear layers
    are ReLU, which is every model bpnet-train makes with its default
    architecture; a model with another activation is refused.

    The scores are computed in the type of the model's parameters and
    nothing less: on a GPU with TF32 switched off, without the
    reduced precision that torch.set_float32_matmul_precision allows,
    and outside any autocast of the caller. PyTorch lets cuDNN
    compute convolutions in TF32 (10 bits of mantissa) by default on
    the GPUs that have it; with it, 4% of the profile scores of a
    trained ENCODE model were more than 1e-5 of the largest score of
    their peak off, by up to 9e-4; without it, all were within 5e-8
    of an independent float64 computation, at the same speed.
"""

import contextlib
import warnings

import numpy as np
import torch

from bpnet.pytorch.model import channels_first, channels_last
from bpnet.utils.shaputils import dinuc_shuffle


#: the kinds of layer for which the scores are verified against an
#: independent DeepLIFT: linear ones, dropout (which does nothing
#: when a model is evaluated) and ReLU
_VERIFIED_LAYERS = (torch.nn.Conv1d, torch.nn.Linear, torch.nn.Dropout,
                    torch.nn.ReLU, torch.nn.ModuleDict)


def check_model(model):
    """
        Refuse a model with a layer for which the scores of this
        module are not verified (an activation other than ReLU:
        deep_lift_shap has rules for many of them, and none of them
        has been compared with bpnet-shap)

        Raises:
            ValueError: the model has such a layer
    """

    others = sorted({type(module).__name__ for module in model.modules()
                     if module is not model and
                     not isinstance(module, _VERIFIED_LAYERS)})
    if others:
        raise ValueError(
            "The scores of the PyTorch backend are verified for models "
            "whose only nonlinear layers are ReLU; this model has layers "
            "of the kind {}".format(", ".join(others)))


class _Layout(torch.nn.Module):
    """
        A model that notes the batches it is given
    """

    def __init__(self):
        super(_Layout, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.seen = []

    def forward(self, sequence, mark):
        self.seen.append((sequence.detach().clone(), mark.detach().clone()))
        return (self.weight * sequence).sum(dim=(1, 2)).unsqueeze(-1)


_LAYOUT_CHECKED = []


def _check_layout(deep_lift_shap):
    """
        Is a batch inside deep_lift_shap what the targets of this
        module take it to be: the sequences of the pairs of the batch,
        then the reference of each pair in the same order, and the
        extra inputs of the sequences for both halves? Looked at once
        per process, with three sequences, two references each and
        batches of four pairs.

        Raises:
            RuntimeError: the layout is another one (a tangermeme
                this module was not written for)
    """

    if _LAYOUT_CHECKED:
        return
    # sequence i is all base i; its reference j has base 3 at
    # position j and base i elsewhere: all rows are different
    length, shuffles = 4, 2
    x = torch.zeros(3, 4, length)
    references = torch.zeros(3, shuffles, 4, length)
    for i in range(3):
        x[i, i] = 1
        for j in range(shuffles):
            references[i, j, i] = 1
            references[i, j, :, j] = 0
            references[i, j, 3, j] = 1
    marks = torch.tensor([[1.0], [2.0], [3.0]])
    layout = _Layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deep_lift_shap(layout, x, args=(marks,), references=references,
                       batch_size=4, device="cpu", dtype=torch.float32,
                       only_warn=True)
    pair, as_expected = 0, bool(layout.seen)
    for sequence, mark in layout.seen:
        half = sequence.shape[0] // 2
        as_expected = as_expected and sequence.shape[0] == 2 * half and \
            mark.shape[0] == 2 * half
        for row in range(half if as_expected else 0):
            i, j = divmod(pair, shuffles)
            as_expected = as_expected and i < 3 and \
                torch.equal(sequence[row], x[i]) and \
                torch.equal(sequence[half + row], references[i, j]) and \
                torch.equal(mark[row], marks[i]) and \
                torch.equal(mark[half + row], marks[i])
            pair += 1
    if not as_expected or pair != 3 * shuffles:
        import tangermeme
        raise RuntimeError(
            "deep_lift_shap of tangermeme {} hands a model other batches "
            "than bpnet.pytorch.attribution was written for (sequences, "
            "then their references, row for row): its scores would not be "
            "those of bpnet-shap".format(
                getattr(tangermeme, "__version__", "(unknown version)")))
    _LAYOUT_CHECKED.append(True)


class _FullPrecision(object):
    """
        While inside: float32 is float32. No TF32 on the GPU
        (convolutions and matrix products), no reduced precision of
        matrix products (torch.set_float32_matmul_precision), no
        autocast of the caller; afterwards every setting as it was.

        PyTorch has three sets of switches for this, which it keeps
        side by side: torch.set_float32_matmul_precision, the older
        'allow_tf32' flags, and (PyTorch 2.9 and later) a setting per
        backend and kind of operation, 'fp32_precision'. Each is read
        before any is changed and put back in the order in which the
        later ones overwrite the earlier ones; a switch that cannot be
        read (PyTorch refuses to report one that disagrees with the
        others) is left alone.
    """

    def __enter__(self):
        backends = torch.backends

        def readable(read):
            # (older PyTorch does not have every switch, and says so
            # in several ways)
            try:
                return read()
            except (RuntimeError, AttributeError, AssertionError):
                return None

        matmul = readable(lambda: torch.get_float32_matmul_precision())
        flags = [(backends.cudnn, "allow_tf32")]
        if matmul is None:
            # (no such setting in this PyTorch, or the backends were
            # set one by one in a way it cannot say)
            flags.append((backends.cuda.matmul, "allow_tf32"))
        holders = [("cuda", "matmul"), ("mkldnn", "matmul")] + [
            (backend, kind) for backend in ("cudnn", "mkldnn")
            for kind in ("conv", "rnn")]
        settings = [getattr(getattr(backends, backend, None), kind, None)
                    for backend, kind in holders]
        # what is there now ...
        flags = [(holder, name, readable(lambda: getattr(holder, name)))
                 for holder, name in flags]
        before = [(setting, readable(lambda: setting.fp32_precision))
                  for setting in settings]
        before = [(setting, value) for setting, value in before
                  if value is not None]
        settings = [setting for setting, _ in before]
        # ... full precision ...
        if matmul is not None:
            torch.set_float32_matmul_precision("highest")
        for holder, name, value in flags:
            if value is not None:
                setattr(holder, name, False)
        for setting in settings:
            setting.fp32_precision = "ieee"

        # ... and how to put it back
        def undo():
            if matmul is not None:
                torch.set_float32_matmul_precision(matmul)
            for holder, name, value in flags:
                if value is not None:
                    setattr(holder, name, value)
            for setting, value in before:
                setting.fp32_precision = value

        self._undo = undo
        self._contexts = contextlib.ExitStack()
        for device_type in ("cpu", "cuda"):
            if device_type == "cpu" or torch.cuda.is_available():
                self._contexts.enter_context(torch.autocast(
                    device_type=device_type, enabled=False))

    def __exit__(self, *failure):
        self._contexts.close()
        self._undo()


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

        Raises:
            ValueError: the model has a layer for which the scores
                are not verified (check_model), the inputs are not
                what the model takes, or a number of shuffles, a
                batch size or a chunk below 1
            RuntimeError: the installed tangermeme lays out its
                batches in another way than this module relies on
    """

    from tangermeme.deep_lift_shap import deep_lift_shap

    for name, value in (("num_shuffles", num_shuffles),
                        ("batch_size", batch_size), ("chunk", chunk)):
        # (deep_lift_shap takes a batch size of 0 or less for "all
        # pairs at once")
        if isinstance(value, bool) or \
                not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(
                "{}: a whole number of at least 1 is expected, not "
                "{!r}".format(name, value))
    several = not isinstance(which, str)
    names_of_targets = list(which) if several else [which]
    kinds = {"counts": CountsTarget, "profile": ProfileTarget}
    if not names_of_targets or \
            any(name not in kinds for name in names_of_targets):
        raise ValueError("which: 'counts' or 'profile', not {!r}".format(
            which))
    check_model(model)
    _check_layout(deep_lift_shap)
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
