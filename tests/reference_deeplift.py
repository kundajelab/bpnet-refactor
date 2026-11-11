"""
    DeepLIFT with the rescale rule, written here independently of the
    shap fork that bpnet-shap uses, in PyTorch: the yardstick for the
    hypothetical contribution scores of every base (the scores of the
    three bases that are not in the sequence are what TF-MoDISco reads,
    and no completeness check can see them).

    The model is the BPNet of tests/reference_model.py (one task,
    'valid' padding, with or without control inputs, a counts head of
    one dense layer, one count per task), built from the saved weights.
    A sequence and its reference go through the model as one batch;
    every ReLU passes back the secant (relu(a_x) - relu(a_r)) / (a_x -
    a_r) of the pair instead of its gradient. The multipliers of the
    one hot input then give the hypothetical contribution of base b at
    position j: the mean over the references of
    sum_k (e_b - r_j)_k * m_jk, as bpnet-shap's combine_mult_and_diffref
    computes it.

    What is explained, as bpnet-shap explains it:

    - counts: the sum of the predicted log counts;
    - profile: the sum over all positions of both strands of w * l, with
      l the logits minus their mean and w their softmax, the weights not
      explained; for the pair of a sequence and a reference the weights
      are the mean of the two (what the fork's scores add up to, see
      tests/test_shap_scores.py).

    The references get zero control inputs, as in bpnet-shap.

    Everything is computed in float64. The explainer of bpnet-shap
    computes in float32, and two of its steps are quotients of
    differences that lose precision where the difference is small:

    - the secant of a ReLU, where the pre-activations of the sequence
      and of a reference lie on opposite sides of zero and close to
      each other;
    - the multiplier of a logit in the product w * l. In exact
      arithmetic it is the mean of the two weights. The differences
      that were seen fit an explainer that computes it as the quotient
      0.5 * ((w_x l_x - w_x l_r) + (w_r l_x - w_r l_r)) / (l_x - l_r),
      which is imprecise where a logit of the sequence and of a
      reference nearly coincide, and that replaces it by zero where
      they differ by less than 1e-7. (Probed on a model with two
      logits near 0.2: a score was zero when the logits differed by
      8.9e-8, off by 9% at 1.04e-7 and at 2.1e-7, by 0.2% at 1e-6 and
      at 1.5e-5, and right at 1.5e-3.)

    An imprecise quotient moves the scores of all positions that the
    unit or the output position sees. float32_allowance() estimates,
    for one sequence, how far these two effects can move each score;
    with it the scores of bpnet-shap can be compared with the ones
    computed here score by score. It is an estimate to first order,
    with constants that were chosen with room and checked against
    runs, not a proven bound: the errors of two quotients on one path
    multiply, and what float32 does to a pre-activation is taken as a
    share of the largest one of the layer.

    What was seen in 30 runs of the tests with freshly trained models
    (about 350 peaks per output): the counts scores never differed
    from the ones computed here by more than the rounding of the file
    and 2e-6 of the largest score of the peak. Of the profile scores,
    those of 14 peaks were further off than the rounding and 1e-5 of
    the largest score, by up to 3.4e-4; in the six peaks that were
    looked at, within the window that an output position sees, and in
    one of them a logit of the sequence and of a reference were 9e-7
    apart at one output position (all others more than 4e-4).
    Simulating the quotient in float32 for that peak reproduced the
    pattern of the differences (correlation 0.97). In 16 of the runs
    the allowance was computed as well: of 192 profile peaks, 31 had
    scores more than 1e-6 beyond the rounding, and none of these
    scores was further off than 0.2 of its allowance.
"""

import numpy as np
import torch
import torch.nn.functional as F

#: pre-activations of a pair closer than this pass back the gradient
#: of the ReLU instead of the secant (the number the explainer
#: compares with in float32)
NEAR = float(np.float32(1e-6))

#: what float32 arithmetic leaves uncertain, for float32_allowance():
#: a pre-activation, as a share of the largest pre-activation of its
#: layer (float32 has 24 bits; eight units in the last place)
RELU_NOISE = 2.0 ** -21
#: and the multiplier of a logit: this times (|l_x| + |l_r|) /
#: |l_x - l_r| of its value. (Each of the four products of the
#: quotient is rounded to 2**-24 of its size, and so is one of the
#: sums: at most 1.25 * 2**-24 by this ratio, to first order.)
PRODUCT_NOISE = 2.0 ** -23
#: logits that are closer than this may have their multiplier replaced
#: by zero (the explainer does that below 1e-7; which side a pair is
#: on is decided in float32)
COINCIDE = 1e-6
#: quotients that are uncertain by less than these shares are left out
#: of the allowance (many small ones remain; what they move is part of
#: the tolerance of the test that uses the allowance)
SMALLEST_RELU, SMALLEST_PRODUCT = 1e-4, 1e-5


class _RescaledReLU(torch.autograd.Function):
    """
        A ReLU over a batch whose first half are the sequences and whose
        second half are their references, row for row. `bump` (None, or
        an array of the shape of the first half) is added to the secants
        that are passed back.
    """

    @staticmethod
    def forward(ctx, a, bump):
        ctx.save_for_backward(a)
        ctx.bump = bump
        return F.relu(a)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        half = a.shape[0] // 2
        a_x, a_r = a[:half], a[half:]
        delta = a_x - a_r
        secant = (F.relu(a_x) - F.relu(a_r)) / torch.where(
            delta.abs() >= NEAR, delta, torch.ones_like(delta))
        slope = torch.where(delta.abs() >= NEAR, secant,
                            (a_x > 0).to(a.dtype))
        if ctx.bump is not None:
            slope = slope + torch.as_tensor(ctx.bump, dtype=a.dtype)
        return grad * torch.cat([slope, slope]), None


def conv(x, kernel, bias, dilation=1):
    """
        x (batch, length, in), kernel (width, in, out) as Keras keeps it
    """

    weight = torch.as_tensor(np.transpose(kernel, (2, 1, 0)), dtype=x.dtype)
    out = F.conv1d(x.transpose(1, 2), weight,
                   torch.as_tensor(bias, dtype=x.dtype), dilation=dilation)
    return out.transpose(1, 2)


def crop(x, length):
    trim = x.shape[1] // 2 - length // 2
    return x[:, trim:x.shape[1] - trim, :]


def forward(weights, params, sequence, profile_bias=None, counts_bias=None,
            name_prefix="main", bumps=None, seen=None):
    """
        The forward pass of tests/reference_model.py in torch, with the
        rescaled ReLUs

        Args:
            bumps (dict): number of a ReLU layer (in the order of the
                forward pass) -> what to add to its secants
            seen (list): if given, the pre-activations of every ReLU
                layer are appended to it

        Returns:
            tuple: (profile logits (batch, output_len, tracks), log
                counts (batch, 1))
    """

    motif = params["motif_module_params"]
    syntax = params["syntax_module_params"]
    assert motif["padding"] == syntax["padding"] == "valid"
    assert len(params["counts_head_params"]["units"]) == 1
    layer = [0]

    def relu(a):
        bump = None if bumps is None else bumps.get(layer[0])
        layer[0] += 1
        if seen is not None:
            seen.append(a.detach().numpy())
        return _RescaledReLU.apply(a, bump)

    x = sequence
    n_motif = len(motif["kernel_sizes"])
    for i in range(n_motif):
        x = conv(x, *weights["{}_conv_{}".format(name_prefix, i)])
        if i < n_motif - 1:
            x = relu(x)
    for i in range(1, syntax["num_dilation_layers"] + 1):
        activated = relu(x)
        out = conv(activated, *weights["{}_dil_conv_{}".format(
            name_prefix, i)], dilation=2 ** i)
        if syntax["pre_activation_residual_unit"]:
            x = out + crop(x, out.shape[1])
        else:
            x = out + crop(activated, out.shape[1])
    x = relu(x)
    profile = crop(conv(x, *weights["{}_profile_head".format(name_prefix)]),
                   params["output_profile_len"])
    pooled = x.mean(dim=1)
    if profile_bias is None:
        kernel, bias = weights["logcounts_predictions"]
        counts = pooled @ torch.as_tensor(kernel, dtype=x.dtype) + \
            torch.as_tensor(bias, dtype=x.dtype)
        return profile, counts
    kernel, bias = weights["{}_counts_head".format(name_prefix)]
    counts = pooled @ torch.as_tensor(kernel, dtype=x.dtype) + \
        torch.as_tensor(bias, dtype=x.dtype)
    profile = conv(torch.cat([profile, profile_bias], dim=-1),
                   *weights["profile_predictions"])
    kernel, bias = weights["logcounts_predictions"]
    counts = torch.cat([counts, torch.logsumexp(counts_bias, dim=-1,
                                                keepdim=True)], dim=-1) @ \
        torch.as_tensor(kernel, dtype=x.dtype) + \
        torch.as_tensor(bias, dtype=x.dtype)
    return profile, counts


def meannormed(profile):
    """
        The logits of all positions of both strands minus their mean,
        (batch, positions * tracks)
    """

    logits = profile.reshape(profile.shape[0], -1)
    return logits - logits.mean(dim=1, keepdim=True)


def paired_weights(logits):
    """
        The weights of the logits of a batch of sequences followed by
        their references: the mean of the softmax of a sequence and of
        its reference, (batch / 2, positions * tracks), not explained
    """

    w = torch.softmax(logits, dim=1).detach()
    half = logits.shape[0] // 2
    return 0.5 * (w[:half] + w[half:])


def explained(which, profile, counts, bump=None):
    """
        The output that bpnet-shap explains, one value per row of a
        batch of sequences followed by their references

        Args:
            bump (numpy.ndarray): for the profile, what to add to the
                weights of the logits (batch / 2, positions * tracks)
    """

    if which == "counts":
        return counts.sum(dim=-1)
    logits = meannormed(profile)
    w = paired_weights(logits)
    if bump is not None:
        w = w + torch.as_tensor(bump, dtype=w.dtype)
    return (torch.cat([w, w]) * logits).sum(dim=1)


def _pairs(sequence, references, profile_bias, counts_bias):
    """
        The batch of a sequence with each of the references: the
        sequence as often as there are references, then the references;
        the control inputs of the sequence, and zeros for the references
    """

    n = len(references)
    x = np.concatenate([np.repeat(sequence[None], n, axis=0), references])
    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    extra = {}
    if profile_bias is not None:
        pb = np.concatenate([np.repeat(profile_bias[None], n, axis=0),
                             np.zeros((n,) + profile_bias.shape)])
        cb = np.concatenate([np.repeat(counts_bias[None], n, axis=0),
                             np.zeros((n,) + counts_bias.shape)])
        extra = {"profile_bias": torch.tensor(pb, dtype=torch.float64),
                 "counts_bias": torch.tensor(cb, dtype=torch.float64)}
    return x, extra


def _hypothetical(weights, params, which, sequence, references,
                  profile_bias, counts_bias, bumps=None, weight_bump=None,
                  seen=None):
    """
        The hypothetical contribution scores of a sequence with respect
        to each of the references, (references, length, 4), and the
        logits minus their mean of the batch
    """

    n = len(references)
    x, extra = _pairs(sequence, references, profile_bias, counts_bias)
    profile, counts = forward(weights, params, x, bumps=bumps, seen=seen,
                              **extra)
    target = explained(which, profile, counts, weight_bump)
    # the multipliers of the sequence half (the target of a row depends
    # on the reference half through the rescaled ReLUs only)
    multipliers, = torch.autograd.grad(target[:n].sum(), x)
    multipliers = (multipliers[:n] + multipliers[n:]).numpy()
    hypothetical = np.zeros(multipliers.shape)
    for base in range(sequence.shape[1]):
        e = np.zeros(sequence.shape)
        e[:, base] = 1.0
        hypothetical[:, :, base] = ((e[None] - references) *
                                    multipliers).sum(axis=-1)
    return hypothetical, meannormed(profile).detach()


def hypothetical_scores(weights, params, which, sequence, references,
                        profile_bias=None, counts_bias=None):
    """
        The hypothetical contribution scores of one sequence

        Args:
            weights (dict): layer name -> [kernel, bias] (Keras layout)
            params (dict): architecture parameters
            which (str): 'counts' or 'profile'
            sequence (numpy.ndarray): one hot (length, 4)
            references (numpy.ndarray): one hot (n, length, 4)
            profile_bias, counts_bias (numpy.ndarray): the control
                inputs of the sequence ((output_len, tracks), (tracks,));
                the references get zeros

        Returns:
            numpy.ndarray: (length, 4), float64
    """

    hypothetical, _ = _hypothetical(weights, params, which, sequence,
                                    references, profile_bias, counts_bias)
    return hypothetical.mean(axis=0)


def relu_uncertainty(a_x, a_r, largest):
    """
        How uncertain float32 leaves what a ReLU passes back for a
        sequence and a reference: the change of the multiplier that
        rounding can cause, between 0 and 1

        The multiplier is the secant (relu(a_x) - relu(a_r)) / (a_x -
        a_r), or the gradient of the ReLU at a_x when the two are
        closer than NEAR. With both pre-activations well on one side
        of zero it is 0 or 1 whatever the rounding, and so is the
        gradient where a_x is well away from zero and the pair surely
        closer than NEAR. Otherwise: a secant moves by about (the
        uncertainty of the difference) / (the difference); and where
        rounding can decide between the secant and the gradient, or
        the sign that the gradient goes by, the multiplier may be
        anything from 0 to 1. (Two pre-activations that are exactly
        zero here need not be zero in float32: they are not taken as
        settled.)

        Args:
            a_x, a_r (numpy.ndarray): pre-activations of the sequence
                and of the reference
            largest (float): the largest pre-activation of the layer,
                to which the rounding of a pre-activation is taken to
                be in proportion

        Returns:
            numpy.ndarray: like a_x
    """

    delta = np.abs(a_x - a_r)
    # how far the difference of a pair may be off
    off = 2.0 * RELU_NOISE * largest + \
        2.0 ** -20 * (np.abs(a_x) + np.abs(a_r))
    # the secant for sure, the gradient for sure, or either
    secant = delta >= NEAR + off
    gradient = delta <= NEAR - off
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(secant, np.minimum(1.0, off / delta), 1.0)
    # both well on one side: the secant and the gradient are the same
    # 0 or 1
    share[((a_x > off) & (a_r > off)) | ((a_x < -off) & (a_r < -off))] = 0.0
    # the gradient, at a sequence that is well away from zero
    share[gradient & (np.abs(a_x) > off)] = 0.0
    return share


def float32_allowance(weights, params, which, sequence, references,
                      profile_bias=None, counts_bias=None):
    """
        How far the quotients that are imprecise in float32 (see the
        top of this module) can move each hypothetical score of one
        sequence, to first order

        For every such quotient (the secant of one unit for one
        reference, the multiplier of one logit for one reference) the
        scores are computed again with the quotient changed by its
        uncertainty; with the other quotients fixed the scores are
        linear in each quotient, so that the change of a score is its
        largest possible change from this quotient alone. The
        allowance is the sum of the absolute changes.

        Args:
            as hypothetical_scores

        Returns:
            numpy.ndarray: (length, 4), float64, not negative; zero
                where no imprecise quotient reaches
    """

    n = len(references)
    seen = []
    exact, logits = _hypothetical(weights, params, which, sequence,
                                  references, profile_bias, counts_bias,
                                  seen=seen)
    #: (reference, ReLU layer or None for the product, index within
    #: the layer or the logits, uncertainty of the quotient)
    uncertain = []
    for layer, a in enumerate(seen):
        share = relu_uncertainty(a[:n], a[n:], np.abs(a).max())
        for index in zip(*np.nonzero(share > SMALLEST_RELU)):
            uncertain.append((index[0], layer, index[1:], share[index]))
    if which == "profile":
        mean_weights = paired_weights(logits).numpy()
        l_x, l_r = logits[:n].numpy(), logits[n:].numpy()
        delta = np.abs(l_x - l_r)
        # (the quotient is not used below 1e-7: no difference smaller
        # than that divides)
        share = PRODUCT_NOISE * (np.abs(l_x) + np.abs(l_r)) / \
            np.maximum(delta, 1e-7)
        # replaced by zero, or not: the whole multiplier, with room
        share[delta < COINCIDE] = np.maximum(share[delta < COINCIDE], 2.0)
        for index in zip(*np.nonzero(share > SMALLEST_PRODUCT)):
            uncertain.append((index[0], None, index[1:],
                              share[index] * mean_weights[index]))
    if not uncertain:
        return np.zeros(sequence.shape)

    # one pair of the sequence and the reference for every uncertain
    # quotient, with this quotient changed
    of = np.array([entry[0] for entry in uncertain])
    bumps = {layer: np.zeros((len(uncertain),) + a.shape[1:])
             for layer, a in enumerate(seen)}
    weight_bump = np.zeros((len(uncertain), logits.shape[1]))
    for row, (_, layer, index, size) in enumerate(uncertain):
        if layer is None:
            weight_bump[(row,) + tuple(index)] = size
        else:
            bumps[layer][(row,) + tuple(index)] = size
    changed, _ = _hypothetical(
        weights, params, which, sequence, references[of], profile_bias,
        counts_bias, bumps=bumps,
        weight_bump=weight_bump if which == "profile" else None)
    return np.abs(changed - exact[of]).sum(axis=0) / n
