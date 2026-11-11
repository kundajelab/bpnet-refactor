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
"""

import numpy as np
import torch
import torch.nn.functional as F

#: pre-activations of a pair closer than this pass back the gradient
#: of the ReLU instead of the secant
NEAR = 1e-6


class _RescaledReLU(torch.autograd.Function):
    """
        A ReLU over a batch whose first half are the sequences and whose
        second half are their references, row for row
    """

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return F.relu(a)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        half = a.shape[0] // 2
        a_x, a_r = a[:half], a[half:]
        delta = a_x - a_r
        secant = (F.relu(a_x) - F.relu(a_r)) / torch.where(
            delta.abs() > NEAR, delta, torch.ones_like(delta))
        slope = torch.where(delta.abs() > NEAR, secant,
                            (a_x > 0).to(a.dtype))
        return grad * torch.cat([slope, slope])


def relu(a):
    return _RescaledReLU.apply(a)


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
            name_prefix="main"):
    """
        The forward pass of tests/reference_model.py in torch, with the
        rescaled ReLUs

        Returns:
            tuple: (profile logits (batch, output_len, tracks), log
                counts (batch, 1))
    """

    motif = params["motif_module_params"]
    syntax = params["syntax_module_params"]
    assert motif["padding"] == syntax["padding"] == "valid"
    assert len(params["counts_head_params"]["units"]) == 1
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


def explained(which, profile, counts):
    """
        The output that bpnet-shap explains, one value per row of a
        batch of sequences followed by their references
    """

    if which == "counts":
        return counts.sum(dim=-1)
    logits = profile.reshape(profile.shape[0], -1)
    logits = logits - logits.mean(dim=1, keepdim=True)
    w = torch.softmax(logits, dim=1).detach()
    half = logits.shape[0] // 2
    w = 0.5 * (w[:half] + w[half:])
    return (torch.cat([w, w]) * logits).sum(dim=1)


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
    profile, counts = forward(weights, params, x, **extra)
    target = explained(which, profile, counts)
    # the multipliers of the sequence half (the target of a row depends
    # on the reference half through the rescaled ReLUs only)
    multipliers, = torch.autograd.grad(target[:n].sum(), x)
    multipliers = (multipliers[:n] + multipliers[n:]).numpy()
    hypothetical = np.zeros(sequence.shape)
    for base in range(sequence.shape[1]):
        e = np.zeros(sequence.shape)
        e[:, base] = 1.0
        hypothetical[:, base] = ((e[None] - references) *
                                 multipliers).sum(axis=-1).mean(axis=0)
    return hypothetical
