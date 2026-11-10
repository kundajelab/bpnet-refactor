"""
    Reference implementation of the BPNet forward pass and loss in
    plain numpy (float64), written from the description of the
    architecture and not from the Keras code. It is the yardstick for
    the model tests: any backend (TensorFlow of any version, PyTorch)
    given the same weights must reproduce these numbers.

    Supported: one task, 'valid' padding, with or without control
    (bias) inputs, a counts head with or without hidden layers, the
    joint multinomial loss over the strands of the task or the original
    one-multinomial-per-strand loss.

    Weights are passed as a dictionary: layer name -> [kernel, bias]
    with Keras conventions (Conv1D kernel (width, in, out), Dense
    kernel (in, out)).
"""

import numpy as np
from scipy.special import gammaln, logsumexp


def conv1d_valid(x, kernel, bias, dilation=1):
    """
        1D convolution without padding (cross-correlation, as in
        deep learning frameworks)

        Args:
            x (numpy.ndarray): input of shape (batch, length, in)
            kernel (numpy.ndarray): shape (width, in, out)
            bias (numpy.ndarray): shape (out,)
            dilation (int): spacing between kernel taps

        Returns:
            numpy.ndarray: shape (batch, length - (width - 1) * dilation,
                out)
    """

    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    width = kernel.shape[0]
    out_len = x.shape[1] - (width - 1) * dilation
    out = np.zeros((x.shape[0], out_len, kernel.shape[2]))
    for k in range(width):
        out += x[:, k * dilation:k * dilation + out_len, :] @ kernel[k]
    return out + np.asarray(bias, dtype=np.float64)


def center_crop(x, new_len):
    """
        Remove the same number of positions from both ends, the way the
        model's Cropping1D layers are set up
    """

    crop = x.shape[1] // 2 - new_len // 2
    return x[:, crop:x.shape[1] - crop, :]


def relu(x):
    return np.maximum(x, 0.0)


#: activations of the hidden layers of the counts head; Keras'
#: LeakyReLU layer has a slope of 0.3 for negative inputs
ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": relu,
    "leakyrelu": lambda x: np.where(x > 0, x, 0.3 * x),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
}


def forward(weights, params, sequence, profile_bias=None, counts_bias=None,
            name_prefix="main", orig_multi_loss=False):
    """
        Forward pass

        Args:
            weights (dict): layer name -> [kernel, bias]
            params (dict): architecture parameters (the content of
                bpnet_params.json; all keys used here must be present)
            sequence (numpy.ndarray): one hot input (batch, input_len, 4)
            profile_bias (numpy.ndarray): control profiles (batch,
                output_len, n control tracks) or None
            counts_bias (numpy.ndarray): log(1 + control counts) (batch,
                n control tracks) or None
            name_prefix (str): prefix of the layer names
            orig_multi_loss (bool): the model predicts one count per
                strand instead of one per task

        Returns:
            tuple: (profile logits (batch, output_len, n tracks),
                log counts (batch, 1) or (batch, n tracks))
    """

    motif = params["motif_module_params"]
    syntax = params["syntax_module_params"]
    assert motif["padding"] == syntax["padding"] == "valid"
    assert params["profile_head_params"]["padding"] == "valid"

    # motif module: convolutions with a ReLU between them; the last
    # one is left without activation
    x = np.asarray(sequence, dtype=np.float64)
    n_motif = len(motif["kernel_sizes"])
    for i in range(n_motif):
        kernel, bias = weights["{}_conv_{}".format(name_prefix, i)]
        x = conv1d_valid(x, kernel, bias)
        if i < n_motif - 1:
            x = relu(x)

    # syntax module: dilated convolutions with residual connections
    for i in range(1, syntax["num_dilation_layers"] + 1):
        activated = relu(x)
        kernel, bias = weights["{}_dil_conv_{}".format(name_prefix, i)]
        conv = conv1d_valid(activated, kernel, bias, dilation=2 ** i)
        if syntax["pre_activation_residual_unit"]:
            x = conv + center_crop(x, conv.shape[1])
        else:
            x = conv + center_crop(activated, conv.shape[1])
    x = relu(x)

    # profile head: wide convolution, cropped to the output length
    kernel, bias = weights["{}_profile_head".format(name_prefix)]
    profile = center_crop(conv1d_valid(x, kernel, bias),
                          params["output_profile_len"])

    # counts head: with a single dense layer, the average over all
    # positions feeds it; with hidden layers, all positions of all
    # filters (flattened position by position) go through the hidden
    # layers first (dropout is inactive at inference)
    head = params["counts_head_params"]
    if len(head["units"]) == 1:
        pooled = x.mean(axis=1)
    else:
        pooled = x.reshape(x.shape[0], -1)
        for i in range(len(head["units"]) - 1):
            kernel, bias = weights["counts_dense_{}".format(i)]
            pooled = ACTIVATIONS[head["activations"][i]](
                pooled @ np.asarray(kernel, dtype=np.float64) + bias)
    has_bias = profile_bias is not None
    name = "{}_counts_head".format(name_prefix) if has_bias \
        else "logcounts_predictions"
    kernel, bias = weights[name]
    counts = pooled @ np.asarray(kernel, dtype=np.float64) + bias

    if not has_bias:
        return profile, counts

    # control tracks: a 1x1 convolution mixes the profile head with the
    # control profiles; a dense layer mixes the counts head with the
    # control counts (summed over strands in log space, or per strand
    # with the original loss)
    kernel, bias = weights["profile_predictions"]
    profile = conv1d_valid(
        np.concatenate([profile, profile_bias], axis=-1), kernel, bias)

    counts_bias = np.asarray(counts_bias, dtype=np.float64)
    if not orig_multi_loss:
        counts_bias = logsumexp(counts_bias, axis=-1, keepdims=True)
    kernel, bias = weights["logcounts_predictions"]
    counts = np.concatenate([counts, counts_bias], axis=-1) @ \
        np.asarray(kernel, dtype=np.float64) + bias
    return profile, counts


def multinomial_nll(counts, logits):
    """
        Negative log likelihood of counts under multinomials whose
        probabilities are softmax(logits), averaged over the rows

        Args:
            counts (numpy.ndarray): (rows, positions)
            logits (numpy.ndarray): (rows, positions)

        Returns:
            float
    """

    counts = np.asarray(counts, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    log_p = logits - logsumexp(logits, axis=-1, keepdims=True)
    total = counts.sum(axis=-1)
    log_likelihood = gammaln(total + 1) - gammaln(counts + 1).sum(axis=-1) \
        + (counts * log_p).sum(axis=-1)
    return -log_likelihood.sum() / counts.shape[0]


def loss(profile_true, logcounts_true, sample_weights, profile_pred,
         counts_pred, loss_weights, orig_multi_loss=False):
    """
        The training loss of a batch for a single task with MSE counts
        loss

        Profile loss: multinomial negative log likelihood over the
        examples whose weight is at least 1; with the default loss the
        strands of the task form one multinomial over all positions of
        all strands, with the original loss there is one multinomial
        per strand and the per-strand losses are added. Counts loss:
        mean squared error over all examples; with the default loss
        between the predicted log count and the log of the summed
        strand counts (each strand count has 1 added, so the target is
        log(total + number of strands)), with the original loss per
        strand, added up.

        Returns:
            tuple: (loss, profile loss, counts loss)
    """

    profile_true = np.asarray(profile_true, dtype=np.float64)
    profile_pred = np.asarray(profile_pred, dtype=np.float64)
    logcounts_true = np.asarray(logcounts_true, dtype=np.float64)
    counts_pred = np.asarray(counts_pred, dtype=np.float64)
    keep = np.asarray(sample_weights) >= 1.0
    n_tracks = profile_true.shape[2]

    if orig_multi_loss:
        counts_loss = sum(
            np.mean((logcounts_true[:, t] - counts_pred[:, t]) ** 2)
            for t in range(n_tracks))
        profile_loss = 0.0
        if keep.any():
            profile_loss = sum(
                multinomial_nll(profile_true[keep][:, :, t],
                                profile_pred[keep][:, :, t])
                for t in range(n_tracks))
    else:
        target = logsumexp(logcounts_true, axis=1)
        counts_loss = np.mean((target - counts_pred[:, 0]) ** 2)
        profile_loss = 0.0
        if keep.any():
            n = int(keep.sum())
            profile_loss = multinomial_nll(
                profile_true[keep].reshape(n, -1),
                profile_pred[keep].reshape(n, -1))

    total = loss_weights[0] * profile_loss + loss_weights[1] * counts_loss
    return total, profile_loss, counts_loss
