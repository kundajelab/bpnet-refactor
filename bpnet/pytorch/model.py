"""
    The models of this package in PyTorch

    A saved model is a list of layers with their settings and
    connections and the weights by layer name
    (bpnet.model.loading.read_architecture, read_weights). This
    module builds the same graph from PyTorch modules, layer for
    layer and under the same names, and moves the weights between
    the two layouts. A model that was trained with TensorFlow can
    then be used from PyTorch, with the same outputs up to rounding.

    Layout of the tensors: PyTorch and the tools written for it keep
    the channels (bases, tracks, filters) on axis 1 and the positions
    last; Keras keeps the positions on axis 1 and the channels last.
    The model here takes and returns PyTorch's layout: a sequence is
    (batch, 4, length), a profile (batch, tracks, length), counts
    (batch, values). channels_first() and channels_last() convert
    arrays of batches between the two.

    Layers with weights, activations and dropout are modules, under
    the names of the layers (`Model.layers`); cropping, addition,
    concatenation, pooling, flattening and the sum of the control
    counts are steps of forward() without a module. Every activation
    is a module of its own (also the one that a Keras layer has built
    in, as '<layer>_activation'), and no module is used twice, so that
    attribution methods that hook the nonlinearities find each of
    them once.

    The parameters are those of the modules ('layers.<layer>.weight',
    'layers.<layer>.bias' in the model's state dictionary, in
    PyTorch's layout); get_weights() and set_weights() exchange them
    with the layout of the saved models.

    What is supported is what the models of this package are made of,
    which is less than Keras accepts for the same layer classes:
    Conv1D with 'valid' or 'same' padding and no stride, Dense after
    pooling or flattening, the activations of _ACTIVATIONS, Add of
    equal shapes, Concatenate along the channels, one task. Anything
    else is an UnsupportedModelError, never another computation.

    forward() takes its inputs by position or by name and checks
    their shapes: the model can be traced (torch.jit.trace keeps the
    computation and drops the check) but not scripted.
"""

import re

import numpy as np
import torch

from bpnet.model import loading
from bpnet.model.loading import UnsupportedModelError

#: activations a Conv1D or Dense layer of Keras may have built in
#: (besides 'linear'), by the names Keras stores, and their modules
_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    # log(1 + exp(x)); PyTorch returns x itself above the threshold,
    # where the two differ by less than 1e-13
    "softplus": lambda: torch.nn.Softplus(beta=1.0, threshold=30.0),
    "softsign": torch.nn.Softsign,
    "swish": torch.nn.SiLU,
    "silu": torch.nn.SiLU,
    # Keras' default: the exact form, with the error function
    "gelu": torch.nn.GELU,
}


def _variable_type(config):
    """
        The type the variables of a layer were saved with, by the
        layer's settings: its type, or float32 under a mixed precision
        policy. None when the file does not say.
    """

    stored = config.get("dtype")
    if isinstance(stored, dict) and \
            stored.get("class_name") in ("Policy", "DTypePolicy") and \
            isinstance(stored.get("config", {}).get("name"), str):
        stored = stored["config"]["name"]
    if stored is None:
        return None
    if not isinstance(stored, str):
        raise UnsupportedModelError(
            "Layer {}: dtype {} is not supported".format(
                config.get("name"), stored))
    return "float32" if stored.startswith("mixed_") else stored


def channels_first(array):
    """
        A batch in PyTorch's layout from one in Keras' layout

        Args:
            array (numpy.ndarray or torch.Tensor): (batch, positions,
                channels); anything with fewer than three axes is
                returned as it is

        Returns:
            the same type: (batch, channels, positions)
    """

    if array.ndim < 3:
        return array
    if isinstance(array, torch.Tensor):
        return array.transpose(1, 2)
    return np.ascontiguousarray(np.swapaxes(array, 1, 2))


def channels_last(array):
    """
        A batch in Keras' layout from one in PyTorch's layout (the
        same exchange of axes as channels_first)
    """

    return channels_first(array)


class Model(torch.nn.Module):
    """
        A layer graph of this package as a PyTorch module

        Args:
            architecture (dict): the layer graph, as
                bpnet.model.loading.read_architecture returns it

        Attributes:
            layers (torch.nn.ModuleDict): the modules by layer name.
                The built-in activation of a layer 'x' is
                'x_activation'.
            input_names (list): names of the inputs, in the order
                forward() takes them
            output_names (list): names of the layers whose outputs
                forward() returns
            shapes (dict): layer name -> shape of its output without
                the batch axis, in PyTorch's layout
            saved_types (dict): name of a layer with weights -> the
                type its variables were saved with ('float32',
                'float64'), None when the layer graph does not say
    """

    def __init__(self, architecture):
        super(Model, self).__init__()
        self.layers = torch.nn.ModuleDict()
        self.shapes = {}
        self.saved_types = {}
        #: (layer name, kind, names of its inputs, settings) in an
        #: order in which every layer comes after its inputs
        self._steps = []
        self.input_names = loading._endpoint_names(
            architecture["input_layers"], "inputs")
        self.output_names = loading._endpoint_names(
            architecture["output_layers"], "outputs")

        pending = list(architecture["layers"])
        names = [layer["name"] for layer in pending]
        if len(set(names)) != len(names):
            raise UnsupportedModelError("Layer names are not unique")
        #: the names of all layers, with or without a module
        self._names = set(names)
        while pending:
            remaining = []
            for layer in pending:
                inputs = loading._input_names(layer)
                if any(name not in self.shapes for name in inputs):
                    remaining.append(layer)
                    continue
                self._add(layer, inputs)
            if len(remaining) == len(pending):
                raise UnsupportedModelError(
                    "Layers {} take the output of layers that do not "
                    "exist".format([layer["name"] for layer in remaining]))
            pending = remaining
        for name in self.input_names + self.output_names:
            if name not in self.shapes:
                raise UnsupportedModelError(
                    "The model has no layer {}".format(name))

    # ----------------------------------------------------------------
    # building
    # ----------------------------------------------------------------

    def _add(self, layer, inputs):
        """
            Add one layer of the graph: its module (if it has one),
            the shape of its output and its step in forward()
        """

        kind, config, name = layer["class_name"], layer["config"], \
            layer["name"]
        shapes = [self.shapes[other] for other in inputs]
        settings = {}

        if kind == "InputLayer":
            loading._expect(config, sparse=False, ragged=False)
            shape = config.get("batch_input_shape", config.get("batch_shape"))
            shape = tuple(int(n) for n in shape[1:])
            if len(shape) == 2:
                # (positions, channels) -> (channels, positions)
                shape = (shape[1], shape[0])
            elif len(shape) != 1:
                raise UnsupportedModelError(
                    "Input {} has shape {}".format(name, shape))
            self.shapes[name] = shape
            self._steps.append((name, kind, inputs, settings))
            return
        if not inputs:
            raise UnsupportedModelError("Layer {} has no input".format(name))
        if kind not in ("Add", "Concatenate") and len(inputs) != 1:
            raise UnsupportedModelError(
                "Layer {} takes {} inputs".format(name, len(inputs)))
        shape = shapes[0]

        if kind == "Conv1D":
            loading._expect(config, strides=1, data_format="channels_last",
                            groups=1)
            self._needs(name, shape, 2)
            width = int(loading._scalar(config["kernel_size"]))
            dilation = int(loading._scalar(config.get("dilation_rate", 1)))
            span = dilation * (width - 1)
            if config["padding"] == "valid":
                settings["padding"] = None
                length = shape[1] - span
            elif config["padding"] == "same":
                # as Keras pads: the odd position goes to the right
                settings["padding"] = (span // 2, span - span // 2)
                length = shape[1]
            else:
                raise UnsupportedModelError(
                    "Layer {}: padding {} is not supported".format(
                        name, config["padding"]))
            if length < 1:
                raise UnsupportedModelError(
                    "Layer {}: its input of {} positions is shorter than "
                    "its kernel".format(name, shape[1]))
            self._register(name, torch.nn.Conv1d(
                shape[0], int(config["filters"]), width, dilation=dilation,
                bias=bool(config.get("use_bias", True))))
            self.shapes[name] = (int(config["filters"]), length)
            self.saved_types[name] = _variable_type(config)
            settings["activation"] = self._activation(name, config)
        elif kind == "Dense":
            self._needs(name, shape, 1)
            self._register(name, torch.nn.Linear(
                shape[0], int(config["units"]),
                bias=bool(config.get("use_bias", True))))
            self.shapes[name] = (int(config["units"]),)
            self.saved_types[name] = _variable_type(config)
            settings["activation"] = self._activation(name, config)
        elif kind == "ReLU":
            loading._expect(config, max_value=None, negative_slope=0.0,
                            threshold=0.0)
            self._register(name, torch.nn.ReLU())
            self.shapes[name] = shape
        elif kind == "LeakyReLU":
            slope = config.get("alpha", config.get("negative_slope", 0.3))
            self._register(name, torch.nn.LeakyReLU(float(slope)))
            self.shapes[name] = shape
        elif kind == "Dropout":
            loading._expect(config, noise_shape=None)
            self._register(name, torch.nn.Dropout(float(config["rate"])))
            self.shapes[name] = shape
        elif kind == "Cropping1D":
            self._needs(name, shape, 2)
            cropping = config["cropping"]
            if not isinstance(cropping, (list, tuple)):
                cropping = (cropping, cropping)
            left, right = int(cropping[0]), int(cropping[1])
            if left < 0 or right < 0:
                raise UnsupportedModelError(
                    "Layer {}: cropping {} is not supported".format(
                        name, (left, right)))
            if left + right >= shape[1]:
                raise UnsupportedModelError(
                    "Layer {} crops {} of {} positions".format(
                        name, left + right, shape[1]))
            settings["positions"] = (left, shape[1] - right)
            self.shapes[name] = (shape[0], shape[1] - left - right)
        elif kind == "Add":
            if any(other != shape for other in shapes):
                raise UnsupportedModelError(
                    "Layer {} adds outputs of shapes {}".format(name, shapes))
            self.shapes[name] = shape
        elif kind == "Concatenate":
            # the channels are the last axis in Keras, axis 1 here
            axis = config.get("axis", -1)
            if axis not in (-1, len(shape)):
                raise UnsupportedModelError(
                    "Layer {} concatenates along axis {}; only the "
                    "channels are supported".format(name, axis))
            if any(other[1:] != shape[1:] or len(other) != len(shape)
                   for other in shapes):
                raise UnsupportedModelError(
                    "Layer {} concatenates outputs of shapes {}".format(
                        name, shapes))
            self.shapes[name] = (sum(other[0] for other in shapes),) + \
                shape[1:]
        elif kind == "GlobalAveragePooling1D":
            loading._expect(config, data_format="channels_last",
                            keepdims=False)
            self._needs(name, shape, 2)
            self.shapes[name] = (shape[0],)
        elif kind == "Flatten":
            loading._expect(config, data_format="channels_last")
            self.shapes[name] = (int(np.prod(shape)),)
            settings["channels_last"] = len(shape) == 2
        elif kind == "Lambda":
            # the function is stored as byte code of the Python that
            # saved the model; the layers of this package are known by
            # their names
            if not re.match(r"^.*_logsumexp_counts_bias_\d+$", name):
                raise UnsupportedModelError(
                    "Lambda layer {} is not supported (models with more "
                    "than one task cannot be loaded yet)".format(name))
            self._needs(name, shape, 1)
            kind = "LogSumExp"
            self.shapes[name] = (1,)
        else:
            raise UnsupportedModelError(
                "Layer {} has class {}, which the models of this package "
                "do not use".format(name, kind))
        self._steps.append((name, kind, inputs, settings))

    def _register(self, name, module):
        """
            The module of a layer, under the layer's name
        """

        if name in self.layers:
            raise UnsupportedModelError(
                "Layer names are not unique: {} (the activation that a "
                "layer 'x' has built in is 'x_activation' here)".format(
                    name))
        try:
            self.layers[name] = module
        except (KeyError, TypeError) as error:
            # a name with a dot, or one that ModuleDict uses itself
            raise UnsupportedModelError(
                "Layer {}: PyTorch cannot register a module under this "
                "name ({})".format(name, error))

    @staticmethod
    def _needs(name, shape, axes):
        if len(shape) != axes:
            raise UnsupportedModelError(
                "Layer {} cannot take an input of shape {}".format(
                    name, shape))

    def _activation(self, name, config):
        """
            The module of the activation a Keras layer has built in
            (None for 'linear'), registered as '<name>_activation'
        """

        activation = loading._activation(config)
        if activation in (None, "linear"):
            return None
        if activation not in _ACTIVATIONS:
            raise UnsupportedModelError(
                "Layer {}: activation {} is not supported".format(
                    name, activation))
        key = name + "_activation"
        if key in self._names:
            # a layer of this name, with or without a module
            raise UnsupportedModelError(
                "Layer names are not unique: {} (the activation that a "
                "layer 'x' has built in is 'x_activation' here)".format(
                    key))
        self._register(key, _ACTIVATIONS[activation]())
        return key

    # ----------------------------------------------------------------
    # running
    # ----------------------------------------------------------------

    def forward(self, *inputs, **named):
        """
            Args:
                *inputs (torch.Tensor): the inputs in the order of
                    input_names, without the batch axis of the shapes
                    in `shapes` (a sequence is (batch, 4, length))
                **named (torch.Tensor): or by their names

            Returns:
                torch.Tensor or tuple: the outputs in the order of
                    output_names (a tensor if there is one)
        """

        values = dict(zip(self.input_names, inputs))
        twice = set(values) & set(named)
        values.update(named)
        if len(inputs) > len(self.input_names) or twice or \
                sorted(values) != sorted(self.input_names):
            raise ValueError(
                "The model takes the inputs {}; given: {} by position, "
                "{} by name".format(self.input_names, len(inputs),
                                    sorted(named)))
        for name in self.input_names:
            if tuple(values[name].shape[1:]) != self.shapes[name]:
                raise ValueError(
                    "Input {} has shape {} after the batch axis; the "
                    "model needs {}".format(
                        name, tuple(values[name].shape[1:]),
                        self.shapes[name]))

        for name, kind, sources, settings in self._steps:
            if kind == "InputLayer":
                continue
            x = values[sources[0]]
            if kind == "Conv1D":
                if settings["padding"] is not None:
                    x = torch.nn.functional.pad(x, settings["padding"])
                x = self.layers[name](x)
            elif kind == "Dense":
                x = self.layers[name](x)
            elif kind in ("ReLU", "LeakyReLU", "Dropout"):
                x = self.layers[name](x)
            elif kind == "Cropping1D":
                x = x[:, :, settings["positions"][0]:settings["positions"][1]]
            elif kind == "Add":
                for other in sources[1:]:
                    x = x + values[other]
            elif kind == "Concatenate":
                x = torch.cat([values[other] for other in sources], dim=1)
            elif kind == "GlobalAveragePooling1D":
                x = x.mean(dim=2)
            elif kind == "Flatten":
                if settings["channels_last"]:
                    # the order in which Keras flattens: position by
                    # position
                    x = x.transpose(1, 2)
                x = x.reshape(x.shape[0], -1)
            elif kind == "LogSumExp":
                x = torch.logsumexp(x, dim=1, keepdim=True)
            if settings.get("activation") is not None:
                x = self.layers[settings["activation"]](x)
            values[name] = x

        outputs = tuple(values[name] for name in self.output_names)
        return outputs[0] if len(outputs) == 1 else outputs


# --------------------------------------------------------------------
# weights
# --------------------------------------------------------------------

def _weighted(model):
    """
        The modules of a model that have weights, by layer name
    """

    return {name: module for name, module in model.layers.items()
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear))}


def _from_keras(module, kernel):
    """
        A kernel in the layout PyTorch keeps it in: Conv1D (width, in,
        out) -> (out, in, width), Dense (in, out) -> (out, in)
    """

    if isinstance(module, torch.nn.Conv1d):
        return np.transpose(kernel, (2, 1, 0))
    return np.transpose(kernel, (1, 0))


def set_weights(model, weights):
    """
        Give every layer of the model its saved weights

        Every layer with weights must find its kernel (and bias) under
        its name, with the shape the layer has, and no saved weights
        may be left over. The weights take the type of the model's
        parameters (float32 unless the model was converted).

        Args:
            model (Model): the model
            weights (dict): layer name -> {'kernel': array, 'bias':
                array} in Keras' layout, as
                bpnet.model.loading.read_weights returns it
    """

    modules = _weighted(model)
    unused = sorted(set(weights) - set(modules))
    if unused:
        raise UnsupportedModelError(
            "Saved weights of layers {} have no layer in the "
            "model".format(unused))
    with torch.no_grad():
        for name, module in modules.items():
            if name not in weights:
                raise UnsupportedModelError(
                    "No saved weights for layer {}".format(name))
            saved = dict(weights[name])
            expected = {"kernel": module.weight}
            if module.bias is not None:
                expected["bias"] = module.bias
            if sorted(saved) != sorted(expected):
                raise UnsupportedModelError(
                    "Layer {}: saved variables {}, the layer has "
                    "{}".format(name, sorted(saved), sorted(expected)))
            for key, parameter in expected.items():
                value = np.asarray(saved[key])
                if key == "kernel":
                    if value.ndim != parameter.dim():
                        raise UnsupportedModelError(
                            "Layer {}: the saved kernel has shape {}".format(
                                name, tuple(value.shape)))
                    value = _from_keras(module, value)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise UnsupportedModelError(
                        "Layer {}: the saved {} does not have the shape "
                        "the layer needs".format(name, key))
                parameter.copy_(torch.as_tensor(
                    np.ascontiguousarray(value), dtype=parameter.dtype))


def get_weights(model):
    """
        The weights of a model by layer name, in Keras' layout

        Args:
            model (Model): the model

        Returns:
            dict: layer name -> {'kernel': array, 'bias': array},
                copies: changing them does not change the model
    """

    weights = {}
    for name, module in _weighted(model).items():
        kernel = module.weight.detach().cpu().numpy()
        # (the exchange of axes is its own inverse for both layouts;
        # a copy: the array of .numpy() is the memory of the parameter)
        weights[name] = {"kernel": np.array(
            _from_keras(module, kernel), order="C")}
        if module.bias is not None:
            weights[name]["bias"] = module.bias.detach().cpu().numpy().copy()
    return weights


def _saved_type(weights):
    """
        The type the weights were saved with (float32, or float64 for
        a model that was built with keras' floatx set to it)
    """

    kinds = sorted({str(np.asarray(value).dtype)
                    for saved in weights.values()
                    for value in saved.values()})
    if kinds in ([], ["float32"]):
        return torch.float32
    if kinds == ["float64"]:
        return torch.float64
    raise UnsupportedModelError(
        "Saved weights of types {} are not supported (all float32 or "
        "all float64)".format(kinds))


def from_architecture(architecture, weights=None):
    """
        The PyTorch model of a layer graph

        The model computes in the type of the saved weights: float32,
        or float64 for a model that was saved in float64. (A model
        that was saved under a mixed precision policy has float32
        weights and computes in float32 here, where Keras computes
        most of it in float16.) Weights of another type than the layer
        graph says the layer was saved with are refused; to convert
        weights on purpose, give them to set_weights().

        Args:
            architecture (dict): see
                bpnet.model.loading.read_architecture
            weights (dict): see bpnet.model.loading.read_weights
                (None: the weights are as PyTorch initializes them)

        Returns:
            Model: in evaluation mode (dropout inactive)
    """

    model = Model(architecture)
    if weights is not None:
        for name, declared in model.saved_types.items():
            found = sorted({str(np.asarray(value).dtype)
                            for value in weights.get(name, {}).values()})
            if declared is not None and found not in ([], [declared]):
                raise UnsupportedModelError(
                    "Layer {}: its saved weights are {}, and the layer "
                    "was saved with the type {}".format(
                        name, found, declared))
        model.to(_saved_type(weights))
        set_weights(model, weights)
    return model.eval()


def load_model(path):
    """
        A saved model of this package as a PyTorch module

        Args:
            path (str): a SavedModel directory, or an HDF5 file
                written by model.save(), of any TensorFlow version.
                (Reading the weights of a SavedModel directory needs
                TensorFlow to be installed; an HDF5 file does not.)

        Returns:
            Model: in evaluation mode, with the saved weights
    """

    return from_architecture(loading.read_architecture(path),
                             loading.read_weights(path))
