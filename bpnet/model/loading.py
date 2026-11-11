"""
    Load a saved model for prediction, whatever TensorFlow version
    saved it

    Why this module exists. Models are saved with keras' model.save(),
    as a SavedModel directory (or, by older versions of this package,
    as an HDF5 file). keras' own load_model() rebuilds the model from
    the configuration stored with it, and that fails outside the
    version that saved the model: TensorFlow 2.5 and later cannot
    re-create the CustomModel class from its configuration
    ("__init__() missing ... required positional arguments"), and the
    Lambda layer is stored as byte code of the Python version that
    saved it.

    What is stored is nevertheless simple and the same in every
    version: the list of layers with their settings and connections
    (JSON), and the weights by layer name. This module reads both
    directly from the files, builds the same graph from layer classes
    and sets the weights by name. The result is a plain keras model for
    prediction (model.predict, model(x)); it holds none of the training
    settings of CustomModel.

    Reading the layer graph and the weights needs neither TensorFlow
    nor keras (the variables of a SavedModel are read by
    bpnet.model.checkpoint); building the keras model does.
"""

import json
import os
import re

import numpy as np


class UnsupportedModelError(ValueError):
    """
        The saved model has a layer or a setting that the models of
        this package do not have
    """


# --------------------------------------------------------------------
# reading the layer graph
# --------------------------------------------------------------------

def _varint(buffer, position):
    """
        A base 128 varint of the protocol buffer wire format

        Returns:
            tuple: (value, position after it)
    """

    result = 0
    shift = 0
    # (a number of 64 bits takes at most ten bytes)
    for _ in range(10):
        if position >= len(buffer):
            raise ValueError("Truncated protocol buffer message")
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            # (64 bits: what the tenth byte has beyond them is dropped,
            # as protocol buffers do)
            return result & 0xFFFFFFFFFFFFFFFF, position
        shift += 7
    raise ValueError(
        "Not a protocol buffer message (a number of more than ten bytes)")


#: the kinds of field of the protocol buffer wire format ("wire types")
_VARINT, _FIXED64, _BYTES, _FIXED32 = 0, 1, 2, 5


def _int32(value):
    """
        A varint as the int32 field it is (the low 32 bits, signed)
    """

    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >> 31 else value


def _int64(value):
    """
        A varint as the int64 field it is (signed)
    """

    return value - (1 << 64) if value >> 63 else value


def _fields(buffer):
    """
        The fields of a serialized protocol buffer message, in order,
        with their kind. A reader that knows the message takes a
        field only when it has the kind the message gives it, and
        passes over it otherwise, as protocol buffers do (a fixed
        width field of eight bytes is not a message of eight bytes).

        Yields:
            tuple: (field number, kind, value); the value is an int
                for a varint field (_VARINT) and a memoryview for a
                length delimited (_BYTES) or fixed width field
                (_FIXED64, _FIXED32)
    """

    position, end = 0, len(buffer)
    while position < end:
        key, position = _varint(buffer, position)
        number, wire_type = key >> 3, key & 7
        if key >> 32 or number == 0:
            raise ValueError(
                "Not a protocol buffer message (field number {})".format(
                    number))
        if wire_type == 0:
            value, position = _varint(buffer, position)
        elif wire_type == 2:
            length, position = _varint(buffer, position)
            value = buffer[position:position + length]
            position += length
        elif wire_type in (1, 5):
            length = 8 if wire_type == 1 else 4
            value = buffer[position:position + length]
            position += length
        else:
            raise ValueError(
                "Not a protocol buffer message (wire type {})".format(
                    wire_type))
        if position > end:
            raise ValueError("Truncated protocol buffer message")
        yield number, wire_type, value


def _first(buffer, number):
    """
        The first length delimited field (a message, a string) with
        the given number, None if there is none
    """

    for field_number, kind, value in _fields(buffer):
        if field_number == number and kind == _BYTES:
            return value
    return None


def _read(path):
    with open(path, "rb") as fh:
        return memoryview(fh.read())


def _metadata_in_saved_model_pb(data):
    """
        The JSON keras stores for the model itself inside
        saved_model.pb (TensorFlow up to 2.4), None if it is not there

        SavedModel.meta_graphs[0].object_graph_def.nodes[0]
            .user_object.metadata
    """

    meta_graph = _first(data, 2)
    object_graph = None if meta_graph is None else _first(meta_graph, 7)
    root = None if object_graph is None else _first(object_graph, 1)
    user_object = None if root is None else _first(root, 4)
    metadata = None if user_object is None else _first(user_object, 3)
    if metadata is None or len(metadata) == 0:
        return None
    return bytes(metadata).decode("utf-8")


def _metadata_in_keras_metadata_pb(data):
    """
        The JSON keras stores for the model itself in keras_metadata.pb
        (TensorFlow 2.5 and later), None if it is not there

        SavedMetadata.nodes[i].metadata of the node whose path is
        'root'
    """

    for number, kind, node in _fields(data):
        if number != 1 or kind != _BYTES:
            continue
        node_path, metadata = None, None
        for field_number, field_kind, value in _fields(node):
            if field_kind != _BYTES:
                continue
            if field_number == 3:
                node_path = bytes(value).decode("utf-8")
            elif field_number == 5:
                metadata = bytes(value).decode("utf-8")
        if node_path == "root":
            return metadata
    return None


def _plain(value):
    """
        keras wraps tuples and shapes in the JSON it stores with a
        SavedModel ({"class_name": "__tuple__", "items": [...]});
        return the value with these replaced by lists
    """

    if isinstance(value, dict):
        if set(value) == {"class_name", "items"} and \
                value["class_name"] in ("__tuple__", "TensorShape"):
            return [_plain(item) for item in value["items"]]
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def is_saved_model_dir(path):
    return os.path.isfile(os.path.join(path, "saved_model.pb"))


def read_architecture(path):
    """
        The layer graph stored with a saved model

        Args:
            path (str): a SavedModel directory, or an HDF5 file written
                by model.save()

        Returns:
            dict: keras' functional configuration with plain lists and
                dicts: 'name', 'layers' (each with 'class_name',
                'name', 'config' and 'inbound_nodes'), 'input_layers',
                'output_layers'
    """

    if os.path.isdir(path):
        if not is_saved_model_dir(path):
            raise FileNotFoundError(
                "{} is not a saved model: it has no saved_model.pb".format(
                    path))
        metadata = None
        keras_metadata = os.path.join(path, "keras_metadata.pb")
        if os.path.isfile(keras_metadata):
            metadata = _metadata_in_keras_metadata_pb(_read(keras_metadata))
        if metadata is None:
            metadata = _metadata_in_saved_model_pb(
                _read(os.path.join(path, "saved_model.pb")))
        if metadata is None:
            raise UnsupportedModelError(
                "{} does not hold a keras model (no keras metadata "
                "found)".format(path))
    elif os.path.isfile(path):
        import h5py

        with h5py.File(path, "r") as f:
            if "model_config" not in f.attrs:
                raise UnsupportedModelError(
                    "{} has no model configuration; a file written by "
                    "model.save() is needed, not one written by "
                    "model.save_weights()".format(path))
            metadata = f.attrs["model_config"]
        if isinstance(metadata, bytes):
            metadata = metadata.decode("utf-8")
    else:
        raise FileNotFoundError("Model not found: {}".format(path))

    model = _plain(json.loads(metadata))
    config = model.get("config")
    if not isinstance(config, dict) or "layers" not in config:
        raise UnsupportedModelError(
            "{} holds a {} model without a layer graph".format(
                path, model.get("class_name")))
    return config


# --------------------------------------------------------------------
# reading the weights
# --------------------------------------------------------------------

def _weights_in_saved_model(path):
    """
        The variables of every layer in a SavedModel's checkpoint

        The checkpoint (read without TensorFlow, see
        bpnet.model.checkpoint) records the objects of the model as a
        graph:
        the model refers to its layers with weights
        ('layer_with_weights-<i>'), each of which refers to its
        variables ('kernel', 'bias'). The graph is followed from the
        model, so that the variables of the optimizer and of the
        metrics are left out, and each variable is filed under the name
        it had in the model ('<layer>/kernel').
    """

    from bpnet.model.checkpoint import Checkpoint, CorruptCheckpointError

    reader = Checkpoint(os.path.join(path, "variables", "variables"))
    if "_CHECKPOINTABLE_OBJECT_GRAPH" not in reader:
        raise CorruptCheckpointError(
            "{} does not say which variables belong to which layer (it "
            "has no object graph)".format(reader.index_path))
    object_graph = reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH")
    if not isinstance(object_graph, bytes):
        raise CorruptCheckpointError(
            "{}: its object graph is not a message".format(
                reader.index_path))
    try:
        variables = _variables_in_object_graph(memoryview(object_graph))
    except ValueError as error:
        raise CorruptCheckpointError(
            "{}: its object graph cannot be read ({})".format(
                reader.index_path, error))

    found = {}
    for variable_name, full_name, checkpoint_key in variables:
        if checkpoint_key not in reader:
            raise CorruptCheckpointError(
                "{} has no tensor {}, which its object graph names "
                "for {}".format(reader.index_path, checkpoint_key,
                                full_name))
        value = reader.get_tensor(checkpoint_key)
        if isinstance(value, bytes) or value.dtype == object:
            raise UnsupportedModelError(
                "{}: variable {} holds text, not numbers".format(
                    reader.index_path, full_name))
        layer = full_name.rsplit("/", 1)[0]
        found.setdefault(layer, {})[variable_name] = value
    return found


def _variables_in_object_graph(object_graph):
    """
        The variables of the layers with weights in the object graph
        of a checkpoint (TrackableObjectGraph): its first object is
        the model, which refers to its layers with weights, each of
        which refers to its variables

        Returns:
            list: for each variable (its name in the layer, its full
                name, the name of its tensor in the checkpoint)

        Raises:
            ValueError: the graph is not what the format says
    """

    # TrackableObjectGraph.nodes
    nodes = [node for number, kind, node in _fields(object_graph)
             if number == 1 and kind == _BYTES]
    if not nodes:
        raise ValueError("it has no objects")

    def children(node):
        # TrackableObject.children[i]: node_id, local_name
        found = []
        for number, kind, reference in _fields(node):
            if number != 1 or kind != _BYTES:
                continue
            node_id, local_name = 0, ""
            for field_number, field_kind, value in _fields(reference):
                if field_number == 1 and field_kind == _VARINT:
                    node_id = _int32(value)
                elif field_number == 2 and field_kind == _BYTES:
                    local_name = bytes(value).decode("utf-8")
            if not 0 <= node_id < len(nodes):
                raise ValueError(
                    "an object refers to object {} of {}".format(
                        node_id, len(nodes)))
            found.append((local_name, node_id))
        return found

    def variable(node):
        # TrackableObject.attributes[i]: name, full_name, checkpoint_key
        for number, kind, attribute in _fields(node):
            if number != 2 or kind != _BYTES:
                continue
            values = {n: bytes(v).decode("utf-8")
                      for n, k, v in _fields(attribute)
                      if n in (1, 2, 3) and k == _BYTES}
            if values.get(1) == "VARIABLE_VALUE":
                if not values.get(2) or not values.get(3):
                    raise ValueError(
                        "a variable without its name ({!r}) or without "
                        "the name of its tensor ({!r})".format(
                            values.get(2, ""), values.get(3, "")))
                return values[2], values[3]
        return None

    found = []
    for local_name, node_id in children(nodes[0]):
        if not local_name.startswith("layer_with_weights-"):
            continue
        for variable_name, variable_id in children(nodes[node_id]):
            names = variable(nodes[variable_id])
            if names is not None:
                found.append((variable_name,) + names)
    return found


def _weights_in_hdf5(path):
    import h5py

    found = {}
    with h5py.File(path, "r") as f:
        group = f["model_weights"]
        for layer in group:
            names = group[layer].attrs.get("weight_names", [])
            for name in names:
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                # '<layer>/kernel:0'
                variable = name.split("/")[-1].split(":")[0]
                found.setdefault(layer, {})[variable] = \
                    np.asarray(group[layer][name])
    return found


def read_weights(path):
    """
        The weights of a saved model by layer name

        Args:
            path (str): a SavedModel directory, or an HDF5 file written
                by model.save()

        Returns:
            dict: layer name -> {variable name: numpy array}, the
                variable names being 'kernel' and 'bias'
    """

    if os.path.isdir(path):
        if not is_saved_model_dir(path):
            raise FileNotFoundError(
                "{} is not a saved model: it has no saved_model.pb".format(
                    path))
        return _weights_in_saved_model(path)
    if os.path.isfile(path):
        return _weights_in_hdf5(path)
    raise FileNotFoundError("Model not found: {}".format(path))


# --------------------------------------------------------------------
# building the graph
# --------------------------------------------------------------------

def logsumexp_over_tracks(x):
    """
        The function of the '<prefix>_logsumexp_counts_bias_<task>'
        Lambda layer: log of the summed control counts of a task
    """

    import tensorflow as tf

    return tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)


def _scalar(value):
    """
        A setting keras stores as a one element tuple or as a number
    """

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise UnsupportedModelError(
                "one value expected, found {}".format(value))
        return value[0]
    return value


def _expect(config, **expected):
    """
        Settings that the layers of this package never change must
        have their usual value (or be absent, as in files written by
        versions that did not have the setting yet)
    """

    for key, value in expected.items():
        if key not in config:
            continue
        found = config[key]
        # keras stores some single settings as a tuple of one
        if isinstance(found, (list, tuple)) and len(found) == 1 and \
                not isinstance(value, (list, tuple)):
            found = found[0]
        if isinstance(value, float) and found is not None:
            same = float(found) == value
        else:
            same = found == value
        if not same:
            raise UnsupportedModelError(
                "Layer {}: {} = {} is not supported (expected {})".format(
                    config.get("name"), key, config[key], value))


def _dtype(config):
    """
        The type a layer was saved with: a name such as 'float32', or
        a mixed precision policy. A layer that is built without it
        would take whatever the process that loads the model has set
        as its default.

        Returns:
            str or tensorflow.keras.mixed_precision.Policy: None when
                the file does not say
    """

    stored = config.get("dtype")
    if stored is None or isinstance(stored, str):
        return stored
    if isinstance(stored, dict) and \
            stored.get("class_name") in ("Policy", "DTypePolicy") and \
            isinstance(stored.get("config", {}).get("name"), str):
        import tensorflow as tf

        return tf.keras.mixed_precision.Policy(stored["config"]["name"])
    raise UnsupportedModelError(
        "Layer {}: dtype {} is not supported".format(
            config.get("name"), stored))


def _activation(config):
    activation = config.get("activation", "linear")
    if activation is not None and not isinstance(activation, str):
        raise UnsupportedModelError(
            "Layer {}: activation {} is not supported".format(
                config.get("name"), activation))
    return activation


def _make_layer(class_name, config):
    """
        The keras layer of one entry of the layer graph
    """

    from tensorflow.keras import layers

    # every layer gets its name and the type it was saved with
    common = {"name": config["name"], "dtype": _dtype(config)}
    name = config["name"]

    if class_name == "Conv1D":
        _expect(config, strides=1, data_format="channels_last", groups=1)
        return layers.Conv1D(
            filters=config["filters"],
            kernel_size=_scalar(config["kernel_size"]),
            padding=config["padding"],
            dilation_rate=_scalar(config.get("dilation_rate", 1)),
            activation=_activation(config),
            use_bias=config.get("use_bias", True), **common)

    if class_name == "Dense":
        return layers.Dense(
            units=config["units"], activation=_activation(config),
            use_bias=config.get("use_bias", True), **common)

    if class_name == "ReLU":
        _expect(config, max_value=None, negative_slope=0.0, threshold=0.0)
        return layers.ReLU(**common)

    if class_name == "LeakyReLU":
        alpha = config.get("alpha", config.get("negative_slope", 0.3))
        return layers.LeakyReLU(alpha=float(alpha), **common)

    if class_name == "Cropping1D":
        cropping = config["cropping"]
        if not isinstance(cropping, (list, tuple)):
            cropping = (cropping, cropping)
        return layers.Cropping1D(
            cropping=(int(cropping[0]), int(cropping[1])), **common)

    if class_name == "Add":
        return layers.Add(**common)

    if class_name == "Concatenate":
        return layers.Concatenate(axis=config.get("axis", -1), **common)

    if class_name == "GlobalAveragePooling1D":
        _expect(config, data_format="channels_last", keepdims=False)
        return layers.GlobalAveragePooling1D(**common)

    if class_name == "Flatten":
        _expect(config, data_format="channels_last")
        return layers.Flatten(**common)

    if class_name == "Dropout":
        _expect(config, noise_shape=None)
        return layers.Dropout(float(config["rate"]), **common)

    if class_name == "Lambda":
        # the function itself is stored as byte code of the Python
        # version that saved the model; the layers of this package are
        # recognized by their names instead
        if re.match(r"^.*_logsumexp_counts_bias_\d+$", name):
            return layers.Lambda(logsumexp_over_tracks, **common)
        raise UnsupportedModelError(
            "Lambda layer {} is not supported (models with more than "
            "one task cannot be loaded yet)".format(name))

    raise UnsupportedModelError(
        "Layer {} has class {}, which the models of this package do "
        "not use".format(name, class_name))


def _input_names(layer):
    """
        Names of the layers whose outputs a layer takes, in order
    """

    nodes = layer.get("inbound_nodes", [])
    if not nodes:
        return []
    if len(nodes) != 1:
        raise UnsupportedModelError(
            "Layer {} is applied {} times; shared layers are not "
            "supported".format(layer["name"], len(nodes)))
    names = []
    for entry in nodes[0]:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3 or \
                not isinstance(entry[0], str):
            raise UnsupportedModelError(
                "Layer {}: the format of its connections is not "
                "supported".format(layer["name"]))
        if entry[1] != 0 or entry[2] != 0:
            raise UnsupportedModelError(
                "Layer {} takes output {} of application {} of layer {}; "
                "only single outputs are supported".format(
                    layer["name"], entry[2], entry[1], entry[0]))
        names.append(entry[0])
    return names


def _endpoint_names(entries, what):
    names = []
    if not isinstance(entries, (list, tuple)):
        raise UnsupportedModelError(
            "The {} of the model are not a list".format(what))
    # a single endpoint may be stored without the enclosing list
    if entries and isinstance(entries[0], str):
        entries = [entries]
    for entry in entries:
        if entry[1] != 0 or entry[2] != 0:
            raise UnsupportedModelError(
                "The {} of the model use output {} of application {} of "
                "layer {}; only single outputs are supported".format(
                    what, entry[2], entry[1], entry[0]))
        names.append(entry[0])
    return names


def build_model(architecture):
    """
        A keras model with the layer graph read by read_architecture;
        its weights are freshly initialized

        Args:
            architecture (dict): see read_architecture

        Returns:
            tensorflow.keras.Model
    """

    import tensorflow as tf
    from tensorflow.keras import layers

    pending = list(architecture["layers"])
    names = [layer["name"] for layer in pending]
    if len(set(names)) != len(names):
        raise UnsupportedModelError("Layer names are not unique")

    tensors = {}
    while pending:
        remaining = []
        for layer in pending:
            inputs = _input_names(layer)
            if any(name not in tensors for name in inputs):
                remaining.append(layer)
                continue
            config = layer["config"]
            if layer["class_name"] == "InputLayer":
                shape = config.get("batch_input_shape",
                                   config.get("batch_shape"))
                _expect(config, sparse=False, ragged=False)
                tensors[layer["name"]] = layers.Input(
                    shape=tuple(shape[1:]), name=layer["name"],
                    dtype=config.get("dtype", "float32"))
                continue
            if not inputs:
                raise UnsupportedModelError(
                    "Layer {} has no input".format(layer["name"]))
            made = _make_layer(layer["class_name"], config)
            if layer["class_name"] in ("Add", "Concatenate"):
                tensors[layer["name"]] = made(
                    [tensors[name] for name in inputs])
            else:
                if len(inputs) != 1:
                    raise UnsupportedModelError(
                        "Layer {} takes {} inputs".format(
                            layer["name"], len(inputs)))
                tensors[layer["name"]] = made(tensors[inputs[0]])
        if len(remaining) == len(pending):
            raise UnsupportedModelError(
                "Layers {} take the output of layers that do not "
                "exist".format([layer["name"] for layer in remaining]))
        pending = remaining

    inputs = [tensors[name] for name in _endpoint_names(
        architecture["input_layers"], "inputs")]
    outputs = [tensors[name] for name in _endpoint_names(
        architecture["output_layers"], "outputs")]
    return tf.keras.Model(
        inputs=inputs, outputs=outputs if len(outputs) > 1 else outputs[0],
        name=architecture.get("name"))


def set_weights(model, weights):
    """
        Give every layer of the model its saved weights

        Every layer with weights must find its kernel (and bias) under
        its name, with the shape the layer has, and no saved weights
        may be left over.

        Args:
            model (tensorflow.keras.Model): from build_model
            weights (dict): from read_weights
    """

    used = set()
    for layer in model.layers:
        variables = layer.weights
        if not variables:
            continue
        if layer.name not in weights:
            raise UnsupportedModelError(
                "No saved weights for layer {}".format(layer.name))
        saved = weights[layer.name]
        values = []
        for variable in variables:
            # 'main_conv_0/kernel:0'
            key = variable.name.split("/")[-1].split(":")[0]
            if key not in saved:
                raise UnsupportedModelError(
                    "No saved {} for layer {}".format(key, layer.name))
            value = np.asarray(saved[key])
            if tuple(value.shape) != tuple(variable.shape):
                raise UnsupportedModelError(
                    "Layer {}: the saved {} has shape {}, the layer "
                    "needs {}".format(layer.name, key, tuple(value.shape),
                                      tuple(variable.shape)))
            if value.dtype != variable.dtype.as_numpy_dtype:
                raise UnsupportedModelError(
                    "Layer {}: the saved {} is {}, the layer's is "
                    "{}".format(layer.name, key, value.dtype,
                                variable.dtype.name))
            values.append(value)
        if len(saved) != len(values):
            raise UnsupportedModelError(
                "Layer {}: saved variables {} are not all used".format(
                    layer.name, sorted(saved)))
        layer.set_weights(values)
        used.add(layer.name)

    unused = sorted(set(weights) - used)
    if unused:
        raise UnsupportedModelError(
            "Saved weights of layers {} have no layer in the "
            "model".format(unused))


def load_model(path):
    """
        Load a saved model for prediction

        Args:
            path (str): a SavedModel directory, or an HDF5 file written
                by model.save(), from any TensorFlow version this
                package has run on

        Returns:
            tensorflow.keras.Model: takes the inputs by name
                ('sequence', 'profile_bias_input_0',
                'counts_bias_input_0') and returns the profile logits
                and the log counts
    """

    model = build_model(read_architecture(path))
    set_weights(model, read_weights(path))
    return model
