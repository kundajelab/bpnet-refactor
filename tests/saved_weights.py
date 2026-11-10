"""
    Read the weights of a saved model by layer name, without building
    or loading a Keras model
"""

import os


def weights_by_layer_name(saved_model_dir):
    """
        Kernel and bias of every layer of a model saved in TensorFlow's
        SavedModel format, read from the variables checkpoint. Names
        come from the checkpoint's object graph (the name each variable
        had in the model), not from the order of the entries.

        Args:
            saved_model_dir (str): directory written by model.save()

        Returns:
            dict: layer name -> [kernel, bias] as numpy arrays
    """

    import tensorflow as tf
    from tensorflow.core.protobuf import trackable_object_graph_pb2

    reader = tf.train.load_checkpoint(
        os.path.join(saved_model_dir, "variables", "variables"))
    graph = trackable_object_graph_pb2.TrackableObjectGraph()
    graph.ParseFromString(reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH"))

    found = {}
    for node in graph.nodes:
        for attribute in node.attributes:
            if attribute.name != "VARIABLE_VALUE":
                continue
            parts = attribute.full_name.split("/")
            # '<layer>/kernel' and '<layer>/bias'; optimizer slots and
            # metric variables have other shapes of name
            if len(parts) == 2 and parts[1] in ("kernel", "bias"):
                found.setdefault(parts[0], {})[parts[1]] = \
                    reader.get_tensor(attribute.checkpoint_key)
    return {layer: [w["kernel"], w["bias"]] for layer, w in found.items()}
