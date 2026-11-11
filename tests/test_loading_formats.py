"""
    Tests for the file reading of bpnet.model.loading that need no
    TensorFlow: the protocol buffer fields, where keras keeps the layer
    graph in a SavedModel directory, and what is not a saved model.
    Loading whole models is tested in test_loading.py.
"""

import json
import os

import pytest

from bpnet.model import loading


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def field(number, value):
    """
        One field of a protocol buffer message: an int is written as a
        varint, bytes as a length delimited field
    """

    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    return varint((number << 3) | 2) + varint(len(value)) + value


def fixed32(number, data):
    """
        A field of four bytes
    """

    assert len(data) == 4
    return varint((number << 3) | 5) + data


def fixed64(number, data):
    """
        A field of eight bytes
    """

    assert len(data) == 8
    return varint((number << 3) | 1) + data


def test_fields_of_a_message():
    message = field(1, 5) + field(2, b"abc") + field(300, 2 ** 40 + 3) + \
        field(7, b"") + varint((4 << 3) | 5) + b"\x01\x02\x03\x04" + \
        varint((5 << 3) | 1) + bytes(range(8)) + field(2, b"x" * 300)
    fields = [(n, k, v if isinstance(v, int) else bytes(v))
              for n, k, v in loading._fields(memoryview(message))]
    # with their kinds: 0 a number, 2 length delimited, 5 and 1 of
    # fixed width (four and eight bytes)
    assert fields == [
        (1, 0, 5), (2, 2, b"abc"), (300, 0, 2 ** 40 + 3), (7, 2, b""),
        (4, 5, b"\x01\x02\x03\x04"), (5, 1, bytes(range(8))),
        (2, 2, b"x" * 300)]
    assert (loading._VARINT, loading._FIXED64, loading._BYTES,
            loading._FIXED32) == (0, 1, 2, 5)
    assert bytes(loading._first(memoryview(message), 2)) == b"abc"
    assert loading._first(memoryview(message), 9) is None
    # a field of another kind is not the message that is looked for:
    # eight bytes of fixed width that read as a message, a number
    other = varint((2 << 3) | 1) + b"\x12\x02\x08\x01\x18\x00\x18\x00" + \
        field(2, 7) + field(2, b"abc")
    assert bytes(loading._first(memoryview(other), 2)) == b"abc"
    assert loading._first(memoryview(other[:-5]), 2) is None
    # the largest field number
    largest = varint((2 ** 29 - 1) << 3) + b"\x01"
    assert list(loading._fields(memoryview(largest))) == [(2 ** 29 - 1, 0, 1)]


def test_numbers_of_a_message():
    """
        A number of a message takes at most ten bytes, of which 64
        bits count (what the tenth byte has beyond them is dropped, as
        protocol buffers do); a field of 32 bits keeps the low 32, and
        the fields of both widths are signed
    """

    assert loading._varint(b"x\xac\x02", 1) == (300, 3)
    assert loading._varint(b"\xff" * 9 + b"\x01", 0) == (2 ** 64 - 1, 10)
    assert loading._varint(b"\xff" * 9 + b"\x7f", 0) == (2 ** 64 - 1, 10)
    assert loading._varint(b"\x81" + b"\x80" * 8 + b"\x7e", 0) == (1, 10)
    with pytest.raises(ValueError, match="more than ten bytes"):
        loading._varint(b"\xff" * 10 + b"\x01", 0)
    for value, as_32, as_64 in (
            (0, 0, 0), (5, 5, 5), (2 ** 31 - 1, 2 ** 31 - 1, 2 ** 31 - 1),
            (2 ** 31, -2 ** 31, 2 ** 31), (2 ** 32 - 1, -1, 2 ** 32 - 1),
            (2 ** 32 + 1, 1, 2 ** 32 + 1), (2 ** 63 - 1, -1, 2 ** 63 - 1),
            (2 ** 63, 0, -2 ** 63), (2 ** 64 - 1, -1, -1),
            (2 ** 64 - 2 ** 31, -2 ** 31, -2 ** 31)):
        assert loading._int32(value) == as_32
        assert loading._int64(value) == as_64


def test_truncated_or_foreign_data_is_not_read_as_a_message():
    with pytest.raises(ValueError, match="Truncated"):
        list(loading._fields(memoryview(field(2, b"abcdef")[:-2])))
    # a number that does not end
    for data in (b"\x80", b"\x08\xff\xff", field(2, b"abc")[:1] + b"\x80"):
        with pytest.raises(ValueError, match="Truncated"):
            list(loading._fields(memoryview(data)))
    # fields of fixed width that are cut short
    for wire_type, length in ((5, 4), (1, 8)):
        with pytest.raises(ValueError, match="Truncated"):
            list(loading._fields(memoryview(
                varint((1 << 3) | wire_type) + bytes(length - 1))))
    # wire types 3 and 4 (groups) are not used by these files; 6 and 7
    # do not exist
    for wire_type in (3, 4, 6, 7):
        with pytest.raises(ValueError, match="wire type"):
            list(loading._fields(memoryview(varint((1 << 3) | wire_type))))
    # there is no field number 0, and none of more than 29 bits (a
    # field number with its kind has 32)
    for data in (b"\x00\x01", field(0, b"abc"),
                 varint((2 ** 29) << 3) + b"\x01",
                 varint(2 ** 32 + (15 << 3)) + b"\x03"):
        with pytest.raises(ValueError, match="field number"):
            list(loading._fields(memoryview(data)))
    # a field number with its kind, and a length, are written in as
    # few bytes as they need (the readers of protocol buffers do not
    # agree on other ways of writing them)
    for data in (b"\x88\x00\x01", b"\x88\x80\x80\x80\x80\x00\x01",
                 b"\x88" + b"\x80" * 8 + b"\x02\x01",
                 b"\x12\x83\x00abc", b"\x12\x83\x80\x80\x80\x80\x00abc"):
        with pytest.raises(ValueError, match="in more bytes than it needs"):
            list(loading._fields(memoryview(data)))
    assert [(n, k, bytes(v)) for n, k, v in loading._fields(memoryview(
        b"\x12\x80\x01" + bytes(128)))] == [(2, 2, bytes(128))]
    # a field has less than 2 GB
    for length, message in ((2 ** 31 - 16, "a field of 2147483632 bytes"),
                            (2 ** 31 - 17, "Truncated")):
        with pytest.raises(ValueError, match=message):
            list(loading._fields(memoryview(b"\x12" + varint(length))))


def keras_json(layers=()):
    return json.dumps({
        "class_name": "CustomModel",
        "config": {"name": "model", "layers": list(layers),
                   "input_layers": [], "output_layers": []}})


def test_metadata_inside_saved_model_pb():
    # SavedModel.meta_graphs[0].object_graph_def.nodes[0].user_object
    # .metadata, among other fields and nodes
    user_object = field(1, b"_tf_keras_network") + field(3, b'{"a": 1}')
    root = field(1, b"child") + field(4, user_object)
    other = field(4, field(1, b"_tf_keras_layer") + field(3, b'{"b": 2}'))
    object_graph = field(1, root) + field(1, other) + field(2, b"functions")
    meta_graph = field(1, b"info") + field(2, b"graph") + \
        field(7, object_graph)
    saved_model = field(1, 1) + field(2, meta_graph)
    assert loading._metadata_in_saved_model_pb(
        memoryview(saved_model)) == '{"a": 1}'

    # TensorFlow 2.5 and later leave the metadata out of this file
    root = field(4, field(1, b"_tf_keras_network"))
    saved_model = field(2, field(7, field(1, root)))
    assert loading._metadata_in_saved_model_pb(
        memoryview(saved_model)) is None
    assert loading._metadata_in_saved_model_pb(memoryview(b"")) is None


def test_metadata_inside_keras_metadata_pb():
    layer = field(2, 3) + field(3, b"root.layer-0") + \
        field(4, b"_tf_keras_layer") + field(5, b'{"layer": 0}')
    # the node id of the root is 0, which is not written
    root = field(3, b"root") + field(4, b"_tf_keras_network") + \
        field(5, b'{"model": 1}')
    data = field(1, layer) + field(1, root)
    assert loading._metadata_in_keras_metadata_pb(
        memoryview(data)) == '{"model": 1}'
    assert loading._metadata_in_keras_metadata_pb(
        memoryview(field(1, layer))) is None


def test_plain_values():
    wrapped = {"kernel_size": {"class_name": "__tuple__", "items": [21]},
               "shape": {"class_name": "TensorShape", "items": [None, 4]},
               "nested": [{"class_name": "__tuple__", "items": [
                   {"class_name": "__tuple__", "items": [1, 2]}, 3]}],
               "initializer": {"class_name": "Zeros", "config": {}},
               "name": "x"}
    assert loading._plain(wrapped) == {
        "kernel_size": [21], "shape": [None, 4], "nested": [[[1, 2], 3]],
        "initializer": {"class_name": "Zeros", "config": {}}, "name": "x"}


def write_saved_model_dir(path, metadata, keras_metadata_file):
    os.makedirs(str(path))
    if keras_metadata_file:
        root = field(4, field(1, b"_tf_keras_network"))
        node = field(3, b"root") + field(5, metadata.encode())
        with open(str(path / "keras_metadata.pb"), "wb") as fh:
            fh.write(field(1, node))
    else:
        root = field(4, field(1, b"_tf_keras_network")
                     + field(3, metadata.encode()))
    with open(str(path / "saved_model.pb"), "wb") as fh:
        fh.write(field(2, field(7, field(1, root))))


@pytest.mark.parametrize("keras_metadata_file", [False, True])
def test_read_architecture_of_a_saved_model_directory(tmp_path,
                                                      keras_metadata_file):
    layers = [{"class_name": "InputLayer", "name": "sequence", "config": {
        "batch_input_shape": {"class_name": "__tuple__",
                              "items": [None, 10, 4]}}, "inbound_nodes": []}]
    write_saved_model_dir(tmp_path / "model", keras_json(layers),
                          keras_metadata_file)
    architecture = loading.read_architecture(str(tmp_path / "model"))
    assert architecture["name"] == "model"
    assert architecture["layers"][0]["config"]["batch_input_shape"] == \
        [None, 10, 4]


def test_paths_that_are_not_saved_models(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        loading.read_architecture(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="Model not found"):
        loading.read_weights(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no saved_model.pb"):
        loading.read_architecture(str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="no saved_model.pb"):
        loading.read_weights(str(tmp_path / "empty"))
    # a SavedModel that keras did not write
    write_saved_model_dir(tmp_path / "plain", "", False)
    with pytest.raises(loading.UnsupportedModelError,
                       match="no keras metadata"):
        loading.read_architecture(str(tmp_path / "plain"))
    # keras metadata of something that has no layer graph
    write_saved_model_dir(tmp_path / "subclassed", json.dumps(
        {"class_name": "MyModel", "config": {}}), True)
    with pytest.raises(loading.UnsupportedModelError,
                       match="MyModel model without a layer graph"):
        loading.read_architecture(str(tmp_path / "subclassed"))


def test_hdf5_files_that_do_not_hold_weights(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "other.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1, 2])
    with pytest.raises(loading.UnsupportedModelError,
                       match="it has no 'model_weights'"):
        loading.read_weights(path)
    # a layer that lists weights it does not have
    with h5py.File(path, "w") as f:
        layer = f.create_group("model_weights").create_group("conv")
        layer.attrs["weight_names"] = [b"conv/kernel:0", b"conv/bias:0"]
        layer.create_group("conv").create_dataset("kernel:0", data=[1.0])
    with pytest.raises(ValueError) as failure:
        loading.read_weights(path)
    assert "layer conv lists the weights conv/bias:0 and does not have " \
        "them" in str(failure.value)
    with h5py.File(path, "a") as f:
        f["model_weights/conv/conv"].create_dataset("bias:0", data=[2.0])
    weights = loading.read_weights(path)
    assert sorted(weights) == ["conv"]
    assert {name: value.tolist() for name, value in
            weights["conv"].items()} == {"kernel": [1.0], "bias": [2.0]}
    # not an HDF5 file
    with open(path, "wb") as fh:
        fh.write(b"something else")
    with pytest.raises(OSError):
        loading.read_weights(path)
