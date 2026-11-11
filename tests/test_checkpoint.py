"""
    Tests for bpnet.model.checkpoint: the variables of a TensorFlow
    checkpoint read without TensorFlow. The files are written here in
    the documented format by a small writer of the tests' own, read
    from the models that three TensorFlow versions saved, and, where
    TensorFlow is installed, compared tensor for tensor with what its
    own reader returns.
"""

import hashlib
import json
import os
import shutil
import struct

import numpy as np
import pytest

from bpnet.model import checkpoint, loading
from bpnet.model.checkpoint import Checkpoint, CorruptCheckpointError
from bpnet.model.loading import UnsupportedModelError

from tests.model_helpers import (FIXTURE_MODELS, FIXTURE_VERSIONS,
                                 unpack_fixture_models)
from tests.test_loading_formats import field, varint


# --------------------------------------------------------------------
# the checksum
# --------------------------------------------------------------------

def crc_bit_by_bit(data, crc=0):
    """
        CRC-32C from its definition (polynomial 0x1EDC6F41, bits
        reflected), one bit at a time
    """

    crc ^= 0xFFFFFFFF
    for byte in bytes(data):
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def test_checksum_of_known_data():
    # the test vectors of RFC 3720 (iSCSI), section B.4
    assert checkpoint.crc32c(b"") == 0
    assert checkpoint.crc32c(b"123456789") == 0xE3069283
    assert checkpoint.crc32c(bytes(32)) == 0x8A9136AA
    assert checkpoint.crc32c(b"\xff" * 32) == 0x62A8AB43
    assert checkpoint.crc32c(bytes(range(32))) == 0x46DD794E
    assert checkpoint.crc32c(bytes(range(31, -1, -1))) == 0x113FDB5C
    # any bytes-like object
    assert checkpoint.crc32c(memoryview(b"123456789")) == 0xE3069283
    assert checkpoint.crc32c(np.arange(32, dtype=np.uint8)) == 0x46DD794E
    assert checkpoint.crc32c(bytearray(b"123456789")) == 0xE3069283
    # in pieces
    assert checkpoint.crc32c(b"6789", checkpoint.crc32c(b"12345")) == \
        0xE3069283


@pytest.mark.parametrize("length", [
    1, 255, 4097, checkpoint._SHORT - 1, checkpoint._SHORT,
    checkpoint._SHORT + 1, 3 * checkpoint._SHORT + 1001,
    checkpoint._LANES * checkpoint._PIECE + 12345])
def test_checksum_of_long_data(length):
    """
        Long data is checked in pieces that are advanced together and
        joined: the result is that of the definition
    """

    rng = np.random.RandomState(length % 1000)
    data = rng.randint(0, 256, size=length).astype(np.uint8).tobytes()
    expected = crc_bit_by_bit(data) if length < 200000 else \
        checkpoint._register(0xFFFFFFFF, data) ^ 0xFFFFFFFF
    assert checkpoint.crc32c(data) == expected
    cut = length // 3
    assert checkpoint.crc32c(data[cut:], checkpoint.crc32c(data[:cut])) == \
        expected
    # one changed bit anywhere changes it
    for position in (0, length // 2, length - 1):
        changed = bytearray(data)
        changed[position] ^= 0x10
        assert checkpoint.crc32c(changed) != expected


def test_zero_bytes_operator():
    # what n zero bytes do to the register, for the n that cutting
    # long data into pieces can give
    for length in (1, 2, 3, 16, 17, 1000, 4096, 4097):
        tables = checkpoint._zeros_operator(length)
        for state in (1, 0x80000000, 0xDEADBEEF, 0xFFFFFFFF):
            expected = checkpoint._register(state, bytes(length))
            found = tables[0][state & 0xFF] ^ \
                tables[1][(state >> 8) & 0xFF] ^ \
                tables[2][(state >> 16) & 0xFF] ^ tables[3][state >> 24]
            assert found == expected


# --------------------------------------------------------------------
# a writer of tables and checkpoints, from the description of the
# formats
# --------------------------------------------------------------------

MAGIC = 0xDB4775248B80FB57


def masked(crc):
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def block(pairs, restart_interval=16):
    """
        A block of a table: entries (bytes shared with the key before,
        bytes not shared, length of the value, the rest of the key, the
        value), the offsets of the entries that share nothing, their
        number
    """

    out, restarts, last = bytearray(), [], b""
    for i, (key, value) in enumerate(pairs):
        shared = 0
        if i % restart_interval == 0:
            restarts.append(len(out))
        else:
            while shared < min(len(key), len(last)) and \
                    key[shared] == last[shared]:
                shared += 1
        out += varint(shared) + varint(len(key) - shared) + \
            varint(len(value)) + key[shared:] + value
        last = key
    for offset in restarts or [0]:
        out += struct.pack("<I", offset)
    out += struct.pack("<I", len(restarts or [0]))
    return bytes(out)


def with_trailer(contents, kind=0):
    return contents + bytes([kind]) + struct.pack(
        "<I", masked(crc_bit_by_bit(contents + bytes([kind]))))


def table(pairs, per_block=1000, kind=0, restart_interval=16):
    """
        A table file: blocks of sorted (key, value) pairs, a block of
        meta data (empty), a block that lists the blocks, the footer
    """

    data, handles = bytearray(), []
    for start in range(0, len(pairs), per_block):
        chunk = pairs[start:start + per_block]
        contents = block(chunk, restart_interval)
        handles.append((chunk[-1][0],
                        varint(len(data)) + varint(len(contents))))
        data += with_trailer(contents, kind)
    meta = block([])
    meta_handle = varint(len(data)) + varint(len(meta))
    data += with_trailer(meta)
    index = block(handles, 1)
    footer = meta_handle + varint(len(data)) + varint(len(index))
    data += with_trailer(index)
    footer += bytes(40 - len(footer)) + struct.pack("<Q", MAGIC)
    return bytes(data + footer)


#: numpy type -> tensorflow.DataType
TYPES = {"float32": 1, "float64": 2, "int32": 3, "uint8": 4, "int16": 5,
         "int8": 6, "int64": 9, "bool": 10, "uint16": 17, "float16": 19,
         "uint32": 22, "uint64": 23}


def tensor_bytes(value):
    """
        (tensorflow.DataType, shape, bytes in the data file, checksum)
        of a numpy array, or of a list / array of bytes objects
    """

    value = np.asarray(value)
    if value.dtype == object or value.dtype.kind == "S":
        strings = [bytes(s) for s in value.reshape(-1)]
        crc, out = 0, bytearray()
        for s in strings:
            out += varint(len(s))
            crc = crc_bit_by_bit(struct.pack("<I", len(s)), crc)
        lengths_checksum = struct.pack("<I", masked(crc))
        crc = crc_bit_by_bit(lengths_checksum + b"".join(strings), crc)
        return 7, value.shape, bytes(out) + lengths_checksum + \
            b"".join(strings), crc
    data = value.astype(value.dtype.newbyteorder("<")).tobytes()
    return TYPES[value.dtype.name], value.shape, data, crc_bit_by_bit(data)


def entry(dtype, shape, offset, size, crc, shard=0, extra=b""):
    """
        tensorflow.BundleEntryProto; like protocol buffers, fields
        that are zero are not written
    """

    dims = b"".join(field(2, field(1, n) if n else b"") for n in shape)
    out = field(1, dtype) + field(2, dims)
    for number, value in ((3, shard), (4, offset), (5, size)):
        if value:
            out += field(number, value)
    return out + varint((6 << 3) | 5) + struct.pack("<I", masked(crc)) + extra


def write_checkpoint(prefix, tensors, header=None, change=None, shards=1,
                     **table_settings):
    """
        Write the files of a checkpoint

        Args:
            prefix (str): path without '.index'
            tensors (dict): name -> array
            header (bytes): the header message (default: the number
                of shards, little endian)
            change (callable): (name, entry settings dict) -> None, to
                write other entries than the data ask for
            shards (int): number of data files; the tensors go to
                them in turn
    """

    data, pairs = [bytearray() for _ in range(shards)], []
    for i, name in enumerate(sorted(tensors)):
        dtype, shape, raw, crc = tensor_bytes(tensors[name])
        shard = i % shards
        settings = {"dtype": dtype, "shape": shape,
                    "offset": len(data[shard]), "size": len(raw),
                    "crc": crc, "shard": shard}
        if change is not None:
            change(name, settings)
        pairs.append((name.encode("utf-8"), entry(**settings)))
        data[shard] += raw
    header = field(1, shards) if header is None else header
    with open(prefix + ".index", "wb") as fh:
        fh.write(table([(b"", header)] + pairs, **table_settings))
    for shard in range(shards):
        with open("{}.data-{:05d}-of-{:05d}".format(prefix, shard, shards),
                  "wb") as fh:
            fh.write(bytes(data[shard]))
    return prefix


# --------------------------------------------------------------------
# tables
# --------------------------------------------------------------------

def some_pairs(n=700):
    rng = np.random.RandomState(0)
    keys = sorted({"layer_with_weights-{}/{}/.ATTRIBUTES/V{}".format(
        rng.randint(40), "kernel" if rng.rand() < 0.5 else "bias",
        rng.randint(10 ** 6)).encode() for _ in range(n)})
    keys = [b"", b"\x00\xff binary"] + keys
    return [(key, rng.randint(0, 256, size=rng.randint(0, 60)).astype(
        np.uint8).tobytes()) for key in keys]


@pytest.mark.parametrize("per_block,restart_interval", [
    (10 ** 6, 16), (50, 16), (1, 1), (7, 3), (300, 1000)])
def test_table_written_here(tmp_path, per_block, restart_interval):
    pairs = some_pairs()
    path = str(tmp_path / "table")
    with open(path, "wb") as fh:
        fh.write(table(pairs, per_block=per_block,
                       restart_interval=restart_interval))
    found = checkpoint.read_table(path)
    assert list(found.items()) == pairs


def test_empty_table(tmp_path):
    path = str(tmp_path / "table")
    with open(path, "wb") as fh:
        fh.write(table([]))
    assert checkpoint.read_table(path) == {}


def damaged(how):
    pairs = some_pairs(60)
    good = table(pairs, per_block=20)
    data = bytearray(good)
    if how == "byte of a block changed":
        data[30] ^= 0x01
    elif how == "byte of the list of blocks changed":
        data[len(data) - 48 - 10] ^= 0x40
    elif how == "checksum of a block changed":
        first = block(pairs[:20])
        data[len(first) + 2] ^= 0xFF
    elif how == "cut in the footer":
        data = data[:-5]
    elif how == "cut to a few bytes":
        data = data[:20]
    elif how == "footer only":
        data = data[-48:]
    elif how == "other end mark":
        data[-1] ^= 0xFF
    elif how == "block handle beyond the file":
        # the footer's handle of the list of blocks: a larger offset
        footer = varint(1) + varint(1) + varint(len(data)) + varint(50)
        data[-48:-8] = footer + bytes(40 - len(footer))
    elif how == "footer that is no handle":
        data[-48:-8] = b"\xff" * 40
    return bytes(data)


@pytest.mark.parametrize("how,message", [
    ("byte of a block changed", "does not have the checksum"),
    ("byte of the list of blocks changed", "does not have the checksum"),
    ("checksum of a block changed", "does not have the checksum"),
    ("cut in the footer", "is not a checkpoint index (or it is cut short)"),
    ("cut to a few bytes", "is too short for a checkpoint index"),
    ("footer only", "is cut short: a block of its table ends at byte"),
    ("other end mark", "is not a checkpoint index (or it is cut short)"),
    ("block handle beyond the file", "is cut short: a block of its table"),
    ("footer that is no handle", "the end of the table is not in the table "
     "format"),
])
def test_table_that_is_damaged(tmp_path, how, message):
    path = str(tmp_path / "table")
    with open(path, "wb") as fh:
        fh.write(damaged(how))
    with pytest.raises(CorruptCheckpointError) as failure:
        checkpoint.read_table(path)
    assert message in str(failure.value) and path in str(failure.value)


def test_blocks_that_are_not_in_the_format(tmp_path):
    """
        Blocks whose checksum is right and whose content is not a
        block: reported, not read beyond their end
    """

    path = str(tmp_path / "table")
    good = block([(b"abc", b"1"), (b"abd", b"2")])
    cases = {
        # more restart points than the block has bytes
        "restarts": good[:-4] + struct.pack("<I", 1000),
        # an entry that shares more than the key before has
        "shared": varint(5) + varint(1) + varint(1) + b"x1" +
        struct.pack("<II", 0, 1),
        # an entry longer than the block
        "length": varint(0) + varint(1) + varint(200) + b"x1" +
        struct.pack("<II", 0, 1),
        # a number that does not end
        "number": b"\xff\xff\xff" + struct.pack("<II", 0, 1),
        "short": b"\x01\x02",
    }
    for name, contents in cases.items():
        data = bytearray(with_trailer(contents))
        index = block([(b"z", varint(0) + varint(len(contents)))], 1)
        footer = varint(0) + varint(0) + varint(len(data)) + \
            varint(len(index))
        data += with_trailer(index)
        data += footer + bytes(40 - len(footer)) + struct.pack("<Q", MAGIC)
        with open(path, "wb") as fh:
            fh.write(bytes(data))
        with pytest.raises(CorruptCheckpointError) as failure:
            checkpoint.read_table(path)
        assert "table" in str(failure.value), name


def test_compressed_table_is_not_supported(tmp_path):
    path = str(tmp_path / "table")
    with open(path, "wb") as fh:
        fh.write(table(some_pairs(30), kind=1))
    with pytest.raises(UnsupportedModelError, match="is compressed"):
        checkpoint.read_table(path)


# --------------------------------------------------------------------
# checkpoints written here
# --------------------------------------------------------------------

def some_tensors():
    rng = np.random.RandomState(1)
    tensors = {
        "conv/kernel": rng.normal(size=(21, 4, 64)).astype(np.float32),
        "conv/bias": rng.normal(size=64).astype(np.float32),
        "double": rng.normal(size=(3, 5)),
        "half": rng.normal(size=(2, 2, 2)).astype(np.float16),
        "iterations": np.array(123456789012, dtype=np.int64),
        "flags": np.array([True, False, True]),
        "empty": np.zeros((0, 4), dtype=np.float32),
        "strings": np.array([b"", b"abc", b"\x00\xff" * 300, b"d"],
                            dtype=object).reshape(2, 2),
        "one string": np.array(b"a string of its own", dtype=object),
        # longer than what is checked byte by byte
        "large": rng.normal(size=(100, 70)).astype(np.float32),
    }
    for name in ("int32", "uint8", "int16", "int8", "uint16", "uint32",
                 "uint64"):
        tensors["types/" + name] = rng.randint(0, 100, size=(4, 3)).astype(
            name)
    return tensors


def assert_same_tensor(found, expected):
    expected = np.asarray(expected)
    if expected.dtype == object:
        if expected.shape == ():
            assert found == expected.item()
        else:
            assert found.shape == expected.shape and found.dtype == object
            assert found.tolist() == expected.tolist()
        return
    assert found.dtype == expected.dtype and found.shape == expected.shape
    assert found.dtype.isnative and found.flags["WRITEABLE"]
    assert found.tobytes() == expected.tobytes()


@pytest.mark.parametrize("per_block", [10 ** 6, 4])
def test_tensors_of_a_checkpoint_written_here(tmp_path, per_block):
    tensors = some_tensors()
    reader = Checkpoint(write_checkpoint(str(tmp_path / "ckpt"), tensors,
                                         per_block=per_block))
    assert reader.keys() == sorted(tensors)
    for name, expected in tensors.items():
        assert name in reader
        assert_same_tensor(reader.get_tensor(name), expected)
    assert "other" not in reader
    with pytest.raises(KeyError, match="has no tensor other"):
        reader.get_tensor("other")


def test_checkpoint_in_several_data_files(tmp_path):
    """
        A checkpoint may keep its tensors in more than one data file
        (the entry of a tensor says in which)
    """

    tensors = some_tensors()
    prefix = write_checkpoint(str(tmp_path / "ckpt"), tensors, shards=3)
    assert sorted(os.listdir(str(tmp_path))) == [
        "ckpt.data-00000-of-00003", "ckpt.data-00001-of-00003",
        "ckpt.data-00002-of-00003", "ckpt.index"]
    reader = Checkpoint(prefix)
    for name, expected in tensors.items():
        assert_same_tensor(reader.get_tensor(name), expected)
    # one of the files is missing: its tensors are not read, the
    # others are
    os.remove(prefix + ".data-00001-of-00003")
    names = sorted(tensors)
    with pytest.raises(FileNotFoundError, match="data-00001-of-00003"):
        reader.get_tensor(names[1])
    assert_same_tensor(reader.get_tensor(names[0]), tensors[names[0]])
    assert_same_tensor(reader.get_tensor(names[2]), tensors[names[2]])


def change_of(name, **settings):
    def change(tensor, entry_settings):
        if tensor == name:
            entry_settings.update(settings)
    return change


@pytest.mark.parametrize("name,settings,message", [
    ("conv/bias", {"crc": 12345}, "does not have the checksum"),
    ("large", {"crc": 12345}, "does not have the checksum"),
    ("strings", {"crc": 12345}, "does not have the checksum"),
    ("conv/bias", {"size": 252}, "has 252 bytes; its shape (64,) needs 256"),
    ("conv/bias", {"shape": (65,)}, "its shape (65,) needs 260"),
    ("conv/bias", {"shape": (63,)}, "has 256 bytes; its shape (63,) needs "
     "252"),
    ("conv/bias", {"shape": (8, 4)}, "its shape (8, 4) needs 128"),
    ("conv/bias", {"dtype": 2}, "its shape (64,) needs 512"),
    ("large", {"offset": 10 ** 7}, "is cut short: tensor large ends at"),
    ("strings", {"size": 3}, "is cut short"),
    ("strings", {"shape": (2, 1)}, "does not have the size its strings "
     "need"),
    ("one string", {"size": 22}, "does not have the size its strings need"),
])
def test_entries_that_do_not_fit_the_data(tmp_path, name, settings, message):
    tensors = some_tensors()
    reader = Checkpoint(write_checkpoint(
        str(tmp_path / "ckpt"), tensors, change=change_of(name, **settings)))
    with pytest.raises(CorruptCheckpointError) as failure:
        reader.get_tensor(name)
    assert message in str(failure.value)
    assert "tensor {}".format(name) in str(failure.value)
    # the other tensors are read
    other = "double" if name != "double" else "half"
    assert_same_tensor(reader.get_tensor(other), tensors[other])


def test_data_that_was_changed(tmp_path):
    tensors = some_tensors()
    prefix = write_checkpoint(str(tmp_path / "ckpt"), tensors)
    data_path = prefix + ".data-00000-of-00001"
    with open(data_path, "rb") as fh:
        good = fh.read()
    sizes = {name: len(tensor_bytes(tensors[name])[2])
             for name in sorted(tensors)}
    offset = 0
    for name in sorted(tensors):
        if sizes[name] == 0:
            continue
        # one bit of this tensor, in its middle
        data = bytearray(good)
        data[offset + sizes[name] // 2] ^= 0x04
        with open(data_path, "wb") as fh:
            fh.write(bytes(data))
        reader = Checkpoint(prefix)
        with pytest.raises(CorruptCheckpointError) as failure:
            reader.get_tensor(name)
        assert "tensor {}".format(name) in str(failure.value)
        assert data_path in str(failure.value)
        for other in tensors:
            if other != name:
                assert_same_tensor(reader.get_tensor(other), tensors[other])
        offset += sizes[name]

    # the lengths of two strings exchanged (0 and 3: the sum is the
    # same): the checksum of the lengths says so
    start = sum(sizes[name] for name in sorted(tensors) if name < "strings")
    data = bytearray(good)
    assert bytes(data[start:start + 2]) == b"\x00\x03"
    data[start:start + 2] = b"\x03\x00"
    with open(data_path, "wb") as fh:
        fh.write(bytes(data))
    with pytest.raises(CorruptCheckpointError) as failure:
        Checkpoint(prefix).get_tensor("strings")
    assert "does not have the checksum that is stored with the lengths " \
        "of its strings" in str(failure.value)
    # a length changed: the tensor no longer has the size they need
    data = bytearray(good)
    data[start + 1] ^= 0x01
    with open(data_path, "wb") as fh:
        fh.write(bytes(data))
    with pytest.raises(CorruptCheckpointError) as failure:
        Checkpoint(prefix).get_tensor("strings")
    assert "does not have the size its strings need" in str(failure.value)

    # the file cut short, and missing
    with open(data_path, "wb") as fh:
        fh.write(good[:-10])
    with pytest.raises(CorruptCheckpointError, match="is cut short"):
        Checkpoint(prefix).get_tensor(sorted(tensors)[-1])
    os.remove(data_path)
    with pytest.raises(FileNotFoundError) as failure:
        Checkpoint(prefix).get_tensor("double")
    assert data_path in str(failure.value)


def test_checkpoints_that_are_not_supported(tmp_path):
    tensors = some_tensors()
    prefix = str(tmp_path / "ckpt")

    # written on a big endian machine
    write_checkpoint(prefix, tensors, header=field(1, 1) + field(2, 1))
    with pytest.raises(UnsupportedModelError, match="big endian"):
        Checkpoint(prefix)

    # a variable saved in slices
    write_checkpoint(prefix, tensors, change=change_of(
        "double", extra=field(7, field(1, b""))))
    reader = Checkpoint(prefix)
    with pytest.raises(UnsupportedModelError, match="was saved in slices"):
        reader.get_tensor("double")
    assert_same_tensor(reader.get_tensor("half"), tensors["half"])

    # types whose bytes are not those of an array (complex64, resource)
    for dtype in (8, 20, 99):
        write_checkpoint(prefix, tensors, change=change_of(
            "double", dtype=dtype))
        with pytest.raises(UnsupportedModelError) as failure:
            Checkpoint(prefix).get_tensor("double")
        assert "tensorflow.DataType {}".format(dtype) in str(failure.value)

    # a shape of unknown rank
    index = prefix + ".index"
    pairs = [(b"", field(1, 1)),
             (b"x", field(1, 1) + field(2, field(3, 1)) + field(5, 4))]
    with open(index, "wb") as fh:
        fh.write(table(pairs))
    with pytest.raises(UnsupportedModelError, match="no known shape"):
        Checkpoint(prefix).get_tensor("x")

    # a table that is not the index of a checkpoint
    with open(index, "wb") as fh:
        fh.write(table([(b"a", b"b")]))
    with pytest.raises(CorruptCheckpointError, match="has no header"):
        Checkpoint(prefix)
    # an entry that is no message
    with open(index, "wb") as fh:
        fh.write(table([(b"", field(1, 1)), (b"x", b"\xff\xff\xff")]))
    with pytest.raises(CorruptCheckpointError,
                       match="the entry of tensor x cannot be read"):
        Checkpoint(prefix).get_tensor("x")
    os.remove(index)
    with pytest.raises(FileNotFoundError):
        Checkpoint(prefix)


def test_fields_that_are_zero_may_be_written(tmp_path):
    """
        Protocol buffers leave out fields that are zero; a writer that
        does not is read the same
    """

    prefix = str(tmp_path / "ckpt")
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    dtype, shape, raw, crc = tensor_bytes(value)
    message = field(1, dtype) + \
        field(2, b"".join(field(2, field(1, n)) for n in shape)) + \
        field(3, 0) + field(4, 0) + field(5, len(raw)) + \
        varint((6 << 3) | 5) + struct.pack("<I", masked(crc))
    with open(prefix + ".index", "wb") as fh:
        fh.write(table([(b"", field(1, 1) + field(2, 0)), (b"x", message)]))
    with open(prefix + ".data-00000-of-00001", "wb") as fh:
        fh.write(raw)
    assert_same_tensor(Checkpoint(prefix).get_tensor("x"), value)


# --------------------------------------------------------------------
# the models that TensorFlow saved
# --------------------------------------------------------------------

@pytest.fixture(scope="module", params=FIXTURE_VERSIONS)
def saved_by(request, tmp_path_factory):
    """
        The models that the given TensorFlow version saved (see
        tests/data/saved_models/make_fixtures.py), unpacked
    """

    target = str(tmp_path_factory.mktemp("fixtures"))
    return {"version": request.param,
            "dir": unpack_fixture_models(request.param, target)}


@pytest.mark.parametrize("directory", sorted(FIXTURE_MODELS))
def test_weights_of_the_models_tensorflow_saved(saved_by, directory):
    """
        Without TensorFlow: every tensor of the checkpoint of a model
        that TensorFlow 2.4.1, 2.11.0 or 2.15.1 saved is read (which
        checks its checksum), and the weights of the layers are the
        ones that were saved, by the digests that were stored with the
        models
    """

    case_dir = os.path.join(saved_by["dir"], directory)
    model = os.path.join(case_dir, "model")
    reader = Checkpoint(os.path.join(model, "variables", "variables"))
    keys = reader.keys()
    assert "_CHECKPOINTABLE_OBJECT_GRAPH" in keys and len(keys) > 10
    kinds = set()
    for key in keys:
        value = reader.get_tensor(key)
        kinds.add(type(value).__name__ if isinstance(value, bytes)
                  else value.dtype.name)
    assert {"bytes", "float32"} <= kinds
    if FIXTURE_MODELS[directory][1]:
        # saved after training steps: the state of the optimizer
        assert "int64" in kinds
        assert any(key.startswith("optimizer/") for key in keys)

    expected = np.load(os.path.join(case_dir, "expected.npz"))
    digests = json.loads(str(expected["weight_digests"]))
    weights = loading.read_weights(model)
    assert sorted(weights) == sorted(digests)
    for layer, variables in digests.items():
        assert sorted(weights[layer]) == sorted(variables)
        for variable, digest in variables.items():
            value = weights[layer][variable]
            assert value.dtype == np.float32
            assert hashlib.sha256(np.ascontiguousarray(
                value).tobytes()).hexdigest() == digest, (layer, variable)


def test_saved_model_with_a_damaged_checkpoint(saved_by, tmp_path):
    source = os.path.join(saved_by["dir"], "released_structure", "model")
    model = str(tmp_path / "model")
    shutil.copytree(source, model)
    data_path = os.path.join(model, "variables",
                             "variables.data-00000-of-00001")
    index_path = os.path.join(model, "variables", "variables.index")
    with open(data_path, "rb") as fh:
        data = fh.read()
    with open(index_path, "rb") as fh:
        index = fh.read()
    assert loading.read_weights(model)

    def write(path, content):
        with open(path, "wb") as fh:
            fh.write(content)

    # a bit of the first kernel
    reader = Checkpoint(os.path.join(model, "variables", "variables"))
    key = "layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"
    offset = reader._entry(key)["offset"]
    changed = bytearray(data)
    changed[offset + 5] ^= 0x01
    write(data_path, bytes(changed))
    with pytest.raises(CorruptCheckpointError) as failure:
        loading.read_weights(model)
    assert "tensor {} does not have the checksum".format(key) in \
        str(failure.value)

    # the data cut short; the index cut short, changed, missing
    write(data_path, data[:len(data) // 2])
    with pytest.raises(CorruptCheckpointError, match="is cut short"):
        loading.read_weights(model)
    write(data_path, data)
    write(index_path, index[:len(index) // 2])
    with pytest.raises(CorruptCheckpointError,
                       match="is not a checkpoint index"):
        loading.read_weights(model)
    changed = bytearray(index)
    changed[len(index) // 2] ^= 0x20
    write(index_path, bytes(changed))
    with pytest.raises(CorruptCheckpointError,
                       match="does not have the checksum"):
        loading.read_weights(model)
    os.remove(index_path)
    with pytest.raises(FileNotFoundError):
        loading.read_weights(model)
    write(index_path, index)
    os.remove(data_path)
    with pytest.raises(FileNotFoundError, match="is missing"):
        loading.read_weights(model)


# --------------------------------------------------------------------
# with TensorFlow: its own reader
# --------------------------------------------------------------------

@pytest.fixture(scope="module")
def tf():
    return pytest.importorskip("tensorflow")


def assert_as_tensorflow_reads(tf, prefix):
    theirs = tf.train.load_checkpoint(prefix)
    reader = Checkpoint(prefix)
    names = sorted(theirs.get_variable_to_shape_map())
    assert reader.keys() == names
    kinds = set()
    for name in names:
        expected = theirs.get_tensor(name)
        found = reader.get_tensor(name)
        if isinstance(expected, bytes):
            assert found == expected
            kinds.add("bytes")
        elif expected.dtype == object:
            assert found.shape == expected.shape
            assert found.tolist() == expected.tolist()
            kinds.add("strings")
        else:
            assert_same_tensor(found, expected)
            kinds.add(expected.dtype.name)
    return kinds


@pytest.mark.parametrize("directory", sorted(FIXTURE_MODELS))
def test_every_tensor_as_tensorflow_reads_it(tf, saved_by, directory):
    prefix = os.path.join(saved_by["dir"], directory, "model", "variables",
                          "variables")
    assert {"bytes", "float32"} <= assert_as_tensorflow_reads(tf, prefix)


def test_checkpoint_that_tensorflow_writes_here(tf, tmp_path):
    """
        Variables of many types and sizes, and enough of them for an
        index of several blocks, written by the TensorFlow that is
        installed and read here
    """

    rng = np.random.RandomState(2)
    holder = tf.Module()
    holder.values = [
        tf.Variable(rng.normal(size=(7, 5)).astype(dtype), name="a")
        for dtype in ("float16", "float32", "float64")]
    holder.values += [
        tf.Variable(rng.randint(0, 100, size=(3, 4)).astype(dtype), name="b")
        for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16",
                      "uint32", "uint64")]
    holder.values += [
        tf.Variable([True, False]), tf.Variable(3.5), tf.Variable(7),
        tf.Variable(np.zeros((0, 3), dtype=np.float32)),
        tf.Variable(rng.normal(size=(300, 500)).astype(np.float32)),
        tf.Variable([["", "abc"], ["d" * 1000, "é"]]),
        tf.Variable("one string")]
    # many variables with long names: the index is written in blocks
    # of 256 kB
    holder.many = {"variable_with_a_long_name_{:05d}_{}".format(i, "x" * 60):
                   tf.Variable(float(i)) for i in range(4000)}
    prefix = tf.train.Checkpoint(holder=holder).write(
        str(tmp_path / "written"))
    kinds = assert_as_tensorflow_reads(tf, prefix)
    assert kinds >= {"bytes", "strings", "bool", "float16", "float32",
                     "float64", "int8", "int16", "int32", "int64", "uint8",
                     "uint16", "uint32", "uint64"}

    # the index has several blocks
    with open(prefix + ".index", "rb") as fh:
        index = memoryview(fh.read())
    footer = index[len(index) - 48:]
    position = 0
    for _ in range(2):
        _, position = loading._varint(footer, position)
    offset, position = loading._varint(footer, position)
    size, position = loading._varint(footer, position)
    blocks = list(checkpoint._entries(
        checkpoint._block(index, offset, size, "index"), "index"))
    assert len(blocks) > 1
