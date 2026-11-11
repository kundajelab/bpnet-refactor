"""
    The variables of a TensorFlow checkpoint, read without TensorFlow

    A SavedModel keeps its variables in a checkpoint ("tensor bundle"):
    'variables/variables.index' lists every tensor (type, shape, where
    its bytes are), and 'variables/variables.data-00000-of-00001' holds
    the bytes. Reading them takes little: the index is a table of
    sorted keys in the format of LevelDB, each value a small protocol
    buffer message, and the data are the arrays as they are in memory.

    Everything is checked the way TensorFlow checks it when it reads a
    checkpoint: the checksum of every block of the index and of every
    tensor that is read (CRC-32C). A file that is cut short or changed
    is reported, not read as something else.

    What is not supported (and reported as such): variables that were
    saved in slices (partitioned variables), an index with compressed
    blocks (TensorFlow does not write one), big endian data.
"""

import os
import struct

import numpy as np

from bpnet.model.loading import UnsupportedModelError, _fields, _varint


class CorruptCheckpointError(ValueError):
    """
        The files of a checkpoint are cut short, or their content is
        not what their checksums say
    """


# --------------------------------------------------------------------
# CRC-32C (Castagnoli), as TensorFlow and LevelDB use it
# --------------------------------------------------------------------

def _table():
    table = np.zeros(256, dtype=np.uint32)
    for byte in range(256):
        value = byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table[byte] = value
    return table


_TABLE = _table()
_TABLE_LIST = [int(value) for value in _TABLE]

#: data shorter than this is checked byte by byte; longer data in
#: pieces that are advanced together: at least _LANES of them, and
#: more for data whose pieces would be longer than _PIECE bytes
_SHORT, _LANES, _PIECE = 1 << 14, 1024, 4096


def _register(state, data):
    """
        The register of the checksum after the bytes of `data`,
        starting from `state`, byte by byte
    """

    table = _TABLE_LIST
    for byte in data:
        state = table[(state ^ byte) & 0xFF] ^ (state >> 8)
    return state


def _zeros_operator(length):
    """
        What `length` zero bytes do to the register, as four tables of
        256 entries (the register after the zeros is the exclusive or
        of one entry of each table, chosen by the four bytes of the
        register before): the register is linear in its start value.
    """

    # the images of the 32 bits of the register under one zero byte
    columns = [_register(1 << bit, b"\x00") for bit in range(32)]

    def apply(matrix, value):
        result, bit = 0, 0
        while value:
            if value & 1:
                result ^= matrix[bit]
            value >>= 1
            bit += 1
        return result

    # the operator of `length` zero bytes by repeated squaring
    result = [1 << bit for bit in range(32)]
    power = columns
    while length:
        if length & 1:
            result = [apply(power, column) for column in result]
        power = [apply(power, column) for column in power]
        length >>= 1
    tables = []
    for position in range(4):
        tables.append([apply(result, byte << (8 * position))
                       for byte in range(256)])
    return tables


def crc32c(data, crc=0):
    """
        The CRC-32C checksum of bytes

        Long data is cut into pieces of equal length whose registers
        are advanced together (one numpy operation per byte position);
        the registers of the pieces are then joined, which the
        linearity of the register allows.

        Args:
            data (bytes-like): the bytes
            crc (int): the checksum of what came before, to extend it

        Returns:
            int: the checksum (32 bits)
    """

    data = memoryview(data).cast("B")
    state = crc ^ 0xFFFFFFFF
    if len(data) >= _SHORT:
        lanes = max(_LANES, len(data) // _PIECE)
        length = len(data) // lanes
        # one row per byte position, one column per piece
        block = np.ascontiguousarray(np.frombuffer(
            data[:lanes * length], dtype=np.uint8).reshape(lanes, length).T)
        registers = np.zeros(lanes, dtype=np.uint32)
        for row in block:
            registers = _TABLE[(registers ^ row) & 0xFF] ^ (registers >> 8)
        t0, t1, t2, t3 = _zeros_operator(length)
        for register in registers.tolist():
            state = t0[state & 0xFF] ^ t1[(state >> 8) & 0xFF] ^ \
                t2[(state >> 16) & 0xFF] ^ t3[state >> 24] ^ register
        data = data[lanes * length:]
    return _register(state, data) ^ 0xFFFFFFFF


def _unmask(masked):
    """
        The checksum from the form it is stored in (TensorFlow rotates
        a stored checksum and adds a constant, so that the checksum of
        data that holds checksums is not degenerate)
    """

    rotated = (masked - 0xA282EAD8) & 0xFFFFFFFF
    return ((rotated >> 17) | (rotated << 15)) & 0xFFFFFFFF


# --------------------------------------------------------------------
# the index: a table of sorted keys
# --------------------------------------------------------------------

_MAGIC = 0xDB4775248B80FB57
_FOOTER = 48


def _block(data, offset, size, path):
    """
        The contents of a block of the table, after its checksum
    """

    end = offset + size + 5
    if end > len(data):
        raise CorruptCheckpointError(
            "{} is cut short: a block of its table ends at byte {} of "
            "{}".format(path, end, len(data)))
    contents = data[offset:offset + size]
    kind = data[offset + size]
    stored, = struct.unpack("<I", data[offset + size + 1:end])
    if crc32c(data[offset:offset + size + 1]) != _unmask(stored):
        raise CorruptCheckpointError(
            "{}: a block of its table does not have the checksum that "
            "is stored with it".format(path))
    if kind != 0:
        raise UnsupportedModelError(
            "{}: a block of its table is compressed (type {})".format(
                path, kind))
    return contents


def _entries(block, path):
    """
        The (key, value) pairs of a block, in order. A key is stored as
        the number of bytes it shares with the key before it and the
        rest.
    """

    if len(block) < 4:
        raise CorruptCheckpointError("{}: an empty block".format(path))
    restarts, = struct.unpack("<I", block[len(block) - 4:])
    end = len(block) - 4 - 4 * restarts
    if end < 0:
        raise CorruptCheckpointError(
            "{}: a block of its table is not in the table format".format(
                path))
    position, key = 0, b""
    try:
        while position < end:
            shared, position = _varint(block, position)
            unshared, position = _varint(block, position)
            length, position = _varint(block, position)
            if shared > len(key) or position + unshared + length > end:
                raise ValueError("entry beyond its block")
            key = key[:shared] + bytes(block[position:position + unshared])
            position += unshared
            yield key, block[position:position + length]
            position += length
    except ValueError as error:
        raise CorruptCheckpointError(
            "{}: a block of its table is not in the table format "
            "({})".format(path, error))


def read_table(path):
    """
        The keys and values of a table file (the '.index' file of a
        checkpoint)

        Args:
            path (str): the file

        Returns:
            dict: key (bytes) -> value (bytes), in the order of the
                file (sorted by key)
    """

    with open(path, "rb") as fh:
        data = memoryview(fh.read())
    if len(data) < _FOOTER:
        raise CorruptCheckpointError(
            "{} is too short for a checkpoint index ({} bytes)".format(
                path, len(data)))
    footer = data[len(data) - _FOOTER:]
    if struct.unpack("<Q", footer[40:])[0] != _MAGIC:
        raise CorruptCheckpointError(
            "{} is not a checkpoint index (or it is cut short): it does "
            "not end with the mark of a table".format(path))
    try:
        # the handle of the block of meta data, which is not needed,
        # then that of the block that lists the blocks
        handles = footer[:_FOOTER - 8]
        _, position = _varint(handles, 0)
        _, position = _varint(handles, position)
        offset, position = _varint(handles, position)
        size, position = _varint(handles, position)
    except ValueError:
        raise CorruptCheckpointError(
            "{}: the end of the table is not in the table format".format(
                path))
    found = {}
    for _, handle in _entries(_block(data, offset, size, path), path):
        try:
            block_offset, position = _varint(handle, 0)
            block_size, position = _varint(handle, position)
        except ValueError:
            raise CorruptCheckpointError(
                "{}: the list of blocks is not in the table format".format(
                    path))
        for key, value in _entries(
                _block(data, block_offset, block_size, path), path):
            found[key] = bytes(value)
    return found


# --------------------------------------------------------------------
# the tensors
# --------------------------------------------------------------------

#: tensorflow.DataType -> numpy type, for the types whose bytes in the
#: file are those of the array
_TYPES = {
    1: "<f4", 2: "<f8", 3: "<i4", 4: "u1", 5: "<i2", 6: "i1", 9: "<i8",
    10: "?", 17: "<u2", 19: "<f2", 22: "<u4", 23: "<u8",
}
_STRING = 7


class Checkpoint(object):
    """
        The tensors of a checkpoint

        Args:
            prefix (str): the path of the checkpoint without
                '.index' (for a SavedModel:
                '<directory>/variables/variables')
    """

    def __init__(self, prefix):
        self.prefix = prefix
        self.index_path = prefix + ".index"
        table = read_table(self.index_path)
        if b"" not in table:
            raise CorruptCheckpointError(
                "{} has no header: it is not the index of a "
                "checkpoint".format(self.index_path))
        header = {number: value for number, value in
                  _fields(memoryview(table.pop(b"")))}
        self.shards = header.get(1, 0)
        if header.get(2, 0) != 0:
            raise UnsupportedModelError(
                "{}: the checkpoint was written on a big endian "
                "machine".format(self.index_path))
        self._entries = table
        self._files = {}

    def keys(self):
        """
            The names of the tensors, sorted
        """

        return [key.decode("utf-8") for key in self._entries]

    def __contains__(self, key):
        return key.encode("utf-8") in self._entries

    def _entry(self, key):
        try:
            message = self._entries[key.encode("utf-8")]
        except KeyError:
            raise KeyError("{} has no tensor {}".format(
                self.index_path, key))
        # (a field that is zero is not stored)
        entry = {"dtype": 0, "shape": [], "shard": 0, "offset": 0,
                 "size": 0, "crc": 0}
        try:
            for number, value in _fields(memoryview(message)):
                if number == 1:
                    entry["dtype"] = value
                elif number == 2:
                    # TensorShapeProto.dim[i].size
                    for dim_number, dim in _fields(value):
                        if dim_number == 2:
                            size = 0
                            for size_number, size_value in _fields(dim):
                                if size_number == 1:
                                    size = size_value
                            entry["shape"].append(size)
                        elif dim_number == 3 and dim:
                            raise UnsupportedModelError(
                                "{}: tensor {} has no known shape".format(
                                    self.index_path, key))
                elif number == 3:
                    entry["shard"] = value
                elif number == 4:
                    entry["offset"] = value
                elif number == 5:
                    entry["size"] = value
                elif number == 6:
                    entry["crc"], = struct.unpack("<I", value)
                elif number == 7:
                    raise UnsupportedModelError(
                        "{}: tensor {} was saved in slices (a partitioned "
                        "variable)".format(self.index_path, key))
        except (ValueError, struct.error) as error:
            if isinstance(error, UnsupportedModelError):
                raise
            raise CorruptCheckpointError(
                "{}: the entry of tensor {} cannot be read ({})".format(
                    self.index_path, key, error))
        return entry

    def _bytes(self, key, entry):
        path = "{}.data-{:05d}-of-{:05d}".format(
            self.prefix, entry["shard"], self.shards)
        if path not in self._files:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "{} is missing (the data of the checkpoint {})".format(
                        path, self.prefix))
            self._files[path] = os.path.getsize(path)
        if entry["offset"] + entry["size"] > self._files[path]:
            raise CorruptCheckpointError(
                "{} is cut short: tensor {} ends at byte {} of {}".format(
                    path, key, entry["offset"] + entry["size"],
                    self._files[path]))
        with open(path, "rb") as fh:
            fh.seek(entry["offset"])
            return fh.read(entry["size"]), path

    def get_tensor(self, key):
        """
            One tensor of the checkpoint

            Args:
                key (str): its name in the checkpoint

            Returns:
                numpy.ndarray: the tensor; for a tensor of strings an
                    array of bytes objects (a bytes object for a
                    single string, as TensorFlow's reader returns it)
        """

        entry = self._entry(key)
        data, path = self._bytes(key, entry)
        shape = tuple(entry["shape"])
        count = int(np.prod(shape, dtype=np.int64))

        def corrupt(what):
            return CorruptCheckpointError(
                "{}: tensor {} {}".format(path, key, what))

        if entry["dtype"] == _STRING:
            # the lengths of the strings, a checksum of the lengths,
            # the strings
            view = memoryview(data)
            lengths, position, crc = [], 0, 0
            try:
                for _ in range(count):
                    length, position = _varint(view, position)
                    lengths.append(length)
                    crc = crc32c(struct.pack(
                        "<I" if length <= 0xFFFFFFFF else "<Q", length), crc)
            except ValueError:
                raise corrupt("is cut short")
            if position + 4 + sum(lengths) != len(data):
                raise corrupt("does not have the size its strings need")
            stored, = struct.unpack("<I", view[position:position + 4])
            if _unmask(stored) != crc:
                raise corrupt("does not have the checksum that is stored "
                              "with the lengths of its strings")
            crc = crc32c(view[position:position + 4], crc)
            position += 4
            strings = []
            for length in lengths:
                strings.append(bytes(view[position:position + length]))
                position += length
            crc = crc32c(view[len(data) - sum(lengths):], crc)
            if _unmask(entry["crc"]) != crc:
                raise corrupt("does not have the checksum that is stored "
                              "with it")
            if shape == ():
                return strings[0]
            out = np.empty(count, dtype=object)
            out[:] = strings
            return out.reshape(shape)

        if entry["dtype"] not in _TYPES:
            raise UnsupportedModelError(
                "{}: tensor {} has a type that is not supported "
                "(tensorflow.DataType {})".format(
                    self.index_path, key, entry["dtype"]))
        dtype = np.dtype(_TYPES[entry["dtype"]])
        if count * dtype.itemsize != len(data):
            raise corrupt("has {} bytes; its shape {} needs {}".format(
                len(data), shape, count * dtype.itemsize))
        if _unmask(entry["crc"]) != crc32c(data):
            raise corrupt("does not have the checksum that is stored "
                          "with it")
        return np.frombuffer(data, dtype=dtype).reshape(shape).astype(
            dtype.newbyteorder("="))
