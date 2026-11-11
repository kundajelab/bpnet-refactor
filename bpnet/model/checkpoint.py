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
    is reported, not read as something else: what these files say is
    read as TensorFlow reads it, or not at all. Where the TensorFlow
    versions do not agree with each other (2.4 and 2.11 read the
    messages of the index with another protocol buffer reader than
    2.15), a file is read only if all of them read it the same. For
    files that no writer makes the reader is therefore the stricter
    one in places, which the tests list.

    What is not supported (and reported as such): variables that were
    saved in slices (partitioned variables), an index with compressed
    blocks (TensorFlow 1.2 and later do not write them), big endian
    data, tensors of more than 32 dimensions or marked as of unknown
    shape.
"""

import os
import struct

import numpy as np

from bpnet.model.loading import (
    UnsupportedModelError, _fields, _int32, _int64, _varint,
    _BYTES, _FIXED32, _VARINT)


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
#: the version of the checkpoint format that this module reads
#: (TensorFlow's kTensorBundleVersion, the same from 2.4 to 2.21)
_VERSION = 1


def _number(buffer, position, bits=64):
    """
        A number in the base 128 encoding of the formats, of the
        given width (64 bits: at most ten bytes; 32 bits: five)

        Returns:
            tuple: (value, position after it)
    """

    result = shift = 0
    for _ in range(10 if bits == 64 else 5):
        if position >= len(buffer):
            raise ValueError("a number is cut short")
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if result >> bits:
                raise ValueError("a number does not fit {} bits".format(bits))
            return result, position
        shift += 7
    raise ValueError("a number takes more bytes than {} bits need".format(
        bits))


def _entries(block, path):
    """
        The (key, value) pairs of a block, in order

        A key is stored as the number of bytes it shares with the key
        before it and the rest. At the "restart points", whose
        positions are listed at the end of the block, a key is stored
        whole; TensorFlow finds a key by them, so they are checked.
    """

    def corrupt(what):
        return CorruptCheckpointError(
            "{}: a block of its table is not in the table format "
            "({})".format(path, what))

    if len(block) < 8:
        raise corrupt("it has {} bytes".format(len(block)))
    count, = struct.unpack("<I", block[len(block) - 4:])
    end = len(block) - 4 - 4 * count
    if count < 1 or end < 0:
        raise corrupt("{} restart points in {} bytes".format(
            count, len(block)))
    restarts = struct.unpack("<{}I".format(count),
                             block[end:len(block) - 4])
    if restarts[0] != 0 or \
            any(b <= a for a, b in zip(restarts, restarts[1:])):
        raise corrupt("its restart points are not in order from 0")
    restarts = set(restarts)
    pairs, starts = [], set()
    position, key = 0, b""
    try:
        while position < end:
            starts.add(position)
            whole = position in restarts
            shared, position = _number(block, position, 32)
            unshared, position = _number(block, position, 32)
            length, position = _number(block, position, 32)
            if shared > len(key) or position + unshared + length > end:
                raise ValueError("an entry goes beyond its block")
            if whole and shared:
                raise ValueError("a key at a restart point is not whole")
            key = key[:shared] + bytes(block[position:position + unshared])
            position += unshared
            pairs.append((key, block[position:position + length]))
            position += length
    except ValueError as error:
        raise corrupt(error)
    if not restarts <= starts | {0}:
        raise corrupt("a restart point is not at an entry")
    return pairs


def _handle(value, path):
    """
        Where a block is: (offset, size)
    """

    try:
        offset, position = _number(value, 0)
        size, position = _number(value, position)
    except ValueError as error:
        raise CorruptCheckpointError(
            "{}: the place of a block is not in the table format "
            "({})".format(path, error))
    return offset, size


def read_table(path):
    """
        The keys and values of a table file (the '.index' file of a
        checkpoint)

        The table is read as TensorFlow reads it or not at all: every
        block against its checksum, the keys in ascending order and
        none twice, every block within the range of keys that the
        list of blocks gives it. (A table that breaks these rules
        could hold a key twice, and TensorFlow would find one of the
        two.)

        Args:
            path (str): the file

        Returns:
            dict: key (bytes) -> value (bytes), in the order of the
                file (sorted by key)
    """

    size = os.path.getsize(path)
    if size < _FOOTER:
        raise CorruptCheckpointError(
            "{} is too short for a checkpoint index ({} bytes)".format(
                path, size))
    with open(path, "rb") as fh:
        fh.seek(size - _FOOTER)
        footer = memoryview(fh.read(_FOOTER))
        if len(footer) != _FOOTER or \
                struct.unpack("<Q", footer[40:])[0] != _MAGIC:
            raise CorruptCheckpointError(
                "{} is not a checkpoint index (or it is cut short): it "
                "does not end with the mark of a table".format(path))

        def block(offset, length):
            # the contents of a block, after its checksum
            end = offset + length + 5
            if end > size - _FOOTER:
                raise CorruptCheckpointError(
                    "{} is cut short: a block of its table ends at byte "
                    "{} of {}".format(path, end, size))
            fh.seek(offset)
            raw = memoryview(fh.read(length + 5))
            if len(raw) != length + 5:
                raise CorruptCheckpointError(
                    "{} is cut short".format(path))
            stored, = struct.unpack("<I", raw[length + 1:])
            if crc32c(raw[:length + 1]) != _unmask(stored):
                raise CorruptCheckpointError(
                    "{}: a block of its table does not have the checksum "
                    "that is stored with it".format(path))
            if raw[length] != 0:
                raise UnsupportedModelError(
                    "{}: a block of its table is compressed (type "
                    "{})".format(path, raw[length]))
            return raw[:length]

        try:
            # the place of the block of meta data, which is not needed,
            # then that of the block that lists the blocks
            handles = footer[:_FOOTER - 8]
            _, position = _number(handles, 0)
            _, position = _number(handles, position)
            offset, position = _number(handles, position)
            length, position = _number(handles, position)
        except ValueError as error:
            raise CorruptCheckpointError(
                "{}: the end of the table is not in the table format "
                "({})".format(path, error))

        found = {}
        last = before = None
        for separator, handle in _entries(block(offset, length), path):
            # the keys of a block lie after those of the block before
            # and not after the key the list gives for the block
            if before is not None and separator <= before:
                raise CorruptCheckpointError(
                    "{}: the list of the blocks of its table is not in "
                    "order".format(path))
            for key, value in _entries(block(*_handle(handle, path)), path):
                if (last is not None and key <= last) or key > separator \
                        or (before is not None and key <= before):
                    raise CorruptCheckpointError(
                        "{}: the keys of its table are not in order, or "
                        "one is there twice ({!r})".format(path, key))
                found[key] = bytes(value)
                last = key
            before = separator
    return found


# --------------------------------------------------------------------
# the tensors
# --------------------------------------------------------------------

#: tensorflow.DataType -> numpy type, for the types whose bytes in the
#: file are those of the array (floats, integers, bool; not complex
#: numbers, bfloat16 or the quantized types)
_TYPES = {
    1: "<f4", 2: "<f8", 3: "<i4", 4: "u1", 5: "<i2", 6: "i1", 9: "<i8",
    10: "?", 17: "<u2", 19: "<f2", 22: "<u4", 23: "<u8",
}
_STRING = 7
#: the most dimensions of an array (numpy 1)
_MAX_RANK = 32


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
        # BundleHeaderProto: num_shards, endianness, version. The
        # messages of the index are read as protocol buffers read
        # them: a field counts when it has the kind its message gives
        # it and is passed over otherwise, a number has the width of
        # its field (the numbers here: 32 bits, signed), and of a
        # field that is there twice the last one counts
        shards, endianness, producer, consumer, bad = 0, 0, 0, 0, []
        try:
            for number, kind, value in _fields(memoryview(table.pop(b""))):
                if number == 1 and kind == _VARINT:
                    shards = _int32(value)
                elif number == 2 and kind == _VARINT:
                    endianness = _int32(value)
                elif number == 3 and kind == _BYTES:
                    # VersionDef: producer, min_consumer, bad_consumers
                    # (a list: one field each, or all in one field)
                    single = False
                    for field, field_kind, part in _fields(value):
                        if field == 1 and field_kind == _VARINT:
                            producer = _int32(part)
                        elif field == 2 and field_kind == _VARINT:
                            consumer = _int32(part)
                        elif field == 3 and field_kind == _VARINT:
                            bad.append(_int32(part))
                        elif field == 3 and field_kind == _BYTES:
                            if single:
                                # (the protocol buffer reader of
                                # TensorFlow before 2.12 takes the
                                # length of such a list for a member
                                # of it, and its members for fields)
                                raise ValueError(
                                    "a list of versions right after a "
                                    "single one, which the TensorFlow "
                                    "versions read differently")
                            position = 0
                            while position < len(part):
                                one, position = _varint(part, position)
                                bad.append(_int32(one))
                        single = field == 3 and field_kind == _VARINT
        except ValueError as error:
            raise CorruptCheckpointError(
                "{}: its header cannot be read ({})".format(
                    self.index_path, error))
        if endianness != 0:
            raise UnsupportedModelError(
                "{}: the checkpoint was not written on a little endian "
                "machine (endianness {})".format(
                    self.index_path, endianness))
        # as TensorFlow checks the versions of a checkpoint it reads
        if consumer > _VERSION or _VERSION in bad or producer < 0:
            raise UnsupportedModelError(
                "{}: the checkpoint asks for another version of the "
                "format than this reader has (version {}; written by {}, "
                "for readers from {}, not for {})".format(
                    self.index_path, _VERSION, producer, consumer, bad))
        self.shards = shards
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
        """
            What the index says about a tensor (BundleEntryProto), and
            the number of its elements
        """

        try:
            message = self._entries[key.encode("utf-8")]
        except KeyError:
            raise KeyError("{} has no tensor {}".format(
                self.index_path, key))
        # (a field that is zero is not stored)
        entry = {"dtype": 0, "shape": [], "shard": 0, "offset": 0,
                 "size": 0, "crc": 0}
        unknown_rank = sliced = False
        try:
            for number, kind, value in _fields(memoryview(message)):
                if number == 1 and kind == _VARINT:
                    entry["dtype"] = _int32(value)
                elif number == 2 and kind == _BYTES:
                    # TensorShapeProto: dim[i] (size, name), unknown_rank
                    for dim_number, dim_kind, dim in _fields(value):
                        if dim_number == 2 and dim_kind == _BYTES:
                            size = 0
                            for part, part_kind, part_value in _fields(dim):
                                if part == 1 and part_kind == _VARINT:
                                    size = _int64(part_value)
                                elif part == 2 and part_kind == _BYTES:
                                    # (the name of a dimension is text:
                                    # protocol buffers do not read a
                                    # message in which it is not)
                                    bytes(part_value).decode("utf-8")
                            entry["shape"].append(size)
                        elif dim_number == 3 and dim_kind == _VARINT:
                            unknown_rank = dim != 0
                elif number == 3 and kind == _VARINT:
                    entry["shard"] = _int32(value)
                elif number == 4 and kind == _VARINT:
                    entry["offset"] = _int64(value)
                elif number == 5 and kind == _VARINT:
                    entry["size"] = _int64(value)
                elif number == 6 and kind == _FIXED32:
                    entry["crc"], = struct.unpack("<I", value)
                elif number == 7 and kind == _BYTES:
                    sliced = True
            if entry["offset"] < 0 or entry["size"] < 0:
                raise ValueError("a negative offset or size")
            # (as TensorFlow: the number of elements fits 63 bits
            # after every dimension, also when a later one is 0)
            count = 1
            for size in entry["shape"]:
                if size < 0:
                    raise ValueError("a dimension of unknown or negative "
                                     "size")
                count *= size
                if count >> 63:
                    raise ValueError("a shape of more elements than a "
                                     "tensor can have")
            entry["count"] = count
        except ValueError as error:
            raise CorruptCheckpointError(
                "{}: the entry of tensor {} cannot be read ({})".format(
                    self.index_path, key, error))
        if sliced:
            raise UnsupportedModelError(
                "{}: tensor {} was saved in slices (a partitioned "
                "variable)".format(self.index_path, key))
        if unknown_rank:
            raise UnsupportedModelError(
                "{}: tensor {} has no known shape".format(
                    self.index_path, key))
        if len(entry["shape"]) > _MAX_RANK:
            raise UnsupportedModelError(
                "{}: tensor {} has {} dimensions".format(
                    self.index_path, key, len(entry["shape"])))
        return entry

    def _read(self, key, entry, start, length):
        """
            `length` bytes of a tensor, from its byte `start`, after
            checking that the data file has them. (As for TensorFlow,
            a tensor of no bytes can be anywhere.)
        """

        path = "{}.data-{:05d}-of-{:05d}".format(
            self.prefix, entry["shard"], self.shards)
        if path not in self._files:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "{} is missing (the data of the checkpoint {})".format(
                        path, self.prefix))
            self._files[path] = os.path.getsize(path)
        end = entry["offset"] + start + length
        if length and end > self._files[path]:
            raise CorruptCheckpointError(
                "{} is cut short: tensor {} is read to byte {} of {}".format(
                    path, key, end, self._files[path]))
        if not length:
            return b""
        with open(path, "rb") as fh:
            fh.seek(entry["offset"] + start)
            data = fh.read(length)
        if len(data) != length:
            raise CorruptCheckpointError(
                "{} is cut short: tensor {} has {} of {} bytes".format(
                    path, key, len(data), length))
        return data

    def get_tensor(self, key):
        """
            One tensor of the checkpoint

            What the index says about the tensor is checked before
            its bytes are read (their number against the shape), and
            the bytes against their checksum.

            Args:
                key (str): its name in the checkpoint

            Returns:
                numpy.ndarray: the tensor; for a tensor of strings an
                    array of bytes objects (a bytes object for a
                    single string, as TensorFlow's reader returns it)
        """

        entry = self._entry(key)
        shape, count = tuple(entry["shape"]), entry["count"]

        def corrupt(what):
            return CorruptCheckpointError(
                "{}.data-{:05d}-of-{:05d}: tensor {} {}".format(
                    self.prefix, entry["shard"], self.shards, key, what))

        if entry["dtype"] == _STRING:
            # in the file: the lengths of the strings (one to ten
            # bytes each), a checksum of the lengths, the strings.
            # The lengths are read first; the strings are read when
            # the lengths add up to the size the index gives (an
            # index can claim any size)
            size = entry["size"]
            if size < count + 4:
                raise corrupt("does not have the size its strings need")
            head = memoryview(self._read(
                key, entry, 0, min(size, 10 * count + 4)))
            lengths, position, crc = [], 0, 0
            try:
                for _ in range(count):
                    length, position = _number(head, position)
                    lengths.append(length)
                    crc = crc32c(struct.pack(
                        "<I" if length <= 0xFFFFFFFF else "<Q", length), crc)
            except ValueError:
                raise corrupt("does not have the size its strings need")
            if position + 4 + sum(lengths) != size:
                raise corrupt("does not have the size its strings need")
            stored, = struct.unpack("<I", head[position:position + 4])
            if _unmask(stored) != crc:
                raise corrupt("does not have the checksum that is stored "
                              "with the lengths of its strings")
            crc = crc32c(head[position:position + 4], crc)
            body = memoryview(self._read(
                key, entry, position + 4, sum(lengths)))
            if _unmask(entry["crc"]) != crc32c(body, crc):
                raise corrupt("does not have the checksum that is stored "
                              "with it")
            strings, position = [], 0
            for length in lengths:
                strings.append(bytes(body[position:position + length]))
                position += length
            if shape == ():
                return strings[0]
            out = np.empty(count, dtype=object)
            out[:] = strings
            return self._shaped(out, shape, key)

        if entry["dtype"] not in _TYPES:
            raise UnsupportedModelError(
                "{}: tensor {} has a type that is not supported "
                "(tensorflow.DataType {})".format(
                    self.index_path, key, entry["dtype"]))
        dtype = np.dtype(_TYPES[entry["dtype"]])
        if count * dtype.itemsize != entry["size"]:
            raise corrupt("has {} bytes; its shape {} needs {}".format(
                entry["size"], shape, count * dtype.itemsize))
        data = self._read(key, entry, 0, entry["size"])
        if _unmask(entry["crc"]) != crc32c(data):
            raise corrupt("does not have the checksum that is stored "
                          "with it")
        return self._shaped(np.frombuffer(data, dtype=dtype), shape,
                            key).astype(dtype.newbyteorder("="))

    def _shaped(self, array, shape, key):
        """
            The array in the shape of the tensor. (A tensor of no
            elements can have a shape that numpy arrays do not have:
            (0, 2**62).)
        """

        try:
            return array.reshape(shape)
        except ValueError as error:
            raise UnsupportedModelError(
                "{}: tensor {} has a shape that numpy arrays do not have "
                "({}: {})".format(self.index_path, key, shape, error))
