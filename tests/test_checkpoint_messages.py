"""
    Tests for the messages inside a checkpoint (bpnet.model.checkpoint
    and its use in bpnet.model.loading): the header and the entries of
    the index, and the object graph of a saved model, are protocol
    buffer messages. The reader reads them as TensorFlow does, or not
    at all; where the TensorFlow versions do not agree with each other
    (2.4 and 2.11 have another protocol buffer reader than 2.15), only
    what all of them read the same.

    A table of checkpoints of one tensor says what the reader does
    with each, and checkpoints and graphs made at random must be read
    or refused with one of the reader's errors. Where TensorFlow is
    installed, it reads the same files (in a process of its own) and
    parses the same graphs: what is read here it must read the same.
"""

import collections
import json
import os
import random
import struct
import subprocess
import sys
import warnings

import numpy as np
import pytest

from bpnet.model import loading
from bpnet.model.checkpoint import Checkpoint, CorruptCheckpointError
from bpnet.model.loading import UnsupportedModelError

from tests.test_checkpoint import (
    BIAS, KERNEL, assert_same_tensor, attribute, crc_bit_by_bit, entry,
    graph_nodes, masked, object_graph, reference, table, tensor_bytes)
from tests.test_loading_formats import field, fixed32, fixed64, varint


# --------------------------------------------------------------------
# the messages of the index: a table of cases
# --------------------------------------------------------------------

def checksum(crc):
    return fixed32(6, struct.pack("<I", masked(crc)))


def version(producer=1, consumer=0, bad=b""):
    """
        The header of a checkpoint with a version: who wrote it, the
        first version of the readers it is for, the versions of the
        readers it is not for
    """

    out = field(1, producer) if producer else b""
    if consumer:
        out += field(2, consumer)
    return HEADER + field(3, out + bad)


#: the data file of the cases below: sixteen numbers, a tensor of no
#: strings, one of three strings, one of a string whose length is
#: written in ten bytes with bits beyond the 64th, one string, and
#: room after them
NUMBERS = np.arange(1, 17, dtype="<f4").tobytes()
_, _, NO_STRINGS, NO_STRINGS_CRC = tensor_bytes(np.array([], dtype=object))
_, _, THREE_STRINGS, THREE_STRINGS_CRC = tensor_bytes(
    np.array([b"a", b"bc", b""], dtype=object))
_lengths = struct.pack("<I", masked(crc_bit_by_bit(struct.pack("<I", 2))))
ODD_STRING = b"\x82" + b"\x80" * 8 + b"\x7e" + _lengths + b"ab"
ODD_STRING_CRC = crc_bit_by_bit(
    _lengths + b"ab", crc_bit_by_bit(struct.pack("<I", 2)))
_, _, ONE_STRING, ONE_STRING_CRC = tensor_bytes(
    np.array(b"one", dtype=object))
DATA = NUMBERS + NO_STRINGS + THREE_STRINGS + ODD_STRING + ONE_STRING + \
    bytes(16)
AT_NO_STRINGS, AT_THREE_STRINGS, AT_ODD_STRING, AT_ONE_STRING = 64, 68, 78, 94
assert DATA[AT_THREE_STRINGS:AT_ODD_STRING] == THREE_STRINGS
assert DATA[AT_ONE_STRING:AT_ONE_STRING + 8] == ONE_STRING

#: the header of one data file; the entry of the first four numbers
#: (float32, shape (4,), 16 bytes from the start of the data file) and
#: what follows its type
HEADER = field(1, 1)
FIRST = np.arange(1, 5, dtype=np.float32)
DIMENSION = field(2, field(1, 4))
CHECKSUM = checksum(crc_bit_by_bit(NUMBERS[:16]))
REST = field(2, DIMENSION) + field(5, 16) + CHECKSUM
FOUR = field(1, 1) + REST
#: bytes that read as a shape of one more dimension, of size 1
LIKE_A_SHAPE = b"\x12\x02\x08\x01\x18\x00\x18\x00"

Message = collections.namedtuple(
    "Message", "name header entry ours stricter dtype")


def damage(text="the entry of tensor x cannot be read"):
    return (CorruptCheckpointError, text)


def not_supported(text):
    return (UnsupportedModelError, text)


def case(name, entry=None, ours=None, header=None, stricter=False,
         dtype="float32"):
    """
        A checkpoint of one tensor 'x' over the data file above

        Args:
            name (str): what is special about it
            entry (bytes): the entry of the tensor in the index
                (default: the first four numbers)
            ours: what this reader returns (default: the first four
                numbers), or (error it raises, text of its message)
            header (bytes): the header of the index
            stricter: for a case this reader refuses, whether
                TensorFlow reads it: False (it refuses too), True, or
                "old" / "new" when only the TensorFlow versions with
                the older protocol buffer reader (2.4, 2.11) or the
                newer one (2.15) read it
            dtype (str): the type TensorFlow is asked to restore
    """

    return Message(name, HEADER if header is None else header,
                   FOUR if entry is None else entry,
                   FIRST if ours is None else ours, stricter, dtype)


BAD_HEADER = damage("its header cannot be read")
OTHER_VERSION = not_supported("asks for another version of the format")
BAD_STRINGS = damage("does not have the size its strings need")

MESSAGES = [
    case("as it is written"),
    case("from the third number on", FOUR[:-5] + field(4, 8) + checksum(
        crc_bit_by_bit(NUMBERS[8:24])), np.arange(3, 7, dtype=np.float32)),
    case("one number, no shape",
         field(1, 1) + field(5, 4) + checksum(crc_bit_by_bit(NUMBERS[:4])),
         np.array(1, dtype=np.float32)),
    case("no numbers",
         field(1, 1) + field(2, field(2, b"") + DIMENSION) + checksum(0),
         np.zeros((0, 4), dtype=np.float32)),
    case("fields that the header does not have",
         header=HEADER + field(15, 3) + field(16, b"xyz") +
         fixed32(17, b"abcd") + fixed64(18, bytes(8))),
    case("another checksum",
         field(1, 1) + field(2, DIMENSION) + field(5, 16) +
         fixed32(6, b"abcd"),
         damage("does not have the checksum that is stored with it")),
    case("beyond the end of the data file", FOUR + field(4, 1000),
         damage("is cut short: tensor x ends at byte 1016 of")),

    # a field of another kind (number, length delimited, fixed width)
    # than its message gives it is passed over, whatever its bytes
    # look like
    case("the shape in a field of eight bytes",
         FOUR + fixed64(2, LIKE_A_SHAPE)),
    case("the shape in a field of four bytes",
         FOUR + fixed32(2, LIKE_A_SHAPE[:4])),
    case("the shape as a number", FOUR + field(2, 7)),
    case("the type as text", FOUR + field(1, b"\x02")),
    case("the type in four bytes", FOUR + fixed32(1, struct.pack("<I", 2))),
    case("the data file as text", FOUR + field(3, b"\x01")),
    case("the offset as text", FOUR + field(4, b"\x08")),
    case("the offset in eight bytes",
         FOUR + fixed64(4, struct.pack("<Q", 8))),
    case("the size as text", FOUR + field(5, b"\x20")),
    case("the size in eight bytes", FOUR + fixed64(5, struct.pack("<Q", 32))),
    case("the checksum as a number", FOUR + field(6, 5)),
    case("the checksum in eight bytes", FOUR + fixed64(6, bytes(8))),
    case("the checksum as text", FOUR + field(6, b"abcd")),
    case("slices as a number", FOUR + field(7, 1)),
    case("slices in four bytes", FOUR + fixed32(7, b"abcd")),
    case("in the shape: a dimension as a number, the unknown rank as text",
         field(1, 1) + field(2, DIMENSION + field(2, 5) + field(3, b"\x01")) +
         field(5, 16) + CHECKSUM),
    case("in a dimension: the size as text and in eight bytes",
         field(1, 1) + field(2, field(2, field(1, 4) + field(1, b"\x09") +
                                      fixed64(1, struct.pack("<Q", 9)))) +
         field(5, 16) + CHECKSUM),
    case("fields that the message does not have",
         FOUR + field(15, 3) + field(16, b"xyz") + fixed32(17, b"abcd") +
         fixed64(18, bytes(8))),
    case("a dimension with a name",
         field(1, 1) + field(2, field(2, field(1, 4) + field(
             2, u"l\u00e4nge".encode("utf-8")))) + field(5, 16) + CHECKSUM),
    case("the number of data files as text, after the number",
         header=HEADER + field(1, b"ab")),
    case("the byte order as text", header=HEADER + field(2, b"\x01")),
    case("the version as a number", header=HEADER + field(3, 7)),
    case("in the version: its numbers as text and in fixed widths",
         header=HEADER + field(3, field(1, b"ab") + field(2, b"\x05") +
                               fixed32(3, struct.pack("<I", 1)) +
                               fixed64(3, struct.pack("<Q", 1)) +
                               fixed32(2, struct.pack("<I", 9)))),
    # ... which leaves the entry without the field
    case("the size only as text",
         field(1, 1) + field(2, DIMENSION) + field(5, b"\x10") + CHECKSUM,
         damage("has 0 bytes; its shape (4,) needs 16")),
    case("the shape only as a number",
         field(1, 1) + field(2, 7) + field(5, 16) + CHECKSUM,
         damage("has 16 bytes; its shape () needs 4")),
    case("the checksum only as a number",
         field(1, 1) + field(2, DIMENSION) + field(5, 16) + field(6, 5),
         damage("does not have the checksum that is stored with it")),
    case("the number of data files only as text", header=field(1, b"ab"),
         ours=(FileNotFoundError, "ckpt.data-00000-of-00000")),

    # a message that is there twice counts as one, and of a number
    # that is there twice the last one
    case("the shape twice: the dimensions of both",
         FOUR + field(2, field(2, field(1, 1))), FIRST.reshape(4, 1)),
    case("numbers twice: the last ones",
         field(1, 2) + field(5, 99) + field(4, 7) + FOUR + field(4, 0)),
    case("the size of a dimension twice: the last one",
         field(1, 1) + field(2, field(2, field(1, 9) + field(1, 4))) +
         field(5, 16) + CHECKSUM),
    case("the checksum twice: the last one",
         field(1, 1) + field(2, DIMENSION) + field(5, 16) +
         fixed32(6, b"abcd") + CHECKSUM),
    case("the unknown rank twice: the last one",
         field(1, 1) + field(2, DIMENSION + field(3, 1) + field(3, 0)) +
         field(5, 16) + CHECKSUM),
    case("the version twice: the last numbers",
         header=HEADER + field(3, field(2, 5)) + field(3, field(2, 0))),
    case("the version twice: the lists of both",
         header=HEADER + field(3, field(3, 1)) + field(3, field(1, 4)),
         ours=OTHER_VERSION),

    # a number has the width of its field: 32 bits for the type, the
    # data file and the versions, 64 for sizes; all are signed, and of
    # a number of ten bytes 64 bits count
    case("of the type the low 32 bits", field(1, 2 ** 32 + 1) + REST),
    case("of the number of the data file the low 32 bits",
         FOUR + field(3, 2 ** 32)),
    case("of the number of data files the low 32 bits",
         header=field(1, 2 ** 32 + 1)),
    case("of the byte order the low 32 bits",
         header=HEADER + field(2, 2 ** 32)),
    case("of a type of ten bytes 64 bits",
         b"\x08\x81" + b"\x80" * 8 + b"\x7e" + REST),
    case("of a size of ten bytes 64 bits",
         field(1, 1) + field(2, DIMENSION) + b"\x28\x90" + b"\x80" * 8 +
         b"\x7e" + CHECKSUM),
    case("the type -1", field(1, 2 ** 32 - 1) + REST,
         not_supported("tensorflow.DataType -1")),
    case("the data file -1", FOUR + field(3, 2 ** 32 - 1),
         (FileNotFoundError, "ckpt.data--0001-of-00001")),
    case("a negative size",
         field(1, 1) + field(2, DIMENSION) + field(5, 2 ** 64 - 1) + CHECKSUM,
         damage()),
    case("a negative offset", FOUR + field(4, 2 ** 63), damage()),
    case("a dimension of negative size",
         field(1, 1) + field(2, field(2, field(1, 2 ** 64 - 1))) +
         field(5, 16) + CHECKSUM, damage()),

    # the versions: this reader is one of version 1, as TensorFlow's
    case("version: as TensorFlow writes it", header=version()),
    case("version: producer 0", header=version(producer=0)),
    case("version: producer 7", header=version(producer=7)),
    case("version: producer 2**32 + 5", header=version(producer=2 ** 32 + 5)),
    case("version: for readers from 1", header=version(consumer=1)),
    case("version: for readers from -1, in ten bytes",
         header=version(consumer=2 ** 64 - 1)),
    case("version: for readers from -1, in five bytes",
         header=version(consumer=2 ** 32 - 1)),
    case("version: for readers from 2**32",
         header=version(consumer=2 ** 32)),
    case("version: not for readers 2 and 5",
         header=version(bad=field(3, 2) + field(3, 5))),
    case("version: not for readers 2 and 300, in one field",
         header=version(bad=field(3, varint(2) + varint(300)))),
    case("version: not for reader -1",
         header=version(bad=field(3, 2 ** 64 - 1))),
    case("version: not for reader 2, in ten bytes",
         header=version(bad=field(3, b"\x82" + b"\x80" * 8 + b"\x7e"))),
    case("version: for readers from 2", header=version(consumer=2),
         ours=OTHER_VERSION),
    case("version: for readers from 999", header=version(consumer=999),
         ours=OTHER_VERSION),
    case("version: for readers from 2**32 + 2",
         header=version(consumer=2 ** 32 + 2), ours=OTHER_VERSION),
    case("version: not for reader 1", header=version(bad=field(3, 1)),
         ours=OTHER_VERSION),
    case("version: not for readers 3 and 1",
         header=version(bad=field(3, 3) + field(3, 1)), ours=OTHER_VERSION),
    case("version: not for readers 7 and 1, in one field",
         header=version(bad=field(3, varint(7) + varint(1))),
         ours=OTHER_VERSION),
    case("version: not for reader 2**32 + 1",
         header=version(bad=field(3, 2 ** 32 + 1)), ours=OTHER_VERSION),
    case("version: not for reader 2**32 + 1, in one field",
         header=version(bad=field(3, varint(2 ** 32 + 1))),
         ours=OTHER_VERSION),
    case("version: producer -1, in ten bytes",
         header=version(producer=2 ** 64 - 1), ours=OTHER_VERSION),
    case("version: producer -1, in five bytes",
         header=version(producer=2 ** 32 - 1), ours=OTHER_VERSION),
    case("version: producer -2**31", header=version(producer=2 ** 31),
         ours=OTHER_VERSION),

    # what is no message
    case("a number that does not end",
         field(1, 1) + b"\x28" + b"\xff" * 20, damage()),
    case("a number of eleven bytes",
         field(1, 1) + b"\x28" + b"\x80" * 10 + b"\x01" + REST, damage()),
    case("a field that goes beyond the message",
         FOUR + field(16, b"abcdef")[:-2], damage()),
    case("a shape that is no message", FOUR + field(2, b"\x12\x05ab"),
         damage()),
    case("a dimension that is no message",
         FOUR + field(2, field(2, b"\x08")), damage()),
    case("field number 0", FOUR + b"\x00\x01", damage()),
    case("field number 0 with text", FOUR + b"\x02\x01a", damage()),
    case("a kind of field that does not exist",
         FOUR + varint((9 << 3) | 6) + b"\x01", damage()),
    case("the end of a group", FOUR + varint((9 << 3) | 4), damage()),
    case("the checksum cut short", FOUR[:-1], damage()),
    case("the name of a dimension that is not text",
         field(1, 1) + field(2, field(2, field(1, 4) + field(
             2, b"\xff\xfe"))) + field(5, 16) + CHECKSUM, damage()),
    case("a header that is no message", header=HEADER + b"\x18",
         ours=BAD_HEADER),
    case("a version that is no message", header=HEADER + field(3, b"\x08"),
         ours=BAD_HEADER),
    case("a list of versions that is cut short",
         header=HEADER + field(3, field(3, b"\x80")), ours=BAD_HEADER),
    case("a list of versions with a number of eleven bytes",
         header=HEADER + field(3, field(3, b"\xff" * 10 + b"\x01")),
         ours=BAD_HEADER),

    # strings
    case("no strings", entry(7, (0,), AT_NO_STRINGS, 4, NO_STRINGS_CRC),
         np.array([], dtype=object), dtype="string"),
    case("three strings",
         entry(7, (3,), AT_THREE_STRINGS, 10, THREE_STRINGS_CRC),
         np.array([b"a", b"bc", b""], dtype=object), dtype="string"),
    case("one string, no shape",
         entry(7, (), AT_ONE_STRING, 8, ONE_STRING_CRC),
         np.array(b"one", dtype=object), dtype="string"),
    case("three strings, another checksum",
         entry(7, (3,), AT_THREE_STRINGS, 10, THREE_STRINGS_CRC ^ 1),
         damage("does not have the checksum that is stored with it"),
         dtype="string"),
    case("three strings where there are numbers",
         entry(7, (3,), 0, 10, THREE_STRINGS_CRC), BAD_STRINGS,
         dtype="string"),

    # a shape of more elements than a tensor can have, also when a
    # later dimension has none; more dimensions than a tensor has
    case("more elements than fit 63 bits, before a dimension of none",
         field(1, 1) + field(2, field(2, field(1, 2 ** 62)) * 2 +
                             field(2, b"")) + checksum(0),
         damage("a shape of more elements than a tensor can have")),
    case("255 dimensions",
         field(1, 1) + field(2, DIMENSION + field(2, field(1, 1)) * 254) +
         field(5, 16) + CHECKSUM, not_supported("has 255 dimensions")),

    # the protocol buffer readers of the TensorFlow versions do not
    # agree on these, so they are refused here: a field number or a
    # length in more bytes than it needs (2.15 refuses more than
    # five), a message that ends after the message it is in (2.4 and
    # 2.11 read what there is of it), a list of versions right after a
    # single one (2.4 and 2.11 take the list apart)
    case("a field number in two bytes", b"\x88\x00\x01" + REST, damage(),
         stricter=True),
    case("a field number in six bytes",
         b"\x88\x80\x80\x80\x80\x00\x01" + REST, damage(), stricter="old"),
    case("a field number in ten bytes, with a bit beyond the 64th",
         b"\x88" + b"\x80" * 8 + b"\x02\x01" + REST, damage(),
         stricter="old"),
    case("the length of the shape in two bytes",
         field(1, 1) + b"\x12\x84\x00" + DIMENSION + field(5, 16) + CHECKSUM,
         damage(), stricter=True),
    case("the length of the shape in six bytes",
         field(1, 1) + b"\x12\x84\x80\x80\x80\x80\x00" + DIMENSION +
         field(5, 16) + CHECKSUM, damage(), stricter="old"),
    case("the length of a field the message does not have, in six bytes",
         FOUR + b"\x82\x01\x83\x80\x80\x80\x80\x00xyz", damage(),
         stricter="old"),
    case("a field number of the header in six bytes",
         header=b"\x88\x80\x80\x80\x80\x00\x01", ours=BAD_HEADER,
         stricter="old"),
    case("the length of the version in six bytes",
         header=HEADER + b"\x1a\x82\x80\x80\x80\x80\x00" + field(1, 1),
         ours=BAD_HEADER, stricter="old"),
    case("a shape that ends after its entry", FOUR + b"\x12\x7f", damage(),
         stricter="old"),
    case("a shape that takes the rest of its entry",
         field(1, 1) + field(5, 16) + CHECKSUM + b"\x12\x7f" + DIMENSION,
         damage(), stricter="old"),
    case("a dimension that ends after its shape",
         field(1, 1) + field(2, b"\x12\x7f\x08\x04") + field(5, 16) +
         CHECKSUM, damage(), stricter="old"),
    case("a version that ends after the header",
         header=HEADER + b"\x1a\x7f" + field(1, 1), ours=BAD_HEADER,
         stricter="old"),
    case("a list of versions that ends after its version",
         header=HEADER + field(3, field(1, 1) + b"\x1a\x7f\x02\x05"),
         ours=BAD_HEADER, stricter="old"),
    case("a list of versions right after a single one",
         header=version(bad=field(3, 5) + field(3, varint(16) + varint(0))),
         ours=BAD_HEADER, stricter=True),
    case("a list of one version right after a single one",
         header=version(bad=field(3, 5) + field(3, b"\x02")),
         ours=BAD_HEADER, stricter="new"),
    case("a single version right after a list",
         header=version(bad=field(3, varint(2) + varint(5)) + field(3, 7))),
    case("a list of versions after a single one and another field",
         header=HEADER + field(3, field(3, 5) + field(1, 1) +
                               field(3, b"\x02"))),

    # where this reader is the stricter one: TensorFlow reads these
    # files, which no writer makes (or which this reader does not
    # support)
    case("a group", FOUR + varint((9 << 3) | 3) + varint((9 << 3) | 4),
         damage(), stricter=True),
    case("a field number of more than 32 bits",
         FOUR + varint(2 ** 32 + (15 << 3)) + b"\x03", damage(),
         stricter=True),
    case("the rank unknown",
         field(1, 1) + field(2, DIMENSION + field(3, 1)) + field(5, 16) +
         CHECKSUM, not_supported("has no known shape"), stricter=True),
    case("33 dimensions",
         field(1, 1) + field(2, DIMENSION + field(2, field(1, 1)) * 32) +
         field(5, 16) + CHECKSUM, not_supported("has 33 dimensions"),
         stricter=True),
    case("254 dimensions",
         field(1, 1) + field(2, DIMENSION + field(2, field(1, 1)) * 253) +
         field(5, 16) + CHECKSUM, not_supported("has 254 dimensions"),
         stricter=True),
    case("32 dimensions",
         field(1, 1) + field(2, DIMENSION + field(2, field(1, 1)) * 31) +
         field(5, 16) + CHECKSUM, FIRST.reshape((4,) + (1,) * 31)),
    case("big endian", header=HEADER + field(2, 1),
         ours=not_supported("was not written on a little endian machine"),
         stricter=True),
    case("a byte order that does not exist", header=HEADER + field(2, 2),
         ours=not_supported("was not written on a little endian machine"),
         stricter=True),
    case("saved in slices", FOUR + field(7, field(1, b"")),
         not_supported("was saved in slices"), stricter=True),
    case("complex numbers",
         field(1, 8) + field(2, field(2, field(1, 2))) + field(5, 16) +
         CHECKSUM, not_supported("tensorflow.DataType 8"), stricter=True,
         dtype="complex64"),
    case("no strings, and no checksum of their lengths",
         entry(7, (0,), AT_NO_STRINGS, 0, 0), BAD_STRINGS, stricter=True,
         dtype="string"),
    case("no strings in a larger size",
         entry(7, (0,), AT_NO_STRINGS, 14, NO_STRINGS_CRC), BAD_STRINGS,
         stricter=True, dtype="string"),
    case("three strings in a larger size",
         entry(7, (3,), AT_THREE_STRINGS, 12, THREE_STRINGS_CRC), BAD_STRINGS,
         stricter=True, dtype="string"),
    case("three strings in a smaller size",
         entry(7, (3,), AT_THREE_STRINGS, 5, THREE_STRINGS_CRC), BAD_STRINGS,
         stricter=True, dtype="string"),
    case("the length of a string in ten bytes, with bits beyond the 64th",
         entry(7, (1,), AT_ODD_STRING, 16, ODD_STRING_CRC), BAD_STRINGS,
         stricter=True, dtype="string"),
]
assert len({message.name for message in MESSAGES}) == len(MESSAGES)


def write_case(prefix, message):
    with open(prefix + ".data-00000-of-00001", "wb") as fh:
        fh.write(DATA)
    with open(prefix + ".index", "wb") as fh:
        fh.write(table([(b"", message.header), (b"x", message.entry)]))
    return prefix


@pytest.mark.parametrize("message", MESSAGES,
                         ids=[message.name for message in MESSAGES])
def test_messages_of_the_index(tmp_path, message):
    """
        The header and the entries of the index are protocol buffer
        messages, and TensorFlow reads them as protocol buffers do. So
        does this reader, for what the messages say is read as
        TensorFlow reads it or not at all: a field counts when it has
        the kind its message gives it and is passed over otherwise, a
        number has the width and the sign of its field, of a field
        that is there twice the last one counts, and what is no
        message is damage; so is what the TensorFlow versions read
        differently from each other. (That TensorFlow reads these
        files the same is tested below, where it is installed.)
    """

    prefix = write_case(str(tmp_path / "ckpt"), message)
    if isinstance(message.ours, np.ndarray):
        assert_same_tensor(Checkpoint(prefix).get_tensor("x"), message.ours)
        return
    error, text = message.ours
    with pytest.raises(error) as failure:
        Checkpoint(prefix).get_tensor("x")
    assert type(failure.value) is error
    assert text in str(failure.value)


# --------------------------------------------------------------------
# messages made at random
# --------------------------------------------------------------------

def padded(value, length):
    """
        A varint written in `length` bytes, which is more than it
        needs (zeros at the top)
    """

    assert value < 1 << (7 * length)
    return bytes(((value >> (7 * i)) & 0x7F) | (0x80 if i < length - 1 else 0)
                 for i in range(length))


class RandomMessages(object):
    """
        Headers and entries of an index made at random: mostly what a
        writer makes, with fields of other kinds, numbers of every
        width and in every way of writing them, fields that are there
        twice or not at all, in any order, cut short

        The sizes of the dimensions are small or negative: TensorFlow
        makes room for a tensor of the shape an entry gives before it
        looks at the rest of the entry, so a shape of many elements
        would take that much memory in the process that reads it.

        Args:
            seed (int): for the random numbers
    """

    NUMBERS = [0, 1, 2, 3, 4, 7, 8, 16, 19, 64, 2 ** 31 - 1, 2 ** 31,
               2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1, 2 ** 63 - 1, 2 ** 63,
               2 ** 64 - 1]
    #: the sizes of dimensions (the last two are negative)
    SIZES = [0, 1, 2, 4, 4, 4, 16, 2 ** 63, 2 ** 64 - 1]
    #: what is no message, and has no size of a dimension in it
    #: whatever it is read as
    JUNK = [b"abc", b"\xff\xff\xff", b"\x08", b"\x0a\x05ab", b"\x12\x7f",
            b"zz", b"\x00"]

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def chance(self, share):
        return self.rng.random() < share

    def number(self, values=None):
        """
            A number as a field holds it (one of `values`, or any),
            written in the usual way, in more bytes than it needs, or
            in ten bytes with bits beyond the 64th
        """

        rng = self.rng
        if values is not None:
            value = rng.choice(values)
        elif self.chance(0.8):
            value = rng.choice(self.NUMBERS)
        else:
            value = rng.getrandbits(rng.choice([7, 14, 32, 64]))
        how = rng.random()
        if how < 0.8:
            return varint(value)
        if how < 0.9:
            return padded(value, rng.randint(len(varint(value)), 10))
        top = ((value >> 63) & 1) | rng.choice([0, 2, 4, 0x7E])
        return padded(value & (2 ** 63 - 1), 9)[:-1] + bytes(
            [((value >> 56) & 0x7F) | 0x80, top])

    def prefix(self, value):
        """
            A field number with its kind, or a length: now and then in
            more bytes than it needs
        """

        how = self.rng.random()
        if how < 0.97:
            return varint(value)
        usual = len(varint(value))
        if how < 0.985:
            return padded(value, self.rng.randint(usual, 5))
        return padded(value, self.rng.randint(6, 10))

    def some_field(self, numbers, sizes=False):
        """
            One field of any kind with a number from `numbers`; with
            `sizes`, a number in field 1 is the size of a dimension
        """

        rng = self.rng
        number = rng.choice(numbers)
        kinds = [0, 0, 2, 2, 5, 1] if self.chance(0.9) else [3, 4, 6, 7]
        kind = rng.choice(kinds)
        tag = self.prefix((number << 3) | kind)
        if kind == 0:
            return tag + self.number(
                self.SIZES if sizes and number == 1 else None)
        if kind == 2:
            how = rng.random()
            payload = b"" if how < 0.25 else self.dimension() \
                if how < 0.5 else self.shape() if how < 0.7 else \
                rng.choice(self.JUNK)
            return tag + self.prefix(len(payload)) + payload
        if kind in (5, 1):
            return tag + bytes(rng.getrandbits(8)
                               for _ in range(4 if kind == 5 else 8))
        return tag + (b"" if self.chance(0.5) else
                      bytes([rng.getrandbits(8)]))

    def changed(self, fields, numbers):
        """
            The message of `fields` after a few random changes: a
            field more, one twice, one less, another order, an end
            that is cut or goes on
        """

        rng = self.rng
        for _ in range(rng.choice([0, 1, 1, 1, 2, 2, 3])):
            how = rng.random()
            if how < 0.45:
                fields.insert(rng.randint(0, len(fields)),
                              self.some_field(numbers))
            elif how < 0.65 and fields:
                fields.insert(rng.randint(0, len(fields)), rng.choice(fields))
            elif how < 0.8 and fields:
                del fields[rng.randrange(len(fields))]
            elif how < 0.95:
                rng.shuffle(fields)
            else:
                # a field that is a message and ends after this one
                fields.append(varint((rng.choice(numbers[1:]) << 3) | 2) +
                              b"\x7f")
        out = b"".join(fields)
        if self.chance(0.04) and out:
            out = out[:-rng.randint(1, min(3, len(out)))]
        return out

    def dimension(self):
        rng = self.rng
        fields = []
        if self.chance(0.85):
            fields.append(b"\x08" + self.number(self.SIZES))
        if self.chance(0.15):
            fields.append(field(2, rng.choice(
                [b"name", b"\xff\xfe", u"l\u00e4nge".encode("utf-8")])))
        while self.chance(0.15):
            fields.insert(rng.randint(0, len(fields)),
                          self.some_field([0, 1, 2, 3, 9], sizes=True))
        return b"".join(fields)

    def shape(self):
        rng = self.rng
        fields = [field(2, self.dimension())
                  for _ in range(rng.choice([0, 1, 1, 1, 2, 3]))]
        if self.chance(0.1):
            fields.append(field(3, rng.choice([0, 1, 2 ** 64 - 1]))
                          if self.chance(0.8) else field(3, b"\x01"))
        while self.chance(0.15):
            fields.insert(rng.randint(0, len(fields)),
                          self.some_field([0, 1, 2, 3, 4, 9], sizes=True))
        return b"".join(fields)

    def entry(self):
        rng = self.rng
        how = rng.random()
        if how < 0.75:
            # numbers: the first four of the data file, or near it
            fields = [field(1, 1), field(2, DIMENSION), field(5, 16),
                      CHECKSUM]
            if self.chance(0.2):
                fields[1] = field(2, self.shape())
            if self.chance(0.1):
                fields[0] = field(1, rng.choice(
                    [1, 2, 3, 4, 5, 6, 9, 10, 17, 19, 22, 23, 8, 7, 0, 21]))
            if self.chance(0.1):
                fields.append(field(4, rng.choice([0, 4, 8, 48, 1000])))
            if self.chance(0.1):
                fields[2] = field(5, rng.choice([0, 4, 8, 16, 32]))
        elif how < 0.9:
            # strings
            shape, offset, size, crc = rng.choice([
                ((0,), AT_NO_STRINGS, 4, NO_STRINGS_CRC),
                ((3,), AT_THREE_STRINGS, 10, THREE_STRINGS_CRC),
                ((), AT_ONE_STRING, 8, ONE_STRING_CRC),
                ((1,), AT_ODD_STRING, 16, ODD_STRING_CRC)])
            if self.chance(0.3):
                size = rng.choice([0, 3, 4, 5, 8, 10, 12, 16, 20])
            if self.chance(0.2):
                shape = rng.choice(
                    [(), (0,), (1,), (2,), (3,), (3, 1), (1, 3)])
            fields = [field(1, 7), field(2, b"".join(
                field(2, field(1, n) if n else b"") for n in shape)),
                field(4, offset)]
            if size:
                fields.append(field(5, size))
            fields.append(checksum(crc))
        else:
            fields = [self.some_field([1, 2, 3, 4, 5, 6, 7])
                      for _ in range(rng.randint(0, 5))]
        return self.changed(
            fields, [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 2 ** 29 - 1, 2 ** 29])

    def version(self):
        rng = self.rng
        fields = []
        if self.chance(0.8):
            fields.append(field(1, rng.choice(
                [0, 1, 1, 1, 7, 2 ** 31, 2 ** 32 - 1, 2 ** 32 + 5,
                 2 ** 64 - 1])))
        if self.chance(0.4):
            fields.append(field(2, rng.choice(
                [0, 1, 1, 2, 999, 2 ** 32, 2 ** 32 + 2, 2 ** 32 - 1,
                 2 ** 64 - 1])))
        while self.chance(0.4):
            if self.chance(0.5):
                fields.append(field(3, rng.choice(
                    [0, 1, 2, 5, 2 ** 32 + 1, 2 ** 64 - 1])))
            else:
                fields.append(field(3, b"".join(
                    self.number() for _ in range(rng.randint(0, 3)))))
        return self.changed(fields, [0, 1, 2, 3, 4, 9])

    def header(self):
        rng = self.rng
        if self.chance(0.6):
            return HEADER
        fields = [field(1, rng.choice([1, 1, 1, 1, 2 ** 32 + 1, 0, 2]))]
        if self.chance(0.3):
            fields.append(field(2, rng.choice([0, 0, 1, 2, 2 ** 32])))
        if self.chance(0.7):
            fields.append(field(3, self.version()))
        return self.changed(fields, [0, 1, 2, 3, 4, 9])

    def message(self, number):
        return Message("random {}".format(number), self.header(),
                       self.entry(), None, None, None)


#: the errors of the reader: the file is damaged, it is not supported,
#: or the data file it names is not there
ERRORS = (CorruptCheckpointError, UnsupportedModelError, FileNotFoundError)


def read_here(prefix):
    """
        What this reader does with the tensor 'x' of a checkpoint: as
        the program below writes down what TensorFlow does, or
        {"error": the text of the reader's error}
    """

    try:
        value = Checkpoint(prefix).get_tensor("x")
    except ERRORS as error:
        return {"error": "{}: {}".format(type(error).__name__, error)}
    if isinstance(value, bytes):
        return {"shape": [], "dtype": "object", "values": [value.hex()]}
    if value.dtype == object:
        return {"shape": list(value.shape), "dtype": "object",
                "values": [item.hex() for item in value.reshape(-1)]}
    return {"shape": list(value.shape), "dtype": value.dtype.name,
            "values": value.tobytes().hex()}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_messages_are_read_or_refused(tmp_path, seed):
    """
        Whatever the index says, the reader returns the tensor or
        raises one of its own errors; no other error escapes. (What it
        returns is compared with TensorFlow below.)
    """

    made = RandomMessages(seed)
    prefix = str(tmp_path / "ckpt")
    outcomes = collections.Counter()
    for number in range(1500):
        write_case(prefix, made.message(number))
        result = read_here(prefix)
        outcomes[result["error"].split(":")[0] if "error" in result
                 else "read"] += 1
    assert sum(outcomes.values()) == 1500
    assert outcomes["read"] > 150
    assert outcomes["CorruptCheckpointError"] > 500
    assert outcomes["UnsupportedModelError"] > 50


# --------------------------------------------------------------------
# object graphs made at random
# --------------------------------------------------------------------

def random_graph(rng):
    """
        The object graph of graph_nodes() after random changes: more
        fields on its objects (children, variables, slot variables,
        fields of later TensorFlow versions, fields of other kinds),
        or bytes that are changed, missing or added
    """

    nodes = graph_nodes()
    if rng.random() < 0.6:
        for _ in range(rng.randint(1, 3)):
            more = rng.choice([
                field(rng.choice([1, 2, 3, 4, 5, 6, 9]),
                      rng.choice([0, 1, 5, 2 ** 32 + 1])),
                field(rng.choice([6, 9]), bytes(
                    rng.getrandbits(8) for _ in range(rng.randint(0, 5)))),
                reference(rng.choice(["x", "layer_with_weights-0", "kernel",
                                      "layer_with_weights-1"]),
                          rng.choice([0, 1, 2, 3, 5, 6, 99, 2 ** 32 - 1,
                                      2 ** 32 + 2])),
                attribute(rng.choice(["a/b", ""]),
                          rng.choice([KERNEL, BIAS, "missing", ""]),
                          name=rng.choice([b"VARIABLE_VALUE", b"OTHER",
                                           b"\xff"])),
                field(3, field(1, rng.choice([0, 2, 7])) + field(
                    2, rng.choice([b"m", b"\xfe"])) +
                    field(3, rng.choice([1, 5, 9]))),
                field(4, field(1, rng.choice([b"saver", b"\xff"])) +
                      field(2, b"object")),
                field(5, rng.choice([field(1, 1), b"\x08"])),
                fixed32(rng.choice([1, 2, 7]), b"abcd"),
                fixed64(rng.choice([1, 2, 7]), bytes(8)),
            ])
            which = rng.randrange(len(nodes))
            nodes[which] = more + nodes[which] if rng.random() < 0.5 \
                else nodes[which] + more
        return object_graph(nodes)
    data = bytearray(object_graph(nodes))
    for _ in range(rng.randint(1, 3)):
        how, position = rng.random(), rng.randrange(len(data))
        if how < 0.5:
            data[position] = rng.getrandbits(8)
        elif how < 0.7:
            del data[position]
        elif how < 0.9:
            data.insert(position, rng.getrandbits(8))
        else:
            del data[position:]
    return bytes(data)


def test_random_object_graphs_are_read_or_refused():
    """
        Whatever the object graph says, its variables are found or it
        is refused as not being one (a ValueError, which the loader
        reports as damage); no other error escapes
    """

    rng = random.Random(4)
    outcomes = collections.Counter()
    for _ in range(3000):
        data = random_graph(rng)
        try:
            variables = loading._variables_in_object_graph(memoryview(data))
        except ValueError:
            outcomes["refused"] += 1
            continue
        outcomes["read"] += 1
        assert all(len(names) == 3 and all(isinstance(name, str)
                                           for name in names)
                   for names in variables)
    assert outcomes["read"] > 500 and outcomes["refused"] > 500


# --------------------------------------------------------------------
# with TensorFlow
# --------------------------------------------------------------------

@pytest.fixture(scope="module")
def tf():
    return pytest.importorskip("tensorflow")


#: what TensorFlow reads from the checkpoints of a list, written down
#: one by one; the program of a process of its own (below). Only an
#: error of TensorFlow itself counts as TensorFlow refusing a file:
#: anything else ends the program with an error.
TENSORFLOW_READS = r'''
import json
import resource
import sys

# TensorFlow makes room for a tensor of the shape that an entry gives
# before it looks at the rest of the entry: the address space of this
# process is bounded, so that a shape of very many elements is an
# error here and not memory taken from the machine
resource.setrlimit(resource.RLIMIT_AS, (12 << 30, 12 << 30))

import tensorflow as tf


def written(tensor):
    flat = tf.reshape(tensor, [-1]).numpy()
    if flat.dtype == object:
        values = [bytes(value).hex() for value in flat.tolist()]
    else:
        values = flat.tobytes().hex()
    return {"shape": tensor.shape.as_list(), "dtype": str(flat.dtype),
            "values": values}


results = []
for prefix, dtype, how in json.load(open(sys.argv[1])):
    if how == "restore":
        # what restores the variables of a model
        try:
            tensor = tf.raw_ops.RestoreV2(
                prefix=prefix, tensor_names=["x"], shape_and_slices=[""],
                dtypes=[getattr(tf, dtype)])[0]
        except tf.errors.OpError as error:
            results.append({"refused": type(error).__name__,
                            "message": str(error)[:300]})
            continue
    else:
        # the reader of tf.train.load_checkpoint
        tensor = tf.constant(tf.train.load_checkpoint(prefix).get_tensor("x"))
    results.append(written(tensor))
with open(sys.argv[2], "w") as out:
    json.dump(results, out)
'''

#: the errors with which TensorFlow refuses the checkpoints of the
#: table above
REFUSALS = ("DataLossError", "InvalidArgumentError", "NotFoundError",
            "OutOfRangeError")


def tensorflow_reads(tasks, work_dir, refusals=None):
    """
        What TensorFlow reads from each checkpoint of a list, in a
        process of its own (TensorFlow ends the process on some
        damaged files; none of the files of these tests is one, and
        a process that ends early fails the test)

        Args:
            tasks (list): (path of the checkpoint, type of its tensor
                'x', "restore" for the operation that restores the
                variables of a model or "reader" for the reader of
                tf.train.load_checkpoint, which is only asked for
                files it must read)
            work_dir (str): a directory for the files of the exchange
            refusals (tuple): the errors with which TensorFlow may
                refuse a file (default: any error of its restore
                operation)

        Returns:
            list: for each task {"shape", "dtype", "values"}, or
                {"refused": the error of TensorFlow, "message"}
    """

    tasks_path = os.path.join(work_dir, "tasks.json")
    out_path = os.path.join(work_dir, "read.json")
    with open(tasks_path, "w") as fh:
        json.dump(tasks, fh)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    done = subprocess.run(
        [sys.executable, "-c", TENSORFLOW_READS, tasks_path, out_path],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=1800)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-3000:]
    with open(out_path) as fh:
        results = json.load(fh)
    assert len(results) == len(tasks)
    for result in results:
        assert ("values" in result) != ("refused" in result), result
        assert refusals is None or result.get("refused", refusals[0]) in \
            refusals, result
    return results


def same_as_here(theirs, ours):
    return "values" in theirs and theirs["shape"] == ours["shape"] and \
        theirs["values"] == ours["values"] and \
        (ours["dtype"] == "object" or theirs["dtype"] == ours["dtype"])


#: the case that tells which protocol buffer reader a TensorFlow has:
#: the one of 2.4 and 2.11 reads what there is of a message that ends
#: after the message it is in, the one of 2.15 refuses it
TELLS_THE_READER = "a shape that ends after its entry"


def test_messages_of_the_index_as_tensorflow_reads_them(tf, tmp_path):
    """
        The checkpoints of test_messages_of_the_index, read by the
        TensorFlow that is installed: what this reader reads,
        TensorFlow reads the same, with the operation that restores
        the variables of a model and with the reader of
        tf.train.load_checkpoint; what this reader refuses, TensorFlow
        refuses too, but for the cases marked as stricter here, which
        it reads (some of them only with one of its two protocol
        buffer readers).
    """

    tasks, cases = [], []
    for number, message in enumerate(MESSAGES):
        directory = tmp_path / "case_{:03d}".format(number)
        directory.mkdir()
        prefix = write_case(str(directory / "ckpt"), message)
        read = isinstance(message.ours, np.ndarray)
        for how in ("restore", "reader") if read else ("restore",):
            tasks.append([prefix, message.dtype, how])
            cases.append((message, how))
    results = tensorflow_reads(tasks, str(tmp_path), REFUSALS)

    old, = ["values" in result for (message, _), result in zip(cases, results)
            if message.name == TELLS_THE_READER]
    counts = collections.Counter()
    for (message, how), result in zip(cases, results):
        where = (message.name, how, result)
        if isinstance(message.ours, np.ndarray):
            ours = {"shape": list(message.ours.shape),
                    "dtype": message.ours.dtype.name}
            if message.ours.dtype == object:
                ours["values"] = [value.hex()
                                  for value in message.ours.reshape(-1)]
            else:
                ours["values"] = message.ours.tobytes().hex()
            assert same_as_here(result, ours), where
            counts["the same"] += 1
        elif message.stricter in (True, "old" if old else "new"):
            assert "values" in result, where
            counts["read by TensorFlow only"] += 1
        else:
            assert "refused" in result, where
            counts["refused by both"] += 1
    assert min(counts["the same"], counts["read by TensorFlow only"],
               counts["refused by both"]) > 10, counts
    # the readers of the TensorFlow versions these tests have run on
    version = tuple(int(part) for part in tf.__version__.split(".")[:2])
    if version in ((2, 4), (2, 11), (2, 15)):
        assert old == (version < (2, 15))


#: refusals of this reader that TensorFlow, or some version of it, may
#: read: where this reader is the stricter one (the texts of its
#: errors)
MAY_BE_READ_BY_TENSORFLOW = (
    "wire type 3", "a field number or a length of more than 32 bits",
    "has no known shape", "was saved in slices", " dimensions",
    "little endian", "the size its strings need", "tensorflow.DataType",
    "a list of versions right after a single one",
    "Truncated protocol buffer message")

#: numpy type -> the name of the type in TensorFlow
TENSORFLOW_TYPES = {"object": "string"}


def test_random_messages_as_tensorflow_reads_them(tf, tmp_path):
    """
        The random checkpoints of test_random_messages_are_read_or_refused,
        read by the TensorFlow that is installed: what this reader
        reads, TensorFlow reads the same; what this reader refuses,
        TensorFlow refuses too, unless the refusal is of one of the
        kinds for which this reader is the stricter one.
    """

    tasks, here = [], []
    for seed in (1, 2, 3):
        made = RandomMessages(seed)
        for number in range(1500):
            directory = tmp_path / "s{}_{:04d}".format(seed, number)
            directory.mkdir()
            prefix = write_case(str(directory / "ckpt"), made.message(number))
            ours = read_here(prefix)
            here.append(ours)
            dtype = ours.get("dtype", "float32")
            tasks.append([prefix, TENSORFLOW_TYPES.get(dtype, dtype),
                          "restore"])
    results = tensorflow_reads(tasks, str(tmp_path))

    counts = collections.Counter()
    for task, ours, theirs in zip(tasks, here, results):
        where = (task[0], ours, theirs)
        if "values" in ours:
            assert same_as_here(theirs, ours), where
            counts["the same"] += 1
        elif "values" in theirs:
            assert any(text in ours["error"]
                       for text in MAY_BE_READ_BY_TENSORFLOW), where
            counts["read by TensorFlow only"] += 1
        else:
            counts["refused by both"] += 1
    assert counts["the same"] > 500 and counts["refused by both"] > 2000, \
        counts


def test_object_graphs_as_tensorflow_parses_them(tf):
    """
        Object graphs, right and changed at random, parsed by the
        class TensorFlow reads them with: every graph that is read
        here it parses, to the same objects (it is not asked to check
        the references between the objects, which this reader does)
    """

    from google.protobuf.message import DecodeError
    from tensorflow.core.protobuf import trackable_object_graph_pb2

    def theirs(data):
        graph = trackable_object_graph_pb2.TrackableObjectGraph.FromString(
            data)
        return [([(child.local_name, child.node_id)
                  for child in node.children],
                 [(one.name, one.full_name, one.checkpoint_key)
                  for one in node.attributes]) for node in graph.nodes]

    assert loading._objects_in_object_graph(memoryview(object_graph(
        graph_nodes()))) == theirs(object_graph(graph_nodes()))
    rng = random.Random(4)
    counts = collections.Counter()
    for _ in range(3000):
        data = random_graph(rng)
        try:
            ours = loading._objects_in_object_graph(memoryview(data))
        except ValueError:
            with warnings.catch_warnings():
                # (some versions parse a part of a graph and warn)
                warnings.simplefilter("ignore")
                try:
                    theirs(data)
                    counts["parsed by TensorFlow only"] += 1
                except DecodeError:
                    counts["refused by both"] += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ours == theirs(data), data
        counts["the same"] += 1
    assert counts["the same"] > 500 and counts["refused by both"] > 300, \
        counts
