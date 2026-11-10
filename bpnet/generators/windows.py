"""
    Reading the values of many windows from bigWig files

    A training batch needs the values of every signal and control
    track in the output window of every example: 512 windows of 1,000
    bases for the released settings. WindowReader reads all windows of
    a batch from a set of files in one call and returns one float32
    array, with zero where a file has no data.

    Windows that overlap are read together: an example and its
    reverse complement copy cover almost the same bases (they differ
    by their jitter), and the compressed blocks of a bigWig that both
    need are then fetched and unpacked once instead of twice.

    (Jacob Schreiber's figwig reads the windows of a batch in the same
    time once overlapping windows are merged, 16 ms for 128 windows in
    4 files, so it is not used here. Its strength, many windows read
    on several threads, would matter for loading all loci of a run at
    once.)
"""

import numpy as np
import pyBigWig


def interval_values(bigwig, chrom, start, end):
    """
        The values of a bigWig file in an interval, with zero at the
        positions that have no data

        pyBigWig hands the values over as a float32 array (the type
        they have in the file) when it was built with numpy; without,
        it returns a Python list of floats that numpy converts again,
        which takes several times longer. Both give the same numbers
        once they are stored in a float32 array.

        Args:
            bigwig (pyBigWig.bigWigFile): an open bigWig file
            chrom (str): chromosome
            start (int): start of the interval
            end (int): end of the interval (exclusive)

        Returns:
            numpy.ndarray: one value per position
    """

    if pyBigWig.numpy:
        values = bigwig.values(chrom, start, end, numpy=True)
        # a file without any data has nothing to return: pyBigWig
        # 0.3.22 then hands back an array without dimensions and with
        # arbitrary content. Anything but a row of values is read the
        # other way, which gives an empty result as it always did
        if values.ndim == 1:
            values[np.isnan(values)] = 0.0
            return values

    return np.nan_to_num(bigwig.values(chrom, start, end))


class WindowReader(object):
    """
        The values of windows of one width in a set of bigWig files

        Args:
            paths (list): bigWig files; they become the channels of
                what read() returns, in this order
    """

    def __init__(self, paths):
        self.paths = list(paths)
        self._files = [pyBigWig.open(path) for path in self.paths]

    def close(self):
        """
            Close the files
        """

        for bigwig in self._files:
            bigwig.close()
        self._files = []

    def read(self, chroms, starts, width):
        """
            The values of every window in every file

            A window on a chromosome a file does not have, or one that
            reaches beyond the chromosome, raises RuntimeError (as
            pyBigWig does for a single interval). A file with a header
            and no entries returns nothing for any interval, which
            cannot be stored into a batch: it raises ValueError, by
            name, when it is read (not when it is opened: the control
            files of a run whose control inputs are set to zero are
            never read).

            Args:
                chroms (list): chromosome of each window
                starts (list): start of each window
                width (int): width of the windows

            Returns:
                numpy.ndarray: float32 array of shape (number of
                    windows, number of files, width)
        """

        chroms = [str(chrom) for chrom in chroms]
        starts = [int(start) for start in starts]
        width = int(width)
        out = np.zeros((len(chroms), len(self.paths), width),
                       dtype=np.float32)
        if len(chroms) == 0 or len(self.paths) == 0:
            return out

        groups = _overlapping_windows(chroms, starts, width)
        for channel, bigwig in enumerate(self._files):
            for chrom, first, last, members in groups:
                values = interval_values(bigwig, chrom, first, last)
                if len(values) != last - first:
                    raise ValueError(
                        "The bigWig file {} holds no data (it has no "
                        "values to return for {}:{}-{})".format(
                            self.paths[channel], chrom, first, last))
                for row in members:
                    offset = starts[row] - first
                    out[row, channel, :] = values[offset:offset + width]
        return out


def _overlapping_windows(chroms, starts, width):
    """
        Group the windows that overlap, so that each group is read
        once

        Args:
            chroms (list): chromosome of each window
            starts (list): start of each window
            width (int): width of the windows

        Returns:
            list: (chromosome, start, end, rows) per group: the
                interval that covers the windows of the group and the
                indices of those windows
    """

    order = sorted(range(len(chroms)), key=lambda i: (chroms[i], starts[i]))
    groups = []
    for row in order:
        chrom, start = chroms[row], starts[row]
        if groups and groups[-1][0] == chrom and start < groups[-1][2]:
            # the windows are sorted and of one width: this one ends
            # last
            last = groups[-1]
            groups[-1] = (chrom, last[1], start + width, last[3] + [row])
        else:
            groups.append((chrom, start, start + width, [row]))
    return groups
