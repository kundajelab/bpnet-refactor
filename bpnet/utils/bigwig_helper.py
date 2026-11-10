import re

import pyBigWig
import numpy as np
from tqdm import tqdm

#: whether the positions of a region are written as a run (a fixedStep 
#: section: one value per position) or, as they always were, as an 
#: entry per position with its own start and end. pyBigWig before 
#: 0.3.29 records the end of a run six positions too far, in the 
#: section and in the index of the file; the values that are read 
#: back are right, but the file is not, so with those versions 
#: nothing changes.
RUNS = tuple(int(part) for part in re.findall(
    r'\d+', pyBigWig.__version__)[:3]) >= (0, 3, 29)


def region_segments(regions, order):
    """
        The part of every region that is written to a bigWig
        
        Regions may overlap, and a bigWig holds one value per 
        position. Regions are taken in the order of the file. Where a 
        region overlaps the next one, it is written up to the 
        midpoint between the centres of the two, and the next one 
        from there: every position gets the value of the region 
        whose centre is closest.
        
        Args:
            regions (list): (chromosome, start, end, centre) of every 
                region
                
            order (list): indices of the regions to write, in the 
                order of the file (chromosomes as in the header, 
                then by start)
                
        Returns:
            list: (index of the region, chromosome, start, end) for 
                each region in `order`: the positions that are 
                written from it. A region between two others with 
                the same centre has none (start equals end).
    """
    
    segments = []
    cur_chr = ""
    cur_end = 0

    for itr in range(len(order)):

        i = order[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
    
        if i_chr != cur_chr: 
            cur_chr = i_chr
            cur_end = 0
    
        # bring current end to at least start of current region
        if cur_end < i_start:
            cur_end = i_start
    
        assert(regions[i][2]>=cur_end)
    
        # figure out where to stop for this region, get next region
        # which may partially overlap with this one
        next_end = i_end
    
        if itr+1 != len(order):
            n = order[itr+1]
            next_chr, next_start, _, next_mid = regions[n]
       
            if next_chr == i_chr and next_start < i_end:
                # if next region overlaps with this, end between their midpoints
                next_end = (i_mid+next_mid)//2
        
        segments.append((i, i_chr, cur_end, next_end))
        
        cur_end = next_end
    
    return segments


def as_float32(values):
    """
        The values as pyBigWig stores a list of Python floats: every 
        value as a double first, then as a float32
        
        Going through the double matters for the few values that a 
        direct conversion treats differently: integers beyond 2**53 
        and long doubles are rounded twice, and a signalling NaN 
        becomes a quiet one. Values beyond the range of float32 
        become infinite. Anything that is not a plain array of 
        numbers (a masked array, a list, an array of objects) is 
        converted value by value with float(), as it always was.
        
        Args:
            values (numpy.ndarray): the values of a run of positions
            
        Returns:
            numpy.ndarray: float32 array
    """
    
    with np.errstate(over='ignore', invalid='ignore'):
        if isinstance(values, np.ma.MaskedArray) or \
                not isinstance(values, np.ndarray) or \
                values.dtype.kind not in 'biuf':
            doubles = np.array([float(x) for x in values], dtype=np.float64)
        else:
            doubles = values.astype(np.float64)
        return doubles.astype(np.float32)


def add_run(bw, chrom, start, end, values):
    """
        Add the values of the consecutive positions start .. end - 1 
        to a bigWig file that is being written
        
        Where pyBigWig can be trusted with it (see RUNS) the 
        positions go into the file as one run of float32 values. An 
        entry per position, with its own start and end, takes three 
        times the space before compression and several times as long 
        to write; what is read back from the file, position by 
        position or as a summary, is the same.
        
        Args:
            bw (pyBigWig.bigWigFile): file opened for writing, with 
                its header added
            
            chrom (str): chromosome
            
            start (int): first position
            
            end (int): position after the last
            
            values (numpy.ndarray): one value per position
    """
    
    start, end = int(start), int(end)
    
    # what pyBigWig refuses when it is given every position (a run 
    # has no ends to compare with, and a negative start is taken for 
    # a very large one)
    if start < 0 or end < start or len(values) != end - start:
        raise RuntimeError(
            "Cannot write {} values to {}:{}-{} of the bigWig file".format(
                len(values), chrom, start, end))
    
    if end == start:
        return
    
    if not RUNS:
        bw.addEntries([chrom] * (end - start), 
                      list(range(start, end)), 
                      ends=list(range(start + 1, end + 1)), 
                      values=[float(x) for x in values])
        return
    
    values = as_float32(values)
    if not pyBigWig.numpy:
        values = [float(x) for x in values]
    
    bw.addEntries(str(chrom), start, values=values, span=1, step=1)


def write_bigwig(data, regions, header, bw_out, outstats_file):
    # regions may overlap but as we go in sorted order, at a given position,
    # we will pick the value from the interval whose summit is closest to 
    # current position
    
    chr_to_idx = {}
    for i,x in enumerate(header):
        chr_to_idx[x[0]] = i

    bw = pyBigWig.open(bw_out, 'w')
    bw.addHeader(header)
    
    # regions may not be sorted, so get their sorted order
    order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

    all_entries = []

    for i, i_chr, cur_end, next_end in tqdm(
            region_segments(regions, order_of_regs)):
        
        i_start = regions[i][1]
        vals = data[i][cur_end - i_start:next_end - i_start]

        add_run(bw, i_chr, cur_end, next_end, vals)
    
        all_entries.append(vals)

    bw.close()

    all_entries = np.hstack(all_entries)

    with open(outstats_file, 'w') as f:
        f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
        f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
        f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
        f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
        f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
        f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
        f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
        f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
        f.write("Max\t{:.6f}\n".format(np.max(all_entries)))